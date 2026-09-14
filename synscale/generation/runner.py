"""Generation runner: frozen pool + one teacher -> filtered JSONL + manifest (doc 09 §6).

Processes the pool in frozen order in shards. For each response it counts student tokens
(rendered = markers + prompt + response), applies the teacher-agnostic filters, flags
benchmark contamination (union-drop is applied later by build_dataset across teachers),
and writes one JSONL line per generation (retained and dropped alike, so filter effects
are auditable). A per-teacher manifest records the backend, decoding, hashes, per-filter
counts, throughput and (optional) energy. Runs are resumable at shard granularity.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from synscale.generation.backend import GenerationBackend, SamplingSpec, request_seed
from synscale.generation.filters import FilterConfig, ResponseFilter, filter_stats
from synscale.generation.prompt_pool import PromptRecord, iter_pool, pool_sha256
from synscale.generation.tokens import TokenCounter, count_rendered, count_words


def _sha(obj: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class PowerMeter:
    """Optional NVML energy integration; a no-op stub if pynvml is unavailable."""

    def __init__(self, device: int = 0):
        self.device = device
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    def power_w(self) -> float:
        if self._nvml is None:
            return 0.0
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        except Exception:
            return 0.0


def run_generation(
    pool_path: str | Path,
    backend: GenerationBackend,
    spec: SamplingSpec,
    counter: TokenCounter,
    out_dir: str | Path,
    *,
    teacher_id: str,
    pool_version: str,
    target_student_tokens: int,
    overgeneration_factor: float = 1.25,
    shard_size: int = 5000,
    filter_cfg: Optional[FilterConfig] = None,
    lang_detector=None,
    contam_index=None,
    system_prompt: Optional[str] = None,
    manifest_extra: Optional[dict] = None,
    resume: bool = True,
    max_prompts: Optional[int] = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # parallel sharding: this worker handles a contiguous slice of the pool; no global stop
    slice_start = slice_end = None
    if shard_count > 1:
        total = sum(1 for _ in iter_pool(pool_path))
        per = -(-total // shard_count)
        slice_start, slice_end = shard_index * per, min((shard_index + 1) * per, total)
    rf = ResponseFilter(filter_cfg, lang_detector=lang_detector)
    meter = PowerMeter()
    target = int(target_student_tokens * overgeneration_factor)

    shard_paths: list[Path] = []
    all_results = []
    retained_tokens = 0
    n_offered = 0
    energy_kwh = 0.0
    t_start = time.time()

    shard: list[PromptRecord] = []
    shard_idx = 0

    def flush(shard_recs: list[PromptRecord], idx: int):
        nonlocal retained_tokens, n_offered, energy_kwh
        prefix = f"shard-w{shard_index}-" if shard_count > 1 else "shard-"
        shard_path = out_dir / f"{prefix}{idx:05d}.jsonl"
        if resume and shard_path.exists():
            # Count its retained tokens for the stopping rule without regenerating.
            toks = 0
            for line in shard_path.read_text().splitlines():
                d = json.loads(line)
                if d.get("retained"):
                    toks += d["tokens_student"]
                    all_results.append(_result_stub(d))
                n_offered_local(d)
            shard_paths.append(shard_path)
            return toks
        msgs = []
        seeds = []
        for r in shard_recs:
            m = r.messages()
            if system_prompt and (not m or m[0].get("role") != "system"):
                m = [{"role": "system", "content": system_prompt}, *m]
            msgs.append(m)
            seeds.append(request_seed(spec.seed, pool_version, r.prompt_id, 0))
        p0 = meter.power_w()
        t0 = time.time()
        gens = backend.generate(msgs, spec, seeds=seeds)
        dt = time.time() - t0
        p1 = meter.power_w()
        energy_kwh += (0.5 * (p0 + p1)) * dt / 3.6e6
        toks = 0
        lines = []
        from synscale.generation.filters import FilterResult
        for r, g in zip(shard_recs, gens):
            n_stud = count_rendered(counter, r.student_prompt(), g.text)
            contaminated = False
            if contam_index is not None:
                contaminated = bool(contam_index.flag(r.student_prompt() + " " + g.text, first_only=True))
            fres = rf.apply(g.text, category=r.category, finish_reason=g.finish_reason,
                            n_student_tokens=n_stud, str_kind=r.str_kind,
                            is_seeded=(r.source_family == "seeded"), contaminated=contaminated)
            all_results.append(fres)
            rec = {
                "example_id": f"{pool_version}:{teacher_id}:{r.prompt_id}:0",
                "prompt_id": r.prompt_id, "category": r.category,
                "prompt_source": r.source, "teacher": teacher_id,
                "student_prompt": r.student_prompt(), "response": g.text,
                "finish_reason": g.finish_reason,
                "tokens_teacher_prompt": g.n_prompt_tokens,
                "tokens_teacher_response": g.n_response_tokens,
                "tokens_student": n_stud,
                "response_len_student": counter.count(g.text),
                "word_count": count_words(g.text),
                "cum_logprob": g.cumulative_logprob,
                "gold": r.to_json().get("gold"),
                "verifiable": r.to_json().get("verifiable", False),
                "flags": fres.flags, "retained": fres.retained,
            }
            lines.append(json.dumps(rec))
            if fres.retained:
                toks += n_stud
        shard_path.write_text("\n".join(lines) + "\n")
        shard_paths.append(shard_path)
        return toks

    def _result_stub(d):
        from synscale.generation.filters import FilterResult
        return FilterResult(retained=d.get("retained", False), flags=d.get("flags", {}))

    def n_offered_local(d):
        nonlocal n_offered
        n_offered += 1

    pos = -1
    for rec in iter_pool(pool_path):
        pos += 1
        if slice_start is not None and not (slice_start <= pos < slice_end):
            continue
        if max_prompts is not None and n_offered >= max_prompts:
            break
        shard.append(rec)
        n_offered += 1
        if len(shard) >= shard_size:
            retained_tokens += flush(shard, shard_idx)
            shard = []
            shard_idx += 1
            if shard_count == 1 and retained_tokens >= target:
                break
    if shard and retained_tokens < target and (max_prompts is None or True):
        retained_tokens += flush(shard, shard_idx)

    wall = time.time() - t_start
    stats = filter_stats(all_results)
    total_teacher_tokens = None  # filled by manifest_extra if measured
    manifest = {
        "manifest_version": 1,
        "teacher": teacher_id,
        "pool_version": pool_version,
        "pool_sha256": pool_sha256(pool_path),
        "student_tokenizer": {"name": counter.name, "sha256": counter.sha256},
        "backend": backend.info(),
        "decoding": asdict(spec),
        "target_student_tokens": target_student_tokens,
        "overgeneration_factor": overgeneration_factor,
        "retained_student_tokens": retained_tokens,
        "n_offered": n_offered,
        "filter_stats": stats,
        "shards": [p.name for p in shard_paths],
        "wall_seconds": wall,
        "energy_kwh": energy_kwh,
        "tokens_per_second": (retained_tokens / wall) if wall else 0.0,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    manifest["dataset_sha256"] = _sha([p.name for p in shard_paths] + [retained_tokens, n_offered])
    mname = f"manifest-w{shard_index}.json" if shard_count > 1 else "manifest.json"
    (out_dir / mname).write_text(json.dumps(manifest, indent=2))
    return manifest


def iter_generated(out_dir: str | Path, retained_only: bool = True) -> Iterable[dict]:
    out_dir = Path(out_dir)
    for shard in sorted(out_dir.glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            d = json.loads(line)
            if (not retained_only) or d.get("retained"):
                yield d
