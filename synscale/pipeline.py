"""End-to-end pipeline orchestrator for the single-GPU profiles (docs/RUN_ON_5080.md).

Runs the whole study from a Profile (synscale.config.profiles), in dependency order,
with resume via per-job marker files under ``results/<profile>/state/``. Stages:

    prepare_data -> build_pool -> generate(teacher) -> [per branch:] build_dataset+branch
    -> pretrain_base(student, seed) -> train(branch) -> evaluate -> build_index -> analyze

Heavy GPU jobs (generation, base pretraining, branch training, evaluation) are, by
default on the real profiles, launched as isolated subprocesses so each frees VRAM on a
16 GB card. The `micro`/`smoke` profiles run in-process on CPU with a MockBackend and a
byte tokenizer, which exercises every stage for validation.

This module is intentionally dependency-light at import time; torch/vllm/lm-eval are only
touched inside the stages that need them.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from synscale.config.loader import load_experiment
from synscale.config.profiles import Profile, TeacherSpec, get_profile

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_ARMS = {
    "base_only": ("C0", "base.seen-shards"),
    "matched_real": ("C1", "real.fineweb-edu-heldout"),
    "human_instruct": ("C1b", "human.stackexchange-qa-v1"),
}


# --------------------------------------------------------------------------------------
# Tokenizer selection
# --------------------------------------------------------------------------------------
def get_tokenizer(profile: Profile, data_dir: Path):
    """Return (tokenizer, token_counter, bos_id, eos_id). Byte tokenizer for micro/smoke."""
    from synscale.generation.build_dataset import ByteTokenizer
    if profile.name in ("micro", "smoke"):
        bt = ByteTokenizer()
        return bt, bt, 256, 257
    # Real profiles: pinned Llama-2 32k tokenizer prepared by prepare_data.
    from synscale.generation.tokens import HFTokenizerCounter
    tok_path = data_dir / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer not found at {tok_path}; run prepare_data first")

    class _HF:
        name = "llama2-sp-32k"
        def __init__(self, p):
            from tokenizers import Tokenizer
            self._t = Tokenizer.from_file(str(p))
            import hashlib
            self.sha256 = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        def encode(self, text: str):
            return self._t.encode(text, add_special_tokens=False).ids
        def count(self, text: str):
            return len(self.encode(text))

    hf = _HF(tok_path)
    # BOS/EOS: Llama-2 SP ids 1/2.
    return hf, hf, 1, 2


# --------------------------------------------------------------------------------------
# State / resume
# --------------------------------------------------------------------------------------
class State:
    def __init__(self, root: Path):
        self.dir = root / "state"
        self.dir.mkdir(parents=True, exist_ok=True)

    def done(self, key: str) -> bool:
        return (self.dir / f"{key}.done").exists()

    def mark(self, key: str, info: Optional[dict] = None) -> None:
        (self.dir / f"{key}.done").write_text(json.dumps(info or {"t": time.time()}))


# --------------------------------------------------------------------------------------
# Experiment-config construction (branch / base run) from a Profile
# --------------------------------------------------------------------------------------
def _student_frag(student: str) -> str:
    return f"configs/students/{student}.yaml"

def _training_frag(student: str) -> str:
    return f"configs/training/train_{student}.yaml"

_EVAL_FRAG = "configs/evaluation/primary_v1.yaml"
_GEN_FRAG = "configs/generation/main_v1.yaml"


def make_experiment_yaml(profile: Profile, *, student: str, arm: str, seed_base: int, seed_phase: int,
                         teacher: Optional[TeacherSpec] = None, d_syn: int, variant: str,
                         out_dir: Path) -> Path:
    """Write a loader-compatible experiment YAML and return its path."""
    import yaml
    spec: dict[str, Any] = {
        "study": profile.name if profile.name in ("pilot", "main") else "main",
        "arm": arm,
        "variant": variant,
        "student": _student_frag(student),
        "training": _training_frag(student),
        "evaluation": _EVAL_FRAG,
    }
    ov: dict[str, Any] = {"training.seed_base": seed_base, "training.seed_phase": seed_phase}
    if arm == "synthetic":
        assert teacher is not None
        spec["teacher"] = teacher.config_path
        spec["generation"] = _GEN_FRAG
        spec["dataset_ref"] = f"data/processed/{profile.name}/treat/{teacher.name}.d{d_syn}.{variant}.s{seed_phase}"
    is_base_run = (arm == "base_only")
    if is_base_run:
        ov["training.synthetic_phase"] = None
    else:
        # phase-2 tokens follow the profile (d2), decay over the whole phase
        ov["training.synthetic_phase.tokens"] = profile.d2_tokens
        ov["training.synthetic_phase.decay_tokens"] = profile.d2_tokens
        ov["training.synthetic_phase.replay_fraction"] = profile.replay_fraction
        if arm in CONTROL_ARMS:
            spec["dataset_ref"] = f"data/processed/{profile.name}/treat/{CONTROL_ARMS[arm][1]}.d{d_syn}.s{seed_phase}"
    # base tokens follow the profile
    ov["training.base_phase.tokens"] = profile.base_tokens[student]
    spec["overrides"] = ov
    spec["notes"] = f"profile={profile.name}; arm={arm}; teacher={teacher.name if teacher else '-'}; d_syn={d_syn}"

    out_dir.mkdir(parents=True, exist_ok=True)
    # Load once to get the deterministic experiment id for the filename.
    tmp = out_dir / "_tmp.yaml"
    tmp.write_text(yaml.safe_dump(spec, sort_keys=False))
    cfg = load_experiment(tmp, repo_root=REPO_ROOT)
    from synscale.tracking.ids import experiment_id
    eid = experiment_id(cfg)
    path = out_dir / f"{eid}.yaml"
    tmp.rename(path)
    return path


@dataclass
class BranchJob:
    student: str
    arm: str
    teacher: Optional[TeacherSpec]
    seed_base: int
    seed_phase: int
    d_syn: int
    variant: str
    cond: str


def enumerate_branches(profile: Profile) -> list[BranchJob]:
    jobs: list[BranchJob] = []
    for s in profile.students:
        # teachers (synthetic), tier-1 seeds
        for t in profile.teachers:
            seeds = list(range(1, profile.phase_seeds + 1))
            if t.name in profile.tier2_teachers:
                seeds += list(range(profile.phase_seeds + 1, profile.phase_seeds + 1 + profile.tier2_seeds))
            for sp in seeds:
                jobs.append(BranchJob(s, "synthetic", t, 1, sp, profile.d_syn, "eqtok", t.name))
        # controls
        for arm in profile.controls:
            if arm == "base_only":
                continue  # C0 handled by the base run (evaluated as theta*)
            n = profile.c1_seeds if arm == "matched_real" else profile.phase_seeds
            for sp in range(1, n + 1):
                jobs.append(BranchJob(s, arm, None, 1, sp, profile.d_syn, "eqtok", CONTROL_ARMS[arm][0]))
        # dsweep
        if profile.dsweep_student == s:
            for t in profile.teachers:
                if t.name not in profile.dsweep_teachers:
                    continue
                for d in profile.dsweep_levels:
                    if d == profile.d_syn:
                        continue
                    for sp in range(1, profile.dsweep_seeds + 1):
                        jobs.append(BranchJob(s, "synthetic", t, 1, sp, d, f"eqtok-d{d}", t.name))
    return jobs


# ======================================================================================
# Stage implementations
# ======================================================================================
def _write_docstream(texts, tokenizer, bos_id, eos_id, stem: Path) -> dict:
    from synscale.training.data import tokenize_documents, write_document_file
    tokens, starts = tokenize_documents(texts, tokenizer, bos_id, eos_id)
    return write_document_file(stem, tokens, starts, extra={"n_docs": int(starts.size), "n_tokens": int(tokens.size)})


def _synth_corpus(n_docs: int, words_per_doc: int, seed: int) -> list[str]:
    import random
    rng = random.Random(seed)
    topics = ["science", "history", "math", "biology", "physics", "economics", "geography", "grammar"]
    out = []
    for _ in range(n_docs):
        t = rng.choice(topics)
        out.append(f"This document about {t}. " + " ".join(
            f"{t} idea number {rng.randrange(1000)} explains concept {rng.randrange(1000)}." for _ in range(words_per_doc // 6)))
    return out


def prepare_data(profile: Profile, data_dir: Path, tokenizer, bos_id, eos_id, *, synthetic: bool = False) -> dict:
    """Base corpus shards (train + fresh held-out C1), C1b StackExchange stream, NLL sets.

    In `synthetic` mode (micro/smoke) all corpora are generated locally; otherwise the caller
    must have prepared real FineWeb-Edu / StackExchange streams (scripts/prepare_data.py).
    """
    proc = data_dir / "processed" / profile.name
    proc.mkdir(parents=True, exist_ok=True)
    max_base = max(profile.base_tokens.values())
    out = {}
    if synthetic:
        # size the synthetic corpus generously above the largest base budget (bytes ~ tokens here)
        n_docs = max(200, int(max_base / 40) + profile.d2_tokens // 40 + 2000)
        _write_docstream(_synth_corpus(n_docs, 60, 1), tokenizer, bos_id, eos_id, proc / "base_train")
        _write_docstream(_synth_corpus(n_docs // 3 + 200, 60, 2), tokenizer, bos_id, eos_id, proc / "base_heldout")
        _write_docstream(_synth_corpus(n_docs // 3 + 200, 50, 3), tokenizer, bos_id, eos_id, proc / "c1b_stackexchange")
        _write_docstream(_synth_corpus(300, 40, 4), tokenizer, bos_id, eos_id, proc / "nll_instr")
        out = {k: str(proc / k) for k in ("base_train", "base_heldout", "c1b_stackexchange", "nll_instr")}
    else:  # real data must already be prepared by scripts/prepare_data.py
        for k in ("base_train", "base_heldout", "c1b_stackexchange", "nll_instr"):
            stem = proc / k
            if not (stem.parent / f"{k}.tokens.npy").exists() and not Path(f"{stem}.tokens.npy").exists():
                raise FileNotFoundError(f"expected prepared corpus at {stem}.*; run scripts/prepare_data.py")
            out[k] = str(stem)
    return out


def build_pool_stage(profile: Profile, data_dir: Path) -> str:
    from synscale.generation import prompt_pool as pp
    from synscale.generation.taxonomy import Taxonomy
    pool_path = data_dir / "prompts" / f"pool-{profile.name}.jsonl"
    if pool_path.exists():
        return str(pool_path)
    tax = Taxonomy.from_yaml(REPO_ROOT / "data" / "prompts" / "taxonomy_v1.yaml")
    gens = {pp.SOURCE_TEMPLATED: pp.TemplatedPromptGenerator(tax)}
    spec = pp.PoolBuildSpec(n_prompts=profile.pool_prompts, seed=0, version=f"{profile.name}-v1.0",
                            source_mix={c: {pp.SOURCE_TEMPLATED: 1.0} for c in pp.CATEGORIES})
    res = pp.PoolBuilder(spec, gens, taxonomy=tax).build()
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pp.freeze_pool(res.records, pool_path, manifest=res.manifest)
    return str(pool_path)


def generate_stage(profile: Profile, teacher: TeacherSpec, pool_path: str, counter, data_dir: Path, *,
                   backend_kind: str, device: str, sampling) -> dict:
    from synscale.generation.backend import build_backend
    from synscale.generation.runner import run_generation
    out_dir = data_dir / "processed" / profile.name / "gen" / teacher.name
    if (out_dir / "manifest.json").exists():
        return json.loads((out_dir / "manifest.json").read_text())
    if backend_kind == "mock":
        # skill scales with log teacher size so the mock has a monotone-ish size signal
        skill = 0.4 + 0.5 * (teacher.nominal_params ** 0.15) / (1e9 ** 0.15)
        backend = build_backend("mock", skill=skill, verbosity=45)
    else:
        backend = build_backend("vllm", hf_repo=teacher.hf_repo, quantization=teacher.quantization,
                                tensor_parallel=teacher.tensor_parallel, max_model_len=teacher.max_model_len,
                                seed=sampling.seed)
    pool_version = f"{profile.name}-v1.0"
    return run_generation(pool_path, backend, sampling, counter, out_dir, teacher_id=teacher.name,
                          pool_version=pool_version, target_student_tokens=profile.d_syn,
                          overgeneration_factor=1.25, shard_size=(200 if profile.name in ("micro","smoke") else 5000),
                          system_prompt=None, manifest_extra={"backend_kind": backend_kind})


def _load_model_from_ckpt(ckpt_path: Path, student_cfg):
    import torch
    from synscale.models.llama_like import LlamaLike
    from synscale.training.trainer import load_checkpoint
    model = LlamaLike(student_cfg)
    load_checkpoint(ckpt_path, model, optimizer=None, restore_rng=False)
    return model


def evaluate_stage(cfg, ckpt_path: Path, tokenizer, bos_id, *, cond: str, results_dir: Path,
                   use_fixtures: bool, device: str, tasks, nll_stems: dict, limit=None) -> dict:
    from synscale.evaluation.local_eval import evaluate_checkpoint
    from synscale.evaluation.benchmarks import get_loader
    from synscale.tracking.ids import experiment_id
    eid = experiment_id(cfg)
    eval_dir = results_dir / eid / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if (eval_dir / "metrics.json").exists():
        return json.loads((eval_dir / "metrics.json").read_text())
    model = _load_model_from_ckpt(ckpt_path, cfg.student)
    model.to(device)
    loader = get_loader(use_fixtures)
    res = evaluate_checkpoint(model, tokenizer, tasks=tasks, loader=loader, bos_id=bos_id,
                              device=device, ctx_len=cfg.student.context_length, limit=limit)
    res["experiment_id"] = eid
    res["cond"] = cond
    res["student"] = cfg.student.name
    res["s_params_nominal"] = cfg.student.target_params
    res["arm"] = cfg.arm
    res["seed_phase"] = cfg.training.seed_phase
    res["teacher"] = cfg.teacher.name if cfg.teacher else None
    res["teacher_params"] = cfg.teacher.nominal_params if cfg.teacher else None
    (eval_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def pretrain_base_stage(profile: Profile, student: str, seed_base: int, corpora: dict, *,
                        exp_dir: Path, results_dir: Path, store_dir: Path, device: str,
                        micro_batch_seqs: int, max_steps=None, peak_tflops=None) -> tuple[Any, Path]:
    from synscale.training.trainer import RunInputs, RuntimeOptions, run_phase
    from synscale.tracking.ids import experiment_id
    yaml_path = make_experiment_yaml(profile, student=student, arm="base_only", seed_base=seed_base,
                                     seed_phase=1, d_syn=0, variant="-", out_dir=exp_dir)
    cfg = load_experiment(yaml_path, repo_root=REPO_ROOT)
    ckpt = store_dir / f"base_{student}_b{seed_base}.pt"
    if not ckpt.exists():
        inputs = RunInputs(data=[corpora["base_train"]], results_dir=results_dir, store_dir=store_dir,
                           out_checkpoint=ckpt, eval_sets={"base_heldout": corpora["base_heldout"]})
        opts = RuntimeOptions(micro_batch_seqs=micro_batch_seqs, max_steps=max_steps, peak_tflops=peak_tflops)
        opts.device = device
        run_phase(cfg, "base", inputs, opts, repo_root=str(REPO_ROOT))
    return cfg, ckpt


def branch_stage(profile: Profile, job: BranchJob, *, corpora: dict, gen_dirs: dict, union_drop: set,
                 tokenizer, counter, bos_id, eos_id, base_ckpt: Path, exp_dir: Path, results_dir: Path,
                 store_dir: Path, proc_dir: Path, device: str, micro_batch_seqs: int,
                 max_steps=None, peak_tflops=None) -> tuple[Any, Path, str]:
    from synscale.training.trainer import RunInputs, RuntimeOptions, run_phase
    from synscale.training.data import BranchSpec, SequenceSource, ShardedSource, build_branch, read_meta
    from synscale.generation.build_dataset import build_treatment_stream
    seq_len = load_student_ctx(job.student)

    # 1. treatment document stream
    treat_dir = proc_dir / "treat"; treat_dir.mkdir(parents=True, exist_ok=True)
    if job.arm == "synthetic":
        treat_stem = treat_dir / f"{job.teacher.name}.d{job.d_syn}.{job.variant}.s{job.seed_phase}"
        if not Path(f"{treat_stem}.tokens.npy").exists():
            build_treatment_stream(gen_dirs[job.teacher.name], treat_stem, tokenizer, bos_id, eos_id,
                                   d_syn=job.d_syn, seed=job.seed_phase, variant=job.variant, drop_ids=union_drop)
    else:
        src = corpora["base_heldout"] if job.arm == "matched_real" else corpora["c1b_stackexchange"]
        treat_stem = Path(src)

    # 2. pack the branch (treatment + replay), sizing d2 so there is never a shortfall
    branch_dir = proc_dir / "branches"; branch_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = make_experiment_yaml(profile, student=job.student, arm=job.arm, seed_base=job.seed_base,
                                     seed_phase=job.seed_phase, teacher=job.teacher, d_syn=job.d_syn,
                                     variant=job.variant, out_dir=exp_dir)
    cfg = load_experiment(yaml_path, repo_root=REPO_ROOT)
    from synscale.tracking.ids import experiment_id
    eid = experiment_id(cfg)
    branch_stem = branch_dir / eid
    treatment = SequenceSource.open(str(treat_stem), seq_len)
    replay = ShardedSource([SequenceSource.open(corpora["base_train"], seq_len)])
    treat_tokens = treatment.n_sequences * seq_len if hasattr(treatment, "n_sequences") else None
    d2 = profile.d2_tokens
    # shrink d2 to available treatment if needed (micro/edge); keeps replay_fraction
    avail_treat = getattr(treatment, "n_sequences", None)
    if avail_treat is not None:
        max_d2 = int(avail_treat * seq_len / (1 - profile.replay_fraction))
        d2 = min(d2, max_d2)
    d2 = (d2 // seq_len) * seq_len  # BranchSpec requires a multiple of seq_len
    spec = BranchSpec(size=job.student, seed=job.seed_phase, seq_len=seq_len,
                      d2_tokens=d2, replay_fraction=profile.replay_fraction)
    if not Path(f"{branch_stem}.packed.npy").exists() and not Path(f"{branch_stem}.tokens.npy").exists():
        build_branch(treatment, replay, spec, str(branch_stem), treatment_ref=str(treat_stem),
                     replay_ref=corpora["base_train"])

    # 3. train the synthetic phase from theta*
    out_ckpt = store_dir / f"{eid}.pt"
    if not out_ckpt.exists():
        inputs = RunInputs(data=[str(branch_stem)], results_dir=results_dir, store_dir=store_dir,
                           base_checkpoint=base_ckpt, out_checkpoint=out_ckpt,
                           eval_sets={"base_heldout": corpora["base_heldout"], "instr": corpora["nll_instr"]})
        opts = RuntimeOptions(micro_batch_seqs=micro_batch_seqs, max_steps=max_steps, peak_tflops=peak_tflops)
        opts.device = device
        run_phase(cfg, "synthetic", inputs, opts, repo_root=str(REPO_ROOT))
    return cfg, out_ckpt, job.cond


def load_student_ctx(student: str) -> int:
    import yaml
    d = yaml.safe_load((REPO_ROOT / _student_frag(student)).read_text())
    return int(d["context_length"])


# ======================================================================================
# Top-level orchestration
# ======================================================================================
def run_profile(profile_name: str, *, data_dir: Optional[Path] = None, results_dir: Optional[Path] = None,
                store_dir: Optional[Path] = None, backend_kind: str = "auto", device: str = "auto",
                micro_batch_seqs: Optional[int] = None, max_steps: Optional[int] = None,
                eval_limit: Optional[int] = None, stop_after: Optional[str] = None) -> dict:
    profile = get_profile(profile_name)
    data_dir = Path(data_dir or "data")
    results_dir = Path(results_dir or f"results/{profile.name}")
    store_dir = Path(store_dir or os.environ.get("SYNSCALE_STORE", f"checkpoints/{profile.name}"))
    for p in (results_dir, store_dir):
        p.mkdir(parents=True, exist_ok=True)
    is_toy = profile.name in ("micro", "smoke")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if backend_kind == "auto":
        backend_kind = "mock" if profile.name == "micro" else "vllm"
    if micro_batch_seqs is None:
        micro_batch_seqs = 8 if is_toy else 16
    state = State(results_dir)
    exp_dir = REPO_ROOT / "configs" / "experiments" / profile.name
    proc_dir = data_dir / "processed" / profile.name

    from synscale.generation.backend import SamplingSpec
    from synscale.config.loader import load_experiment as _le
    gen_cfg = _le  # noqa
    # decoding from the generation fragment
    import yaml as _yaml
    gc = _yaml.safe_load((REPO_ROOT / _GEN_FRAG).read_text())
    sampling = SamplingSpec(temperature=gc["temperature"], top_p=gc["top_p"], top_k=gc.get("top_k", -1),
                            repetition_penalty=gc.get("repetition_penalty", 1.0),
                            max_new_tokens=(256 if is_toy else gc["max_new_tokens"]),
                            n=gc.get("samples_per_prompt", 1), seed=gc.get("sampling_seed", 0),
                            stop_token_ids=tuple(gc.get("stop_token_ids", [])))

    tokenizer, counter, bos_id, eos_id = get_tokenizer(profile, data_dir)

    log = {"profile": profile.name, "device": device, "backend": backend_kind, "stages": []}

    # 1. data
    corpora = prepare_data(profile, data_dir, tokenizer, bos_id, eos_id, synthetic=is_toy)
    log["stages"].append("prepare_data"); state.mark("prepare_data")
    if stop_after == "prepare_data":
        return log

    # 2. pool
    pool_path = build_pool_stage(profile, data_dir)
    log["stages"].append("build_pool")
    if stop_after == "build_pool":
        return log

    # 3. generation (per teacher)
    gen_dirs = {}
    for t in profile.teachers:
        man = generate_stage(profile, t, pool_path, counter, data_dir, backend_kind=backend_kind,
                             device=device, sampling=sampling)
        gen_dirs[t.name] = str(data_dir / "processed" / profile.name / "gen" / t.name)
    log["stages"].append("generate"); log["teachers"] = list(gen_dirs)
    if stop_after == "generate":
        return log
    from synscale.generation.build_dataset import union_dropped_prompt_ids
    union_drop = union_dropped_prompt_ids(gen_dirs)

    # 4. base pretraining (per student, per base seed) + evaluate theta* as C0
    peak = _peak_tflops(device)
    base_ckpts: dict[str, Path] = {}
    tasks = _primary_tasks()
    for s in profile.students:
        for b in range(1, profile.base_seeds[s] + 1):
            cfg, ckpt = pretrain_base_stage(profile, s, b, corpora, exp_dir=exp_dir, results_dir=results_dir,
                                            store_dir=store_dir, device=device, micro_batch_seqs=micro_batch_seqs,
                                            max_steps=max_steps, peak_tflops=peak)
            if b == 1:
                base_ckpts[s] = ckpt
            evaluate_stage(cfg, ckpt, tokenizer, bos_id, cond="C0", results_dir=results_dir,
                           use_fixtures=is_toy, device=device, tasks=tasks,
                           nll_stems={"instr": corpora["nll_instr"]}, limit=eval_limit)
    log["stages"].append("pretrain_base")
    if stop_after == "pretrain_base":
        return log

    # 5. branches: build + train + evaluate
    jobs = enumerate_branches(profile)
    for i, job in enumerate(jobs):
        cfg, ckpt, cond = branch_stage(profile, job, corpora=corpora, gen_dirs=gen_dirs, union_drop=union_drop,
                                       tokenizer=tokenizer, counter=counter, bos_id=bos_id, eos_id=eos_id,
                                       base_ckpt=base_ckpts[job.student], exp_dir=exp_dir, results_dir=results_dir,
                                       store_dir=store_dir, proc_dir=proc_dir, device=device,
                                       micro_batch_seqs=micro_batch_seqs, max_steps=max_steps, peak_tflops=peak)
        evaluate_stage(cfg, ckpt, tokenizer, bos_id, cond=cond, results_dir=results_dir,
                       use_fixtures=is_toy, device=device, tasks=tasks,
                       nll_stems={"instr": corpora["nll_instr"]}, limit=eval_limit)
        # free VRAM between jobs
        try:
            import torch, gc as _gc
            _gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
        except Exception:
            pass
    log["stages"].append("branches"); log["n_branches"] = len(jobs)
    if stop_after == "branches":
        return log

    # 6. index + analyze
    from synscale.tracking.index import build_index
    idx_path = build_index(results_dir, out_path=results_dir / "index.csv")
    log["index"] = str(idx_path)
    try:
        from synscale.analysis.run import analyze_index
        analyze_index(idx_path, out_dir=results_dir / "analysis")
        log["stages"].append("analyze")
    except Exception as e:  # analysis is best-effort; never blocks the pipeline
        log["analyze_error"] = str(e)
    return log


def _peak_tflops(device: str):
    if device != "cuda":
        return None
    try:
        import torch
        from synscale.models.param_count import peak_tflops_for_device
        return peak_tflops_for_device(torch.cuda.get_device_name(0))
    except Exception:
        return None


def _primary_tasks():
    import yaml
    d = yaml.safe_load((REPO_ROOT / _EVAL_FRAG).read_text())
    return list(d.get("primary_tasks", ["sciq", "arc_easy", "piqa", "lambada_openai"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the whole single-GPU study for a profile.")
    ap.add_argument("--profile", default="local5080")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "vllm"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--micro-batch-seqs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--stop-after", default=None)
    a = ap.parse_args()
    out = run_profile(a.profile, data_dir=a.data_dir, results_dir=a.results_dir, store_dir=a.store_dir,
                      backend_kind=a.backend, device=a.device, micro_batch_seqs=a.micro_batch_seqs,
                      max_steps=a.max_steps, eval_limit=a.eval_limit, stop_after=a.stop_after)
    print(json.dumps(out, indent=2, default=str))
