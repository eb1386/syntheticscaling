"""Pinned lm-eval harness runner, result parser and per-run eval writer.

Implements benchmarks.md §3.1–3.2 exactly:

* primary suite (zero-shot, no chat template, ``--seed 1234``, ``--batch_size 64``, bf16,
  ``--log_samples``): sciq (acc), arc_easy (acc_norm), piqa (acc_norm), lambada_openai
  (acc + perplexity), openbookqa (acc_norm), social_iqa (acc), hellaswag (acc_norm);
* secondary log-likelihood block, the 5-shot ICL probe, and the generative block
  (gsm8k 5-shot, ifeval with ``max_gen_toks=512``, triviaqa ``--limit 3000``).

The harness is invoked through its CLI (so the archived command is byte-for-byte the one in
the protocol document) and its output files are parsed into one JSON per task under
``results/<experiment_id>/eval/<task>.json`` that carries the aggregate metrics *and* the
per-item log-likelihoods (needed for task loss, paired item bootstraps and the contamination
sensitivity analysis).  ``lm_eval`` itself is imported lazily and only by the real runner;
``MockRunner`` produces files with the same layout for tests and dry runs.
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np

PINNED_HARNESS_VERSION = "0.4.13"
EVAL_SEED = 1234
BATCH_SIZE = 64
DTYPE = "bfloat16"
CUSTOM_TASK_DIR = Path(__file__).resolve().parent / "tasks"


# ----------------------------------------------------------------------------------------
# Task specifications
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """One harness task with its pre-registered metric and invocation settings."""

    name: str
    metric: str  # pre-registered metric used in the primary/secondary analysis
    output_type: str = "multiple_choice"  # multiple_choice | loglikelihood | generate_until
    num_fewshot: int = 0
    extra_metrics: tuple[str, ...] = ()  # logged but never used for selection
    gen_kwargs: Optional[dict[str, Any]] = None
    limit: Optional[int] = None
    tier: str = "secondary"  # primary_a | primary_b | secondary | icl | generative | report
    gated: bool = False  # only run/report if a pre-registered gate passes (gsm8k: 1B only)


PRIMARY_SUITE: tuple[TaskSpec, ...] = (
    TaskSpec("sciq", "acc", tier="primary_a", extra_metrics=("acc_norm",)),
    TaskSpec("arc_easy", "acc_norm", tier="primary_a", extra_metrics=("acc",)),
    TaskSpec("piqa", "acc_norm", tier="primary_a", extra_metrics=("acc",)),
    TaskSpec("lambada_openai", "acc", output_type="loglikelihood", tier="primary_a",
             extra_metrics=("perplexity",)),
    TaskSpec("openbookqa", "acc_norm", tier="primary_b", extra_metrics=("acc",)),
    TaskSpec("social_iqa", "acc", tier="primary_b", extra_metrics=("acc_norm",)),
    TaskSpec("hellaswag", "acc_norm", tier="primary_b", extra_metrics=("acc",)),
)

SECONDARY_LL_SUITE: tuple[TaskSpec, ...] = (
    TaskSpec("arc_challenge", "acc_norm", extra_metrics=("acc",)),
    TaskSpec("winogrande", "acc"),
    TaskSpec("boolq", "acc"),
    TaskSpec("mmlu_stem_continuation", "acc_norm", extra_metrics=("acc",)),
    TaskSpec("mmlu_continuation", "acc_norm", extra_metrics=("acc",)),
    TaskSpec("commonsense_qa_cloze", "acc_norm", extra_metrics=("acc",)),
    TaskSpec("blimp", "acc"),
    TaskSpec("truthfulqa_mc2", "acc", tier="report"),
)

ICL_PROBE_SUITE: tuple[TaskSpec, ...] = (
    TaskSpec("arc_easy", "acc_norm", num_fewshot=5, tier="icl", extra_metrics=("acc",)),
    TaskSpec("sciq", "acc", num_fewshot=5, tier="icl", extra_metrics=("acc_norm",)),
)

GENERATIVE_SUITE: tuple[TaskSpec, ...] = (
    # greedy (harness default do_sample=false), until ["Question:", "</s>", "<|im_end|>"], 256 tokens
    TaskSpec("gsm8k", "exact_match,flexible-extract", output_type="generate_until", num_fewshot=5,
             extra_metrics=("exact_match,strict-match",), tier="generative", gated=True),
    TaskSpec("ifeval", "prompt_level_loose_acc", output_type="generate_until",
             extra_metrics=("prompt_level_strict_acc", "inst_level_strict_acc", "inst_level_loose_acc"),
             gen_kwargs={"max_gen_toks": 512}, tier="report"),
    TaskSpec("triviaqa", "exact_match", output_type="generate_until", limit=3000, tier="generative"),
)

SUITES: dict[str, tuple[TaskSpec, ...]] = {
    "primary": PRIMARY_SUITE,
    "secondary_ll": SECONDARY_LL_SUITE,
    "icl": ICL_PROBE_SUITE,
    "generative": GENERATIVE_SUITE,
}


def task_key(spec: TaskSpec) -> str:
    """Name under which a task's JSON is stored (few-shot probes get a suffix)."""
    return spec.name if spec.num_fewshot == 0 else f"{spec.name}_{spec.num_fewshot}shot"


# ----------------------------------------------------------------------------------------
# Command construction
# ----------------------------------------------------------------------------------------
def build_lm_eval_command(
    checkpoint: str,
    tokenizer: str,
    tasks: Iterable[TaskSpec],
    output_path: str | Path,
    *,
    num_fewshot: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    seed: int = EVAL_SEED,
    dtype: str = DTYPE,
    include_path: str | Path | None = CUSTOM_TASK_DIR,
    device: Optional[str] = None,
    gen_kwargs: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Return the ``lm_eval`` argv for one block of tasks (benchmarks.md §3.1–3.2).

    All tasks in one block must share ``num_fewshot``, ``gen_kwargs`` and ``limit``; the
    runner groups tasks accordingly.  No chat template is ever applied.
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError("no tasks")
    shots = {t.num_fewshot for t in tasks}
    if len(shots) != 1:
        raise ValueError("tasks in one block must share num_fewshot")
    nf = shots.pop() if num_fewshot is None else num_fewshot
    model_args = f"pretrained={checkpoint},dtype={dtype},tokenizer={tokenizer}"
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(t.name for t in tasks),
        "--num_fewshot", str(nf),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--log_samples",
        "--output_path", str(output_path),
    ]
    if include_path is not None:
        cmd += ["--include_path", str(include_path)]
    if device:
        cmd += ["--device", device]
    if gen_kwargs:
        cmd += ["--gen_kwargs", ",".join(f"{k}={v}" for k, v in gen_kwargs.items())]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def group_blocks(tasks: Iterable[TaskSpec]) -> list[list[TaskSpec]]:
    """Split tasks into CLI blocks with identical (num_fewshot, gen_kwargs, limit)."""
    blocks: dict[tuple, list[TaskSpec]] = {}
    for t in tasks:
        key = (t.num_fewshot, json.dumps(t.gen_kwargs, sort_keys=True), t.limit)
        blocks.setdefault(key, []).append(t)
    return list(blocks.values())


# ----------------------------------------------------------------------------------------
# Parsing harness outputs
# ----------------------------------------------------------------------------------------
def _latest(paths: list[str]) -> Optional[str]:
    return max(paths, key=os.path.getmtime) if paths else None


def find_harness_outputs(output_path: str | Path) -> tuple[Optional[Path], dict[str, Path]]:
    """Locate ``results_*.json`` and ``samples_<task>_*.jsonl`` under an output path.

    The harness nests them under ``<output_path>/<model_name_sanitized>/``; we search
    recursively and take the newest file per task.
    """
    root = Path(output_path)
    res = _latest(glob.glob(str(root / "**" / "results_*.json"), recursive=True))
    samples: dict[str, Path] = {}
    for p in glob.glob(str(root / "**" / "samples_*.jsonl"), recursive=True):
        stem = Path(p).name[len("samples_"):-len(".jsonl")]
        # samples_<task>_<date_id>.jsonl, date_id = YYYY-MM-DDTHH-MM-SS.ffffff
        task = stem.rsplit("_", 1)[0]
        if task not in samples or os.path.getmtime(p) > os.path.getmtime(samples[task]):
            samples[task] = Path(p)
    return (Path(res) if res else None), samples


def parse_results_json(path: str | Path) -> dict[str, Any]:
    """Return ``{task: {metric: value, ...}}`` from a harness ``results_*.json``.

    Metric keys of the form ``"acc,none"`` are flattened to ``"acc"``; filtered generative
    metrics keep their filter suffix (``"exact_match,flexible-extract"``).
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, Any] = {"_config": data.get("config", {}), "_versions": data.get("versions", {}),
                           "_n_samples": data.get("n-samples", {}), "results": {}}
    for task, metrics in data.get("results", {}).items():
        flat: dict[str, Any] = {}
        for k, v in metrics.items():
            if k == "alias":
                continue
            if "," in k:
                metric, filt = k.split(",", 1)
                flat[metric if filt == "none" else f"{metric},{filt}"] = v
            else:
                flat[k] = v
        out["results"][task] = flat
    return out


def _first_float(x: Any) -> float:
    """Harness resps are ``[ll, is_greedy]`` (possibly nested once more); return ll."""
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def _is_greedy(x: Any) -> Optional[bool]:
    while isinstance(x, (list, tuple)) and len(x) and isinstance(x[0], (list, tuple)):
        x = x[0]
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return bool(x[1])
    return None


def parse_samples_jsonl(path: str | Path, output_type: str) -> list[dict[str, Any]]:
    """Per-item records from a ``--log_samples`` file.

    multiple_choice → ``{doc_id, gold, lls, is_greedy, choice_chars, choice_bytes, acc, acc_norm}``;
    loglikelihood → ``{doc_id, ll, is_greedy, target_bytes, acc, perplexity}``;
    generate_until → ``{doc_id, resp, <metrics...>}``.
    """
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            rec: dict[str, Any] = {"doc_id": s.get("doc_id")}
            metrics = s.get("metrics") or [k for k in ("acc", "acc_norm", "perplexity", "exact_match")
                                              if k in s]
            if output_type == "multiple_choice":
                resps = s.get("filtered_resps") or s.get("resps") or []
                lls = [_first_float(r) for r in resps]
                choices = _choices_from_arguments(s.get("arguments"), len(lls))
                gold = s.get("target")
                try:
                    gold = int(gold)
                except (TypeError, ValueError):
                    gold = choices.index(str(gold).strip()) if str(gold).strip() in choices else gold
                rec.update({
                    "gold": gold,
                    "lls": lls,
                    "is_greedy": [_is_greedy(r) for r in resps],
                    "choice_chars": [len(c) for c in choices],
                    "choice_bytes": [len(c.encode("utf-8")) for c in choices],
                })
            elif output_type == "loglikelihood":
                resps = s.get("filtered_resps") or s.get("resps") or []
                target = str(s.get("target", ""))
                rec.update({
                    "ll": _first_float(resps[0]) if resps else float("nan"),
                    "is_greedy": _is_greedy(resps[0]) if resps else None,
                    "target_bytes": len(target.encode("utf-8")),
                    "target_chars": len(target),
                })
            else:  # generate_until
                resps = s.get("filtered_resps") or s.get("resps") or []
                rec["resp"] = resps[0] if resps else ""
                while isinstance(rec["resp"], (list, tuple)):
                    rec["resp"] = rec["resp"][0] if rec["resp"] else ""
            for m in metrics:
                if m in s:
                    v = s[m]
                    rec[m] = float(v) if isinstance(v, (int, float, bool)) else v
            items.append(rec)
    return items


def _choices_from_arguments(arguments: Any, n: int) -> list[str]:
    """Recover continuation strings from the sanitized ``arguments`` field."""
    out: list[str] = []
    if isinstance(arguments, dict):
        for i in range(n):
            a = arguments.get(f"gen_args_{i}", {})
            out.append(str(a.get("arg_1", "")))
    elif isinstance(arguments, list):
        for a in arguments[:n]:
            out.append(str(a[1]) if isinstance(a, (list, tuple)) and len(a) > 1 else "")
    while len(out) < n:
        out.append("")
    return out


# ----------------------------------------------------------------------------------------
# Per-task JSON written under results/<experiment_id>/eval/
# ----------------------------------------------------------------------------------------
def task_loss_bpb(per_item: list[dict[str, Any]], output_type: str) -> tuple[float, float, int]:
    """Bits-per-byte of the correct continuation (statistics.md §1.4): mean, SE, n.

    For multiple_choice the gold option's log-likelihood is divided by its UTF-8 byte length;
    for loglikelihood tasks the target's.  Items with missing values are skipped.
    """
    vals: list[float] = []
    for it in per_item:
        if output_type == "multiple_choice":
            g = it.get("gold")
            lls, nb = it.get("lls") or [], it.get("choice_bytes") or []
            if isinstance(g, int) and 0 <= g < len(lls) and g < len(nb) and nb[g] > 0:
                vals.append(-lls[g] / (math.log(2) * nb[g]))
        elif output_type == "loglikelihood":
            nb = it.get("target_bytes", 0)
            if nb and it.get("ll") is not None and not math.isnan(it["ll"]):
                vals.append(-it["ll"] / (math.log(2) * nb))
    if not vals:
        return float("nan"), float("nan"), 0
    a = np.asarray(vals)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan"), len(a)


def assemble_task_result(
    spec: TaskSpec,
    metrics: dict[str, Any],
    per_item: list[dict[str, Any]],
    *,
    version: Any = None,
    harness_version: str = PINNED_HARNESS_VERSION,
    config: Optional[dict[str, Any]] = None,
    sources: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build the JSON document stored as ``eval/<task>.json``."""
    tl_mean, tl_se, tl_n = task_loss_bpb(per_item, spec.output_type)
    return {
        "task": spec.name,
        "task_key": task_key(spec),
        "version": version,
        "harness_version": harness_version,
        "output_type": spec.output_type,
        "num_fewshot": spec.num_fewshot,
        "primary_metric": spec.metric,
        "tier": spec.tier,
        "metrics": metrics,
        "n_items": len(per_item) if per_item else metrics.get("n_items"),
        "task_loss_bpb": tl_mean,
        "task_loss_bpb_se": tl_se,
        "task_loss_n": tl_n,
        "per_item": per_item,
        "lm_eval_config": config or {},
        "sources": sources or {},
    }


def write_task_result(results_dir: str | Path, experiment_id: str, doc: dict[str, Any]) -> Path:
    out = Path(results_dir) / experiment_id / "eval" / f"{doc['task_key']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return out


def load_task_result(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_block(
    specs: list[TaskSpec], output_path: str | Path, *, harness_version: str = PINNED_HARNESS_VERSION
) -> dict[str, dict[str, Any]]:
    """Parse one harness invocation's outputs into ``{task_key: task-result doc}``."""
    res_path, sample_paths = find_harness_outputs(output_path)
    if res_path is None:
        raise FileNotFoundError(f"no results_*.json under {output_path}")
    parsed = parse_results_json(res_path)
    docs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        metrics = parsed["results"].get(spec.name)
        if metrics is None:
            # group tasks (blimp, mmlu_*) report the aggregate under the group name
            metrics = {}
            for name, m in parsed["results"].items():
                if name == spec.name or name.startswith(spec.name):
                    metrics = m
                    break
        per_item: list[dict[str, Any]] = []
        sp = sample_paths.get(spec.name)
        if sp is not None:
            per_item = parse_samples_jsonl(sp, spec.output_type)
        docs[task_key(spec)] = assemble_task_result(
            spec, metrics or {}, per_item,
            version=parsed["_versions"].get(spec.name),
            harness_version=harness_version,
            config=parsed["_config"],
            sources={"results_json": str(res_path), "samples_jsonl": str(sp) if sp else ""},
        )
    return docs


# ----------------------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------------------
class Runner(Protocol):
    def run(self, checkpoint: str, tokenizer: str, specs: list[TaskSpec], output_path: Path,
            **kw: Any) -> dict[str, dict[str, Any]]: ...


def check_harness_version(expected: str = PINNED_HARNESS_VERSION) -> str:
    """Import lm_eval lazily and fail on a version mismatch (doc 18 §18.6)."""
    import lm_eval  # type: ignore  # lazy: only the evaluation machine has it

    found = getattr(lm_eval, "__version__", "unknown")
    if found != expected:
        raise RuntimeError(f"lm-eval version {found} != pinned {expected}")
    return found


@dataclass
class LMEvalRunner:
    """Runs the pinned harness CLI and parses its outputs."""

    harness_version: str = PINNED_HARNESS_VERSION
    device: Optional[str] = None
    batch_size: int = BATCH_SIZE
    seed: int = EVAL_SEED
    dtype: str = DTYPE
    include_path: Optional[Path] = CUSTOM_TASK_DIR
    env: dict[str, str] = field(default_factory=dict)
    check_version: bool = True

    def run(self, checkpoint: str, tokenizer: str, specs: list[TaskSpec], output_path: Path,
            **kw: Any) -> dict[str, dict[str, Any]]:
        if self.check_version:
            check_harness_version(self.harness_version)
        if shutil.which("lm_eval") is None:
            raise RuntimeError("lm_eval CLI not on PATH")
        docs: dict[str, dict[str, Any]] = {}
        for block in group_blocks(specs):
            b0 = block[0]
            block_dir = Path(output_path) / f"block_{b0.num_fewshot}shot_{'_'.join(t.name for t in block)}"
            block_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_lm_eval_command(
                checkpoint, tokenizer, block, block_dir, batch_size=self.batch_size, seed=self.seed,
                dtype=self.dtype, include_path=self.include_path, device=self.device,
                gen_kwargs=b0.gen_kwargs, limit=b0.limit,
            )
            (block_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            subprocess.run(cmd, check=True, env={**os.environ, **self.env})
            docs.update(collect_block(block, block_dir, harness_version=self.harness_version))
        return docs


@dataclass
class MockRunner:
    """Deterministic stand-in that writes harness-shaped files and parses them.

    ``accuracy`` maps task name → target accuracy of the fake model; per-item log-likelihoods
    are drawn so that the argmax matches the gold option with that probability.  Item counts
    are shrunk (``n_items``) to keep tests fast.
    """

    accuracy: dict[str, float] = field(default_factory=dict)
    n_items: int = 40
    seed: int = 0
    harness_version: str = PINNED_HARNESS_VERSION

    def _write_block(self, block: list[TaskSpec], block_dir: Path) -> None:
        rng = np.random.default_rng(self.seed + sum(map(ord, "".join(t.name for t in block))))
        model_dir = block_dir / "mock_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {"results": {}, "versions": {}, "n-samples": {},
                                   "config": {"model": "hf", "batch_size": BATCH_SIZE,
                                              "random_seed": EVAL_SEED, "numpy_seed": EVAL_SEED,
                                              "torch_seed": EVAL_SEED, "fewshot_seed": EVAL_SEED}}
        for spec in block:
            p = self.accuracy.get(spec.name, 0.5)
            samples: list[dict[str, Any]] = []
            accs, acc_norms, lls_all, ems = [], [], [], []
            for i in range(self.n_items):
                if spec.output_type == "multiple_choice":
                    k = 4
                    gold = int(rng.integers(k))
                    choices = [f"option {chr(97 + j)} text" + " x" * int(rng.integers(0, 4)) for j in range(k)]
                    lls = -rng.uniform(5, 30, size=k)
                    correct = rng.random() < p
                    if correct:
                        lls[gold] = lls.max() + 1.0
                    else:
                        lls[gold] = lls.min() - 1.0
                    acc = float(np.argmax(lls) == gold)
                    acc_norm = float(np.argmax(lls / np.array([len(c) for c in choices])) == gold)
                    accs.append(acc); acc_norms.append(acc_norm)
                    samples.append({
                        "doc_id": i, "doc": {}, "target": gold,
                        "arguments": {f"gen_args_{j}": {"arg_0": "Question: q\nAnswer:", "arg_1": " " + choices[j]}
                                      for j in range(k)},
                        "resps": [[[float(l), False]] for l in lls],
                        "filtered_resps": [[float(l), False] for l in lls],
                        "filter": "none", "metrics": ["acc", "acc_norm"],
                        "acc": acc, "acc_norm": acc_norm,
                    })
                elif spec.output_type == "loglikelihood":
                    target = " word" + "s" * int(rng.integers(0, 3))
                    correct = rng.random() < p
                    ll = -float(rng.uniform(0.5, 3.0)) if correct else -float(rng.uniform(4, 12))
                    lls_all.append(ll); accs.append(float(correct))
                    samples.append({
                        "doc_id": i, "doc": {}, "target": target,
                        "arguments": {"gen_args_0": {"arg_0": "some context", "arg_1": target}},
                        "resps": [[[ll, correct]]], "filtered_resps": [[ll, correct]],
                        "filter": "none", "metrics": ["perplexity", "acc"],
                        "perplexity": ll, "acc": float(correct),
                    })
                else:
                    correct = rng.random() < p
                    ems.append(float(correct))
                    samples.append({
                        "doc_id": i, "doc": {}, "target": "42",
                        "arguments": {"gen_args_0": {"arg_0": "Question: ...", "arg_1": {"until": ["\n"]}}},
                        "resps": [["42" if correct else "seven"]], "filtered_resps": ["42" if correct else "seven"],
                        "filter": "flexible-extract", "metrics": ["exact_match"], "exact_match": float(correct),
                    })
            n = self.n_items
            if spec.output_type == "multiple_choice":
                m = {"acc,none": float(np.mean(accs)), "acc_stderr,none": float(np.std(accs, ddof=1) / math.sqrt(n)),
                     "acc_norm,none": float(np.mean(acc_norms)),
                     "acc_norm_stderr,none": float(np.std(acc_norms, ddof=1) / math.sqrt(n)), "alias": spec.name}
            elif spec.output_type == "loglikelihood":
                m = {"perplexity,none": float(math.exp(-np.mean(lls_all))), "perplexity_stderr,none": 0.1,
                     "acc,none": float(np.mean(accs)), "acc_stderr,none": float(np.std(accs, ddof=1) / math.sqrt(n)),
                     "alias": spec.name}
            else:
                m = {"exact_match,flexible-extract": float(np.mean(ems)), "exact_match_stderr,flexible-extract": 0.01,
                     "exact_match,strict-match": 0.0, "exact_match_stderr,strict-match": 0.0, "alias": spec.name}
            results["results"][spec.name] = m
            results["versions"][spec.name] = 1.0
            results["n-samples"][spec.name] = {"original": n, "effective": n}
            with open(model_dir / f"samples_{spec.name}_2026-09-10T00-00-00.000000.jsonl", "w", encoding="utf-8") as fh:
                for s in samples:
                    fh.write(json.dumps(s) + "\n")
        (model_dir / "results_2026-09-10T00-00-00.000000.json").write_text(json.dumps(results, indent=1))

    def run(self, checkpoint: str, tokenizer: str, specs: list[TaskSpec], output_path: Path,
            **kw: Any) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for block in group_blocks(specs):
            b0 = block[0]
            block_dir = Path(output_path) / f"block_{b0.num_fewshot}shot_{'_'.join(t.name for t in block)}"
            cmd = build_lm_eval_command(checkpoint, tokenizer, block, block_dir, gen_kwargs=b0.gen_kwargs,
                                        limit=b0.limit)
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            self._write_block(block, block_dir)
            docs.update(collect_block(block, block_dir, harness_version=self.harness_version))
        return docs


def select_specs(suites: Iterable[str], *, student_is_1b: bool = False, gsm8k_gate_passed: bool = False
                 ) -> list[TaskSpec]:
    """Resolve suite names to task specs, applying the pre-registered gates."""
    specs: list[TaskSpec] = []
    for s in suites:
        if s == "all":
            for k in SUITES:
                specs += select_specs([k], student_is_1b=student_is_1b, gsm8k_gate_passed=gsm8k_gate_passed)
            continue
        if s not in SUITES:
            raise KeyError(f"unknown suite {s}; known: {list(SUITES)}")
        for t in SUITES[s]:
            if t.name == "gsm8k" and not (student_is_1b and gsm8k_gate_passed):
                continue
            if t.name == "triviaqa" and not student_is_1b:
                continue
            specs.append(t)
    return specs


def run_evaluation(
    experiment_id: str,
    checkpoint: str,
    tokenizer: str,
    results_dir: str | Path,
    suites: Iterable[str] = ("primary",),
    runner: Optional[Runner] = None,
    *,
    scratch_dir: str | Path | None = None,
    student_is_1b: bool = False,
    gsm8k_gate_passed: bool = False,
) -> dict[str, Path]:
    """Evaluate one checkpoint and write ``results/<experiment_id>/eval/<task>.json`` files."""
    runner = runner or LMEvalRunner()
    specs = select_specs(suites, student_is_1b=student_is_1b, gsm8k_gate_passed=gsm8k_gate_passed)
    scratch = Path(scratch_dir) if scratch_dir else Path(results_dir) / experiment_id / "eval" / "_harness"
    docs = runner.run(checkpoint, tokenizer, specs, scratch)
    written: dict[str, Path] = {}
    for key, doc in docs.items():
        written[key] = write_task_result(results_dir, experiment_id, doc)
    return written
