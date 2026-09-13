"""Student-referenced data-quality index q(T, S) (docs/24, the screening centerpiece).

For the central claim ("teacher identity is screened off by measurable data properties") we
need, for each teacher T and student S, a small set of properties measured on the teacher's
synthetic data:

  learnability  : mean per-token NLL of the teacher's responses under the STUDENT's base
                  checkpoint theta*(S). Low NLL = the data is compatible with what this student
                  can already model (a student-referenced difficulty / compatibility measure).
                  This is the only student-dependent property and the one prior work on
                  capacity gap (Xu 2024 CAR; Busbridge 2025) points to.
  diversity     : distinct-n and self-repetition on the responses (teacher-only).
  correctness   : fraction correct on the verifiable subset, from generation gold labels
                  (teacher-only).

These become covariates for the screening regression in synscale.analysis.novelty. The scalar
index q is the mean of their z-scores (sign-aligned so higher q = better data); the regression
also uses the raw components so no information is lost.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

_WORD = re.compile(r"\S+")
_FINAL = re.compile(r"final answer\s*:\s*(.+)", re.I)


def _responses(gen_dir: Path, limit: int) -> list[dict]:
    from synscale.generation.runner import iter_generated
    out = []
    for i, rec in enumerate(iter_generated(gen_dir, retained_only=True)):
        if i >= limit:
            break
        out.append(rec)
    return out


def distinct_n(texts: Iterable[str], n: int) -> float:
    grams, total = set(), 0
    for t in texts:
        w = _WORD.findall(t)
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n])); total += 1
    return len(grams) / total if total else 0.0


def diversity_metrics(gen_dir: str | Path, limit: int = 3000) -> dict[str, float]:
    recs = _responses(Path(gen_dir), limit)
    resp = [r["response"] for r in recs]
    return {
        "distinct_1": distinct_n(resp, 1), "distinct_2": distinct_n(resp, 2),
        "distinct_3": distinct_n(resp, 3),
        "mean_len": float(np.mean([len(_WORD.findall(t)) for t in resp])) if resp else 0.0,
        "n": len(resp),
    }


def correctness_rate(gen_dir: str | Path, limit: int = 20000) -> dict[str, float]:
    """Fraction correct on verifiable items (gold present); uses the MSR 'Final answer:' line."""
    recs = _responses(Path(gen_dir), limit)
    n = c = 0
    for r in recs:
        gold = r.get("gold")
        if gold is None or not r.get("verifiable"):
            continue
        n += 1
        m = _FINAL.search(r["response"] or "")
        pred = (m.group(1).strip() if m else "")
        if pred and _norm_ans(pred) == _norm_ans(str(gold)):
            c += 1
    return {"correctness": (c / n if n else float("nan")), "n_verifiable": n}


def _norm_ans(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z.\-]", "", s).lower().lstrip("0") or "0"


def learnability_nll(gen_dir: str | Path, model, tokenizer, bos_id: int, *, device="cpu",
                     limit: int = 500, ctx_len: int = 2048) -> float:
    """Mean per-token NLL of teacher responses under the student's base model theta*(S)."""
    import torch
    from synscale.evaluation.nll_eval import score_tokens
    recs = _responses(Path(gen_dir), limit)
    model.eval()
    def nll_fn(ids):
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        tgt = torch.tensor(ids[1:], device=device)
        return (-logp[range(len(tgt)), tgt]).cpu().numpy()
    vals = []
    for r in recs:
        ids = tokenizer.encode(r["response"])
        if len(ids) < 4:
            continue
        arr = score_tokens(ids, nll_fn, ctx_len, bos_id=bos_id)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            vals.append(float(np.mean(arr)))
    return float(np.mean(vals)) if vals else float("nan")


def quality_index(rows: list[dict]) -> list[dict]:
    """Attach a scalar q = mean of z-scores of (-learnability_nll, distinct_2, correctness).

    Sign-aligned so higher q = better data. Missing components are dropped from the mean.
    `rows` is a list of dicts each with keys student, teacher, and any of those metrics.
    """
    def z(vals):
        v = np.array([x if x is not None and x == x else np.nan for x in vals], float)
        m, sd = np.nanmean(v), np.nanstd(v)
        return (v - m) / sd if sd > 0 else np.zeros_like(v)
    learn = z([-(r.get("learnability_nll") or np.nan) for r in rows])   # negative: lower NLL is better
    div = z([r.get("distinct_2") for r in rows])
    corr = z([r.get("correctness") for r in rows])
    for i, r in enumerate(rows):
        comps = [c for c in (learn[i], div[i], corr[i]) if c == c]
        r["q"] = float(np.mean(comps)) if comps else float("nan")
    return rows
