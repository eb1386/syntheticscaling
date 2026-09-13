"""Deterministic subsampling of a filtered teacher pool to a fixed budget (doc 09 §3.3).

Primary control: equal student-tokenizer tokens, with per-category quotas computed on
TOTAL rendered tokens (prompt + markers + response) per doc 00 §0.7 Amendment A. The
priority order within a category is H64(seed, prompt_id), identical across teachers, so
teacher subsets overlap maximally (common random numbers) and nested-D subsets share
prompts. Also provides equal-examples and length-matched variants.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

TOLERANCE_TOKENS = 1024


def priority(seed: int, prompt_id: str) -> int:
    h = hashlib.blake2b(f"{seed}:{prompt_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


@dataclass
class Example:
    prompt_id: str
    category: str
    tokens_student: int          # rendered total (prompt + markers + response)
    response_len_student: int    # response only, for length-matching
    idx: int = 0                 # position in the JSONL (for retrieval)


def category_token_shares(examples: Sequence[Example]) -> dict[str, float]:
    """Realised per-category token shares, used to fix s_c^tok after the pilot."""
    tot: dict[str, int] = {}
    for e in examples:
        tot[e.category] = tot.get(e.category, 0) + e.tokens_student
    total = sum(tot.values()) or 1
    return {c: v / total for c, v in tot.items()}


def subsample_equal_tokens(examples: Sequence[Example], d_syn: int, *, seed: int,
                           category_shares: Optional[Mapping[str, float]] = None,
                           tolerance: int = TOLERANCE_TOKENS) -> list[Example]:
    """Select whole examples to hit d_syn rendered student tokens, stratified by category.

    If `category_shares` is given, each category c gets quota s_c * d_syn; otherwise the
    realised shares of `examples` are used (so the mix is preserved). Deterministic in
    (examples, d_syn, seed).
    """
    by_cat: dict[str, list[Example]] = {}
    for e in examples:
        by_cat.setdefault(e.category, []).append(e)
    shares = dict(category_shares) if category_shares else category_token_shares(examples)
    chosen: list[Example] = []
    for cat, items in by_cat.items():
        quota = int(round(shares.get(cat, 0.0) * d_syn))
        if quota <= 0:
            continue
        order = sorted(items, key=lambda e: priority(seed, e.prompt_id))
        acc = 0
        for e in order:
            if acc >= quota:
                break
            if acc + e.tokens_student <= quota + tolerance:
                chosen.append(e)
                acc += e.tokens_student
    return chosen


def subsample_nested_d(examples: Sequence[Example], d_levels: Sequence[int], *, seed: int,
                       category_shares: Optional[Mapping[str, float]] = None) -> dict[int, list[Example]]:
    """Nested prefixes: the set for a smaller D is a prefix of the set for a larger D."""
    levels = sorted(d_levels)
    out: dict[int, list[Example]] = {}
    for d in levels:
        out[d] = subsample_equal_tokens(examples, d, seed=seed, category_shares=category_shares)
    return out


def subsample_equal_examples(examples: Sequence[Example], n_per_cat: Mapping[str, int], *, seed: int) -> list[Example]:
    by_cat: dict[str, list[Example]] = {}
    for e in examples:
        by_cat.setdefault(e.category, []).append(e)
    chosen: list[Example] = []
    for cat, items in by_cat.items():
        order = sorted(items, key=lambda e: priority(seed, e.prompt_id))
        chosen.extend(order[: n_per_cat.get(cat, 0)])
    return chosen


def length_bins(edges: Optional[Sequence[float]] = None) -> np.ndarray:
    if edges is not None:
        return np.asarray(edges, dtype=float)
    return np.unique(np.round(np.geomspace(8, 1024, 13))).astype(float)


def length_matched(teacher_examples: Mapping[str, Sequence[Example]], *, seed: int,
                   d_target: int, slack: float = 0.2) -> dict[str, list[Example]]:
    """Give every teacher the same response-length distribution per category × bin (doc 09 §3.4).

    `teacher_examples` maps teacher id -> its filtered examples. Returns per-teacher subsets
    with matched (category, length-bin) counts = floor(f * intersection support).
    """
    edges = length_bins()
    teachers = list(teacher_examples)
    # bucket[t][(cat, bin)] -> list
    buckets: dict[str, dict[tuple[str, int], list[Example]]] = {t: {} for t in teachers}
    for t, exs in teacher_examples.items():
        for e in exs:
            b = int(np.searchsorted(edges, e.response_len_student, side="right"))
            buckets[t].setdefault((e.category, b), []).append(e)
    keys = set().union(*[set(b) for b in buckets.values()]) if buckets else set()
    inter = {k: min(len(buckets[t].get(k, [])) for t in teachers) for k in keys}
    inter = {k: v for k, v in inter.items() if v > 0}
    # mean length per bucket to convert counts->tokens
    def mean_len(k):
        vals = [e.response_len_student for t in teachers for e in buckets[t].get(k, [])]
        return float(np.mean(vals)) if vals else 0.0
    total_tokens = sum(inter[k] * mean_len(k) for k in inter)
    f = min(1.0 - slack, (d_target / total_tokens) if total_tokens else 0.0)
    out: dict[str, list[Example]] = {t: [] for t in teachers}
    for k, m in inter.items():
        take = int(np.floor(f * m))
        for t in teachers:
            order = sorted(buckets[t][k], key=lambda e: priority(seed, e.prompt_id))
            out[t].extend(order[:take])
    return out


def ks_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def subsample_report(chosen: Sequence[Example]) -> dict[str, Any]:
    lens = [e.response_len_student for e in chosen]
    toks = sum(e.tokens_student for e in chosen)
    return {
        "n_examples": len(chosen),
        "tokens_student_total": toks,
        "mean_response_len": float(np.mean(lens)) if lens else 0.0,
        "median_response_len": float(np.median(lens)) if lens else 0.0,
        "by_category": {c: sum(1 for e in chosen if e.category == c) for c in {e.category for e in chosen}},
    }
