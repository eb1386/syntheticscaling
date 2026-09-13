"""Per-teacher × benchmark contamination audit (benchmarks.md §7; synthetic_data.md §1.5).

Inputs are the per-example generation JSONL records (one line per generated sample, retained
and dropped alike).  Three quantities are reported per teacher and benchmark:

* ``rho`` — item-level contamination rate: fraction of benchmark items with ≥ 1 n-gram (or
  full-short-item) hit anywhere in the teacher's pool;
* ``r`` — example-level rate: fraction of the teacher's examples with ≥ 1 hit on that benchmark;
* ``density`` — benchmark-style density: fraction of responses that look like a 4-option
  multiple-choice science question (regex), a stylistic leakage that survives n-gram filtering.

Hits come from (in order of preference) an explicit ``contam_hits`` list on the record
(``[{"benchmark": ..., "item_id": ...}, ...]``), an ``NGramIndex`` built here from the benchmark
items, or — as a last resort — the boolean ``flags.contam_8gram`` / ``flags.contam_semantic``
of the generation schema, which only support benchmark ``"any"``.  The paraphrase audit
(embedding + LLM judge) is out of scope for this module; ``rho_para`` may be supplied externally
and is carried through the table.

``clean_item_subset`` builds the ``clean-b`` item set (zero hits in *any* teacher's pool) used
to re-evaluate all conditions in the sensitivity analysis.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_NGRAM = 13  # GPT-3 / harness convention; 8 for the stricter tier (synthetic_data.md §1.5)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, NFKC, strip punctuation, collapse whitespace (harness janitor convention)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def word_ngrams(words: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


class NGramIndex:
    """Word n-gram index over benchmark items with full-item matching for short items."""

    def __init__(self, n: int = DEFAULT_NGRAM) -> None:
        self.n = n
        self._grams: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
        self._short: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.n_items: dict[str, int] = defaultdict(int)

    def add(self, benchmark: str, item_id: str, text: str) -> None:
        words = normalize(text).split()
        self.n_items[benchmark] += 1
        key = (benchmark, str(item_id))
        if len(words) < self.n:
            if words:
                self._short[" ".join(words)].add(key)
            return
        for g in word_ngrams(words, self.n):
            self._grams[g].add(key)

    def add_items(self, benchmark: str, items: Mapping[str, str]) -> None:
        for k, v in items.items():
            self.add(benchmark, k, v)

    def hits(self, text: str) -> set[tuple[str, str]]:
        norm = normalize(text)
        words = norm.split()
        out: set[tuple[str, str]] = set()
        for g in word_ngrams(words, self.n):
            if g in self._grams:
                out |= self._grams[g]
        if self._short:
            padded = f" {norm} "
            for s, keys in self._short.items():
                if f" {s} " in padded:
                    out |= keys
        return out


# ----------------------------------------------------------------------------------------
# Benchmark-style density
# ----------------------------------------------------------------------------------------
_OPTION_RE = re.compile(r"(?m)^\s*\(?([A-Da-d])[).:]\s+\S")


def is_benchmark_style(response: str) -> bool:
    """True if the response contains a question mark and four option lines labelled A–D."""
    labels = [m.group(1).upper() for m in _OPTION_RE.finditer(response)]
    return "?" in response and {"A", "B", "C", "D"}.issubset(set(labels))


def benchmark_style_density(responses: Iterable[str]) -> float:
    rs = list(responses)
    return float(np.mean([is_benchmark_style(r) for r in rs])) if rs else float("nan")


# ----------------------------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------------------------
@dataclass
class AuditResult:
    teacher: str
    n_examples: int
    hit_items: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # benchmark -> item ids
    examples_hit: dict[str, int] = field(default_factory=lambda: defaultdict(int))  # benchmark -> n examples
    example_ids_hit: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    density: float = float("nan")
    n_items: dict[str, int] = field(default_factory=dict)

    def rho(self, benchmark: str) -> float:
        n = self.n_items.get(benchmark)
        return len(self.hit_items.get(benchmark, ())) / n if n else float("nan")

    def r(self, benchmark: str) -> float:
        return self.examples_hit.get(benchmark, 0) / self.n_examples if self.n_examples else float("nan")


def _record_hits(rec: dict[str, Any], index: Optional[NGramIndex]) -> set[tuple[str, str]]:
    if rec.get("contam_hits"):
        return {(str(h["benchmark"]), str(h["item_id"])) for h in rec["contam_hits"]}
    if index is not None:
        text = f"{rec.get('student_prompt', rec.get('prompt', ''))}\n{rec.get('response', '')}"
        return index.hits(text)
    flags = rec.get("flags") or {}
    if flags.get("contam_8gram") or flags.get("contam_semantic") or flags.get("contam_13gram"):
        return {("any", f"flag:{rec.get('example_id', rec.get('prompt_id', ''))}")}
    return set()


def audit_records(
    records: Iterable[dict[str, Any]],
    *,
    teacher: Optional[str] = None,
    index: Optional[NGramIndex] = None,
    n_items: Optional[Mapping[str, int]] = None,
) -> AuditResult:
    """Audit one teacher's generation records (see module docstring for hit sources)."""
    res = AuditResult(teacher or "?", 0)
    if index is not None:
        res.n_items.update(index.n_items)
    if n_items:
        res.n_items.update(n_items)
    styles: list[bool] = []
    for rec in records:
        res.n_examples += 1
        if teacher is None and rec.get("teacher"):
            res.teacher = str(rec["teacher"])
        ex_id = str(rec.get("example_id", rec.get("prompt_id", res.n_examples)))
        hits = _record_hits(rec, index)
        per_bench: set[str] = set()
        for b, item in hits:
            res.hit_items[b].add(item)
            per_bench.add(b)
        for b in per_bench:
            res.examples_hit[b] += 1
            res.example_ids_hit[b].add(ex_id)
        styles.append(is_benchmark_style(str(rec.get("response", ""))))
    res.density = float(np.mean(styles)) if styles else float("nan")
    return res


def read_generation_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream a (possibly zstd-compressed) per-example JSONL."""
    p = Path(path)
    if p.suffix == ".zst":
        import zstandard  # type: ignore  # lazy: only needed for compressed pools

        with open(p, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
            import io

            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)
    else:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def audit_jsonl(path: str | Path, **kw: Any) -> AuditResult:
    return audit_records(read_generation_jsonl(path), **kw)


def contamination_table(
    audits: Mapping[str, AuditResult],
    benchmarks: Optional[Sequence[str]] = None,
    rho_para: Optional[Mapping[tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """Teacher × benchmark long table with ``rho``, ``r``, ``density`` (and ``rho_para`` if given)."""
    if benchmarks is None:
        bs: set[str] = set()
        for a in audits.values():
            bs |= set(a.hit_items) | set(a.n_items)
        benchmarks = sorted(bs)
    rows = []
    for t, a in audits.items():
        for b in benchmarks:
            rows.append({
                "teacher": t, "benchmark": b, "n_examples": a.n_examples,
                "n_items": a.n_items.get(b, np.nan), "items_hit": len(a.hit_items.get(b, ())),
                "rho": a.rho(b), "examples_hit": a.examples_hit.get(b, 0), "r": a.r(b),
                "density": a.density,
                "rho_para": (rho_para or {}).get((t, b), np.nan),
            })
    return pd.DataFrame(rows)


def clean_item_subset(audits: Mapping[str, AuditResult], benchmark: str, all_item_ids: Iterable[str]) -> set[str]:
    """Items of ``benchmark`` with zero hits in every teacher's pool (the ``clean-b`` set)."""
    hit: set[str] = set()
    for a in audits.values():
        hit |= {str(i) for i in a.hit_items.get(benchmark, ())}
    return {str(i) for i in all_item_ids} - hit


def flag_paraphrase_divergence(table: pd.DataFrame, threshold_pts: float = 2.0,
                               primary: Sequence[str] = ()) -> list[str]:
    """Benchmarks whose ``rho_para`` differs across teachers by more than ``threshold_pts`` points."""
    flagged = []
    if "rho_para" not in table:
        return flagged
    for b, g in table.groupby("benchmark"):
        v = g["rho_para"].dropna()
        if len(v) >= 2 and 100 * (v.max() - v.min()) > threshold_pts and (not primary or b in primary):
            flagged.append(str(b))
    return flagged


def accuracy_on_subset(per_item: Sequence[dict[str, Any]], metric: str, item_ids: Iterable[Any]) -> tuple[float, int]:
    """Re-score one archived task on a subset of ``doc_id``s (for the clean-b sensitivity analysis)."""
    keep = {str(i) for i in item_ids}
    vals = [float(it[metric]) for it in per_item if str(it.get("doc_id")) in keep and metric in it]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def union_drop_prompts(records_by_teacher: Mapping[str, Iterable[dict[str, Any]]],
                       index: Optional[NGramIndex] = None) -> set[str]:
    """Prompt ids to drop for *all* teachers: any teacher's response hit (union-drop rule)."""
    drop: set[str] = set()
    for recs in records_by_teacher.values():
        for rec in recs:
            if _record_hits(rec, index):
                drop.add(str(rec.get("prompt_id")))
    return drop
