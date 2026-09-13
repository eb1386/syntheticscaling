"""Benchmark decontamination (spec §1.5).

Stage 1 (exact): word-level 8-gram overlap between normalised pool text (seed passages, prompts,
responses) and the benchmark union B; benchmark items shorter than 8 words are matched as whole
items. A 13-gram variant (GPT-3 convention) is available for the sensitivity analysis.

Stage 2 (semantic): embedding similarity against B questions behind a pluggable ``Embedder``;
flagged pairs are meant to be adjudicated by the fixed non-Qwen judge — here the stage reports
candidates and the judge step is a hook.

Union-drop bookkeeping: response-level flags are teacher-dependent; if any teacher's response to
prompt p is flagged, p is dropped for every teacher (:class:`UnionDropLedger`).
"""
from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

DECONTAM_VERSION = "decontam-v1.0"
DEFAULT_NGRAM = 8
SENSITIVITY_NGRAM = 13

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """lowercase → NFKC → strip punctuation → collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(_PUNCT_TABLE)
    # remove any remaining non-ASCII punctuation / symbols
    text = "".join(c if (c.isalnum() or c.isspace()) else " " for c in text)
    return _WS_RE.sub(" ", text).strip()


def _ngram_hash(words: Sequence[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class BenchmarkItem:
    """One benchmark item: ``text`` = question + all options + answer (pre-normalisation)."""

    benchmark: str
    split: str
    text: str
    item_id: str = ""


def load_benchmark_items(path: str | Path) -> list[BenchmarkItem]:
    """Read JSONL with fields benchmark, split, and either ``text`` or question/options/answer."""
    items: list[BenchmarkItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "text" in d:
                text = d["text"]
            else:
                parts = [d.get("question", "")]
                opts = d.get("options") or []
                parts.extend(opts if isinstance(opts, list) else [str(opts)])
                parts.append(str(d.get("answer", "")))
                text = " ".join(p for p in parts if p)
            items.append(
                BenchmarkItem(
                    benchmark=d.get("benchmark", "unknown"),
                    split=d.get("split", "test"),
                    text=text,
                    item_id=str(d.get("id", d.get("item_id", ""))),
                )
            )
    return items


@dataclass(frozen=True)
class ContamHit:
    benchmark: str
    kind: str  # "ngram" | "whole_item" | "semantic"
    matched: str
    score: float = 1.0


class NGramIndex:
    """Hash set of all word n-grams of B (+ whole short items)."""

    def __init__(self, n: int = DEFAULT_NGRAM):
        self.n = n
        self._ngrams: dict[int, str] = {}
        self._short: dict[str, str] = {}
        self.n_items = 0
        self.per_benchmark: Counter[str] = Counter()

    @classmethod
    def build(cls, items: Iterable[BenchmarkItem], n: int = DEFAULT_NGRAM) -> "NGramIndex":
        idx = cls(n)
        for it in items:
            idx.add(it)
        return idx

    def add(self, item: BenchmarkItem) -> None:
        words = normalise(item.text).split()
        self.n_items += 1
        self.per_benchmark[item.benchmark] += 1
        if len(words) < self.n:
            if words:
                self._short.setdefault(" ".join(words), item.benchmark)
            return
        for i in range(len(words) - self.n + 1):
            self._ngrams.setdefault(_ngram_hash(words[i : i + self.n]), item.benchmark)

    @property
    def n_ngrams(self) -> int:
        return len(self._ngrams)

    def flag(self, text: str, first_only: bool = True) -> list[ContamHit]:
        """Return contamination hits for ``text`` (empty list = clean)."""
        norm = normalise(text)
        words = norm.split()
        hits: list[ContamHit] = []
        if len(words) >= self.n:
            for i in range(len(words) - self.n + 1):
                bm = self._ngrams.get(_ngram_hash(words[i : i + self.n]))
                if bm is not None:
                    hits.append(ContamHit(bm, "ngram", " ".join(words[i : i + self.n])))
                    if first_only:
                        return hits
        if self._short:
            padded = f" {norm} "
            for short, bm in self._short.items():
                if f" {short} " in padded:
                    hits.append(ContamHit(bm, "whole_item", short))
                    if first_only:
                        return hits
        return hits

    def is_contaminated(self, text: str) -> bool:
        return bool(self.flag(text))


# --------------------------------------------------------------------------------------
# Semantic stage (interface + stub)
# --------------------------------------------------------------------------------------
class Embedder(Protocol):
    """Maps texts to L2-normalised embeddings, shape (n, d)."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol
        ...


class StubEmbedder:
    """Hashed bag-of-words embedding (deterministic; paraphrases sharing words score high)."""

    name = "stub-hashed-bow"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for w in normalise(t).split():
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "big")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SemanticStage:
    """Exact cosine search of pool texts against benchmark questions; flags ≥ threshold."""

    def __init__(self, embedder: Embedder, items: Sequence[BenchmarkItem], threshold: float = 0.85,
                 judge: Optional[Callable[[str, str], bool]] = None):
        self.embedder = embedder
        self.items = list(items)
        self.threshold = threshold
        self.judge = judge  # (pool_text, benchmark_text) -> is_rephrase
        self._emb = embedder.embed([it.text for it in self.items]) if self.items else np.zeros((0, 1))

    def flag(self, texts: Sequence[str]) -> list[Optional[ContamHit]]:
        if not self.items or not texts:
            return [None] * len(texts)
        q = self.embedder.embed(texts)
        sims = q @ self._emb.T
        out: list[Optional[ContamHit]] = []
        for i in range(len(texts)):
            j = int(np.argmax(sims[i]))
            s = float(sims[i, j])
            if s >= self.threshold and (self.judge is None or self.judge(texts[i], self.items[j].text)):
                out.append(ContamHit(self.items[j].benchmark, "semantic", self.items[j].text, s))
            else:
                out.append(None)
        return out


# --------------------------------------------------------------------------------------
# Union-drop bookkeeping and reports
# --------------------------------------------------------------------------------------
@dataclass
class UnionDropLedger:
    """Tracks response-level contamination flags per teacher and the union of dropped prompts."""

    flags: dict[str, dict[str, list[ContamHit]]] = field(default_factory=lambda: defaultdict(dict))
    offered: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def offer(self, teacher: str, prompt_id: str) -> None:
        self.offered[teacher].add(prompt_id)

    def add(self, teacher: str, prompt_id: str, hits: Sequence[ContamHit]) -> None:
        self.offered[teacher].add(prompt_id)
        if hits:
            self.flags[teacher][prompt_id] = list(hits)

    def dropped_prompt_ids(self) -> set[str]:
        out: set[str] = set()
        for per_prompt in self.flags.values():
            out.update(per_prompt.keys())
        return out

    def per_teacher_rates(self) -> dict[str, dict]:
        out = {}
        for teacher in sorted(self.offered):
            n = len(self.offered[teacher])
            flagged = self.flags.get(teacher, {})
            by_bm: Counter[str] = Counter(h.benchmark for hits in flagged.values() for h in hits)
            out[teacher] = {
                "n_offered": n,
                "n_flagged": len(flagged),
                "flag_rate": (len(flagged) / n) if n else 0.0,
                "by_benchmark": dict(sorted(by_bm.items())),
            }
        return out

    def report(self) -> dict:
        dropped = self.dropped_prompt_ids()
        return {
            "version": DECONTAM_VERSION,
            "rule": "union-drop: a prompt flagged for any teacher is dropped for all teachers",
            "n_union_dropped": len(dropped),
            "per_teacher": self.per_teacher_rates(),
        }


def contamination_table(
    index: NGramIndex,
    texts_by_teacher: dict[str, Sequence[tuple[str, str]]],
) -> dict:
    """Per-teacher × benchmark flag counts over (prompt_id, text) pairs; feeds the audit script."""
    ledger = UnionDropLedger()
    benchmarks = sorted(index.per_benchmark)
    table: dict[str, dict[str, int]] = {}
    for teacher, pairs in texts_by_teacher.items():
        row = {bm: 0 for bm in benchmarks}
        for prompt_id, text in pairs:
            hits = index.flag(text, first_only=False)
            ledger.add(teacher, prompt_id, hits)
            for bm in {h.benchmark for h in hits}:
                row[bm] += 1
        table[teacher] = row
    return {
        "ngram_n": index.n,
        "benchmarks": benchmarks,
        "n_benchmark_items": index.n_items,
        "table": table,
        "ledger": ledger.report(),
    }
