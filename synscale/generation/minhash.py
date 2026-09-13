"""MinHash-LSH near-duplicate detection (pure Python + numpy).

Configuration follows the datatrove defaults referenced in the spec (§1.4, F6): word 5-gram
shingles, 14 bands × 8 rows = 112 hash functions, which approximates a Jaccard threshold of
≈ 0.72. Hash functions are universal hashes ``(a·x + b) mod p`` with ``p = 2^61 − 1`` over
32-bit shingle fingerprints, so ``a·x`` never overflows ``uint64``.

Two entry points:

* :func:`near_duplicate_clusters` — batch clustering with an explicit priority (the head of a
  cluster is the member with the *lowest* priority value, e.g. ``H64(seed, prompt_id)``).
* :class:`MinHashIndex` — a streaming index (used shard-by-shard by the runner) that reports,
  for each new document, the head it collides with, or ``None`` if it becomes its own head.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

_MERSENNE_61 = np.uint64((1 << 61) - 1)
_MAX32 = 1 << 32
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class MinHashConfig:
    n_gram: int = 5
    n_bands: int = 14
    rows_per_band: int = 8
    seed: int = 0

    @property
    def n_hashes(self) -> int:
        return self.n_bands * self.rows_per_band

    def approx_jaccard_threshold(self) -> float:
        """(1/b)^(1/r): the Jaccard similarity at which collision probability is ≈ 0.5."""
        return (1.0 / self.n_bands) ** (1.0 / self.rows_per_band)

    def to_dict(self) -> dict:
        return {
            "n_gram": self.n_gram,
            "n_bands": self.n_bands,
            "rows_per_band": self.rows_per_band,
            "seed": self.seed,
            "approx_jaccard_threshold": round(self.approx_jaccard_threshold(), 4),
        }


def _shingle_fingerprint(shingle: str) -> int:
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest(), "big")


def word_shingles(text: str, n: int) -> list[str]:
    """Lower-cased word n-grams; short texts (< n words) yield the whole text as one shingle."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return [" ".join(words)] if words else [""]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


class MinHasher:
    """Computes MinHash signatures and LSH band keys for texts."""

    def __init__(self, config: MinHashConfig = MinHashConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._a = rng.integers(1, _MAX32, size=config.n_hashes, dtype=np.uint64)
        self._b = rng.integers(0, _MAX32, size=config.n_hashes, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        fps = np.fromiter(
            {_shingle_fingerprint(s) for s in word_shingles(text, self.config.n_gram)},
            dtype=np.uint64,
        )
        if fps.size == 0:
            fps = np.zeros(1, dtype=np.uint64)
        # (a*x) < 2^64 because a, x < 2^32; reduce mod p before adding b.
        h = (self._a[:, None] * fps[None, :]) % _MERSENNE_61
        h = (h + self._b[:, None]) % _MERSENNE_61
        return h.min(axis=1)

    def band_keys(self, signature: np.ndarray) -> list[bytes]:
        r = self.config.rows_per_band
        return [
            bytes([j]) + signature[j * r : (j + 1) * r].tobytes() for j in range(self.config.n_bands)
        ]

    def keys(self, text: str) -> list[bytes]:
        return self.band_keys(self.signature(text))


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_clusters(
    texts: Sequence[str],
    priority: Optional[Sequence[int]] = None,
    config: MinHashConfig = MinHashConfig(),
    hasher: Optional[MinHasher] = None,
) -> list[Optional[int]]:
    """Cluster texts by LSH collisions; return for each text the index of its cluster head.

    The head of a cluster is the member with the lowest ``priority`` (default: list index, i.e.
    "keep first"). Entries that are their own head get ``None`` so callers can use
    ``head is not None`` as the near-duplicate flag.
    """
    hasher = hasher or MinHasher(config)
    n = len(texts)
    uf = _UnionFind(n)
    buckets: dict[bytes, int] = {}
    for i, text in enumerate(texts):
        for key in hasher.keys(text):
            j = buckets.get(key)
            if j is None:
                buckets[key] = i
            else:
                uf.union(i, j)
    prio = list(priority) if priority is not None else list(range(n))
    heads: dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        best = heads.get(root)
        if best is None or (prio[i], i) < (prio[best], best):
            heads[root] = i
    return [None if heads[uf.find(i)] == i else heads[uf.find(i)] for i in range(n)]


class MinHashIndex:
    """Streaming LSH index: each added document either joins an existing head or becomes one."""

    def __init__(self, config: MinHashConfig = MinHashConfig(), hasher: Optional[MinHasher] = None):
        self.config = config
        self.hasher = hasher or MinHasher(config)
        self._buckets: dict[bytes, str] = {}
        self.n_docs = 0

    def query(self, text: str) -> Optional[str]:
        """Return the head id colliding with ``text`` (without inserting)."""
        for key in self.hasher.keys(text):
            head = self._buckets.get(key)
            if head is not None:
                return head
        return None

    def add(self, doc_id: str, text: str) -> Optional[str]:
        """Insert ``text``; return the colliding head id, or ``None`` if ``doc_id`` is a new head."""
        keys = self.hasher.keys(text)
        head: Optional[str] = None
        for key in keys:
            h = self._buckets.get(key)
            if h is not None:
                head = h
                break
        if head is None:
            for key in keys:
                self._buckets.setdefault(key, doc_id)
        self.n_docs += 1
        return head

    def add_heads(self, docs: Iterable[tuple[str, str]]) -> None:
        """Register already-accepted (id, text) heads, e.g. when resuming from earlier shards."""
        for doc_id, text in docs:
            for key in self.hasher.keys(text):
                self._buckets.setdefault(key, doc_id)
            self.n_docs += 1
