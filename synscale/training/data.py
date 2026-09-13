"""Deterministic token data: on-disk formats, packing, ordering, tokenisation and the
phase-2 branch builder (student_training.md §2.7, §2.11). No torch dependency.

On-disk formats (``<stem>`` is a path without extension)
---------------------------------------------------------
Document stream — a tokenised corpus, documents concatenated in a fixed order::

    <stem>.tokens.bin   uint16 tokens (flat)
    <stem>.docs.bin     int32 document start offsets (first is 0, strictly increasing)
    <stem>.json         {"format": "documents", "n_tokens", "n_docs", "tokens_sha256", ...}

Packed sequences — rows of ``seq_len + 1`` tokens (``seq_len`` inputs and their shifted
targets), written by the branch builder so that every slot trains exactly ``seq_len``
tokens and no target crosses into another slot::

    <stem>.tokens.bin   uint16, shape (n_seq, seq_len + 1)
    <stem>.bounds.bin   int32 document starts in *input-stream coordinates* (row·seq_len + r)
    <stem>.json         {"format": "packed", "seq_len", "n_seq", ...}

Both are read through :class:`SequenceSource`, a stateless index → (tokens, cu_seqlens)
map. Sequence ``i`` of a document stream is ``tokens[i·L : i·L + L + 1]`` (rows overlap by
the one target token, exactly like the nanoGPT memmap loader); sequence ``i`` of a packed
file is row ``i``. ``cu_seqlens`` lists the document boundaries inside the ``L`` input
positions (always starting with 0 and ending with L), which drive the intra-document
attention mask and document-relative RoPE positions. A document straddling a row boundary
is cut: its continuation starts a new segment at position 0 with no access to the prefix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np

TOKEN_DTYPE = np.uint16
INDEX_DTYPE = np.int32

# ----------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- writers


def _paths(stem: str | Path) -> tuple[Path, Path, Path]:
    stem = Path(stem)
    return stem.with_name(stem.name + ".tokens.bin"), stem.with_name(stem.name + ".docs.bin"), stem.with_name(stem.name + ".json")


def _packed_paths(stem: str | Path) -> tuple[Path, Path, Path]:
    stem = Path(stem)
    return stem.with_name(stem.name + ".tokens.bin"), stem.with_name(stem.name + ".bounds.bin"), stem.with_name(stem.name + ".json")


def write_document_file(stem: str | Path, tokens: np.ndarray, doc_starts: np.ndarray, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a document stream; returns the metadata dict that was stored in ``<stem>.json``."""
    tokens = np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE)
    doc_starts = np.ascontiguousarray(doc_starts, dtype=INDEX_DTYPE)
    if len(doc_starts) == 0 or doc_starts[0] != 0 or np.any(np.diff(doc_starts) <= 0) or doc_starts[-1] >= len(tokens):
        raise ValueError("doc_starts must start at 0, be strictly increasing and lie inside the token array")
    tok_path, doc_path, meta_path = _paths(stem)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(tok_path)
    doc_starts.tofile(doc_path)
    meta = {
        "format": "documents", "n_tokens": int(len(tokens)), "n_docs": int(len(doc_starts)),
        "tokens_sha256": sha256_array(tokens), "docs_sha256": sha256_array(doc_starts),
        **(dict(extra) if extra else {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def read_meta(stem: str | Path) -> dict[str, Any]:
    return json.loads(_paths(stem)[2].read_text())


# ----------------------------------------------------------------------------- sequence sources


@dataclass(frozen=True)
class PackedSequence:
    """``tokens``: ``seq_len + 1`` uint16 tokens; ``cu_seqlens``: int32 ``[0, ..., seq_len]``."""

    tokens: np.ndarray
    cu_seqlens: np.ndarray


class SequenceSourceLike(Protocol):
    seq_len: int

    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> PackedSequence: ...


class SequenceSource:
    """Stateless index → sequence map over a document stream or a packed file.

    ``tokens`` is a flat uint16 array (memmap); ``doc_starts`` are document start positions in
    input-stream coordinates (``row·seq_len + r``); ``row_stride`` is ``seq_len`` for a
    document stream and ``seq_len + 1`` for a packed file.
    """

    def __init__(self, tokens: np.ndarray, doc_starts: np.ndarray, seq_len: int, row_stride: int) -> None:
        if row_stride not in (seq_len, seq_len + 1):
            raise ValueError("row_stride must be seq_len (document stream) or seq_len + 1 (packed)")
        self.tokens = tokens
        self.doc_starts = np.asarray(doc_starts, dtype=np.int64)
        self.seq_len = seq_len
        self.row_stride = row_stride
        self._n = int(len(tokens) // (seq_len + 1)) if row_stride == seq_len + 1 else int(max(len(tokens) - 1, 0) // seq_len)

    @classmethod
    def from_documents(cls, stem: str | Path, seq_len: int) -> "SequenceSource":
        tok_path, doc_path, _ = _paths(stem)
        tokens = np.memmap(tok_path, dtype=TOKEN_DTYPE, mode="r")
        return cls(tokens, np.fromfile(doc_path, dtype=INDEX_DTYPE), seq_len, seq_len)

    @classmethod
    def from_packed(cls, stem: str | Path) -> "SequenceSource":
        tok_path, bounds_path, meta_path = _packed_paths(stem)
        meta = json.loads(meta_path.read_text())
        if meta.get("format") != "packed":
            raise ValueError(f"{meta_path}: not a packed-sequence file")
        tokens = np.memmap(tok_path, dtype=TOKEN_DTYPE, mode="r")
        return cls(tokens, np.fromfile(bounds_path, dtype=INDEX_DTYPE), meta["seq_len"], meta["seq_len"] + 1)

    @classmethod
    def open(cls, stem: str | Path, seq_len: int) -> "SequenceSource":
        """Open either format; ``seq_len`` must match a packed file's stored value."""
        meta = read_meta(stem)
        if meta.get("format") == "packed":
            src = cls.from_packed(stem)
            if src.seq_len != seq_len:
                raise ValueError(f"{stem}: packed with seq_len {src.seq_len}, requested {seq_len}")
            return src
        return cls.from_documents(stem, seq_len)

    def __len__(self) -> int:
        return self._n

    def cu_seqlens(self, i: int) -> np.ndarray:
        lo, hi = i * self.seq_len, (i + 1) * self.seq_len
        a, b = np.searchsorted(self.doc_starts, lo, side="left"), np.searchsorted(self.doc_starts, hi, side="left")
        inner = self.doc_starts[a:b] - lo
        inner = inner[inner > 0]
        return np.concatenate(([0], inner, [self.seq_len])).astype(INDEX_DTYPE)

    def __getitem__(self, i: int) -> PackedSequence:
        if not 0 <= i < self._n:
            raise IndexError(i)
        start = i * self.row_stride
        return PackedSequence(np.asarray(self.tokens[start:start + self.seq_len + 1]), self.cu_seqlens(i))


class ShardedSource:
    """Concatenation of several sources (e.g. replay shards); each shard's tail remainder is
    dropped, so sequence ``i`` is a pure function of the shard list."""

    def __init__(self, shards: Sequence[SequenceSourceLike]) -> None:
        if not shards:
            raise ValueError("at least one shard is required")
        self.shards = list(shards)
        self.seq_len = shards[0].seq_len
        self._offsets = np.cumsum([0] + [len(s) for s in shards])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, i: int) -> PackedSequence:
        if not 0 <= i < len(self):
            raise IndexError(i)
        k = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return self.shards[k][i - int(self._offsets[k])]


def position_ids_from_cu_seqlens(cu_seqlens: np.ndarray, seq_len: int) -> np.ndarray:
    """Document-relative positions for the ``seq_len`` input tokens of one sequence."""
    pos = np.arange(seq_len, dtype=np.int64)
    starts = np.repeat(cu_seqlens[:-1], np.diff(cu_seqlens))
    return pos - starts


@dataclass(frozen=True)
class Batch:
    """A micro-batch as numpy arrays; ``cu_seqlens`` is flattened over ``B·L`` tokens."""

    input_ids: np.ndarray
    targets: np.ndarray
    position_ids: np.ndarray
    cu_seqlens: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(self.targets.size)


def collate(items: Sequence[PackedSequence], seq_len: int) -> Batch:
    """Stack sequences into a batch; document boundaries are offset by ``b·seq_len`` so the
    flattened ``cu_seqlens`` follows the flash-attn varlen convention."""
    tokens = np.stack([it.tokens for it in items]).astype(np.int64)
    cu = np.concatenate([it.cu_seqlens[:-1].astype(np.int64) + b * seq_len for b, it in enumerate(items)] + [[len(items) * seq_len]])
    pos = np.stack([position_ids_from_cu_seqlens(it.cu_seqlens, seq_len) for it in items])
    return Batch(input_ids=tokens[:, :-1], targets=tokens[:, 1:], position_ids=pos, cu_seqlens=cu)


# ----------------------------------------------------------------------------- ordering


class DataOrder:
    """Stateless map from a global sequence position to a source index.

    With ``shuffle=True`` position ``p`` maps to ``perm_e[p mod n]`` where ``perm_e`` is the
    seeded permutation of epoch ``e = p // n`` (seed and epoch both enter the RNG seed, so
    the order is a pure function of ``(seed, p)``). With ``shuffle=False`` (branch files,
    whose order was fixed by the builder) it is the identity modulo ``n``.
    """

    def __init__(self, n_sequences: int, seed: int, shuffle: bool = True) -> None:
        if n_sequences <= 0:
            raise ValueError("empty source")
        self.n = n_sequences
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = -1
        self._perm: np.ndarray | None = None

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n)
            self._epoch = epoch
        assert self._perm is not None
        return self._perm

    def index(self, position: int) -> int:
        if not self.shuffle:
            return position % self.n
        epoch, offset = divmod(position, self.n)
        return int(self._permutation(epoch)[offset])

    def indices(self, start: int, count: int) -> list[int]:
        return [self.index(p) for p in range(start, start + count)]


# ----------------------------------------------------------------------------- tokenisation


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


INSTRUCTION_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n{response}"


def render_instruction(prompt: str, response: str) -> str:
    """Student-visible rendering of one synthetic/instruction example (doc 00, Amendment B).
    BOS/EOS are added as token ids by :func:`encode_document`, giving
    ``<s>### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}</s>``."""
    return INSTRUCTION_TEMPLATE.format(prompt=prompt, response=response)


def encode_document(tokenizer: TokenizerLike, text: str, bos_id: int, eos_id: int) -> list[int]:
    return [bos_id, *tokenizer.encode(text), eos_id]


def tokenize_documents(texts: Iterable[str], tokenizer: TokenizerLike, bos_id: int, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode documents in the given order; returns ``(tokens uint16, doc_starts int32)``."""
    chunks: list[np.ndarray] = []
    starts: list[int] = []
    total = 0
    for text in texts:
        ids = np.asarray(encode_document(tokenizer, text, bos_id, eos_id), dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= np.iinfo(TOKEN_DTYPE).max + 1):
            raise ValueError("token id outside the uint16 range")
        starts.append(total)
        chunks.append(ids.astype(TOKEN_DTYPE))
        total += int(ids.size)
    if not chunks:
        raise ValueError("no documents")
    return np.concatenate(chunks), np.asarray(starts, dtype=INDEX_DTYPE)


def render_instruction_examples(examples: Iterable[Mapping[str, Any]], prompt_key: str = "prompt", response_key: str = "response") -> Iterator[str]:
    for ex in examples:
        yield render_instruction(ex[prompt_key], ex[response_key])


# ----------------------------------------------------------------------------- branch builder (§2.11)

_STREAM_SLOTS, _STREAM_TREAT, _STREAM_REPLAY = 0, 1, 2
SLOT_TREAT, SLOT_REPLAY = 0, 1


@dataclass(frozen=True)
class BranchSpec:
    """Inputs that define a phase-2 branch (everything except the two data files)."""

    size: str
    seed: int
    seq_len: int
    d2_tokens: int
    replay_fraction: float

    @property
    def n_slots(self) -> int:
        if self.d2_tokens % self.seq_len:
            raise ValueError("d2_tokens must be a multiple of seq_len")
        return self.d2_tokens // self.seq_len

    @property
    def n_replay(self) -> int:
        return int(round(self.replay_fraction * self.n_slots))

    @property
    def n_treatment(self) -> int:
        return self.n_slots - self.n_replay


def slot_pattern(n_treat: int, n_rep: int, seed: int) -> np.ndarray:
    """Seeded permutation of the multiset {treat × n_treat, replay × n_rep} (uint8; 0 = treat,
    1 = replay). Drawn once per (size, seed) and shared by every condition of that size."""
    pattern = np.concatenate([np.full(n_treat, SLOT_TREAT, np.uint8), np.full(n_rep, SLOT_REPLAY, np.uint8)])
    return np.random.default_rng([seed, _STREAM_SLOTS]).permutation(pattern)


def build_branch(
    treatment: SequenceSourceLike,
    replay: SequenceSourceLike,
    spec: BranchSpec,
    out_stem: str | Path,
    treatment_ref: str = "",
    replay_ref: str = "",
    slots: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assemble the packed phase-2 file for one condition (student_training.md §2.11).

    1. take the first ``n_treat`` treatment sequences in file order (shortfall is an error);
    2. take the first ``n_rep`` replay sequences (identical for every condition of the size);
    3. draw (or accept) the seeded slot pattern and seeded within-slot permutations;
    4. write ``<out_stem>.tokens.bin`` (uint16, ``n_slots × (seq_len + 1)``),
       ``<out_stem>.bounds.bin`` (int32 document starts), ``<out_stem>.slots.npy`` and
       ``<out_stem>.json`` (format "packed") plus ``<out_stem>.branch.json`` recording every
       input and SHA-256.

    Returns the ``branch.json`` content.
    """
    n_treat, n_rep, n_slots = spec.n_treatment, spec.n_replay, spec.n_slots
    if len(treatment) < n_treat:
        raise ValueError(f"treatment has {len(treatment)} sequences, need {n_treat} (over-generate by >= 5%)")
    if len(replay) < n_rep:
        raise ValueError(f"replay has {len(replay)} sequences, need {n_rep}")
    if treatment.seq_len != spec.seq_len or replay.seq_len != spec.seq_len:
        raise ValueError("source seq_len differs from spec.seq_len")
    if slots is None:
        slots = slot_pattern(n_treat, n_rep, spec.seed)
    slots = np.asarray(slots, dtype=np.uint8)
    if len(slots) != n_slots or int((slots == SLOT_REPLAY).sum()) != n_rep:
        raise ValueError("slot pattern does not match n_treat / n_rep")
    treat_perm = np.random.default_rng([spec.seed, _STREAM_TREAT]).permutation(n_treat)
    rep_perm = np.random.default_rng([spec.seed, _STREAM_REPLAY]).permutation(n_rep)

    tok_path, bounds_path, meta_path = _packed_paths(out_stem)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    row = spec.seq_len + 1
    out = np.memmap(tok_path, dtype=TOKEN_DTYPE, mode="w+", shape=(n_slots, row))
    bounds: list[np.ndarray] = []
    t_next = r_next = 0
    for k, kind in enumerate(slots):
        if kind == SLOT_TREAT:
            item = treatment[int(treat_perm[t_next])]
            t_next += 1
        else:
            item = replay[int(rep_perm[r_next])]
            r_next += 1
        out[k] = item.tokens
        inner = item.cu_seqlens[1:-1]
        if inner.size:
            bounds.append(inner.astype(np.int64) + k * spec.seq_len)
    out.flush()
    del out
    bounds_arr = (np.concatenate(bounds) if bounds else np.zeros(0, np.int64)).astype(INDEX_DTYPE)
    bounds_arr.tofile(bounds_path)
    slots_path = Path(str(out_stem) + ".slots.npy")
    np.save(slots_path, slots)

    tokens_sha, bounds_sha = sha256_file(tok_path), sha256_array(bounds_arr)
    combined = hashlib.sha256((tokens_sha + bounds_sha).encode()).hexdigest()
    packed_meta = {"format": "packed", "seq_len": spec.seq_len, "n_seq": n_slots, "tokens_sha256": tokens_sha, "bounds_sha256": bounds_sha, "sha256": combined}
    meta_path.write_text(json.dumps(packed_meta, indent=2, sort_keys=True))
    branch = {
        "size": spec.size, "seed": spec.seed, "seq_len": spec.seq_len, "d2_tokens": spec.d2_tokens,
        "replay_fraction": spec.replay_fraction, "n_slots": n_slots, "n_treatment": n_treat, "n_replay": n_rep,
        "treatment_tokens": n_treat * spec.seq_len, "replay_tokens": n_rep * spec.seq_len,
        "treatment": {"ref": treatment_ref, "sequences_available": len(treatment)},
        "replay": {"ref": replay_ref, "sequences_available": len(replay)},
        "slots_sha256": sha256_array(slots), "slots_path": str(slots_path),
        "tokens_sha256": tokens_sha, "bounds_sha256": bounds_sha, "sha256": combined,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(str(out_stem) + ".branch.json").write_text(json.dumps(branch, indent=2, sort_keys=True))
    return branch
