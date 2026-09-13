"""Token counting behind an interface (student tokenizer = the accounting unit).

The real study counts tokens with the pinned Llama-2 32k SentencePiece tokenizer
(``HFTokenizerCounter`` via the ``tokenizers`` package). Tests and CPU-only runs use
``ApproxTokenCounter``, a deterministic word/punctuation splitter with a ≈1.3 tokens/word ratio.

Amendment A: D_syn counts *all* tokens of the rendered example — prompt + response + markers +
BOS/EOS — so ``count_rendered`` is the only function that should feed quota accounting.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from synscale.generation.templates import render_student_body

_PIECE_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", re.UNICODE)


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens of a string under one fixed tokenizer."""

    name: str
    sha256: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


class ApproxTokenCounter:
    """Deterministic, tokenizer-free approximation used in tests and smoke runs.

    Every alphabetic run counts ``ceil(len/6)`` pieces (long words split like BPE), every digit
    counts one (Llama-2 splits digits), every punctuation mark one. Whitespace is free.
    """

    name = "approx-v1"
    sha256 = hashlib.sha256(b"approx-v1").hexdigest()

    def count(self, text: str) -> int:
        n = 0
        for piece in _PIECE_RE.findall(text):
            if piece.isalpha():
                n += max(1, math.ceil(len(piece) / 6))
            else:
                n += 1
        return n


class HFTokenizerCounter:
    """Counts tokens with a ``tokenizers`` JSON file (e.g. the pinned Llama-2 tokenizer)."""

    def __init__(self, tokenizer_path: str | Path, name: Optional[str] = None):
        from tokenizers import Tokenizer  # lazy: optional dependency

        path = Path(tokenizer_path)
        self._tok = Tokenizer.from_file(str(path))
        self.name = name or path.stem
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


def count_rendered(counter: TokenCounter, prompt: str, response: str) -> int:
    """Total student tokens of the rendered example: markers + prompt + response + BOS + EOS."""
    return counter.count(render_student_body(prompt, response)) + 2


def count_words(text: str) -> int:
    return len(text.split())
