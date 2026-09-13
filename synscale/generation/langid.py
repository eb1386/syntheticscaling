"""Language identification behind an interface.

The study uses fastText ``lid.176`` (p(en) ≥ 0.65, datatrove default). Tests use a deterministic
stop-word heuristic that needs no model file. Both return ``(language, probability)``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

_WORD_RE = re.compile(r"[A-Za-z']+")
_STOPWORDS = frozenset(
    """the a an and or but if then of to in on at by for with from as is are was were be been being
    it its this that these those there here he she they we you i me him her them us my your his
    their our not no yes do does did done have has had will would can could should may might must
    what which who whom whose when where why how all any each some more most other such only own
    same so than too very just also because while about into over after before between under again
    further once during out up down off through above below both few many much one two three first
    last new old good great long little large small every another every well still even never
    always often sometimes""".split()
)


@runtime_checkable
class LanguageDetector(Protocol):
    name: str

    def detect(self, text: str) -> tuple[str, float]:  # pragma: no cover - protocol
        ...


class HeuristicEnglishDetector:
    """Stop-word + Latin-script heuristic; p(en) ≈ 1 for ordinary English prose."""

    name = "heuristic-stopword-v1"

    def detect(self, text: str) -> tuple[str, float]:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return ("und", 0.0)
        latin_ratio = sum(1 for c in letters if c.isascii()) / len(letters)
        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            return ("und", 0.0)
        stop_ratio = sum(1 for w in words if w in _STOPWORDS) / len(words)
        p = min(1.0, 0.30 + 2.5 * stop_ratio) * latin_ratio
        return ("en", p) if p >= 0.5 else ("und", p)


class FastTextDetector:
    """fastText ``lid.176.bin`` wrapper (lazy import; model file supplied by the caller)."""

    name = "fasttext-lid.176"

    def __init__(self, model_path: str | Path):
        import fasttext  # type: ignore  # lazy optional dependency

        self._model = fasttext.load_model(str(model_path))

    def detect(self, text: str) -> tuple[str, float]:
        labels, probs = self._model.predict(text.replace("\n", " "), k=1)
        return (labels[0].replace("__label__", ""), float(probs[0]))
