"""Teacher-agnostic response filters F1-F12 (doc 09 §2.2, doc 00 §0.7).

Every filter is a cheap regex/hash/threshold pass. Every example carries all flags;
`retained` is the conjunction of the drop rules. Filters are applied identically to
every teacher (so filtering is part of the treatment, not a nuisance), and each
teacher's per-filter rates are reported as covariates. The only union-drop filter is
contamination (F11), handled by the runner via decontam, not here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# --- F7 refusal / self-identification -----------------------------------------
_REFUSAL = re.compile(
    r"\b(i'?m sorry,? but|i cannot (help|assist|provide|comply)|as an ai|i am an ai language model|"
    r"i'?m unable to|i can'?t help with|i will not|i won'?t be able to)\b", re.I)
_SELF_ID = re.compile(r"\b(qwen|alibaba cloud|tongyi|通义)\b", re.I)
# --- F8 passage reference (seeded prompts only) --------------------------------
_PASSAGE_REF = re.compile(
    r"\b(the (passage|text|background|excerpt) (above|below|provided|given)|"
    r"according to the (passage|text)|as (mentioned|stated) (above|in the text)|the author)\b", re.I)
# --- CJK detection (F3) -------------------------------------------------------
_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")
_WORD = re.compile(r"\S+")
_FINAL_ANSWER = re.compile(r"final answer\s*:", re.I)


def _char_ngram_frac(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0
    grams: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        g = text[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    if not grams:
        return 0.0
    top = max(grams.values())
    return top * n / max(len(text), 1)


def _dup_line_frac(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    dup = sum(c for c in seen.values() if c > 1) - sum(1 for c in seen.values() if c > 1)
    return dup / len(lines)


# Gopher-style repetition thresholds (Rae et al. 2021; datatrove defaults).
GOPHER = {
    "dup_line_frac": 0.30,
    "top_2gram": 0.20, "top_3gram": 0.18, "top_4gram": 0.16,
}


@dataclass
class FilterConfig:
    ruleset: str = "filters-v1.0"
    min_tokens_saqa: int = 8
    min_tokens_other: int = 32
    lang_min_p_en: float = 0.65
    cjk_max_ratio: float = 0.01
    require_final_answer_for_msr: bool = True


@dataclass
class FilterResult:
    retained: bool
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def drop_reason(self) -> Optional[str]:
        order = ["truncated", "too_short", "non_english", "cjk", "degenerate",
                 "exact_dup", "refusal", "self_id", "passage_ref", "format_invalid",
                 "missing_answer", "contaminated"]
        for k in order:
            if self.flags.get(k):
                return k
        return None


class ResponseFilter:
    """Applies F1-F10, F12 (F6 near-dup and F11 contamination are runner-level, cross-example)."""

    def __init__(self, cfg: Optional[FilterConfig] = None, lang_detector=None):
        self.cfg = cfg or FilterConfig()
        self.lang = lang_detector
        self._seen_hashes: set[str] = set()

    def reset(self) -> None:
        self._seen_hashes.clear()

    def _word_count(self, text: str) -> int:
        return len(_WORD.findall(text))

    def apply(self, response: str, *, category: str, finish_reason: str,
              n_student_tokens: int, str_kind: Optional[str] = None,
              constraints: Optional[list] = None, is_seeded: bool = False,
              contaminated: bool = False) -> FilterResult:
        f: dict[str, Any] = {}
        text = response or ""
        # F1 truncation
        f["truncated"] = finish_reason == "length"
        # F2 too short
        min_tok = self.cfg.min_tokens_saqa if category == "saqa" else self.cfg.min_tokens_other
        f["too_short"] = n_student_tokens < min_tok or not text.strip()
        # F3 language / CJK
        cjk_chars = len(_CJK.findall(text))
        f["cjk_ratio"] = cjk_chars / max(len(text), 1)
        f["cjk"] = f["cjk_ratio"] > self.cfg.cjk_max_ratio
        if self.lang is not None and text.strip():
            p_en = self.lang.prob_english(text)
            f["lang_p_en"] = p_en
            f["non_english"] = p_en < self.cfg.lang_min_p_en
        else:
            f["non_english"] = False
        # F4 degeneration (Gopher subset)
        f["dup_line_frac"] = _dup_line_frac(text)
        deg = (f["dup_line_frac"] > GOPHER["dup_line_frac"]
               or _char_ngram_frac(text, 2) > GOPHER["top_2gram"]
               or _char_ngram_frac(text, 3) > GOPHER["top_3gram"]
               or _char_ngram_frac(text, 4) > GOPHER["top_4gram"])
        f["degenerate"] = bool(deg)
        # F5 exact dup (within teacher)
        h = hashlib.blake2b(re.sub(r"\s+", " ", text.strip().lower()).encode(), digest_size=16).hexdigest()
        f["exact_dup"] = h in self._seen_hashes
        if not f["exact_dup"]:
            self._seen_hashes.add(h)
        # F7 refusal / self-id
        head = text[:300]
        f["refusal"] = bool(_REFUSAL.search(head))
        f["self_id"] = bool(_SELF_ID.search(text))
        # F8 passage reference
        f["passage_ref"] = bool(is_seeded and _PASSAGE_REF.search(text))
        # F9 format validity (STR); F10 MSR final-answer line
        f["format_invalid"] = False
        if str_kind:
            f["format_invalid"] = not _valid_structured(text, str_kind)
        f["missing_answer"] = bool(category == "msr" and self.cfg.require_final_answer_for_msr
                                   and not _FINAL_ANSWER.search(text))
        # F11 contamination (decided by runner via decontam; union-dropped)
        f["contaminated"] = bool(contaminated)
        # F12 length-range compliance (recorded, never dropped)
        f["word_count"] = self._word_count(text)

        drop = any(f.get(k) for k in ("truncated", "too_short", "non_english", "cjk", "degenerate",
                                      "exact_dup", "refusal", "self_id", "passage_ref",
                                      "format_invalid", "missing_answer", "contaminated"))
        return FilterResult(retained=not drop, flags=f)


def _valid_structured(text: str, kind: str) -> bool:
    t = text.strip()
    if kind == "json":
        import json
        # Extract the first {...} or [...] block leniently.
        start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
        if start < 0:
            return False
        try:
            json.loads(t[start:])
            return True
        except Exception:
            return False
    if kind == "table":
        lines = [ln for ln in t.splitlines() if ln.strip().startswith("|")]
        return len(lines) >= 2 and any(set(ln) <= set("|-: ") for ln in lines)
    if kind == "list":
        items = [ln for ln in t.splitlines() if re.match(r"\s*\d+[.)]\s+", ln)]
        return len(items) >= 1
    return True


def filter_stats(results: list[FilterResult]) -> dict[str, Any]:
    """Aggregate per-filter drop counts for the per-teacher covariate table."""
    n = len(results)
    keys = ["truncated", "too_short", "non_english", "cjk", "degenerate", "exact_dup",
            "refusal", "self_id", "passage_ref", "format_invalid", "missing_answer", "contaminated"]
    counts = {k: sum(1 for r in results if r.flags.get(k)) for k in keys}
    retained = sum(1 for r in results if r.retained)
    counts.update(n_offered=n, n_retained=retained, yield_frac=(retained / n if n else 0.0))
    return counts
