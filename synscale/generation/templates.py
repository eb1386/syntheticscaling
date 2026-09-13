"""Frozen prompt-side templates: system prompt, instruction suffixes, rendering, hashes.

Everything in this module is part of the *invariant* protocol shared by every teacher
condition (spec §0, §1.6, §1.7 with locked-methodology Amendments A and B):

* ``SYSTEM_PROMPT`` — the single system prompt supplied to every teacher for every prompt.
* ``SUFFIXES`` — the verbatim per-category instruction suffixes (format + soft length).
* ``render_student_example`` — the student-visible rendering with Alpaca-style markers
  (Amendment B): ``<s>### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}</s>``.
* SHA-256 helpers so that the exact strings are recorded in every generation manifest and
  every student training config.

Changing any string here is a protocol change and must bump ``SUFFIX_SET_VERSION`` /
``RENDER_VERSION``; the fingerprints below make silent edits detectable.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

SUFFIX_SET_VERSION = "suffix-v1.0"
RENDER_VERSION = "render-v1.0"
SYSTEM_PROMPT_ID = "sys_v1"

# --------------------------------------------------------------------------------------
# System prompt (spec §1.6, verbatim)
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Write a clear, accurate, self-contained response in English.\n"
    "Follow the length and format instructions exactly. Do not ask questions back, do not add\n"
    "disclaimers, and do not mention these instructions or any background text you were given."
)


# --------------------------------------------------------------------------------------
# Categories, design shares, source mixture (spec Table 1.2)
# --------------------------------------------------------------------------------------
class Category(str, Enum):
    """The seven prompt categories of the pool."""

    IF = "IF"  # instruction following with verifiable constraints
    SAQA = "SAQA"  # short-answer QA
    CS = "CS"  # commonsense / everyday reasoning
    SCI = "SCI"  # science explanation
    MSR = "MSR"  # multi-step reasoning with a final answer line
    EXP = "EXP"  # explanation + answer (textbook style)
    STR = "STR"  # structured output (JSON / table / list)


CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)

#: Design prompt shares (fraction of prompts in the pool).
DESIGN_PROMPT_SHARES: dict[str, float] = {
    "IF": 0.15,
    "SAQA": 0.15,
    "CS": 0.15,
    "SCI": 0.20,
    "MSR": 0.15,
    "EXP": 0.12,
    "STR": 0.08,
}

#: Design token shares (Table 1.2 "implied token share"), normalised to sum to 1. These are
#: the *prior*; the pilot fixes s_c^tok on realised total rendered lengths (Amendment A).
_RAW_TOKEN_SHARES = {"IF": 13.2, "SAQA": 4.7, "CS": 9.4, "SCI": 25.1, "MSR": 18.9, "EXP": 22.6, "STR": 6.0}
DESIGN_TOKEN_SHARES: dict[str, float] = {
    k: v / sum(_RAW_TOKEN_SHARES.values()) for k, v in _RAW_TOKEN_SHARES.items()
}

#: Source codes: (a) templated, (b) seeded, (c) harvested, (d) writer.
SOURCE_TEMPLATED = "templated"
SOURCE_SEEDED = "seeded"
SOURCE_HARVESTED = "harvested"
SOURCE_WRITER = "writer"
SOURCES: tuple[str, ...] = (SOURCE_TEMPLATED, SOURCE_SEEDED, SOURCE_HARVESTED, SOURCE_WRITER)

#: Per-category source mixture (Table 1.2 "Source mix").
SOURCE_MIX: dict[str, dict[str, float]] = {
    "IF": {SOURCE_WRITER: 0.60, SOURCE_HARVESTED: 0.30, SOURCE_TEMPLATED: 0.10},
    "SAQA": {SOURCE_TEMPLATED: 0.50, SOURCE_SEEDED: 0.30, SOURCE_HARVESTED: 0.20},
    "CS": {SOURCE_WRITER: 0.60, SOURCE_TEMPLATED: 0.40},
    "SCI": {SOURCE_SEEDED: 0.50, SOURCE_TEMPLATED: 0.40, SOURCE_WRITER: 0.10},
    "MSR": {SOURCE_TEMPLATED: 0.70, SOURCE_WRITER: 0.30},
    "EXP": {SOURCE_SEEDED: 0.70, SOURCE_TEMPLATED: 0.30},
    "STR": {SOURCE_TEMPLATED: 0.50, SOURCE_WRITER: 0.50},
}

#: Audience conditioning values (spec §1.4) with per-category sampling weights.
AUDIENCES: tuple[str, ...] = (
    "young children",
    "middle-school students",
    "high-school students",
    "college students",
    "general adult readers",
    "professionals",
)
AUDIENCE_WEIGHTS: dict[str, tuple[float, ...]] = {
    "SCI": (0.10, 0.25, 0.30, 0.20, 0.10, 0.05),
    "EXP": (0.10, 0.20, 0.25, 0.20, 0.20, 0.05),
    "default": (0.05, 0.15, 0.25, 0.25, 0.25, 0.05),
}

# --------------------------------------------------------------------------------------
# Instruction suffixes (spec §1.6, verbatim; ``{}`` slots filled from the pool record)
# --------------------------------------------------------------------------------------
SUFFIXES: dict[str, str] = {
    "IF": (
        "Your response must satisfy all of the following: {constraint_1}; {constraint_2}. "
        "Keep it between 80 and 200 words unless a constraint says otherwise."
    ),
    "SAQA": "Answer in at most 60 words. Give the answer first, then at most two sentences of justification.",
    "SAQA_MCQ": (
        "Answer in at most 60 words. Give the answer first, then at most two sentences of justification. "
        "Start with the letter of the correct option."
    ),
    "CS": (
        "Answer in at most 120 words. State which option or outcome is more plausible and explain why "
        "in everyday terms."
    ),
    "SCI": "Answer in 120 to 250 words at the level of {audience}. Name the key principle or mechanism.",
    "MSR": (
        "Solve step by step in at most 250 words. Put the final answer alone on the last line in the "
        'form "Final answer: <answer>".'
    ),
    "EXP": (
        "Write a self-contained explanation of 200 to 400 words for {audience}. Do not refer to any "
        'background text. End with one sentence beginning "Answer:" that answers: {question}'
    ),
    "STR_JSON": (
        "Respond with valid JSON only, matching this schema exactly: {schema}. No text before or after "
        "the JSON."
    ),
    "STR_TABLE": "Respond with a Markdown table only, with columns {cols} and exactly {N} rows.",
    "STR_LIST": "Respond with a numbered list of exactly {N} items, one sentence each, and nothing else.",
}

#: Soft word ranges implied by each suffix (F12 length-range compliance). ``None`` = unbounded.
SUFFIX_WORD_RANGES: dict[str, tuple[Optional[int], Optional[int]]] = {
    "IF": (80, 200),
    "SAQA": (None, 60),
    "SAQA_MCQ": (None, 60),
    "CS": (None, 120),
    "SCI": (120, 250),
    "MSR": (None, 250),
    "EXP": (200, 400),
    "STR_JSON": (None, None),
    "STR_TABLE": (None, None),
    "STR_LIST": (None, None),
}

#: Background wrapper for passage-seeded prompts (teacher-only; the student never sees it).
BACKGROUND_TEMPLATE = (
    'Background (for your reference only; do not quote or mention it):\n"""{passage}"""\n\n{task}\n{suffix}'
)

# --------------------------------------------------------------------------------------
# Student-visible rendering (Amendment B)
# --------------------------------------------------------------------------------------
BOS = "<s>"
EOS = "</s>"
INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "### Response:\n"
STUDENT_EXAMPLE_TEMPLATE = BOS + INSTRUCTION_MARKER + "{prompt}\n\n" + RESPONSE_MARKER + "{response}" + EOS


def suffix_key(category: str, *, mcq: bool = False, str_kind: Optional[str] = None) -> str:
    """Return the SUFFIXES key for a category (+ SAQA MCQ variant / STR kind)."""
    category = str(category)
    if category == "SAQA":
        return "SAQA_MCQ" if mcq else "SAQA"
    if category == "STR":
        if str_kind not in ("json", "table", "list"):
            raise ValueError(f"STR prompts need str_kind in json|table|list, got {str_kind!r}")
        return f"STR_{str_kind.upper()}"
    if category not in SUFFIXES:
        raise KeyError(f"unknown category {category!r}")
    return category


def render_suffix(key: str, **slots: Any) -> str:
    """Fill the ``{}`` slots of a suffix template.

    For IF the ``constraints`` slot (list of rendered constraint strings, 1–2 items) is joined
    into the ``{constraint_1}; {constraint_2}`` positions; other keys take their named slots.
    """
    template = SUFFIXES[key]
    if key == "IF":
        constraints = list(slots.get("constraints") or [])
        if not 1 <= len(constraints) <= 2:
            raise ValueError("IF suffix needs 1 or 2 constraints")
        joined = "; ".join(constraints)
        return template.replace("{constraint_1}; {constraint_2}", joined)
    if key == "STR_JSON":
        schema = slots["schema"]
        if not isinstance(schema, str):
            schema = json.dumps(schema, sort_keys=True, separators=(", ", ": "))
        return template.format(schema=schema)
    if key == "STR_TABLE":
        cols = slots["cols"]
        if not isinstance(cols, str):
            cols = ", ".join(cols)
        return template.format(cols=cols, N=slots["N"])
    return template.format(**slots)


def render_student_prompt(task: str, suffix: str) -> str:
    """Student-visible prompt: ``{task}\\n{suffix}`` (passage removed for seeded prompts)."""
    return f"{task}\n{suffix}"


def render_user_turn(task: str, suffix: str, passage: Optional[str] = None) -> str:
    """Exact user turn sent to the teacher (passage wrapped as background for seeded prompts)."""
    if passage:
        return BACKGROUND_TEMPLATE.format(passage=passage, task=task, suffix=suffix)
    return render_student_prompt(task, suffix)


def build_messages(user_turn: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Chat messages (system + user) for the teacher; recorded verbatim in the JSONL for replay."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_turn}]


def render_student_example(prompt: str, response: str) -> str:
    """One student training document (Amendment B). Token accounting covers all of it (Amendment A)."""
    return STUDENT_EXAMPLE_TEMPLATE.format(prompt=prompt, response=response)


def render_student_body(prompt: str, response: str) -> str:
    """The rendered example without BOS/EOS; used by token counters that add BOS/EOS as specials."""
    return INSTRUCTION_MARKER + prompt + "\n\n" + RESPONSE_MARKER + response


# --------------------------------------------------------------------------------------
# Hashing / versioning
# --------------------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


SYSTEM_PROMPT_SHA256 = sha256_text(SYSTEM_PROMPT)


def suffix_set_sha256() -> str:
    """Fingerprint of the whole suffix set (keys + strings), recorded in manifests."""
    return sha256_json(SUFFIXES)


def render_template_sha256() -> str:
    return sha256_text(STUDENT_EXAMPLE_TEMPLATE)


def template_fingerprint() -> dict[str, str]:
    """All protocol-string hashes for a manifest."""
    return {
        "system_prompt_id": SYSTEM_PROMPT_ID,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "suffix_set_version": SUFFIX_SET_VERSION,
        "suffix_set_sha256": suffix_set_sha256(),
        "render_version": RENDER_VERSION,
        "render_template_sha256": render_template_sha256(),
        "background_template_sha256": sha256_text(BACKGROUND_TEMPLATE),
    }


def normalise_shares(shares: Mapping[str, float], keys: Iterable[str] = CATEGORIES) -> dict[str, float]:
    """Return shares restricted to ``keys`` and normalised to sum to one."""
    keys = list(keys)
    total = float(sum(shares.get(k, 0.0) for k in keys))
    if total <= 0:
        raise ValueError("shares must have positive mass")
    return {k: float(shares.get(k, 0.0)) / total for k in keys}


def largest_remainder_counts(shares: Mapping[str, float], n: int) -> dict[str, int]:
    """Integer allocation of ``n`` items to keys proportional to ``shares`` (Hamilton rounding)."""
    shares = normalise_shares(shares, shares.keys())
    raw = {k: shares[k] * n for k in shares}
    counts = {k: int(raw[k]) for k in shares}
    rest = n - sum(counts.values())
    order = sorted(shares, key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:rest]:
        counts[k] += 1
    return counts
