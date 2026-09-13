"""Frozen prompt pool: record schema, builders, ordering, dedup, freezing (spec §1).

Pool record schema (one JSON object per line; field order fixed by :meth:`PromptRecord.to_json`)::

    prompt_id      zero-padded 10-digit position in the frozen order
    category       IF | SAQA | CS | SCI | MSR | EXP | STR
    subcategory    "domain.subdomain" (taxonomy) or a source-specific label
    leaf           taxonomy leaf title (templated / seeded / writer prompts)
    source         templated | seeded | harvested:<name> | writer
    license        per-item license string
    seed_doc_id, persona_id, audience   conditioning provenance (nullable)
    task           the task text shown to the student (passage removed)
    suffix         the rendered instruction suffix (spec §1.6)
    passage        teacher-only background text (seeded prompts) — never student-visible
    verifiable     True if a programmatic checker exists (MSR gold, STR format, SAQA-MCQ letter)
    gold           gold answer / expected content (nullable)
    format_spec    STR: {"kind": json|table|list, "schema" | "cols" + "n" | "n"}
    constraints    IF: list of {"type", "value"} constraint dicts
    mcq            SAQA multiple-choice variant flag
    meta           free-form provenance (template id, msr params, ...)

Builders: (a) :class:`TemplatedPromptGenerator` over a :class:`Taxonomy`, (b)
:class:`PassageSeededPromptGenerator`, (c) :class:`HarvestedPromptIngestor` with the §1.3
filters, (d) :class:`WriterPromptGenerator` over an abstract :class:`WriterBackend`.
:class:`PoolBuilder` composes them by the §1.2 mixture, deduplicates (exact + MinHash),
decontaminates, applies per-leaf caps, orders the pool as a seeded stratified interleave
(any prefix is a stratified sample) and freezes it to JSONL + sha256 + manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from synscale.generation.decontam import NGramIndex, normalise
from synscale.generation.langid import HeuristicEnglishDetector, LanguageDetector
from synscale.generation.minhash import MinHashConfig, MinHashIndex
from synscale.generation.msr_templates import MSR_TEMPLATE_VERSION, MSR_TEMPLATES, instantiate_msr
from synscale.generation.taxonomy import Leaf, Taxonomy, leaf_counts, leaf_entropy
from synscale.generation.templates import (
    AUDIENCE_WEIGHTS,
    AUDIENCES,
    CATEGORIES,
    DESIGN_PROMPT_SHARES,
    SOURCE_HARVESTED,
    SOURCE_MIX,
    SOURCE_SEEDED,
    SOURCE_TEMPLATED,
    SOURCE_WRITER,
    build_messages,
    largest_remainder_counts,
    normalise_shares,
    render_student_prompt,
    render_suffix,
    render_user_turn,
    suffix_key,
    template_fingerprint,
)

POOL_SCHEMA_VERSION = "pool-schema-v1"
PROMPT_ID_WIDTH = 10
OURS_LICENSE = "ours (Apache-2.0)"


# ======================================================================================
# Record
# ======================================================================================
@dataclass
class PromptRecord:
    prompt_id: str
    category: str
    subcategory: str
    source: str
    license: str
    task: str
    suffix: str
    leaf: Optional[str] = None
    seed_doc_id: Optional[str] = None
    persona_id: Optional[str] = None
    audience: Optional[str] = None
    passage: Optional[str] = None
    verifiable: bool = False
    gold: Any = None
    format_spec: Optional[dict] = None
    constraints: list[dict] = field(default_factory=list)
    mcq: bool = False
    meta: dict = field(default_factory=dict)

    # ---- rendering -----------------------------------------------------------------------
    def student_prompt(self) -> str:
        return render_student_prompt(self.task, self.suffix)

    def user_turn(self) -> str:
        return render_user_turn(self.task, self.suffix, self.passage)

    def messages(self) -> list[dict[str, str]]:
        return build_messages(self.user_turn())

    @property
    def source_family(self) -> str:
        """``harvested:wildchat`` → ``harvested``."""
        return self.source.split(":", 1)[0]

    @property
    def str_kind(self) -> Optional[str]:
        return (self.format_spec or {}).get("kind")

    # ---- serialisation -------------------------------------------------------------------
    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PromptRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__} | {
            "constraints": list(d.get("constraints") or []),
            "meta": dict(d.get("meta") or {}),
            "verifiable": bool(d.get("verifiable", False)),
            "mcq": bool(d.get("mcq", False)),
        })

    def dedup_key(self) -> str:
        return hashlib.sha1(normalise(self.task).encode("utf-8")).hexdigest()


def format_prompt_id(position: int) -> str:
    return f"{position:0{PROMPT_ID_WIDTH}d}"


# ======================================================================================
# Shared banks
# ======================================================================================
def _pick_audience(category: str, rng: random.Random) -> str:
    w = AUDIENCE_WEIGHTS.get(category, AUDIENCE_WEIGHTS["default"])
    return rng.choices(AUDIENCES, weights=w, k=1)[0]


_IF_WORDS = ["balance", "pattern", "careful", "future", "simple", "energy", "history", "practice",
             "journey", "curious", "signal", "quiet"]
_IF_END_PHRASES = ["Is there anything else I can help with?", "That is all.", "End of note.",
                   "Thank you for reading.", "Any other questions?"]


def sample_if_constraints(rng: random.Random, k: Optional[int] = None) -> list[dict]:
    """1–2 IFEval-style constraints of distinct types."""
    k = k or rng.choice([1, 2, 2])
    types = rng.sample(["exact_bullets", "include_word", "max_words", "all_lowercase", "end_with"], k)
    out = []
    for t in types:
        if t == "exact_bullets":
            out.append({"type": t, "value": rng.choice([2, 3, 4, 5])})
        elif t == "include_word":
            out.append({"type": t, "value": rng.choice(_IF_WORDS)})
        elif t == "max_words":
            out.append({"type": t, "value": rng.choice([60, 80, 100, 120, 150])})
        elif t == "all_lowercase":
            out.append({"type": t, "value": None})
        else:
            out.append({"type": t, "value": rng.choice(_IF_END_PHRASES)})
    return out


def render_constraint(c: Mapping[str, Any]) -> str:
    t, v = c["type"], c.get("value")
    if t == "exact_bullets":
        return f"use exactly {v} bullet points, each on its own line starting with '- '"
    if t == "include_word":
        return f'include the word "{v}"'
    if t == "max_words":
        return f"use at most {v} words"
    if t == "all_lowercase":
        return "write everything in lowercase letters, with no capital letters at all"
    if t == "end_with":
        return f'end your response with the exact phrase "{v}"'
    raise KeyError(f"unknown constraint type {t}")


def if_suffix(constraints: Sequence[Mapping[str, Any]]) -> str:
    return render_suffix("IF", constraints=[render_constraint(c) for c in constraints])


_IF_TASKS = [
    "Write a short note explaining {leaf} to a friend who has never heard of it.",
    "Describe how a beginner could learn the basics of {leaf} in one week.",
    "Give practical advice to someone starting a project related to {leaf}.",
    "Write a brief announcement for a community talk about {leaf}.",
    "Summarise why {leaf} is worth knowing about, for a general audience.",
    "Write a short encouraging message to a student struggling with {leaf}.",
]

_SAQA_TASKS = [
    ("What is {leaf}?", None),
    ("In one or two sentences, define {leaf}.", None),
    ("Which subfield of {domain_h} does {leaf} belong to?", "subdomain"),
    ("Name the broader area of study that includes {leaf}.", "domain"),
    ("Give one key fact about {leaf}.", None),
]

_CS_PLAUSIBLE = [  # (A, B, more plausible)
    ("{name} leaves an ice cream cone in the sun and it melts", "{name} leaves an ice cream cone in the sun and it freezes", "A"),
    ("{name} drops a glass on a tiled floor and it shatters", "{name} drops a glass on a tiled floor and it bounces back intact", "A"),
    ("{name} waters a houseplant every week and it stays healthy", "{name} never waters a houseplant and it grows faster", "A"),
    ("{name} forgets an umbrella in heavy rain and gets soaked", "{name} forgets an umbrella in heavy rain and stays completely dry", "A"),
    ("{name} runs up ten flights of stairs and feels rested", "{name} runs up ten flights of stairs and feels out of breath", "B"),
    ("{name} puts a metal spoon in the freezer and it becomes warm", "{name} puts a metal spoon in the freezer and it becomes cold", "B"),
    ("{name} studies all week and does well on the test", "{name} skips studying entirely and knows every answer", "A"),
    ("{name} leaves bread out for a month and it stays fresh", "{name} leaves bread out for a month and it grows mould", "B"),
]
_CS_WHAT_IF = [
    "What happens if {name} leaves a bicycle outside in the rain for a whole year?",
    "What happens if {name} adds too much salt while cooking soup?",
    "What happens if {name} forgets to charge a phone overnight before a long trip?",
    "What happens if {name} plants seeds but never lets sunlight reach them?",
    "What happens if {name} stacks heavy boxes on top of a cardboard box?",
    "What is the most likely reason {name}'s car will not start on a very cold morning?",
    "What is the most likely reason the lights went out in {name}'s whole street at once?",
]
_CS_GOAL = [  # (goal, option 1, option 2, better option)
    ("keep a drink cold on a picnic", "wrap it in a wet towel and keep it in the shade", "leave it in direct sunlight", "1"),
    ("open a stuck jar lid", "run the lid under hot water and then twist", "put the jar in the freezer and twist immediately", "1"),
    ("dry wet shoes overnight", "stuff them with newspaper and leave them in a warm room", "seal them in a plastic bag", "1"),
    ("remember a shopping list", "write it down before leaving home", "try to memorise it while driving", "1"),
    ("stop a small kitchen grease fire", "pour water on it", "cover the pan with a lid", "2"),
    ("carry groceries up several floors", "make two trips with a bag in each hand", "carry all bags on one finger", "1"),
]
_CS_COMPLETION = [  # (sentence with blank, option A, option B, answer)
    ("{name} could not lift the suitcase because the ___ was too heavy.", "suitcase", "handle", "A"),
    ("The ice cream melted faster than the frozen peas because the ___ was smaller.", "ice cream", "freezer", "A"),
    ("{name} put on a coat before going outside because the ___ was cold.", "weather", "coat", "A"),
    ("The trophy did not fit in the suitcase because the ___ was too small.", "trophy", "suitcase", "B"),
    ("{name} whispered in the library because the ___ was quiet.", "street", "library", "B"),
]
_CS_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Avery", "Quinn"]

_SCI_TASKS = [
    "Explain why {leaf} matters in {subdomain_h}.",
    "What would happen to a system that relies on {leaf} if its key conditions were suddenly changed?",
    "Which principle or mechanism best explains {leaf}, and how does it work?",
    "Describe a simple everyday observation that can be explained by {leaf}.",
    "Explain the main cause-and-effect relationship involved in {leaf}.",
]
_EXP_QUESTIONS = [
    "What is the single most important idea to understand about {leaf}?",
    "Why does {leaf} matter for {subdomain_h}?",
    "What is one common misconception about {leaf}?",
]
_STR_JSON_PROPS = [
    ("topic", {"type": "string"}),
    ("summary", {"type": "string"}),
    ("field", {"type": "string"}),
    ("difficulty", {"type": "string", "enum": ["beginner", "intermediate", "advanced"]}),
    ("year_range", {"type": "string"}),
]
_STR_TABLE_COLS = [["Term", "Definition"], ["Concept", "Example", "Why it matters"],
                   ["Step", "Description"], ["Question", "Short answer"]]


# ======================================================================================
# (a) Templated generator
# ======================================================================================
class PromptGenerator(Protocol):
    source: str

    def generate(self, category: str, n: int, rng: random.Random) -> list[PromptRecord]:  # pragma: no cover
        ...


class TemplatedPromptGenerator:
    """Templated prompts over a taxonomy: SAQA (+MCQ), CS bank, SCI, EXP, IF, STR, and MSR."""

    source = SOURCE_TEMPLATED

    def __init__(self, taxonomy: Taxonomy, license: str = OURS_LICENSE, mcq_fraction: float = 0.3):
        self.taxonomy = taxonomy
        self.license = license
        self.mcq_fraction = mcq_fraction

    def _leaf_cycle(self, rng: random.Random, science_only: bool = False) -> Iterator[Leaf]:
        leaves = list(self.taxonomy.science_leaves() if science_only else self.taxonomy.leaves)
        while True:
            rng.shuffle(leaves)
            yield from leaves

    def _base(self, category: str, leaf: Optional[Leaf], task: str, suffix: str, **kw: Any) -> PromptRecord:
        return PromptRecord(
            prompt_id="", category=category, subcategory=leaf.subcategory if leaf else kw.pop("subcategory", "bank"),
            source=self.source, license=self.license, task=task, suffix=suffix,
            leaf=leaf.key if leaf else None, **kw,
        )

    def generate(self, category: str, n: int, rng: random.Random) -> list[PromptRecord]:
        fn = {
            "IF": self._gen_if, "SAQA": self._gen_saqa, "CS": self._gen_cs, "SCI": self._gen_sci,
            "MSR": self._gen_msr, "EXP": self._gen_exp, "STR": self._gen_str,
        }[category]
        return [fn(rng, leaves) for leaves in [self._leaf_cycle(rng, science_only=category in ("SCI", "EXP"))]
                for _ in range(n)]

    def _gen_if(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        leaf = next(leaves)
        cons = sample_if_constraints(rng)
        task = rng.choice(_IF_TASKS).format(leaf=leaf.name)
        return self._base("IF", leaf, task, if_suffix(cons), constraints=cons, verifiable=True,
                          gold={"constraints": cons})

    def _gen_saqa(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        leaf = next(leaves)
        if rng.random() < self.mcq_fraction:
            others = [lf for lf in self.taxonomy.leaves if lf.domain != leaf.domain]
            distractors = rng.sample(others, 3) if len(others) >= 3 else rng.sample(self.taxonomy.siblings(leaf), 3)
            options = [leaf] + distractors
            rng.shuffle(options)
            letters = "ABCD"
            gold_letter = letters[options.index(leaf)]
            opts = " ".join(f"({letters[i]}) {o.name}" for i, o in enumerate(options))
            task = f"Which of the following is a topic in {leaf.subdomain_h}? {opts}"
            return self._base("SAQA", leaf, task, render_suffix("SAQA_MCQ"), mcq=True, verifiable=True,
                              gold={"letter": gold_letter, "text": leaf.name})
        template, gold_kind = rng.choice(_SAQA_TASKS)
        task = template.format(leaf=leaf.name, domain_h=leaf.domain_h)
        gold = None
        if gold_kind == "subdomain":
            gold = {"text": leaf.subdomain_h}
        elif gold_kind == "domain":
            gold = {"text": leaf.domain_h}
        return self._base("SAQA", leaf, task, render_suffix("SAQA"), verifiable=gold is not None, gold=gold)

    def _gen_cs(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        name = rng.choice(_CS_NAMES)
        kind = rng.choice(["plausible", "what_if", "goal", "completion"])
        if kind == "plausible":
            a, b, ans = rng.choice(_CS_PLAUSIBLE)
            task = f"Which is more plausible: (A) {a.format(name=name)}, or (B) {b.format(name=name)}? Why?"
            gold = {"letter": ans}
        elif kind == "what_if":
            task = rng.choice(_CS_WHAT_IF).format(name=name)
            gold = None
        elif kind == "goal":
            goal, o1, o2, ans = rng.choice(_CS_GOAL)
            task = f"To {goal}, would {name} do better to (1) {o1}, or (2) {o2}? Explain."
            gold = {"option": ans}
        else:
            sent, a, b, ans = rng.choice(_CS_COMPLETION)
            task = f"Fill in the blank: \"{sent.format(name=name)}\" Option A: {a}. Option B: {b}."
            gold = {"letter": ans}
        return self._base("CS", None, task, render_suffix("CS"), subcategory=f"commonsense.{kind}",
                          verifiable=False, gold=gold, meta={"cs_kind": kind})

    def _gen_sci(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        leaf = next(leaves)
        audience = _pick_audience("SCI", rng)
        task = rng.choice(_SCI_TASKS).format(leaf=leaf.name, subdomain_h=leaf.subdomain_h)
        return self._base("SCI", leaf, task, render_suffix("SCI", audience=audience), audience=audience)

    def _gen_exp(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        leaf = next(leaves)
        audience = _pick_audience("EXP", rng)
        question = rng.choice(_EXP_QUESTIONS).format(leaf=leaf.name, subdomain_h=leaf.subdomain_h)
        task = f"Explain {leaf.name}."
        return self._base("EXP", leaf, task, render_suffix("EXP", audience=audience, question=question),
                          audience=audience, meta={"question": question})

    def _gen_msr(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        inst = instantiate_msr(rng)
        return self._base("MSR", None, inst.task, render_suffix("MSR"), subcategory=f"msr.{inst.template_id}",
                          verifiable=True, gold={"answer": inst.gold},
                          meta={"template_id": inst.template_id, "params": inst.params,
                                "template_version": MSR_TEMPLATE_VERSION})

    def _gen_str(self, rng: random.Random, leaves: Iterator[Leaf]) -> PromptRecord:
        leaf = next(leaves)
        kind = rng.choice(["json", "table", "list"])
        if kind == "json":
            props = dict(rng.sample(_STR_JSON_PROPS, rng.choice([2, 3])))
            n_terms = rng.choice([2, 3, 4, 5])
            props["key_terms"] = {"type": "array", "items": {"type": "string"}, "minItems": n_terms, "maxItems": n_terms}
            schema = {"type": "object", "properties": props, "required": sorted(props), "additionalProperties": False}
            task = f"Describe the topic \"{leaf.name}\" as a JSON object."
            spec = {"kind": "json", "schema": schema}
            suffix = render_suffix("STR_JSON", schema=schema)
        elif kind == "table":
            cols = rng.choice(_STR_TABLE_COLS)
            n = rng.choice([3, 4, 5, 6])
            task = f"Give {n} important points about {leaf.name}."
            spec = {"kind": "table", "cols": cols, "n": n}
            suffix = render_suffix("STR_TABLE", cols=cols, N=n)
        else:
            n = rng.choice([3, 4, 5, 6, 7])
            task = f"List {n} facts about {leaf.name}."
            spec = {"kind": "list", "n": n}
            suffix = render_suffix("STR_LIST", N=n)
        return self._base("STR", leaf, task, suffix, format_spec=spec, verifiable=True, gold=spec)


# ======================================================================================
# (b) Passage-seeded generator
# ======================================================================================
@dataclass(frozen=True)
class Passage:
    doc_id: str
    text: str
    topic: Optional[str] = None
    cluster: Optional[int] = None
    license: str = "ODC-BY"


def load_passages(path: str | Path) -> list[Passage]:
    out: list[Passage] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(Passage(str(d["doc_id"]), d["text"], d.get("topic"), d.get("cluster"),
                                   d.get("license", "ODC-BY")))
    return out


class PassageSeededPromptGenerator:
    """Cosmopedia-style prompts conditioned on a background passage the student never sees.

    Interface only: the real pipeline supplies 1.1M cluster-balanced FineWeb-Edu passages with a
    writer-extracted ``topic``; here a passage without a topic falls back to its opening words.
    Each passage is used at most once per category.
    """

    source = SOURCE_SEEDED

    def __init__(self, passages: Sequence[Passage], min_words: int = 20, max_words: int = 400):
        self.passages = [p for p in passages if min_words <= len(p.text.split()) <= max_words]
        self._used: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def topic_of(p: Passage) -> str:
        if p.topic:
            return p.topic
        return " ".join(p.text.split()[:6]).rstrip(",.;:")

    def generate(self, category: str, n: int, rng: random.Random) -> list[PromptRecord]:
        if category not in ("SAQA", "SCI", "EXP"):
            return []
        avail = [p for p in self.passages if p.doc_id not in self._used[category]]
        rng.shuffle(avail)
        out: list[PromptRecord] = []
        for p in avail[:n]:
            topic = self.topic_of(p)
            self._used[category].add(p.doc_id)
            if category == "SAQA":
                task = rng.choice([f"What is the main idea behind {topic}?", f"Briefly, what is {topic}?",
                                   f"State one important fact about {topic}."])
                suffix, audience, meta = render_suffix("SAQA"), None, {}
            elif category == "SCI":
                audience = _pick_audience("SCI", rng)
                task = rng.choice([f"Explain the science behind {topic}.",
                                   f"What principle explains {topic}, and why does it matter?"])
                suffix, meta = render_suffix("SCI", audience=audience), {}
            else:
                audience = _pick_audience("EXP", rng)
                question = f"What is the single most important idea to understand about {topic}?"
                task = f"Explain {topic}."
                suffix, meta = render_suffix("EXP", audience=audience, question=question), {"question": question}
            out.append(PromptRecord(
                prompt_id="", category=category, subcategory=f"seeded.cluster{p.cluster if p.cluster is not None else 'na'}",
                source=self.source, license=p.license, task=task, suffix=suffix, seed_doc_id=p.doc_id,
                audience=audience, passage=p.text, meta={"topic": topic, **meta},
            ))
        return out


# ======================================================================================
# (c) Harvested human prompts with §1.3 filters
# ======================================================================================
URL_RE = re.compile(r"(https?://|www\.)\S+|\b[a-z0-9-]+\.(com|org|net|io|edu|gov)\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
CODE_REQUEST_RE = re.compile(
    r"```|\bwrite (me )?(a|an|the|some)? ?(python|java|javascript|c\+\+|c#|rust|go|sql|bash|html|css|typescript)?\s*"
    r"(function|script|program|class|code|regex|query|snippet)\b|\bimplement\b.*\bin (python|java|c\+\+|javascript|rust|go)\b"
    r"|\b(def |#include|import numpy|SELECT .* FROM|console\.log|public static void)\b|\bunit tests?\b|\bdebug (this|my) code\b",
    re.I,
)
MULTITURN_RE = re.compile(r"(^|\n)\s*(user|assistant|human|ai|system|bot)\s*:", re.I)


@dataclass
class HarvestRejection:
    reason: str


class HarvestedPromptIngestor:
    """Applies the §1.3 filters to harvested human prompts and assigns IF / SAQA categories."""

    source = SOURCE_HARVESTED

    def __init__(self, detector: Optional[LanguageDetector] = None, min_words: int = 8, max_words: int = 400,
                 p_en_min: float = 0.65):
        self.detector = detector or HeuristicEnglishDetector()
        self.min_words, self.max_words, self.p_en_min = min_words, max_words, p_en_min
        self.accepted: dict[str, list[PromptRecord]] = defaultdict(list)
        self.rejections: Counter[str] = Counter()
        self._cursor: dict[str, int] = defaultdict(int)

    def check(self, text: str, turns: int = 1) -> Optional[str]:
        """Return a rejection reason or ``None`` if the prompt passes."""
        words = text.split()
        if not (self.min_words <= len(words) <= self.max_words):
            return "length"
        if turns != 1 or MULTITURN_RE.search(text):
            return "multi_turn"
        if URL_RE.search(text):
            return "url"
        if EMAIL_RE.search(text) or SSN_RE.search(text) or IPV4_RE.search(text) or CARD_RE.search(text) \
                or PHONE_RE.search(text):
            return "pii"
        if CODE_REQUEST_RE.search(text):
            return "code_request"
        lang, p = self.detector.detect(text)
        if lang != "en" or p < self.p_en_min:
            return "non_english"
        return None

    @staticmethod
    def classify(text: str) -> str:
        t = text.strip()
        return "SAQA" if (t.endswith("?") and len(t.split()) <= 40) else "IF"

    def ingest(self, raw: Iterable[Mapping[str, Any]], rng: random.Random) -> int:
        """Ingest dicts with ``text``, ``source``, ``license`` (+ optional ``category``, ``turns``, ``id``)."""
        n_ok = 0
        for d in raw:
            text = str(d.get("text", "")).strip()
            reason = self.check(text, int(d.get("turns", 1)))
            if reason:
                self.rejections[reason] += 1
                continue
            category = d.get("category") or self.classify(text)
            if category not in ("IF", "SAQA"):
                self.rejections["category_not_harvested"] += 1
                continue
            src = f"{self.source}:{d.get('source', 'unknown')}"
            if category == "IF":
                cons = sample_if_constraints(rng)
                rec = PromptRecord(prompt_id="", category="IF", subcategory="harvested.instruction", source=src,
                                   license=str(d.get("license", "unknown")), task=text, suffix=if_suffix(cons),
                                   constraints=cons, verifiable=True, gold={"constraints": cons},
                                   meta={"harvest_id": str(d.get("id", ""))})
            else:
                rec = PromptRecord(prompt_id="", category="SAQA", subcategory="harvested.question", source=src,
                                   license=str(d.get("license", "unknown")), task=text, suffix=render_suffix("SAQA"),
                                   meta={"harvest_id": str(d.get("id", ""))})
            self.accepted[category].append(rec)
            n_ok += 1
        return n_ok

    def generate(self, category: str, n: int, rng: random.Random) -> list[PromptRecord]:
        pool = self.accepted.get(category, [])
        start = self._cursor[category]
        out = pool[start : start + n]
        self._cursor[category] = start + len(out)
        return list(out)


def load_harvested(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ======================================================================================
# (d) Writer-model prompts
# ======================================================================================
@dataclass(frozen=True)
class Persona:
    persona_id: str
    text: str


DEFAULT_PERSONAS: tuple[Persona, ...] = tuple(
    Persona(f"p{i:04d}", t) for i, t in enumerate([
        "a retired schoolteacher who reads popular-science magazines",
        "a first-year nursing student preparing for exams",
        "a small-town librarian organising a children's reading week",
        "a bicycle mechanic curious about how things work",
        "a high-school debate coach",
        "a farmer tracking weather patterns for planting",
        "an amateur astronomer with a backyard telescope",
        "a museum guide who explains exhibits to families",
        "a software tester learning about history in the evenings",
        "a community nurse who volunteers at a science fair",
        "a parent helping a child with homework",
        "a chef interested in the chemistry of cooking",
    ])
)


def load_personas(path: str | Path) -> list[Persona]:
    with open(path, "r", encoding="utf-8") as fh:
        return [Persona(str(d["persona_id"]), d["text"]) for d in (json.loads(l) for l in fh if l.strip())]


class WriterBackend(Protocol):
    """Fixed non-teacher prompt-writer model (Mistral-Small in the study); ``name`` is recorded."""

    name: str

    def write(self, instructions: Sequence[str], seeds: Sequence[int]) -> list[str]:  # pragma: no cover
        ...


class StubWriterBackend:
    """Deterministic writer used in tests: composes a question from persona + leaf + category."""

    name = "stub-writer-v1"

    _FORMS = {
        "IF": "Write a short piece about {leaf} that {persona} could share with friends.",
        "CS": "If {persona} tried to use ideas from {leaf} in daily life, which would work better: a careful plan or a quick guess, and why?",
        "SCI": "As {persona}, how would you explain the mechanism behind {leaf} in simple terms?",
        "MSR": "A person who is {persona} needs {n} items related to {leaf} costing {p} each and has {b} to spend. How much money is left after buying them?",
        "STR": "Summarise {leaf} for {persona}.",
    }

    def write(self, instructions: Sequence[str], seeds: Sequence[int]) -> list[str]:
        out = []
        for ins, seed in zip(instructions, seeds):
            m = re.search(r"CATEGORY=(\w+)\|PERSONA=(.*?)\|LEAF=(.*?)\|", ins)
            if not m:
                out.append(f"Question {seed}: {ins[:60]}")
                continue
            cat, persona, leaf = m.group(1), m.group(2), m.group(3)
            rng = random.Random(seed)
            n, p = rng.randint(2, 9), rng.randint(2, 20)
            out.append(self._FORMS.get(cat, "Ask about {leaf} as {persona}.").format(
                leaf=leaf, persona=persona, n=n, p=p, b=n * p + rng.randint(1, 50)))
        return out


class WriterPromptGenerator:
    """Persona + leaf + category-template conditioned prompts written by the fixed writer model."""

    source = SOURCE_WRITER
    _PROPERTY = {
        "IF": "a clear task with a format or length requirement",
        "CS": "everyday commonsense reasoning about what is plausible",
        "SCI": "a scientific explanation of a mechanism",
        "MSR": "a multi-step numerical solution with a single final answer",
        "STR": "a structured answer",
    }

    def __init__(self, backend: WriterBackend, taxonomy: Taxonomy, personas: Sequence[Persona] = DEFAULT_PERSONAS,
                 license: str = OURS_LICENSE, base_seed: int = 0):
        self.backend, self.taxonomy, self.personas, self.license, self.base_seed = backend, taxonomy, list(personas), license, base_seed

    def instruction(self, category: str, persona: Persona, leaf: Leaf) -> str:
        return (f"CATEGORY={category}|PERSONA={persona.text}|LEAF={leaf.name}| "
                f"You are {persona.text}. Write one question this person might ask about {leaf.name} that "
                f"requires {self._PROPERTY.get(category, 'a thoughtful answer')}. Output only the question.")

    def generate(self, category: str, n: int, rng: random.Random) -> list[PromptRecord]:
        if category not in self._PROPERTY:
            return []
        leaves = list(self.taxonomy.science_leaves() if category == "SCI" else self.taxonomy.leaves)
        picks = [(rng.choice(self.personas), rng.choice(leaves)) for _ in range(n)]
        seeds = [(self.base_seed * 1_000_003 + rng.getrandbits(31)) % (1 << 31) for _ in range(n)]
        texts = self.backend.write([self.instruction(category, p, lf) for p, lf in picks], seeds)
        out = []
        for (persona, leaf), text, seed in zip(picks, texts, seeds):
            task = text.strip()
            kw: dict[str, Any] = {"persona_id": persona.persona_id, "leaf": leaf.key, "meta": {"writer": self.backend.name, "writer_seed": seed}}
            if category == "IF":
                cons = sample_if_constraints(rng)
                suffix = if_suffix(cons)
                kw.update(constraints=cons, verifiable=True, gold={"constraints": cons})
            elif category == "STR":
                kind = rng.choice(["json", "table", "list"])
                if kind == "json":
                    schema = {"type": "object", "properties": {"topic": {"type": "string"}, "summary": {"type": "string"}},
                              "required": ["summary", "topic"], "additionalProperties": False}
                    spec, suffix = {"kind": "json", "schema": schema}, render_suffix("STR_JSON", schema=schema)
                elif kind == "table":
                    cols, nrows = rng.choice(_STR_TABLE_COLS), rng.choice([3, 4, 5])
                    spec, suffix = {"kind": "table", "cols": cols, "n": nrows}, render_suffix("STR_TABLE", cols=cols, N=nrows)
                else:
                    k = rng.choice([3, 4, 5, 6])
                    spec, suffix = {"kind": "list", "n": k}, render_suffix("STR_LIST", N=k)
                kw.update(format_spec=spec, verifiable=True, gold=spec)
            elif category == "SCI":
                audience = _pick_audience("SCI", rng)
                suffix = render_suffix("SCI", audience=audience)
                kw["audience"] = audience
            else:
                suffix = render_suffix(suffix_key(category))
            out.append(PromptRecord(prompt_id="", category=category, subcategory=leaf.subcategory, source=self.source,
                                    license=self.license, task=task, suffix=suffix, **kw))
        return out


# ======================================================================================
# Ordering, dedup, diversity
# ======================================================================================
def stratified_interleave(groups: Mapping[Any, Sequence[Any]], weights: Optional[Mapping[Any, float]] = None) -> list:
    """Deterministic interleave such that every prefix has group proportions ≈ ``weights``.

    At each step the non-exhausted group with the smallest ``(taken + 1) / weight`` is drawn
    (smooth weighted round-robin). Groups are visited in sorted key order on ties.
    """
    keys = sorted(groups, key=str)
    w = {k: float(weights[k]) if weights and weights.get(k, 0) > 0 else 0.0 for k in keys} if weights else {}
    if not weights:
        w = {k: float(len(groups[k])) for k in keys}
    w = {k: v for k, v in w.items() if v > 0 and len(groups[k]) > 0}
    pos = {k: 0 for k in keys}
    out: list = []
    total = sum(len(groups[k]) for k in w)
    for _ in range(total):
        k_best = min((k for k in w if pos[k] < len(groups[k])), key=lambda k: ((pos[k] + 1) / w[k], str(k)))
        out.append(groups[k_best][pos[k_best]])
        pos[k_best] += 1
    return out


def distinct_n(texts: Iterable[str], n: int = 4) -> float:
    total, seen = 0, set()
    for t in texts:
        words = normalise(t).split()
        for i in range(len(words) - n + 1):
            total += 1
            seen.add(" ".join(words[i : i + n]))
    return (len(seen) / total) if total else 0.0


class _CleanState:
    """Exact + MinHash dedup, decontamination and leaf-cap bookkeeping across build rounds."""

    def __init__(self, minhash_cfg: MinHashConfig, contam_index: Optional[NGramIndex], leaf_cap: int):
        self.exact: set[str] = set()
        self.index = MinHashIndex(minhash_cfg)
        self.contam = contam_index
        self.leaf_cap = leaf_cap
        self.leaf_counter: Counter[str] = Counter()
        self.drops: Counter[str] = Counter()
        self.n_seen = 0

    def accept(self, rec: PromptRecord) -> bool:
        self.n_seen += 1
        key = rec.dedup_key()
        if key in self.exact:
            self.drops["exact_dup"] += 1
            return False
        if self.index.query(rec.task) is not None:
            self.drops["near_dup"] += 1
            return False
        if self.contam is not None and (self.contam.flag(rec.task) or (rec.passage and self.contam.flag(rec.passage))):
            self.drops["contaminated"] += 1
            return False
        if rec.leaf and self.leaf_counter[rec.leaf] >= self.leaf_cap:
            self.drops["leaf_cap"] += 1
            return False
        self.exact.add(key)
        self.index.add(key, rec.task)
        if rec.leaf:
            self.leaf_counter[rec.leaf] += 1
        return True


@dataclass
class PoolBuildSpec:
    n_prompts: int
    seed: int = 0
    version: str = "v1.0"
    prompt_shares: Mapping[str, float] = field(default_factory=lambda: dict(DESIGN_PROMPT_SHARES))
    source_mix: Mapping[str, Mapping[str, float]] = field(default_factory=lambda: {k: dict(v) for k, v in SOURCE_MIX.items()})
    leaf_cap_fraction: float = 0.002
    leaf_cap_min: int = 1
    minhash: MinHashConfig = MinHashConfig()
    max_topup_rounds: int = 4

    @property
    def leaf_cap(self) -> int:
        return max(self.leaf_cap_min, int(math.floor(self.leaf_cap_fraction * self.n_prompts)))


@dataclass
class PoolBuildResult:
    records: list[PromptRecord]
    manifest: dict

    def freeze(self, out_path: str | Path) -> "FrozenPool":
        return freeze_pool(self.records, out_path, self.manifest)


class PoolBuilder:
    """Composes generators into a frozen, ordered pool according to :class:`PoolBuildSpec`."""

    def __init__(self, spec: PoolBuildSpec, generators: Mapping[str, PromptGenerator],
                 contam_index: Optional[NGramIndex] = None, taxonomy: Optional[Taxonomy] = None):
        self.spec = spec
        self.generators = dict(generators)
        self.contam_index = contam_index
        self.taxonomy = taxonomy
        if SOURCE_TEMPLATED not in self.generators and SOURCE_WRITER not in self.generators:
            raise ValueError("need at least a templated or writer generator as fallback source")
        self.fallback = SOURCE_TEMPLATED if SOURCE_TEMPLATED in self.generators else SOURCE_WRITER

    def _rng(self, *parts: Any) -> random.Random:
        return random.Random(f"{self.spec.seed}|" + "|".join(str(p) for p in parts))

    def _request(self, category: str, source: str, n: int, round_id: int) -> list[PromptRecord]:
        gen = self.generators.get(source)
        if gen is None or n <= 0:
            return []
        return gen.generate(category, n, self._rng(category, source, round_id))

    def build(self) -> PoolBuildResult:
        spec = self.spec
        shares = normalise_shares(spec.prompt_shares)
        quotas = largest_remainder_counts(shares, spec.n_prompts)
        state = _CleanState(spec.minhash, self.contam_index, spec.leaf_cap)
        kept: dict[str, list[PromptRecord]] = {c: [] for c in CATEGORIES}
        requested: dict[str, Counter] = {c: Counter() for c in CATEGORIES}
        shortfall: dict[str, Counter] = {c: Counter() for c in CATEGORIES}

        # round 0: nominal source mixture
        for c in CATEGORIES:
            mix = spec.source_mix.get(c, {self.fallback: 1.0})
            want = largest_remainder_counts(mix, quotas[c])
            deficit = 0
            for src, n in sorted(want.items()):
                recs = self._request(c, src, n, 0)
                requested[c][src] += n
                if len(recs) < n:
                    shortfall[c][src] += n - len(recs)
                    deficit += n - len(recs)
                kept[c].extend(r for r in recs if state.accept(r))
            if deficit:
                recs = self._request(c, self.fallback, deficit, 0)
                requested[c][self.fallback] += deficit
                kept[c].extend(r for r in recs if state.accept(r))

        # top-up rounds for dedup / decontam / cap losses
        for rnd in range(1, spec.max_topup_rounds + 1):
            deficits = {c: quotas[c] - len(kept[c]) for c in CATEGORIES if quotas[c] > len(kept[c])}
            if not deficits:
                break
            for c, d in deficits.items():
                n = int(math.ceil(d * 1.2)) + 1
                recs = self._request(c, self.fallback, n, rnd)
                requested[c][self.fallback] += n
                kept[c].extend(r for r in recs if state.accept(r))

        # ordering: sources interleaved within category, categories interleaved by share
        per_cat_ordered: dict[str, list[PromptRecord]] = {}
        for c in CATEGORIES:
            by_src: dict[str, list[PromptRecord]] = defaultdict(list)
            for r in kept[c]:
                by_src[r.source_family].append(r)
            mix = spec.source_mix.get(c, {})
            weights = {s: mix.get(s, 0.0) or 0.01 for s in by_src}
            per_cat_ordered[c] = stratified_interleave(by_src, weights)[: quotas[c]]
        ordered = stratified_interleave(per_cat_ordered, shares)
        for i, r in enumerate(ordered):
            r.prompt_id = format_prompt_id(i)

        manifest = self._manifest(ordered, quotas, requested, shortfall, state)
        return PoolBuildResult(ordered, manifest)

    def _manifest(self, records: list[PromptRecord], quotas: dict, requested: dict, shortfall: dict,
                  state: _CleanState) -> dict:
        n = len(records)
        cat_counts = Counter(r.category for r in records)
        src_counts = Counter(r.source_family for r in records)
        src_detail = Counter(r.source for r in records)
        cat_src = {c: dict(Counter(r.source_family for r in records if r.category == c)) for c in CATEGORIES}
        lc = leaf_counts(r.leaf for r in records)
        report = {
            "distinct_4": distinct_n((r.task for r in records), 4),
            "taxonomy_entropy": leaf_entropy(lc),
            "max_leaf_count": max(lc.values()) if lc else 0,
            "leaf_cap": self.spec.leaf_cap,
            "audience_marginals": dict(Counter(r.audience for r in records if r.audience)),
            "n_personas": len({r.persona_id for r in records if r.persona_id}),
            "n_seed_docs": len({r.seed_doc_id for r in records if r.seed_doc_id}),
            "n_verifiable": sum(1 for r in records if r.verifiable),
            "msr_templates": dict(Counter(r.meta.get("template_id") for r in records if r.category == "MSR" and r.meta.get("template_id"))),
            "near_dup_rate_candidates": (state.drops["near_dup"] / state.n_seen) if state.n_seen else 0.0,
        }
        return {
            "schema_version": POOL_SCHEMA_VERSION,
            "version": self.spec.version,
            "seed": self.spec.seed,
            "n_prompts_target": self.spec.n_prompts,
            "n_prompts": n,
            "quotas": quotas,
            "counts_by_category": dict(cat_counts),
            "shares_realised": {c: (cat_counts[c] / n if n else 0.0) for c in CATEGORIES},
            "counts_by_source": dict(src_counts),
            "counts_by_source_detail": dict(src_detail),
            "counts_by_category_source": cat_src,
            "requested": {c: dict(v) for c, v in requested.items()},
            "shortfall_reallocated": {c: dict(v) for c, v in shortfall.items() if v},
            "drops": dict(state.drops),
            "n_candidates_seen": state.n_seen,
            "minhash": self.spec.minhash.to_dict(),
            "leaf_cap_fraction": self.spec.leaf_cap_fraction,
            "decontaminated": self.contam_index is not None,
            "decontam_ngram_n": self.contam_index.n if self.contam_index else None,
            "taxonomy_sha256": self.taxonomy.sha256 if self.taxonomy else None,
            "taxonomy_version": self.taxonomy.version if self.taxonomy else None,
            "generators": {s: getattr(g, "name", type(g).__name__) for s, g in self.generators.items()},
            "diversity": report,
            "templates": template_fingerprint(),
            "msr_template_version": MSR_TEMPLATE_VERSION,
            "msr_template_ids": sorted(MSR_TEMPLATES),
        }


# ======================================================================================
# Freezing / loading
# ======================================================================================
@dataclass(frozen=True)
class FrozenPool:
    path: Path
    sha256: str
    version: str
    manifest_path: Path
    n_prompts: int

    @property
    def pool_version(self) -> str:
        return pool_version_string(self.version, self.sha256)


def pool_version_string(version: str, sha256: str) -> str:
    return f"pool-{version}-{sha256[:12]}"


def record_line(rec: PromptRecord) -> str:
    return json.dumps(rec.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def freeze_pool(records: Sequence[PromptRecord], out_path: str | Path, manifest: Optional[dict] = None,
                version: Optional[str] = None) -> FrozenPool:
    """Write the ordered pool as JSONL, its sha256 and a build manifest (``<name>.manifest.json``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            if rec.prompt_id != format_prompt_id(i):
                raise ValueError(f"record {i} has prompt_id {rec.prompt_id!r}; pool must be in frozen order")
            line = record_line(rec) + "\n"
            fh.write(line)
            h.update(line.encode("utf-8"))
    sha = h.hexdigest()
    version = version or (manifest or {}).get("version", "v0")
    (out_path.parent / (out_path.name + ".sha256")).write_text(f"{sha}  {out_path.name}\n")
    manifest = dict(manifest or {})
    manifest.update({"pool_file": out_path.name, "pool_sha256": sha, "pool_version": pool_version_string(version, sha),
                     "n_prompts": len(records)})
    mpath = out_path.parent / (out_path.stem + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return FrozenPool(out_path, sha, version, mpath, len(records))


def pool_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_pool(path: str | Path) -> Iterator[PromptRecord]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield PromptRecord.from_json(json.loads(line))


def load_pool(path: str | Path, limit: Optional[int] = None) -> list[PromptRecord]:
    out = []
    for rec in iter_pool(path):
        out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out
