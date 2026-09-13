"""200-item format-adherence probe (benchmarks.md §3.3).

Eight templates × 25 items, generated deterministically from small word banks (no teacher
output is used), scored by regex / JSON parsing.  Greedy decoding with 64 new tokens through a
pluggable ``generate_fn(prompts) -> list[str]`` so the probe is independent of the harness and
of torch.  The pass rate per template is the reported quantity (chance ≈ 0); where an item has
a known answer, ``correct`` is scored separately so that *format* and *knowledge* can be told
apart.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

PROBE_SEED = 20260910
ITEMS_PER_TEMPLATE = 25
MAX_NEW_TOKENS = 64

TEMPLATES: tuple[str, ...] = (
    "one_word", "yes_no", "numbered_list", "json_keys", "max_words",
    "pick_label", "repeat_verbatim", "end_phrase",
)

_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"]
_ANIMALS = ["dog", "cat", "horse", "cow", "sheep", "goat", "lion", "tiger", "bear", "wolf", "fox", "deer"]
_FRUITS = ["apple", "banana", "cherry", "grape", "lemon", "mango", "peach", "pear", "plum", "melon"]
_TOOLS = ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "file", "level"]
_PLANTS = ["oak", "pine", "rose", "tulip", "fern", "moss", "maple", "birch"]
_FISH = ["trout", "salmon", "cod", "tuna", "carp", "pike"]
_CATEGORIES = {
    "colors": _COLORS, "animals": _ANIMALS, "fruits": _FRUITS, "tools": _TOOLS, "plants": _PLANTS,
}
_CAPITALS = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "Egypt": "Cairo",
             "Peru": "Lima", "Kenya": "Nairobi", "Norway": "Oslo", "Greece": "Athens", "Chile": "Santiago",
             "Cuba": "Havana", "Iran": "Tehran", "Ireland": "Dublin", "Poland": "Warsaw", "Austria": "Vienna",
             "Portugal": "Lisbon", "Thailand": "Bangkok", "Canada": "Ottawa", "Ghana": "Accra", "Nepal": "Kathmandu",
             "Finland": "Helsinki", "Hungary": "Budapest", "Denmark": "Copenhagen", "Sweden": "Stockholm",
             "Belgium": "Brussels"}
_YESNO_FACTS = [("Is the sun a star", "yes"), ("Do fish live in water", "yes"), ("Is ice cold", "yes"),
                ("Can birds fly", "yes"), ("Is snow white", "yes"), ("Do cows eat grass", "yes"),
                ("Is the moon made of cheese", "no"), ("Do cats bark", "no"), ("Is fire cold", "no"),
                ("Can rocks talk", "no"), ("Is water dry", "no"), ("Do trees walk", "no"),
                ("Is seven larger than three", "yes"), ("Is two plus two five", "no"),
                ("Is a week seven days", "yes"), ("Is a triangle round", "no"),
                ("Does rain fall from clouds", "yes"), ("Is sugar sour", "no"),
                ("Do humans have two eyes", "yes"), ("Is a whale a fish", "no"),
                ("Is Paris in France", "yes"), ("Is gold a metal", "yes"),
                ("Is the ocean made of milk", "no"), ("Do spiders have eight legs", "yes"),
                ("Is midnight in the afternoon", "no")]
_TOPICS = ["the water cycle", "photosynthesis", "gravity", "a bicycle", "the moon", "a volcano",
           "friendship", "the ocean", "a library", "electricity", "bread baking", "a rainbow",
           "a river", "recycling", "the seasons", "a thunderstorm", "honeybees", "a desert",
           "a telescope", "a kite", "the alphabet", "a compass", "a garden", "a glacier", "a clock"]
_END_PHRASES = ["That is all.", "End of answer.", "Thank you.", "Done.", "Over and out."]
_JSON_KEYSETS = [("name", "age"), ("city", "country"), ("item", "price"), ("title", "year"),
                 ("animal", "legs"), ("color", "hex"), ("fruit", "count"), ("first", "last")]


@dataclass
class ProbeItem:
    template: str
    idx: int
    prompt: str
    spec: dict = field(default_factory=dict)  # scoring parameters (N, keys, string, phrase, gold)


@dataclass
class ItemScore:
    template: str
    idx: int
    passed: bool
    correct: Optional[bool]
    response: str


@dataclass
class ProbeResult:
    pass_rate: dict[str, float]
    correct_rate: dict[str, Optional[float]]
    overall_pass_rate: float
    n_items: int
    scores: list[ItemScore]

    def to_json(self) -> dict:
        return {"pass_rate": self.pass_rate, "correct_rate": self.correct_rate,
                "overall_pass_rate": self.overall_pass_rate, "n_items": self.n_items,
                "scores": [asdict(s) for s in self.scores]}


# ----------------------------------------------------------------------------------------
# Item generation (deterministic)
# ----------------------------------------------------------------------------------------
def build_probe_items(seed: int = PROBE_SEED, per_template: int = ITEMS_PER_TEMPLATE) -> list[ProbeItem]:
    """Return the 8 × ``per_template`` items; identical for a given seed."""
    rng = random.Random(seed)
    items: list[ProbeItem] = []
    countries = list(_CAPITALS)
    for i in range(per_template):
        # 1. one-word answer
        c = countries[i % len(countries)]
        items.append(ProbeItem("one_word", i,
                               f"Answer with exactly one word. What is the capital of {c}?\nAnswer:",
                               {"gold": _CAPITALS[c]}))
        # 2. yes/no
        q, gold = _YESNO_FACTS[i % len(_YESNO_FACTS)]
        items.append(ProbeItem("yes_no", i, f"Answer with yes or no only. {q}?\nAnswer:", {"gold": gold}))
        # 3. numbered list of exactly N items
        n = rng.choice([2, 3, 4, 5])
        cat = rng.choice(list(_CATEGORIES))
        items.append(ProbeItem("numbered_list", i,
                               f"List exactly {n} {cat} as a numbered list (1., 2., ...) and nothing else.\nAnswer:",
                               {"n": n, "category": cat}))
        # 4. JSON with fixed keys
        keys = _JSON_KEYSETS[i % len(_JSON_KEYSETS)]
        items.append(ProbeItem("json_keys", i,
                               f"Respond with a JSON object that has exactly the keys \"{keys[0]}\" and "
                               f"\"{keys[1]}\" and no other text.\nAnswer:", {"keys": list(keys)}))
        # 5. answer in <= N words
        n = rng.choice([3, 5, 8, 10])
        topic = _TOPICS[i % len(_TOPICS)]
        items.append(ProbeItem("max_words", i,
                               f"In at most {n} words, describe {topic}.\nAnswer:", {"n": n}))
        # 6. pick one of listed options by label
        mammal = rng.choice(_ANIMALS)
        distractors = [rng.choice(_FISH), rng.choice(_PLANTS), rng.choice(_TOOLS)]
        opts = distractors + [mammal]
        rng.shuffle(opts)
        labels = ["A", "B", "C", "D"]
        gold_label = labels[opts.index(mammal)]
        listed = " ".join(f"({l}) {o}" for l, o in zip(labels, opts))
        items.append(ProbeItem("pick_label", i,
                               f"Which of these is a mammal? {listed}\nAnswer with the letter only.\nAnswer:",
                               {"gold": gold_label, "labels": labels}))
        # 7. repeat a given string verbatim
        s = f"{rng.choice(_COLORS)} {rng.choice(_ANIMALS)} {rng.randint(10, 99)}"
        items.append(ProbeItem("repeat_verbatim", i,
                               f"Repeat the following text exactly, with no other words: {s}\nAnswer:",
                               {"string": s}))
        # 8. end with a fixed phrase
        phrase = _END_PHRASES[i % len(_END_PHRASES)]
        topic = _TOPICS[(i * 7) % len(_TOPICS)]
        items.append(ProbeItem("end_phrase", i,
                               f"Write one short sentence about {topic}. End your answer with the exact phrase "
                               f"\"{phrase}\"\nAnswer:", {"phrase": phrase}))
    return items


# ----------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*[.!]?$")
_YESNO_RE = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
_LIST_LINE_RE = re.compile(r"^\s*(\d+)[.)]\s+(\S.*)$")
_LABEL_RE = re.compile(r"^\W*\(?([A-D])\)?[.:]?(?=\s|$)")
_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _first_paragraph(text: str) -> str:
    return text.strip().split("\n\n")[0].strip()


def score_item(item: ProbeItem, response: str) -> ItemScore:
    """Regex / JSON scoring for one response; returns format pass and (if known) correctness."""
    t, sp = item.template, item.spec
    resp = response if isinstance(response, str) else str(response)
    passed, correct = False, None
    first = _first_line(resp)
    if t == "one_word":
        passed = bool(_WORD_RE.match(first))
        if passed:
            correct = first.rstrip(".!").lower() == sp["gold"].lower()
    elif t == "yes_no":
        m = _YESNO_RE.match(resp.strip())
        passed = m is not None
        if m:
            correct = m.group(1).lower() == sp["gold"]
    elif t == "numbered_list":
        nums, extra = [], 0
        for line in resp.strip().splitlines():
            if not line.strip():
                continue
            m = _LIST_LINE_RE.match(line)
            if m:
                nums.append(int(m.group(1)))
            else:
                extra += 1
        # exactly N consecutive numbered lines and no other text
        passed = nums[: sp["n"]] == list(range(1, sp["n"] + 1)) and len(nums) == sp["n"] and extra == 0
    elif t == "json_keys":
        m = _JSON_BLOCK_RE.search(resp)
        if m:
            try:
                obj = json.loads(m.group(0))
                passed = isinstance(obj, dict) and set(obj.keys()) == set(sp["keys"])
            except json.JSONDecodeError:
                passed = False
    elif t == "max_words":
        words = [w for w in re.split(r"\s+", _first_paragraph(resp)) if w]
        passed = 1 <= len(words) <= sp["n"]
    elif t == "pick_label":
        m = _LABEL_RE.match(first)
        passed = m is not None
        if m:
            correct = m.group(1) == sp["gold"]
    elif t == "repeat_verbatim":
        got = first.strip().strip("\"'`")
        passed = got.rstrip(".") == sp["string"]
        correct = passed
    elif t == "end_phrase":
        body = resp.strip().strip("\"'")
        passed = body.endswith(sp["phrase"]) and len(body) > len(sp["phrase"])
    return ItemScore(t, item.idx, bool(passed), correct, resp)


def run_format_probe(
    generate_fn: Callable[[Sequence[str]], Sequence[str]],
    items: Optional[list[ProbeItem]] = None,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> ProbeResult:
    """Generate greedily for every item and score.  ``generate_fn`` receives all prompts at
    once and must return one string per prompt (truncate to ``max_new_tokens`` itself)."""
    items = items or build_probe_items()
    try:
        responses = list(generate_fn(list(it.prompt for it in items), max_new_tokens=max_new_tokens))  # type: ignore[call-arg]
    except TypeError:
        responses = list(generate_fn([it.prompt for it in items]))
    if len(responses) != len(items):
        raise ValueError("generate_fn must return one response per prompt")
    scores = [score_item(it, r) for it, r in zip(items, responses)]
    return summarize_scores(scores)


def summarize_scores(scores: list[ItemScore]) -> ProbeResult:
    pass_rate: dict[str, float] = {}
    correct_rate: dict[str, Optional[float]] = {}
    for t in TEMPLATES:
        ss = [s for s in scores if s.template == t]
        if not ss:
            continue
        pass_rate[t] = sum(s.passed for s in ss) / len(ss)
        known = [s for s in ss if s.correct is not None]
        correct_rate[t] = (sum(bool(s.correct) for s in known) / len(ss)) if known else None
    overall = sum(s.passed for s in scores) / len(scores) if scores else float("nan")
    return ProbeResult(pass_rate, correct_rate, overall, len(scores), scores)


def write_probe_result(results_dir: str, experiment_id: str, result: ProbeResult) -> str:
    from pathlib import Path

    out = Path(results_dir) / experiment_id / "eval" / "format_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=1))
    return str(out)
