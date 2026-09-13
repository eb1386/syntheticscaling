"""Symbolic multi-step-reasoning (MSR) word-problem templates with computed gold answers.

GSM-Symbolic-style: each template samples parameters with a seeded RNG, renders a word problem
and computes the gold answer from the parameters. Every template also exposes an *independent*
brute-force evaluator (simulation / repeated addition / exact fractions) so the tests can verify
the closed-form gold against a second implementation.

Answer normalisation (:func:`normalise_answer`, :func:`answers_match`) is shared with the
verification of teacher responses (``Final answer: <x>`` lines).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

MSR_TEMPLATE_VERSION = "msr-templates-v1.0"

_NAMES = ["Maya", "Liam", "Sofia", "Noah", "Amara", "Ethan", "Priya", "Lucas", "Zara", "Omar",
          "Hana", "Mateo", "Ines", "Kenji", "Leila", "Tomas"]
_ITEMS = [("notebook", "notebooks"), ("pencil", "pencils"), ("apple", "apples"), ("ticket", "tickets"),
          ("mug", "mugs"), ("book", "books"), ("sticker", "stickers"), ("plant", "plants")]
_VEHICLES = ["car", "train", "cyclist", "bus", "delivery van"]
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_UNITS = [  # (from, to, factor, from_plural)
    ("kilometer", "meters", 1000, "kilometers"),
    ("hour", "minutes", 60, "hours"),
    ("kilogram", "grams", 1000, "kilograms"),
    ("meter", "centimeters", 100, "meters"),
    ("day", "hours", 24, "days"),
    ("week", "days", 7, "weeks"),
]


@dataclass
class MSRInstance:
    template_id: str
    task: str
    gold: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MSRTemplate:
    template_id: str
    sample: Callable[[random.Random], dict]
    render: Callable[[dict], str]
    gold: Callable[[dict], str]
    brute_force: Callable[[dict], str]

    def instantiate(self, rng: random.Random) -> MSRInstance:
        params = self.sample(rng)
        return MSRInstance(self.template_id, self.render(params), self.gold(params), params)


def _fmt_money(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.2f}"


def _fmt_num(x: Fraction | int) -> str:
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s


# --- T1 shopping total --------------------------------------------------------------------
def _t1_sample(rng: random.Random) -> dict:
    i1, i2 = rng.sample(_ITEMS, 2)
    return {"name": rng.choice(_NAMES), "n1": rng.randint(2, 9), "p1": rng.randint(2, 15),
            "n2": rng.randint(2, 9), "p2": rng.randint(2, 15), "item1": i1[1], "item2": i2[1]}


def _t1_render(p: dict) -> str:
    return (f"{p['name']} buys {p['n1']} {p['item1']} at ${p['p1']} each and {p['n2']} {p['item2']} "
            f"at ${p['p2']} each. How many dollars does {p['name']} spend in total?")


def _t1_gold(p: dict) -> str:
    return str(p["n1"] * p["p1"] + p["n2"] * p["p2"])


def _t1_brute(p: dict) -> str:
    total = 0
    for _ in range(p["n1"]):
        total += p["p1"]
    for _ in range(p["n2"]):
        total += p["p2"]
    return str(total)


# --- T2 two-leg trip distance -------------------------------------------------------------
def _t2_sample(rng: random.Random) -> dict:
    return {"vehicle": rng.choice(_VEHICLES), "v1": rng.randint(10, 90), "t1": rng.randint(1, 6),
            "v2": rng.randint(10, 90), "t2": rng.randint(1, 6)}


def _t2_render(p: dict) -> str:
    return (f"A {p['vehicle']} travels at {p['v1']} km/h for {p['t1']} hours, then at {p['v2']} km/h "
            f"for {p['t2']} hours. How many kilometers does it travel in total?")


def _t2_gold(p: dict) -> str:
    return str(p["v1"] * p["t1"] + p["v2"] * p["t2"])


def _t2_brute(p: dict) -> str:
    d = 0
    for _ in range(p["t1"]):
        d += p["v1"]
    for _ in range(p["t2"]):
        d += p["v2"]
    return str(d)


# --- T3 sharing leftover ------------------------------------------------------------------
def _t3_sample(rng: random.Random) -> dict:
    k, m = rng.randint(2, 6), rng.randint(2, 6)
    return {"name": rng.choice(_NAMES), "item": rng.choice(_ITEMS)[1], "k": k, "m": m,
            "total": k * m + rng.randint(0, 20)}


def _t3_render(p: dict) -> str:
    return (f"{p['name']} has {p['total']} {p['item']} and gives {p['k']} to each of {p['m']} friends. "
            f"How many {p['item']} does {p['name']} have left?")


def _t3_gold(p: dict) -> str:
    return str(p["total"] - p["k"] * p["m"])


def _t3_brute(p: dict) -> str:
    left = p["total"]
    for _ in range(p["m"]):
        for _ in range(p["k"]):
            left -= 1
    return str(left)


# --- T4 ages in the future ----------------------------------------------------------------
def _t4_sample(rng: random.Random) -> dict:
    n1, n2 = rng.sample(_NAMES, 2)
    return {"name1": n1, "name2": n2, "a": rng.randint(5, 40), "d": rng.randint(1, 15), "y": rng.randint(1, 12)}


def _t4_render(p: dict) -> str:
    return (f"{p['name1']} is {p['a']} years old. {p['name2']} is {p['d']} years older than {p['name1']}. "
            f"What will the sum of their ages be in {p['y']} years?")


def _t4_gold(p: dict) -> str:
    return str((p["a"] + p["y"]) + (p["a"] + p["d"] + p["y"]))


def _t4_brute(p: dict) -> str:
    a1, a2 = p["a"], p["a"] + p["d"]
    for _ in range(p["y"]):
        a1 += 1
        a2 += 1
    return str(a1 + a2)


# --- T5 discount total ----------------------------------------------------------------------
def _t5_sample(rng: random.Random) -> dict:
    return {"item": rng.choice(_ITEMS), "price": rng.randint(5, 60), "pct": rng.choice([5, 10, 15, 20, 25, 30, 40, 50]),
            "n": rng.randint(1, 6)}


def _t5_render(p: dict) -> str:
    return (f"A {p['item'][0]} normally costs ${p['price']}. During a sale every {p['item'][0]} is "
            f"{p['pct']}% off. How many dollars do {p['n']} {p['item'][1]} cost during the sale?")


def _t5_gold(p: dict) -> str:
    return _fmt_money(Fraction(p["n"] * p["price"] * (100 - p["pct"]), 100))


def _t5_brute(p: dict) -> str:
    discount_each = Fraction(p["price"]) * Fraction(p["pct"], 100)
    total = Fraction(0)
    for _ in range(p["n"]):
        total += Fraction(p["price"]) - discount_each
    return _fmt_money(total)


# --- T6 average of scores -----------------------------------------------------------------
def _t6_sample(rng: random.Random) -> dict:
    k = rng.randint(3, 5)
    scores = [rng.randint(50, 100) for _ in range(k)]
    # adjust the last score so the sum is divisible by k (integer average)
    rem = sum(scores) % k
    scores[-1] -= rem
    return {"name": rng.choice(_NAMES), "scores": scores}


def _t6_render(p: dict) -> str:
    s = ", ".join(str(x) for x in p["scores"][:-1]) + f" and {p['scores'][-1]}"
    return f"{p['name']} scored {s} on {len(p['scores'])} quizzes. What is the average score?"


def _t6_gold(p: dict) -> str:
    return _fmt_num(Fraction(sum(p["scores"]), len(p["scores"])))


def _t6_brute(p: dict) -> str:
    total = Fraction(0)
    for s in p["scores"]:
        total += s
    return _fmt_num(total / len(p["scores"]))


# --- T7 weekday arithmetic ----------------------------------------------------------------
def _t7_sample(rng: random.Random) -> dict:
    return {"today": rng.choice(_WEEKDAYS), "n": rng.randint(3, 60)}


def _t7_render(p: dict) -> str:
    return f"If today is {p['today']}, what day of the week will it be {p['n']} days from now?"


def _t7_gold(p: dict) -> str:
    return _WEEKDAYS[(_WEEKDAYS.index(p["today"]) + p["n"]) % 7]


def _t7_brute(p: dict) -> str:
    i = _WEEKDAYS.index(p["today"])
    for _ in range(p["n"]):
        i = 0 if i == 6 else i + 1
    return _WEEKDAYS[i]


# --- T8 unit conversion chain -------------------------------------------------------------
def _t8_sample(rng: random.Random) -> dict:
    u = rng.choice(_UNITS)
    return {"x": rng.randint(2, 30), "unit_from": u[0], "unit_from_pl": u[3], "unit_to": u[1], "factor": u[2],
            "extra": rng.randint(1, 9)}


def _t8_render(p: dict) -> str:
    return (f"A recipe step takes {p['x']} {p['unit_from_pl']} and is repeated {p['extra']} times. "
            f"How many {p['unit_to']} is that in total?")


def _t8_gold(p: dict) -> str:
    return str(p["x"] * p["factor"] * p["extra"])


def _t8_brute(p: dict) -> str:
    total = 0
    for _ in range(p["extra"]):
        for _ in range(p["x"]):
            total += p["factor"]
    return str(total)


# --- T9 ratio split -----------------------------------------------------------------------
def _t9_sample(rng: random.Random) -> dict:
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    while a == b:
        b = rng.randint(1, 5)
    n1, n2 = rng.sample(_NAMES, 2)
    return {"name1": n1, "name2": n2, "a": a, "b": b, "total": (a + b) * rng.randint(2, 20)}


def _t9_render(p: dict) -> str:
    return (f"{p['name1']} and {p['name2']} split ${p['total']} in the ratio {p['a']}:{p['b']}. "
            f"How many dollars does {p['name1']} receive?")


def _t9_gold(p: dict) -> str:
    return str(p["total"] * p["a"] // (p["a"] + p["b"]))


def _t9_brute(p: dict) -> str:
    share1 = 0
    remaining = p["total"]
    while remaining > 0:
        share1 += p["a"]
        remaining -= p["a"] + p["b"]
    return str(share1)


MSR_TEMPLATES: dict[str, MSRTemplate] = {
    t.template_id: t
    for t in [
        MSRTemplate("shopping_total", _t1_sample, _t1_render, _t1_gold, _t1_brute),
        MSRTemplate("trip_distance", _t2_sample, _t2_render, _t2_gold, _t2_brute),
        MSRTemplate("sharing_leftover", _t3_sample, _t3_render, _t3_gold, _t3_brute),
        MSRTemplate("ages_future_sum", _t4_sample, _t4_render, _t4_gold, _t4_brute),
        MSRTemplate("discount_total", _t5_sample, _t5_render, _t5_gold, _t5_brute),
        MSRTemplate("average_score", _t6_sample, _t6_render, _t6_gold, _t6_brute),
        MSRTemplate("weekday_after", _t7_sample, _t7_render, _t7_gold, _t7_brute),
        MSRTemplate("unit_conversion", _t8_sample, _t8_render, _t8_gold, _t8_brute),
        MSRTemplate("ratio_split", _t9_sample, _t9_render, _t9_gold, _t9_brute),
    ]
}


def instantiate_msr(rng: random.Random, template_id: Optional[str] = None) -> MSRInstance:
    """Draw a template (uniform unless given) and instantiate it with ``rng``."""
    tid = template_id or rng.choice(sorted(MSR_TEMPLATES))
    return MSR_TEMPLATES[tid].instantiate(rng)


# --------------------------------------------------------------------------------------
# Answer normalisation / matching (shared with response verification)
# --------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalise_answer(text: str) -> str:
    t = text.strip().strip(".").strip()
    t = t.replace("$", "").replace(",", "").replace("%", "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _as_number(text: str) -> Optional[Fraction]:
    """Parse a numeric answer, tolerating a trailing unit word ("42 dollars"); else ``None``."""
    t = normalise_answer(text)
    m = _NUM_RE.match(t)
    if m is None:
        return None
    rest = t[m.end():].strip()
    if rest and not re.fullmatch(r"[a-z ]*", rest):
        return None
    try:
        return Fraction(m.group(0))
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(pred: str, gold: str, rel_tol: float = 1e-4) -> bool:
    """Exact match after normalisation; numeric answers compared with a relative tolerance."""
    p_num, g_num = _as_number(pred), _as_number(gold)
    if g_num is not None:
        if p_num is None:
            return False
        if g_num == 0:
            return p_num == 0
        return abs(float(p_num - g_num)) <= rel_tol * abs(float(g_num))
    return normalise_answer(pred) == normalise_answer(gold)
