"""Primary aggregate, floor gate and cell tables (benchmarks.md §1.4, §3.1, §4; statistics.md §1).

* ``CHANCE_TABLE`` — chance level c_k = max(uniform guess, majority class) and item count N_k per
  harness task (Table C; BoolQ uses the majority baseline 0.622).
* Floor gate: task t enters the primary set at the smallest retained student size S iff
  mean_C0(a_t) − c_t ≥ max(5 points, 3·SE_item,t), SE_item = sqrt(a(1−a)/N).  Tier A tasks
  always enter (with a near-floor flag); Tier B tasks are gated.  The same set K is used at
  every S.
* P_cc = (100/|K|) Σ_k (a_k − c_k)/(1 − c_k); unweighted mean M; SD-unit gain D over C0.
* Cell table (mean ± SD, r) and control bands (C0 / C1 / C1b) with t-based 95 % CIs.

Column conventions of the run-level index (see ``synscale.tracking.index``): ``student``,
``cond`` (teacher id or ``C0``/``C1``/``C1b``), ``arm``, ``seed_phase``, ``s_params_nominal``,
``t_params_nominal`` and one ``<task>_<metric>`` column per archived metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------------------
# Task table (benchmarks.md Table C + statistics.md §1.1)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskInfo:
    task: str
    metric: str
    chance: float
    n_items: int
    tier: str  # primary_a | primary_b | secondary | report


TASK_TABLE: dict[str, TaskInfo] = {
    "sciq": TaskInfo("sciq", "acc", 0.25, 1000, "primary_a"),
    "arc_easy": TaskInfo("arc_easy", "acc_norm", 0.25, 2376, "primary_a"),
    "piqa": TaskInfo("piqa", "acc_norm", 0.50, 1838, "primary_a"),
    "lambada_openai": TaskInfo("lambada_openai", "acc", 0.0, 5153, "primary_a"),
    "openbookqa": TaskInfo("openbookqa", "acc_norm", 0.25, 500, "primary_b"),
    "social_iqa": TaskInfo("social_iqa", "acc", 1.0 / 3.0, 1954, "primary_b"),
    "hellaswag": TaskInfo("hellaswag", "acc_norm", 0.25, 10042, "primary_b"),
    "arc_challenge": TaskInfo("arc_challenge", "acc_norm", 0.25, 1172, "secondary"),
    "winogrande": TaskInfo("winogrande", "acc", 0.50, 1267, "secondary"),
    "boolq": TaskInfo("boolq", "acc", 0.622, 3270, "secondary"),  # majority ("yes") baseline
    "mmlu_continuation": TaskInfo("mmlu_continuation", "acc_norm", 0.25, 14042, "secondary"),
    "mmlu_stem_continuation": TaskInfo("mmlu_stem_continuation", "acc_norm", 0.25, 3018, "secondary"),
    "commonsense_qa_cloze": TaskInfo("commonsense_qa_cloze", "acc_norm", 0.20, 1221, "secondary"),
    "blimp": TaskInfo("blimp", "acc", 0.50, 67000, "secondary"),
    "truthfulqa_mc2": TaskInfo("truthfulqa_mc2", "acc", 0.40, 817, "report"),
    "gsm8k": TaskInfo("gsm8k", "exact_match,flexible-extract", 0.0, 1319, "secondary"),
    "triviaqa": TaskInfo("triviaqa", "exact_match", 0.0, 3000, "secondary"),
}
CHANCE_TABLE: dict[str, float] = {k: v.chance for k, v in TASK_TABLE.items()}
ITEM_COUNTS: dict[str, int] = {k: v.n_items for k, v in TASK_TABLE.items()}
TIER_A: tuple[str, ...] = tuple(k for k, v in TASK_TABLE.items() if v.tier == "primary_a")
TIER_B: tuple[str, ...] = tuple(k for k, v in TASK_TABLE.items() if v.tier == "primary_b")
PRIMARY_TASKS: tuple[str, ...] = TIER_A + TIER_B
GATE_MIN_POINTS = 0.05
GATE_SE_MULTIPLIER = 3.0

# Control-arm labels
C0, C1, C1B = "C0", "C1", "C1b"
ARM_TO_COND = {"base_only": C0, "matched_real": C1, "human_instruct": C1B}
S_REF = 25e6  # s = log2(S / 25M)
T_REF = 1e9  # x = log2(T / 1B)
PRIMARY_TEACHER_PARAMS: tuple[float, ...] = (1.5e9, 3e9, 7e9, 14e9, 32e9, 72e9)


def metric_column(task: str) -> str:
    return f"{task}_{TASK_TABLE[task].metric}"


# ----------------------------------------------------------------------------------------
# Floor gate
# ----------------------------------------------------------------------------------------
def se_item(acc: float, n: int) -> float:
    """Binomial item-sampling SE of an accuracy."""
    acc = min(max(acc, 0.0), 1.0)
    return math.sqrt(acc * (1 - acc) / n) if n > 0 else float("nan")


def floor_gate(c0_mean_acc: float, task: str, *, chance: Optional[float] = None, n_items: Optional[int] = None,
               min_points: float = GATE_MIN_POINTS, se_mult: float = GATE_SE_MULTIPLIER) -> tuple[bool, float, float]:
    """Return (passes, margin, threshold) for one task given the C0 mean accuracy at the smallest S."""
    c = CHANCE_TABLE[task] if chance is None else chance
    n = ITEM_COUNTS[task] if n_items is None else n_items
    if c0_mean_acc is None or (isinstance(c0_mean_acc, float) and math.isnan(c0_mean_acc)):
        return False, float("nan"), float("nan")
    thr = max(min_points, se_mult * se_item(c0_mean_acc, n))
    margin = c0_mean_acc - c
    return bool(margin >= thr), margin, thr


@dataclass
class GateResult:
    smallest_student: Optional[str]
    retained: list[str]  # the primary set K (same at every S)
    tier_a_near_floor: list[str]
    details: pd.DataFrame  # one row per task: c0_mean, chance, se, threshold, margin, passes, tier, in_K
    status: str = "ok"  # ok | no_c0 (Tier A only)


def gate_from_index(df: pd.DataFrame, tasks: Iterable[str] = PRIMARY_TASKS,
                    smallest_student: Optional[str] = None) -> GateResult:
    """Evaluate the floor gate on the C0 rows of the smallest retained student size.

    ``smallest_student`` defaults to the student with the smallest ``s_params_nominal`` among the
    rows that have a C0 run; without any C0 row the gate cannot be evaluated and K = Tier A.
    """
    tasks = list(tasks)
    c0 = df[df["cond"] == C0]
    if smallest_student is None and len(c0):
        smallest_student = str(c0.sort_values("s_params_nominal").iloc[0]["student"])
    rows = []
    retained, near_floor = [], []
    if smallest_student is None or c0.empty:
        for t in tasks:
            info = TASK_TABLE[t]
            in_k = info.tier == "primary_a"
            rows.append({"task": t, "tier": info.tier, "c0_mean": np.nan, "chance": info.chance, "se_item": np.nan,
                         "threshold": np.nan, "margin": np.nan, "passes": False, "in_K": in_k})
            if in_k:
                retained.append(t)
        return GateResult(None, retained, [], pd.DataFrame(rows), status="no_c0")
    sub = c0[c0["student"] == smallest_student]
    for t in tasks:
        info = TASK_TABLE[t]
        col = metric_column(t)
        m = float(sub[col].mean()) if col in sub and sub[col].notna().any() else float("nan")
        passes, margin, thr = floor_gate(m, t)
        in_k = passes or info.tier == "primary_a"
        if info.tier == "primary_a" and not passes:
            near_floor.append(t)
        if in_k and not math.isnan(m):
            retained.append(t)
        elif in_k and info.tier == "primary_a":
            retained.append(t)  # Tier A always enters even if not yet evaluated
        rows.append({"task": t, "tier": info.tier, "c0_mean": m, "chance": info.chance,
                     "se_item": se_item(m, info.n_items) if not math.isnan(m) else np.nan,
                     "threshold": thr, "margin": margin, "passes": passes, "in_K": in_k})
    return GateResult(smallest_student, retained, near_floor, pd.DataFrame(rows))


# ----------------------------------------------------------------------------------------
# Aggregates
# ----------------------------------------------------------------------------------------
def chance_corrected(acc: float, chance: float) -> float:
    return (acc - chance) / (1.0 - chance)


def p_cc(accs: Mapping[str, float], tasks: Sequence[str], chance: Mapping[str, float] = CHANCE_TABLE) -> float:
    """Chance-corrected mean over ``tasks`` in points (NaN if any task is missing)."""
    vals = []
    for t in tasks:
        a = accs.get(t)
        if a is None or (isinstance(a, float) and math.isnan(a)):
            return float("nan")
        vals.append(chance_corrected(float(a), chance[t]))
    return 100.0 * float(np.mean(vals)) if vals else float("nan")


def se_p_cc_item(accs: Mapping[str, float], tasks: Sequence[str]) -> float:
    """Item-sampling SE of P_cc in points: (100/K) sqrt(Σ SE_t²/(1−c_t)²)."""
    if not tasks:
        return float("nan")
    tot = 0.0
    for t in tasks:
        tot += (se_item(float(accs[t]), ITEM_COUNTS[t]) / (1 - CHANCE_TABLE[t])) ** 2
    return 100.0 * math.sqrt(tot) / len(tasks)


def unweighted_mean(accs: Mapping[str, float], tasks: Sequence[str]) -> float:
    vals = [float(accs[t]) for t in tasks if t in accs and not math.isnan(float(accs[t]))]
    return 100.0 * float(np.mean(vals)) if len(vals) == len(tasks) and vals else float("nan")


def add_aggregates(df: pd.DataFrame, K: Sequence[str], *, tier_a: Sequence[str] = TIER_A) -> pd.DataFrame:
    """Add ``p_cc`` (over K), ``p_cc_tier_a``, ``mean_acc`` and ``task_loss`` (mean over K) columns."""
    out = df.copy()
    cols = {t: metric_column(t) for t in set(K) | set(tier_a)}

    def row_accs(row: pd.Series) -> dict[str, float]:
        return {t: row[c] for t, c in cols.items() if c in row.index}

    out["p_cc"] = out.apply(lambda r: p_cc(row_accs(r), list(K)), axis=1)
    out["p_cc_tier_a"] = out.apply(lambda r: p_cc(row_accs(r), list(tier_a)), axis=1)
    out["mean_acc"] = out.apply(lambda r: unweighted_mean(row_accs(r), list(K)), axis=1)
    tl_cols = [f"task_loss_{t}" for t in K if f"task_loss_{t}" in out.columns]
    out["task_loss"] = out[tl_cols].mean(axis=1) if tl_cols else np.nan
    return out


# ----------------------------------------------------------------------------------------
# Cell tables and SD-unit gains
# ----------------------------------------------------------------------------------------
def t_ci(values: Sequence[float], level: float = 0.95) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) with a t-based CI across seeds; lo = hi = mean when n = 1."""
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if n == 1:
        return m, m, m, 1
    h = stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h, n


def cell_table(df: pd.DataFrame, outcome: str = "p_cc", extra: Sequence[str] = ()) -> pd.DataFrame:
    """Per (student, cond): r, mean, SD, 95 % t-CI of ``outcome`` (+ mean ± SD of ``extra`` columns)."""
    rows = []
    for (s, c), g in df.groupby(["student", "cond"], sort=False):
        m, lo, hi, n = t_ci(g[outcome].tolist())
        row = {"student": s, "cond": c, "arm": g["arm"].iloc[0], "r": n, "mean": m,
               "sd": float(g[outcome].std(ddof=1)) if n > 1 else np.nan, "ci_lo": lo, "ci_hi": hi,
               "t_params_nominal": g["t_params_nominal"].iloc[0] if "t_params_nominal" in g else np.nan,
               "s_params_nominal": g["s_params_nominal"].iloc[0]}
        for e in extra:
            if e in g:
                row[f"{e}_mean"] = float(g[e].mean())
                row[f"{e}_sd"] = float(g[e].std(ddof=1)) if n > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["s_params_nominal", "t_params_nominal"], na_position="first").reset_index(drop=True)


def control_bands(df: pd.DataFrame, outcome: str = "p_cc") -> pd.DataFrame:
    """C0 / C1 / C1b mean and 95 % CI per student (for the horizontal bands in the figures)."""
    ct = cell_table(df[df["cond"].isin([C0, C1, C1B])], outcome)
    return ct[["student", "cond", "r", "mean", "sd", "ci_lo", "ci_hi"]].reset_index(drop=True)


def within_size_residual_sd(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Pooled within-size residual SD of ``outcome`` across all conditions and seeds at each S."""
    rows = []
    for s, g in df.groupby("student"):
        resid, dfree = 0.0, 0
        for _, cell in g.groupby("cond"):
            v = cell[outcome].dropna().to_numpy(dtype=float)
            if len(v) > 1:
                resid += float(((v - v.mean()) ** 2).sum())
                dfree += len(v) - 1
        rows.append({"student": s, "sigma": math.sqrt(resid / dfree) if dfree > 0 else np.nan, "df": dfree})
    return pd.DataFrame(rows)


def sd_unit_gain(df: pd.DataFrame, K: Sequence[str]) -> pd.DataFrame:
    """Base-normalised gain in SD units, d_t = (ā_t(cond,S) − ā_t(C0,S))/σ̂_t(S), and D = mean_t d_t."""
    rows = []
    for s, g in df.groupby("student"):
        c0 = g[g["cond"] == C0]
        if c0.empty:
            continue
        sig = {t: within_size_residual_sd(g, metric_column(t)).iloc[0]["sigma"] for t in K}
        for c, cell in g.groupby("cond"):
            ds = {}
            for t in K:
                col = metric_column(t)
                if sig[t] and not math.isnan(sig[t]) and sig[t] > 0:
                    ds[t] = 100 * (cell[col].mean() - c0[col].mean()) / (100 * sig[t])
            rows.append({"student": s, "cond": c, "D": float(np.mean(list(ds.values()))) if ds else np.nan,
                         **{f"d_{t}": v for t, v in ds.items()}})
    return pd.DataFrame(rows)


def per_task_cell_table(df: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    """Mean ± SD of every per-task accuracy per cell (appendix table; no task is dropped)."""
    cols = [metric_column(t) for t in tasks if metric_column(t) in df.columns]
    agg = df.groupby(["student", "cond"])[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
