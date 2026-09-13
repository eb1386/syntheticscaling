"""Simulated run-level indices with known response surfaces (for tests and power analysis).

Nothing here is an experimental result: every number is generated from the surfaces of
statistics.md §3.2 (``null``, ``lin``, ``sat``, ``satS``, ``peak``) plus Gaussian seed noise, and
the tables are shaped exactly like ``results/index.parquet`` so that the whole analysis
pipeline can be exercised end to end.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from synscale.analysis.aggregate import (
    C0, C1, C1B, CHANCE_TABLE, PRIMARY_TASKS, S_REF, T_REF, TASK_TABLE, metric_column,
)

STUDENTS: dict[str, tuple[float, float]] = {  # name -> (nominal, non-embedding) params (locked §0.5)
    "s025m": (25e6, 12_589_440), "s100m": (100e6, 75_516_672), "s250m": (250e6, 216_312_000),
    "s500m": (500e6, 462_487_872), "s1b": (1e9, 956_419_584),
}
TEACHERS: dict[str, float] = {"t0.5b": 0.5e9, "t1.5b": 1.5e9, "t3b": 3e9, "t7b": 7e9, "t14b": 14e9,
                              "t32b": 32e9, "t72b": 72e9}
PRIMARY_TEACHERS: tuple[str, ...] = ("t1.5b", "t3b", "t7b", "t14b", "t32b", "t72b")
# Model-predicted unit costs (compute_cost.md Task E table), $ per M delivered student tokens at $2/GPU-h
COST_USD_PER_MTOK: dict[str, float] = {"t0.5b": 0.015, "t1.5b": 0.02, "t3b": 0.03, "t7b": 0.06,
                                       "t14b": 0.12, "t32b": 0.28, "t72b": 0.72}
GPU_S_PER_MTOK: dict[str, float] = {"t0.5b": 30, "t1.5b": 40, "t3b": 47, "t7b": 115, "t14b": 224,
                                    "t32b": 496, "t72b": 1294}
D_SYN = 600_000_000

# plausible C0 accuracies per student (only used to place the simulated tasks above the floor)
_BASE_ACC: dict[str, dict[str, float]] = {
    "s025m": {"sciq": 0.55, "arc_easy": 0.34, "piqa": 0.58, "lambada_openai": 0.10, "openbookqa": 0.27,
              "social_iqa": 0.35, "hellaswag": 0.27},
    "s100m": {"sciq": 0.68, "arc_easy": 0.40, "piqa": 0.62, "lambada_openai": 0.25, "openbookqa": 0.30,
              "social_iqa": 0.38, "hellaswag": 0.30},
    "s250m": {"sciq": 0.74, "arc_easy": 0.44, "piqa": 0.65, "lambada_openai": 0.35, "openbookqa": 0.33,
              "social_iqa": 0.40, "hellaswag": 0.34},
    "s500m": {"sciq": 0.78, "arc_easy": 0.48, "piqa": 0.68, "lambada_openai": 0.42, "openbookqa": 0.35,
              "social_iqa": 0.42, "hellaswag": 0.38},
    "s1b": {"sciq": 0.82, "arc_easy": 0.52, "piqa": 0.70, "lambada_openai": 0.48, "openbookqa": 0.37,
            "social_iqa": 0.43, "hellaswag": 0.44},
}


def x_of(t_params: float) -> float:
    return math.log2(t_params / T_REF)


def s_of(s_params: float) -> float:
    return math.log2(s_params / S_REF)


def true_gain(surface: str, s_params: float, t_params: float, *, R: float = 3.0, beta: float = 0.5,
              kappa: float = 0.15, slope: float = 0.6, alpha: float = 0.64, tstar_25m: float = 3e9,
              t_grid: Sequence[float] = (1.5e9, 72e9)) -> float:
    """True cell mean gain over C1 (points) for the surfaces of statistics.md §3.2."""
    x, s = x_of(t_params), s_of(s_params)
    t1, t6 = t_grid[0], t_grid[-1]
    if surface == "null":
        return 0.0
    if surface == "lin":
        return slope * (x - x_of(t1))
    if surface == "sat":
        return R * (1 - (t_params / t1) ** (-beta)) / (1 - (t6 / t1) ** (-beta))
    if surface == "satS":
        r_s = R * (0.6 + 0.8 * s / s_of(1e9))
        return r_s * (1 - (t_params / t1) ** (-beta)) / (1 - (t6 / t1) ** (-beta))
    if surface == "peak":
        xstar = math.log2(tstar_25m / T_REF) + alpha * s
        return 2.0 - kappa * (x - xstar) ** 2
    raise ValueError(surface)


def simulate_index(
    surface: str = "peak",
    *,
    sigma: float = 0.5,
    r: int = 3,
    students: Sequence[str] = tuple(STUDENTS),
    teachers: Sequence[str] = PRIMARY_TEACHERS,
    seed: int = 0,
    c1_gain_over_c0: float = 1.0,
    c1b_gain_over_c0: float = 0.5,
    r_controls: Optional[int] = None,
    hetero: bool = False,
    dsweep: bool = False,
    dsweep_levels: Sequence[int] = (75_000_000, 150_000_000, 300_000_000),
    dsweep_slope: float = 1.0,
    study: str = "main",
    tier2_extra: int = 0,
    per_task_jitter: float = 0.0,
    **surface_kw: float,
) -> pd.DataFrame:
    """Run-level index with a known gain surface on P_cc (over C1) and consistent per-task accuracies.

    Per-task accuracies are ``a_k = base_k(S) + (g + ε)(1 − c_k)/100`` with the *same* seed noise
    ε ~ N(0, σ²) for every task, so that P_cc = P_cc(C0) + gain + ε exactly; ``per_task_jitter``
    adds independent per-task noise on top.  The D-sweep rows follow
    ``gain(D) = gain(D_syn) + dsweep_slope · log2(D / D_syn)``.  ``tier2_extra`` adds confirmation
    seeds (``tier == 2``) at every teacher cell.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    r_ctrl = r if r_controls is None else r_controls

    def noise(s_name: str) -> float:
        f = 1.0
        if hetero:
            f = {"s025m": 1.5, "s100m": 1.2}.get(s_name, 1.0)
        return float(rng.normal(0, sigma * f))

    def make_row(s_name: str, cond: str, arm: str, seed_phase: int, gain_over_c0: float, d_syn: int,
                 tier: int = 1, teacher: Optional[str] = None) -> dict:
        s_nom, s_ne = STUDENTS[s_name]
        eps = noise(s_name)
        row = {
            "experiment_id": f"{study}.{s_name}.{teacher or cond.lower()}.d{d_syn // 1_000_000}m.eqtok.s{seed_phase}-sim{len(rows):05d}",
            "study": study, "arm": arm, "student": s_name, "cond": cond, "teacher": teacher,
            "s_params_nominal": s_nom, "s_params_nonemb": s_ne,
            "t_params_nominal": TEACHERS[teacher] if teacher else np.nan,
            "d_syn": d_syn, "variant": "eqtok" if arm != "base_only" else "-",
            "seed_base": 1, "seed_phase": seed_phase, "tier": tier, "status": "finished",
        }
        for t in PRIMARY_TASKS:
            base = _BASE_ACC[s_name][t]
            a = base + (gain_over_c0 + eps) * (1 - CHANCE_TABLE[t]) / 100.0
            if per_task_jitter:
                a += float(rng.normal(0, per_task_jitter)) * (1 - CHANCE_TABLE[t]) / 100.0
            row[metric_column(t)] = float(min(max(a, 0.0), 1.0))
            # task loss: a smooth decreasing function of accuracy (bits/byte), purely illustrative
            row[f"task_loss_{t}"] = float(1.5 - 1.2 * row[metric_column(t)] + rng.normal(0, 0.005))
        row["nll_instr"] = float(2.6 - 0.01 * (gain_over_c0 + eps) + rng.normal(0, 0.003))
        row["nll_real_id"] = float(2.9 + 0.004 * max(gain_over_c0, 0) + rng.normal(0, 0.002))
        if teacher:
            row["c_usd"] = COST_USD_PER_MTOK[teacher] / 1e6
            row["c_tau"] = GPU_S_PER_MTOK[teacher] / 1e6
            row["cost_usd_cell"] = row["c_usd"] * d_syn
        else:
            row["c_usd"] = np.nan
            row["c_tau"] = np.nan
            row["cost_usd_cell"] = 0.0
        return row

    for s_name in students:
        s_nom = STUDENTS[s_name][0]
        rows.append(make_row(s_name, C0, "base_only", 1, 0.0, 0))
        for k in range(1, r_ctrl + 1):
            rows.append(make_row(s_name, C1, "matched_real", k, c1_gain_over_c0, D_SYN))
        for k in range(1, r + 1):
            rows.append(make_row(s_name, C1B, "human_instruct", k, c1b_gain_over_c0, D_SYN))
        for t_name in teachers:
            g = true_gain(surface, s_nom, TEACHERS[t_name], **surface_kw)
            for k in range(1, r + 1):
                rows.append(make_row(s_name, t_name, "synthetic", k, c1_gain_over_c0 + g, D_SYN, teacher=t_name))
            for k in range(r + 1, r + 1 + tier2_extra):
                rows.append(make_row(s_name, t_name, "synthetic", k, c1_gain_over_c0 + g, D_SYN, tier=2,
                                     teacher=t_name))
    if dsweep:
        for s_name in ("s100m", "s1b"):
            if s_name not in students:
                continue
            s_nom = STUDENTS[s_name][0]
            for t_name in ("t3b", "t14b", "t72b"):
                if t_name not in teachers:
                    continue
                g = true_gain(surface, s_nom, TEACHERS[t_name], **surface_kw)
                for d in dsweep_levels:
                    gd = g + dsweep_slope * math.log2(d / D_SYN)
                    for k in range(1, 3):
                        row = make_row(s_name, t_name, "synthetic", k, c1_gain_over_c0 + gd, int(d), teacher=t_name)
                        row["study"] = "dsweep"
                        row["experiment_id"] = row["experiment_id"].replace(f"{study}.", "dsweep.", 1)
                        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def true_gain_table(surface: str, students: Sequence[str] = tuple(STUDENTS),
                    teachers: Sequence[str] = PRIMARY_TEACHERS, **kw: float) -> pd.DataFrame:
    rows = [{"student": s, "teacher": t, "gain": true_gain(surface, STUDENTS[s][0], TEACHERS[t], **kw)}
            for s in students for t in teachers]
    return pd.DataFrame(rows)
