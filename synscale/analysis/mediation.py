"""Exploratory mediation models M0–M4 (synthetic_data.md §4.3).

Fitted by least squares across the (S, T) cells with student fixed effects a_S, evaluated by
leave-one-teacher-out cross-validation (LOTO-CV: all cells of one teacher held out; RMSE
averaged over held-out teachers).

    M0 (size only):        G(S,T) = a_S + b · log T
    M1 (learnability):     G(S,T) = a_S + β₁ ℓ_S(T) + β₂ ℓ_S(T)²
    M2 (correctness):      G(S,T) = a_S + γ · c_T
    M3 (size + mediator):  G(S,T) = a_S + b · log T + β₁ ℓ_S(T)
    M4 (diversity):        G(S,T) = a_S + δ · d_T

Pre-registered tests: (i) M1 beats M0 in LOTO-CV and β₂ < 0; (ii) in M3, b is not
distinguishable from 0 once ℓ_S(T) is included; (iii) the argmax_T of the M1 fit shifts with S.
Input: a cell table with columns ``student``, ``t_params_nominal``, ``G`` (seed-averaged gain over
C1) and mediator columns ``ell`` (ℓ_S(T)), ``c_T``, ``d_T``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from synscale.analysis.aggregate import T_REF

MEDIATION_MODELS = {
    "M0": ("logT",), "M1": ("ell", "ell2"), "M2": ("c_T",), "M3": ("logT", "ell"), "M4": ("d_T",),
}


@dataclass
class MediationFit:
    model: str
    terms: tuple[str, ...]
    params: pd.Series
    se: pd.Series
    p: pd.Series
    rss: float
    n: int
    k: int
    df: int
    loto_rmse: float
    fitted: np.ndarray


def _features(cells: pd.DataFrame) -> pd.DataFrame:
    f = cells.copy()
    f["logT"] = np.log2(f["t_params_nominal"].astype(float) / T_REF)
    if "ell" in f:
        f["ell2"] = f["ell"].astype(float) ** 2
    return f


def _design(f: pd.DataFrame, terms: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    levels = sorted(f["student"].unique(), key=lambda s: float(f.loc[f["student"] == s, "s_params_nominal"].iloc[0])
                    if "s_params_nominal" in f else s)
    fe = np.stack([(f["student"] == lv).to_numpy(dtype=float) for lv in levels], axis=1)
    X = np.hstack([fe] + [f[t].to_numpy(dtype=float)[:, None] for t in terms])
    return X, [f"a[{lv}]" for lv in levels] + list(terms)


def fit_mediation(cells: pd.DataFrame, model: str, outcome: str = "G") -> MediationFit:
    terms = MEDIATION_MODELS[model]
    f = _features(cells).dropna(subset=[outcome, *[t for t in terms]])
    X, names = _design(f, terms)
    y = f[outcome].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    rss = float(((y - fitted) ** 2).sum())
    n, k = X.shape
    dfree = n - k
    s2 = rss / dfree if dfree > 0 else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    p = 2 * stats.t.sf(np.abs(coef / np.where(se > 0, se, np.nan)), dfree) if dfree > 0 else np.full(k, np.nan)
    return MediationFit(model, tuple(terms), pd.Series(coef, index=names), pd.Series(se, index=names),
                        pd.Series(p, index=names), rss, n, k, dfree, loto_cv(f, terms, outcome), fitted)


def loto_cv(f: pd.DataFrame, terms: Sequence[str], outcome: str = "G") -> float:
    """Leave-one-teacher-out RMSE averaged over held-out teachers."""
    errs = []
    for t in f["t_params_nominal"].unique():
        train, test = f[f["t_params_nominal"] != t], f[f["t_params_nominal"] == t]
        if len(train) == 0 or len(test) == 0:
            continue
        Xtr, _ = _design(train, terms)
        coef, *_ = np.linalg.lstsq(Xtr, train[outcome].to_numpy(dtype=float), rcond=None)
        # build the test design with the same student ordering as training
        levels = [n[2:-1] for n in _design(train, terms)[1] if n.startswith("a[")]
        fe = np.stack([(test["student"] == lv).to_numpy(dtype=float) for lv in levels], axis=1)
        Xte = np.hstack([fe] + [test[tm].to_numpy(dtype=float)[:, None] for tm in terms])
        pred = Xte @ coef
        errs.append(float(np.sqrt(np.mean((test[outcome].to_numpy(dtype=float) - pred) ** 2))))
    return float(np.mean(errs)) if errs else float("nan")


def fit_all(cells: pd.DataFrame, outcome: str = "G", models: Sequence[str] = tuple(MEDIATION_MODELS)) -> dict[str, MediationFit]:
    out = {}
    for m in models:
        needed = [t for t in MEDIATION_MODELS[m] if t not in ("logT", "ell2")]
        if all(t in cells.columns for t in needed) and (m != "M1" or "ell" in cells.columns):
            out[m] = fit_mediation(cells, m, outcome)
    return out


def mediation_summary(fits: dict[str, MediationFit]) -> pd.DataFrame:
    rows = []
    for m, f in fits.items():
        row = {"model": m, "n": f.n, "k": f.k, "rss": f.rss, "loto_rmse": f.loto_rmse}
        for t in f.terms:
            row[f"{t}_est"] = f.params[t]; row[f"{t}_se"] = f.se[t]; row[f"{t}_p"] = f.p[t]
        rows.append(row)
    return pd.DataFrame(rows)


def mediation_tests(fits: dict[str, MediationFit]) -> dict[str, Any]:
    """The three pre-registered mediation tests (where the needed models were fitted)."""
    out: dict[str, Any] = {}
    if "M0" in fits and "M1" in fits:
        f1 = fits["M1"]
        out["H4_mechanism"] = {
            "M1_beats_M0_loto": bool(f1.loto_rmse < fits["M0"].loto_rmse),
            "beta2": float(f1.params["ell2"]), "beta2_se": float(f1.se["ell2"]),
            "beta2_p_one_sided": float(stats.t.cdf(f1.params["ell2"] / f1.se["ell2"], f1.df)) if f1.se["ell2"] > 0 else float("nan"),
            "beta2_negative": bool(f1.params["ell2"] < 0),
        }
    if "M3" in fits:
        f3 = fits["M3"]
        out["H1_size_given_mediator"] = {"b": float(f3.params["logT"]), "b_se": float(f3.se["logT"]),
                                         "b_p": float(f3.p["logT"]),
                                         "b_indistinguishable_from_0": bool(f3.p["logT"] > 0.05)}
    return out


def m1_argmax_by_student(cells: pd.DataFrame, fit: MediationFit) -> pd.DataFrame:
    """argmax_T of the M1 fit per student (H3: does the learnability optimum shift with S?)."""
    f = _features(cells)
    b1, b2 = float(fit.params["ell"]), float(fit.params["ell2"])
    rows = []
    for st, g in f.groupby("student"):
        pred = fit.params[f"a[{st}]"] + b1 * g["ell"] + b2 * g["ell"] ** 2
        i = int(np.argmax(pred.to_numpy()))
        rows.append({"student": st, "argmax_T": float(g["t_params_nominal"].iloc[i]),
                     "ell_at_argmax": float(g["ell"].iloc[i]),
                     "ell_star_unconstrained": -b1 / (2 * b2) if b2 < 0 else float("nan")})
    return pd.DataFrame(rows)


def cells_from_index(df: pd.DataFrame, outcome: str = "p_cc", mediators: Optional[pd.DataFrame] = None
                     ) -> pd.DataFrame:
    """Seed-averaged gain over C1 per (S, T) cell, optionally joined with a mediator table
    (columns ``student``/``t_params_nominal`` (or ``teacher``) + ``ell``/``c_T``/``d_T``)."""
    from synscale.analysis.ste import gains_table

    g = gains_table(df, outcome)[["student", "s_params_nominal", "t_params_nominal", "teacher", "G1"]].rename(columns={"G1": "G"})
    if mediators is not None:
        key = ["student", "t_params_nominal"] if "t_params_nominal" in mediators else ["student", "teacher"]
        if "student" not in mediators:
            key = key[1:]
        g = g.merge(mediators, on=key, how="left")
    return g
