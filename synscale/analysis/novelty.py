"""The novel analyses (docs/24). Everything here operates on fitted scaling cells plus the
student-referenced data-quality index, and every test is designed to have a clean negative.

N2 screening (centerpiece): is teacher identity (size, family) screened off by the measured
    data-quality index q? Compare out-of-sample prediction of a cell outcome from
    (log size + family) vs (q) vs (both), by leave-one-teacher-out CV. If q predicts as well as
    identity and identity adds nothing beyond q, the scaling is governed by measurable data
    properties, not model identity.
N3 predictive law: fit the joint law on a subset of cells and predict held-out cells.
N4 compute-optimal allocation: at a fixed total FLOP budget split between teacher generation
    and student training, which teacher size minimises student loss, and how does the optimum
    move with student size?
N5 cross-family multiplier invariance: is the synthetic-data multiplier a function of q alone?
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _ols_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Leave-one-group-out R^2 (out-of-sample). groups = teacher id per row."""
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return float("nan")
    preds = np.full_like(y, np.nan, dtype=float)
    for g in uniq:
        tr = groups != g; te = groups == g
        Xtr = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        preds[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
    ss_res = np.nansum((y - preds) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def screening_test(cells: list[dict], outcome: str = "L_inf_hat") -> dict[str, Any]:
    """N2. cells: dicts with student, teacher, t_params, family, q, and `outcome`.

    Returns leave-one-teacher-out R^2 for three predictor sets, per student (student intercept
    absorbed by working within a student). Interpretation: if R2(q) ~ R2(both) >> R2(identity
    alone is not needed), identity is screened off by q.
    """
    out: dict[str, Any] = {"outcome": outcome, "per_student": {}}
    by_s: dict[str, list[dict]] = {}
    for c in cells:
        if c.get("q") is None or not (c["q"] == c["q"]):
            continue
        by_s.setdefault(c["student"], []).append(c)
    for s, rows in by_s.items():
        if len(rows) < 4:
            continue
        y = np.array([r[outcome] for r in rows], float)
        groups = np.array([r["teacher"] for r in rows])
        logT = np.log2(np.array([r["t_params"] for r in rows], float) / 1e9)
        fams = sorted({r.get("family", "?") for r in rows})
        fam_oh = np.array([[1.0 if r.get("family") == f else 0.0 for f in fams[1:]] for r in rows]) \
            if len(fams) > 1 else np.zeros((len(rows), 0))
        q = np.array([r["q"] for r in rows], float).reshape(-1, 1)
        X_id = np.column_stack([logT] + ([fam_oh] if fam_oh.size else []))
        X_q = q
        X_both = np.column_stack([X_q, X_id])
        out["per_student"][s] = {
            "n_teachers": len(rows),
            "r2_identity(logT+family)": _ols_cv(X_id, y, groups),
            "r2_quality(q)": _ols_cv(X_q, y, groups),
            "r2_both": _ols_cv(X_both, y, groups),
        }
    # verdict helper
    verdicts = {}
    for s, d in out["per_student"].items():
        rq, rid, rb = d["r2_quality(q)"], d["r2_identity(logT+family)"], d["r2_both"]
        if rq == rq and rid == rid:
            screened = (rq >= rid - 0.05) and (rb <= rq + 0.05)
            verdicts[s] = "q screens off identity" if screened else "identity adds beyond q"
    out["verdict_per_student"] = verdicts
    return out


def predictive_law(records_by_cell: dict[tuple, list[tuple]], holdout_frac: float = 0.25,
                   seed: int = 0) -> dict[str, Any]:
    """N3. records_by_cell maps (student, teacher) -> loss-vs-D curve [(D, L)].

    Fit L = a_S + b*logT + c*logD + d*(logT*logD) on a random subset of (cell, D) points and
    predict held-out cells' points out of sample. Returns train/test RMSE.
    """
    rng = np.random.default_rng(seed)
    rows = []
    students = sorted({s for s, _ in records_by_cell})
    s_index = {s: i for i, s in enumerate(students)}
    for (s, t), curve in records_by_cell.items():
        tp = _teacher_params_from_name(t)
        if tp is None:
            continue
        for D, L in curve:
            rows.append((s_index[s], math.log2(tp / 1e9), math.log2(max(D, 1) / 1e6), L, f"{s}:{t}"))
    if len(rows) < 12:
        return {"note": "not enough points for a predictive fit"}
    cells = sorted({r[4] for r in rows})
    rng.shuffle(cells)
    n_hold = max(1, int(len(cells) * holdout_frac))
    hold = set(cells[:n_hold])
    def design(rr):
        s_oh = np.zeros((len(rr), len(students)))
        for i, r in enumerate(rr):
            s_oh[i, r[0]] = 1.0
        logT = np.array([r[1] for r in rr]); logD = np.array([r[2] for r in rr])
        return np.column_stack([s_oh, logT, logD, logT * logD]), np.array([r[3] for r in rr])
    tr = [r for r in rows if r[4] not in hold]; te = [r for r in rows if r[4] in hold]
    Xtr, ytr = design(tr); Xte, yte = design(te)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    rmse_tr = float(np.sqrt(np.mean((ytr - Xtr @ beta) ** 2)))
    rmse_te = float(np.sqrt(np.mean((yte - Xte @ beta) ** 2)))
    return {"train_rmse": rmse_tr, "heldout_rmse": rmse_te, "n_heldout_cells": len(hold),
            "heldout_cells": sorted(hold)}


def compute_allocation_frontier(fit_by_cell: dict[tuple, dict], student_params: dict[str, float],
                                budgets_flop: Optional[list[float]] = None) -> dict[str, Any]:
    """N4. For each student and total FLOP budget C, split C between teacher generation
    (2*T per token) and student training (6*N per token) so that D tokens cost (2T+6N)*D,
    then choose the teacher size minimising the fitted loss L(S,T,D=C/(2T+6N)).

    fit_by_cell maps (student, teacher_name) -> {'E','A','alpha','t_params'} (power-asymptote).
    Returns, per student and budget, the cost-optimal teacher and the implied generation share.
    """
    budgets_flop = budgets_flop or [1e17, 3e17, 1e18, 3e18, 1e19]
    out: dict[str, Any] = {"per_student": {}}
    for s, N in student_params.items():
        cells = {t: f for (ss, t), f in fit_by_cell.items() if ss == s and f.get("t_params")}
        if len(cells) < 3:
            continue
        rows = []
        for C in budgets_flop:
            best = None
            for t, f in cells.items():
                T = f["t_params"]
                D = C / (2 * T + 6 * N)             # tokens affordable at this split
                if D <= 0:
                    continue
                L = f["E"] + f["A"] * D ** (-f["alpha"])
                gen_share = (2 * T) / (2 * T + 6 * N)
                if best is None or L < best["loss"]:
                    best = {"teacher": t, "t_params": T, "loss": float(L), "D_tokens": float(D),
                            "gen_flop_share": float(gen_share)}
            if best:
                rows.append({"budget_flop": C, **best})
        out["per_student"][s] = rows
    return out


def _teacher_params_from_name(name: str) -> Optional[float]:
    import re
    m = re.search(r"(\d+(?:p\d+)?)b", name)
    if not m:
        return None
    return float(m.group(1).replace("p", ".")) * 1e9
