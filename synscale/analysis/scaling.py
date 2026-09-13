"""Synthetic-data scaling-law analysis (the core NeurIPS question).

For every run we already have a dense loss-vs-tokens curve in ``results/<id>/train_log.jsonl``
(the trainer evaluates held-out NLL every ~20M tokens during the synthetic phase). This module:

1. reads each run's curve and covariates (student S, teacher size T, arm/condition, seed);
2. fits the data-scaling law  L(D) = E + A * D^(-alpha)  per cell (E = irreducible loss,
   A = coefficient, alpha = data-scaling exponent), where D is synthetic tokens seen;
3. tests whether E, A, alpha depend on teacher size T (do bigger teachers give data that
   scales better?), within each student size, and pools across S;
4. compares synthetic curves against the matched-real control (C1): does synthetic data
   scale like real data, and does a larger teacher close the gap?

These three tests are the operational meaning of "do synthetic-data scaling laws depend on
the teacher?". Everything is loss-based (floor-free), which is why it works at 25M-250M.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _read_curve(run_dir: Path, eval_set: str, replay_fraction: float) -> list[tuple[float, float]]:
    """Return [(D_syn_tokens, loss)] from train_log.jsonl for one eval set (synthetic phase)."""
    log = run_dir / "train_log.jsonl"
    if not log.exists():
        return []
    pts = []
    for line in log.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "held_out_nll" in d and eval_set in d["held_out_nll"]:
            d_syn = d["tokens"] * (1.0 - replay_fraction)   # synthetic tokens seen so far
            loss = d["held_out_nll"][eval_set]
            if d_syn > 0 and loss is not None and math.isfinite(loss):
                pts.append((float(d_syn), float(loss)))
    return sorted(pts)


def _covariates(run_dir: Path) -> Optional[dict]:
    man = run_dir / "manifest.json"
    if not man.exists():
        return None
    m = json.loads(man.read_text())
    cfg = m.get("config", {})
    arm = cfg.get("arm")
    teacher = cfg.get("teacher")
    sp = (cfg.get("training", {}) or {}).get("synthetic_phase") or {}
    return {
        "experiment_id": m.get("experiment_id", run_dir.name),
        "student": (cfg.get("student", {}) or {}).get("name"),
        "s_params": (cfg.get("student", {}) or {}).get("target_params"),
        "arm": arm,
        "teacher": (teacher or {}).get("name") if teacher else None,
        "t_params": (teacher or {}).get("nominal_params") if teacher else None,
        "seed_phase": (cfg.get("training", {}) or {}).get("seed_phase"),
        "replay_fraction": sp.get("replay_fraction", 0.25),
        "cond": {"base_only": "C0", "matched_real": "C1", "human_instruct": "C1b", "synthetic": None}.get(arm),
    }


def collect_curves(results_dir: str | Path, eval_set: str = "instr") -> list[dict]:
    """One record per run: covariates + its loss-vs-D curve on `eval_set`."""
    results_dir = Path(results_dir)
    out = []
    for run_dir in sorted(results_dir.glob("*/")):
        cov = _covariates(run_dir)
        if not cov or cov["arm"] == "base_only":
            continue
        curve = _read_curve(run_dir, eval_set, cov["replay_fraction"])
        if len(curve) >= 4:                       # need enough points to fit 3 params
            cov["curve"] = curve
            cov["cond"] = cov["cond"] or cov["teacher"]
            out.append(cov)
    return out


def fit_power_law(D: np.ndarray, L: np.ndarray) -> dict[str, float]:
    """Fit L = E + A * D^(-alpha) by least squares over a grid of alpha (robust, no SciPy needed).

    For each alpha, (E, A) are linear in [1, D^-alpha] -> closed-form; pick the alpha with
    the smallest residual. Returns E, A, alpha, rmse, and the implied loss at D->inf (=E).
    """
    D = np.asarray(D, float); L = np.asarray(L, float)
    best = None
    for alpha in np.linspace(0.02, 1.0, 197):
        X = np.column_stack([np.ones_like(D), D ** (-alpha)])
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        resid = L - X @ coef
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        if coef[1] < 0:            # A must be >= 0 (loss decreases with data)
            continue
        if best is None or rmse < best["rmse"]:
            best = {"E": float(coef[0]), "A": float(coef[1]), "alpha": float(alpha), "rmse": rmse}
    if best is None:               # fallback: monotone but no valid A
        coef = np.polyfit(np.log(D), L, 1)
        best = {"E": float(L.min()), "A": float(abs(coef[0])), "alpha": 0.02, "rmse": float("nan")}
    return best


def _mean_curve(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Average loss across seeds at matched D. Single-seed cells return the curve directly;
    multi-seed cells are averaged into bins sized relative to the token range (so it works at
    both micro and full scale)."""
    if len(records) == 1:
        c = records[0]["curve"]
        return np.array([d for d, _ in c]), np.array([l for _, l in c])
    maxD = max(d for r in records for d, _ in r["curve"])
    bin_w = max(maxD / 40.0, 1.0)
    allD = sorted({round(d / bin_w) * bin_w for r in records for d, _ in r["curve"]})
    D, L = [], []
    for target in allD:
        vals = []
        for r in records:
            near = min(r["curve"], key=lambda p: abs(p[0] - target))
            if abs(near[0] - target) <= bin_w:
                vals.append(near[1])
        if vals:
            D.append(target); L.append(float(np.mean(vals)))
    return np.asarray(D), np.asarray(L)


def fit_cells(records: list[dict]) -> list[dict]:
    """Fit the power law per (student, cond) cell, averaging seeds."""
    cells: dict[tuple, list[dict]] = {}
    for r in records:
        cells.setdefault((r["student"], r["cond"]), []).append(r)
    rows = []
    for (student, cond), recs in cells.items():
        D, L = _mean_curve(recs)
        if len(D) < 4:
            continue
        fit = fit_power_law(D, L)
        rows.append({"student": student, "s_params": recs[0]["s_params"], "cond": cond,
                     "teacher": recs[0]["teacher"], "t_params": recs[0]["t_params"],
                     "n_seeds": len(recs), "n_points": len(D), **fit})
    return rows


def test_teacher_dependence(cell_fits: list[dict]) -> dict[str, Any]:
    """Within each student, regress each scaling parameter on log2(teacher size).

    A significant positive slope of alpha on log T means bigger teachers produce data that
    scales better (a genuine 'synthetic-data scaling law depends on the teacher' result);
    a slope on E means bigger teachers lower the irreducible loss. Reported descriptively
    (few teacher points), with the sign, slope and R^2.
    """
    import numpy as np
    syn = [r for r in cell_fits if r["t_params"]]
    out: dict[str, Any] = {"per_student": {}, "pooled": {}}
    by_s: dict[str, list[dict]] = {}
    for r in syn:
        by_s.setdefault(r["student"], []).append(r)
    for s, rows in by_s.items():
        if len(rows) < 3:
            continue
        x = np.log2(np.array([r["t_params"] for r in rows]) / 1e9)
        d = {}
        for param in ("alpha", "E", "A"):
            y = np.array([r[param] for r in rows])
            if np.ptp(x) == 0:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            ss = float(1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
            d[param] = {"slope_per_log2T": float(slope), "r2": ss, "values": y.tolist(), "logT": x.tolist()}
        out["per_student"][s] = d
    return out


def synthetic_vs_real(cell_fits: list[dict]) -> list[dict]:
    """Per student: gap between each teacher's irreducible loss E and the real-data control's E.

    Negative gap = the teacher's synthetic data reaches a lower asymptotic loss than matched
    real tokens; how the gap moves with teacher size is the headline scaling comparison.
    """
    out = []
    for s in {r["student"] for r in cell_fits}:
        real = next((r for r in cell_fits if r["student"] == s and r["cond"] == "C1"), None)
        if not real:
            continue
        for r in cell_fits:
            if r["student"] == s and r["t_params"]:
                out.append({"student": s, "teacher": r["teacher"], "t_params": r["t_params"],
                            "E_syn": r["E"], "E_real": real["E"], "E_gap": r["E"] - real["E"],
                            "alpha_syn": r["alpha"], "alpha_real": real["alpha"]})
    return out


def run_scaling_analysis(results_dir: str | Path, out_dir: str | Path, eval_set: str = "instr") -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = collect_curves(results_dir, eval_set=eval_set)
    report: dict[str, Any] = {"eval_set": eval_set, "n_runs_with_curves": len(records)}
    if len(records) < 2:
        report["note"] = "not enough curves yet (need runs with >=4 eval points)"
        (out_dir / "scaling_report.json").write_text(json.dumps(report, indent=2))
        return report
    fits = fit_cells(records)
    report["cell_fits"] = fits
    report["teacher_dependence"] = test_teacher_dependence(fits)
    report["synthetic_vs_real"] = synthetic_vs_real(fits)
    (out_dir / "scaling_report.json").write_text(json.dumps(report, indent=2, default=str))
    # a compact CSV of the per-cell fits
    if fits:
        import csv
        keys = ["student", "s_params", "cond", "teacher", "t_params", "n_seeds", "n_points", "E", "A", "alpha", "rmse"]
        with open(out_dir / "scaling_fits.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in fits:
                w.writerow({k: r.get(k) for k in keys})
    return report
