"""Synthetic-data scaling-law analysis (the core investigation).

Each run records a dense loss-vs-tokens curve in results/<id>/train_log.jsonl (held-out NLL
every ~20M tokens during the synthetic phase). This module fits data-scaling curves and asks
whether they depend on teacher size, with three deliberate statistical safeguards that the
project's reviewer asked us to lock:

1. CHECKPOINTS ARE NOT INDEPENDENT. The ~20 points on one curve come from a single
   optimisation trajectory and are highly autocorrelated. We therefore fit a curve SEPARATELY
   PER SEED (one independent run = one observation of the parameters) and take uncertainty from
   the spread ACROSS SEEDS, never from the count of checkpoints. Single-seed cells are reported
   as point estimates with an explicit "no seed CI" flag.
2. "E" IS AN ESTIMATED ASYMPTOTE, NOT THE IRREDUCIBLE LOSS. Over a limited D range E, A and
   alpha trade off. We fit three functional forms and compare them on HELD-OUT checkpoints
   (the last third of each curve), and report E as "L_inf_hat (observed regime)".
3. THE MULTIPLIER IS PREFERRED OVER A BESPOKE EFFICIENCY METRIC. For each teacher we report the
   synthetic-data multiplier: real tokens / synthetic tokens needed to reach the same loss.

Functional forms:
    power_asymptote :  L(D) = E + A * D^(-alpha)      (primary)
    log            :  L(D) = a - b * log D
    power_pure     :  L(D) = A * D^(-alpha)           (no free asymptote)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


# -------------------------------------------------------------------- curve + covariate reading
def _read_curve(run_dir: Path, eval_set: str, replay_fraction: float) -> list[tuple[float, float]]:
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
            d_syn = d["tokens"] * (1.0 - replay_fraction)     # synthetic tokens seen so far
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
    arm = cfg.get("arm"); teacher = cfg.get("teacher")
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
        "cond": {"matched_real": "C1", "human_instruct": "C1b", "synthetic": None}.get(arm),
    }


def collect_curves(results_dir: str | Path, eval_set: str = "instr") -> list[dict]:
    results_dir = Path(results_dir)
    out = []
    for run_dir in sorted(results_dir.glob("*/")):
        cov = _covariates(run_dir)
        if not cov or cov["arm"] == "base_only":
            continue
        curve = _read_curve(run_dir, eval_set, cov["replay_fraction"])
        if len(curve) >= 4:
            cov["curve"] = curve
            cov["cond"] = cov["cond"] or cov["teacher"]
            out.append(cov)
    return out


# ------------------------------------------------------------------------------ functional forms
def _fit_power_asymptote(D, L):
    """L = E + A*D^-alpha; (E,A) linear given alpha, so grid-search alpha (no SciPy needed)."""
    best = None
    for alpha in np.linspace(0.02, 1.2, 237):
        X = np.column_stack([np.ones_like(D), D ** (-alpha)])
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        if coef[1] < 0:
            continue
        rmse = float(np.sqrt(np.mean((L - X @ coef) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"form": "power_asymptote", "E": float(coef[0]), "A": float(coef[1]),
                    "alpha": float(alpha), "rmse": rmse}
    if best is None:
        best = {"form": "power_asymptote", "E": float(np.min(L)), "A": 0.0, "alpha": 0.02, "rmse": float("nan")}
    return best


def _fit_log(D, L):
    b, a = np.polyfit(np.log(D), L, 1)  # L = a + b*log D  (b<0 expected)
    rmse = float(np.sqrt(np.mean((L - (a + b * np.log(D))) ** 2)))
    return {"form": "log", "a": float(a), "b": float(-b), "rmse": rmse}


def _fit_power_pure(D, L):
    best = None
    for alpha in np.linspace(0.02, 1.2, 237):
        A = float(np.mean(L * D ** (alpha)))       # least squares for single coeff on D^-alpha
        # closed form: minimise ||L - A*D^-alpha||: A = <L, D^-a>/<D^-a, D^-a>
        w = D ** (-alpha)
        A = float(np.dot(L, w) / np.dot(w, w))
        rmse = float(np.sqrt(np.mean((L - A * w) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"form": "power_pure", "A": A, "alpha": float(alpha), "rmse": rmse}
    return best


def fit_forms(curve: list[tuple[float, float]]) -> dict[str, Any]:
    """Fit all three forms and score each on held-out checkpoints (last third of the curve).

    Held-out RMSE is the honest comparison: it penalises a form that only interpolates. The
    'best_form' is the one with the smallest held-out RMSE.
    """
    D = np.array([d for d, _ in curve], float)
    L = np.array([l for _, l in curve], float)
    k = max(3, int(round(len(D) * 0.67)))
    Dtr, Ltr, Dte, Lte = D[:k], L[:k], D[k:], L[k:]
    forms = {}
    for name, fn in (("power_asymptote", _fit_power_asymptote), ("log", _fit_log), ("power_pure", _fit_power_pure)):
        full = fn(D, L)
        tr = fn(Dtr, Ltr) if len(Dte) else full
        ho = float(np.sqrt(np.mean((Lte - _predict(tr, Dte)) ** 2))) if len(Dte) else float("nan")
        full["heldout_rmse"] = ho
        forms[name] = full
    valid = {k2: v for k2, v in forms.items() if v.get("heldout_rmse") == v.get("heldout_rmse")}
    best = min(valid, key=lambda k2: valid[k2]["heldout_rmse"]) if valid else "power_asymptote"
    return {"forms": forms, "best_form": best}


def _predict(fit: dict, D: np.ndarray) -> np.ndarray:
    if fit["form"] == "power_asymptote":
        return fit["E"] + fit["A"] * D ** (-fit["alpha"])
    if fit["form"] == "log":
        return fit["a"] - fit["b"] * np.log(D)
    return fit["A"] * D ** (-fit["alpha"])


def _invert_power_asymptote(fit: dict, L_target: float) -> Optional[float]:
    """D such that E + A*D^-alpha = L_target (needs L_target > E and A>0)."""
    if fit["A"] <= 0 or L_target <= fit["E"]:
        return None
    return float((fit["A"] / (L_target - fit["E"])) ** (1.0 / fit["alpha"]))


# ------------------------------------------------------------ per-seed fits + across-seed CIs
def fit_cells(records: list[dict]) -> list[dict]:
    """Fit the primary form PER SEED, then aggregate parameters across seeds (the correct unit).

    Uncertainty is the across-seed spread; single-seed cells carry seed_ci=False. Checkpoints
    within a run are NOT counted as independent (safeguard 1).
    """
    cells: dict[tuple, list[dict]] = {}
    for r in records:
        cells.setdefault((r["student"], r["cond"]), []).append(r)
    rows = []
    for (student, cond), recs in cells.items():
        per_seed = [_fit_power_asymptote(np.array([d for d, _ in r["curve"]]),
                                         np.array([l for _, l in r["curve"]])) for r in recs]
        forms_cmp = fit_forms(recs[0]["curve"])
        def agg(key):
            vals = np.array([f[key] for f in per_seed], float)
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            hw = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else float("nan")  # normal approx; n small
            return m, sd, hw
        E, E_sd, E_hw = agg("E"); A, A_sd, A_hw = agg("A"); al, al_sd, al_hw = agg("alpha")
        rows.append({
            "student": student, "s_params": recs[0]["s_params"], "cond": cond,
            "teacher": recs[0]["teacher"], "t_params": recs[0]["t_params"],
            "n_seeds": len(recs), "seed_ci": len(recs) > 1,
            "L_inf_hat": E, "L_inf_sd": E_sd, "L_inf_ci_hw": E_hw,   # "E": estimated asymptote, not true irreducible loss
            "A": A, "A_sd": A_sd, "alpha": al, "alpha_sd": al_sd, "alpha_ci_hw": al_hw,
            "best_form": forms_cmp["best_form"],
            "heldout_rmse_by_form": {k: v.get("heldout_rmse") for k, v in forms_cmp["forms"].items()},
            "_fit": {"form": "power_asymptote", "E": E, "A": A, "alpha": al},
        })
    return rows


def test_teacher_dependence(cell_fits: list[dict]) -> dict[str, Any]:
    """Per student, regress each scaling parameter on log2(teacher size). Descriptive (few T points)."""
    syn = [r for r in cell_fits if r["t_params"]]
    out: dict[str, Any] = {"per_student": {}, "caveat": "checkpoints within a run are autocorrelated; "
                           "slopes use per-seed-aggregated cell parameters and only 4 teacher points, so they are descriptive."}
    by_s: dict[str, list[dict]] = {}
    for r in syn:
        by_s.setdefault(r["student"], []).append(r)
    for s, rows in by_s.items():
        if len(rows) < 3:
            continue
        x = np.log2(np.array([r["t_params"] for r in rows]) / 1e9)
        d = {}
        for param, key in (("alpha", "alpha"), ("L_inf_hat", "L_inf_hat"), ("A", "A")):
            y = np.array([r[key] for r in rows])
            if np.ptp(x) == 0:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            r2 = float(1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
            d[param] = {"slope_per_log2T": float(slope), "r2": r2, "logT": x.tolist(), "values": y.tolist()}
        out["per_student"][s] = d
    return out


def synthetic_data_multiplier(cell_fits: list[dict], n_levels: int = 5) -> list[dict]:
    """Real tokens / synthetic tokens to reach the same loss (safeguard 3, preferred metric).

    For each (student, teacher) we invert the fitted power-asymptote curves for the teacher and
    the matched-real control (C1) at loss levels spanning their common range, and report the
    ratio D_real(L)/D_syn(L). >1 means one synthetic token is worth more than one real token
    over that loss range.
    """
    out = []
    for s in {r["student"] for r in cell_fits}:
        real = next((r for r in cell_fits if r["student"] == s and r["cond"] == "C1"), None)
        if not real:
            continue
        rf = real["_fit"]
        for r in cell_fits:
            if r["student"] != s or not r["t_params"]:
                continue
            tf = r["_fit"]
            lo = max(rf["E"], tf["E"]) + 0.02
            hi = min(rf["E"] + rf["A"], tf["E"] + tf["A"])  # loss at D=1 upper bound-ish
            if not (hi > lo):
                continue
            mults = []
            for L in np.linspace(lo, hi, n_levels):
                d_syn = _invert_power_asymptote(tf, L); d_real = _invert_power_asymptote(rf, L)
                if d_syn and d_real and d_syn > 0:
                    mults.append(d_real / d_syn)
            if mults:
                out.append({"student": s, "teacher": r["teacher"], "t_params": r["t_params"],
                            "multiplier_median": float(np.median(mults)),
                            "multiplier_min": float(np.min(mults)), "multiplier_max": float(np.max(mults)),
                            "seed_ci": r["seed_ci"] and real["seed_ci"]})
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
    report["synthetic_data_multiplier"] = synthetic_data_multiplier(fits)
    n_single = sum(1 for f in fits if not f["seed_ci"])
    if n_single:
        report["warning"] = f"{n_single}/{len(fits)} cells have a single seed (no across-seed CI); run >=2 seeds for inference."
    (out_dir / "scaling_report.json").write_text(json.dumps(report, indent=2, default=str))
    if fits:
        keys = ["student", "s_params", "cond", "teacher", "t_params", "n_seeds", "seed_ci",
                "L_inf_hat", "L_inf_sd", "A", "alpha", "alpha_sd", "best_form"]
        with open(out_dir / "scaling_fits.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in fits:
                w.writerow({k: r.get(k) for k in keys})
    return report


# backward-compatible alias for the earlier public name
fit_power_law = _fit_power_asymptote
