"""Generation cost and Synthetic Transfer Efficiency (compute_cost.md Task E; locked §0.13).

* Unit costs per delivered student token from a generation manifest: c_τ [GPU-s/token],
  c_$ = c_τ·p [$/token] (p = $2.00/GPU-h reference, $0.95 owned sensitivity), c_E [kWh/token],
  c_F = 2·T·(prompt + response teacher tokens)/D_pool [FLOP/token].
* Gains G₀ = M(S,T) − M(S,C0), G₁ = M(S,T) − M(S,C1); **STE(S,T) = G₁ / (c_$(T)·D_syn)**.
* Marginal efficiency mᵢ between cost-adjacent teachers; price-of-a-point curve
  T$(S; λ) = argmax_T [G₁ − λ·C] on a log λ grid; ε-rule T$_ε (cheapest T with G₁ ≥ (1−ε)·max);
  λ₁ = marginal efficiency of the last upgrade into T*; H5 supported iff the 95 % CI of λ₁ lies
  below λ_ref = G₁(T₁)/C(T₁) (average efficiency of the cheapest teacher).
* Bootstrap propagating seed uncertainty (pooled-variance parametric, primary; non-parametric
  seed resampling as a check) and cost uncertainty (chunk/shard resampling), with selection
  probabilities for T$(λ) ("stable" if ≥ 0.8).
* Iso-cost comparison: P(S,T,D) interpolated from the D-sweep and evaluated at D(T) = B/c(T).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from synscale.analysis.aggregate import C0, C1
from synscale.analysis.regression import within_s_variance

PRICE_PER_GPU_H_REFERENCE = 2.00
PRICE_PER_GPU_H_OWNED = 0.95
PRICE_SENSITIVITY = (0.95, 2.00, 3.00)
USD_PER_KWH = 0.15
LAMBDA_GRID = tuple([0.0] + list(np.geomspace(1e-3, 10.0, 41)))


# ----------------------------------------------------------------------------------------
# Unit costs from generation manifests (synthetic_data.md §6.2 schema; tolerant to extra fields)
# ----------------------------------------------------------------------------------------
@dataclass
class UnitCosts:
    teacher: str
    tokens_delivered: float  # student tokens after filtering (D_pool)
    gpu_seconds: float
    energy_kwh: float
    teacher_tokens_processed: float
    c_tau: float  # GPU-s per delivered student token
    c_usd: float  # $ per token at the reference price
    c_kwh: float
    c_flop: float
    chunks: list[tuple[float, float, float]]  # (gpu_seconds, tokens_delivered, kwh) per shard
    price_per_gpu_h: float

    def as_dict(self) -> dict[str, float]:
        return {"c_tau": self.c_tau, "c_usd": self.c_usd, "c_kwh": self.c_kwh, "c_flop": self.c_flop,
                "gen_gpu_seconds": self.gpu_seconds, "gen_tokens_delivered": self.tokens_delivered,
                "gen_energy_kwh": self.energy_kwh}


def unit_costs_from_manifest(man: Mapping[str, Any], *, price_per_gpu_h: float = PRICE_PER_GPU_H_REFERENCE,
                             pue: float = 1.0, teacher_params: Optional[float] = None) -> UnitCosts:
    """Compute c_τ, c_$, c_E, c_F from a generation manifest.

    Shard fields used: ``wall_s`` (× ``n_gpus``/``tensor_parallel`` of the manifest), ``tokens_student``
    (delivered, i.e. retained; ``tokens_student_retained`` preferred if present), ``energy_kwh``,
    ``tokens_teacher`` (+ ``tokens_teacher_prompt``).  Aggregate keys ``gpu_seconds``,
    ``tokens_student`` and ``energy_kwh`` are used if there are no shards.
    """
    n_gpus = float(man.get("n_gpus") or man.get("tensor_parallel") or 1)
    shards = man.get("shards") or []
    chunks: list[tuple[float, float, float]] = []
    tt = 0.0
    for sh in shards:
        gpu_s = float(sh.get("gpu_seconds", float(sh.get("wall_s", 0.0)) * n_gpus))
        tok = float(sh.get("tokens_student_retained", sh.get("tokens_student", 0.0)))
        kwh = float(sh.get("energy_kwh", 0.0)) * pue
        tt += float(sh.get("tokens_teacher", 0.0)) + float(sh.get("tokens_teacher_prompt", 0.0))
        chunks.append((gpu_s, tok, kwh))
    if chunks:
        gpu_s = sum(c[0] for c in chunks); tok = sum(c[1] for c in chunks); kwh = sum(c[2] for c in chunks)
    else:
        gpu_s = float(man.get("gpu_seconds", 0.0)); tok = float(man.get("tokens_student", 0.0))
        kwh = float(man.get("energy_kwh", 0.0)) * pue
        tt = float(man.get("tokens_teacher", 0.0)) + float(man.get("tokens_teacher_prompt", 0.0))
    if teacher_params is None:
        teacher_params = float(man.get("teacher_params", man.get("nominal_params", 0.0)) or 0.0)
    c_tau = gpu_s / tok if tok else float("nan")
    return UnitCosts(
        teacher=str(man.get("teacher", "?")), tokens_delivered=tok, gpu_seconds=gpu_s, energy_kwh=kwh,
        teacher_tokens_processed=tt, c_tau=c_tau, c_usd=c_tau * price_per_gpu_h / 3600.0,
        c_kwh=kwh / tok if tok else float("nan"),
        c_flop=2.0 * teacher_params * tt / tok if tok and teacher_params else float("nan"),
        chunks=chunks, price_per_gpu_h=price_per_gpu_h,
    )


# ----------------------------------------------------------------------------------------
# Gains and STE
# ----------------------------------------------------------------------------------------
def gains_table(df: pd.DataFrame, outcome: str = "p_cc", *, cost_col: str = "c_usd") -> pd.DataFrame:
    """Per (student, teacher): seed values, G₀, G₁, cell cost and STE (main grid only)."""
    main = df[df["study"] != "dsweep"] if "study" in df else df
    rows = []
    for st, g in main.groupby("student"):
        c0 = g[g["cond"] == C0][outcome].dropna().to_numpy(dtype=float)
        c1 = g[g["cond"] == C1][outcome].dropna().to_numpy(dtype=float)
        syn = g[g["arm"] == "synthetic"]
        for t, cell in syn.groupby("t_params_nominal"):
            m = cell[outcome].dropna().to_numpy(dtype=float)
            if len(m) == 0:
                continue
            d_syn = float(cell["d_syn"].iloc[0]) if "d_syn" in cell else float("nan")
            c_unit = float(cell[cost_col].iloc[0]) if cost_col in cell else float("nan")
            cost_cell = c_unit * d_syn
            g1 = float(m.mean() - c1.mean()) if len(c1) else float("nan")
            rows.append({
                "student": st, "s_params_nominal": float(g["s_params_nominal"].iloc[0]), "t_params_nominal": float(t),
                "teacher": cell["teacher"].iloc[0] if "teacher" in cell else f"t{t / 1e9:g}b",
                "r": len(m), "M_mean": float(m.mean()), "M_seeds": m.tolist(), "C0_mean": float(c0.mean()) if len(c0) else np.nan,
                "C1_mean": float(c1.mean()) if len(c1) else np.nan, "C1_seeds": c1.tolist(), "C1_r": len(c1),
                "G0": float(m.mean() - c0.mean()) if len(c0) else np.nan, "G1": g1, "d_syn": d_syn,
                "c_unit": c_unit, "cost_cell": cost_cell, "STE": ste_primary(g1, cost_cell),
            })
    return pd.DataFrame(rows).sort_values(["s_params_nominal", "t_params_nominal"]).reset_index(drop=True)


def ste_primary(gain_points: float, cost_cell: float) -> float:
    """STE = G₁ / C_gen  [points per $ (or per GPU-s / FLOP for the robustness variants)]."""
    return gain_points / cost_cell if cost_cell and not math.isnan(cost_cell) else float("nan")


def marginal_efficiency(gains: Mapping[Any, float], costs: Mapping[Any, float]) -> list[dict[str, Any]]:
    """mᵢ = [G(Tᵢ₊₁) − G(Tᵢ)] / [C(Tᵢ₊₁) − C(Tᵢ)] for teachers ordered by cost."""
    order = sorted(gains, key=lambda t: costs[t])
    out = []
    for a, b in zip(order[:-1], order[1:]):
        dc = costs[b] - costs[a]
        out.append({"from": a, "to": b, "dG": gains[b] - gains[a], "dC": dc,
                    "m": (gains[b] - gains[a]) / dc if dc > 0 else float("nan")})
    return out


def t_dollar(gains: Mapping[Any, float], costs: Mapping[Any, float], lam: float) -> Any:
    """argmax_T [G(T) − λ·C(T)]; ties go to the cheaper teacher."""
    return max(gains, key=lambda t: (gains[t] - lam * costs[t], -costs[t]))


def t_dollar_curve(gains: Mapping[Any, float], costs: Mapping[Any, float],
                   lam_grid: Sequence[float] = LAMBDA_GRID) -> pd.DataFrame:
    return pd.DataFrame([{"lambda": lam, "T": t_dollar(gains, costs, lam),
                          "net": max(gains[t] - lam * costs[t] for t in gains)} for lam in lam_grid])


def t_epsilon(gains: Mapping[Any, float], costs: Mapping[Any, float], eps: float = 0.10) -> Any:
    """Cheapest T with G(T) ≥ (1−ε)·max_T G (undefined → the raw optimum if max gain ≤ 0)."""
    best = max(gains.values())
    if best <= 0:
        return t_dollar(gains, costs, 0.0)
    ok = [t for t in gains if gains[t] >= (1 - eps) * best]
    return min(ok, key=lambda t: costs[t])


def lambda1(gains: Mapping[Any, float], costs: Mapping[Any, float]) -> float:
    """Marginal efficiency of the last upgrade into T* (∞ if T* is the cheapest teacher)."""
    tstar = t_dollar(gains, costs, 0.0)
    cheaper = [t for t in gains if costs[t] < costs[tstar]]
    if not cheaper:
        return float("inf")
    prev = max(cheaper, key=lambda t: costs[t])
    dc = costs[tstar] - costs[prev]
    return (gains[tstar] - gains[prev]) / dc if dc > 0 else float("inf")


def lambda_ref(gains: Mapping[Any, float], costs: Mapping[Any, float]) -> float:
    """Average efficiency of the cheapest teacher, G₁(T₁)/C(T₁)."""
    t1 = min(gains, key=lambda t: costs[t])
    return gains[t1] / costs[t1] if costs[t1] > 0 else float("nan")


def h5_rule(lambda1_ci_hi: float, lam_ref: float) -> bool:
    """H5 supported iff the upper 95 % bound of λ₁ lies below λ_ref."""
    return bool(np.isfinite(lambda1_ci_hi) and np.isfinite(lam_ref) and lambda1_ci_hi < lam_ref)


# ----------------------------------------------------------------------------------------
# Bootstrap propagating seed and cost uncertainty
# ----------------------------------------------------------------------------------------
def bootstrap_ste(
    cell_seeds: Mapping[Any, Sequence[float]],
    c1_seeds: Sequence[float],
    cost_chunks: Mapping[Any, Sequence[tuple[float, float]]],
    *,
    d_syn: float,
    price_per_gpu_h: float = PRICE_PER_GPU_H_REFERENCE,
    sigma_pooled: Optional[float] = None,
    B: int = 5000,
    seed: int = 0,
    lam_grid: Sequence[float] = LAMBDA_GRID,
    eps: float = 0.10,
    seed_method: str = "parametric",
    stable_threshold: float = 0.8,
) -> dict[str, Any]:
    """Percentile bootstrap of STE, mᵢ, λ₁, T$(λ) selection probabilities and T$_ε.

    ``cost_chunks[T]`` is a list of (gpu_seconds, tokens_delivered) per chunk (shard); the
    resampled unit cost is Σ gpu_s / Σ tokens.  ``seed_method='parametric'`` draws cell means
    from N(mean, σ²_pooled / K) (σ pooled over all conditions of S — homoscedastic within S);
    ``'nonparametric'`` resamples seeds with replacement.  Price is a scale factor and is not
    bootstrapped.
    """
    rng = np.random.default_rng(seed)
    Ts = list(cell_seeds)
    means = {t: float(np.mean(cell_seeds[t])) for t in Ts}
    ks = {t: len(cell_seeds[t]) for t in Ts}
    c1 = np.asarray(c1_seeds, dtype=float)
    if sigma_pooled is None:
        rss, dfree = 0.0, 0
        for v in list(cell_seeds.values()) + [c1]:
            v = np.asarray(v, dtype=float)
            if len(v) > 1:
                rss += float(((v - v.mean()) ** 2).sum()); dfree += len(v) - 1
        sigma_pooled = math.sqrt(rss / dfree) if dfree > 0 else 0.0
    ste = {t: np.empty(B) for t in Ts}
    lam1 = np.empty(B)
    marg = {t: np.empty(B) for t in Ts}
    sel = {lam: {t: 0 for t in Ts} for lam in lam_grid}
    eps_sel = {t: 0 for t in Ts}
    for i in range(B):
        if seed_method == "parametric":
            g = {t: rng.normal(means[t], sigma_pooled / math.sqrt(ks[t])) for t in Ts}
            gc1 = rng.normal(c1.mean(), sigma_pooled / math.sqrt(len(c1)))
        else:
            g = {t: float(np.mean(rng.choice(cell_seeds[t], size=ks[t]))) for t in Ts}
            gc1 = float(np.mean(rng.choice(c1, size=len(c1))))
        gains = {t: g[t] - gc1 for t in Ts}
        costs = {}
        for t in Ts:
            ch = cost_chunks[t]
            idx = rng.integers(0, len(ch), size=len(ch))
            gpu_s = sum(ch[j][0] for j in idx); tok = sum(ch[j][1] for j in idx)
            costs[t] = (gpu_s / tok) * price_per_gpu_h / 3600.0 * d_syn
        for t in Ts:
            ste[t][i] = gains[t] / costs[t]
        lam1[i] = lambda1(gains, costs)
        for m in marginal_efficiency(gains, costs):
            marg[m["to"]][i] = m["m"]
        for lam in lam_grid:
            sel[lam][t_dollar(gains, costs, lam)] += 1
        eps_sel[t_epsilon(gains, costs, eps)] += 1
    q = lambda a: tuple(float(v) for v in np.nanpercentile(a[np.isfinite(a)], [2.5, 97.5])) if np.isfinite(a).any() else (np.nan, np.nan)
    sel_prob = pd.DataFrame([{"lambda": lam, **{str(t): sel[lam][t] / B for t in Ts},
                              "T_dollar": max(Ts, key=lambda t: sel[lam][t]),
                              "stable": max(sel[lam].values()) / B >= stable_threshold} for lam in lam_grid])
    return {
        "STE_ci": {t: q(ste[t]) for t in Ts}, "STE_mean": {t: float(ste[t].mean()) for t in Ts},
        "lambda1_ci": q(lam1), "lambda1_median": float(np.nanmedian(lam1[np.isfinite(lam1)])) if np.isfinite(lam1).any() else float("inf"),
        "marginal_ci": {t: q(marg[t]) for t in Ts if np.isfinite(marg[t]).any()},
        "selection": sel_prob, "t_epsilon_prob": {t: eps_sel[t] / B for t in Ts},
        "sigma_pooled": sigma_pooled, "B": B,
    }


def ste_analysis(df: pd.DataFrame, outcome: str = "p_cc", *, cost_chunks: Optional[Mapping[Any, Sequence[tuple[float, float]]]] = None,
                 price_per_gpu_h: float = PRICE_PER_GPU_H_REFERENCE, B: int = 2000, seed: int = 0,
                 lam_grid: Sequence[float] = LAMBDA_GRID, eps: float = 0.10) -> dict[str, Any]:
    """Per-student STE summary: point estimates, T*, T$(λ) curve, T$_ε, λ₁, λ_ref, H5 with CIs."""
    gt = gains_table(df, outcome)
    summary_rows, curves, boots = [], {}, {}
    for st, g in gt.groupby("student", sort=False):
        gains = dict(zip(g["t_params_nominal"], g["G1"]))
        costs = dict(zip(g["t_params_nominal"], g["cost_cell"]))
        if any(math.isnan(v) for v in list(gains.values()) + list(costs.values())):
            continue
        tstar = t_dollar(gains, costs, 0.0)
        lam1 = lambda1(gains, costs)
        lref = lambda_ref(gains, costs)
        curve = t_dollar_curve(gains, costs, lam_grid)
        curves[st] = curve
        d_syn = float(g["d_syn"].iloc[0])
        chunks = cost_chunks or {t: [(c / d_syn * 3600.0 / price_per_gpu_h, 1.0)] for t, c in costs.items()}
        sigma2, _ = within_s_variance(df[df["student"] == st], st, outcome)
        boot = bootstrap_ste({t: s for t, s in zip(g["t_params_nominal"], g["M_seeds"])}, g["C1_seeds"].iloc[0],
                             chunks, d_syn=d_syn, price_per_gpu_h=price_per_gpu_h,
                             sigma_pooled=math.sqrt(sigma2) if not math.isnan(sigma2) else None, B=B, seed=seed,
                             lam_grid=lam_grid, eps=eps)
        boots[st] = boot
        summary_rows.append({
            "student": st, "T_star": tstar, "T_epsilon": t_epsilon(gains, costs, eps), "lambda1": lam1,
            "lambda1_ci_lo": boot["lambda1_ci"][0], "lambda1_ci_hi": boot["lambda1_ci"][1], "lambda_ref": lref,
            "H5_supported": h5_rule(boot["lambda1_ci"][1], lref),
            "STE_best": max(gains, key=lambda t: ste_primary(gains[t], costs[t])),
            **{f"STE_{t / 1e9:g}B": ste_primary(gains[t], costs[t]) for t in gains},
        })
    return {"gains": gt, "summary": pd.DataFrame(summary_rows), "curves": curves, "bootstrap": boots}


# ----------------------------------------------------------------------------------------
# Iso-cost interpolation from the D-sweep
# ----------------------------------------------------------------------------------------
def fit_dose_response(df: pd.DataFrame, outcome: str = "p_cc") -> pd.DataFrame:
    """Per (S,T): P = a + b·log2 D fitted on all rows (main + D-sweep) of that cell; falls back to
    a per-S shared slope when a cell has a single D level.  Returns the coefficient table."""
    syn = df[df["arm"] == "synthetic"].dropna(subset=[outcome])
    rows = []
    for st, gs in syn.groupby("student"):
        # shared slope per S from all cells with >= 2 levels (within-cell centred regression)
        num = den = 0.0
        for t, gc in gs.groupby("t_params_nominal"):
            ld = np.log2(gc["d_syn"].astype(float)); y = gc[outcome].astype(float)
            if ld.nunique() >= 2:
                num += float(((ld - ld.mean()) * (y - y.mean())).sum()); den += float(((ld - ld.mean()) ** 2).sum())
        b_shared = num / den if den > 0 else float("nan")
        for t, gc in gs.groupby("t_params_nominal"):
            ld = np.log2(gc["d_syn"].astype(float)).to_numpy(); y = gc[outcome].astype(float).to_numpy()
            if len(np.unique(ld)) >= 2:
                b = float(np.polyfit(ld, y, 1)[0]); shared = False
            else:
                b = b_shared; shared = True
            a = float(np.mean(y - b * ld)) if not math.isnan(b) else float("nan")
            rows.append({"student": st, "t_params_nominal": float(t), "a": a, "b": b, "shared_slope": shared,
                         "log2D_min": float(ld.min()), "log2D_max": float(ld.max()), "n": len(y)})
    return pd.DataFrame(rows)


def iso_cost(df: pd.DataFrame, budgets_usd: Sequence[float], unit_cost_usd: Mapping[float, float],
             outcome: str = "p_cc", *, allow_extrapolation: bool = True) -> pd.DataFrame:
    """Evaluate P̂(S, T, D(T) = B / c(T)) for each budget B; mark the best teacher per (S, B)."""
    coef = fit_dose_response(df, outcome)
    rows = []
    for (st, t), g in coef.groupby(["student", "t_params_nominal"]):
        c = None
        for k, v in unit_cost_usd.items():
            if abs(math.log(float(k) / float(t))) < 0.05:
                c = v
        if c is None or math.isnan(g["b"].iloc[0]):
            continue
        for Bud in budgets_usd:
            D = Bud / c
            ld = math.log2(D)
            extrap = not (g["log2D_min"].iloc[0] - 1e-9 <= ld <= g["log2D_max"].iloc[0] + 1e-9)
            if extrap and not allow_extrapolation:
                continue
            rows.append({"student": st, "t_params_nominal": float(t), "budget_usd": Bud, "D": D,
                         "P_hat": float(g["a"].iloc[0] + g["b"].iloc[0] * ld), "extrapolated": extrap,
                         "shared_slope": bool(g["shared_slope"].iloc[0])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["best"] = False
    for (st, Bud), g in out.groupby(["student", "budget_usd"]):
        out.loc[g["P_hat"].idxmax(), "best"] = True
    return out.sort_values(["student", "budget_usd", "t_params_nominal"]).reset_index(drop=True)
