"""Regression methodology (statistics.md §4, §6): model sequence, tests, variance, contrasts.

Run-level observations y_i (P_cc or any outcome column) with per-student fixed intercepts
α_S; x = log2(T / 1B), s = log2(S / 25M), d = log2(D / D_syn).

| model  | mean structure                                   |
|--------|--------------------------------------------------|
| M0     | α_S                                              |
| M1     | α_S + β₁x                                        |
| M2     | α_S + β₁x + β₂x²                                 |
| M3     | α_S + β₁x + β₂x² + γ₁ s·x + γ₂ s·x²              |
| M3cat  | α_S + β_{1,S} x + β_{2,S} x²                     |
| M4     | M3 + δ₁d + δ₂ d·x + δ₃ d·s (+ δ₄ d²)             |
| M2sat  | α_S + A_S (1 − T^{−β})   (nonlinear, profile β)  |

OLS with HC3 sandwich SEs for coefficient tests; nested F / LR tests; AICc (n ≤ 150); per-S
residual variance from M3cat (df = n_S − 3) for per-S tests; Brown–Forsythe across S; isotonic
regression per S with permutation / parametric-bootstrap tests; the saturation non-inferiority
test (δ = 1.0, sensitivity 0.5); capability-covariate variants (M1z–M3z); Holm within the
contrast families F-A…F-F; Hedges' g.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar

from synscale.analysis.aggregate import C0, C1, PRIMARY_TEACHER_PARAMS, S_REF, T_REF

MODELS = ("M0", "M1", "M2", "M3", "M3cat", "M4", "M2sat")


# ----------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------
def _is_primary_teacher(t: float) -> bool:
    return any(abs(math.log(t / p)) < 0.05 for p in PRIMARY_TEACHER_PARAMS)


def prepare(df: pd.DataFrame, outcome: str = "p_cc", *, teachers: str = "primary",
            include_dsweep: bool = False, d_ref: Optional[float] = None) -> pd.DataFrame:
    """Synthetic-arm rows with ``x``, ``s``, ``d``, ``y`` columns (NaN outcomes dropped).

    ``teachers='primary'`` keeps the 6-rung grid (1.5B–72B); ``'extended'`` adds the 0.5B rung.
    """
    sub = df[df["arm"] == "synthetic"].copy()
    if teachers == "primary":
        sub = sub[sub["t_params_nominal"].apply(_is_primary_teacher)]
    if not include_dsweep and "study" in sub:
        sub = sub[sub["study"] != "dsweep"]
    sub["x"] = np.log2(sub["t_params_nominal"].astype(float) / T_REF)
    sub["s"] = np.log2(sub["s_params_nominal"].astype(float) / S_REF)
    if d_ref is None:
        main = sub[sub.get("study", "main") != "dsweep"] if "study" in sub else sub
        d_ref = float(main["d_syn"].mode().iloc[0]) if len(main) and "d_syn" in main else float("nan")
    sub["d"] = np.log2(sub["d_syn"].astype(float) / d_ref) if "d_syn" in sub else 0.0
    sub["y"] = sub[outcome].astype(float)
    return sub.dropna(subset=["y", "x"]).reset_index(drop=True)


def student_levels(sub: pd.DataFrame) -> list[str]:
    return list(sub.sort_values("s_params_nominal")["student"].unique())


def design(sub: pd.DataFrame, model: str, *, x_col: str = "x", d_quadratic: Optional[bool] = None
           ) -> tuple[np.ndarray, list[str]]:
    """Design matrix and column names for one model (per-student fixed effects, no global intercept)."""
    levels = student_levels(sub)
    fe = np.stack([(sub["student"] == lv).to_numpy(dtype=float) for lv in levels], axis=1)
    names = [f"alpha[{lv}]" for lv in levels]
    x = sub[x_col].to_numpy(dtype=float)
    s = sub["s"].to_numpy(dtype=float)
    cols = [fe]
    if model == "M0":
        pass
    elif model == "M1":
        cols += [x[:, None]]; names += ["beta1"]
    elif model == "M2":
        cols += [x[:, None], (x ** 2)[:, None]]; names += ["beta1", "beta2"]
    elif model == "M3":
        cols += [x[:, None], (x ** 2)[:, None], (s * x)[:, None], (s * x ** 2)[:, None]]
        names += ["beta1", "beta2", "gamma1", "gamma2"]
    elif model == "M3cat":
        cols += [fe * x[:, None], fe * (x ** 2)[:, None]]
        names += [f"beta1[{lv}]" for lv in levels] + [f"beta2[{lv}]" for lv in levels]
    elif model == "M4":
        d = sub["d"].to_numpy(dtype=float)
        cols += [x[:, None], (x ** 2)[:, None], (s * x)[:, None], (s * x ** 2)[:, None],
                 d[:, None], (d * x)[:, None], (d * s)[:, None]]
        names += ["beta1", "beta2", "gamma1", "gamma2", "delta1", "delta2", "delta3"]
        if d_quadratic is None:
            d_quadratic = bool(np.ptp(d) >= 3.0)  # D range >= 8x
        if d_quadratic:
            cols += [(d ** 2)[:, None]]; names += ["delta4"]
    else:
        raise ValueError(model)
    return np.hstack(cols), names


# ----------------------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------------------
@dataclass
class FitResult:
    model: str
    params: pd.Series
    se_hc3: pd.Series
    se_ols: pd.Series
    ci_lo: pd.Series
    ci_hi: pd.Series
    rss: float
    n: int
    k: int  # number of mean parameters
    df_resid: int
    aicc: float
    loglik: float
    fitted: np.ndarray
    resid: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def sigma2(self) -> float:
        return self.rss / self.df_resid if self.df_resid > 0 else float("nan")


def aicc(rss: float, n: int, k_mean: int) -> float:
    """Gaussian AICc with k = k_mean + 1 (σ² counted), n ln(RSS/n) + 2k + 2k(k+1)/(n−k−1)."""
    k = k_mean + 1
    if n - k - 1 <= 0 or rss <= 0:
        return float("inf")
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def gaussian_loglik(rss: float, n: int) -> float:
    return -0.5 * n * (math.log(2 * math.pi * rss / n) + 1) if rss > 0 else float("inf")


def fit_ols(sub: pd.DataFrame, model: str, *, x_col: str = "x", **kw: Any) -> FitResult:
    X, names = design(sub, model, x_col=x_col, **kw)
    y = sub["y"].to_numpy(dtype=float)
    ols = sm.OLS(y, X).fit()
    n, k = X.shape
    dfr = n - k
    try:
        hc3 = sm.OLS(y, X).fit(cov_type="HC3")
        se_hc3 = np.asarray(hc3.bse)
    except Exception:  # singular leverage etc.
        se_hc3 = np.full(k, np.nan)
    se_ols = np.asarray(ols.bse)
    tcrit = stats.t.ppf(0.975, dfr) if dfr > 0 else float("nan")
    params = np.asarray(ols.params)
    rss = float(ols.ssr)
    return FitResult(
        model=model, params=pd.Series(params, index=names), se_hc3=pd.Series(se_hc3, index=names),
        se_ols=pd.Series(se_ols, index=names), ci_lo=pd.Series(params - tcrit * se_hc3, index=names),
        ci_hi=pd.Series(params + tcrit * se_hc3, index=names), rss=rss, n=n, k=k, df_resid=dfr,
        aicc=aicc(rss, n, k), loglik=gaussian_loglik(rss, n), fitted=np.asarray(ols.fittedvalues),
        resid=np.asarray(ols.resid), extra={"sm": ols},
    )


def _sat_design(sub: pd.DataFrame, beta: float) -> tuple[np.ndarray, list[str]]:
    levels = student_levels(sub)
    fe = np.stack([(sub["student"] == lv).to_numpy(dtype=float) for lv in levels], axis=1)
    t = 2.0 ** sub["x"].to_numpy(dtype=float)  # T in units of 1B
    g = (1.0 - t ** (-beta))[:, None]
    return np.hstack([fe, fe * g]), [f"alpha[{lv}]" for lv in levels] + [f"A[{lv}]" for lv in levels]


def fit_m2sat(sub: pd.DataFrame, beta_grid: Sequence[float] = tuple(np.geomspace(0.02, 8.0, 60))) -> FitResult:
    """Nonlinear LS for M2sat by profiling β (linear in α_S, A_S for fixed β) from many starts."""
    y = sub["y"].to_numpy(dtype=float)

    def rss_at(beta: float) -> float:
        X, _ = _sat_design(sub, beta)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ coef) ** 2).sum())

    grid = sorted(beta_grid)
    vals = [rss_at(b) for b in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best_beta, best_rss = grid[i], vals[i]
    if hi > lo:
        res = minimize_scalar(rss_at, bounds=(lo, hi), method="bounded")
        if res.fun < best_rss:
            best_beta, best_rss = float(res.x), float(res.fun)
    X, names = _sat_design(sub, best_beta)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    n, k = X.shape
    k_mean = k + 1  # + beta
    dfr = n - k_mean
    params = pd.Series(np.append(coef, best_beta), index=names + ["beta"])
    nan = pd.Series(np.nan, index=params.index)
    return FitResult("M2sat", params, nan, nan, nan, nan, best_rss, n, k_mean, dfr, aicc(best_rss, n, k_mean),
                     gaussian_loglik(best_rss, n), fitted, y - fitted, {"beta_profile": list(zip(grid, vals))})


# ----------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------
def nested_f_test(full: FitResult, reduced: FitResult) -> dict[str, float]:
    df1 = reduced.df_resid - full.df_resid
    df2 = full.df_resid
    if df1 <= 0 or df2 <= 0:
        return {"F": float("nan"), "df1": df1, "df2": df2, "p": float("nan")}
    F = ((reduced.rss - full.rss) / df1) / (full.rss / df2)
    return {"F": float(F), "df1": int(df1), "df2": int(df2), "p": float(stats.f.sf(F, df1, df2))}


def lr_test(full: FitResult, reduced: FitResult) -> dict[str, float]:
    df = full.k - reduced.k
    lr = full.n * math.log(reduced.rss / full.rss) if full.rss > 0 else float("inf")
    return {"LR": float(lr), "df": int(df), "p": float(stats.chi2.sf(lr, df)) if df > 0 else float("nan")}


def one_sided_t(fit: FitResult, name: str, *, alternative: str = "less", robust: bool = True) -> dict[str, float]:
    est = float(fit.params[name])
    se = float(fit.se_hc3[name] if robust else fit.se_ols[name])
    if not se or math.isnan(se):
        return {"estimate": est, "se": se, "t": float("nan"), "p": float("nan")}
    t = est / se
    p = stats.t.cdf(t, fit.df_resid) if alternative == "less" else stats.t.sf(t, fit.df_resid)
    return {"estimate": est, "se": se, "t": float(t), "p": float(p), "df": fit.df_resid}


def model_sequence(sub: pd.DataFrame, *, dsweep: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Fit M0–M3cat (+ M4 if D-sweep rows are given, + M2sat) and run the pre-registered tests."""
    fits = {m: fit_ols(sub, m) for m in ("M0", "M1", "M2", "M3", "M3cat")}
    fits["M2sat"] = fit_m2sat(sub)
    tests: dict[str, Any] = {
        "H1_M1_vs_M0": nested_f_test(fits["M1"], fits["M0"]),
        "H2_beta2_lt_0": one_sided_t(fits["M2"], "beta2", alternative="less"),
        "H2_beta2_lt_0_ols_se": one_sided_t(fits["M2"], "beta2", alternative="less", robust=False),
        "M2_vs_M1": nested_f_test(fits["M2"], fits["M1"]),
        "H4pre_M3_vs_M2": nested_f_test(fits["M3"], fits["M2"]),
        "M3cat_vs_M2": nested_f_test(fits["M3cat"], fits["M2"]),
        "M3cat_vs_M3": nested_f_test(fits["M3cat"], fits["M3"]),
        "LR_M1_vs_M0": lr_test(fits["M1"], fits["M0"]),
        "AICc_M2sat_minus_M3cat": fits["M2sat"].aicc - fits["M3cat"].aicc,
    }
    if dsweep is not None and len(dsweep):
        both = pd.concat([sub, dsweep], ignore_index=True)
        fits["M4"] = fit_ols(both, "M4")
        fits["M3_with_dsweep"] = fit_ols(both, "M3")
        tests["M4_delta2_ne_0"] = {
            "estimate": float(fits["M4"].params["delta2"]), "se": float(fits["M4"].se_hc3["delta2"]),
            "p": float(2 * stats.t.sf(abs(fits["M4"].params["delta2"] / fits["M4"].se_hc3["delta2"]),
                                      fits["M4"].df_resid)) if fits["M4"].se_hc3["delta2"] else float("nan"),
        }
    aic_table = pd.DataFrame([{"model": m, "k": f.k, "n": f.n, "rss": f.rss, "aicc": f.aicc, "loglik": f.loglik}
                              for m, f in fits.items()])
    return {"fits": fits, "tests": tests, "aicc": aic_table}


def coefficient_table(fits: Mapping[str, FitResult]) -> pd.DataFrame:
    rows = []
    for m, f in fits.items():
        for name in f.params.index:
            rows.append({"model": m, "term": name, "estimate": f.params[name], "se_hc3": f.se_hc3.get(name, np.nan),
                         "se_ols": f.se_ols.get(name, np.nan), "ci_lo": f.ci_lo.get(name, np.nan),
                         "ci_hi": f.ci_hi.get(name, np.nan)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Variance structure
# ----------------------------------------------------------------------------------------
def per_s_residual_variance(sub: pd.DataFrame, fit: Optional[FitResult] = None) -> pd.DataFrame:
    """σ̂²_S from the M3cat residuals with df = n_S − 3 (pre-registered per-S variance)."""
    fit = fit or fit_ols(sub, "M3cat")
    rows = []
    for lv in student_levels(sub):
        mask = (sub["student"] == lv).to_numpy()
        r = fit.resid[mask]
        dfree = int(mask.sum()) - 3
        rows.append({"student": lv, "n": int(mask.sum()), "df": dfree,
                     "sigma2": float((r ** 2).sum() / dfree) if dfree > 0 else np.nan,
                     "sigma": float(math.sqrt((r ** 2).sum() / dfree)) if dfree > 0 else np.nan})
    return pd.DataFrame(rows)


def brown_forsythe(sub: pd.DataFrame, fit: Optional[FitResult] = None) -> dict[str, float]:
    """Brown–Forsythe (median-centred Levene) test of equal residual variance across S."""
    fit = fit or fit_ols(sub, "M3cat")
    groups = [fit.resid[(sub["student"] == lv).to_numpy()] for lv in student_levels(sub)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return {"stat": float("nan"), "p": float("nan")}
    stat, p = stats.levene(*groups, center="median")
    return {"stat": float(stat), "p": float(p)}


def within_s_variance(df: pd.DataFrame, student: str, outcome: str = "p_cc") -> tuple[float, int]:
    """Pooled within-cell variance over all conditions at one S (df = n_S − n_cells)."""
    g = df[(df["student"] == student)]
    rss, dfree = 0.0, 0
    for _, cell in g.groupby("cond"):
        v = cell[outcome].dropna().to_numpy(dtype=float)
        if len(v) > 1:
            rss += float(((v - v.mean()) ** 2).sum())
            dfree += len(v) - 1
    return (rss / dfree if dfree > 0 else float("nan")), dfree


def cooks_distance_flags(fit: FitResult, threshold_factor: float = 4.0) -> np.ndarray:
    """Boolean mask of runs with Cook's D > 4/n (reported; the analysis is repeated without them)."""
    ols = fit.extra.get("sm")
    if ols is None:
        return np.zeros(fit.n, dtype=bool)
    infl = ols.get_influence()
    d = infl.cooks_distance[0]
    return np.asarray(d > threshold_factor / fit.n)


# ----------------------------------------------------------------------------------------
# Isotonic regression and saturation
# ----------------------------------------------------------------------------------------
def pava(y: Sequence[float], w: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pool-adjacent-violators: weighted non-decreasing fit."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    blocks: list[list[float]] = []  # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.asarray(out)


def _rss_iso(x_cells: np.ndarray, ys: list[np.ndarray]) -> tuple[float, float, float]:
    means = np.array([y.mean() for y in ys])
    counts = np.array([len(y) for y in ys], dtype=float)
    grand = np.concatenate(ys).mean()
    iso = pava(means, counts)
    rss_const = float(sum(((y - grand) ** 2).sum() for y in ys))
    rss_iso = float(sum(((y - m) ** 2).sum() for y, m in zip(ys, iso)))
    rss_cells = float(sum(((y - y.mean()) ** 2).sum() for y in ys))
    return rss_const, rss_iso, rss_cells


def isotonic_tests(df: pd.DataFrame, student: str, outcome: str = "p_cc", *, n_perm: int = 2000,
                   n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Per-S isotonic (non-decreasing in T) analysis (statistics.md §4.3.1).

    Returns the permutation p-value for "some monotone T effect" (RSS(const) − RSS(iso)), the
    parametric-bootstrap p-value for "non-monotone" (RSS(iso) − RSS(cell means)), and Spearman's
    ρ between cell means and log T.
    """
    sub = df[(df["arm"] == "synthetic") & (df["student"] == student)].dropna(subset=[outcome])
    if "study" in sub:
        sub = sub[sub["study"] != "dsweep"]
    xs = sorted(sub["t_params_nominal"].unique())
    ys = [sub[sub["t_params_nominal"] == t][outcome].to_numpy(dtype=float) for t in xs]
    rng = np.random.default_rng(seed)
    rss_const, rss_iso, rss_cells = _rss_iso(np.log2(xs), ys)
    stat_mono, stat_nonmono = rss_const - rss_iso, rss_iso - rss_cells
    pooled = np.concatenate(ys)
    sizes = [len(y) for y in ys]
    # (i) permutation of runs among cells
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        parts, k = [], 0
        for n_ in sizes:
            parts.append(perm[k:k + n_]); k += n_
        c, i, _ = _rss_iso(np.log2(xs), parts)
        cnt += (c - i) >= stat_mono - 1e-12
    p_mono = (cnt + 1) / (n_perm + 1)
    # (ii) parametric bootstrap from the isotonic fit with σ̂_S (within-cell)
    means = np.array([y.mean() for y in ys]); counts = np.array(sizes, dtype=float)
    iso = pava(means, counts)
    dfree = sum(n_ - 1 for n_ in sizes)
    sigma = math.sqrt(rss_cells / dfree) if dfree > 0 else 0.0
    cnt = 0
    for _ in range(n_boot):
        parts = [m + sigma * rng.standard_normal(n_) for m, n_ in zip(iso, sizes)]
        _, i, c = _rss_iso(np.log2(xs), parts)
        cnt += (i - c) >= stat_nonmono - 1e-12
    p_nonmono = (cnt + 1) / (n_boot + 1)
    rho = stats.spearmanr(np.log2(xs), means).correlation if len(xs) > 2 else float("nan")
    return {"student": student, "stat_monotone": stat_mono, "p_monotone_perm": float(p_mono),
            "stat_nonmonotone": stat_nonmono, "p_nonmonotone_boot": float(p_nonmono),
            "spearman_rho": float(rho), "sigma_within": sigma, "n": int(len(pooled))}


def saturation_test(df: pd.DataFrame, student: str, outcome: str = "p_cc", *, t_hi: float = 72e9,
                    t_lo: float = 14e9, delta: float = 1.0, alpha: float = 0.05) -> dict[str, float]:
    """One-sided non-inferiority test H0: μ(t_hi) − μ(t_lo) ≥ δ vs H1: < δ (statistics.md §4.3.2).

    Uses the within-S pooled variance over all cells at S; reports the (1−2α) two-sided CI so
    that the upper bound is the one-sided 95 % bound.
    """
    sub = df[(df["student"] == student)]
    if "study" in sub:
        sub = sub[sub["study"] != "dsweep"]
    hi = sub[np.isclose(sub["t_params_nominal"], t_hi)][outcome].dropna().to_numpy(dtype=float)
    lo = sub[np.isclose(sub["t_params_nominal"], t_lo)][outcome].dropna().to_numpy(dtype=float)
    s2, dfree = within_s_variance(sub, student, outcome)
    if len(hi) == 0 or len(lo) == 0 or dfree <= 0 or math.isnan(s2):
        return {"student": student, "diff": float("nan"), "se": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "saturated": False, "delta": delta, "df": dfree}
    diff = float(hi.mean() - lo.mean())
    se = math.sqrt(s2 * (1 / len(hi) + 1 / len(lo)))
    t = (diff - delta) / se
    p = float(stats.t.cdf(t, dfree))
    tcrit = stats.t.ppf(1 - alpha, dfree)
    return {"student": student, "diff": diff, "se": se, "t": float(t), "p": p, "df": dfree,
            "ci_lo": diff - tcrit * se, "ci_hi": diff + tcrit * se, "saturated": bool(diff + tcrit * se < delta),
            "delta": delta, "r_hi": int(len(hi)), "r_lo": int(len(lo))}


# ----------------------------------------------------------------------------------------
# Capability covariate (statistics.md §4.5)
# ----------------------------------------------------------------------------------------
def capability_models(sub: pd.DataFrame, z_by_teacher: Mapping[float, float]) -> dict[str, Any]:
    """Refit M1–M3 with the teacher capability z_T in place of log T; compare AICc."""
    zs = np.array([_lookup_z(t, z_by_teacher) for t in sub["t_params_nominal"]], dtype=float)
    sub2 = sub.copy()
    sub2["z"] = zs
    fits = {f"{m}z": fit_ols(sub2, m, x_col="z") for m in ("M1", "M2", "M3")}
    base = {m: fit_ols(sub, m) for m in ("M1", "M2", "M3")}
    rho = stats.spearmanr(sub["x"], zs).correlation if len(sub) > 2 else float("nan")
    return {"fits": fits, "aicc_diff_z_minus_logT": {m: fits[f"{m}z"].aicc - base[m].aicc for m in base},
            "spearman_x_z": float(rho)}


def _lookup_z(t: float, z_by_teacher: Mapping[float, float]) -> float:
    for k, v in z_by_teacher.items():
        if abs(math.log(float(t) / float(k))) < 0.05:
            return v
    return float("nan")


# ----------------------------------------------------------------------------------------
# Multiplicity and effect sizes
# ----------------------------------------------------------------------------------------
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries are left NaN and not counted)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out


def hedges_g(a: Sequence[float], b: Sequence[float], *, sigma2: Optional[float] = None,
             df: Optional[int] = None, level: float = 0.95) -> dict[str, float]:
    """Hedges' g = J·Δ̂/σ̂ with J = 1 − 3/(4df − 1); approximate CI via the large-sample SE."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if sigma2 is None:
        dfree = n1 + n2 - 2
        sigma2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / dfree if dfree > 0 else float("nan")
        df = dfree
    df = df if df is not None else n1 + n2 - 2
    if not sigma2 or math.isnan(sigma2) or df <= 1:
        return {"g": float("nan"), "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    J = 1 - 3 / (4 * df - 1)
    d = (a.mean() - b.mean()) / math.sqrt(sigma2)
    g = J * d
    se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(0.5 + level / 2)
    return {"g": float(g), "se": float(se), "ci_lo": float(g - z * se), "ci_hi": float(g + z * se)}


def contrast(a: Sequence[float], b: Sequence[float], sigma2: float, df: int, level: float = 0.95) -> dict[str, float]:
    """Difference of two cell means with a pooled-variance t interval (two-sided p)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0 or df <= 0 or math.isnan(sigma2):
        return {"delta": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "df": df, "n_a": len(a), "n_b": len(b)}
    delta = float(a.mean() - b.mean())
    se = math.sqrt(sigma2 * (1 / len(a) + 1 / len(b)))
    t = delta / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df)) if se > 0 else float("nan")
    h = stats.t.ppf(0.5 + level / 2, df) * se
    return {"delta": delta, "se": se, "t": float(t), "p": p, "ci_lo": delta - h, "ci_hi": delta + h,
            "df": int(df), "n_a": int(len(a)), "n_b": int(len(b))}


def contrast_families(
    df: pd.DataFrame,
    outcome: str = "p_cc",
    *,
    econ_pairs: Optional[Mapping[str, tuple[float, float]]] = None,
    confirm_pairs: Optional[Mapping[str, float]] = None,
    t_largest: float = 72e9,
) -> pd.DataFrame:
    """All contrasts of families F-A…F-F with raw and Holm-adjusted p-values and Hedges' g.

    ``econ_pairs`` maps student → (T_econ, T_raw*) for F-E; ``confirm_pairs`` maps student → T₀
    for the F-F "intermediate beats largest" confirmation on tier-2 seeds (``tier == 2``).
    """
    main = df[df.get("study", pd.Series("main", index=df.index)) != "dsweep"] if "study" in df else df
    rows: list[dict[str, Any]] = []

    def cell(student: str, cond: Optional[str] = None, t: Optional[float] = None, tier: Optional[int] = None
             ) -> np.ndarray:
        g = main[main["student"] == student]
        if cond is not None:
            g = g[g["cond"] == cond]
        if t is not None:
            g = g[(g["arm"] == "synthetic") & np.isclose(g["t_params_nominal"].astype(float), t)]
        if tier is not None and "tier" in g:
            g = g[g["tier"] == tier]
        return g[outcome].dropna().to_numpy(dtype=float)

    def add(family: str, student: str, label: str, a: np.ndarray, b: np.ndarray, s2: float, dfree: int) -> None:
        c = contrast(a, b, s2, dfree)
        g = hedges_g(a, b, sigma2=s2, df=dfree) if len(a) and len(b) else hedges_g([], [])
        rows.append({"family": family, "student": student, "contrast": label, **c, "g": g["g"],
                     "g_ci_lo": g["ci_lo"], "g_ci_hi": g["ci_hi"]})

    students = list(main.sort_values("s_params_nominal")["student"].unique())
    for st in students:
        s2, dfree = within_s_variance(main, st, outcome)
        teachers = sorted(main[(main["student"] == st) & (main["arm"] == "synthetic")]["t_params_nominal"].dropna().unique())
        teachers = [t for t in teachers if _is_primary_teacher(t)]
        for t1, t2 in zip(teachers[:-1], teachers[1:]):
            add("F-A", st, f"T{t2 / 1e9:g}B - T{t1 / 1e9:g}B", cell(st, t=t2), cell(st, t=t1), s2, dfree)
        for t in teachers:
            add("F-B", st, f"T{t / 1e9:g}B - C1", cell(st, t=t), cell(st, cond=C1), s2, dfree)
            add("F-C", st, f"T{t / 1e9:g}B - C0", cell(st, t=t), cell(st, cond=C0), s2, dfree)
        add("F-D", st, "C1 - C0", cell(st, cond=C1), cell(st, cond=C0), s2, dfree)
        if econ_pairs and st in econ_pairs:
            te, ts = econ_pairs[st]
            if not np.isclose(te, ts):
                add("F-E", st, f"T{te / 1e9:g}B(econ) - T{ts / 1e9:g}B(raw*)", cell(st, t=te), cell(st, t=ts), s2, dfree)
        if confirm_pairs and st in confirm_pairs and "tier" in main:
            t0 = confirm_pairs[st]
            a, b = cell(st, t=t0, tier=2), cell(st, t=t_largest, tier=2)
            if len(a) and len(b):
                s2b, dfb = within_s_variance(main[main["tier"] == 2], st, outcome)
                add("F-F", st, f"T{t0 / 1e9:g}B - T{t_largest / 1e9:g}B (tier-2 seeds)", a, b, s2b, dfb)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    # Holm within each S for F-A/F-B/F-C; over students for F-D/F-E/F-F
    for fam in ("F-A", "F-B", "F-C"):
        for st in students:
            m = (out["family"] == fam) & (out["student"] == st)
            if m.any():
                out.loc[m, "p_holm"] = holm(out.loc[m, "p"].to_numpy())
    for fam in ("F-D", "F-E", "F-F"):
        m = out["family"] == fam
        if m.any():
            out.loc[m, "p_holm"] = holm(out.loc[m, "p"].to_numpy())
    m = out["family"] == "F-A"
    out["p_holm_global25"] = np.nan
    if m.any():
        out.loc[m, "p_holm_global25"] = holm(out.loc[m, "p"].to_numpy())
    return out
