"""Per-student T* estimation, classification and the descriptive α fit (statistics.md §5).

Per S the r_T seed values at the six teacher sizes are fitted with

* F1: P = a + b·x + c·x²  (x = log2 T/1B); interior maximum at x* = −b/(2c) iff c < 0 and
  x_min < x* < x_max;
* F2: P = P∞ − A·2^{−βx} = P∞ − A·T^{−β}, β > 0 (monotone saturating; profile-β NLS from many
  starts).

Classification (pre-registered, evaluated in the order flat → interior → saturating →
monotone_unsaturated):

1. **flat** iff the joint F-test of (b, c) = 0 (within-S variance) is not significant;
2. **interior** iff (i) ĉ < 0 with one-sided p < 0.05, (ii) x_min < x̂* < x_max, (iii) the 95 %
   bootstrap CI of x̂* has upper bound < x_max, (iv) AICc(F1) ≤ AICc(F2) + 2;
3. **saturating** iff not interior and the one-sided 95 % upper bound of the model-based
   top-step gain Δ_top (32B → 72B) is < δ (δ = 1.0; sensitivity 0.5); T_sat = smallest grid T_j
   with UB95(Δ₂(x_j)) < δ where Δ₂(x) = b + c(2x+1) is the gain from doubling T;
4. **monotone_unsaturated** otherwise (T* right-censored at the largest teacher).

Uncertainty: cluster (seed) bootstrap when r ≥ 3 in every cell, otherwise the parametric
bootstrap of (b, c) from N((b̂, ĉ), Σ̂); B = 2000, percentile CIs on the log2 scale.  The α fit
(log2 T* = α s + c, WLS with w = 1/Var(x̂*)) uses only sizes classified interior; censored sizes
are listed, never imputed.  ``tier2_confirmation`` estimates the "T₀ beats 72B" contrast on the
tier-2 seeds only.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from synscale.analysis.aggregate import PRIMARY_TEACHER_PARAMS, S_REF, T_REF
from synscale.analysis.regression import _is_primary_teacher, aicc, contrast, within_s_variance

X_GRID = tuple(math.log2(t / T_REF) for t in PRIMARY_TEACHER_PARAMS)
DELTA_PRIMARY = 1.0
DELTA_SENSITIVITY = 0.5
LABELS = ("interior", "saturating", "monotone_unsaturated", "flat")


# ----------------------------------------------------------------------------------------
# Functional forms
# ----------------------------------------------------------------------------------------
@dataclass
class F1Fit:
    a: float
    b: float
    c: float
    cov: np.ndarray  # 3x3 covariance of (a, b, c) with within-S σ²
    sigma2: float
    df: int
    rss: float
    n: int
    aicc: float
    x_min: float
    x_max: float
    xstar: float  # vertex if interior, else boundary with the larger fitted value
    interior_vertex: bool
    p_curv: float  # one-sided p for c < 0
    F_effect: float  # joint F for (b, c) = 0
    p_effect: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.a + self.b * x + self.c * x ** 2


@dataclass
class F2Fit:
    p_inf: float
    A: float
    beta: float
    rss: float
    n: int
    aicc: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.p_inf - self.A * 2.0 ** (-self.beta * x)


def vertex(b: float, c: float, x_min: float, x_max: float) -> tuple[float, bool]:
    """Vertex of a + bx + cx² if concave and inside (x_min, x_max), else the better boundary."""
    if c < 0 and abs(c) > 1e-12:
        v = -b / (2 * c)
        if x_min < v < x_max:
            return float(v), True
    f_min, f_max = b * x_min + c * x_min ** 2, b * x_max + c * x_max ** 2
    return (x_max if f_max >= f_min else x_min), False


def fit_f1(x: Sequence[float], y: Sequence[float], *, x_min: Optional[float] = None,
           x_max: Optional[float] = None) -> F1Fit:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    X = np.stack([np.ones_like(x), x, x ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, dfree = len(y), len(y) - 3
    rss = float((resid ** 2).sum())
    sigma2 = rss / dfree if dfree > 0 else float("nan")
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov = sigma2 * xtx_inv
    xmn = float(x.min()) if x_min is None else x_min
    xmx = float(x.max()) if x_max is None else x_max
    xs, inter = vertex(coef[1], coef[2], xmn, xmx)
    se_c = math.sqrt(max(cov[2, 2], 0.0))
    p_curv = float(stats.t.cdf(coef[2] / se_c, dfree)) if se_c > 0 and dfree > 0 else float("nan")
    # joint F for (b, c) = 0
    sub = cov[1:, 1:]
    try:
        F = float(coef[1:] @ np.linalg.solve(sub, coef[1:]) / 2.0)
        p_eff = float(stats.f.sf(F, 2, dfree)) if dfree > 0 else float("nan")
    except np.linalg.LinAlgError:
        F, p_eff = float("nan"), float("nan")
    return F1Fit(float(coef[0]), float(coef[1]), float(coef[2]), cov, sigma2, dfree, rss, n, aicc(rss, n, 3),
                 xmn, xmx, xs, inter, p_curv, F, p_eff)


def fit_f2(x: Sequence[float], y: Sequence[float],
           beta_grid: Sequence[float] = tuple(np.geomspace(0.02, 8.0, 60))) -> F2Fit:
    """Profile-β nonlinear LS for P∞ − A·2^{−βx} (multi-start over the β grid + local refinement)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def solve(beta: float) -> tuple[float, np.ndarray]:
        X = np.stack([np.ones_like(x), -(2.0 ** (-beta * x))], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ coef) ** 2).sum()), coef

    grid = sorted(beta_grid)
    vals = [solve(b)[0] for b in grid]
    i = int(np.argmin(vals))
    best_beta, best_rss = grid[i], vals[i]
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(lambda b: solve(b)[0], bounds=(lo, hi), method="bounded")
        if res.fun < best_rss:
            best_beta, best_rss = float(res.x), float(res.fun)
    rss, coef = solve(best_beta)
    return F2Fit(float(coef[0]), float(coef[1]), float(best_beta), rss, len(y), aicc(rss, len(y), 3))


def delta2(b: float, c: float, x: float) -> float:
    """Gain from doubling T at x under F1: P(x+1) − P(x) = b + c(2x + 1)."""
    return b + c * (2 * x + 1)


def top_step_gain(fit: F1Fit, x5: float = X_GRID[-2], x6: float = X_GRID[-1]) -> tuple[float, float]:
    """Δ_top = b(x₆−x₅) + c(x₆²−x₅²) and its delta-method SE."""
    L = np.array([0.0, x6 - x5, x6 ** 2 - x5 ** 2])
    est = float(L @ np.array([fit.a, fit.b, fit.c]))
    se = float(math.sqrt(max(L @ fit.cov @ L, 0.0)))
    return est, se


def t_sat(fit: F1Fit, x_grid: Sequence[float] = X_GRID, delta: float = DELTA_PRIMARY) -> Optional[float]:
    """Smallest grid T_j (in params) with one-sided 95 % UB of Δ₂(x_j) < δ; None if never."""
    tcrit = stats.t.ppf(0.95, fit.df) if fit.df > 0 else float("nan")
    for xj in x_grid:
        L = np.array([0.0, 1.0, 2 * xj + 1])
        est = float(L @ np.array([fit.a, fit.b, fit.c]))
        se = math.sqrt(max(L @ fit.cov @ L, 0.0))
        if est + tcrit * se < delta:
            return float(2 ** xj * T_REF)
    return None


# ----------------------------------------------------------------------------------------
# Bootstraps
# ----------------------------------------------------------------------------------------
def cluster_bootstrap(cells: Mapping[float, np.ndarray], *, B: int = 2000, seed: int = 0,
                      x_min: float, x_max: float, x5: float, x6: float) -> dict[str, np.ndarray]:
    """Resample runs within each cell with replacement; refit F1; record x̂*, Δ_top, c, interior."""
    rng = np.random.default_rng(seed)
    xs = np.array(sorted(cells))
    ys = [np.asarray(cells[x], dtype=float) for x in xs]
    out = {k: np.empty(B) for k in ("xstar", "dtop", "c", "b")}
    out["interior"] = np.empty(B, dtype=bool)
    for i in range(B):
        yy, xx = [], []
        for x, y in zip(xs, ys):
            idx = rng.integers(0, len(y), size=len(y))
            yy.append(y[idx]); xx.append(np.full(len(y), x))
        f = fit_f1(np.concatenate(xx), np.concatenate(yy), x_min=x_min, x_max=x_max)
        out["xstar"][i], out["interior"][i] = f.xstar, f.interior_vertex
        out["dtop"][i] = top_step_gain(f, x5, x6)[0]
        out["c"][i], out["b"][i] = f.c, f.b
    return out


def parametric_bootstrap(fit: F1Fit, *, B: int = 2000, seed: int = 0, x5: float, x6: float) -> dict[str, np.ndarray]:
    """Draw (a, b, c) ~ N((â, b̂, ĉ), Σ̂) and record the same quantities as the cluster bootstrap."""
    rng = np.random.default_rng(seed)
    cov = (fit.cov + fit.cov.T) / 2
    draws = rng.multivariate_normal([fit.a, fit.b, fit.c], cov, size=B, method="svd")
    xstar = np.empty(B); interior = np.empty(B, dtype=bool)
    for i, (_, b, c) in enumerate(draws):
        xstar[i], interior[i] = vertex(b, c, fit.x_min, fit.x_max)
    dtop = draws[:, 1] * (x6 - x5) + draws[:, 2] * (x6 ** 2 - x5 ** 2)
    return {"xstar": xstar, "interior": interior, "dtop": dtop, "c": draws[:, 2], "b": draws[:, 1]}


# ----------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------
@dataclass
class TStarResult:
    student: str
    s_params: float
    label: str
    xstar: float
    tstar: float
    xstar_ci: tuple[float, float]
    tstar_ci: tuple[float, float]
    dtop: float
    dtop_se: float
    dtop_ub95: float
    dtop_ci_boot: tuple[float, float]
    t_sat: Optional[float]
    aicc_f1: float
    aicc_f2: float
    p_curv: float
    p_effect: float
    c: float
    c_se: float
    conditions: dict[str, bool]
    bootstrap: str
    censored: bool
    delta: float
    r_min: int
    n: int
    f1: F1Fit = field(repr=False)
    f2: F2Fit = field(repr=False)
    boot: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    @property
    def var_xstar(self) -> float:
        return float(np.var(self.boot["xstar"], ddof=1)) if "xstar" in self.boot else float("nan")

    def to_row(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("f1", "f2", "boot", "conditions")}
        d.update({f"cond_{k}": v for k, v in self.conditions.items()})
        d["xstar_ci_lo"], d["xstar_ci_hi"] = self.xstar_ci
        d["tstar_ci_lo"], d["tstar_ci_hi"] = self.tstar_ci
        d["dtop_boot_lo"], d["dtop_boot_hi"] = self.dtop_ci_boot
        d["var_xstar"] = self.var_xstar
        for k in ("xstar_ci", "tstar_ci", "dtop_ci_boot"):
            d.pop(k)
        return d


def classify_student(
    df: pd.DataFrame,
    student: str,
    outcome: str = "p_cc",
    *,
    delta: float = DELTA_PRIMARY,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    bootstrap: str = "auto",
    teachers: str = "primary",
) -> TStarResult:
    """Fit F1/F2 on the runs of one student and apply the four-label rule."""
    sub = df[(df["arm"] == "synthetic") & (df["student"] == student)].dropna(subset=[outcome])
    if "study" in sub:
        sub = sub[sub["study"] != "dsweep"]
    if teachers == "primary":
        sub = sub[sub["t_params_nominal"].apply(_is_primary_teacher)]
    x = np.log2(sub["t_params_nominal"].astype(float) / T_REF).to_numpy()
    y = sub[outcome].to_numpy(dtype=float)
    cells = {float(xv): y[np.isclose(x, xv)] for xv in np.unique(x)}
    xs = sorted(cells)
    x_min, x_max = xs[0], xs[-1]
    x5, x6 = xs[-2], xs[-1]
    r_min = min(len(v) for v in cells.values())
    f1 = fit_f1(x, y, x_min=x_min, x_max=x_max)
    f2 = fit_f2(x, y)
    method = bootstrap if bootstrap != "auto" else ("cluster" if r_min >= 3 else "parametric")
    if method == "cluster":
        boot = cluster_bootstrap(cells, B=B, seed=seed, x_min=x_min, x_max=x_max, x5=x5, x6=x6)
    else:
        boot = parametric_bootstrap(f1, B=B, seed=seed, x5=x5, x6=x6)
    lo, hi = np.percentile(boot["xstar"], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    dtop, dtop_se = top_step_gain(f1, x5, x6)
    tcrit = stats.t.ppf(1 - alpha, f1.df) if f1.df > 0 else float("nan")
    dtop_ub = dtop + tcrit * dtop_se
    dlo, dhi = np.percentile(boot["dtop"], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    conds = {
        "curv_neg_sig": bool(f1.c < 0 and f1.p_curv < alpha),
        "vertex_inside": bool(f1.interior_vertex),
        "ci_excludes_largest": bool(hi < x_max - 1e-9),
        "aicc_f1_le_f2_plus_2": bool(f1.aicc <= f2.aicc + 2),
        "effect_sig": bool(f1.p_effect < alpha),
        "top_step_ub_lt_delta": bool(dtop_ub < delta),
    }
    if not conds["effect_sig"]:
        label = "flat"
    elif conds["curv_neg_sig"] and conds["vertex_inside"] and conds["ci_excludes_largest"] and conds["aicc_f1_le_f2_plus_2"]:
        label = "interior"
    elif conds["top_step_ub_lt_delta"]:
        label = "saturating"
    else:
        label = "monotone_unsaturated"
    tsat = t_sat(f1, xs, delta) if label == "saturating" else None
    xstar = f1.xstar if label == "interior" else x_max
    censored = label != "interior"
    return TStarResult(
        student=student, s_params=float(sub["s_params_nominal"].iloc[0]), label=label, xstar=float(xstar),
        tstar=float(2 ** xstar * T_REF), xstar_ci=(float(lo), float(hi)),
        tstar_ci=(float(2 ** lo * T_REF), float(2 ** hi * T_REF)), dtop=dtop, dtop_se=dtop_se, dtop_ub95=dtop_ub,
        dtop_ci_boot=(float(dlo), float(dhi)), t_sat=tsat, aicc_f1=f1.aicc, aicc_f2=f2.aicc, p_curv=f1.p_curv,
        p_effect=f1.p_effect, c=f1.c, c_se=math.sqrt(max(f1.cov[2, 2], 0)), conditions=conds, bootstrap=method,
        censored=censored, delta=delta, r_min=r_min, n=len(y), f1=f1, f2=f2, boot=boot,
    )


def classify_all(df: pd.DataFrame, outcome: str = "p_cc", **kw: Any) -> tuple[pd.DataFrame, dict[str, TStarResult]]:
    results: dict[str, TStarResult] = {}
    main = df[df["arm"] == "synthetic"]
    for st in main.sort_values("s_params_nominal")["student"].unique():
        results[st] = classify_student(df, st, outcome, **kw)
    table = pd.DataFrame([r.to_row() for r in results.values()])
    return table, results


# ----------------------------------------------------------------------------------------
# α fit: log2 T* vs log2 S (descriptive)
# ----------------------------------------------------------------------------------------
def alpha_fit(results: Mapping[str, TStarResult], *, B: int = 2000, seed: int = 0, min_points: int = 3
              ) -> dict[str, Any]:
    """WLS of log2 T̂*_S on s = log2(S/25M) over sizes classified interior; t-CI and bootstrap CI."""
    used = [r for r in results.values() if r.label == "interior"]
    excluded = [{"student": r.student, "label": r.label, "censored_at": r.xstar} for r in results.values()
                if r.label != "interior"]
    out: dict[str, Any] = {"n_used": len(used), "excluded": excluded, "alpha": float("nan"), "c": float("nan"),
                           "alpha_ci_t": (float("nan"), float("nan")), "alpha_ci_boot": (float("nan"), float("nan"))}
    if len(used) < min_points:
        out["status"] = f"fewer than {min_points} interior sizes; alpha not estimated"
        return out
    s = np.array([math.log2(r.s_params / S_REF) for r in used])
    xs = np.array([r.xstar for r in used])
    w = np.array([1.0 / max(r.var_xstar, 1e-6) for r in used])
    X = np.stack([s, np.ones_like(s)], axis=1)
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ xs)
    resid = xs - X @ beta
    dfree = len(used) - 2
    s2 = float((w * resid ** 2).sum() / dfree) if dfree > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ W @ X)
    se_a = math.sqrt(max(cov[0, 0], 0))
    tcrit = stats.t.ppf(0.975, dfree) if dfree > 0 else float("nan")
    # bootstrap: propagate per-S bootstrap draws of x*
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(B):
        xb = np.array([rng.choice(r.boot["xstar"]) for r in used])
        bb = np.linalg.solve(X.T @ W @ X, X.T @ W @ xb)
        alphas.append(bb[0])
    lo, hi = np.percentile(alphas, [2.5, 97.5])
    out.update({"alpha": float(beta[0]), "c": float(beta[1]), "alpha_se": se_a, "df": dfree,
                "alpha_ci_t": (float(beta[0] - tcrit * se_a), float(beta[0] + tcrit * se_a)),
                "alpha_ci_boot": (float(lo), float(hi)), "status": "ok",
                "points": [{"student": r.student, "s": float(si), "xstar": float(xi), "w": float(wi)}
                           for r, si, xi, wi in zip(used, s, xs, w)]})
    return out


# ----------------------------------------------------------------------------------------
# Tier-2 confirmation of "intermediate beats largest"
# ----------------------------------------------------------------------------------------
def tier2_confirmation(df: pd.DataFrame, student: str, t0_params: float, *, t_ref: float = 72e9,
                       outcome: str = "p_cc", tier_col: str = "tier") -> dict[str, Any]:
    """Contrast (cell T₀ − cell T_ref) estimated on tier-2 seeds only (never used for selection)."""
    sub = df[(df["student"] == student) & (df["arm"] == "synthetic")]
    if tier_col not in sub:
        return {"student": student, "status": "no tier column", "confirmed": False}
    t2 = sub[sub[tier_col] == 2]
    a = t2[np.isclose(t2["t_params_nominal"].astype(float), t0_params)][outcome].dropna().to_numpy(dtype=float)
    b = t2[np.isclose(t2["t_params_nominal"].astype(float), t_ref)][outcome].dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"student": student, "status": "no tier-2 seeds in one of the cells", "confirmed": False}
    s2, dfree = within_s_variance(t2, student, outcome)
    if dfree <= 0:
        dfree = len(a) + len(b) - 2
        s2 = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / dfree if dfree > 0 else float("nan")
    c = contrast(a, b, s2, dfree)
    return {"student": student, "t0": t0_params, "t_ref": t_ref, **c, "status": "ok",
            "confirmed": bool(c["ci_lo"] > 0)}
