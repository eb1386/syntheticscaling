"""Figures for the write-up (doc 20). matplotlib only; each returns a Figure and saves a PNG.

Every function takes the results index (a DataFrame with per-run rows) and the aggregate
columns added by synscale.analysis.aggregate.add_aggregates. Error bars are 95% t-CIs across
seeds with r printed; C0/C1/C1b are drawn as horizontal bands (doc 20 §0). These run only
once real results exist; they are import-safe without matplotlib.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ci(vals):
    import numpy as np
    from scipy import stats
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), 0.0, len(v))
    m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
    h = se * stats.t.ppf(0.975, len(v) - 1)
    return m, h, len(v)


def transfer_curves(df, out_dir, outcome="p_cc"):
    """Per-student outcome vs log2 teacher size, with C0/C1/C1b bands (F1/F2 in doc 20)."""
    plt = _mpl()
    import numpy as np
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    syn = df[df["arm"] == "synthetic"]
    students = sorted(syn["student"].unique(), key=lambda s: syn[syn.student == s]["s_params_nominal"].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in students:
        sub = syn[syn.student == s]
        xs, ys, es = [], [], []
        for t in sorted(sub["teacher_params"].dropna().unique()):
            cell = sub[sub.teacher_params == t][outcome]
            m, h, r = _ci(cell)
            xs.append(math.log2(t / 1e9)); ys.append(m); es.append(h)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{s}")
        for arm, style in (("C1", "--"), ("C0", ":")):
            band = df[(df.student == s) & (df.cond == arm)][outcome]
            if len(band):
                ax.axhline(float(np.nanmean(band)), linestyle=style, alpha=0.3)
    ax.set_xlabel("log2 teacher size / 1B"); ax.set_ylabel(outcome)
    ax.set_title("Teacher-size transfer curves (95% CI across seeds)"); ax.legend()
    p = out_dir / f"transfer_curves_{outcome}.png"; fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def sxt_heatmap(df, out_dir, outcome="p_cc"):
    """Student × teacher gain over C1 heatmap (F in doc 20)."""
    plt = _mpl(); import numpy as np
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    syn = df[df["arm"] == "synthetic"]
    students = sorted(syn["student"].unique(), key=lambda s: syn[syn.student == s]["s_params_nominal"].iloc[0])
    teachers = sorted(syn["teacher_params"].dropna().unique())
    M = np.full((len(students), len(teachers)), np.nan)
    for i, s in enumerate(students):
        c1 = df[(df.student == s) & (df.cond == "C1")][outcome]
        base = float(np.nanmean(c1)) if len(c1) else np.nan
        for j, t in enumerate(teachers):
            cell = syn[(syn.student == s) & (syn.teacher_params == t)][outcome]
            if len(cell):
                M[i, j] = float(np.nanmean(cell)) - base
    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(teachers))); ax.set_xticklabels([f"{t/1e9:g}B" for t in teachers])
    ax.set_yticks(range(len(students))); ax.set_yticklabels(students)
    ax.set_xlabel("teacher"); ax.set_ylabel("student"); ax.set_title(f"gain over C1 ({outcome})")
    fig.colorbar(im, ax=ax)
    p = out_dir / f"sxt_heatmap_{outcome}.png"; fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def cost_per_token(cost_rows, out_dir):
    """Generation cost per 1M delivered student tokens by teacher (bars: GPU-s / $ / kWh)."""
    plt = _mpl()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = [r["teacher"] for r in cost_rows]; gpus = [r.get("gpu_s_per_mtok", 0) for r in cost_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, gpus); ax.set_ylabel("GPU-s / 1M student tokens"); ax.set_title("Generation cost by teacher")
    p = out_dir / "cost_per_token.png"; fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p
