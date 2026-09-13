"""Novelty analyses (docs/24): screening finds the true driver; predictive law + allocation run."""
import numpy as np
from synscale.analysis import novelty as nv
from synscale.analysis.data_properties import quality_index, distinct_n


def test_quality_index_signs():
    rows = [{"learnability_nll": 3.0, "distinct_2": 0.4, "correctness": 0.3},
            {"learnability_nll": 2.0, "distinct_2": 0.7, "correctness": 0.9}]
    q = quality_index(rows)
    assert q[1]["q"] > q[0]["q"]  # lower NLL + higher diversity + higher correctness = higher q


def test_screening_detects_quality_driver():
    # outcome driven by q, not by size: q should screen off identity
    cells = []
    for t, tp, fam, q in [("t1", 1e9, "a", -1.2), ("t2", 3e9, "a", -0.3),
                          ("t3", 7e9, "a", 0.5), ("t4", 14e9, "a", 0.9), ("l1", 8e9, "b", 0.0)]:
        cells.append(dict(student="s100m", teacher=t, t_params=tp, family=fam, q=q,
                          L_inf_hat=2.0 - 0.3 * q + 0.01 * np.log2(tp / 1e9)))
    r = nv.screening_test(cells, outcome="L_inf_hat")["per_student"]["s100m"]
    assert r["r2_quality(q)"] >= r["r2_identity(logT+family)"] - 0.1


def test_screening_detects_size_driver():
    # outcome driven by size, weak q: identity should not be screened off by q
    cells = []
    rng = np.random.default_rng(0)
    for t, tp, fam in [("t1", 1e9, "a"), ("t2", 3e9, "a"), ("t3", 7e9, "a"), ("t4", 14e9, "a"), ("l1", 8e9, "b")]:
        cells.append(dict(student="s100m", teacher=t, t_params=tp, family=fam,
                          q=float(rng.normal()), L_inf_hat=3.0 - 0.2 * np.log2(tp / 1e9)))
    r = nv.screening_test(cells, outcome="L_inf_hat")["per_student"]["s100m"]
    assert r["r2_identity(logT+family)"] > r["r2_quality(q)"] - 0.05


def test_predictive_and_allocation_run():
    by_cell = {}
    for s in ("s100m",):
        for t, tp, al in (("t1p5b", 1.5e9, 0.35), ("t7b", 7e9, 0.5), ("t72b", 72e9, 0.6)):
            by_cell[(s, t)] = [(D, 2.0 + 4000 * (D) ** (-al)) for D in (25e6, 50e6, 100e6, 200e6, 400e6)]
    pr = nv.predictive_law(by_cell, holdout_frac=0.34, seed=1)
    assert "heldout_rmse" in pr or "note" in pr
    fit_by_cell = {(s, t): {"E": 2.0, "A": 4000.0, "alpha": 0.5, "t_params": tp}
                   for (s, t), tp in (((s, t), tp) for s in ("s100m",) for t, tp in (("t1p5b", 1.5e9), ("t7b", 7e9), ("t72b", 72e9)))}
    alloc = nv.compute_allocation_frontier(fit_by_cell, {"s100m": 100e6})
    assert "s100m" in alloc["per_student"] and len(alloc["per_student"]["s100m"]) >= 1
