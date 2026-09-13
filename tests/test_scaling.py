"""Scaling-law analysis: power-law recovery and teacher-dependence test on synthetic curves."""
import json
from pathlib import Path
import numpy as np
from synscale.analysis import scaling as sc


def test_power_law_recovers_alpha():
    D = np.array([25e6, 50e6, 100e6, 200e6, 400e6, 800e6])
    E, A, alpha = 2.0, 5000.0, 0.5
    L = E + A * D ** (-alpha)
    fit = sc.fit_power_law(D, L)
    assert abs(fit["alpha"] - alpha) < 0.06
    assert abs(fit["E"] - E) < 0.05


def _write_run(root: Path, eid, student, s_params, arm, teacher, t_params, curve):
    d = root / eid
    d.mkdir(parents=True)
    cfg = {"arm": arm, "student": {"name": student, "target_params": s_params},
           "teacher": ({"name": teacher, "nominal_params": t_params} if teacher else None),
           "training": {"seed_phase": 1, "synthetic_phase": {"replay_fraction": 0.25}}}
    (d / "manifest.json").write_text(json.dumps({"experiment_id": eid, "config": cfg}))
    with open(d / "train_log.jsonl", "w") as f:
        for tokens, loss in curve:
            f.write(json.dumps({"step": 1, "tokens": tokens, "held_out_nll": {"instr": loss}}) + "\n")


def test_teacher_dependence_end_to_end(tmp_path):
    # Two students, three teachers whose data scales better with size (alpha rises with T),
    # plus a matched-real control. Curves are on total tokens; D_syn = 0.75*tokens.
    root = tmp_path / "results"
    Ttoks = [int(x / 0.75) for x in (25e6, 50e6, 100e6, 200e6, 400e6, 530e6)]
    for student, s_params in (("s025m", 25_000_000), ("s100m", 100_000_000)):
        for teacher, t_params, alpha in (("t0p5b", 5e8, 0.35), ("t3b", 3e9, 0.5), ("t7b", 7e9, 0.62)):
            curve = [(tk, 2.0 + 4000.0 * (tk * 0.75) ** (-alpha)) for tk in Ttoks]
            _write_run(root, f"main.{student}.{teacher}.d1.s1", student, s_params, "synthetic", teacher, t_params, curve)
        curve = [(tk, 2.1 + 4000.0 * (tk * 0.75) ** (-0.4)) for tk in Ttoks]
        _write_run(root, f"main.{student}.real.d1.s1", student, s_params, "matched_real", None, None, curve)
    rep = sc.run_scaling_analysis(root, out_dir=tmp_path / "out", eval_set="instr")
    assert rep["n_runs_with_curves"] == 8
    fits = {(r["student"], r["cond"]): r for r in rep["cell_fits"]}
    # alpha should increase with teacher size within a student
    a = fits[("s025m", "t0p5b")]["alpha"]; b = fits[("s025m", "t7b")]["alpha"]
    assert b > a
    # per-cell fits report the estimated asymptote (not "irreducible loss") and a best form
    assert "L_inf_hat" in fits[("s025m", "t7b")] and "best_form" in fits[("s025m", "t7b")]
    # single-seed cells are flagged (no across-seed CI)
    assert fits[("s025m", "t7b")]["seed_ci"] is False
    # teacher-dependence test reports a positive alpha slope on log T
    td = rep["teacher_dependence"]["per_student"]["s025m"]["alpha"]
    assert td["slope_per_log2T"] > 0
    # synthetic-data multiplier present for at least one teacher at s025m
    assert any(r["student"] == "s025m" for r in rep["synthetic_data_multiplier"])
