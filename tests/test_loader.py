import yaml

from synscale.config.loader import load_experiment
from synscale.tracking.ids import experiment_id


def test_loader_composes_and_overrides(tmp_path, student_cfg, teacher_cfg, gen_cfg, train_cfg, eval_cfg):
    (tmp_path / "configs").mkdir()
    for name, cfg in [("student", student_cfg), ("teacher", teacher_cfg), ("generation", gen_cfg),
                      ("training", train_cfg), ("evaluation", eval_cfg)]:
        (tmp_path / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    spec = {
        "study": "main", "arm": "synthetic", "variant": "eqtok",
        "student": "configs/student.yaml", "teacher": "configs/teacher.yaml",
        "generation": "configs/generation.yaml", "training": "configs/training.yaml",
        "evaluation": "configs/evaluation.yaml",
        "overrides": {"training.seed_phase": 3, "training.synthetic_phase.tokens": 250_000_000},
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(spec))
    cfg = load_experiment(p, repo_root=tmp_path)
    assert cfg.training.seed_phase == 3
    assert cfg.training.synthetic_phase.tokens == 250_000_000
    assert experiment_id(cfg).startswith("main.s100m.t7b.d250m.eqtok.s3-")
