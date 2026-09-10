import pytest

from synscale.config.schemas import ExperimentConfig, StudentConfig
from synscale.tracking.ids import config_hash, experiment_id, human_tokens
from synscale.tracking.manifest import new_manifest


def test_human_tokens():
    assert human_tokens(0) == "0"
    assert human_tokens(500_000_000) == "500m"
    assert human_tokens(2_000_000_000) == "2b"
    assert human_tokens(123) == "123"


def test_experiment_id_is_deterministic(exp_cfg):
    a = experiment_id(exp_cfg)
    b = experiment_id(exp_cfg.model_copy(deep=True))
    assert a == b
    assert a.startswith("main.s100m.t7b.d500m.eqtok.s1-")
    assert len(a.split("-")[-1]) == 8


def test_notes_do_not_change_hash(exp_cfg):
    h0 = config_hash(exp_cfg)
    c2 = exp_cfg.model_copy(deep=True, update={"notes": "anything"})
    assert config_hash(c2) == h0


def test_controlled_change_changes_hash(exp_cfg):
    h0 = config_hash(exp_cfg)
    c2 = exp_cfg.model_copy(deep=True)
    c2.generation.temperature = 1.0
    assert config_hash(c2) != h0


def test_control_arm_ids(exp_cfg):
    base = exp_cfg.model_copy(deep=True, update={"arm": "base_only", "teacher": None, "generation": None})
    base.training.synthetic_phase = None
    base = ExperimentConfig.model_validate(base.model_dump())
    assert experiment_id(base).startswith("main.s100m.base.d0.-.s1-")
    real = exp_cfg.model_copy(deep=True, update={"arm": "matched_real", "teacher": None, "generation": None})
    real = ExperimentConfig.model_validate(real.model_dump())
    assert experiment_id(real).startswith("main.s100m.real.d500m.eqtok.s1-")


def test_synthetic_arm_requires_teacher(exp_cfg):
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({**exp_cfg.model_dump(), "teacher": None})


def test_student_gqa_validation():
    with pytest.raises(ValueError):
        StudentConfig(
            name="x", target_params=1, n_layers=1, d_model=768, n_heads=12, n_kv_heads=5,
            head_dim=64, d_ff=2048, vocab_size=32768, context_length=2048, tie_embeddings=True,
        )


def test_manifest_roundtrip(exp_cfg, tmp_path):
    m = new_manifest(exp_cfg)
    assert m.experiment_id == experiment_id(exp_cfg)
    out = m.write(tmp_path)
    assert out.exists()
    assert (tmp_path / m.experiment_id / "manifest.json").read_text().startswith("{")
