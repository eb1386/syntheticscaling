import pytest

from synscale.config.schemas import (
    EvalConfig,
    ExperimentConfig,
    GenerationConfig,
    PhaseConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
)


@pytest.fixture
def student_cfg() -> StudentConfig:
    return StudentConfig(
        name="s100m", target_params=100_000_000, n_layers=12, d_model=768, n_heads=12,
        n_kv_heads=12, head_dim=64, d_ff=2048, vocab_size=32768, context_length=2048,
        tie_embeddings=True,
    )


@pytest.fixture
def teacher_cfg() -> TeacherConfig:
    return TeacherConfig(
        name="t7b", family="qwen2.5", hf_repo="Qwen/Qwen2.5-7B-Instruct",
        revision="0123456789abcdef0123456789abcdef01234567", nominal_params=7_000_000_000,
        weight_dtype="bf16",
    )


@pytest.fixture
def gen_cfg() -> GenerationConfig:
    return GenerationConfig(
        name="main_v1", prompt_pool="data/prompts/pool_v1.jsonl", prompt_pool_sha256="a" * 64,
        system_prompt_id="sys_v1", temperature=0.7, top_p=0.95, max_new_tokens=512,
        sampling_seed=1, filter_ruleset="filter_v1", target_student_tokens=600_000_000,
    )


def _phase(tokens: int) -> PhaseConfig:
    return PhaseConfig(
        tokens=tokens, batch_tokens=524_288, lr_peak=3e-3, lr_min_ratio=0.1, schedule="wsd",
        warmup_tokens=min(tokens // 10, 100_000_000), decay_tokens=tokens // 5, weight_decay=0.1,
        betas=(0.9, 0.95), loss_mask="all_tokens", packing="packed_masked",
    )


@pytest.fixture
def train_cfg() -> TrainingConfig:
    return TrainingConfig(
        name="main_v1", tokenizer="tok32k_v1", tokenizer_sha256="b" * 64, context_length=2048,
        base_phase=_phase(2_000_000_000), synthetic_phase=_phase(500_000_000),
        eval_every_tokens=100_000_000, seed_base=0, seed_phase=1,
    )


@pytest.fixture
def eval_cfg() -> EvalConfig:
    return EvalConfig(name="primary_v1", harness_version="0.4.8", primary_tasks=["arc_easy", "piqa"])


@pytest.fixture
def exp_cfg(student_cfg, teacher_cfg, gen_cfg, train_cfg, eval_cfg) -> ExperimentConfig:
    return ExperimentConfig(
        study="main", arm="synthetic", student=student_cfg, teacher=teacher_cfg,
        generation=gen_cfg, training=train_cfg, evaluation=eval_cfg,
    )
