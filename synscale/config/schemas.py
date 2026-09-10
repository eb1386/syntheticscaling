"""Configuration schemas for every experimental object.

Every experimental condition is fully described by an ExperimentConfig, which composes
a StudentConfig, an optional TeacherConfig + GenerationConfig (absent for control arms),
a TrainingConfig (base phase and synthetic phase), and an EvalConfig. Configs are
loaded from YAML, validated here, and hashed (see synscale.tracking.ids) so that the
experiment ID is a deterministic function of the resolved configuration.

Design rules
------------
* No defaults that silently change a scientific control. Fields that define a control
  (decoding, token budgets, optimizer, schedule) are required.
* Hardware/runtime fields are NOT part of a config; they are recorded in the run manifest.
* Anything that varies across teacher conditions must live in TeacherConfig or in the
  dataset reference, never in TrainingConfig.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class StudentConfig(BaseModel):
    """Architecture of one member of the student family (LLaMA-like decoder)."""

    name: str = Field(..., description="Short id, e.g. 's025m'.")
    target_params: int = Field(..., description="Nominal total parameter count (e.g. 25_000_000).")
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    context_length: int
    tie_embeddings: bool
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    depth_scaled_init: bool = True
    activation: Literal["swiglu"] = "swiglu"
    norm: Literal["rmsnorm"] = "rmsnorm"
    attention_bias: bool = False
    mlp_bias: bool = False

    @model_validator(mode="after")
    def _check_heads(self) -> "StudentConfig":
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads (GQA).")
        if self.n_heads * self.head_dim != self.d_model:
            # Allowed in principle, but our family keeps q-proj square for simplicity.
            raise ValueError("n_heads * head_dim must equal d_model in this family.")
        return self


class TeacherConfig(BaseModel):
    """One teacher checkpoint. Everything that could differ across teachers is explicit."""

    name: str = Field(..., description="Short id, e.g. 't7b'.")
    family: str = Field(..., description="Model family, e.g. 'qwen2.5'.")
    hf_repo: str = Field(..., description="Hugging Face repo id.")
    revision: str = Field(..., description="Git commit SHA of the HF repo; never 'main'.")
    nominal_params: int = Field(..., description="Nominal parameter count, e.g. 7_000_000_000.")
    exact_params: Optional[int] = Field(None, description="Exact parameter count if known.")
    variant: Literal["instruct", "base"] = "instruct"
    weight_dtype: Literal["bf16", "fp8_e4m3"] = Field(
        ..., description="Serving precision; must be identical across all teachers in a study."
    )
    tensor_parallel: int = 1
    max_model_len: int = 4096
    chat_template: Literal["hf_default"] = "hf_default"
    thinking_mode: bool = Field(False, description="Must be False for all teachers.")
    license: str = ""


class GenerationConfig(BaseModel):
    """Decoding and filtering controls shared by every teacher condition."""

    name: str
    prompt_pool: str = Field(..., description="Path or id of the versioned prompt pool.")
    prompt_pool_sha256: str
    system_prompt_id: str
    temperature: float
    top_p: float
    top_k: int = -1
    max_new_tokens: int
    samples_per_prompt: int = 1
    sampling_seed: int
    stop_sequences: list[str] = Field(default_factory=list)
    filter_ruleset: str = Field(..., description="Id of the teacher-agnostic filter ruleset.")
    target_student_tokens: int = Field(
        ..., description="Deliverable size after filtering, counted in student-tokenizer tokens."
    )


class PhaseConfig(BaseModel):
    """Hyperparameters of one training phase (base pretraining or synthetic phase)."""

    tokens: int = Field(..., description="Training tokens for this phase (student tokenizer).")
    batch_tokens: int = Field(..., description="Effective batch size in tokens.")
    lr_peak: float
    lr_min_ratio: float = Field(..., description="Final LR as a fraction of peak.")
    schedule: Literal["wsd", "cosine"]
    warmup_tokens: int
    decay_tokens: Optional[int] = Field(None, description="WSD decay length; required if schedule=wsd.")
    weight_decay: float
    betas: tuple[float, float]
    eps: float = 1e-8
    grad_clip: float = 1.0
    loss_mask: Literal["all_tokens", "response_only"]
    packing: Literal["packed_masked", "packed_unmasked", "padded"]
    replay_fraction: float = Field(0.0, description="Fraction of batch drawn from base corpus (synthetic phase only).")

    @model_validator(mode="after")
    def _check_schedule(self) -> "PhaseConfig":
        if self.schedule == "wsd" and self.decay_tokens is None:
            raise ValueError("decay_tokens is required for WSD schedule.")
        if self.warmup_tokens > self.tokens:
            raise ValueError("warmup_tokens exceeds phase tokens.")
        return self


class TrainingConfig(BaseModel):
    name: str
    tokenizer: str = Field(..., description="Tokenizer id; must be identical for all students.")
    tokenizer_sha256: str
    context_length: int
    precision: Literal["bf16_mixed"] = "bf16_mixed"
    optimizer: Literal["adamw"] = "adamw"
    base_phase: PhaseConfig
    synthetic_phase: Optional[PhaseConfig] = None
    checkpoint_rule: Literal["final"] = Field("final", description="Final checkpoint only; no best-on-benchmark selection.")
    eval_every_tokens: int
    seed_base: int = Field(..., description="Seed for base-phase init and data order.")
    seed_phase: int = Field(..., description="Seed for synthetic-phase data order (and nothing else).")


class EvalConfig(BaseModel):
    name: str
    harness: Literal["lm-eval"] = "lm-eval"
    harness_version: str
    primary_tasks: list[str]
    secondary_tasks: list[str] = Field(default_factory=list)
    num_fewshot: int = 0
    batch_size: int = 64
    eval_seed: int = 1234
    nll_sets: list[str] = Field(default_factory=list, description="Held-out NLL evaluation set ids.")


Arm = Literal["synthetic", "base_only", "matched_real", "human_instruct"]


class ExperimentConfig(BaseModel):
    """One launchable experimental condition."""

    study: Literal["pilot", "main", "dsweep", "ablation"]
    arm: Arm
    student: StudentConfig
    teacher: Optional[TeacherConfig] = None
    generation: Optional[GenerationConfig] = None
    training: TrainingConfig
    evaluation: EvalConfig
    dataset_ref: Optional[str] = Field(None, description="Path of the (subsampled) training file for the second phase.")
    dataset_sha256: Optional[str] = None
    variant: str = Field("eqtok", description="Budget-control variant: eqtok | eqex | lenmatch | ...")
    notes: str = ""

    @model_validator(mode="after")
    def _check_arm(self) -> "ExperimentConfig":
        if self.arm == "synthetic" and (self.teacher is None or self.generation is None):
            raise ValueError("synthetic arm requires teacher and generation configs.")
        if self.arm != "synthetic" and self.teacher is not None:
            raise ValueError("control arms must not carry a teacher config.")
        if self.arm != "base_only" and self.training.synthetic_phase is None:
            raise ValueError("second-phase arms require training.synthetic_phase.")
        return self
