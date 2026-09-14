"""Hardware/scope profiles: which grid actually runs on which machine.

The locked methodology (docs/00) targets a 2x RTX PRO 6000 + 2x RTX 5080 cluster and a
5x7 grid with teachers up to 72B. A single 16 GB RTX 5080 cannot serve teachers >~14B
even in int4, cannot train a 1B student in reasonable time, and runs everything serially.
This module defines self-contained profiles that DO run on one card, plus a tiny `smoke`
profile that exercises the whole pipeline in minutes on CPU/GPU.

A Profile is consumed by scripts/run_pipeline.py (the orchestrator), scripts/make_matrix.py
and synscale.analysis.compute_budget. Scientific scoping choices and their rationale are in
docs/RUN_ON_5080.md and docs/05_pilot_design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class TeacherSpec:
    name: str            # short id used in experiment ids, e.g. "t7b"; matches configs/teachers/*.yaml name
    hf_repo: str
    nominal_params: int
    quantization: Optional[str]   # None=bf16, "awq_marlin", "fp8"
    config_path: str = ""         # teacher config fragment (BF16 reference; quantization from this profile)
    tensor_parallel: int = 1
    max_model_len: int = 4096
    family: str = "qwen2.5"       # for the cross-family screening analysis (docs/24)
    student_subset: Optional[tuple] = None  # if set, this teacher is used only for these students (partial 2nd family)


@dataclass(frozen=True)
class Profile:
    name: str
    students: tuple[str, ...]
    teachers: tuple[TeacherSpec, ...]
    controls: tuple[str, ...]                 # arms without a teacher
    d_syn: int                                # treatment tokens per cell (student tokenizer)
    d2_tokens: int                            # phase-2 total (treatment + replay)
    replay_fraction: float
    base_tokens: Mapping[str, int]            # per student
    phase_seeds: int                          # tier-1 seeds per cell
    base_seeds: Mapping[str, int]             # base runs per student
    tier2_seeds: int                          # extra seeds at end teachers + argmax
    tier2_teachers: tuple[str, ...]
    c1_seeds: int
    pool_prompts: int
    dsweep_student: Optional[str]
    dsweep_teachers: tuple[str, ...]
    dsweep_levels: tuple[int, ...]
    dsweep_seeds: int
    primary_eval: str = "primary_v1"
    generation: str = "main_v1"
    training_prefix: str = "train_"      # experiment YAMLs use configs/training/<prefix><student>.yaml
    primary_outcome: str = "p_cc"        # "loss" for the scaling study (loss-vs-D), "p_cc" otherwise
    teacher_precision: str = "awq_marlin"
    gpu: str = "rtx5080"                 # target GPU id (see synscale.analysis.config_planner.GPUS)
    persistent_dir: str = "data"         # root for tokenizer/corpora/pools/checkpoints on a persistent volume
    measure_data_properties: bool = False  # compute the student-referenced data-quality index q (docs/24)
    notes: str = ""


# --- Qwen2.5-Instruct rungs servable on one RTX 5080 (16 GB) via AWQ int4 ---------------
# 0.5-3B also run in BF16; >=7B require int4 (7B int4 ~= 4.5 GB, 14B int4 ~= 8 GB).
_Q = "Qwen/Qwen2.5-{}-Instruct{}"
def _awq(size: str) -> str: return _Q.format(size, "-AWQ")

def _tc(name, size, params):
    return TeacherSpec(name, _awq(size), params, "awq_marlin",
                       config_path=f"configs/teachers/qwen2.5-{size.lower()}-instruct.yaml", max_model_len=4096)

LOCAL_TEACHERS = (
    _tc("t0p5b", "0.5B", 500_000_000),
    _tc("t1p5b", "1.5B", 1_500_000_000),
    _tc("t3b",   "3B",   3_000_000_000),
    _tc("t7b",   "7B",   7_000_000_000),
    _tc("t14b",  "14B",  14_000_000_000),
)

LOCAL5080 = Profile(
    name="local5080",
    students=("s025m", "s100m", "s250m"),
    teachers=LOCAL_TEACHERS,
    controls=("base_only", "matched_real", "human_instruct"),   # C0, C1, C1b
    d_syn=300_000_000,
    d2_tokens=400_000_000,
    replay_fraction=0.25,
    base_tokens={"s025m": 5_000_000_000, "s100m": 5_000_000_000, "s250m": 5_000_000_000},
    phase_seeds=3,
    base_seeds={"s025m": 3, "s100m": 1, "s250m": 1},
    tier2_seeds=2,
    tier2_teachers=("t1p5b", "t14b"),
    c1_seeds=5,
    pool_prompts=1_200_000,
    dsweep_student="s100m",
    dsweep_teachers=("t1p5b", "t14b"),
    dsweep_levels=(75_000_000, 150_000_000, 300_000_000),
    dsweep_seeds=2,
    notes="Complete single-5080 study: 3 students x 5 teachers (0.5-14B, AWQ int4) + 3 controls.",
)

# A reduced knob for ~1 week instead of ~3-4 weeks (still a complete grid, lower power).
LOCAL5080_FAST = Profile(
    name="local5080_fast",
    students=("s025m", "s100m"),
    teachers=LOCAL_TEACHERS[:4],           # 0.5-7B
    controls=("base_only", "matched_real", "human_instruct"),
    d_syn=150_000_000,
    d2_tokens=200_000_000,
    replay_fraction=0.25,
    base_tokens={"s025m": 3_000_000_000, "s100m": 3_000_000_000},
    phase_seeds=3,
    base_seeds={"s025m": 3, "s100m": 1},
    tier2_seeds=2,
    tier2_teachers=("t1p5b", "t7b"),
    c1_seeds=3,
    pool_prompts=500_000,
    dsweep_student="s100m",
    dsweep_teachers=("t1p5b", "t7b"),
    dsweep_levels=(75_000_000, 150_000_000),
    dsweep_seeds=2,
    notes="Reduced single-5080 study (~1 week): 2 students x 4 teachers.",
)

# Tiny end-to-end validation (minutes, CPU or GPU, MockBackend allowed).
SMOKE = Profile(
    name="smoke",
    students=("s025m",),
    teachers=(TeacherSpec("t0p5b", _awq("0.5B"), 500_000_000, "awq_marlin", max_model_len=2048),
              TeacherSpec("t3b", _awq("3B"), 3_000_000_000, "awq_marlin", max_model_len=2048)),
    controls=("base_only", "matched_real"),
    d_syn=200_000,
    d2_tokens=260_000,
    replay_fraction=0.25,
    base_tokens={"s025m": 2_000_000},
    phase_seeds=2,
    base_seeds={"s025m": 1},
    tier2_seeds=0,
    tier2_teachers=(),
    c1_seeds=2,
    pool_prompts=3_000,
    dsweep_student=None,
    dsweep_teachers=(),
    dsweep_levels=(),
    dsweep_seeds=0,
    notes="Minutes-long CPU smoke test of the whole pipeline with MockBackend.",
)


# CPU/CI end-to-end validation (seconds-minutes): a 2-layer toy student, MockBackend.
MICRO = Profile(
    name="micro",
    students=("s_micro",),
    teachers=(TeacherSpec("t0p5b", _awq("0.5B"), 500_000_000, "awq_marlin",
                          config_path="configs/teachers/qwen2.5-0.5b-instruct.yaml", max_model_len=2048),
              TeacherSpec("t3b", _awq("3B"), 3_000_000_000, "awq_marlin",
                          config_path="configs/teachers/qwen2.5-3b-instruct.yaml", max_model_len=2048)),
    controls=("base_only", "matched_real"),
    d_syn=100_000, d2_tokens=140_000, replay_fraction=0.25,
    base_tokens={"s_micro": 200_000},
    phase_seeds=1, base_seeds={"s_micro": 1}, tier2_seeds=0, tier2_teachers=(),
    c1_seeds=1, pool_prompts=1500,
    dsweep_student=None, dsweep_teachers=(), dsweep_levels=(), dsweep_seeds=0,
    measure_data_properties=True,
    notes="Tiny CPU end-to-end validation with MockBackend and a byte tokenizer.",
)


# --- Smaller teachers for the scaling study: 0.5-7B served FP8 (quantized on load), uniform precision.
#     0.5-3B also fit BF16; a 3B BF16-vs-FP8 bridge cell (ablation) measures the quantisation effect.
def _fp8(name, size, params):
    return TeacherSpec(name, _Q.format(size, ""), params, "fp8",
                       config_path=f"configs/teachers/qwen2.5-{size.lower()}-instruct.yaml", max_model_len=4096)

SCALING_TEACHERS = (
    _fp8("t0p5b", "0.5B", 500_000_000),
    _fp8("t1p5b", "1.5B", 1_500_000_000),
    _fp8("t3b",   "3B",   3_000_000_000),
    _fp8("t7b",   "7B",   7_000_000_000),
)

# Flagship study for one RTX 5080: does synthetic-data scaling depend on teacher size?
# Student S x teacher T grid; the data-quantity axis D comes from the loss-vs-tokens curve
# recorded during each synthetic run (train_log.jsonl), so no separate D-sweep runs are needed.
SCALING5080 = Profile(
    name="scaling5080",
    students=("s025m", "s050m", "s100m", "s250m"),
    teachers=SCALING_TEACHERS,
    controls=("base_only", "matched_real", "human_instruct"),
    d_syn=400_000_000, d2_tokens=530_000_000, replay_fraction=0.25,
    base_tokens={"s025m": 500_000_000, "s050m": 1_000_000_000, "s100m": 2_000_000_000, "s250m": 4_900_000_000},
    phase_seeds=3, base_seeds={"s025m": 3, "s050m": 1, "s100m": 1, "s250m": 1},
    tier2_seeds=0, tier2_teachers=(),
    c1_seeds=3, pool_prompts=1_500_000,
    dsweep_student=None, dsweep_teachers=(), dsweep_levels=(), dsweep_seeds=0,
    training_prefix="train_scaling_", primary_outcome="loss", teacher_precision="fp8",
    notes="Scaling-laws study: 4 students x 4 FP8 teachers (0.5-7B); D-axis from loss-vs-tokens curves.",
)

SCALING5080_FAST = Profile(
    name="scaling5080_fast",
    students=("s025m", "s050m", "s100m"),
    teachers=SCALING_TEACHERS,
    controls=("base_only", "matched_real", "human_instruct"),
    d_syn=250_000_000, d2_tokens=330_000_000, replay_fraction=0.25,
    base_tokens={"s025m": 500_000_000, "s050m": 1_000_000_000, "s100m": 2_000_000_000},
    phase_seeds=2, base_seeds={"s025m": 2, "s050m": 1, "s100m": 1},
    tier2_seeds=0, tier2_teachers=(),
    c1_seeds=2, pool_prompts=800_000,
    dsweep_student=None, dsweep_teachers=(), dsweep_levels=(), dsweep_seeds=0,
    training_prefix="train_scaling_", primary_outcome="loss", teacher_precision="fp8",
    notes="Reduced scaling study (~1 week): 3 students x 4 FP8 teachers.",
)


# ============================================================================================
# C4: two-family, main-track-targeted study on rented H100s (docs/24). FP8 for all teachers
# (uniform precision, quantized on load), 70B/72B served tensor-parallel over 2 cards. The
# Qwen family spans all students; a partial Llama family at two student sizes breaks the
# size-vs-family confound at the key points without doubling the whole grid (cost-efficient C4).
# ============================================================================================
def _fp8t(name, size, params, family, config, tp=1, subset=None):
    return TeacherSpec(name, size, params, "fp8", config_path=config, tensor_parallel=tp,
                       max_model_len=4096, family=family, student_subset=subset)

_C4_STUDENTS = ("s025m", "s050m", "s100m", "s250m", "s500m", "s1b")
_LLAMA_SUBSET = ("s100m", "s1b")   # cross-family screen at one small and one large student

def _int4t(name, repo, params, family, config, subset=None):
    # int4 AWQ: every teacher (incl 70B/72B) fits ONE H100 (~35 GB at 70B) -> single-card, spot-safe.
    return TeacherSpec(name, repo, params, "awq_marlin", config_path=config, tensor_parallel=1,
                       max_model_len=4096, family=family, student_subset=subset)

C4_TEACHERS = (
    # Qwen2.5-Instruct AWQ (official int4 repos), uniform precision across the full size axis, all students
    _int4t("t0p5b", "Qwen/Qwen2.5-0.5B-Instruct-AWQ", 500_000_000, "qwen2.5", "configs/teachers/qwen2.5-0.5b-instruct.yaml"),
    _int4t("t1p5b", "Qwen/Qwen2.5-1.5B-Instruct-AWQ", 1_500_000_000, "qwen2.5", "configs/teachers/qwen2.5-1.5b-instruct.yaml"),
    _int4t("t3b",   "Qwen/Qwen2.5-3B-Instruct-AWQ",   3_000_000_000, "qwen2.5", "configs/teachers/qwen2.5-3b-instruct.yaml"),
    _int4t("t7b",   "Qwen/Qwen2.5-7B-Instruct-AWQ",   7_000_000_000, "qwen2.5", "configs/teachers/qwen2.5-7b-instruct.yaml"),
    _int4t("t14b",  "Qwen/Qwen2.5-14B-Instruct-AWQ",  14_000_000_000, "qwen2.5", "configs/teachers/qwen2.5-14b-instruct.yaml"),
    _int4t("t32b",  "Qwen/Qwen2.5-32B-Instruct-AWQ",  32_000_000_000, "qwen2.5", "configs/teachers/qwen2.5-32b-instruct.yaml"),
    _int4t("t72b",  "Qwen/Qwen2.5-72B-Instruct-AWQ",  72_000_000_000, "qwen2.5", "configs/teachers/qwen2.5-72b-instruct.yaml"),
    # Llama-3.x partial second family at two students; 70B int4 (hugging-quants AWQ) fits one card
    _fp8t("l3b",   "meta-llama/Llama-3.2-3B-Instruct", 3_210_000_000, "llama3.2", "configs/teachers/l3b-instruct.yaml", subset=_LLAMA_SUBSET),
    _fp8t("l8b",   "meta-llama/Llama-3.1-8B-Instruct", 8_030_000_000, "llama3.1", "configs/teachers/l8b-instruct.yaml", subset=_LLAMA_SUBSET),
    _int4t("l70b", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", 70_600_000_000, "llama3.1", "configs/teachers/l70b-instruct.yaml", subset=_LLAMA_SUBSET),
)

C4 = Profile(
    name="c4",
    students=_C4_STUDENTS,
    teachers=C4_TEACHERS,
    controls=("base_only", "matched_real", "human_instruct"),
    d_syn=400_000_000, d2_tokens=530_000_000, replay_fraction=0.25,
    base_tokens={"s025m": 500_000_000, "s050m": 1_000_000_000, "s100m": 2_000_000_000,
                 "s250m": 4_900_000_000, "s500m": 10_000_000_000, "s1b": 15_000_000_000},
    phase_seeds=3, base_seeds={"s025m": 3, "s050m": 1, "s100m": 1, "s250m": 1, "s500m": 1, "s1b": 1},
    tier2_seeds=0, tier2_teachers=(),
    c1_seeds=3, pool_prompts=2_000_000,
    dsweep_student=None, dsweep_teachers=(), dsweep_levels=(), dsweep_seeds=0,
    training_prefix="train_scaling_", primary_outcome="loss", teacher_precision="awq_marlin",
    gpu="h100_80", persistent_dir="/workspace/synscale", measure_data_properties=True,
    notes="C4 main-track target: Qwen 0.5-72B (all students) + partial Llama 3/8/70B at 100M and 1B; "
          "loss-based scaling, student-referenced data-quality index, cross-family screening (docs/24).",
)

PROFILES: dict[str, Profile] = {p.name: p for p in (C4, SCALING5080, SCALING5080_FAST, LOCAL5080, LOCAL5080_FAST, SMOKE, MICRO)}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choices: {sorted(PROFILES)}")
    return PROFILES[name]
