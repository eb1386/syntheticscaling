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
    d_syn=60_000, d2_tokens=80_000, replay_fraction=0.25,
    base_tokens={"s_micro": 200_000},
    phase_seeds=1, base_seeds={"s_micro": 1}, tier2_seeds=0, tier2_teachers=(),
    c1_seeds=1, pool_prompts=1500,
    dsweep_student=None, dsweep_teachers=(), dsweep_levels=(), dsweep_seeds=0,
    notes="Tiny CPU end-to-end validation with MockBackend and a byte tokenizer.",
)

PROFILES: dict[str, Profile] = {p.name: p for p in (LOCAL5080, LOCAL5080_FAST, SMOKE, MICRO)}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choices: {sorted(PROFILES)}")
    return PROFILES[name]
