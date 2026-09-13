"""Experiment matrix: the locked design (docs/00_locked_methodology.md) as code.

This module is the single source of truth for

* the student table (locked shapes -> D_base, batch, LR, warmup; tests assert that
  ``configs/training/train_<student>.yaml`` matches it),
* the teacher ladder and which rungs belong to which study,
* the seed tiers, the D-sweep levels and the documented ablation list,
* the naming conventions for ``variant`` strings and ``dataset_ref`` paths.

``scripts/make_matrix.py`` turns the specs into one YAML per run under
``configs/experiments/<study>/<experiment_id>.yaml``; ``synscale.analysis.compute_budget``
imports the same specs so that the compute budget is derived from the actual matrix.

Conventions (see docs/06_full_experimental_matrix.md)
------------------------------------------------------
* Every phase-2 run trains D2 = 800M tokens (600M treatment + 200M replay), so the ``d<...>``
  field of every branch ID reads ``d800m``. The *synthetic dose* D_syn is carried by the
  ``variant`` string: ``eqtok`` means D_syn = 600M (the whole delivered pool);
  ``eqtok-d75m`` / ``eqtok-d150m`` / ``eqtok-d300m`` are nested prefixes with the remaining
  treatment slots filled from fresh web shards (doc 00 §0.9).
* ``dataset_ref`` names the composed 600M-token treatment file:
  ``data/processed/phase2/<source>.<variant>.bin`` where ``<source>`` is
  ``<teacher>.<generation>`` for synthetic arms and a fixed id for the controls.
* Base pretraining runs are ``arm: base_only`` with ``training.synthetic_phase: null``
  (ID ``...base.d0.-.s1``). The C0 "nothing new" cooldown (doc 00 §0.8: 600M tokens of
  already-seen base shards) is ``arm: base_only`` *with* a synthetic phase whose
  ``dataset_ref`` is the seen-shards file (ID ``...base.d800m.-.s<k>``).
* Teacher ids avoid dots (the experiment ID is dot-delimited): ``t0p5b``, ``t1p5b``,
  ``t3b``, ``t7b``, ``t14b``, ``t32b``, ``t72b``; ``llama8b`` is the cross-family anchor;
  ``t7b_base`` / ``t32b_base`` are the base-checkpoint teachers of the instruct-vs-base ablation.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from synscale.config.loader import _apply_override
from synscale.config.schemas import ExperimentConfig
from synscale.tracking.ids import experiment_id

REPO_ROOT = Path(__file__).resolve().parents[2]

STUDIES = ("pilot", "main", "dsweep", "ablation")

# ----------------------------------------------------------------------------------------
# Locked constants
# ----------------------------------------------------------------------------------------
D2_TOKENS = 800_000_000          # phase-2 tokens per branch (doc 00 §0.3)
D_SYN_MAIN = 600_000_000         # treatment tokens per branch (doc 00 §0.3)
REPLAY_FRACTION = 0.25           # rho (doc 00 §0.3)
DSWEEP_LEVELS = (75_000_000, 150_000_000, 300_000_000)   # doc 00 §0.9
ABLATION_D_SYN = 150_000_000     # generation-protocol ablations run at 150M (cheap pools)


@dataclass(frozen=True)
class StudentRow:
    name: str
    n_tot: int
    n_ne: int
    batch_tokens: int
    d_base: int                  # locked D_base (doc 00 §0.4 table; rounded values)
    eval_every_steps: int

    @property
    def lr_peak(self) -> float:
        """eta = 2e-3 * (N_ne / 25e6)^-0.4, rounded to two significant figures."""
        return float(f"{2e-3 * (self.n_ne / 25e6) ** -0.4:.2g}")

    @property
    def warmup_tokens(self) -> int:
        """5 % of the base tokens, capped at 1000 optimizer steps."""
        return min(int(0.05 * self.d_base), 1000 * self.batch_tokens)

    @property
    def eval_every_tokens(self) -> int:
        return self.batch_tokens * self.eval_every_steps


STUDENT_TABLE: dict[str, StudentRow] = {
    "s025m": StudentRow("s025m", 24_877_440, 12_589_440, 262_144, 5_000_000_000, 80),
    "s100m": StudentRow("s100m", 100_092_672, 75_516_672, 524_288, 5_000_000_000, 40),
    "s250m": StudentRow("s250m", 247_032_000, 216_312_000, 524_288, 5_000_000_000, 40),
    "s500m": StudentRow("s500m", 505_495_872, 462_487_872, 1_048_576, 10_000_000_000, 20),
    "s1b": StudentRow("s1b", 1_005_571_584, 956_419_584, 1_048_576, 20_000_000_000, 20),
}
STUDENTS = tuple(STUDENT_TABLE)


def base_tokens_rule(n_tot: int) -> int:
    """D_base = max(20 * N_tot, 5e9) (doc 00 §0.4); the table stores the locked rounded values."""
    return max(20 * n_tot, 5_000_000_000)


@dataclass(frozen=True)
class TeacherRow:
    name: str
    config: str                  # path relative to repo root
    family: str
    nominal_params: int
    exact_params: int
    non_embedding_params: int    # teacher_selection.md §3 (approximate; not in TeacherConfig)
    tensor_parallel: int
    variant: str = "instruct"


TEACHER_TABLE: dict[str, TeacherRow] = {
    "t0p5b": TeacherRow("t0p5b", "configs/teachers/qwen2.5-0.5b-instruct.yaml", "qwen2.5", 500_000_000, 494_032_768, 360_000_000, 1),
    "t1p5b": TeacherRow("t1p5b", "configs/teachers/qwen2.5-1.5b-instruct.yaml", "qwen2.5", 1_500_000_000, 1_543_714_304, 1_310_000_000, 1),
    "t3b": TeacherRow("t3b", "configs/teachers/qwen2.5-3b-instruct.yaml", "qwen2.5", 3_000_000_000, 3_085_938_688, 2_770_000_000, 1),
    "t7b": TeacherRow("t7b", "configs/teachers/qwen2.5-7b-instruct.yaml", "qwen2.5", 7_000_000_000, 7_615_616_512, 6_530_000_000, 1),
    "t14b": TeacherRow("t14b", "configs/teachers/qwen2.5-14b-instruct.yaml", "qwen2.5", 14_000_000_000, 14_770_033_664, 13_100_000_000, 1),
    "t32b": TeacherRow("t32b", "configs/teachers/qwen2.5-32b-instruct.yaml", "qwen2.5", 32_000_000_000, 32_763_876_352, 31_000_000_000, 1),
    "t72b": TeacherRow("t72b", "configs/teachers/qwen2.5-72b-instruct.yaml", "qwen2.5", 72_000_000_000, 72_706_203_648, 70_000_000_000, 2),
    # ablation-only teachers
    "llama8b": TeacherRow("llama8b", "configs/teachers/llama-3.1-8b-instruct.yaml", "llama3.1", 8_000_000_000, 8_030_261_248, 6_980_000_000, 1),
    "t7b_base": TeacherRow("t7b_base", "configs/teachers/qwen2.5-7b-base.yaml", "qwen2.5", 7_000_000_000, 7_615_616_512, 6_530_000_000, 1, "base"),
    "t32b_base": TeacherRow("t32b_base", "configs/teachers/qwen2.5-32b-base.yaml", "qwen2.5", 32_000_000_000, 32_763_876_352, 31_000_000_000, 1, "base"),
}
TEACHER_LADDER = ("t0p5b", "t1p5b", "t3b", "t7b", "t14b", "t32b", "t72b")   # 7-rung, in size order
PRIMARY_TEACHERS = TEACHER_LADDER[1:]                                        # 6-rung primary grid
PILOT_TEACHERS = ("t1p5b", "t7b", "t32b")
TIER2_TEACHERS = ("t1p5b", "t32b", "t72b")
DSWEEP_TEACHERS = ("t3b", "t14b", "t72b")

PILOT_STUDENTS = ("s025m", "s100m")
DSWEEP_STUDENTS = ("s100m", "s1b")
MAIN_BASE_RUN_STUDENTS = ("s250m", "s500m", "s1b")   # s025m/s100m reuse theta*(b=1) from the pilot

PILOT_BASE_SEEDS = (1, 2, 3)
PILOT_PHASE_SEEDS = (1, 2, 3)
TIER1_SEEDS = (1, 2, 3)
TIER2_EXTRA_SEEDS = (4, 5)
MAIN_C1_SEEDS = (1, 2, 3, 4, 5)
MAIN_C0_SEEDS = (1, 2, 3)
MAIN_C1B_SEEDS = (1, 2, 3)
DSWEEP_SEEDS = (1, 2)

GENERATION_CONFIGS = {
    "main_v1": "configs/generation/main_v1.yaml",
    "neutral_v1": "configs/generation/neutral_v1.yaml",
    "k4_v1": "configs/generation/k4_v1.yaml",
    "unfiltered_v1": "configs/generation/unfiltered_v1.yaml",
    "base3shot_v1": "configs/generation/base3shot_v1.yaml",
}
EVALUATION_CONFIG = "configs/evaluation/primary_v1.yaml"

CONTROL_SOURCES = {
    # arm -> (dataset source id, description)
    "matched_real": ("real.fineweb-edu-heldout", "C1: fresh held-out FineWeb-Edu (matched-real control)"),
    "human_instruct": ("human.stackexchange-qa-v1", "C1b: StackExchange Q&A (non-code), Alpaca-style markers"),
    "base_only": ("base.seen-shards", "C0: already-seen base shards (second epoch), disjoint from the replay shards"),
}


def student_config(name: str) -> str:
    return f"configs/students/{name}.yaml"


def training_config(name: str) -> str:
    return f"configs/training/train_{name}.yaml"


def d_syn_variant(d_syn: int, base: str = "eqtok") -> str:
    """'eqtok' for the main dose, 'eqtok-d150m' for a nested prefix."""
    if d_syn == D_SYN_MAIN:
        return base
    return f"{base}-d{d_syn // 1_000_000}m"


def dataset_ref(source: str, variant: str) -> str:
    return f"data/processed/phase2/{source}.{variant}.bin"


# ----------------------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """One run before composition. ``group`` and ``d_syn`` are bookkeeping (not written to YAML
    except through ``notes``); everything else maps 1:1 onto the experiment YAML."""

    study: str
    arm: str
    student: str
    seed_base: int
    seed_phase: int
    variant: str = "eqtok"
    teacher: Optional[str] = None
    generation: Optional[str] = None
    d_syn: int = D_SYN_MAIN                 # synthetic (or treatment) tokens; 0 for base runs
    is_base_run: bool = False               # arm base_only with no phase 2
    group: str = ""                         # e.g. 'pilot.branch', 'ablation.A3-replay'
    overrides: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    dataset_sha256: str = "PENDING"
    extra_generation_pool: Optional[tuple[str, str, int]] = None   # (teacher, generation, tokens) if a new pool is needed

    # --- derived -------------------------------------------------------------------
    @property
    def source(self) -> str:
        if self.arm == "synthetic":
            assert self.teacher and self.generation
            return f"{self.teacher}.{self.generation}"
        return CONTROL_SOURCES[self.arm][0]

    @property
    def dataset_ref(self) -> Optional[str]:
        if self.is_base_run:
            return None
        return dataset_ref(self.source, self.variant)

    def to_yaml(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "study": self.study,
            "arm": self.arm,
            "variant": self.variant,
            "student": student_config(self.student),
        }
        if self.teacher is not None:
            spec["teacher"] = TEACHER_TABLE[self.teacher].config
        if self.generation is not None:
            spec["generation"] = GENERATION_CONFIGS[self.generation]
        spec["training"] = training_config(self.student)
        spec["evaluation"] = EVALUATION_CONFIG
        if not self.is_base_run:
            spec["dataset_ref"] = self.dataset_ref
            spec["dataset_sha256"] = self.dataset_sha256
        spec["notes"] = f"group={self.group}; d_syn={self.d_syn}; {self.notes}".strip("; ")
        ov: dict[str, Any] = {"training.seed_base": self.seed_base, "training.seed_phase": self.seed_phase}
        if self.is_base_run:
            ov["training.synthetic_phase"] = None
        ov.update(self.overrides)
        spec["overrides"] = ov
        return spec


# ----------------------------------------------------------------------------------------
# Composition (mirrors synscale.config.loader.load_experiment, with a fragment cache)
# ----------------------------------------------------------------------------------------
_FRAGMENT_KEYS = ("student", "teacher", "generation", "training", "evaluation")


@lru_cache(maxsize=None)
def _fragment(rel_path: str, repo_root: str) -> dict[str, Any]:
    with open(Path(repo_root) / rel_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{rel_path}: top level must be a mapping")
    return data


def compose(spec_yaml: dict[str, Any], repo_root: Path | str = REPO_ROOT) -> ExperimentConfig:
    tree: dict[str, Any] = {}
    for key, val in spec_yaml.items():
        if key in _FRAGMENT_KEYS and isinstance(val, str):
            tree[key] = copy.deepcopy(_fragment(val, str(repo_root)))
        elif key == "overrides":
            continue
        else:
            tree[key] = val
    for dotted, value in (spec_yaml.get("overrides") or {}).items():
        _apply_override(tree, dotted, value)
    return ExperimentConfig.model_validate(tree)


def resolve(spec: ExperimentSpec, repo_root: Path | str = REPO_ROOT) -> ExperimentConfig:
    return compose(spec.to_yaml(), repo_root)


# ----------------------------------------------------------------------------------------
# Study builders
# ----------------------------------------------------------------------------------------
def _base_run(study: str, student: str, seed_base: int, group: str, variant: str = "eqtok",
              overrides: Optional[dict] = None, notes: str = "") -> ExperimentSpec:
    return ExperimentSpec(study=study, arm="base_only", student=student, seed_base=seed_base, seed_phase=1,
                          variant=variant, d_syn=0, is_base_run=True, group=group,
                          overrides=dict(overrides or {}), notes=f"base pretraining run -> theta*({student}, b={seed_base}). {notes}".strip())


def _branch(study: str, student: str, arm: str, seed_base: int, seed_phase: int, group: str,
            teacher: Optional[str] = None, generation: Optional[str] = None, variant: str = "eqtok",
            d_syn: int = D_SYN_MAIN, overrides: Optional[dict] = None, notes: str = "",
            extra_pool: Optional[tuple[str, str, int]] = None) -> ExperimentSpec:
    if arm == "synthetic":
        assert teacher is not None
        generation = generation or "main_v1"
    return ExperimentSpec(study=study, arm=arm, student=student, seed_base=seed_base, seed_phase=seed_phase,
                          variant=variant, teacher=teacher, generation=generation, d_syn=d_syn, group=group,
                          overrides=dict(overrides or {}), notes=notes, extra_generation_pool=extra_pool)


def pilot_specs() -> list[ExperimentSpec]:
    """Pilot (doc 00 §0.10): {s025m, s100m} x {C0, C1, C1b, t1p5b, t7b, t32b} x 3 base seeds x 3 phase
    seeds (= 54 branches per size) + 3 base runs per size + D-sweep at (s100m, t7b) x {75M,150M,300M} x 2 seeds."""
    out: list[ExperimentSpec] = []
    for s in PILOT_STUDENTS:
        for b in PILOT_BASE_SEEDS:
            out.append(_base_run("pilot", s, b, "pilot.base"))
        for b in PILOT_BASE_SEEDS:
            for k in PILOT_PHASE_SEEDS:
                out.append(_branch("pilot", s, "base_only", b, k, "pilot.C0", notes="C0 cooldown on already-seen base shards"))
                out.append(_branch("pilot", s, "matched_real", b, k, "pilot.C1", notes="C1 matched-real control"))
                out.append(_branch("pilot", s, "human_instruct", b, k, "pilot.C1b", notes="C1b human Q&A control"))
                for t in PILOT_TEACHERS:
                    out.append(_branch("pilot", s, "synthetic", b, k, "pilot.treatment", teacher=t))
    for d in DSWEEP_LEVELS:
        for k in DSWEEP_SEEDS:
            out.append(_branch("pilot", "s100m", "synthetic", 1, k, "pilot.dsweep", teacher="t7b",
                               variant=d_syn_variant(d), d_syn=d,
                               notes=f"pilot D-sweep: {d // 1_000_000}M synthetic (nested prefix) + fresh-web fill to 600M treatment tokens"))
    return out


def tier2_argmax_teachers(argmax: dict[str, str]) -> dict[str, list[str]]:
    """argmax +/- 1 rung per student on the 7-rung ladder (overlaps with the E-tier are dropped)."""
    out: dict[str, list[str]] = {}
    for s, t in argmax.items():
        if t not in TEACHER_LADDER:
            raise ValueError(f"unknown teacher '{t}' in tier-2 argmax for {s}")
        i = TEACHER_LADDER.index(t)
        cand = [TEACHER_LADDER[j] for j in (i - 1, i, i + 1) if 0 <= j < len(TEACHER_LADDER)]
        out[s] = [c for c in cand if c not in TIER2_TEACHERS]
    return out


def main_specs(tier2_argmax: Optional[dict[str, str]] = None) -> list[ExperimentSpec]:
    """Main study (doc 00 §0.10): 5 students x 7 teachers x 3 phase seeds (+2 seeds at t1p5b/t32b/t72b and,
    once known, at argmax +/- 1), C0 x3, C1b x3, C1 x5 per student, base runs for s250m/s500m/s1b (b=1)."""
    out: list[ExperimentSpec] = []
    argmax_extra = tier2_argmax_teachers(tier2_argmax or {})
    for s in STUDENTS:
        if s in MAIN_BASE_RUN_STUDENTS:
            out.append(_base_run("main", s, 1, "main.base"))
        for t in TEACHER_LADDER:
            seeds = list(TIER1_SEEDS)
            if t in TIER2_TEACHERS or t in argmax_extra.get(s, []):
                seeds += list(TIER2_EXTRA_SEEDS)
            for k in seeds:
                grp = "main.tier1" if k in TIER1_SEEDS else ("main.tier2-E" if t in TIER2_TEACHERS else "main.tier2-argmax")
                out.append(_branch("main", s, "synthetic", 1, k, grp, teacher=t,
                                   notes="0.5B extension rung (secondary 7-rung fit)" if t == "t0p5b" else ""))
        for k in MAIN_C0_SEEDS:
            out.append(_branch("main", s, "base_only", 1, k, "main.C0", notes="C0 cooldown on already-seen base shards"))
        for k in MAIN_C1B_SEEDS:
            out.append(_branch("main", s, "human_instruct", 1, k, "main.C1b", notes="C1b human Q&A control"))
        for k in MAIN_C1_SEEDS:
            out.append(_branch("main", s, "matched_real", 1, k, "main.C1", notes="C1 matched-real control (5 seeds: enters 7 contrasts per S)"))
    return out


def dsweep_specs() -> list[ExperimentSpec]:
    """D-sweep (doc 00 §0.9): {s100m, s1b} x {t3b, t14b, t72b} x {75M, 150M, 300M} x 2 seeds = 36 runs."""
    out: list[ExperimentSpec] = []
    for s in DSWEEP_STUDENTS:
        for t in DSWEEP_TEACHERS:
            for d in DSWEEP_LEVELS:
                for k in DSWEEP_SEEDS:
                    out.append(_branch("dsweep", s, "synthetic", 1, k, "dsweep", teacher=t, variant=d_syn_variant(d), d_syn=d,
                                       notes=f"{d // 1_000_000}M synthetic (nested prefix) + fresh-web fill to 600M treatment tokens; schedule unchanged"))
    return out


# Ablation list. Each entry: (id, description). The generator below implements them in this order.
ABLATIONS: list[tuple[str, str]] = [
    ("A1-lenmatch", "length-matched subsample (synthetic_data.md §3.4) at the pilot cells {s025m,s100m} x {t1p5b,t7b,t32b}, 2 seeds; variant 'lenmatch'"),
    ("A2-eqex", "equal-examples subsample (doc 00 §0.9 secondary) at the same 2x3 cells, 2 seeds; variant 'eqex'"),
    ("A3-replay", "replay fraction rho in {0, 0.5} (student_training.md X4) at {s025m,s100m} x {t1p5b,t7b,t32b, C1}, 1 seed; D_syn fixed at 600M so D2 = 600M (rho=0) / 1.2B (rho=0.5); variants 'eqtok-rho0', 'eqtok-rho50'"),
    ("A4-neutral", "neutral decoding (T 1.0, top-p 0.95, top-k off, rep 1.0) at s100m x {t1p5b,t72b}, D_syn 150M, 2 seeds; variant 'eqtok-d150m-neutral' + main-protocol reference cells at 150M"),
    ("A5-k4", "k = 4 samples per prompt at {s100m,s500m} x {t1p5b,t32b}, D_syn 150M, 2 seeds; variant 'eqtok-d150m-k4' + reference cells"),
    ("A6-instruct-vs-base", "base checkpoints Qwen2.5-7B/-32B with a fixed 3-shot template at s100m, D_syn 150M, 2 seeds; variant 'eqtok-d150m-base3shot' + reference cells"),
    ("A7-crossfamily", "cross-family anchor Llama-3.1-8B-Instruct at {s100m, s500m}, D_syn 600M, 2 seeds; variant 'eqtok' (compare with t7b main cells)"),
    ("A8-unfiltered", "minimal filtering (F1 + F11 only) at s100m x {t1p5b,t32b}, D_syn 600M, 2 seeds, same samples as main; variant 'unfilt'"),
    ("X2-overtrained-base", "s100m base at ~200 tok/param (20B tokens) then C1 + {t1p5b,t7b,t32b} x 2 seeds; variant 'eqtok-x2base' (base_phase.tokens override)"),
    ("X3-fineweb-base", "s100m base on plain FineWeb (non-Edu) then C1 + {t1p5b,t7b,t32b} x 2 seeds; variant 'eqtok-x3fineweb' [needs training.base_corpus field]"),
    ("X1-optionC", "Option C: from-scratch on D_base + D2 with the treatment file mixed uniformly from step 0, C1 + {t1p5b,t7b,t32b}; s025m x 3 seeds, s100m x 1 seed; variant 'optc' [needs training.mixing_mode field]"),
]

# Config fields the schema lacks; the corresponding ablations are represented by `variant` + `notes`.
MISSING_FIELDS = {
    "training.base_corpus": "id of the base corpus (fineweb-edu-v1 | fineweb-v1); needed by X3",
    "training.mixing_mode": "'annealing' (B') | 'uniform_from_start' (Option C); needed by X1",
    "teacher.non_embedding_params": "non-embedding parameter count (size covariate, teacher_selection.md §3)",
}


def ablation_specs() -> list[ExperimentSpec]:
    out: list[ExperimentSpec] = []
    A = "ablation"
    pilot_cells = [(s, t) for s in PILOT_STUDENTS for t in PILOT_TEACHERS]
    # A1 / A2
    for s, t in pilot_cells:
        for k in (1, 2):
            out.append(_branch(A, s, "synthetic", 1, k, "ablation.A1-lenmatch", teacher=t, variant="lenmatch",
                               notes="length-matched subsample: 12 log-spaced length bins x category, common support"))
            out.append(_branch(A, s, "synthetic", 1, k, "ablation.A2-eqex", teacher=t, variant="eqex",
                               notes="equal-examples subsample: n_c = min_T N_{T,c} per category"))
    # A3 replay fraction
    for s in PILOT_STUDENTS:
        for rho, tag, d2 in ((0.0, "rho0", 600_000_000), (0.5, "rho50", 1_200_000_000)):
            ov = {"training.synthetic_phase.replay_fraction": rho, "training.synthetic_phase.tokens": d2,
                  "training.synthetic_phase.decay_tokens": d2}
            note = f"replay rho={rho}: D_syn fixed at 600M, D2 = {d2 // 1_000_000}M (linear decay over D2); same treatment file as eqtok"
            for t in PILOT_TEACHERS:
                out.append(_branch(A, s, "synthetic", 1, 1, "ablation.A3-replay", teacher=t, variant=f"eqtok-{tag}", overrides=ov, notes=note))
            out.append(_branch(A, s, "matched_real", 1, 1, "ablation.A3-replay", variant=f"eqtok-{tag}", overrides=ov, notes=note))
    # reference cells at D_syn = 150M with the main protocol (nested prefix; no generation cost)
    ref150 = [("s100m", "t1p5b"), ("s500m", "t1p5b"), ("s100m", "t32b"), ("s500m", "t32b"), ("s100m", "t7b")]
    v150 = d_syn_variant(ABLATION_D_SYN)
    for s, t in ref150:
        for k in (1, 2):
            out.append(_branch(A, s, "synthetic", 1, k, "ablation.ref-d150m", teacher=t, variant=v150, d_syn=ABLATION_D_SYN,
                               notes="main-protocol reference at 150M for the generation-protocol ablations (t72b x s100m at 150M is in the D-sweep)"))
    # A4 neutral decoding
    for t in ("t1p5b", "t72b"):
        for k in (1, 2):
            out.append(_branch(A, "s100m", "synthetic", 1, k, "ablation.A4-neutral", teacher=t, generation="neutral_v1",
                               variant=f"{v150}-neutral", d_syn=ABLATION_D_SYN, extra_pool=(t, "neutral_v1", ABLATION_D_SYN),
                               notes="neutral decoding pool (150M); compare with the main-protocol cell at 150M"))
    # A5 k = 4
    for s in ("s100m", "s500m"):
        for t in ("t1p5b", "t32b"):
            for k in (1, 2):
                out.append(_branch(A, s, "synthetic", 1, k, "ablation.A5-k4", teacher=t, generation="k4_v1",
                                   variant=f"{v150}-k4", d_syn=ABLATION_D_SYN, extra_pool=(t, "k4_v1", ABLATION_D_SYN),
                                   notes="1/4 of the prompts x 4 samples (150M); compare with the main-protocol cell at 150M"))
    # A6 instruct vs base
    for t in ("t7b_base", "t32b_base"):
        for k in (1, 2):
            out.append(_branch(A, "s100m", "synthetic", 1, k, "ablation.A6-instruct-vs-base", teacher=t, generation="base3shot_v1",
                               variant=f"{v150}-base3shot", d_syn=ABLATION_D_SYN, extra_pool=(t, "base3shot_v1", ABLATION_D_SYN),
                               notes="base checkpoint, fixed 3-shot template (150M); compare with the instruct cell at 150M"))
    # A7 cross-family anchor
    for s in ("s100m", "s500m"):
        for k in (1, 2):
            out.append(_branch(A, s, "synthetic", 1, k, "ablation.A7-crossfamily", teacher="llama8b", variant="eqtok",
                               extra_pool=("llama8b", "main_v1", D_SYN_MAIN),
                               notes="cross-family anchor; bridge pair llama8b vs t7b (teacher_selection.md §4 F1)"))
    # A8 unfiltered
    for t in ("t1p5b", "t32b"):
        for k in (1, 2):
            out.append(_branch(A, "s100m", "synthetic", 1, k, "ablation.A8-unfiltered", teacher=t, generation="unfiltered_v1",
                               variant="unfilt", notes="same samples as main_v1 with filters F2-F10 disabled (no extra generation)"))
    # X2 over-trained base at s100m
    x2_tokens = 20_000_000_000
    x2_warm = min(int(0.05 * x2_tokens), 1000 * STUDENT_TABLE["s100m"].batch_tokens)
    x2_ov = {"training.base_phase.tokens": x2_tokens, "training.base_phase.warmup_tokens": x2_warm}
    out.append(_base_run(A, "s100m", 1, "ablation.X2-overtrained-base", variant="x2base", overrides=x2_ov,
                         notes="X2: over-trained base, ~200 tok/param (20B tokens)"))
    for k in (1, 2):
        for t in PILOT_TEACHERS:
            out.append(_branch(A, "s100m", "synthetic", 1, k, "ablation.X2-overtrained-base", teacher=t, variant="eqtok-x2base",
                               overrides=x2_ov, notes="branch from the over-trained base (base_phase.tokens = 20B)"))
        out.append(_branch(A, "s100m", "matched_real", 1, k, "ablation.X2-overtrained-base", variant="eqtok-x2base",
                           overrides=x2_ov, notes="C1 branch from the over-trained base"))
    # X3 plain-FineWeb base at s100m
    x3_note = "X3: base corpus = plain FineWeb (non-Edu) [needs training.base_corpus; represented by variant]"
    out.append(_base_run(A, "s100m", 1, "ablation.X3-fineweb-base", variant="x3fineweb", notes=x3_note))
    for k in (1, 2):
        for t in PILOT_TEACHERS:
            out.append(_branch(A, "s100m", "synthetic", 1, k, "ablation.X3-fineweb-base", teacher=t, variant="eqtok-x3fineweb", notes=x3_note))
        out.append(_branch(A, "s100m", "matched_real", 1, k, "ablation.X3-fineweb-base", variant="eqtok-x3fineweb", notes=x3_note))
    # X1 Option C (from scratch, treatment mixed uniformly from step 0)
    for s, seeds in (("s025m", (1, 2, 3)), ("s100m", (1,))):
        row = STUDENT_TABLE[s]
        total = row.d_base + D2_TOKENS
        ov = {"training.base_phase.tokens": total, "training.base_phase.decay_tokens": D2_TOKENS,
              "training.base_phase.lr_min_ratio": 0.0, "training.synthetic_phase.tokens": 0,
              "training.synthetic_phase.decay_tokens": 0}
        note = ("X1 Option C: single from-scratch run over D_base + D2 tokens (WSD, decay over the last 800M) with the "
                "600M treatment file mixed uniformly from step 0; no separate phase 2 [needs training.mixing_mode]")
        for k in seeds:
            for t in PILOT_TEACHERS:
                out.append(_branch(A, s, "synthetic", k, k, "ablation.X1-optionC", teacher=t, variant="optc", overrides=ov, notes=note))
            out.append(_branch(A, s, "matched_real", k, k, "ablation.X1-optionC", variant="optc", overrides=ov, notes=note))
    return out


def study_specs(study: str, tier2_argmax: Optional[dict[str, str]] = None) -> list[ExperimentSpec]:
    if study == "pilot":
        return pilot_specs()
    if study == "main":
        return main_specs(tier2_argmax)
    if study == "dsweep":
        return dsweep_specs()
    if study == "ablation":
        return ablation_specs()
    raise ValueError(study)


def all_specs(tier2_argmax: Optional[dict[str, str]] = None) -> dict[str, list[ExperimentSpec]]:
    return {s: study_specs(s, tier2_argmax) for s in STUDIES}


# ----------------------------------------------------------------------------------------
# Generation pools implied by a set of specs (used by the compute budget)
# ----------------------------------------------------------------------------------------
def generation_pools(specs: Iterable[ExperimentSpec]) -> dict[tuple[str, str], int]:
    """{(teacher, generation): delivered student tokens} required by the specs. Main-protocol pools deliver
    600M; nested-prefix ablation pools deliver their own target."""
    pools: dict[tuple[str, str], int] = {}
    for sp in specs:
        if sp.arm != "synthetic":
            continue
        if sp.extra_generation_pool is not None:
            t, g, tok = sp.extra_generation_pool
            pools[(t, g)] = max(pools.get((t, g), 0), tok)
        elif sp.generation == "unfiltered_v1":
            continue   # same samples as main_v1
        else:
            key = (sp.teacher, "main_v1")
            pools[key] = max(pools.get(key, 0), D_SYN_MAIN)
    return pools


# ----------------------------------------------------------------------------------------
# Counting / writing
# ----------------------------------------------------------------------------------------
def count_by(specs: Iterable[ExperimentSpec], key: str = "group") -> dict[str, int]:
    out: dict[str, int] = {}
    for sp in specs:
        k = getattr(sp, key)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summarize(specs: list[ExperimentSpec]) -> dict[str, Any]:
    return {
        "total": len(specs),
        "base_runs": sum(sp.is_base_run for sp in specs),
        "phase2_runs": sum(not sp.is_base_run for sp in specs),
        "by_arm": count_by(specs, "arm"),
        "by_group": count_by(specs, "group"),
        "by_student": count_by(specs, "student"),
        "pools": {f"{t}.{g}": tok for (t, g), tok in sorted(generation_pools(specs).items())},
    }


def _yaml_text(spec: ExperimentSpec, exp_id: str) -> str:
    header = (
        f"# {exp_id}\n"
        f"# Generated by scripts/make_matrix.py from synscale/config/matrix.py — DO NOT EDIT BY HAND.\n"
        f"# group: {spec.group}\n"
    )
    return header + yaml.safe_dump(spec.to_yaml(), sort_keys=False, width=120)


def write_study(study: str, out_root: Path | str = REPO_ROOT / "configs" / "experiments",
                repo_root: Path | str = REPO_ROOT, tier2_argmax: Optional[dict[str, str]] = None,
                dataset_shas: Optional[dict[str, str]] = None, clean: bool = True) -> dict[str, Any]:
    """Write one YAML per run into ``<out_root>/<study>/``. Deterministic and idempotent: the study
    directory is cleared of ``*.yaml`` first, and file names are the experiment IDs."""
    out_dir = Path(out_root) / study
    out_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for f in out_dir.glob("*.yaml"):
            f.unlink()
    specs = study_specs(study, tier2_argmax)
    ids: dict[str, ExperimentSpec] = {}
    for sp in specs:
        if dataset_shas and sp.dataset_ref in dataset_shas:
            sp.dataset_sha256 = dataset_shas[sp.dataset_ref]
        cfg = resolve(sp, repo_root)
        exp_id = experiment_id(cfg)
        if exp_id in ids:
            raise RuntimeError(f"duplicate experiment id {exp_id}: {sp} vs {ids[exp_id]}")
        ids[exp_id] = sp
        (out_dir / f"{exp_id}.yaml").write_text(_yaml_text(sp, exp_id), encoding="utf-8")
    summary = summarize(specs)
    summary["dir"] = str(out_dir)
    return summary


def load_tier2_argmax(path: Optional[str | Path]) -> Optional[dict[str, str]]:
    if not path:
        return None
    return json.loads(Path(path).read_text())
