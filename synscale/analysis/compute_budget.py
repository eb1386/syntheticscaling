"""Compute-budget calculator for the single-RTX-5080 profiles.

Every number is derived from stated hardware constants and MFU/throughput assumptions
(marked [assume]); replace them with the pilot's measured values (docs/16). The point is
an honest wall-clock estimate for a solo researcher on one 16 GB card, plus knobs.

Training time uses per-size effective throughput calibrated to the doc-13 5080 table
(h per 1e9 training tokens). Generation time uses an int4 memory-bound decode model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from synscale.config.profiles import Profile, get_profile

# --- hardware constants (RTX 5080, [fact]/[assume]) -----------------------------------
BF16_TFLOPS_5080 = 112.6          # dense, FP32-accumulate [fact: Puget/WareDB]
MEM_BW_5080 = 960e9               # bytes/s [derived 256-bit x 30 Gbps]
ELEC_PRICE = 0.15                 # $/kWh [assume]
PUE = 1.2
TDP_5080 = 360.0                  # W
POWER_UTIL = {"train": 0.85, "gen": 0.80, "eval": 0.55}

# training effective throughput: h per 1e9 tokens (from doc 13 Table B.1, 5080 column)
H_PER_1E9_TOK = {"s025m": 3.01, "s050m": 4.20, "s100m": 6.56, "s250m": 12.79, "s500m": 25.51, "s1b": 47.0}
# generation decode throughput on one 5080, int4 (awq_marlin), batched [assume, +-2x]
GEN_TOK_PER_S_INT4 = {5e8: 9000.0, 1.5e9: 6000.0, 3e9: 4000.0, 7e9: 2200.0, 14e9: 1200.0}

# generation accounting
STUDENT_PER_TEACHER_TOKEN = 1.12   # student/teacher tokenizer ratio [assume]
FILTER_YIELD = 0.90                # retained fraction [assume]


def _gen_tok_per_s(params: int) -> float:
    # nearest tabulated size
    key = min(GEN_TOK_PER_S_INT4, key=lambda k: abs(k - params))
    return GEN_TOK_PER_S_INT4[key]


@dataclass
class Line:
    stage: str
    detail: str
    gpu_hours: float
    kwh: float


def _energy(gpu_hours: float, kind: str) -> float:
    return gpu_hours * (TDP_5080 * POWER_UTIL[kind] + 100.0) / 1000.0 * PUE


def base_lines(p: Profile) -> list[Line]:
    out = []
    for s in p.students:
        h1 = H_PER_1E9_TOK[s] * (p.base_tokens[s] / 1e9)
        n = p.base_seeds[s]
        gh = h1 * n
        out.append(Line("base", f"{s} x{n} seed(s), {p.base_tokens[s]/1e9:.0f}B tok", gh, _energy(gh, "train")))
    return out


def generation_lines(p: Profile) -> list[Line]:
    out = []
    for t in p.teachers:
        teacher_tokens = p.d_syn / STUDENT_PER_TEACHER_TOKEN / FILTER_YIELD * 1.25  # overgen
        gh = teacher_tokens / _gen_tok_per_s(t.nominal_params) / 3600.0
        out.append(Line("generation", f"{t.name} ({t.nominal_params/1e9:.1f}B int4), ~{teacher_tokens/1e6:.0f}M tok", gh, _energy(gh, "gen")))
    return out


def _branch_hours(student: str, d2: int) -> float:
    return H_PER_1E9_TOK[student] * (d2 / 1e9)


def branch_counts(p: Profile) -> dict[str, int]:
    """Number of phase-2 branch runs per student (tier-1 + tier-2 + controls + C1 extra + dsweep)."""
    counts = {}
    n_teachers = len(p.teachers)
    for s in p.students:
        # tier-1: every teacher + every control at phase_seeds
        n = (n_teachers + len(p.controls)) * p.phase_seeds
        # tier-2: extra seeds at tier2 teachers
        n += len(p.tier2_teachers) * p.tier2_seeds
        # C1 extra seeds beyond tier-1 (matched_real is one of controls, already counted phase_seeds)
        n += max(0, p.c1_seeds - p.phase_seeds)
        counts[s] = n
    # dsweep on one student
    if p.dsweep_student:
        extra_levels = [d for d in p.dsweep_levels if d != p.d_syn]
        counts[p.dsweep_student] = counts.get(p.dsweep_student, 0) + len(p.dsweep_teachers) * len(extra_levels) * p.dsweep_seeds
    return counts


def phase2_lines(p: Profile) -> list[Line]:
    out = []
    counts = branch_counts(p)
    for s in p.students:
        n = counts[s]
        gh = n * _branch_hours(s, p.d2_tokens)
        out.append(Line("phase2", f"{s}: {n} branches x {p.d2_tokens/1e6:.0f}M tok", gh, _energy(gh, "train")))
    return out


def eval_lines(p: Profile) -> list[Line]:
    # minutes per full primary+continuous eval, per doc 13 B.3 (5080)
    mins = {"s025m": 12, "s050m": 13, "s100m": 14, "s250m": 18, "s500m": 32, "s1b": 48}
    counts = branch_counts(p)
    out = []
    for s in p.students:
        # each branch evaluated once at final + base checkpoints (final + a few); ~1.3x branches
        n_evals = int(round(counts[s] * 1.3)) + p.base_seeds[s]
        gh = n_evals * mins[s] / 60.0
        out.append(Line("eval", f"{s}: {n_evals} evaluations", gh, _energy(gh, "eval")))
    return out


def budget(profile_name: str) -> dict:
    p = get_profile(profile_name)
    lines = base_lines(p) + generation_lines(p) + phase2_lines(p) + eval_lines(p)
    by_stage: dict[str, float] = {}
    kwh = 0.0
    for ln in lines:
        by_stage[ln.stage] = by_stage.get(ln.stage, 0.0) + ln.gpu_hours
        kwh += ln.kwh
    total_h = sum(by_stage.values())
    # single-card serial wall-clock == total gpu-hours; add 1.25x for overhead/restarts
    return {
        "profile": p.name,
        "lines": lines,
        "by_stage": by_stage,
        "total_gpu_hours": total_h,
        "wall_days_serial": total_h / 24.0,
        "wall_days_with_overhead": total_h * 1.25 / 24.0,
        "energy_kwh": kwh,
        "energy_usd": kwh * ELEC_PRICE,
        "branch_counts": branch_counts(p),
    }


def format_budget(b: dict) -> str:
    L = [f"=== Compute budget: profile '{b['profile']}' on ONE RTX 5080 ==="]
    L.append(f"{'stage':<12}{'detail':<48}{'GPU-h':>8}")
    for ln in b["lines"]:
        L.append(f"{ln.stage:<12}{ln.detail[:47]:<48}{ln.gpu_hours:>8.1f}")
    L.append("-" * 68)
    for st, h in b["by_stage"].items():
        L.append(f"{st:<12}{'subtotal':<48}{h:>8.1f}")
    L.append("-" * 68)
    L.append(f"{'TOTAL':<12}{'GPU-hours (== wall-clock on one card)':<48}{b['total_gpu_hours']:>8.1f}")
    L.append(f"wall-clock serial:        {b['wall_days_serial']:.1f} days")
    L.append(f"wall-clock w/ 1.25x overhead: {b['wall_days_with_overhead']:.1f} days")
    L.append(f"energy: {b['energy_kwh']:.0f} kWh (~${b['energy_usd']:.0f} at $0.15/kWh, PUE 1.2)")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="scaling5080",
                    choices=["c4", "scaling5080", "scaling5080_fast", "local5080", "local5080_fast", "smoke"])
    a = ap.parse_args()
    print(format_budget(budget(a.profile)))
