"""Plan and cost candidate experiment configurations, and compare GPU providers.

Answers the decision question: for each of several study scopes (which teachers, which
students, one vs two families, seeds) and a chosen GPU, what are the GPU-hours, the dollar
cost, the wall-clock, and a blunt read on how it lands for NeurIPS. Also prints C4 across
providers (RunPod/TensorDock/Lambda/Vast/Modal) with a reliability read. Every number is a
transparent estimate from FLOPs + bandwidth scaling and is +-2x until measured in the pilot;
the point is relative comparison, not precision. Run: python -m synscale.analysis.config_planner
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- GPUs: VRAM GB, dense BF16 TFLOPS [approx], mem bandwidth GB/s, reference $/hr (Sep 2026) ---
# h100_80's reference rate is RunPod Community on-demand H100 SXM, the recommended primary provider.
GPUS = {
    "rtx5080":  dict(vram=16,  tflops=112, bw=960,  usd=0.30),  # a 5080 (owned); $ shown as electricity-ish
    "rtx4090":  dict(vram=24,  tflops=165, bw=1008, usd=0.69),
    "a40":      dict(vram=48,  tflops=150, bw=696,  usd=0.44),
    "a6000":    dict(vram=48,  tflops=155, bw=768,  usd=0.49),
    "l40s":     dict(vram=48,  tflops=362, bw=864,  usd=0.99),
    "a100_80":  dict(vram=80,  tflops=312, bw=1935, usd=1.39),
    "h100_80":  dict(vram=80,  tflops=990, bw=3350, usd=2.69),  # RunPod Community H100 SXM on-demand
    "h200":     dict(vram=141, tflops=990, bw=4800, usd=4.39),
}

# --- Per-GPU-hour H100-80GB rates by provider, with a blunt reliability read (Sep 2026, verified) ---
# For a multi-day training run needing a multi-GPU box + persistent storage + programmatic control.
PROVIDERS = {
    #                         $/GPU-h   reliability note
    "RunPod Secure":     dict(usd=2.99, note="RunPod-owned HW, 99% SLA. Network volumes survive pod death. Recommended reliable tier."),
    "RunPod Community":  dict(usd=2.69, note="RunPod marketplace + network volumes (durable). Cheaper; host-dependent uptime. Recommended default."),
    "TensorDock":        dict(usd=2.25, note="Curated marketplace, cheaper than RunPod. Less proven than Secure; storage model varies."),
    "Lambda Cloud":      dict(usd=3.99, note="Best SLA (99.9%) BUT H100 SXM is 8-GPU-only, no spot, and frequently SOLD OUT. Capacity is the risk."),
    "Vast.ai spot":      dict(usd=1.49, note="Cheapest. Peer-to-peer, NO uptime SLA, no cross-instance shared volume. Least reliable."),
    "Modal":             dict(usd=3.95, note="Serverless, per-second, preempts by default. Paradigm mismatch for multi-day SSH training."),
}

# student total params and per-size MFU assumption (tiny models under-use big GPUs)
STUDENTS = {"s025m":25.2e6, "s050m":51.9e6, "s100m":100.7e6, "s250m":251.7e6, "s500m":505e6, "s1b":1006e6}
MFU = {"s025m":0.15, "s050m":0.20, "s100m":0.28, "s250m":0.35, "s500m":0.40, "s1b":0.45}

# teacher decode tok/s baseline on a 5080 at int4 (memory-bound; scaled by bandwidth for other GPUs)
GEN_TOKS_5080_INT4 = {0.5e9:9000, 1.5e9:6000, 3e9:4000, 7e9:2200, 14e9:1200, 32e9:520, 70e9:230}


def _near(d, key):
    return d[min(d, key=lambda k: abs(k - key))]


def teacher_weight_gb(params, precision):
    return params / 1e9 * {"int4": 0.5, "fp8": 1.0, "bf16": 2.0}[precision]


def teacher_fit(params, gpu, precision):
    """Return n_cards needed (1 or 2) to serve a teacher, or None if it won't fit on 2."""
    usable = 0.92 * GPUS[gpu]["vram"] - 4.0  # KV + workspace headroom
    w = teacher_weight_gb(params, precision)
    if w <= usable:
        return 1
    if w <= (2 * 0.92 * GPUS[gpu]["vram"] - 8):
        return 2
    return None


def choose_precision(params, serve="auto"):
    # The ultra-optimized C-configs serve EVERY teacher single-card int4 (AWQ/awq_marlin), so no
    # teacher ever needs tensor-parallel=2 and every generation job is spot-safe. "auto" keeps the
    # older bf16-for-small assumption for comparison.
    if serve == "int4":
        return "int4"
    return "bf16" if params <= 7e9 else "int4"


def train_gpu_hours(student, tokens, gpu):
    N = STUDENTS[student]
    peak = GPUS[gpu]["tflops"] * 1e12 * MFU[student]
    return 6 * N * tokens / (peak * 3600)


def gen_gpu_hours(params, teacher_tokens, gpu, precision):
    cards = teacher_fit(params, gpu, precision)
    if cards is None:
        return None
    tok_s = _near(GEN_TOKS_5080_INT4, params) * (GPUS[gpu]["bw"] / 960)
    if precision == "bf16":
        tok_s *= 0.6  # bf16 weights move 4x the bytes of int4 -> slower decode (rough)
    return teacher_tokens / tok_s / 3600 * cards


@dataclass
class Config:
    name: str
    gpu: str
    students: list
    teachers: list          # family-A teacher param counts, run across ALL students
    families: int           # 1, or 2 when family_b is a partial screen (see family_b_*)
    seeds: int
    d_syn: int
    base_tok_per_param: int
    novelty: str
    neurips: str
    validation: bool = False
    teacher_serve: str = "auto"     # "int4" = single-card int4 for all teachers (spot-safe, ultra-opt)
    spot_usd: float = None          # marketplace/spot $/hr for this config's GPU, if it has one
    ondemand_usd: float = None      # reliable on-demand $/hr, if different from spot
    family_b_teachers: list = field(default_factory=list)  # partial 2nd family teacher params
    family_b_students: list = field(default_factory=list)  # only these students get family B

    def _cells(self):
        """(teacher_params, student) training cells across both families. Family B is PARTIAL:
        only family_b_teachers at family_b_students, which is what makes C4 affordable versus a
        full 2x grid."""
        cells = [(t, s) for s in self.students for t in self.teachers]
        for t in self.family_b_teachers:
            for s in self.family_b_students:
                cells.append((t, s))
        return cells

    def estimate(self):
        g = GPUS[self.gpu]
        d2 = int(self.d_syn / 0.75)
        teacher_tokens = self.d_syn / 0.75 / 0.9 * 1.25   # overgen + filter loss, per teacher
        base_by_student = {s: train_gpu_hours(s, STUDENTS[s] * self.base_tok_per_param, self.gpu)
                           for s in self.students}
        base_h = sum(base_by_student.values())
        cells = self._cells()
        # branch runs: one per teacher-student cell + 3 controls per student, times seeds
        n_branch = (len(cells) + 3 * len(self.students)) * self.seeds
        phase2_h = 0.0
        for t, s in cells:
            phase2_h += train_gpu_hours(s, d2, self.gpu) * self.seeds
        for s in self.students:                        # 3 controls per student
            phase2_h += train_gpu_hours(s, d2, self.gpu) * 3 * self.seeds
        # generation: each teacher produces ONE synthetic corpus from the shared prompt pool, reused
        # across every student it feeds (learnability q is measured per student, not regenerated), so
        # decode is billed once per teacher. Family B teachers still generate, just for fewer students.
        gen_h = 0.0; unfit = []; max_cards = 1
        for t in list(self.teachers) + list(self.family_b_teachers):
            prec = choose_precision(t, self.teacher_serve)
            cards = teacher_fit(t, self.gpu, prec)
            if cards is None:
                unfit.append(t); continue
            max_cards = max(max_cards, cards)
            gen_h += gen_gpu_hours(t, teacher_tokens, self.gpu, prec)
        eval_h = n_branch * 1.3 * 0.25            # ~15 min per eval, scaled loosely
        total = base_h + phase2_h + gen_h + eval_h
        if self.validation:
            total *= 1.12
        cost = total * g["usd"]
        # Hybrid plan: only the longest single base run (the 1B student) is worth an on-demand pod;
        # every other job is short, sharded, or checkpointed, so it rides spot. See fleet.py.
        ondemand_h = base_by_student.get("s1b", 0.0)
        spot_h = total - ondemand_h
        return dict(base=base_h, gen=gen_h, phase2=phase2_h, eval=eval_h, total=total,
                    cost=cost, days=total / 24, n_branch=n_branch, unfit=unfit,
                    max_cards=max_cards, ondemand_h=ondemand_h, spot_h=spot_h)

    def cost_lines(self):
        """(label, $) RunPod cost rows: all-Community / hybrid / all-Secure, when rates are set."""
        e = self.estimate()
        if self.spot_usd is None:
            return [("community on-demand", e["cost"])]
        od = self.ondemand_usd if self.ondemand_usd is not None else self.spot_usd
        return [
            ("RunPod Community (all jobs)", e["total"] * self.spot_usd),
            ("hybrid (1B base on Secure, rest Community)", e["spot_h"] * self.spot_usd + e["ondemand_h"] * od),
            ("RunPod Secure (all jobs)", e["total"] * od),
        ]


def default_configs():
    S4 = ["s025m", "s050m", "s100m", "s250m"]
    S6 = S4 + ["s500m", "s1b"]
    return [
        Config("C1 workshop (cheap)", "a40", S4, [0.5e9,1.5e9,3e9,7e9,14e9], 1, 3, 400_000_000, 20,
               "teachers to 14B, 1 family, 4 students; direction only",
               "Workshop / arXiv. Below main-track: capped teachers, one family."),
        Config("C2 workshop+ (mid)", "a100_80", S4+["s500m"], [0.5e9,1.5e9,3e9,7e9,14e9,32e9], 1, 3, 400_000_000, 20,
               "teachers to 32B, 1 family, 5 students",
               "Strong workshop / borderline main. Reaches the contested region but one family."),
        Config("C3 borderline main (1 family, full teacher axis)", "h100_80", S6, [0.5e9,1.5e9,3e9,7e9,14e9,32e9,70e9], 1, 3, 400_000_000, 20,
               "teachers to 70B, students to 1B, predictive validation",
               "Borderline main track. Full teacher axis + validation; single-family confound remains.", True,
               teacher_serve="int4", spot_usd=2.69, ondemand_usd=2.99),
        Config("C4 main plausible (Qwen full + Llama partial)", "h100_80", S6, [0.5e9,1.5e9,3e9,7e9,14e9,32e9,70e9], 2, 3, 400_000_000, 20,
               "Qwen 0.5-72B across all students + Llama 3/8/70B at 100M,1B (cross-family screen)",
               "Main-track plausible. Breaks size-vs-family confound; one crisp validated finding needed.", True,
               teacher_serve="int4", spot_usd=2.69, ondemand_usd=2.99,
               family_b_teachers=[3e9, 8e9, 70e9], family_b_students=["s100m", "s1b"]),
        Config("C5 strongest (partial 2nd family, 5 seeds core)", "h100_80", S6, [0.5e9,1.5e9,3e9,7e9,14e9,32e9,70e9], 2, 5, 400_000_000, 20,
               "Qwen full + Llama partial, 5 seeds on core cells, students to 1B, validation",
               "Strongest single-GPU-rentable case. Tight CIs + confound broken + validation.", True,
               teacher_serve="int4", spot_usd=2.69, ondemand_usd=2.99,
               family_b_teachers=[3e9, 8e9, 70e9], family_b_students=["s100m", "s1b"]),
    ]


USD_TO_CAD = 1.39   # Sep 2026 approx; for the Canadian-dollar figures in the README


def report():
    print(f"{'config':<44}{'gpu':<9}{'$/hr':>6}{'GPU-h':>8}{'wall(d,1gpu)':>13}{'cost $':>9}")
    print("-" * 92)
    rows = []
    for c in default_configs():
        e = c.estimate(); rows.append((c, e))
        print(f"{c.name:<44}{c.gpu:<9}{GPUS[c.gpu]['usd']:>6.2f}{e['total']:>8.0f}{e['days']:>13.1f}{e['cost']:>9.0f}"
              + (f"  [unfit teachers: {[int(t/1e9) for t in e['unfit']]}B]" if e['unfit'] else ""))
    print("-" * 92)
    print("Notes: GPU-h = billable card-hours; wall(d) assumes 1 rented GPU running serially")
    print("(rent 2-3 in parallel to cut wall-clock ~linearly for extra $0). +-2x until measured on RunPod.")
    print()
    for c, e in rows:
        print(f"* {c.name} [{c.gpu}] ~${e['cost']:,.0f}, {e['days']:.0f} days serial, {e['n_branch']} branch runs")
        print(f"    novelty: {c.novelty}")
        print(f"    NeurIPS: {c.neurips}")
        if c.spot_usd is not None:
            serve = "single-card int4 (spot-safe)" if c.teacher_serve == "int4" else "mixed precision"
            print(f"    serving: {serve}, max {e['max_cards']} card(s)/teacher; "
                  f"on-demand hours {e['ondemand_h']:.0f}, spot hours {e['spot_h']:.0f}")
            for label, usd in c.cost_lines():
                print(f"      {label:<40} ${usd:>6,.0f} USD  (~${usd*USD_TO_CAD:>6,.0f} CAD)")
    # C4 across providers, ranked cheapest-first, with the reliability read.
    c4 = next(c for c in default_configs() if c.name.startswith("C4"))
    h = c4.estimate()["total"]
    print()
    print(f"== C4 (~{h:.0f} GPU-h) across providers, cheapest first ==")
    for name, p in sorted(PROVIDERS.items(), key=lambda kv: kv[1]["usd"]):
        usd = h * p["usd"]
        print(f"  {name:<18} ${p['usd']:.2f}/GPU-h  ->  ${usd:>6,.0f} USD (~${usd*USD_TO_CAD:>6,.0f} CAD)   {p['note']}")
    print()
    print("RECOMMENDED: RunPod. Best reliability-per-dollar for a multi-day run — RunPod-owned Secure")
    print("HW (99% SLA) or Community, both with NETWORK VOLUMES that survive a pod crash so work is")
    print("never lost, plus real availability (Lambda sells out) and a CLI (runpodctl). The fleet's")
    print("checkpoint+requeue design absorbs RunPod's occasional pod failures. Vast is cheaper but has")
    print("no SLA and no shared volume; use it only to shave cost knowingly. See docs/28_runpod_runbook.md.")
    print("Hybrid = the single longest base run (1B) on Secure, every other job on Community; add ~25%")
    print("for preemption re-runs. CAD at ~1.39/USD. +-2x until measured in the pilot.")


if __name__ == "__main__":
    report()
