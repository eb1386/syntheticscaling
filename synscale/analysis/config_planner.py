"""Plan and cost candidate experiment configurations across RunPod GPUs.

Answers the decision question: for each of several study scopes (which teachers, which
students, one vs two families, seeds) and a chosen GPU, what are the GPU-hours, the RunPod
dollar cost, the wall-clock, and a blunt read on how it lands for NeurIPS. Every number is a
transparent estimate from FLOPs + bandwidth scaling and is +-2x until measured on RunPod; the
point is relative comparison, not precision. Run: python -m synscale.analysis.config_planner
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- GPUs: VRAM GB, dense BF16 TFLOPS [approx], mem bandwidth GB/s, RunPod community $/hr (Sep 2026) ---
GPUS = {
    "rtx5080":  dict(vram=16,  tflops=112, bw=960,  usd=0.30),  # your card (owned); $ shown as electricity-ish
    "rtx4090":  dict(vram=24,  tflops=165, bw=1008, usd=0.69),
    "a40":      dict(vram=48,  tflops=150, bw=696,  usd=0.44),
    "a6000":    dict(vram=48,  tflops=155, bw=768,  usd=0.49),
    "l40s":     dict(vram=48,  tflops=362, bw=864,  usd=0.99),
    "a100_80":  dict(vram=80,  tflops=312, bw=1935, usd=1.39),
    "h100_80":  dict(vram=80,  tflops=990, bw=3350, usd=2.89),
    "h200":     dict(vram=141, tflops=990, bw=4800, usd=4.39),
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


def choose_precision(params):
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
    teachers: list          # param counts
    families: int
    seeds: int
    d_syn: int
    base_tok_per_param: int
    novelty: str
    neurips: str
    validation: bool = False

    def estimate(self):
        g = GPUS[self.gpu]
        d2 = int(self.d_syn / 0.75)
        teacher_tokens = self.d_syn / 0.75 / 0.9 * 1.25   # overgen + filter loss, per teacher
        base_h = sum(train_gpu_hours(s, STUDENTS[s] * self.base_tok_per_param, self.gpu) for s in self.students)
        # branches: (teachers*families + 3 controls) * seeds, per student
        n_branch = len(self.students) * (len(self.teachers) * self.families + 3) * self.seeds
        phase2_h = 0.0
        for s in self.students:
            per = train_gpu_hours(s, d2, self.gpu)
            phase2_h += per * (len(self.teachers) * self.families + 3) * self.seeds
        gen_h = 0.0; unfit = []
        for _ in range(self.families):
            for t in self.teachers:
                h = gen_gpu_hours(t, teacher_tokens, self.gpu, choose_precision(t))
                if h is None:
                    unfit.append(t)
                else:
                    gen_h += h
        eval_h = n_branch * 1.3 * 0.25            # ~15 min per eval, scaled loosely
        total = base_h + phase2_h + gen_h + eval_h
        if self.validation:
            total *= 1.12
        cost = total * g["usd"]
        return dict(base=base_h, gen=gen_h, phase2=phase2_h, eval=eval_h, total=total,
                    cost=cost, days=total / 24, n_branch=n_branch, unfit=unfit)


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
               "Borderline main track. Full teacher axis + validation; single-family confound remains.", True),
        Config("C4 main plausible (2 families)", "h100_80", S6, [0.5e9,1.5e9,3e9,7e9,14e9,32e9,70e9], 2, 3, 400_000_000, 20,
               "2 teacher families to 70B, students to 1B, validation",
               "Main-track plausible. Breaks size-vs-family confound; one crisp validated finding needed.", True),
        Config("C5 strongest (2 families, 5 seeds core)", "h100_80", S6, [0.5e9,1.5e9,3e9,7e9,14e9,32e9,70e9], 2, 5, 400_000_000, 20,
               "2 families to 70B, 5 seeds, students to 1B, validation",
               "Strongest single-GPU-rentable case. Tight CIs + confound broken + validation.", True),
    ]


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
    print()
    print("Prices: RunPod community cloud, Sep 2026 (H100 $2.89, A100-80 $1.39, L40S $0.99, A40 $0.44).")


if __name__ == "__main__":
    report()
