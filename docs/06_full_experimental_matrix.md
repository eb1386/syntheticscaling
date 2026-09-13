# 06. Full Experimental Matrix

Two matrices are defined: the **cluster** design (docs/00, docs/13; 5×7 grid, teachers to 72B)
and the **single-5080** design implemented here (`synscale/config/profiles.py`). This document
gives the single-5080 matrix; regenerate counts with `python scripts/make_matrix.py --profile <p>`.

## 06.1 local5080 (complete study on one card)

- **Students (3):** s025m, s100m, s250m (base D = 5B tokens each, nested prefixes of one shuffle).
- **Teachers (5):** Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, served AWQ int4 (uniform precision).
- **Arms per cell:** synthetic (per teacher) + controls C0 (θ*), C1 (matched real), C1b (human Q&A).
- **D_syn:** 300M treatment tokens; D₂ = 400M (300M treatment + 100M replay, ρ=0.25).
- **Seeds:** tier-1 = 3 phase seeds/cell; tier-2 = +2 at teachers {1.5B, 14B}; C1 at 5; base seeds = 3 at 25M, 1 elsewhere.
- **D-sweep:** s100m × {1.5B, 14B} × {75M, 150M, 300M} × 2 seeds (nested prefixes).

Run counts (`make_matrix`): **5 base runs + 89 branch runs = 94 training runs**, plus 94 evaluations.
Branches by condition: C1 15, C1b 9, and per teacher {0.5B 9, 1.5B 19, 3B 9, 7B 9, 14B 19}.

## 06.2 local5080_fast (~1 week)

2 students (25M, 100M) × 4 teachers (≤7B), D_syn 150M, D₂ 200M, 3B base 3B tokens. **4 base + 48 branches = 52 runs.**

## 06.3 Experiment IDs and arms

Experiment IDs follow `docs/18` (`<study>.<student>.<source>.d<Dsyn>.<variant>.s<seed>-<hash8>`), e.g.
`main.s100m.t14b.d300000000.eqtok.s1-<hash>`, `main.s100m.real.d300000000.eqtok.s1-<hash>`,
`main.s025m.base.d0.-.s1-<hash>`. Arm→condition: base_only→C0, matched_real→C1, human_instruct→C1b, synthetic→teacher id.

## 06.4 Ablations

Ranked in `docs/15`. On one card the affordable ones are the length-matched and equal-examples
re-analyses (no new training — they re-subsample existing pools), the 3B BF16-vs-int4 bridge
(one extra pool + a few branches), and the replay-ρ and phase-2-LR checks at 25M/100M.
