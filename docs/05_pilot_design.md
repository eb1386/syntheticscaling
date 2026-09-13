# 05. Pilot Design and Go/No-Go Rules

> **The `scaling5080_fast` profile IS the pilot** (user lock item 5): a ~1-week, 3-student x
> 4-teacher, 2-seed run whose purpose is to measure throughput, seed variance, OOM behaviour,
> filter yield, loss-curve smoothness and effect sizes on the real 5080. Only after it passes
> the go/no-go rules below do we launch the full 4-student `scaling5080` grid for the paper.

The pilot validates the pipeline and estimates the noise before committing weeks of GPU time.
On one 5080 the pilot is the first slice of the `local5080` run (it reuses the same code and
configs), plus the `smoke` check.

## 05.1 Sequence

1. `make micro` — CPU, minutes: proves every stage runs end to end (data → pool → generate → base → branches → eval → index → analyze). Already verified in this repo.
2. `make smoke` — GPU, ~1–3 h: real Qwen int4 teachers, a 3k-prompt pool, a 25M student, a tiny token budget. Confirms vLLM sm_120 kernels, the real tokenizer, real benchmark loading, and VRAM headroom.
3. First slice of `local5080`: the 25M student, teachers {0.5B, 3B, 14B}, C1, 3 seeds, D-sweep at 100M×{1.5B,14B}. ~2–3 days.

## 05.2 What the pilot measures (calibrates the estimates in docs/16)

| Quantity | Used to fix |
|---|---|
| int4 decode tok/s per teacher on the 5080 | generation wall-clock (`compute_budget.GEN_TOK_PER_S_INT4`) |
| training MFU / h-per-1B-tokens per student | base + phase-2 wall-clock (`H_PER_1E9_TOK`) |
| student/teacher token ratio r, filter yield y | pool sizing (eq. 1, doc 09) and generation budget |
| seed SD of P_cc and of held-out NLL | seed count for the full grid (doc 12 §3.4) |
| floor status of each benchmark at 25M (C0) | the retained set K (doc 11 floor gate) |
| synthetic-vs-C1 gain and teacher spread at 100M | whether the T question is answerable at this D_syn |

## 05.3 Pre-registered go / no-go rules (doc 12 §7, adapted to the local arms)

Proceed to the full `local5080` grid only if all hold:

| Rule | Criterion | If it fails |
|---|---|---|
| P1 detectable synthetic effect | at ≥1 student, mean(synthetic − C1) 95% CI excludes 0 on P_cc **or** held-out NLL | review the generation pipeline; raise D_syn |
| P2 noise | σ̂_agg upper 95% ≤ 0.9 chance-corrected points | switch primary to NLL/task-loss; floor-gate harder; then raise D_syn (not seeds) |
| P3 floor | ≥4 benchmarks retained at 25M, ≥6 at 100M | drop 25M to exploratory (NLL-only); grid becomes 2×5 |
| P4 teacher spread | max−min cell mean at 100M ≥ 1.0 point | the grid cannot map a curve at this D_syn; raise D_syn |
| P5 dose | P_cc at (100M, mid teacher) non-decreasing across the D-sweep and 300M−75M CI excludes 0 | lower D_syn if already saturated |
| P6 integrity | re-eval reproduces; decontamination clean; NLL on human text not worse than C1 by >5% rel. | fix before proceeding |

## 05.4 Deliverables of the pilot

Calibrated `compute_budget` constants; the measured seed SD and the chosen seed tier; the floor-gated benchmark set K; the pilot cell table and the synthetic-vs-C1 contrast with CIs; a decision record appended to `docs/00`.
