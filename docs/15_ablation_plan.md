# 15. Ablation and Cross-Check Plan (ranked)

Rule: main grid first; ablations only after the pilot passes P1–P6. Costs are PRO-6000-equivalent GPU-hours from the locked calculator's per-run rates (phase-2 branch ≈ 0.45 / 1.1 / 2.2 / 4.0 / 7.0 h at 25M / 100M / 250M / 500M / 1B; generation per doc 16). Value is judged by which pre-registered claim the ablation protects or extends. Tier 0 = run inside the pilot; Tier 1 = run with the main grid; Tier 2 = run after the main grid if budget allows; Tier 3 = follow-up work.

| Rank | Ablation | Tier | Cells (× seeds) | ≈ GPU-h | What it tests / protects | Decision it can change |
|---|---|---|---|---|---|---|
| 1 | Length-matched subsampling (doc 09 §3.4) | 0 + 1 | pilot cells; then {100M, 500M} × all T × 2 | 25 + 60 | Whether the teacher ordering is a verbosity artefact of equal-token budgeting (S6) | Primary claim survives or is re-framed as "on common length support" |
| 2 | Equal-examples budgeting | 1 | all pilot cells; {100M, 500M} × all T × 2 | 25 + 60 | The other direction of the tokens-vs-examples confound (S6) | Same |
| 3 | Replay fraction ρ ∈ {0, 0.5} (X4) | 0 | {25M, 100M} × {C1, T1.5B, T7B, T32B} × 2 | 25 | Whether forgetting artefacts depend on T; whether ρ = 0.25 hides or manufactures an ordering | ρ for the full grid |
| 4 | Phase-2 peak LR × {0.5, 0.25} (X5) | 0 | 100M × 3 T × 2 | 15 | Schedule sensitivity of the ordering | Phase-2 LR |
| 5 | Base LR sweep × {0.5, 1, 2} | 0 | base runs at 25M, 100M | 20 | Fits the LR exponent before 250M–1B bases | LR table |
| 6 | Unfiltered vs filtered pool | 1 | {1.5B, 32B} × 100M × 3 | 7 | Whether teacher-agnostic filters still act asymmetrically (yield differs by T) | Filter ruleset |
| 7 | Teacher instruct vs base (3-shot) | 2 | {7B, 32B} base × {100M, 500M} × 3; + 2 pools | 30 + 60 gen | Size effect vs post-training effect; addresses the Qwen2.5 "all sizes saw 72B data" limitation | Interpretation of H1–H3 |
| 8 | Cross-family anchor (Llama-3.1-8B-Instruct) | 2 | 1 pool; × {100M, 500M} × 3 | 30 + 20 gen | Bounds "family > size" (Pieler 2024); makes a flat T-axis interpretable | Whether any size claim is stated |
| 9 | Over-trained base at 100M (X2: 25B tokens, 250 tok/param) | 2 | 1 base (≈ 90 h) + {C1, 3 T} × 2 | 100 | External validity to SmolLM-like regimes; tokens/param confound of doc 00 §0.4 | Limitations wording; whether T* depends on base strength |
| 10 | Neutral decoding (T 1.0, top-p 0.95, no top-k, no repetition penalty) | 2 | {1.5B, 72B} × 100M × 3; 2 partial pools | 7 + 40 gen | Whether the size effect depends on the vendor-tuned decoding regime | Generation config for follow-ups |
| 11 | Within-teacher learnability quartiles (doc 09 §4.3) | 2 | 7B pool split by ℓ_S into quartiles at D/4; {25M, 250M} × 4 × 2 | 25 | Causal leverage for the learnability mediator (S4) | Mechanism claim |
| 12 | Verified-only arm (correctness-filtered MSR/STR) | 2 | {1.5B, 7B, 72B} × 100M × 3 | 10 | Whether correctness, not size, carries the gain (S5) | Mechanism claim |
| 13 | k = 4 samples per prompt at fixed D (¼ prompts) | 2 | {3B, 32B} × {100M, 500M} × 2 | 25 + 30 gen | Prompt diversity vs sample diversity (Bansal 2024 regime) | Pool design for follow-ups |
| 14 | Plain-FineWeb base at 100M (X3) | 2 | 1 base (≈ 15 h) + {C1, 3 T} × 2 | 25 | Whether an "educational" base shrinks the effect or changes the ordering | Base corpus for follow-ups |
| 15 | Response-only loss (prompt masked) | 2 | {7B} × {100M, 500M} × 2 | 10 | Sensitivity of Amendment A (doc 00 §0.7) | Token-accounting convention |
| 16 | C1c small-D human-instruction comparison | 1 | {100M, 500M} × {human, 7 T, web} × 3 at D = 20M | 60 | The only honest "synthetic vs human instruction data" statement | Discussion |
| 17 | Qwen3 non-thinking 3-rung replication of the pilot | 3 | {1.7B, 8B, 32B} × {25M, 100M} × 3; 3 pools | 40 + 150 gen | Whether distilled small teachers flatten the curve | Family generality section |
| 18 | Option-C from-scratch mixture at 25M/100M (X1) | 3 | {C1, 3 T} × 25M × 3; × 100M × 1 | 90 | Whether the ordering depends on early vs late exposure ("finetuner's fallacy") | Scope of the conclusion |
| 19 | 16k custom BPE family (25M, 100M only) | 3 | 2 bases + {C1, 3 T} × 3 | 40 | Embedding-share sensitivity of the smallest student | Follow-up |
| 20 | Curriculum ordering (easy→hard by ℓ_S) | 3 | 7B × {100M} × 3 | 4 | Whether ordering within the annealing phase matters | Follow-up |

Not planned: teacher confidence filtering as a training arm (it is a covariate; as a filter it changes the estimand to rejection sampling); multiple-teacher mixtures (different question); temperature sweeps beyond rank 10 (cost of regeneration).

Budget: Tier 0 ≈ 110 GPU-h (inside the pilot); Tier 1 ≈ 210 GPU-h; Tier 2 ≈ 330 GPU-h training + ≈ 150 GPU-h generation. Tier 2 is run in rank order until the budget in docs/16 is exhausted.
