# 00. Locked Methodology (Decision Record v1.0)

> **ACTIVE STUDY: C4 (docs/24).** The project now runs, by default, the two-family,
> loss-based scaling study defined in `docs/24_novel_experiment.md` on rented H100s
> (`c4` profile). It reframes the question from teacher-size selection to *what governs*
> synthetic-data scaling (teacher size, family, or a student-referenced data-quality
> index q), adds a second teacher family, out-of-sample predictive validation, and a
> compute-allocation frontier. This document and docs 04-21 remain the methodological
> and cluster-scale record; where C4 differs, docs/24 and docs/22 win.

Status: **locked for the pilot**; every item below supersedes any conflicting text in the workstream documents (docs 04–13), which are retained as the derivation record. Each decision names the alternatives that were considered, the workstream disagreement it resolves, and its status tag: **[choice]** methodological choice, **[assumption]** number to be measured in the pilot, **[hypothesis]**, **[fact]** established/literature-supported.

## 0.1 Title and framing

**[choice]** Working title: *"Which Teacher for Which Student? Teacher-Size Effects of Synthetic Annealing Data for 25M–1B Language Models."* Fallback if no interior optimum is found: *"Teacher-Size Transfer Curves for Synthetic-Data Continued Training of 25M–1B Language Models."*

The original title, "Synthetic Scaling Laws: Optimal Teacher Model Size…", is not defensible: (i) "scaling laws" and "distillation scaling laws" are already claimed by Qin et al. 2025 and Busbridge et al. 2025 for other channels; (ii) five student sizes and six teacher sizes in one family cannot establish a law, only a descriptive relation with wide confidence bands (doc 12 §5.4: the exponent's CI half-width is ≈ ±0.35 in the best realistic case); (iii) "optimal teacher size" presupposes an interior optimum that four prior generator-size studies (Kang 2025, Niklaus 2026, BeyondWeb 2025, Pieler 2024) suggest may not exist above ~3B. The method is named precisely: *sequence-level distillation via synthetic annealing data*.

## 0.2 Research question and hypotheses (final form)

**Question.** For a decoder-only student of size S ∈ {25M, 100M, 250M, 500M, 1B} trained to a fixed real-data base state θ*(S), which teacher size T ∈ {1.5B, 3B, 7B, 14B, 32B, 72B} (one instruct family, one generation protocol, equal student-tokenizer tokens) yields the largest improvement when D_syn = 600M tokens of its synthetic data replace an equal number of fresh web tokens in the student's annealing phase, and how does that answer change with S and with generation cost?

Hypotheses are stated as falsifiable tests in doc 01. Prior expectation from the literature (doc 04 §4): H1 (strict monotonicity) is likely false; H2 (saturation) is likely true somewhere in 3–14B; H3 (intermediate beats largest) is plausible for 25M–100M; H4 (a predictable T*(S) relation) is exploratory and may be unidentifiable; H5 (cost-optimal ≠ performance-optimal) is nearly certain and the contribution is its magnitude as a function of S.

## 0.3 Student-training design — Option B′ (WSD-branching annealing)

**[choice]** Adopt the training workstream's B′ (doc 10 §1.4). Per student size, one base run on FineWeb-Edu with a warmup-stable schedule, stopped at the stable-phase checkpoint θ*(S) (weights + AdamW state + loader position). Every condition is a branch from θ*(S) running an identical linear-decay-to-zero phase over D₂ = 800M tokens = 600M treatment + 200M replay of already-seen base shards (ρ = 0.25, same shards and slot pattern for every condition). Conditions differ in exactly one file.

Rejected: Option A (pure synthetic pretraining: floor effects at every size; a Chinchilla-sized synthetic corpus for the 1B student would require 20B tokens per teacher, generation-infeasible); Option C (real+synthetic from scratch: ≈ 24× the base-training cost; teacher effect entangled with early-vs-late exposure). C is retained as cross-check X1 at 25M/100M.

**Identification statement.** τ_S(T) = E[Y(Π₂(θ*(S), F_T)) − Y(Π₂(θ*(S), F_real))] is the average causal effect of replacing 600M fresh web tokens with 600M synthetic tokens from teacher T in the annealing phase of a size-S student; Δ_S(T,T′) = τ_S(T) − τ_S(T′) is the total effect of teacher identity, interpreted as teacher size within the fixed family. The estimand is conditional on the base corpus, D_base(S), D₂, ρ, the decay schedule, the student tokenizer, and the family.

## 0.4 Base-token rule (resolves disagreement between docs 10 and 11)

**[choice]** D_base(S) = max(20·N_tot(S), 5B) tokens, drawn as **nested prefixes of one seeded shuffle** of the FineWeb-Edu train split:

| S | N_tot | 20·N_tot | D_base | tokens/param | D₂/D_base |
|---|---|---|---|---|---|
| 25M | 24.9M | 0.5B | **5B** | 201 | 16% |
| 100M | 100.1M | 2.0B | **5B** | 50 | 16% |
| 250M | 247.0M | 4.9B | **5B** | 20 | 16% |
| 500M | 505.5M | 10.1B | **10B** | 20 | 8% |
| 1B | 1005.6M | 20.1B | **20B** | 20 | 4% |

Reasons: (i) the benchmark workstream shows that Chinchilla budgets leave the 25M and 100M students at chance on nearly every accuracy benchmark (Pythia-70M at 1–2B tokens; doc 11 §1.3), whereas 5B tokens lifts SciQ, ARC-Easy, PIQA and LAMBADA off the floor; (ii) the extra compute is ≈ 20 PRO-6000 GPU-hours (small models are cheap); (iii) the phase-2 fraction spread across S falls from 26× (100% at 25M vs 3.8% at 1B under pure Chinchilla) to 4× (16% vs 4%), which weakens the relative-dose confound on the S×T interaction; (iv) the three small students see byte-identical base data, and the two large students see supersets, so base *content* is not confounded with S. Cost: tokens/param varies from 201 to 20, so "training regime" is confounded with S; this is disclosed, and cross-check X2 (over-trained 100M base at 200 tok/param) measures its effect on the teacher ranking.

## 0.5 Students (doc 07)

**[choice]** Llama-2 32k SentencePiece tokenizer (V = 32,000, byte fallback, per-digit splitting; ungated copies in llama2.c/TinyLlama; SHA-256 pinned), tied embeddings at every size, GQA 3:1 at every size, head_dim 64, d_ff = 8/3·d exactly, RMSNorm, RoPE θ = 10⁴, no biases, context 2048, bf16 autocast with fp32 master/Adam, init N(0, 0.02) with 1/√(2L) residual scaling. Final shapes (all within ±1.2% of target; totals exact from `synscale/models/param_count.py`):

| name | L | d_model | heads | kv | d_ff | total | non-embedding |
|---|---|---|---|---|---|---|---|
| s025m | 8 | 384 | 6 | 2 | 1024 | 24,877,440 | 12,589,440 |
| s100m | 12 | 768 | 12 | 4 | 2048 | 100,092,672 | 75,516,672 |
| s250m | 22 | 960 | 15 | 5 | 2560 | 247,032,000 | 216,312,000 |
| s500m | 24 | 1344 | 21 | 7 | 3584 | 505,495,872 | 462,487,872 |
| s1b | 38 | 1536 | 24 | 8 | 4096 | 1,005,571,584 | 956,419,584 |

Resolved: the training workstream's preference for a custom 32k BPE is rejected in favour of a fixed public artefact (no tokenizer-training step; MobileLLM/TinyLlama precedent); the Llama 2 Community License permits research use of the tokenizer. Both parameter counts are reported everywhere; the S axis in regressions uses N_tot (nominal) with N_ne as a sensitivity axis.

## 0.6 Teachers (doc 08)

**[choice]** Qwen2.5-Instruct at 1.5B, 3B, 7B, 14B, 32B, 72B (HF revisions pinned to commit SHAs; safetensors hashed), **plus 0.5B as a pre-registered seventh rung** (cheap; gives a teacher smaller than the 500M/1B students and anchors the low end of any T*(S) fit). The 30-cell 6-rung grid is the primary analysis; the 0.5B rung is reported as an extension and included in a secondary 7-rung fit. All teachers served in **BF16 weights and BF16 KV cache** (72B: tensor-parallel 2 across both 96 GB cards). FP8-for-all is the pre-registered fallback only if measured 72B throughput is < 50% of the model (doc 13 §D.2 option A), with a 32B BF16-vs-FP8 bridging cell.

Known limitations to state: the "1B" rung is 1.5B; the 3B rung carries the Qwen Research License; all Qwen2.5 sizes were exposed to 72B-synthesised data during their own training (uniform across sizes, may attenuate the gradient); teacher capability is nearly collinear with log T within one family, so a measured capability covariate (each teacher evaluated on the student suite with the same harness) is analysed alongside log T. A cross-family anchor (Llama-3.1-8B-Instruct) at two student sizes is a tier-2 ablation.

## 0.7 Synthetic data (doc 09), with two amendments

Pool v1.0 = 4.0M frozen, versioned, decontaminated prompts (categories IF 15 / SAQA 15 / CS 15 / SCI 20 / MSR 15 / EXP 12 / STR 8 %; sources: templated taxonomy 41%, FineWeb-Edu passage-seeded 27%, fixed non-Qwen prompt-writer 24%, human prompts 8%). One system prompt, one ChatML template (hash-checked across sizes), one decoding config (T 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, max 1024 new tokens, n = 1), teacher-agnostic filters F1–F12 with no quality judge, union-drop contamination removal, common-random-numbers prompt order, per-teacher stopping rule at 1.25× quota.

**Amendment A (resolves doc 09 §1.7 vs doc 10 §2.9): full-sequence loss; D_syn counts all tokens of the rendered example (prompt + response + markers).** Reasons: the control arms (C1 fresh web, C1b human Q&A) have no prompt/response split, so a response-only objective would make phase 2 a *different objective* in treatment vs control; full-sequence counting keeps trained tokens = processed tokens = optimizer steps identical across all arms. The prompt-token fraction per teacher (expected 15–35%) is reported as a covariate, and the length-matched and equal-example secondary analyses bound its effect. Category token quotas in the subsampler are re-derived on total tokens.

**Amendment B (resolves doc 10 §2.9 vs doc 11 §8.4): student-visible rendering uses Alpaca-style plain-text markers, not the harness template.** Each example is one document: `<s>### Instruction:\n{prompt}\n\n### Response:\n{response}</s>`. The lm-eval zero-shot prompts use `Question:/Answer:`, so format familiarity with the evaluation template is not part of the treatment; C1b (human StackExchange Q&A) and the held-out human-instruction NLL set are rendered with the same markers.

## 0.8 Controls and arms per student size

| Arm | Treatment file (600M tokens) | Purpose |
|---|---|---|
| C0 | already-seen base shards (second epoch) | "nothing new" cooldown reference |
| C1 | fresh held-out FineWeb-Edu | matched-real control; the primary comparator |
| C1b | StackExchange Q&A (non-code sites), same markers | separates Q&A *format* from synthetic *provenance* |
| C1c (100M, 500M only) | 20M human instruction tokens (Dolly + OASST1 + No Robots) + 580M fresh web, vs each teacher subsampled to 20M | the only honest human-instruction comparison |
| T_x, x ∈ {0.5, 1.5, 3, 7, 14, 32, 72}B | teacher pool subsampled to 600M | treatment cells |

## 0.9 Budget control and D-sweep

Primary: equal student-tokenizer tokens (600M) with seeded per-category quotas. Secondary: equal examples. Ablation: length-matched (stratified by category × 12 log-spaced length bins). **Iso-cost analysis is co-reported** (literature workstream's objection accepted): P(S,T,D) is interpolated from the D-sweep and evaluated at D(T) = B/c(T) for a grid of budgets B, using measured cost c(T).

**D-sweep [choice]:** at fixed D₂ = 800M, the synthetic fraction is varied: D_syn ∈ {75M, 150M, 300M} (600M is the main grid) with the remaining treatment slots filled from fresh web shards — schedule and step count unchanged, no extra generation. Cells: S ∈ {100M, 1B} × T ∈ {3B, 14B, 72B} × 3 levels × 2 seeds = 36 runs; the nested subsets share prompts across levels. The three small students at fixed 600M are at ≈ 16% synthetic fraction; the D-sweep's 300M point at 1B (1.5%) and 75M at 100M (1.5%) provide the dose-matched comparison the statistics workstream requires.

## 0.10 Seeds and replication (docs 10 §2.10, 12 §3.4)

| Stage | Base seeds b | Phase seeds s | Cells |
|---|---|---|---|
| Pilot (25M, 100M) | 3 | 3 | {C0, C1, C1b, T1.5B, T7B, T32B} → 54 branches per size; nested variance decomposition σ²_base, σ²_base×T, σ²_order |
| Full tier 1 | 1 (b=1 from pilot at 25M/100M) | 3 | all 35 treatment cells, C0, C1b |
| Full tier 2 | 1 | +2 (=5) | T ∈ {1.5B, 32B, 72B} per S; argmax±1 per S (confirmation seeds, never used for selection); C1 at every S |
| D-sweep | 1 | 2 | 36 runs |

A phase seed re-draws data order and slot permutation only (the subsample at D_syn = 600M is the whole delivered pool, so seeds do not re-draw it; at D-sweep levels the nested prefix is fixed). Seed = data-order variance is therefore a *lower bound* on run-to-run variance; the pilot's base replicates supply σ²_base and σ²_base×T, which inflate the cross-S standard errors (doc 12 §0). Decision rule for tier-1 r after the pilot: doc 12 §3.4.

## 0.11 Evaluation (doc 11)

Primary discrete outcome: chance-corrected mean P_cc over a floor-gated set identical across S. Tier A (always): SciQ (acc), ARC-Easy (acc_norm), PIQA (acc_norm), LAMBADA-OpenAI (acc, ppl). Tier B (enter iff C0 clears max(5 pts, 3·SE_item) at the smallest retained S): OpenBookQA, SocialIQA, HellaSwag. Zero-shot, lm-eval 0.4.13 pinned, no chat template, seed 1234, batch 64, bf16. Co-primary continuous outcomes: held-out NLL on human-written instruction/QA text (Dolly/OASST1/NQ-open, response tokens, rendered with the training markers) and per-task loss (bits-per-byte of the correct continuation). Secondary: ARC-Challenge, Winogrande, BoolQ (vs majority), MMLU-continuation (STEM, full), CommonsenseQA-cloze, BLiMP, TruthfulQA-MC2 (report), GSM8K (1B, gated), IFEval-lite (report), TriviaQA-3k, 5-shot ICL probe, 200-item format probe, teacher-validation NLL matrix (diagnostic). Final checkpoint only. Per-teacher contamination audit (n-gram, paraphrase, benchmark-style density) mandatory before any teacher comparison is reported.

## 0.12 Statistics (doc 12)

Model sequence M0–M4 (+ saturating M2sat), per-S variance, HC3 SEs, isotonic and saturation tests, per-S T* classification {interior / saturating / monotone-unsaturated / flat} with the four-condition interior rule and tier-2 confirmation seeds, δ = 1.0 chance-corrected point (sensitivity 0.5), Holm within pre-defined contrast families, seed CI primary + paired-item bootstrap robustness, α of log T* vs log S reported descriptively only. Pilot go/no-go rules P1–P6 and redesign triggers are pre-registered (doc 12 §7).

## 0.13 Cost and Synthetic Transfer Efficiency (doc 13)

C_gen(T) measured as GPU-seconds, kWh, cloud-equivalent $ (reference $2.00/GPU-h RTX PRO 6000; sensitivity $0.95 owned), and FLOPs, per delivered student token, from 10k-prompt chunks. **STE(S,T) = [M(S,T) − M(S,C1)] / (c_$(T)·D_syn)** (gain over matched-real control per generation dollar) is the primary efficiency metric; marginal efficiency between adjacent teachers and the price-of-a-point curve T$(λ) = argmax_T[G₁ − λ·C] are the selection tools; H5 is supported at S if the CI of the last-upgrade marginal efficiency into T* lies below the cheapest teacher's average efficiency.

## 0.14 Compute envelope (doc 16 recomputes under these decisions)

Reference hardware 2× RTX PRO 6000 Blackwell (96 GB) + 2× RTX 5080 (16 GB). Under the locked design the full study is ≈ 1,400–1,750 PRO-6000-equivalent GPU-hours (generation ≈ 400–750 depending on measured 72B throughput; base runs ≈ 270; phase-2 ≈ 620; evaluation ≈ 100 on 5080s), i.e. 5–6 weeks wall-clock with the 72B pool and the 1B base run on the critical path. A `local-5080` profile (one 16 GB card) supports students ≤ 250M and teachers ≤ 3B locally, with 7B+ generation rented.

## 0.15 Things this study cannot establish (to be stated in the paper)

Generality beyond Qwen2.5-Instruct; separation of size from post-training recipe; separation of size from capability within one family; a scaling *law* for T*(S); anything about model collapse; anything about from-scratch synthetic pretraining (Option A) or early-mixing (Option C) beyond the X1 cross-check.
