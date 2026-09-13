# 3 Methods

This section describes the locked design (decision record v1.0). Every number is a design parameter, an assumption the pilot will measure ([assumption], [estimate], [hypothesis]) or a placeholder; no results are reported. Items still to be verified before the pre-registration is frozen are marked [CHECK].

## 3.1 Overview and identification statement

We measure how teacher size affects the value of synthetic text used as annealing data for a small student. Five decoder-only students, S ∈ {25M, 100M, 250M, 500M, 1B}, are each pretrained once on FineWeb-Edu with a warmup-stable schedule and stopped at the stable-phase checkpoint θ*(S). Every condition is a branch from θ*(S) that runs an identical linear-decay phase over D₂ = 800M tokens: 600M from a condition-specific treatment file plus 200M replay of already-seen base shards. Treatment files come from seven rungs of one teacher family (Qwen2.5-Instruct, 0.5B to 72B) generated under one frozen protocol, or from control corpora (§3.6). Within a size, conditions differ in exactly one file. We call the method sequence-level distillation via synthetic annealing data; it is the mid-training pattern of SmolLM2 (Allal et al. 2025), MiniCPM (Hu et al. 2024) and OLMo 2 (OLMo team 2025) with the teacher as the only manipulated factor.

**Estimand.** Let Π₂(θ, F) be the phase-2 procedure applied to checkpoint θ with treatment file F, and Y an outcome (§3.7). With θ*(S) and Π₂ fixed,

$$\tau_S(T) = \mathbb{E}\big[\,Y(\Pi_2(\theta^*(S), F_T)) - Y(\Pi_2(\theta^*(S), F_{\mathrm{real}}))\,\big]$$

is the average causal effect of replacing 600M fresh web tokens with 600M synthetic tokens from teacher T in the annealing phase of a size-S student, where F_real is the matched-real control C1 and the expectation is over phase seeds (and base seeds at the pilot sizes). The contrast

$$\Delta_S(T, T') = \tau_S(T) - \tau_S(T')$$

is the total effect of teacher identity through every channel (content, style, length, correctness, regurgitation), interpreted as teacher size because family, post-training recipe, prompts, decoding and filters are held fixed; channel decompositions are secondary analyses. The estimand is conditional on the base corpus, D_base(S), D₂, ρ, the decay schedule, the student tokenizer and the teacher family. T*(S) is the argmax of τ_S(T) over the tested grid, examined only within that grid; we do not claim a scaling law, and from-scratch or early-mixing regimes are probed only by cross-check X1.

## 3.2 Student family

All students share the Llama-2 block (Touvron et al. 2023): pre-norm RMSNorm (ε = 1e-5), RoPE (θ = 10⁴), SwiGLU with d_ff = 8d/3 exactly, no biases, no dropout, grouped-query attention with n_kv = n_heads/3 and head_dim 64 at every size, tied embeddings, context 2048. One rule at every size avoids an architectural discontinuity confounded with S. Shapes come from an exhaustive search with d = 192k within an aspect band d/L ∈ [32, 64], where loss is flat at fixed non-embedding size (Kaplan et al. 2020); all totals are within ±1.2% of target (Table 1; `synscale/models/param_count.py`).

**Table 1. Student family** (V = 32,000, tied embeddings, GQA 3:1, head_dim 64, context 2048).

| Name | L | d_model | Heads | KV heads | d_ff | N_tot | N_ne | Embedding share | d/L |
|---|---|---|---|---|---|---|---|---|---|
| s025m | 8 | 384 | 6 | 2 | 1024 | 24,877,440 | 12,589,440 | 49.4% | 48 |
| s100m | 12 | 768 | 12 | 4 | 2048 | 100,092,672 | 75,516,672 | 24.6% | 64 |
| s250m | 22 | 960 | 15 | 5 | 2560 | 247,032,000 | 216,312,000 | 12.4% | 44 |
| s500m | 24 | 1344 | 21 | 7 | 3584 | 505,495,872 | 462,487,872 | 8.5% | 56 |
| s1b | 38 | 1536 | 24 | 8 | 4096 | 1,005,571,584 | 956,419,584 | 4.9% | 40 |

Parameter counts follow from

$$N_{\mathrm{ne}} = L\,(2d^2 + 2d\,d_{kv} + 3d\,d_{ff} + 2d) + d,\qquad N_{\mathrm{tot}} = N_{\mathrm{ne}} + V d,$$

with d_kv = n_kv · 64 = d/3, V = 32,000 and the linear terms the RMSNorm gains; under the family's rules the per-layer term is (32/3)d² + 2d. Both counts are reported everywhere; regressions use N_tot as the nominal S axis and N_ne as a sensitivity axis.

**Tokenizer.** Llama-2 SentencePiece BPE (V = 32,000, byte fallback, per-digit splitting) from an ungated copy (llama2.c, TinyLlama), SHA-256 pinned; it is the tokenizer of MobileLLM and TinyLlama and requires no training step. It is the largest mainstream vocabulary that leaves a non-degenerate 25M model (49% embedding share), and tying is forced by the same constraint (Press and Wolf 2017; Liu et al. 2024). All token budgets are counted in this tokenizer.

**Initialisation and precision.** N(0, 0.02) for all matrices and the embedding, with the residual-writing projections (W_o, W_down) scaled by 1/√(2L) (Radford et al. 2019) and norm gains 1. bf16 autocast with fp32 master weights, gradients and AdamW moments; norm statistics, softmax and loss in fp32 through a fused or chunked cross-entropy. No FP8 for students. Activation checkpointing only for the 500M model on a 16 GB card; every branch of a size runs on the same GPU model.

## 3.3 Teachers

**Table 2. Teacher checkpoints** (Qwen2.5-Instruct; Qwen Team 2024). Non-embedding counts are from the model cards [CHECK to the last digit]. All rungs are served by vLLM in BF16 weights and BF16 KV cache on RTX PRO 6000 (96 GB).

| Rung | Checkpoint | Params total / non-emb. | Layers | Tied emb. | Licence | Serving | Role |
|---|---|---|---|---|---|---|---|
| 0.5B | Qwen/Qwen2.5-0.5B-Instruct | 0.49B / 0.36B | 24 | yes | Apache-2.0 | BF16, TP 1 | pre-registered extension (7th rung) |
| 1.5B | Qwen/Qwen2.5-1.5B-Instruct | 1.54B / 1.31B | 28 | yes | Apache-2.0 | BF16, TP 1 | core |
| 3B | Qwen/Qwen2.5-3B-Instruct | 3.09B / 2.77B | 36 | yes | Qwen Research License | BF16, TP 1 | core |
| 7B | Qwen/Qwen2.5-7B-Instruct | 7.61B / 6.53B | 28 | no | Apache-2.0 | BF16, TP 1 | core |
| 14B | Qwen/Qwen2.5-14B-Instruct | 14.7B / 13.1B | 48 | no | Apache-2.0 | BF16, TP 1 | core |
| 32B | Qwen/Qwen2.5-32B-Instruct | 32.5B / 31.0B | 64 | no | Apache-2.0 | BF16, TP 1 (TP 2 if throughput favours it; recorded) | core |
| 72B | Qwen/Qwen2.5-72B-Instruct | 72.7B / 70.0B | 80 | no | Qwen License (≤ 100M MAU) | BF16, TP 2 | core |

Varying size within one family holds the post-training recipe and tokenizer fixed; families whose small members are distilled from their large members would make a small rung a proxy for the large one and flatten curves. The 30-cell, 6-rung grid is primary; the 0.5B rung, smaller than the 500M and 1B students, is an extension and enters a secondary 7-rung fit. Repositories are pinned to commit SHAs, safetensors shards hashed, and tokenizer, chat template and generation config stored with the data. Uniform BF16 avoids the size-correlated perturbation of quantisation. FP8 for all rungs (W8A8, per-channel weights, per-token dynamic activations) is the pre-registered fallback only if measured 72B BF16 throughput falls below 50% of the cost model, with a 32B BF16-vs-FP8 bridging cell on the same prompts. `max_model_len` is identical across rungs; YaRN is never enabled.

**Capability covariate.** Within one family capability is nearly collinear with log T, so each teacher is evaluated once, at serving precision, on the student primary suite with the same harness (§3.7), giving z_T = P_cc(T); the regressions of §3.8 are refitted with z_T (M1z to M3z), compared by AICc, and T* is also reported in capability units. Limitations: the smallest core rung is 1.5B; the 3B rung is research-licensed [CHECK: output-use clauses of the 3B and 72B licences]; all Qwen2.5 sizes saw 72B-synthesised data in their own training, which is uniform across sizes but may attenuate the gradient. A cross-family anchor (Llama-3.1-8B-Instruct) at two student sizes is a tier-2 ablation.

## 3.4 Synthetic data generation

**Invariance principle.** The training file of teacher T is

$$F_T = \sigma\Big(\Phi\big(G_T(P;\ \theta_{\mathrm{dec}}, s_{\mathrm{sys}}, s_{\mathrm{suffix}}, \tau_{\mathrm{chat}})\big);\ D_{\mathrm{syn}}, \mathrm{seed}\Big),$$

where only the generator G_T is teacher-indexed. The prompt pool P (prompts, categories, order), system prompt, suffixes, ChatML template (extracted from the 7B tokenizer, hashed, asserted identical across rungs), decoding config, filters Φ, subsampler σ, student tokenizer and rendering are frozen, versioned and hashed before generation. Whatever cannot be made teacher-independent (yield, truncation, refusal, contamination, length, correctness) is measured per teacher and reported as a covariate: it is part of the treatment.

**Prompt pool.** Pool v1.0 is 4.0M frozen, decontaminated prompts (Table 3), sized so that the least verbose, lowest-yield teacher fills the 600M quota with a 1.25× margin provided retained length ≥ 220 student tokens per example and yield ≥ 0.85 [assumption]. Sources: templates over a 12-domain, ≈ 2,000-leaf taxonomy (41%); FineWeb-Edu passage-seeded prompts in the Cosmopedia style (27%; Ben Allal et al. 2024) from a SEED split disjoint from the base and control splits; a fixed prompt-writer outside the teacher family (24%; Mistral-Small-3.1-24B-Instruct [CHECK licence], fallback Gemma-3-27B-it) with persona conditioning (Ge et al. 2024); and harvested human prompts (8%; WildChat-1M, oasst2, Dolly-15k, No Robots). FLAN, UltraChat, OpenHermes, LMSYS-Chat and Alpaca are excluded (benchmark templates, a third model's style, or licence). Per-leaf caps (≤ 0.2%), audience conditioning and MinHash deduplication (< 0.5% near-duplicates) guard diversity. The pool order is a seeded stratified interleave, so any prefix is a stratified sample and every teacher processes a prefix of the same sequence (common random numbers). Seed passages are teacher-only background; students never see them.

**Table 3. Prompt pool composition.** Sources: (a) templated taxonomy, (b) passage-seeded, (c) human, (d) prompt-writer. Token estimates assume ≈ 1.3 Qwen tokens per word and r ≈ 1.15 student tokens per teacher token [assumption]; token shares are re-derived from realised lengths in the pilot.

| Category | Prompt share | Source mix (%) | Target length (words) | Est. student tokens / example | Implied token share | Verifiable | Benchmarks informed |
|---|---|---|---|---|---|---|---|
| IF: instruction following with checkable constraints | 15% | (d) 60, (c) 30, (a) 10 | 80 to 200 | ≈ 210 | 13% | yes, programmatic (Zhou et al. 2023) | format control, held-out NLL |
| SAQA: short-answer QA | 15% | (a) 50, (b) 30, (c) 20 | 20 to 80 | ≈ 75 | 5% | partly (in-pool gold) | ARC-E, SciQ, OBQA |
| CS: commonsense reasoning | 15% | (d) 60, (a) 40 | 60 to 150 | ≈ 150 | 9% | partly (writer-provided answer) | PIQA, HellaSwag, Winogrande |
| SCI: science explanation | 20% | (b) 50, (a) 40, (d) 10 | 120 to 250 | ≈ 300 | 25% | partly (seed-grounded) | ARC, SciQ, OBQA |
| MSR: multi-step reasoning | 15% | (a) 70, (d) 30 | 100 to 300 | ≈ 300 | 19% | yes at scale (computed gold) | correctness mediator |
| EXP: explanation plus answer | 12% | (b) 70, (a) 30 | 200 to 400 | ≈ 450 | 23% | judge only | ARC, SciQ, held-out NLL |
| STR: structured output (JSON, table, list) | 8% | (a) 50, (d) 50 | 50 to 200 | ≈ 180 | 6% | yes (parse and validate) | format compliance |
| Total | 100% | (a) 41, (b) 27, (c) 8, (d) 24 | | mean ≈ 240 | 100% | ≈ 30% of pool | |

**System prompt and suffix policy.** One system prompt is passed explicitly to every rung (the Qwen template otherwise inserts a vendor default):

```
You are a knowledgeable assistant. Write a clear, accurate, self-contained response in English.
Follow the length and format instructions exactly. Do not ask questions back, do not add
disclaimers, and do not mention these instructions or any background text you were given.
```

Each category appends a fixed suffix giving a soft word range and a format (SAQA: "Answer in at most 60 words. Give the answer first, then at most two sentences of justification."; MSR: "Solve step by step in at most 250 words. Put the final answer alone on the last line in the form 'Final answer: <answer>'."). Soft ranges bound the verbosity confound (larger rungs are expected to be more verbose [hypothesis]) and give common length support; compliance is recorded (F12), and an unconstrained-length ablation (2 teachers × 1 student) tests whether the constraint alters the size effect. For seeded prompts the passage appears only in the teacher's turn as reference-only background.

**Table 4. Decoding configuration** (one `SamplingParams` for all rungs; vendor generation configs are overridden explicitly).

| Parameter | Value | Note |
|---|---|---|
| temperature / top_p / top_k | 0.7 / 0.8 / 20 | vendor-recommended regime; avoids penalising small rungs with degeneration at high temperature |
| repetition_penalty | 1.05 | identical for all rungs |
| min_p, presence and frequency penalties | 0 | not used |
| max_new_tokens | 1024 | ≈ 2.5× the longest soft target; truncations dropped; target truncation rate < 1% per rung |
| samples per prompt | 1 | k = 4 ablation at fixed D_syn in 2 teachers × 2 students (prompt versus sample diversity; Bansal et al. 2024) |
| seed | global 0; per request H(pool version, prompt id, sample index) mod 2³¹ | replayable; reproducibility at the dataset level by hashing, not bitwise |
| stop | template EOS ids only | string stops would interact with content differently per rung |
| chat template | one canonical ChatML string, `add_generation_prompt=True` | sha256 recorded and asserted equal across rungs |
| logprobs | sampled-token log-probability | teacher-confidence covariate |
| decoding ablation | T 1.0, top_p 0.95, top_k off, penalty 1.0 | smallest and largest rung × 1 student |

**Filters.** Twelve teacher-agnostic rules with fixed thresholds (Table 5); no model-based quality or correctness judge, because judge bias is teacher-correlated (Zheng et al. 2023; Panickssery et al. 2024) and a judge filter would change the estimand to the compute-matched regime of Bansal et al. (2024). Wrong answers on verifiable items are retained: the correctness rate is part of the treatment.

**Table 5. Filter set Φ** (applied per teacher except F11).

| Rule | Criterion | Action |
|---|---|---|
| F1 truncation | `finish_reason = length` | drop |
| F2 too short | < 8 student tokens (SAQA) or < 32 (others), or whitespace only | drop |
| F3 language | fastText p(en) < 0.65, or CJK character ratio > 1% | drop |
| F4 degeneration | Gopher repetition thresholds (Rae et al. 2021) | drop |
| F5 exact duplicate | sha1 of normalised response, within teacher | keep first in pool order |
| F6 near duplicate | MinHash, 5-gram shingles, ≈ Jaccard 0.72, within teacher and category (Lee et al. 2022) | keep lowest H(prompt id) |
| F7 refusal or self-identification | fixed regex list in the first 300 characters; "Qwen", "Alibaba", "Tongyi" anywhere | drop |
| F8 passage reference | regex, seeded prompts only | drop |
| F9 format validity | STR: JSON schema, table and list validation; IF: IFEval-style constraint checkers | drop on hard STR failure; IF failures recorded only |
| F10 answer line | "Final answer:" present and parseable (MSR) | drop |
| F11 benchmark contamination | decontamination stages 1 and 2 | union-drop across teachers |
| F12 length compliance | word count within the suffix range | recorded only |

**Decontamination.** Targets are the test and validation splits of every benchmark in §3.7 (and of any later addition), the held-out NLL sets and the V-set, normalised (lowercase, NFKC, punctuation stripped). Stage 1: word 8-gram overlap (Dubey et al. 2024), stricter than the GPT-3 13-gram rule (Brown et al. 2020) because items are short, with whole-item matching under 8 words. Stage 2: embedding retrieval (bge-base-en-v1.5, cosine ≥ 0.85 [assumption; calibrated on known paraphrases]) followed by a rephrase judgement by a fixed non-Qwen judge (Yang et al. 2023). Both stages apply to seed passages and prompts before freezing, and to responses per teacher; because memorisation grows with size (Carlini et al. 2022), a prompt flagged for any teacher is dropped for all (union-drop), keeping prompt sets identical. Training-split overlap is reported, not removed; a 13-gram sensitivity run is reported.

**Budget control and subsampling.** The primary control is equal student-tokenizer tokens, D_syn = 600M for every teacher, which holds exposure, optimizer steps and compute constant and is hardware-independent. Under Amendment A of the design record a token is any token of the rendered example (prompt, response, markers), so trained tokens equal processed tokens in every arm, controls included. Per-category quotas are seeded and identical across teachers:

```
inputs: retained examples E_T = {(prompt_id, category c, len)}, len = rendered tokens
        (prompt + response + markers, student tokenizer); token shares s_c (fixed
        after the pilot); D_syn; global seed s_0 (same for all T); tol = 1024 tokens
for each category c:
    q_c   = round(s_c * D_syn)                        # identical quota for all T
    order = sort(E_T[c], key = H64(s_0, prompt_id))   # teacher-independent priority
    acc = 0; S_c = []
    for e in order:
        if acc + e.len <= q_c + tol: S_c.append(e); acc += e.len
        if acc >= q_c: break
    assert q_c - tol <= acc <= q_c + tol
    record N_{T,c} = |S_c|, tokens_{T,c} = acc
F_T = union over c of S_c, shuffled with seed s_0     # same shuffle rule for all T
```

Per-category tokens agree across teachers to ±1,024; example counts N_{T,c} differ with verbosity and are reported. Hash priority maximises prompt overlap between teachers, and D-sweep subsets are prefixes of the same order, hence nested. Secondary sets: equal examples (the first min_T N_{T,c} per category) and a length-matched ablation (category × 12 log-spaced length bins on the intersection support with ≥ 20% slack; target KS distance between teachers < 0.01), run in the pilot cells and a 2 × 3 subgrid.

**Rendering and stopping rule.** Each retained example is one document with Alpaca-style markers, `<s>### Instruction:\n{prompt}\n\n### Response:\n{response}</s>`, so familiarity with the evaluation template (`Question:/Answer:`) is not part of the treatment; C1b and the instruction NLL set use the same markers. Generation runs in frozen pool order in shards of 50,000 prompts, written in pool order; teacher T stops at the first shard boundary at which retained tokens in every category reach 1.25 × q_c. Expected teacher tokens per rung are ≈ 0.77B, roughly independent of verbosity [estimate]. Per-teacher covariates (Table 6) are reported for every rung and category.

**Table 6. Per-teacher covariates reported.**

| Group | Quantities |
|---|---|
| Pipeline | prompts offered; yield; removals per filter F1 to F12; truncation, refusal, non-English, CJK, exact- and near-duplicate rates; contamination flag rate; prompt-set overlap between rung pairs |
| Length | mean, SD, quantiles and suffix compliance before and after σ; KS distance to the pooled distribution; example count N_T; prompt-token fraction (expected 15 to 35% [assumption]); teacher-to-student token ratio r_T |
| Confidence | mean sampled-token log-probability |
| Correctness | pass rate on the ≈ 1.1M in-pool programmatically verifiable items (MSR, STR, IF); pass@1 and majority-vote correctness on a frozen 2,000-item V-set (arithmetic, conversions, string tasks, templated word problems after Mirzadeh et al. 2024, structured output, team-written factual and science items; k = 4 samples), scored programmatically or by a fixed non-Qwen judge (Llama-3.3-70B-Instruct, temperature 0, blinded; 200-item human audit, κ ≥ 0.75 required) |

## 3.5 Training protocol

**Base phase.** One base run per size on the FineWeb-Edu `sample-100BT` train shards (ODC-By; Penedo et al. 2024), decontaminated as in §3.4, with D_base(S) = max(20 · N_tot, 5B) drawn as nested prefixes of one seeded shuffle: the three small students see byte-identical data and the two large ones supersets. The 5B floor exists because Chinchilla budgets (Hoffmann et al. 2022) leave 25M and 100M students at chance on nearly every accuracy benchmark; the resulting tokens-per-parameter spread (201 to 20) is a disclosed confound measured by cross-check X2. Schedule: linear warmup to η_peak, then constant until D_base, when θ*(S) (weights, AdamW moments, loader position, RNG state) is saved (Table 7).

**Table 7. Base phase.** Batch is in sequences of 2048 tokens; steps are ⌊D/B⌋ [CHECK: exact counts fixed by the build script]; warmup is min(5% of base steps, 1000).

| S | N_tot | 20 · N_tot | D_base | Batch (seq = tokens) | Base steps | Warmup | η_peak | Tokens / param | D₂ / D_base |
|---|---|---|---|---|---|---|---|---|---|
| 25M | 24.9M | 0.5B | 5B | 128 = 0.26M | ≈ 19,070 | 954 | 2.0e-3 | 201 | 16% |
| 100M | 100.1M | 2.0B | 5B | 256 = 0.52M | ≈ 9,540 | 477 | 1.2e-3 | 50 | 16% |
| 250M | 247.0M | 4.9B | 5B | 256 = 0.52M | ≈ 9,540 | 477 | 8e-4 | 20 | 16% |
| 500M | 505.5M | 10.1B | 10B | 512 = 1.05M | ≈ 9,540 | 477 | 6e-4 | 20 | 8% |
| 1B | 1005.6M | 20.1B | 20B | 512 = 1.05M | ≈ 19,070 | 954 | 4e-4 | 20 | 4% |

**Optimizer.** AdamW, β = (0.9, 0.95), ε = 1e-8, decoupled weight decay 0.1 on 2-D matrices only (not norm gains or the tied embedding, as in OLMo 2), clipping 1.0, no dropout. η_peak = 2e-3 · (S / 25M)^(−0.4), keyed to nominal size, is an envelope of Pythia (Biderman et al. 2023), llm.c, MobileLLM, SmolLM and OLMo 2 practice between the DeepSeek compute fit (Bi et al. 2024) and a width rule; a pilot sweep {0.5, 1, 2} · η_peak at 25M and 100M on the base objective only (never on phase-2 outcomes, so no hyperparameter can favour a teacher) refits the exponent (Porian et al. 2024) [CHECK: re-keying by N_ne]. Batches sit at or below critical-batch estimates (Zhang et al. 2024) and are constant across phases; the micro-batch is fixed per (size, GPU) and identical across conditions.

**Phase 2 (WSD branching).** Every condition branches from θ*(S) with no re-warm and decays the learning rate linearly to zero over D₂ = 800M (Bergsma et al. 2025; Hägele et al. 2024), with identical batch, micro-batch and step count within a size (Table 8). Branching at peak learning rate places the treatment in the annealing phase, where high-quality data has its largest effect, and outside the over-training regime in which adaptation degrades (Springer et al. 2025); WSD is also the schedule Ibrahim et al. (2024) recommend for continual pretraining, and it removes the re-warm loss spike (Gupta et al. 2023).

**Table 8. Phase 2** (identical for every arm within a size). D₂ = 390,625 sequences: treatment 292,968 (≈ 600.0M tokens), replay 97,657 (≈ 200.0M) [CHECK rounding].

| S | D₂ | Treatment | Replay (ρ = 0.25) | Batch (seq) | Phase-2 steps | LR | Re-warm | Held-out loss every |
|---|---|---|---|---|---|---|---|---|
| 25M | 800M | 600M | 200M | 128 | ≈ 3,050 | 2.0e-3 → 0, linear | none | 80 steps |
| 100M | 800M | 600M | 200M | 256 | ≈ 1,525 | 1.2e-3 → 0, linear | none | 40 steps |
| 250M | 800M | 600M | 200M | 256 | ≈ 1,525 | 8e-4 → 0, linear | none | 40 steps |
| 500M | 800M | 600M | 200M | 512 | ≈ 763 | 6e-4 → 0, linear | none | 20 steps |
| 1B | 800M | 600M | 200M | 512 | ≈ 763 | 4e-4 → 0, linear | none | 20 steps |

**Replay.** ρ = 0.25: 200M tokens per branch are already-seen base shards, the same shards in the same slot positions for every condition; 25% follows Ibrahim et al. (2024) for a strong distribution shift. Replay is not the treatment, so C1 receives it too and C_T − C1 is exactly "600M synthetic versus 600M fresh web with 200M shared replay". Repeated tokens cost almost nothing below four epochs (Muennighoff et al. 2023), which makes C0 definable.

**Packing, masking and loss.** Documents (`<s> … </s>`) are concatenated in prompt-id order and split into 2048-token windows; attention is intra-document (block-diagonal, FlashAttention varlen or FlexAttention) with document-relative positions (Dubey et al. 2024; Zhao et al. 2024). Without masking, cross-document leakage would co-vary with the teacher's length distribution. The objective is the full-sequence next-token loss over prompt, response and markers (Shi et al. 2024; Huerta-Enochian and Ko 2024), identical in treatment and control arms; D_syn is the number of trained treatment tokens, prompts included. There is no padding, so token counts are exact.

**Seeds.** A phase seed re-draws the permutation of treatment sequences into treatment slots, the shared replay permutation, the slot pattern and the CUDA RNG stream; it changes nothing about initialisation, and at 600M the subsample is the whole delivered pool, so seeds do not re-draw it (at D-sweep levels the nested prefix is fixed). Seed variance is therefore data-order variance, a lower bound on run-to-run variance. Seeds use common random numbers: the same slot pattern and replay order in every condition. Base seeds are replicated only at the pilot sizes (Table 9), giving σ²_base and σ²_base×T, which inflate cross-S standard errors; within S the base run cancels in every contrast.

**Table 9. Seeds and replication.**

| Stage | Base seeds b | Phase seeds s | Cells |
|---|---|---|---|
| Pilot (25M, 100M) | 3 | 3 | {C0, C1, C1b, T1.5B, T7B, T32B}: 54 branches per size; nested decomposition of σ²_base, σ²_base×T, σ²_order |
| Full, tier 1 | 1 (b = 1 from the pilot at 25M, 100M) | 3 | all 35 treatment cells (5 S × 7 T), C0, C1b |
| Full, tier 2 | 1 | +2 (= 5) | T ∈ {1.5B, 32B, 72B} at every S; argmax ± 1 per S (confirmation seeds, never used for selection); C1 at every S |
| D-sweep | 1 | 2 | 36 runs (§3.6) |

**Checkpoint rule.** The final checkpoint is always the evaluated one: no best-on-benchmark selection, early stopping or averaging. Held-out loss is logged at the cadence of Table 8 on held-out base text (10M tokens), human Q&A (5M), 2M tokens of each teacher's held-out synthetic text (monitoring only) and the instruction NLL set; benchmarks run at θ*, mid-decay (diagnostic) and final (primary).

**The "only one file differs" guarantee.** One script builds every branch from (treatment file, replay shard list, tokenizer hash, s, D₂, ρ, L): it packs the treatment to exactly n_treat sequences (surplus dropped; a shortfall is a hard error, so generation over-delivers by ≥ 5%), packs replay to n_rep, draws the slot pattern once per (S, s), and emits one token memmap, one boundary array and their SHA-256. The branch config has exactly one condition-specific field, `treatment_path`; a pre-flight check hashes the config with that field removed and refuses to launch unless it equals the size-level reference, and also checks code commit, container digest, library versions, GPU model, θ*, tokenizer, micro-batch and slot-pattern hashes. The loader is a stateless index-to-sequence map. A run whose loss exceeds twice its running minimum for more than 50 steps, or produces NaN, is stopped, logged and re-run once with the same seed after root-causing, never silently replaced.

## 3.6 Control arms and the inference ladder

**Table 10. Arms per student size** (every arm receives the same 200M replay stream).

| Arm | Treatment file (600M tokens) | Purpose |
|---|---|---|
| C0 | already-seen base shards (second epoch) | "nothing new" cooldown reference |
| C1 | fresh held-out FineWeb-Edu | matched-real control; the primary comparator |
| C1b | Stack Exchange question and accepted-answer pairs (non-code sites, CC BY-SA), rendered with the study's markers | separates Q&A format from synthetic provenance |
| C1c (100M and 500M only) | 20M human instruction tokens (Dolly + OASST1 + No Robots, one epoch) + 580M fresh web, versus each teacher subsampled to 20M + 580M fresh web, versus 600M fresh web; 3 seeds | the only honest human-instruction comparison |
| T_x, x ∈ {0.5, 1.5, 3, 7, 14, 32, 72}B | teacher pool subsampled to 600M | treatment cells |

C0 is required because under WSD branching "base only" cannot mean "no further training": the un-decayed θ* is not a finished model. C1 is the primary comparator. C1b separates format from provenance: C_T − C1b is the value of teacher-written over human-written answers of the same form, and C1b − C1 the value of the form. A 600M-token human instruction control does not exist (human-written instruction data totals a few tens of millions of tokens [estimate]; FLAN-style collections are templated from benchmark training splits), so C1c makes the synthetic-versus-human-instruction comparison at the only dose where it is honest. The inference ladder, all paired over seeds: synthetic versus nothing new (C_T − C0), versus more web (C_T − C1), versus human Q&A (C_T − C1b), versus human instructions at small D (C1c), and the teacher contrast Δ_S(T, T′).

**D-sweep.** At fixed D₂, D_syn ∈ {75M, 150M, 300M} (600M is the main grid) with the remaining treatment slots filled from fresh web shards; schedule and steps unchanged, no extra generation. Cells: S ∈ {100M, 1B} × T ∈ {3B, 14B, 72B} × 3 levels × 2 seeds = 36 runs with nested prompt subsets. It identifies the D term (§3.8), feeds the iso-cost analysis (§3.9) and gives dose-matched points: the phase-2 fraction is 16% of base tokens for the small students and 4% at 1B, while 300M at 1B and 75M at 100M are both 1.5%. Pre-registered cross-checks: X1, real plus synthetic from scratch (25M, 100M); X2, over-trained 100M base; X3, plain-FineWeb base (100M); X4, ρ ∈ {0, 0.5}; X5, phase-2 peak LR × {0.5, 0.25} (100M) [CHECK: list against the design record].

## 3.7 Evaluation

**Table 11. Evaluation suites.** Log-likelihood tasks are zero-shot with harness-default prompts. Chance c_k is the better of uniform guessing and the majority class, recomputed from the downloaded splits.

| Suite | Task (harness id) | Metric | Shots | Items | Chance |
|---|---|---|---|---|---|
| Primary, tier A (always in K) | SciQ (`sciq`) | acc | 0 | 1,000 (test) | 0.25 |
| | ARC-Easy (`arc_easy`) | acc_norm | 0 | 2,376 (test) | 0.25 |
| | PIQA (`piqa`) | acc_norm | 0 | 1,838 (val) | 0.50 |
| | LAMBADA-OpenAI (`lambada_openai`) | acc (ppl logged) | 0 | 5,153 (test) | 0 |
| Primary, tier B (floor-gated) | OpenBookQA (`openbookqa`) | acc_norm | 0 | 500 (test) | 0.25 |
| | SocialIQA (`social_iqa`) | acc | 0 | 1,954 (val) | 0.333 |
| | HellaSwag (`hellaswag`) | acc_norm | 0 | 10,042 (val) | 0.25 |
| Co-primary continuous | held-out human instruction NLL (Dolly 2,000, OASST1 1,000, NQ-open 1,000) | nats per token on response tokens; bits per byte | n/a | ≈ 1M tokens | n/a |
| | task loss | bits per byte of the correct continuation, mean over K | 0 | as above | n/a |
| Secondary | ARC-Challenge (`arc_challenge`) | acc_norm | 0 | 1,172 | 0.25 |
| | Winogrande (`winogrande`, xl) | acc | 0 | 1,267 | 0.50 |
| | BoolQ (`boolq`) | acc versus majority | 0 | 3,270 | 0.622 (majority "yes"; Clark et al. 2019) |
| | MMLU-continuation, STEM and full (`mmlu_stem_continuation`, `mmlu_continuation`) | acc_norm | 0 | ≈ 3,018; 14,042 | 0.25 |
| | CommonsenseQA-cloze (custom YAML) | acc_norm | 0 | 1,221 | 0.20 |
| | BLiMP (`blimp`) | acc, mean of 67 paradigms | 0 | 67,000 pairs | 0.50 |
| | TruthfulQA-MC2 (`truthfulqa_mc2`) | mc2 (report only) | 6 (fixed prefix) | 817 | ≈ 0.40 (uniform-probability) [CHECK] |
| | GSM8K (`gsm8k`; 1B only, gated on C0 or C1 ≥ 3%) | exact match, flexible-extract (strict logged) | 5 | 1,319 | 0 |
| | IFEval-lite (`ifeval`, `max_gen_toks` 512) | prompt-level loose accuracy (report only; all four logged) | 0 | 541 | empty-output baseline, computed once [CHECK] |
| | TriviaQA-3k (`triviaqa`, `--limit 3000`; 1B, optional) | exact match | 0 | 3,000 | 0 |
| | ICL probe: ARC-Easy, SciQ | acc_norm, acc | 5 | as above | as above |
| | format-adherence probe (200 hand-written items, 8 templates) | pass rate per template | 0 | 200 | ≈ 0 |
| Diagnostic | teacher-validation NLL matrix (2% of each teacher's pool) | nats per token | n/a | ≈ 0.5M tokens per teacher | n/a |
| Forgetting | held-out FineWeb-Edu (5M tokens); FineWeb non-Edu (2M) + Wikipedia (1M) | nats per token | n/a | | n/a |

**Settings.** lm-evaluation-harness 0.4.13, pinned with `transformers`, `datasets` and every dataset's HF commit hash; zero-shot for all primary tasks and conditions; harness-default prompts, no chat template, seed 1234, batch 64, bf16, `--log_samples`. Metrics are fixed in advance: acc_norm for variable-length free-text options; acc for SciQ, SocialIQA, BoolQ, Winogrande, LAMBADA and BLiMP; the other metric is logged, never used for selection. Zero-shot is used because in-context learning is absent below ≈ 400M in token-matched public checkpoints; a 5-shot probe is secondary. Letter-answer MMLU is excluded (at chance for all public models ≤ 1B). Public Pythia checkpoints are run through the same pipeline as validation. Only the final checkpoint is analysed.

**Floor gate and aggregate.** Tier A always enters K. Tier B task t enters at all sizes iff, on the C0 seeds at the smallest retained size,

$$\overline{a}_t^{\,C0} - c_t \;\ge\; \max\big(5\ \text{points},\ 3\cdot \mathrm{SE}_{\mathrm{item},t}\big),\qquad \mathrm{SE}_{\mathrm{item},t} = \sqrt{a_t(1-a_t)/N_t},$$

evaluated once before any synthetic condition is analysed; the same K is used at every S. The 25M row is flagged near-floor if a Tier A task fails and removed from the primary grid if |K| < 4. The primary discrete outcome is the chance-corrected mean (centred accuracy; Li et al. 2024)

$$P_{cc} = \frac{100}{|K|} \sum_{k \in K} \frac{a_k - c_k}{1 - c_k},$$

in chance-corrected points; its item SE is (1/|K|) · √(Σ_k SE_k² / (1 − c_k)²), ≈ 0.85 points for the four-task set [assumption]. Raw mean, base-normalised gain and a precision-weighted variant are secondary.

**Continuous outcomes** (co-primary, floor-free): (i) held-out NLL on human-written instruction and QA text, rendered with the training markers, on response tokens (also without marker tokens), with no assumed direction; (ii) task loss, the mean bits per byte of the correct continuation averaged over K. Continuous metrics have higher signal-to-noise at small scale (Madaan et al. 2024; Bhagia et al. 2024; Heineman et al. 2025) and are what scaling analyses model (Hoffmann et al. 2022; Busbridge et al. 2025). All NLLs are in the student tokenizer and comparable within the study only.

**Contamination audit** (mandatory before any teacher comparison is reported). Larger teachers can regurgitate more benchmark text. On the raw pools, per teacher and benchmark, we report the item-level 13-gram or whole-short-item hit rate ρ_{T,b}, the example-level rate r_{T,b}, a paraphrase rate ρ^para_{T,b} (5k outputs per teacher, embedding top-1 retrieval against ARC, SciQ, OBQA and MMLU-STEM items, candidates at cosine ≥ 0.8 judged by an LLM; Yang et al. 2023) and the benchmark-style density (fraction of outputs that are four-option science questions). All conditions are re-evaluated on items with zero hits in any pool (clean-b); if ρ^para differs across teachers by more than ≈ 2 points on a primary benchmark, clean-b becomes the headline. Removed examples are replaced by whole-example resampling so that D_syn stays matched.

## 3.8 Statistical analysis

**Units and notation.** The replication unit is a training run; item-level standard errors describe one fixed model and are not used to test treatment effects. Let s = log₂(S / 25M), x = log₂(T / 1B) with actual sizes (0.62 for 1.5B to 6.18 for 72B; −1.03 for 0.5B in the secondary fit), d = log₂(D_syn / 600M), and α_S a per-S intercept absorbing the base run and the S main effect.

**Table 12. Model sequence** (primary outcome P_cc; repeated for each secondary outcome).

| Model | Mean structure | Parameters | Test |
|---|---|---|---|
| M0 | α_S | 5 | reference |
| M1 | α_S + β₁x | 6 | M1 vs M0: H1 (any monotone T effect) |
| M2 | α_S + β₁x + β₂x² | 7 | β₂ < 0, one-sided: H2 (diminishing returns); M2 vs M1 |
| M3 | α_S + β₁x + β₂x² + γ₁ s·x + γ₂ s·x² | 9 | M3 vs M2 (2 df): does the T curve depend on S (H4 precursor) |
| M3cat | α_S + β_{1,S} x + β_{2,S} x² | 15 | M3cat vs M2 (8 df): unrestricted S × T; M3cat vs M3 (6 df): adequacy of the log-linear interaction; the per-S model for T*(S) |
| M4 | M3 + δ₁d + δ₂ d·x + δ₃ d·s + δ₄ d², on the D-sweep cells and the main grid at d = 0 | 13 | δ₂ ≠ 0: does the teacher effect change with dose (Bansal et al. 2024) |
| M2sat | α_S + A_S (1 − T^(−β)), nonlinear least squares | 11 | AICc against M3cat: saturating versus interior form |

OLS with F-tests for nested comparisons; nonlinear least squares with multiple starts for M2sat; AICc for non-nested comparisons (n ≤ 150). Per-S tests use the per-S residual variance from M3cat (df = 6r_S − 3); pooled coefficient tests use HC3 sandwich standard errors; a Brown-Forsythe test across S is reported. Each D level is its own run, never a checkpoint cut from one run. Diagnostics (residual plots, Cook's distance above 4/n with a repeated analysis) go to the appendix. Nonparametric per-S checks: isotonic regression on T with permutation tests for "some monotone effect" and "non-monotone", Spearman's ρ between cell means and log T, and a rank-based duplicate of every pooled test.

**Per-S classification of T*.** Two three-parameter forms are fitted per S: F1, quadratic in x with vertex x̂* = −b / (2c); F2, saturating, P = P_∞ − A · 2^(−βx). Labels are assigned in order:

1. *Flat* iff the T-effect test is not significant for this S; report the CI of the total range.
2. *Interior optimum* iff all four hold: (i) ĉ < 0 with p < 0.05 on the within-S variance; (ii) x_min < x̂* < x_max; (iii) the 95% CI of x̂* has upper bound below x_max; (iv) AICc(F1) ≤ AICc(F2) + 2. Then T*(S) = 2^(x̂*) with its CI.
3. *Saturating* iff not interior and the one-sided 95% upper bound of the model-based top-step gain (32B to 72B) is below δ; report T_sat(S), the smallest grid T at which the upper bound of the gain from doubling T, Δ₂(x) = b + c(2x + 1), falls below δ.
4. *Monotone, unsaturated* otherwise: T* is right-censored at 72B.

δ = 1.0 chance-corrected point (sensitivity 0.5), fixed now: about 2σ_agg under the central noise assumption and a quarter of the expected total range. Any claim that an intermediate teacher beats 72B at S must be confirmed on the tier-2 seeds alone, which were not used to select the cell (the maximum of six noisy means is biased upward by about 1.27σ/√r): tier-1 seeds select, tier-2 seeds confirm.

**Uncertainty.** The primary CI is seed-based: a cluster bootstrap over the r_T runs within each cell (B = 2,000, percentile CIs on the log₂ scale) when r ≥ 3 in every cell of that S, otherwise a parametric bootstrap from the coefficient covariance with σ̂²_S (93 to 95% coverage in design simulations). A paired item bootstrap (items resampled with replacement, stratified by benchmark, clustered by activity for HellaSwag) is a required robustness check, because runs sharing a base are highly correlated item by item (Miller 2024); a directional claim requires both intervals to exclude 0. The variance of a cell contrast is ≈ 2σ²_seed / r + σ²_{c×j} / N, the second term from the item bootstrap.

**Multiplicity.** H1 to H4 map to one model-level test each (M1 vs M0; β₂ < 0; the classification; M3 vs M2). Per-cell contrasts are secondary and Holm-corrected within pre-defined families: F-A, adjacent teachers within S (5 per S); F-B, teacher versus C1 (6 per S); F-C, teacher versus C0 (descriptive); F-D, C1 versus C0 (5); F-E, economic versus raw optimum (5); F-F, intermediate-beats-largest confirmations on tier-2 seeds (≤ 5). All contrasts are tabulated with raw and adjusted p-values, CIs, df and Hedges' g alongside the absolute effect; "significant" means adjusted p < 0.05.

**T* versus S.** If at least three sizes are classified interior, log₂ T̂*_S is regressed on s by weighted least squares (weights 1 / Var(x̂*_S)), and α̂ is reported with t-based and bootstrap CIs as a descriptive summary of five estimates, with censored sizes shown as arrows and excluded. With five points the CI half-width of α̂ is about ±0.35 to ±0.5 even in favourable simulations, so H4 is assessed as consistent or inconsistent with a monotone relation, never as a law.

**Reporting standard** (binding). A full cell table (r, mean ± SD of P_cc, every a_k, task loss, NLL) and the run-level CSV are released; per-S curves show t-based 95% seed CIs with r on the panel, individual seed points, C0 and C1 bands, and F1/F2 bootstrap bands; "optimum", "saturation" and "monotone" are used only under the rules above, "law" never, and "no effect" is phrased as "not detected; CI for the total range [...]"; every test reports estimate, CI, df, raw and adjusted p; seed accounting states which cells have 3 versus 5 seeds and which seeds selected versus confirmed; negative results receive equal prominence. The plan is frozen and hashed before the first full-study run; deviations are logged and both analyses presented.

## 3.9 Cost measurement and Synthetic Transfer Efficiency

**Generation cost.** τ_gen(T) is GPU-seconds summed over all cards occupied by the serving engine from first request to last response (load and warm-up excluded and reported separately); E_gen(T) is 1 Hz NVML power integrated over cards and multiplied by PUE 1.2 [assumption]; D_pool(T) is delivered student tokens after filtering; N_proc(T) is teacher tokens processed. Per delivered student token,

$$c_\tau(T) = \frac{\tau_{\mathrm{gen}}(T)}{D_{\mathrm{pool}}(T)},\qquad c_\$(T) = c_\tau(T)\,p,\qquad c_E(T) = \frac{E_{\mathrm{gen}}(T)}{D_{\mathrm{pool}}(T)},\qquad c_F(T) = \frac{2\,N_T\,N_{\mathrm{proc}}(T)}{D_{\mathrm{pool}}(T)},$$

with p = $2.00 per GPU-hour on RTX PRO 6000 (sensitivity $0.95 owned and $3.00), N_T the teacher's parameter count and attention FLOPs (< 1%) ignored; costs are logged in 10k-prompt chunks. A cell's cost is C_gen(S, T) = c(T) · D_syn, independent of S at fixed D_syn. Time cost is expected to be sub-linear in T below 7B and super-linear above 32B while FLOP cost is linear [assumption], so both units are reported.

**Gains and STE.** With M(S, cond) the primary outcome, G₀(S, T) = M(S, T) − M(S, C0) and G₁(S, T) = M(S, T) − M(S, C1); G₁ is the economically relevant gain because real tokens are approximately free. The primary efficiency metric is

$$\mathrm{STE}(S,T) = \frac{G_1(S,T)}{c_\$(T)\,D_{\mathrm{syn}}} = \frac{M(S,T) - M(S,\mathrm{C1})}{c_\$(T)\,D_{\mathrm{syn}}},$$

in chance-corrected points per dollar (STE_τ per kGPU-s and STE_F per EFLOP as hardware-free variants). A ratio of a concave gain over cost peaks at the cheapest teacher, so STE is reported, not used to select. Selection uses the marginal efficiency between cost-adjacent teachers,

$$m_i(S) = \frac{G_1(S,T_{i+1}) - G_1(S,T_i)}{C_\$(T_{i+1}) - C_\$(T_i)},$$

and the price-of-a-point curve

$$T_\$(S;\lambda) = \arg\max_T \big[G_1(S,T) - \lambda\,C_\$(S,T)\big],$$

reported as a step function over λ ∈ [0, 10] points per dollar; λ = 0 recovers the raw optimum and the switch points are the m_i when G₁ is concave in cost. Constant-free summaries: λ₁(S), the marginal efficiency of the last upgrade into T* (the smallest price of a point at which T* stops being optimal), and T_ε, the cheapest teacher with G₁ ≥ (1 − ε) · G₁(T*), ε = 0.10.

**H5 rule.** H5 (cost-optimal ≠ performance-optimal) is supported at S if the 95% CI of λ₁(S) lies below λ_ref(S) = G₁(S, T₁) / C_$(S, T₁), the average efficiency of the cheapest teacher, and the loss at T_$ relative to T* is below δ with a CI excluding δ. Five contrasts, Holm-corrected. CIs propagate gain and cost uncertainty by a pooled-variance parametric bootstrap of cell and control means (item resampling nested; non-parametric seed resampling as a check), chunk resampling of cost, price by sensitivity only, B = 5,000; for T_$ the selection probability of each teacher at each λ is reported and a choice is called stable at ≥ 0.8. **Iso-cost analysis** (co-reported): P(S, T, D) interpolated from the D-sweep is evaluated at D(T) = B / c(T) over a grid of budgets B (the compute-matched question of Bansal et al. 2024), as model-based extrapolation with CIs.

## 3.10 Pilot and pre-registration

**Design.** S ∈ {25M, 100M}; three base and three phase seeds; conditions {C0, C1, C1b, T1.5B, T7B, T32B} (32B to span the widest range and exercise the single-card BF16 path): 54 branches per size and a nested decomposition of σ²_base, σ²_base×T and σ²_order; a base LR sweep; ρ ∈ {0, 0.5}; the length-matched and unconstrained-length ablations; and a dose curve at (100M, 7B) over D_syn ∈ {75M, 150M, 300M, 600M} with two seeds [CHECK]. All runs use the full suite and the continuous outcomes.

**What the pilot estimates.** σ̂_seed,k per benchmark and σ̂_agg (df = 16); σ̂_base; the floor status of each benchmark at 25M and 100M; the synthetic-versus-C1 and versus-C0 gains; the teacher spread at each S; the dose response; realised lengths, yields, prompt-token fractions and r_T (which fix the category token shares and confirm the pool sizing); 72B BF16 throughput (which decides the precision fallback); MFU and evaluation overhead. Proceeding to the full study requires P1 to P4 (Table 13); P5 and P6 gate the dose design and pipeline integrity.

**Table 13. Pre-registered go/no-go rules.**

| Rule | Criterion | Rationale |
|---|---|---|
| P1 Detectable synthetic effect | At ≥ 1 student size, the pooled synthetic-versus-C1 difference (9 vs 3 runs, within-S variance) has a 95% CI excluding 0 in either direction, on P_cc or on task loss | If synthetic data does nothing detectable against matched real tokens at this D_syn, the T question is moot; a robustly negative effect passes but triggers a generation-pipeline review |
| P2 Noise | Upper 95% bound of σ̂_agg ≤ 0.9 points on the floor-gated P_cc, or the equivalent on task loss | Above this, 1-point effects would need r ≥ 10 per cell |
| P3 Floor | ≥ 4 benchmarks retained at 25M by the gate, ≥ 6 at 100M | Otherwise P_cc at 25M is one to three noisy numbers |
| P4 Spread | Teacher spread at 100M ≥ 1.0 point (point estimate) on P_cc or the corresponding task-loss spread | If the three most different teachers cannot be told apart by 1 point, a six-teacher grid at r = 3 cannot map a curve |
| P5 Dose | P_cc at (100M, 7B) is non-decreasing in D_syn across the four levels (Spearman > 0) and the 600M − 75M difference has a CI excluding 0 | Confirms the phase is not saturated at D_syn and that 600M sits on the responsive part of the dose curve |
| P6 Integrity | Re-evaluation of a checkpoint reproduces every number exactly; decontamination report clean; NLL on human text not worse than C1 by more than a pre-set margin (5% relative) | Pipeline validity |

**Seeds after the pilot.** With σ̂^upper the upper 95% bound of σ̂_agg: tier 1 uses r = 3 if σ̂^upper ≤ 0.6; r = 4 and tier 2 = +2 if 0.6 < σ̂^upper ≤ 0.9; otherwise no extra seeds but variance reduction (floor gating, task loss as primary, larger D_syn) and a re-pilot of one cell. If σ̂²_base×T is at least a quarter of the teacher-effect variance, base seeds are added at 250M before the full grid.

**Redesign triggers.** P3 failing at 25M only demotes that row to exploratory (4 × 6 primary grid); failing at 100M too moves the suite toward continuous outcomes. P1 or P4 failing raises D_syn by 2 to 4× and revises generation before re-running (100M, 7B) versus C1; if still undetectable at 4 D_syn, the premise is reported as unsupported at this scale. P5 failing lowers D_syn. If X3 shows a materially larger effect and a different ranking on plain FineWeb, the base corpus switches before the 250M to 1B base runs. Seed outliers are reported, never dropped. Pilot data are not pooled into the full-study inference. The design record and analysis plan are frozen with recorded hashes before the first full-study run; every deviation is logged with its reason.

## 3.11 Compute and reproducibility

**Hardware and envelope.** Reference hardware is two RTX PRO 6000 Blackwell (96 GB) and two RTX 5080 (16 GB). Teachers ≤ 32B serve on one 96 GB card, 72B on both (TP 2, PCIe); students 25M to 250M train on the 5080s, 500M and 1B on the PRO 6000s. The study is ≈ 1,400 to 1,750 PRO-6000-equivalent GPU-hours (generation ≈ 400 to 750 depending on measured 72B throughput, base runs ≈ 270, phase 2 ≈ 620, evaluation ≈ 100 on 5080s) [assumption], five to six weeks wall-clock with the 72B pool and the 1B base run on the critical path. Per-run GPU-seconds and, where measured, energy are recorded. A `local-5080` profile supports students ≤ 250M and teachers ≤ 3B on one 16 GB card, with larger generation rented.

**Manifests and hashes** (Table 14). Pool, filters, subsampler, suffixes, system prompt and tokenizer are versioned strings with SHA-256 of their sources; each derived training set (equal-token, equal-example, length-matched, nested-D prefix) is materialised once, hashed and referenced by hash in training configs. A training run is reproducible from (θ* hash, dataset hash, config hash, seed).

**Table 14. Released manifests.**

| Manifest | Contents |
|---|---|
| Per-teacher generation manifest | repository and commit SHA; weights hash; dtype; tensor parallelism; GPU; engine and library versions; chat-template and system-prompt hashes; decoding; pool, filter and suffix versions; stopping rule; per-shard prompts, retained examples, tokens in both tokenizers, wall-clock, tokens/s, kWh; filter statistics; dataset hash; subsample hashes |
| Per-example JSONL (every generated sample, retained or dropped) | exact teacher input; student-visible prompt; raw response; finish reason; both token counts; log-probabilities; all filter flags; verification result; shard id |
| Per-run training manifest | size; condition; base and phase seeds; git commit; container digest; library versions; GPU model and count; tokenizer hash; θ* hash; treatment path, hash, document count and prompt-token fraction; replay and slot-pattern hashes; D_base, D₂, ρ, exact trained tokens; batch, micro-batch, accumulation; LR, schedule, optimizer, clipping, precision, masking; steps; evaluation-set hashes; checkpoint rule; config hash excluding the treatment field; wall-clock, GPU-seconds, energy, MFU; divergence events |

**Pinned versions and determinism.** One vLLM version, one CUDA and driver, one torch and flash-attn (or FlexAttention) build and one container digest for the whole study; lm-eval 0.4.13 with pinned `transformers`, `datasets` and dataset commit hashes; teacher repositories at pinned commits; the same GPU model for every branch of a size and for all teachers. Bit-exact reproducibility is not claimed, since attention backward kernels and continuous batching are nondeterministic; reproducibility is at the artefact level (hashes) and the statistical level (seeds). Code, manifests, the prompt pool, the filtered pools [CHECK licences], run-level results and the harness configuration are released.
