# 21. Paper Outline

Status: structure proposal for the write-up. Binding inputs: doc 00 (locked methodology) and the pre-registered analysis plan (doc 12); where a workstream draft and doc 00 differ, doc 00 wins and the difference is noted with [CHECK]. Figure and table ids (F*, T*, AF*, AT*) refer to doc 20.

Tags used below: [fact] established; [lit] literature claim (with the source); [hyp] hypothesis; [choice] methodological choice; [placeholder] number to be read from `results/index.parquet` after the runs exist. No number in the paper may come from anywhere else.

Target venue: workshop paper (main text 8 to 9 pages) or arXiv preprint with full appendices. Word targets below sum to roughly 7,400 to 9,900 words of main text, excluding references, figure captions and appendices. Sections 4 to 7 are written last, after `scripts/analyze.py --study main` has produced every table.

Working title (doc 00 §0.1): "Which Teacher for Which Student? Teacher-Size Effects of Synthetic Annealing Data for 25M–1B Language Models". Fallback if no interior optimum is confirmed at any student size: "Teacher-Size Transfer Curves for Synthetic-Data Continued Training of 25M–1B Language Models". The choice is made after the per-S classification (doc 12 §5.1), never before.

## Global conventions for the write-up

1. Every effect is a difference in P_cc (chance-corrected points over the floor-gated benchmark set K, identical across S) or in held-out NLL (nats per token in the student tokenizer), with a 95% seed-based t CI, r stated, and the paired-item bootstrap CI for every headline contrast (doc 12 §8.6).
2. Vocabulary rules (doc 12 §8.3): "optimum" or "T*" only when the four-condition interior rule holds and the tier-2 confirmation contrast excludes 0; "saturation" only with δ stated; "law" is not used; a null result is phrased "not detected; 95% CI of the total range [a, b]".
3. Every synthetic cell is reported against both C0 and C1 [lit: Honda 2026 shows that reporting only post-distillation numbers hides degradation].
4. Established results from neighbouring regimes (larger teacher not always better; saturation of generator size; cheaper teacher can be compute-optimal; best teacher depends on the student) are cited as such and never presented as findings of this study (lit_review §7).
5. The method is named "sequence-level distillation via synthetic annealing data" (doc 00 §0.1); "synthetic scaling laws", "distillation scaling laws" and "law of capacity gap" are not used (taken by Qin et al. 2025, Busbridge et al. 2025, Zhang et al. 2025).

---

## Abstract (200 to 250 words; draft in `paper/abstract_draft.md`)

- One sentence on the practical decision (a small pretrained model, a fixed generation budget, which teacher).
- One sentence on why prior answers come from neighbouring regimes (logit KD, 7B-class instruction tuning, web rephrasing) and what they predict (saturation between roughly 1B and 8B) [lit].
- Design in two sentences: five students 25M to 1B, one base run per size, identical annealing branch of 800M tokens with 600M treatment tokens; arms C0, C1, C1b and six (plus one) Qwen2.5-Instruct teachers; equal student-tokenizer tokens; one frozen prompt pool, one decoding configuration, teacher-agnostic filters [choice].
- Outcomes and seeds: P_cc over a floor-gated set, held-out NLL on human instruction text, r per cell [placeholder].
- Results as bracketed placeholders: gain over C1 per S; per-S classification (interior, saturating, monotone-unsaturated, flat); T_sat or T* where identified; descriptive α̂ with CI; T$ versus T* from measured cost; mediation outcome.
- What is not claimed: one family, one protocol, no scaling law, nothing about model collapse or from-scratch synthetic pretraining.
- No headers, no citations, no invented numbers.

---

## 1 Introduction (900 to 1,300 words; draft in `paper/introduction_draft.md`)

- The practical question, stated as a decision problem: a lab holds θ*(S) for a small student and a generation budget B; the choice variable is teacher size T within a family; the objective is the student's downstream gain per token or per dollar.
- Why the literature does not settle it: logit-KD laws [Busbridge et al. 2025; Zhang et al. 2025] need teacher probabilities the student cannot see; compute-matched sampling [Bansal et al. 2024] and instruction-tuning generator sweeps [Xu et al. 2024] use 7B-class students; rephrasing and textbook generator sweeps [Pieler et al. 2024; BeyondWeb 2025; Kang et al. 2025; Niklaus et al. 2026; SmolLM/Cosmopedia 2024] use from-scratch mixtures or a single student size and report saturation between roughly 1B and 8B [lit].
- What is held fixed (base corpus and state, D₂, ρ, schedule, tokenizer, prompt pool, decoding, filters, rendering, evaluation) and what varies (the generator only, and D_syn in a sub-grid) [choice].
- The identification statement in one sentence (doc 00 §0.3): τ_S(T) = E[Y(Π₂(θ*(S), F_T)) − Y(Π₂(θ*(S), F_real))]; Δ_S(T,T′) = τ_S(T) − τ_S(T′) is the total effect of teacher identity, read as teacher size within the fixed family.
- H1 to H5 as falsifiable statements with the named test for each (doc 12 §4 to §5, doc 13 §E.4) and the prior expectation from the literature for each (doc 00 §0.2).
- Contributions as questions answered (not as results): (i) does the teacher-size curve have an interior optimum, a plateau, or no detectable effect at each S; (ii) does the logit-KD relation T*(S) transfer to the sequence channel; (iii) where do T* and T$ diverge with measured cost; (iv) does size have an effect once student-base NLL on the teacher's data is controlled.
- A paragraph on what the study cannot establish (doc 00 §0.15).
- References: Table T1 (grid and arms); optional F0 (design schematic).

---

## 2 Related Work (900 to 1,200 words)

Organised by the three transfer channels of lit_review §2, then by training regime. Every entry below names the papers the subsection cites; verification tags from lit_review §3 carry into the bibliography ([CHECK] items are listed in Appendix H).

### 2.1 Logit and distribution distillation: the capacity gap (250 to 300 words)

- Origin and vision-era capacity gap: Hinton et al. 2015 (arXiv:1503.02531); Cho and Hariharan 2019 (arXiv:1910.01348); Mirzadeh et al. 2020 (arXiv:1902.03393); Stanton et al. 2021 (arXiv:2106.05945); Beyer et al. 2022 (arXiv:2106.05237).
- LM logit KD and teacher-size effects: Zhang et al. 2023/ACL 2025 (arXiv:2311.07052, linear T*(S) for students below 3B); Zhong et al. 2024 (arXiv:2402.11890); Panigrahi et al. 2024 (arXiv:2410.05464, progressive distillation as curriculum); Busbridge et al. 2025 (arXiv:2502.08606, distillation scaling law, optimum in teacher-loss space, teacher inference compute included); Peng et al. 2025 (arXiv:2410.16215); Lu and Liu 2026 (arXiv:2605.23857); He et al. 2026 (arXiv:2609.01532, mid-training KD favours reasoning over recall).
- Theory and mechanism: Ildiz et al. 2024 (arXiv:2410.18837, teacher shifts the constant not the exponent); Huang et al. 2026 (arXiv:2605.29548, rare-task retention in small models); Allen-Zhu and Li 2024 (arXiv:2404.05405, roughly 2 bits per parameter).
- Position statement: this channel is unavailable here (tokenizer mismatch) and the sequence channel discards the token distributions the capacity-gap literature usually invokes; whether the logit-KD T*(S) relation transfers is therefore a real question (lit_review §7 item 2).
- References: none beyond the bibliography; the condensed novelty matrix is Table T2.

### 2.2 Sequence-level synthetic data: the channel of this study (400 to 500 words)

- Definition: Kim and Rush 2016 (arXiv:1606.07947); Hsieh et al. 2023 (arXiv:2305.02301) as a large-gap success case.
- Generator-size studies for pretraining data (the closest prior work): Pieler et al. 2024 (arXiv:2410.20796, family differences exceed size differences); BeyondWeb, Maini et al. 2025 (arXiv:2508.10975, 1B to 3B gain, saturation by 8B); Kang et al. 2025 (arXiv:2510.01631, generator size × student size 100M to 3B, from-scratch mixtures, larger than roughly 8B not better); Niklaus et al. 2026 (arXiv:2604.13977, no gain beyond 1B for rephrasing, up to 30× cheaper); Ben Allal et al. 2024 (Cosmopedia and SmolLM blog posts; alternative 70B-class generators gave no significant improvement).
- Synthetic pretraining and continued pretraining lineage: Eldan and Li 2023 (arXiv:2305.07759); Gunasekar et al. 2023 (arXiv:2306.11644); Li et al. 2023 (arXiv:2309.05463); Abdin et al. 2024 (arXiv:2412.08905); Maini et al. 2024 WRAP (arXiv:2401.16380); Su et al. 2024 Nemotron-CC (arXiv:2412.02595); Nguyen et al. 2025 REWIRE (arXiv:2506.04689); Yang et al. 2024 EntiGraph (arXiv:2409.07431, the term "synthetic continued pretraining" and the log-linear D curve); Qin et al. 2025 (arXiv:2503.19551); Kim et al. 2026 (arXiv:2603.18534); Yu and Xiong 2026 SynPro (arXiv id [CHECK]); Tao et al. 2026 (arXiv id [CHECK]); Gu et al. 2024 MiniPLM (arXiv:2410.17215, data-level KD for 200M to 1.2B students with a fixed teacher).
- Teacher choice for instruction and reasoning SFT: Peng et al. 2023 (arXiv:2304.03277); Mukherjee et al. 2023 Orca (arXiv:2306.02707); Wang et al. 2023 (arXiv:2306.04751); Xu et al. 2024 (arXiv:2411.07133, larger-models paradox, CAR); Bansal et al. 2024 (arXiv:2408.16737, compute-matched, weaker and cheaper wins); Li et al. 2025 (arXiv:2502.12143, small-model learnability gap); Honda 2026 (arXiv:2604.08880); Li et al. 2024 Montessori-Instruct (arXiv:2410.14208); PerSyn 2025 (arXiv:2510.10925); Panigrahi et al. 2025 GRACE (arXiv:2511.02833); Hu 2026 SCAS (arXiv:2605.26872).
- Data properties as mediators: Chen et al. 2024 (arXiv:2410.15226, diversity); "Data Value in the Age of Scaling" 2025 (arXiv:2511.13640, truncation and tails); "Scaling Laws Revisited" 2025 (arXiv:2510.03313, quality term).
- Position statement: none of these varies teacher size and student size jointly for sequence-level data on a shared pretrained base with a matched-real control and measured cost; lit_review §5 table (Busbridge, Bansal, Xu versus this study) is condensed into two sentences.

### 2.3 On-policy distillation (120 to 180 words)

- Agarwal et al. 2024 GKD (arXiv:2306.13649); Gu et al. 2024 MiniLLM (arXiv:2306.08543); Ko et al. 2024 DistiLLM (arXiv:2402.03898); Qwen Team 2025 (arXiv:2505.09388); Thinking Machines Lab 2025 (blog); Li et al. 2026 (arXiv:2604.13016, same-family 1.5B and 7B teachers distributionally indistinguishable to the student); Armandpour et al. 2026 (arXiv:2605.10889); Ding and Zhang 2026 (arXiv:2608.31046, teacher noise grows with teacher scale); Yu et al. 2026 (arXiv:2607.26246).
- Position statement: this study is off-policy by design (limitation, §8); the OPD findings inform the power analysis (within-family effects may be small) and the mediator list.

### 2.4 Training regime, scaling methodology and small-model references (120 to 180 words)

- Kaplan et al. 2020 (arXiv:2001.08361); Hoffmann et al. 2022 (arXiv:2203.15556); Muennighoff et al. 2023 (arXiv:2305.16264, repeats below 4 epochs are nearly free, used for C0); Sardana et al. 2024 (arXiv:2401.00448); Gadre et al. 2024 (arXiv:2403.08540, NLL to downstream mapping); Hu et al. 2024 MiniCPM (arXiv:2404.06395, WSD); Biderman et al. 2023 Pythia (arXiv:2304.01373); Zhang et al. 2024 TinyLlama (arXiv:2401.02385); Liu et al. 2024 MobileLLM (arXiv:2402.14905); Penedo et al. 2024 FineWeb (arXiv:2406.17557); Allal et al. 2025 SmolLM2 (arXiv:2502.02737); Gemma Team 2024 (arXiv:2408.00118) and Mistral AI 2026 Ministral 3 (arXiv:2601.08584) as examples of distilled small models that make a family an imperfect size axis.

### 2.5 Self-training, weak-to-strong, and model collapse (100 to 150 words)

- Zelikman et al. 2022 (arXiv:2203.14465); Huang et al. 2022 (arXiv:2210.11610); Singh et al. 2023 (arXiv:2312.06585); Burns et al. 2023 (arXiv:2312.09390); Shumailov et al. 2024 (Nature 631); Gerstgrasser et al. 2024 (arXiv:2404.01413); Dohmatob et al. 2024 (ICML 2024, tails; arXiv:2410.04840, strong collapse); Feng et al. 2024 (arXiv:2406.07515); Schaeffer et al. 2025 (arXiv:2503.03150).
- Position statement: single-generation synthetic data with the real base retained is not the collapse setting; identical decoding does not imply identical effective entropy across teachers, so output entropy is a reported covariate (lit_review §9 item 7).

### 2.6 What this study adds (80 to 120 words)

- Table T2 (condensed from lit_review §6): the four items lit_review §7 lists as unaddressed, stated as questions; the six items listed as incremental, stated as confirmations or extensions.
- References: Table T2.

---

## 3 Method (1,800 to 2,400 words)

### 3.1 Students (200 to 260 words)

- Tokenizer [choice]: Llama-2 32k SentencePiece, byte fallback, per-digit splitting, SHA-256 pinned; not the teacher tokenizer (a 151k vocabulary would dominate a 25M model).
- Architecture [choice]: tied embeddings, GQA 3:1, head_dim 64, d_ff = 8/3·d, RMSNorm, RoPE θ = 10⁴, no biases, context 2048, bf16 autocast with fp32 master weights, init N(0, 0.02) with 1/√(2L) residual scaling.
- Table T3: L, d_model, heads, kv, d_ff, N_tot, N_ne for s025m to s1b (doc 00 §0.5); both counts reported; S axis uses N_tot with N_ne as sensitivity axis.
- Base-token rule [choice]: D_base(S) = max(20·N_tot(S), 5B) as nested prefixes of one seeded FineWeb-Edu shuffle; tokens per parameter 201, 50, 20, 20, 20 (disclosed confound with S; cross-check X2 at 200 tokens per parameter for 100M).
- Why 5B rather than Chinchilla for the small students [lit: Pythia-70M at 1 to 2B tokens sits at chance on most accuracy benchmarks, doc 11 §1.3]; phase-2 fraction spread reduced from 26× to 4×.
- References: Table T3; Appendix AT6 (hyperparameters).

### 3.2 Teachers (180 to 240 words)

- Family and rungs [choice]: Qwen2.5-Instruct 1.5B, 3B, 7B, 14B, 32B, 72B (primary 6-rung grid) plus 0.5B as a pre-registered seventh rung (extension; secondary 7-rung fit); HF revisions pinned to commit SHAs; safetensors hashed.
- Serving [choice]: BF16 weights and KV cache for all sizes; 72B tensor-parallel 2; FP8-for-all is the pre-registered fallback if measured 72B throughput is below 50% of the model, with a 32B BF16-versus-FP8 bridging cell.
- Teacher capability covariate z_T: each teacher evaluated on the student suite with the same harness and settings (doc 12 §4.5); Table T4.
- Cross-family anchor: Llama-3.1-8B-Instruct at two student sizes (tier 2); bounds "family versus size" [lit: Pieler et al. 2024].
- Known limitations to state here and in §8: the "1B" rung is 1.5B; 3B carries the Qwen Research License; all Qwen2.5 sizes were exposed to 72B-synthesised data during their own training [CHECK against the Qwen2.5 report; lit_review tags this UNVERIFIED]; capability is nearly collinear with log T within a family.
- References: Table T4.

### 3.3 Synthetic data protocol (420 to 540 words)

- Invariance equation (doc 09 §0): D_T = σ(Φ(G_T(P; θ_dec, s_sys, s_suffix, τ_chat)); D_syn, seed), with only G_T teacher-indexed; every other component frozen, versioned and hashed (Table T5 lists the hashes).
- Prompt pool v1.0 [choice]: 4.0M frozen, decontaminated prompts; categories IF 15 / SAQA 15 / CS 15 / SCI 20 / MSR 15 / EXP 12 / STR 8 percent; sources templated taxonomy 41%, FineWeb-Edu passage-seeded 27% (SEED split disjoint from base and C1 shards), fixed non-Qwen prompt writer 24%, human prompts 8%; per-leaf caps; roughly 30% of the pool programmatically verifiable (MSR, STR, IF).
- Generation [choice]: one system prompt, one ChatML template (hash asserted equal across sizes), one decoding configuration (T 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, max 1024 new tokens, n = 1), common-random-numbers prompt order, per-teacher stopping at 1.25× the per-category quota.
- Filters Φ [choice]: F1 to F12 teacher-agnostic rules with fixed thresholds (truncation, length, language and CJK ratio, Gopher repetition, exact and MinHash duplicates, refusal and self-identification, passage reference, format validity, MSR answer line, benchmark contamination with union-drop, length compliance recorded not dropped); no quality judge, no perplexity threshold, no correctness rejection; reasons (doc 09 §2.3): judge bias is teacher-correlated, a judge filter changes the estimand into Bansal's compute-matched regime.
- Contamination removal [choice]: 8-gram exact stage plus embedding stage with a non-Qwen judge on the pool; per-teacher response-level flags with union-drop so prompt sets stay identical; per-teacher flag rates reported (Appendix AT4).
- Amendment A [choice, doc 00 §0.7]: full-sequence loss; D_syn = 600M counts all tokens of the rendered example (prompt, response, markers) so that trained tokens, processed tokens and optimizer steps are identical across arms; prompt-token fraction per teacher (expected 15 to 35%) is a covariate; length-matched and equal-example analyses bound its effect. [CHECK: doc 09 §1.7 still describes response-only counting; the paper follows doc 00.]
- Amendment B [choice]: rendering `<s>### Instruction:\n{prompt}\n\n### Response:\n{response}</s>`, applied identically to C1b and to the held-out instruction NLL set; the lm-eval prompts use a different format, so format familiarity is not part of the treatment.
- Subsampler σ [choice]: seeded, whole-example, stratified by category with identical token quotas per teacher (tolerance 1024 tokens), priority H(s₀, pid) so teacher subsets overlap maximally; nested prefixes give the D-sweep sets; equal-examples secondary set; length-matched ablation with 12 log-spaced bins and KS distance reported.
- References: Table T5; Appendix AT5 (filter yields), AT7 (prompt templates), AF2 (length distributions).

### 3.4 Training design B′ and controls (360 to 460 words)

- Design [choice]: per S one base run with a warmup-stable schedule on FineWeb-Edu, stopped at the stable-phase checkpoint θ*(S) (weights, AdamW state, loader position); every condition is a branch running an identical linear-decay-to-zero phase over D₂ = 800M tokens = 600M treatment + 200M replay of already-seen base shards (ρ = 0.25; same shards and slot pattern for every condition); conditions differ in exactly one file (enforced by the matrix invariant test, doc 18 §18.1).
- Why B′ [choice, with reasons]: Option A rejected (floor effects; Chinchilla-sized synthetic corpora infeasible); Option C rejected (≈ 24× base cost; early-versus-late exposure entangled) but retained as cross-check X1 at 25M and 100M; annealing-phase placement is where small-model recipes put their highest-quality data [lit: MiniCPM, SmolLM2].
- Arms table (Table T1): C0 second epoch of seen shards ("nothing new"); C1 fresh held-out FineWeb-Edu (primary comparator); C1b StackExchange Q&A with the same markers (format versus provenance); C1c at 100M and 500M only (20M human instruction tokens plus 580M fresh web versus each teacher at 20M); T_x for x in {0.5, 1.5, 3, 7, 14, 32, 72}B.
- Identification statement (doc 00 §0.3, equation) and the factorial reading: C1 − C0 = value of fresh web tokens over repeats; T − C1 = synthetic versus fresh web; T − C0 = synthetic versus nothing new. The estimand is conditional on the base corpus, D_base(S), D₂, ρ, the decay schedule, the tokenizer and the family.
- Seeds and tiers (doc 00 §0.10): pilot 3 base × 3 phase seeds at 25M and 100M; tier 1 r = 3 in all 35 treatment cells plus C0 and C1b; tier 2 +2 seeds for T ∈ {1.5B, 32B, 72B} and argmax ± 1 per S (confirmation only) and C1 at every S; a phase seed re-draws data order and slot permutation only, so seed variance is a lower bound and σ²_base from the pilot inflates cross-S SEs. [CHECK: doc 12 §0 specifies that a seed also re-draws the subsample; under doc 00 the 600M subsample is the whole delivered pool, so it cannot be re-drawn; state this in §8.]
- D-sweep [choice]: at fixed D₂ = 800M, D_syn ∈ {75M, 150M, 300M} with remaining treatment slots filled by fresh web shards, S ∈ {100M, 1B} × T ∈ {3B, 14B, 72B} × 2 seeds = 36 runs; nested prompt subsets; provides the dose-matched comparison across S.
- Final checkpoint only (doc 18 §18.1 item 4); cross-checks X1 to X5 summarised in Appendix AF7.
- References: Table T1; F0; Appendix AT6, AT10.

### 3.5 Evaluation (260 to 340 words)

- Primary discrete outcome [choice]: P_cc = (100/|K|) Σ_k (a_k − c_k)/(1 − c_k) over a floor-gated set K identical across S.
- Floor gate [choice, equation]: benchmark k is retained iff a_k^{C0} − c_k ≥ 3·SE_item,k (and ≥ 3·SD_seed,k where base replicates exist) at the smallest retained S; if |K| < 4 at 25M the 25M row leaves the primary grid (Table T8).
- Tier A (always): SciQ acc, ARC-Easy acc_norm, PIQA acc_norm, LAMBADA-OpenAI acc (ppl logged); Tier B (enters iff C0 clears max(5 pts, 3·SE_item) at the smallest retained S): OpenBookQA, SocialIQA, HellaSwag. Zero-shot, lm-eval 0.4.13 pinned, no chat template, seed 1234, batch 64, bf16, final checkpoint only.
- Co-primary continuous outcomes [choice]: held-out NLL on human-written instruction and QA text (Dolly, OASST1, NQ-open; response tokens, rendered with the training markers; also excluding template tokens); per-task loss (bits per byte of the correct continuation); HeldOut-Real-ID and OOD NLL as forgetting diagnostics.
- Secondary suite: ARC-Challenge, Winogrande, BoolQ (versus majority), MMLU-continuation (STEM and full), CommonsenseQA-cloze, BLiMP, TruthfulQA-MC2 (report), GSM8K (1B, gated), IFEval-lite (report), TriviaQA-3k, 5-shot ICL probe, 200-item format probe, teacher-validation NLL matrix (diagnostic only).
- Contamination audit [choice, mandatory before any teacher comparison]: item-level rate ρ_T,b, example-level r_T,b, paraphrase rate ρ^para_T,b (embedding retrieval plus LLM judgment), benchmark-style density; clean-b sensitivity re-evaluation; headline switches to clean-b if ρ^para differs across teachers by more than about 2 points on a primary benchmark (doc 11 §7).
- References: Table T6; Appendix AT2, AT4, AF10, AF11.

### 3.6 Statistical analysis (300 to 400 words)

- Notation: s = log₂(S/25M), x = log₂(T/1B), d = log₂(D/D_syn); per-S fixed intercepts α_S absorb the base run.
- Model sequence (Table T10): M0 α_S; M1 + β₁x; M2 + β₂x²; M3 + γ₁ s·x + γ₂ s·x²; M3cat per-S quadratics (the per-S model for T*); M4 adds δ₁d + δ₂ d·x + δ₃ d·s on the D-sweep union; M2sat α_S + A_S(1 − T^−β) by nonlinear least squares; nested comparisons by F or LR tests, non-nested by AICc.
- Error structure [choice]: per-S residual variance from M3cat for per-S tests; HC3 sandwich SEs for pooled coefficients; Brown-Forsythe across S; Cook's D flag with both analyses reported.
- Nonparametric per-S analyses: isotonic regression with permutation test (any monotone effect) and parametric-bootstrap test (non-monotone); saturation non-inferiority test μ(72B) − μ(14B) < δ, δ = 1.0 point (sensitivity 0.5); Spearman ρ; aligned-rank-transform duplicate of pooled tests.
- Per-S classification [choice]: fit F1 (quadratic in x) and F2 (saturating); interior iff ĉ < 0 with p < 0.05, x_min < x̂* < x_max, upper 95% bound of x̂* below x_max, and AICc(F1) ≤ AICc(F2) + 2; saturating iff the model-based top-step gain has a one-sided 95% upper bound below δ, with T_sat the smallest grid T whose doubling gain bound falls below δ; monotone-unsaturated (T* right-censored at 72B); flat (T effect not significant). Evaluated in the order flat → interior → saturating → monotone. Cluster (seed) bootstrap B = 2000 when r ≥ 3, parametric bootstrap otherwise; item bootstrap as a second CI.
- T* versus S [choice]: if ≥ 3 sizes are interior, WLS log₂ T̂*_S = α s + c with t CI (≤ 3 df) and a bootstrap CI; censored sizes shown as arrows and excluded; α̂ is descriptive (expected half-width about ±0.35 at best, doc 12 §5.4).
- Multiplicity [choice]: one model-level test per hypothesis; per-cell contrasts in six pre-defined families (F-A adjacent teachers, F-B teacher versus C1, F-C teacher versus C0, F-D C1 versus C0, F-E economic versus raw optimum, F-F intermediate-beats-largest confirmation) with Holm within family; all 100 contrasts in Appendix AT3 with raw and adjusted p, CI and Hedges' g.
- Pre-registration: doc 12 frozen with a recorded hash before the first full-study run; pilot go/no-go rules P1 to P6 and the seed decision rule; every deviation logged (Appendix AT8) and the analysis shown both ways.
- References: Table T10, T11; Appendix AF3, AT3, AT8.

### 3.7 Cost model and Synthetic Transfer Efficiency (220 to 300 words)

- Measured generation cost [choice]: τ_gen(T) GPU-seconds over all vLLM cards from first request to last response (load and warm-up reported separately); E_gen(T) from 1 Hz power sampling × PUE; unit costs per delivered student token c_τ (GPU-s), c_$ = c_τ·p with p = $2.00 per GPU-hour (sensitivity $0.95 owned, $3.00), c_E (kWh), c_F = 2·T·(prompt + response tokens)/D_pool (FLOP, hardware-independent); cell cost C_gen(S,T) = c(T)·D_syn.
- Gains: G₀ = M(S,T) − M(S,C0); G₁ = M(S,T) − M(S,C1) (the economically relevant counterfactual, real tokens cost ≈ 0).
- Primary efficiency metric [choice]: STE(S,T) = G₁(S,T)/(c_$(T)·D_syn), points per dollar, with STE_τ and STE_F as hardware-free variants; a ratio of a concave gain over cost is maximised at the cheapest teacher, so STE is reported, not used for selection.
- Selection tools: marginal efficiency mᵢ(S) = ΔG₁/ΔC_$ between cost-adjacent teachers; price-of-a-point curve T$(S; λ) = argmax_T [G₁ − λ·C_$] over λ ∈ [0, 10] points per dollar, reported as a step function with bootstrap selection probabilities (stable if ≥ 0.8); λ₁(S) = marginal efficiency of the last upgrade into T*; ε-rule T$_ε = cheapest T with G₁ ≥ (1 − ε)·G₁(T*), ε = 0.10.
- H5 criterion (doc 00 §0.13): supported at S iff the 95% CI of λ₁(S) lies below λ_ref(S) = G₁(S,T₁)/C_$(S,T₁), the average efficiency of the cheapest teacher; Holm over the five students. [CHECK: doc 12 §5.5 states H5 as a non-inferiority test P̂(S,T*) − P̂(S,T_econ) < δ; the paper reports the doc 00 criterion as primary and the doc 12 form as secondary; log in AT8.]
- Bootstrap: pooled-variance parametric bootstrap on gains nested with item resampling, chunk resampling on cost, B = 5,000; price not bootstrapped, handled by sensitivity.
- Iso-cost analysis [choice, co-reported]: P(S,T,D) interpolated from the D-sweep (M4 or P_∞ − B·D^−γ) and evaluated at D(T) = B/c(T) for a grid of budgets B, restricted to D ≤ 600M (no extrapolation beyond the delivered pool).
- References: Table T13, T14; F5 to F8; Appendix AT9, AF9.

---

## 4 Pilot results (400 to 600 words; all numbers [placeholder])

- Pilot design (doc 00 §0.10): S ∈ {25M, 100M} × {C0, C1, C1b, T1.5B, T7B, T32B} × 3 base seeds × 3 phase seeds = 54 branches per size; evaluated on the full suite plus continuous outcomes. [CHECK: doc 12 §7 additionally lists a 4-level D sweep at (100M, 7B) with 2 seeds; state whether it was run or superseded by the 36-run D-sweep of doc 00 §0.9.]
- Table T7: nested variance decomposition σ̂²_base, σ̂²_base×T, σ̂²_order per outcome (P_cc, task loss, held-out NLL) with df; the implication for cross-S SEs stated in one sentence.
- Table T8: floor-gate table at 25M and 100M (a_k^{C0}, c_k, SE_item,k, 3·SE, SD_seed,k, retained?) and the resulting K; whether the 25M row stays in the primary grid.
- Table T9: go/no-go outcomes P1 to P6 (detectable synthetic effect, noise bound σ̂_agg^upper ≤ 0.9, floor, spread ≥ 1.0 at 100M, dose, integrity) and the seed decision (σ̂^upper ≤ 0.6 → r = 3; 0.6 to 0.9 → r = 4 with tier 2 +2; > 0.9 → variance reduction and re-pilot).
- Pooled synthetic-versus-C1 contrast per S [placeholder ± CI], teacher spread at 100M [placeholder], and, if run, the dose response at (100M, 7B) [placeholder].
- Statement that pilot data are not pooled into the main inference (doc 12 §7) and where the pilot curves are shown (Appendix AF4).
- Any redesign trigger that fired and the action taken (cross-referenced to Appendix AT8).
- References: Tables T7, T8, T9; Appendix AF4.

---

## 5 Main results by hypothesis (1,200 to 1,600 words; all numbers [placeholder])

### 5.0 Overview (150 to 200 words)

- F1 (S × T heatmap of G₁ = P_cc − P_cc(C1), second panel for held-out NLL), F2 (per-S transfer curves, P_cc), F3 (per-S curves, held-out NLL), Table T10 (coefficients), Table T11 (per-S classification).
- Report first whether fresh tokens help at all: family F-D (C1 − C0 per S) and F-B (each teacher versus C1, Holm within S); a teacher can be "best" while below C1 [lit: Honda 2026].
- Seed accounting sentence: which cells have r = 3 versus 5, which seeds were used for selection versus confirmation (Appendix AT10).

### 5.1 H1: larger teachers monotonically improve students (150 to 220 words)

- Falsifiable statement: P_cc(S,T) is strictly increasing in T at every S.
- Pre-registered tests: pooled M1 versus M0 F-test on β₁ (HC3); per-S isotonic test, statistic RSS(constant) − RSS(isotonic) with the within-S permutation null; per-S Spearman ρ between cell means and log T; adjacent-teacher family F-A with Holm within S. [CHECK: doc 13 §E.4 lists Jonckheere-Terpstra per S as the H1 test; reconcile with doc 12 §4.3 before freezing, and log the choice in AT8.]
- Prior expectation stated as [lit]: strict monotonicity is refuted in every neighbouring regime; the informative quantity is the saturation point.
- Report: β̂₁ [CI], isotonic p per S, Spearman ρ per S, number of adjacent contrasts with adjusted p < 0.05 and their direction; the held-out NLL duplicate.
- References: F2, F3, T10, T11, AT3.

### 5.2 H2: returns saturate beyond some T (150 to 220 words)

- Falsifiable statement: the gain from doubling T falls below δ = 1.0 chance-corrected point above some T_sat(S).
- Pre-registered tests: one-sided β₂ < 0 in M2 (HC3); per-S non-inferiority test H0: μ(72B) − μ(14B) ≥ δ versus H1: < δ (equivalently the 90% two-sided CI upper bound below δ; sensitivity δ = 0.5); model-based Δ_top = b(x₆ − x₅) + c(x₆² − x₅²) with delta-method or bootstrap SE; AICc of M2sat versus M3cat.
- Report: β̂₂ [CI] in points per octave², per-S T_sat with the doubling-gain bound, the AICc difference, the NLL duplicate.
- Prior expectation [lit]: saturation between roughly 3B and 14B for QA-style generation; must be distinguished from a small positive slope with the CI, not by eye.
- References: F2, F3, T10, T11.

### 5.3 H3: an intermediate teacher beats the largest for a constrained student (180 to 250 words)

- Falsifiable statement: at some S there exists T₀ < 72B with P_cc(S,T₀) − P_cc(S,72B) > 0.
- Pre-registered tests: per-S interior classification (four conditions on F1: ĉ < 0 with p < 0.05, interior vertex, vertex CI upper bound below x_max, AICc(F1) ≤ AICc(F2) + 2); non-monotone isotonic test RSS(isotonic) − RSS(cell means) with parametric bootstrap; confirmation contrast (T₀ − 72B) estimated on tier-2 seeds only with a 95% CI excluding 0 (family F-F, Holm over the number tested). Tier-1 seeds select, tier-2 seeds confirm (winner's-curse protection, doc 12 §5.3).
- Report per S: classification label, T̂* with CI where interior, the confirmation contrast [placeholder ± CI], the item-bootstrap CI.
- Prior expectation [lit]: plausible for 25M to 100M; mechanism expected to be compatibility, not size (Xu 2024; GRACE; Li 2025).
- Any sentence of the form "T₀ beats 72B at S" appears only if both tests pass (doc 12 §8.3).
- References: F2, T11, AT3, AT10.

### 5.4 H4: T*(S) follows a predictable relation with S (180 to 250 words)

- Falsifiable statement: the argmax_T of τ_S(T) increases with S; the alternative of Zhang et al. 2025 (linear T*(S), logit channel) is the hypothesis under test, not an assumption.
- Pre-registered tests: interaction test M3 versus M2 (2 df) and M3cat versus M2 (8 df), and M3cat versus M3 (6 df, adequacy of the log-linear interaction); M3 refit with and without the D-sweep cells (dose confound); then, only if ≥ 3 sizes are classified interior, WLS log₂ T̂*_S = α s + c with t CI (≤ 3 df) and bootstrap CI; censored sizes plotted as arrows and excluded.
- Capability covariate: M1z to M3z with z_T in place of x; AICc comparison; T* in capability units; teacher-loss-space fit as in Busbridge et al. 2025 (secondary).
- Report: γ̂₁, γ̂₂ [CI]; F-test p-values; α̂ [CI] or the statement that α is not identifiable because fewer than 3 sizes are interior; F4.
- Vocabulary: "log T* increases with log S with slope α̂ [CI]" as a descriptive summary of five estimates; never "T* ∝ S^α".
- References: F4, T10, T11, AF6.

### 5.5 H5: the cost-optimal teacher differs from the performance-optimal teacher (120 to 180 words)

- Falsifiable statement: T$(S; λ) ≠ T*(S) for λ in the range where the last upgrade into T* is worth less per marginal dollar than the cheapest teacher yields per average dollar.
- Pre-registered test (doc 00 §0.13): the 95% bootstrap CI of λ₁(S) lies below λ_ref(S); Holm over 5 S (family F-E); secondary non-inferiority form of doc 12 §5.5 reported alongside [CHECK, see §3.7].
- Report per S: T*, T$_ε, λ₁(S) [CI], λ_ref(S), supported or not (Table T14); the full frontier is deferred to §7.
- Prior expectation [lit]: nearly certain once c(T) grows with T and P(T) saturates (Bansal 2024; Niklaus 2026); the contribution is the magnitude and its dependence on S with measured cost.
- References: T14, F6, F7.

### 5.6 Secondary outcomes and robustness (200 to 300 words)

- Equal-examples secondary analysis and length-matched ablation (AF5): does the ranking change; KS distances achieved.
- Task loss and held-out NLL as duplicates of every P_cc effect (direction check); correlation between ΔNLL and ΔP_cc reported.
- Paired-item bootstrap CIs for headline contrasts; aligned-rank-transform duplicates; Cook's D exclusions.
- Prompt-token fraction covariate (Amendment A); C1b (format versus provenance) and C1c (human instruction tokens at 20M) contrasts.
- Cross-family anchor (Llama-3.1-8B-Instruct) at two S; the 0.5B rung and the 7-rung fit.
- Knowledge-heavy versus reasoning/commonsense benchmark split [lit: He et al. 2026]; per-benchmark small multiples (AF1).
- Cross-checks X1 to X5 as rank agreement (AF7).
- References: AF1, AF5, AF6, AF7, AT3.

---

## 6 Data-property analysis and mediation (500 to 700 words; exploratory, pre-registered)

- Table T12: per-teacher covariates (yield, truncation, refusal, non-English, CJK, degeneration, near-duplicate, format-fail, contamination flag rate, ℓ_T, N_T, compliance, correctness c_T in-pool and V-set, ℓ_S(T) for each S, NLL_ref(T), distinct-4, MTLD, self-BLEU-4, mean pairwise cosine, cluster count, coverage, FK grade, steps, mean logprob, prompt-token fraction).
- F10: mediator-versus-gain scatter for the core mediators (ℓ_S(T), c_T, diversity, length, z_T), points coloured by S.
- Mediation models (doc 09 §4.3, renamed Med-0 to Med-4 in the paper to avoid a clash with the regression sequence M0 to M4 [CHECK naming]): Med-0 size only; Med-1 learnability ℓ_S(T) + ℓ_S(T)²; Med-2 correctness; Med-3 size + ℓ_S(T); Med-4 diversity; fitted across the 30 cells with student fixed effects; leave-one-teacher-out CV error (Table T15).
- Pre-registered tests: (i) Med-1 beats Med-0 in LOTO-CV and β₂ < 0 (inverted U in student-base NLL, the falsifiable form of "capacity gap"); (ii) in Med-3, b indistinguishable from 0 once ℓ_S(T) is included; (iii) argmax_T of the Med-1 fit shifts with S.
- Within-teacher quartile test (7B teacher, students 25M and 250M, quartiles of per-example ℓ_S at D_syn/4): causal support for the mediator that the cross-teacher regression cannot give.
- Interpretation caveats: low ℓ_S(T) can mean "learnable" or "uninformative" (the quadratic term and the gap ℓ_S(T) − NLL_ref(T) separate these); at most two mediators per model with six teacher levels; identical decoding does not imply identical effective entropy (entropy reported).
- Contamination interplay: the audit rates from Appendix AT4 enter as a covariate; the clean-b sensitivity result is stated here.
- Relation to the literature: CAR [Xu et al. 2024] and GRACE [Panigrahi et al. 2025] as cheap predictors; Busbridge's teacher-loss framing.
- References: T12, T15, F10, AF2, AF10, AT4, AT5.

---

## 7 Cost-benefit frontier (500 to 700 words; all numbers [placeholder])

- Table T13 and F5: measured c(T) in GPU-s, kWh, $ and PFLOP per Mtoken with chunk-bootstrap CIs, TP and dtype per teacher, throughput; the model-predicted values of doc 13 §E.1 are not shown in the paper (they are planning numbers).
- Observation to test, stated as [hyp] from doc 13: time cost is sub-linear in T below 7B and super-linear above 32B on this hardware while FLOP cost is linear; this is why FLOP-optimal and dollar-optimal teachers can differ; report both units.
- F6: STE(S,T) curves and marginal efficiency mᵢ(S) with λ_ref(S) and λ₁(S) marked; G₁ versus log C_$.
- F7: T$(S; λ) step functions with bootstrap selection probabilities; the ε-rule teacher per S.
- F8: iso-cost curves for S ∈ {100M, 1B} and T ∈ {3B, 14B, 72B} from the D-sweep, D(T) = B/c_$(T), with crossing budgets; Appendix AF9 repeats with FLOP cost and p = $0.95.
- F9: D-sweep dose-response panels; the M4 δ₂ term (does the teacher effect change with dose); the data-scaling form P_∞ − B·D^−γ as secondary.
- Where T* and T$ diverge as a function of S (the H5 magnitude), with the caveat that the frontier is hardware- and price-specific and only the FLOP view is portable.
- Compute and energy of the study itself (Appendix AT9) in one sentence.
- References: F5 to F9, T13, T14, AF9, AT9, AT12.

---

## 8 Limitations (500 to 700 words)

Every item below appears; items 1 to 6 are from doc 00 §0.15, items 7 to 13 from lit_review §7 "must not claim", items 14 onward are disclosed confounds from the locked design.

1. Generality beyond Qwen2.5-Instruct is not established; the cross-family anchor bounds but does not remove the family effect [lit: Pieler et al. 2024].
2. Teacher size is not separated from the post-training recipe; small instruct models may themselves be distilled from larger ones (documented for Qwen3, Gemma 2, Llama 3.2, Ministral 3; status for Qwen2.5 [CHECK]).
3. Teacher size is not separated from teacher capability within one family (near-collinear; the z_T refit has little discriminating power).
4. No scaling law for T*(S): at most a descriptive relation from ≤ 5 points with a CI half-width of about ±0.35 in the best case; undefined where the curve is flat or censored.
5. Nothing about model collapse: single-generation data with the real base retained is not the collapse setting [lit: Gerstgrasser 2024; Schaeffer 2025].
6. Nothing about from-scratch synthetic pretraining (Option A) or early mixing (Option C) beyond cross-check X1 at 25M and 100M [lit: rankings need not transfer across regimes].
7. The paper does not propose "synthetic scaling laws", "distillation scaling laws" or a "law of capacity gap".
8. The paper does not claim to be first to show that a smaller teacher can produce a better student (known since Cho and Hariharan 2019).
9. "T* ∝ S^α" is not claimed as a law under any outcome.
10. No claim of generality across prompt styles or data types (QA versus rephrasing versus CoT); the single-family, single-protocol design strengthens internal validity and prevents any claim about the relative importance of family, prompt and source.
11. Equal-token is not claimed to be the correct control; the iso-cost view is co-reported because it is the practitioner's question [lit: Bansal 2024].
12. The design is off-policy; on-policy distillation may rank teachers differently [lit: Agarwal 2024; Li 2026].
13. The "1B" rung is 1.5B; the 3B rung carries the Qwen Research License; all Qwen2.5 sizes were exposed to 72B-synthesised data during their own training, which may attenuate the size gradient uniformly [CHECK].
14. Tokens per parameter vary from 201 (25M) to 20 (≥ 250M), so training regime is confounded with S; X2 measures the effect on the ranking at 100M only.
15. The base run is an unreplicated block above 100M: cross-S comparisons (S main effect, S × T interaction, T*(S)) are conditional on the five base runs; σ̂²_base from the pilot is reported and inflates cross-S SEs.
16. A phase seed re-draws data order and slot permutation only; the delivered pool is used in full at 600M, so seed variance is a lower bound on run-to-run variance and the study is conditional on one pool per teacher (pool-to-pool variance not estimated).
17. Held-out NLL is tokenizer-dependent (not comparable with public models) and the instruction set rewards style match to human text; it is paired with the discrete suite and both template-inclusive and template-free values are shown.
18. Small students sit near floor on several benchmarks; the primary set K is floor-gated and identical across S; per-benchmark results are in the appendix.
19. Identical decoding settings do not give identical effective sampling entropy across teacher sizes [lit: Dohmatob et al. 2024]; entropy is reported as a covariate, not controlled.
20. Generation cost is specific to the stated hardware, vLLM version, precision and price; only the FLOP unit is portable.
21. The prompt pool, prompt writer and filters were fixed once; different pools may rank teachers differently [lit: Niklaus 2026; SmolLM].
22. FineWeb-Edu is itself LLM-classifier filtered, a mild synthetic supervision in the base [lit: Penedo et al. 2024]; X3 checks a non-Edu base at 100M.
23. Final-checkpoint evaluation only; no checkpoint selection; effects of the annealing schedule are absorbed into the estimand.

---

## 9 Conclusion (200 to 300 words)

- Restate the question as a decision problem and the answer per S in one sentence each, using only the permitted vocabulary [placeholder].
- The two design contributions that survive any result: the matched-real control on a shared annealing branch and the co-reported equal-token and iso-cost views with measured cost.
- What the mediator analysis did or did not show [placeholder] and what would break the collinearity (a second family at two or three sizes).
- Negative results are stated with the same prominence as positive ones (doc 12 §8.8).
- Release: configs, prompt pool, manifests, run-level CSV, analysis code; no claim beyond the stated conditions.

---

## Appendices

| Id | Title | Contents | Target |
|---|---|---|---|
| A | Full cell table (AT1) | Every cell and control: r, mean ± SD of P_cc, every per-benchmark a_k (mean ± SD), task loss, held-out NLLs, σ̂_S; run-level CSV released; seed accounting (AT10) | 1 to 2 pages of tables |
| B | Per-benchmark tables (AT2) and small multiples (AF1) | Accuracy by condition per S with chance, majority and SE_item; secondary suite; ICL and format probes; NLL variants (AF11) | 2 to 3 pages |
| C | Contamination audit (AT4, AF10) | Removal counts per layer (base corpus, prompt pool, responses per teacher); ρ_T,b, r_T,b, ρ^para_T,b, style density; clean-b sensitivity; 200-pair manual audit summary | 1 page plus tables |
| D | Hyperparameters (AT6) | Per size per phase: batch tokens, LR, warmup, decay, weight decay, betas, clip, precision, packing, replay; base-run lengths; evaluation settings; software versions | 1 page |
| E | Prompt templates (AT7) | System prompt, per-category suffixes, ChatML template hash, student-visible rendering, decoding configuration, filter thresholds F1 to F12, prompt-source licences | 1 to 2 pages |
| F | Pre-registration deviations log (AT8) | Date, item, pre-registered value, deviation, reason, affected analyses, both-ways reporting; includes the [CHECK] reconciliations listed in this outline | 0.5 to 1 page |
| G | Compute and energy accounting (AT9, AF12) | GPU-hours by card type and stage (base runs, branches, generation per teacher, evaluation), kWh, $ at $0.95 and $2.00, FLOPs; wall-clock; hardware and software stack | 0.5 to 1 page |
| H | Statistical details (AF3, AT3) | Model diagnostics, the 100-contrast table, bootstrap settings, power simulation summary, pilot variance tables in full, verification status of literature details flagged [CHECK] | 1 to 2 pages |
| I | Synthetic-data statistics and ablations (AT5, AF2, AT12, AT13, AF5 to AF9) | Filter yields, length distributions, prompt-set overlap, mediator covariates in full, D-sweep cell table, decoding and sampling ablations, cross-checks X1 to X5, cross-family anchor, teacher-validation NLL matrix | 2 to 3 pages |

---

## Cross-reference: hypothesis → section → pre-registered test → evidence

| Hypothesis | Section | Primary test (pre-registered) | Secondary / robustness | Figures and tables |
|---|---|---|---|---|
| H1 monotone | 5.1 | M1 vs M0 (F, HC3); per-S isotonic permutation test | Spearman ρ; F-A Holm; NLL duplicate; JT [CHECK] | F2, F3, T10, T11, AT3 |
| H2 saturation | 5.2 | β₂ < 0 (one-sided, HC3); per-S non-inferiority μ(72B) − μ(14B) < δ = 1.0 | δ = 0.5; Δ_top model-based; AICc M2sat vs M3cat | F2, F3, T10, T11 |
| H3 intermediate beats largest | 5.3 | Four-condition interior rule on F1; tier-2 confirmation contrast (T₀ − 72B) | Non-monotone isotonic test; item bootstrap | F2, T11, AT3, AT10 |
| H4 T*(S) relation | 5.4 | M3 vs M2 (2 df), M3cat vs M2 (8 df); WLS α̂ if ≥ 3 interior sizes | M1z to M3z; teacher-loss space; with/without D-sweep | F4, T10, T11, AF6 |
| H5 cost ≠ performance optimum | 5.5, 7 | CI of λ₁(S) below λ_ref(S), Holm over S | Non-inferiority form (doc 12 §5.5); ε-rule; price sensitivity; FLOP unit | F5 to F8, T13, T14, AF9 |
| Mediation (exploratory) | 6 | Med-1 vs Med-0 LOTO-CV, β₂ < 0; b ≈ 0 in Med-3 | Within-teacher quartile test; clean-b | F10, T12, T15 |
