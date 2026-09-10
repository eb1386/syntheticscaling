# 10. Student-Training Methodology

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- **Amendment 1 (base tokens, supersedes §2.3 D_base column):** D_base(S) = max(20·N_tot, 5B) drawn as nested prefixes of one seeded shuffle: 5B / 5B / 5B / 10B / 20B for 25M / 100M / 250M / 500M / 1B (doc 00 §0.4). Recomputed schedule with the locked batches (0.25M / 0.5M / 0.5M / 1M / 1M tokens):

| S | D_base | base steps | warmup steps (5%, cap 1000) | D₂ | phase-2 steps | eval every (steps) |
|---|---|---|---|---|---|---|
| 25M | 5B | 19,073 | 954 | 0.8B | 3,052 | 80 |
| 100M | 5B | 9,537 | 477 | 0.8B | 1,526 | 40 |
| 250M | 5B | 9,537 | 477 | 0.8B | 1,526 | 40 |
| 500M | 10B | 9,537 | 477 | 0.8B | 763 | 20 |
| 1B | 20B | 19,073 | 954 | 0.8B | 763 | 20 |

- **Amendment 2 (learning rates):** the rule η_peak = 2e-3·(N_ne/25M)^−0.4 is evaluated at the locked non-embedding counts (12.59M, 75.5M, 216.3M, 462.5M, 956.4M) → **2.6e-3, 1.3e-3, 8.4e-4, 6.2e-4, 4.7e-4**; the pilot LR sweep at 25M/100M (×{0.5, 1, 2}) re-fits the exponent before the 250M–1B base runs start.
- **Amendment 3 (tokenizer, supersedes §2.2):** Llama-2 32k SentencePiece, not a custom BPE (doc 00 §0.5). Special tokens: the tokenizer's BOS/EOS only.
- **Amendment 4 (rendering, supersedes the 'plain Question:/Answer:' remark in §2.9):** Alpaca-style markers per doc 00 §0.7 Amendment B. Full-sequence loss and all-token counting in §2.9 are adopted as written.
- **Amendment 5 (D-sweep):** the fixed-D₂ synthetic-fraction sweep in §2.3 is adopted with levels {75M, 150M, 300M} at S ∈ {100M, 1B} × T ∈ {3B, 14B, 72B} × 2 seeds (doc 00 §0.9).
- Pilot seeds: 3 base × 3 phase seeds at 25M and 100M for arms {C0, C1, C1b, T1.5B, T7B, T32B}; cross-checks X1–X5 are ranked in docs/15.
- Open issue 4 (weight-decay timescale): decision = keep λ = 0.1 uniform and report τ = 1/(ηλ) per size.

---

# Student-training methodology and protocol

Workstream: STUDENT-TRAINING METHODOLOGY. Inputs: `00_brief.md` (P1–P9). Dependencies on other workstreams are marked **[ARCH]** (architecture: dims, tokenizer, context), **[GEN]** (generation), **[EVAL]** (evaluation).
Status legend used throughout: **fact / lit** = literature-supported, **choice** = methodological decision, **assumption**, **hyp** = hypothesis, **[UNVERIFIED]** = could not be checked against the source in this session.

## 0. Decisions at a glance

| Item | Decision | Section |
|---|---|---|
| Design | **B′ ("WSD-branching annealing design")**: one shared base run per size to a *stable-phase* checkpoint at 20 tok/param; every condition is a branch that runs the *identical* decay phase on D₂ tokens; conditions differ only in the treatment file | §1.4 |
| Phase-2 content | Treatment file (1−ρ)·D₂ tokens + ρ·D₂ replay of already-seen base shards, ρ = 0.25; D_syn := (1−ρ)·D₂ = 600 M student tokens (provisional, P5), D₂ = 800 M | §1.5 |
| Controls | C0 = replay-only cooldown (no new data), C1 = held-out fresh web tokens, C1b = human-written Q&A web text (StackExchange), C1c = small equal-D (≈20 M) human-instruction vs synthetic comparison; C2–C7 = teachers | §4 |
| Estimand | Effect of teacher identity T on the value of D_syn synthetic tokens *as annealing data* for a student of size S at base state θ*(S, 20 tok/param), relative to fresh web tokens under an identical schedule | §1.4 |
| Seeds | Pilot sizes (25 M, 100 M): 3 base seeds × 3 phase-2 seeds; other sizes: 1 base seed × 3 phase-2 seeds; nested variance decomposition reported | §2.10 |
| Schedule | WSD; base = warmup + constant; branches = linear decay to 0 over D₂ (no re-warm because branching happens at peak LR) | §2.5 |
| Optimizer | AdamW β=(0.9, 0.95), ε=1e-8, WD=0.1 decoupled on matrices only, clip 1.0, bf16 autocast / fp32 master | §2.4 |
| LR / batch | 25 M: 2e-3 / 0.25 M tok; 100 M: 1.2e-3 / 0.5 M; 250 M: 8e-4 / 0.5 M; 500 M: 6e-4 / 1 M; 1 B: 4e-4 / 1 M (pilot LR sweep ×{0.5,1,2} at 25 M & 100 M) | §2.4, §2.13 |
| Packing | Concatenate + split at 2048, **intra-document causal masking** (no cross-document attention), per-document position ids | §2.7 |
| Loss | Full-sequence LM loss (prompt + response); D_syn counted as all trained tokens | §2.9 |
| Checkpoint | Final checkpoint after decay; no selection on benchmarks | §2.12 |
| Base corpus | FineWeb-Edu (ODC-By), sample-100BT, 13-gram Bloom-filter decontamination; one plain-FineWeb cross-check base at 100 M | §3 |
| Trainer | Custom minimal PyTorch trainer (nanoGPT/llm.c lineage) with flash-attn varlen (or FlexAttention), torch.compile, fused AdamW, memmap deterministic loader, signed run manifest | §5 |

---

## 1. Part 1 — Design choice

### 1.1 Notation and estimands

- S: student size (non-embedding parameters, see §2.1). T: teacher (identity within the fixed family; its size is the manipulated attribute).
- G: the fixed generation protocol (prompt pool, system prompt, decoding, filtering; P2). F_T = G(T): the synthetic treatment file for teacher T. F_real: matched fresh web tokens. F_hum: human-written Q&A text.
- Π: the fixed student training protocol (everything in §2). Y(·): an outcome metric (held-out NLL, continuous benchmark score; **[EVAL]**).
- Base state θ*(S, D_base): the shared checkpoint after D_base base tokens.

Three different causal quantities are on the table:

| Design | Potential outcome | Estimand for teacher contrast T vs T′ | Reads as |
|---|---|---|---|
| A (pure synthetic from scratch) | Y_A(T) = Y(Π_scratch(F_T, D)) | E[Y_A(T) − Y_A(T′)] | effect of teacher on a **fully synthetic curriculum** of D tokens |
| B / B′ (shared base + synthetic phase) | Y_B(T) = Y(Π_phase2(θ*, F_T)) | E[Y_B(T) − Y_B(T′)]; and vs. real: E[Y_B(T) − Y_B(real)] | effect of teacher on the **marginal value of D_syn synthetic tokens added to base state θ*** |
| C (fixed real mix + synthetic, from scratch) | Y_C(T) = Y(Π_scratch(Mix(F_real,base, F_T), D_base + D_syn)) | E[Y_C(T) − Y_C(T′)] | effect of teacher on a **from-scratch mixture** at synthetic fraction D_syn/(D_base+D_syn) |

All three are legitimate; they answer different questions, and there is no guarantee that argmax_T is the same across them (§1.6 tests this at the cheapest sizes). The research question in the brief ("which T produces the most useful data for a student of size S under controlled data and compute") is agnostic; the practical use case (a lab has a small pretrained model and a generation budget) is B. The 2026 "Finetuner's Fallacy" result (Datology; mixing 1–5 % domain data early in pretraining beats pretrain-then-finetune, arXiv 2603.16177 — **lit, details [UNVERIFIED]**) is a warning that B-rankings need not transfer to C.

### 1.2 Option-by-option analysis

| Criterion | A: pure synthetic pretraining | B: shared base + synthetic continued phase | C: real+synthetic mixture from scratch |
|---|---|---|---|
| Identified quantity | Teacher effect on an all-synthetic curriculum; also the *only* knowledge source is the teacher, so "teacher size" ≈ "world-knowledge coverage" | Marginal value of D_syn synthetic tokens at base state θ*; teacher contrast is a within-base paired comparison | Teacher effect on a mixture; entangled with mixing fraction and with early-vs-late exposure |
| Main confounds | Format mismatch to human-written benchmarks; tokenizer/format effects scale with T; student never sees web text → benchmarks measure teacher regurgitation | θ* shared (no confound), but result is conditional on (D_base, schedule state, ρ, D_syn); prompt tokens identical across T | Synthetic fraction varies with S under fixed D_syn (≈ 3 % at 1 B, ≈ 43 % at 25 M) |
| Floor effects (25 M–1 B) | Severe: TinyStories-style models (1–33 M, 480 M tokens, Eldan & Li 2023) speak coherent English but are at chance on standard MC benchmarks; with D_syn = 600 M tokens a 1 B student would be trained at 0.6 tok/param | Mitigated: base at 20 tok/param provides the floor lift; small models still near floor on ARC-C/OBQA (**[EVAL]** to quantify) | Same base tokens as B, so same floors |
| # full pretraining runs (5 sizes × 6 T + controls, 3 seeds) | 5 × 8 × 3 = 120 from-scratch runs, but at D_syn only (meaningless for ≥ 250 M); to be Chinchilla-sized it needs 20 N synthetic tokens per teacher (1 B: 20 B tokens × 6 teachers — generation-infeasible) | **5 base runs** (+2 extra base seeds at pilot sizes) + 120–135 short branches | 120 from-scratch runs of (D_base + D_syn) each ≈ 24 × cost of one base sweep ≈ 13 k GPU-h (§1.3) — infeasible on 2–4 GPUs |
| Prior work | TinyStories; phi-1 (1.3 B; 6 B filtered web + 1 B synthetic), phi-1.5 (≈ 30 B, mostly synthetic) [lit, from search snippets]; Kang et al. 2025 "Demystifying…" find synthetic-only rephrased pretraining is *not* faster than web and that textbook-style pure-synthetic mixtures show collapse-like patterns | Yang et al. 2024 EntiGraph (continued pretraining of Llama-3-8B on 455 M synthetic tokens with RedPajama replay, ratio 0.1); MiniCPM/SmolLM2/OLMo 2 annealing-phase data injection; Ibrahim et al. 2024, Gupta et al. 2023 | WRAP (Maini et al. 2024; real + rephrased C4 at 350 M/1.3 B); Cosmopedia/SmolLM (FineWeb-Edu-dedup 220 B + Cosmopedia-v2 28 B + Python-Edu 4 B); Kang et al. 2025 (≈ 30 % synthetic optimum); BeyondWeb 2025; MiniPLM (Gu et al. 2024; teacher-refined corpus, students 200 M–1.2 B from scratch) |
| Sensitivity of outcome to synthetic phase | Maximal (everything is treatment) but the outcome is degenerate (floor) | Risk: base dominates and all T look alike. Mitigations: annealing-phase placement (high leverage; MiniCPM/SmolLM2 practice), 20 tok/param base (not over-trained), continuous metrics, D_syn/D_base ≥ 3 %, pre-registered escalation (§1.7) | Diluted for large S; strong for small S |
| Verdict | Reject as primary; cheap 25 M "coherence" sanity study only | **Adopt (B′ variant)** | Reject as primary; run as design cross-check at 25 M (3 seeds) and 100 M (1 seed) |

### 1.3 Cost model (assumption-laden; ±2×)

FLOPs/token ≈ 6·(N_ne + V·d)·1.1 (LM head included because at 25 M the 32 k head is ≈ 40 % of the matmul FLOPs; +10 % attention). GPU: RTX PRO 6000 Blackwell assumed 250 TFLOPS dense BF16 **[assumption, UNVERIFIED]**; assumed MFU 15/25/30/35/40 % for 25 M/100 M/250 M/500 M/1 B (small models are launch/memory bound). Illustrative dims from **[ARCH]** placeholders (d = 512/768/1024/1280/2048).

| S (N_ne) | N_tot = N_ne + V·d (32 k vocab) | D_base = 20·N_tot (tokens) | Base run (GPU-h) | D₂ = 0.8 B branch (GPU-h) | Branches full grid (27 = 9 conditions × 3 seeds) |
|---|---|---|---|---|---|
| 25 M | 41 M | 0.8 B | ≈ 1.6 | ≈ 1.6 | ≈ 43 |
| 100 M | 125 M | 2.5 B | ≈ 9 | ≈ 3 | ≈ 81 |
| 250 M | 283 M | 5.7 B | ≈ 39 | ≈ 5.5 | ≈ 150 |
| 500 M | 541 M | 10.8 B | ≈ 123 | ≈ 9 | ≈ 245 |
| 1 B | 1.06 B | 21.2 B | ≈ 410 | ≈ 16 | ≈ 420 |
| **Total** | | | ≈ 585 (1 seed) + 21 (2 extra base seeds at 25 M and 100 M) | | ≈ 940 |

Total ≈ 1.55 k GPU-h ≈ 32 days on 2 × PRO 6000 (25 M/100 M work fits on the 16 GB RTX 5080s: a 100 M model with 0.5 M-token batches via gradient accumulation needs < 8 GB). The 1 B base run (≈ 17 days on one GPU) is the critical path; start it first. Fallback if compute-limited: 1 B base at 10 tok/param (halves it) — but then tokens/param is no longer uniform across S (see Open issues). Option C at full grid would be ≈ 24 × 585 ≈ 14 k GPU-h; Option A at Chinchilla token counts is dominated by generation (20 B synthetic tokens per teacher for the 1 B student) and is out of reach.

### 1.4 Chosen design B′ and identification statement

**Choice.** For each S: one base run with a warmup-stable-decay (WSD) schedule, *stopped at the end of the stable phase* at D_base = 20·N_tot tokens, saving θ*(S) with optimizer state. Every experimental condition is a *branch* from θ*(S) that runs the same phase-2 protocol Π₂: D₂ tokens, LR decaying linearly from η_peak to 0, batch/precision/packing/seed-indexed ordering identical, replay stream identical. The branch's input is a single "treatment file" of (1−ρ)·D₂ tokens:

| Condition | Treatment file | Replay (ρ·D₂, already-seen base shards, identical stream) |
|---|---|---|
| C0 (base only) | already-seen base shards (second epoch) | yes |
| C1 (matched real) | fresh, held-out base-corpus shards | yes |
| C1b (human Q&A) | StackExchange Q/A pairs (§4) | yes |
| C_T, T ∈ {1.5 B … 72 B} | F_T from teacher T (whole-example subsample to exactly D_syn tokens) | yes |

**Why "stopped short" here means the stable-phase checkpoint, not a small token budget.** The lead's B′ idea (stop at 10–20 tok/param so the synthetic phase has leverage) is adopted as 20 tok/param — Chinchilla-optimal (Hoffmann et al. 2022) is *already* short relative to current small-model practice (SmolLM2-135M: 2 T tokens ≈ 15 k tok/param; 360M: 4 T; MobileLLM 125M/350M: 1 T). Two further sources of leverage come from the WSD structure: (i) the base is branched at peak LR, before the loss reduction that the cooldown delivers (Hägele et al. 2024: WSD matches cosine only after the cooldown), so phase 2 is the model's annealing phase — the phase where SmolLM2, MiniCPM (Hu et al. 2024) and OLMo 2 mid-training deliberately place their highest-quality data because its effect is disproportionate; (ii) the model is not in the "catastrophic overtraining" regime (Springer et al. 2025: OLMo-1B at 3 T tokens fine-tunes > 2 pp worse than at 2.3 T), so plasticity is high. The matched-real control C1 separates "more tokens + cooldown" from "synthetic tokens".

**Identification statement (primary).** Let s index phase-2 seeds and (at pilot sizes) b index base seeds. With θ*(S,b) fixed and Π₂ fixed, the only input that differs between branches is the treatment file. Therefore

  τ_S(T) := E_{b,s}[ Y(Π₂(θ*(S,b), F_T; s)) − Y(Π₂(θ*(S,b), F_real; s)) ]

is the average causal effect of *replacing (1−ρ)·D₂ = D_syn fresh web tokens with D_syn synthetic tokens from teacher T during the annealing phase of a size-S student trained to 20 tok/param on the base corpus*, and

  Δ_S(T, T′) := τ_S(T) − τ_S(T′) = E[ Y(F_T) − Y(F_T′) ]

is the effect of teacher identity, which under the fixed family/recipe of P3 is interpreted as the effect of teacher size. Because F_T = G(T) with G fixed (P2), Δ_S(T,T′) is the *total* effect of T through every channel (content quality, style, length distribution, regurgitation of benchmark-like text, etc.); channel decompositions (length-matched subsampling, equal-examples) are secondary analyses. The estimand is explicitly conditional on: the base corpus, D_base/N = 20, D₂, ρ = 0.25, the linear-to-zero decay, the student tokenizer, and the family Qwen2.5-Instruct. It does **not** identify Y_A or Y_C quantities, nor the effect at other base states; §1.6 measures how much the ranking moves under those changes. H4 ("T*(S)") is the argmax over the tested T-grid of τ_S(T), and its dependence on S is tested only within the grid.

**Why the C0/C1 collapse is avoided.** Under WSD branching, "base only" cannot mean "no further training" (the un-decayed θ* is not a finished model). C0 is defined as *decay on already-seen data* (second epoch of base shards, which by Muennighoff et al. 2023 costs ≈ nothing at < 4 epochs), so the trio C0/C1/C_T is a clean factorial: C1 − C0 = value of fresh web tokens over repeated tokens; C_T − C1 = synthetic vs fresh web; C_T − C0 = synthetic vs nothing new. θ* itself is also evaluated (free) as an un-annealed reference, but it is not a fair comparator.

### 1.5 Pure synthetic phase vs mixture (replay) — choice ρ = 0.25

- **lit**: Ibrahim et al. 2024 (405 M and 10 B; Pile→SlimPajama = weak shift, Pile→German = strong shift): re-warm + re-decay + replay matches full retraining; 5 % replay suffices for weak shift, 25 % for strong shift. Gupta et al. 2023 (410 M): re-warming raises upstream loss initially but improves downstream in the long run; re-warming to the original max LR is the recommended default. Yang et al. 2024 (EntiGraph) used RedPajama replay at ratio 0.1 in synthetic continued pretraining. "Efficient Continual Pre-training by Mitigating the Stability Gap" (Guo et al. 2024) and "Replaying pre-training data improves fine-tuning" (2026, arXiv 2603.04964) point the same way **[details UNVERIFIED]**.
- Instruction/Q&A-style synthetic text vs web pretraining text is a *format* shift stronger than Pile→SlimPajama but weaker than a language switch; we take the conservative 25 %.
- Replay is *not* the treatment, so it is held constant across all conditions (same shards, same order, same slot positions). Because C1 also receives the replay stream, the C_T − C1 contrast is exactly "600 M synthetic vs 600 M fresh web, with 200 M shared replay".
- Why not ρ = 0 (pure synthetic)? It maximises signal but makes forgetting a teacher-correlated artifact: teachers whose outputs are more repetitive or longer push the student further from the base distribution; with pure synthetic data, benchmark losses from forgetting could invert the ranking on web-style held-out NLL. Pilot ablation: ρ ∈ {0, 0.25, 0.5} at 25 M and 100 M for the three pilot teachers (cheap) — if the teacher ranking is ρ-invariant, the point is moot and is reported as such.
- Why replay seen shards rather than fresh shards? So that C0 ("nothing new") is definable and so that the replay component adds no new information that could mask the treatment.

### 1.6 Design-sensitivity cross-checks (pre-registered, cheap)

| Check | What | Cost | Interpretation |
|---|---|---|---|
| X1: Option C at 25 M (3 seeds) and 100 M (1 seed) | From-scratch runs on D_base + D₂ tokens with the treatment file mixed uniformly from step 0, for C1 and the 3 pilot teachers | 25 M: 12 × 3.3 h ≈ 40 h; 100 M: 4 × 12 h ≈ 48 h | Does the teacher ranking depend on early vs late exposure? |
| X2: over-trained base at 100 M | Second base run to 200 tok/param (25 B tokens, ≈ 90 h), then the standard branches for C1 + 3 pilot teachers, 2 seeds | ≈ 115 h | Does base strength change the ranking (external validity to SmolLM2-like regimes)? |
| X3: plain-FineWeb base at 100 M | Base on FineWeb (non-Edu) then same branches (C1 + 3 teachers, 2 seeds) | ≈ 35 h | Is the effect size / ranking sensitive to an "educational" base? (§3) |
| X4: replay fraction ρ ∈ {0, 0.5} at 25 M, 100 M | as §1.5 | ≈ 25 h | Forgetting artifacts |
| X5: phase-2 peak LR ∈ {0.5, 0.25} × η_peak at 100 M | 3 teachers × 2 seeds | ≈ 40 h | Is the ranking schedule-sensitive? |

### 1.7 Pre-registered escalation rule against "base dominates"

After the pilot at 100 M: let Δ̂ = mean over pilot teachers of (C_T − C1) on the primary continuous metric and let SE be the phase-2-seed standard error of that difference (paired over seeds). If |Δ̂| < 2·SE **and** the pairwise teacher contrasts are also < 2·SE, the phase-2 leverage is insufficient; the pre-registered remedies, in order, are: (1) increase D_syn to 1.2 B (requires **[GEN]** pool of 1.2 B per teacher), (2) reduce D_base to 10 tok/param for all sizes, (3) reduce ρ to 0.1. The rule is fixed *before* seeing full-grid results and applies uniformly across sizes.

---

## 2. Part 2 — Training controls

### 2.1 Definitions

- **N_ne**: non-embedding parameters; the nominal sizes {25 M, 100 M, 250 M, 500 M, 1 B} refer to N_ne (Kaplan convention) **[ARCH to confirm; if the architecture workstream defines sizes as total parameters, the LR/batch table is keyed by the nearest N_ne]**. **N_tot** = N_ne + V·d (tied embeddings). Token budgets use N_tot (Hoffmann convention: D_base = 20·N_tot) — **choice**; either convention is fine if used consistently, and the difference is < 10 % except at 25 M (where it is ≈ 60 %; decide once, document).
- **D_base**: base tokens to θ*. **D₂**: phase-2 tokens per branch. **D_syn = (1−ρ)·D₂**: treatment tokens, counted in student-tokenizer tokens *including* prompt, separator, BOS/EOS tokens (§2.9). Tokens are counted after packing, so every branch has exactly the same number of trained tokens.
- **Seed** s: phase-2 seed (data-order permutation, packing composition, RNG stream). **Base seed** b: initialization + base data order.

### 2.2 Tokenizer (interface with P4/[ARCH])

**Choice**: one 32 k byte-level BPE trained with HF `tokenizers` on 10 B tokens of the base-corpus *train* shards (digits split individually, no byte-fallback failures, `<|bos|>`, `<|eos|>`, `<|sep|>` special tokens), frozen and hash-pinned **before any synthetic data is generated**. Rationale: (i) no third-party license entanglement (Llama-2's 32 k SentencePiece is under the Llama 2 Community License); (ii) same vocabulary for all sizes; (iii) D_syn is defined in this tokenizer (P2/P4). Tied input/output embeddings at every size (P4: one rule). Known bias: a base-corpus-trained tokenizer fragments synthetic/Q&A text slightly more than web text; this affects all teachers equally and is absorbed by the "count in student tokens" rule. If **[ARCH]** shows that 32 k × d dominates the 25 M model, fall back to 16 k for *all* sizes (never mix vocab sizes across S).

### 2.3 Token budgets and steps per phase

Context L = 2048 **[ARCH assumption]**. Effective batch B (tokens) from §2.6.

| S | D_base = 20·N_tot | B (tokens; seqs × 2048) | Base steps | Warmup steps (5 % of base, cap 1000) | D₂ | Phase-2 steps | Eval every (steps) |
|---|---|---|---|---|---|---|---|
| 25 M | 0.8 B (N_tot ≈ 41 M) | 0.25 M (128) | 3,200 | 160 | 0.8 B | 3,200 | 80 |
| 100 M | 2.5 B (N_tot ≈ 125 M) | 0.5 M (256) | 5,000 | 250 | 0.8 B | 1,600 | 40 |
| 250 M | 5.7 B (N_tot ≈ 283 M) | 0.5 M (256) | 11,300 | 565 | 0.8 B | 1,600 | 40 |
| 500 M | 10.8 B (N_tot ≈ 541 M) | 1 M (512) | 10,800 | 540 | 0.8 B | 800 | 20 |
| 1 B | 21.2 B (N_tot ≈ 1.06 B) | 1 M (512) | 21,200 | 1,000 | 0.8 B | 800 | 20 |

(N_tot values are placeholders pending **[ARCH]**; the rule is what matters.) Note the asymmetry: D₂/D_base = 100 % at 25 M and 3.8 % at 1 B. This is a direct consequence of P5 (fixed D_syn across S, forced by generation cost) and is the main threat to interpreting T*(S) (Open issue 1). The nested D-sweep is implemented *at fixed D₂* as a synthetic-fraction sweep: D_syn ∈ {100, 200, 400, 600} M with the remainder of the treatment slots filled from fresh held-out shards — schedule and step count unchanged, no extra generation.

### 2.4 Optimizer and learning rate

**Optimizer (all phases, all sizes)**: AdamW, β₁ = 0.9, β₂ = 0.95, ε = 1e-8, weight decay 0.1 (decoupled, PyTorch semantics: p ← p − lr·wd·p), applied to 2-D weight matrices only — **not** to RMSNorm gains, biases (none), or the tied embedding/unembedding (OLMo 2 excludes embeddings from WD to stop embedding norms shrinking; llm.c/nanoGPT apply WD to all ≥ 2-D tensors including embeddings — we follow OLMo 2). Gradient clipping at global-norm 1.0. No dropout. Fused AdamW kernel. Z-loss and QK-norm are **[ARCH]** decisions (OLMo 2 uses both; at ≤ 1 B with these LRs they are optional; if adopted, adopt for all sizes).

**Verified reference LRs (peak) and batch sizes** (all AdamW, decoder-only):

| Model | Peak LR | Batch (tokens) | Schedule | Source |
|---|---|---|---|---|
| Pythia-70M / 160M / 410M / 1B / 1.4B | 1e-3 / 6e-4 / 3e-4 / 3e-4 / 2e-4 | 2 M (uniform in the current suite; 160M/410M/1.4B were 4 M in v0) | cosine (to 10 % of max **[UNVERIFIED]**) | Biderman et al. 2023, Table 1 (https://arxiv.org/abs/2304.01373) |
| GPT-2 124M (llm.c reproduction) | 6e-4 (3× higher also fine per Karpathy) | 0.5 M (524,288) | linear warmup + cosine; WD 0.1; β=(0.9,0.95); clip 1.0 | https://github.com/karpathy/llm.c/discussions/481 |
| MobileLLM-125M/350M | 2e-3 | 32 × 32 GPUs × 2 k ≈ 2 M | cosine; Adam WD 0.1; 480 k iters on 1 T tokens | Liu et al. 2024 (https://arxiv.org/abs/2402.14905) |
| SmolLM-135M/360M (v1) | 3e-3 | 1 M | trapezoidal (WSD), 20 % cooldown; 600 B tokens | HF SmolLM blog (https://huggingface.co/blog/smollm) |
| SmolLM2-1.7B | 5e-4 | ≈ 2 M, seq 2048 | WSD, 2000 warmup steps, 10 % decay | Allal et al. 2025 (https://arxiv.org/abs/2502.02737) |
| SmolLM2-135M/360M | [UNVERIFIED — paper says WSD with 20 % decay; the peak LR value was cut off in the retrieved snippet; likely 3e-3 as in v1] | | | same |
| OLMo-2-1B | 4e-4 | 512 × 4096 ≈ 2.1 M | 2000-step warmup, cosine to 10 %; β=(0.9,0.95), ε=1e-8, WD 0.1 not on embeddings, init N(0, 0.02) truncated | OLMo team 2025 (https://arxiv.org/abs/2501.00656) |
| DeepSeek LLM fit (C in FLOPs) | η_opt = 0.3118·C^−0.125; B_opt = 0.2920·C^0.3271 | → at C = 1e17: η ≈ 2.3e-3, B ≈ 0.1 M; at C = 1e20: η ≈ 1e-3, B ≈ 1.0 M | multi-step | Bi et al. 2024 (https://arxiv.org/abs/2401.02954) |

**Chosen peak LRs** (with the batch sizes of §2.6), from the envelope of the above plus a muP-flavoured width rule (hidden-layer LR ∝ 1/width; Yang et al. 2022 Tensor Programs V **[lit, not re-verified here]**):

  η_peak(N_ne) = 2e-3 · (N_ne / 25 M)^−0.4   →  25 M: 2.0e-3, 100 M: 1.2e-3, 250 M: 8e-4, 500 M: 6e-4, 1 B: 4e-4.

The exponent −0.4 sits between the DeepSeek compute-fit (η ∝ C^−0.125 ≈ N^−0.25 at Chinchilla) and the width rule (N ∝ d² at fixed depth → η ∝ N^−0.5). **Pilot LR sweep** (base phase only, 25 M and 100 M): η ∈ {0.5, 1, 2} × η_peak, 1 seed each, choose by held-out loss at end of a short cooldown; then re-fit the exponent through the two selected points and apply to 250 M–1 B without further tuning (Porian et al. 2024 show scale-dependent tuning of LR/warmup/β₂ matters for small-scale scaling fits; we tune where it is cheap). LR is tuned on the *base* objective only — never on phase-2 outcomes, to avoid selecting hyperparameters that favour a teacher.

**Weight-decay timescale caveat** (Bergsma et al. 2025 "Power Lines": the AdamW EMA timescale τ = 1/(η·λ) steps should be a fixed fraction of the run): with λ = 0.1, τ = 5,000 steps at 25 M (run ≈ 6,400 steps) vs 25,000 steps at 1 B (run ≈ 22,000). Uniform λ = 0.1 is standard and kept for comparability; it is listed as an open issue.

### 2.5 Schedule: WSD with branch-at-peak

- **Base**: linear warmup over W steps (table §2.3) from 0 to η_peak, then constant η_peak until D_base. Save θ* = (weights, AdamW moments, data-loader position, RNG state).
- **Phase 2 (every branch)**: **no re-warm** (the branch starts at η_peak); LR decays *linearly to 0* over the entire D₂ ("Straight to Zero", Bergsma et al. 2025; Hägele et al. 2024 find linear or 1−√ cooldowns of ≈ 20 % match cosine; Hägele et al. also report that longer cooldowns do not hurt **[UNVERIFIED from memory]**). With D₂ = 0.8 B, the cooldown fraction is 50 % of total tokens at 25 M and 3.6 % at 1 B — different, but identical across conditions within a size.
- Why WSD rather than cosine-to-completion + re-warm: (i) it is exactly the "infinite LR schedule" recommended by Ibrahim et al. 2024 for continual pretraining; (ii) it removes the re-warm loss spike (Gupta et al. 2023) as a nuisance; (iii) θ* is reusable for every condition and for the D-sweep; (iv) it mirrors how practitioners inject high-quality data (annealing).
- **Fallback recipe if a cosine base is imposed**: re-warm linearly over 5 % of phase-2 steps (≥ 20 steps) to the *full* η_peak (Ibrahim et al. 2024: re-warming to the original max LR gives the best adaptation; replay handles the forgetting), then linear decay to 0.

### 2.6 Effective batch size

**lit**: Zhang et al. 2024/ICLR 2025 ("How Does Critical Batch Size Scale in Pre-training?", 85 M–1.2 B on C4): CBS scales primarily with data size, not model size; Chinchilla-regime fit B* ≈ 93.2·N^0.47 (tokens **[units as retrieved; UNVERIFIED]**), giving ≈ 0.28 M (25 M), 0.54 M (100 M), 0.82 M (250 M), 1.1 M (500 M), 1.6 M (1 B). DeepSeek fit gives 0.1–1 M over the same compute range. Ai2's OLMo large-batch study **[content UNVERIFIED]** is consistent with ≥ 1 M being fine at 1 B.

**Choice**: 0.25 M (25 M), 0.5 M (100 M, 250 M), 1 M (500 M, 1 B) — at or below CBS (efficient regime) while keeping ≥ 800 optimizer steps in phase 2 and ≥ 3,200 in the base. **Batch is held constant across phases within a size** (one fewer confound; keeps η/B and τ consistent). Gradient-accumulation micro-batch is fixed per (size, GPU model) and identical across conditions — bf16 reduction order depends on it.

### 2.7 Sequence packing, masking, context

**Choice**: pre-tokenize documents (`<|bos|>` doc `<|eos|>`), concatenate in the seeded order, split into 2048-token sequences (documents may straddle a boundary; the continuation gets no access to its prefix), and train with **intra-document causal masking** (block-diagonal within each 2048 window; flash-attn `varlen` with `cu_seqlens`, or FlexAttention block mask) and **document-relative position ids** (RoPE positions restart at 0 for each document).

Justification: (a) Zhao et al. 2024 (ACL) show cross-document attention under plain causal masking injects distracting context and hurts LM and downstream performance; Llama 3 (Dubey et al. 2024) trains with a mask preventing attention across documents. (b) Treatment-specific reason: synthetic Q&A examples are short (hundreds of tokens), so a 2048 window contains many documents in phase 2 but few in the base or C1; without masking, the amount of cross-document leakage would *co-vary with the teacher's response-length distribution* — a packing artifact correlated with treatment. Intra-document masking removes it. (c) No padding, so trained-token counts are exact. Best-fit-decreasing packing to reduce straddling (Ding et al. 2024 **[UNVERIFIED]**) is optional; straddling is rare in phase 2 because synthetic docs are < 2048 tokens.

### 2.8 Precision

bf16 autocast for matmuls/attention, fp32 master weights and AdamW states, fp32 RMSNorm statistics and softmax (SDPA/flash-attn do this internally), fp32 loss reduction. No fp8 for students (Blackwell fp8/fp4 would add a numerics confound across sizes). Same GPU model for all branches of a given size (bf16 kernels differ across GPU generations).

### 2.9 Loss masking and the definition of D_syn

**Choice**: full-sequence next-token loss on prompt + response (+ separator/EOS), i.e., the pretraining objective. **D_syn is the number of trained tokens in the treatment file, prompts included** (= treatment slots × 2048 exactly).

Reasons: (1) The prompt pool and system prompt are identical across teachers (P2), so prompt tokens are a *constant* component of every synthetic condition and cannot confound Δ(T,T′); (2) matched-real and human-Q&A controls have no prompt/response split, so a response-only objective would make phase 2 a different *objective* in synthetic vs control branches — a confound; (3) Shi et al. 2024 (NeurIPS) show including instruction loss helps when prompts are long relative to responses, and Huerta-Enochian & Ko 2024 (EMNLP) find small non-zero prompt-loss weights beneficial on short-completion data — so full loss is not harmful; (4) it keeps "tokens trained" = "tokens processed" (compute-matched). Consequence: under whole-example subsampling to fixed D_syn, teachers with longer responses contribute fewer examples and hence fewer prompt tokens (prompt fraction varies, e.g., 15–25 %). This is reported per cell, and P2's secondary analyses (equal examples; length-matched subsampling) plus a response-token-matched re-analysis address it. **[GEN]** should keep prompts short (≤ ~100 tokens) and formatting minimal (plain "Question:/Answer:" style, no chat template, one document per example) to keep the prompt fraction small.

### 2.10 Seeds and variance decomposition

What a phase-2 seed controls: the permutation of treatment sequences into treatment slots, the (shared) permutation of replay sequences into replay slots, the slot pattern itself (which steps get replay), and the CUDA RNG stream (no dropout, so effectively nothing else). It controls **nothing about initialization** — all branches share θ*. Hence phase-2-seed variance = *data-order variance*, a lower bound on run-to-run variance.

Model: Y_{bTs} = μ + α_T + β_b + (αβ)_{Tb} + ε_{bTs}.
- **Pilot sizes (25 M, 100 M)**: b ∈ {1,2,3} base seeds (init + base data order; 3 base runs per size: ≈ 5 h at 25 M and ≈ 27 h at 100 M in total), × conditions {C0, C1, T₁, T₂, T₃} × s ∈ {1,2,3} → 45 branches per size. Estimate σ²_β (base seed), σ²_{αβ} (base-seed × teacher interaction), σ²_ε (order), by nested ANOVA / mixed model.
- **Other sizes**: b = 1, s ∈ {1,2,3}. Teacher contrasts are paired within base seed, so β_b cancels; what does *not* cancel is (αβ)_{Tb}. The honest standard error for generalising Δ(T,T′) to other base seeds is SE = sqrt(2·(σ̂²_{αβ} + σ̂²_ε/3)), with σ̂²_{αβ} borrowed from the pilot sizes (stated assumption: interaction variance does not grow with S). If the pilot finds σ̂²_{αβ} ≳ ¼ of the teacher effect variance, add base seeds at 250 M (≈ 40 h each) before the full grid.
- Seeds are used with **common random numbers**: seed s produces the same slot pattern and the same replay ordering in every condition, and treatment sequences are placed by a seeded permutation of *slot indices*, so paired differences across conditions at equal s share as much nuisance as possible.
- Benchmark noise: Madaan et al. 2024 show seed variance of ≈ 1 pp on standard MC benchmarks even at 7 B and recommend continuous metrics; at 25 M–250 M the binomial noise of ARC-C (1,172 items at ≈ 25–30 % accuracy: ± 1.3 pp) alone exceeds plausible teacher contrasts. Therefore every eval point records continuous metrics (held-out NLL; mean log-prob of the correct choice) as primary **[EVAL]**.

### 2.11 Data ordering and the "only difference is the data file" guarantee

Procedure (executed by one script, `build_branch.py`, whose output hash is recorded in the manifest):
1. Inputs: `treatment.jsonl` (documents), replay shard list (fixed per size), tokenizer hash, seed s, D₂, ρ, L = 2048.
2. Tokenize treatment documents (`<|bos|>` … `<|eos|>`), concatenate in *prompt-id order* (prompt ids are shared across teachers; for C1/C1b the document id plays this role), split into 2048-token sequences; **truncate the list to exactly n_treat = (1−ρ)·D₂ / L sequences** (surplus dropped; a shortfall is a hard error — **[GEN]** must over-generate by ≥ 5 %).
3. Tokenize/pack replay shards to exactly n_rep = ρ·D₂ / L sequences (same for every condition of that size).
4. Slot pattern: a seeded permutation of the multiset {treat × n_treat, rep × n_rep} is drawn once per (size, s) and stored (`slots_{S}_{s}.npy`); it is the same file for every condition. Within treatment slots, treatment sequences are placed by a seeded permutation of [0, n_treat); within replay slots, replay sequences likewise.
5. Output: one `uint16` memmap of tokens, one `int32` document-boundary array (for cu_seqlens), one SHA-256 over both; a `branch.json` that records all inputs.
6. The training config for a branch has exactly one condition-specific field, `treatment_path` (and its hash). A pre-flight check computes the canonical hash of the config *with that field removed* and refuses to launch unless it equals the size-level reference hash; the same check covers code commit, container image digest, torch/cuda/flash-attn versions, GPU model, θ* hash, tokenizer hash, micro-batch size, and `slots` hash.
7. The loader is a stateless index → sequence map (sequence i of step t is `memmap[perm[t·b + i]]`), single-process, no worker nondeterminism; resumption restores the exact step. Numeric nondeterminism from attention backward (atomics) is accepted and documented; bit-exact reproducibility is not claimed (flash-attn's `deterministic=True` mode exists in recent versions but slows training **[UNVERIFIED]**; PyTorch SDPA's flash backend is nondeterministic in backward).
8. Divergence rule: a run whose loss exceeds 2× its running minimum for > 50 steps or produces NaN is stopped, logged as failed, and re-run once with the *same* seed after root-causing; never silently replaced by another seed.

### 2.12 Checkpoint selection and evaluation cadence

- **Selection**: the *final* checkpoint at the end of the phase-2 decay, always. No best-on-benchmark, no early stopping, no averaging (weight averaging along the trajectory would be admissible only if applied identically to all branches; not used in the primary).
- **Held-out loss** every k steps (table §2.3; ≈ 40 points per branch) on: (a) 10 M tokens of held-out base corpus, (b) 5 M tokens of held-out human Q&A text (C1b source, disjoint), (c) 2 M tokens of held-out synthetic text from *each* teacher (monitoring only; never used for selection), (d) the **[EVAL]** human-written instruction/QA NLL set.
- **Benchmarks** at 3 points: θ* (shared, pre-branch), mid-decay (step ≈ ½), final. The final is the primary; the mid-point is for monotonicity diagnostics only.
- θ* and C0 are also evaluated so that the paper can report the base model's absolute level next to Pythia/SmolLM references.

### 2.13 Full hyperparameter table (per size, per phase)

Shared: AdamW β=(0.9,0.95), ε=1e-8, WD 0.1 (matrices only), clip 1.0, bf16/fp32, L=2048, intra-doc masking, tied embeddings, init N(0, 0.02) truncated with 1/√(2·n_layer) scaling on residual-output projections (GPT-2/OLMo-style) **[ARCH]**, no dropout, seed-indexed ordering.

| S (N_ne) | Phase | Tokens | B (tok) | Steps | Peak LR | Schedule | Warmup | Data | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| 25 M | base | 0.8 B | 0.25 M | 3,200 | 2.0e-3 | constant after warmup | 160 | FineWeb-Edu train shards | θ*(25M, b), b ∈ {1,2,3} |
| 25 M | phase 2 | 0.8 B (0.6 B treat + 0.2 B replay) | 0.25 M | 3,200 | 2.0e-3 → 0 | linear decay, no re-warm | 0 | treatment file + replay | final |
| 100 M | base | 2.5 B | 0.5 M | 5,000 | 1.2e-3 | constant | 250 | same | θ*(100M, b), b ∈ {1,2,3} |
| 100 M | phase 2 | 0.8 B | 0.5 M | 1,600 | 1.2e-3 → 0 | linear | 0 | | final |
| 250 M | base | 5.7 B | 0.5 M | 11,300 | 8e-4 | constant | 565 | | θ*(250M, 1) |
| 250 M | phase 2 | 0.8 B | 0.5 M | 1,600 | 8e-4 → 0 | linear | 0 | | final |
| 500 M | base | 10.8 B | 1 M | 10,800 | 6e-4 | constant | 540 | | θ*(500M, 1) |
| 500 M | phase 2 | 0.8 B | 1 M | 800 | 6e-4 → 0 | linear | 0 | | final |
| 1 B | base | 21.2 B | 1 M | 21,200 | 4e-4 | constant | 1,000 | | θ*(1B, 1) |
| 1 B | phase 2 | 0.8 B | 1 M | 800 | 4e-4 → 0 | linear | 0 | | final |

Pilot-only sweeps: base LR ×{0.5, 1, 2} at 25 M/100 M; phase-2 peak ×{0.5, 0.25} at 100 M (X5); ρ ∈ {0, 0.5} (X4). Token counts assume the placeholder N_tot; recompute from **[ARCH]** dims with D_base = 20·N_tot, keeping B and LR.

---

## 3. Part 3 — Base corpus

| Candidate | License | Size | Selection mechanism | Concern for this study |
|---|---|---|---|---|
| FineWeb (plain) | ODC-By 1.0 | ≈ 15 T GPT-2 tokens; sample-10BT/100BT/350BT | heuristic filters + dedup only | weakest per-token quality → more floor at 25 M–250 M; least "LLM-shaped" |
| **FineWeb-Edu** | ODC-By 1.0 | 1.3 T; sample-10BT/100BT/350BT | classifier trained on Llama-3-70B-Instruct educational scores (threshold 3 **[UNVERIFIED]**) | base already "textbook-like" → may shrink the synthetic effect; selection by an LLM's judgement (not LLM text) |
| DCLM-baseline 1.0 | CC-BY-4.0 | ≈ 4 T | fastText classifier with OpenHermes-2.5 + r/ELI5 as positives | selects web text that *resembles instruction/Q&A data* (OH-2.5 is largely GPT-4-generated) → directly enriched for the treatment's format; excluded |
| Dolma (v1.7) | ODC-By | 3 T | multi-source mixture (web, code, papers, books, Wikipedia) | mixture composition is a second design axis; fine but unnecessary |
| Web + small code/math fraction | — | — | — | adds a design axis; the benchmark suite (P6) is not code/math; skip |

**Choice: FineWeb-Edu, `sample-100BT`** (≈ 100 B GPT-2 tokens; ≈ 90–110 B in the student tokenizer **[assumption]**), sharded: train shards (≥ 22 B student tokens; the 1 B base uses ≈ 21 B), fresh held-out shards for C1 and the D-sweep (≥ 3 B), an eval shard (10 M tokens), never overlapping. Reasons: (1) it is the de-facto standard for small-model pretraining (SmolLM/SmolLM2, many 2024–26 papers), so the base and absolute levels are comparable to published models; (2) floor effects, not effect size, are the binding risk at 25 M–250 M, and FineWeb-Edu gives the largest benchmark lift per token among open web corpora (FineWeb-Edu release results **[lit, numbers not re-verified]**); (3) its selection is by educational *score*, not by instruction-likeness (unlike DCLM), so it does not pre-load the treatment's format. The "textbook-like base shrinks the synthetic effect" risk is real for *power*, not for identification; it is measured directly by cross-check X3 (plain-FineWeb base at 100 M). If X3 shows a materially larger main effect *and* a different teacher ranking, the full grid switches to plain FineWeb before the 250 M–1 B base runs start (decision point after the pilot). A middle option (FineWeb-Edu-score-2, the looser threshold **[dataset existence UNVERIFIED]**) is available. Precedent that synthetic textbook-style data still adds value on top of FineWeb-Edu: SmolLM-Corpus mixes FineWeb-Edu-dedup (220 B) with Cosmopedia-v2 (28 B) **[the ablation numbers were not re-verified]**.

**Decontamination plan** (applied to base train/held-out shards, replay shards, the human Q&A corpus, the prompt pool, and every synthetic pool):
- Method: Dolma/OLMo-style paragraph-level exact-match Bloom filter over benchmark *test and validation* items (question + choices + passages), ignoring paragraphs < 13 tokens and punctuation-only paragraphs; remove the whole document on any hit; additionally a 13-gram Jaccard ≥ 0.8 document filter (GPT-3-style) as a second pass. Benchmarks: all of P6 (ARC-E/C, PIQA, HellaSwag, SciQ, OBQA, BoolQ, Winogrande, LAMBADA) plus MMLU, GSM8K, and the **[EVAL]** NLL sets. Report removal rates per corpus.
- Teacher regurgitation is a *teacher-size-correlated* contamination channel (larger teachers memorise more benchmark text). Beyond removal, report the pre-removal hit rate per teacher pool as a diagnostic, and keep the removed items for a "contamination-sensitivity" re-analysis (Open issue 5).

---

## 4. Part 4 — Matched-real and human controls

**C1 (matched real)**: exactly n_treat sequences (= D_syn = 600 M tokens) from *fresh, held-out* FineWeb-Edu shards (same corpus, same tokenizer, same packing, same slot pattern, same replay). It answers "synthetic vs the same number of additional web tokens under the same schedule". "Same D tokens" means same trained tokens after packing; document counts differ (web docs are longer) and are reported.

**C0 (nothing new)**: n_treat sequences drawn from *already-seen* train shards (a second epoch of a random subset), so that C1 − C0 isolates freshness (expected ≈ 0 at < 4 epochs by Muennighoff et al. 2023, which is itself a useful calibration) and C_T − C0 gives "synthetic vs nothing new".

**C1b (human-written Q&A text, format control)**: 600 M tokens of Stack Exchange question + accepted-answer pairs (Stack Exchange data dump, CC BY-SA 4.0; ≈ 20 B tokens in the RedPajama-1T slice **[size and licensing details UNVERIFIED]**), formatted with the same "Question:/Answer:" template as the synthetic data, decontaminated, restricted to non-code sites to match the prompt pool's domain mix **[GEN to align]**. It separates "Q&A format" from "synthetic provenance": C_T − C1b is the value of teacher-generated answers over human-written answers of the same format, and C1b − C1 the value of the format itself.

**Why a 600 M-token human *instruction-following* control is infeasible**: genuinely human-written instruction data is tiny — Dolly-15k (≈ 15 k examples), OASST1 (≈ 161 k messages / ≈ 10 k trees **[counts from memory, UNVERIFIED]**), No Robots (≈ 10 k), LIMA (1 k): on the order of 20–30 M tokens in total **[estimate]**. FLAN v2 is large but templated from NLP datasets (many of them the P6 benchmarks' training splits) — a contamination and format mismatch, not a human-instruction control. Tulu-3 SFT (939 k samples) is majority machine-generated. Repeating 25 M tokens 24× is not a fair control (repetition confound).

**C1c (small equal-D comparison)**: at D = 20 M treatment tokens (the remaining 580 M treatment slots filled with fresh web tokens, identical across arms), compare {human instruction data (Dolly + OASST1 + No Robots, deduplicated, 1 epoch)} vs {each teacher's synthetic data subsampled to 20 M} vs {fresh web}. Run at 100 M and 500 M, 3 seeds. This gives the paper an honest "synthetic vs human instruction data" statement at the only D where it can be made.

Resulting inference ladder: synthetic > nothing (C_T − C0), synthetic > more web (C_T − C1), synthetic vs human Q&A (C_T − C1b), synthetic vs human instructions at small D (C1c), and the teacher contrast Δ(T,T′) — all paired over seeds.

---

## 5. Part 5 — Trainer and run manifest

| Framework | Fit for 1-GPU, reproducibility-first study | Verdict |
|---|---|---|
| Custom minimal PyTorch (nanoGPT / llm.c lineage) | ≈ 1.5 k lines; full control of loader, masking, schedule, manifest; easy to audit; widely reproduced recipe (GPT-2 124 M) | **Adopt** |
| TorchTitan | production-grade, regional torch.compile, FSDP2/TP/PP — irrelevant on 1 GPU; HF-datasets streaming loader is not the deterministic memmap we need | no |
| HF Trainer | overhead, many implicit behaviours (grad-accum loss scaling history, packing/`position_ids` handling), harder to guarantee the config-diff invariant | no |
| Lightning / Composer | Composer's streaming loader is deterministic and good, but adds abstraction; the manifest/diff invariant is easier to enforce in a small codebase | no (acceptable fallback) |

**Implementation**: plain-PyTorch LLaMA block (RMSNorm, RoPE, SwiGLU, GQA) **[ARCH]**; attention via `flash_attn_varlen_func` with `cu_seqlens` for intra-document masking (fallback: FlexAttention block mask under `torch.compile`; PyTorch ≥ 2.5); `torch.compile` on the block (regional, as TorchTitan does; note that compile does not guarantee bitwise determinism); `torch.optim.AdamW(fused=True)`; bf16 autocast; gradient accumulation with loss normalised by *tokens* in the effective batch (not micro-batches); memmap loader as in §2.11; CUDA 12.8+/PyTorch 2.7+ for Blackwell sm_120 and a flash-attn build for it **[availability UNVERIFIED — FlexAttention is the fallback]**. Throughput targets: ≥ 25 % MFU at 100 M, ≥ 40 % at 1 B (verify in pilot; the cost model in §1.3 assumes these).

**Run manifest (one JSON per run; hashed; archived with the checkpoint)**:
`run_id`, `size`, `condition` (C0/C1/C1b/C1c/T_x/X-check), `base_seed`, `phase2_seed`, `git_commit`, `container_digest`, `torch/cuda/flash_attn versions`, `gpu_model`, `gpu_count`, `tokenizer_hash`, `theta_star_hash` (weights + optimizer + loader position), `treatment_path` + `treatment_sha256` + `treatment_doc_count` + `treatment_prompt_token_fraction`, `replay_sha256`, `slots_sha256`, `D_base`, `D2`, `rho`, `D_syn_tokens_trained` (exact), `batch_tokens`, `micro_batch`, `grad_accum`, `seq_len`, `peak_lr`, `schedule` (type, warmup, decay), `optimizer` (betas, eps, wd, wd_param_groups), `clip`, `precision`, `masking` (intra_doc: true), `steps_total`, `eval_every`, `eval_sets` (hashes), `benchmark_points`, `checkpoint_rule` ("final"), `config_hash_excluding_treatment` (must equal size-level reference), `wallclock_s`, `gpu_seconds`, `energy_kwh` (if measured), `mfu_estimate`, `loss_curve_path`, `divergence_events`, `status`.

---

## References (author-year; URLs as retrieved)

- Allal et al. 2025, SmolLM2. https://arxiv.org/abs/2502.02737 ; SmolLM v1 blog https://huggingface.co/blog/smollm ; SmolLM-Corpus https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- Bergsma et al. 2025, Power Lines. https://arxiv.org/abs/2505.13738 ; Straight to Zero. https://arxiv.org/abs/2502.15938
- Bi et al. 2024 (DeepSeek LLM). https://arxiv.org/abs/2401.02954
- Biderman et al. 2023, Pythia. https://arxiv.org/abs/2304.01373
- BeyondWeb (DatologyAI) 2025. https://arxiv.org/abs/2508.10975
- Datology 2026, The Finetuner's Fallacy. https://arxiv.org/abs/2603.16177
- DCLM-baseline. https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 ; Li et al. 2024 DataComp-LM https://arxiv.org/abs/2406.11794
- Dubey et al. 2024, The Llama 3 Herd of Models (document masking). https://arxiv.org/abs/2407.21783
- Eldan & Li 2023, TinyStories. https://arxiv.org/abs/2305.07759
- FineWeb / FineWeb-Edu. https://huggingface.co/datasets/HuggingFaceFW/fineweb ; https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Gu et al. 2024, MiniPLM. https://arxiv.org/abs/2410.17215
- Gunasekar et al. 2023, phi-1. https://arxiv.org/abs/2306.11644 ; Li et al. 2023 phi-1.5 https://arxiv.org/abs/2309.05463
- Gupta et al. 2023, How to (re)warm your model. https://arxiv.org/abs/2308.04014
- Hägele et al. 2024, Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. https://arxiv.org/abs/2405.18392
- Hoffmann et al. 2022, Chinchilla. https://arxiv.org/abs/2203.15556
- Hu et al. 2024, MiniCPM (WSD). https://arxiv.org/abs/2404.06395
- Huerta-Enochian & Ko 2024, Does Prompt Loss Matter? https://arxiv.org/abs/2401.13586
- Ibrahim et al. 2024, Simple and Scalable Strategies to Continually Pre-train LLMs. https://arxiv.org/abs/2403.08763
- Kang et al. 2025 (EMNLP), Demystifying Synthetic Data in LLM Pre-training. https://arxiv.org/abs/2510.01631
- Karpathy 2024, llm.c GPT-2 124M reproduction. https://github.com/karpathy/llm.c/discussions/481
- Liu et al. 2024, MobileLLM. https://arxiv.org/abs/2402.14905
- Madaan et al. 2024, Quantifying Variance in Evaluation Benchmarks. https://arxiv.org/abs/2406.10229
- Maini et al. 2024, Rephrasing the Web (WRAP). https://arxiv.org/abs/2401.16380
- Muennighoff et al. 2023, Scaling Data-Constrained LMs. https://arxiv.org/abs/2305.16264
- OLMo team 2025, 2 OLMo 2 Furious. https://arxiv.org/abs/2501.00656
- Pareja et al. 2024, Unveiling the Secret Recipe (SFT of 3–7 B). https://arxiv.org/abs/2412.13337
- Porian et al. 2024, Resolving Discrepancies in Compute-Optimal Scaling. https://arxiv.org/abs/2406.19146
- Qin et al. 2025, Scaling Laws of Synthetic Data (SynthLLM). https://arxiv.org/abs/2503.19551
- Shi et al. 2024, Instruction Tuning With Loss Over Instructions. https://arxiv.org/abs/2405.14394
- Springer et al. 2025, Overtrained LMs Are Harder to Fine-Tune. https://arxiv.org/abs/2503.19206
- Wortsman et al. 2023, Small-scale proxies. https://arxiv.org/abs/2309.14322
- Yang et al. 2024, Synthetic Continued Pretraining (EntiGraph). https://arxiv.org/abs/2409.07431
- Zhang et al. 2024, How Does Critical Batch Size Scale in Pre-training? https://arxiv.org/abs/2410.21676
- Zhao et al. 2024 (ACL), Analysing the Impact of Sequence Composition on LM Pre-training. https://arxiv.org/abs/2402.13991
- 2026, How Can We Synthesize High-Quality Pretraining Data? (generator-size ablation, FinePhrase). https://arxiv.org/abs/2604.13977
- Dolma decontamination notes. https://github.com/allenai/dolma/blob/decon-instructions/docs/paloma_decontamination.md

## Open issues and disagreements with the brief

1. **Fixed D_syn across student sizes (P5) confounds S with phase-2 fraction.** D₂/D_base is 100 % at 25 M and 3.8 % at 1 B, so "T*(S)" (H4) is entangled with "how much of training is synthetic". Generation cost forbids scaling D_syn ∝ N (20 B tokens per teacher at 1 B). Proposal: keep P5 for the primary, but present H4 results *also* at matched synthetic fraction using the fixed-D₂ fraction sweep (§2.3) at 100 M and 1 B, and state in the paper that T*(S) is defined at fixed D_syn.
2. **Novelty threat from 2025–26 generator-size ablations.** Kang et al. 2025 (larger generators not better than ≈ 8 B), BeyondWeb 2025 (generator size/family matters little), and the 2026 FinePhrase study (no gain beyond a 1 B generator; Gemma-3 270 M–27 B, SmolLM2 135 M–1.7 B) already report saturation in teacher size — for *rephrasing* web text at fixed student size. The defensible contribution is therefore the S × T interaction for 25 M–1 B students under free (prompt-conditioned) generation with matched-real/human controls, not "teacher size saturates". P8's title concern is reinforced; suggested framing: "Which teacher for which student? Controlled teacher-size effects of synthetic annealing data for 25 M–1 B language models".
3. **C0's definition differs from the brief.** The brief's C0 "base only" is replaced by "replay-only cooldown of the same length" because under WSD an un-annealed base is not a finished model. If the lead prefers a cosine-to-completion base, the fallback recipe in §2.5 applies, at the cost of re-warm artifacts and one extra decay per size.
4. **Uniform WD = 0.1 gives size-dependent AdamW timescales** (τ = 1/(ηλ) ≈ 5 k steps at 25 M vs 25 k at 1 B). Comparability across sizes argues for scaling λ so that τ is a fixed fraction of the run (Bergsma et al. 2025); simplicity argues for 0.1. Decision needed; I lean 0.1 with a footnote.
5. **Teacher regurgitation of benchmarks is a size-correlated contamination channel.** Even after 13-gram decontamination, paraphrased benchmark content survives; larger teachers likely produce more of it. Recommend an **[EVAL]** contamination-sensitivity analysis (e.g., per-item overlap scores → re-estimate Δ(T,T′) on the least-overlapping half of items) and reporting pre-removal hit rates per teacher.
6. **Disagreement with P7 on what "seed" means.** Three phase-2 seeds on a single base seed measure data-order variance only. Without the pilot's 3 base seeds, the paper cannot claim generalisation over initializations; the pilot must include them (cost ≈ 21 h of extra base runs plus their branches) and the variance decomposition must be reported, not just "3 seeds".
7. **Sizes as N_ne vs N_tot, and vocab size, are undecided ([ARCH]).** The 25 M model's budget is ≈ 60 % embeddings at 32 k; the LR/batch table is keyed by N_ne. Recommend deciding N_ne = nominal size and V = 32 k (or 16 k for all) before tokenizer freeze.
8. **Cost.** The 1 B base at 20 tok/param (≈ 410 GPU-h under the §1.3 assumptions) is ≈ 26 % of the whole budget; if the PRO 6000 delivers less than the assumed 250 TFLOPS dense BF16 / 40 % MFU, the 1 B row should be dropped from the primary grid rather than trained at fewer tokens/param than the others.
9. **Verification gaps.** The following could not be checked at source in this session and should be verified before pre-registration: SmolLM2-135M/360M peak LR; Pythia WD/warmup/min-LR; Hägele et al. cooldown-length results; units of the Zhang et al. CBS fit; flash-attn deterministic mode and Blackwell wheel availability; StackExchange slice size/license; human-instruction corpus token counts; FineWeb-Edu threshold and the existence of the score-2 variant; RTX PRO 6000 dense BF16 throughput.
10. **Option C cross-check (X1) may contradict the primary.** If the teacher ranking differs between B′ and C at 25 M/100 M, the paper must report both and the "which teacher" answer becomes design-conditional; this is a feature (it tests the Finetuner's-Fallacy claim in the synthetic setting) but the lead should decide now whether X1 is in scope.
