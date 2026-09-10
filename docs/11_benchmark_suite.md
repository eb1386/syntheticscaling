# 11. Benchmark Suite and Evaluation Protocol

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- Adopted unchanged: Tier A / Tier B primary suite with the floor gate, secondary suite, zero-shot harness settings, P_cc aggregate, continuous co-primary outcomes, contamination audit.
- Open issue 2 (base budget) is resolved by doc 00 §0.4 (5B-token minimum base). Open issue 3 (25M): 25M stays in the grid; if the pilot floor gate P3 fails at 25M, the 25M row becomes exploratory with NLL-only primary outcomes and the primary grid is 4 × 6.
- Open issue 4 (template confound) is resolved by doc 00 §0.7 Amendment B (distinct training markers) and by the C1b human Q&A control.
- HellaSwag is evaluated on a fixed 5,000-item seeded subsample (same items for every checkpoint) to halve evaluation cost; `--cache_requests` is enabled.

---

# Benchmark selection and evaluation protocol (workstream: BENCHMARKS)

Scope: primary/secondary evaluation suites for decoder-only students of 25M–1B parameters trained on ~0.5–20B real tokens (P5: ~20 tokens/parameter) plus a synthetic instruction-style phase (D_syn = 100M–1B student tokens). Every number below is tagged:

* **[V]** verified in this session from a primary artifact (EleutherAI/pythia eval JSONs, GitHub READMEs, the `lm_eval==0.4.13` wheel, OLMES task configs, DCLM eval metadata).
* **[R]** recalled from a paper/model card that could not be re-fetched in this session (arxiv.org, huggingface.co, qwenlm.github.io were unreachable). Treat as approximate; must be re-checked before any figure is published.
* **[E]** our extrapolation / estimate.

Harness: `lm-eval==0.4.13` (latest on PyPI as of 2026-09-10; wheel inspected). All task ids, metric names, prompt templates, regexes and generation defaults quoted below are read from that wheel [V].

---

## 0. Decisions in one paragraph

Primary suite = 7 log-likelihood multiple-choice/cloze tasks in two tiers, zero-shot, harness-default prompts, no chat template: **Tier A (defined at every student size): SciQ (acc), ARC-Easy (acc_norm), PIQA (acc_norm), LAMBADA-OpenAI (acc; ppl logged)**; **Tier B (enters the aggregate at a size only if the base model C0 clears a pre-registered floor gate; expected ≥250M): OpenBookQA (acc_norm), SocialIQA (acc), HellaSwag (acc_norm)**. Primary aggregate = mean chance-normalized accuracy (PRIMARY-4 at all sizes; PRIMARY-7 where Tier B passes). Co-primary continuous outcomes = per-token NLL on (a) a held-out real shard and (b) a held-out human-written instruction/QA set. Secondary suite = ARC-Challenge, Winogrande, BoolQ, MMLU-continuation (cloze; STEM subset and full), CommonsenseQA-cloze (custom YAML), BLiMP, TruthfulQA-MC2 (report only), GSM8K (1B students only, gated), IFEval-lite (report only), TriviaQA-3k (1B only, optional), a 5-shot ICL probe (ARC-Easy, SciQ), teacher-validation NLL (fit, not outcome), and a 200-item format-adherence probe. Floor finding for 25M: under P5 budgets only SciQ and LAMBADA-ppl are safely off the floor; ARC-Easy and PIQA are marginal (+3–8 points); everything else is at chance. This is the strongest argument in this document for over-training the small students (Section 8).

---

## 1. Reference numbers

### 1.1 Sources actually consulted

| Source | What it gave | Status |
|---|---|---|
| EleutherAI/pythia repo, `evals/pythia-v1/*/zero-shot|five-shot/*_step{N}.json` (sparse clone) | Pythia 70M/160M/410M/1B(bf16) at step 143000 (≈300B tokens) and at intermediate steps; LAMBADA acc+ppl, PIQA, WinoGrande, WSC, ARC-E, ARC-C, SciQ, LogiQA, MMLU (`hendrycksTest-*`, 57 subjects) 0-shot and 5-shot. Repo README warns the harness commit is old and results "may not be reproducible by the current version". | [V] |
| facebookresearch/MobileLLM README (Liu et al. 2024) | Zero-shot table for MobileLLM 125M/350M/600M/1B/1.5B and baselines (OPT, GPT-Neo, Pythia-160M/410M/1B, TinyLlama, Falcon, Qwen1.5) on ARC-E, ARC-C, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande. Metric not stated in README; values match harness `acc_norm` for Pythia on ARC-E/PIQA (39.6/69.3 vs 40.0/69.2) → inferred acc_norm where defined. | [V] table, [E] metric |
| jzhang38/TinyLlama `EVAL.md` | acc_norm (lm-eval-harness) for Pythia-1.0B and TinyLlama checkpoints at 103B–3T tokens on HellaSwag, OBQA, WinoGrande, ARC-C, ARC-E, BoolQ, PIQA; MMLU/BBH/DROP/HumanEval (InstructEval). | [V] |
| EleutherAI/gpt-neo README | GPT-3 125M/350M/1.3B/2.7B (values transcribed from Brown et al. 2020) and GPT-2 1.5B, GPT-Neo 125M/350M/1.3B/2.7B on LAMBADA ppl/acc, WinoGrande, HellaSwag (acc), PIQA. | [V] |
| `lm_eval` 0.4.13 wheel (PyPI) | Task YAMLs, metric definitions, `acc_norm` implementation, GSM8K/IFEval/TriviaQA generation settings, seed CLI, decontamination module (13-gram). | [V] |
| allenai/olmes `oe_eval/configs/tasks.py` | OLMES per-task cloze normalization (`acc_per_char`, `acc_uncond` = PMI, `acc_raw`), 5-shot curated, `limit: 1000` subsampling for several tasks. | [V] |
| mlfoundations/dclm `eval/eval_meta_data.csv` | Random baselines and item counts used by DCLM "centered accuracy". | [V] |
| huggingface/smollm `text/evaluation/smollm2/*` | SmolLM2 eval protocol: lighteval, `loglikelihood_acc_norm_nospace`, 0-shot for all MC tasks, GSM8K 5-shot, MMLU-Pro letter-MCF, MMLU cloze prompt for STEM ablations. (No numbers in repo.) | [V] protocol |
| huggingface/datatrove `pipeline/decont/n_grams.py` | Decontamination defaults: `n_grams=12`, `find_query_ngrams=False`, `find_overlap_ngrams=True`; any single hit removes the document. | [V] |
| lm-sys/llm-decontaminator README | Embedding-top-k + LLM-judge rephrase detection; contamination rates (e.g., HumanEval in The Stack 18.9%, MMLU in FLAN-CoT 0.5%). | [V] |
| SmolLM2 model cards (Allal et al. 2025), Qwen2.5 blog/report (Qwen Team 2024), TruthfulQA (Lin et al. 2022), BLiMP (Warstadt et al. 2020), Cerebras-GPT (Dey et al. 2023), Open LLM Leaderboard v1 | Not reachable this session; numbers reproduced from memory where used. | [R] |

### 1.2 Table A — fully trained public models (zero-shot unless stated; percent)

Metric convention per column is stated in the header. "an" = harness `acc_norm`; "acc" = unnormalized.

| Model (tokens) | HellaSwag an | PIQA an | ARC-E an | ARC-C an | OBQA an | BoolQ acc | WinoG acc | SciQ acc | LAMBADA acc / ppl | MMLU 0-shot letters | SIQA acc | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pythia-70M (300B) | ~27 [E] | 59.1 | 35.0 (acc 37.4) | 22.1 | — | — | 52.8 | 60.1 (an 55.2) | 18.5 / 142.4 | 25.3 (5-shot 24.7) | — | [V] pythia JSON |
| Pythia-160M (300B) | 29.9 | 61.6 | 39.6 (acc 43.5) | 23.3 | 31.2 | 59.5 | 53.1 | 74.1 (an 66.8) | 32.8 / 38.1 | 24.4 (5-shot 25.8) | 41.5 | [V] pythia JSON; MobileLLM README for HS/OBQA/BoolQ/SIQA |
| MobileLLM-125M (1T) | 38.9 | 65.3 | 43.9 | 27.1 | 39.5 | 60.2 | 53.1 | — | — | — | 42.4 | [V] MobileLLM README |
| GPT-3 125M (300B) | 33.7 (acc) | 64.6 | — | — | — | — | 52.0 | — | 42.7 / 18.6 | — | — | [V] gpt-neo README (from Brown et al. 2020) |
| SmolLM2-135M (2T) | 42.1 | 68.4 | ARC avg 43.9 | (in avg) | 34.6 | — | 51.3 | — | — | MMLU-Pro 11.3 (chance 10) | — | [R] model card (lighteval an_nospace) |
| Pythia-410M (300B) | 40.1 | 67.1 | 45.8 (acc 52.1) | 24.3 | 36.2 | 55.3 | 53.7 | 81.1 (an 72.1) | 51.6 / 10.8 | 25.5 (5-shot 25.0) | 43.1 | [V] pythia JSON; MobileLLM README |
| MobileLLM-350M (1T) | 49.6 | 68.6 | 53.8 | 33.5 | 40.0 | 62.4 | 57.6 | — | — | — | 44.7 | [V] MobileLLM README |
| GPT-3 350M (300B) | 43.6 (acc) | 70.2 | — | — | — | — | 52.1 | — | 54.3 / 9.09 | — | — | [V] gpt-neo README |
| SmolLM2-360M (4T) | 54.5 | 71.7 | ARC avg 53.0 | (in avg) | 37.4 | — | 52.5 | — | — | MMLU-Pro 11.7 | — | [R] model card |
| Qwen2.5-0.5B (18T) | 52.1 (10-shot) | — | — | 35.6 (25-shot) | — | — | 56.3 (5-shot) | — | — | MMLU 5-shot 47.5 | — | [R] Qwen2.5 report (few-shot, letters) |
| Pythia-1B (300B) | 47.2 | 69.3 | 49.0 (acc 56.9) | 27.1 | 31.4–38.6 | 58.7–60.8 | 53.4–53.7 | 84.0 (an 76.0) | 56.2 / 7.92 | 25.9 (5-shot 25.4); InstructEval 5-shot 25.7 | 43.3 | [V] pythia JSON; TinyLlama EVAL.md; MobileLLM README |
| TinyLlama-1.1B (3T) | 59.2 | 73.3 | 55.3 | 30.1 | 36.0 | 57.8 | 59.1 | — | — | ~25.9 (5-shot) | 43.9 | [V] TinyLlama EVAL.md; MobileLLM README |
| MobileLLM-1B (1T) | 61.4 | 74.4 | 63.0 | 39.0 | 46.8 | 66.7 | 62.3 | — | — | — | 45.0 | [V] MobileLLM README |
| GPT-3 1.3B (300B) | 54.7 (acc) | 75.1 | — | — | — | — | 58.7 | — | 63.6 / 5.44 | — | — | [V] gpt-neo README |
| GPT-2 1.5B (~10B×epochs) | 40.0 (acc) | 70.8 | — | — | — | — | 59.4 | — | 51.2 / 10.6 | — | — | [V] gpt-neo README (harness) |
| SmolLM2-1.7B (11T) | 68.7 | 77.6 | ARC avg 60.5 | (in avg) | 42.2 | — | 59.4 | — | — | MMLU-Pro 19.4 | — | [V] via search snippet of model card |

Other benchmarks, small models:

| Benchmark | 70–160M | 350–500M | ~1B | Notes |
|---|---|---|---|---|
| CommonsenseQA (5-way, chance 20) | SmolLM2-135M 33.9 [R] | SmolLM2-360M 38.0 [R] | SmolLM2-1.7B 43.6 [V] | lighteval cloze (choices scored as text). Harness default `commonsense_qa` uses letters A–E → floor for all sizes ≤1B [E, by analogy with MMLU]. |
| TriviaQA (generative EM, chance 0) | SmolLM2-135M 4.1 [R] | 360M 16.9 [R] | 1.7B 36.7 [V] | Knowledge-bound; scales with tokens more than with params. Chinchilla-budget students will be ≤5% below 1B [E]. |
| GSM8K (5-shot, EM, chance 0) | SmolLM2-135M 1.4 [R]; Pythia-160M ≈1 [R] | 360M 3.2 [R] | SmolLM2-1.7B 31.0 [R]; Qwen2.5-0.5B 41.6 (4-shot) [R]; Pythia-1B ≈1–2 [R] | Non-zero only with math-heavy pretraining at ≫ Chinchilla budgets. |
| MMLU letter-MCF (chance 25) | 24.4–25.3 [V] | 25.5 [V] | 25.4–25.9 [V] (two independent harnesses) | Floor at every size ≤1B for Pile-scale data. Qwen2.5-0.5B reaches 47.5 only with 18T curated tokens [R]. |
| MMLU cloze/continuation | SmolLM-135M ≈30 [R] | SmolLM-360M ≈34 [R] | SmolLM-1.7B ≈40 [R] | HF SmolLM v1 blog reported MMLU in cloze form because MCF was uninformative [R]. |
| TruthfulQA MC2 (6-shot QA prefix) | Pythia-160M ≈44 [R] | Pythia-410M ≈41 [R] | Pythia-1B ≈39; TinyLlama ≈37 [R] | Inverse scaling (Lin et al. 2022): smaller/weaker models score *higher*. Uninformative here. |
| IFEval (instruct models only) | SmolLM2-135M-Instruct ≈30 (avg of 4 metrics) [R] | 360M-Instruct ≈41 [R]; Qwen2.5-0.5B-Instruct 27.9 prompt-strict [R] | SmolLM2-1.7B-Instruct ≈57 [R] | Requires chat-tuned models with long, formatted outputs; base-continued students will be near the empty-output baseline [E]. |
| BLiMP (67 phenomena, chance 50) | GPT-2 ≈80–82 overall [R]; BabyLM OPT-125M on 100M words ≈60–65 [R] | — | — | Grammar; well above chance even for tiny models; useful as a harm/drift check. Pythia JSONs in the repo do not contain BLiMP despite the README listing it [V]. |

### 1.3 Table B — token-matched Pythia checkpoints (the regime that matters for P5)

Pythia step s ≈ 2.1M tokens × s (batch 1024×2048). Our students under P5: 25M→0.5B, 100M→2B, 250M→5B, 500M→10B, 1B→20B tokens. Zero-shot, `lambada_openai` acc, `piqa` acc_norm, `arc_easy` acc_norm (acc), `arc_challenge` acc_norm, `sciq` acc, `winogrande` acc, MMLU 0-shot letters (57-subject mean). [V] from `evals/pythia-v1`.

| Model | Step (≈tokens) | LAMBADA acc | PIQA an | ARC-E an (acc) | ARC-C an | SciQ acc | WinoG | MMLU |
|---|---|---|---|---|---|---|---|---|
| 70M | 512 (1.1B) | 0.0 (ppl 1.2e5) | 52.4 | 30.1 (29.8) | 21.8 | 26.4 | 51.0 | 22.7 |
| 70M | 1000 (2.1B) | 5.0 (ppl 4465) | 53.9 | 30.1 (30.9) | 21.2 | 45.2 | 51.1 | 24.1 |
| 70M | 3000 (6.3B) | 12.1 | 58.4 | 33.5 (35.1) | 21.2 | 59.2 | 50.8 | — |
| 70M | 13000 (27B) | 21.0 | 59.3 | 35.2 (39.5) | 22.2 | 65.0 | 50.3 | — |
| 70M | 143000 (300B) | 18.5 | 59.1 | 35.0 (37.4) | 22.1 | 60.1 | 52.8 | 25.3 |
| 160M | 1000 (2.1B) | 6.5 (ppl 2591) | 55.0 | 28.5 (30.6) | 21.2 | 42.7 | 50.8 | 23.9 |
| 160M | 3000 (6.3B) | 22.8 | 58.4 | 35.1 (39.3) | 21.2 | 65.3 | 52.6 | — |
| 160M | 13000 (27B) | 30.0 | 61.0 | 39.8 (43.6) | 23.4 | 77.0 | 51.1 | — |
| 160M | 143000 (300B) | 32.8 | 61.6 | 39.6 (43.5) | 23.3 | 74.1 | 53.1 | 24.4 |
| 410M | 3000 (6.3B) | 26.3 | 60.2 | 37.2 (41.1) | 22.2 | 70.3 | 50.7 | — |
| 410M | 13000 (27B) | 41.6 | 63.4 | 42.7 (45.6) | 22.5 | 77.7 | 51.5 | — |
| 410M | 143000 (300B) | 51.6 | 67.1 | 45.8 (52.1) | 24.3 | 81.1 | 53.7 | 25.5 |
| 1B | 3000 (6.3B) | 30.6 | 60.7 | 37.1 (41.9) | 22.2 | 72.3 | 52.2 | — |
| 1B | 13000 (27B) | 44.6 | 66.2 | 43.9 (49.4) | 25.6 | 81.3 | 52.8 | — |
| 1B | 143000 (300B) | 56.2 | 69.3 | 49.0 (56.9) | 27.1 | 84.0 | 53.7 | 25.9 |

Caveats: (i) Pythia's LR warmup ends at step ~1430, so steps ≤1000 are *lower bounds* for a properly decayed run of the same token count; (ii) Chinchilla-optimal public references [R]: Cerebras-GPT-111M (2.2B tokens) HellaSwag 26.8 / PIQA 59.4 / LAMBADA 19.4 / ARC-E 38.0 (acc) / ARC-C 16.6 / OBQA 11.8 (acc); Cerebras-GPT-1.3B (26B tokens) 32.5 / 66.4 / 46.2 / 50.8 / 22.4 / 16.6 (Dey et al. 2023, zero-shot acc). These bracket what a 100M/2B and a 1B/20B student should reach: ARC-E ≈ 35–38 acc, PIQA ≈ 59, LAMBADA ≈ 19% at 100M; HellaSwag stays ≤ 33 acc even at 1B/26B.

### 1.4 Table C — chance, majority baseline, size, SE

Binomial SE at p = 0.5 is `sqrt(0.25/n)`; at the realistic accuracy p the SE is `sqrt(p(1−p)/n)` (smaller near chance). Item counts from the harness YAML splits and DCLM metadata [V]; majority values marked ≈ are to be recomputed from the downloaded datasets with the script in 1.6.

| Task (harness id, version) | Split used | n | Options | Chance | Majority / trivial baseline | SE @50% (pts) |
|---|---|---|---|---|---|---|
| `hellaswag` v1.0 | validation | 10,042 | 4 | 25.0 | ≈25–26 (labels near-uniform) | 0.50 |
| `piqa` v1.0 | validation | 1,838 | 2 | 50.0 | ≈50.5 | 1.17 |
| `arc_easy` v1.0 | test | 2,376 | 3–5 (mostly 4) | ≈25 (mean 1/k) | ≈27 (most frequent key) | 1.03 |
| `arc_challenge` v1.0 | test | 1,172 | 3–5 (mostly 4) | ≈25 | ≈27 | 1.46 |
| `openbookqa` v1.0 | test | 500 | 4 | 25.0 | ≈27 | 2.24 |
| `boolq` v2.0 | validation | 3,270 | 2 | 50.0 | **62.2 (always "yes")** | 0.87 |
| `winogrande` v1.0 | validation (xl) | 1,267 | 2 | 50.0 | ≈50.4 | 1.40 |
| `sciq` v1.0 | test | 1,000 | 4 | 25.0 | Positional: correct answer is always index 3 in the harness YAML → irrelevant for LL scoring, fatal for letter-MCF | 1.58 |
| `lambada_openai` v1.0 | test | 5,153 | vocab | ≈0 | ≈1 (predict "the") | 0.70 |
| `mmlu` / `mmlu_continuation` v1.0 | test | 14,042 | 4 | 25.0 | ≈25–26 | 0.42 |
| `mmlu_stem_continuation` | test | ≈3,018 (18 subjects) | 4 | 25.0 | ≈25–27 | 0.91 |
| `commonsense_qa` (custom cloze) | validation | 1,221 | 5 | 20.0 | ≈21 | 1.43 |
| `social_iqa` v0.0 | validation | 1,954 | 3 | 33.3 | ≈34 | 1.13 |
| `blimp` (67 × 1,000) v1.0 | train (only split) | 67,000 | 2 | 50.0 | 50.0 | 0.19 |
| `truthfulqa_mc1` v2.0 / `mc2` v3.0 | validation | 817 | 4–13 | MC1 ≈22.6 (mean 1/k) [R]; MC2 uniform-prob ≈40 [R] | — | 1.75 |
| `gsm8k` v3.0 | test | 1,319 | free | 0 | 0 | 1.38 |
| `ifeval` v4.0 | train (only split) | 541 prompts | free | — | empty-output baseline (some instructions pass vacuously) — compute once and report | 2.15 |
| `triviaqa` v3.0 (rc.nocontext) | validation | 17,944 (subsample 3,000) | free | 0 | 0 | 0.37 (0.91) |

### 1.5 Floor determination

Rule: "clearly above chance" = C0 mean − chance ≥ max(5 points, 3·SE_item). Smallest size at which this is expected to hold, in the two regimes.

| Benchmark (metric) | Fully trained (≥300B tok) | P5 regime (≈20 tok/param) [E from Table B + Cerebras-GPT] | Verdict for our study |
|---|---|---|---|
| SciQ (acc) | 70M (+35) | 25M@0.5B: ≈+10–20 (70M@1B tokens was still at chance; @2B +20); 100M@2B: +20 | **Primary Tier A**; the only accuracy task safely usable at 25M |
| LAMBADA (acc; ppl) | 70M (18.5) | 25M@0.5B: acc 2–8%, ppl 1e2–1e3; 100M@2B: ≈10–19% | **Primary Tier A**; acc near floor at 25M, ppl is continuous and always informative |
| ARC-Easy (acc_norm) | 70M (+10) | 100M@2B: +8–10 (Cerebras 111M acc 38); 25M@0.5B: +3–6 | **Primary Tier A**; marginal at 25M |
| PIQA (acc_norm) | 70M (+9) | 100M@2B: +8–9; 25M@0.5B: +3–5 | **Primary Tier A**; marginal at 25M |
| OpenBookQA (acc_norm) | 160M (+6) | 250–500M | Primary Tier B |
| SocialIQA (acc) | 160M (+8) | ≈250M (+5) [E] | Primary Tier B |
| HellaSwag (acc_norm) | 160M (+5, borderline); 410M (+15) | 1B (+8–10); ≤500M within +2–5 | Primary Tier B; effectively a 1B-only task in P5 regime |
| Winogrande (acc) | ≥1B with ≥1T tokens (MobileLLM-1B +12); Pythia-1B/300B only +3.7 | never below 1B | Secondary (transfer/harm check) |
| BoolQ (acc vs majority 62.2) | only MobileLLM-1B (66.7) exceeds majority; all Pythia ≤1B below majority | never | Secondary, report relative to majority; not in any aggregate |
| ARC-Challenge (acc_norm) | 350M–1B with ≥1T tokens (+5–8); Pythia-1B +2 | never below 1B | Secondary |
| MMLU letters | never ≤1B | never | Excluded |
| MMLU-continuation (cloze) | ≈135M (+5) with 600B+ tokens [R] | ≈1B (+3–6) [E] | Secondary (STEM subset + full) |
| CommonsenseQA cloze | 135M (+14) with 2T tokens [R] | 250–500M [E] | Secondary |
| GSM8K | ≥1.7B with math-rich ≥10T tokens | never | Secondary, gated (1B only, only if C0 or C1 ≥ 3%) |
| TriviaQA | 360M with 4T tokens (16.9) | 1B (≈3–6%) [E] | Optional secondary (1B only) |
| BLiMP | all sizes | all sizes | Secondary (grammar/harm) |
| TruthfulQA | inverse scaling | — | Report only |
| IFEval | chat-tuned ≥135M | — | Report only ("IFEval-lite") |

### 1.6 Baseline-recompute script (run once at dataset download; record in the eval log)

```python
# chance = mean(1/k), majority = max label frequency; run per task on the harness split above
from datasets import load_dataset; from collections import Counter
def stats(ds, key_fn, k_fn):
    keys=[key_fn(d) for d in ds]; ks=[k_fn(d) for d in ds]
    c=Counter(keys); return sum(1/k for k in ks)/len(ks), max(c.values())/len(keys), len(keys)
print("arc_easy", stats(load_dataset("allenai/ai2_arc","ARC-Easy")["test"], lambda d:d["answerKey"], lambda d:len(d["choices"]["text"])))
print("boolq", stats(load_dataset("aps/super_glue","boolq")["validation"], lambda d:d["label"], lambda d:2))
# ... piqa (baber/piqa), hellaswag (Rowan/hellaswag), openbookqa, winogrande_xl, social_iqa (lighteval/siqa), commonsense_qa, sciq
```

---

## 2. Per-benchmark assessment

### 2.1 Metric notes (established facts about the harness, [V])

* **Log-likelihood MC scoring**: each option string is scored as a continuation of the prompt; `acc` = argmax raw log-likelihood; `acc_norm` = argmax of log-likelihood divided by **character** length of the option (`completion_len = len(choice)`; a byte-length variant `pred_byte` is computed but not surfaced). Historically called "byte-length normalization"; identical for ASCII. Length normalization removes the bias toward short continuations (raw LL is a product of per-token probabilities). It is the convention used by TinyLlama, MobileLLM (inferred), SmolLM (lighteval `acc_norm_nospace`, which excludes the leading space from the character count), and OLMES (`acc_per_char`). OLMES instead uses **PMI** (`acc_uncond`: LL(choice|prompt) − LL(choice|"")) for ARC-Challenge, OpenBookQA and CommonsenseQA, where options are common short phrases with strongly varying prior probability. The harness exposes PMI as `acc_mutual_info` but not in the default task YAMLs; adopting it would require custom YAMLs and break comparability with published numbers, so we do not.
* **Generative scoring** (GSM8K, IFEval, TriviaQA) requires greedy decoding, stop sequences and regex extraction; for base-continued students it is dominated by formatting failures rather than knowledge — hence secondary/report-only.
* **Task-level choice** (pre-registered): `acc_norm` for tasks with free-text options of variable length (ARC-E, ARC-C, OBQA, HellaSwag, PIQA, MMLU-continuation, CSQA-cloze); `acc` for SciQ (harness/Pythia convention; on Pythia acc_norm is systematically 5–8 points *below* acc at every size, i.e. the normalization hurts on this task), BoolQ, Winogrande, SocialIQA, LAMBADA, BLiMP. Both metrics come from the same log-likelihood calls; the non-chosen metric is logged and reported in an appendix, never used for selection.

### 2.2 Zero-shot vs few-shot (methodological choice, fixed for all conditions)

Evidence [V] (Pythia, Table 1.2 vs 5-shot JSONs): 5-shot changes 70M results by −6 (LAMBADA), −2.4 (SciQ), +1.3 (ARC-E) — i.e. noise or harm; at 160M: SciQ +3.8, ARC-E +2.2, LAMBADA −7; at 410M: SciQ +8, ARC-E +7.7; at 1B: SciQ +8, ARC-E +8.9, ARC-C +1.1, MMLU 0. So in-context learning becomes useful only from ~400M upward and only on some tasks. Decision: **all primary-suite tasks are zero-shot** for every condition (C0, C1, C2..C7) and every size; a 5-shot re-run of ARC-Easy and SciQ is a *secondary ICL probe* (does synthetic instruction data change few-shot ability?). Zero-shot with the harness's default `Question: … Answer:` templates is also the setting in which instruction-style data is expected to transfer; the templates are identical across conditions, so template familiarity is part of the treatment (see Open issue 4 for the resulting confound with C1).

### 2.3 Prompt standardization and pinning

* `lm-eval==0.4.13`, `transformers` and `datasets` versions pinned in `requirements-eval.txt`; run with `--log_samples` so every prompt/continuation/LL is archived; the harness writes task `version` fields into results JSON.
* Dataset revisions: pin HF dataset commit hashes for `Rowan/hellaswag`, `baber/piqa`, `allenai/ai2_arc`, `allenai/openbookqa`, `allenai/sciq`, `aps/super_glue` (boolq), `allenai/winogrande` (winogrande_xl), `EleutherAI/lambada_openai`, `lighteval/siqa`, `cais/mmlu`, `tau/commonsense_qa`, `nyu-mll/blimp`, `truthfulqa/truthful_qa`, `openai/gsm8k`, `google/IFEval`, `mandarjoshi/trivia_qa`.
* No chat template (`--apply_chat_template` not set) for any condition. `--num_fewshot 0` explicitly. `--seed 1234` (sets all four harness seed slots: python, numpy, torch, fewshot; default is `0,1234,1234,1234`). `--batch_size 64` fixed (bf16 log-likelihoods change slightly with batch composition; fixing batch size and GPU class per student size keeps this constant across conditions). `dtype=bfloat16`.
* Exact harness prompts [V]: ARC `Question: {q}\nAnswer:` with option text as continuation; PIQA `Question: {goal}\nAnswer:`; SciQ `{support}\nQuestion: {q}\nAnswer:`; OBQA `{question_stem}` + option; HellaSwag `{activity_label}: {ctx}` + ending; SIQA `Q: {context} {question}\nA:`; BoolQ `{passage}\nQuestion: {q}?\nAnswer:` with `no`/`yes`; LAMBADA: last whitespace-delimited word as target; MMLU-continuation `Question: {q}\nAnswer:` with a per-subject description line, full option text as continuation; MMLU default (excluded) `A. … D. …\nAnswer:` with letter continuations.

### 2.4 Assessment table

Responsiveness column = our hypothesis (H) about whether instruction-style synthetic data should move the score; it is exactly what the experiment tests, so it must not be used to *select* tasks post hoc.

| Benchmark | Include? | Metric | Shots | Harness id / ver | Contamination risk (known corpora) | n / SE@50% | Responsive to instruction-style synthetic data? |
|---|---|---|---|---|---|---|---|
| SciQ | Primary A | acc | 0 | `sciq` 1.0 | Source pages (science textbooks/quiz sites) are in Common Crawl; FineWeb-Edu's classifier up-weights exactly this genre → elevated verbatim/near-verbatim risk. Support passages are long → 13-gram screening effective. | 1,000 / 1.58 | H: yes (science QA; support passage reading) |
| ARC-Easy | Primary A | acc_norm | 0 | `arc_easy` 1.0 | ARC items are grade-school exam questions widely mirrored online; ARC test set overlap found in several web corpora [R: e.g. GPT-3 and Llama-2 contamination appendices]. FineWeb-Edu boosts ARC by several points [R] partly through genre match. | 2,376 / 1.03 | H: yes |
| PIQA | Primary A | acc_norm | 0 | `piqa` 1.0 | Goals derived from instructables.com — present in CC; short items (≈8 words) → use full-item matching. | 1,838 / 1.17 | H: weak (physical commonsense; mostly pretraining) — transfer-breadth check |
| LAMBADA | Primary A | acc (+ppl) | 0 | `lambada_openai` 1.0 | Book passages (BookCorpus-derived); low risk in FineWeb-Edu, medium in book-heavy corpora. | 5,153 / 0.70 | H: no (pretraining-driven) — harm/forgetting detector |
| OpenBookQA | Primary B | acc_norm | 0 | `openbookqa` 1.0 | Small test set (500), questions are short and quiz-like; moderate risk. | 500 / 2.24 | H: yes (elementary science facts) |
| SocialIQA | Primary B | acc | 0 | `social_iqa` 0.0 | Crowd-written; low web presence; low risk. | 1,954 / 1.13 | H: moderate (social reasoning appears in chat data) |
| HellaSwag | Primary B | acc_norm | 0 | `hellaswag` 1.0 | ActivityNet/WikiHow contexts; WikiHow is in CC and in FineWeb-Edu; contexts are long → screenable. | 10,042 / 0.50 | H: no — precise harm/transfer check |
| ARC-Challenge | Secondary | acc_norm | 0 | `arc_challenge` 1.0 | as ARC-Easy | 1,172 / 1.46 | H: yes but floor ≤1B |
| Winogrande | Secondary | acc | 0 | `winogrande` 1.0 | crowd-written; low | 1,267 / 1.40 | no; harm check |
| BoolQ | Secondary (report vs majority) | acc | 0 | `boolq` 2.0 | Wikipedia passages (high presence everywhere); questions are Google queries. | 3,270 / 0.87 | unclear; unusable below majority |
| MMLU-continuation (STEM; full) | Secondary | acc_norm | 0 | `mmlu_stem_continuation`, `mmlu_continuation` 1.0 | MMLU items mirrored on many exam-prep sites; FLAN-CoT 0.5% rephrased contamination [V lm-sys]. | 3,018 / 0.91; 14,042 / 0.42 | H: yes at 1B; floor below |
| MMLU letters | Excluded | — | — | `mmlu` | — | — | floor at all sizes ≤1B [V] |
| CommonsenseQA-cloze | Secondary | acc_norm | 0 | custom `commonsense_qa_cloze` (YAML in repo; based on `commonsense_qa` 1.0 with `doc_to_choice: choices.text`) | ConceptNet-derived; low–moderate | 1,221 / 1.43 | H: moderate |
| BLiMP | Secondary | acc (mean of 67) | 0 | `blimp` 1.0 | synthetic minimal pairs; none | 67,000 / 0.19 | no; grammar drift/harm detector |
| TruthfulQA MC2 | Report only | acc (mc2) | 6-shot fixed prefix in YAML | `truthfulqa_mc2` 3.0 | questions widely mirrored | 817 / 1.75 | inverse scaling; not interpretable |
| GSM8K | Secondary, gated | exact_match (flexible-extract primary; strict-match logged) | 5 (train) | `gsm8k` 3.0 | GSM8K test in several instruction corpora [V lm-sys reports MATH/GSM8K-style leakage into instruction sets]; **teachers' post-training data likely includes GSM8K-style items** | 1,319 / 1.38 | H: yes at 1B if synthetic data contains math |
| IFEval-lite | Report only | prompt_level_loose_acc (all 4 logged) | 0 | `ifeval` 4.0 with `max_gen_toks=512` override | none (prompts are novel) | 541 / 2.15 | H: yes for format following; near-zero baseline |
| TriviaQA-3k | Optional secondary (1B) | exact_match | 0 | `triviaqa` 3.0, `--limit 3000` (fixed seed) | Wikipedia/web trivia; high presence | 3,000 / 0.91 | H: weak (knowledge-bound) |
| ICL probe: ARC-E, SciQ 5-shot | Secondary | acc_norm / acc | 5 (harness sampler, seed 1234) | same ids | as above | as above | tests whether synthetic data changes ICL |

---

## 3. Suites and exact evaluation settings

### 3.1 Primary suite

```
lm_eval --model hf \
  --model_args pretrained=<ckpt>,dtype=bfloat16,tokenizer=<student_tok> \
  --tasks sciq,arc_easy,piqa,lambada_openai,openbookqa,social_iqa,hellaswag \
  --num_fewshot 0 --batch_size 64 --seed 1234 --log_samples \
  --output_path runs/<size>/<condition>/<seed>/primary
```
Fixed per-task metrics: SciQ `acc`; ARC-E `acc_norm`; PIQA `acc_norm`; LAMBADA `acc` and `perplexity`; OBQA `acc_norm`; SIQA `acc`; HellaSwag `acc_norm`.

**Floor gate (pre-registered, evaluated once per student size on the C0 seeds before any synthetic condition is analysed):** task t enters PRIMARY-7 at size S iff mean_C0(acc_t) − chance_t ≥ max(5 points, 3·SE_item,t). Tier A tasks always enter PRIMARY-4 regardless (the 25M cell is reported with an explicit "near-floor" flag if a Tier A task fails the gate).

### 3.2 Secondary suite

```
# log-likelihood block
--tasks arc_challenge,winogrande,boolq,mmlu_stem_continuation,mmlu_continuation,commonsense_qa_cloze,blimp,truthfulqa_mc2 --num_fewshot 0
# ICL probe
--tasks arc_easy,sciq --num_fewshot 5
# generative block (1B students; GSM8K only if gate passes)
--tasks gsm8k --num_fewshot 5                       # greedy (do_sample=false, temperature=0), until ["Question:","</s>","<|im_end|>"], max_gen_toks=256 (harness default)
--tasks ifeval --gen_kwargs max_gen_toks=512        # greedy; deviation from task default 1280 documented; all four IFEval metrics logged
--tasks triviaqa --limit 3000                        # greedy; until ["\n",".",","]; exact_match ignore_case+punctuation
```
Answer extraction (verbatim from the wheel): GSM8K `strict-match` regex `#### (\-?[0-9\.\,]+)`; `flexible-extract` regex `(-?[$0-9.,]{2,})|(-?[0-9]+)` with `group_select: -1` (last number), `take_first`; exact_match with `ignore_case: true`, `ignore_punctuation: false`, `regexes_to_ignore: [",", "\$", "(?s).*#### ", "\.$"]`. Base-continued students will not emit `####`, so `flexible-extract` is the pre-registered primary GSM8K metric. IFEval loose accuracy strips markdown and first/last lines before checking instructions (harness `utils.process_results`). Seeds: `--seed 1234` for all runs; generative tasks are greedy so the seed only affects few-shot sampling and `--limit` subsampling.

### 3.3 Custom probes (shipped with the repo)

* **Format-adherence probe (200 items, hand-written, not teacher-generated)**: 8 templates × 25 items — one-word answer; yes/no; numbered list of exactly N items; JSON with fixed keys; answer in ≤N words; pick one of listed options by label; repeat a given string verbatim; end with a fixed phrase. Greedy, `max_gen_toks=64`, scored by regex/JSON parse; report pass rate per template; chance ≈ 0. Purpose: separate "format following" from "knowledge" gains.
* **Held-out human-written instruction NLL set** — Section 5.

---

## 4. Aggregate scores

Notation: a_t = accuracy on task t, c_t = chance (Table C), K = number of tasks, s = seed, S = student size, cond ∈ {C0, C1, C2..C7}.

1. **Unweighted mean**: `M = (1/K) Σ_t a_t`. Simple, but tasks with chance 50 (PIQA) and chance 25 (ARC-E) contribute on different scales; a 5-point gain means different things.
2. **Chance-normalized mean** (DCLM "centered accuracy" precedent [V]): `z_t = (a_t − c_t)/(1 − c_t)`, `Z = (1/K) Σ_t z_t`. For LAMBADA, c = 0 so z = acc. Item-sampling SE of Z: `SE_Z = (1/K) sqrt(Σ_t SE_t²/(1−c_t)²)`; with the SE values of Table C, SE_Z ≈ 0.85 points for PRIMARY-4 and ≈ 0.69 for PRIMARY-7 (before seed variance).
3. **Base-normalized gain in SD units**: `d_t(cond,S) = (ā_t(cond,S) − ā_t(C0,S)) / σ̂_t(S)`, with σ̂_t(S) the pooled within-size residual SD across *all* conditions and seeds at size S (with 8 conditions × 3 seeds the residual has 16 df, versus 2 df if only C0 seeds were used). `D = (1/K) Σ_t d_t`.

Which for regression across S? (i) M and Z are on a common accuracy scale across S, so a scaling curve `Z(S, T)` is directly interpretable; Z removes the chance offset so floor-bound tasks contribute ≈0 rather than a spurious 25–50 point constant, and DCLM uses it for exactly this reason. Both are bounded and sigmoidal in log-compute; for fitting, use `logit(Z)` or fit per-task on the logit scale with task random effects. (ii) D standardizes by noise, which differs by size (small students are noisier), so effect *magnitudes* are not comparable across S and D cannot be the target of a scaling fit; it is however the right quantity for detectability/power and for asking "does teacher T beat C1 at size S". **Decision: Z (PRIMARY-4 at all sizes; PRIMARY-7 where the gate passes) is the primary discrete outcome; D is the sensitivity analysis; M and all per-task values are reported in full (no task is dropped from the appendix).** The continuous NLL outcomes of Section 5 are co-primary for the scaling regression because they are floor-free.

**MMLU-style letter answers below 1B.** With options in the prompt and `A/B/C/D` as continuation, a model must map content to a label symbol; Pythia 70M–1B score 24.4–25.9 at 0- and 5-shot [V], TinyLlama-1.1B at 3T tokens 25.9 [V], SmolLM2-360M on 10-way MMLU-Pro 11.7 vs chance 10 [R]. Public models only leave the floor at ≥1.5B or with ≫1T curated tokens (Qwen2.5-0.5B 47.5 at 18T tokens [R]). OLMES therefore evaluates small models with the cloze ("RC") formulation and reports max(MCF, CF) [V]; the harness offers `mmlu_continuation` (full answer text as continuation, options not shown; scored by acc_norm), and SmolLM used an equivalent cloze prompt for ablations [V]. Cloze MMLU is only +3–8 above chance for ≤500M models [R/E], hence secondary. Reporting max(MCF, CF) per model, as OLMES does, is a form of selection and is not used here.

---

## 5. Continuous metrics (co-primary, floor-free)

All NLLs are per-token means in the **student tokenizer** (shared by all students, P4), also reported as bits-per-byte (`Σ NLL / (ln 2 · bytes)`) for readability. Computation with the harness `loglikelihood_rolling`-style pass or a 40-line custom script; no sampling.

| Set | Content | Size | Purpose | Notes |
|---|---|---|---|---|
| (a) HeldOut-Real-ID | FineWeb-Edu documents held out from the base corpus (never trained on; decontaminated like the corpus) | 5M tokens | forgetting / drift of the pretraining distribution | C1 (extra real tokens) is expected to improve it; synthetic conditions to worsen it — the *difference vs C1* is the treatment effect |
| (a′) HeldOut-Real-OOD | FineWeb (non-Edu) sample + Wikipedia sample | 2M + 1M tokens | forgetting off the Edu distribution | licence: ODC-By (FineWeb), CC BY-SA (Wikipedia) |
| (b) HeldOut-Instr | human-written instruction/QA: databricks-dolly-15k (CC BY-SA 3.0; 15k human prompts+responses; stratified 2,000 over its 7 categories), OASST1 (Apache-2.0; English first-turn prompt + top-ranked human reply; 1,000), NQ-open (CC BY-SA 3.0; 1,000 short-answer items) | ≈1M tokens | fit to human instruction distribution; the primary continuous outcome | score **response tokens only**, conditional on the prompt rendered with the same template as the synthetic phase; also report NLL excluding template tokens. Exclude Alpaca (model-generated, CC BY-NC), FLAN (templated), No Robots (CC BY-NC 4.0; usable for research but flag), LIMA (CC BY-NC-SA). Verify licences at download. |
| (c) TeacherVal-T | 2% (≥2k examples) of each teacher's filtered pool, never used for training | ≈0.5M tokens each | fit to each teacher's distribution; full cross matrix NLL(student trained on T_i, val of T_j) | report as diagnostic, **not** as outcome (teacher-specific; a student trained on T_i trivially fits T_i best) |

Why these are the right regression targets: (1) no floor or ceiling — NLL moves monotonically from the first tokens of training, whereas 25M/100M accuracy sits at chance; (2) low noise — with 1M tokens and document-level bootstrap the SE is of order 0.002–0.005 nats/token (to be measured on C0 seeds), i.e. an order of magnitude below the per-task accuracy SE in Z units; (3) it is the quantity scaling laws actually model (Hoffmann et al. 2022; Busbridge et al. 2025 model teacher/student cross-entropy), so a fit `NLL_b(S, T, D_syn)` is directly comparable with that literature; (4) it is computed on the same tokens for every condition, so seed and item variance are paired.

Limitations (must appear in the paper): NLL depends on the tokenizer — comparable across our students (one tokenizer) but not with public models or across tokenizers; a 32k tokenizer inflates per-token NLL relative to a 151k one. (b) rewards *style* match to human-written text: a teacher whose outputs are more human-like (shorter, less markdown) will look better even if less informative — hence (b) is paired with the discrete suite and the correlation between Δ(b) and ΔZ is reported. Template tokens can dominate (b) if the synthetic phase teaches a format; report both with and without template tokens. (a) improvements under C1 are in-distribution by construction; do not interpret C1 vs synthetic on (a) as "synthetic is worse", only as forgetting magnitude.

---

## 6. Evaluation cost

Request counts (LL calls) and approximate tokens per request (32k tokenizer): HellaSwag 10,042×4 = 40,168 (≈90 tok); PIQA 1,838×2 = 3,676 (45); ARC-E ≈9,500 (45); OBQA 2,000 (30); SciQ 4,000 (120); LAMBADA 5,153 (75); SIQA 5,862 (50). **Primary suite ≈ 70k requests, ≈5.4M tokens.** Secondary LL block: MMLU-continuation 56,168 (80) + STEM subset (subset of same) ; BLiMP 134,000 (15); ARC-C ≈4,700; Winogrande 2,534; BoolQ 6,540 (150); CSQA 6,105; TruthfulQA-MC2 ≈5,700 (200; 6-shot prefix) → ≈216k requests, ≈9M tokens. Generative: GSM8K 1,319 × (≈700 prompt + ≤256 gen); IFEval 541 × ≤512 gen; TriviaQA-3k 3,000 × ≤20 gen. NLL sets: ≈11M tokens forward.

Time model: `t ≈ 2·N_params·tokens / (η·F_peak) + requests / r_overhead`, with η·F_peak the effective bf16 throughput and r_overhead ≈ 300–500 requests/s of tokenization/collation/Python overhead in the HF backend (assumption; measure in the pilot). Assumed effective throughput: RTX 5080 ≈ 40 TFLOPS (consumer parts run bf16 with fp32 accumulate at half rate; ~35% MFU), RTX PRO 6000 Blackwell ≈ 150 TFLOPS. Batch size 64 does not saturate either GPU for a 25M model; those runs are overhead-bound.

| Block | 25M, RTX 5080 | 1B, RTX 5080 | 1B, RTX PRO 6000 |
|---|---|---|---|
| Primary (7 tasks) | ≈3 min (overhead-bound) | ≈4.5 min compute + ≈3 min overhead ≈ 7–8 min | ≈4–5 min |
| Secondary LL block | ≈10 min (BLiMP overhead dominates) | ≈8 min compute + 9 min overhead ≈ 15–18 min | ≈12 min |
| Generative (GSM8K, IFEval-lite, TriviaQA-3k) | ≈4 min | ≈8–10 min (decode-bound; 1280→512 tokens for IFEval keeps this bounded) | ≈5 min |
| NLL sets (≈11M tokens) | <1 min | ≈9 min | ≈2.5 min |
| **Total per model** | ≈18 min | ≈40–45 min | ≈25 min |

Full grid: 5 sizes × 8 conditions × 3 seeds = 120 evaluations (+ D-sweep cells, + intermediate checkpoints if wanted) ≈ 50–60 GPU-hours on 5080-class cards, i.e. negligible next to training. Evaluating the four public Pythia checkpoints (70M/160M/410M/1B) through the same pinned pipeline (≈1.5 GPU-hours) is mandatory as a pipeline validation against Table A.

---

## 7. Contamination controls

Definitions: an *item* is a test question plus its correct answer (plus support passage where present); text is normalized as in the harness janitor (lowercase, punctuation stripped, whitespace collapsed). Conventions in the literature: GPT-3 13-gram document filtering (Brown et al. 2020) — also the harness's own `decontamination` module (`ngram_n=13`) [V]; datatrove default 12-gram, any hit removes the document [V]; Llama 2 tokenized 10-grams with an eval sample called contaminated when ≥80% of its tokens are covered (Touvron et al. 2023) [R]; Dolma/OLMo paragraph-level Bloom-filter matching against eval paragraphs of ≥13 tokens (Soldaini et al. 2024) [R].

**Index.** One n-gram index built from *all* evaluation items in this document (primary, secondary, ICL probe, format probe, HeldOut-Instr, HeldOut-Real): word 13-grams over `query + label` with overlap n-grams enabled (datatrove `find_overlap_ngrams=True`), plus **full-item exact match for items shorter than 13 words** (PIQA goals, OBQA stems, Winogrande sentences, BLiMP sentences), plus 8-grams over question text only for a second, stricter tier.

| Layer | Unit removed | Rule | Report |
|---|---|---|---|
| (a) Base corpus (FineWeb-Edu subset) and HeldOut-Real | document | any 13-gram / full-short-item hit | removed docs per benchmark; expect ARC/SciQ/OBQA/MMLU hits to be over-represented because the FineWeb-Edu classifier (trained on Llama-3-70B "educational value" labels) favours quiz/exam pages; whether FineWeb-Edu itself was benchmark-decontaminated is [UNVERIFIED] — assume not |
| (b) Prompt pool | prompt | 8-gram hit against any test question; embedding near-duplicate (MiniLM cosine ≥ 0.85 vs any test question → drop; lm-sys style) | removal counts; pool must also be disjoint from HeldOut-Instr sources (Section 5) both ways |
| (c) Synthetic outputs, per teacher T | example | 13-gram / full-short-item hit in prompt+response | removal rate r_T per benchmark; applied identically to all teachers **after** the audit below |

**Post-hoc audit (mandatory; the teacher-size confound).** Larger teachers have more capacity to memorize benchmark items seen in their own pretraining/post-training, so verbatim and paraphrased leakage into synthetic outputs is expected to increase with T; if not measured, "bigger teacher is better" could be partly "bigger teacher leaks more test items". Procedure, on the *raw* pools before filtering: (1) item-level contamination rate `ρ_T,b` = fraction of benchmark-b test items with ≥1 13-gram (or full-short-item) hit anywhere in T's pool; (2) example-level rate `r_T,b`; (3) paraphrase audit: for 5k sampled outputs per teacher, embedding top-1 retrieval against ARC-E/ARC-C/SciQ/OBQA/MMLU-STEM items, candidates with cosine ≥ 0.8 judged by an LLM for "same question" (lm-sys llm-decontaminator protocol), giving `ρ^para_T,b`; (4) "benchmark-style density": fraction of outputs that are 4-option science questions (regex), a stylistic leakage that survives n-gram filtering. Report all four as a teacher × benchmark table. Sensitivity analyses: re-evaluate all conditions on the item subset with zero hits in *any* teacher pool (`clean-b`); if `ρ^para` differs across teachers by more than ~2 points on any primary benchmark, also report Z on `clean-b` as the headline and flag the benchmark. Removed examples are replaced by whole-example resampling from the same teacher's pool so that D_syn stays matched (P2).

---

## 8. Open issues and disagreements with the brief

1. **P6 (suite composition).** BoolQ, Winogrande and ARC-Challenge should not be primary: every public model ≤1B trained on ≤300B tokens is at or below the BoolQ majority baseline (62.2), within +4 of chance on Winogrande and within +2 on ARC-Challenge [V]. In the P5 regime they will be pure noise (SE 0.9–1.5). Proposed: demote all three to secondary; add SocialIQA (+8 at 160M) and keep HellaSwag only as Tier B.
2. **P5 (20 tokens/parameter) makes the 25M and 100M cells nearly un-evaluable by accuracy.** Pythia-70M at 1B tokens is at chance on every task including SciQ; at 2B tokens SciQ is +20 but ARC-E/PIQA are +4–5 and LAMBADA acc is 5% [V]. Going from 2B to 6B–27B tokens lifts a 70M model to SciQ 59–65, LAMBADA 12–21, ARC-E 33–35 [V]. Training FLOPs for small students are negligible (6·N·D: 25M × 20B tokens ≈ 3×10^18 FLOP ≈ 10–20 GPU-hours on a 5080), so the compute-optimal budget is the wrong criterion for the small students of an *evaluation* study. Alternatives: (i) same base corpus D_base ≈ 10–20B tokens for every size (cleanest control: identical data, only S varies; the 1B run costs the same as under P5); (ii) at least 100 tokens/parameter for ≤100M. Either way, the smallest size should be re-checked against the floor gate in the pilot before the full grid.
3. **The 25M student.** Even with (2), PRIMARY-4 at 25M rests mainly on SciQ and LAMBADA-ppl. Recommend pre-registering the NLL outcomes (Section 5) as the primary outcome for 25M, or replacing 25M by 50M.
4. **Template confound.** Harness zero-shot prompts (`Question:/Answer:`) resemble instruction data, so part of any synthetic-vs-C1 gain may be format familiarity. Mitigations: render synthetic training examples with distinct chat markers (not the harness template); include the format-adherence probe; consider a control C1-QA (base + matched *human-written* Q/A tokens, e.g. StackExchange/ELI5 under permissive licences) — without it, "instruction-style synthetic data helps" and "Q/A-formatted data helps" are not separable.
5. **P3 teacher choice vs contamination.** Qwen2.5-Instruct post-training is known to be benchmark-heavy [R]; larger Qwen teachers will plausibly emit more ARC/GSM8K-like items. The audit in Section 7 is therefore not optional, and GSM8K should not be interpreted at all unless `ρ_T,gsm8k` is reported.
6. **Aggregates and power (P7).** Per-task item SE is 1–2 points for the 1–2k-item tasks and seed SD is ~1 point [R: Madaan et al. 2024 "Quantifying variance in evaluation benchmarks"]; expected teacher effects at ≤250M on single tasks may be 1–3 points. Power analysis must be done on Z (SE ≈ 0.7–0.85 from items alone) and on HeldOut-Instr NLL, not on single benchmarks. Three seeds per cell will not resolve single-task differences below ~3 points.
7. **Reference-number provenance.** Public numbers mix harnesses (lm-eval vs lighteval `acc_norm_nospace` vs OLMES per-char/PMI), prompts and shot counts; they are useful only as floor indicators. No cross-paper comparison should appear in the paper; the four Pythia checkpoints must be re-run through our pinned pipeline as anchors. Numbers tagged [R] above (SmolLM2 135M/360M, Qwen2.5-0.5B, Cerebras-GPT, IFEval, TruthfulQA, BLiMP, GSM8K for tiny models) were not re-fetched this session and need verification.
8. **Licences.** Dolly (CC BY-SA 3.0) and NQ (CC BY-SA 3.0) impose share-alike on derived datasets; OASST1 is Apache-2.0; No Robots and LIMA are non-commercial. The held-out set as specified is research-safe; confirm before release of any derived artifact.
9. **Equal-token control (P2) and NLL.** Because D_syn is matched in tokens, teachers with longer outputs contribute fewer examples; NLL outcomes are per token and unaffected, but the "equal number of examples" secondary analysis will change the token count per cell — the NLL sets must be evaluated in both analyses, not reused.
10. **BLiMP and grammar drift** are cheap and precise (SE 0.19) but the Pythia JSONs do not contain BLiMP despite the paper listing it; establish our own C0 baseline rather than relying on published values.
