# 09. Synthetic-Data Generation Methodology

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- **Amendment A (token accounting, supersedes §1.7):** the student trains with full-sequence loss on the rendered document, and D_syn counts all rendered tokens (prompt + markers + response) in the student tokenizer. Reason: doc 00 §0.7. Consequences: (i) category token quotas in §3.3 are computed on total rendered tokens; (ii) the design mean example length becomes ≈ 340 student tokens (≈ 100 prompt + markers, ≈ 240 response), so eq. (1) gives N_pool ≈ 1.25 × 600M / (0.85 × 300) ≈ 2.9M prompts at the least-verbose teacher; the 4.0M pool v1.0 retains ≥ 35% margin; (iii) the prompt-token fraction per teacher is a reported covariate; (iv) the response-only variant becomes an ablation.
- **Amendment B (rendering, supersedes the rendering line of §1.7):** student-visible documents use Alpaca-style markers (`### Instruction:` / `### Response:`) wrapped in the tokenizer's BOS/EOS; the harness zero-shot template (`Question:/Answer:`) is never used in training data. C1b and the held-out instruction NLL set use the same markers.
- Decoding: the study's SamplingParams are passed explicitly to vLLM for every teacher (vendor generation_config defaults are overridden); repetition_penalty 1.05 is recorded in the backend extras file and hashed into the generation manifest.
- The judge and writer model choices (Mistral-Small-3.1 writer; Llama-3.3-70B judge) stand, licences [UNVERIFIED] until checked (docs/25 task 4).

---

# Synthetic-data generation methodology and data-quality measurement plan

Workstream: SYNTHETIC-DATA GENERATION & DATA QUALITY. Companion to `00_brief.md` (P1–P9).
Status: design document (v0.1, 2026-09-10). Nothing here is an experimental result.

Legend used throughout: **[FACT]** established / literature-supported; **[CHOICE]** methodological choice made here; **[ASSUMPTION]** number or behaviour assumed pending pilot measurement; **[HYP]** hypothesis to be tested; **[UNVERIFIED]** could not be checked against a primary source in this session (web search budget exhausted; arxiv.org / huggingface.co / readthedocs egress blocked — only github.com was reachable).

---

## 0. Invariance principle: what "only teacher size varies" means operationally

Every teacher condition T ∈ {1.5B, 3B, 7B, 14B, 32B, 72B} (P3) is a function

    D_T = σ( Φ( G_T( P ; θ_dec, s_sys, s_suffix, τ_chat ) ) ; D_syn, seed )

where the only teacher-indexed component is the generator G_T. All other components are **frozen, versioned, hashed, and identical across T**:

| Symbol | Component | Fixed how | Section |
|---|---|---|---|
| P | prompt pool (prompts, categories, order) | frozen JSONL, `pool-v1.0-<sha256>` | §1 |
| s_sys | system prompt | literal string, sha256 in manifest | §1.6 |
| s_suffix | per-category instruction suffix (format + length) | literal templates | §1.6 |
| τ_chat | chat template | one canonical ChatML string for all sizes | §2.1 |
| θ_dec | decoding config | one `SamplingParams` for all T | §2.1 |
| Φ | filter set (rules + thresholds + versions) | `filters-v1.0` | §2.2 |
| σ | subsampler (equal-token, stratified, seeded) | `subsample-v1.0` | §3.3 |
| tok_S | student tokenizer used for all token accounting | one tokenizer file, sha256 | P4 |
| R | student-visible rendering of an example | one template | §1.7 |

Anything that *cannot* be made teacher-independent (filter yield, truncation rate, refusal rate, contamination-removal rate, length distribution, correctness) is **measured per teacher and reported as a covariate**; it is part of the treatment, not a nuisance (§2.3, §4).

---

## 1. Prompt pool

### 1.1 Sizing arithmetic (revises the brief's "≈2M prompts")

Let D_syn be the synthetic-token budget per cell in **student-tokenizer tokens** (P2, P5). Let, for teacher T and category c:

- L̄_{T,c}: mean *retained* response length in student tokens,
- y_{T,c}: filter yield (fraction of generated examples retained by Φ),
- r = (student tokens)/(teacher tokens) for the same text; **[ASSUMPTION]** r ≈ 1.15 for a 32k BPE vs. Qwen's ~151k vocab on English prose (to be measured on the pilot; r ≈ 1.25–1.3 for a 16k vocab).

Because every teacher must be able to fill the same per-category token quota q_c = s_c^tok · D_syn from whole examples (§3.3), the pool is sized by the **least verbose / lowest-yield teacher**, with a margin m for (i) the length-matched ablation (needs surplus in every length bin), (ii) the equal-examples secondary analysis, (iii) sampling slack:

    N_pool  ≥  m · D_syn / ( min_T  y_T · L̄_T )                     (1)

The brief's 600M / 350 = 1.71M assumes no yield loss, no margin, no verbosity gap, and does not state which tokenizer the 350 is in. With the category mixture of §1.2 the design mean response length is ≈ 240 student tokens (Table 1.2), not 350, because the mixture deliberately contains short-answer categories that match the benchmark suite (P6). Table 1.1 gives N_pool from (1) with m = 1.25, y_min = 0.85 **[ASSUMPTION]**:

**Table 1.1 — Required pool size (millions of prompts).**

| D_syn | L̄_min = 190 (small teacher 20% terser) | L̄_min = 240 (design mean) | L̄_min = 300 |
|---|---|---|---|
| 300M | 2.32M | 1.84M | 1.47M |
| 600M | 4.64M | 3.68M | 2.94M |
| 1B | 7.74M | 6.13M | 4.90M |

**[CHOICE]** Pool v1.0 = **4.0M prompts**, which covers D_syn = 600M whenever the least-verbose teacher retains ≥ 220 student tokens/example at yield ≥ 0.85. Versioning is append-only: if the pilot shows D_syn = 1B is needed, v1.1 appends prompts generated by the same recipe and seeds; v1.0 is a prefix of v1.1, so nothing generated on v1.0 is wasted. The pilot pool v0 is the first 100k prompts of the v1.0 order (stratified-interleaved, §1.5), so pilot outputs are also reusable.

**Generation stopping rule (teacher-agnostic, saves the most on the largest teacher).** Prompts are generated in the frozen pool order. Teacher T stops when, for every category c, retained tokens ≥ m·q_c. Because the order is a seeded stratified interleave, any prefix is itself a stratified random sample of P, and every teacher's prompt set is a prefix of the same sequence (common-random-numbers design; maximises prompt overlap between teacher conditions, which reduces the variance of teacher contrasts). Expected teacher tokens generated per teacher (including the ~15% that Φ discards) is then approximately independent of verbosity:

    Tok_gen(T) ≈ m · D_syn / ( y_T · r )  ≈ 1.25 · 600M / (0.85 · 1.15) ≈ 0.77B teacher tokens        (2)

plus prefill of ≈ N_T × 150–450 prompt tokens. **[ESTIMATE — cost workstream must measure]** At plausible vLLM BF16 decode throughputs on 2× RTX PRO 6000 (72B TP=2 ≈ 0.4–0.7k tok/s; 32B ≈ 1.2–2k; 14B ≈ 3–4k; 7B ≈ 5–8k; ≤3B ≥ 10k), the 72B teacher alone needs ≈ 13–22 days of both large GPUs for D_syn = 600M; the whole teacher sweep ≈ 25–35 GPU-pair-days. Pool size (via D_syn) is therefore the dominant cost lever, which is why the stopping rule and D_syn choice should be set jointly with the cost workstream (see Open issues).

### 1.2 Category mixture

**Table 1.2 — Categories, sources, templates, shares, lengths, verifiability.** Source codes: (a) templated over taxonomy; (b) passage-seeded (Cosmopedia-style); (c) harvested human prompts (prompts only); (d) fixed prompt-writer model (not a teacher, not Qwen). Lengths are soft targets enforced by the instruction suffix (§1.6); token estimates use ≈1.3 Qwen tokens/word and r = 1.15.

| Category | Prompt share | Source mix | Template family (abridged; full text in §1.6) | Target length (words) | Est. student tok/ex | Implied token share | Verifiable? | Benchmarks informed |
|---|---|---|---|---|---|---|---|---|
| IF: instruction following with verifiable constraints | 15% | (d) 60%, (c) 30%, (a) 10% | task + 1–2 IFEval-type constraints ("exactly 3 bullet points", "include the word X", "≤ N words", "all lowercase", "end with '…'") | 80–200 | ≈ 210 | 13.2% | Yes — programmatic constraint checks (Zhou et al. 2023) | held-out NLL; generalisation of format control |
| SAQA: short-answer QA (closed-book factoid, definition, "which of the following") | 15% | (a) 50%, (b) 30%, (c) 20% | "Q: {question}" with answer-first suffix; MCQ variant with 4 options (a-subset only, options generated by writer (d) from the taxonomy entry) | 20–80 | ≈ 75 | 4.7% | Partly — in-pool gold for (a)/(b) items whose answer is in the seed/taxonomy entry; judge for the rest | ARC-E, SciQ, OBQA, BoolQ, LAMBADA-style completion |
| CS: commonsense / everyday physical & social reasoning | 15% | (d) 60%, (a) 40% | "Which is more plausible: A or B? Why?", "What happens if…", "What is the most likely reason…", "To achieve goal G, would you…" (PIQA-like), sentence-completion with two candidates (Winogrande-like, freshly written) | 60–150 | ≈ 150 | 9.4% | Partly — writer-provided intended answer for two-option items (judge-audited) | PIQA, HellaSwag, Winogrande |
| SCI: science (grade-school → intro college) | 20% | (b) 50%, (a) 40%, (d) 10% | "Explain why…", "What would happen to … if …", "Which principle explains …", conditioned on {audience} | 120–250 | ≈ 300 | 25.1% | Partly — seed-grounded facts; judge on V-set | ARC-C, SciQ, OBQA |
| MSR: multi-step reasoning (arithmetic word problems, unit/date reasoning, logic, planning) | 15% | (a) 70% (own symbolic templates), (d) 30% | "Solve step by step … Final answer: <x>" | 100–300 | ≈ 300 | 18.9% | **Yes at scale** — templated items carry a computed gold answer (≈ 400k in-pool verifiable items) | reasoning transfer; correctness mediator |
| EXP: explanation + answer (textbook / "explain X for audience Y") | 12% | (b) 70%, (a) 30% | Cosmopedia-style: "Write a self-contained explanation of {topic} for {audience}; end with a one-sentence answer to {question}" | 200–400 | ≈ 450 | 22.6% | Judge only (V-set) | ARC, SciQ, OBQA, held-out NLL |
| STR: structured-format (JSON / list / table) | 8% | (a) 50%, (d) 50% | "Return only valid JSON matching schema S", "Give a Markdown table with columns … and exactly N rows", "numbered list of exactly N items" | 50–200 | ≈ 180 | 6.0% | **Yes** — parse + schema validation | format compliance; instruction adherence covariate |
| **Total** | 100% | (a) 41% (b) 27% (c) 8% (d) 24% | | | **mean ≈ 240** | 100% | ≈ 30% of pool programmatically verifiable | |

**Argument for this mixture [CHOICE].**
1. *Benchmark alignment without benchmark leakage.* P6 is a commonsense/science multiple-choice suite scored by log-likelihood. The SAQA/CS/SCI/EXP categories supply the same knowledge domains and short question→answer form in fresh wording (Cosmopedia/SmolLM-style synthetic textbooks and QA are the documented recipe for moving these benchmarks at ≤1B scale — Ben Allal et al. 2024; Allal et al. 2025 [FACT]), while §1.5 decontamination and the FLAN exclusion (§1.3) keep benchmark items and formats out.
2. *Verifiable subset at scale.* MSR (own symbolic templates) and STR give ≈ 1.1M programmatically scorable responses per teacher, so teacher correctness can be measured per size without any judge (§4, §5). IF adds programmatic adherence.
3. *Length support.* The categories span 20–400 words, giving the length-matched ablation (§3.4) overlapping support across teachers.
4. *Mediation.* Category is stratified identically across teachers, so per-category teacher effects are estimable and the category mix cannot explain teacher differences.
5. *Diversity via four independent generators of prompts* (templates, seeds, humans, writer LLM) reduces dependence of the conclusions on any one prompt style; (d) alone would import a single model's prompt distribution.

The shares (15/15/15/20/15/12/8) are a design prior; **the pilot fixes the token shares s_c^tok from realised lengths** (Table 1.2's implied shares), and those fixed s_c^tok are what §3.3 controls.

### 1.3 Prompt sources and licenses

Verified via the dataset cards (search-result snippets) on 2026-09-10 unless flagged.

**Table 1.3 — Candidate prompt sources.**

| Source | Content used | License (verified) | Decision | Reason |
|---|---|---|---|---|
| FineWeb-Edu (HuggingFaceTB) | seed passages for (b) | ODC-BY **[UNVERIFIED this session; widely stated]** | **IN** — SEED split, disjoint from BASE and C1-REAL splits (seeded 3-way partition by doc id) | teacher-only conditioning; students never see passages |
| Own topic taxonomy + templates (a) | prompts | n/a (ours) | **IN** | fully controllable, unbounded scale |
| Fixed prompt-writer model (d) | prompts | model-dependent (see §1.4) | **IN** | diversity, persona/audience conditioning |
| WildChat-1M user turns (allenai) | first user turn, English, non-toxic subset | ODC-BY (license changed from AI2 ImpACT to ODC-BY on 2024-06-26, retroactive) https://huggingface.co/datasets/allenai/WildChat-1M | **IN** (c) | real human requests; ODC-BY permits redistribution with attribution |
| databricks-dolly-15k | instruction (+context where needed) | CC BY-SA 3.0 https://huggingface.co/datasets/databricks/databricks-dolly-15k | **IN** (c) | human-written; SA propagates to the released pool (per-item license field) |
| OpenAssistant oasst2 | prompter root messages, English | Apache 2.0 https://huggingface.co/datasets/OpenAssistant/oasst2 | **IN** (c) | human-written |
| No Robots (HuggingFaceH4) | prompts | CC BY-NC 4.0 https://huggingface.co/datasets/HuggingFaceH4/no_robots | **IN for research, flagged NC** | 10k human prompts; NC limits pool redistribution → keep in a separate NC shard |
| UltraChat-200k | prompts | MIT https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | **OUT** | prompts are ChatGPT-generated (opening questions written by GPT); style would import a third model's prompt distribution — (d) already covers model-written prompts under our control |
| FLAN collection / Open-Orca/FLAN | prompts | google-research/FLAN code Apache 2.0 (https://github.com/google-research/FLAN); Open-Orca/FLAN card says CC BY-4.0 but "subject to the licensing of the component datasets" | **OUT** | `flan/templates.py` contains template keys `bool_q`, `arc`, `piqa`, `hellaswag`, `openbookqa`, `winogrande`, `trivia_qa`, `natural_questions` (verified) → using FLAN prompts would inject benchmark train items and formats (task contamination) |
| OpenHermes-2.5 | prompts | "FAFO"/mixed, undetermined per subset https://huggingface.co/datasets/teknium/OpenHermes-2.5/discussions/9 | **OUT** | license indeterminable; large fraction OpenAI-generated |
| LMSYS-Chat-1M | prompts | custom LMSYS-Chat-1M License Agreement (gated; deletion-on-request and termination clauses) https://huggingface.co/datasets/lmsys/lmsys-chat-1m | **OUT** | incompatible with a frozen, redistributable, versioned pool |
| Persona Hub (200k / 370M elite) | persona strings for (d) conditioning | data CC BY-NC-SA 4.0, code MIT (README, https://github.com/tencent-ailab/persona-hub) | **Optional** — default is own personas | NC-SA would propagate to prompts that embed persona text; instead derive ≈ 200k personas by the Text-to-Persona method from SEED passages using the fixed writer model (Ge et al. 2024, arXiv:2406.20094) |
| Cosmopedia v1 prompts | prompts (with seed text) | Apache 2.0 (v1); smollm-corpus/Cosmopedia-v2 ODC-BY | **OUT as prompts; IN as design reference** | responses would be far longer than our targets; Mixtral-conditioned prompt engineering |
| SmolTalk (new subsets) / Tulu-3 SFT mixture | prompts | Apache 2.0 for new SmolTalk subsets; Tulu-3 ODC-BY with NC subsets | **OUT** | largely model-generated (Magpie) prompts; mixed sub-licenses |
| Nemotron post-training datasets | prompts | CC BY 4.0 | **OUT** | model-generated prompts (same reason as UltraChat) |
| Stanford Alpaca | prompts | CC BY-NC 4.0 | **OUT** | GPT-generated prompts, NC |

Harvested (c) prompts pass: English (fastText, p ≥ 0.65), 8–400 words, single-turn, no URLs/PII (Presidio-style regex), not a code request (regex on code fences / "write a function"), not toxic (WildChat-nontoxic subset only), MinHash-deduplicated against the whole pool, and §1.5 decontamination. Expected retained: ≈ 300k (WildChat) + 60k (oasst2) + 15k (Dolly) + 10k (No Robots) ≈ 0.35M ≈ 8% of pool.

### 1.4 Diversity design

- **Topic taxonomy (a), (b), (d).** Three levels: 12 domains → ≈ 200 sub-domains → ≈ 2,000 leaf topics; leaf titles seeded from Wikipedia "vital articles" lists (CC BY-SA titles only) and audited by hand **[CHOICE; counts are design targets]**. Every templated prompt records `(domain, subdomain, leaf)`. Per-leaf caps (≤ 0.2% of pool) prevent topic collapse.
- **Seed passages (b).** 1.1M FineWeb-Edu passages of 150–400 words from the SEED split, sampled uniformly over 200 k-means clusters of passage embeddings (cluster-balanced sampling, as in Cosmopedia's web-cluster approach — Ben Allal et al. 2024, https://huggingface.co/blog/cosmopedia). Teachers are told the passage is background only and must write a self-contained response (§1.6); the student never sees the passage.
- **Audience conditioning.** {young children, middle-school, high-school, college, general adult, professional} sampled with fixed probabilities per category (Cosmopedia-style).
- **Persona conditioning (d).** ≈ 200k personas derived by Text-to-Persona from SEED passages with the writer model; each writer prompt is conditioned on one persona + one taxonomy leaf + one category template ("write one question this person might ask about {leaf} that requires {category-specific property}").
- **Prompt-writer model (d).** Must be (i) not a teacher and (ii) not from the teacher family (to avoid a style affinity that would favour the whole Qwen line uniformly, which is harmless for *contrasts* but confounds absolute comparisons with the C1 real-data control). **[CHOICE]** Mistral-Small-3.1-24B-Instruct (Apache 2.0 **[UNVERIFIED this session]**); fallback Gemma-3-27B-it. Writer decoding: T = 1.0, top-p 0.95, seed per prompt. Writer outputs are frozen into the pool before any teacher runs; the writer is never used again.
- **Prompt-level dedup.** Exact (sha1 of normalised text) + MinHash near-dup (5-gram shingles, 14 buckets × 8 hashes ≈ Jaccard 0.72 threshold; datatrove defaults, verified at https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/dedup/minhash.py) across the whole pool; target < 0.5% near-duplicate prompts.
- **Prompt diversity report** (frozen with the pool): distinct-4 over prompts, taxonomy entropy, persona/audience marginals, embedding cluster-size distribution.

### 1.5 Decontamination protocol

**Targets (benchmark union B).** Test *and* validation splits of every benchmark in P6 (ARC-E/C, PIQA, HellaSwag, SciQ, OpenBookQA, BoolQ, Winogrande, LAMBADA) **plus** every benchmark any workstream might add later (MMLU, CommonsenseQA, GSM8K, TruthfulQA, TriviaQA, NQ-open, SQuAD, RACE, IFEval, MathQA, SVAMP, StrategyQA, MT-Bench/AlpacaEval prompts), the held-out NLL evaluation sets, and the V-set (§5). Each item string = question + all options + answer, normalised (lowercase, Unicode NFKC, punctuation stripped, whitespace collapsed).

**Conventions in the literature [FACT, verified via secondary sources this session]:**
- GPT-3: 13-gram overlap (Brown et al. 2020, arXiv:2005.14165).
- Llama 2: token contaminated if in a ≥10-token n-gram shared with an eval sample; sample flagged if > 80% of its tokens contaminated (Touvron et al. 2023, arXiv:2307.09288).
- Llama 3: same scheme with 8-grams; per-dataset ratio threshold (Dubey et al. 2024, arXiv:2407.21783).
- PaLM: ≥ 70% of a data point's 8-grams found in training → contaminated (Chowdhery et al. 2022, arXiv:2204.02311).
- Dolma/OLMo: paragraph-level Bloom filter, ignoring paragraphs < 13 tokens; removes paragraphs found in eval sets (Soldaini et al. 2024, arXiv:2402.00159; https://github.com/allenai/dolma).
- Qwen2/2.5: 13-gram overlap (Qwen2 report, arXiv:2407.10671; Qwen2.5-Math 13-gram with normalisation; Qwen2.5-Coder 10-gram).
- Rephrased contamination evades n-grams; LMSYS "LLM decontaminator" = embedding top-k retrieval + LLM judgment of rephrase (Yang et al. 2023, arXiv:2311.04850).
- phi-1 used embedding + syntax-based (AST) similarity for HumanEval (Gunasekar et al. 2023, arXiv:2306.11644) **[UNVERIFIED details]**.

**[CHOICE] Our rule — stricter than GPT-3 because benchmark items are short (PIQA goals ≈ 10 words, Winogrande ≈ 20).**
1. *Exact n-gram stage.* Word-level 8-gram (Llama-3 n) over normalised text. A pool item (seed passage, prompt, or response) is flagged if it shares **any** 8-gram with any item in B, or, for B-items shorter than 8 words, contains the whole item. Implemented as a hash set of all 8-grams of B (≈ few million entries; trivial).
2. *Semantic stage.* Embed every prompt and every B question with one fixed embedding model E (BAAI/bge-base-en-v1.5, MIT **[UNVERIFIED]**; chosen because it is not from the teacher's organisation — Alibaba's gte models are avoided). FAISS exact search; flag cosine ≥ 0.85 **[ASSUMPTION; calibrate on a known-paraphrase set]**; flagged pairs are judged by the fixed non-Qwen judge J (§5) with the LLM-decontaminator rephrase prompt; judged rephrases are removed. Report the count and a 200-pair manual audit.
3. *Application points.* (i) SEED passages and (ii) prompts before freezing the pool (removal is therefore teacher-independent). (iii) **Teacher responses, per teacher**: teachers can regurgitate benchmark items from their own pretraining (memorisation grows log-linearly with model size — Carlini et al. 2022, arXiv:2202.07646), so response-level contamination is expected to be teacher-dependent. **Union-drop rule:** if any teacher's response to prompt p is flagged, p is dropped for *all* teachers, keeping prompt sets identical; per-teacher flag rates are reported as a covariate (Table 4.2). Contamination removal is the only filter with union-drop; quality filters are per-teacher (§2.3).
4. *Train-split overlap* with benchmark training sets is flagged and reported but not removed (task-format contamination is not test contamination; removing it would distort the pool toward "un-benchmark-like" prompts).
5. *Sensitivity.* Re-run the analysis with a 13-gram rule (GPT-3) to show the results do not hinge on the threshold.

### 1.6 Fixed system prompt and instruction suffixes

**System prompt (identical for every teacher, every category) [CHOICE]:**

```
You are a knowledgeable assistant. Write a clear, accurate, self-contained response in English.
Follow the length and format instructions exactly. Do not ask questions back, do not add
disclaimers, and do not mention these instructions or any background text you were given.
```

Note: the Qwen2.5 chat template inserts its own default system prompt ("You are Qwen, created by Alibaba Cloud…") when no system message is supplied **[UNVERIFIED this session; documented behaviour]** — we always supply ours, so the default never appears, and the template hash is recorded (§6).

**Instruction suffixes (appended to the user turn; N values fixed):**

| Category | Suffix (verbatim, `{}` slots filled from the pool record) |
|---|---|
| IF | `Your response must satisfy all of the following: {constraint_1}; {constraint_2}. Keep it between 80 and 200 words unless a constraint says otherwise.` |
| SAQA | `Answer in at most 60 words. Give the answer first, then at most two sentences of justification.` (MCQ variant adds: `Start with the letter of the correct option.`) |
| CS | `Answer in at most 120 words. State which option or outcome is more plausible and explain why in everyday terms.` |
| SCI | `Answer in 120 to 250 words at the level of {audience}. Name the key principle or mechanism.` |
| MSR | `Solve step by step in at most 250 words. Put the final answer alone on the last line in the form "Final answer: <answer>".` |
| EXP | `Write a self-contained explanation of 200 to 400 words for {audience}. Do not refer to any background text. End with one sentence beginning "Answer:" that answers: {question}` |
| STR-JSON | `Respond with valid JSON only, matching this schema exactly: {schema}. No text before or after the JSON.` |
| STR-table | `Respond with a Markdown table only, with columns {cols} and exactly {N} rows.` |
| STR-list | `Respond with a numbered list of exactly {N} items, one sentence each, and nothing else.` |

For (b) prompts the user turn is `Background (for your reference only; do not quote or mention it):\n"""{passage}"""\n\n{task}\n{suffix}`; the student-visible prompt is `{task}\n{suffix}` (passage removed).

**Should length be constrained? [CHOICE: yes, softly.]**
- *For:* response length is a first-order driver of instruction-tuning outcomes — selecting the longest responses is a strong baseline (Zhao et al. 2024, arXiv:2402.04833 **[UNVERIFIED this session]**), and Xu et al. 2024 (arXiv:2411.07133) report that larger response generators are not reliably better teachers and that response characteristics such as length/compatibility matter **[UNVERIFIED details]**. Without a target, larger Qwen models are expected to be more verbose **[HYP; pilot measures]**, which would confound T with length. Soft targets also cut truncation waste and put all teachers on comparable support for the length-matched ablation.
- *Against:* constraints change the "natural" output; compliance may differ by size (small models overshoot/undershoot more), which is itself a teacher-dependent covariate; hard caps ("at most N words") can truncate reasoning. We therefore use *soft ranges* (not hard maxima where reasoning is needed), a generous hard cap in tokens (§2.1), and report compliance per teacher. An **unconstrained-length ablation** (2 teachers × 1 student, suffixes without length clauses) tests whether the constraint changes the teacher-size effect.

### 1.7 Student-visible rendering and token accounting [CHOICE]

Each retained example is rendered as one document `R(x) = "{student_prompt}\n\n{response}"` with the **loss masked on prompt tokens**; D_syn counts **response tokens only** (student tokenizer). Rationale: prompt tokens are teacher-independent (real or writer-generated) and would dilute the treatment; masking makes the counted budget exactly the teacher-generated text. Prompt tokens are still consumed as context and are reported (they are ≈ 35% of response tokens on average). The C1 control (matched real tokens) counts all its tokens, since it has no masked prefix; this small asymmetry in context length is disclosed. Ablation: unmasked training on 1 cell to check sensitivity. (Disagreement/gap with the brief: P2 does not say whether prompts count; see Open issues.)

---

## 2. Decoding and controls

### 2.1 Decoding configuration (one `SamplingParams` for all teachers) [CHOICE]

| Parameter | Value | Rationale |
|---|---|---|
| temperature | 0.7 | Qwen's recommended non-thinking setting (Qwen3 cards: T = 0.7, top-p 0.8, top-k 20, min-p 0; Qwen2.5-Coder `generation_config.json`: 0.7/0.8/20/rep 1.05 — verified; Qwen2.5-Instruct assumed identical **[UNVERIFIED; read `generation_config.json` of each size and record]**). Using the vendor-recommended regime avoids penalising small teachers with degeneration at high T while keeping sampling stochastic (greedy is discouraged by the vendor and collapses diversity). |
| top_p | 0.8 | as above |
| top_k | 20 | as above; recorded, and the "neutral" ablation below removes it |
| repetition_penalty | 1.05 | as above (vLLM `repetition_penalty`); applies identically to all T |
| min_p, presence/frequency penalty | 0 | not used |
| max_new_tokens | 1024 | hard cap ≈ 2.5× the longest soft target (400 words ≈ 520 Qwen tokens); truncations are dropped, so the cap must be rarely binding (target < 1% per teacher) |
| n (samples per prompt) | 1 (main) | k = 4 ablation at fixed D_syn (¼ the prompts × 4 samples) in 2 teachers × 2 students, testing prompt-diversity vs. sample-diversity (Bansal et al. 2024 regime, arXiv:2408.16737) |
| seed | global `seed=0` + per-request `SamplingParams.seed = H(pool_version, prompt_id, sample_idx) mod 2^31` | identical seeds across teachers (no cross-model meaning, but makes each teacher's run replayable). vLLM reproducibility additionally requires same hardware + same vLLM version and, offline, `VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch-invariant kernels (https://github.com/vllm-project/vllm/blob/main/docs/usage/reproducibility.md, verified). We do **not** pay the throughput cost of deterministic scheduling; reproducibility is guaranteed at the *dataset* level by hashing (§6), not at the sampling level. |
| chat template | one canonical ChatML string (`<\|im_start\|>system … <\|im_end\|>`) applied with `add_generation_prompt=True`; the template text is extracted from the 7B tokenizer, its sha256 recorded, and asserted equal for every size (if a size's template differs, the canonical string is used for all) | |
| thinking mode | n/a for Qwen2.5-Instruct. If Qwen3 is chosen: `chat_template_kwargs={"enable_thinking": False}`; strip the empty `<think>\n\n</think>` block the template emits; any non-empty think content → drop and count | verified: Qwen3 cards document `enable_thinking=False` and the 0.7/0.8/20 non-thinking settings |
| stop | EOS ids of the template (`<\|im_end\|>`, `<\|endoftext\|>`); no string stops | string stops would interact with content differently per teacher |
| logprobs | `logprobs=0` (sampled-token logprob only) → `cumulative_logprob` per example | free; feeds teacher-confidence covariate (§4) |
| precision | BF16 for all sizes (72B: TP = 2 on 2×96 GB) per P3; if FP8 is needed for cost, the *same* FP8 recipe for all sizes | |

**Decoding ablation ("neutral" settings):** T = 1.0, top-p 0.95, top-k off, repetition_penalty 1.0, on the smallest and largest teacher × 1 student, to check that the teacher-size effect is not an artefact of the family's tuned decoding regime.

### 2.2 Filtering rules Φ (teacher-agnostic, cheap, fixed thresholds) [CHOICE]

All filters are O(N) regex/hash/classifier passes except MinHash (standard). Order matters only for attribution; every example carries all flags (§6).

**Table 2.2 — Filters.**

| # | Rule | Implementation | Threshold / action | Cost | Reported per teacher |
|---|---|---|---|---|---|
| F1 | Truncation / no EOS | `finish_reason == "length"` | drop | free | truncation rate |
| F2 | Empty / too short | student tokens < 8 (SAQA) or < 32 (others); or response is only whitespace/punctuation | drop | free | short rate |
| F3 | Language ID | fastText `lid.176` on response; p(en) < 0.65 (FineWeb/datatrove default, verified) → drop; additionally any CJK-script character ratio > 1% → drop (Qwen models occasionally code-switch to Chinese; expected more frequent in small sizes **[HYP]**) | drop | ~µs/doc | non-English rate, CJK rate |
| F4 | Degeneration / repetition | Gopher repetition thresholds (Rae et al. 2021; datatrove defaults verified): dup line frac 0.30, dup para frac 0.30, dup line/para char frac 0.20, top 2/3/4-gram char frac 0.20/0.18/0.16, dup 5…10-gram char frac 0.15…0.10 | drop | cheap | degeneration rate |
| F5 | Exact duplicate | sha1 of normalised response, within teacher | keep first in pool order | free | exact-dup rate |
| F6 | Near duplicate | MinHash, 5-gram shingles, 14 buckets × 8 hashes (≈ Jaccard 0.72; Lee et al. 2022 used MinHash with a ≈0.8 Jaccard criterion and ExactSubstr ≥ 50 tokens, arXiv:2107.06499 **[n-gram/threshold UNVERIFIED]**), within teacher, within category | keep lowest `H(prompt_id)` per cluster | moderate | near-dup rate (also a diversity metric) |
| F7 | Refusal / meta phrases | regex in first 300 chars: `I'm sorry, but`, `I cannot (help\|assist\|provide)`, `As an AI`, `I am an AI language model`, `I'm unable to`, `I can't help with`; plus self-identification `Qwen\|Alibaba\|Tongyi` anywhere | drop | free | refusal rate, self-id rate |
| F8 | Passage reference (category b only) | regex: `the (passage\|text\|background) (above\|provided\|given)`, `according to the (passage\|text)`, `as mentioned (above\|in the text)`, `the author` | drop | free | passage-ref rate |
| F9 | Format validity (STR; IF constraints) | JSON: `json.loads` + `jsonschema` validate; table: header + separator regex, column count, row count; list: item count; IF: IFEval-style checkers for each constraint type (Zhou et al. 2023, arXiv:2311.07911) | drop on hard format failure (STR); IF constraint failures are **not** dropped (they are the adherence metric) | cheap | format-fail rate, IF adherence |
| F10 | MSR answer line | `Final answer:` line present and parseable | drop if absent (unusable for verification and format) | free | missing-answer rate |
| F11 | Benchmark contamination (responses) | §1.5 stage 1 (+ stage 2 on flagged) | **union-drop across teachers** | cheap | contamination flag rate |
| F12 | Length-range compliance | word count vs. suffix range | **not dropped**; recorded | free | compliance rate |

Explicitly **excluded from Φ**: any model-based quality/helpfulness/correctness judge; any perplexity threshold; any correctness-based rejection on verifiable items (wrong answers stay in — the correctness *rate* is the treatment; a "verified-only" arm is a separate ablation).

### 2.3 Why filtering must not use a quality judge (and what to do if one is used)

1. **Judge bias is teacher-correlated.** LLM judges exhibit verbosity and self-enhancement bias (Zheng et al. 2023, arXiv:2306.05685) and favour their own generations (Panickssery et al. 2024, arXiv:2404.13076) **[UNVERIFIED this session]**; a same-family judge would favour the larger Qwen teachers' style, and even a foreign judge rewards polish that co-varies with size. Any such filter would be a second treatment confounded with T.
2. **A judge filter changes the estimand.** Rejection sampling by quality converts "data the teacher produces under a fixed protocol" into "data the teacher produces that a judge accepts", equalises quality across sizes, and makes the *generation cost per retained token* teacher-dependent — precisely the compute-matched regime of Bansal et al. 2024, which is a different question from P2.
3. **Cost.** 6 teachers × ≈ 4M responses ≈ 24M judge calls.

If a judge is used at all (only as a separately labelled arm): fixed judge J outside the teacher family, pointwise rubric with a fixed absolute threshold, blinded inputs (no teacher name, responses shuffled across teachers, category-specific rubric with explicit length-neutral wording), J's acceptance rate a_T reported per teacher and used as a covariate; the unfiltered dataset remains primary.

### 2.4 Per-teacher filter statistics (reported table, Table 4.2 schema)

For each T and category: N_offered, N_generated, removals by F1–F11, N_retained, retained tokens (teacher and student tokenizers), yield y_{T,c}, and the overlap |P_T ∩ P_T'| / |P_T ∪ P_T'| of retained prompt sets between teacher pairs. Filters are per-teacher (except F11), so retained prompt sets differ slightly; the common-random-numbers order (§1.1) maximises overlap. Robustness check: a "union-retained" variant that keeps only prompts retained by every teacher (identical prompt sets, biased toward prompts all teachers can answer) in 2 cells.

---

## 3. Budget control

### 3.1 Options and the confounds each leaves open

Let ℓ_T = mean retained response length (student tokens), N_T = number of examples in the training set, FLOPs_gen(T) ≈ 2·P_T·(prompt + response tokens) with P_T teacher parameters.

**Table 3.1 — Budget-control options.**

| Control | Fixes | What varies with T | Confound left open | Verdict |
|---|---|---|---|---|
| Equal examples N | N_T = N | tokens D_T = N·ℓ_T; student steps/compute | verbose teachers give more tokens and more student compute; a "better teacher" result may be a "more data" result | secondary analysis |
| Equal synthetic tokens D (student tokenizer) — **P2** | D_T = D_syn; student compute | N_T = D/ℓ_T; length distribution | terse teachers give more examples/prompts (more prompt diversity per token); length itself | **primary** |
| Equal generation FLOPs | FLOPs_gen | tokens ∝ 1/P_T: 1.5B produces 48× the tokens of 72B | with a fixed prompt pool this forces k ≫ 1 samples/prompt for small T or a much larger pool; the question becomes Bansal et al. 2024's (weak-but-cheap vs. strong-but-expensive at matched compute), not "which teacher size at fixed data" | separate analysis; not a control |
| Equal wall-clock / $ | time or cost | tokens ∝ throughput(T), hardware-specific | same as FLOPs but also hardware-, batch- and precision-dependent; not reproducible across labs | reported, not controlled |
| Equal tokens **and** matched length distribution | D_T and F_len | N_T, prompt sets, selection within teacher | selection effect: subsample is not representative of each teacher's natural output | ablation (§3.4) |

### 3.2 Choice and defence [CHOICE — agrees with P2]

Equal student-tokenizer tokens is primary because (i) it holds the *student's* training exposure (tokens seen, optimizer steps, compute) exactly constant, so P(S,T) measures the value per token of what the teacher writes; (ii) it is hardware- and precision-independent and thus reproducible; (iii) the remaining confound (length) is addressed by the length-matched ablation and by reporting N_T and ℓ_T, and the "examples vs. tokens" confound is bounded by the equal-examples secondary analysis (both directions of the confound are then observed). Generation cost is analysed separately as measured GPU-seconds/energy/$ per retained student token (H5), with FLOPs_gen as a hardware-independent companion.

### 3.3 Subsampling algorithm σ (seeded, whole examples, stratified by category, common random numbers)

Inputs: retained examples E_T = {(pid, c, len_student)}; token shares s_c^tok (fixed after pilot); D_syn; global seed s_0 (same for all teachers); tolerance tol = 1024 tokens.

```
for each category c:
    q_c = round(s_c^tok · D_syn)                       # identical quota for all teachers
    order = sort(E_T[c], key = H64(s_0, pid))          # teacher-independent priority order
    acc = 0; S_c = []
    for e in order:
        if acc + e.len <= q_c + tol: S_c.append(e); acc += e.len
        if acc >= q_c: break
    assert q_c - tol <= acc <= q_c + tol
    record N_{T,c} = |S_c|, tokens_{T,c} = acc
S_T = ∪_c S_c ; shuffle S_T with seed s_0 for training order (same shuffle rule for all T)
```

Properties: per-category tokens equal across teachers to within ±1024 tokens (≤ 0.001% of a 100M quota); per-category example counts N_{T,c} = q_c/ℓ_{T,c} differ and are reported; priority by `H(s_0, pid)` means teacher subsets overlap maximally on the same prompts. **Nested D-sweep (P5):** the subset for budget D' < D_syn is the prefix of the same per-category order with quotas s_c^tok·D' — subsets are nested, so the D-term fit uses the same prompts at every D. **Equal-examples secondary set:** per category, the first n_c = min_T N_{T,c} examples in the same order for every teacher (identical example counts; tokens vary).

### 3.4 Length-matched ablation

Goal: isolate content quality from length by giving every teacher the same response-length distribution, category by category.

```
bins b = 12 log-spaced edges over [8, 1024] student tokens
for each (c, b): n_{c,b} = min_T |E_T[c, b]|                 # intersection support
D_lm = min(D_syn, 0.8 · Σ_{c,b} n_{c,b} · mean_len(c,b))       # keep ≥ 20% slack in every bin
f = D_lm / Σ_{c,b} n_{c,b} · mean_len(c,b)                     # scale factor ≤ 0.8
for each T, c, b: take the first floor(f · n_{c,b}) examples of E_T[c,b] in H(s_0, pid) order
```

Report the achieved two-sample Kolmogorov–Smirnov distance between teachers' length distributions (target < 0.01) and between each teacher's matched subset and its natural distribution (quantifies the selection effect). Run in the pilot cells and in a 2×3 subgrid of the full study. The inference is conditional: it estimates the teacher effect *on the common length support*, not for each teacher's natural verbosity.

### 3.5 Measuring and reporting response length

For every T × category, before filtering, after filtering, after σ, and in the length-matched set: histogram in student tokens (32 log-spaced bins) and in words; mean, SD, median, p10/p90, max; fraction at `max_new_tokens`; fraction within the suffix's word range; KS distance to the pooled-over-teachers distribution; N_T and ℓ_T = D_syn/N_T. Also the teacher/student token ratio r_T (should be ≈ constant across T; a drift would indicate vocabulary-usage differences and is itself informative). One figure: per-category violin plots of length by teacher; one table: means and compliance.

---

## 4. Data-quality metrics

Metrics are computed on (i) the equal-token training set S_T (primary) and (ii) a fixed 50k-example stratified sample of it where quadratic-cost metrics are needed. Every model-based metric uses a model outside the teacher family.

**Table 4.1 — Candidate metrics.**

| Metric | Definition / tooling | Cost | Teacher-bias risk | Hypothesis informed | Plan |
|---|---|---|---|---|---|
| Correctness on verifiable subset | exact match after normalisation on MSR templated items, STR schema validity, IF constraints (in-pool, ≈ 1.1M/teacher); judge-scored V-set (§5) | low (programmatic) / low (2k judge calls) | none (programmatic); judge bias controlled by blinding + human audit | H1–H3 (does student gain track teacher correctness?), mediation | **IN (core mediator c_T)** |
| Format compliance / validity | F9/F10 rates | free | none | adherence as a quality facet | **IN** |
| Response length ℓ_T, distribution | §3.5 | free | none | length confound; H3 | **IN** |
| Lexical diversity: distinct-n (n = 1..4; Li et al. 2016), TTR, MTLD (McCarthy & Jarvis 2010), self-BLEU (Zhu et al. 2018) | computed on fixed 5k-example samples (TTR/distinct-n are sample-size dependent; MTLD is length-robust); self-BLEU-4 on 5k | low | none (but length-sensitive → also report on the length-matched set) | H3 mechanism (do larger teachers collapse diversity?) — cf. Chen et al. 2024 | **IN** |
| Semantic diversity: mean pairwise cosine distance; cluster count | E-embeddings of 50k responses; mean of 1 − cos over 10^6 random pairs; cluster count = number of k-means clusters (k = 1,000) with ≥ 5 members, and HDBSCAN cluster count; entropy of cluster-size distribution | low–moderate | E is not Qwen-family; still style-sensitive | H3 mechanism; Chen et al. 2024 report cluster-based diversity correlates with downstream performance (arXiv:2410.15226) | **IN** |
| LLM cluster-agent diversity (Chen et al. 2024) | LLM-guided clustering with self-verification on 10k samples | moderate (LLM calls) | judge-style bias | same as above, finer-grained | **OUT of main; optional with J on 10k** |
| Duplication: MinHash near-dup rate | F6 cluster statistics (rate before removal) | already computed | none | collapse/diversity | **IN** |
| Perplexity of the data under the BASE student of each size S | NLL_S(T) = mean per-token NLL of S_T responses under the C0 base checkpoint of size S; 5 sizes × 6 teachers on 20k examples (≈ 5M tokens) each | low | none (student is fixed before treatment) | **capacity-gap / learnability proxy** (Busbridge et al. 2025, arXiv:2502.08606; Mirzadeh et al. 2020): H3/H4 mechanism | **IN (core mediator ℓ_S(T))** |
| Perplexity under a fixed reference LM | NLL_ref(T) under a ~1B open base model outside the teacher family (OLMo-2-1B or SmolLM2-1.7B **[availability/licence UNVERIFIED]**; note SmolLM2 was trained on Cosmopedia-style text → style prior) | low | style prior of the reference | naturalness/"fluency"; the gap ℓ_S(T) − NLL_ref(T) isolates student-specific difficulty from intrinsic text difficulty | **IN** |
| Teacher self-confidence | mean sampled-token logprob from vLLM `cumulative_logprob` | free | intrinsic to teacher (by design) | calibration of teachers vs. size | **IN (descriptive)** |
| Reasoning-depth proxies | number of numbered/newline steps in MSR/SCI; count of CoT markers (`therefore`, `so`, `step`, `because`); equations per response | free | none | do larger teachers write longer chains? | **IN (descriptive)** |
| Readability (Flesch–Kincaid grade, textstat) | per category | free | none | audience compliance; capacity gap (grade level vs. student size) | **IN (descriptive)** |
| Token entropy | unigram entropy of student-token distribution over S_T; conditional entropy not computed | free | none | vocabulary spread | **IN (descriptive)** |
| Embedding-space coverage | fraction of the 1,000 prompt-embedding clusters of P with ≥ 1 retained response; and benchmark-proximity: fraction of B-questions whose nearest response has cos ≥ 0.6 (distributional proximity, *not* contamination) | low | none | coverage/relevance mediator | **IN** |
| Instruction adherence (programmatic) | IF constraint pass rate; F12 length compliance; STR validity | free | none | teacher controllability vs. size | **IN** |
| Contradiction / hallucination on verifiable subset | subsumed by correctness for programmatic items; for V-set science items, judge flags "contains a false statement" separately from "final answer correct" | low | judge (blinded, audited) | correctness facet | **IN (V-set only)** |
| Answer complexity (parse depth, clause count) | spaCy dependency depth | low | none | weak, redundant with length/readability | **OUT** |
| Human quality ratings | 200-item audit per teacher | high (human time) | — | sanity check of judge | **IN (audit only)** |

### 4.2 Per-teacher covariate table (schema)

Rows = teachers (and optionally category); columns = y_T, truncation, refusal, non-English, CJK, degeneration, near-dup, format-fail, contamination flag rate, ℓ_T, N_T, compliance, c_T (correctness, in-pool and V-set), ℓ_S(T) for each S, NLL_ref(T), distinct-4, MTLD, self-BLEU-4, mean pairwise cosine, cluster count, coverage, FK grade, steps, teacher mean logprob. This table is the input to §4.3.

### 4.3 Mediation analysis design

Let G(S,T) be the student outcome (benchmark-average z-score gain over C1 at fixed D_syn; also held-out NLL) averaged over seeds. With only 6 teacher levels, mediation is **exploratory and pre-registered to at most two mediators per model**.

Pre-registered models (fit by least squares across the 30 cells with student fixed effects a_S; evaluated by leave-one-teacher-out CV error, averaged across held-out teachers):

    M0 (size only):        G(S,T) = a_S + b · log T
    M1 (learnability):     G(S,T) = a_S + β_1 · ℓ_S(T) + β_2 · ℓ_S(T)^2
    M2 (correctness):      G(S,T) = a_S + γ · c_T
    M3 (size + mediator):  G(S,T) = a_S + b · log T + β_1 · ℓ_S(T)
    M4 (diversity):        G(S,T) = a_S + δ · d_T          (d_T = cluster count or mean cosine distance)

Pre-registered tests: (i) *H4 mechanism*: M1 beats M0 in LOTO-CV and β_2 < 0 (inverted-U in student-base NLL — data that is too easy or too hard for S transfers less) — this is the falsifiable form of "capacity gap"; (ii) *H1*: in M3, b is not distinguishable from 0 once ℓ_S(T) is included; (iii) *H3*: the argmax_T of the M1 fit shifts with S. Note the interpretation caveat: low ℓ_S(T) can mean "learnable" or "uninformative"; the quadratic term and the gap ℓ_S(T) − NLL_ref(T) are there to separate these.

Stronger within-teacher test (adds cells): for one teacher (7B) and two students (25M, 250M), split S_T into quartiles of per-example ℓ_S and train on each quartile at D_syn/4 (equal tokens); if gain varies across quartiles as M1 predicts, the mediator has within-teacher causal support that the cross-teacher regression cannot give.

---

## 5. Held-out verifiable set (V-set)

Purpose: measure teacher correctness per size on a fixed instrument, independent of the training pool, with human-checkable gold answers; disjoint from every benchmark in B and frozen before generation (also added to the decontamination targets).

**Table 5.1 — V-set composition (2,000 prompts).**

| Block | n | Construction | Gold | Scoring |
|---|---|---|---|---|
| V1 arithmetic | 300 | seeded script: 2–5 operand add/sub/mul/div, 1–6 digits, mixed decimals | computed | exact match on `Final answer:` after numeric normalisation |
| V2 unit & date conversion | 200 | script over a table of units and calendar rules | computed | exact match (tolerance 0.5% for floats) |
| V3 string/list tasks | 200 | count letters, reverse words, sort numbers, set operations | computed | exact match |
| V4 templated word problems | 400 | 40 own GSM-Symbolic-style templates (Mirzadeh et al. 2024, arXiv:2410.05229; Apple's templates are not reused — licence unverified and they derive from GSM8K-test) × 10 instantiations | template formula | exact match |
| V5 structured format | 200 | JSON-schema / table / list tasks with content from V1–V3 | validator + computed content | validity AND content match |
| V6 factual short answer | 300 | 300 items **written by the team** (entity, date, quantity, definition) with a documented source each; no public QA set reused (public sets are in teacher pretraining; also candidates for future benchmarks) | human | judge J with gold; human audit |
| V7 science explanation | 400 | team-written questions with a short gold answer and 2–3 must-not-contradict facts | human | judge J: final-answer correctness (0/1) and "contains a false statement" (0/1) |

All V-set prompts use the same system prompt, suffixes, and decoding as the pool; each teacher answers each item k = 4 times (8k generations/teacher — negligible cost) so that per-item pass@1 and majority-vote correctness are both reported.

**Judge J [CHOICE].** A fixed instruction-tuned model outside the Qwen family, used **only** for scoring (V6, V7, decontamination rephrase checks) and never for filtering or data generation: Llama-3.3-70B-Instruct (fallback Gemma-3-27B-it; a closed API model is acceptable but hurts reproducibility). Temperature 0; prompt shows question, gold, must-not-contradict facts, and the response with the teacher identity removed and item order shuffled; output a JSON verdict. **Audit:** 200 items (stratified over V6/V7 and over teachers, blinded) labelled by two humans; report inter-annotator κ and judge–human Cohen's κ (pre-registered acceptance: κ ≥ 0.75; if lower, the rubric is revised *before* any teacher comparison is reported, and the revision is versioned). Programmatic blocks (V1–V5) need no judge.

In-pool verifiable subset (MSR templated items, STR, IF) gives the large-sample correctness estimate c_T used in §4.3; the V-set gives the human-anchored estimate and the factual/scientific facets.

---

## 6. Generation pipeline specification

### 6.1 Execution

- **Engine:** vLLM offline `LLM.generate` (pinned version, pinned CUDA/driver), BF16, TP = 1 for ≤ 32B if it fits with adequate KV cache, TP = 2 for 72B (and 32B if throughput favours it — recorded). Same engine version and the same GPU type for all teachers; GPU count per teacher is a cost variable, not a data variable, and is recorded.
- **Ordering:** prompts processed in frozen pool order, in shards of 50,000 consecutive prompts; vLLM schedules freely inside a shard; outputs are written **in pool order**, one JSONL file per shard, so shard k of every teacher covers exactly the same prompt ids.
- **Stopping rule:** §1.1 (per-category retained-token quota with margin m), evaluated at shard boundaries after running Φ on the shard; recorded in the manifest.
- **Timing/energy:** per shard: wall-clock, generated tokens, tokens/s, NVML power samples at 1 Hz integrated to kWh, GPU count; per example: `gen_shard` id (per-request latency is meaningless under continuous batching; cost is attributed at shard level and pro-rated by tokens).
- **Determinism controls:** global `seed=0`, per-request seeds (§2.1); no deterministic-scheduling flag (throughput); reproducibility is at the artefact level via hashes.

### 6.2 Per-teacher manifest (`manifest_<teacher>.json`)

```json
{
  "manifest_version": 1,
  "teacher": "Qwen2.5-7B-Instruct",
  "hf_repo": "Qwen/Qwen2.5-7B-Instruct", "hf_revision": "<commit sha>",
  "weights_sha256": "<sha256 over sorted safetensors shard hashes>",
  "dtype": "bfloat16", "tensor_parallel": 1, "gpu": "RTX PRO 6000 Blackwell 96GB", "n_gpus": 1,
  "vllm_version": "x.y.z", "torch_version": "...", "cuda_version": "...", "driver": "...",
  "chat_template_sha256": "...", "system_prompt_sha256": "...", "suffix_set_version": "suffix-v1.0",
  "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05,
               "max_tokens": 1024, "n": 1, "global_seed": 0, "seed_rule": "H(pool_version,prompt_id,sample_idx) mod 2^31",
               "stop_token_ids": [151645, 151643], "logprobs": 0, "enable_thinking": null},
  "pool_version": "pool-v1.0-<sha256[:12]>", "student_tokenizer_sha256": "...",
  "filter_version": "filters-v1.0", "stopping_rule": {"margin": 1.25, "quota_tokens_by_category": {...}},
  "shards": [{"path": "shard-00000.jsonl.zst", "sha256": "...", "n_prompts": 50000, "n_retained": 43120,
              "tokens_teacher": 10612345, "tokens_student": 12203987, "wall_s": 1834.2, "tok_per_s": 5786.1,
              "energy_kwh": 0.61, "started_at": "...", "finished_at": "..."}],
  "filter_stats": {"F1": 412, "F2": 88, "...": "..."},
  "n_prompts_offered": 2350000, "n_retained": 2011400,
  "dataset_sha256": "<sha256 over sorted shard sha256s>",
  "subsample_versions": {"equal_tokens_600M_seed0": "<sha256>", "equal_examples_seed0": "<sha256>", "length_matched_seed0": "<sha256>"}
}
```

### 6.3 Per-example JSONL schema (one line per generated sample; retained and dropped alike, so filter effects are auditable)

| Field | Type | Notes |
|---|---|---|
| `example_id` | str | `"{pool_version}:{teacher}:{prompt_id}:{sample_idx}"` |
| `prompt_id` | str | zero-padded 10 digits; position in pool order |
| `category`, `subcategory` | str | e.g. `science`, `physics.thermodynamics` |
| `prompt_source` | str | `templated` / `seeded` / `harvested:wildchat` / `writer` |
| `prompt_license` | str | per-item (ODC-BY, CC BY-SA 3.0, Apache-2.0, CC BY-NC 4.0, ours) |
| `seed_doc_id`, `persona_id`, `audience` | str/null | conditioning provenance |
| `teacher`, `teacher_revision` | str | |
| `sample_idx`, `seed` | int | |
| `messages` | list | exact teacher input (system + user), for replay |
| `student_prompt` | str | student-visible prompt (passage removed) |
| `response` | str | raw teacher output (think-block stripped if Qwen3) |
| `finish_reason` | str | `stop` / `length` |
| `tokens_teacher_prompt`, `tokens_teacher_response` | int | teacher tokenizer |
| `tokens_student_prompt`, `tokens_student_response` | int | student tokenizer (the accounting unit) |
| `cum_logprob`, `mean_logprob` | float | teacher self-confidence |
| `gen_shard` | str | links to shard timing/energy in the manifest |
| `verifiable`, `gold`, `verified_correct` | bool/obj/bool-or-null | programmatic items only |
| `flags` | obj | `{truncated, short, lang, lang_p, cjk, gopher_rep, exact_dup, near_dup_cluster, refusal, self_id, passage_ref, format_ok, answer_line, contam_8gram, contam_semantic, len_compliant}` |
| `retained` | bool | conjunction of drop rules |
| `hash` | str | sha256(`prompt_id|teacher|response`) |

### 6.4 Hashing and versioning

- Pool: `pool-v1.0-<sha256 of the prompt JSONL>`; the order file is part of the hashed content.
- Filters/subsampler/suffixes/system prompt/tokenizer: versioned strings with sha256 of their source files; all appear in the manifest and in every student training-run config.
- Dataset per teacher: sha256 over sorted shard hashes; each derived training set (equal-token, equal-examples, length-matched, nested-D prefixes) is materialised once, hashed, and referenced by hash in training configs. A training run is reproducible from `(base_checkpoint_hash, dataset_hash, train_config_hash, seed)`.

---

## 7. Pilot checklist for this workstream (before the full pool is frozen)

1. Measure r_T, ℓ_{T,c}, y_{T,c}, truncation and compliance on pool v0 (100k prompts) for the pilot teachers; fix s_c^tok, N_pool (eq. 1), max_new_tokens.
2. Verify template hash equality across Qwen2.5 sizes; read every size's `generation_config.json` and record.
3. Calibrate the semantic-decontamination threshold and audit 200 flagged pairs.
4. Run the judge audit (κ) on V6/V7 with the pilot teachers.
5. Confirm that the union-drop contamination rate and the CJK/refusal rates are < 1% for every teacher; otherwise revise regexes before freezing `filters-v1.0`.

---

## Open issues and disagreements with the brief

1. **Pool arithmetic (disagreement).** "≈2M prompts for ≥600M tokens" is 2–3× too low once yield, the least-verbose teacher, the design mean length (≈ 240 student tokens with a benchmark-aligned mixture) and ablation margins are included (eq. 1, Table 1.1). Recommend pool v1.0 = 4.0M prompts with the teacher-agnostic stopping rule; or raise D_syn only if the cost workstream confirms the 72B budget (≈ 13–22 GPU-pair-days at D_syn = 600M **[ESTIMATE]**).
2. **What counts toward D_syn (gap in P2).** Recommend response tokens only, with prompt-token loss masked, so the counted budget is exactly teacher-generated text; C1 then has a small context-length asymmetry that must be disclosed. If the lead prefers unmasked training, the token quota must include prompt tokens and category quotas must be re-derived.
3. **FLAN should be excluded (disagreement with source list (c)).** FLAN v2 templates include BoolQ, ARC, PIQA, HellaSwag, OpenBookQA and Winogrande (verified in `flan/templates.py`); harvesting FLAN prompts would inject benchmark train items and formats. LMSYS-Chat-1M (custom revocable licence) and OpenHermes-2.5 (indeterminate licence) are also excluded; WildChat-1M (ODC-BY), Dolly (CC BY-SA), oasst2 (Apache-2.0), No Robots (CC BY-NC, research-only shard) are in.
4. **Identical filtering ≠ identical prompt sets.** Even with identical rules, retained prompt sets differ by teacher; this is part of the treatment and is handled by reporting yields, the common-random-numbers order, and a union-retained robustness variant. Contamination removal is the one filter that must be union-dropped across teachers.
5. **Teacher-family confound not removable by this workstream.** Small Qwen2.5-Instruct models may have been post-trained on data produced by larger Qwen models **[UNVERIFIED]**, so "teacher size" partially carries "distilled from the largest teacher" for the small sizes. Same issue is worse for Qwen3 (brief already notes) and Llama 3.2. Must be disclosed; a second family at 3 sizes (e.g., Gemma 3 1B/4B/27B) would be the only real check.
6. **Decoding regime.** Using the family's recommended sampling (0.7/0.8/20/1.05) is a defensible fixed protocol but is tuned by the vendor; the "neutral" decoding ablation (§2.1) is needed before claiming size effects are decoding-independent. `repetition_penalty` in particular may interact with model size in unknown direction.
7. **Length constraints change the estimand.** Soft length targets are recommended, but the study then answers "which teacher size under a fixed length-controlled protocol"; the unconstrained ablation is required to say anything about natural verbosity.
8. **Mediation is under-powered with 6 teacher levels.** Cross-teacher regressions (§4.3) are exploratory; the within-teacher quartile split by student-base NLL is the only design here with real causal leverage and should be budgeted (≈ 8 extra training runs).
9. **Benchmark suite vs. correctness story (for the evaluation workstream).** P6 has no generative, verifiable evaluation; if teacher correctness (c_T) is to be linked to student capability, at least one held-out generative verifiable eval (own templated arithmetic/word problems, never GSM8K test) should be added.
10. **Release licensing.** The pool mixes ODC-BY, CC BY-SA 3.0, Apache-2.0, and CC BY-NC 4.0 items; release per-item licence fields and keep NC items in a separable shard; if Persona Hub personas are used, the resulting prompts inherit CC BY-NC-SA 4.0 — the self-derived persona route avoids this.
11. **Judge and writer model licences [UNVERIFIED this session]:** Mistral-Small-3.1 (Apache 2.0) as writer and Llama-3.3-70B-Instruct as judge are proposed; the Llama licence's naming clause matters only if Llama outputs enter training data, which they do not (judge is scoring-only). Confirm before freezing.
12. **Unverified items to close before the pool is frozen:** Qwen2.5-Instruct `generation_config.json` per size; Qwen2.5 template default-system-prompt behaviour; FineWeb-Edu licence; bge-base licence; exact Lee et al. 2022 MinHash settings; Zhao 2024 / Xu 2024 / Panickssery 2024 details; OLMo-2-1B availability.

---

### References (author–year; arXiv ids from memory where the abstract page could not be fetched)

Allal et al. 2025 SmolLM2 (arXiv:2502.02737) · Bansal et al. 2024 (arXiv:2408.16737) · Ben Allal et al. 2024 Cosmopedia (https://huggingface.co/blog/cosmopedia; https://github.com/huggingface/cosmopedia) · Brown et al. 2020 (arXiv:2005.14165) · Busbridge et al. 2025 (arXiv:2502.08606) · Carlini et al. 2022 (arXiv:2202.07646) · Chen et al. 2024 (arXiv:2410.15226) · Chowdhery et al. 2022 (arXiv:2204.02311) · Dubey et al. 2024 Llama 3 (arXiv:2407.21783) · Ge et al. 2024 Persona Hub (arXiv:2406.20094; https://github.com/tencent-ailab/persona-hub) · Gunasekar et al. 2023 (arXiv:2306.11644) · Lee et al. 2022 (arXiv:2107.06499; https://github.com/google-research/deduplicate-text-datasets) · Li et al. 2016 distinct-n (arXiv:1510.03055) · Longpre et al. 2023 Flan Collection (arXiv:2301.13688; https://github.com/google-research/FLAN) · McCarthy & Jarvis 2010 MTLD (Behav. Res. Methods 42:381–392) · Mirzadeh et al. 2024 GSM-Symbolic (arXiv:2410.05229; https://github.com/apple/ml-gsm-symbolic) · Panickssery et al. 2024 (arXiv:2404.13076) · Penedo et al. 2024 FineWeb (arXiv:2406.17557; https://github.com/huggingface/datatrove) · Qwen Team 2024 Qwen2 (arXiv:2407.10671), Qwen2.5 (arXiv:2412.15115) · Rae et al. 2021 Gopher (arXiv:2112.11446) · Soldaini et al. 2024 Dolma (arXiv:2402.00159; https://github.com/allenai/dolma) · Touvron et al. 2023 Llama 2 (arXiv:2307.09288) · vLLM reproducibility (https://github.com/vllm-project/vllm/blob/main/docs/usage/reproducibility.md) · Xu et al. 2024 (arXiv:2411.07133) · Yang et al. 2023 LLM decontaminator (arXiv:2311.04850; https://www.lmsys.org/blog/2023-11-14-llm-decontaminator/) · Zhao et al. 2024 Long is more (arXiv:2402.04833) · Zhao et al. 2024 WildChat (arXiv:2405.01470) · Zheng et al. 2023 MT-Bench (arXiv:2306.05685) · Zhou et al. 2023 IFEval (arXiv:2311.07911) · Zhu et al. 2018 Texygen (arXiv:1802.01886). Dataset cards: WildChat-1M, Dolly-15k, oasst2, No Robots, UltraChat-200k, OpenHermes-2.5, LMSYS-Chat-1M, PersonaHub, smollm-corpus, SmolTalk, Tulu-3 SFT mixture, Nemotron post-training, Alpaca (huggingface.co, accessed via search snippets 2026-09-10).
