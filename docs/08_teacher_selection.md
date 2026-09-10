# 08. Teacher-Model Selection

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- Adopted: Qwen2.5-Instruct at 1.5B, 3B, 7B, 14B, 32B, 72B as the primary six-rung grid, BF16 weights and KV for every teacher, 72B at tensor-parallel 2.
- **Amendment:** the 0.5B rung (§3, 'optional') is included as a pre-registered seventh rung in the main study; the six-rung grid remains the primary analysis and the seven-rung fit is secondary (doc 00 §0.6).
- The teacher-capability covariate is measured in-house on the student suite with the pinned harness (§8 layer (b)); vendor scores in §8 are reference only and every cell there is [UNVERIFIED].
- FP8-for-all is the pre-registered fallback only under the throughput trigger in docs/16; the 32B BF16-vs-FP8 bridging cell is then mandatory.

---

# Teacher-model selection (workstream deliverable)

Date: 2026-09-10. Scope: choose the open-weight family and exact checkpoints that serve as teachers at ≈1B, 3B, 7B, 14B, 32B, 70B, minimizing confounds other than parameter count.

Verification notes. In this session `huggingface.co`, `arxiv.org`, `modelscope.cn`, `qwen.ai`, `qwenlm.github.io`, `ai.meta.com`, `ai.google.dev`, `allenai.org`, `mistral.ai` and all inference-benchmark blogs were blocked by the egress proxy. Facts below were established from web-search snippets of those pages and from GitHub READMEs (reachable). Anything reconstructed from memory of a primary source that could not be re-opened is tagged **[UNVERIFIED]** and must be re-checked against the model card / technical report before the numbers are used in a paper. Numbers that are derived (VRAM, throughput) show the derivation.

---

## 1. Candidate inventory (what exists on Hugging Face as of 2026-09)

Only *dense* checkpoints are listed as ladder candidates; MoE models are excluded because "teacher size" is ill-defined for them (total vs. active parameters). "Distill/prune" = were the *smaller* sizes derived from larger family members (a direct confound for a teacher-size study).

| Family (release) | Dense sizes (B) | Instruct at every size | Tokenizer shared | Pretraining tokens per size | Small sizes distilled / pruned from larger? | License (per size) | Context | Sources |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5** (2024-09) | 0.5, 1.5, 3, 7, 14, 32, 72 | yes | yes (151,643-token BPE) | 18T reported family-wide; report does not give a per-size breakdown → treat "same corpus for all" as reported-but-unverified per size **[UNVERIFIED per size]** | No pruning/logit-distillation reported. Post-training: same SFT (>1M samples) + offline DPO + online GRPO for all sizes; SFT/pretraining data partly *synthesized by Qwen2-72B/Qwen2.5-72B-class models*, applied uniformly to all sizes | 0.5/1.5/7/14/32: Apache-2.0; **3B: Qwen Research License (non-commercial)**; **72B: Qwen License (commercial OK below 100M MAU)** | 32K (≤3B), 128K via YaRN (≥7B) | [Qwen2.5 report](https://arxiv.org/abs/2412.15115); [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/) (license table); [Qwen2.5-LLM blog](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| **Qwen3 dense** (2025-04; 4B "-2507" refresh 2025-08) | 0.6, 1.7, 4, 8, 14, 32 | yes (hybrid think/no-think) | yes (same Qwen BPE) | 36T reported family-wide (S1 30T @4K, S2 5T STEM/code, S3 long-context) | **Yes (post-training)**: 0.6B–14B dense (+30B-A3B) built by "strong-to-weak distillation" (off-policy response distillation, then on-policy logit-KL distillation) from Qwen3-32B / Qwen3-235B-A22B; only 32B (and 235B) get the full 4-stage pipeline | Apache-2.0 for all sizes | 32K (≤4B), 128K (≥8B); 2507-4B: 256K | [Qwen3 report](https://arxiv.org/abs/2505.09388); [QwenLM/Qwen3 README](https://github.com/QwenLM/Qwen3) |
| **Qwen3.5 dense** (2026-02/03) | 0.8, 2, 4, 9, 27 | yes (thinking default) | Qwen3.5 tokenizer **[UNVERIFIED whether identical to Qwen3 vocab]** | not published in README **[UNVERIFIED]** | Recipe for 0.8B–9B not documented in the README; Qwen team's Qwen3 precedent is strong-to-weak distillation → assume distilled until the Qwen3.5 report says otherwise **[UNVERIFIED]** | Apache-2.0 per HF card **[UNVERIFIED — README defers to model card]** | 262K | [QwenLM/Qwen3.5 README](https://github.com/QwenLM/Qwen3.5); [Qwen tweet on small series](https://x.com/Alibaba_Qwen/status/2028460046510965160) |
| Qwen3.6 / Qwen3.8 (2026-04 / 2026-08) | 27 only (+MoE 35B-A3B, 2.4T-A95B) | yes | — | — | — | Apache-2.0 | 262K | [Qwen3.6](https://github.com/QwenLM/Qwen3.6), [Qwen3.8](https://github.com/QwenLM/Qwen3.8) |
| **Llama 3.1 / 3.2 / 3.3** (2024-07/09/12) | 1, 3, 8, 70 (405 not servable here) | yes | yes (128K tiktoken) | 3.2-1B/3B: up to 9T; 3.1-8B/70B: 15T+ | **Yes**: 1B and 3B = single-shot structured pruning of Llama-3.1-8B + logit distillation from 3.1-8B and 3.1-70B during pretraining | Llama 3.x Community License (derivative models trained on outputs must be named "Llama-…") | 128K | [Meta Llama 3.2 blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Llama 4 (2025-04) | none (Scout 109B-A17B, Maverick 400B-A17B, both MoE) | — | — | — | — | Llama 4 license | 1M/10M | [HF blog](https://huggingface.co/blog/llama4-release) |
| **Gemma 2** (2024-06) | 2, 9, 27 | yes | yes (256K SentencePiece) | 2T / 8T / 13T | **Yes**: 2B and 9B trained with logit distillation from a larger teacher; 27B from scratch | Gemma Terms of Use | 8K | [Gemma 2 report](https://arxiv.org/abs/2408.00118) |
| **Gemma 3** (2025-03) | 1, 4, 12, 27 | yes | yes (262K) | 2T / 4T / 12T / 14T | **Yes**: "all models trained with knowledge distillation" (256 sampled logits/token) | Gemma Terms of Use | 32K (1B), 128K | [Gemma 3 report](https://arxiv.org/abs/2503.19786) |
| **Gemma 4** (2026, report 2026-06) | E2B(2.3 eff.), E4B(4.5 eff.), 12, 31 (+26B-A4B MoE) | yes (-it) | yes | 12T for 12B; others **[UNVERIFIED]** | **Yes**: "similar recipe as Gemma 2 … pre-training with knowledge distillation" | Gemma Terms of Use **[UNVERIFIED for v4]** | 128K (E-sizes), 256K | [Gemma 4 report](https://arxiv.org/abs/2607.02770); [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) |
| **OLMo 2** (2024-11 → 2025-04) | 1, 7, 13, 32 | yes (SFT+DPO+RLVR) | yes | 1B: 4T; 7B: 4.05T; 13B: 5.6T; 32B: 6T (1.5 epochs) | No distillation; all from scratch; data fully released (Dolma/Dolmino) | Apache-2.0 | 4K | [OLMo 2 paper](https://arxiv.org/abs/2501.00656); [OLMo 2 32B blog](https://allenai.org/blog/olmo2-32b) |
| OLMo 3 / 3.1 (2025-11 → 2026) | 7, 32 | yes (Instruct, Think) | yes | Dolma 3 | no | Apache-2.0 | 64K | [Ai2 blog](https://allenai.org/blog/olmo3) |
| **Ministral 3** (2025-12) | 3, 8, 14 | yes (+reasoning) | yes | continued-training budgets **[UNVERIFIED]** | **Yes**: "cascade distillation" — iterative prune→distill from Mistral Small 3.1 (24B) → 14B → 8B → 3B | Apache-2.0 | 128K | [Ministral 3 paper](https://arxiv.org/abs/2601.08584); [The Batch](https://www.deeplearning.ai/the-batch/mistral-uses-cascade-distillation-on-mistral-3-to-build-ministral-family) |
| **Falcon-H1** (2025-05) | 0.5, 1.5, 1.5-Deep, 3, 7, 34 | yes (SFT+DPO) | yes | 0.5B: 2.5T; 1.5B: 3T; 1.5B-Deep: 3T; 3B: 2.5T; 7B: ≈12T; 34B: ≈18T | No distillation/pruning reported **[UNVERIFIED — report not re-opened]**; but *token budgets differ 7×* across sizes | "Falcon-LLM" license, Apache-2.0-based | 256K | [Falcon-H1 paper](https://arxiv.org/abs/2507.22448); [GitHub](https://github.com/tiiuae/falcon-H1) |
| **Hunyuan dense** (2025-07) | 0.5, 1.8, 4, 7 | yes (thinking default) | yes | not published | "training strategies similar to Hunyuan-A13B"; distillation not stated **[UNVERIFIED]** | Tencent Hunyuan Community License: **excludes EU/UK/South Korea**, 100M-MAU cap | 256K | [Hunyuan-7B GitHub](https://github.com/Tencent-Hunyuan/Hunyuan-7B); [license](https://github.com/Tencent-Hunyuan/Hunyuan-0.5B/blob/main/LICENSE) |
| Granite 4.1 / 4.2 (2026) | 3, 8, 30 | yes | yes | ~15T (4.1) | not stated | Apache-2.0 | 128K | [Granite 4.1 blog](https://huggingface.co/blog/ibm-granite/granite-4-1) |
| Phi-4 family (2024-12 → 2025) | 3.8 (mini), 14 | yes | **no** (mini uses a different 200K vocab than Phi-4-14B) | mostly synthetic | Phi-4-reasoning distilled from o3-mini traces | MIT | 16K/128K | [Phi-4-mini card](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| EXAONE 4.0 (2025-07) | 1.2, 32 | yes | yes | — | — | EXAONE AI Model License (non-commercial) | 64K/128K | [EXAONE 4.0](https://arxiv.org/abs/2507.11407) |
| SmolLM3 (2025-07) | 3 | yes | — | 11.2T | no | Apache-2.0 | 64K–128K | [HF](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Nemotron 3 (2025-12 → 2026-06) | none (30B-A3B, 100B, 500B; hybrid Mamba-MoE) | — | — | — | — | NVIDIA Open Model | 1M | [NVIDIA report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) |
| GLM-4.5 / 4.6 / 4.7 (2025–26) | none open at ≤72B dense (355B-A32B, 106B-A12B MoE) | — | — | — | — | MIT | 128K | [zai-org/GLM-4.5](https://github.com/zai-org/GLM-4.5) |

Established fact: as of 2026-09 **no family other than Qwen2.5 offers six dense, instruction-tuned checkpoints spanning ≈1B–70B under one tokenizer and one post-training recipe.** Every 2025–2026 family with ≥5 dense sizes (Qwen3, Qwen3.5, Falcon-H1) either (a) built its small sizes by distillation from its large sizes (Qwen3; Qwen3.5 presumed), or (b) trained sizes on very different token budgets (Falcon-H1), and none reaches 70B.

---

## 2. Scoring table (0–3 per criterion; higher = better for this study)

Criteria: **Av** availability of all needed sizes; **Lic** licensing; **Arch** architecture consistency across sizes; **Inst** instruct checkpoint at every size; **Bench** benchmark strength; **Cost** inference cost/throughput; **Ctx** context length; **Tok** tokenizer consistency; **Repro** fixed checkpoints; **Syn** suitability for synthetic-data generation (format compliance, controllable verbosity, low refusal); **Indep** small sizes *not* distilled/pruned from larger family members; **Corp** same pretraining corpus/token budget across sizes.

| Family | Av | Lic | Arch | Inst | Bench | Cost | Ctx | Tok | Repro | Syn | Indep | Corp | Σ/36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 3 | 2 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 2 | **31** |
| Qwen3 dense | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 0 | 3 | 30 |
| Qwen3.5 dense | 2 | 3 | 2 | 3 | 3 | 2 | 3 | 3 | 2 | 2 | 0* | 1* | 26 |
| Llama 3.1/3.2/3.3 | 1 | 1 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 2 | 0 | 1 | 24 |
| Gemma 3 | 1 | 1 | 2 | 3 | 2 | 2 | 3 | 3 | 3 | 2 | 0 | 0 | 22 |
| Gemma 4 | 1 | 1 | 1 | 3 | 3 | 2 | 3 | 3 | 2 | 2 | 0 | 0* | 21 |
| Gemma 2 | 0 | 1 | 2 | 3 | 1 | 2 | 1 | 3 | 3 | 2 | 0 | 0 | 18 |
| OLMo 2 | 1 | 3 | 3 | 3 | 1 | 3 | 0 | 3 | 3 | 1 | 3 | 1 | 25 |
| OLMo 3/3.1 | 0 | 3 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | 2 | 3 | 2 | 29 (2 sizes) |
| Ministral 3 | 0 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 0 | 0 | 25 |
| Falcon-H1 | 2 | 2 | 2 | 3 | 2 | 1 | 3 | 3 | 3 | 1 | 2* | 0 | 24 |
| Hunyuan dense | 0 | 0 | 2 | 3 | 2 | 3 | 3 | 3 | 2 | 1 | 1* | 1* | 21 |
| Granite 4.1/4.2 | 0 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 2 | 2* | 3 | 30 (3 sizes) |
| Phi-4 | 0 | 3 | 1 | 3 | 2 | 3 | 2 | 0 | 3 | 1 | 1 | 1 | 20 |
| EXAONE 4.0 | 0 | 0 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 2 | 2* | 2* | 25 (2 sizes) |

\* = based on incomplete documentation [UNVERIFIED]. Σ is a tie-breaker only; **Av ≥ 2 and Indep ≥ 2 are gating criteria**, which leaves Qwen2.5 (and, at 5 rungs with a corpus confound, Falcon-H1).

One-line justifications for the leading candidates:

- **Qwen2.5** — Av 3: 0.5–72B, seven dense rungs. Lic 2: two non-Apache rungs (3B research-only; 72B ≤100M MAU). Arch 2: ≤3B use tied embeddings + 2 KV heads, ≥7B untied + 4–8 KV heads (head_dim 128 everywhere except 0.5B = 64). Bench 2: strong for 2024, below Qwen3/3.5 at equal size. Cost 3: standard dense transformer, best-supported path in vLLM. Repro 3: static since 2024-09 (config/README-only commits, to be confirmed by pinning a revision **[UNVERIFIED]**). Syn 2: good format compliance ≥7B; small sizes loop/mix Chinese (see §8). Indep 3: no size-specific pruning/distillation reported. Corp 2: 18T reported family-wide but not per size; pretraining and SFT data include synthetic text from 72B-class Qwen models *for all sizes*.
- **Qwen3 dense** — Av 2: no 70B (235B is MoE). Indep 0: 0.6–14B are explicitly strong-to-weak distilled from 32B/235B (report §4.4), so "small teacher" ≈ "compressed image of the large teacher" — exactly the confound this study must avoid. Repro 2: the 4B was replaced by a 2507 refresh with a different recipe; other sizes were not, so the ladder would mix recipes. Corp 3: 36T for all sizes.
- **Qwen3.5 dense** — Av 2: 0.8–27B, no 70B. Arch 2: hybrid Gated-DeltaNet/attention layers and native multimodality — "parameter count" no longer maps cleanly onto attention compute, and inference cost per parameter differs from dense transformers. Indep 0*: recipe undocumented; Qwen precedent is distillation.
- **Llama 3.x** — Av 1: gaps at 14B/32B; Indep 0: 1B/3B pruned from 8B and logit-distilled from 8B+70B; Corp 1: 9T vs 15T. Lic 1: Llama-naming clause for models trained on outputs.
- **Gemma 2/3/4** — Indep 0: every size below the largest is distilled (Gemma 3: all sizes); Corp 0: 2T→14T; Lic 1: Gemma Terms define models trained on Gemma outputs (synthetic data) as "Model Derivatives" that inherit the terms **[UNVERIFIED wording]**.
- **OLMo 2** — Indep 3, Lic 3, but Av 1 (1/7/13/32, no 3B, no 70B), Ctx 0 (4K), Bench 1, Corp 1 (4T→6T). Valuable as an open-data *replication* family (contamination can be audited).
- **Ministral 3** — Indep 0 by construction (cascade prune-distill from a 24B).
- **Falcon-H1** — six rungs but Corp 0 (2.5T–18T) and hybrid Mamba-2 architecture (Cost 1: slower, less mature vLLM kernels; SSM state instead of KV cache complicates the VRAM/throughput story).

---

## 3. Decision (1): PRIMARY family and exact checkpoints

**Primary = Qwen2.5-Instruct**, confirming the brief's P3. The six rungs, with architecture facts from the Qwen2.5 report/model cards (parameter counts, layers, heads and tying are confident; treat the exact non-embedding counts as **[UNVERIFIED to the last digit]**):

| Rung | HF repo id | Params (total / non-embedding) | Layers | Q/KV heads | Hidden | Tied emb. | License | Native ctx |
|---|---|---|---|---|---|---|---|---|
| "1B" | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B / 1.31B | 28 | 12 / 2 | 1536 | yes | Apache-2.0 | 32K |
| 3B | `Qwen/Qwen2.5-3B-Instruct` | 3.09B / 2.77B | 36 | 16 / 2 | 2048 | yes | **Qwen Research License** | 32K |
| 7B | `Qwen/Qwen2.5-7B-Instruct` | 7.61B / 6.53B | 28 | 28 / 4 | 3584 | no | Apache-2.0 | 128K (YaRN) |
| 14B | `Qwen/Qwen2.5-14B-Instruct` | 14.7B / 13.1B | 48 | 40 / 8 | 5120 | no | Apache-2.0 | 128K |
| 32B | `Qwen/Qwen2.5-32B-Instruct` | 32.5B / 31.0B | 64 | 40 / 8 | 5120 | no | Apache-2.0 | 128K |
| 70B | `Qwen/Qwen2.5-72B-Instruct` | 72.7B / 70.0B | 80 | 64 / 8 | 8192 | no | **Qwen License** (≤100M MAU) | 128K |
| optional 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B / 0.36B | 24 | 14 / 2 | 896 | yes | Apache-2.0 | 32K |

Methodological choices:
- **Size covariate = non-embedding parameters** (log scale). Embeddings are 15% of the 1.5B model but 4% of the 72B; using total parameters would distort the low end. Report both.
- **Pin revisions.** Record the commit SHA of every repo at download time and the SHA-256 of each safetensors shard; load with `revision=<sha>`. Store `tokenizer.json`, `tokenizer_config.json` (chat template), `generation_config.json` alongside the data. The Qwen2.5 repos have had post-release commits (README/config) **[UNVERIFIED which]**; pinning removes the question.
- **Add the 0.5B rung** (free, Apache-2.0) as a seventh point: a teacher *smaller* than the 1B student directly probes H3/weak-to-strong and anchors the low end of any T*(S) fit.
- **Licensing is acceptable for an academic preprint.** The 3B rung is research-only; the 72B rung is fine below 100M MAU. Action item: read the actual `LICENSE` files of `Qwen2.5-3B-Instruct` and `Qwen2.5-72B-Instruct` for any clause on *use of outputs to train other models* before releasing student checkpoints or the synthetic corpus **[UNVERIFIED — text not re-opened]**. Apache-2.0 rungs carry no such clause.

Why not a newer family (hypothesis-level reasoning, stated explicitly): the research question is about *teacher size*, not teacher recency. A 2026 family whose small members are distilled from its large members (Qwen3; Gemma 4; Ministral 3; Qwen3.5 presumably) makes "T = 1.7B" partially a proxy for "T = 235B filtered through a 1.7B bottleneck", which would bias H1–H3 toward flat curves. Qwen2.5's weakness is that *all* its sizes were exposed to synthetic text from 72B-class Qwen models during pretraining/SFT — a uniform, not size-dependent, effect, but one that may attenuate the size gradient. This is a limitation to state, not a confound of the size axis.

---

## 4. Decision (2): least-confounded FALLBACKS

**F0 — Qwen2.5 covers all six rungs (default).** No fallback needed unless (a) the 72B is dropped for cost, (b) the 3B research license is judged unacceptable, or (c) a modern-family replication is wanted.

**F1 — In-family ladder + one cross-family anchor with a bridging experiment.** If rung *r* cannot come from Qwen2.5, take it from family B and estimate the family offset at a scale where both families have a from-scratch, non-distilled checkpoint:

Let \(y_A(T)\) be the student outcome with a family-A teacher of size T. For an anchor \(T_a\) from family B, report the corrected value

\[
\hat y_A(T_a) = y_B(T_a) - \hat\delta,\qquad \hat\delta = \tfrac{1}{|M|}\sum_{T_m\in M}\big[y_B(T_m) - y_A(T_m)\big],
\]

where \(M\) is the set of matched bridge scales. With two bridge scales one can also test whether \(\delta\) depends on T (fit \(\delta(T)=\delta_0+\delta_1\log T\); if \(\delta_1\) is not distinguishable from 0, the anchor is usable; otherwise report the anchor only qualitatively).

| Missing rung | Anchor checkpoint | Bridge pairs (both from scratch, no size-specific distillation) | Notes |
|---|---|---|---|
| 70B | `meta-llama/Llama-3.1-70B-Instruct` (era-matched to Qwen2.5; 3.3-70B has a newer post-training recipe) | `meta-llama/Llama-3.1-8B-Instruct` vs `Qwen/Qwen2.5-7B-Instruct` (M = {7–8B}) | one bridge scale only → δ assumed scale-invariant [assumption]. Llama license naming clause applies to students trained on its outputs. |
| 70B (alt.) | none in OLMo | `allenai/OLMo-2-1124-7B-Instruct` vs Qwen2.5-7B and `allenai/OLMo-2-0325-32B-Instruct` vs Qwen2.5-32B (M = {7B, 32B}) | two bridge scales → δ(T) testable; but OLMo has no 70B, so this only calibrates the *method*. |
| 3B | `meta-llama/Llama-3.2-3B-Instruct` | 1.5B↔Llama-3.2-1B and 7B↔Llama-3.1-8B | Llama 1B/3B are pruned+distilled → the anchor itself carries the confound; use only if the research license is a hard blocker. |

**F2 — Modern-family replication of the pilot (3 rungs) with Qwen3 dense in non-thinking mode** (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`; `enable_thinking=False`). Purpose: check whether the pilot's size ordering survives in a family where small sizes *are* distilled. Interpretation: if the Qwen3 curve is flatter than the Qwen2.5 curve, that is evidence the distillation confound matters — a useful secondary finding, not a replacement primary.

**F3 — Fully-open replication (OLMo 2: 1B, 7B, 13B, 32B).** Only family where teacher pretraining data is public, so contamination of the student benchmark suite via the teacher can be audited. 4K context and weaker instruction following limit it to a robustness check.

Rejected fallbacks: Falcon-H1 (7× token-budget spread across sizes; hybrid SSM), Gemma 3/4 and Ministral 3 (all sizes distilled), Hunyuan (EU/UK/KR exclusion; 4 sizes ≤7B), Qwen3.5 (no 70B, hybrid architecture, small-model recipe undocumented).

---

## 5. Decision (3): instruct vs. base teachers

Default: **Instruct checkpoints** for all rungs (the project generates instruction-style data; base models need few-shot scaffolding and produce more off-format text, which would interact with the filtering rules in a size-dependent way).

Ablation (methodological choice): at two rungs (7B and 32B; optionally 1.5B), generate the same prompt pool with `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-32B` **base** models using a fixed 3-shot template (identical exemplars for both), same decoding, same filters. Train the same student. Purpose: separate "what post-training adds" (including the 72B-synthesized SFT data all instruct sizes saw) from "what scale adds". Prediction to falsify: if the instruct–base gap is larger at 7B than at 32B, post-training compresses the size effect.

---

## 6. Decision (4): serving precision and VRAM

**Recommendation: BF16 weights and BF16 KV cache for all six teachers.** Reason (literature-supported): quantization degradation is size-dependent — small models lose more accuracy at a given bit-width than large ones — so a uniform FP8 recipe still injects a *size-correlated* perturbation. BF16 removes this. The only cost is that the 72B needs both GPUs (tensor-parallel 2 over PCIe; the RTX PRO 6000 has no NVLink).

If BF16 is infeasible for the 72B (e.g., only one card available), fall back to **one FP8 recipe for all six**: W8A8 with per-channel weight scales and per-token dynamic activation scales (llm-compressor "FP8_DYNAMIC"), *not* block-scaled FP8 (vLLM's block-scaled/DeepGEMM path has open crashes on SM120), BF16 KV. Include a bridging cell (32B BF16 vs 32B FP8, same prompts) to measure the precision effect. vLLM on SM120 (compute capability 12.0) has had a series of FP8/NVFP4 kernel gaps ([#32109](https://github.com/vllm-project/vllm/issues/32109), [#47436](https://github.com/vllm-project/vllm/issues/47436), [PR #17280](https://github.com/vllm-project/vllm/pull/17280)); pin one vLLM version for the whole study and verify FP8 output parity on a fixed prompt set before generation.

GPU facts used below: RTX PRO 6000 Blackwell Workstation — 96 GB GDDR7, 1.79 TB/s, dense BF16 tensor throughput ≈ 250 TFLOPS (FP32 accumulate; the 504 figure is with sparsity) and dense FP8 ≈ 500 TFLOPS (sources: [flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition), [BIZON](https://bizon-tech.com/blog/new-rtx-pro-6000-blackwell-gpus-tech-specs)); the FP16-with-FP32-accumulate rate is inferred from the GB202 pattern **[UNVERIFIED for this SKU]**.

KV bytes per token (BF16) = layers × KV_heads × head_dim × 2 (K,V) × 2 bytes:

| Checkpoint | Weights BF16 (GB) | Weights FP8 (GB) | KV/token (KB) | GPUs (BF16) | Usable KV pool, BF16 (GB)† | Max concurrent seqs at mean context 1k / 4k tokens |
|---|---|---|---|---|---|---|
| 1.5B | 3.1 | 1.5 | 28.7 | 1 | ≈82 | 2,900 / 720 (scheduler-limited; use 512) |
| 3B | 6.2 | 3.1 | 36.9 | 1 | ≈79 | 2,100 / 530 (use 512) |
| 7B | 15.2 | 7.6 | 57.3 | 1 | ≈70 | 1,200 / 300 (use 256–384) |
| 14B | 29.4 | 14.7 | 196.6 | 1 | ≈56 | 285 / 71 (use 256 / 64) |
| 32B | 65.0 | 32.5 | 262.1 | 1 (or TP=2) | ≈20 (1 GPU); ≈105 (TP=2) | 78 / 19 (1 GPU); 400 / 100 (TP=2) |
| 72B | 145.4 | 72.7 | 327.7 | **2 (TP=2)** | ≈26 (BF16, TP=2); ≈100 (FP8, TP=2) | 80 / 20 (BF16); 300 / 75 (FP8) |

† Usable = 0.92 × 96 GB per card − weights − ≈3 GB activations/CUDA-graph workspace (vLLM `gpu_memory_utilization=0.92`, as used in the SM120 community guide). The 72B in BF16 is feasible but batch-limited (≈80 sequences at 1k context); this is the single strongest argument for FP8 — weigh it against the size-dependent quantization argument above.

---

## 7. Decision (5): expected generation throughput (vLLM, high batch)

Model (derivation, decode phase): per decode step with batch B and mean live context L,

\[
t_{\text{step}} \approx \max\!\left(\frac{W + B\,L\,k}{\beta\,\mathrm{BW}},\ \frac{2N\,B}{\phi\,F}\right) + t_o,\qquad
\text{tok/s} = B / t_{\text{step}},
\]

with W = weight bytes, k = KV bytes/token, N = parameters, BW = 1.79 TB/s, F = 250 TFLOPS (BF16), efficiency factors β ≈ 0.6 (bandwidth realised), φ ≈ 0.6 (GEMM realised), and per-step overhead \(t_o\) = 3 ms (B ≤ 128), 5 ms (B = 256), 8 ms (B = 512) for scheduling + sampling. For TP=2 over PCIe Gen5, add all-reduce cost ≈ 2·layers·(B·d·2 B)/64 GB/s + 2·layers·50 µs.

Calibration against measured points: (i) vLLM issue [#17788](https://github.com/vllm-project/vllm/issues/17788): Qwen3-32B BF16 on one H100 NVL (3.9 TB/s), `max-num-seqs 32`, ShareGPT-like lengths → 654 tok/s measured; the model above with β = 0.6 gives ≈700 tok/s. (ii) [databasemart Pro 6000 vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000) (snippet only): "Qwen-14B, 5,160 tok/s at high concurrency" — the model gives ≈5.3k tok/s at B = 256, L ≈ 0.5k. (iii) [SM120 community guide](https://github.com/lastloop-ai/vllm-blackwell-guide): single-stream Qwen3.6-27B INT4 ≈ 100–115 tok/s on one RTX PRO 6000, i.e. ≈ 0.6 × (1.79 TB/s ÷ 14 GB) — consistent with β ≈ 0.6. Estimates below assume instruction-style generation with ≈250-token prompts and ≈500-token mean responses (L ≈ 0.75k), which is the relevant regime for P2.

| Checkpoint | Config | B | Bandwidth term (ms) | Compute term (ms) | t_step (ms) | Decode tok/s (est.) | Range quoted | Basis |
|---|---|---|---|---|---|---|---|---|
| 1.5B | 1 GPU | 512 | 13 | 10 | 21 | ≈24,000 | 15–25k | overhead-bound; scheduler/sampling dominates |
| 3B | 1 GPU | 512 | 19 | 21 | 29 | ≈17,500 | 12–18k | mixed |
| 7B | 1 GPU | 256 | 25 | 26 | 31 | ≈8,300 | 6–9k | compute ≈ bandwidth |
| 14B | 1 GPU | 256 | 63 | 50 | 68 | ≈3,800 | 3–5k | bandwidth (KV) bound; matches 5.2k at shorter L |
| 32B | 1 GPU | 112 | 82 | 49 | 85 | ≈1,300 | 1.0–1.6k | weight-bandwidth bound, batch-limited by KV pool |
| 32B | 2 GPUs, DP | 2×112 | — | — | — | ≈2,600 | 2–3.2k | two independent instances |
| 72B | 2 GPUs, TP=2, BF16 | 112 | 81 (+13 PCIe) | 54 | 97 | ≈1,150 | 0.8–1.3k | weight-bandwidth + PCIe all-reduce |
| 72B | 2 GPUs, TP=2, FP8 | 256 | 63 (+13) | 62 | 81 | ≈3,200 | 2.5–4k | only if FP8 chosen for all rungs |

Prefill adds ≈10–20% to wall-clock at these prompt lengths; multiply decode figures by ≈0.85 for end-to-end. All numbers are estimates **[to be replaced by measured `vllm bench throughput` runs with the study's exact prompt pool — the measurement is itself a deliverable of P2]**.

Generation wall-clock for a 1.25 B-token teacher pool (≈1 B student tokens after a 20% filter loss and the ≈1.2× student/teacher tokenizer ratio [assumption]), 2 GPUs: 1.5B ≈ 7 h; 3B ≈ 10 h; 7B ≈ 22 h; 14B ≈ 46 h; 32B ≈ 5.6 days; 72B (BF16) ≈ 13 days; total ≈ 22 days of 2-GPU time, ≈ 1,070 GPU-h. FLOPs per generated token (2N): 3.1 / 6.2 / 15 / 29 / 65 / 145 GFLOP — a 47× spread, versus a ≈21× spread in realised throughput, because small models are overhead-bound. This gap is itself an input to H5 (economic optimum): cost should be measured in GPU-seconds, not FLOPs.

---

## 8. Decision (6): teacher capability covariate (vendor-reported scores)

Use a **capability score** as the alternative x-axis to log-parameters. Two layers: (a) vendor numbers below for reference; (b) **re-measure every teacher locally** with one lm-eval-harness version on (i) MMLU-Redux, GSM8K, IFEval and (ii) the *student* suite (ARC-E/C, PIQA, HellaSwag, SciQ, OBQA, BoolQ, Winogrande, LAMBADA) so that teacher capability and student outcome are on the same axes. Only (b) should be used in fits; (a) is for sanity.

Qwen2.5-Instruct, reported by Qwen ([Qwen2.5-LLM blog](https://qwenlm.github.io/blog/qwen2.5-llm/), Sept 2024). Cells reproduced from memory because the page was blocked — **every cell [UNVERIFIED] until re-checked**:

| Checkpoint | MMLU-Pro | MMLU-redux | GPQA | MATH | GSM8K | HumanEval | IFEval (strict-prompt) | Arena-Hard |
|---|---|---|---|---|---|---|---|---|
| 0.5B-Instruct | 15.0 | 24.1 | 29.8 | 34.4 | 49.6 | 35.4 | 27.9 | — |
| 1.5B-Instruct | 32.4 | 50.7 | 29.8 | 55.2 | 73.2 | 61.6 | 42.5 | — |
| 3B-Instruct | 43.7 | 64.4 | 30.3 | 65.9 | 86.7 | 74.4 | 58.2 | — |
| 7B-Instruct | 56.3 | 75.4 | 36.4 | 75.5 | 91.6 | 84.8 | 71.2 | 52.0 |
| 14B-Instruct | 63.7 | 80.0 | 45.5 | 80.0 | 94.8 | 83.5 | 81.0 | 68.3 |
| 32B-Instruct | 69.0 | 83.9 | 49.5 | 83.1 | 95.9 | 88.4 | 79.5 | 74.5 |
| 72B-Instruct | 71.1 | 86.8 | 49.0 | 83.1 | 95.8 | 86.6 | 84.1 | 81.2 |

Qwen2.5 base models (same source, same caveat): MMLU 5-shot 47.5 / 60.9 / 65.6 / 74.2 / 79.7 / 83.3 / 86.1 and GSM8K 4-shot 41.6 / 68.5 / 79.1 / 85.4 / 90.2 / 92.9 / 91.5 for 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B.

Observation relevant to H2: on GSM8K and MATH the 32B and 72B instruct scores are indistinguishable (95.9 vs 95.8; 83.1 vs 83.1), so a saturation of *student* gains between 32B and 72B would be expected from teacher capability alone; a capability-axis analysis is therefore essential to distinguish "returns saturate in T" from "returns saturate in teacher accuracy".

Cross-family anchors (model cards, from memory, **[UNVERIFIED]**): Llama-3.2-1B-Instruct MMLU 49.3 / GSM8K 44.4 / IFEval 59.5; Llama-3.2-3B-Instruct 63.4 / 77.7 / 77.4; Llama-3.1-8B-Instruct 69.4 / 84.5 / 80.4; Llama-3.1-70B-Instruct 83.6 / 95.1 / 87.5. Qwen3 non-thinking (third-party lm-eval reproduction in [EleutherAI issue #3576](https://github.com/EleutherAI/lm-evaluation-harness/issues/3576)): 0.6B MMLU-Redux 62.5 / GSM8K 71.8 / IFEval-loose 37.8; 1.7B 71.3 / 78.5 / 43.8; 4B 82.1 / 87.2 / 52.5 — for 8B–32B see Qwen3 report Tables 18–19. Hunyuan dense (GitHub README): MMLU 54.0 / 64.6 / 74.0 / 79.8 and GSM8K 55.6 / 77.3 / 87.5 / 88.3 for 0.5B / 1.8B / 4B / 7B.

---

## 9. Decision (7): chat template and system prompt — making prompts equivalent

Established facts about Qwen2.5-Instruct:
1. All seven sizes share one ChatML template (`<|im_start|>system … <|im_end|>`), so the token-level prompt is identical across rungs.
2. **If no system message is supplied, the template silently inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."** ([Qwen docs](https://qwen.readthedocs.io/en/v2.5/inference/chat.html)). Always pass an explicit system message; log the rendered prompt string and its token ids for every request.
3. `generation_config.json` ships non-neutral defaults (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) and vLLM ≥0.7 applies a model's generation config by default. Launch with `--generation-config vllm` (or pass every SamplingParams field explicitly) so all rungs use the study's decoding, not the vendor's.
4. Set `max_model_len` identically (e.g., 8,192) for all rungs regardless of native context; never enable YaRN for the ≥7B models (it changes attention scaling and is not available for ≤3B).

Cross-family rules (only for fallback anchors): use each family's own template — never force ChatML onto Llama/Gemma — but keep the *system text* byte-identical; Gemma has no system role (prepend to the first user turn); Qwen3 requires `enable_thinking=False`, which inserts an empty `<think>\n\n</think>\n\n` prefix that must be stripped from outputs; Hunyuan defaults to thinking and needs `enable_thinking=False` or `/no_think`. Record the exact template hash per family.

---

## 10. Decision (8): known issues and controls

| Issue | Where | Control |
|---|---|---|
| Default system prompt injection | Qwen2.5 template | explicit system message (§9) |
| Vendor decoding defaults (rep. penalty 1.05, top_k 20) | Qwen2.5 `generation_config.json` | override for all rungs; identical seed schedule |
| Repetition loops / degenerate output at ≤1.5B (Qwen docs recommend presence_penalty 1.0–1.5 for small Qwen models) | Qwen2.5-0.5B/1.5B/3B | do **not** tune penalties per size (would confound); instead apply one uniform n-gram-loop filter and *report rejection rate per teacher* |
| Chinese-script tokens in English responses, more frequent at small sizes [hypothesis; anecdotal] | all Qwen | uniform script filter; report per-teacher rate |
| Verbosity grows with size [hypothesis — must be measured]; length affects P2's equal-token control | all families | log response-length distribution per teacher; run the length-matched ablation in P2 |
| Format non-compliance (JSON/markdown) higher at small sizes → survivorship confound through filtering | all | report filter pass-rate per teacher; include an "unfiltered" ablation at 1.5B and 32B |
| Refusals on benign educational prompts [hypothesis; expected low with an educational prompt pool] | all | classifier-based refusal count per teacher, reported |
| Self-identification ("I am Qwen…") and knowledge-cutoff text in outputs | all Qwen | uniform regex removal or uniform retention; document |
| Benchmark contamination via teacher pretraining (closed corpus) | Qwen2.5 | n-gram decontamination of the synthetic pool against every student eval set, applied identically per teacher; OLMo-2 replication (F3) as an auditable check |
| Thinking-mode defaults | Qwen3 (F2), Hunyuan | disable; strip think tags; verify no hidden reasoning tokens in the data |
| Silent checkpoint updates | any HF repo | pin revision SHA + file hashes (§3) |
| FP8 kernel bugs on SM120 | vLLM on RTX PRO 6000 | BF16 primary; if FP8, per-channel/per-token dynamic only, one vLLM version, parity test |
| PCIe-only multi-GPU | 72B TP=2 | budget ≈15% all-reduce overhead; prefer DP for ≤32B |

---

## Open issues and disagreements with the brief

1. **P3 confirmed, with a caveat the brief does not state.** Qwen2.5 is the only viable six-rung family, but *all* Qwen2.5 sizes were pretrained/SFT-ed on synthetic text from 72B-class Qwen models. This does not confound the size axis but may attenuate it; the paper must state it. The instruct-vs-base ablation (§5) partially addresses it.
2. **The "1B" rung is 1.5B (1.31B non-embedding).** Use non-embedding parameters as the covariate and stop calling the rung "1B" in tables. Recommend adding the 0.5B rung (cheap; strengthens H3 and weak-to-strong analyses).
3. **The 3B rung is research-licensed.** Fine for a preprint; blocks any commercial release of the 3B-derived synthetic corpus. Verify output-use clauses in the 3B and 72B licenses before releasing artifacts [UNVERIFIED].
4. **Precision: I recommend BF16 for all, even though it makes the 72B batch-limited (~1.1k tok/s, ~13 days for a 1.25B-token pool on 2 GPUs).** The brief allows "same FP8 recipe for every teacher"; I disagree that this is neutral, because FP8 degradation is size-dependent. If FP8 is chosen for cost, add the 32B BF16-vs-FP8 bridging cell.
5. **H2 is partly pre-determined by teacher capability.** Vendor scores show 32B ≈ 72B on GSM8K/MATH. The analysis plan must fit student outcome against a *measured* teacher-capability axis as well as log-T, or "saturation beyond 32B" is uninterpretable.
6. **Newer-family alternatives are all confounded.** Qwen3 (distilled ≤14B), Qwen3.5 (hybrid GDN, recipe undocumented, no 70B), Gemma 4 (distilled), Ministral 3 (cascade-distilled), Falcon-H1 (7× token-budget spread). Proposed use: Qwen3 non-thinking as a 3-rung *replication* of the pilot to test whether distilled small teachers flatten the curve — a secondary result the brief does not currently plan.
7. **Throughput numbers in §7 are model-based estimates calibrated on three public data points**, not measurements on the target GPU; the first pipeline milestone should replace them with `vllm bench` runs on the actual prompt pool. The 47× FLOP spread vs ≈21× realised-throughput spread means H5 must use measured GPU-seconds, as the brief already states.
8. **Egress restrictions prevented re-opening the primary sources for the benchmark table (§8) and Qwen3 tables.** Every cell there must be re-verified against the model cards before use.
