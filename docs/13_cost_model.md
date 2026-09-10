# 13. Cost Model, Hardware Facts, and Generation Throughput (derivation)

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- Hardware facts, formulas, throughput model, STE derivation (Task E) and redesign analysis are adopted. The budget tables in this document use D_syn = 150M and 20 tokens/param and are superseded by docs/16, which recomputes everything under the locked design (D₂ = 800M with 600M treatment; D_base = max(20·N_tot, 5B); seven pools; pilot with 54 branches per size; C1b; tier-2 seeds; D-sweep; the local-5080 profile).
- Open issue 1 (base seeds) is resolved by doc 00 §0.10. Open issue 4 (pool vs D-sweep overlap) is resolved by using nested prefixes for the D-sweep and the whole pool at 600M. Open issue 5 (evaluation size) is resolved by the 5k HellaSwag subsample. Open issue 6 (context) is resolved at 2048 with intra-document masking.

---

# Compute budget and cost model (workstream: COMPUTE BUDGET & COST MODEL)

Date of hardware/price lookup: 2026-09-10. Calculator: `scratchpad/code/compute_budget.py` (pure Python; every table below is its output or is derived from constants printed by it). Conventions: **GPU-h** = hours summed over all cards a job occupies (a TP=2 job running 10 h = 20 GPU-h); **wall** = wall-clock from a list-scheduling simulation with dependencies; **tokens** = student-tokenizer tokens unless stated.

Legend for claims: **[fact]** verified against a source; **[derived]** arithmetic from facts; **[assumption]** modelling choice; **[UNVERIFIED]** could not be checked (many primary pages — nvidia.com, tomshardware, techpowerup, vast.ai, huggingface.co, arxiv.org — are blocked by this session's egress proxy; where a number comes only from search-engine snippets of those pages it is flagged).

---

## Task A — Hardware facts

| GPU | VRAM | Mem. BW | BF16 tensor, dense | FP8 dense | TDP | Street price (Sept 2026) | Cloud rental |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell **Workstation** | 96 GB GDDR7 ECC, 512-bit [fact] | 1,792 GB/s [fact] | **≈504 TFLOPS** [derived, see below] | ≈1,008 TFLOPS [derived] | 600 W [fact] | NVIDIA store $16,000; Newegg $13,998; B&H $15,499 [fact via snippets]; launch $8,565 (Mar 2025); calculator uses **$15,500** | Direct rentals exist: median on-demand $1.87–2.20/GPU-h (getdeploying, 8 Sept 2026); RunPod $1.99/h (Jul 2026); Vast.ai Server-edition listings from $0.93/h; Spheron $2.29/h. Calculator uses **$2.00/GPU-h** |
| RTX PRO 6000 Blackwell **Max-Q** | 96 GB, same die | 1,792 GB/s [fact] | ≈439 TFLOPS [derived from 3,511 AI TOPS] | ≈878 | 300 W [fact] | Launch listing $8,299.99 at Newegg (2025, slickdeals); trackers report ≈$14.7k (2026) [UNVERIFIED] | as above |
| RTX PRO 6000 Blackwell **Server** | 96 GB, passive, headless | 1,792 GB/s | ≈469 TFLOPS [derived from 3,753 AI TOPS] | ≈938 | 600 W (configurable) [UNVERIFIED for lower cap] | ≈$13.1k (June 2026 tracker) [UNVERIFIED] | Vast.ai $0.93/h |
| RTX 5080 | 16 GB GDDR7, 256-bit, 30 Gbps [fact] | **960 GB/s** [derived: 256 bit × 30 Gbps / 8] | **112.6 TFLOPS** dense with FP32 accumulate (225 with FP16 accumulate) [fact: Puget/WareDB spec sheets] | 225 (FP32 acc.) | 360 W [fact] | MSRP $999; street $1,580–1,800 (Sept 2026 trackers) [fact via snippets]; calculator uses $1,600 | $0.10–0.59/GPU-h (getdeploying/RunPod); calculator uses $0.35 |
| H100 SXM (reference) | 80 GB HBM3 | 3,350 GB/s | 989 TFLOPS | 1,979 | 700 W | purchase [UNVERIFIED], ~$25–30k | $2.69–3.99/GPU-h (RunPod, Together, Thunder) |

**Deriving dense BF16 for the PRO 6000.** NVIDIA's headline "4,000 AI TOPS" is FP4 with 2:4 sparsity. The Blackwell tensor-core rate ladder is ×½ per step (sparse→dense, FP4→FP8, FP8→FP16): 4,000 → 2,000 (FP4 dense) → 1,000 (FP8 dense) → **500 TFLOPS BF16 dense**. Cross-check: 188 SMs × 2.617 GHz boost × 1,024 dense FP16 FLOP/clk/SM (same per-SM tensor rate as Ada; the RTX 5090 whitepaper gives 419 TFLOPS dense FP16 for 170 SMs × 2.407 GHz, i.e. 1,024/clk/SM) = 503.8 TFLOPS. Several spec aggregators list "504 TFLOPS FP16 / 1,008 FP8", consistent with dense. Unlike GeForce, professional Blackwell runs FP32-accumulate at full rate (GeForce halves it: RTX 5080 = 225 → 112.6), so 504 is the right peak for bf16 training with fp32 accumulation. Max-Q and Server editions use the same die at lower clocks: 3,511 and 3,753 AI TOPS → 439 and 469 TFLOPS BF16 dense [derived]. All three have identical VRAM and bandwidth, so generation (memory-bound) throughput is nearly edition-independent; training is 8–13 % slower on Max-Q.

**Power/electricity assumption [assumption].** $0.15/kWh, PUE 1.2 (workstation in an office with cooling overhead), +100 W host share per GPU; average draw while running = 85 % TDP (training), 75 % (generation), 50 % (evaluation). Note the practical constraint: 2 × 600 W Workstation cards + host ≈ 1.6 kW on one machine; on a 120 V/15 A circuit (1.8 kW) that is marginal, which is an argument for the 300 W Max-Q edition if the two big cards share a chassis.

**Cloud-equivalent rental.** Because the PRO 6000 is directly rentable ($0.93–2.29/h), no proxy is needed; L40S (~$0.8–1.2/h) and RTX 6000 Ada (~$0.7–1.0/h) are the previous-generation analogues. Ownership-amortised cost for comparison: $15,500 / (3 y × 8,760 h × 70 % utilisation) ≈ $0.84/GPU-h + ≈$0.11/h electricity ≈ **$0.95/GPU-h owned** vs **$2.00 cloud-equivalent**.

Sources (author/site, year, URL): NVIDIA product pages/datasheets (blocked here) https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ and https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf ; Tom's Hardware 2026 price articles https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year and https://www.tomshardware.com/pc-components/gpus/nvidia-raises-rtx-pro-6000-blackwell-gpu-pricing-to-usd13-250-55-percent-increase-over-msrp-in-a-years-time ; TechPowerUp https://www.techpowerup.com/351549/nvidia-rtx-pro-6000-blackwell-96-gb-gpu-now-costs-usd-16-000 ; getdeploying cloud index https://getdeploying.com/gpus/nvidia-rtx-pro-6000 and https://getdeploying.com/gpus/nvidia-rtx-5080 ; Vast.ai https://vast.ai/pricing/gpu/RTX-PRO-6000-S ; RunPod https://www.runpod.io/gpu-models/rtx-pro-6000 ; edition comparison (AI TOPS per edition) https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/ and https://acecloud.ai/blog/rtx-pro-6000-blackwell-server-vs-workstation-vs-max-q/ ; RTX 5080 specs https://www.asus.com/us/motherboards-components/graphics-cards/prime/prime-rtx5080-16g/techspec/ , https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/ ; RTX 5080 price trackers https://videocardprices.com/card/nvidia-rtx-5080/ , https://gpupricehistory.com/us/rtx-5080 ; H100 https://www.thundercompute.com/blog/nvidia-h100-specs-full-guide , https://www.runpod.io/gpu-models/h100-sxm .

---

## Task B — Formulas and efficiency assumptions

### B.1 Student training compute

Per-token training FLOPs (PaLM Appendix-B accounting, LM-head matmul included, input-embedding lookup free):

C_tok(S) = 6 · (N_nonemb + V·d) + κ · 12 · L · d · s,  κ = 0.5 [assumption: FlashAttention/SDPA skips fully-masked causal blocks, so ≈ half of the non-causal 12·L·d·s]

C_train = C_tok · D;  wall-clock = C_train / (P_peak · MFU).

Provisional student shapes (architecture workstream owns the final dims; these only set the FLOP counts; V = 32,768, s = 2,048, SwiGLU ffn ≈ 8/3·d rounded to 64, N_nonemb = L(4d² + 3d·ffn)):

| S | d | L | ffn | N_nonemb | N_flop = N_nonemb + V·d | FLOP/token | attention share | MFU PRO 6000 | h per 1 B tokens (PRO 6000) | MFU 5080 | h per 1 B tokens (5080) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25M | 512 | 8 | 1408 | 25.7 M | 42.5 M | 0.31 GFLOP | 16.5 % | 0.20 | 0.84 | 0.25 | 3.0 |
| 100M | 768 | 14 | 2048 | 99.1 M | 124.3 M | 0.88 | 15.1 % | 0.28 | 1.73 | 0.33 | 6.6 |
| 250M | 1024 | 20 | 2752 | 253 M | 286.5 M | 1.97 | 12.8 % | 0.35 | 3.10 | 0.38 | 12.8 |
| 500M | 1280 | 25 | 3456 | 496 M | 538 M | 3.62 | 10.9 % | 0.40 | 4.99 | 0.35 (activation ckpt.) | 25.5 |
| 1B | 1792 | 26 | 4800 | 1,005 M | 1,064 M | 6.95 | 8.2 % | 0.42 | 9.13 | does not fit (16 B/param > 16 GB) | — |

Note the LM head: with a 32k vocabulary the head is 40 % of the 25M model's FLOPs — the tokens/param budget should be stated against N_nonemb (Kaplan convention) but FLOPs must count V·d.

**MFU assumptions [assumption, with evidence].** Single GPU, bf16 autocast with fp32 accumulate, `torch.compile`, flash/SDPA attention, fused AdamW, ≥0.5 M-token batches:
- Evidence: llm.c GPT-2 124M reaches ~49 % MFU on one A100-40GB PCIe and ~60 % on 8×A100 SXM (Karpathy 2024, https://github.com/karpathy/llm.c/discussions/481) — hand-written CUDA; PyTorch+compile is typically 0.6–0.8× of that. LLMQ (Xu et al. 2025, arXiv 2512.15306) reports ≈50 % MFU for 0.5–7B bf16 pretraining on RTX 4090s [UNVERIFIED: arXiv blocked, from abstract snippet]. nanoGPT gives 135 ms/iter for 124M on 8×A100 with compile (README), which is ≈40 % MFU [derived].
- Blackwell PRO has a higher FLOP-per-byte ratio (504/1.79 = 281 vs A100 153, H100 295), so non-matmul work (norms, RoPE, softmax, optimizer step, embedding gathers) costs relatively more for tiny models → 25M at 20 %, 100M 28 %, 250M 35 %, 500M 40 %, 1B 42 %. The 5080 (117 FLOP/byte) gets slightly higher MFU on small models but 4.5× lower peak. Blackwell consumer (sm_120) support in FlashAttention-2 lagged in 2025; PyTorch ≥ 2.7 SDPA (cuDNN/flash backends) is assumed to work on both cards [assumption; verify on day 1 of the pilot].
- Sensitivity: every MFU number here is ±30 %; the 1B base run (183 h) is the single largest training item, so a 30 % MFU miss moves the full budget by ±55 GPU-h.

### B.2 Teacher generation compute and throughput

Per generated token, a decoder of T parameters does ≈ 2·T FLOPs (+ 4·L·d·ctx attention, < 1 % at 500-token contexts). Decode is memory-bound at low batch and compute-bound at high batch. Roofline model used by the calculator (vLLM-style continuous batching):

- Weight bytes W = 2·T (BF16) or 1·(T − T_emb) + 2·T_emb (FP8 with BF16 embeddings/head); per-GPU W/tp.
- KV cache available: tp · (0.92 · VRAM − W/tp − 3 GB); KV bytes/token = 2 · L · n_kv · 128 · 2 (bf16). Resident tokens per sequence ≈ prompt + response/2 + 16 (block slack). Batch B = min(max_num_seqs = 512, KV_avail / (KV_tok · resident)).
- Per-step time t = max(t_mem, t_comp) + t_comm, with t_mem = (W/tp + B·resident·KV_tok/tp)/(BW · 0.75), t_comp = 2·T·B/tp / (P_peak · 0.40), t_comm (TP=2 over PCIe 5.0 P2P) = 2L · (B·d·2 B / 50 GB/s + 30 µs). Decode tok/s = 0.85 · B / t (engine overhead), capped at 30k tok/s per instance for tiny models (CPU-bound scheduler) [assumption].
- Prefill tok/s = P_peak · 0.60 · tp / (2T); prompts 150 tokens of which 50 are a shared system prefix served from prefix cache; responses 350 tokens [parametrised].
- Teacher tokens needed: delivered student tokens / (1 − 0.20 filter loss) / ρ, with **ρ = 1.12 student tokens per teacher token** [assumption: a 32k English BPE emits ≈10–15 % more tokens than Qwen's 151k vocabulary on English prose; measure on the first 10k samples and re-run the calculator].

Calibration against published numbers (all on the same card class, so no bandwidth scaling needed; [fact via snippets], configs incompletely reported):
- PRO 6000, vLLM: Llama-8B ≈ 8,990 tok/s; Qwen-14B ≈ 5,160 tok/s (VRLA Tech 2026 benchmark, https://vrlatech.com/gpu-benchmark-ai-llm-2026/). Model: 7.6B bf16 → 11,200; 14.8B bf16 → 5,800. Within 25 %.
- PRO 6000, Qwen3.5-35B: BF16 1,103 vs FP8 1,581 tok/s (1.43×) (Jarvislabs 2026, https://jarvislabs.ai/blog/nvfp4-rtxpro-6000) — concurrency unstated; FP8 gain ratio matches the model's 1.3–2.0×.
- H100 NVL, Qwen3-32B, vLLM 0.8.5 `benchmark_serving` on ShareGPT: 654 (BF16) / 879 (FP8) output tok/s at only ≈31 effective concurrent sequences (TPOT 47 ms) because ShareGPT prompts are long and the KV cache is small after 65 GB of weights (vLLM issue #17788, https://github.com/vllm-project/vllm/issues/17788). This is the important cautionary data point: **32B and 72B in BF16 are KV-cache-starved on 80–96 GB cards**; with our 500-token sequences the model predicts B ≈ 220 on the PRO 6000 (2,600 tok/s), but if prompts are longer than assumed throughput collapses toward the H100 NVL number. Treat every 32B/72B throughput below as ±2×.
- Single-stream (B = 1) numbers from the community (100–200 tok/s for 27–35B with speculative decoding, https://github.com/lastloop-ai/vllm-blackwell-guide) are irrelevant to batch generation and are not used.
- Qwen's own speed tables (qwen.readthedocs.io) are batch-1 latency tables and could not be fetched [UNVERIFIED].

Throughput and cost per **600 M delivered student tokens** per teacher (BF16 KV, max_num_seqs 512; "cpu-cap" = CPU-bound 30k ceiling, "mem" = bandwidth-bound, "compute" = tensor-bound):

| Teacher | Config | W (GB) | KV avail (GB) | B | Bound | Decode tok/s | Prefill tok/s | Wall h | GPU-h | $ cloud | kWh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1× PRO 6000 BF16 | 3.1 | 82 | 512 | cpu-cap | 30,000 | 98,000 | 6.7 | 6.7 | 13 | 4 |
| 1.5B | 1× 5080 BF16 | 3.1 | 8.6 | 512 | compute | 12,400 | 21,900 | 17.4 | 17.4 | 6 | 8 |
| Qwen2.5-3B | 1× PRO 6000 BF16 | 6.2 | 79 | 512 | compute | 27,700 | 48,900 | 7.8 | 7.8 | 16 | 5 |
| 3B | 1× 5080 BF16 | 6.2 | 5.5 | 441 | compute | 6,200 | 10,900 | 34.9 | 34.9 | 12 | 15 |
| Qwen2.5-7B | 1× PRO 6000 BF16 | 15.2 | 70 | 512 | compute | 11,200 | 19,800 | 19.2 | 19.2 | 38 | 13 |
| 7B | 1× PRO 6000 FP8 | 8.7 | 77 | 512 | compute | 22,500 | 39,700 | 9.6 | 9.6 | 19 | 6 |
| Qwen2.5-14B | 1× PRO 6000 BF16 | 29.5 | 56 | 512 | compute | 5,800 | 10,200 | 37.3 | 37.3 | 75 | 25 |
| 14B | 1× PRO 6000 FP8 | 16.3 | 69 | 512 | mem | 11,500 | 20,500 | 18.7 | 18.7 | 37 | 12 |
| Qwen2.5-32B | 1× PRO 6000 BF16 | 65.5 | 19.8 | 221 | compute | 2,600 | 4,600 | 82.6 | 82.6 | 165 | 55 |
| 32B | 2× PRO 6000 BF16 TP2 | 65.5 | 105 | 512 | compute | 4,300 | 9,200 | 48.7 | 97.4 | 195 | 64 |
| 32B | 1× PRO 6000 FP8 | 34.3 | 51 | 512 | compute | 5,200 | 9,200 | 41.3 | 41.3 | 83 | 27 |
| 32B | 1× H100 SXM BF16 | 65.5 | 5.1 | 57 | mem | 1,700 | 9,100 | 114 | 114 | 341 | 86 |
| Qwen2.5-72B | 1× PRO 6000 BF16 | 145 | — | — | **does not fit** | | | | | | |
| 72B | **2× PRO 6000 BF16 TP2** | 145 | 25 | 226 | compute | 1,960 | 4,200 | 108 | 216 | 431 | 142 |
| 72B | 1× PRO 6000 FP8 | 75.2 | 10.1 | 91 | mem | 1,200 | 4,200 | 166 | 166 | 332 | 110 |
| 72B | 2× PRO 6000 FP8 TP2 | 75.2 | 95 | 512 | compute | 3,500 | 8,300 | 59 | 119 | 238 | 78 |
| 72B | 1× H100 SXM (either) | | | | does not fit | | | | | | |

Take-aways: (i) 72B BF16 requires both PRO 6000s (TP=2) and is the single most expensive job (216 GPU-h); 72B FP8 on one card fits but is KV-starved (B ≈ 91) and is *slower per GPU-hour* than TP=2 BF16 — so FP8 only pays if used with TP=2. (ii) 32B BF16 on one card is KV-limited (B ≈ 220); FP8 doubles its throughput. (iii) 1.5B–3B are CPU/scheduler-bound, not GPU-bound; running them on the 5080s costs only 2.5–4.5× more wall-clock and frees the big cards. (iv) The 14B and 7B teachers cannot run on 16 GB cards in BF16.

### B.3 Evaluation cost

The suite (ARC-E 2,376; ARC-C 1,172; PIQA 1,838; HellaSwag 10,042; SciQ 1,000; OBQA 500; BoolQ 3,270; Winogrande 1,267; LAMBADA 5,153 — full validation/test sets) is ≈26.6k items and ≈78k scored continuations (item × choices) ≈ 6.3 M tokens per evaluation. (The brief's "~15k items" holds if HellaSwag is subsampled to 5k and BoolQ dropped.) Compute is not the bottleneck: 1B student, forward-only: 2·1.06e9·6.3e6 = 1.3e16 FLOP ≈ 7 min on a 5080 at 30 % MFU, 16 s for 25M. Harness overhead (tokenisation, request construction, length-sorted batching, Python) dominates: the flex.ai lm-eval blueprint quotes ≈10 min for HellaSwag alone on one GPU (model size unstated) [UNVERIFIED]. Assumed minutes per full 9-task eval + held-out NLL, on a 5080: **25M 12, 100M 14, 250M 18, 500M 24, 1B 32**; ×0.7 on a PRO 6000. Checkpoints evaluated: base runs every 10 % + final (11); synthetic-phase runs at 25/50/75/100 % (4). Evaluation is 45–55 % of all 5080 GPU-hours in both studies, so caching tokenised requests across checkpoints (the harness supports `--cache_requests`) is worth implementing.

---

## Task C — Pilot estimate

Design (per brief): students {25M, 100M}; base pretraining at 20 tokens/param (0.5 B, 2 B tokens) with **3 base seeds each** (cheap at these sizes and needed to estimate base-init variance — see disagreements); teachers {Qwen2.5-1.5B, 7B, 32B}, each pool = 600 M delivered student tokens (670 M teacher response tokens + 192 M non-cached prompt tokens); synthetic phase D_syn ∈ {50 M, 150 M, 500 M} × 3 seeds per teacher; C1 matched-real control at each D × 3 seeds; C0 = base checkpoint (eval only). BF16 everywhere. 159 jobs.

**GPU-hours by component (2× PRO 6000 + 2× 5080, list-scheduled placement):**

| Component | Jobs | PRO 6000 GPU-h | 5080 GPU-h | kWh | Electricity $ | Cloud-eq. $ |
|---|---|---|---|---|---|---|
| Generation (1.5B on 5080; 7B, 32B on PRO 6000) | 3 | 101.9 | 17.4 | 75 | 11 | 210 |
| Base pretraining (6 runs) | 6 | 6.9 | 17.6 | 14 | 2 | 20 |
| Eval of base checkpoints | 6 | 1.8 | 11.7 | 5 | 1 | 8 |
| Synthetic phase, D = 150 M (18 runs) | 18 | 0.5 | 11.0 | 6 | 1 | 5 |
| D-sweep (50 M, 500 M; 36 runs) | 36 | 9.9 | 10.6 | 12 | 2 | 23 |
| Controls C1 (18 runs) | 18 | 3.9 | 5.7 | 6 | 1 | 10 |
| Eval of synthetic-phase checkpoints (72 runs × 4) | 72 | 18.9 | 35.3 | 21 | 3 | 50 |
| **Total** | 159 | **144** | **110** | **138** | **$21** | **$326** |

**Wall-clock by hardware configuration** (list schedule; ×1.3 contingency in parentheses):

| Config | PRO 6000 GPU-h | 5080 GPU-h | Wall days |
|---|---|---|---|
| 1× PRO 6000 | 195 | 0 | 8.1 (10.6) |
| 1× PRO 6000 + 2× 5080 | 110 | 198 | 4.6 (6.0) |
| 2× PRO 6000 | 195 | 0 | 4.1 (5.3) |
| 2× PRO 6000 + 2× 5080 | 144 | 110 | 3.7 (4.8) |

The pilot's critical path is the 32B generation (83 h on one PRO 6000). With 14B instead of 32B (the brief allows either) the pilot drops to ≈2 days; with 32B in FP8 (41 h) to ≈2.5 days — but the pilot should use the same precision as the full study, so BF16 is recommended and the 32B pool should be started on day 0.

**Storage (pilot):** base corpus 2 B tokens uint16 = 4 GB (+9 GB raw text); synthetic pools 3 × 750 M pre-filter tokens = 4 GB uint16 + 11 GB JSONL; checkpoints 65 GB (base runs: last full checkpoint 14 B/param + 10 bf16 snapshots; synthetic runs: 4 bf16 snapshots each, full checkpoint deleted after completion); eval logs 20 GB → **≈115 GB**.

**Energy:** 138 kWh ≈ $21 at $0.15/kWh, PUE 1.2.

---

## Task D — Full-study estimate, affordability, redesign, schedule

Design: students {25M, 100M, 250M, 500M, 1B} at 20 tokens/param (0.5, 2, 5, 10, 20 B tokens; one base run each, plus 2 extra base seeds at 25M/100M); teachers {1.5B, 3B, 7B, 14B, 32B, 72B} each with a 600 M-token pool; D_syn = 150 M × 3 seeds in all 30 cells, 5 seeds in 6 cells (+12 runs); C1 × 3 seeds per size; C0 = eval of base; D-sweep {50 M, 500 M} × 2 seeds in {100M, 500M} × {3B, 14B, 72B} (+24 runs) with matching C1; ablations = 20 % of synthetic-phase train+eval GPU-h; BF16 generation with 72B at TP=2. 330 jobs.

**GPU-hours by component (2× PRO 6000 + 2× 5080):**

| Component | Jobs | PRO 6000 GPU-h | 5080 GPU-h | kWh | Electricity $ | Cloud-eq. $ |
|---|---|---|---|---|---|---|
| Generation (6 pools; 1.5B/3B on 5080s) | 6 | 355 | 52 | 257 | 39 | 728 |
| Base pretraining (9 runs) | 9 | 232 | 108 | 223 | 33 | 502 |
| Eval of base checkpoints | 9 | 4 | 22 | 9 | 1 | 16 |
| Synthetic phase main (102 runs incl. tiered seeds) | 102 | 42 | 52 | 56 | 8 | 102 |
| Controls C1 (23 runs) | 23 | 6 | 45 | 27 | 4 | 29 |
| D-sweep (24 runs) | 24 | 0 | 106 | 52 | 8 | 37 |
| Eval of synthetic-phase checkpoints | 149 | 30 | 148 | 64 | 10 | 111 |
| Ablations (20 % of syn train+eval) | 8 | 38 | 0 | 28 | 4 | 75 |
| **Total** | 330 | **706** | **534** | **715** | **$107** | **$1,600** |

Two items are 83 % of the big-card budget: generation (355 GPU-h, of which 72B = 216 and 32B = 83) and base pretraining (232 GPU-h, of which 1B = 183). The synthetic-phase training itself — the 30-cell experiment proper — is only ≈95 GPU-h across both card classes; seeds, controls and the D-sweep are cheap.

**Wall-clock by configuration:**

| Config | PRO 6000 GPU-h | 5080 GPU-h | Wall days (×1.3) | Note |
|---|---|---|---|---|
| 1× PRO 6000 | 863 | 0 | 36.0 (46.7) | 72B must be served FP8 on one card → precision inconsistent with other teachers |
| 1× PRO 6000 + 2× 5080 | 629 | 613 | 26.2 (34.1) | same caveat |
| 2× PRO 6000 | 912 | 0 | 19.0 (24.7) | |
| **2× PRO 6000 + 2× 5080** | **706** | **534** | **14.7 (19.1)** | baseline |
| 2× PRO 6000 + 2× 5080, all teachers FP8 | 550 | 487 | 11.6 (15.1) | option A |

**Storage (full):** base corpus 20 B tokens = 40 GB uint16 (+90 GB raw); pools 6 × 750 M = 9 GB uint16 + 22 GB JSONL; checkpoints 515 GB (dominated by 36+ runs at 1B × 4 bf16 snapshots × 2.1 GB and the 1B base run's 75 GB); logs 20 GB → **≈700 GB**; 1 TB NVMe suffices, 2 TB recommended. **Energy:** 715 kWh ≈ $107.

### D.1 Is the 30-cell study affordable in 8 weeks?

**Yes, with margin.** 8 weeks = 1,344 h per card = 2,688 PRO 6000 GPU-h available vs 706 needed (26 % utilisation), and 2,688 5080 GPU-h vs 534. The list-scheduled makespan is 14.7 days, 19 days with a 1.3× contingency for restarts/debugging, i.e. ≈2.9× slack. The caveat is that the slack is *parallel*, not *serial*: the critical path is irreducible at ≈12 days (72B generation needs both big cards for 108 h, then the 1B base run occupies one card for 183 h, then the 1B synthetic-phase runs, evaluation and ablations), so a 2× throughput miss on 72B generation or a 30 % MFU miss on the 1B run pushes the makespan to ≈17–20 days, still inside 8 weeks. Failure modes that would break the budget: (a) 72B BF16 TP=2 throughput below ≈800 tok/s (KV starvation with longer prompts; PCIe all-reduce slower than modelled) → +150 GPU-h and the 72B pool alone takes 10 days of both cards; (b) needing a second base run per size for seeds (+232 GPU-h per extra seed set); (c) D_syn = 500 M as the main budget (+97 GPU-h, +2 days).

### D.2 Redesign options, quantified (vs. baseline 706 PRO 6000 GPU-h, 14.7 days)

| Option | PRO 6000 GPU-h | Saved | % | Wall days | Saved days | Scientific cost |
|---|---|---|---|---|---|---|
| A. Serve all six teachers in FP8 (W8A8, bf16 KV) | 550 | 156 | 22 | 11.6 | 3.1 | Same recipe for all teachers keeps P3's consistency; FP8 affects small teachers' sample quality more than large ones (a mild confound); must be decided before the pilot |
| A2. FP8 weights + FP8 KV cache | 518 | 189 | 27 | 10.8 | 3.9 | as A, plus KV quantisation noise |
| B. 1B base at 10 tokens/param (others 20) | 623 | 84 | 12 | 13.0 | 1.7 | Breaks equal-tokens/param across S; 1B cell no longer comparable — **not recommended** |
| B2. All bases at 10 tokens/param | 588 | 118 | 17 | 12.3 | 2.5 | Keeps comparability; under-trained bases increase floor effects and inflate synthetic gains |
| C. Drop 72B (5 teachers) | 483 | 223 | 32 | 10.1 | 4.6 | Loses the top of the range, exactly where H2/H3/H5 are decided — last resort |
| D. Drop 3B and 14B (fractional design, 4 teachers) | 630 | 77 | 11 | 13.3 | 1.5 | Saves little (small/medium teachers are cheap); halves the resolution of the T-axis — poor trade |
| E. Halve pools to 300 M delivered/teacher | 542 | 164 | 23 | 11.4 | 3.3 | Caps the D-sweep at ≈250 M and forces seed subsamples to overlap ≥50 % |
| F. 1 seed everywhere | 647 | 60 | 8 | 13.5 | 1.2 | Destroys inference — shows seeds are cheap, keep them |
| G. (cost) D_syn = 500 M main | 803 | −97 | −14 | 16.7 | −2.0 | Affordable if the pilot says 150 M is too small |
| H. Drop D-sweep | 667 | 40 | 6 | 13.9 | 0.8 | Loses the D-term fit for negligible savings |
| I. A + B combined | 473 | 234 | 33 | 9.9 | 4.9 | |

Recommendation: **no redesign of the grid**. Keep 30 cells, 3 seeds (5 tiered), the D-sweep, and 20 tokens/param for all sizes. Bank option A (FP8 serving) as the pre-registered fallback if measured 72B/32B throughput in the pilot is below 50 % of the model; option E as the second fallback. Reject B, C, D, F, H (they save little or damage the design).

### D.3 Scheduling plan (2× PRO 6000 + 2× 5080)

Placement rules used by the simulator and recommended in practice:
- **PRO 6000 (96 GB):** all generation for 7B–72B (72B as TP=2 across both cards; 32B/14B/7B on one card), base pretraining for 250M–1B, synthetic-phase runs for 500M–1B, ablations, evaluation only when idle.
- **RTX 5080 (16 GB):** generation for 1.5B and 3B (BF16 fits; 6–12k tok/s), base and synthetic-phase training for 25M–250M, all evaluations (≤1B fits comfortably at bf16), C1 controls. 14B+ teachers and 1B training never go here.
- Order: (1) day 0: 72B pool on both PRO 6000s (108 h) while the 5080s run the 3B and 1.5B pools and the 250M/100M/25M base runs; (2) day 4.5: 1B base run on card A (183 h, to day 12); card B runs the 32B (83 h), 500M base (50 h), 14B (37 h) and 7B (19 h) pools/runs back-to-back; (3) as each (base, pool) pair completes, its cells' synthetic-phase runs and C1 controls start on the 5080s (25M–250M) or card B (500M); (4) day 12–14: 1B cells on both big cards (18+ runs × 1.4 h), then ablations; evaluations trail each run by ≤1 h on the 5080s. Simulated makespan 353 h = 14.7 days; big-card utilisation 706/(2 × 353) = 100 % on the critical path, 5080s 76 %.
- Practicalities: pin CUDA_VISIBLE_DEVICES per queue; use a job queue (e.g. a simple SLURM-less runner or `snakemake` with GPU resources) that encodes the two dependencies (base checkpoint, pool); checkpoint the 72B generation every 50k prompts so a crash costs < 2 h; keep the vLLM server for each teacher up until its pool is complete, never co-locate a training job with a generation job on the same card (KV-cache pressure kills throughput).

---

## Task E — Generation cost metric and Synthetic Transfer Efficiency (STE)

### E.1 Precise definition of generation cost C_gen(T)

Let a teacher T's pool deliver D_pool(T) student tokens after filtering. Measured quantities: τ_gen(T) = GPU-seconds summed over all cards occupied by the vLLM server(s), from first request to last response (model download/load and warm-up excluded and reported separately as a fixed cost); E_gen(T) = ∫ P(t) dt summed over cards from `nvidia-smi --query-gpu=power.draw` at 1 Hz, × PUE.

Unit cost per delivered student token, four units:
- c_τ(T) = τ_gen(T) / D_pool(T) [GPU-s / token] — hardware-specific (state: RTX PRO 6000 WS, vLLM version, BF16, TP).
- c_$(T) = c_τ(T) · p, p = $2.00/GPU-h = 5.56e-4 $/GPU-s (reference cloud-equivalent; sensitivity at $0.95 owned) [$/token].
- c_E(T) = E_gen(T) / D_pool(T) [kWh/token], and c_E·$0.15/kWh if a $ energy view is wanted.
- c_F(T) = 2·T·(N_prompt + N_response tokens processed) / D_pool(T) [FLOP/token] — hardware- and software-independent, reproducible by other labs; attention FLOPs (<1 %) ignored.

Cell cost: **C_gen(S,T) = c(T) · D_syn**, i.e. the cost of the tokens actually consumed, assuming linear cost in tokens (true for batched serving after warm-up). At fixed D_syn the cost does not depend on S; it does when D_syn varies (D-sweep).

Model-predicted values per 1 M delivered student tokens (BF16, PRO 6000, best of TP1/TP2):

| Teacher | TP | GPU-s / Mtok | $ / Mtok | kWh / Mtok | PFLOP / Mtok | time rel. 1.5B | FLOP rel. 1.5B |
|---|---|---|---|---|---|---|---|
| 1.5B | 1 | 40 | 0.02 | 0.007 | 4.9 | 1.0 | 1.0 |
| 3B | 1 | 47 | 0.03 | 0.009 | 9.9 | 1.2 | 2.0 |
| 7B | 1 | 115 | 0.06 | 0.021 | 24.3 | 2.9 | 4.9 |
| 14B | 1 | 224 | 0.12 | 0.041 | 47.1 | 5.5 | 9.6 |
| 32B | 1 | 496 | 0.28 | 0.091 | 104.5 | 12.3 | 21.3 |
| 72B | 2 | 1,294 | 0.72 | 0.237 | 231.9 | 32.0 | 47.2 |

At D_syn = 150 M the cell costs are $3 (1.5B) … $108 (72B). The time-cost is sub-linear in T below 7B (scheduler-bound) and super-linear above 32B (KV starvation, TP communication), while FLOP-cost is exactly linear — this divergence is why "economically optimal" and "FLOP-optimal" teachers can differ, and why both cost units must be reported.

### E.2 Gain definitions

M(S, cond) = the primary normalised metric on an interval scale (benchmark workstream): mean over benchmarks b of 100·(acc_b − chance_b)/(1 − chance_b) ("points"), or a continuous NLL-based score. Two gains:
- G₀(S,T) = M(S,T) − M(S,C0) (over base only) — measures "any extra training helps".
- **G₁(S,T) = M(S,T) − M(S,C1)** (over matched-real control, same D_syn of free real tokens) — measures the value of the teacher's tokens *beyond* free data. This is the economically relevant counterfactual: real tokens cost ≈0 to obtain, so generation dollars must be justified against them.

### E.3 Candidate STE definitions and their properties

| # | Definition | Units | Interpretability | Arbitrary constants | Affine rescaling of M (aM+b) | Gain ≤ 0 | Comparable across S |
|---|---|---|---|---|---|---|---|
| 1 | G₀(S,T)/C_$(T) | points per $ | High ("return per dollar") but conflates extra-tokens effect with teacher quality (C0 is not token-matched) | p ($/GPU-h) as a pure scale → ranking within S invariant; depends on D_syn (G saturates, C linear) → only meaningful at fixed D_syn | b cancels; a scales → rank-invariant | Sign meaningful ("paid to get worse"), magnitude not | No: gain headroom differs by S |
| 2 | G₁(S,T)/(D_syn/10⁶) | points per M synthetic tokens | Data-efficiency; at fixed D_syn it is just G₁ in other units — contains **no teacher cost**, cannot answer H5 | none beyond D_syn | rank-invariant | negative = synthetic worse than real (meaningful) | No |
| 3 | [G₁(T)/C(T)] / [G₁(T_ref)/C(T_ref)], T_ref = raw-best T* | dimensionless | "Efficiency relative to the best teacher"; p and a cancel exactly | choice of T_ref; undefined when G₁(T_ref) ≤ 0 | fully invariant (a, b cancel) | numerator sign carries; reference ≤ 0 → unusable | Within-S normalised, so cross-S compares *shapes* only |
| 4 | AUC of G₁ vs log C over the teacher range | points·log($) | A per-student summary of "returns to spending" — does **not** select a teacher; sensitive to endpoints and to interpolation between 6 points | integration range; log base | b cancels (using G), a scales | negative regions subtract | Yes if the same cost range is used, but it answers H2 not H5 |
| 5 | mᵢ = [G₁(Tᵢ₊₁) − G₁(Tᵢ)] / [C(Tᵢ₊₁) − C(Tᵢ)] | points per marginal $ | "Return on upgrading"; exactly the object that decides whether a bigger teacher is worth it | grid spacing; blows up where ΔC is tiny (1.5B→3B differ by ≈$0.6 per 150 M tokens) | rank-invariant | negative = upgrade hurts (meaningful) | No |

Structural remark: any ratio metric (1, 3) of a concave gain curve over cost is maximised at the cheapest teacher, so "argmax STE" is a poor selector; the ratio should be *reported* (it is the most communicable summary) but selection must use a net-benefit or marginal criterion.

### E.4 Selection

**Primary metric — STE (ratio, matched-real numerator):**

STE(S,T) = G₁(S,T) / C_$(S,T) = [M(S,T) − M(S,C1)] / (c_$(T) · D_syn),  units: normalised points per $ at p = $2.00/GPU-h on RTX PRO 6000 (state p; report STE_τ in points per kGPU-s and STE_F in points per EFLOP as hardware-free robustness variants; they differ only by a T-dependent factor for STE_F, which can re-rank).

**Secondary metric — marginal efficiency and the price-of-a-point curve:**

mᵢ(S) = [G₁(S,Tᵢ₊₁) − G₁(S,Tᵢ)] / [C_$(Tᵢ₊₁) − C_$(Tᵢ)],  points per $, for adjacent teachers ordered by cost.

Economically optimal teacher for a price-of-a-point λ (points per $): **T$(S; λ) = argmax_T [G₁(S,T) − λ·C_$(S,T)]**. λ = 0 gives the raw-optimal T*(S) = argmax_T G₁(S,T). As λ increases T$ steps down through cheaper teachers; the switch points are exactly the marginal efficiencies mᵢ (when G₁ is concave in C). We report the full step function T$(λ) over λ ∈ [0, 10] points/$ (log grid) rather than one λ, plus two constant-free summaries: (i) λ₁(S) = the marginal efficiency of the last upgrade into T* — the smallest price-of-a-point at which the raw-optimal teacher stops being optimal; (ii) the ε-rule T$_ε = the cheapest T with G₁(S,T) ≥ (1−ε)·G₁(S,T*), ε = 0.10, which needs no λ. **H5 is supported for student S if the 95 % CI of λ₁(S) lies below λ_ref(S) := G₁(S,T₁)/C_$(S,T₁), the average efficiency of the cheapest teacher** — i.e. upgrading to the raw-best teacher yields less per marginal dollar than the cheapest teacher yields per average dollar. T$ ≠ T* is then not an artefact of a chosen λ.

**CI procedure (bootstrap propagating gain and cost uncertainty).** For each (S,T): seeds k = 1..K give M_k(S,T) and M_k(S,C1) (K = 3 or 5; C1 has its own seeds, unpaired).
1. Gain resampling. Because K = 3 gives only 10 distinct resamples, use a *pooled-variance parametric bootstrap* as primary: estimate the seed SD σ_S pooled across all conditions of student S (assume homoscedastic across T within S; check with Levene), draw M* ~ N(mean_k M_k, σ_S²/K) for the cell and the control, G₁* = difference. Nested inside, resample benchmark items (per-item harness outputs) to include item-sampling variance. Non-parametric seed resampling is run as a check.
2. Cost resampling. Generation is logged in chunks (e.g. every 10k prompts: GPU-s, delivered tokens, energy). Resample chunks with replacement, c* = Σ GPU-s / Σ tokens. Price p is a common scale factor and is handled by sensitivity (p ∈ {0.95, 2.00, 3.00}), not bootstrapped.
3. Combine: STE* = G₁*/(c*·D_syn); mᵢ* from adjacent cells; T$*(λ) = argmax over the resampled curve. B = 5,000 replicates; percentile 95 % CIs for STE and λ₁; for T$ report the *selection probability* of each teacher at each λ (fraction of replicates choosing it) and call T$ "stable" if ≥ 0.8.
4. Multiplicity: 30 cells × 6 teachers; the CIs are descriptive; the pre-registered tests are H1 (monotone trend, Jonckheere–Terpstra over T within S), H2 (plateau: last marginal mᵢ CI includes 0 in G units), H3 (interior maximum with selection probability ≥ 0.8), H5 (as above). Family-wise control via Holm over the 5 students for each hypothesis.

Reference implementation: `compute_budget.py` → `ste_primary`, `t_dollar`, `t_epsilon`, `bootstrap_ste` (no experimental numbers are embedded).

---

## Calculator output (verbatim; `python3 compute_budget.py --redesign`, default assumptions; schedule listing truncated)

```
=== Hardware constants (Task A) ===
GPU                               VRAM  BW GB/s BF16 dense FP8 dense  TDP W  price $ rent $/h  note
RTX PRO 6000 Blackwell WS           96     1792      504.0      1008    600    15500     2.00  Workstation Edition, 600 W; price = NVIDIA store $16,000 / Newegg $13,998 / B&H $15,499 (Sept 2026)
RTX PRO 6000 Blackwell Max-Q        96     1792      439.0       878    300    14700     2.00  3511 AI TOPS -> 439 dense BF16 [derived]; 300 W; price [UNVERIFIED]
RTX PRO 6000 Blackwell Server       96     1792      469.0       938    600    13100     0.93  3753 AI TOPS -> 469 dense BF16 [derived]; passive; Vast.ai $0.93/h
RTX 5080                            16      960      112.6       225    360     1600     0.35  GeForce halves FP32-accumulate tensor rate: 112.6 dense BF16 (225 with FP16 acc.); $999 MSRP, ~$1,580-1,800 street
H100 SXM 80GB                       80     3350      989.0      1979    700    28000     2.99  reference only; purchase price [UNVERIFIED]; rental $2.69-3.99/h
electricity $0.15/kWh, PUE 1.2, host 100 W/GPU, power util {'train': 0.85, 'gen': 0.75, 'eval': 0.5}

=== Student shapes and training cost (Task B) ===
S         d   L   ffn  N_nonemb    N_flop  FLOP/tok  attn% | MFU 6000 h/1B tok | MFU 5080 h/1B tok
25M     512   8  1408     25.7M     42.5M     0.31G   16.5 |     0.20     0.84 |     0.25     3.01
100M    768  14  2048     99.1M    124.3M     0.88G   15.1 |     0.28     1.73 |     0.33     6.56
250M   1024  20  2752    253.0M    286.5M     1.97G   12.8 |     0.35     3.10 |     0.38    12.79
500M   1280  25  3456    495.6M    537.6M     3.62G   10.9 |     0.40     4.99 |     0.35    25.51
1B     1792  26  4800   1004.9M   1063.6M     6.95G    8.2 |     0.42     9.13 |     None     None
ctx=2048, vocab=32768, causal factor 0.5; FLOP/tok = 6(N_nonemb+V d) + 0.5*12 L d ctx

=== Teacher generation throughput and cost per 600M delivered student tokens (Task B) ===
prompt/resp = 150/350, prefix cache 50, student/teacher token ratio 1.12, filter loss 20%, max_num_seqs 512, kv bf16
teacher       gpu      tp prec   W GB  KV GB    B    bound dec tok/s pre tok/s  wall h   GPU-h $ cloud    kWh
1.5B          pro6000   1 bf16    3.1   82.2  512  cpu-cap     30000     98182     6.7     6.7      13      4
1.5B          pro6000   2 bf16    3.1  167.6  512  compute     59189    196364     3.4     6.8      14      5
1.5B          pro6000   1  fp8    1.8   83.5  512  cpu-cap     30000    196364     6.5     6.5      13      4
1.5B          pro6000   2  fp8    1.8  168.9  512  cpu-cap     60000    392727     3.2     6.5      13      4
1.5B          rtx5080   1 bf16    3.1    8.6  512  compute     12430     21935    17.4    17.4       6      8
1.5B          h100_sxm  1 bf16    3.1   67.5  512  cpu-cap     30000    192662     6.5     6.5      19      5
1.5B          h100_sxm  1  fp8    1.8   68.8  512  cpu-cap     30000    385519     6.3     6.3      19      5
3B            pro6000   1 bf16    6.2   79.1  512  compute     27728     48932     7.8     7.8      16      5
3B            pro6000   2 bf16    6.2  164.5  512  compute     33406     97864     6.1    12.2      24      8
3B            pro6000   1  fp8    3.4   81.9  512  cpu-cap     30000     97864     6.7     6.7      13      4
3B            pro6000   2  fp8    3.4  167.2  512  compute     47805    195728     4.2     8.3      17      5
3B            rtx5080   1 bf16    6.2    5.5  441  compute      6195     10932    34.9    34.9      12     15
3B            h100_sxm  1 bf16    6.2   64.4  512  cpu-cap     30000     96019     6.8     6.8      20      5
3B            h100_sxm  1  fp8    3.4   67.2  512  cpu-cap     30000    192136     6.5     6.5      19      5
7B            pro6000   1 bf16   15.2   70.1  512  compute     11244     19843    19.2    19.2      38     13
7B            pro6000   2 bf16   15.2  155.4  512  compute     17309     39685    12.1    24.2      48     16
7B            pro6000   1  fp8    8.7   76.6  512  compute     22488     39685     9.6     9.6      19      6
7B            pro6000   2  fp8    8.7  161.9  512  compute     28138     79370     7.3    14.6      29     10
7B            rtx5080   1 bf16   does not fit
7B            h100_sxm  1 bf16   15.2   55.4  512  compute     22064     38937     9.8     9.8      29      7
7B            h100_sxm  1  fp8    8.7   61.9  512  cpu-cap     30000     77913     6.9     6.9      21      5
14B           pro6000   1 bf16   29.5   55.8  512  compute      5801     10237    37.3    37.3      75     25
14B           pro6000   2 bf16   29.5  141.1  512  compute      8625     20474    24.2    48.3      97     32
14B           pro6000   1  fp8   16.3   69.0  512      mem     11548     20474    18.7    18.7      37     12
14B           pro6000   2  fp8   16.3  154.3  512      mem     13690     40948    14.9    29.8      60     20
14B           rtx5080   1 bf16   does not fit
14B           h100_sxm  1 bf16   29.5   41.1  512  compute     11383     20088    19.0    19.0      57     14
14B           h100_sxm  1  fp8   16.3   54.3  512      mem     21587     40196     9.9     9.9      30      7
32B           pro6000   1 bf16   65.5   19.8  221  compute      2615      4615    82.6    82.6     165     55
32B           pro6000   2 bf16   65.5  105.1  512  compute      4332      9231    48.7    97.4     195     64
32B           pro6000   1  fp8   34.3   51.0  512  compute      5231      9231    41.3    41.3      83     27
32B           pro6000   2  fp8   34.3  136.3  512  compute      7394     18462    28.0    56.1     112     37
32B           rtx5080   1 bf16   does not fit
32B           h100_sxm  1 bf16   65.5    5.1   57      mem      1719      9057   114.1   114.1     341     86
32B           h100_sxm  1  fp8   34.3   36.3  406  compute     10270     18123    21.0    21.0      63     16
72B           pro6000   1 bf16   does not fit
72B           pro6000   2 bf16  145.4   25.2  226  compute      1957      4159   107.8   215.7     431    142
72B           pro6000   1  fp8   75.2   10.1   91      mem      1213      4159   166.2   166.2     332    110
72B           pro6000   2  fp8   75.2   95.4  512  compute      3510      8318    59.4   118.8     238     78
72B           rtx5080   1 bf16   does not fit
72B           h100_sxm  1 bf16   does not fit
72B           h100_sxm  1  fp8   does not fit

=== Generation cost C_gen(T) per 1M delivered student tokens (bf16, pro6000, best-of TP1/TP2) (Task E) ===
teacher  tp GPU-s/Mtok  $/Mtok kWh/Mtok PFLOP/Mtok rel. time rel. FLOP
1.5B      1         40    0.02    0.007        4.9       1.0       1.0
3B        1         47    0.03    0.009        9.9       1.2       2.0
7B        1        115    0.06    0.021       24.3       2.9       4.9
14B       1        224    0.12    0.041       47.1       5.5       9.6
32B       1        496    0.28    0.091      104.5      12.3      21.3
72B       2       1294    0.72    0.237      231.9      32.0      47.2

--- PILOT: 2x PRO 6000 + 2x RTX 5080 | gen bf16 | D_syn 150M | seeds 3 ---
component               jobs PRO6000 GPU-h 5080 GPU-h     kWh  elec $ cloud-eq $
generation                 3         101.9       17.4      75      11        210
base pretraining           6           6.9       17.6      14       2         20
eval base                  6           1.8       11.7       5       1          8
syn main                  18           0.5       11.0       6       1          5
eval syn                  72          18.9       35.3      21       3         50
D-sweep                   36           9.9       10.6      12       2         23
controls                  18           3.9        5.7       6       1         10
TOTAL                    159         143.7      109.5     138      21        326
wall-clock (list schedule, no contingency): 88 h = 3.7 days; with x1.3 contingency: 4.8 days
longest jobs: gen_32B@pro6000x1 83h [0-83]; gen_7B@pro6000x1 19h [0-19]; gen_1.5B@rtx5080x1 17h [0-17]; base_100M_s0@rtx5080x1 13h [0-13]; base_100M_s1@pro6000x1 3h [19-23]; base_100M_s2@pro6000x1 3h [23-26]

=== PILOT: wall-clock vs GPU counts (days, list schedule; x1.3 contingency in parentheses) ===
config                   PRO6000 GPU-h 5080 GPU-h  wall days w/ conting.
1x PRO6000 + 0x 5080                195          0        8.1 (10.6)
1x PRO6000 + 2x 5080                110        198        4.6 (6.0)
2x PRO6000 + 0x 5080                195          0        4.1 (5.3)
2x PRO6000 + 2x 5080                144        109        3.7 (4.8)

=== PILOT: storage (GB) ===
base corpus tokens (uint16, nested prefix)                4
base corpus raw text (~4.5 B/token, optional)             9
synthetic pools tokens (uint16)                           4
synthetic pools raw JSONL (~5 B/token)                   11
checkpoints retained                                     65
eval logs / samples (est.)                               20
TOTAL                                                   114

--- FULL: 2x PRO 6000 + 2x RTX 5080 | gen bf16 | D_syn 150M | seeds 3 ---
component               jobs PRO6000 GPU-h 5080 GPU-h     kWh  elec $ cloud-eq $
generation                 6         354.8       52.3     257      39        728
base pretraining           9         232.4      107.9     223      33        502
eval base                  9           4.1       22.0       9       1         16
syn main                 102          41.7       52.2      56       8        102
eval syn                 149          29.5      148.0      64      10        111
controls                  23           6.4       45.3      27       4         29
D-sweep                   24           0.0      105.8      52       8         37
ablations (20%)            8          37.6        0.0      28       4         75
TOTAL                    330         706.4      533.5     715     107       1600
wall-clock (list schedule, no contingency): 353 h = 14.7 days; with x1.3 contingency: 19.1 days
longest jobs: base_1B_s0@pro6000x1 183h [108-290]; gen_72B@pro6000x2 108h [0-108]; gen_32B@pro6000x1 83h [108-190]; base_250M_s0@rtx5080x1 64h [0-64]; base_500M_s0@pro6000x1 50h [190-240]; gen_14B@pro6000x1 37h [240-278]

=== FULL: wall-clock vs GPU counts (days, list schedule; x1.3 contingency in parentheses) ===
config                   PRO6000 GPU-h 5080 GPU-h  wall days w/ conting.
1x PRO6000 + 0x 5080                863          0       36.0 (46.7) *72B served FP8 on 1 card (precision inconsistent)
1x PRO6000 + 2x 5080                629        613       26.2 (34.1) *72B served FP8 on 1 card (precision inconsistent)
2x PRO6000 + 0x 5080                912          0       19.0 (24.7)
2x PRO6000 + 2x 5080                706        534       14.7 (19.1)

=== FULL: storage (GB) ===
base corpus tokens (uint16, nested prefix)               40
base corpus raw text (~4.5 B/token, optional)            90
synthetic pools tokens (uint16)                           9
synthetic pools raw JSONL (~5 B/token)                   22
checkpoints retained                                    515
eval logs / samples (est.)                               20
TOTAL                                                   697

=== Schedule (first jobs by start time) ===
start h   end h class      n  job
      0      35 rtx5080    1  gen_3B
      0     108 pro6000    2  gen_72B
      0      64 rtx5080    1  base_250M_s0
     35      52 rtx5080    1  gen_1.5B
     52      65 rtx5080    1  base_100M_s0
     ...
    108     190 pro6000    1  gen_32B
    108     290 pro6000    1  base_1B_s0
    190     240 pro6000    1  base_500M_s0
    240     278 pro6000    1  gen_14B
    ...
utilisation pro6000: 706 GPU-h busy
utilisation rtx5080: 534 GPU-h busy

=== Redesign options for 'FULL' on 2x PRO 6000 + 2x 5080 (savings vs. baseline) ===
option                                                     PRO6000 GPU-h   saved     %  wall d  saved d
baseline                                                             706                  14.7
A. serve ALL teachers in FP8 (W8A8, bf16 KV)                         550     156    22    11.6      3.1
A2. FP8 weights + FP8 KV cache                                       518     189    27    10.8      3.9
B. 1B base at 10 tokens/param (others 20)                            623      84    12    13.0      1.7
B2. ALL bases at 10 tokens/param                                     588     118    17    12.3      2.5
C. drop 72B teacher (5 teachers)                                     483     223    32    10.1      4.6
D. drop 3B and 14B (4 teachers: 1.5,7,32,72)                         630      77    11    13.3      1.5
E. halve generated pool (300M delivered/teacher)                     542     164    23    11.4      3.3
F. 1 seed everywhere (NOT recommended; shows seeds are cheap)           647      60     8    13.5      1.2
G. (cost, not saving) D_syn = 500M instead of 150M                   803     -97   -14    16.7     -2.0
H. drop D-sweep                                                      667      40     6    13.9      0.8
I. A + B combined                                                    473     234    33     9.9      4.9
```

`python3 compute_budget.py --study full --gen-precision fp8` (option A) → PRO 6000 550 GPU-h, 5080 487 GPU-h, 590 kWh ($89), cloud-eq. $1,271, 11.6 days (15.1 with contingency).

---

## Open issues and disagreements with the brief

1. **"Base-only ×3 seeds" contradicts P1's single shared base run.** A shared base gives synthetic-phase seeds that vary only data order/subsample, not initialisation; seed variance will be under-estimated and CIs too narrow. Proposal implemented in the calculator: 3 base seeds at 25M and 100M only (12 PRO 6000 GPU-h or ~30 5080 GPU-h) to *measure* init variance and inflate the CIs at larger sizes by the measured ratio; at 250M–1B keep one base run (an extra 1B base seed costs 183 GPU-h each). C0 should be the evaluated base checkpoint (plus its LR-decay branch if WSD is used), not a seeded run.
2. **The 72B teacher is the budget.** It is 216 of 355 generation GPU-h and forces TP=2 (both big cards for 4.5 days). The brief's "identical precision for all teachers" is honoured with BF16, but it should be decided *before the pilot* whether the study is BF16 or a single FP8 recipe (option A, −156 GPU-h), because the pilot's 32B pool must be generated in the study's precision. I recommend BF16 unless the pilot measures < 50 % of the modelled 72B throughput.
3. **The token-ratio and filter-loss assumptions (ρ = 1.12, 20 %) are unmeasured** and scale all generation costs linearly. Measure both on the first 10k samples of the pilot and re-run the calculator; if the student tokenizer is 16k instead of 32k, ρ rises to ≈1.2 and generation gets 7 % cheaper per student token.
4. **Pool size vs D-sweep.** A 600 M pool with D_syn = 500 M in the sweep means the three seeds' subsamples overlap ≥ 80 %, so "seed" variance at D = 500 M is mostly data-order variance. Either enlarge pools for the three sweep teachers (3B/14B/72B: +≈50 % of their generation cost, ≈+130 GPU-h for 72B alone) or restrict the sweep's top point to 300 M.
5. **The evaluation suite is ≈26.6k items, not ~15k,** with HellaSwag alone 10k; evaluation is ≈50 % of 5080 GPU-hours. Recommend fixing a 5k HellaSwag subsample (same items for every checkpoint) and `--cache_requests`.
6. **Context length.** The calculator assumes s = 2,048; the attention term is 8–17 % of student FLOPs. If the synthetic data are 500-token examples, packing into 2,048 windows crosses document boundaries; with s = 1,024 training is ≈5–8 % cheaper and the packing artefact smaller. Architecture/data workstreams should decide.
7. **Hardware numbers not verifiable from primary sources in this session** (proxy-blocked): NVIDIA datasheet values were reconciled from spec aggregators and the SM×clock derivation; the Max-Q/Server prices and the RTX 5080 rental prices are tracker snapshots. Two measured PRO 6000 vLLM data points calibrate the decode model to ±25 % for 7–14B; the 32B/72B BF16 predictions are ±2× (KV-cache starvation risk) and are the first thing the pilot must measure.
8. **Title/framing (P8):** the cost analysis supports calling the economic result a "cost–benefit frontier of teacher size" rather than a scaling law; six teachers give five marginal slopes, enough to locate a knee, not to fit a power law in cost.
9. **8-week feasibility is not the binding constraint; the serial critical path is.** With 2 + 2 cards the study takes ≈15–19 days; the *pilot* should therefore also validate the scheduling assumptions (co-location, TP=2 over PCIe, vLLM on sm_120) since a one-card configuration (36–47 days, FP8-only 72B) has no slack at all.
