# 07. Student Architectures (derivation)

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- Adopted unchanged: Llama-2 32k SentencePiece (V = 32,000), tied embeddings, GQA 3:1, head_dim 64, ctx 2048, RoPE θ = 10⁴, the five shapes in §0, bf16-mixed precision, init rule.
- The training workstream's proposal of a custom 32k BPE is rejected (doc 00 §0.5); the 16k alternative is recorded as a future ablation, not adopted.
- Sizes are named by total parameters (N_tot); N_ne is reported alongside and used as a sensitivity axis. The LR rule in doc 10 is keyed by N_ne using the values in §0.
- The throughput/MFU numbers in §8 are assumptions to be replaced by pilot measurements; the compute budget in docs/16 uses the calculator's own MFU assumptions.

---

# Student architecture family: derivation and specification

Workstream: student architecture. Deliverables: this document; `scratchpad/code/param_count.py` (exact parameter/FLOP formulas, constrained shape search, memory and throughput planner, YAML emitter); `scratchpad/code/test_param_count.py` (pytest, 5 passed / 1 skipped because `torch` is not installed in the sandbox — the torch cross-check skips gracefully); `scratchpad/code/fertility.py` (tokenizer measurements, results in `scratchpad/data/fertility_results.json`). Every table below is copied verbatim from `python3 param_count.py` and `python3 fertility.py` output (`code/param_count_output.md`, `code/fertility_output.md`).

Labels used: **[fact]** = arithmetic or verifiable published configuration; **[lit]** = literature-supported claim; **[choice]** = methodological choice made here; **[assumption]** = number that must be measured in the pilot; **[UNVERIFIED]** = could not be checked from the sandbox (most academic and Hugging Face hosts are blocked by the egress policy; only PyPI and raw.githubusercontent.com were reachable).

## 0. Decisions at a glance

| item | decision | kind |
|---|---|---|
| block | pre-norm RMSNorm (eps 1e-5), RoPE (theta = 10 000), SwiGLU with d_ff = 8/3 d_model exactly, no bias terms, no dropout | [choice] (Llama-2 standard) |
| attention | grouped-query attention with n_kv = n_heads / 3 at **every** size; head_dim = 64; FlashAttention (causal); KV cache | [choice], Section 5 |
| vocabulary | Llama-2 SentencePiece BPE, 32 000 tokens, byte fallback, per-digit number splitting; ungated copies ship in `karpathy/llama2.c` and TinyLlama | [choice], Section 3 |
| embedding tying | input embedding and LM head tied at every size | [choice], Section 4 |
| shape rule | d_model = 192 k (so n_heads = 3 k and d_ff = 8 d/3 is an exact multiple of 64); aspect ratio d_model / n_layers in [32, 64], target 48 | [choice], Section 6 |
| context | 2048 tokens for both the base and the synthetic phase | [choice], Section 7 |
| precision, init | bf16 autocast with fp32 master weights, gradients and Adam moments; N(0, 0.02) with residual projections scaled by 1/sqrt(2 L) | [choice], Section 8 |
| checkpointing | none, except full checkpointing when the 500M model is trained on a 16 GB card | [choice], Section 8 |

**Final family (vocab 32 000, tied, GQA 3:1, head_dim 64, ctx 2048).** Totals are exact; all within +-1.2 % of target.

| name | n_layers | d_model | n_heads | n_kv_heads | d_ff | total params | non-embedding params | embedding share | d/L |
|---|---|---|---|---|---|---|---|---|---|
| 25M | 8 | 384 | 6 | 2 | 1024 | 24 877 440 | 12 589 440 | 49.4 % | 48 |
| 100M | 12 | 768 | 12 | 4 | 2048 | 100 092 672 | 75 516 672 | 24.6 % | 64 |
| 250M | 22 | 960 | 15 | 5 | 2560 | 247 032 000 | 216 312 000 | 12.4 % | 44 |
| 500M | 24 | 1344 | 21 | 7 | 3584 | 505 495 872 | 462 487 872 | 8.5 % | 56 |
| 1B | 38 | 1536 | 24 | 8 | 4096 | 1 005 571 584 | 956 419 584 | 4.9 % | 40 |

The 100M model has exactly the GPT-2-small / Pythia-160M skeleton (12 x 768, 12 heads), the 250M model has the SmolLM-360M / MobileLLM-350M width (960, 15 heads, 5 KV heads), and the 1B model has Llama-3.2-1B's KV layout (24 query heads / 8 KV heads at head_dim 64, at d = 1536 instead of 2048) — the family sits inside the mainstream of small decoder configurations (Section 6, reference table).

## 1. Parameter and FLOP accounting [fact]

Per layer, with d = d_model, d_attn = n_heads x head_dim, d_kv = n_kv_heads x head_dim:

```
attn_layer = d * d_attn  [W_q]  +  2 * d * d_kv  [W_k, W_v]  +  d_attn * d  [W_o]
ffn_layer  = 3 * d * d_ff                                   (SwiGLU: W_gate, W_up, W_down; no biases)
norm_layer = 2 * d                                          (RMSNorm gains; attention norm + FFN norm)
final_norm = d
embeddings = V * d   (+ V * d for a separate LM head if untied)

total      = L * (attn_layer + ffn_layer + norm_layer) + final_norm + embeddings
non_emb    = total - embeddings                             (excludes the input embedding AND an untied head)
```

Worked example used by the unit test: L = 2, d = 64, 4 heads of 16, 2 KV heads, d_ff = 128, V = 100, tied: attn = 4096 + 4096 + 4096 = 12 288; ffn = 24 576; norms = 128; layer = 36 992; total = 2 x 36 992 + 64 + 6 400 = **80 448**; non-embedding = 74 048. The same code reproduces GPT-2 small to the parameter (124 439 808, with its learned positional table and biased LayerNorm/MLP) and Pythia-70M (70 426 624), and matches the SmolLM, MobileLLM, TinyLlama, OLMo-1B, Qwen2.5-0.5B and Llama-3.2-1B model-card totals to within rounding (Section 6).

FLOPs per token (T = context length):

```
C_fwd   = 2 * (non_emb + V * d)  +  4 * L * T * d_attn
          |_ every weight matmul, incl. the LM head (a real matmul even when tied)
                                     |_ Q K^T and P V: 2 * T * d_attn each per layer (full-T upper bound;
                                        a causal FlashAttention kernel skips masked blocks -> ~half)
C_train = 3 * C_fwd = 6 * (non_emb + V * d) + 12 * L * T * d_attn      (backward ~ 2x forward)
C_train (full activation checkpointing) = 4 * C_fwd                     (one extra forward)
```

Two conventions coexist in the literature: Chinchilla's `6 N D` uses N = total parameters and drops the attention term (Hoffmann et al. 2022, https://arxiv.org/abs/2203.15556); Kaplan et al. (2020, https://arxiv.org/abs/2001.08361) exclude embeddings. Both are printed by the code. At our sizes the causal attention term is 20 % (25M) to 11 % (1B) of C_fwd at T = 2048, so it is not negligible for the small end and is kept explicit.

## 2. Shape search

Rules enforced by `search()` [choice]:

1. head_dim = 64; n_heads = d_model / 64.
2. GQA ratio 3: n_kv_heads = n_heads / 3, and shapes where 3 does not divide n_heads are rejected. Consequently d_model is a multiple of 192.
3. d_ff = LLaMA rule `multiple_of * ceil(int(8 d / 3) / multiple_of)` with multiple_of = 64 (Touvron et al. 2023, https://arxiv.org/abs/2302.13971 use 256 at 7B; 64 keeps the small models closer to 8/3). For d = 192 k the rule gives d_ff = 8 d / 3 exactly.
4. Total parameters within +-10 % of target (the brief's tolerance).
5. Aspect ratio d_model / n_layers in [32, 64] with target 48 (Section 6).
6. Objective: minimise `score = |rel_err| + 0.1 * |ln(aspect / 48)|` among in-band shapes (fallback: all shapes). The second term is a weak tie-breaker: a 10 % deviation in aspect ratio costs the same as a 1 % deviation in parameter count.

Candidate tables (top 6 by score; `*` = chosen):

### target 25M: 17 shapes within +-10%; top 6 by score (score = |rel_err| + 0.1*|ln(aspect/48)|; * = chosen)
| L | d_model | heads | kv | d_ff | total | non-emb | emb% | rel.err | d/L | in band | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 384 | 6 | 2 | 1024 | 24.88M | 12.59M | 49% | -0.5% | 48 | y | 0.0049 * |
| 9 | 384 | 6 | 2 | 1024 | 26.45M | 14.16M | 46% | +5.8% | 43 | y | 0.0698 |
| 7 | 384 | 6 | 2 | 1024 | 23.30M | 11.02M | 53% | -6.8% | 55 | y | 0.0812 |
| 2 | 576 | 9 | 3 | 1536 | 25.51M | 7.08M | 72% | +2.1% | 288 | n | 0.1997 |
| 48 | 192 | 3 | 1 | 512 | 25.04M | 18.89M | 25% | +0.1% | 4 | n | 0.2500 |
| 47 | 192 | 3 | 1 | 512 | 24.64M | 18.50M | 25% | -1.4% | 4 | n | 0.2606 |

### target 100M: 27 shapes within +-10%; top 6 by score (score = |rel_err| + 0.1*|ln(aspect/48)|; * = chosen)
| L | d_model | heads | kv | d_ff | total | non-emb | emb% | rel.err | d/L | in band | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 768 | 12 | 4 | 2048 | 100.09M | 75.52M | 25% | +0.1% | 64 | y | 0.0297 * |
| 23 | 576 | 9 | 3 | 1536 | 99.85M | 81.42M | 18% | -0.1% | 25 | n | 0.0665 |
| 13 | 768 | 12 | 4 | 2048 | 106.39M | 81.81M | 23% | +6.4% | 59 | y | 0.0846 |
| 22 | 576 | 9 | 3 | 1536 | 96.31M | 77.88M | 19% | -3.7% | 26 | n | 0.0975 |
| 11 | 768 | 12 | 4 | 2048 | 93.80M | 69.22M | 26% | -6.2% | 70 | n | 0.0995 |
| 24 | 576 | 9 | 3 | 1536 | 103.39M | 84.96M | 18% | +3.4% | 24 | n | 0.1033 |

### target 250M: 41 shapes within +-10%; top 6 by score (score = |rel_err| + 0.1*|ln(aspect/48)|; * = chosen)
| L | d_model | heads | kv | d_ff | total | non-emb | emb% | rel.err | d/L | in band | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 960 | 15 | 5 | 2560 | 247.03M | 216.31M | 12% | -1.2% | 44 | y | 0.0214 * |
| 23 | 960 | 15 | 5 | 2560 | 256.86M | 226.14M | 12% | +2.7% | 42 | y | 0.0414 |
| 15 | 1152 | 18 | 6 | 3072 | 249.24M | 212.37M | 15% | -0.3% | 77 | n | 0.0501 |
| 21 | 960 | 15 | 5 | 2560 | 237.20M | 206.48M | 13% | -5.1% | 46 | y | 0.0561 |
| 24 | 960 | 15 | 5 | 2560 | 266.70M | 235.98M | 12% | +6.7% | 40 | y | 0.0850 |
| 36 | 768 | 12 | 4 | 2048 | 251.12M | 226.55M | 10% | +0.4% | 21 | n | 0.0856 |

### target 500M: 61 shapes within +-10%; top 6 by score (score = |rel_err| + 0.1*|ln(aspect/48)|; * = chosen)
| L | d_model | heads | kv | d_ff | total | non-emb | emb% | rel.err | d/L | in band | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1344 | 21 | 7 | 3584 | 505.50M | 462.49M | 9% | +1.1% | 56 | y | 0.0264 * |
| 33 | 1152 | 18 | 6 | 3072 | 504.08M | 467.22M | 7% | +0.8% | 35 | y | 0.0400 |
| 23 | 1344 | 21 | 7 | 3584 | 486.23M | 443.22M | 9% | -2.8% | 58 | y | 0.0472 |
| 32 | 1152 | 18 | 6 | 3072 | 489.92M | 453.06M | 8% | -2.0% | 36 | y | 0.0489 |
| 25 | 1344 | 21 | 7 | 3584 | 524.77M | 481.76M | 8% | +5.0% | 54 | y | 0.0609 |
| 18 | 1536 | 24 | 8 | 4096 | 502.19M | 453.04M | 10% | +0.4% | 85 | n | 0.0619 |

### target 1B: 76 shapes within +-10%; top 6 by score (score = |rel_err| + 0.1*|ln(aspect/48)|; * = chosen)
| L | d_model | heads | kv | d_ff | total | non-emb | emb% | rel.err | d/L | in band | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 38 | 1536 | 24 | 8 | 4096 | 1.006B | 956.42M | 5% | +0.6% | 40 | y | 0.0228 * |
| 30 | 1728 | 27 | 9 | 4608 | 1.011B | 955.62M | 5% | +1.1% | 58 | y | 0.0291 |
| 37 | 1536 | 24 | 8 | 4096 | 980.40M | 931.25M | 5% | -2.0% | 42 | y | 0.0341 |
| 29 | 1728 | 27 | 9 | 4608 | 979.06M | 923.77M | 6% | -2.1% | 60 | y | 0.0426 |
| 39 | 1536 | 24 | 8 | 4096 | 1.031B | 981.59M | 5% | +3.1% | 39 | y | 0.0505 |
| 24 | 1920 | 30 | 10 | 5120 | 1.005B | 943.81M | 6% | +0.5% | 80 | n | 0.0563 |

Chosen family with the per-layer breakdown, FLOPs per token, and KV-cache cost:

| name | L | d_model | heads | kv_heads | head_dim | d_ff | vocab | total | non-emb | emb% | d/L | attn/layer | ffn/layer | C_fwd/tok (full T) | C_train/tok (causal) | attn share | 6N_total | KV B/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25M | 8 | 384 | 6 | 2 | 64 | 1024 | 32000 | 24,877,440 | 12,589,440 | 49.4% | 48 | 393,216 | 1,179,648 | 7.492e+07 | 1.870e+08 | 20% | 1.493e+08 | 4,096 |
| 100M | 12 | 768 | 12 | 4 | 64 | 2048 | 32000 | 100,092,672 | 75,516,672 | 24.6% | 64 | 1,572,864 | 4,718,592 | 2.757e+08 | 7.138e+08 | 16% | 6.006e+08 | 12,288 |
| 250M | 22 | 960 | 15 | 5 | 64 | 2560 | 32000 | 247,032,000 | 216,312,000 | 12.4% | 44 | 2,457,600 | 7,372,800 | 6.671e+08 | 1.742e+09 | 15% | 1.482e+09 | 28,160 |
| 500M | 24 | 1344 | 21 | 7 | 64 | 3584 | 32000 | 505,495,872 | 462,487,872 | 8.5% | 56 | 4,816,896 | 14,450,688 | 1.275e+09 | 3.429e+09 | 12% | 3.033e+09 | 43,008 |
| 1B | 38 | 1536 | 24 | 8 | 64 | 4096 | 32000 | 1,005,571,584 | 956,419,584 | 4.9% | 40 | 6,291,456 | 18,874,368 | 2.489e+09 | 6.751e+09 | 11% | 6.033e+09 | 77,824 |

Reading the table: the attention share of forward FLOPs at T = 2048 with a causal kernel is 20 %, 16 %, 15 %, 12 %, 11 %; `6N_total` is the Chinchilla-convention training cost. The KV cache is 4 KB / token at 25M and 78 KB / token at 1B in bf16 (2 x L x n_kv x head_dim x 2 bytes), i.e. a 2048-token sequence costs at most 160 MB even at 1B.

## 3. Tokenizer and vocabulary size

### 3.1 Embedding share [fact]

Tied embedding parameters are V x d. For the chosen 25M (d = 384) and 100M (d = 768) widths, and for the best feasible shape the search finds under each vocabulary:

| vocab | V*384 (25M @ d=384) | share of 25M | V*768 (100M @ d=768) | share of 100M | best 25M shape (L x d) | 25M non-emb | best 100M shape | 100M non-emb |
|---|---|---|---|---|---|---|---|---|
| 8192 | 3.15M | 13% | 6.29M | 6% | 14x384 | 22.03M | 15x768 | 94.40M |
| 16384 | 6.29M | 25% | 12.58M | 13% | 12x384 | 18.88M | 14x768 | 88.10M |
| 32000 | 12.29M | 49% | 24.58M | 25% | 8x384 | 12.59M | 12x768 | 75.52M |
| 32768 | 12.58M | 50% | 25.17M | 25% | 8x384 | 12.59M | 12x768 | 75.52M |
| 49152 | 18.87M | 75% | 37.75M | 38% | 4x384 | 6.29M | 18x576 | 63.72M |
| 50257 | 19.30M | 77% | 38.60M | 39% | 4x384 | 6.29M | 18x576 | 63.72M |
| 65536 | 25.17M | 101% | 50.33M | 50% | 31x192 | 12.20M | 18x576 | 63.72M |
| 128256 | 49.25M | 197% | 98.50M | 99% | 4x192 | 1.57M | 9x576 | 31.86M |
| 151936 | 58.34M | 233% | 116.69M | 117% | infeasible | - | 26x384 | 40.91M |

- A 128k (Llama-3) or 151k (Qwen2.5) vocabulary is **infeasible** at 25M: the embedding alone is 49.3M resp. 58.3M parameters, i.e. 197 % and 233 % of the budget; the only "solutions" are degenerate (4 x 192 with 1.6M non-embedding parameters, or nothing). At 100M they still consume 99 % / 117 % of the budget, leaving a 9 x 576 or 26 x 384 model. This is the quantitative form of P4's rejection of the teacher tokenizer.
- 49k–50k vocabularies (SmolLM, GPT-2) take 75–77 % of a 25M budget and leave a 4-layer model with 6.3M non-embedding parameters — not acceptable either.
- 32k leaves 12.6M non-embedding parameters (49 % embedding share); 16k leaves 18.9M (25 %); 8k leaves 22.0M (13 %).

### 3.2 Tokenisation efficiency (fertility) — measured on proxy corpora

The FineWeb-Edu sample could not be downloaded from the sandbox, so fertility (tokens per whitespace-delimited word) was measured on English proxy corpora that *are* reachable (NLTK Brown, Reuters-21578, ABC news, web-forum/review text, State-of-the-Union speeches, Gutenberg books, and de-tokenised WikiText-2). Custom byte-level BPEs (GPT-2 pre-tokeniser + Llama-2-style per-digit splitting, `tokenizers` library) were trained on the 90 % held-in split (31 MB) and all tokenizers were evaluated on the 10 % held-out split (0.68M words). Numbers for the custom BPEs are *optimistic* (trained in-domain on a small corpus); numbers for the pretrained tokenizers are pessimistic for FineWeb-Edu because Reuters (all-caps headlines, dense numerals) is 20 % of the held-out words. **[assumption] The pipeline must re-run `fertility.py` on the real base corpus and on a sample of teacher outputs before budgets are fixed.**

```
docs total 14018, train 12607 (31.1 MB), eval 1411 (4.0 MB, 0.68M words)
  eval genre abc_news             words   65,638  bytes/word 6.09
  eval genre books                words  267,080  bytes/word 5.47
  eval genre brown_mixed          words  115,814  bytes/word 5.96
  eval genre reuters_news         words  140,225  bytes/word 6.37
  eval genre speeches             words   47,408  bytes/word 5.91
  eval genre web_forum_reviews    words   30,062  bytes/word 5.63
  eval genre wikipedia            words   11,711  bytes/word 5.81
```

| tokenizer | vocab | tokens/word (pooled) | bytes/token | words per 2048-token ctx | t/w abc_news | t/w books | t/w brown_mixed | t/w reuters_news | t/w speeches | t/w web_forum_reviews | t/w wikipedia | numbers probe | code probe | qa probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-2 SP 32000 | 32000 | 1.614 | 3.62 | 1269 | 1.483 | 1.597 | 1.410 | 1.977 | 1.341 | 1.620 | 1.505 | 49 | 35 | 16 |
| GPT-2 BPE 50257 | 50257 | 1.457 | 4.01 | 1406 | 1.297 | 1.469 | 1.247 | 1.793 | 1.197 | 1.446 | 1.228 | 31 | 46 | 15 |
| Llama-3 tiktoken 128256 | 128256 | 1.383 | 4.23 | 1481 | 1.234 | 1.390 | 1.248 | 1.628 | 1.184 | 1.375 | 1.278 | 36 | 31 | 14 |
| custom byte-BPE 8192 (proxy) | 8192 | 1.603 | 3.65 | 1277 | 1.557 | 1.587 | 1.515 | 1.789 | 1.350 | 1.697 | 1.651 | 51 | 42 | 22 |
| custom byte-BPE 16384 (proxy) | 16384 | 1.488 | 3.93 | 1376 | 1.421 | 1.487 | 1.382 | 1.671 | 1.259 | 1.558 | 1.508 | 51 | 39 | 21 |
| custom byte-BPE 32768 (proxy) | 32768 | 1.419 | 4.12 | 1443 | 1.338 | 1.426 | 1.302 | 1.598 | 1.218 | 1.464 | 1.422 | 51 | 36 | 19 |
| custom byte-BPE 49152 (proxy) | 49152 | 1.393 | 4.20 | 1471 | 1.308 | 1.403 | 1.268 | 1.573 | 1.204 | 1.430 | 1.383 | 51 | 36 | 17 |
| custom byte-BPE 65536 (proxy) | 65536 | 1.379 | 4.24 | 1485 | 1.292 | 1.391 | 1.251 | 1.559 | 1.199 | 1.412 | 1.368 | 50 | 35 | 16 |

Pooled fertility excluding Reuters, relative to the Llama-2 tokenizer:

| tokenizer | vocab | t/w pooled | t/w excl. Reuters | relative to Llama-2 | words per 2048-token context |
|---|---|---|---|---|---|
| Llama-2 SP 32000 | 32000 | 1.614 | 1.520 | 1.000 | 1348 |
| GPT-2 BPE 50257 | 50257 | 1.457 | 1.370 | 0.901 | 1495 |
| Llama-3 tiktoken 128256 | 128256 | 1.383 | 1.319 | 0.868 | 1553 |
| custom byte-BPE 8192 (proxy) | 8192 | 1.603 | 1.555 | 1.023 | 1317 |
| custom byte-BPE 16384 (proxy) | 16384 | 1.488 | 1.441 | 0.948 | 1421 |
| custom byte-BPE 32768 (proxy) | 32768 | 1.419 | 1.372 | 0.903 | 1493 |
| custom byte-BPE 49152 (proxy) | 49152 | 1.393 | 1.346 | 0.886 | 1522 |
| custom byte-BPE 65536 (proxy) | 65536 | 1.379 | 1.333 | 0.877 | 1537 |

Observations [fact on this proxy; direction expected to hold on FineWeb-Edu]:

- Vocabulary returns are logarithmic while the embedding cost is linear in V: within the custom-BPE series (same corpus, same pre-tokeniser), 8k -> 16k saves 7 % of tokens, 16k -> 32k saves 5 %, 32k -> 65k saves 3 %. The larger gaps between Llama-2 (32k) and GPT-2 (50k) / Llama-3 (128k) — 10 % and 13 % — mostly reflect Llama-2's multilingual + code training mix, not vocabulary size.
- The Llama-2 tokenizer is ~10 % *less* efficient on English than an English-only 32k BPE of the same size, because its SentencePiece model was trained on a multilingual + code mix. An English-only 16k BPE trained on the base corpus reaches Llama-2-32k fertility (0.95x) with half the embedding table.
- Effective context: 2048 tokens hold ~1350 English words with Llama-2-32k and ~1500 with a 50k BPE. Instruction/response synthetic examples (< 512 tokens, i.e. < ~340 words) fit 4–8 per packed sequence in every case.
- Budgets: because D_syn is counted in student tokens (P2), the tokenizer sets the amount of *text* a fixed budget buys; this is a constant factor across teachers within a student and across students within the study, so it does not affect the teacher comparisons. It does matter for the conversion from teacher-token generation cost to student tokens: Qwen2.5's 151k byte-level BPE is in the Llama-3 efficiency class, so ~1.15 student (Llama-2) tokens per Qwen token is the expected conversion on English prose [assumption; measure on actual generations].

### 3.3 Representation and evaluation considerations

- Byte fallback / coverage: Llama-2 SP has 256 byte tokens (any UTF-8 string is encodable); GPT-2, SmolLM and any custom byte-level BPE are byte-complete by construction. No candidate produces `<unk>`.
- Numbers: Llama-2 splits every digit (numbers probe: 49 tokens for 85 chars, custom BPEs 51), GPT-2 merges digit runs arbitrarily (31), Llama-3 chunks 3 digits (36). Per-digit splitting is the convention adopted by Llama-2 and PaLM for arithmetic consistency; for models of 25M–1B that will not do multi-digit arithmetic reliably in any case, the main requirement is *consistency* across the family, which every candidate satisfies.
- Code / whitespace: the code probe (97 chars) costs 35 tokens with Llama-2, 46 with GPT-2 (GPT-2 encodes runs of spaces poorly), 31 with Llama-3, 36 with the custom 32k BPE. Llama-2 is adequate for the small amount of code that teacher outputs may contain.
- Benchmarks (P6) are log-likelihood multiple-choice tasks and are tokenizer-agnostic; lm-eval-harness accepts any Hugging Face tokenizer.

### 3.4 Decision [choice]

**Llama-2 SentencePiece BPE, V = 32 000, tied.** Reasons, in order of weight:

1. Feasibility at 25M: 32k is the largest mainstream vocabulary that leaves a non-degenerate 25M model (8 layers, 12.6M non-embedding parameters). 49k–151k vocabularies do not.
2. Reproducibility and comparability: it is an off-the-shelf, fixed artefact (also used by MobileLLM at 125M–1B and by TinyLlama), so our 100M–1B rows are directly comparable to published configurations, and no tokenizer-training step enters the pipeline.
3. Byte fallback, per-digit splitting and adequate code handling (Section 3.3).
4. V = 32 000 = 500 x 64, so the LM-head matmul needs no padding.

Costs accepted: (a) ~10 % more tokens than an English-only BPE of the same size (a constant factor in all budgets and throughput figures); (b) the 25M model is 49 % embeddings (Section 10, issue 1). The strongest alternative is a **custom 16k byte-level BPE trained on the base corpus**: same fertility as Llama-2-32k on the proxy (0.95x), embedding share at 25M drops from 49 % to 25 % and non-embedding parameters rise from 12.6M to 18.9M (+50 %), at the price of one extra artefact to release and validate and of a vocabulary smaller than any published LM's (Tao et al. 2024, "Scaling Laws with Vocabulary", https://arxiv.org/abs/2407.13623, predict that the compute-optimal vocabulary grows with non-embedding size; their fitted optimum for ~10^7–10^8 non-embedding parameters lies in the 10k–30k range [UNVERIFIED exact values], so 16k and 32k are both defensible for this family). A custom 32k BPE would be ~10 % more token-efficient than Llama-2 but does not change the embedding share; not worth the reproducibility cost.

## 4. Embedding tying: one rule for all sizes [choice]

**Tie the input embedding and the LM head at every size.**

- Arithmetic [fact]: untied, the 25M model would carry 2 x 12.29M = 24.6M embedding parameters — the entire budget — and the search returns no in-band shape. Untied at 100M would be 49 % embeddings. Since the rule must be uniform (P4), tying is forced by the small end.
- Literature [lit]: Press and Wolf (2017, https://arxiv.org/abs/1608.05859) introduced output-embedding tying and showed it improves perplexity while removing V x d parameters; GPT-2 (Radford et al. 2019) ties; MobileLLM (Liu et al. 2024, https://arxiv.org/abs/2402.14905, Section 3.3) report that sharing the embedding at 125M removes ~16M parameters (~12 %) for a ~0.2-point loss in average zero-shot accuracy that is recovered when the saved parameters are re-invested in depth; SmolLM/SmolLM2 (Allal et al. 2025, https://arxiv.org/abs/2502.02737) tie at 135M, 360M and 1.7B; OLMo-1B ties while OLMo-7B does not (Groeneveld et al. 2024, https://arxiv.org/abs/2402.00838); Llama-3.2-1B/3B tie while 8B+ do not; Qwen2.5-0.5B/1.5B/3B tie while 7B+ do not; Gemma ties at all sizes. Pythia (Biderman et al. 2023, https://arxiv.org/abs/2304.01373) is the notable untied family, and its 70M model is 73 % embeddings as a result (reference table). The pattern in current practice is "tie below ~3B", which covers our whole family.
- Consequence for reporting: because tying makes "total" and "non-embedding" differ by V x d only once, the 25M–1B family has non-embedding counts 12.6M, 75.5M, 216M, 462M, 956M; both numbers are reported everywhere.

## 5. Attention: GQA rule and head_dim [choice]

**GQA with n_kv_heads = n_heads / 3 at every size, head_dim = 64.**

- Why one ratio everywhere rather than "MHA below 250M, GQA above": the study fits trends across student size S; changing the architecture function at some S introduces a discontinuity in parameters-per-d^2 and in attention capacity that would be confounded with S. Modern small models use GQA at all sizes (SmolLM 3:1 at 135M/360M, MobileLLM 3:1 at 125M–600M and 4:1 at 1B, Qwen2.5-0.5B 7:1, Llama-3.2-1B 4:1, TinyLlama 8:1); the only MHA holdouts are older (Pythia, GPT-2) or at the 1B+ end (OLMo-1B, SmolLM-1.7B).
- Why 3:1 rather than 4:1 [fact]: with head_dim 64 and a 32k vocabulary, 4:1 needs n_heads divisible by 4, i.e. d = 512 at the small end, and the 25M budget then only affords **3 layers** (table below); 3:1 works at d = 384 with 8 layers. 3:1 is also SmolLM's and MobileLLM's rule, so our 100M–600M rows are comparable. 2:1 and MHA give sensible shapes too (table) but forgo the KV-cache and parameter savings.
- Evidence GQA is not harmful at small scale [lit]: Ainslie et al. (2023, https://arxiv.org/abs/2305.13245) match MHA quality with 8 KV groups at T5-XXL; MobileLLM (Section 3.4) find that reducing KV heads to 1/4 at 125M leaves accuracy unchanged and, when the saved parameters are spent on width, gains ~0.4 points [UNVERIFIED exact figure].
- head_dim = 64 [choice]: used by every reference model in the table except Pythia-1B (256) and OLMo-1B (128); FlashAttention-2 kernels are tuned for 64/128 (Dao 2023, https://arxiv.org/abs/2307.08691); head_dim 128 would give the 25M model 3 heads and 1 KV head.
- Cost accounting: attention parameters per layer are 2 d^2 + 2 d d_kv = 2.67 d^2 with 3:1 versus 4 d^2 for MHA; the KV cache is one third of MHA's.

| rule | 25M | 100M | 250M | 500M | 1B |
|---|---|---|---|---|---|
| MHA (1:1) | 7x384 h6/kv6 24.68M (-1.3%) | 16x640 h10/kv10 99.80M (-0.2%) | 20x960 h15/kv15 251.94M (+0.8%) | 26x1216 h19/kv19 502.34M (+0.5%) | 31x1600 h25/kv25 1.007B (+0.7%) |
| GQA 2:1 | 8x384 h6/kv3 25.27M (+1.1%) | 17x640 h10/kv5 97.79M (-2.2%) | 19x1024 h16/kv8 253.21M (+1.3%) | 25x1280 h20/kv10 495.68M (-0.9%) | 31x1664 h26/kv13 1.004B (+0.4%) |
| GQA 3:1 (chosen) | 8x384 h6/kv2 24.88M (-0.5%) | 12x768 h12/kv4 100.09M (+0.1%) | 22x960 h15/kv5 247.03M (-1.2%) | 24x1344 h21/kv7 505.50M (+1.1%) | 38x1536 h24/kv8 1.006B (+0.6%) |
| GQA 4:1 | 3x512 h8/kv2 24.84M (-0.6%) | 12x768 h12/kv3 98.91M (-1.1%) | 20x1024 h16/kv4 254.32M (+1.7%) | 26x1280 h20/kv5 492.57M (-1.5%) | 38x1536 h24/kv6 990.63M (-0.9%) |
| MHA, head_dim 128 | 7x384 h3/kv3 24.68M (-1.3%) | 16x640 h5/kv5 99.80M (-0.2%) | 17x1024 h8/kv8 247.83M (-0.9%) | 23x1280 h10/kv10 496.99M (-0.6%) | 34x1536 h12/kv12 1.012B (+1.2%) |

(The 4:1 entry at 25M is the out-of-band fallback 3 x 512; no in-band 4:1 shape exists at 25M.)

## 6. Depth versus width [choice, informed by lit]

Evidence used:

- Kaplan et al. (2020, Section 3.1, Fig. 5) [lit]: at fixed non-embedding N, loss varies by only a few percent across aspect ratios d/L spanning more than an order of magnitude; shape is a second-order effect.
- MobileLLM (Liu et al. 2024, Section 3.2) [lit]: at 125M and 350M, deep-and-thin (30–32 layers, d/L about 19–30) beats wide-and-shallow (12 layers, d/L 64) by roughly 1–2 average zero-shot points at equal parameters. Tay et al. (2021, "Scale Efficiently", https://arxiv.org/abs/2109.10686) reach the same "DeepNarrow" conclusion for T5; Levine et al. (2020, https://arxiv.org/abs/2006.12467) derive a width-dependent optimal depth. Petty et al. (2023, https://arxiv.org/abs/2310.19956) find depth helps generalisation with diminishing returns beyond a few tens of layers.
- Practice: GPT-2 / Pythia use d/L = 43–128; SmolLM / MobileLLM use 19–30; TinyLlama, OLMo-1B and Llama-3.2-1B use 93–128 at 1B.
- Constraint from the small end [fact]: with a 32k tied vocabulary the only in-tolerance 25M shapes are 7–9 layers at d = 384 (d/L 43–55) or absurd 47–48 x 192 towers; deep-and-thin at MobileLLM's ratio is not available at 25M.
- Throughput: at d = 384–960, per-layer matmuls are small; deeper-thinner shapes lower GPU utilisation further (Section 8), which matters because 30 cells x seeds must be trained.

Decision: aspect band [32, 64] with target 48 — between the GPT-2/Pythia and the MobileLLM/SmolLM regimes, close to Kaplan's flat region, and feasible at every size. The resulting ratios are 48, 64, 44, 56, 40. The 100M model (12 x 768) is deliberately the GPT-2-small skeleton; the search's runner-up 23 x 576 (MobileLLM-style, d/L = 25) is documented in the candidate table and would be the choice if the lead prefers to follow MobileLLM strictly (Section 10, issue 4).

Reference configurations (totals recomputed with `param_count.py`; "reported" from the papers / model cards; MobileLLM-600M/1B dimensions are from memory of the model cards and marked UNVERIFIED):

| model | L | d | heads | kv | head_dim | d_ff | vocab | tied | d/L | total (formula) | non-emb | emb% | reported | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pythia-70M | 6 | 512 | 8 | 8 | 64 | 2048 | 50304 | n | 85 | 70.43M | 18.92M | 73% | 70.40M | untied; GELU 4x; LN w/ bias; parallel attn/MLP |
| Pythia-160M | 12 | 768 | 12 | 12 | 64 | 3072 | 50304 | n | 64 | 162.32M | 85.06M | 48% | 162.30M | untied |
| Pythia-410M | 24 | 1024 | 16 | 16 | 64 | 4096 | 50304 | n | 43 | 405.33M | 302.31M | 25% | 405.30M | untied |
| Pythia-1B | 16 | 2048 | 8 | 8 | 256 | 8192 | 50304 | n | 128 | 1.012B | 805.74M | 20% | 1.012B | untied; head_dim 256 |
| GPT-2 small | 12 | 768 | 12 | 12 | 64 | 3072 | 50257 | y | 64 | 124.44M | 85.06M | 32% | 124.40M | tied; learned pos. |
| GPT-2 medium | 24 | 1024 | 16 | 16 | 64 | 4096 | 50257 | y | 43 | 354.82M | 302.31M | 15% | 354.80M | tied; learned pos. |
| SmolLM(2)-135M | 30 | 576 | 9 | 3 | 64 | 1536 | 49152 | y | 19 | 134.52M | 106.20M | 21% | 134.50M | GQA 3:1; tied |
| SmolLM(2)-360M | 32 | 960 | 15 | 5 | 64 | 2560 | 49152 | y | 30 | 361.82M | 314.64M | 13% | 361.80M | GQA 3:1; tied |
| SmolLM(2)-1.7B | 24 | 2048 | 32 | 32 | 64 | 8192 | 49152 | y | 85 | 1.711B | 1.611B | 6% | 1.711B | MHA; d_ff = 4d; tied |
| MobileLLM-125M | 30 | 576 | 9 | 3 | 64 | 1536 | 32000 | y | 19 | 124.64M | 106.20M | 15% | 124.60M | GQA 3:1; tied; Llama-2 tokenizer |
| MobileLLM-350M | 32 | 960 | 15 | 5 | 64 | 2560 | 32000 | y | 30 | 345.36M | 314.64M | 9% | 345.30M | GQA 3:1; tied |
| MobileLLM-600M | 40 | 1152 | 18 | 6 | 64 | 3072 | 32000 | y | 29 | 603.19M | 566.32M | 6% | 603.00M | GQA 3:1; tied [dims UNVERIFIED] |
| MobileLLM-1B | 54 | 1280 | 20 | 5 | 64 | 3440 | 32000 | y | 24 | 975.60M | 934.64M | 4% | 1.000B | GQA 4:1; tied [dims UNVERIFIED] |
| TinyLlama-1.1B | 22 | 2048 | 32 | 4 | 64 | 5632 | 32000 | n | 93 | 1.100B | 968.98M | 12% | 1.100B | GQA 8:1; untied |
| OLMo-1B (Feb-24) | 16 | 2048 | 16 | 16 | 128 | 8192 | 50304 | y | 128 | 1.177B | 1.074B | 9% | 1.177B | MHA; non-parametric LN; tied |
| Qwen2.5-0.5B | 24 | 896 | 14 | 2 | 64 | 4864 | 151936 | y | 37 | 494.01M | 357.87M | 28% | 494.00M | GQA 7:1; tied; +QKV biases (not counted) |
| Llama-3.2-1B | 16 | 2048 | 32 | 8 | 64 | 8192 | 128256 | y | 128 | 1.236B | 973.15M | 21% | 1.236B | GQA 4:1; tied |

Sources: Pythia — Biderman et al. 2023 (https://arxiv.org/abs/2304.01373); GPT-2 — Radford et al. 2019; SmolLM2 — Allal et al. 2025 (https://arxiv.org/abs/2502.02737); MobileLLM — Liu et al. 2024 (https://arxiv.org/abs/2402.14905); TinyLlama — Zhang et al. 2024 (https://arxiv.org/abs/2401.02385); OLMo — Groeneveld et al. 2024 (https://arxiv.org/abs/2402.00838); Qwen2.5 — Yang et al. 2024 (https://arxiv.org/abs/2412.15115); Llama 3 — Grattafiori et al. 2024 (https://arxiv.org/abs/2407.21783). Where a model's block differs from ours (Pythia/GPT-2: GELU 4x MLP with biases, biased LayerNorm, learned positions; OLMo: non-parametric LayerNorm; Qwen2.5: QKV biases) the formula variant is stated in the note column; the agreement with reported totals is within 0.1 % in every case except MobileLLM-1B (2.4 % low, consistent with the dimensions being unverified).

## 7. Context length and positional encoding [choice]

**ctx = 2048, RoPE theta = 10 000, no positional interpolation.**

- Every reference small model in Section 6 trains at 2048 (Pythia, TinyLlama, MobileLLM, SmolLM, OLMo-1B), so evaluation comparisons need no length adjustment.
- Few-shot evaluation: 5-shot HellaSwag and 5-shot BoolQ prompts routinely exceed 1024 tokens [assumption; to be checked by the evaluation workstream on the actual harness], so a 1024 context would force zero-shot-only evaluation or truncation.
- Base pretraining cost: attention is 20 % (25M) to 11 % (1B) of causal forward FLOPs at T = 2048; halving T saves 10 %, 8 %, 7 %, 6 %, 5 % of training FLOPs — a modest saving that does not justify a non-standard context.
- Synthetic phase: instruction + response examples are typically < 512 tokens (< ~340 English words), so 4–8 examples are packed per 2048-token sequence. Whether packed examples may attend across example boundaries (standard pretraining practice) or are isolated with a block-diagonal mask (FlashAttention `varlen`) is a training-recipe decision; it must be identical in every cell, and the isolated variant is recommended because it removes a spurious dependence of the loss on packing order.
- RoPE base 10 000 is the Llama-2 value and is sufficient for T <= 4096; Llama-3's 500 000 is needed only for 8k–128k contexts (Grattafiori et al. 2024). Documented so that the value is not silently inherited from a Llama-3 config.

## 8. Precision, initialisation, activation checkpointing, memory, throughput

### 8.1 Precision [choice]
bf16 autocast for matmuls and attention; fp32 master weights, fp32 gradients, fp32 AdamW moments (16 bytes / parameter of state); RMSNorm statistics, softmax/cross-entropy and the loss in fp32; no loss scaling (bf16 has the fp32 exponent range; Micikevicius et al. 2018, https://arxiv.org/abs/1710.03740, is the fp16 case). Logits are computed by a fused or chunked linear-cross-entropy kernel so that the V x T fp32 logit tensor is never materialised (unfused it costs 8 V = 256 KB per token, i.e. 4 GB per 16k-token micro-batch).

### 8.2 Initialisation [choice]
N(0, 0.02) for all projection matrices and the (tied) embedding; the two residual-writing projections per layer (W_o, W_down) scaled to std 0.02 / sqrt(2 L) (Radford et al. 2019; the same scheme is used by nanoGPT/llm.c and by OLMo's "full_megatron"-style init); RMSNorm gains = 1. The per-size scale factors are printed in the YAML blocks (0.25 at L = 8 down to 0.115 at L = 38). A width-scaled init (std = 1/sqrt(d), as in SmolLM-135M's 0.0417, or muP) was considered and rejected to stay with the most common recipe; with d in 384–1536 the fixed 0.02 lies within the range GPT-2 used successfully (768–1600). Stability additions (QK-LayerNorm, z-loss; Wortsman et al. 2023, https://arxiv.org/abs/2309.14322) are not included by default; see Section 10, issue 7. Peak learning rates must still be tuned per size (training-recipe workstream); tied embeddings with a 0.02 init give initial logit std of 0.02 sqrt(d) = 0.39 (25M) to 0.78 (1B), which is benign.

### 8.3 Memory model [fact, with stated assumptions]
- States: 16 x N_total bytes (fp32 weights + fp32 grads + Adam m, v). With 8-bit AdamW (Dettmers et al. 2022, https://arxiv.org/abs/2110.02861): 10 x N_total.
- Activations saved for backward per token per layer (bf16, FlashAttention, no T^2 tensor): residual input, norm output, Q, K, V, attention output, attention-projection output, FFN-norm output, gate, up, silu(gate), silu(gate) * up, down output = 5 d + 2 d_attn + 2 d_kv + 4 d_ff elements (36.7 d bytes at d_ff = 8 d / 3 and 3:1 GQA), plus fp32 log-sum-exp per head, times a 1.25 safety factor for temporaries and allocator fragmentation. This is the SwiGLU analogue of the 34 s b h estimate of Korthikanti et al. (2022, https://arxiv.org/abs/2205.05198). Full checkpointing keeps 2 d bytes per layer per token plus one live layer.
- Fused cross-entropy: 2 V bytes per token live; CUDA context, kernels and workspace: 2 GB.
- GPU peaks [assumption, UNVERIFIED]: RTX PRO 6000 Blackwell 96 GB, 500 TFLOPS dense bf16 with fp32 accumulate, 1.79 TB/s; RTX 5080 16 GB, 110 TFLOPS (consumer parts run fp32-accumulate tensor math at half rate), 0.96 TB/s. If the workstation part is in fact rated at 250 TFLOPS for fp32-accumulate, halve its tokens/s.
- MFU [assumption]: 0.10, 0.20, 0.28, 0.33, 0.38 for 25M … 1B. Small models are bound by memory traffic and by the 32k-way softmax (the LM head is 49 % of the 25M model's weight-matmul FLOPs: 2 V d = 24.6 MFLOP of 49.8 MFLOP), so their utilisation is low regardless of GPU; these values are conservative for a `torch.compile` + FlashAttention stack and must be replaced by pilot measurements. Tokens/s = MFU x peak / C_train(causal).

| model | GPU | ckpt | states GB | act B/tok | max micro-batch (seq x 2048) | accum for 1M tok | C_train/tok | MFU | tokens/s | hours for 20 tok/param |
|---|---|---|---|---|---|---|---|---|---|---|
| 25M | RTX 5080 (16 GB) | none | 0.4 | 200K | 34 (69k) | 16 | 1.87e+08 | 0.10 | 59k | 2.3 |
| 25M | RTX 5080 (16 GB) | full | 0.4 | 87K | 80 (163k) | 7 | 2.49e+08 | 0.10 | 44k | 3.1 |
| 25M | RTX PRO 6000 Blackwell (96 GB) | none | 0.4 | 200K | 239 (489k) | 3 | 1.87e+08 | 0.10 | 267k | 0.5 |
| 25M | RTX PRO 6000 Blackwell (96 GB) | full | 0.4 | 87K | 549 (1124k) | 1 | 2.49e+08 | 0.10 | 201k | 0.7 |
| 100M | RTX 5080 (16 GB) | none | 1.5 | 476K | 13 (26k) | 40 | 7.14e+08 | 0.20 | 31k | 18.0 |
| 100M | RTX 5080 (16 GB) | full | 1.5 | 119K | 53 (108k) | 10 | 9.52e+08 | 0.20 | 23k | 24.1 |
| 100M | RTX PRO 6000 Blackwell (96 GB) | none | 1.5 | 476K | 99 (202k) | 6 | 7.14e+08 | 0.20 | 140k | 4.0 |
| 100M | RTX PRO 6000 Blackwell (96 GB) | full | 1.5 | 119K | 396 (811k) | 2 | 9.52e+08 | 0.20 | 105k | 5.3 |
| 250M | RTX 5080 (16 GB) | none | 3.7 | 1010K | 5 (10k) | 103 | 1.74e+09 | 0.28 | 18k | 77.6 |
| 250M | RTX 5080 (16 GB) | full | 3.7 | 157K | 33 (67k) | 16 | 2.32e+09 | 0.28 | 13k | 103.5 |
| 250M | RTX PRO 6000 Blackwell (96 GB) | none | 3.7 | 1010K | 45 (92k) | 12 | 1.74e+09 | 0.28 | 80k | 17.1 |
| 250M | RTX PRO 6000 Blackwell (96 GB) | full | 3.7 | 157K | 294 (602k) | 2 | 2.32e+09 | 0.28 | 60k | 22.8 |
| 500M | RTX 5080 (16 GB) | none | 7.5 | 1509K | 2 (4k) | 256 | 3.43e+09 | 0.33 | 11k | 265.3 |
| 500M | RTX 5080 (16 GB) | full | 7.5 | 202K | 16 (32k) | 32 | 4.57e+09 | 0.33 | 8k | 353.7 |
| 500M | RTX PRO 6000 Blackwell (96 GB) | none | 7.5 | 1509K | 29 (59k) | 18 | 3.43e+09 | 0.33 | 48k | 58.4 |
| 500M | RTX PRO 6000 Blackwell (96 GB) | full | 7.5 | 202K | 219 (448k) | 3 | 4.57e+09 | 0.33 | 36k | 77.8 |
| 1B | RTX 5080 (16 GB) | none | 15.0 | 2680K | DOES NOT FIT | - | 6.75e+09 | 0.38 | 6k | 902.2 |
| 1B | RTX 5080 (16 GB) | full | 15.0 | 274K | DOES NOT FIT | - | 9.00e+09 | 0.38 | 5k | 1203.0 |
| 1B | RTX PRO 6000 Blackwell (96 GB) | none | 15.0 | 2680K | 15 (30k) | 35 | 6.75e+09 | 0.38 | 28k | 198.5 |
| 1B | RTX PRO 6000 Blackwell (96 GB) | full | 15.0 | 274K | 147 (301k) | 4 | 9.00e+09 | 0.38 | 21k | 264.6 |

| model | ckpt | states GB | max micro-batch | accum for 1M tok |
|---|---|---|---|---|
| 500M | none | 4.7 | 3 (6k) | 171 |
| 500M | full | 4.7 | 23 (47k) | 23 |
| 1B | none | 9.4 | DOES NOT FIT | - |
| 1B | full | 9.4 | 8 (16k) | 64 |

Reading the plan:

- **Fit**: 25M–250M fit on a 16 GB RTX 5080 without checkpointing (micro-batches of 34 / 13 / 5 sequences); 500M fits on the 5080 only with full checkpointing (16 sequences, +33 % FLOPs) or 8-bit Adam; 1B does not fit on 16 GB with fp32 optimizer state and is marginal even with 8-bit Adam plus checkpointing (8 sequences). On the 96 GB card every size fits without checkpointing (1B: 15 sequences = 30k tokens per micro-batch, 35 accumulation steps for 1M tokens/step).
- **Checkpointing policy** [choice]: none by default; full checkpointing only for the 500M model on 16 GB cards. Run 25M/100M/250M cells on the RTX 5080 machines and 500M/1B cells on the RTX PRO 6000s; with two PRO 6000s in DDP the 1B base run takes about 4 days wall-clock at the assumed MFU.
- **Tokens per step**: 2^20 tokens (512 sequences) is reachable at every size by gradient accumulation; 0.5M tokens/step halves the accumulation counts. Whether 0.5M or 1M is used must be fixed per size across all cells (the critical batch size grows with model size; recipe workstream).
- **Base pretraining time at 20 tokens / total parameter** (P5): 0.5 h, 4 h, 17 h, 58 h, 199 h on one PRO 6000 (279 GPU-hours for the family), or 2.3 h, 18 h, 78 h for 25M–250M on a 5080. Per 10^9 synthetic student tokens: 1.0 h, 2.0 h, 3.5 h, 5.8 h, 9.9 h on the PRO 6000 — so with 8 arms x 3 seeds x 0.5 B tokens the synthetic phase costs roughly as much GPU time as base pretraining (Section 10, issue 5).
- Inference (KV cache) is not a constraint: the largest KV cache is 78 KB per token (1B), 160 MB per 2048-token sequence.

## 9. Config blocks (YAML)

```yaml
# student_25M.yaml  (target total 25.00M; actual 24.88M, -0.5%)
model:
  name: 25M
  arch: llama                    # RMSNorm pre-norm, RoPE, SwiGLU, GQA, no biases
  n_layers: 8
  d_model: 384
  n_heads: 6
  n_kv_heads: 2
  head_dim: 64
  d_ff: 1024                  # = 8/3 * d_model exactly (d_model is a multiple of 192)
  vocab_size: 32000
  tie_embeddings: true
  ctx_len: 2048
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  activation: silu               # SwiGLU
  bias: false
  dropout: 0.0
  init:
    std: 0.02                    # N(0, 0.02) for all matrices and the embedding
    residual_proj_scale: 0.25000   # Wo, W2 std = 0.02/sqrt(2*n_layers)
    norm_gain: 1.0
  precision: bf16-mixed          # bf16 autocast, fp32 master weights / grads / Adam moments
  attention_impl: flash          # causal, GQA-aware
  kv_cache: true
derived:
  total_params: 24877440
  non_embedding_params: 12589440
  embedding_params: 12288000
  embed_fraction: 0.494
  aspect_ratio_d_over_L: 48.0
  train_flops_per_token_causal: 1.870e+08
  kv_cache_bytes_per_token_bf16: 4096
```

```yaml
# student_100M.yaml  (target total 100.00M; actual 100.09M, +0.1%)
model:
  name: 100M
  arch: llama                    # RMSNorm pre-norm, RoPE, SwiGLU, GQA, no biases
  n_layers: 12
  d_model: 768
  n_heads: 12
  n_kv_heads: 4
  head_dim: 64
  d_ff: 2048                  # = 8/3 * d_model exactly (d_model is a multiple of 192)
  vocab_size: 32000
  tie_embeddings: true
  ctx_len: 2048
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  activation: silu               # SwiGLU
  bias: false
  dropout: 0.0
  init:
    std: 0.02                    # N(0, 0.02) for all matrices and the embedding
    residual_proj_scale: 0.20412   # Wo, W2 std = 0.02/sqrt(2*n_layers)
    norm_gain: 1.0
  precision: bf16-mixed          # bf16 autocast, fp32 master weights / grads / Adam moments
  attention_impl: flash          # causal, GQA-aware
  kv_cache: true
derived:
  total_params: 100092672
  non_embedding_params: 75516672
  embedding_params: 24576000
  embed_fraction: 0.246
  aspect_ratio_d_over_L: 64.0
  train_flops_per_token_causal: 7.138e+08
  kv_cache_bytes_per_token_bf16: 12288
```

```yaml
# student_250M.yaml  (target total 250.00M; actual 247.03M, -1.2%)
model:
  name: 250M
  arch: llama                    # RMSNorm pre-norm, RoPE, SwiGLU, GQA, no biases
  n_layers: 22
  d_model: 960
  n_heads: 15
  n_kv_heads: 5
  head_dim: 64
  d_ff: 2560                  # = 8/3 * d_model exactly (d_model is a multiple of 192)
  vocab_size: 32000
  tie_embeddings: true
  ctx_len: 2048
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  activation: silu               # SwiGLU
  bias: false
  dropout: 0.0
  init:
    std: 0.02                    # N(0, 0.02) for all matrices and the embedding
    residual_proj_scale: 0.15076   # Wo, W2 std = 0.02/sqrt(2*n_layers)
    norm_gain: 1.0
  precision: bf16-mixed          # bf16 autocast, fp32 master weights / grads / Adam moments
  attention_impl: flash          # causal, GQA-aware
  kv_cache: true
derived:
  total_params: 247032000
  non_embedding_params: 216312000
  embedding_params: 30720000
  embed_fraction: 0.124
  aspect_ratio_d_over_L: 43.6
  train_flops_per_token_causal: 1.742e+09
  kv_cache_bytes_per_token_bf16: 28160
```

```yaml
# student_500M.yaml  (target total 500.00M; actual 505.50M, +1.1%)
model:
  name: 500M
  arch: llama                    # RMSNorm pre-norm, RoPE, SwiGLU, GQA, no biases
  n_layers: 24
  d_model: 1344
  n_heads: 21
  n_kv_heads: 7
  head_dim: 64
  d_ff: 3584                  # = 8/3 * d_model exactly (d_model is a multiple of 192)
  vocab_size: 32000
  tie_embeddings: true
  ctx_len: 2048
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  activation: silu               # SwiGLU
  bias: false
  dropout: 0.0
  init:
    std: 0.02                    # N(0, 0.02) for all matrices and the embedding
    residual_proj_scale: 0.14434   # Wo, W2 std = 0.02/sqrt(2*n_layers)
    norm_gain: 1.0
  precision: bf16-mixed          # bf16 autocast, fp32 master weights / grads / Adam moments
  attention_impl: flash          # causal, GQA-aware
  kv_cache: true
derived:
  total_params: 505495872
  non_embedding_params: 462487872
  embedding_params: 43008000
  embed_fraction: 0.085
  aspect_ratio_d_over_L: 56.0
  train_flops_per_token_causal: 3.429e+09
  kv_cache_bytes_per_token_bf16: 43008
```

```yaml
# student_1B.yaml  (target total 1.000B; actual 1.006B, +0.6%)
model:
  name: 1B
  arch: llama                    # RMSNorm pre-norm, RoPE, SwiGLU, GQA, no biases
  n_layers: 38
  d_model: 1536
  n_heads: 24
  n_kv_heads: 8
  head_dim: 64
  d_ff: 4096                  # = 8/3 * d_model exactly (d_model is a multiple of 192)
  vocab_size: 32000
  tie_embeddings: true
  ctx_len: 2048
  rope_theta: 10000.0
  rms_norm_eps: 1.0e-5
  activation: silu               # SwiGLU
  bias: false
  dropout: 0.0
  init:
    std: 0.02                    # N(0, 0.02) for all matrices and the embedding
    residual_proj_scale: 0.11471   # Wo, W2 std = 0.02/sqrt(2*n_layers)
    norm_gain: 1.0
  precision: bf16-mixed          # bf16 autocast, fp32 master weights / grads / Adam moments
  attention_impl: flash          # causal, GQA-aware
  kv_cache: true
derived:
  total_params: 1005571584
  non_embedding_params: 956419584
  embedding_params: 49152000
  embed_fraction: 0.049
  aspect_ratio_d_over_L: 40.4
  train_flops_per_token_causal: 6.751e+09
  kv_cache_bytes_per_token_bf16: 77824
```

## Open issues and disagreements with the brief

1. **The "25M" model is 49 % embeddings (12.6M non-embedding parameters) under every feasible mainstream tokenizer.** P4 asks that it not be "dominated" by embeddings; 49 % is borderline. Options, in order of my preference: (a) keep 32k and report non-embedding counts alongside totals everywhere (done here; Pythia-70M at 73 % and GPT-2-small at 32 % show this is normal at the small end); (b) adopt a custom 16k byte-level BPE trained on the base corpus (fertility equal to Llama-2-32k on the proxy, embedding share 25 %, non-embedding +50 %); (c) redefine targets in non-embedding parameters. The choice interacts with P5: "20 tokens per parameter" buys 500M tokens for the 25M model if counted on totals but 252M if counted on non-embedding parameters — the budget workstream must state which N it uses.
2. **Non-embedding sizes are not log-uniformly spaced.** The family's non-embedding counts step by 6.0x, 2.9x, 2.1x, 2.1x. If H4 (T* as a function of S) is to be fitted, a log-uniform S grid (e.g. non-embedding 12.6M, 32M, 80M, 200M, 500M; or adding a 50M model) is statistically better than the round-number totals in the brief. I did not change the targets because the brief fixes them.
3. **Tokenizer alternative.** The proxy measurement shows the Llama-2 tokenizer is ~10 % less efficient on English than a same-size English BPE; if the lead prefers efficiency over off-the-shelf reproducibility, a custom 32k (or 16k, see issue 1) BPE is the alternative; the tokenizer must be frozen before any teacher generation is counted, and `fertility.py` must be re-run on FineWeb-Edu and on real teacher outputs (the 1.15 teacher-to-student token conversion is an estimate).
4. **Depth/width.** I chose a middle aspect band ([32, 64]); MobileLLM's evidence favours d/L about 20–30 at 100M–350M (roughly 1–2 accuracy points). The alternatives are documented (23 x 576 at 100M, 36 x 768 at 250M, 33 x 1152 at 500M); switching costs throughput, and the 25M model cannot follow (no in-tolerance thin shape exists). The disagreement is with "deep-and-thin everywhere", not with the brief.
5. **Compute realism.** At the assumed MFU the base runs cost 279 PRO-6000-hours and each 10^9 synthetic tokens at 1B costs 10 hours; 30 cells x 3 seeds x (8 arms) at D_syn = 0.5–1 B will exceed the base-pretraining cost and is dominated by the 500M/1B rows. The pilot should measure MFU and tokens/s before the full grid is committed; the GPU peak numbers are unverified.
6. **"GQA where appropriate" (brief) vs GQA everywhere (here).** I argue a single rule is required for a clean S axis, and 3:1 is the only ratio feasible at every size; if the lead insists on MHA at the small end, the MHA row of the alternatives table gives 7 x 384 / 16 x 640 / 20 x 960 / 26 x 1216 / 31 x 1600.
7. **Stability extras.** No QK-norm or z-loss by default (mainstream Llama-2 block). Small models trained at high learning rates can show loss spikes (Wortsman et al. 2023); if the pilot shows instability, add QK-LayerNorm to *all* sizes rather than to the affected size only.
8. **Packing / cross-example attention** in the synthetic phase (Section 7) and **document-boundary handling** in the base phase must be fixed identically for C0–C7; the brief does not mention this control.
9. **Sandbox limitations.** Hugging Face, arXiv and most publisher hosts were blocked, so model-card dimensions for MobileLLM-600M/1B, the exact MobileLLM/Tao et al. figures and the GPU datasheets are marked UNVERIFIED; everything else is arithmetic reproducible from the code.
