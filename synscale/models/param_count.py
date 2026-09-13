"""Exact parameter, FLOP and memory accounting for the LLaMA-like student family.

Architecture assumed (per layer, pre-norm; see docs/00_locked_methodology.md §0.5)::

    h = x + W_o · Attn(RoPE(W_q n(x)), RoPE(W_k n(x)), W_v n(x))     n = RMSNorm (gain only)
    y = h + W_down · (SiLU(W_gate n(h)) ⊙ W_up n(h))                  SwiGLU, no biases

followed by a final RMSNorm, a token embedding (V × d) and an LM head that is either
tied to the embedding or a separate V × d matrix.

Exact parameter formulas (d = d_model, d_attn = n_heads·head_dim, d_kv = n_kv_heads·head_dim)::

    attn_layer = d·d_attn [Q] + 2·d·d_kv [K, V] + d_attn·d [O]
    ffn_layer  = 3·d·d_ff
    norm_layer = 2·d
    final_norm = d
    embeddings = V·d (+ V·d if untied)
    total      = L·(attn_layer + ffn_layer + norm_layer) + final_norm + embeddings
    non_emb    = total − embeddings     (excludes the input embedding AND an untied head)

FLOPs per token (T = context length)::

    C_fwd   = 2·(non_emb + V·d) + 4·L·T·d_attn      (weight matmuls incl. LM head; QKᵀ and PV)
    C_train = 3·C_fwd                                (backward ≈ 2× forward)
    C_train = 4·C_fwd with full activation checkpointing

The attention term is the full-T upper bound; a causal kernel that skips masked blocks costs
about half of it (``causal_half=True``). Chinchilla's ``6·N·D`` uses N = total parameters and
drops the attention term; Kaplan's N excludes embeddings. Both conventions are reported.

Public API
----------
``count_params``, ``flops_per_token``, ``search_shapes``, ``locked_students``,
``count_params_gpt2_style`` (cross-check only), the design rules ``d_ff_llama`` and
``kv_heads_for``, and the memory planner ``training_plan``.
"""
from __future__ import annotations

import math
from typing import Any, Protocol

from synscale.config.schemas import StudentConfig


class Shape(Protocol):
    """Structural type: anything carrying the block dimensions (a StudentConfig qualifies)."""

    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    tie_embeddings: bool


# ----------------------------------------------------------------------------- parameters


def attn_params_per_layer(c: Shape) -> int:
    q = c.d_model * c.n_heads * c.head_dim
    kv = 2 * c.d_model * c.n_kv_heads * c.head_dim
    o = c.n_heads * c.head_dim * c.d_model
    return q + kv + o


def ffn_params_per_layer(c: Shape) -> int:
    return 3 * c.d_model * c.d_ff


def norm_params_per_layer(c: Shape) -> int:
    return 2 * c.d_model


def count_params(c: Shape) -> dict[str, Any]:
    """Closed-form parameter counts.

    Returns a dict with ``total``, ``non_embedding``, ``embedding`` (input embedding plus an
    untied head), the per-layer and whole-model breakdown (``attn_per_layer``,
    ``ffn_per_layer``, ``norm_per_layer``, ``layer``, ``attn_total``, ``ffn_total``,
    ``norm_total``, ``embed_in``, ``lm_head``, ``final_norm``), ``embed_fraction`` and the
    bf16 KV-cache footprint per token.
    """
    attn = attn_params_per_layer(c)
    ffn = ffn_params_per_layer(c)
    norms = norm_params_per_layer(c)
    embed_in = c.vocab_size * c.d_model
    lm_head = 0 if c.tie_embeddings else c.vocab_size * c.d_model
    final_norm = c.d_model
    non_emb = c.n_layers * (attn + ffn + norms) + final_norm
    total = non_emb + embed_in + lm_head
    return dict(
        attn_per_layer=attn,
        ffn_per_layer=ffn,
        norm_per_layer=norms,
        layer=attn + ffn + norms,
        attn_total=c.n_layers * attn,
        ffn_total=c.n_layers * ffn,
        norm_total=c.n_layers * norms + final_norm,
        embed_in=embed_in,
        lm_head=lm_head,
        final_norm=final_norm,
        embedding=embed_in + lm_head,
        non_embedding=non_emb,
        total=total,
        embed_fraction=(embed_in + lm_head) / total,
        kv_cache_bytes_per_token_bf16=2 * c.n_layers * c.n_kv_heads * c.head_dim * 2,
    )


def count_params_gpt2_style(
    n_layers: int, d_model: int, vocab_size: int, tied: bool = True, learned_pos_ctx: int | None = None
) -> int:
    """GPT-2 / GPT-NeoX (Pythia) block: LayerNorm (gain+bias) ×2, attention with biases
    (4d² + 4d), GELU MLP 4× with biases (8d² + 5d); final LayerNorm; optional learned
    positional table. Used only to cross-check the formulas against published totals."""
    per_layer = 12 * d_model * d_model + 13 * d_model
    total = n_layers * per_layer + 2 * d_model + vocab_size * d_model * (1 if tied else 2)
    if learned_pos_ctx:
        total += learned_pos_ctx * d_model
    return total


# ----------------------------------------------------------------------------- flops


def flops_per_token(
    c: Shape, ctx: int, causal_half: bool = False, checkpointing: bool = False
) -> dict[str, Any]:
    """Forward and training FLOPs per token at context length ``ctx``.

    Returns ``fwd``, ``train``, ``attn_share_fwd`` and the two literature conventions
    ``chinchilla_6N_total`` and ``kaplan_6N_nonemb``.
    """
    p = count_params(c)
    d_attn = c.n_heads * c.head_dim
    matmul_fwd = 2 * (p["non_embedding"] + c.vocab_size * c.d_model)
    attn_fwd = 4 * c.n_layers * ctx * d_attn
    if causal_half:
        attn_fwd //= 2
    fwd = matmul_fwd + attn_fwd
    train = (4 if checkpointing else 3) * fwd
    return dict(
        fwd=fwd,
        train=train,
        attn_share_fwd=attn_fwd / fwd,
        chinchilla_6N_total=6 * p["total"],
        kaplan_6N_nonemb=6 * p["non_embedding"],
    )


# ----------------------------------------------------------------------------- design rules


def d_ff_llama(d_model: int, multiple_of: int = 64, ratio: float = 8 / 3) -> int:
    """LLaMA rule: hidden = int(ratio·d), rounded UP to a multiple of ``multiple_of``."""
    h = int(ratio * d_model)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def kv_heads_for(n_heads: int, gqa_ratio: int) -> int | None:
    """GQA rule: n_kv = n_heads / gqa_ratio if it divides exactly, else None (shape rejected)."""
    if gqa_ratio <= 1:
        return n_heads
    if n_heads % gqa_ratio:
        return None
    return n_heads // gqa_ratio


def _make_config(
    name: str, target: int, n_layers: int, d_model: int, n_heads: int, n_kv_heads: int,
    head_dim: int, d_ff: int, vocab: int, tie: bool, ctx: int, rope_theta: float = 10_000.0,
) -> StudentConfig:
    return StudentConfig(
        name=name, target_params=target, n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=head_dim, d_ff=d_ff, vocab_size=vocab,
        context_length=ctx, tie_embeddings=tie, rope_theta=rope_theta,
    )


def search_shapes(
    target: float,
    vocab: int,
    tie: bool = True,
    head_dim: int = 64,
    gqa_ratio: int = 3,
    d_multiple: int = 64,
    ffn_multiple: int = 64,
    aspect_band: tuple[float, float] = (32.0, 64.0),
    aspect_target: float = 48.0,
    tol: float = 0.10,
    lam: float = 0.10,
    d_range: tuple[int, int] = (128, 4096),
    layer_range: tuple[int, int] = (2, 96),
    ctx: int = 2048,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Enumerate (L, d) shapes under the family rules and rank them against ``target``.

    Keeps shapes within ±``tol`` of the target TOTAL parameter count and scores them by
    ``|rel_err| + lam·|ln(aspect / aspect_target)|`` where aspect = d/L. The best shape is
    chosen among those inside ``aspect_band`` (falling back to all candidates).

    Returns ``(best, candidates)``; each candidate is a dict with ``config`` (StudentConfig),
    ``total``, ``non_emb``, ``rel_err``, ``aspect``, ``in_band``, ``score``, ``embed_fraction``.
    """
    cands: list[dict[str, Any]] = []
    for d in range(d_range[0], d_range[1] + 1, d_multiple):
        if d % head_dim:
            continue
        nh = d // head_dim
        nkv = kv_heads_for(nh, gqa_ratio)
        if nkv is None:
            continue
        dff = d_ff_llama(d, ffn_multiple)
        for n_layers in range(layer_range[0], layer_range[1] + 1):
            c = _make_config(f"L{n_layers}d{d}", int(target), n_layers, d, nh, nkv, head_dim,
                             dff, vocab, tie, ctx)
            p = count_params(c)
            rel = p["total"] / target - 1
            if abs(rel) > tol:
                continue
            aspect = d / n_layers
            in_band = aspect_band[0] <= aspect <= aspect_band[1]
            score = abs(rel) + lam * abs(math.log(aspect / aspect_target))
            cands.append(dict(config=c, total=p["total"], non_emb=p["non_embedding"], rel_err=rel,
                              aspect=aspect, in_band=in_band, score=score,
                              embed_fraction=p["embed_fraction"]))
    cands.sort(key=lambda x: x["score"])
    pool = [x for x in cands if x["in_band"]] or cands
    best = min(pool, key=lambda x: x["score"]) if pool else None
    return best, cands


# ----------------------------------------------------------------------------- locked family

# (name, target, L, d_model, n_heads, n_kv_heads, d_ff); vocab 32 000, tied, head_dim 64, ctx 2048.
_LOCKED_SHAPES = (
    ("s025m", 25_000_000, 8, 384, 6, 2, 1024),
    ("s100m", 100_000_000, 12, 768, 12, 4, 2048),
    ("s250m", 250_000_000, 22, 960, 15, 5, 2560),
    ("s500m", 500_000_000, 24, 1344, 21, 7, 3584),
    ("s1b", 1_000_000_000, 38, 1536, 24, 8, 4096),
)

LOCKED_TOTALS = {
    "s025m": 24_877_440,
    "s100m": 100_092_672,
    "s250m": 247_032_000,
    "s500m": 505_495_872,
    "s1b": 1_005_571_584,
}


def locked_students(vocab: int = 32_000, ctx: int = 2048, rope_theta: float = 10_000.0) -> dict[str, StudentConfig]:
    """The five locked student configs (doc 00 §0.5) as validated StudentConfig objects."""
    return {
        name: _make_config(name, target, n_layers, d, nh, nkv, 64, dff, vocab, True, ctx, rope_theta)
        for name, target, n_layers, d, nh, nkv, dff in _LOCKED_SHAPES
    }


# ----------------------------------------------------------------------------- memory & throughput

GPUS: dict[str, dict[str, float]] = {
    # peak = dense BF16 tensor-core TFLOPS with FP32 accumulate, as used for training
    # (vendor-sheet figures, unverified; consumer parts run FP32-accumulate at half rate).
    "RTX 5080 (16 GB)": dict(mem_gb=16.0, peak_tflops=110.0, bw_tbs=0.96),
    "RTX PRO 6000 Blackwell (96 GB)": dict(mem_gb=96.0, peak_tflops=500.0, bw_tbs=1.79),
}

# Model-FLOPs-utilisation assumptions keyed by nominal target; to be replaced by pilot measurements.
MFU_BY_TARGET: dict[float, float] = {25e6: 0.10, 100e6: 0.20, 250e6: 0.28, 500e6: 0.33, 1e9: 0.38}


def peak_tflops_for_device(device_name: str) -> float | None:
    """Best-effort lookup of the dense bf16 peak for a GPU name as reported by torch."""
    lowered = device_name.lower()
    if "5080" in lowered:
        return GPUS["RTX 5080 (16 GB)"]["peak_tflops"]
    if "pro 6000" in lowered or "rtx 6000" in lowered:
        return GPUS["RTX PRO 6000 Blackwell (96 GB)"]["peak_tflops"]
    return None


def optimizer_state_bytes(c: Shape, mode: str = "amp_adamw") -> int:
    """Bytes of persistent training state per model: weights + grads + Adam moments."""
    n = count_params(c)["total"]
    if mode == "amp_adamw":  # fp32 weights 4 + fp32 grads 4 + Adam m, v 8 (bf16 autocast compute)
        return 16 * n
    if mode == "adamw8bit":  # fp32 weights 4 + fp32 grads 4 + 8-bit m, v 2
        return 10 * n
    raise ValueError(f"unknown optimizer mode {mode!r}")


def activation_bytes_per_token(c: Shape, checkpointing: bool = False, safety: float = 1.25) -> float:
    """bf16 activations saved for backward with a FlashAttention-style kernel (no T×T scores).

    Per layer: residual in, norm out, Q, K, V, attn out, attn-proj out, ffn-norm out, gate, up,
    silu(gate), silu·up, down out → 5d + 2·d_attn + 2·d_kv + 4·d_ff elements, plus fp32
    log-sum-exp per head and RMSNorm statistics. ``safety`` covers temporaries/fragmentation.
    """
    d_attn = c.n_heads * c.head_dim
    elems = 5 * c.d_model + 2 * d_attn + 2 * c.n_kv_heads * c.head_dim + 4 * c.d_ff
    per_layer = (2 * elems + 4 * c.n_heads + 8) * safety
    if checkpointing:  # only layer inputs kept + one layer live during recompute
        return c.n_layers * 2 * c.d_model * safety + per_layer
    return c.n_layers * per_layer


def logits_bytes_per_token(c: Shape, fused_ce: bool = True) -> int:
    """Unfused: bf16 logits (2V) + fp32 softmax (4V) + grad (2V) = 8V bytes/token; a fused or
    chunked linear-cross-entropy keeps one chunk live, counted as 2V."""
    return (2 if fused_ce else 8) * c.vocab_size


def training_plan(
    c: Shape,
    gpu: dict[str, float],
    mfu: float,
    ctx: int = 2048,
    tokens_per_step: int = 2**20,
    opt_mode: str = "amp_adamw",
    overhead_gb: float = 2.0,
    fused_ce: bool = True,
) -> list[dict[str, Any]]:
    """Memory/throughput plan for no-checkpointing and full-checkpointing.

    For each mode: largest micro-batch (sequences of ``ctx`` tokens) that fits, gradient
    accumulation steps for ``tokens_per_step``, tokens/s at the assumed ``mfu`` and the base
    pretraining time at 20 tokens per total parameter.
    """
    rows = []
    states = optimizer_state_bytes(c, opt_mode)
    n_total = count_params(c)["total"]
    for ckpt in (False, True):
        free = gpu["mem_gb"] * 2**30 - states - overhead_gb * 2**30
        per_tok = activation_bytes_per_token(c, ckpt) + logits_bytes_per_token(c, fused_ce)
        max_tokens = free / per_tok if free > 0 else 0
        max_seqs = int(max_tokens // ctx)
        fl = flops_per_token(c, ctx, causal_half=True, checkpointing=ckpt)["train"]
        tps = mfu * gpu["peak_tflops"] * 1e12 / fl
        rows.append(dict(
            checkpointing=ckpt,
            states_gb=states / 2**30,
            act_bytes_per_token=per_tok,
            max_seqs=max_seqs,
            max_tokens=int(max_seqs * ctx),
            accum=(math.ceil(tokens_per_step / (max_seqs * ctx)) if max_seqs else None),
            fits=max_seqs >= 1,
            train_flops_per_token=fl,
            mfu=mfu,
            tokens_per_s=tps,
            hours_20tok_per_param=20 * n_total / tps / 3600,
        ))
    return rows
