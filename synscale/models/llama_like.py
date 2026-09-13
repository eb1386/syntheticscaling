"""Plain-PyTorch LLaMA-like decoder used for every student size.

Block (pre-norm): RMSNorm (fp32 statistics) → GQA attention with RoPE → residual →
RMSNorm → SwiGLU (exact ``d_ff``) → residual. No biases, no dropout, tied embeddings when
``cfg.tie_embeddings``. Parameter counts match ``synscale.models.param_count.count_params``
exactly (tested for the five locked configs).

Attention runs through ``torch.nn.functional.scaled_dot_product_attention``. Two masking
modes are supported:

* plain causal (``cu_seqlens=None``): every token attends to all earlier tokens of its row;
* intra-document (``cu_seqlens`` given): a block-diagonal causal mask restricted to the
  document each token belongs to, built from the flattened cumulative sequence lengths that
  ``synscale.training.data`` produces (same convention as ``flash_attn_varlen_func``).

Positions are given explicitly through ``position_ids`` so that RoPE can be
document-relative (positions restart at 0 for every document in a packed row).

Inference uses a ``KVCache`` (``generate`` supports greedy and temperature/top-k/top-p
sampling for a batch of equal-length prompts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from synscale.config.schemas import StudentConfig

# ----------------------------------------------------------------------------- masks & positions


def document_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    """Map flattened cumulative lengths ``[0, l0, l0+l1, ..., batch·seq_len]`` to per-token
    document ids of shape ``(batch, seq_len)``. Row boundaries must be included in
    ``cu_seqlens`` (the data pipeline guarantees this), so no document spans two rows."""
    positions = torch.arange(batch * seq_len, device=cu_seqlens.device)
    doc = torch.searchsorted(cu_seqlens, positions, right=True) - 1
    return doc.view(batch, seq_len)


def block_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Boolean attention mask ``(batch, 1, T, T)``: True where query i may attend key j,
    i.e. j ≤ i and both tokens lie in the same document."""
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.ones(doc_ids.shape[-1], doc_ids.shape[-1], dtype=torch.bool, device=doc_ids.device).tril()
    return (same_doc & causal).unsqueeze(1)


def position_ids_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """Document-relative positions (restart at 0 at every document start)."""
    batch, seq_len = doc_ids.shape
    ar = torch.arange(seq_len, device=doc_ids.device).expand(batch, seq_len)
    is_start = torch.ones_like(doc_ids, dtype=torch.bool)
    is_start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    start_pos = torch.where(is_start, ar, torch.zeros_like(ar)).cummax(dim=1).values
    return ar - start_pos


# ----------------------------------------------------------------------------- RoPE


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape ``(batch, 1, T, head_dim)`` for the given positions (fp32 math)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim))
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq  # (B, T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype).unsqueeze(1), emb.sin().to(dtype).unsqueeze(1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, T, hd)`` (rotate-half convention)."""
    return x * cos + _rotate_half(x) * sin


# ----------------------------------------------------------------------------- KV cache


@dataclass
class KVCache:
    """Per-layer key/value buffers ``(B, n_kv, max_len, head_dim)`` and the filled length."""

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    length: int = 0

    @classmethod
    def empty(cls, cfg: StudentConfig, batch: int, max_len: int, device: torch.device, dtype: torch.dtype) -> "KVCache":
        shape = (batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        return cls(
            k=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)],
            v=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)],
        )

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store the new keys/values for ``layer`` and return all keys/values seen so far."""
        n_new = k.shape[2]
        end = self.length + n_new
        self.k[layer][:, :, self.length:end] = k
        self.v[layer][:, :, self.length:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n_new: int) -> None:
        self.length += n_new


# ----------------------------------------------------------------------------- modules


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learned gain; statistics computed in fp32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


class Attention(nn.Module):
    """Grouped-query attention with RoPE, backed by scaled_dot_product_attention."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.attention_bias)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        kv_cache: Optional[KVCache],
        layer: int,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.append(layer, k, v)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None and seq_len > 1)
        return self.wo(out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.head_dim))


class SwiGLU(nn.Module):
    """``w2(silu(w1 x) ⊙ w3 x)`` with hidden width exactly ``d_ff``."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.mlp_bias)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.mlp_bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask, kv_cache, layer) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, kv_cache, layer)
        return x + self.ffn(self.ffn_norm(x))


class LlamaLike(nn.Module):
    """Decoder-only LM. ``forward`` returns logits of shape ``(B, T, vocab)``."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.reset_parameters()

    # -- initialisation ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """N(0, init_std) truncated at ±3σ for every matrix and the embedding; the two
        residual-writing projections per layer (attention out-proj, FFN down-proj) use
        std/√(2·n_layers) when ``depth_scaled_init``; norm gains are 1."""
        std = self.cfg.init_std
        residual_std = std / math.sqrt(2 * self.cfg.n_layers) if self.cfg.depth_scaled_init else std
        residual_weights = {id(layer.attn.wo.weight) for layer in self.layers} | {id(layer.ffn.w2.weight) for layer in self.layers}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                s = residual_std if id(module.weight) in residual_weights else std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=s, a=-3 * s, b=3 * s)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.trunc_normal_(self.tok_emb.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    # -- forward ---------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: ``(B, T)`` token ids.
            position_ids: ``(B, T)`` RoPE positions; defaults to ``arange(T)`` (offset by the
                cache length when a ``kv_cache`` is given).
            cu_seqlens: 1-D int tensor of cumulative document lengths over the flattened
                ``B·T`` tokens (``[0, ..., B·T]``, row boundaries included). Enables the
                block-diagonal intra-document mask. Mutually exclusive with ``kv_cache``.
            kv_cache: inference cache; keys/values of this call are appended and the cache
                length is advanced.
        """
        bsz, seq_len = input_ids.shape
        past = kv_cache.length if kv_cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + seq_len, device=input_ids.device).expand(bsz, seq_len)
        x = self.tok_emb(input_ids)
        cos, sin = rope_cos_sin(position_ids, self.cfg.head_dim, self.cfg.rope_theta, x.dtype)
        attn_mask = self._attention_mask(cu_seqlens, bsz, seq_len, past, input_ids.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attn_mask, kv_cache, i)
        if kv_cache is not None:
            kv_cache.advance(seq_len)
        return self.lm_head(self.norm(x))

    @staticmethod
    def _attention_mask(cu_seqlens, bsz: int, seq_len: int, past: int, device) -> Optional[torch.Tensor]:
        if cu_seqlens is not None:
            if past:
                raise ValueError("intra-document masking is a training-time feature; do not combine with kv_cache")
            return block_causal_mask(document_ids_from_cu_seqlens(cu_seqlens, bsz, seq_len))
        if past and seq_len > 1:  # cached prefix + several new tokens: causal with offset
            q_pos = torch.arange(past, past + seq_len, device=device).unsqueeze(-1)
            k_pos = torch.arange(past + seq_len, device=device).unsqueeze(0)
            return (k_pos <= q_pos).view(1, 1, seq_len, past + seq_len)
        return None  # SDPA's is_causal (or a single decode token attending to everything)

    # -- generation ------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Greedy (``temperature == 0``) or sampled continuation of a batch of equal-length
        prompts; returns ``(B, T0 + n_generated)``. Stops early once every row emitted
        ``eos_id`` (rows keep being extended with eos afterwards)."""
        bsz, prompt_len = input_ids.shape
        max_len = prompt_len + max_new_tokens
        if max_len > self.cfg.context_length:
            raise ValueError(f"prompt + max_new_tokens = {max_len} exceeds context {self.cfg.context_length}")
        param = next(self.parameters())
        cache = KVCache.empty(self.cfg, bsz, max_len, param.device, param.dtype)
        out = input_ids
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        step_input = input_ids
        for _ in range(max_new_tokens):
            logits = self(step_input, kv_cache=cache)[:, -1, :].float()
            next_tok = _select_next_token(logits, temperature, top_k, top_p, generator)
            if eos_id is not None:
                next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
                finished |= next_tok == eos_id
            out = torch.cat([out, next_tok.unsqueeze(1)], dim=1)
            step_input = next_tok.unsqueeze(1)
            if eos_id is not None and bool(finished.all()):
                break
        return out


def _select_next_token(logits, temperature, top_k, top_p, generator) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        drop = cum - sorted_logits.softmax(-1) > top_p  # keep the smallest set with mass ≥ top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(1)


# ----------------------------------------------------------------------------- utilities


def count_parameters(model: nn.Module) -> int:
    """Number of parameters, counting tied tensors once (as ``nn.Module.parameters`` does)."""
    return sum(p.numel() for p in model.parameters())


def cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Summed next-token cross-entropy in fp32 (the caller normalises by the token count)."""
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum", ignore_index=ignore_index)


def build_student(cfg: StudentConfig, device: torch.device | str = "cpu") -> LlamaLike:
    """Instantiate a student on ``device`` (use ``'meta'`` to inspect shapes without memory)."""
    with torch.device(device):
        return LlamaLike(cfg)
