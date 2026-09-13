"""Closed-form parameter/FLOP accounting: hand computation, published totals, locked family."""
import pytest

from synscale.config.schemas import StudentConfig
from synscale.models import param_count as pc


def _cfg(n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab, tie=True, ctx=2048):
    return StudentConfig(
        name="x", target_params=1, n_layers=n_layers, d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, d_ff=d_ff, vocab_size=vocab, context_length=ctx, tie_embeddings=tie,
    )


def test_hand_computed_example():
    # L=2, d=64, 4 heads of 16, 2 kv heads, d_ff=128, V=100, tied
    p = pc.count_params(_cfg(2, 64, 4, 2, 16, 128, 100))
    assert p["attn_per_layer"] == 12288  # Q 4096 + K,V 4096 + O 4096
    assert p["ffn_per_layer"] == 3 * 64 * 128 == 24576
    assert p["norm_per_layer"] == 128
    assert p["layer"] == 36992
    assert p["total"] == 2 * 36992 + 64 + 6400 == 80448
    assert p["non_embedding"] == 74048
    assert p["embedding"] == 6400 and p["lm_head"] == 0
    untied = pc.count_params(_cfg(2, 64, 4, 2, 16, 128, 100, tie=False))
    assert untied["total"] == 80448 + 6400
    assert untied["non_embedding"] == 74048
    assert untied["embedding"] == 12800


def test_flops_formula():
    c = _cfg(2, 64, 4, 2, 16, 128, 100, ctx=32)
    p = pc.count_params(c)
    f = pc.flops_per_token(c, 32)
    assert f["fwd"] == 2 * (p["non_embedding"] + 100 * 64) + 4 * 2 * 32 * 64
    assert f["train"] == 3 * f["fwd"]
    assert pc.flops_per_token(c, 32, causal_half=True)["fwd"] == 2 * (p["non_embedding"] + 100 * 64) + 2 * 2 * 32 * 64
    assert pc.flops_per_token(c, 32, checkpointing=True)["train"] == 4 * f["fwd"]
    assert f["chinchilla_6N_total"] == 6 * p["total"]
    assert f["kaplan_6N_nonemb"] == 6 * p["non_embedding"]


def test_reference_models_reproduced():
    assert pc.count_params_gpt2_style(12, 768, 50257, tied=True, learned_pos_ctx=1024) == 124_439_808
    assert pc.count_params_gpt2_style(6, 512, 50304, tied=False) == 70_426_624
    smol = pc.count_params(_cfg(30, 576, 9, 3, 64, 1536, 49152))["total"]
    assert abs(smol - 134.5e6) / 134.5e6 < 0.005
    tiny = pc.count_params(_cfg(22, 2048, 32, 4, 64, 5632, 32000, tie=False))["total"]
    assert abs(tiny - 1.1e9) / 1.1e9 < 0.01


def test_design_rules():
    assert pc.d_ff_llama(4096, 256) == 11008  # LLaMA-7B
    assert pc.d_ff_llama(576, 64) == 1536  # SmolLM-135M
    assert pc.kv_heads_for(9, 3) == 3
    assert pc.kv_heads_for(8, 3) is None
    assert pc.kv_heads_for(8, 1) == 8


LOCKED = {
    "s025m": (8, 384, 6, 2, 1024, 24_877_440, 12_589_440),
    "s100m": (12, 768, 12, 4, 2048, 100_092_672, 75_516_672),
    "s250m": (22, 960, 15, 5, 2560, 247_032_000, 216_312_000),
    "s500m": (24, 1344, 21, 7, 3584, 505_495_872, 462_487_872),
    "s1b": (38, 1536, 24, 8, 4096, 1_005_571_584, 956_419_584),
}


@pytest.mark.parametrize("name", list(LOCKED))
def test_locked_family_exact(name):
    n_layers, d, nh, nkv, dff, total, non_emb = LOCKED[name]
    cfg = pc.locked_students()[name]
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff) == (n_layers, d, nh, nkv, dff)
    assert cfg.vocab_size == 32000 and cfg.tie_embeddings and cfg.context_length == 2048
    assert cfg.head_dim == 64 and cfg.rope_theta == 10_000.0
    assert cfg.d_ff * 3 == 8 * cfg.d_model  # d_ff = 8/3 d exactly
    p = pc.count_params(cfg)
    assert p["total"] == total == pc.LOCKED_TOTALS[name]
    assert p["non_embedding"] == non_emb
    assert abs(p["total"] / cfg.target_params - 1) <= 0.012


@pytest.mark.parametrize("name", list(LOCKED))
def test_search_reproduces_locked_shapes(name):
    cfg = pc.locked_students()[name]
    best, cands = pc.search_shapes(cfg.target_params, 32000)
    assert best is not None and best["in_band"] and abs(best["rel_err"]) <= 0.10
    assert (best["config"].n_layers, best["config"].d_model) == (cfg.n_layers, cfg.d_model)
    for x in cands:
        c = x["config"]
        assert c.n_heads * c.head_dim == c.d_model and c.n_heads % c.n_kv_heads == 0


def test_training_plan_fit():
    fam = pc.locked_students()
    small = pc.training_plan(fam["s025m"], pc.GPUS["RTX 5080 (16 GB)"], mfu=0.10)
    assert small[0]["fits"] and small[0]["accum"] >= 1
    big = pc.training_plan(fam["s1b"], pc.GPUS["RTX 5080 (16 GB)"], mfu=0.38)
    assert not big[0]["fits"]
    assert pc.optimizer_state_bytes(fam["s1b"]) == 16 * pc.LOCKED_TOTALS["s1b"]
    assert pc.peak_tflops_for_device("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == 500.0
    assert pc.peak_tflops_for_device("unknown") is None
