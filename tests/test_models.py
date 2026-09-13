"""nn.Module correctness: parameter counts, init, masking, RoPE, KV cache."""
import math

import pytest

torch = pytest.importorskip("torch")

from synscale.config.schemas import StudentConfig  # noqa: E402
from synscale.models import llama_like as ll  # noqa: E402
from synscale.models.param_count import count_params, locked_students  # noqa: E402


def _toy(**kw) -> StudentConfig:
    base = dict(name="toy", target_params=1, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                d_ff=128, vocab_size=100, context_length=64, tie_embeddings=True)
    base.update(kw)
    return StudentConfig(**base)


@pytest.mark.parametrize("name", ["s025m", "s100m", "s250m", "s500m", "s1b"])
def test_module_matches_closed_form(name):
    cfg = locked_students()[name]
    model = ll.build_student(cfg, "meta")
    assert ll.count_parameters(model) == count_params(cfg)["total"]


@pytest.mark.parametrize("tie", [True, False])
def test_small_shapes_tied_and_untied(tie):
    cfg = _toy(tie_embeddings=tie)
    assert ll.count_parameters(ll.LlamaLike(cfg)) == count_params(cfg)["total"]


def test_initialisation_scales():
    torch.manual_seed(0)
    cfg = _toy(n_layers=2, d_model=512, n_heads=8, n_kv_heads=4, head_dim=64, d_ff=1024, vocab_size=2000)
    model = ll.LlamaLike(cfg)
    residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    for layer in model.layers:
        assert abs(layer.attn.wo.weight.std().item() - residual_std) < 0.1 * residual_std
        assert abs(layer.ffn.w2.weight.std().item() - residual_std) < 0.1 * residual_std
        assert abs(layer.attn.wq.weight.std().item() - 0.02) < 0.002
        assert abs(layer.ffn.w1.weight.std().item() - 0.02) < 0.002
        assert torch.all(layer.attn_norm.weight == 1)
    assert abs(model.tok_emb.weight.std().item() - 0.02) < 0.002
    assert model.tok_emb.weight.abs().max().item() <= 0.06 + 1e-6  # truncated at 3σ
    assert model.lm_head.weight is model.tok_emb.weight
    plain = ll.LlamaLike(_toy(depth_scaled_init=False, d_model=512, n_heads=8, n_kv_heads=4, head_dim=64, d_ff=1024, vocab_size=2000))
    assert abs(plain.layers[0].attn.wo.weight.std().item() - 0.02) < 0.002


def test_document_helpers():
    cu = torch.tensor([0, 5, 16, 23, 32])
    doc = ll.document_ids_from_cu_seqlens(cu, 2, 16)
    assert doc.tolist() == [[0] * 5 + [1] * 11, [2] * 7 + [3] * 9]
    pos = ll.position_ids_from_doc_ids(doc)
    assert pos.tolist() == [list(range(5)) + list(range(11)), list(range(7)) + list(range(9))]
    mask = ll.block_causal_mask(doc)
    assert mask.shape == (2, 1, 16, 16)
    assert mask[0, 0, 6, 5] and not mask[0, 0, 6, 4] and not mask[0, 0, 5, 6]


def test_intra_document_mask_isolates_documents():
    torch.manual_seed(0)
    model = ll.LlamaLike(_toy()).eval()
    ids = torch.randint(0, 100, (2, 16))
    cu = torch.tensor([0, 5, 16, 23, 32])
    doc = ll.document_ids_from_cu_seqlens(cu, 2, 16)
    with torch.no_grad():
        packed = model(ids, position_ids=ll.position_ids_from_doc_ids(doc), cu_seqlens=cu)
        alone = model(ids[1:2, 7:16])
        plain = model(ids)
    assert torch.allclose(packed[1, 7:16], alone[0], atol=1e-5)
    assert not torch.allclose(plain[1, 7:16], alone[0], atol=1e-3)  # without the mask the prefix leaks in


def test_rope_is_relative():
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(pq, pk):
        cq, sq = ll.rope_cos_sin(torch.tensor([[pq]]), 16, 10_000.0, torch.float32)
        ck, sk = ll.rope_cos_sin(torch.tensor([[pk]]), 16, 10_000.0, torch.float32)
        return (ll.apply_rope(q, cq, sq) * ll.apply_rope(k, ck, sk)).sum()

    assert torch.allclose(score(3, 1), score(10, 8), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)


def test_kv_cache_generation_matches_full_forward():
    torch.manual_seed(0)
    model = ll.LlamaLike(_toy()).eval()
    prompt = torch.randint(0, 100, (2, 8))
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (2, 14)
    with torch.no_grad():
        full = model(out[:, :-1]).argmax(-1)
    assert torch.equal(full[:, 7:], out[:, 8:])


def test_sampling_and_eos():
    torch.manual_seed(0)
    model = ll.LlamaLike(_toy()).eval()
    prompt = torch.randint(0, 100, (2, 4))
    a = model.generate(prompt, 5, temperature=0.8, top_k=10, top_p=0.9, generator=torch.Generator().manual_seed(1))
    b = model.generate(prompt, 5, temperature=0.8, top_k=10, top_p=0.9, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and a.shape == (2, 9)
    eos = int(model.generate(prompt, 1)[0, -1])
    stopped = model.generate(prompt, 5, eos_id=eos)
    assert stopped.shape[1] <= 9 and int(stopped[0, 4]) == eos
    with pytest.raises(ValueError):
        model.generate(prompt, 100)
