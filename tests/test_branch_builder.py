"""Data formats, ordering, tokenisation and the phase-2 branch builder (no torch needed)."""
import json

import numpy as np
import pytest

from synscale.training import data as D
from synscale.training.schedules import LRSchedule, schedule_from_phase


class StubTokenizer:
    """Character-level tokenizer with ids in [3, 130); 1 = BOS, 2 = EOS."""

    def encode(self, text: str) -> list[int]:
        return [3 + (ord(c) % 127) for c in text]


def _write_docs(stem, n_docs, seed, min_len=4, max_len=30):
    rng = np.random.default_rng(seed)
    docs = [np.concatenate([[1], rng.integers(3, 130, int(rng.integers(min_len, max_len))), [2]]).astype(np.uint16) for _ in range(n_docs)]
    starts = np.cumsum([0] + [len(d) for d in docs[:-1]])
    D.write_document_file(stem, np.concatenate(docs), starts)
    return D.SequenceSource.from_documents(stem, 16)


# ----------------------------------------------------------------------------- rendering / tokenisation


def test_render_and_encode_document():
    text = D.render_instruction("What is 2+2?", "4.")
    assert text == "### Instruction:\nWhat is 2+2?\n\n### Response:\n4."
    ids = D.encode_document(StubTokenizer(), text, bos_id=1, eos_id=2)
    assert ids[0] == 1 and ids[-1] == 2 and len(ids) == len(text) + 2
    rendered = list(D.render_instruction_examples([{"prompt": "p", "response": "r"}]))
    assert rendered == ["### Instruction:\np\n\n### Response:\nr"]


def test_tokenize_documents_layout(tmp_path):
    tokens, starts = D.tokenize_documents(["ab", "cde"], StubTokenizer(), 1, 2)
    assert tokens.dtype == np.uint16 and starts.tolist() == [0, 4]
    assert tokens.tolist()[:4] == [1, 3 + ord("a") % 127, 3 + ord("b") % 127, 2]
    meta = D.write_document_file(tmp_path / "d", tokens, starts)
    assert meta["n_docs"] == 2 and meta["n_tokens"] == 9
    src = D.SequenceSource.open(tmp_path / "d", 4)
    assert len(src) == 2  # (9 - 1) // 4
    assert src[0].tokens.tolist() == tokens[:5].tolist() and src[0].cu_seqlens.tolist() == [0, 4]
    assert src[1].cu_seqlens.tolist() == [0, 4]  # doc 2 starts at position 4 == row start → no interior cut
    assert D.position_ids_from_cu_seqlens(np.array([0, 3, 4]), 4).tolist() == [0, 1, 2, 0]


# ----------------------------------------------------------------------------- sources, collate, order


def test_collate_offsets_and_positions(tmp_path):
    src = _write_docs(tmp_path / "c", 40, 0)
    batch = D.collate([src[0], src[1]], 16)
    assert batch.input_ids.shape == (2, 16) and batch.targets.shape == (2, 16)
    assert np.array_equal(batch.targets[:, :-1], batch.input_ids[:, 1:])
    assert batch.cu_seqlens[0] == 0 and batch.cu_seqlens[-1] == 32 and np.all(np.diff(batch.cu_seqlens) > 0)
    assert 16 in batch.cu_seqlens.tolist()  # row boundary is always a segment boundary
    assert batch.n_tokens == 32
    for row in range(2):
        cu = src[row].cu_seqlens
        assert np.array_equal(batch.position_ids[row], D.position_ids_from_cu_seqlens(cu, 16))


def test_sharded_source(tmp_path):
    a = _write_docs(tmp_path / "a", 30, 1)
    b = _write_docs(tmp_path / "b", 30, 2)
    s = D.ShardedSource([a, b])
    assert len(s) == len(a) + len(b)
    assert np.array_equal(s[len(a)].tokens, b[0].tokens)
    assert np.array_equal(s[len(a) - 1].tokens, a[len(a) - 1].tokens)
    with pytest.raises(IndexError):
        s[len(s)]


def test_data_order_is_stateless_and_seeded():
    order = D.DataOrder(100, seed=7)
    first = order.indices(0, 100)
    assert sorted(first) == list(range(100))
    assert order.index(37) == first[37]  # same answer regardless of call history
    assert order.indices(100, 100) != first  # epoch 2 is a different permutation
    assert D.DataOrder(100, seed=7).indices(0, 10) == first[:10]
    assert D.DataOrder(100, seed=8).indices(0, 100) != first
    assert D.DataOrder(100, seed=7, shuffle=False).indices(98, 4) == [98, 99, 0, 1]


# ----------------------------------------------------------------------------- branch builder


def _spec(seed, size="s025m"):
    return D.BranchSpec(size=size, seed=seed, seq_len=16, d2_tokens=16 * 40, replay_fraction=0.25)


def _rows(stem):
    src = D.SequenceSource.from_packed(stem)
    return src, [src[i] for i in range(len(src))]


def test_branch_counts_and_membership(tmp_path):
    treat = _write_docs(tmp_path / "treat", 200, 10)
    replay = _write_docs(tmp_path / "rep", 200, 11)
    spec = _spec(seed=1)
    assert (spec.n_slots, spec.n_treatment, spec.n_replay) == (40, 30, 10)
    branch = D.build_branch(treat, replay, spec, tmp_path / "b1", "treat", "rep")
    assert (branch["n_slots"], branch["n_treatment"], branch["n_replay"]) == (40, 30, 10)
    assert branch["treatment_tokens"] == 30 * 16
    src, rows = _rows(tmp_path / "b1")
    assert len(src) == 40 and src.seq_len == 16
    slots = np.load(tmp_path / "b1.slots.npy")
    assert slots.shape == (40,) and int((slots == D.SLOT_REPLAY).sum()) == 10
    treat_rows = {treat[i].tokens.tobytes() for i in range(30)}
    rep_rows = {replay[i].tokens.tobytes() for i in range(10)}
    placed_treat = {rows[k].tokens.tobytes() for k in range(40) if slots[k] == D.SLOT_TREAT}
    placed_rep = {rows[k].tokens.tobytes() for k in range(40) if slots[k] == D.SLOT_REPLAY}
    assert placed_treat == treat_rows and placed_rep == rep_rows  # exactly the first n rows, all used once
    # document boundaries survive the move
    for k in range(40):
        assert rows[k].cu_seqlens[0] == 0 and rows[k].cu_seqlens[-1] == 16
    meta = json.loads((tmp_path / "b1.json").read_text())
    assert meta["format"] == "packed" and meta["sha256"] == branch["sha256"]


def test_slot_pattern_shared_across_conditions_and_seeded(tmp_path):
    treat_a = _write_docs(tmp_path / "ta", 200, 20)
    treat_b = _write_docs(tmp_path / "tb", 200, 21)
    replay = _write_docs(tmp_path / "rep", 200, 22)
    a = D.build_branch(treat_a, replay, _spec(seed=3), tmp_path / "a", "ta", "rep")
    b = D.build_branch(treat_b, replay, _spec(seed=3), tmp_path / "b", "tb", "rep")
    assert a["slots_sha256"] == b["slots_sha256"]
    slots = np.load(tmp_path / "a.slots.npy")
    assert np.array_equal(slots, np.load(tmp_path / "b.slots.npy"))
    _, rows_a = _rows(tmp_path / "a")
    _, rows_b = _rows(tmp_path / "b")
    for k in np.flatnonzero(slots == D.SLOT_REPLAY):  # replay rows land in the same slots with the same content
        assert np.array_equal(rows_a[k].tokens, rows_b[k].tokens)
    assert a["sha256"] != b["sha256"]  # the treatment file is the only difference

    c = D.build_branch(treat_a, replay, _spec(seed=4), tmp_path / "c", "ta", "rep")
    assert c["slots_sha256"] != a["slots_sha256"]
    _, rows_c = _rows(tmp_path / "c")
    assert any(not np.array_equal(rows_a[k].tokens, rows_c[k].tokens) for k in range(40))
    # same pattern, different seed for placement only: reuse a's slots with seed 4
    d = D.build_branch(treat_a, replay, _spec(seed=4), tmp_path / "d", "ta", "rep", slots=slots)
    _, rows_d = _rows(tmp_path / "d")
    treat_slots = np.flatnonzero(slots == D.SLOT_TREAT)
    assert any(not np.array_equal(rows_a[k].tokens, rows_d[k].tokens) for k in treat_slots)


def test_branch_sha_reproducible(tmp_path):
    treat = _write_docs(tmp_path / "t", 200, 30)
    replay = _write_docs(tmp_path / "r", 200, 31)
    a = D.build_branch(treat, replay, _spec(seed=5), tmp_path / "x" / "a", "t", "r")
    b = D.build_branch(treat, replay, _spec(seed=5), tmp_path / "y" / "b", "t", "r")
    for key in ("tokens_sha256", "bounds_sha256", "sha256", "slots_sha256"):
        assert a[key] == b[key]
    assert a["tokens_sha256"] == D.sha256_file(tmp_path / "x" / "a.tokens.bin")


def test_branch_shortfall_is_an_error(tmp_path):
    treat = _write_docs(tmp_path / "t", 10, 40)  # far fewer than 30 sequences
    replay = _write_docs(tmp_path / "r", 200, 41)
    with pytest.raises(ValueError, match="treatment has"):
        D.build_branch(treat, replay, _spec(seed=1), tmp_path / "b", "t", "r")
    with pytest.raises(ValueError, match="multiple of seq_len"):
        D.BranchSpec(size="s", seed=1, seq_len=16, d2_tokens=100, replay_fraction=0.25).n_slots


# ----------------------------------------------------------------------------- schedules (torch-free)


def test_wsd_schedule_shape():
    s = LRSchedule(kind="wsd", lr_peak=1e-3, lr_min=0.0, warmup_steps=10, total_steps=100, decay_steps=0)
    c = s.curve()
    assert c.shape == (100,)
    assert c[0] == pytest.approx(1e-4) and c[9] == pytest.approx(1e-3)
    assert np.all(c[10:] == 1e-3)  # base phase: constant after warmup
    p2 = LRSchedule(kind="wsd", lr_peak=1e-3, lr_min=0.0, warmup_steps=0, total_steps=50, decay_steps=50)
    c2 = p2.curve()
    assert c2[0] == 1e-3 and np.all(np.diff(c2) < 0) and c2[-1] == pytest.approx(1e-3 / 50)
    assert p2.lr(50) == 0.0  # reaches zero one step after the last update
    partial = LRSchedule(kind="wsd", lr_peak=1.0, lr_min=0.1, warmup_steps=5, total_steps=100, decay_steps=20)
    cp = partial.curve()
    assert np.all(cp[5:80] == 1.0) and cp[80] == 1.0 and cp[99] == pytest.approx(1.0 - 0.9 * 19 / 20)
    with pytest.raises(ValueError):
        LRSchedule(kind="wsd", lr_peak=1.0, lr_min=0.0, warmup_steps=10, total_steps=20, decay_steps=15)


def test_cosine_schedule_shape():
    s = LRSchedule(kind="cosine", lr_peak=1.0, lr_min=0.1, warmup_steps=10, total_steps=110)
    c = s.curve()
    assert c[9] == pytest.approx(1.0) and c[10] == pytest.approx(1.0)
    assert c[60] == pytest.approx(0.55) and c[-1] > 0.1 and np.all(np.diff(c[10:]) <= 0)
    assert s.lr(110) == pytest.approx(0.1)


def test_schedule_from_phase_conftest_phase(train_cfg):
    base = schedule_from_phase(train_cfg.base_phase)
    assert base.total_steps == 2_000_000_000 // 524_288
    assert base.warmup_steps == 100_000_000 // 524_288
    assert base.decay_steps == 400_000_000 // 524_288 and base.lr_min == pytest.approx(3e-4)
    assert base.lr(base.total_steps - 1) < 3e-3
