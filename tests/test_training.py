"""Trainer: optimizer groups, divergence rule, toy training, checkpoint round trip, run_phase."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from synscale.config.schemas import EvalConfig, ExperimentConfig, PhaseConfig, StudentConfig, TrainingConfig  # noqa: E402
from synscale.models.llama_like import LlamaLike  # noqa: E402
from synscale.training import trainer as T  # noqa: E402
from synscale.training.data import BranchSpec, DataOrder, SequenceSource, build_branch, write_document_file  # noqa: E402
from synscale.training.schedules import schedule_from_phase  # noqa: E402

L = 32
V = 100


def _student() -> StudentConfig:
    return StudentConfig(name="toy", target_params=1, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                         d_ff=128, vocab_size=V, context_length=L, tie_embeddings=True)


def _phase(steps, warmup_steps=0, decay_steps=0, lr_min_ratio=1.0, batch_seqs=8, lr=3e-3) -> PhaseConfig:
    bt = batch_seqs * L
    return PhaseConfig(tokens=steps * bt, batch_tokens=bt, lr_peak=lr, lr_min_ratio=lr_min_ratio, schedule="wsd",
                       warmup_tokens=warmup_steps * bt, decay_tokens=decay_steps * bt, weight_decay=0.1,
                       betas=(0.9, 0.95), loss_mask="all_tokens", packing="packed_masked")


def _write_corpus(stem, n_docs, seed):
    """Zipf-distributed tokens: structured enough for a 30-step loss decrease to be unambiguous."""
    rng = np.random.default_rng(seed)
    probs = 1.0 / np.arange(1, V - 2)
    probs /= probs.sum()
    docs, starts, total = [], [], 0
    for _ in range(n_docs):
        body = rng.choice(np.arange(3, V), size=int(rng.integers(6, 40)), p=probs)
        d = np.concatenate([[1], body, [2]]).astype(np.uint16)
        starts.append(total)
        total += len(d)
        docs.append(d)
    write_document_file(stem, np.concatenate(docs), np.asarray(starts))
    return SequenceSource.from_documents(stem, L)


def _run(tmp_path, model, phase, source, seed=0, micro=4, max_steps=None):
    device = torch.device("cpu")
    opt = T.build_optimizer(model, phase, device)
    return T.PhaseRun(model=model, optimizer=opt, schedule=schedule_from_phase(phase), phase=phase, source=source,
                      order=DataOrder(len(source), seed), seq_len=L, device=device,
                      options=T.RuntimeOptions(micro_batch_seqs=micro, device="cpu", max_steps=max_steps), out_dir=tmp_path / "out")


def test_optimizer_groups_exclude_norms_and_embedding():
    model = LlamaLike(_student())
    phase = _phase(1)
    opt = T.build_optimizer(model, phase, torch.device("cpu"))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"])
    decay_ids = {id(p) for p in decay["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(model.tok_emb.weight) not in decay_ids and id(model.tok_emb.weight) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids
    assert len(decay["params"]) == 2 * 7  # per layer: q, k, v, o, w1, w2, w3
    assert opt.defaults["betas"] == (0.9, 0.95) and opt.defaults["eps"] == 1e-8


def test_divergence_rule():
    s = T.TrainState()
    for step in range(20):
        s.step = step
        T._check_divergence(s, 1.0)
    assert s.status == "running" and s.running_min == 1.0
    for _ in range(T.DIVERGENCE_PATIENCE):
        T._check_divergence(s, 2.5)
    assert s.status == "running"
    T._check_divergence(s, 2.5)
    assert s.status == "failed" and "running minimum" in s.failure_reason
    s2 = T.TrainState()
    T._check_divergence(s2, math.nan)
    assert s2.status == "failed"
    s3 = T.TrainState(running_min=1.0, steps_above=40)
    T._check_divergence(s3, 1.5)  # recovery resets the counter
    assert s3.steps_above == 0


def test_toy_training_decreases_loss_and_matches_schedule(tmp_path):
    torch.manual_seed(0)
    source = _write_corpus(tmp_path / "corpus", 600, seed=1)
    phase = _phase(30, warmup_steps=5, decay_steps=10, lr_min_ratio=0.1)
    model = LlamaLike(_student())
    run = _run(tmp_path, model, phase, source)
    assert run.accumulation == 2 and run.intra_document
    state, metrics = T.train_phase(run, T.TrainState(), "base", 0, tmp_path / "final.pt")
    assert state.status == "finished" and state.step == 30 and state.tokens == 30 * 8 * L
    rows = [json.loads(line) for line in (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) == 30
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.2
    assert losses[0] < math.log(V) + 0.5  # init gives near-uniform predictions
    sched = run.schedule
    assert [r["lr"] for r in rows if "lr" in r] == pytest.approx([sched.lr(s) for s in range(30)])
    assert rows[0]["lr"] == pytest.approx(3e-3 / 5) and rows[19]["lr"] == pytest.approx(3e-3)
    assert rows[29]["lr"] == pytest.approx(3e-3 - 0.9 * 3e-3 * 9 / 10)
    assert (tmp_path / "final.pt").exists() and metrics["checkpoint_sha256"]
    assert metrics["final_train_loss"] == losses[-1]


def test_checkpoint_round_trip_is_bit_identical(tmp_path):
    torch.manual_seed(0)
    source = _write_corpus(tmp_path / "corpus", 400, seed=2)
    phase = _phase(8, warmup_steps=2, decay_steps=4, lr_min_ratio=0.0)
    model = LlamaLike(_student())
    run = _run(tmp_path, model, phase, source, max_steps=5)
    state, _ = T.train_phase(run, T.TrainState(), "base", 0, tmp_path / "ckpt.pt")
    assert state.step == 5
    # continue the original run for one step
    run.options.max_steps = 6
    loss_a, lr_a, _ = T.train_step(run, 5)
    # fresh model + optimizer, restore, take the same step
    torch.manual_seed(123)
    model_b = LlamaLike(_student())
    run_b = _run(tmp_path, model_b, phase, source)
    state_b = T.load_checkpoint(tmp_path / "ckpt.pt", model_b, run_b.optimizer)
    assert state_b.step == 5 and state_b.tokens == state.tokens
    T.apply_phase_hparams(run_b.optimizer, phase)
    loss_b, lr_b, _ = T.train_step(run_b, 5)
    assert loss_a == loss_b and lr_a == lr_b
    for pa, pb in zip(model.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_run_phase_base_then_synthetic(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus", 500, seed=3)
    treat = _write_corpus(tmp_path / "treat", 500, seed=4)
    _write_corpus(tmp_path / "heldout", 100, seed=5)
    student = _student()
    training = TrainingConfig(name="t", tokenizer="tok", tokenizer_sha256="b" * 64, context_length=L,
                              base_phase=_phase(6, warmup_steps=2), synthetic_phase=_phase(4, decay_steps=4, lr_min_ratio=0.0),
                              eval_every_tokens=3 * 8 * L, seed_base=0, seed_phase=1)
    evaluation = EvalConfig(name="e", harness_version="0.4.13", primary_tasks=["sciq"])
    base_cfg = ExperimentConfig(study="pilot", arm="base_only", student=student, evaluation=evaluation,
                                training=training.model_copy(update={"synthetic_phase": None}))
    inputs = T.RunInputs(data=[str(tmp_path / "corpus")], results_dir=tmp_path / "results", store_dir=tmp_path / "store",
                         eval_sets={"heldout": str(tmp_path / "heldout")})
    m = T.run_phase(base_cfg, "base", inputs, T.RuntimeOptions(micro_batch_seqs=4, device="cpu", eval_sequences=8))
    assert m.runtime.status == "finished" and m.metrics["steps"] == 6 and "heldout" in m.metrics["held_out_nll"]
    out = tmp_path / "results" / m.experiment_id
    assert (out / "manifest.json").exists() and (out / "checkpoint_sha256.txt").read_text().strip() == m.metrics["checkpoint_sha256"]
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert sum("held_out_nll" in r for r in log) == 2  # steps 3 and 6
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["runtime"]["status"] == "finished" and saved["metrics"]["micro_batch_seqs"] == 4

    spec = BranchSpec(size="toy", seed=1, seq_len=L, d2_tokens=4 * 8 * L, replay_fraction=0.25)
    build_branch(treat, corpus, spec, tmp_path / "branch", "treat", "corpus")
    syn_cfg = ExperimentConfig(study="pilot", arm="matched_real", student=student, training=training, evaluation=evaluation,
                               dataset_ref=str(tmp_path / "branch"))
    m2 = T.run_phase(syn_cfg, "synthetic", T.RunInputs(data=[str(tmp_path / "branch")], results_dir=tmp_path / "results",
                                                         store_dir=tmp_path / "store", base_checkpoint=tmp_path / "store" / m.experiment_id / "base_final.pt"),
                     T.RuntimeOptions(micro_batch_seqs=8, device="cpu"))
    assert m2.runtime.status == "finished" and m2.metrics["steps"] == 4
    assert m2.base_checkpoint_sha256 == m.metrics["checkpoint_sha256"]
    assert m2.dataset is not None and m2.dataset.n_examples == 32 and m2.dataset.n_student_tokens == 24 * L
    ckpt = torch.load(tmp_path / "store" / m2.experiment_id / "synthetic_final.pt", weights_only=False)
    steps = {int(s["step"]) for s in ckpt["optimizer"]["state"].values()}
    assert steps == {10}  # AdamW moments carried over from θ*: 6 base + 4 phase-2 steps
    log2 = [json.loads(l) for l in (tmp_path / "results" / m2.experiment_id / "train_log.jsonl").read_text().splitlines()]
    assert log2[0]["lr"] == pytest.approx(3e-3) and log2[3]["lr"] == pytest.approx(3e-3 / 4)  # no re-warm, linear to zero
