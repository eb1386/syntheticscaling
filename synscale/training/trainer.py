"""Single-device training loop for the student family (student_training.md §2, §5).

* bf16 autocast on CUDA, fp32 master weights / gradients / AdamW moments; fp32 loss;
* AdamW (fused on CUDA) with β = (0.9, 0.95), ε, weight decay on 2-D matrices only
  (not RMSNorm gains, not the tied embedding), global-norm gradient clipping;
* gradient accumulation with the loss normalised by the number of target tokens in the
  effective batch; the micro-batch is fixed per (size, GPU) and recorded in the manifest;
* stateless data order (``DataOrder``) so resumption restores the exact sequence stream;
* held-out NLL every ``eval_every_tokens`` on fixed prefixes of the eval sets;
* final-checkpoint-only policy: the phase writes one checkpoint at its end (an optional
  in-place ``resume.pt`` exists purely for crash recovery and is never evaluated);
* divergence rule: loss > 2× running minimum for more than 50 consecutive steps, or a
  non-finite loss, stops the run with status ``failed``;
* ``train_log.jsonl`` (step, tokens, loss, lr, grad_norm, tok/s, mfu) and the run manifest
  (written at start with status ``running``, completed at exit).
"""
from __future__ import annotations

import signal as _signal
_PREEMPTED = {"flag": False}
def _install_preemption_handler():
    def _h(signum, frame):
        _PREEMPTED["flag"] = True
    try:
        _signal.signal(_signal.SIGTERM, _h); _signal.signal(_signal.SIGUSR1, _h)
    except Exception:
        pass

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from synscale.config.schemas import ExperimentConfig, PhaseConfig, StudentConfig
from synscale.models.llama_like import LlamaLike, cross_entropy_sum
from synscale.models.param_count import flops_per_token, peak_tflops_for_device
from synscale.tracking.manifest import DatasetInfo, RunManifest, new_manifest
from synscale.training.data import Batch, DataOrder, SequenceSource, SequenceSourceLike, ShardedSource, collate, sha256_file
from synscale.training.schedules import LRSchedule, schedule_from_phase

PhaseName = Literal["base", "synthetic"]
DIVERGENCE_FACTOR = 2.0
DIVERGENCE_PATIENCE = 50


# ----------------------------------------------------------------------------- options


@dataclass
class RuntimeOptions:
    """Runtime knobs that never enter the experiment identity (they are recorded in the manifest)."""

    micro_batch_seqs: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_sequences: int = 64
    log_every: int = 1
    resume_every_steps: Optional[int] = None
    resume_from: Optional[Path] = None
    peak_tflops: Optional[float] = None
    max_steps: Optional[int] = None


@dataclass
class RunInputs:
    """Paths a phase needs beyond the ExperimentConfig (candidate schema fields; see summary)."""

    data: list[str]
    results_dir: Path
    store_dir: Path
    base_checkpoint: Optional[Path] = None
    eval_sets: dict[str, str] = field(default_factory=dict)
    out_checkpoint: Optional[Path] = None


@dataclass
class TrainState:
    step: int = 0
    tokens: int = 0
    running_min: float = math.inf
    steps_above: int = 0
    last_loss: float = math.nan
    status: str = "running"
    failure_reason: str = ""


# ----------------------------------------------------------------------------- optimizer


def build_optimizer(model: LlamaLike, phase: PhaseConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay restricted to 2-D matrices, excluding the (tied) embedding."""
    embedding_ids = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    decay = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in embedding_ids]
    no_decay = [p for p in model.parameters() if p.ndim < 2 or id(p) in embedding_ids]
    groups = [
        {"params": decay, "weight_decay": phase.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=phase.lr_peak, betas=phase.betas, eps=phase.eps, fused=device.type == "cuda")


def apply_phase_hparams(optimizer: torch.optim.Optimizer, phase: PhaseConfig) -> None:
    """Re-assert the phase's betas/eps/weight-decay after ``load_state_dict`` (which restores
    the saved group options)."""
    for i, group in enumerate(optimizer.param_groups):
        group["betas"] = tuple(phase.betas)
        group["eps"] = phase.eps
        group["weight_decay"] = phase.weight_decay if i == 0 else 0.0


# ----------------------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, model: LlamaLike, optimizer: torch.optim.Optimizer, state: TrainState, phase: PhaseName, seed: int) -> str:
    """Weights + optimizer state + loader position (``state.step``) + RNG; returns the SHA-256."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": asdict(state),
        "phase": phase,
        "seed": seed,
        "student": model.cfg.model_dump(mode="json"),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def load_checkpoint(path: Path, model: LlamaLike, optimizer: Optional[torch.optim.Optimizer] = None, restore_rng: bool = True) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        torch.set_rng_state(payload["rng"]["torch"])
        if torch.cuda.is_available() and payload["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return TrainState(**payload["state"])


# ----------------------------------------------------------------------------- batches & loss


def to_device(batch: Batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.from_numpy(batch.input_ids).to(device),
        "targets": torch.from_numpy(batch.targets).to(device),
        "position_ids": torch.from_numpy(batch.position_ids).to(device),
        "cu_seqlens": torch.from_numpy(batch.cu_seqlens).to(device),
    }


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)


def loss_sum_on(model: LlamaLike, batch: Batch, device: torch.device, intra_document: bool) -> torch.Tensor:
    """Summed cross-entropy over the batch's target tokens (fp32)."""
    t = to_device(batch, device)
    with _autocast(device):
        logits = model(t["input_ids"], position_ids=t["position_ids"] if intra_document else None,
                       cu_seqlens=t["cu_seqlens"] if intra_document else None)
    return cross_entropy_sum(logits, t["targets"])


@torch.no_grad()
def evaluate_nll(model: LlamaLike, source: SequenceSourceLike, n_sequences: int, micro_batch: int, device: torch.device, intra_document: bool) -> float:
    """Mean per-token NLL over the first ``n_sequences`` of ``source`` (fixed across runs)."""
    model.eval()
    n = min(n_sequences, len(source))
    total, count = 0.0, 0
    for start in range(0, n, micro_batch):
        items = [source[i] for i in range(start, min(start + micro_batch, n))]
        batch = collate(items, source.seq_len)
        total += float(loss_sum_on(model, batch, device, intra_document))
        count += batch.n_tokens
    model.train()
    return total / max(count, 1)


# ----------------------------------------------------------------------------- the loop


@dataclass
class PhaseRun:
    """Everything the loop needs for one phase."""

    model: LlamaLike
    optimizer: torch.optim.Optimizer
    schedule: LRSchedule
    phase: PhaseConfig
    source: SequenceSourceLike
    order: DataOrder
    seq_len: int
    device: torch.device
    options: RuntimeOptions
    out_dir: Path
    eval_sets: dict[str, SequenceSourceLike] = field(default_factory=dict)
    eval_every_steps: int = 0
    train_flops_per_token: Optional[int] = None

    @property
    def seqs_per_step(self) -> int:
        return self.phase.batch_tokens // self.seq_len

    @property
    def accumulation(self) -> int:
        if self.seqs_per_step % self.options.micro_batch_seqs:
            raise ValueError(f"micro_batch_seqs={self.options.micro_batch_seqs} must divide {self.seqs_per_step} sequences per step")
        return self.seqs_per_step // self.options.micro_batch_seqs

    @property
    def intra_document(self) -> bool:
        return self.phase.packing == "packed_masked"


def train_step(run: PhaseRun, step: int) -> tuple[float, float, float]:
    """One optimizer step (all micro-batches); returns (loss per token, lr, grad norm)."""
    lr = run.schedule.lr(step)
    for group in run.optimizer.param_groups:
        group["lr"] = lr
    run.optimizer.zero_grad(set_to_none=True)
    tokens_in_step = run.seqs_per_step * run.seq_len
    first = step * run.seqs_per_step
    loss_total = 0.0
    for m in range(run.accumulation):
        start = first + m * run.options.micro_batch_seqs
        items = [run.source[i] for i in run.order.indices(start, run.options.micro_batch_seqs)]
        loss = loss_sum_on(run.model, collate(items, run.seq_len), run.device, run.intra_document) / tokens_in_step
        loss.backward()
        loss_total += float(loss.detach())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(run.model.parameters(), run.phase.grad_clip))
    run.optimizer.step()
    return loss_total, lr, grad_norm


def _check_divergence(state: TrainState, loss: float) -> None:
    if not math.isfinite(loss):
        state.status, state.failure_reason = "failed", f"non-finite loss at step {state.step}"
        return
    state.running_min = min(state.running_min, loss)
    state.steps_above = state.steps_above + 1 if loss > DIVERGENCE_FACTOR * state.running_min else 0
    if state.steps_above > DIVERGENCE_PATIENCE:
        state.status, state.failure_reason = "failed", f"loss > {DIVERGENCE_FACTOR}x running minimum for {state.steps_above} steps"


def train_phase(run: PhaseRun, state: TrainState, phase_name: PhaseName, seed: int, final_checkpoint: Path) -> tuple[TrainState, dict[str, Any]]:
    """Run the loop from ``state.step`` to the end of the schedule (or ``max_steps``).

    Returns the final state and a metrics dict (final loss, last held-out NLLs, throughput,
    mfu, checkpoint sha256). Appends to ``<out_dir>/train_log.jsonl``.
    """
    run.model.train()
    total_steps = run.schedule.total_steps if run.options.max_steps is None else min(run.schedule.total_steps, run.options.max_steps)
    log_path = run.out_dir / "train_log.jsonl"
    run.out_dir.mkdir(parents=True, exist_ok=True)
    peak = run.options.peak_tflops
    if peak is None and run.device.type == "cuda":
        peak = peak_tflops_for_device(torch.cuda.get_device_name(run.device))
    held_out: dict[str, float] = {}
    _install_preemption_handler()
    t_start = time.perf_counter()
    tokens_at_start = state.tokens
    window_t, window_tokens = time.perf_counter(), 0
    with open(log_path, "a", encoding="utf-8") as log:
        while state.step < total_steps and state.status == "running":
            loss, lr, grad_norm = train_step(run, state.step)
            state.step += 1
            state.tokens += run.seqs_per_step * run.seq_len
            state.last_loss = loss
            window_tokens += run.seqs_per_step * run.seq_len
            _check_divergence(state, loss)
            if state.step % run.options.log_every == 0 or state.status != "running":
                now = time.perf_counter()
                tok_s = window_tokens / max(now - window_t, 1e-9)
                window_t, window_tokens = now, 0
                mfu = (run.train_flops_per_token * tok_s / (peak * 1e12)) if (peak and run.train_flops_per_token) else None
                log.write(json.dumps({"step": state.step, "tokens": state.tokens, "loss": loss, "lr": lr, "grad_norm": grad_norm, "tok_s": tok_s, "mfu": mfu}) + "\n")
                log.flush()
            if run.eval_every_steps and state.step % run.eval_every_steps == 0 and run.eval_sets:
                held_out = {name: evaluate_nll(run.model, src, run.options.eval_sequences, run.options.micro_batch_seqs, run.device, run.intra_document) for name, src in run.eval_sets.items()}
                log.write(json.dumps({"step": state.step, "tokens": state.tokens, "held_out_nll": held_out}) + "\n")
                log.flush()
            if run.options.resume_every_steps and state.step % run.options.resume_every_steps == 0 and state.status == "running":
                save_checkpoint(run.out_dir / "resume.pt", run.model, run.optimizer, state, phase_name, seed)
            if _PREEMPTED["flag"]:
                # spot reclaim: persist a resume checkpoint and exit non-zero so the fleet requeues.
                save_checkpoint(run.out_dir / "resume.pt", run.model, run.optimizer, state, phase_name, seed)
                import sys as _sys
                print(f"[preempted] saved resume.pt at step {state.step}; exiting for requeue")
                _sys.exit(75)  # EX_TEMPFAIL: transient, retry
    if state.status == "running":
        state.status = "finished"
    if run.eval_sets and (not held_out or state.step % max(run.eval_every_steps, 1) != 0):
        held_out = {name: evaluate_nll(run.model, src, run.options.eval_sequences, run.options.micro_batch_seqs, run.device, run.intra_document) for name, src in run.eval_sets.items()}
    wall = time.perf_counter() - t_start
    tok_s = (state.tokens - tokens_at_start) / max(wall, 1e-9)
    sha = save_checkpoint(final_checkpoint, run.model, run.optimizer, state, phase_name, seed)
    metrics = {
        "final_train_loss": state.last_loss, "held_out_nll": held_out, "steps": state.step, "tokens_trained": state.tokens,
        "tokens_per_second": tok_s, "mfu": (run.train_flops_per_token * tok_s / (peak * 1e12)) if (peak and run.train_flops_per_token) else None,
        "wall_seconds": wall, "checkpoint_sha256": sha, "checkpoint_path": str(final_checkpoint),
        "micro_batch_seqs": run.options.micro_batch_seqs, "grad_accumulation": run.accumulation, "status": state.status,
        "failure_reason": state.failure_reason,
    }
    return state, metrics


# ----------------------------------------------------------------------------- phase orchestration


def open_sources(stems: list[str], seq_len: int) -> SequenceSourceLike:
    sources = [SequenceSource.open(s, seq_len) for s in stems]
    return sources[0] if len(sources) == 1 else ShardedSource(sources)


def _phase_config(cfg: ExperimentConfig, phase: PhaseName) -> PhaseConfig:
    if phase == "base":
        return cfg.training.base_phase
    if cfg.training.synthetic_phase is None:
        raise ValueError("config has no synthetic_phase")
    return cfg.training.synthetic_phase


def _dataset_info(stem: str) -> Optional[DatasetInfo]:
    branch_path = Path(stem + ".branch.json")
    if not branch_path.exists():
        return None
    b = json.loads(branch_path.read_text())
    return DatasetInfo(path=stem, sha256=b["sha256"], n_examples=b["n_slots"], n_student_tokens=b["treatment_tokens"])


def run_phase(cfg: ExperimentConfig, phase: PhaseName, inputs: RunInputs, options: RuntimeOptions, repo_root: Optional[str] = None) -> RunManifest:
    """Train one phase of an experiment end to end and return the completed manifest.

    ``base``: fresh model seeded with ``seed_base``, seeded shuffle of the document stream(s)
    in ``inputs.data``, WSD warmup + constant LR, checkpoint θ* at the end.
    ``synthetic``: weights + AdamW state loaded from ``inputs.base_checkpoint``, branch file
    consumed in builder order (``seed_phase`` fixed that order), linear decay to zero.
    """
    phase_cfg = _phase_config(cfg, phase)
    device = torch.device(options.device)
    seq_len = cfg.training.context_length
    if phase_cfg.packing == "padded":
        raise NotImplementedError("padded packing is not supported; use packed_masked or packed_unmasked")

    manifest = new_manifest(cfg, repo_root)
    out_dir = inputs.results_dir / manifest.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if phase == "base":
        seed = cfg.training.seed_base
        torch.manual_seed(seed)
        model = LlamaLike(cfg.student).to(device)
        optimizer = build_optimizer(model, phase_cfg, device)
        state = TrainState()
        shuffle = True
    else:
        seed = cfg.training.seed_phase
        if inputs.base_checkpoint is None:
            raise ValueError("synthetic phase requires inputs.base_checkpoint")
        torch.manual_seed(seed)
        model = LlamaLike(cfg.student).to(device)
        optimizer = build_optimizer(model, phase_cfg, device)
        load_checkpoint(inputs.base_checkpoint, model, optimizer, restore_rng=False)
        apply_phase_hparams(optimizer, phase_cfg)
        manifest.base_checkpoint_sha256 = sha256_file(inputs.base_checkpoint)
        manifest.dataset = _dataset_info(inputs.data[0]) if len(inputs.data) == 1 else None
        state = TrainState()
        shuffle = False
    if options.resume_from is not None:
        state = load_checkpoint(options.resume_from, model, optimizer)
        apply_phase_hparams(optimizer, phase_cfg)

    source = open_sources(inputs.data, seq_len)
    order = DataOrder(len(source), seed, shuffle=shuffle)
    eval_sets = {name: SequenceSource.open(stem, seq_len) for name, stem in inputs.eval_sets.items()}
    run = PhaseRun(
        model=model, optimizer=optimizer, schedule=schedule_from_phase(phase_cfg), phase=phase_cfg, source=source,
        order=order, seq_len=seq_len, device=device, options=options, out_dir=out_dir, eval_sets=eval_sets,
        eval_every_steps=cfg.training.eval_every_tokens // phase_cfg.batch_tokens,
        train_flops_per_token=flops_per_token(cfg.student, seq_len, causal_half=True)["train"],
    )
    manifest.metrics = {"phase": phase, "micro_batch_seqs": options.micro_batch_seqs, "grad_accumulation": run.accumulation,
                        "steps_planned": run.schedule.total_steps, "sequences_available": len(source)}
    manifest.software["torch"] = torch.__version__
    manifest.write(inputs.results_dir)

    final_ckpt = inputs.out_checkpoint or (inputs.store_dir / manifest.experiment_id / f"{phase}_final.pt")
    state, metrics = train_phase(run, state, phase, seed, final_ckpt)

    manifest.runtime.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.runtime.wall_seconds = metrics["wall_seconds"]
    manifest.runtime.tokens_per_second = metrics["tokens_per_second"]
    manifest.runtime.mfu = metrics["mfu"]
    manifest.runtime.status = state.status
    manifest.metrics.update(metrics)
    manifest.write(inputs.results_dir)
    (out_dir / "checkpoint_sha256.txt").write_text(metrics["checkpoint_sha256"] + "\n")
    return manifest

