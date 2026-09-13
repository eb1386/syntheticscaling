"""Learning-rate schedules (student_training.md §2.5).

* **WSD** (warmup–stable–decay): linear warmup from ``lr_peak/warmup_steps`` to ``lr_peak``,
  constant until ``total_steps − decay_steps``, then linear decay towards ``lr_min``.
  The base phase uses ``decay_steps = 0`` (stop at the stable checkpoint θ*); phase 2 uses
  ``warmup_steps = 0`` (no re-warm, the branch starts at the peak) and
  ``decay_steps = total_steps`` with ``lr_min = 0`` (linear decay to zero over D₂).
* **cosine** fallback: linear warmup then cosine to ``lr_min``.

Steps are 0-indexed optimizer steps; ``lr(step)`` is the rate applied at that step
(nanoGPT convention: the decay reaches ``lr_min`` one step after the last update).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from synscale.config.schemas import PhaseConfig


def steps_for(tokens: int, batch_tokens: int) -> int:
    """Number of full optimizer steps in a phase (partial final batches are never trained)."""
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    return tokens // batch_tokens


def warmup_lr(step: int, lr_peak: float, warmup_steps: int) -> float:
    return lr_peak * (step + 1) / warmup_steps


def wsd_lr(step: int, lr_peak: float, lr_min: float, warmup_steps: int, total_steps: int, decay_steps: int) -> float:
    if step < warmup_steps:
        return warmup_lr(step, lr_peak, warmup_steps)
    decay_start = total_steps - decay_steps
    if decay_steps <= 0 or step < decay_start:
        return lr_peak
    progress = min((step - decay_start) / decay_steps, 1.0)
    return lr_peak + (lr_min - lr_peak) * progress


def cosine_lr(step: int, lr_peak: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return warmup_lr(step, lr_peak, warmup_steps)
    span = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / span, 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class LRSchedule:
    kind: Literal["wsd", "cosine"]
    lr_peak: float
    lr_min: float
    warmup_steps: int
    total_steps: int
    decay_steps: int = 0

    def __post_init__(self) -> None:
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps exceeds total_steps")
        if self.kind == "wsd" and self.decay_steps > self.total_steps - self.warmup_steps:
            raise ValueError("decay_steps exceeds the post-warmup span")

    def lr(self, step: int) -> float:
        if self.kind == "wsd":
            return wsd_lr(step, self.lr_peak, self.lr_min, self.warmup_steps, self.total_steps, self.decay_steps)
        return cosine_lr(step, self.lr_peak, self.lr_min, self.warmup_steps, self.total_steps)

    def curve(self) -> np.ndarray:
        """LR at every step of the phase, shape ``(total_steps,)``."""
        return np.asarray([self.lr(s) for s in range(self.total_steps)], dtype=np.float64)


def schedule_from_phase(phase: PhaseConfig) -> LRSchedule:
    """Translate a token-denominated PhaseConfig into a step-denominated schedule."""
    total = steps_for(phase.tokens, phase.batch_tokens)
    warmup = steps_for(phase.warmup_tokens, phase.batch_tokens)
    decay = steps_for(phase.decay_tokens or 0, phase.batch_tokens) if phase.schedule == "wsd" else 0
    return LRSchedule(
        kind=phase.schedule, lr_peak=phase.lr_peak, lr_min=phase.lr_peak * phase.lr_min_ratio,
        warmup_steps=warmup, total_steps=total, decay_steps=min(decay, total - warmup),
    )
