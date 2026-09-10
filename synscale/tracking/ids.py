"""Deterministic experiment identifiers.

Human-readable stem + content hash:

    <study>.<student>.<source>.d<Dsyn>.<variant>.s<seed>-<hash8>

    main.s100m.t7b.d500m.eqtok.s1-3fa9c2d1
    main.s100m.base.d0.-.s1-9b12ee40          (base-only control)
    main.s100m.real.d500m.eqtok.s1-77ac0b3e   (matched-real control)

The hash is SHA-256 over the canonical JSON of the resolved ExperimentConfig with
volatile/annotation fields removed, so re-launching an identical condition yields the
identical ID, and any change to a controlled quantity yields a different one.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from synscale.config.schemas import ExperimentConfig

_VOLATILE_FIELDS = {"notes"}


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items()) if k not in _VOLATILE_FIELDS}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def config_hash(cfg: ExperimentConfig, n_hex: int = 8) -> str:
    payload = json.dumps(_canonical(cfg.model_dump(mode="json")), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n_hex]


def human_tokens(n: int) -> str:
    """500_000_000 -> '500m'; 2_000_000_000 -> '2b'; 0 -> '0'."""
    if n == 0:
        return "0"
    if n % 1_000_000_000 == 0:
        return f"{n // 1_000_000_000}b"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    return str(n)


def source_token(cfg: ExperimentConfig) -> str:
    if cfg.arm == "synthetic":
        assert cfg.teacher is not None
        return cfg.teacher.name
    return {"base_only": "base", "matched_real": "real", "human_instruct": "human"}[cfg.arm]


def experiment_id(cfg: ExperimentConfig) -> str:
    d_syn = cfg.training.synthetic_phase.tokens if cfg.training.synthetic_phase else 0
    variant = cfg.variant if cfg.arm != "base_only" else "-"
    stem = ".".join(
        [
            cfg.study,
            cfg.student.name,
            source_token(cfg),
            f"d{human_tokens(d_syn)}",
            variant,
            f"s{cfg.training.seed_phase}",
        ]
    )
    return f"{stem}-{config_hash(cfg)}"
