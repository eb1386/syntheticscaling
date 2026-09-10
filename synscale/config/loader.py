"""Compose an ExperimentConfig from YAML fragments.

An experiment YAML references fragments by path and may override leaf values:

    study: main
    arm: synthetic
    variant: eqtok
    student: configs/students/s100m.yaml
    teacher: configs/teachers/qwen2.5-7b-instruct.yaml
    generation: configs/generation/main_v1.yaml
    training: configs/training/main_v1.yaml
    evaluation: configs/evaluation/primary_v1.yaml
    overrides:
      training.seed_phase: 2
      training.synthetic_phase.tokens: 500_000_000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from synscale.config.schemas import ExperimentConfig

_FRAGMENT_KEYS = ("student", "teacher", "generation", "training", "evaluation")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _apply_override(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"override path '{dotted}' does not exist in the composed config")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"override path '{dotted}' does not exist in the composed config")
    node[keys[-1]] = value


def load_experiment(path: str | Path, repo_root: str | Path | None = None) -> ExperimentConfig:
    path = Path(path)
    root = Path(repo_root) if repo_root else path.parent
    spec = _load_yaml(path)
    tree: dict[str, Any] = {}
    for key, val in spec.items():
        if key in _FRAGMENT_KEYS and isinstance(val, str):
            frag_path = (root / val) if not Path(val).is_absolute() else Path(val)
            tree[key] = _load_yaml(frag_path)
        elif key == "overrides":
            continue
        else:
            tree[key] = val
    for dotted, value in (spec.get("overrides") or {}).items():
        _apply_override(tree, dotted, value)
    return ExperimentConfig.model_validate(tree)
