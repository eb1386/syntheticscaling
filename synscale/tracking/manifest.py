"""Run manifest: the immutable record written for every run.

A manifest is written at launch (fields known a priori) and completed at exit. It is the
unit of provenance for the analysis stage: no metric enters the results tables unless it
is attached to a manifest with a matching experiment_id and config_hash.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from synscale.config.schemas import ExperimentConfig
from synscale.tracking.ids import config_hash, experiment_id


class GitInfo(BaseModel):
    commit: str
    dirty: bool
    branch: str


class HardwareInfo(BaseModel):
    gpu_name: str = "unknown"
    gpu_count: int = 0
    gpu_mem_gb: float = 0.0
    driver: str = "unknown"
    cuda: str = "unknown"
    torch: str = "unknown"
    hostname: str = ""
    cpu: str = ""


class RuntimeInfo(BaseModel):
    started_at: str
    finished_at: Optional[str] = None
    wall_seconds: Optional[float] = None
    gpu_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    mfu: Optional[float] = None
    energy_kwh: Optional[float] = None
    status: str = "running"


class DatasetInfo(BaseModel):
    path: str
    sha256: str
    n_examples: int
    n_student_tokens: int
    prompt_pool_sha256: Optional[str] = None
    generation_manifest: Optional[str] = None


class RunManifest(BaseModel):
    experiment_id: str
    config_hash: str
    config: ExperimentConfig
    git: GitInfo
    seeds: dict[str, int]
    base_checkpoint_sha256: Optional[str] = None
    dataset: Optional[DatasetInfo] = None
    hardware: HardwareInfo
    runtime: RuntimeInfo
    software: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    eval_results_path: Optional[str] = None

    def write(self, results_dir: Path) -> Path:
        out = results_dir / self.experiment_id / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True))
        return out


def _git(args: list[str], cwd: Optional[str] = None) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def collect_git_info(cwd: Optional[str] = None) -> GitInfo:
    commit = _git(["rev-parse", "HEAD"], cwd) or "unknown"
    dirty = bool(_git(["status", "--porcelain"], cwd))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "unknown"
    return GitInfo(commit=commit, dirty=dirty, branch=branch)


def collect_hardware_info() -> HardwareInfo:
    info = HardwareInfo(hostname=platform.node(), cpu=platform.processor())
    try:  # optional torch
        import torch  # type: ignore

        info.torch = torch.__version__
        if torch.cuda.is_available():
            info.gpu_count = torch.cuda.device_count()
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            info.cuda = torch.version.cuda or "unknown"
    except Exception:
        pass
    return info


def new_manifest(cfg: ExperimentConfig, repo_root: Optional[str] = None) -> RunManifest:
    return RunManifest(
        experiment_id=experiment_id(cfg),
        config_hash=config_hash(cfg),
        config=cfg,
        git=collect_git_info(repo_root),
        seeds={
            "base": cfg.training.seed_base,
            "phase": cfg.training.seed_phase,
            "eval": cfg.evaluation.eval_seed,
            "sampling": cfg.generation.sampling_seed if cfg.generation else -1,
        },
        hardware=collect_hardware_info(),
        runtime=RuntimeInfo(started_at=datetime.now(timezone.utc).isoformat()),
        software={"python": platform.python_version(), "pid": str(os.getpid())},
    )
