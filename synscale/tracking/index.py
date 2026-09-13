"""Build the flat results index from per-run eval metrics (doc 18 §18.5).

Walks ``results/<experiment_id>/eval/metrics.json`` and produces one row per run with the
per-task metric columns the analysis expects (``<task>_<metric>``), the condition label,
the student/teacher covariates, and NLLs. Written as CSV (pandas optional) so it is
inspectable without extra dependencies.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from synscale.analysis.aggregate import TASK_TABLE, metric_column


def _rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(results_dir.glob("*/eval/metrics.json")):
        m = json.loads(metrics_path.read_text())
        row: dict[str, Any] = {
            "experiment_id": m.get("experiment_id", metrics_path.parent.parent.name),
            "cond": m.get("cond"),
            "arm": m.get("arm"),
            "student": m.get("student"),
            "s_params_nominal": m.get("s_params_nominal"),
            "seed_phase": m.get("seed_phase"),
            "teacher": m.get("teacher"),
            "teacher_params": m.get("teacher_params"),
            "t_params_nominal": m.get("teacher_params"),
        }
        for task, res in (m.get("tasks") or {}).items():
            if task in TASK_TABLE:
                metric = TASK_TABLE[task].metric
                key = metric.split(",")[0]
                row[metric_column(task)] = res.get(key if key in res else "acc_norm", res.get("acc"))
        for k, v in (m.get("nll") or {}).items():
            row[f"nll_{k}"] = v
        rows.append(row)
    return rows


def build_index(results_dir: str | Path, out_path: Optional[str | Path] = None) -> Path:
    results_dir = Path(results_dir)
    rows = _rows(results_dir)
    out_path = Path(out_path or results_dir / "index.csv")
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # also parquet if pandas is available
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_path.with_suffix(".parquet"))
    except Exception:
        pass
    return out_path


def load_index(path: str | Path):
    import pandas as pd
    p = Path(path)
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_csv(p)
