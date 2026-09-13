"""Self-contained multiple-choice + NLL evaluator for our own student checkpoints.

Avoids exporting to HF format or pinning an lm-eval version: we own the model, so we
score each answer option by its per-token log-likelihood under the student and pick the
argmax (``acc``) and the length-normalised argmax (``acc_norm``, per-character), exactly
the harness convention (doc 11 §2.1). This runs on CPU with the toy model for validation
and on GPU for the real students. `synscale.evaluation.lm_eval_runner` remains available
as an independent cross-check on an exported checkpoint.

Benchmark items are loaded by `synscale.evaluation.benchmarks` (HuggingFace `datasets` for
real tasks; JSON fixtures for micro/smoke).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class MCItem:
    context: str            # prompt shown before the answer
    options: list[str]      # candidate continuations (full answer text)
    gold: int               # index of the correct option


def _loglik_of_continuation(model, tokenizer, bos_id, context_ids, cont_ids, device, ctx_len) -> float:
    """Sum log p(cont | context) under the model (teacher-forced)."""
    import torch
    ids = [bos_id] + context_ids + cont_ids
    ids = ids[-ctx_len:]
    n_cont = len(cont_ids)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    # positions predicting the continuation tokens are the last n_cont targets
    targets = ids[1:]
    for j in range(len(targets) - n_cont, len(targets)):
        total += float(logp[j, targets[j]])
    return total


def score_mc(model, tokenizer, items: Sequence[MCItem], *, bos_id: int, device="cpu",
             ctx_len: int = 2048) -> dict[str, float]:
    import torch
    model.eval()
    correct = correct_norm = 0
    for it in items:
        ctx_ids = tokenizer.encode(it.context)
        lls, lls_norm = [], []
        for opt in it.options:
            cont_ids = tokenizer.encode(opt)
            if not cont_ids:
                lls.append(-1e9); lls_norm.append(-1e9); continue
            ll = _loglik_of_continuation(model, tokenizer, bos_id, ctx_ids, cont_ids, device, ctx_len)
            lls.append(ll)
            lls_norm.append(ll / max(len(opt), 1))   # per-character normalisation (acc_norm)
        if int(np.argmax(lls)) == it.gold:
            correct += 1
        if int(np.argmax(lls_norm)) == it.gold:
            correct_norm += 1
    n = len(items)
    return {"acc": correct / n if n else float("nan"),
            "acc_norm": correct_norm / n if n else float("nan"),
            "n": n}


# metric to report per task (doc 11 §2.1)
TASK_METRIC = {
    "sciq": "acc", "arc_easy": "acc_norm", "piqa": "acc_norm", "lambada_openai": "acc",
    "openbookqa": "acc_norm", "social_iqa": "acc", "hellaswag": "acc_norm",
    "arc_challenge": "acc_norm", "winogrande": "acc", "boolq": "acc",
}


def evaluate_checkpoint(model, tokenizer, *, tasks: Sequence[str], loader, bos_id: int,
                        device="cpu", ctx_len: int = 2048, limit: Optional[int] = None) -> dict[str, Any]:
    """Score `tasks` and return {task: {metric...}} plus the reported metric per task."""
    out: dict[str, Any] = {"tasks": {}, "reported": {}}
    for task in tasks:
        items = loader(task, limit=limit)
        res = score_mc(model, tokenizer, items, bos_id=bos_id, device=device, ctx_len=ctx_len)
        out["tasks"][task] = res
        out["reported"][task] = res[TASK_METRIC.get(task, "acc_norm")]
    return out
