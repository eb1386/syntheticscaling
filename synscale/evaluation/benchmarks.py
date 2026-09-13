"""Load evaluation benchmarks as multiple-choice items (real via `datasets`, micro via fixtures).

Real tasks use the same public datasets and framing as lm-eval-harness (doc 11 §2.3). The
loader returns `MCItem`s; the local scorer turns them into acc/acc_norm. Downloaded once by
scripts/prepare_data.py; cached by the `datasets` library.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from synscale.evaluation.local_eval import MCItem

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_fixtures"


def _fixture_loader(task: str, limit: Optional[int] = None) -> list[MCItem]:
    path = FIXTURE_DIR / f"{task}.json"
    rows = json.loads(path.read_text())
    items = [MCItem(context=r["context"], options=r["options"], gold=r["gold"]) for r in rows]
    return items[:limit] if limit else items


def _hf_loader(task: str, limit: Optional[int] = None) -> list[MCItem]:  # pragma: no cover (needs network)
    from datasets import load_dataset
    items: list[MCItem] = []
    if task == "arc_easy" or task == "arc_challenge":
        name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", name, split="test")
        for r in ds:
            opts = r["choices"]["text"]; labels = r["choices"]["label"]
            try:
                gold = labels.index(r["answerKey"])
            except ValueError:
                continue
            items.append(MCItem(f"Question: {r['question']}\nAnswer:", [f" {o}" for o in opts], gold))
    elif task == "sciq":
        ds = load_dataset("allenai/sciq", split="test")
        for r in ds:
            opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
            support = (r.get("support") or "").strip()
            ctx = (support + "\n" if support else "") + f"Question: {r['question']}\nAnswer:"
            items.append(MCItem(ctx, [f" {o}" for o in opts], 3))
    elif task == "piqa":
        ds = load_dataset("piqa", split="validation")
        for r in ds:
            items.append(MCItem(f"Question: {r['goal']}\nAnswer:", [f" {r['sol1']}", f" {r['sol2']}"], r["label"]))
    elif task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="test")
        for r in ds:
            opts = r["choices"]["text"]; labels = r["choices"]["label"]
            gold = labels.index(r["answerKey"])
            items.append(MCItem(r["question_stem"], [f" {o}" for o in opts], gold))
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for r in ds:
            ctx = f"{r['activity_label']}: {r['ctx']}"
            items.append(MCItem(ctx, [f" {e}" for e in r["endings"]], int(r["label"])))
    elif task == "social_iqa":
        ds = load_dataset("social_i_qa", split="validation", trust_remote_code=True)
        for r in ds:
            ctx = f"Q: {r['context']} {r['question']}\nA:"
            items.append(MCItem(ctx, [f" {r['answerA']}", f" {r['answerB']}", f" {r['answerC']}"], int(r["label"]) - 1))
    elif task == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        for r in ds:
            # cloze: replace blank with each option, score the suffix (approximation)
            for opt_key, idx in (("option1", 0), ("option2", 1)):
                pass
            ctx = r["sentence"].split("_")[0]
            suff = r["sentence"].split("_")[1] if "_" in r["sentence"] else ""
            items.append(MCItem(ctx, [f"{r['option1']}{suff}", f"{r['option2']}{suff}"], int(r["answer"]) - 1))
    elif task == "boolq":
        ds = load_dataset("boolq", split="validation")
        for r in ds:
            items.append(MCItem(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", [" no", " yes"], int(r["answer"])))
    else:
        raise ValueError(f"no loader for task {task!r}")
    return items[:limit] if limit else items


def get_loader(use_fixtures: bool) -> Callable[..., list[MCItem]]:
    return _fixture_loader if use_fixtures else _hf_loader
