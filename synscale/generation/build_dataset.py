"""Turn filtered teacher generations into a packed phase-2 treatment document stream.

Steps (doc 09 §3.3, doc 00 §0.7):
1. Union-drop: a prompt flagged as contaminated in ANY teacher's pool is dropped for
   EVERY teacher, so all teachers keep identical prompt sets w.r.t. contamination.
2. Build the Example list (rendered student-token counts) from retained generations.
3. Subsample to D_syn student tokens (equal-token primary; equal-examples / length-matched
   / nested-D available), stratified by category, common-random-numbers order.
4. Render each example ("### Instruction:/### Response:") and tokenize to a uint16
   document stream + doc_starts (consumed by build_branch).

The result is a document-stream stem; scripts/build_branch.py packs it with replay into
the final phase-2 file. Deterministic in (generations, D_syn, seed, tokenizer).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from synscale.generation.subsample import (Example, category_token_shares,
                                           subsample_equal_tokens, subsample_nested_d,
                                           subsample_report)
from synscale.training.data import (TokenizerLike, render_instruction, tokenize_documents,
                                    write_document_file)


def union_dropped_prompt_ids(teacher_dirs: Mapping[str, str | Path]) -> set[str]:
    """Prompt ids contaminated in ANY teacher's pool (retained_only=False so we see flags)."""
    from synscale.generation.runner import iter_generated
    dropped: set[str] = set()
    for _, d in teacher_dirs.items():
        for rec in iter_generated(d, retained_only=False):
            if rec.get("flags", {}).get("contaminated"):
                dropped.add(rec["prompt_id"])
    return dropped


def examples_from_generations(gen_dir: str | Path, *, drop_ids: Optional[set[str]] = None) -> tuple[list[Example], dict[str, dict]]:
    """Retained examples for one teacher, excluding union-dropped prompts.

    Returns (examples, by_prompt) where by_prompt maps prompt_id -> the JSONL record
    (for later rendering).
    """
    from synscale.generation.runner import iter_generated
    drop_ids = drop_ids or set()
    examples: list[Example] = []
    by_prompt: dict[str, dict] = {}
    for i, rec in enumerate(iter_generated(gen_dir, retained_only=True)):
        pid = rec["prompt_id"]
        if pid in drop_ids:
            continue
        examples.append(Example(prompt_id=pid, category=rec["category"],
                                tokens_student=rec["tokens_student"],
                                response_len_student=rec.get("response_len_student", rec["tokens_student"]),
                                idx=i))
        by_prompt[pid] = rec
    return examples, by_prompt


def build_treatment_stream(
    gen_dir: str | Path,
    out_stem: str | Path,
    tokenizer: TokenizerLike,
    bos_id: int,
    eos_id: int,
    *,
    d_syn: int,
    seed: int,
    variant: str = "eqtok",
    category_shares: Optional[Mapping[str, float]] = None,
    drop_ids: Optional[set[str]] = None,
    d_levels: Optional[list[int]] = None,
) -> dict[str, Any]:
    examples, by_prompt = examples_from_generations(gen_dir, drop_ids=drop_ids)
    if not examples:
        raise ValueError(f"no retained examples in {gen_dir}")
    shares = dict(category_shares) if category_shares else category_token_shares(examples)

    if variant.startswith("eqtok") and d_levels:
        nested = subsample_nested_d(examples, d_levels, seed=seed, category_shares=shares)
        chosen = nested[d_syn]
    else:
        chosen = subsample_equal_tokens(examples, d_syn, seed=seed, category_shares=shares)

    # Render in the seeded subsample order (order is fixed by build_branch's slot pattern,
    # but we keep a deterministic document order here too).
    texts = [render_instruction(by_prompt[e.prompt_id]["student_prompt"],
                                by_prompt[e.prompt_id]["response"]) for e in chosen]
    tokens, doc_starts = tokenize_documents(texts, tokenizer, bos_id, eos_id)
    meta = write_document_file(out_stem, tokens, doc_starts, extra={
        "source": str(gen_dir), "variant": variant, "d_syn_target": d_syn,
        "n_examples": len(chosen), "n_tokens": int(tokens.size), "seed": seed,
        "tokenizer": getattr(tokenizer, "name", "unknown"),
        "prompt_token_fraction": _prompt_token_fraction(by_prompt, chosen, tokenizer),
        "report": subsample_report(chosen),
    })
    return meta


def _prompt_token_fraction(by_prompt, chosen, tokenizer) -> float:
    """Fraction of rendered tokens that are prompt+markers (reported covariate, doc 00 §0.7)."""
    try:
        pt = rt = 0
        for e in chosen[: min(len(chosen), 2000)]:
            rec = by_prompt[e.prompt_id]
            full = len(tokenizer.encode(render_instruction(rec["student_prompt"], rec["response"])))
            resp = len(tokenizer.encode(rec["response"]))
            rt += full
            pt += full - resp
        return pt / rt if rt else 0.0
    except Exception:
        return float("nan")


class ByteTokenizer:
    """A trivial byte-level tokenizer for CPU/CI smoke tests.

    Doubles as a TokenizerLike (``encode``) and a TokenCounter (``count``). Byte ids are
    0-255; BOS/EOS ids passed to the data layer are 256/257 (fit the toy vocab of 512)."""

    name = "byte-v1"
    sha256 = "byte-v1"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))
