#!/usr/bin/env python3
"""Download and tokenise everything the real single-5080 run needs (docs/RUN_ON_5080.md).

Writes, under data/processed/<profile>/, the four document-stream stems the pipeline expects:
  base_train, base_heldout, c1b_stackexchange, nll_instr
plus the pinned tokenizer at data/tokenizer/tokenizer.json, and pre-caches the benchmark
datasets used by the self-contained evaluator.

Sources (all permissive; see docs/09):
  base corpus     HuggingFaceFW/fineweb-edu (ODC-BY), streamed to a token budget
  C1b Q&A         a StackExchange Q&A dataset (CC BY-SA), non-code
  human instr     databricks-dolly-15k (CC BY-SA) + OpenAssistant/oasst1 (Apache-2.0)
  benchmarks      ai2_arc, sciq, piqa, openbookqa, hellaswag, social_i_qa, lambada_openai

Requires: transformers, tokenizers, datasets, numpy, huggingface_hub. Network access.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from synscale.config.profiles import get_profile
from synscale.training.data import write_document_file

TOKENIZER_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ungated Llama-2 32k SentencePiece


def get_tokenizer(data_dir: Path):
    from tokenizers import Tokenizer
    tok_dir = data_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_json = tok_dir / "tokenizer.json"
    if not tok_json.exists():
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        hf.backend_tokenizer.save(str(tok_json))
        print(f"[prepare] saved tokenizer -> {tok_json}")
    return Tokenizer.from_file(str(tok_json))


class _Tok:
    def __init__(self, t): self._t = t
    def encode(self, text): return self._t.encode(text, add_special_tokens=False).ids


def _stream_docstream(texts, tok, bos, eos, stem: Path, max_tokens: int, log_every=100_000):
    """Tokenise a text iterator into a uint16 memmap + doc_starts, stopping at max_tokens."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    buf, starts, total = [], [], 0
    for i, text in enumerate(texts):
        ids = [bos] + tok.encode(text) + [eos]
        starts.append(total); buf.append(np.asarray(ids, dtype=np.uint16)); total += len(ids)
        if total >= max_tokens:
            break
        if i % log_every == 0 and i:
            print(f"[prepare] {stem.name}: {total/1e6:.1f}M tokens", flush=True)
    tokens = np.concatenate(buf); doc_starts = np.asarray(starts, dtype=np.int32)
    write_document_file(stem, tokens, doc_starts, extra={"n_docs": int(doc_starts.size), "n_tokens": int(tokens.size)})
    print(f"[prepare] wrote {stem} ({tokens.size/1e6:.1f}M tokens, {doc_starts.size} docs)")


def prepare(profile_name: str, data_dir: Path, base_train_tokens: int, heldout_tokens: int):
    from datasets import load_dataset
    p = get_profile(profile_name)
    tok = _Tok(get_tokenizer(data_dir))
    bos, eos = 1, 2
    proc = data_dir / "processed" / profile_name
    need = max(p.base_tokens.values())
    base_train_tokens = max(base_train_tokens, need + p.d2_tokens)  # +replay headroom

    # base corpus (FineWeb-Edu, streamed)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it = (r["text"] for r in fw)
    _stream_docstream(it, tok, bos, eos, proc / "base_train", base_train_tokens)
    fw2 = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    it2 = (r["text"] for r in fw2.skip(400_000))   # disjoint held-out shard for C1
    _stream_docstream(it2, tok, bos, eos, proc / "base_heldout", heldout_tokens)

    # C1b StackExchange Q&A (fallback to dolly if unavailable)
    try:
        se = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
        texts = (f"### Instruction:\n{r['question']}\n\n### Response:\n{(r.get('answers') or [{'text':''}])[0].get('text','')}"
                 for r in se)
        _stream_docstream(texts, tok, bos, eos, proc / "c1b_stackexchange", heldout_tokens)
    except Exception as e:
        print(f"[prepare] stack-exchange unavailable ({e}); using dolly as C1b fallback")
        d = load_dataset("databricks/databricks-dolly-15k", split="train")
        texts = [f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['response']}" for r in d]
        _stream_docstream(iter(texts * 50), tok, bos, eos, proc / "c1b_stackexchange", heldout_tokens)

    # human-instruction NLL set (dolly + oasst1 English first replies)
    d = load_dataset("databricks/databricks-dolly-15k", split="train")
    texts = [f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['response']}" for r in d]
    _stream_docstream(iter(texts), tok, bos, eos, proc / "nll_instr", 3_000_000)

    # pre-cache benchmarks (the evaluator loads them at eval time)
    from synscale.evaluation.benchmarks import get_loader
    loader = get_loader(use_fixtures=False)
    for task in ("sciq", "arc_easy", "piqa", "openbookqa", "hellaswag", "social_iqa", "lambada_openai"):
        try:
            n = len(loader(task, limit=10)); print(f"[prepare] benchmark {task}: reachable ({n} sample items)")
        except Exception as e:
            print(f"[prepare] WARNING benchmark {task} not cached: {e}")
    print("[prepare] done.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="local5080")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base-train-tokens", type=int, default=25_000_000_000,
                    help="tokens to tokenise for the base corpus (>= largest base budget + replay)")
    ap.add_argument("--heldout-tokens", type=int, default=1_000_000_000)
    a = ap.parse_args()
    prepare(a.profile, Path(a.data_dir), a.base_train_tokens, a.heldout_tokens)


if __name__ == "__main__":
    main()
