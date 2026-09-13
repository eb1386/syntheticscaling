"""Held-out NLL / bits-per-byte evaluation and task loss (benchmarks.md §5, statistics.md §1.4).

Two kinds of held-out set are supported:

* **token memmaps** (HeldOut-Real-ID / -OOD, TeacherVal-T): a flat ``uint16``/``uint32`` array of
  student-tokenizer ids with documents separated by ``eos_id``; every token is scored;
* **JSONL instruction sets** (HeldOut-Instr: Dolly / OASST1 / NQ-open): each line has ``prompt``
  and ``response``; the example is rendered with the *training* markers
  ``### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}`` (locked methodology §0.7,
  amendment B) and only the response tokens are scored (``scope="response"``); the
  ``"response_with_marker"`` scope adds the ``### Response:`` marker tokens, ``"all"`` scores the
  whole document.

Model access is a pluggable ``nll_fn(token_ids) -> per-token NLL (nats)`` of length
``len(token_ids) - 1`` (position t scores token t given tokens < t); ``torch_nll_fn`` wraps a
causal LM behind a lazy import.  Uncertainty is a document-level bootstrap SE of the
token-weighted mean.  ``task_loss_from_eval_json`` computes the bits-per-byte of the correct
continuation from the per-item log-likelihoods archived by the harness runner.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

LN2 = math.log(2.0)
INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "\n\n### Response:\n"


class NLLFn(Protocol):
    def __call__(self, token_ids: Sequence[int]) -> np.ndarray: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass
class DocScore:
    n_tokens: int  # scored tokens
    nll_sum: float  # sum of per-token NLL (nats) over scored tokens
    n_bytes: int  # UTF-8 bytes of the scored text


@dataclass
class NLLResult:
    set_name: str
    scope: str
    n_docs: int
    n_tokens: int
    n_bytes: int
    nll_per_token: float
    bits_per_byte: float
    se_nll_bootstrap: float
    se_bpb_bootstrap: float
    n_boot: int
    context_length: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------
def render_instruction(prompt: str, response: str) -> tuple[str, str, str]:
    """Return (prefix_up_to_and_including_marker, response_text, full_document).

    ``full = "### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}"``; the BOS/EOS tokens are
    added by the caller's tokenizer if it does so for training documents.
    """
    prefix = f"{INSTRUCTION_MARKER}{prompt}{RESPONSE_MARKER}"
    return prefix, response, prefix + response


# ----------------------------------------------------------------------------------------
# Scoring primitives
# ----------------------------------------------------------------------------------------
def score_tokens(
    ids: Sequence[int], nll_fn: NLLFn, context_length: int, *, start: int = 0, bos_id: Optional[int] = None
) -> np.ndarray:
    """Per-token NLL for positions ``start..len(ids)-1`` (each conditioned on all earlier tokens
    within a window of ``context_length``).

    Long documents are scored in disjoint windows of ``context_length`` tokens (the harness
    ``loglikelihood_rolling`` convention).  If ``bos_id`` is given it is prepended so that the
    first token is scored too; otherwise position 0 is unscored (as in training).
    """
    ids = list(ids)
    if bos_id is not None:
        ids = [bos_id] + ids
        start += 1
    n = len(ids)
    out = np.full(n, np.nan)
    pos = 1
    while pos < n:
        lo = max(0, pos - 1)  # window starts one token before the first scored position
        hi = min(n, lo + context_length)
        window = ids[lo:hi]
        nll = np.asarray(nll_fn(window), dtype=float)
        if len(nll) != len(window) - 1:
            raise ValueError("nll_fn must return len(ids)-1 values")
        out[lo + 1:hi] = nll
        pos = hi
    return out[start:]


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def score_jsonl_set(
    records: Iterable[dict[str, Any]],
    tokenizer: Tokenizer,
    nll_fn: NLLFn,
    *,
    context_length: int = 2048,
    scope: str = "response",
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> list[DocScore]:
    """Score a JSONL instruction set (fields ``prompt``, ``response``).

    Response-only scoring is exact at the token level: the prefix is tokenised separately and the
    response tokens are those of the full rendering beyond the prefix length (the two are
    concatenated by the tokenizer at a marker newline boundary, so the split is stable).
    """
    if scope not in ("response", "response_with_marker", "all"):
        raise ValueError(scope)
    scores: list[DocScore] = []
    for i, rec in enumerate(records):
        if max_docs is not None and i >= max_docs:
            break
        prefix, response, full = render_instruction(rec["prompt"], rec["response"])
        if scope == "all":
            ids = tokenizer.encode(full)
            start_tok = 0
            text = full
        else:
            cut = prefix if scope == "response" else prefix[: -len(RESPONSE_MARKER)] + "\n\n"
            ids_prefix = tokenizer.encode(cut)
            ids = tokenizer.encode(full)
            start_tok = len(ids_prefix)
            text = full[len(cut):]
        if eos_id is not None:
            ids = ids + [eos_id]
        # scored positions: start_tok .. end; position 0 is never scored unless bos_id given
        nll = score_tokens(ids, nll_fn, context_length, start=max(start_tok, 0 if bos_id is not None else 1),
                           bos_id=bos_id)
        nll = nll[~np.isnan(nll)]
        scores.append(DocScore(int(len(nll)), float(nll.sum()), _bytes(text) + (1 if eos_id is not None else 0)))
    return scores


def score_memmap_set(
    tokens: np.ndarray,
    nll_fn: NLLFn,
    *,
    eos_id: int,
    context_length: int = 2048,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    token_bytes: Optional[np.ndarray] = None,
    max_tokens: Optional[int] = None,
    bos_id: Optional[int] = None,
) -> list[DocScore]:
    """Score a flat token array split into documents at ``eos_id`` (EOS belongs to its document).

    Byte counts come from ``decode`` (exact) or from a per-vocabulary ``token_bytes`` table.
    """
    if decode is None and token_bytes is None:
        raise ValueError("need decode() or token_bytes for bits-per-byte")
    arr = np.asarray(tokens)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    if len(arr) == 0:
        return []
    ends = np.flatnonzero(arr == eos_id)
    if len(ends) == 0 or ends[-1] != len(arr) - 1:
        ends = np.append(ends, len(arr) - 1)  # trailing document without EOS
    starts = np.concatenate([[0], ends[:-1] + 1])
    scores: list[DocScore] = []
    for s, e in zip(starts, ends):
        ids = arr[s:e + 1].astype(np.int64).tolist()
        if len(ids) < 2 and bos_id is None:
            continue
        nll = score_tokens(ids, nll_fn, context_length, start=0 if bos_id is not None else 1, bos_id=bos_id)
        nll = nll[~np.isnan(nll)]
        if decode is not None:
            nb = _bytes(decode(ids))
        else:
            nb = int(np.asarray(token_bytes)[np.asarray(ids)].sum())
        scores.append(DocScore(int(len(nll)), float(nll.sum()), nb))
    return scores


# ----------------------------------------------------------------------------------------
# Aggregation and bootstrap
# ----------------------------------------------------------------------------------------
def aggregate(scores: Sequence[DocScore]) -> tuple[float, float, int, int]:
    """Token-weighted mean NLL (nats/token), bits-per-byte, n_tokens, n_bytes."""
    nt = sum(s.n_tokens for s in scores)
    nb = sum(s.n_bytes for s in scores)
    tot = sum(s.nll_sum for s in scores)
    nll = tot / nt if nt else float("nan")
    bpb = tot / (LN2 * nb) if nb else float("nan")
    return nll, bpb, nt, nb


def bootstrap_se(scores: Sequence[DocScore], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Document-level bootstrap SE of (nll/token, bits/byte)."""
    if len(scores) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nt = np.array([s.n_tokens for s in scores], dtype=float)
    nb = np.array([s.n_bytes for s in scores], dtype=float)
    tot = np.array([s.nll_sum for s in scores], dtype=float)
    n = len(scores)
    idx = rng.integers(0, n, size=(n_boot, n))
    t_sum = tot[idx].sum(1)
    nll = t_sum / np.maximum(nt[idx].sum(1), 1)
    bpb = t_sum / (LN2 * np.maximum(nb[idx].sum(1), 1))
    return float(nll.std(ddof=1)), float(bpb.std(ddof=1))


def summarize(set_name: str, scores: Sequence[DocScore], *, scope: str, context_length: int,
              n_boot: int = 1000, seed: int = 0) -> NLLResult:
    nll, bpb, nt, nb = aggregate(scores)
    se_nll, se_bpb = bootstrap_se(scores, n_boot=n_boot, seed=seed)
    return NLLResult(set_name, scope, len(scores), nt, nb, nll, bpb, se_nll, se_bpb, n_boot, context_length)


def write_nll_result(results_dir: str | Path, experiment_id: str, result: NLLResult) -> Path:
    name = result.set_name if result.scope in ("response", "all") else f"{result.set_name}_{result.scope}"
    out = Path(results_dir) / experiment_id / "eval" / f"nll_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=1))
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ----------------------------------------------------------------------------------------
# Task loss from harness per-item log-likelihoods
# ----------------------------------------------------------------------------------------
def task_loss_from_items(per_item: Sequence[dict[str, Any]], output_type: str) -> np.ndarray:
    """Per-item bits-per-byte of the correct continuation (NaN where undefined)."""
    out = np.full(len(per_item), np.nan)
    for i, it in enumerate(per_item):
        if output_type == "multiple_choice":
            g, lls, nb = it.get("gold"), it.get("lls") or [], it.get("choice_bytes") or []
            if isinstance(g, int) and 0 <= g < len(lls) and g < len(nb) and nb[g] > 0:
                out[i] = -lls[g] / (LN2 * nb[g])
        elif output_type == "loglikelihood":
            nb, ll = it.get("target_bytes", 0), it.get("ll")
            if nb and ll is not None and not math.isnan(ll):
                out[i] = -ll / (LN2 * nb)
    return out


def task_loss_from_eval_json(path: str | Path) -> dict[str, float]:
    """Mean ± SE bits-per-byte of the correct continuation for one archived task JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    vals = task_loss_from_items(doc.get("per_item", []), doc.get("output_type", "multiple_choice"))
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"task_loss_bpb": float(doc.get("task_loss_bpb", float("nan"))),
                "task_loss_bpb_se": float(doc.get("task_loss_bpb_se", float("nan"))), "n": 0}
    return {"task_loss_bpb": float(vals.mean()),
            "task_loss_bpb_se": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            "n": int(len(vals))}


# ----------------------------------------------------------------------------------------
# torch adapter (lazy)
# ----------------------------------------------------------------------------------------
def torch_nll_fn(model: Any, device: str = "cuda", dtype: Optional[str] = "bfloat16") -> NLLFn:
    """Wrap a causal LM (HF-style ``model(input_ids).logits`` or ``model(x) -> logits``)."""
    import torch  # type: ignore  # lazy

    model = model.to(device)
    model.eval()
    dt = getattr(torch, dtype) if dtype else None

    @torch.no_grad()
    def fn(token_ids: Sequence[int]) -> np.ndarray:
        x = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        ctx = torch.autocast(device_type=device.split(":")[0], dtype=dt) if dt and device != "cpu" else _nullctx()
        with ctx:
            out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = x[0, 1:]
        nll = -logp.gather(1, tgt[:, None])[:, 0]
        return nll.cpu().numpy()

    return fn


class _nullctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None
