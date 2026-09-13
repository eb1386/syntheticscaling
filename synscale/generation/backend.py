"""Generation backends: a teacher-agnostic interface plus a vLLM implementation and a mock.

The backend is the ONLY teacher-indexed component of the generation pipeline
(doc 00 §0.7). Everything else (prompt pool, system prompt, decoding config,
filters, subsampler) is frozen and shared across teachers.

`GenerationBackend.generate` takes a list of chat-message lists (already including
the study system prompt) and one `SamplingSpec`, and returns one `Generation` per
request in input order. Determinism at the dataset level is guaranteed downstream by
hashing (doc 18); per-request seeds make a run replayable on identical hardware.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence


@dataclass(frozen=True)
class SamplingSpec:
    """Decoding parameters shared by every teacher in a study (doc 00 §0.7)."""

    temperature: float
    top_p: float
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    n: int = 1
    seed: int = 0
    stop_token_ids: Sequence[int] = field(default_factory=tuple)
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    @classmethod
    def from_generation_config(cls, gc: Any) -> "SamplingSpec":
        extra = getattr(gc, "backend_extras", {}) or {}
        return cls(
            temperature=gc.temperature,
            top_p=gc.top_p,
            top_k=gc.top_k,
            repetition_penalty=gc.repetition_penalty,
            max_new_tokens=gc.max_new_tokens,
            n=gc.samples_per_prompt,
            seed=gc.sampling_seed,
            stop_token_ids=tuple(gc.stop_token_ids),
            min_p=float(extra.get("min_p", 0.0)),
            presence_penalty=float(extra.get("presence_penalty", 0.0)),
            frequency_penalty=float(extra.get("frequency_penalty", 0.0)),
        )


@dataclass
class Generation:
    text: str
    finish_reason: str            # "stop" | "length"
    n_prompt_tokens: int          # teacher tokenizer
    n_response_tokens: int        # teacher tokenizer
    cumulative_logprob: float = 0.0


def request_seed(base_seed: int, pool_version: str, prompt_id: str, sample_idx: int) -> int:
    """Deterministic per-request seed: H(pool_version, prompt_id, sample_idx) with base offset."""
    h = hashlib.blake2b(f"{pool_version}:{prompt_id}:{sample_idx}".encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") ^ (base_seed & 0xFFFFFFFF)) % (2 ** 31)


class GenerationBackend(Protocol):
    name: str

    def generate(self, messages: Sequence[Sequence[dict]], spec: SamplingSpec,
                 seeds: Optional[Sequence[int]] = None) -> list[Generation]: ...

    def info(self) -> dict[str, Any]: ...


class MockBackend:
    """Deterministic pseudo-teacher for tests and CPU smoke runs (no GPU, no network).

    Produces responses whose length and 'quality' scale with a configurable `skill`
    (a stand-in for teacher size) so the whole pipeline, filters and subsampler can be
    exercised without a real model.
    """

    def __init__(self, skill: float = 1.0, verbosity: int = 40, refusal_rate: float = 0.0,
                 cjk_rate: float = 0.0, name: str = "mock"):
        self.skill = skill
        self.verbosity = verbosity
        self.refusal_rate = refusal_rate
        self.cjk_rate = cjk_rate
        self.name = name

    def generate(self, messages, spec, seeds=None):
        import random
        out: list[Generation] = []
        seeds = seeds or [spec.seed] * len(messages)
        for msg, sd in zip(messages, seeds):
            rng = random.Random(sd)
            user = next((m["content"] for m in msg if m["role"] == "user"), "")
            n_words = max(8, int(self.verbosity * (0.7 + 0.6 * rng.random())))
            if rng.random() < self.refusal_rate:
                text, finish = "I'm sorry, but I cannot help with that request.", "stop"
            elif rng.random() < self.cjk_rate:
                text, finish = "答案是 " + " ".join(["word"] * n_words), "stop"
            else:
                # A response whose informativeness rises with skill (more distinct words).
                # Varied pseudo-words avoid a constant prefix that would trip the
                # repetition filter; a larger skill draws from a larger vocabulary.
                vocab = max(8, int(8 + 120 * self.skill))
                cons, vows = "bcdfghklmnprstvwz", "aeiou"
                def word():
                    k = rng.randrange(vocab)
                    r = random.Random(k)
                    return "".join(r.choice(cons) + r.choice(vows) for _ in range(r.randint(2, 4)))
                words = [word() for _ in range(n_words)]
                body = " ".join(words)
                text = f"The answer relates to {words[0]} and {words[min(1,len(words)-1)]}. {body}. Final answer: {rng.randrange(100)}"
                finish = "length" if n_words >= spec.max_new_tokens else "stop"
            npt = len(user.split())
            nrt = len(text.split())
            out.append(Generation(text=text, finish_reason=finish, n_prompt_tokens=npt,
                                  n_response_tokens=nrt, cumulative_logprob=-0.5 * nrt / max(self.skill, 0.1)))
        return out

    def info(self):
        return {"backend": "mock", "skill": self.skill, "verbosity": self.verbosity}


class VLLMBackend:
    """vLLM offline batch generation. Import is lazy so the module loads without vLLM.

    Serving precision is set by `quantization` (None=bf16, 'awq_marlin', 'fp8'); on a
    single 16 GB card the local profile uses AWQ int4 for teachers >3B (doc RUN_ON_5080).
    The study system prompt must already be the first message of every request.
    """

    def __init__(self, hf_repo: str, revision: Optional[str] = None, *, dtype: str = "bfloat16",
                 quantization: Optional[str] = None, tensor_parallel: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.92, max_num_seqs: int = 256,
                 kv_cache_dtype: str = "auto", enforce_eager: bool = False, seed: int = 0):
        self.hf_repo = hf_repo
        self.revision = revision
        self.dtype = dtype
        self.quantization = quantization
        self.tensor_parallel = tensor_parallel
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.kv_cache_dtype = kv_cache_dtype
        self.enforce_eager = enforce_eager
        self.seed = seed
        self.name = hf_repo
        self._llm = None
        self._tok = None

    def _ensure(self):
        if self._llm is not None:
            return
        from vllm import LLM  # lazy
        from transformers import AutoTokenizer
        kw: dict[str, Any] = dict(
            model=self.hf_repo, dtype=self.dtype, tensor_parallel_size=self.tensor_parallel,
            max_model_len=self.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs, kv_cache_dtype=self.kv_cache_dtype,
            enforce_eager=self.enforce_eager, seed=self.seed,
        )
        if self.revision and self.revision not in ("main", "PIN_ME_TO_COMMIT_SHA"):
            kw["revision"] = self.revision
        if self.quantization:
            kw["quantization"] = self.quantization
        self._llm = LLM(**kw)
        self._tok = AutoTokenizer.from_pretrained(self.hf_repo, revision=kw.get("revision"))

    def generate(self, messages, spec, seeds=None):
        self._ensure()
        from vllm import SamplingParams
        seeds = seeds or [spec.seed] * len(messages)
        prompts = [self._tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True)
                   for m in messages]
        params = [SamplingParams(
            temperature=spec.temperature, top_p=spec.top_p,
            top_k=(spec.top_k if spec.top_k and spec.top_k > 0 else -1),
            repetition_penalty=spec.repetition_penalty, max_tokens=spec.max_new_tokens, n=spec.n,
            min_p=spec.min_p, presence_penalty=spec.presence_penalty,
            frequency_penalty=spec.frequency_penalty,
            stop_token_ids=list(spec.stop_token_ids) or None, seed=int(sd),
        ) for sd in seeds]
        results = self._llm.generate(prompts, params, use_tqdm=True)
        out: list[Generation] = []
        for req, prompt in zip(results, prompts):
            o = req.outputs[0]
            finish = "length" if o.finish_reason == "length" else "stop"
            out.append(Generation(
                text=o.text, finish_reason=finish,
                n_prompt_tokens=len(req.prompt_token_ids),
                n_response_tokens=len(o.token_ids),
                cumulative_logprob=float(o.cumulative_logprob or 0.0)))
        return out

    def info(self):
        return {"backend": "vllm", "hf_repo": self.hf_repo, "revision": self.revision,
                "dtype": self.dtype, "quantization": self.quantization,
                "tensor_parallel": self.tensor_parallel}


def build_backend(kind: str, **kw: Any) -> GenerationBackend:
    if kind == "mock":
        return MockBackend(**{k: v for k, v in kw.items() if k in {"skill", "verbosity", "refusal_rate", "cjk_rate", "name"}})
    if kind == "vllm":
        return VLLMBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}")
