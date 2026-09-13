"""Generation runtime + subsample + dataset builder + micro end-to-end (CPU, MockBackend)."""
import os, tempfile
import pytest
from synscale.generation import prompt_pool as pp
from synscale.generation.taxonomy import Taxonomy
from synscale.generation.backend import build_backend, SamplingSpec, request_seed
from synscale.generation.tokens import ApproxTokenCounter
from synscale.generation.runner import run_generation, iter_generated
from synscale.generation.filters import ResponseFilter
from synscale.generation.subsample import Example, subsample_equal_tokens, subsample_nested_d
from synscale.generation.build_dataset import build_treatment_stream, ByteTokenizer

REPO = os.path.dirname(os.path.dirname(__file__))
TAX = os.path.join(REPO, "data", "prompts", "taxonomy_v1.yaml")


def _pool(d, n=1500):
    tax = Taxonomy.from_yaml(TAX)
    gens = {pp.SOURCE_TEMPLATED: pp.TemplatedPromptGenerator(tax)}
    spec = pp.PoolBuildSpec(n_prompts=n, seed=0, version="t-v0",
                            source_mix={c: {pp.SOURCE_TEMPLATED: 1.0} for c in pp.CATEGORIES})
    res = pp.PoolBuilder(spec, gens, taxonomy=tax).build()
    path = os.path.join(d, "pool.jsonl")
    pp.freeze_pool(res.records, path, manifest=res.manifest)
    return path


def test_request_seed_deterministic():
    assert request_seed(0, "v", "0000000001", 0) == request_seed(0, "v", "0000000001", 0)
    assert request_seed(0, "v", "0000000001", 0) != request_seed(0, "v", "0000000002", 0)


def test_filters_drop_refusal_and_short():
    rf = ResponseFilter()
    r = rf.apply("I'm sorry, but I cannot help.", category="sci", finish_reason="stop", n_student_tokens=40)
    assert not r.retained and r.drop_reason == "refusal"


def test_subsample_nested_prefix():
    exs = [Example(f"{i:010d}", ["sci", "msr"][i % 2], 100, 80, i) for i in range(2000)]
    nested = subsample_nested_d(exs, [20000, 60000], seed=0)
    small = {e.prompt_id for e in nested[20000]}
    big = {e.prompt_id for e in nested[60000]}
    assert small <= big


def test_generation_and_dataset_end_to_end():
    d = tempfile.mkdtemp()
    pool = _pool(d)
    man = run_generation(pool, build_backend("mock", skill=2.0, verbosity=45),
                         SamplingSpec(0.7, 0.8, max_new_tokens=200), ApproxTokenCounter(),
                         os.path.join(d, "gen"), teacher_id="t1", pool_version="t-v0",
                         target_student_tokens=15000, shard_size=200)
    assert man["retained_student_tokens"] >= 15000 * 1.25 * 0.5
    assert 0.5 < man["filter_stats"]["yield_frac"] <= 1.0
    meta = build_treatment_stream(os.path.join(d, "gen"), os.path.join(d, "treat"),
                                  ByteTokenizer(), 256, 257, d_syn=8000, seed=1)
    assert meta["n_examples"] > 0 and meta["n_tokens"] > 0
