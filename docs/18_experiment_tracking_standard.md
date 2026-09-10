# 18. Experiment-Tracking Standard

Status: locked (methodological choice). Implemented in `synscale/tracking/`.

## 18.1 Principles

1. **A result is only a result if it has a manifest.** No number enters an analysis table unless it is read from `results/<experiment_id>/manifest.json` and `results/<experiment_id>/eval/*.json` with matching `experiment_id` and `config_hash`.
2. **Identity is a function of the resolved configuration.** Two launches with byte-identical resolved configs produce the same experiment ID; any change to a controlled quantity produces a different one. Hardware and runtime are recorded but never contribute to identity.
3. **Everything that varies across teacher conditions is explicit.** The only fields permitted to differ between two `synthetic`-arm runs at the same student size and seed are `teacher.*` and `dataset_ref`/`dataset_sha256`. A CI test (`tests/test_matrix_invariants.py`, to be added with the experiment matrix) asserts this over the whole matrix.
4. **Final checkpoint only.** `checkpoint_rule: final` is the only accepted value; best-of-k checkpoint selection on benchmarks is disallowed because it biases every cell upward by an amount that depends on run-to-run noise.

## 18.2 Experiment ID convention

```
<study>.<student>.<source>.d<Dsyn>.<variant>.s<seed>-<hash8>
```

| Field | Values | Meaning |
|---|---|---|
| `study` | `pilot`, `main`, `dsweep`, `ablation` | which pre-registered study the run belongs to |
| `student` | `s025m`, `s100m`, `s250m`, `s500m`, `s1b` | student family member |
| `source` | `t1b`…`t72b` (teacher id), `base`, `real`, `human` | teacher condition or control arm |
| `d<Dsyn>` | `d0`, `d125m`, `d500m`, … | second-phase token budget in student tokens |
| `variant` | `eqtok`, `eqex`, `lenmatch`, `-` | budget-control variant; `-` for base-only |
| `s<seed>` | `s1`, `s2`, … | second-phase seed (data order) |
| `hash8` | 8 hex chars | SHA-256 prefix of the canonical resolved config |

Examples:

```
pilot.s100m.t7b.d150m.eqtok.s2-3fa9c2d1
main.s500m.base.d0.-.s1-9b12ee40
main.s500m.real.d500m.eqtok.s1-77ac0b3e
ablation.s100m.t32b.d150m.lenmatch.s1-c01d77aa
```

The hash excludes only the `notes` field. Seeds are part of the hash, so seed replicates have distinct IDs and distinct hashes.

## 18.3 Run manifest (`RunManifest`)

Written at launch and completed at exit. Fields:

| Group | Fields | Source |
|---|---|---|
| identity | `experiment_id`, `config_hash` | `synscale.tracking.ids` |
| config | full resolved `ExperimentConfig` (student, teacher, generation, training, evaluation, arm, variant) | YAML composition |
| git | `commit`, `dirty`, `branch` | `git` at launch; `dirty=true` runs are rejected by the launcher for `main`/`pilot` studies |
| seeds | `base`, `phase`, `eval`, `sampling` | config |
| provenance | `base_checkpoint_sha256`, `dataset.{path, sha256, n_examples, n_student_tokens, prompt_pool_sha256, generation_manifest}` | dataset builder |
| hardware | `gpu_name`, `gpu_count`, `gpu_mem_gb`, `driver`, `cuda`, `torch`, `hostname`, `cpu` | probed at launch |
| runtime | `started_at`, `finished_at`, `wall_seconds`, `gpu_seconds`, `tokens_per_second`, `mfu`, `energy_kwh`, `status` | trainer |
| software | `python`, package lock hash, `pid` | launcher |
| metrics | final train loss, held-out NLLs, throughput | trainer |
| eval | `eval_results_path` → `results/<id>/eval/<task>.json` (lm-eval raw outputs incl. per-item log-likelihoods) | evaluator |

Generation runs have their own manifest (`GenerationManifest`, to be added in `synscale/generation/`): teacher repo + revision SHA, weight dtype, vLLM version, decoding config, prompt-pool SHA, per-example JSONL with `prompt_id, category, teacher, n_tokens_teacher, n_tokens_student, gen_time_s, finish_reason, filter_flags`, and aggregate `gpu_seconds`, `tokens_per_second`, `energy_kwh` (from `nvidia-smi --query-gpu=power.draw` sampled at 1 Hz).

## 18.4 Seeds

| Seed | Controls | Varies across |
|---|---|---|
| `seed_base` | base-phase initialization and base data order | base replicates only (pilot: 3 per size; main: 1 per size, see 10_student_training) |
| `seed_phase` | second-phase data order (and dropout if any; none in this family) | every replicate |
| `sampling_seed` | vLLM sampling seed for generation | fixed per teacher condition (same value for every teacher) |
| `eval_seed` | any stochastic evaluation component | fixed globally |

Subsampling of the synthetic pool to `target_student_tokens` uses `hash(prompt_id) mod 2^32` ordering, so the subsample is a deterministic function of the pool and the budget, not of the seed.

## 18.5 Results layout

```
results/
  <experiment_id>/
    manifest.json
    train_log.jsonl        # step, tokens, loss, lr, tok/s, mfu
    eval/
      <task>.json          # lm-eval output with per-item loglikelihoods
      nll_<set>.json
    checkpoint_sha256.txt  # sha256 of the final checkpoint (checkpoint itself lives outside git)
  generation/
    <teacher>.<genconfig>.<poolsha8>/
      manifest.json
      samples.jsonl.zst
      stats.json           # length distribution, filter rates, throughput, energy
  index.parquet            # flat table rebuilt by `scripts/build_index.py`, one row per run
```

Checkpoints and datasets are stored on local NVMe (`$SYNSCALE_STORE`) and referenced by SHA-256; git holds configs, code, manifests, small eval JSON, and the analysis notebooks.

## 18.6 Launch invariants (enforced by `scripts/launch.py`)

- refuses to launch a `pilot`/`main` run from a dirty git tree;
- refuses to launch if `results/<experiment_id>/manifest.json` exists with `status: finished` (idempotent re-launch);
- verifies `dataset_sha256` and `base_checkpoint_sha256` before training;
- pins `lm-eval` and `vllm` versions to those in `configs/evaluation/*.yaml` and `configs/generation/*.yaml`, failing on mismatch.
