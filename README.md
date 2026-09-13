# Synthetic Data Scaling Laws for Small Language Models

**Question.** When you train a small language model on synthetic data written by a teacher
model, does the data follow a *scaling law* — does loss keep falling predictably as you add
more synthetic tokens — and **does that scaling law depend on how big the teacher is?**

This repository runs that experiment end to end on a **single RTX 5080 (16 GB)**. It trains
a family of small students (25M–250M) on synthetic data from four teachers (0.5B–7B), and
fits the data-scaling law `L(D) = E + A·D^(−α)` for every student × teacher cell, where `D`
is the number of synthetic tokens. It then tests whether the exponent `α`, the coefficient
`A`, and the irreducible loss `E` change with teacher size, and whether synthetic data scales
like an equal amount of real web text.

Everything is measured in **loss** (not accuracy), which is the right, floor-free quantity
for scaling laws and is what makes this feasible at 25M–250M parameters.

## Run it (two commands)

```bash
./install.sh scaling5080     # deps + tokenizer + data + teacher models + prompt pool (one time)
./run_all.sh scaling5080     # runs the whole study; resumable — re-run to continue
```

Before committing days of GPU time, validate the whole pipeline and see the time estimate:

```bash
make micro                        # CPU, minutes: runs every stage on a toy model (no GPU)
make smoke                        # GPU, ~1-3h: real Qwen teachers, tiny budget
make budget PROFILE=scaling5080   # prints the GPU-hour / wall-clock estimate
```

Run inside `tmux`/`screen` so it survives disconnects.

## How long on one 5080

`make budget` (numbers are GPU-hours = wall-clock, since one card runs serially):

| Profile | Scope | Wall-clock |
|---|---|---|
| `scaling5080` | 4 students (25M–250M) × 4 teachers (0.5–7B) + real/human controls, 3 seeds | **~23 days (~28 with overhead)** |
| `scaling5080_fast` | 3 students (25M–100M) × 4 teachers, 2 seeds | **~7 days (~9 with overhead)** |
| `smoke` | real models, tiny budget | ~1–3 hours |
| `micro` | CPU validation, toy model | minutes |

Start with `scaling5080_fast` for a complete first result in a week; run `scaling5080` for the
full-depth version. The data-quantity axis `D` is read from each run's training curve, so you
get ~20 points on the scaling curve per cell for free — no extra runs. See
[`docs/RUN_ON_5080.md`](docs/RUN_ON_5080.md) for the scoping, knobs, and what needs a bigger card.

## What you get

After a run, `results/<profile>/analysis/` contains:

- **`scaling_instr/scaling_fits.csv`** — `E`, `A`, `α` for every student × teacher cell (the scaling laws).
- **`scaling_instr/scaling_report.json`** — the teacher-dependence tests (does `α`/`E` move with teacher size?) and the synthetic-vs-real comparison.
- **`cell_table.csv`, `transfer_curves_*.png`, `sxt_heatmap_*.png`** — the accuracy view (secondary).
- One `manifest.json` per run with git commit, config hash, seeds, and hardware (full provenance).

## What the experiment varies (the axes)

| Axis | Values | Why |
|---|---|---|
| Teacher size `T` | Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B (FP8, quantized on load) | the question is about teacher size |
| Student size `S` | 25M, 50M, 100M, 250M (one LLaMA-style family, shared tokenizer) | does the answer depend on the student? |
| Synthetic tokens `D` | ~20 points from 25M up to 400M, read from the training curve | the scaling-law axis |
| Controls | C1 = equal real web tokens, C1b = human Q&A, C0 = base model | is synthetic data better than real data, and than nothing? |

Teacher data is generated once per teacher under one frozen prompt pool, one decoding config,
and teacher-agnostic filters, at equal student-tokenizer tokens — so the **only** thing that
differs between teacher conditions is the teacher (see `docs/09`).

## Repository layout

```
synscale/            the package
  config/            config schemas, YAML loader, hardware/scope profiles
  models/            LLaMA-style student (RMSNorm, RoPE, SwiGLU, GQA) + exact param counts
  generation/        prompt pool, vLLM teacher backend, filters, subsampling, runner
  training/          WSD trainer, deterministic data/branch builder, schedules
  evaluation/        self-contained multiple-choice + NLL scorer
  analysis/          scaling-law fits, aggregates, T*/STE, plots, compute budget
  pipeline.py        the orchestrator that runs the whole study from a profile
scripts/             install/prepare/generate/train/evaluate/analyze entry points
configs/             student, teacher, training, generation, evaluation fragments
docs/                the full methodology, scaling-law method (docs/22), run guide, budget
paper/               abstract / introduction / methods drafts
tests/               60 unit tests (CPU; torch/vllm tests skip if unavailable)
```

Install: `pip install -e ".[train,gen,eval,analysis,dev]"`. Tests: `make test`.

## Status and honesty

The whole pipeline runs end to end (verified on CPU with `make micro`; 60 tests pass). It has
**not** yet been run on a GPU here — the throughput/time numbers are formula estimates marked
in `synscale/analysis/compute_budget.py` and should be confirmed by the pilot on day one
(`docs/05`). What remains to turn a launch into a paper is listed in `docs/25`. A single 5080
cannot serve teachers above ~7B or train students above ~250M in reasonable time; the full
teacher/student range needs a bigger or rented card (`docs/RUN_ON_5080.md`).

Full methodology and design rationale: [`docs/00_locked_methodology.md`](docs/00_locked_methodology.md).
Scaling-law analysis method: [`docs/22_scaling_laws_analysis.md`](docs/22_scaling_laws_analysis.md).
