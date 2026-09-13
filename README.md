# What Governs Synthetic-Data Scaling for Small Language Models

When a small language model is trained on synthetic data written by a larger teacher, does the
data obey a scaling law, and what actually controls it? This repository is a controlled
experiment that asks whether synthetic-data scaling is governed by teacher size, by teacher
family, or by measurable properties of the data itself, measured relative to the student.

The framing matters. "Which teacher size is best" is a question the field has largely answered
in direction, so it is incremental. This study instead tries to identify the variable that
governs the scaling and to test whether the identity of the teacher is screened off by it. That
is a mechanism question, and it has not, to our knowledge, been answered cleanly for
sequence-level synthetic data at pretraining scale.

Honesty first. The code targets novel questions and is built to a pre-registered, controlled
standard, but novelty is proven by results, not by ambition, and acceptance at a venue also
depends on the writeup. Every test here is designed so that a clean negative is a real finding.

## The question, precisely

We measure held-out loss as a function of student size, teacher, and synthetic-token count,

```
L = f(S, T, D)
```

and fit the data-scaling curve `L(D) = E + A·D^(−α)` for every cell, where `E` is the estimated
asymptote over the observed range, `A` the coefficient, and `α` the exponent. Then we ask five
things, each falsifiable.

1. **Rate versus level.** Does the teacher change the exponent `α`, or only the asymptote `E`?
2. **Screening, the centerpiece.** We build a student-referenced data-quality index `q` from
   three measured properties of each teacher's data: its learnability (how well the student's
   own base model already predicts that data), its diversity, and its correctness. We then ask
   whether teacher identity, meaning size and family, is screened off by `q`. If `q` predicts a
   cell's scaling as well as identity does, and identity adds nothing beyond `q`, then the
   scaling is governed by measurable data properties, not by which model wrote the data.
3. **A predictive law.** We fit `L(S, T, D)` on a subset of cells and predict held-out cells
   out of sample, turning measured curves into a law that transfers.
4. **Compute-optimal allocation.** At a fixed total compute budget split between generating
   teacher tokens and training the student, which teacher size minimizes loss, and how does the
   optimum move with student size?
5. **Cross-family invariance.** Is the synthetic-data multiplier, real tokens per synthetic
   token at matched loss, a function of `q` alone, invariant to teacher family?

Full design and rationale are in `docs/24_novel_experiment.md`.

## Why loss, matched checkpoints, and a data index

Everything is measured in loss, not accuracy. Loss is the quantity scaling laws are defined on,
it moves smoothly from the first tokens, and it has far higher signal to noise than
multiple-choice accuracy, which sits at chance for the smallest students. Accuracy is kept only
as a secondary transfer-breadth check.

Each student is pretrained once on real text to a stable checkpoint. Every condition branches
from that same checkpoint into an identical short annealing phase whose only difference is the
data it sees: synthetic text from one teacher, an equal number of fresh real tokens, human
question-answer text, or nothing new. A difference in outcome is therefore attributable to the
data source alone. The data-quantity axis `D` is read from the training curve, so one run yields
about 20 points on the scaling curve for free.

The data-quality index `q` is what makes the screening question answerable. Its key term is
learnability, the per-token loss of a teacher's data under the specific student's base model.
This is a student-referenced compatibility measure, so `q` is student-specific, exactly what a
capacity-dependent theory predicts.

## The experiment (C4)

| Axis | Values |
|---|---|
| Students `S` | 25M, 50M, 100M, 250M, 500M, 1B, one family, shared 16k tokenizer |
| Teacher family A | Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, across all students |
| Teacher family B | Llama-3.x 3B, 8B, 70B, at 100M and 1B, the cross-family screen |
| Data `D` | about 20 points from 25M to 400M, from the training curve |
| Controls | C1 equal real tokens, C1b human Q&A, C0 base |

Teachers are served at uniform FP8 precision, quantized on load, with 70B and 72B running across
two GPUs. The partial second family breaks the size-versus-family confound at two student sizes
without doubling the whole grid, which is what keeps C4 affordable. Total: 188 training runs.

## Running it on RunPod

The study is targeted at rented H100s. Two commands. The first installs dependencies, trains the
tokenizer, downloads and tokenizes the corpora, fetches the teacher models, and builds the
prompt pool onto a persistent volume. The second runs the whole study and is resumable, so a
reclaimed spot instance costs almost nothing.

```bash
./install.sh c4
./run_all.sh c4
```

Cost and time, from the planner (`python -m synscale.analysis.config_planner`), at September
2026 rates:

| Setup | GPU-hours | Cost | Wall-clock |
|---|---|---|---|
| C4 on H100 spot (~$1.49/hr) | ~1,185 | **~$1,770** | ~12 days on 4 GPUs in parallel |
| C4 on H100 on-demand (~$2.50/hr) | ~1,185 | ~$2,960 | ~12 days on 4 GPUs |

Cost is billable GPU-hours times the rate, so renting several GPUs in parallel cuts wall-clock
for the same total dollars. The cheapest reliable plan is a hybrid: run the few long base-training
jobs on on-demand, and the many short generation and branch jobs on spot, all on a persistent
volume so reclaims do not lose work. Validate first without spending days:

```bash
make micro                   # CPU, minutes: runs every stage, including the data index and novelty analyses
make budget PROFILE=c4       # GPU-hours and cost estimate
make smoke                   # GPU, about 1 to 3 hours: real teachers, tiny budget
```

Run `scaling5080_fast` on a cheap card first as the pilot to measure real throughput and effect
sizes, then commit the H100 budget to C4. Details and knobs are in `docs/RUN_ON_5080.md` and
`docs/24_novel_experiment.md`.

## What you get

After a run, `results/c4/analysis/` holds:

* `scaling_instr/scaling_fits.csv`, the `E`, `A`, `α` per cell with seed-level uncertainty.
* `novelty_report.json`, the screening result (does `q` screen off size and family), the
  out-of-sample predictive-law error, and the compute-allocation frontier.
* `scaling_instr/scaling_report.json`, the rate-versus-level tests and the multipliers.
* `data_props.json`, the learnability, diversity, and correctness of every teacher's data per
  student, and the derived index `q`.
* The secondary accuracy tables and figures, and one manifest per run with git commit, config
  hash, seeds, and hardware.

## Repository layout

```
synscale/
  config/            schemas, YAML loader, hardware and scope profiles (C4 is the default)
  models/            LLaMA-style student and exact parameter counts
  generation/        prompt pool, two-family teacher backend, filters, subsampling, runner
  training/          WSD trainer, deterministic data and branch builder, schedules
  evaluation/        self-contained multiple-choice and NLL scorer
  analysis/          scaling-law fits, data-quality index, screening and predictive and
                     allocation analyses, plots, cost planner
  pipeline.py        the orchestrator that runs the whole study from a profile
scripts/             install, prepare, generate, train, evaluate, analyze entry points
configs/             student, teacher (two families), training, generation, evaluation fragments
docs/                methodology (00), the experiment (24), scaling method (22), locked decisions (23)
paper/               abstract, introduction, methods drafts
tests/               64 unit tests, CPU only, torch and vllm tests skip if unavailable
```

Install for development with `pip install -e ".[train,gen,eval,analysis,dev]"`. Run tests with
`make test`.

## Status

The whole pipeline, including the data-quality index and the novelty analyses, runs end to end
and is verified on CPU with `make micro`, and 64 unit tests pass. It has not yet run on a GPU
here, so timing numbers are estimates in `synscale/analysis/config_planner.py` to confirm in the
pilot. What remains before a paper is in `docs/25_implementation_tasks.md`.
