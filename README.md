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

Every teacher, up to and including 70B and 72B, is served single-card at int4 (AWQ), so no teacher
needs a two-GPU tensor-parallel pod. That one decision makes every generation job fit on a single
spot instance and be safe to preempt, which is what the fleet below exploits. The partial second
family breaks the size-versus-family confound at two student sizes without doubling the whole grid,
which is what keeps C4 affordable. Total: about 198 training runs.

## Running it (RunPod recommended)

The study runs on one multi-GPU H100 box, with local disk (or a persistent network volume) as the
shared volume and one GPU-pinned fleet worker per card (`scripts/fleet_local.sh`). A file-locked
job queue drives the dependency DAG (prepare, pool, generation shards, base training, branches,
finalize); generation for the 72B and 70B teachers is sharded across the GPUs; training catches
SIGTERM, checkpoints, and requeues, so a lost pod costs minutes. Because every teacher is
single-card int4, every job fits one GPU. The mechanism is in `docs/26_fleet_orchestration.md`.

**Provider choice matters more than any code tweak.** For a multi-day run needing a multi-GPU box,
persistent storage, and reliability, the recommendation is **RunPod**: RunPod-owned Secure Cloud
(99% SLA) or cheaper Community Cloud, both with **network volumes that survive a pod crash so work
is never lost** — which, combined with the fleet's checkpoint-and-requeue design, makes an
imperfect pod a non-event. Lambda has the best SLA but is 8-GPU-only, has no spot, and is often sold
out; Vast.ai is cheapest but has no SLA and no shared volume (least reliable). Verified H100-80GB
rates and the full C4 cost, from `python -m synscale.analysis.config_planner`:

| Provider | $/GPU-h | C4 (~1,230 GPU-h) | Reliability |
|---|---|---|---|
| Vast.ai spot | ~$1.49 | ~$1,840 / ~$2,550 CAD | No SLA, no shared volume. Least reliable. |
| TensorDock | ~$2.25 | ~$2,770 / ~$3,850 CAD | Curated marketplace; cost-conscious middle. |
| **RunPod Community** | ~$2.69 | **~$3,310 / ~$4,610 CAD** | **Recommended default.** Network volumes; durable. |
| **RunPod Secure** | ~$2.99 | ~$3,680 / ~$5,120 CAD | Most reliable. RunPod-owned, 99% SLA. |
| Lambda / Modal | ~$3.99 | ~$4,900+ / ~$6,800+ CAD | Reliable but sold out / serverless mismatch. |

Add ~25% for preemption re-runs. The full RunPod procedure (network volume, launch, monitor,
teardown, and the money-safe smoke gate) is in `docs/28_runpod_runbook.md`; a single-box vast.ai
alternative is in `docs/27_vast_runbook.md`. On the box itself:

```bash
./install.sh c4
./run_all.sh c4 --fleet        # build the queue and drain it with one worker per GPU
```

Validate without spending real money first:

```bash
make micro                   # CPU, minutes: runs every stage end to end, including the fleet queue
make budget PROFILE=c4       # the provider cost comparison above
make smoke                   # GPU, ~1-3 hours: real teachers, tiny budget (the pre-spend gate)
```

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
  fleet.py           file-locked spot-fleet job queue, DAG, sharding, preemption requeue
  pipeline.py        the orchestrator that runs the whole study from a profile
scripts/             install, prepare, train, analyze, worker; runpod/ and vast/ launch toolkits
configs/             student, teacher (two families), training, generation, evaluation fragments
docs/                methodology (00), experiment (24), scaling (22), fleet (26), RunPod (28), vast (27)
paper/               abstract, introduction, methods drafts
tests/               68 unit tests, CPU only, torch and vllm tests skip if unavailable
```

Install for development with `pip install -e ".[train,gen,eval,analysis,dev]"`. Run tests with
`make test`.

## Status

The whole pipeline, including the data-quality index, the novelty analyses, and the spot-fleet
queue, runs end to end and is verified on CPU with `make micro`, and 68 unit tests pass. It has
not yet run on a GPU here, so timing and cost numbers are transparent estimates in
`synscale/analysis/config_planner.py` to confirm in the pilot. What remains before a paper is in
`docs/25_implementation_tasks.md`.
