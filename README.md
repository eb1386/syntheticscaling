# Synthetic Data Scaling Laws for Small Language Models

When you train a small language model on synthetic data written by a larger teacher model,
does the data obey a scaling law? That is, does loss keep falling predictably as you add more
synthetic tokens, and does that behavior depend on how big the teacher is? This repository is a
controlled experiment built to answer that.

## The question, precisely

We measure held-out loss `L` as a joint function of three quantities:

```
L = f(S, T, D)
```

* `S` is the student size: 25M, 50M, 100M, 250M parameters.
* `T` is the teacher size: Qwen2.5-Instruct at 0.5B, 1.5B, 3B, 7B.
* `D` is the number of synthetic tokens the student has trained on.

For every student and teacher pair we fit a data-scaling curve over `D` and then ask three
things. Does the scaling exponent depend on teacher size, so that better data keeps paying off
for longer? Does the asymptote depend on teacher size, so that better data reaches a lower floor?
And does synthetic data behave like real web text, or better, or worse? The headline number is a
synthetic-data multiplier: how many real tokens one synthetic token is worth.

This is a different, stronger question than "which teacher gives the highest benchmark score." It
asks whether teacher scale changes the marginal value of another synthetic token, whether that
change depends on student capacity, and how synthetic data compares to an equal amount of real
data.

## Why loss, and why matched checkpoints

Everything is measured in loss, not accuracy. Loss is the quantity scaling laws are defined on,
it moves smoothly from the first tokens of training, and it has far higher signal to noise than
multiple-choice accuracy, which sits at chance for 25M to 100M models on most tasks. Accuracy is
kept only as a secondary transfer-breadth check.

Each student is pretrained once on real web text (FineWeb-Edu) to a stable checkpoint. Every
condition then branches from that same checkpoint and runs an identical short annealing phase, in
which the only thing that differs is the data it sees: synthetic text from one teacher, an equal
number of fresh real web tokens (the control C1), human question-answer text (C1b), or nothing new
(C0). Because all conditions start from the same checkpoint and share every hyperparameter, a
difference in outcome is attributable to the data source alone.

The data-quantity axis `D` comes for free. The trainer evaluates held-out loss every 20M tokens
during the annealing phase, so one training run yields roughly 20 points on the `L(D)` curve. We
do not run a separate job per data quantity. This is what makes a broad grid affordable on one
consumer GPU.

## What the scaling analysis does

For each student and teacher cell we fit three functional forms and compare them on held-out
checkpoints (the last third of each curve), so a form that only interpolates is penalized:

```
power with asymptote :  L(D) = E + A * D^(-alpha)     (primary)
log                  :  L(D) = a - b * log D
pure power           :  L(D) = A * D^(-alpha)
```

Here `E` is the estimated asymptote over the observed range, reported as `L_inf_hat`. It is not
claimed to be the model's true irreducible loss, because over a limited range of `D` the three
parameters trade off against each other.

Three statistical safeguards are built in.

1. Checkpoints on one training curve are not independent, because they come from a single
   optimization trajectory. We fit a curve separately per seed, treat each independent run as one
   observation of the parameters, and take uncertainty from the spread across seeds. Cells with a
   single seed are flagged so their parameters are never given a false confidence interval.
2. Model choice is decided by held-out fit, not by in-sample residuals, and the asymptote is
   labelled as an estimate rather than a physical constant.
3. Evaluation is teacher-independent. Loss is measured on fixed held-out real web text, fixed
   held-out human instruction text, and downstream accuracy. We never score a student on its own
   teacher's synthetic distribution, which would reward imitation rather than learning.

## The synthetic-data multiplier

For each student and teacher we invert the fitted curve for the teacher and for the matched-real
control, then compute the ratio of real tokens to synthetic tokens needed to reach the same loss.
If a 100M student needs 200M real tokens to reach a loss that 120M tokens from the 7B teacher also
reach, the multiplier is about 1.67. The statement it licenses is concrete: one token from the 7B
teacher is worth roughly 1.6 real tokens for a 100M student over this loss range. Combined with the
measured cost of generating those tokens, it answers whether the extra value was worth paying for.
This replaces the earlier bespoke efficiency metric, which is kept only as an appendix.

## What the experiment varies

| Axis | Values | Why it is here |
|---|---|---|
| Teacher size `T` | Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B (FP8) | the question is about teacher size |
| Student size `S` | 25M, 50M, 100M, 250M, one LLaMA-style family, shared 16k tokenizer | does the answer depend on the student |
| Synthetic tokens `D` | about 20 points from 25M up to 400M, read from the training curve | the scaling-law axis |
| Controls | C1 equal real tokens, C1b human question-answer, C0 base model | is synthetic data better than real data, or than nothing |

Teacher data is generated once per teacher under one frozen prompt pool, one decoding
configuration, and teacher-agnostic filters, at equal student-tokenizer tokens, so the only thing
that differs between teacher conditions is the teacher. All students share one 16k byte-level BPE
trained on the real corpus, so the 25M model is not dominated by its embedding table and the
student-size axis stays clean. Teachers generate in their own tokenizer and their text is
retokenized with the student tokenizer.

## What a result looks like

Any of the following is a clean, reportable outcome, and the study is designed so a negative is as
informative as a positive:

* Synthetic data does not scale: loss is flat in `D`. The premise fails.
* It scales but is teacher-independent: the exponent and asymptote do not move with teacher size.
  A tidy negative on the headline question.
* It scales and is teacher-dependent: the exponent rises and the asymptote falls with teacher
  size. Report the slopes and whether they themselves depend on student size.
* Synthetic beats real: some teacher reaches a lower asymptote than matched real tokens. Report
  which teacher sizes achieve it and the multiplier.

A particularly interesting shape would be teachers above 3B producing near-identical curves for a
25M student while remaining distinguishable for a 250M student, which would be a teacher-capacity by
student-capacity interaction.

After a run, `results/<profile>/analysis/` holds `scaling_instr/scaling_fits.csv` (E, A, alpha per
cell), `scaling_instr/scaling_report.json` (the teacher-dependence tests and the multipliers), the
secondary accuracy tables and figures, and one manifest per run recording git commit, config hash,
seeds, and hardware.

## Running it

Two commands. The first installs dependencies, trains the tokenizer, downloads and tokenizes the
corpora, fetches the teacher models, and builds the prompt pool. The second runs the whole study
and is resumable, so re-running continues where it left off.

```bash
./install.sh scaling5080
./run_all.sh scaling5080
```

Validate the pipeline first, and get the time estimate, without committing days of compute:

```bash
make micro                        # CPU, minutes: runs every stage on a toy model
make smoke                        # GPU, about 1 to 3 hours: real teachers, tiny budget
make budget PROFILE=scaling5080   # GPU-hours and wall-clock estimate
```

Run `scaling5080_fast` first as the pilot, about one week, to measure throughput, seed variance,
and effect sizes on the real card. If it is clean, run the full `scaling5080` grid, about three to
four weeks on a single 16 GB card, for the paper. Details, hardware notes, and knobs to shorten the
run are in `docs/RUN_ON_5080.md`.

## Repository layout

```
synscale/            the package
  config/            config schemas, YAML loader, hardware and scope profiles
  models/            LLaMA-style student (RMSNorm, RoPE, SwiGLU, GQA) and exact parameter counts
  generation/        prompt pool, vLLM teacher backend, filters, subsampling, runner
  training/          WSD trainer, deterministic data and branch builder, schedules
  evaluation/        self-contained multiple-choice and NLL scorer
  analysis/          scaling-law fits, aggregates, plots, compute budget
  pipeline.py        the orchestrator that runs the whole study from a profile
scripts/             install, prepare, generate, train, evaluate, analyze entry points
configs/             student, teacher, training, generation, evaluation fragments
docs/                methodology, scaling-law method (docs/22), locked decisions (docs/23)
paper/               abstract, introduction, methods drafts
tests/               60 unit tests, CPU only, torch and vllm tests skip if unavailable
```

Install for development with `pip install -e ".[train,gen,eval,analysis,dev]"`. Run tests with
`make test`.

## Status and honesty

The whole pipeline runs end to end and is verified on CPU with `make micro`, and 60 unit tests
pass. It has not yet run on a GPU here, so the timing numbers are formula estimates marked in
`synscale/analysis/compute_budget.py` and should be confirmed by the pilot. This is built to a
NeurIPS-quality methodological standard, meaning pre-registered tests, matched checkpoints,
teacher-independent evaluation, and seed-level uncertainty. Whether any result belongs at a main
conference depends on what it shows and its novelty against prior work, and the repository does not
claim the result in advance.

Full methodology: `docs/00_locked_methodology.md`. Scaling-law method: `docs/22_scaling_laws_analysis.md`.
Locked design decisions: `docs/23_locked_review_decisions.md`. What remains before a paper:
`docs/25_implementation_tasks.md`.
