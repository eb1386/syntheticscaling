# Which Teacher for Which Student?
### Teacher-size effects of synthetic annealing data for 25M–1B language models

Research-ready specification, implementation, and paper drafts for a controlled study of **teacher model size** in synthetic-data generation for small student language models. The project was previously titled "Synthetic Scaling Laws: Optimal Teacher Model Size for Synthetic Data Generation in Small Language Models"; the title was changed because the design supports transfer curves and a descriptive optimum relation, not a scaling law (see `docs/00_locked_methodology.md` §0.1).

**Core question.** For a student of size S ∈ {25M, 100M, 250M, 500M, 1B} trained to a fixed base state on real data, which teacher size T ∈ {1.5B, 3B, 7B, 14B, 32B, 72B} (Qwen2.5-Instruct; 0.5B as an extension) produces the most useful 600M tokens of synthetic annealing data under one frozen generation protocol, how does that depend on S, and how does it change when generation cost is charged?

## Run it on a single RTX 5080

```bash
./install.sh local5080     # deps + tokenizer + corpora + Qwen2.5 int4 teachers + prompt pool (one time)
./run_all.sh local5080     # runs the whole study end to end, resumable
```

Validate first (minutes to an hour), and see the time estimate:

```bash
make micro                       # CPU end-to-end validation (no GPU) — verified in this repo
make smoke                       # GPU validation with real models, tiny budget (~1-3h)
make budget PROFILE=local5080    # GPU-hours and wall-clock estimate
```

**Honest time cost on one 16 GB RTX 5080** (`synscale/analysis/compute_budget.py`; serial):

| Profile | Scope | Total GPU-hours | Wall-clock |
|---|---|---|---|
| `local5080` | 3 students (25M-250M) x 5 teachers (0.5-14B int4) + controls, 3-5 seeds, D-sweep | ~651 | **~27 days (~34 with overhead)** |
| `local5080_fast` | 2 students x 4 teachers (<=7B), lower power | ~167 | ~7 days (~9 with overhead) |
| `smoke` | real models, tiny budget | ~1-3 | ~1-3 hours |

The full study is about a month of continuous GPU time — the real cost of a complete
teacher x student grid with controls and seeds on one consumer card. `docs/RUN_ON_5080.md`
explains the scoping (a single 5080 cannot serve 32B/72B teachers or train 1B students in
reasonable time), the knobs to shorten it, and what to rent a bigger card for.

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 0 | Locked methodology (decision record; binding) | `docs/00_locked_methodology.md` |
| 1–2 | Final research question; testable hypotheses | `docs/01_research_question_and_hypotheses.md` |
| — | One-page design overview | `docs/02_design_overview.md` |
| 3 | Novelty assessment | `docs/03_novelty_assessment.md` |
| 4 | Literature-review tables and novelty matrix | `docs/04_literature_review.md` |
| 5 | Pilot design | `docs/05_pilot_design.md` |
| 6 | Full experimental matrix | `docs/06_full_experimental_matrix.md` |
| 7 | Exact student architectures (derivation) | `docs/07_student_architectures.md`, `synscale/models/param_count.py` |
| 8 | Teacher-model selection | `docs/08_teacher_selection.md` |
| 9 | Synthetic-data generation methodology | `docs/09_synthetic_data_methodology.md` |
| 10 | Student-training methodology | `docs/10_student_training_methodology.md` |
| 11 | Benchmark suite | `docs/11_benchmark_suite.md` |
| 12 | Statistical analysis plan (pre-registration grade) | `docs/12_statistical_analysis_plan.md` |
| 13 | Cost model and hardware facts | `docs/13_cost_model.md` |
| 14 | Synthetic Transfer Efficiency definition | `docs/14_synthetic_transfer_efficiency.md` |
| 15 | Ablation plan (ranked) | `docs/15_ablation_plan.md` |
| 16 | Compute budget (pilot, full, per hardware profile) | `docs/16_compute_budget.md` |
| 17 | Repository architecture | `docs/17_repository_architecture.md` |
| 18 | Experiment-tracking standard | `docs/18_experiment_tracking_standard.md` |
| 19 | Failure-mode register | `docs/19_failure_mode_register.md` |
| 19 | Failure-mode register | `docs/19_failure_mode_register.md` |
| 20 | Figure/table plan | `docs/20_figure_table_plan.md` |
| 21 | Paper outline | `docs/21_paper_outline.md` |
| 22 | Draft Methods | `paper/methods_draft.md` |
| 23 | Draft Introduction | `paper/introduction_draft.md` |
| 24 | Draft Abstract | `paper/abstract_draft.md` |
| 25 | Next implementation tasks in dependency order | `docs/25_implementation_tasks.md` |

## Repository

```
configs/        students, teachers, generation, training, evaluation fragments; generated experiment YAMLs
synscale/       package: config, tracking, models, generation, training, evaluation, analysis
scripts/        make_matrix, launch, build_pool, generate, build_dataset, pretrain_base, train, evaluate, build_index, analyze, compute_budget
tests/          unit tests (CPU only; torch-dependent tests skip if torch is absent)
docs/           deliverables 00–25
paper/          abstract, introduction, methods drafts; figures/
results/        manifests and small eval JSON only (checkpoints and datasets live outside git)
```

Install: `pip install -e .[dev]` (add `[train]`, `[gen]`, `[eval]`, `[analysis]` on the machines that need them). Tests: `python -m pytest -q`. Experiment matrix: `python scripts/make_matrix.py --all`. Compute budget: `python scripts/compute_budget.py --study both --profile reference`.

## Status

Specification and pipeline code are complete to the extent that can be verified without GPUs (see `docs/25_implementation_tasks.md` for what remains: throughput calibration, real vLLM/lm-eval runs, the pilot). No experimental results exist; every number in the documents is a design parameter, a literature value, or an assumption marked for measurement.
