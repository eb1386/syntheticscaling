# 22. Synthetic-Data Scaling-Law Analysis (the core method)

This is the analysis that makes the project a scaling-laws study rather than a
teacher-picking study. It is implemented in `synscale/analysis/scaling.py` and runs
automatically at the end of the pipeline (`results/<profile>/analysis/scaling_*/`).

## 22.1 The object we fit

For each student `S`, each data source (a teacher `T`, or the real control `C1`, or the human
control `C1b`) we measure held-out loss `L` as a function of the number of synthetic (or real)
tokens `D` the student has trained on in the annealing phase, and fit the standard data-scaling
form (Hoffmann et al. 2022; Kaplan et al. 2020):

```
L(D) = E + A · D^(-alpha)
```

- **E** — estimated asymptote `L_inf_hat` over the *observed* D range. NOT the true irreducible
  loss: with a limited D range E, A and alpha trade off, so E is reported as an estimate and the
  form is chosen by held-out fit (see 22.3a).
- **A** — coefficient: the offset / difficulty of the source.
- **alpha** — data-scaling exponent: how fast loss falls as you add tokens. Bigger alpha = data that keeps paying off.

`D` is synthetic tokens seen, `D = (1 - rho) · tokens_trained` with replay fraction `rho = 0.25`.

## 22.2 Where the data-quantity axis comes from (why this is cheap)

The trainer evaluates held-out NLL every ~20M tokens during the synthetic phase and writes it
to `results/<id>/train_log.jsonl` (`held_out_nll` per eval set, with `tokens`). So each single
run yields ~20 points on the `L(D)` curve. We do **not** run a separate D-sweep: one run per
`(S, T, seed)` gives the whole curve. This is what lets a broad grid fit on one 5080.

## 22.3a Statistical safeguards (locked)

1. **Checkpoints are not independent.** The ~20 points on one curve come from one optimisation
   trajectory and are autocorrelated. We fit a curve **separately per seed** (one run = one
   observation of the parameters) and take uncertainty from the **spread across seeds**. Cells
   with a single seed are flagged `seed_ci=false`; inference needs >=2 seeds (>=3 in the full run).
2. **E is an estimated asymptote, not irreducible loss.** We fit three forms — `E+A*D^-alpha`,
   `a-b*log D`, and pure `A*D^-alpha` — and compare them on **held-out checkpoints** (the last
   third of each curve). The reported `best_form` is the one that predicts held-out points best;
   `L_inf_hat` is labelled as an estimate over the observed regime.
3. **Teacher-independent evaluation.** Loss is measured on fixed, teacher-independent sets:
   held-out real web text (`base_heldout`), held-out human instruction/answer text (`instr`),
   and downstream benchmark accuracy (secondary). We never evaluate on a teacher's own synthetic
   distribution, which would reward imitation.

## 22.3b Synthetic-data multiplier (preferred metric)

For each (student, teacher) we invert the fitted curve for the teacher and for the matched-real
control (C1) and report `D_real(L) / D_syn(L)` at matched loss: how many real tokens one
synthetic token is worth over that loss range (median, min, max). A value of 1.6 means "one
7B-teacher token ~ 1.6 real tokens for this student here". Combined with generation cost it
answers whether the extra value was economically worthwhile — more interpretable than a bespoke
efficiency metric.

## 22.3 The three tests (the research question, operationalised)

"Do synthetic-data scaling laws depend on the teacher?" becomes three concrete tests
(`test_teacher_dependence`, `synthetic_vs_real`):

1. **Does the exponent depend on teacher size?** Within each student, regress `alpha` on
   `log2 T`. A positive slope means larger teachers produce data that *scales better* (loss
   keeps falling with more tokens); a flat slope means teacher size sets a level, not a rate.
2. **Does the estimated asymptote depend on teacher size?** Regress `L_inf_hat` on `log2 T`. A negative
   slope means larger teachers reach a lower asymptotic loss (more headroom), regardless of rate.
3. **Does synthetic data scale like real data?** Compare each teacher's `(E, alpha)` to the
   matched-real control `C1` at the same student. The gap `E_syn - E_real` and the exponents
   say whether synthetic data is a substitute for, or complement to, web text — and whether a
   bigger teacher closes the gap.

Reported per student and pooled across students (with the student intercept absorbed). Because
there are only four teacher points, slopes are reported **descriptively** with sign, magnitude
and R^2, and confidence bands come from a seed bootstrap — never as a "law" from four points
(consistent with the honesty rules in `docs/00 §0.1`, `docs/03`).

## 22.4 Joint form and student dependence

Beyond per-cell fits, the pooled analysis fits `L(S, T, D)` with a student term and a
teacher term to ask whether the teacher effect on the scaling curve grows or shrinks with
student size (the `S × T` interaction on the scaling parameters). With 4 students × 4 teachers
this is exploratory; it is reported as a trend with CIs, and the per-cell fits are the primary
evidence.

## 22.5 Why loss, not accuracy

Accuracy on multiple-choice benchmarks sits at chance for 25M–100M students on most tasks
(`docs/11`), so it cannot trace a scaling curve at the small end. Loss moves smoothly from the
first tokens, has far higher signal-to-noise, and is the quantity scaling laws are defined on.
Accuracy is kept as a secondary "transfer breadth" outcome at the final `D` only.

## 22.6 Falsifiable outcomes

- **Synthetic data does not scale** (alpha ~ 0, loss flat in D): the premise fails; reported as such.
- **Scales, teacher-independent** (alpha, E flat in log T): synthetic-data scaling laws exist but do not depend on teacher size — a clean negative on the headline question.
- **Scales, teacher-dependent** (alpha rises and/or E falls with log T): the headline positive; report the slopes and whether they depend on student size.
- **Synthetic ≥ real** (E_syn ≤ E_real for some T): synthetic annealing data beats matched web text; report which teacher sizes achieve it.

Each is a publishable result; the study is designed so the negative is as clear as the positive.
