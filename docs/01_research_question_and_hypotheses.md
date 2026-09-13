# 01. Final Research Question and Testable Hypotheses

> **Scaling-laws framing (single-5080 study).** On one RTX 5080 the study is run as a
> *data-scaling-law* experiment: the primary outcome is held-out loss as a function of
> synthetic-token count D, fit as L(D)=E+A·D^(-alpha) per student x teacher cell, and the
> headline question is whether the scaling parameters (alpha, E) depend on teacher size and
> whether synthetic data scales like real data. See `docs/22_scaling_laws_analysis.md`. The
> hypotheses below (H1-H5, teacher-size effects on a fixed-budget outcome) remain the
> accuracy/secondary view and the cluster-scale framing.


Tags: **[fact]** established; **[lit]** literature-supported; **[hyp]** hypothesis; **[choice]** methodological choice; **[assume]** assumption to be measured.

## 1.1 Final research question

For a decoder-only student language model of size S ∈ {25M, 100M, 250M, 500M, 1B} parameters that has been pretrained on a fixed real-data corpus to a fixed base state θ*(S), and for a fixed synthetic-data budget of D_syn = 600M student-tokenizer tokens generated under one frozen protocol (prompt pool, system prompt, decoding, filtering), **which teacher size T ∈ {1.5B, 3B, 7B, 14B, 32B, 72B} (Qwen2.5-Instruct; 0.5B as an extension rung) yields the largest improvement over an equal budget of fresh real tokens when the synthetic data is used as annealing data, how does that dependence change with S, and how does the answer change when generation cost is charged?**

The quantity estimated in each cell is (doc 00 §0.3):

  τ_S(T) = E[ Y(Π₂(θ*(S), F_T)) − Y(Π₂(θ*(S), F_real)) ],  Δ_S(T, T′) = τ_S(T) − τ_S(T′),

with Π₂ the fixed phase-2 protocol, F_T the teacher-T treatment file and F_real the matched fresh-web file. Y is the primary outcome P_cc (chance-corrected mean accuracy on the floor-gated suite) or a co-primary continuous outcome (held-out instruction NLL; task loss).

This is narrower than the project's original question ("optimal teacher size for synthetic data generation in small LMs"). The narrowing is deliberate: it is the largest question that the available compute, one teacher family, and pre-registered statistics can answer with controlled inference. What is excluded is listed in doc 00 §0.15.

## 1.2 Primary hypotheses as falsifiable tests

Each hypothesis is stated with its null, the pre-registered test (doc 12), and the observation that would falsify it. δ = 1.0 chance-corrected point (sensitivity 0.5) is the minimal practically important difference.

| id | Hypothesis (alternative) | Null | Test (doc 12) | Falsified / not supported if | Prior from literature (doc 04 §4) |
|---|---|---|---|---|---|
| **H1** | Student gain τ_S(T) increases monotonically with log T at every S | No monotone T effect (M1 vs M0), or isotonic fit not better than constant | M1 vs M0 F-test; per-S isotonic permutation test; Spearman ρ(cell mean, log T) | M1 does not improve on M0 at α = 0.05, **or** the per-S non-monotonicity test (isotonic vs unrestricted) rejects at any S | Strict monotonicity is refuted in every neighbouring channel; expect **H1 false** as a strict statement [lit] |
| **H2** | Returns to T diminish: the gain from doubling T falls below δ beyond some T_sat(S) | Curvature β₂ ≥ 0 in M2; top-step gain μ(72B) − μ(14B) ≥ δ | One-sided test of β₂ < 0; non-inferiority test of the 32B→72B and 14B→72B steps against δ; model-based Δ₂(x) with bootstrap CI | β₂ CI includes 0 **and** the upper CI bound of the top-step gain exceeds δ at every S | Saturation at 1–8B generators is reported for rephrasing (Kang 2025, Niklaus 2026, BeyondWeb 2025); expect **T_sat ≈ 3–14B** for QA/instruction-style generation [hyp] |
| **H3** | For at least one S, an intermediate teacher beats the 72B teacher | The 72B cell mean is ≥ every other cell mean at that S | Four-condition interior-optimum rule (doc 12 §5.1) on tier-1 seeds; **confirmation** on tier-2 seeds only: CI of (cell T₀ − cell 72B) excludes 0 | No S satisfies the interior rule, or the tier-2 confirmation CI includes 0 | Plausible for 25M–100M via capacity/learnability gap (Busbridge 2025, Zhang 2025, Xu 2024) [hyp]; could be masked if the T-axis is flat [hyp] |
| **H4** | The optimal or saturating teacher size shifts with S: T*(S) (or T_sat(S)) is increasing in S | γ₁ = γ₂ = 0 in M3 (no S × T interaction); per-S classifications identical | M3 vs M2 (2 df); M3cat vs M2 (8 df); descriptive WLS fit log₂ T* = α s + c with CI, only over sizes classified "interior" | M3 does not improve on M2, or fewer than 3 sizes have an identifiable T*, or the α CI includes 0 | Linear T*(S) shown for logit KD below 3B (Zhang 2025) [lit]; no sequence-level result; **exploratory**; a power-law "law" is explicitly not claimed |
| **H5** | The economically optimal teacher T$(S) is smaller than the performance-optimal T*(S) by more than a negligible performance loss | λ₁(S) ≥ λ_ref(S): upgrading into T* is at least as efficient per marginal dollar as the cheapest teacher's average efficiency | Bootstrap CI of the last-upgrade marginal efficiency λ₁(S) vs λ_ref(S) (doc 13 Task E / doc 14); T$(λ) step function; ε-rule T$_ε | The λ₁ CI overlaps λ_ref at every S, or T$_ε = T* at every S | Near-certain once c(T) grows ≈ linearly and P(T) saturates [lit: Bansal 2024, Niklaus 2026]; the contribution is the **measured magnitude** and its S-dependence |

Interpretation rules (binding, doc 12 §8): "optimum" only under the interior rule plus tier-2 confirmation; "saturation" only with δ stated; "law" never; H4 reported as consistent/inconsistent with a monotone relation.

## 1.3 Secondary hypotheses (pre-registered, exploratory)

| id | Statement | Test | Purpose |
|---|---|---|---|
| S1 (synthetic vs real) | τ_S(T) > 0 for the best T at each S, i.e. synthetic tokens beat an equal number of fresh web tokens | Family F-B contrasts (cell − C1), Holm within S | Establishes that the T question is non-trivial; a robust negative result is reported with equal prominence |
| S2 (format vs provenance) | Part of τ_S(T) is explained by Q&A format: C1b − C1 > 0 and C_T − C1b < C_T − C1 | Contrasts vs C1b | Separates "instruction-shaped text helps" from "teacher-generated text helps" |
| S3 (dose) | τ_S(T) is increasing and concave in D_syn over {75, 150, 300, 600}M; the teacher ordering is stable across D | M4 (δ₂ = 0 test); Spearman across D | Iso-cost analysis; checks that 600M is on the responsive part of the dose curve (pilot rule P5) |
| S4 (learnability mediation) | Student-base NLL of teacher data, ℓ_S(T), predicts τ_S(T) with an inverted-U (β₂ < 0) and explains the log T effect (b ≈ 0 in M3-mediation) | Mediation models M0–M4 with leave-one-teacher-out CV (doc 09 §4.3); within-teacher quartile split at 7B × {25M, 250M} | The falsifiable form of the "capacity gap" mechanism |
| S5 (correctness mediation) | Teacher correctness c_T on the verifiable subset predicts τ_S(T) beyond log T | M2/M3 mediation | Tests whether "bigger teacher helps" is "more correct teacher helps" |
| S6 (length) | The teacher ordering is unchanged under length-matched subsampling and under equal-examples budgeting | Re-estimate Δ_S(T,T′) on the ablation sets; rank correlation with the primary ordering | Rules out a verbosity artefact of the equal-token control |
| S7 (capability axis) | Replacing log T by the teacher's measured capability z_T does not change the per-S classification | M1z–M3z; AICc comparison | Distinguishes "returns saturate in size" from "returns saturate in teacher accuracy" (32B ≈ 72B on vendor math scores) |
| S8 (forgetting) | Synthetic annealing worsens held-out real-text NLL relative to C1 by less than 5% relative, and the effect does not depend on T | NLL on HeldOut-Real-ID/OOD vs C1 | Detects teacher-dependent distribution shift; pilot integrity rule P6 |
| S9 (contamination) | Teacher-size-correlated benchmark regurgitation, if present, does not change the per-S classification when evaluated on the clean-b item subset | Re-analysis on clean-b (doc 11 §7) | Mandatory sensitivity analysis |

## 1.4 What would make the study uninformative, and the pre-registered response

Pilot rules P1–P6 (doc 12 §7; doc 05) decide whether the full grid runs. The study is designed so that "flat T-axis above 3B, no interior optimum, no interaction" is a publishable result with the same reporting standard as a positive one (doc 12 §8.8).
