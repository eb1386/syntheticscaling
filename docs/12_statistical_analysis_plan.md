# 12. Statistical Analysis Plan (pre-registration grade)

> **Provenance.** Workstream derivation document, synthesised by the research lead. Where this text conflicts with `docs/00_locked_methodology.md`, the locked document wins. Amendments applied by the lead are listed first; the workstream text follows unchanged so the derivation remains auditable.

## Lead amendments (supersede the text below)

- Adopted unchanged: outcomes, noise model, power results, tiered seeds, model sequence M0–M4/M2sat, per-S T* classification, δ = 1.0 (sensitivity 0.5), Holm families, reporting standards, pilot rules P1–P6.
- Grid amendments: teacher rungs are {0.5, 1.5, 3, 7, 14, 32, 72}B with the six-rung core as the primary analysis (doc 00 §0.6); the pilot uses 3 base × 3 phase seeds at both pilot sizes and arms {C0, C1, C1b, T1.5B, T7B, T32B} (doc 00 §0.10); the D-sweep is a fixed-D₂ synthetic-fraction sweep at levels {75M, 150M, 300M} (doc 00 §0.9), so §4.4's 2D_syn level is not available and the dose range is 8×.
- Open issue 4 (no teacher smaller than the 1B student) is resolved by the 0.5B rung. Open issue 5 (relative dose) is mitigated by the base-token rule (16% vs 4% synthetic fraction) and the D-sweep at both S extremes.
- The 'seed' definition in §0 (re-draw of the subsample) applies only at D-sweep levels via nested prefixes; at D_syn = 600M the delivered pool is used whole and the seed varies order only (doc 00 §0.10). Base replicates at 25M/100M supply σ²_base.

---

# Statistical Analysis Plan (pre-registration grade)

**Workstream:** statistical design. **Applies to:** the 5 (student size S) × 6 (teacher size T) × r (seeds) synthetic-data study described in `00_brief.md`, its per-student-size controls (C0 base only; C1 base + matched real tokens), the nested synthetic-token (D) sweep, and the 2 × 3 pilot.
**Status of every statement:** each is tagged as *established* (textbook statistics), *literature-supported* (cited), *assumption* (to be replaced by pilot estimates), *methodological choice* (ours; pre-registered), or *hypothesis*.
**Companion code:** `scratchpad/code/power_sim.py` (numpy only; all simulation numbers in §3 are its output, `power_sim_results.md`).

---

## 0. Design, notation, and what a "seed" is

| Symbol | Meaning |
|---|---|
| S ∈ {25M, 100M, 250M, 500M, 1B} | student non-embedding-inclusive parameter count (as finally derived by the architecture workstream); s = log₂(S / 25M) ∈ {0, 2, 3.32, 4.32, 5.32} |
| T ∈ {1.5, 3, 7, 14, 32, 72}B | teacher size (Qwen2.5-Instruct family, P3); x = log₂(T / 1B) ∈ {0.58, 1.58, 2.81, 3.81, 5.00, 6.17}. The brief's grid "1/3/7/14/32/70B" does not exist in one family; the analysis uses the actual sizes. |
| D | synthetic tokens (student tokenizer) in the continued-training phase; d = log₂(D / D_syn); D_syn is the fixed primary budget (P5) |
| cell (S,T) | one treatment: base(S) → continued training on D_syn tokens from teacher T |
| r_{S,T} | seeds (independent runs) in cell (S,T) |
| C0(S) | base model only (the shared base run; see below) |
| C1(S) | base + D_syn matched real tokens (held-out FineWeb-Edu), r seeds |
| y_{k,j}(run) ∈ {0,1} | correctness of item j of benchmark k for a given run |
| a_k(run) | accuracy of a run on benchmark k, N_k items |
| P(run) | primary aggregate (§1.2) |

**What a seed varies (methodological choice).** Under Option B (P1) the base run is shared within S, so a "seed" in cell (S,T) re-draws *only* the synthetic phase: (i) the whole-example subsample of D_syn tokens from teacher T's generated pool, (ii) the data order, (iii) any residual nondeterminism. We require (i) to be re-drawn per seed. Reason: the treatment is "teacher T's generation distribution," not one particular D_syn-token sample; with a fixed sample, the seed SD understates the between-sample variance and the inference is conditional on one draw. The teacher pool itself (generation seed) is drawn once per teacher because regeneration is the dominant cost; this is a stated limitation (the study is conditional on one pool per teacher; pool-to-pool variance is not estimated).

**The single base run is an unreplicated block.** Because every cell within S shares C0(S), base-run randomness is a constant within S. Consequence (established): it cancels exactly in *all within-S contrasts* (adjacent-T, teacher vs C1, curvature in T) — this is the design's main efficiency — but it does **not** cancel in *cross-S* comparisons (the S main effect, the S × T interaction, and any T*(S) relation), which are therefore conditional on the five particular base runs. Mitigation (recommended, cheap): train 3 base seeds at 25M and 100M (≈ 4 extra base runs, the two cheapest sizes) to estimate the base-seed variance component σ²_base(S) and report it alongside the synthetic-phase seed SD; do not use them to replicate the whole grid.

Total cells: 30 treatment cells + 5 C1 cells + 5 C0 runs (+ optional base replicates) + D-sweep cells (§4.4).

---

## 1. Outcome variables

### 1.1 Per-benchmark accuracy and its scoring rule (pre-specified)

For each benchmark k with N_k items, a run's accuracy is a_k = N_k⁻¹ Σ_j y_{k,j}. The scoring rule is fixed per benchmark before the pilot: length-normalised multiple-choice log-likelihood (`acc_norm`) for ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA and PIQA; raw `acc` for SciQ, BoolQ and Winogrande; last-token exact-match accuracy for LAMBADA (OpenAI variant). All zero-shot, `lm-evaluation-harness` task definitions, fixed item sets (test splits where labels are public, otherwise validation), fixed prompt templates, deterministic scoring. Both `acc` and `acc_norm` are logged for every run; only the pre-specified one enters the primary analysis.

**Chance level c_k** (methodological choice): the best label-only baseline, c_k = max(uniform guess, majority class):

| k | items N_k | c_k | note |
|---|---|---|---|
| ARC-Easy | 2376 | 0.25 | 3–5 options; 4-way typical |
| ARC-Challenge | 1172 | 0.25 | |
| PIQA | 1838 | 0.50 | |
| HellaSwag | 10042 | 0.25 | |
| SciQ | 1000 | 0.25 | |
| OpenBookQA | 500 | 0.25 | smallest set |
| BoolQ | 3270 | **0.622** | majority class ("yes") ≈ 62% of the validation set [literature-supported: Clark et al. 2019, https://arxiv.org/abs/1905.10044]; uniform 0.5 would understate chance |
| Winogrande | 1267 | 0.50 | |
| LAMBADA | 5153 | ≈ 0 | open-vocabulary; c_k := 0 |

### 1.2 The primary aggregate: three candidates, one choice

Let K be the set of retained benchmarks (§1.3). Candidates:

**(A1) Raw mean accuracy.** P_raw = (1/|K|) Σ_k a_k. Standard (Pythia, SmolLM tables). Defect: level dominated by high-chance tasks; a benchmark at floor contributes pure noise; a 1-point gain means different things on a 4-way and a 2-way task.

**(A2) Chance-corrected mean (chosen).**
$$P_{cc} = \frac{100}{|K|}\sum_{k\in K}\frac{a_k - c_k}{1 - c_k}.$$
Each term is the fraction of the above-chance range achieved (a per-benchmark Cohen-κ-like rescaling). Zero means "at chance on every retained benchmark"; 100 means perfect. Does **not** depend on the base run. Defect: the rescaling multiplies each benchmark's noise by 1/(1−c_k) — ×1.33 for 4-way, ×2 for binary, ×2.65 for BoolQ — and the binary tasks are exactly the ones a 25M model sits at floor on (§1.3). Hence A2 must be paired with a floor-gating rule; without it A2 is *worse* than A1 at 25M.

**(A3) Normalised gain over base ("headroom recovered").** G_k = (a_k − a_k^{base})/(1 − a_k^{base}), aggregate = mean_k G_k. Rejected as primary for three reasons: (i) it divides by a quantity estimated from the single, unreplicated base run per S, so base-seed noise enters every cell's *level* nonlinearly (within S it is a common transform, so T-orderings are preserved, but cross-S comparisons of "gain" are distorted); (ii) the denominator shrinks with S (e.g. SciQ base ≈ 0.85 at 1B vs ≈ 0.6 at 25M), so identical absolute gains are inflated ×2.7 for the largest student, manufacturing an S × T interaction; (iii) for benchmarks where the base is below chance (ARC-Challenge, BoolQ at 25M) it is undefined in spirit. Reported as a secondary descriptive.

**(A4) Precision-weighted chance-corrected mean.** P_w = Σ_k w_k (a_k − c_k)/(1 − c_k) with w_k ∝ 1/σ̂²_{seed,k} (pooled per-benchmark seed variance from the pilot, frozen before the full study). Maximises signal-to-noise; not interpretable in points; used only as a sensitivity analysis and for the pilot detectability check (§7).

**Decision (methodological choice):** the primary outcome is **P_cc over a floor-gated benchmark set that is identical for all S** (§1.3), reported in "chance-corrected points". All effects (gain over C1, gain over C0, differences between teachers) are differences in P_cc. Secondary: A1, A3, A4, every per-benchmark a_k, and the continuous outcomes of §1.4.

### 1.3 Floor effects at 25M and the benchmark-retention rule

Reference numbers (literature-supported: Pythia paper Table, zero-shot, Biderman et al. 2023, https://arxiv.org/abs/2304.01373; the ± are *item* SEs): Pythia-70M scores ARC-Easy 0.374 ± 0.010, ARC-Challenge 0.181 ± 0.011 (below the 0.25 chance line under `acc`), PIQA 0.595 ± 0.011, Winogrande 0.528 ± 0.014, SciQ 0.601 ± 0.015, LAMBADA 0.185 ± 0.005; HellaSwag, OpenBookQA and BoolQ for Pythia-70M/160M are within ≈ 1–3 points of chance/majority [UNVERIFIED exact values]. SmolLM-135M (2T tokens): ARC-Challenge 0.29, ARC-Easy 0.56, HellaSwag 0.43, PIQA 0.68, Winogrande 0.53 (search-summary values from the SmolLM model card; [UNVERIFIED digit-level]). Pythia-70M has ≈ 19M non-embedding parameters and was trained on 300B tokens; a 25M student trained Chinchilla-style on ≈ 0.5B tokens will be at or below it. Expectation (hypothesis): at 25M only SciQ, ARC-Easy, PIQA and LAMBADA clear the floor decisively; HellaSwag marginally; ARC-Challenge, OpenBookQA, BoolQ and Winogrande do not.

**Retention rule (pre-registered, evaluated on C0 at the pilot):** benchmark k is retained in the primary set K iff, for the *smallest student size retained in the primary grid*,
$$a_k^{C0} - c_k \;\ge\; 3\cdot \mathrm{SE}_{item,k},\qquad \mathrm{SE}_{item,k}=\sqrt{a_k(1-a_k)/N_k},$$
and, if base replicates exist, also ≥ 3 · SD_seed,k. The same K is used at every S in the primary analysis (otherwise cross-S comparisons change the outcome definition). If |K| < 4 at 25M, the 25M row is removed from the primary grid (§7). Per-benchmark results at every S are reported regardless.

### 1.4 Continuous outcomes (co-primary, pre-specified)

1. **Held-out NLL on human-written instruction/QA text** (nats/token in the shared student tokenizer; also bits-per-byte for comparability). Direction is not assumed: synthetic-phase training can improve benchmarks while *worsening* NLL on human text (distribution shift). It is therefore a distinct outcome answering "does synthetic data move the student toward or away from human text?", not a proxy for P_cc.
2. **Task loss**: per benchmark, the mean bits-per-byte of the correct continuation (the continuous counterpart of `acc`), aggregated by simple mean over K. Continuous metrics have higher signal-to-noise and monotonicity than discrete accuracy at small scale [literature-supported: Madaan et al. 2024, https://arxiv.org/abs/2406.10229; Heineman et al. 2025, https://arxiv.org/abs/2508.13144; Bhagia et al. 2024, https://arxiv.org/abs/2412.04403]. Used to (i) confirm the *direction* of every P_cc effect and (ii) provide the smoother surface for T* estimation as a secondary analysis.

### 1.5 From N items to a per-run standard error; seeds vs items

**Decomposition (established).** For run i and item j,
$$y_{ij} = \mu + a_i + b_j + e_{ij},$$
a_i = run (seed) effect, b_j = item difficulty, e_{ij} = run × item residual (Bernoulli given the run's item-level probabilities). A run's observed accuracy on the *fixed* item set is ā_i = μ + a_i + b̄ + ē_i.

- **Item-sampling SE of one run:** SE_item = √(p(1−p)/N) (binomial; exact enough for N ≥ 500). Numerically, at 25M-like accuracies: ARC-Easy 1.0 pt, ARC-Challenge 1.2, PIQA 1.1, HellaSwag 0.45, SciQ 1.5, OpenBookQA 2.0, BoolQ 0.9, Winogrande 1.4, LAMBADA 0.6 (percentage points). After chance-correction (÷(1−c_k)): 1.3, 1.6, 2.3, 0.6, 2.0, 2.7, 2.3, 2.8, 0.6.
- **Seed-to-seed SD of observed accuracy**, σ²_{seed,obs} = σ²_a + σ²_{ē} ≈ σ²_a + p(1−p)/N_eff. Because every run is scored on the same items, b̄ is a common constant and does not appear. Therefore an empirical seed SD (from r runs on fixed items) *already contains* the item-level Bernoulli component for that N; it is the right per-run SD for comparing cells. Cell-mean SE: σ_{seed,obs}/√r. Note the floor: the aggregate seed SD cannot be smaller than the Bernoulli-only aggregate, which at 25M-like accuracies is ≈ 0.65 pt for K = 9 and ≈ 0.84 pt for the 4-benchmark floor-gated set (computed from the N_k above). OpenBookQA (N = 500) alone contributes 2.7 chance-corrected points of pure Bernoulli noise per run — a reason to drop it or to use its task loss instead.
- **Why the seed is the unit for cell comparisons.** The claim "teacher T₂ beats T₁ for student S" is a claim about a *training procedure*; its replication unit is a training run. Item-level SEs describe uncertainty about the item population for one fixed model. Using N items as N replicates of a treatment effect with r = 1 is pseudo-replication: two single runs that differ by 3 pt on ARC-Easy (item SE 1.0 pt each, "significant" by an item test) can be re-ordered by re-running with a new seed whenever σ_seed ≳ 1.5 pt.
- **When item variance matters.** Generalisation to *new items* is governed by the condition × item interaction variance σ²_{c×j}, which is neither in σ_seed,obs nor cancelled by using the same items. It is estimated by the **paired item bootstrap**: resample items (with replacement, stratified by benchmark), recompute every run's accuracy on the resampled items, recompute the contrast of cell means. Pairing matters: for two runs A and B on the same items the per-benchmark SE of the difference is √((p_A + p_B − 2p_AB)/N), where p_AB is the joint-correct rate, which is far below the unpaired √((p_A(1−p_A) + p_B(1−p_B))/N) because runs sharing a base are highly correlated item-by-item [literature-supported: Miller 2024, https://arxiv.org/abs/2411.00640]. Also cluster by passage where items share a context (BoolQ, LAMBADA are single-item; ARC/SciQ are independent; HellaSwag activities → cluster by activity label).
- **Combined uncertainty of a cell contrast Δ̄ = ȳ_{S,T₂} − ȳ_{S,T₁}:**
$$\mathrm{Var}(\bar\Delta) \approx \underbrace{\frac{2\,\sigma^2_{seed,obs}}{r}}_{\text{training randomness (incl. Bernoulli)}} + \underbrace{\frac{\sigma^2_{c\times j}}{N}}_{\text{item generalisation}} .$$
The second term is taken from the paired item bootstrap of Δ̄ after subtracting its Bernoulli part 2p(1−p)/(rN) (already inside the first term); the cleaner alternative is a crossed random-effects logistic model with random intercepts for run and item (run-slope for condition), fitted per benchmark. **Rule:** the primary CI is seed-based; the paired item bootstrap CI is a required robustness check; a directional claim requires both intervals to exclude 0.

---

## 2. Noise model

### 2.1 Literature values (what could be verified from here, and what could not)

| Source | Setting | What it says | Verification status |
|---|---|---|---|
| PolyPythias, van der Wal et al. 2025 (ICLR), https://arxiv.org/abs/2503.09543 | 10 seeds × {14M, 31M, 70M, 160M, 410M}, 300B Pile tokens | Downstream performance "largely stable" across seeds; ARC-Easy, LAMBADA, SciQ above chance at all sizes; only two outlier seed-runs (410M). Separate data-order vs init seeds available. | Existence and qualitative claims verified via search summary; per-benchmark seed SDs [UNVERIFIED from this environment — arXiv/OpenReview blocked] |
| Madaan et al. 2024, "Quantifying Variance in Evaluation Benchmarks", https://arxiv.org/abs/2406.10229 | 10 Llama-2-7B-architecture models from scratch, 210B tokens, 13 benchmarks | Defines seed variance; continuous metrics (choice log-likelihood) have higher SNR and monotonicity; reformulating MMLU as completion reduces variance at ~7B | Verified qualitatively; the per-benchmark seed-SD table [UNVERIFIED]; my recollection is ≈ 0.5–1.5 acc points for ARC/HellaSwag/PIQA-type tasks and several points for generative tasks |
| DataDecide, Magnusson et al. 2025, https://arxiv.org/abs/2504.11393 | 25 recipes × {4M…1B} × 3 seeds | Noise := SD across 3 seeds at 150M averaged over recipes; MMLU low seed noise; ARC-Easy large recipe spread; ≈ 80% decision accuracy at 150M for 1B rankings | Verified qualitatively; numeric per-benchmark noise [UNVERIFIED] |
| Heineman et al. 2025, "Signal and Noise", https://arxiv.org/abs/2508.13144 | 30 benchmarks, 375 models | Noise ≈ relative SD over the final n checkpoints; correlates with init-seed noise (R² 0.82) and data-order noise (R² 0.86); checkpoint averaging and BPB metrics raise SNR | Verified via search summary |
| Biderman et al. 2023 (Pythia) | 70M–410M | Item SEs ≈ 0.5–1.5 pt per benchmark (their ±) | Verified (search summary reproduced the table) |
| Miller 2024 | — | Clustered SEs; paired differences; planning formulas | Verified |

### 2.2 Pre-registered noise assumptions (to be overwritten by the pilot estimate, §7)

- **Per-benchmark seed SD of observed accuracy** for the continued-training phase: σ_{seed,k} ∈ [0.5, 1.5] pt, central 1.0 pt (assumption). Rationale: the phase-specific SD should be *smaller* than full-pretraining seed SD (shared base), but the per-seed subsample re-draw adds variance, and the Bernoulli floor alone is 0.45–2.0 pt (§1.5).
- **Aggregate seed SD** for P_cc: σ_agg = (1/|K|)·√(Σ_k σ²_{seed,k}/(1−c_k)² + ρ Σ_{k≠l} σ'_k σ'_l), with ρ the between-benchmark seed correlation (a "good seed" tends to be good everywhere). Numerically (σ_{seed,k} = 1.0): ρ = 0 → 0.55 pt (K = 9) / 0.73 pt (K = 4); ρ = 0.3 → 0.99 pt for either K. With σ_{seed,k} = 0.7: 0.39–0.69 pt. **Planning range: σ_agg ∈ {0.3, 0.5, 0.8}** (central 0.5), consistent with the brief's 0.3–0.8. For task loss / NLL we assume half these values in relative terms (assumption, from the SNR literature above) and verify in the pilot.
- **Heteroscedasticity:** assume σ_agg(25M) = 1.5 × σ_agg(≥250M), σ_agg(100M) = 1.2 × (assumption, to be checked). The simulation shows that ignoring this inflates the per-S Type I error at 25M to ≈ 17% if the variance is pooled across S (§3.3, Sim Table 4), so per-S variance estimates are used for per-S tests (§4.2).
- **Distribution:** Gaussian on the aggregate scale (a mean of ≥ 4 bounded quantities); heavy-tailed outlier runs (loss spikes) are possible [PolyPythias] → every test is repeated with a rank-based alternative (§4.3), and outlier runs are reported, never silently dropped.

---

## 3. Power and precision

### 3.1 Two-cell contrasts (adjacent teachers; teacher vs C1; teacher vs C0)

**Formula (established).** For a difference Δ between two cell means with r seeds each, two-sided α = 0.05, power 1−β = 0.8, unpaired:
$$r \;\ge\; \frac{2\,(z_{0.975}+z_{0.80})^2\,\sigma_{agg}^2}{\Delta^2} \approx 15.7\,\frac{\sigma_{agg}^2}{\Delta^2}\ (+\,1\text{ for the }t\text{ correction}).$$
Paired seeds with correlation ρ: replace 2σ² by 2σ²(1−ρ). **Pairing across cells at the seed level is not expected to help** (methodological note): two cells with the same seed index train on different data, so ρ ≈ 0; the real "pairing" in this design is the shared base (a block, already exploited because it cancels within S) and the shared items (already exploited in the seed-level SD, §1.5). Planning therefore uses the unpaired formula; the paired columns below show what ρ = 0.3–0.5 would buy if the synthetic subsample index were deliberately shared across teachers (same prompt subset for every teacher within a seed — a design option worth adopting, since it costs nothing and can only reduce variance).

The Monte-Carlo version (exact t-test, `power_sim.py` Sim Table 1; "pooled" uses the residual variance from the whole 30-cell grid, df = 30(r−1), as the regression models do):

#### Sim Table 1. Seeds per cell for 80% power, two-sided alpha=0.05, difference of two cell means
(isolated = t test using only the two cells, df=2r-2; pooled = residual variance pooled over the 30-cell grid, df=30(r-1);
 paired rho=0.3 = seeds paired across the two conditions with correlation 0.3, isolated paired t test)

| sigma_seed (agg pt) | delta (pt) | r isolated | r pooled | r paired(rho=0.3) | r paired(rho=0.5) |
|---|---|---|---|---|---|
| 0.3 | 0.5 | 7 | 6 | 7 | 6 |
| 0.3 | 1.0 | 3 | 2 | 4 | 3 |
| 0.3 | 2.0 | 2 | 1 | 3 | 3 |
| 0.5 | 0.5 | 17 | 17 | 13 | 10 |
| 0.5 | 1.0 | 6 | 5 | 5 | 5 |
| 0.5 | 2.0 | 3 | 2 | 3 | 3 |
| 0.8 | 0.5 | >40 | >40 | 31 | 22 |
| 0.8 | 1.0 | 12 | 10 | 10 | 8 |
| 0.8 | 2.0 | 4 | 3 | 5 | 4 |
| 1.0 | 0.5 | >40 | >40 | >40 | 34 |
| 1.0 | 1.0 | 17 | 16 | 13 | 10 |
| 1.0 | 2.0 | 6 | 4 | 6 | 5 |

#### Sim Table 1b. Power (%) of the pooled-variance contrast at fixed r
| sigma | delta | r=1 | r=2 | r=3 | r=5 |
|---|---|---|---|---|---|
| 0.3 | 0.5 | 20 | 37 | 52 | 74 |
| 0.3 | 1.0 | 59 | 90 | 98 | 100 |
| 0.3 | 2.0 | 99 | 100 | 100 | 100 |
| 0.5 | 0.5 | 10 | 16 | 23 | 35 |
| 0.5 | 1.0 | 26 | 49 | 67 | 88 |
| 0.5 | 2.0 | 75 | 97 | 100 | 100 |
| 0.8 | 0.5 | 7 | 9 | 12 | 16 |
| 0.8 | 1.0 | 13 | 23 | 33 | 50 |
| 0.8 | 2.0 | 38 | 68 | 85 | 98 |

Note on the paired columns: with ρ = 0.3 the isolated paired t test (df = r − 1) sometimes needs *more* seeds than the unpaired test (df = 2r − 2) because the variance reduction 1 − ρ does not compensate for the lost degrees of freedom at small r; pairing pays only for ρ ≳ 0.5 or when the pooled residual variance is used.

Reading: with the central assumption σ_agg = 0.5, **1-point adjacent differences need 5–6 seeds per cell; 2-point differences need 2–3; 0.5-point differences need ≈ 17** and are out of reach. With σ_agg = 0.8 even 1-point effects need 10–12 seeds. The teacher-size grid steps are ×2–2.3, so adjacent effects on a saturating curve are expected at 0.2–1.0 pt (§3.3 surfaces): **adjacent-pair tests are not where the study's power lies; the fitted curve is** (§4).

Uncertainty of the pilot's own σ estimate (Sim Table 2): with 6 cells × 3 seeds (df = 12), the 95% CI for σ_agg spans 0.72–1.65 × σ̂. The full-study seed count must be set from the *upper* bound, not the point estimate.

#### Sim Table 2. Relative 95% CI for sigma_seed estimated from m cells x r seeds (df = m(r-1))
| cells m | seeds r | df | CI multiplier low | CI multiplier high |
|---|---|---|---|---|
| 6 | 2 | 6 | 0.64 | 2.23 |
| 6 | 3 | 12 | 0.72 | 1.65 |
| 6 | 4 | 18 | 0.76 | 1.48 |
| 8 | 3 | 16 | 0.74 | 1.52 |
| 30 | 3 | 60 | 0.85 | 1.22 |
| 30 | 5 | 120 | 0.89 | 1.14 |

### 3.2 Full-grid simulation: what was simulated

`power_sim.py` simulates the 5 × 6 grid under four response surfaces (true cell means in aggregate points), Gaussian seed noise σ_agg ∈ {0.3, 0.5, 0.8}, and designs U1/U2/U3/U5 (uniform r), **T3+2** (3 seeds everywhere, then +2 seeds at the empirically best teacher and its two neighbours for each S — adaptive) and **E3+2** (3 everywhere, +2 at T = 1.5B, 32B, 72B for each S — fixed, non-adaptive, targets curvature/saturation). 2000 replicates per configuration.

Surfaces (all with an arbitrary S intercept):
- `null`: no T effect.
- `lin`: +0.6 pt per doubling of T, no curvature.
- `sat R,β`: P = a_S + R·[1 − (T/1.5B)^{−β}] / [1 − (72/1.5)^{−β}]; monotone saturating, total range R over the grid. R = 2, β = 0.5 gives adjacent gains 0.68 / 0.57 / 0.32 / 0.26 / 0.17 pt; R = 4 doubles them.
- `satS`: as `sat` (R = 3) but the range grows with S (0.6R at 25M → 1.4R at 1B): a genuine S × T interaction without an interior optimum.
- `peak κ`: P = a_S − κ·(log₂T − log₂T*(S))², T*(S) = 3B·(S/25M)^{0.64} (3B at 25M … 32B at 1B). κ = 0.08 / 0.15 / 0.30 means a drop of 1.7 / 3.2 / 6.4 pt at 4.6 octaves from the optimum (i.e. between 3B and 72B), and only 0.08 / 0.15 / 0.30 pt one octave away.

Tests, as they will be run on the real data (§4–5): (a) T effect conditional on S = joint F-test of the linear and quadratic log T terms for that S in the per-S quadratic model (pooled residual variance; and a within-S-variance variant for 25M); (b) diminishing returns = one-sided test of negative curvature (common-curvature model; per-S model; and the nonparametric contrast "first adjacent gain − last adjacent gain > 0"); (c) S × T interaction = F-test of the continuous interaction (s·x, s·x²; 2 df) and of the full per-S curves (8 df) against a common curve; (d) T* = vertex of the per-S quadratic with a parametric bootstrap CI; "interior optimum declared" = curvature significantly negative AND the 95% CI of x* excludes the largest teacher; "saturation declared" = one-sided 95% upper bound of the top-step gain (32B → 72B) below δ, either from the two cell means (nonparametric) or from the fitted per-S quadratic (model-based), δ = 0.5 or 1.0 pt.

### 3.3 Results

#### Sim Table 3. Full 5x6 grid simulation: rejection / success rates (%) by surface, sigma, design
(Under 'null' and 'lin', columns (b),(c),(d-interior) are FALSE-POSITIVE rates; under 'sat' (c) is a false-positive rate;
 'sat np .5' = nonparametric two-cell saturation test (T6 vs T5, delta=0.5); 'sat mod .5'/'sat mod 1' = model-based top-step
 gain from the per-S quadratic with delta=0.5 / 1.0.  Saturation is TRUE for null/sat/peak (top-step gain <= 0.26) and FALSE
 for lin at delta=0.5 (true gain 0.70) but TRUE for lin at delta=1.0.  Surfaces: sat R=2 -> adjacent gains 0.68/0.57/0.32/0.26/0.17;
 sat R=4 -> double; peak k -> drop of k*(distance in octaves)^2 from the optimum, i.e. 1.7/3.2/6.4 pt at 4.6 octaves for k=.08/.15/.3.)

| surface | sigma | design | runs | (a)T|S mean | (a)T|25M | (a)T|25M wS | (b)curv common | (b)curv perS | (b)nonpar | (c)SxT 2df | (c)SxT 8df | (d)T* in 2x | (d)T* CI cov | (d)CI w(oct) | (d)interior | (d)sat np .5 | (d)sat mod .5 | (d)sat mod 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null | 0.3 | U1 | 30 | 5 | 5 | 4 | 5 | 5 | 4 | 5 | 5 | - | - | 5.6 | 3 | 31 | 72 | 100 |
| null | 0.3 | U2 | 60 | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 6 | - | - | 5.6 | 3 | 50 | 94 | 100 |
| null | 0.3 | U3 | 90 | 5 | 5 | 4 | 4 | 5 | 4 | 5 | 5 | - | - | 5.6 | 3 | 64 | 99 | 100 |
| null | 0.3 | U5 | 150 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | - | - | 5.6 | 3 | 83 | 100 | 100 |
| null | 0.3 | T3+2 | 117 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | - | - | 5.6 | 3 | 73 | 100 | 100 |
| null | 0.3 | E3+2 | 120 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | - | - | 5.6 | 3 | 83 | 100 | 100 |
| null | 0.5 | U1 | 30 | 5 | 5 | 5 | 6 | 5 | 5 | 5 | 4 | - | - | 5.6 | 4 | 17 | 39 | 85 |
| null | 0.5 | U2 | 60 | 5 | 4 | 5 | 4 | 5 | 5 | 5 | 4 | - | - | 5.6 | 3 | 26 | 61 | 99 |
| null | 0.5 | U3 | 90 | 5 | 5 | 5 | 4 | 5 | 4 | 4 | 5 | - | - | 5.6 | 3 | 33 | 77 | 100 |
| null | 0.5 | U5 | 150 | 5 | 5 | 5 | 6 | 5 | 5 | 4 | 5 | - | - | 5.6 | 3 | 47 | 93 | 100 |
| null | 0.5 | T3+2 | 117 | 5 | 6 | 5 | 5 | 5 | 5 | 5 | 5 | - | - | 5.6 | 3 | 37 | 85 | 100 |
| null | 0.5 | E3+2 | 120 | 5 | 5 | 5 | 5 | 5 | 5 | 6 | 6 | - | - | 5.6 | 3 | 48 | 87 | 100 |
| null | 0.8 | U1 | 30 | 5 | 5 | 6 | 5 | 5 | 4 | 5 | 5 | - | - | 5.6 | 4 | 10 | 20 | 50 |
| null | 0.8 | U2 | 60 | 5 | 6 | 6 | 6 | 5 | 6 | 5 | 6 | - | - | 5.6 | 3 | 15 | 34 | 79 |
| null | 0.8 | U3 | 90 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | - | - | 5.6 | 4 | 18 | 43 | 90 |
| null | 0.8 | U5 | 150 | 5 | 6 | 5 | 5 | 5 | 5 | 6 | 5 | - | - | 5.6 | 3 | 26 | 61 | 98 |
| null | 0.8 | T3+2 | 117 | 5 | 6 | 5 | 5 | 5 | 4 | 5 | 5 | - | - | 5.6 | 4 | 20 | 51 | 96 |
| null | 0.8 | E3+2 | 120 | 5 | 4 | 5 | 4 | 5 | 5 | 4 | 5 | - | - | 5.6 | 3 | 26 | 54 | 97 |
| lin | 0.3 | U1 | 30 | 100 | 100 | 100 | 5 | 5 | 2 | 5 | 5 | - | - | 0.0 | 0 | 1 | 1 | 37 |
| lin | 0.3 | U2 | 60 | 100 | 100 | 100 | 5 | 5 | 1 | 4 | 4 | - | - | 0.0 | 0 | 1 | 0 | 61 |
| lin | 0.3 | U3 | 90 | 100 | 100 | 100 | 5 | 5 | 1 | 5 | 5 | - | - | 0.0 | 0 | 1 | 0 | 76 |
| lin | 0.3 | U5 | 150 | 100 | 100 | 100 | 5 | 5 | 1 | 4 | 4 | - | - | 0.0 | 0 | 0 | 0 | 92 |
| lin | 0.3 | T3+2 | 110 | 100 | 100 | 100 | 5 | 5 | 1 | 4 | 5 | - | - | 0.0 | 0 | 0 | 0 | 85 |
| lin | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 5 | 5 | 1 | 6 | 5 | - | - | 0.0 | 0 | 0 | 0 | 87 |
| lin | 0.5 | U1 | 30 | 100 | 100 | 81 | 5 | 5 | 3 | 6 | 5 | - | - | 0.5 | 0 | 2 | 2 | 20 |
| lin | 0.5 | U2 | 60 | 100 | 100 | 100 | 6 | 5 | 3 | 4 | 5 | - | - | 0.0 | 0 | 2 | 1 | 31 |
| lin | 0.5 | U3 | 90 | 100 | 100 | 100 | 5 | 5 | 2 | 5 | 5 | - | - | 0.0 | 0 | 1 | 0 | 41 |
| lin | 0.5 | U5 | 150 | 100 | 100 | 100 | 5 | 5 | 2 | 5 | 5 | - | - | 0.0 | 0 | 1 | 0 | 57 |
| lin | 0.5 | T3+2 | 110 | 100 | 100 | 100 | 6 | 5 | 1 | 5 | 5 | - | - | 0.0 | 0 | 1 | 0 | 49 |
| lin | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 5 | 5 | 1 | 5 | 6 | - | - | 0.0 | 0 | 1 | 0 | 50 |
| lin | 0.8 | U1 | 30 | 81 | 82 | 45 | 5 | 5 | 4 | 5 | 5 | - | - | 1.6 | 0 | 3 | 2 | 13 |
| lin | 0.8 | U2 | 60 | 99 | 99 | 97 | 5 | 5 | 3 | 5 | 6 | - | - | 0.9 | 0 | 3 | 2 | 18 |
| lin | 0.8 | U3 | 90 | 100 | 100 | 100 | 4 | 5 | 3 | 5 | 5 | - | - | 0.4 | 0 | 2 | 1 | 22 |
| lin | 0.8 | U5 | 150 | 100 | 100 | 100 | 5 | 5 | 3 | 5 | 5 | - | - | 0.0 | 0 | 2 | 1 | 31 |
| lin | 0.8 | T3+2 | 112 | 100 | 100 | 100 | 5 | 5 | 3 | 5 | 5 | - | - | 0.0 | 0 | 2 | 1 | 26 |
| lin | 0.8 | E3+2 | 120 | 100 | 100 | 100 | 5 | 5 | 3 | 4 | 5 | - | - | 0.0 | 0 | 2 | 1 | 27 |
| sat R=2.0 b=0.5 | 0.3 | U1 | 30 | 100 | 100 | 80 | 94 | 41 | 59 | 5 | 5 | - | - | 1.6 | 3 | 19 | 53 | 99 |
| sat R=2.0 b=0.5 | 0.3 | U2 | 60 | 100 | 100 | 100 | 100 | 65 | 84 | 4 | 5 | - | - | 1.3 | 2 | 29 | 80 | 100 |
| sat R=2.0 b=0.5 | 0.3 | U3 | 90 | 100 | 100 | 100 | 100 | 81 | 95 | 4 | 4 | - | - | 1.2 | 2 | 38 | 93 | 100 |
| sat R=2.0 b=0.5 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 95 | 99 | 5 | 4 | - | - | 1.0 | 1 | 53 | 99 | 100 |
| sat R=2.0 b=0.5 | 0.3 | T3+2 | 113 | 100 | 100 | 100 | 100 | 86 | 98 | 5 | 5 | - | - | 1.0 | 1 | 55 | 97 | 100 |
| sat R=2.0 b=0.5 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 90 | 99 | 5 | 5 | - | - | 1.0 | 1 | 52 | 97 | 100 |
| sat R=2.0 b=0.5 | 0.5 | U1 | 30 | 78 | 79 | 41 | 63 | 21 | 32 | 4 | 5 | - | - | 2.1 | 3 | 11 | 27 | 77 |
| sat R=2.0 b=0.5 | 0.5 | U2 | 60 | 99 | 99 | 96 | 88 | 33 | 50 | 5 | 5 | - | - | 1.8 | 2 | 16 | 44 | 96 |
| sat R=2.0 b=0.5 | 0.5 | U3 | 90 | 100 | 100 | 100 | 96 | 45 | 63 | 5 | 5 | - | - | 1.6 | 2 | 20 | 58 | 99 |
| sat R=2.0 b=0.5 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 64 | 82 | 5 | 5 | - | - | 1.4 | 2 | 28 | 78 | 100 |
| sat R=2.0 b=0.5 | 0.5 | T3+2 | 114 | 100 | 100 | 100 | 98 | 49 | 72 | 5 | 5 | - | - | 1.4 | 2 | 28 | 67 | 100 |
| sat R=2.0 b=0.5 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 99 | 54 | 75 | 4 | 5 | - | - | 1.4 | 2 | 27 | 69 | 100 |
| sat R=2.0 b=0.5 | 0.8 | U1 | 30 | 39 | 40 | 20 | 32 | 13 | 17 | 6 | 6 | - | - | 2.7 | 3 | 8 | 16 | 43 |
| sat R=2.0 b=0.5 | 0.8 | U2 | 60 | 75 | 74 | 60 | 55 | 19 | 27 | 6 | 5 | - | - | 2.2 | 3 | 10 | 24 | 69 |
| sat R=2.0 b=0.5 | 0.8 | U3 | 90 | 91 | 91 | 84 | 70 | 25 | 34 | 4 | 5 | - | - | 2.0 | 2 | 13 | 32 | 84 |
| sat R=2.0 b=0.5 | 0.8 | U5 | 150 | 99 | 99 | 98 | 85 | 33 | 48 | 4 | 5 | - | - | 1.8 | 2 | 16 | 44 | 96 |
| sat R=2.0 b=0.5 | 0.8 | T3+2 | 115 | 95 | 95 | 92 | 75 | 28 | 40 | 5 | 5 | - | - | 1.9 | 2 | 16 | 38 | 92 |
| sat R=2.0 b=0.5 | 0.8 | E3+2 | 120 | 99 | 99 | 98 | 78 | 28 | 42 | 5 | 5 | - | - | 1.9 | 2 | 15 | 37 | 92 |
| sat R=4.0 b=0.5 | 0.3 | U1 | 30 | 100 | 100 | 100 | 100 | 86 | 96 | 4 | 4 | - | - | 1.1 | 2 | 10 | 32 | 96 |
| sat R=4.0 b=0.5 | 0.3 | U2 | 60 | 100 | 100 | 100 | 100 | 99 | 100 | 4 | 4 | - | - | 0.8 | 1 | 13 | 55 | 100 |
| sat R=4.0 b=0.5 | 0.3 | U3 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 4 | 4 | - | - | 0.7 | 1 | 16 | 71 | 100 |
| sat R=4.0 b=0.5 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 4 | 4 | - | - | 0.5 | 1 | 21 | 89 | 100 |
| sat R=4.0 b=0.5 | 0.3 | T3+2 | 111 | 100 | 100 | 100 | 100 | 100 | 100 | 4 | 4 | - | - | 0.5 | 1 | 20 | 77 | 100 |
| sat R=4.0 b=0.5 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 4 | 3 | - | - | 0.5 | 1 | 20 | 80 | 100 |
| sat R=4.0 b=0.5 | 0.5 | U1 | 30 | 100 | 100 | 91 | 98 | 51 | 73 | 5 | 5 | - | - | 1.5 | 3 | 7 | 18 | 65 |
| sat R=4.0 b=0.5 | 0.5 | U2 | 60 | 100 | 100 | 100 | 100 | 79 | 94 | 5 | 4 | - | - | 1.2 | 2 | 10 | 28 | 91 |
| sat R=4.0 b=0.5 | 0.5 | U3 | 90 | 100 | 100 | 100 | 100 | 92 | 99 | 5 | 5 | - | - | 1.0 | 2 | 11 | 38 | 98 |
| sat R=4.0 b=0.5 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 99 | 100 | 6 | 4 | - | - | 0.8 | 2 | 13 | 54 | 100 |
| sat R=4.0 b=0.5 | 0.5 | T3+2 | 112 | 100 | 100 | 100 | 100 | 94 | 100 | 5 | 5 | - | - | 0.9 | 1 | 12 | 41 | 99 |
| sat R=4.0 b=0.5 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 100 | 97 | 100 | 4 | 4 | - | - | 0.9 | 1 | 13 | 45 | 100 |
| sat R=4.0 b=0.5 | 0.8 | U1 | 30 | 94 | 94 | 57 | 77 | 27 | 40 | 4 | 5 | - | - | 1.9 | 3 | 6 | 12 | 36 |
| sat R=4.0 b=0.5 | 0.8 | U2 | 60 | 100 | 100 | 100 | 96 | 45 | 62 | 6 | 4 | - | - | 1.6 | 2 | 7 | 16 | 59 |
| sat R=4.0 b=0.5 | 0.8 | U3 | 90 | 100 | 100 | 100 | 100 | 60 | 78 | 5 | 5 | - | - | 1.4 | 2 | 8 | 21 | 75 |
| sat R=4.0 b=0.5 | 0.8 | U5 | 150 | 100 | 100 | 100 | 100 | 78 | 94 | 5 | 5 | - | - | 1.2 | 2 | 9 | 29 | 91 |
| sat R=4.0 b=0.5 | 0.8 | T3+2 | 113 | 100 | 100 | 100 | 100 | 66 | 88 | 5 | 4 | - | - | 1.3 | 2 | 10 | 24 | 84 |
| sat R=4.0 b=0.5 | 0.8 | E3+2 | 120 | 100 | 100 | 100 | 100 | 70 | 90 | 6 | 5 | - | - | 1.3 | 2 | 9 | 24 | 84 |
| satS R=3.0 b=0.5 | 0.3 | U1 | 30 | 100 | 99 | 73 | 99 | 68 | 88 | 99 | 86 | - | - | 1.3 | 3 | 13 | 40 | 97 |
| satS R=3.0 b=0.5 | 0.3 | U2 | 60 | 100 | 100 | 100 | 100 | 87 | 100 | 100 | 100 | - | - | 1.0 | 2 | 19 | 66 | 100 |
| satS R=3.0 b=0.5 | 0.3 | U3 | 90 | 100 | 100 | 100 | 100 | 94 | 100 | 100 | 100 | - | - | 0.9 | 1 | 24 | 80 | 100 |
| satS R=3.0 b=0.5 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | - | - | 0.7 | 1 | 34 | 94 | 100 |
| satS R=3.0 b=0.5 | 0.3 | T3+2 | 112 | 100 | 100 | 100 | 100 | 95 | 100 | 100 | 100 | - | - | 0.7 | 1 | 34 | 86 | 100 |
| satS R=3.0 b=0.5 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | - | - | 0.7 | 1 | 34 | 88 | 100 |
| satS R=3.0 b=0.5 | 0.5 | U1 | 30 | 93 | 69 | 36 | 84 | 37 | 54 | 74 | 37 | - | - | 1.7 | 3 | 8 | 22 | 70 |
| satS R=3.0 b=0.5 | 0.5 | U2 | 60 | 99 | 97 | 91 | 99 | 61 | 81 | 98 | 85 | - | - | 1.4 | 2 | 12 | 36 | 94 |
| satS R=3.0 b=0.5 | 0.5 | U3 | 90 | 100 | 100 | 99 | 100 | 73 | 92 | 100 | 98 | - | - | 1.3 | 2 | 15 | 46 | 99 |
| satS R=3.0 b=0.5 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 86 | 99 | 100 | 100 | - | - | 1.0 | 2 | 19 | 64 | 100 |
| satS R=3.0 b=0.5 | 0.5 | T3+2 | 113 | 100 | 100 | 100 | 100 | 78 | 96 | 100 | 100 | - | - | 1.1 | 2 | 19 | 52 | 100 |
| satS R=3.0 b=0.5 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 100 | 81 | 98 | 100 | 100 | - | - | 1.1 | 1 | 19 | 55 | 100 |
| satS R=3.0 b=0.5 | 0.8 | U1 | 30 | 73 | 33 | 17 | 54 | 21 | 31 | 36 | 16 | - | - | 2.2 | 3 | 7 | 14 | 39 |
| satS R=3.0 b=0.5 | 0.8 | U2 | 60 | 92 | 65 | 52 | 84 | 34 | 50 | 69 | 39 | - | - | 1.8 | 3 | 9 | 20 | 64 |
| satS R=3.0 b=0.5 | 0.8 | U3 | 90 | 96 | 83 | 76 | 95 | 44 | 62 | 86 | 61 | - | - | 1.7 | 2 | 10 | 25 | 78 |
| satS R=3.0 b=0.5 | 0.8 | U5 | 150 | 99 | 97 | 96 | 100 | 60 | 81 | 98 | 88 | - | - | 1.4 | 2 | 12 | 34 | 93 |
| satS R=3.0 b=0.5 | 0.8 | T3+2 | 114 | 98 | 88 | 84 | 97 | 49 | 71 | 92 | 72 | - | - | 1.5 | 2 | 12 | 30 | 87 |
| satS R=3.0 b=0.5 | 0.8 | E3+2 | 120 | 99 | 97 | 95 | 98 | 53 | 76 | 97 | 87 | - | - | 1.5 | 2 | 13 | 30 | 88 |
| peak k=0.08 | 0.3 | U1 | 30 | 72 | 98 | 70 | 95 | 62 | 87 | 100 | 98 | 86 | 93 | 2.4 | 43 | 63 | 96 | 100 |
| peak k=0.08 | 0.3 | U2 | 60 | 92 | 100 | 100 | 100 | 88 | 99 | 100 | 100 | 94 | 94 | 1.9 | 63 | 85 | 100 | 100 |
| peak k=0.08 | 0.3 | U3 | 90 | 98 | 100 | 100 | 100 | 97 | 100 | 100 | 100 | 96 | 94 | 1.5 | 74 | 93 | 100 | 100 |
| peak k=0.08 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.2 | 85 | 99 | 100 | 100 |
| peak k=0.08 | 0.3 | T3+2 | 119 | 99 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 98 | 94 | 1.3 | 80 | 97 | 100 | 100 |
| peak k=0.08 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 98 | 95 | 1.3 | 81 | 99 | 100 | 100 |
| peak k=0.08 | 0.5 | U1 | 30 | 39 | 66 | 34 | 69 | 32 | 52 | 92 | 60 | 68 | 92 | 3.2 | 21 | 34 | 73 | 96 |
| peak k=0.08 | 0.5 | U2 | 60 | 66 | 96 | 89 | 95 | 53 | 79 | 100 | 97 | 81 | 94 | 2.6 | 34 | 54 | 92 | 100 |
| peak k=0.08 | 0.5 | U3 | 90 | 78 | 99 | 99 | 100 | 69 | 90 | 100 | 100 | 87 | 94 | 2.4 | 45 | 67 | 97 | 100 |
| peak k=0.08 | 0.5 | U5 | 150 | 91 | 100 | 100 | 100 | 86 | 99 | 100 | 100 | 93 | 94 | 2.0 | 60 | 82 | 100 | 100 |
| peak k=0.08 | 0.5 | T3+2 | 119 | 83 | 100 | 100 | 100 | 77 | 94 | 100 | 100 | 90 | 94 | 2.2 | 52 | 74 | 99 | 100 |
| peak k=0.08 | 0.5 | E3+2 | 120 | 88 | 100 | 100 | 100 | 79 | 97 | 100 | 100 | 90 | 94 | 2.1 | 53 | 83 | 99 | 100 |
| peak k=0.08 | 0.8 | U1 | 30 | 18 | 29 | 19 | 43 | 18 | 29 | 54 | 25 | 55 | 93 | 4.6 | 13 | 19 | 43 | 73 |
| peak k=0.08 | 0.8 | U2 | 60 | 35 | 62 | 49 | 71 | 28 | 46 | 87 | 60 | 66 | 94 | 3.8 | 18 | 30 | 67 | 94 |
| peak k=0.08 | 0.8 | U3 | 90 | 49 | 81 | 74 | 86 | 36 | 57 | 98 | 86 | 72 | 95 | 3.0 | 22 | 38 | 80 | 98 |
| peak k=0.08 | 0.8 | U5 | 150 | 67 | 97 | 95 | 98 | 53 | 77 | 100 | 99 | 80 | 94 | 2.6 | 34 | 54 | 91 | 100 |
| peak k=0.08 | 0.8 | T3+2 | 118 | 55 | 85 | 81 | 89 | 44 | 66 | 99 | 91 | 76 | 94 | 2.9 | 28 | 44 | 86 | 100 |
| peak k=0.08 | 0.8 | E3+2 | 120 | 61 | 93 | 90 | 94 | 45 | 73 | 100 | 98 | 77 | 94 | 2.7 | 29 | 54 | 86 | 100 |
| peak k=0.15 | 0.3 | U1 | 30 | 98 | 100 | 99 | 100 | 98 | 100 | 100 | 100 | 97 | 93 | 1.4 | 78 | 79 | 98 | 100 |
| peak k=0.15 | 0.3 | U2 | 60 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.0 | 90 | 93 | 100 | 100 |
| peak k=0.15 | 0.3 | U3 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.8 | 95 | 97 | 100 | 100 |
| peak k=0.15 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.6 | 99 | 100 | 100 | 100 |
| peak k=0.15 | 0.3 | T3+2 | 119 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.7 | 97 | 99 | 100 | 100 |
| peak k=0.15 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.7 | 97 | 100 | 100 | 100 |
| peak k=0.15 | 0.5 | U1 | 30 | 78 | 100 | 79 | 97 | 70 | 93 | 100 | 100 | 89 | 92 | 2.2 | 48 | 51 | 86 | 98 |
| peak k=0.15 | 0.5 | U2 | 60 | 96 | 100 | 100 | 100 | 94 | 100 | 100 | 100 | 96 | 94 | 1.6 | 70 | 72 | 97 | 100 |
| peak k=0.15 | 0.5 | U3 | 90 | 99 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 98 | 94 | 1.4 | 79 | 82 | 99 | 100 |
| peak k=0.15 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 95 | 1.1 | 89 | 92 | 100 | 100 |
| peak k=0.15 | 0.5 | T3+2 | 119 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.2 | 85 | 88 | 100 | 100 |
| peak k=0.15 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.2 | 85 | 92 | 100 | 100 |
| peak k=0.15 | 0.8 | U1 | 30 | 49 | 80 | 44 | 79 | 40 | 63 | 98 | 75 | 75 | 93 | 2.8 | 27 | 29 | 62 | 84 |
| peak k=0.15 | 0.8 | U2 | 60 | 75 | 99 | 96 | 99 | 64 | 88 | 100 | 100 | 85 | 94 | 2.4 | 41 | 45 | 82 | 97 |
| peak k=0.15 | 0.8 | U3 | 90 | 87 | 100 | 100 | 100 | 80 | 97 | 100 | 100 | 91 | 95 | 2.1 | 54 | 58 | 90 | 99 |
| peak k=0.15 | 0.8 | U5 | 150 | 96 | 100 | 100 | 100 | 94 | 100 | 100 | 100 | 95 | 94 | 1.7 | 69 | 73 | 97 | 100 |
| peak k=0.15 | 0.8 | T3+2 | 119 | 90 | 100 | 100 | 100 | 87 | 98 | 100 | 100 | 93 | 94 | 1.9 | 61 | 62 | 95 | 100 |
| peak k=0.15 | 0.8 | E3+2 | 120 | 93 | 100 | 100 | 100 | 89 | 99 | 100 | 100 | 94 | 95 | 1.8 | 63 | 71 | 95 | 100 |
| peak k=0.3 | 0.3 | U1 | 30 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 93 | 0.7 | 97 | 91 | 100 | 100 |
| peak k=0.3 | 0.3 | U2 | 60 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.5 | 100 | 98 | 100 | 100 |
| peak k=0.3 | 0.3 | U3 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.4 | 100 | 100 | 100 | 100 |
| peak k=0.3 | 0.3 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.3 | 100 | 100 | 100 | 100 |
| peak k=0.3 | 0.3 | T3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.4 | 100 | 100 | 100 | 100 |
| peak k=0.3 | 0.3 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.3 | 100 | 100 | 100 | 100 |
| peak k=0.3 | 0.5 | U1 | 30 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 93 | 1.2 | 85 | 73 | 95 | 100 |
| peak k=0.3 | 0.5 | U2 | 60 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.9 | 95 | 88 | 99 | 100 |
| peak k=0.3 | 0.5 | U3 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.7 | 98 | 93 | 100 | 100 |
| peak k=0.3 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.5 | 100 | 98 | 100 | 100 |
| peak k=0.3 | 0.5 | T3+2 | 119 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.6 | 99 | 97 | 100 | 100 |
| peak k=0.3 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.6 | 99 | 98 | 100 | 100 |
| peak k=0.3 | 0.8 | U1 | 30 | 89 | 100 | 92 | 100 | 86 | 98 | 100 | 100 | 93 | 93 | 1.8 | 62 | 50 | 81 | 93 |
| peak k=0.3 | 0.8 | U2 | 60 | 99 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 98 | 94 | 1.3 | 80 | 69 | 93 | 99 |
| peak k=0.3 | 0.8 | U3 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.1 | 88 | 78 | 97 | 100 |
| peak k=0.3 | 0.8 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.9 | 94 | 87 | 99 | 100 |
| peak k=0.3 | 0.8 | T3+2 | 119 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 1.0 | 92 | 85 | 99 | 100 |
| peak k=0.3 | 0.8 | E3+2 | 120 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 0.9 | 92 | 88 | 99 | 100 |

#### Sim Table 4. Heteroscedastic check (sigma_25M = 1.5 sigma, sigma_100M = 1.2 sigma), sigma=0.5
| surface | sigma | design | runs | (a)T|S mean | (a)T|25M | (a)T|25M wS | (b)curv common | (b)curv perS | (b)nonpar | (c)SxT 2df | (c)SxT 8df | (d)T* in 2x | (d)T* CI cov | (d)CI w(oct) | (d)interior | (d)sat np .5 | (d)sat mod .5 | (d)sat mod 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null | 0.5 | U3 | 90 | 5 | 16 | 5 | 4 | 4 | 5 | 10 | 5 | - | - | 5.6 | 3 | 28 | 67 | 99 |
| null | 0.5 | U5 | 150 | 6 | 17 | 5 | 5 | 5 | 5 | 9 | 6 | - | - | 5.6 | 3 | 39 | 85 | 100 |
| null | 0.5 | T3+2 | 117 | 6 | 17 | 5 | 6 | 5 | 4 | 9 | 6 | - | - | 5.6 | 3 | 31 | 76 | 100 |
| null | 0.5 | E3+2 | 120 | 5 | 16 | 5 | 5 | 5 | 5 | 9 | 6 | - | - | 5.6 | 3 | 39 | 78 | 100 |
| sat R=2.0 b=0.5 | 0.5 | U3 | 90 | 99 | 98 | 89 | 90 | 36 | 53 | 10 | 6 | - | - | 1.7 | 2 | 16 | 48 | 98 |
| sat R=2.0 b=0.5 | 0.5 | U5 | 150 | 100 | 100 | 99 | 98 | 52 | 70 | 9 | 5 | - | - | 1.5 | 2 | 21 | 67 | 100 |
| sat R=2.0 b=0.5 | 0.5 | T3+2 | 114 | 100 | 99 | 95 | 94 | 41 | 63 | 10 | 5 | - | - | 1.6 | 2 | 22 | 58 | 99 |
| sat R=2.0 b=0.5 | 0.5 | E3+2 | 120 | 100 | 100 | 98 | 96 | 44 | 66 | 9 | 5 | - | - | 1.6 | 2 | 22 | 58 | 99 |
| peak k=0.15 | 0.5 | U3 | 90 | 98 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 97 | 93 | 1.6 | 72 | 75 | 98 | 100 |
| peak k=0.15 | 0.5 | U5 | 150 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 94 | 1.2 | 84 | 88 | 100 | 100 |
| peak k=0.15 | 0.5 | T3+2 | 119 | 99 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 98 | 94 | 1.4 | 79 | 82 | 100 | 100 |
| peak k=0.15 | 0.5 | E3+2 | 120 | 100 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 98 | 94 | 1.3 | 80 | 89 | 100 | 100 |

#### Sim Table 5. Exponent alpha recovered from the 5 per-S T* estimates (peak surface, true alpha=0.64)
| kappa | sigma | design | mean alpha_hat | SD alpha_hat | 95% CI half-width (t, 3 df, using SD) | P(|alpha_hat-0.64|<0.2) |
|---|---|---|---|---|---|---|
| 0.08 | 0.3 | U3 | 0.68 | 0.12 | 0.39 | 88 |
| 0.08 | 0.3 | U5 | 0.67 | 0.10 | 0.31 | 94 |
| 0.08 | 0.3 | T3+2 | 0.67 | 0.11 | 0.34 | 93 |
| 0.08 | 0.3 | E3+2 | 0.68 | 0.11 | 0.34 | 92 |
| 0.08 | 0.5 | U3 | 0.70 | 0.17 | 0.54 | 72 |
| 0.08 | 0.5 | U5 | 0.69 | 0.15 | 0.46 | 80 |
| 0.08 | 0.5 | T3+2 | 0.69 | 0.15 | 0.49 | 79 |
| 0.08 | 0.5 | E3+2 | 0.69 | 0.15 | 0.49 | 78 |
| 0.08 | 0.8 | U3 | 0.70 | 0.21 | 0.68 | 62 |
| 0.08 | 0.8 | U5 | 0.71 | 0.19 | 0.59 | 67 |
| 0.08 | 0.8 | T3+2 | 0.69 | 0.21 | 0.65 | 64 |
| 0.08 | 0.8 | E3+2 | 0.70 | 0.20 | 0.62 | 65 |
| 0.15 | 0.3 | U3 | 0.65 | 0.07 | 0.21 | 99 |
| 0.15 | 0.3 | U5 | 0.65 | 0.05 | 0.17 | 100 |
| 0.15 | 0.3 | T3+2 | 0.65 | 0.06 | 0.19 | 100 |
| 0.15 | 0.3 | E3+2 | 0.65 | 0.06 | 0.19 | 99 |
| 0.15 | 0.5 | U3 | 0.68 | 0.11 | 0.36 | 91 |
| 0.15 | 0.5 | U5 | 0.66 | 0.09 | 0.28 | 97 |
| 0.15 | 0.5 | T3+2 | 0.67 | 0.10 | 0.33 | 94 |
| 0.15 | 0.5 | E3+2 | 0.67 | 0.10 | 0.31 | 95 |
| 0.15 | 0.8 | U3 | 0.69 | 0.16 | 0.50 | 78 |
| 0.15 | 0.8 | U5 | 0.68 | 0.13 | 0.41 | 87 |
| 0.15 | 0.8 | T3+2 | 0.68 | 0.14 | 0.45 | 82 |
| 0.15 | 0.8 | E3+2 | 0.69 | 0.14 | 0.44 | 83 |
| 0.3 | 0.3 | U3 | 0.64 | 0.03 | 0.11 | 100 |
| 0.3 | 0.3 | U5 | 0.64 | 0.03 | 0.08 | 100 |
| 0.3 | 0.3 | T3+2 | 0.64 | 0.03 | 0.09 | 100 |
| 0.3 | 0.3 | E3+2 | 0.64 | 0.03 | 0.09 | 100 |
| 0.3 | 0.5 | U3 | 0.65 | 0.06 | 0.18 | 100 |
| 0.3 | 0.5 | U5 | 0.65 | 0.05 | 0.14 | 100 |
| 0.3 | 0.5 | T3+2 | 0.65 | 0.05 | 0.16 | 100 |
| 0.3 | 0.5 | E3+2 | 0.65 | 0.05 | 0.15 | 100 |
| 0.3 | 0.8 | U3 | 0.66 | 0.09 | 0.29 | 96 |
| 0.3 | 0.8 | U5 | 0.66 | 0.07 | 0.23 | 99 |
| 0.3 | 0.8 | T3+2 | 0.66 | 0.08 | 0.26 | 98 |
| 0.3 | 0.8 | E3+2 | 0.66 | 0.08 | 0.25 | 98 |


**Reading the tables (main findings).**

1. *Calibration.* Under `null` all tests reject at 4–6% (Sim Table 3): the F/t machinery with a pooled residual variance is correctly sized when the variance is homogeneous. Under heteroscedastic noise (Sim Table 4) the per-S test at 25M with pooled variance rejects 16–17% of the time under the null, and the 2-df interaction test 9–10%; the within-S-variance version of the per-S test stays at 5%. Per-S tests will therefore use within-S variance and pooled tests HC3/variance-model SEs (§4.2).
2. *T main effect (a).* Easy: for any surface with ≥ 2 pt range and σ ≤ 0.5, power ≈ 100% already at r = 2–3. At σ = 0.8 and R = 2, r = 3 gives ≈ 91%; r = 1 only 39%.
3. *Diminishing returns (b).* The **pooled common-curvature test** is well powered at r = 3 (96% for R = 2, σ = 0.5; 70% at σ = 0.8), the **per-S curvature test is not** (45% / 25%). Curvature is a pooled, hierarchical claim in this study, not a per-cell one.
4. *S × T interaction (c).* Detectable at r = 3 when it is large (`satS`: 100% at σ ≤ 0.5, 86% at 0.8; `peak` surfaces: 100% because T* moves by 3.4 octaves across S). Under `sat` it is a correctly-sized false-positive rate. The study cannot detect *subtle* interactions (e.g. a 20% change in range across S) at any affordable r — a limitation to state.
5. *T* recovery (d).* Where an interior optimum exists with κ ≥ 0.15, the vertex is within 2× of the truth in ≥ 98% of replicates at r = 3, σ ≤ 0.5, and the "interior optimum" rule fires in 79% (κ = 0.15) / 98% (κ = 0.30) of replicates with ≈ 3% false declarations under `null`/`sat`. With a **weak peak (κ = 0.08, ≈ 1.7 pt drop over 4.6 octaves)** localisation degrades: at r = 3, σ = 0.5 the vertex is within 2× in 87% of replicates but the CI is 2.4 octaves wide and the interior-optimum rule fires in only 45% (22% at σ = 0.8). This is the regime where "an intermediate teacher is best" is true but the data will mostly return "flat plateau" — which is the honest answer, and the pre-registered vocabulary (§8) must allow it. CI coverage of the parametric bootstrap is 92–95% throughout; the adaptive T3+2 design does not distort coverage.
6. *Saturation.* The two-cell nonparametric rule with δ = 0.5 is hopeless at any affordable r (20–47% at σ = 0.5 for r = 3–5 even when saturation is true): an equivalence test with margin δ needs SE ≈ (δ − true gain)/1.65, i.e. r ≳ 12 at σ = 0.5. The **model-based** version (top-step gain from the fitted per-S quadratic, which borrows strength from all six teacher levels) reaches 58–78% (`sat R=2`, true top-step gain 0.17) and 77–93% (`null`) at r = 3–5, σ = 0.5 for δ = 0.5, and ≥ 98% for δ = 1.0, while correctly *not* declaring saturation under `lin` at δ = 0.5 (0–2%). **δ is therefore set to 1.0 pt for the primary saturation claim and 0.5 pt as a sensitivity margin**, and the claim is model-based.
7. *Tiered designs.* At σ = 0.5: for `peak κ=0.15` the interior-optimum rule fires in 79% (U3, 90 runs) → 85% (T3+2, 119 runs) → 85% (E3+2, 120 runs) → 89% (U5, 150 runs), with median CI widths 1.4 → 1.2 → 1.2 → 1.1 octaves; for `sat R=2` the per-S curvature power is 45 → 49 → 54 → 64% and the model-based saturation (δ = 0.5) 58 → 67 → 69 → 78%. So either tier recovers roughly half of the U5 gain for 40% of the extra cost, and **E3+2 is at least as good as T3+2 on every criterion while being non-adaptive** (the ends of the T grid carry the leverage for both curvature and the vertex). Neither is dramatic: **the dominant lever is σ_agg, not r** (compare σ = 0.3 at r = 1 with σ = 0.8 at r = 5 anywhere in Sim Table 3). Variance reduction — floor-gating, dropping OpenBookQA-scale sets, continuous task loss, averaging the last few synthetic-phase checkpoints — buys more than seeds.
8. *α (Sim Table 5).* Even with a clean interior optimum at every S (κ = 0.15) and r = 3, the SD of α̂ is 0.11 (σ = 0.5) to 0.16 (σ = 0.8), i.e. a 95% CI half-width of ≈ ±0.36 to ±0.50 from 5 points; with a weak peak (κ = 0.08) it is ±0.54 to ±0.68 and α̂ is biased upward by ≈ 0.05 (boundary clamping). The exponent will be reported as descriptive with its CI; the design cannot distinguish, say, α = 0.5 from α = 0.8.

### 3.4 Recommended seeds

| Stage | Cells | Seeds | Runs | Justification |
|---|---|---|---|---|
| **Pilot** | 2 S × 3 T | **3** | 18 | + C1 × 2 S × 3 seeds (6) + C0 (2, plus 2 extra base seeds at 25M) + D sweep (6) ≈ 34 runs. Gives df = 16 for σ̂_agg (CI ×0.74–1.52). The pooled synthetic-vs-C1 test (9 vs 3 runs, df = 16) detects a 2-pt gain with 94% power at σ = 0.8 and a 1-pt gain with 80% at σ = 0.5 (42% at σ = 0.8). Two seeds would leave df = 8 and a CI of ×0.64–2.2 for σ̂ — too wide to plan the full study. |
| **Full, tier 1** | all 30 cells | **3** | 90 | Sufficient for (a), pooled (b), large (c); the minimum at which a within-cell SD exists in every cell (r = 2 gives 1 df per cell). |
| **Full, tier 2** | per S: T = 1.5B, 32B, 72B (E-tier) **and** the argmax cell ± 1 from tier 1 (T-tier; overlaps allowed) | **+2 → 5** | ≈ 40–60 | Ends: curvature and the saturation claim (E3+2 ≈ U5 on those). Argmax ± 1: T* localisation and a *confirmation sample* for any "intermediate beats largest" claim (§5.4: winner's-curse protection). |
| **Controls** | C1 at every S | **5** | 25 | Every synthetic cell is compared with C1; C1 enters 6 contrasts per S, so its mean deserves the lowest SE (optimal allocation ∝ √(number of comparisons)). |
| **Base replicates** | C0 at 25M, 100M | 3 (i.e. +2 each) | 4 | Estimates σ_base for the cross-S caveat; cheapest sizes only. |
| **D sweep** | 2 S × 3 T × 3 extra D levels | 2 | 36 | §4.4. |
| Total | | | ≈ 195–215 synthetic-phase runs + 5 base runs (+ 4 base replicates) | Every synthetic-phase run is short (D_syn tokens); the base runs dominate compute. |

Decision rule for r after the pilot: take σ̂_agg^{upper} (upper 95% bound). If σ̂^{upper} ≤ 0.6: tier 1 = 3. If 0.6 < σ̂^{upper} ≤ 0.9: tier 1 = 4 and tier 2 = +2. If σ̂^{upper} > 0.9: do not add seeds; instead reduce variance (drop noisy benchmarks; switch the primary to task loss; increase D_syn) and re-pilot one cell.

---

## 4. Regression methodology

### 4.1 Model sequence (primary outcome P_cc; repeated for each secondary outcome)

Observation: run i in cell (S,T), y_i = P_cc,i; s_i, x_i as in §0; α_S = per-S fixed intercept (absorbs the base run and the S main effect, both of which are nuisance here).

| Model | Mean structure | Parameters | Tests |
|---|---|---|---|
| M0 | α_S | 5 | reference |
| M1 | α_S + β₁x | 6 | M1 vs M0: H1 (any monotone T effect) |
| M2 | α_S + β₁x + β₂x² | 7 | β₂ < 0 one-sided: H2 (diminishing returns); M2 vs M1 |
| M3 | α_S + β₁x + β₂x² + γ₁ s·x + γ₂ s·x² | 9 | M3 vs M2 (2 df): H4-precursor (does the T curve depend on S?) |
| M3cat | α_S + β_{1,S} x + β_{2,S} x² | 15 | M3cat vs M2 (8 df): unrestricted S × T; M3cat vs M3 (6 df): is the log-linear interaction adequate? Also the per-S model used for T*(S) |
| M4 | M3 + δ₁d + δ₂ d·x + δ₃ d·s (+ δ₄ d²) on the D-sweep subset ∪ main grid at d = 0 | 12–13 | δ₂ ≠ 0: does the teacher effect change with data quantity (the Bansal et al. 2024 compute-matched question) |
| M2sat | α_S + A_S·(1 − T^{−β}) (nonlinear least squares) | 11 | AICc comparison with M3cat: saturating vs interior-optimum form (§5.1) |

Fitting: OLS for M0–M3cat; nonlinear LS (Levenberg–Marquardt, multiple starts) for M2sat; ML for the variance model below. Nested comparisons by F-tests (OLS) or likelihood-ratio tests (ML); non-nested (M2sat vs M3cat) by AICc (n ≤ 150, so AICc not AIC) and by the T* decision rule of §5.2. Report all coefficient estimates with 95% CIs, not only p-values.

### 4.2 Error structure

- **Seed random intercept?** Each run contributes one observation of P_cc, so a "seed" random effect is not separable from the residual; no mixed model is needed at the aggregate level. Mixed models are used where a run contributes several observations: (i) the per-benchmark stacked analysis y_{ik} = α_{S,k} + curve_k(x) + u_i + ε_{ik}, u_i ~ N(0, τ²) a run random intercept (or, equivalently for inference, cluster-robust SEs by run); (ii) the D sweep if several D levels are cut from one run (they must not be — each D level is its own run, otherwise checkpoints within a run are serially correlated).
- **Heteroscedasticity across S** (expected, §2.2): fit the variance model log σ_S = θ₀ + θ₁ s by ML jointly with the mean model (or, simpler and pre-registered as the primary: per-S residual variance σ̂²_S from M3cat, df = 6r_S − 3, used for all per-S tests; HC3 sandwich SEs for all pooled coefficient tests). Test for heteroscedasticity with Brown–Forsythe across S on M3cat residuals; report σ̂_S per S in the cell table regardless.
- **Diagnostics (reported in the appendix):** residuals vs fitted; residual SD by S and by T; QQ plot of standardised residuals; Cook's distance per run (with r = 3 a single divergent run has leverage ≈ 1/3 of its cell) — any run with Cook's D > 4/n is flagged and the analysis is repeated without it; both versions are reported.
- **Multiplicity of outcomes:** the primary analysis is P_cc; secondary outcomes are reported with the same models, unadjusted, labelled secondary.

### 4.3 Nonparametric per-S analyses (robustness to the quadratic form)

1. **Monotone (isotonic) fit and tests.** For each S, fit the isotonic (non-decreasing in T) regression of the r_T seed values on T. (i) Test of "some monotone T effect": statistic = RSS(constant) − RSS(isotonic); null distribution by permuting runs among the six cells within S (exchangeable under H0). (ii) Test of "non-monotone" (H3): statistic = RSS(isotonic) − RSS(unrestricted cell means); reference distribution by parametric bootstrap from the isotonic fit with σ̂_S. Also report Spearman's ρ between cell means and log T per S.
2. **Saturation test (pre-specified).** H0: μ(72B) − μ(14B) ≥ δ vs H1: < δ (one-sided non-inferiority, α = 0.05 — equivalently the 90% two-sided CI upper bound < δ). Primary δ = 1.0 pt, sensitivity δ = 0.5 pt (§3.3 item 6). Report the CI in all cases, and the model-based analogue from §5.2.
3. **Rank-based duplicate of every pooled test** (aligned-rank transform ANOVA on P_cc after removing α_S) as a check against non-Gaussian residuals.

### 4.4 The D term

The D sweep exists in a subset of cells (P5). To identify δ₂ (d·x) and δ₃ (d·s) it must span both factors: recommended subset = S ∈ {100M, 1B} × T ∈ {3B, 14B, 72B} × D ∈ {D_syn/4, D_syn/2, 2D_syn} (D_syn itself is in the main grid), 2 seeds each = 36 runs. Analysis: M4 (linear in d with interactions; a d² term if the D range is ≥ 8×), plus the data-scaling form P = P_∞(S,T) − B(S,T)·D^{−γ} as a secondary nonlinear fit. The **relative dose** D/(base tokens) differs by S by ×40 under a fixed D_syn (§Open issues, item 5); the D sweep is what allows the S × T interaction to be separated from a dose effect — report M3 with and without the D-sweep cells.

### 4.5 Capability covariate instead of log T

Replace x by z_T := the teacher's own P_cc on the same benchmark suite (evaluated once per teacher, same precision as generation), or by the teacher's MMLU. Refit M1–M3 with z_T (models M1z–M3z). Compare AICc with the log T versions. Expectation (hypothesis, to be stated as such): within one family with 6 sizes, z_T and log T are almost collinear (rank correlation ≈ 1), so the comparison has little discriminating power; its value is (i) a T* expressed in capability units, which is portable across families, and (ii) the basis for the recommended second-family replication (two or three sizes of Llama-3 / Gemma-3) that would break the collinearity.

---

## 5. Estimating T*(S) and its uncertainty

### 5.1 Definition and the choice of functional form per S

Two candidate forms per S (fitted to the r_T seed values at 6 teacher sizes):

- **F1 (quadratic in log T):** P = a + b·x + c·x². Interior maximum at x* = −b/(2c) iff c < 0 and x_min < x* < x_max. Three parameters, 6 support points, 6r − 3 residual df.
- **F2 (saturating):** P = P_∞ − A·T^{−β} (equivalently P_∞ − A·2^{−βx}), β > 0. Monotone; no interior optimum (T* = ∞, i.e. "the largest teacher"). Three parameters.

Both have three parameters, so the comparison is a plain likelihood/AICc comparison, not a nested test. Because F1 can also fit a monotone curve (with c ≈ 0 or vertex beyond x_max), the decision uses the *location and uncertainty of the vertex*, not the AICc alone:

**Per-S classification rule (pre-registered):**
1. Fit F1 and F2. Compute ĉ and its one-sided p-value (within-S variance), x̂*, and the bootstrap CI of x̂* (§5.2).
2. **Interior optimum** iff (i) ĉ < 0 with p < 0.05, (ii) x_min < x̂* < x_max, (iii) the 95% CI of x̂* has upper bound < x_max (excludes the largest teacher), and (iv) AICc(F1) ≤ AICc(F2) + 2. Then T*(S) = 2^{x̂*} with its CI.
3. **Saturating** iff not interior, and the model-based top-step gain (§5.2) has a one-sided 95% upper bound < δ = 1.0. Then report T_sat(S) = the smallest grid T at which the upper bound of the gain from doubling T falls below δ.
4. **Monotone, unsaturated** iff neither: the largest teacher is best and larger teachers might still help; T* is right-censored at 72B.
5. **Flat** iff the T effect test (a) is not significant for this S: no T*; report the CI of the total range.

The four labels are mutually exclusive and exhaustive by construction (evaluated in the order 5 → 2 → 3 → 4).

### 5.2 Bootstrap and the two decision quantities

- **Cluster (seed) bootstrap** — primary when r ≥ 3 in every cell of that S: resample the r_T runs within each cell with replacement (6 cells independently), refit F1 (and F2), record x̂*, the fitted top-step gain, and the classification; B = 2000. Percentile CIs on the log₂ scale. With r = 3 the per-cell bootstrap has only 10 distinct multisets, 10⁶ across six cells — coarse but usable; with r ≤ 2 it is not, and the **parametric bootstrap** (draw (b, c) from N((b̂, ĉ), Σ̂) with within-S σ̂²_S) is used instead; the simulation (Sim Table 3, "T* CI cov") shows 93–95% coverage for the parametric version.
- **Gain from doubling T at x:** Δ₂(x) = P(x+1) − P(x) = b + c(2x+1) under F1; top-step gain over the actual grid step 32B → 72B: Δ_top = b(x₆−x₅) + c(x₆²−x₅²). Its SE from the coefficient covariance (delta method / bootstrap).
- **Optional item bootstrap:** the whole procedure repeated over item-resampled accuracies (§1.5) to show that T* is not driven by a few items; reported as a second CI, not merged.

### 5.3 Decision rules, stated once

- *"Saturation at T_sat"*: the smallest T_j such that the one-sided 95% upper bound of Δ₂(x_j) (model-based) is < δ = 1.0 pt (sensitivity: 0.5). δ is fixed now; it corresponds to ≈ 2 σ_agg under the central noise assumption and to about a quarter of the expected total range.
- *"Interior optimum"*: §5.1 rule 2, all four conditions. Additionally, any sentence of the form "a T₀ teacher beats the 72B teacher for student S" must be supported by the **tier-2 confirmation seeds**: the contrast (cell T₀ − cell 72B) is estimated on the tier-2 seeds only (they were not used to select T₀), with a 95% CI excluding 0. Tier-1 seeds select; tier-2 seeds confirm. This removes the winner's-curse bias of selecting the argmax cell post hoc (established: the maximum of six noisy means is biased upward by ≈ 1.27 σ/√r for six equal means).
- Any *"optimum"* wording without both of these is disallowed in the write-up (§8).

### 5.4 The relation between T* and S

If ≥ 3 student sizes are classified "interior optimum", fit by weighted least squares
$$\log_2 \hat T^*_S = \alpha\, s + c,\qquad w_S = 1/\widehat{\mathrm{Var}}(\hat x^*_S),$$
and report α̂ with a t-based 95% CI (n − 2 = 3 df at most) plus a bootstrap CI that propagates the per-S bootstrap distributions. Sizes classified "saturating" or "unsaturated" contribute an interval-censored observation (x* ≥ x_sat or x* ≥ x_max); with ≤ 5 points a censored regression is not identifiable in any useful sense, so those sizes are shown on the plot as censored arrows and excluded from α̂, and this is stated. **What 5 points can support (established, and Sim Table 5):** even with a clean optimum at every S and r = 3, the CI half-width of α̂ is ≈ ±0.35 (σ = 0.5) to ±0.5 (σ = 0.8). The result is reported as "log T* increases with log S with slope α̂ [CI]" — a descriptive summary of five estimates, not a law, and H4 is assessed as "consistent / inconsistent with a monotone relation", not as "T* ∝ S^α".

### 5.5 Economic optimum (H5)

Let C(T) be the measured generation cost per D_syn tokens (GPU-s, energy, cloud-equivalent $). Define T_econ(S; λ) = argmax_T [P̂(S,T) − λ·C(T)] for a pre-specified grid of λ (points per $), and the *iso-cost* comparison of Bansal et al. 2024 (https://arxiv.org/abs/2408.16737): at fixed generation cost, a smaller teacher yields more tokens — this needs the D term (M4) and is reported as model-based extrapolation with its CI. The H5 contrast per S is P̂(S, T*_raw) − P̂(S, T_econ), estimated from the fitted curve with the bootstrap CI (§5.2); H5 is "supported at S" if T_econ ≠ T*_raw and the performance loss is < δ with a CI excluding δ (non-inferiority), and "not supported" otherwise. Five contrasts, Holm-corrected (§6).

---

## 6. Multiple comparisons and effect sizes

**Hierarchy (methodological choice).** Primary inference is at the model level: H1–H4 map to *one test each* (M1 vs M0; β₂ < 0; the per-S classification; M3 vs M2), which sidesteps per-cell corrections. Per-cell contrasts are secondary and corrected within pre-defined families:

| Family | Contrasts | Count | Correction |
|---|---|---|---|
| F-A: adjacent teachers within S | (T_{j+1} − T_j), j = 1..5, per S | 5 × 5 = 25 | Holm within each S (family size 5); global Holm over 25 as sensitivity |
| F-B: teacher vs matched-real control | (cell − C1), 6 per S | 30 | Holm within each S (6) |
| F-C: synthetic vs base | (cell − C0), 6 per S | 30 | Holm within S; descriptive (C0 is unreplicated at ≥ 250M, so the CI uses σ_seed only) |
| F-D: C1 vs C0 (do more real tokens help?) | 1 per S | 5 | Holm over 5 |
| F-E: economically vs raw-optimal | 1 per S | 5 | Holm over 5 |
| F-F: "intermediate beats largest" confirmation | ≤ 1 per S, tier-2 seeds only | ≤ 5 | Holm over the number actually tested |

Holm (step-down) controls the family-wise error rate at 0.05 without independence assumptions; it is preferred to Bonferroni (uniformly more powerful) and to FDR (the contrasts are confirmatory, few, and pre-specified). All 100 contrasts are reported in the appendix table with raw and adjusted p-values and CIs; the text only calls a contrast "significant" if adjusted p < 0.05.

**Effect sizes.** For each contrast: the difference in chance-corrected points with its 95% CI (t, using the pooled df of the relevant model, or within-S df for per-S contrasts); Hedges' g = J·Δ̂/σ̂_pooled with J = 1 − 3/(4·df − 1) and its CI (noncentral-t or bootstrap). With r = 3 the CI of g spans several units and g is reported only alongside the absolute effect. For model terms: β₂ in points per octave², γ₁ in points per octave per doubling of S, etc., all with CIs.

---

## 7. Pilot: design and pre-specified go / redesign rules

**Pilot design:** S ∈ {25M, 100M} × T ∈ {1.5B, 7B, 32B} (32B rather than 14B to span the widest range and to test the 96 GB single-GPU BF16 path) × 3 seeds; C1 × 3 seeds at both S; C0 with 3 base seeds at 25M and 1 at 100M; a 4-level D sweep (D_syn/4, D_syn/2, D_syn, 2D_syn) at (100M, 7B) with 2 seeds; ≈ 34 runs. All pilot runs are evaluated on the full 9-benchmark suite plus the continuous outcomes.

**Pre-specified quantities to estimate:** σ̂_{seed,k} per benchmark, σ̂_agg (df = 16), σ̂_base(25M) (df = 2 — only an order of magnitude), the floor status of each benchmark at 25M and 100M, the synthetic-vs-C1 and synthetic-vs-C0 gains, the teacher spread (max − min cell mean) at each S, and the D dose-response.

**Proceed to the full study only if all of P1–P4 hold:**

| Rule | Criterion | Rationale |
|---|---|---|
| P1 Detectable synthetic effect | At ≥ 1 student size, the average synthetic-vs-C1 difference (pooled over the 3 teachers, 9 vs 3 runs, within-S variance) has a 95% CI excluding 0 in *either* direction, on P_cc **or** on task loss | If synthetic data does nothing detectable relative to matched real tokens at this D_syn, the T question is moot; a robustly *negative* effect also passes P1 but triggers a generation-pipeline review before proceeding |
| P2 Noise | σ̂_agg^{upper} (95%) ≤ 0.9 pt on the floor-gated P_cc, or ≤ the equivalent on task loss | Above this the full study needs r ≥ 10 per cell for 1-pt effects (Sim Table 1) |
| P3 Floor | ≥ 4 benchmarks retained at 25M by the §1.3 rule, ≥ 6 at 100M | Otherwise P_cc at 25M is 1–3 noisy numbers |
| P4 Spread | Teacher spread at 100M ≥ 1.0 pt (point estimate) on P_cc or the corresponding task-loss spread | If the three most different teachers cannot be told apart by 1 pt, the six-teacher grid at r = 3 cannot map a curve |
| P5 Dose | P_cc at (100M, 7B) is non-decreasing in D across the 4 levels (Spearman > 0) and the 2D_syn − D_syn/4 difference has CI excluding 0 | Confirms the synthetic phase is not saturated at D_syn and that D_syn sits on the responsive part of the dose curve |
| P6 Integrity | Re-evaluation of a checkpoint reproduces every number exactly; decontamination report clean; NLL on human text not worse than C1 by more than a pre-set margin (e.g. 5% relative) | Pipeline validity |

**Redesign triggers (each mapped to an action):**

| Failure | Action |
|---|---|
| P3 fails at 25M only | Drop 25M from the primary grid; keep it as an exploratory row with the same seeds; add 2B? No — stay within hardware; the primary grid becomes 4 × 6 |
| P3 fails at 100M too | Change the suite: replace OpenBookQA/Winogrande/BoolQ by larger continuous-scored sets (task loss on the same items; add held-out QA NLL); increase base tokens per parameter (deviation from P5, must be applied to all S) |
| P2 fails | Variance reduction first (floor gating; task loss as primary; average the last 3 synthetic-phase checkpoints; drop N < 1000 sets), then re-estimate on the existing pilot runs; if still failing, increase D_syn (larger effects) rather than r |
| P1 or P4 fails | Increase D_syn (×2–4) and/or revise generation (prompt pool, filtering) and re-run the pilot cell (100M, 7B) vs C1 only; if the effect remains undetectable at 4D_syn, the study's premise (synthetic continued training moves small students) is not supported at this scale — report that |
| P5 fails (flat in D) | D_syn is already past saturation for that cell: lower D_syn and re-check, since the teacher effect is expected to be largest before saturation |
| Seed outliers (loss spikes) | Report; do not drop; add a rank-based analysis; consider gradient clipping/LR changes only if applied to *all* cells (restart pilot) |

The pilot data are **not** pooled into the full-study inference (different D_syn or suite may result); they are reported separately.

---

## 8. Reporting standards (binding for the write-up)

1. **Full cell table** in the appendix: for every cell and control, r, mean ± SD of P_cc, every per-benchmark a_k (mean ± SD), task loss, held-out NLL, and the run-level CSV released.
2. **Figures:** per-S curves of P_cc vs log T with **95% CIs across seeds** (t-based, r printed on the panel), all individual seed points overlaid, C0 and C1 as horizontal bands with their CIs; the fitted F1/F2 curves with bootstrap bands; per-benchmark small multiples. Error bars are never SEs without saying so, and never item-only.
3. **Claims vocabulary:** "optimum"/"T*" only under §5.3; "saturation" only with δ stated; "law" not used; α reported as descriptive with CI; "monotone" only if the isotonic test passes and the curvature test is reported; "no effect" phrased as "not detected; the 95% CI for the total range is […]".
4. **Every test reports** the estimate, CI, df, and the raw and Holm-adjusted p-value; per-S tests use within-S variance.
5. **Seed accounting:** which cells have 3 vs 5 seeds, which seeds were used for selection vs confirmation, and the base-replicate σ̂_base.
6. **Both uncertainty sources**: seed CI (primary) and paired-item bootstrap CI (robustness) for the headline contrasts.
7. **Pre-registration:** this document is frozen (hash recorded) before the first full-study run; every deviation is logged with the reason and the analysis is presented both as pre-registered and as deviated.
8. **Negative results** (flat curves, no interior optimum, no S × T interaction) are reported with the same prominence as positive ones; the study is designed so that "saturating, no interior optimum, no detectable interaction" is a publishable outcome.

---

## Open issues and disagreements with the brief

1. **P1 (one base run per S) makes cross-S inference conditional on five unreplicated runs.** Within-S contrasts are clean; the S main effect, S × T interaction and T*(S) are not. I recommend 3 base seeds at 25M and 100M (≈ 4 cheap runs) to at least bound σ_base, and an explicit "conditional on the base runs" caveat for H4.
2. **P7 (3 seeds everywhere, 5 in selected cells) is adequate for the model-level hypotheses but not for per-cell claims.** With the central noise assumption (σ_agg = 0.5) r = 3 has 67% power for a 1-pt two-cell difference and 23% for 0.5 pt (Sim Table 1b); adjacent-teacher differences on a saturating curve are 0.2–0.7 pt. The paper's claims must be curve-level. The tiered allocation should target the *ends* (1.5B, 32B, 72B) and C1, not only "cells adjacent to the optimum" — the end cells drive curvature and saturation, and the argmax-adjacent cells are needed mainly as a confirmation sample against the winner's curse.
3. **The dominant lever is variance, not seeds.** Halving σ_agg is worth more than going from r = 3 to r = 5 in every row of Sim Table 3. The brief's benchmark suite (P6) contains three sets that are near floor at 25M–100M (Winogrande, BoolQ, ARC-Challenge) and one with N = 500 (OpenBookQA, 2.7 chance-corrected points of Bernoulli noise per run). I recommend making the continuous task loss a co-primary outcome and floor-gating the discrete aggregate.
4. **Teacher grid mismatch:** the brief's 1/3/7/14/32/70B is not Qwen2.5 (1.5/3/7/14/32/72B). The 1.5B lower end is 1.5× the 1B student; the design has no teacher smaller than the largest student. If H3 is important, a 0.5B teacher (Qwen2.5-0.5B-Instruct) should be added as a 7th level for the 1B student at least; statistically it costs 5 cells × 3 seeds.
5. **P5 (fixed D_syn across S) confounds the S × T interaction with relative dose.** D_syn is ≈ 40× larger relative to base tokens for 25M than for 1B (both Chinchilla-trained). A 25M student at 2× its base tokens and a 1B student at 5% of its base tokens are in different regimes; an apparent "T* grows with S" could be a dose artefact. The D sweep must span both S extremes (§4.4), and the interaction must be reported with and without the D term.
6. **δ (minimal practically important difference) must be fixed before the full study.** I set δ = 1.0 pt (sensitivity 0.5) because the equivalence-style saturation claim at δ = 0.5 is not achievable with two-cell tests at any affordable r and only just achievable model-based at r = 5. The lead should confirm δ; if the research question demands resolving 0.5-pt saturation, the design needs σ_agg ≈ 0.3 (i.e. continuous metrics), not more seeds.
7. **H4 as "T* ∝ S^α" is not testable with 5 student sizes at this noise level** (α̂ CI half-width ≈ ±0.35 in the best realistic case; undefined when any S is classified saturating). The plan reports α descriptively; the paper's title and abstract should not promise a scaling law (agrees with P8).
8. **Seed definition:** the brief does not say whether the synthetic subsample is re-drawn per seed. It should be (§0); otherwise the study is about six specific datasets rather than six teachers. Sharing the same prompt subset across teachers within a seed index is a free variance reduction and should be adopted.
9. **Literature numbers for seed SD could not be verified from this environment** (arXiv, OpenReview, HF, allenai.org were blocked); the noise model rests on the stated assumptions and the Bernoulli floor, and must be replaced by pilot estimates. The pilot's df = 16 gives σ̂ only to within ×0.74–1.52.
10. **Second family.** Within one family, log T and teacher capability are collinear; §4.5 cannot separate "size" from "capability". A 2–3-point replication with a second family (Llama-3.x or Gemma-3) at one student size is the cheapest way to make any T* claim about *size* rather than about Qwen2.5 specifically; it is out of the brief's scope but should be listed as the first follow-up.
