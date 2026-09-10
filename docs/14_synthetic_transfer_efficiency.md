# 14. Synthetic Transfer Efficiency (STE): Definition

Derivation and candidate analysis: doc 13 Task E (compute workstream). This document fixes the definition used in the paper. Tags: **[choice]** unless stated.

## 14.1 Cost of a teacher's tokens

For teacher T, a generation run delivers D_pool(T) student-tokenizer tokens after filtering. Measured per 10k-prompt chunk and summed: GPU-seconds τ_gen(T) (all cards occupied, from first request to last response; model load excluded and reported separately), energy E_gen(T) (NVML power at 1 Hz, integrated, × PUE 1.2), and processed tokens N_proc(T) (prompt + response, teacher tokenizer).

Unit costs per delivered student token:

- c_τ(T) = τ_gen(T) / D_pool(T)  [GPU-s/token]  (hardware-specific: RTX PRO 6000 Blackwell, BF16, vLLM version, TP)
- c_$(T) = c_τ(T) · p  [$/token], p = $2.00 per GPU-hour cloud-equivalent (sensitivity: $0.95 owned-and-amortised; $3.00)
- c_E(T) = E_gen(T) / D_pool(T)  [kWh/token]
- c_F(T) = 2 · N_params(T) · N_proc(T) / D_pool(T)  [FLOP/token]  (hardware-independent; attention FLOPs < 1% ignored)

Cell cost: C_x(S,T) = c_x(T) · D_syn, linear in tokens after warm-up. At fixed D_syn the cost does not depend on S.

## 14.2 Gains

M(S, cond) is the primary outcome (P_cc in chance-corrected points, or a continuous outcome) averaged over seeds.

- G₀(S,T) = M(S,T) − M(S,C0): gain over the replay-only cooldown ("any new data").
- **G₁(S,T) = M(S,T) − M(S,C1)**: gain over the matched-real control. This is the economically relevant counterfactual: fresh web tokens are essentially free, so generation dollars must be justified against them.

## 14.3 Candidates considered

| # | Definition | Interpretable as | Defect |
|---|---|---|---|
| 1 | G₁/C_$ | return per dollar | ratio of a concave gain over a linear cost is maximised at the cheapest teacher, so argmax is a poor selector |
| 2 | G₁ per 1M synthetic tokens | data efficiency | contains no teacher cost; cannot address H5 |
| 3 | (G₁/C)/(G₁(T*)/C(T*)) | efficiency relative to the best teacher | undefined when G₁(T*) ≤ 0; hides absolute scale |
| 4 | area under G₁ vs log C | returns to spending | does not select a teacher; endpoint-sensitive |
| 5 | m_i = ΔG₁/ΔC between adjacent teachers | return on upgrading | unstable where ΔC is tiny (1.5B→3B); needs ordering by cost |

## 14.4 Definition adopted

**Primary reporting metric:**

  STE(S,T) = G₁(S,T) / C_$(S,T) = [M(S,T) − M(S,C1)] / (c_$(T) · D_syn)   [chance-corrected points per $]

reported with STE_τ (points per kGPU-s) and STE_F (points per EFLOP) as hardware-free companions. STE is the communicable summary; it is **not** used to select a teacher.

**Selection tools:**

- Marginal efficiency between cost-adjacent teachers: m_i(S) = [G₁(S,T_{i+1}) − G₁(S,T_i)] / [C_$(T_{i+1}) − C_$(T_i)].
- Price-of-a-point curve: T$(S; λ) = argmax_T [G₁(S,T) − λ·C_$(S,T)] for λ on a log grid over [0, 10] points per dollar; λ = 0 recovers T*(S). Reported as a step function per S with bootstrap selection probabilities; T$ is "stable" at a λ if the selection probability is ≥ 0.8.
- λ₁(S): the marginal efficiency of the last upgrade into T*(S), i.e. the smallest price-of-a-point at which the performance-optimal teacher stops being optimal.
- ε-rule (constant-free): T$_ε(S) = the cheapest T with G₁(S,T) ≥ (1 − ε)·G₁(S,T*), ε = 0.10.

**H5 decision rule:** H5 is supported at S if the 95% bootstrap CI of λ₁(S) lies entirely below λ_ref(S) = G₁(S,T₁)/C_$(S,T₁), the average efficiency of the cheapest teacher. Five per-S tests, Holm-corrected.

## 14.5 Uncertainty

Bootstrap with B = 5,000: gains by pooled-variance parametric resampling of seed means (σ_S pooled within S; Levene check) nested with per-item resampling; costs by resampling the 10k-prompt chunks; p treated by sensitivity, not resampled. Percentile CIs for STE, m_i, λ₁; selection probabilities for T$(λ). Implementation: `synscale/analysis/ste.py`.

## 14.6 Iso-cost companion analysis

Using the D-sweep (doc 00 §0.9), P(S,T,D) is interpolated per (S,T) and evaluated at D(T) = B / c_$(T) for budgets B on a grid, producing the Bansal-style "at equal spend, which teacher?" curve. Reported side by side with the equal-token result; extrapolation beyond the D range is not performed.
