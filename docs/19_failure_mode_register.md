# 19. Failure-Mode Register

Each mode: mechanism (which hypothesis it biases and how), likelihood, the signal that reveals
it, the mitigation in the design, and residual risk. Ordered roughly by severity for the
single-5080 study. (The full red-team pass was cut short by an infrastructure limit; this
register is the research lead's consolidation of the workstreams' "Open issues" sections.)

| id | Failure mode | Mechanism / which H | Lik. | Detection signal | Mitigation (where) | Residual |
|---|---|---|---|---|---|---|
| FM-01 | Teacher-family confound | small Qwen2.5 sizes saw 72B-synthesised data in their own training → "size" partly encodes distillation depth; attenuates H1–H3 | high | flatter curve than a cross-family check | disclose (doc 00 §0.6); cross-family anchor ablation (doc 15 #8); capability covariate (doc 12 §4.5) | not removable within one family |
| FM-02 | Capability ≈ size collinearity | within one family log T and teacher accuracy are collinear; "saturates in size" vs "in accuracy" not separable (H2/H4) | high | 32B≈72B vendor scores; here 7B≈14B on some tasks | fit against measured capability z_T as well as log T (doc 12 §4.5, S7) | interpretation caveat only |
| FM-03 | Benchmark floor at 25M–100M | most MC tasks at chance → P_cc is noise; false "no effect" (all H) | high | floor gate; C0 within 3·SE of chance | 5B base tokens (doc 00 §0.4); floor gate + Tier A/B; NLL co-primary (doc 11) | 25M may drop to NLL-only |
| FM-04 | Teacher-size-correlated benchmark leakage | bigger teachers regurgitate more test items → inflates their apparent gain (H1) | med | per-teacher contamination-rate table rising with T | union-drop decontam (doc 09 §1.5); clean-b re-analysis (doc 11 §7, S9) | paraphrase leakage may survive |
| FM-05 | int4 quantisation size-dependence (local only) | int4 hurts small teachers more → distorts the low end of the T-axis | med | 3B BF16-vs-int4 bridge cell diverges | uniform int4 for all; bridge cell measures it (doc RUN_ON_5080) | small-teacher rung noisier |
| FM-06 | Verbosity confound under equal-token | verbose teachers → fewer prompts/examples at fixed tokens (H1/H3) | med | length distribution + N_T per teacher | length-matched + equal-examples re-analyses (doc 15 #1,2, S6) | conditional on length support |
| FM-07 | Underpowered seeds | 3 seeds can't resolve <1-pt teacher gaps → false null (H1/H3) | med | pilot σ̂; power table (doc 12 §3) | tier-2 seeds at ends + argmax; curve-level tests; NLL primary at small S | small effects still hard |
| FM-08 | Base dominates, phase-2 no leverage | θ* so strong all teachers look alike (all H) | med | pilot rule P1/P4 | branch at peak LR (annealing phase); matched-real control; escalation rule (doc 10 §1.7) | may need larger D_syn |
| FM-09 | Relative-dose confounds S×T | fixed D_syn is 16% of a 25M vs 4% of a 250M → "T* grows with S" could be dose (H4) | med | D-sweep at both S extremes | base-token rule shrinks spread to ~4×; D-sweep separates dose (doc 00 §0.9, doc 12 §4.4) | H4 reported with/without D term |
| FM-10 | Prompt/template artifacts | Qwen default system prompt, generation_config defaults, chat-template drift across sizes | med | rendered-prompt hash per size differs | explicit system prompt; override decoding; template hash asserted (doc 08 §9, doc 09 §2.1) | low if enforced |
| FM-11 | Winner's curse on argmax | picking the best of 5–7 noisy cells biases the optimum upward (H3) | med | tier-1 argmax not confirmed by tier-2 | tier-2 confirmation seeds, selection vs confirmation split (doc 12 §5.3) | needs the extra seeds |
| FM-12 | Filtering introduces teacher bias | yield/refusal/format-fail differ by teacher → survivorship (all H) | low-med | per-teacher filter-rate table | teacher-agnostic filters, rates reported as covariates; unfiltered ablation (doc 09 §2.2, doc 15 #6) | reported, not eliminated |
| FM-13 | vLLM sm_120 / int4 kernel bugs | wrong or crashing generation on Blackwell | low-med | smoke parity check fails | pin vLLM; smoke test; GPTQ-Int4 fallback (doc RUN_ON_5080) | driver-dependent |
| FM-14 | Forgetting on human/real text | synthetic annealing shifts the student off human text (misreads NLL) | low-med | held-out real/instr NLL vs C1 | replay ρ=0.25; C1 comparator; report NLL both ways (doc 09 §1.5, S8) | direction reported, not assumed |
| FM-15 | Non-determinism (flash-attn backward) | runs not bit-reproducible → seed variance mis-attributed | low | re-run drift > seed SD | documented; seed=data-order variance is a lower bound; manifests hash inputs (doc 18) | accepted, disclosed |
| FM-16 | Scope overclaim | one-family, int4, ≤250M student result stated as general (all H) | med | reviewer objection | Limitations enumerated (doc 03, doc 00 §0.15); title scoped (doc 00 §0.1) | writing discipline |

## Claims the paper must not make (beyond doc 03 §3.4)

Add for the single-5080 variant: no claim about BF16-served teachers (local uses int4); no claim
about teachers >14B or students >250M; no claim that the int4 low-end rung is precision-clean
without citing the bridge cell (FM-05).
