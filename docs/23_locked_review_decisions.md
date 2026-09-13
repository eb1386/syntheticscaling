# 23. Locked Decisions from Design Review

Five methodological points were raised in review and are now locked in code and docs. No further
redesign of the core experiment unless the real 5080 pilot reveals a problem.

| # | Decision | Where it lives |
|---|---|---|
| 1 | **Checkpoints on one training curve are not independent.** Fit the scaling curve per seed; take uncertainty from the spread across seeds (single-seed cells flagged `seed_ci=false`). Never treat ~20 checkpoints as 20 observations. | `synscale/analysis/scaling.py` (`fit_cells`); `docs/22 §22.3a`; `tests/test_scaling.py` |
| 2 | **`E` is an estimated asymptote, not the irreducible loss.** Fit three forms (`E+A·D^-α`, `a-b·logD`, pure `A·D^-α`) and choose by held-out-checkpoint RMSE; report `L_inf_hat` as an estimate over the observed range. | `scaling.py` (`fit_forms`, `best_form`); `docs/22 §22.3a` |
| 3 | **Teacher-independent evaluation on ≥3 fixed distributions:** held-out real web NLL (`base_heldout`), held-out human instruction NLL (`instr`), downstream accuracy (secondary). Never evaluate on a teacher's own synthetic distribution. | pipeline eval sets + `docs/11`, `docs/22 §22.3a` |
| 4 | **One shared 16k custom BPE for every student**, trained once on the real base corpus. A 16k vocab keeps the 25M model from being ~50% embeddings (32k), so the student-size axis is not distorted. Teachers generate with their own tokenizer; output is retokenised with the student BPE. | `scripts/prepare_data.py` (`train_tokenizer`); `configs/students/*.yaml` (vocab 16384); `synscale/pipeline.py` (`get_tokenizer`) |
| 5 | **`scaling5080_fast` is the pilot, not the paper.** Use it to measure throughput, variance, OOM, filter yield, loss-curve smoothness and effect sizes; only then run the full 4-student `scaling5080`. | `docs/05`, `README`, `docs/RUN_ON_5080.md` |

Additional locked choices from the same review:

- **Preferred efficiency metric = synthetic-data multiplier** `D_real(L)/D_syn(L)` at matched loss ("one 7B-teacher token ≈ 1.6 real tokens for a 100M student here"), combined with generation cost. Replaces the bespoke "Synthetic Transfer Efficiency" as the headline (`scaling.py` `synthetic_data_multiplier`; `docs/22 §22.3b`). STE is retained only as a secondary appendix metric.
- **No "NeurIPS-scale" claim.** Built to a NeurIPS-quality methodological standard; whether a result belongs at a main conference depends on the finding and its novelty. A clean negative (teachers ≥1.5B give indistinguishable scaling curves) is a valid, reportable outcome.

Frozen research question: *We train 25M–250M students from matched base checkpoints on synthetic
data from 0.5B–7B teachers, measure held-out loss throughout training to estimate L(S,T,D), and
test whether teacher scale changes the marginal value of additional synthetic tokens, whether
that depends on student capacity, and how synthetic data compares with equal quantities of real
data.*
