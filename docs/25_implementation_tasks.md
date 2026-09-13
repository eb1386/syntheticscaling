# 25. Next Implementation Tasks (dependency order)

Status after this session: the pipeline runs end to end on CPU (`make micro`), 58 unit tests
pass, and the single-5080 study is launchable with `./install.sh` + `./run_all.sh`. The tasks
below are what remains to turn a launch into a NeurIPS-grade result; most require the GPU that
is not present in the authoring environment.

**Done (verified here):** config schemas + IDs + manifests; LLaMA-like model with exact param
counts; WSD trainer + branch builder; generation runtime (backend/filters/subsample/runner);
prompt pool + taxonomy + decontam; self-contained MC + NLL evaluator; index + analysis + T*/STE;
profiles + orchestrator + install/run commands; compute budget. CPU end-to-end micro run.

Remaining, in order:

1. **GPU smoke** (`make smoke`): confirm vLLM sm_120 int4 kernels, real tokenizer, benchmark loading, VRAM headroom on the actual 5080. Fix any kernel/repo-id issues (GPTQ-Int4 fallback).
2. **Pilot calibration** (doc 05): measure int4 decode tok/s, training MFU, r, yield, seed SD; update `compute_budget` constants and the seed tier; run the floor gate at 25M.
3. **Pin provenance**: replace `revision: PIN_ME_TO_COMMIT_SHA` in `configs/teachers/*.yaml` with HF commit SHAs; fill `tokenizer_sha256`, `prompt_pool_sha256`, `chat_template_sha256` (doc 18).
4. **Real base corpus at scale**: verify `scripts/prepare_data.py` streams FineWeb-Edu to 25B tokens and writes the memmaps within the disk budget; add sharding if needed.
5. **Decontamination at scale**: build the benchmark n-gram index over the full eval set and run it on the pool and every teacher's outputs; produce the per-teacher contamination table (doc 11 §7).
6. **Base pretraining**: run θ* for 25M (×3 seeds), 100M, 250M; verify loss curves against Pythia/SmolLM references; confirm the floor gate passes (else drop 25M to NLL-only).
7. **Generation**: run all 5 teachers; verify yield ≥0.85, length distributions, refusal/CJK <1%; freeze pools.
8. **Branches**: run the 89 branch trainings with resume; watch for divergence; keep final checkpoints only.
9. **Evaluation**: score every checkpoint (primary + NLL); add the secondary suite; cross-check a few cells with `lm_eval_runner` on an exported checkpoint.
10. **Contamination-sensitivity**: re-evaluate on the clean-b item subset; report both.
11. **Analysis**: floor-gated P_cc, cell table, control bands; fit M0–M4 + M2sat; per-student T* classification with bootstrap CIs; STE and T$(λ); mediation (learnability, correctness, diversity).
12. **Figures**: implement `synscale/analysis/plots.py` functions (doc 20) — transfer curves, S×T heatmap, T*(S), cost-per-token, STE/T$ frontiers, dose–response, mediator scatter.
13. **Ablations (Tier 0/1)**: length-matched and equal-examples re-analyses; replay-ρ and phase-2-LR checks; the 3B BF16-vs-int4 bridge (doc 15).
14. **Seeds tier-2**: after tier-1, add +2 seeds at the ends and around each student's argmax; run the confirmation contrasts (doc 12 §5.3).
15. **Cross-family anchor** (if a bigger card is rented): Llama-3.1-8B-Instruct at 100M/500M to bound the family>size effect.
16. **Pre-registration freeze**: hash docs/00, docs/01, docs/12 before the first full run; log deviations.
17. **Write-up**: fill the placeholders in `paper/` from the results; Limitations from docs/03 + FM-16.
18. **Reproducibility pack**: release configs, manifests, index, small eval JSON, pool hash, and the analysis notebook; checkpoints on request.
19. **Verify [UNVERIFIED] literature/dataset details** (arXiv/HF were blocked during authoring): teacher licenses, dataset licenses, the 2025–26 related-work citations in docs/04.
20. **Optional full grid**: rent a ≥48 GB card and run the 32B/72B teacher and 500M/1B student rows with the same configs to recover the cluster design.
