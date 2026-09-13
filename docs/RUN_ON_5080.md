# Running the study on a single RTX 5080

This is the operational guide for running the whole experiment on **one 16 GB RTX 5080**.
It explains exactly what runs locally, the two commands, the honest time cost, the knobs
to make it shorter, and what genuinely cannot be done on one consumer card.

## TL;DR — two commands

```bash
./install.sh scaling5080   # deps + tokenizer + corpora + teacher models + prompt pool  (one time)
./run_all.sh scaling5080   # runs the entire study, resumable; logs to results/scaling5080/run.log
```

`scaling5080` is the flagship: a data-scaling-law study (4 students x 4 FP8 teachers 0.5-7B),
primary outcome = loss vs synthetic-token count, D-axis read from the training curve
(`docs/22`). `local5080` (teacher-size effects on accuracy, teachers to 14B int4) is retained
as the alternative framing. Use `scaling5080_fast` for a complete first result in ~1 week.

Before committing weeks, validate the whole pipeline in minutes and then in ~1 hour:

```bash
make micro                 # CPU, MockBackend, toy model — proves every stage works end to end
make smoke                 # GPU, real Qwen models, tiny token budget (~1 h) — proves the real path
make budget PROFILE=local5080   # prints the GPU-hour / wall-clock estimate below
```

Run inside `tmux`/`screen`; `run_all.sh` is **resumable** (re-run to continue after any interruption — finished jobs are skipped via markers under `results/<profile>/state/`).

## What actually runs on a 5080, and why this is scoped

The locked methodology (`docs/00_locked_methodology.md`) targets a 2×RTX PRO 6000 + 2×RTX 5080 cluster and a 5×7 grid with teachers up to 72B. One 16 GB card cannot serve a 72B/32B teacher even in int4, cannot train a 1B student in a sane time, and runs everything serially. The `local5080` profile is the **largest complete study that fits one card**:

| Axis | Cluster design (docs/00) | `local5080` (this card) | Why |
|---|---|---|---|
| Students | 25M–1B (5) | **25M, 100M, 250M (3)** | a 500M/1B base run is ~10–17 days each on one 5080 |
| Teachers | 1.5–72B BF16 (6) | **0.5, 1.5, 3, 7, 14B, AWQ int4 (5)** | 14B int4 ≈ 8 GB fits; 32B/72B do not fit 16 GB |
| Precision | BF16 | **int4 (AWQ Marlin), uniform** | uniform quantisation keeps teacher precision a constant, not a size-correlated confound; a 3B BF16-vs-int4 bridge cell measures the quantisation effect |
| D_syn | 600M | **300M** | halves generation and phase-2 time |
| Controls | C0/C1/C1b/C1c | **C0 (=θ*), C1, C1b** | the key contrasts (synthetic vs matched-real, vs base, vs human Q&A) |
| Seeds | 3 + tiered 5 | **3 + tiered 5 at ends, C1×5, 25M base×3** | powered for the curve-level tests (doc 12) |

This is a genuine 3×5 teacher×student grid with matched-real and human-Q&A controls, a nested D-sweep, mediation covariates, per-teacher cost, and 3–5 seeds — a complete, pre-registered study, not a demo. The teacher span 0.5→14B (28×) brackets the saturation region the literature places at ~1–8B (`docs/04`), so H1/H2/H3 are answerable; H4/H5 are answerable across the 3 student sizes. The scope limits are stated honestly in the paper's Limitations (`docs/03`, `docs/00 §0.15`): no 32B/72B teacher, students ≤250M, int4 serving.

## Time and energy (measured-formula estimate; calibrate in the pilot)

`make budget PROFILE=local5080` (numbers are GPU-hours on one 5080 = wall-clock, since it runs serially):

| Profile | base | generation | phase-2 train | eval | **total GPU-h** | **wall-clock** | energy |
|---|---|---|---|---|---|---|---|
| **local5080** (accuracy framing, full) | 142 | 188 | 289 | 32 | **651** | **~27 days (≈34 with 1.25× overhead)** | ~309 kWh (~$46) |
| **local5080_fast** (complete, lower power) | 47 | 51 | 53 | 16 | **167** | **~7 days (≈9 with overhead)** | ~78 kWh (~$12) |
| **smoke** (real models, tiny budget) | — | — | — | — | ~1–3 | ~1–3 hours | negligible |
| **micro** (CPU validation) | — | — | — | — | — | minutes (CPU) | negligible |

The full study is **about a month of continuous GPU time** on one 5080. That is the honest cost of a complete teacher×student grid with controls and seeds on a single consumer card. The throughput assumptions (int4 decode tok/s, training MFU) are in `synscale/analysis/compute_budget.py` and are **±2×** until the pilot measures them; the first day of a real run should confirm them (the pilot section of `docs/05`).

## Knobs to make it shorter (each is one CLI flag or a profile edit)

| Change | How | Saves | Cost to the science |
|---|---|---|---|
| Reduced complete grid | `./run_all.sh local5080_fast` | 27→7 days | 2 students, 4 teachers (≤7B), D_syn 150M — lower power, same design |
| Drop the 250M student | edit `students` in `synscale/config/profiles.py` | ~9 days | loses the largest student point (weakens H4) |
| Drop the 14B teacher | remove from `LOCAL_TEACHERS` | ~4–5 days (its generation is the single biggest job) | loses the clearest "beyond-saturation" point |
| Smaller D_syn | set `d_syn`/`d2_tokens` in the profile | ~linear | less dose signal (keep ≥150M; see pilot rule P5) |
| Fewer seeds | lower `phase_seeds`/`c1_seeds` | ~linear in phase-2 | wider CIs; 3 is the floor for a within-cell SD |
| Rent one big card for the 1B/32B/72B rows | run those cells on a cloud RTX PRO 6000 / H100 with the same configs | — | recovers the full 5×7 grid (the cluster design) |

## How the pipeline is organised (what `run_all.sh` does)

`python -m synscale.pipeline --profile local5080` runs, in dependency order, with resume:

1. **prepare_data** — done by `install.sh` via `scripts/prepare_data.py` (tokenizer, FineWeb-Edu base + held-out, StackExchange C1b, Dolly/OASST NLL set, benchmark cache).
2. **build_pool** — freezes the decontaminated prompt pool (`data/prompts/pool-local5080.jsonl`).
3. **generate** — for each teacher, vLLM int4 batch generation → filtered JSONL + manifest. All teachers are generated **before** any training, so vLLM is fully torn down before the trainer starts (no VRAM overlap on the 16 GB card).
4. **pretrain_base** — one θ* per student (3 seeds at 25M) on FineWeb-Edu; θ* is evaluated as C0.
5. **branches** — per cell: subsample the teacher pool to D_syn (equal-token), pack with 25% replay, train the WSD decay phase from θ*, evaluate.
6. **index + analyze** — `results/local5080/index.csv` → floor gate, P_cc cell table, control bands, per-student T* classification, STE. Figures and tables under `results/local5080/analysis/`.

Every run writes a manifest (`results/<id>/manifest.json`) with git commit, config hash, seeds, hardware, runtime, and metrics (`docs/18`).

## Requirements and gotchas

- **GPU/driver**: RTX 5080 (Blackwell, sm_120) needs a recent CUDA (12.4+) PyTorch and a vLLM build with sm_120 kernels. `install.sh` pins the cu124 wheel; if AWQ/FP8 kernels error on your driver, update vLLM or switch a teacher to its `-GPTQ-Int4` repo (configs/teachers).
- **Disk**: ~60–120 GB (base corpus memmap ~50 GB at 25B tokens, teacher pools, checkpoints). Set `SYNSCALE_STORE` to a large disk.
- **Teacher repos**: the profile uses `Qwen/Qwen2.5-*-Instruct-AWQ`. If a repo id 404s, swap to `-GPTQ-Int4` in `synscale/config/profiles.py` (both are int4, uniform precision preserved).
- **`revision: PIN_ME_TO_COMMIT_SHA`** in `configs/teachers/*.yaml`: for a release-grade run, replace with the HF commit SHAs (provenance, `docs/18`); the pipeline runs without it but records `main`.

## What one 5080 cannot do (be honest in the paper)

- 32B and 72B teachers (VRAM); the full 6-rung teacher axis needs a ≥48 GB card or the cloud.
- 500M and 1B students in reasonable wall-clock.
- The BF16-throughout teacher serving of the cluster design (local uses int4; the 3B BF16-vs-int4 bridge cell quantifies the difference).
- Real `lm-eval-harness` numbers are produced by the self-contained scorer (`synscale/evaluation/local_eval.py`), which uses the same log-likelihood convention; `lm_eval_runner.py` remains for an independent cross-check on an exported checkpoint.
