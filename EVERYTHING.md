# EVERYTHING — Synthetic Scaling Laws: complete project dossier

The single, self-contained reference for this repository. If you read only one file, read this one.
It contains the science, the full experiment spec, the methodology, the engineering (fleet), the
provider decision and costs, both cloud runbooks, the repo map, the complete command cheat-sheet,
the entire message-by-message history, every key decision and why, what is verified vs not, and
exactly what to do next. A shorter version is `HANDOFF.md`.

- **Repo:** `eb1386/syntheticscaling` (private) · **Branch:** `claude/admiring-fermi-btrw8q`
- **Latest commit:** `d2b6329` · **Tracked files:** 180 · **Tests:** 68 passing (CPU)
- **Recommended provider:** RunPod (reliability + network volumes); vast.ai is the cheaper alternative
- **Runnable now:** whole pipeline end-to-end on CPU via `make micro`; GPU run pending a cloud launch
- **Live blocker:** the build session's egress policy blocks provider APIs; fixed by starting a fresh
  "All domains" session (see §16)

---

## 1. TL;DR

We are testing, under controlled conditions, **what governs synthetic-data scaling for small
language models**: teacher size, teacher family, or measurable properties of the data itself
(measured relative to the student). The primary outcome is **held-out loss**, fit to
`L(D) = E + A·D^(−α)` per cell, with the data axis read for free from training curves. The headline
construct is a **student-referenced data-quality index `q`**, and the centerpiece test asks whether
teacher identity is **screened off** by `q`. The study (**C4**) uses two teacher families (Qwen2.5
full 0.5–72B + Llama-3.x partial) across six students (25M–1B), all teachers served **single-card
int4** so every job fits one GPU. It runs as a **file-locked fleet** on one multi-GPU box, ~**1,230
GPU-hours**, recommended on **RunPod** (~$4,600–5,100 CAD). Everything is built and CPU-verified; the
one thing not yet done is a real-GPU run, gated by a cheap smoke test.

---

## 2. The research question and why it is novel

When a small model is trained on synthetic data written by a larger teacher, does the data obey a
scaling law, and **what controls it**? "Which teacher size is best" is largely answered in direction
(incremental). This study instead identifies the *variable that governs the scaling* and tests
whether the **identity** of the teacher (its size and family) is **screened off** by measurable data
properties. That is a mechanism claim, not a ranking, and has not been answered cleanly for
sequence-level synthetic data at pretraining scale.

Five falsifiable sub-questions (a clean negative on any is a real finding):

- **N1 — Rate vs level.** Does the teacher change the exponent `α` (the *rate* of improvement) or
  only the asymptote `E` (the *level* it saturates at)?
- **N2 — Screening (the centerpiece).** Build a student-referenced quality index `q` from three
  measured properties of each teacher's data — learnability, diversity, correctness. Does `q` screen
  off teacher identity? If `q` predicts a cell's scaling as well as identity does and identity adds
  nothing beyond `q`, then measurable data properties govern the scaling, not which model wrote it.
- **N3 — Predictive law.** Fit `L(S, T, D)` on a subset of cells and predict held-out cells
  out-of-sample, turning measured curves into a law that transfers.
- **N4 — Compute-optimal allocation.** At a fixed compute budget split between generating teacher
  tokens and training the student, which teacher size minimizes loss, and how does the optimum move
  with student size?
- **N5 — Cross-family invariance.** Is the synthetic-data multiplier (real tokens per synthetic
  token at matched loss) a function of `q` alone, invariant to teacher family?

Honest framing (the user has no mentor and is working with high-school students): novelty is proven
by results, not ambition; main-track NeurIPS without a mentor is hard; a strong workshop is the
realistic floor; the design is pre-registration-grade so negatives still publish.

---

## 3. The experiment (C4) — full specification

| Axis | Values |
|---|---|
| Students `S` | 25M, 50M, 100M, 250M, 500M, 1B — one LLaMA-style family, shared 16k byte-level BPE |
| Teacher family A | Qwen2.5-Instruct **0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B** (AWQ int4), across **all** students |
| Teacher family B | Llama **3.2-3B, 3.1-8B, 3.1-70B** (70B AWQ int4) at **100M & 1B only** (cross-family screen) |
| Data `D` | ~20 points 25M→400M per cell, read from the training curve (`d_syn = 400M`) |
| Controls | C1 matched real tokens, C1b human Q&A, C0 base (no new data) |
| Seeds | 3 per cell (`phase_seeds=3`); base seeds: 3 at 25M, 1 elsewhere |
| Serving | **every teacher single-card int4 (`awq_marlin`)** → nothing needs 2 GPUs → spot/crash-safe |
| Primary outcome | **loss** (accuracy kept only as a secondary transfer-breadth check) |
| Scale | ~200 annealing runs; the fleet DAG builds **295 jobs** at `gen_shards=16` |

**Student architectures (16k vocab, exact params):** s025m L12 d384 (25.2M), s050m L12 d576 (51.9M),
s100m L14 d768 (100.7M), s250m L24 d960 (251.7M), s500m L25 d1344 (503.8M), s1b L39 d1536 (1.006B).
LLaMA-style: RMSNorm, RoPE, SwiGLU, GQA 3:1, head_dim 64, tied embeddings.

**Base pretraining tokens per student (`base_tokens`):** s025m 0.5B, s050m 1B, s100m 2B, s250m 4.9B,
s500m 10B, s1b 15B (roughly Chinchilla, 20 tok/param).

**Teachers (exact repos):** Qwen `Qwen/Qwen2.5-{0.5,1.5,3,7,14,32,72}B-Instruct-AWQ`; Llama
`meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`. Llama repos are **gated** → need an HF token.

**Profile knobs (`c4`):** `gpu=h100_80`, `persistent_dir=/workspace/synscale`,
`teacher_precision=awq_marlin`, `measure_data_properties=True`, `training_prefix=train_scaling_`.

---

## 4. Methodology (how the numbers are made trustworthy)

- **Loss, not accuracy.** Loss is what scaling laws are defined on, moves smoothly from the first
  tokens, and has far higher signal-to-noise than multiple-choice accuracy (which sits at chance for
  the smallest students). Accuracy is a secondary check only.
- **Matched checkpoints + WSD branching.** Each student is pretrained once on real text to a stable
  checkpoint θ*. Every condition **branches from that same θ*** into an identical short annealing
  phase whose only difference is the data it sees (synthetic from one teacher / equal fresh real
  tokens / human Q&A / nothing). So any outcome difference is attributable to the data source alone.
- **Free D-axis.** The data-quantity axis `D` is read from the training curve
  (`train_log.jsonl` held-out NLL), so one run yields ~20 points on the scaling curve for free.
- **The scaling fit.** `L(D) = E + A·D^(−α)` per cell, fit at the **seed level** (uncertainty from
  seeds, not pooled). `E` is reported as an **asymptote over the observed range** (`L_inf_hat`), never
  called "irreducible loss." Three functional forms are compared on held-out checkpoints.
- **The data-quality index `q` (student-referenced).** Its key term is **learnability**: the
  per-token NLL of a teacher's data under that specific student's base model θ* — a compatibility
  measure, so `q` is student-specific, exactly what a capacity-dependent theory predicts. Plus
  **diversity** (distinct-n) and **correctness**. Combined via z-scores into a scalar `q`.
- **Synthetic-data multiplier.** `D_real(L)/D_syn(L)` at matched loss — real tokens' worth per
  synthetic token.
- **Teacher-independent evaluation, 16k BPE, intra-document masking** — all locked in review.

Design docs: methodology `docs/00`, scaling method `docs/22`, the C4 thesis `docs/24`, locked review
decisions `docs/23`, statistical plan `docs/12`.

---

## 5. Analyses and outputs

After a run, `results/c4/analysis/` holds:

- `scaling_instr/scaling_fits.csv` — `E`, `A`, `α` per cell with seed-level uncertainty.
- `novelty_report.json` — the **screening** result (does `q` screen off size and family), the
  out-of-sample **predictive-law** error, and the **compute-allocation frontier**.
- `scaling_instr/scaling_report.json` — the **rate-vs-level** tests and the **multipliers**.
- `data_props.json` — learnability, diversity, correctness per teacher per student, and derived `q`.
- Secondary accuracy tables/figures, plus one manifest per run (git commit, config hash, seeds, HW).

Code: `synscale/analysis/scaling.py` (fits, `L_inf_hat`, multiplier), `analysis/novelty.py`
(screening / predictive / allocation), `analysis/data_properties.py` (q), `analysis/run.py` (driver).

---

## 6. The ultra-optimized fleet (engineering)

**Enabling decision:** serve **every** teacher (up to 72B) **single-card int4 (AWQ)**. A 70B int4 ≈
35 GB fits one 80 GB H100, so there are **no two-GPU jobs**, so **every job fits one GPU** and is
safe to preempt or lose. Quality cost of AWQ int4 is a fraction of a point and far downstream of the
student's loss; the pilot confirms it.

- **`synscale/fleet.py`** — a file-locked job queue on the shared volume, no server. Claim is a
  **hardlink lock** (atomic even on RunPod's NFS network volumes, where plain `O_EXCL` is not
  reliable; also correct on local disk). Dependency DAG:
  `prepare → pool → gen_shard[sharded] → base → branch → finalize`. `requeue_stale()` reclaims dead
  jobs after `CLAIM_TTL=3600s`. Only `base:s1b:1` is tagged `tier="ondemand"`; `ondemand` workers run
  spot jobs too. C4 DAG at `gen_shards=16` = **295 jobs** (1 prepare, 1 pool, 104 gen_shard, 8 base,
  180 branch, 1 finalize).
- **Generation sharding** — the 72B/70B jobs split into contiguous prompt-pool slices, each a job
  with its own shard file + manifest, so the longest work parallelizes across GPUs.
- **Preemption-safe training** — `trainer.py` catches **SIGTERM/SIGUSR1**, writes `resume.pt`, exits
  **75** (`EX_TEMPFAIL`); the worker requeues and another resumes. A reclaim costs minutes.
- **`scripts/fleet_local.sh`** — the on-box engine: one GPU-pinned worker per card
  (`CUDA_VISIBLE_DEVICES`), all draining the same queue. Used identically on RunPod and vast.ai.
- **`scripts/worker.py` / `worker.sh`** — one-job-per-process worker (frees VRAM between jobs) and a
  loop; `--build-only` seeds the queue and prints a summary.

Full design + the "why not all spot" reasoning: `docs/26_fleet_orchestration.md`.

---

## 7. Provider decision and cost (verified Sep 2026)

C4 ≈ **1,230 GPU-hours** (single-card int4). From `python -m synscale.analysis.config_planner`,
cheapest-first, with a reliability read:

| Provider | $/GPU-h | C4 total | Reliability |
|---|---|---|---|
| Vast.ai spot | ~$1.49 | ~$1,840 / ~$2,550 CAD | No SLA, peer-to-peer, no shared volume. Least reliable. |
| TensorDock | ~$2.25 | ~$2,770 / ~$3,850 CAD | Curated marketplace; cost-conscious middle. |
| **RunPod Community** | ~$2.69 | **~$3,310 / ~$4,610 CAD** | **Recommended default.** Network volumes (durable). |
| **RunPod Secure** | ~$2.99 | ~$3,680 / ~$5,120 CAD | Most reliable. RunPod-owned, 99% SLA. |
| Modal | ~$3.95 | ~$4,870 / ~$6,760 CAD | Serverless, preempts by default. Paradigm mismatch. |
| Lambda Cloud | ~$3.99 | ~$4,920 / ~$6,830 CAD | Best SLA (99.9%) but 8-GPU-only, no spot, often sold out. |

**RunPod is recommended.** Its **network volumes survive a pod crash**, so work is never lost — and
the fleet's checkpoint+requeue turns an occasional pod failure into a non-event. Lambda is most
reliable on paper but you often can't get H100 capacity at all; Vast is cheapest but unreliable;
TensorDock is the middle. Add ~25% for preemption re-runs → budget ~**$5,500–6,000 CAD** to finish
comfortably on RunPod, or ~**$2,500–3,300 CAD** on Vast if you accept the risk. Wall-clock on an
8×H100 box ≈ 6–7 days; on 4×H100 ≈ 13 days (same total dollars). Network storage ≈ $0.07/GB/mo (~$10
for 600 GB over the run).

---

## 8. RunPod runbook (recommended) — step by step

Architecture: **one multi-GPU pod + a persistent network volume at `/workspace` + one GPU-pinned
worker per card.** The pod can die and be replaced; the volume (and all work) persists.

**You provide:** funded RunPod balance (~$5,500–6,000 CAD to finish comfortably); RunPod API key; a
~600 GB **network volume** created in the console in an H100-stocked region; a Hugging Face token
with Llama-3.x access; a read-only GitHub PAT scoped to this repo (so the pod can clone; dies with
the pod).

```bash
# in the fresh all-domains session:
runpodctl config --apiKey <RUNPOD_KEY>
./scripts/runpod/preflight.sh                     # verify auth + reminders

# SMOKE GATE FIRST (a few $): prove the code on real CUDA. Quickest is a single cheap GPU with
# PROFILE=smoke, or run the fully-automated Vast smoke gate (scripts/vast/smoke.sh, ~$1-5).

# create a ~600GB network volume in the console (Storage -> Network Volumes), note its id, then:
./scripts/runpod/launch.sh c4 --volume <VOL_ID> --gpus 8 --gh-token <GH_PAT> --hf <HF_TOKEN>
#   add --secure for RunPod Secure Cloud (more reliable, pricier)

./scripts/runpod/monitor.sh                       # pod state + where to read live progress
./scripts/runpod/teardown.sh                      # remove pod (billing stops); volume persists
```

The pod self-runs `scripts/runpod/bootstrap.sh` (clones with the PAT, installs, launches the fleet
detached), so the study survives the driver session dying. RunPod's remote exec is SSH-based, so live
progress (`/workspace/{status.txt,run.log,DONE}`) is read from the console Logs tab or over SSH;
monitoring may need an occasional pasted snippet. **Console fallback** (paste the printed Container
Start Command) and a **no-token manual-upload path** are both documented in `docs/28_runpod_runbook.md`.

---

## 9. Vast.ai runbook (cheaper alternative) — fully automated

Vast has no cross-instance shared volume, so it runs on **one multi-GPU box, local disk = shared
volume**. The `scripts/vast/` toolkit drives everything over the vast HTTPS API:

```bash
vastai set api-key <KEY>
./scripts/vast/preflight.sh                        # auth + balance + ssh key + offers
./scripts/vast/smoke.sh                            # ~$1-5 real-GPU gate; AUTO-DESTROYS pass or fail
./scripts/vast/launch.sh c4 --gpus 8 --hf <TOK>    # only if smoke passes (on-demand)
#   or interruptible/cheaper:  ... --bid 12.0
./scripts/vast/monitor.sh                          # one snapshot: RUNNING|DONE|STOPPED|GONE
./scripts/vast/teardown.sh                         # fetch results, destroy, confirm billing stopped
```

Every `create` pairs with a timed wait that **auto-destroys** on any terminal status. The onstart
hook installs + runs the fleet detached (survives the driver session); code is uploaded via
`vastai copy` + a GO marker (no GitHub token on the box). Full detail: `docs/27_vast_runbook.md`.

---

## 10. What the user provides (summary)

| Item | RunPod | Vast (alt) |
|---|---|---|
| Funded balance | ~$5,500–6,000 CAD to finish | ~$2,400–2,800 USD |
| API key | console.runpod.io settings | console.vast.ai/manage-keys |
| Persistent storage | ~600 GB **network volume** (console) | local disk (auto, `--disk 600`) |
| HF token (Llama access) | yes (`--hf`) | yes (`--hf`) |
| GitHub PAT (read-only, this repo) | yes (or manual-upload path) | not needed (uses `vastai copy`) |

Both need a **fresh all-domains session** so the provider APIs are reachable (see §16).

---

## 11. Money-safety rails (built in)

- **Smoke gate first** — a few dollars on one GPU proves the code on real CUDA before any H100 spend.
- **Auto-destroy on failure** — every provisioning create is paired with a timed wait that destroys
  the box on `exited/offline/unknown`/timeout; the vast smoke self-destructs pass or fail.
- **Balance is a hard cap** — if credit runs out the box/pod stops (RunPod volume / vast disk
  preserved) and the run resumes on top-up. You cannot overspend your balance.
- **Cost ceilings** — `--cost` (RunPod) / on-demand default (vast) prevent taking a pricier host.
- **Tear down on DONE** — remove the box the moment `/workspace/DONE` appears; volume keeps results.
- The fleet is **resumable and crash-tolerant** by design (requeue + checkpoint + durable volume).

---

## 12. Repository map

```
synscale/
  config/        profiles.py (c4/smoke/micro/5080 + TeacherSpec/Profile), schemas, loader, matrix
  models/        llama_like.py (student), param_count.py (exact param counts)
  generation/    prompt_pool, backend (vLLM), templates, filters, subsample, runner (sharding), decontam
  training/      trainer.py (WSD + SIGTERM checkpoint), data.py, schedules.py, cli.py
  evaluation/    local_eval, nll_eval, benchmarks, contamination_audit, format_probe (self-contained MC)
  analysis/      scaling.py, novelty.py, data_properties.py, config_planner.py (cost+providers),
                 run.py, aggregate, regression, mediation, ste, tstar, plots, compute_budget, simulate
  tracking/      index.py (results index), manifest.py, ids.py
  fleet.py       file-locked queue, DAG, sharding, hardlink lock, preemption requeue
  pipeline.py    orchestrator; build_context() lets any stage run standalone from a worker
scripts/
  install/prepare_data/pretrain_base/build_branch/train/analyze/make_matrix/worker(.py/.sh)/fleet_local.sh
  runpod/  bootstrap, preflight, launch, monitor, teardown          (recommended provider)
  vast/    common, preflight, smoke, onstart, launch, monitor, teardown   (alternative)
configs/   students/ (s025m..s1b, s_micro), teachers/ (qwen+llama), training/ (train_scaling_*),
           generation/, evaluation/   — all 16k vocab
docs/      00-28 + RUN_ON_5080 (see §13)
paper/     abstract_draft, introduction_draft, methods_draft
tests/     68 tests, CPU only (torch/vllm tests skip if unavailable)
install.sh · run_all.sh · Makefile · pyproject.toml
HANDOFF.md (short handoff) · EVERYTHING.md (this file)
```

180 tracked files. Deps (pyproject extras): `train` (torch, datasets, tokenizers, safetensors),
`gen` (vllm, transformers), `eval` (lm-eval), `analysis` (pandas, scipy, statsmodels, matplotlib,
scikit-learn), `dev` (pytest, ruff).

---

## 13. Documentation index (docs/)

```
00 Locked Methodology (decision record)     15 Ablation and Cross-Check Plan
01 Research Question and Hypotheses          16 Compute Budget (single RTX 5080)
02 Design Overview (one page)                17 Repository Architecture
03 Novelty Assessment                        18 Experiment-Tracking Standard
04 Literature Review and Novelty Matrix      19 Failure-Mode Register
05 Pilot Design and Go/No-Go Rules           20 Figure and Table Plan
06 Full Experimental Matrix                  21 Paper Outline
07 Student Architectures (derivation)        22 Scaling-Law Analysis (the core method)
08 Teacher-Model Selection                   23 Locked Decisions from Design Review
09 Synthetic-Data Generation Methodology     24 The Experiment (C4): what governs scaling  ← the thesis
10 Student-Training Methodology              25 Next Implementation Tasks (dependency order)
11 Benchmark Suite and Evaluation Protocol   26 Fleet Orchestration (the ultra-optimized path)
12 Statistical Analysis Plan (pre-reg grade) 27 Vast.ai Runbook (alternative)
13 Cost Model, Hardware, Throughput          28 RunPod Runbook (recommended)  ← start here for the run
14 Synthetic Transfer Efficiency (STE)       RUN_ON_5080 (single-GPU pilot)
```

---

## 14. Profiles reference

- **`c4`** — the main study (H100, 2 families, 6 students; see §3). Default for install/run.
- **`smoke`** — GPU sanity (1 student s025m, 2 teachers, `d_syn=200k`, real vLLM). ~1–3h, the pre-spend gate.
- **`micro`** — CPU end-to-end with a mock backend (1 student, 2 teachers, `d_syn=100k`). Minutes;
  runs every stage including the data index, novelty analyses, and the fleet queue. Used for all
  offline verification here.
- **`scaling5080` / `scaling5080_fast` / `local5080*`** — the earlier single-RTX-5080 pilots (kept as
  a cheap pilot to measure real throughput before committing the H100 budget).

---

## 15. Command cheat-sheet

```bash
# offline validation (no GPU, no money)
make micro                         # whole pipeline end-to-end on CPU (mock backend)
make budget PROFILE=c4             # the provider cost comparison table
python -m synscale.analysis.config_planner   # same, directly
make test                          # 68 unit tests

# on a GPU box (any provider)
./install.sh c4                    # deps + tokenizer + data + teacher models + prompt pool
./run_all.sh c4 --fleet            # build the queue and drain it with one worker per GPU
./scripts/fleet_local.sh c4        # (equivalent) on-box: one GPU-pinned worker per GPU
make smoke                         # real teachers, tiny budget (GPU sanity)

# fleet internals
python scripts/worker.py --profile c4 --queue-dir <dir> --build-only   # seed queue, print summary
python scripts/worker.py --profile c4 --tier ondemand --queue-dir <dir> # claim+run one job

# provider control (in the all-domains session)
runpodctl config --apiKey <K>;  ./scripts/runpod/{preflight,launch,monitor,teardown}.sh
vastai set api-key <K>;          ./scripts/vast/{preflight,smoke,launch,monitor,teardown}.sh
```

---

## 16. The egress blocker (current live issue)

The **build session cannot reach provider APIs** — its egress policy was "trusted" (GitHub + package
registries only), returning `403 CONNECT` for `api.runpod.io`, `vast.ai`, `console.vast.ai`,
`huggingface.co`. The user set **Domain allowlist → All domains** (correct), but a running session's
egress proxy is **bound at session start** and does not pick up the change live (re-tested: still
`000`). **Resolution: start a fresh Claude Code session on `claude/admiring-fermi-btrw8q`.** First
check there:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://api.runpod.io/    # want not-000
```

`runpodctl` 2.14.0 and `vastai` 1.7.0 are installed; both need the new session to reach their APIs.

---

## 17. Full message-by-message history

Messages 1–18 are reconstructed from the pre-compaction summary; 19+ are first-hand.

1. **"is all the code there [for a cluster]?"** → reviewed cluster-readiness.
2. **"could i run it locally? i have a 5080"** → began single-5080 feasibility.
3. **"finish it, ready to run on my 5080, one install + one run command, enough data for NeurIPS,
   estimate time"** → built install.sh/run_all.sh/Makefile, 5080 profiles, estimates; locked the
   loss-based matched-checkpoint WSD methodology.
4. **"use smaller models … simple readme + comments … broad enough to actually investigate synthetic
   scaling laws"** → re-derived smaller students at 16k vocab; broadened the grid.
5. **Reviewer message (5 locks + additions)** → seed-level fits; `E`→`L_inf_hat` (not "irreducible");
   teacher-independent eval; 16k BPE; fast=pilot; added the synthetic-data multiplier; dropped the
   "NeurIPS-scale" claim; froze the core design.
6. **"fix readme, no em dash, less 5080 setup, mostly the study"** → rewrote the README.
7. **"verify it's perfect, then share the private repo with Dylanduancanada@gmail.com"** → verified;
   **could not add a collaborator** (no tool) — reported honestly.
8. **"does it help pick model + synthetic-data size to optimize cost?"** → yes: the allocation
   frontier + multiplier answer exactly that.
9. **"will results be shocking?"** & 10. **"do NeurIPS results need to be shocking — brutally honest,
   is this complex enough?"** → honest: not shocking but clean+novel; mechanism > ranking; main-track
   hard without a mentor; workshop realistic.
11. **"I'll run on RunPod, compare 5 configs, drop the 5080, add novelty, pilot then main; no mentor,
    high-schoolers"** → built `config_planner.py` (C1–C5).
12. **"cheaper than RunPod?"** → compared marketplaces/spot.
13/14. **"c2 or c4 … c3 or c4"** → recommended C4 (or C3 + partial family).
15. **"make C4 cost-efficient, ultra-optimize, rework ALL the code into a new main-track-worthy
    experiment, things never done before"** → reworked into **C4** (2 families, novelty analyses, q
    index, `docs/24`); commit `1d2b5ef`.
16. **"which site, how do I load it"** → cloud runbook thread begins.
17. **"estimate main-training cost in CAD"** → CAD estimates.
18. **"ULTRA OPTIMIZED HYBRID … why not all spot h100"** → the fleet: single-card int4, file-lock
    queue, sharding, SIGTERM checkpointing.
19. **[continuation]** → finished + committed the fleet (`1349979`); reconciled the planner (partial
    family; single-card int4); wrote `docs/26`; `--fleet` mode; delivered the "why not all spot"
    answer (all-spot ≈ hybrid; the anchor buys availability, not price).
20. **"run `npx skills add vast-ai/vast-cli`, make it 100% ready on vast.ai, hybrid, monitor it,
    don't waste money"** → installed the vastai skill + CLI; found vast has no shared cross-instance
    volume → single-box model; built `scripts/vast/` + `docs/27`; commit `8aa78e2`; then found the
    vast egress block and documented it (`d30eab0`).
21. **[settings screenshot] "will it work now, do I need a new session?"** → tested: still blocked;
    the All-domains change applies to a new session only.
22. **"write a comprehensive md, message by message, as a handoff"** → `HANDOFF.md`, commit `7e3cf34`.
23. **"vast is unreliable, maybe RunPod — check ACTUAL costs across koyeb/shadeform/tensordock/lambda/
    modal/runpod, change code + handoff so the new one is best/most reliable"** → researched current
    prices; **switched to RunPod**; hardlink lock for NFS; provider comparison in the planner;
    `scripts/runpod/` + `docs/28`; updated README + HANDOFF; commit `d2b6329`.
24. **"make the md with EVERYTHING"** → this file (`EVERYTHING.md`).

---

## 18. Key decisions and rationale

- **Reframe "which teacher is best" → "what governs the scaling" (screening).** A mechanism claim is
  more novel and yields a clean negative either way.
- **Loss as the primary outcome.** Higher SNR, defined for scaling laws, free D-axis from curves.
- **Single-card int4 serving.** Eliminates all 2-GPU jobs → every job fits one GPU → spot/crash-safe
  → the whole fleet story becomes possible. The pivot that unlocked cheap, robust execution.
- **Partial second family (Llama at 2 student sizes).** Breaks the size-vs-family confound without
  doubling the grid — what keeps C4 affordable.
- **File-lock fleet on one box (not a server, not multi-pod NFS).** Robust everywhere; local file
  locking is reliable; the hardlink lock makes even NFS volumes safe.
- **Provider = RunPod.** Reliability + network volumes (work survives a crash) + real availability;
  the fleet absorbs RunPod's occasional pod failures. Chosen over Vast (unreliable) and Lambda
  (sold out / no spot / 8-GPU-min).
- **Smoke gate before any big spend.** The single most important money-saver.

---

## 19. Errors fixed along the way (so they don't recur)

- optimizer test tripping on `tensor in list` → used `id()` sets.
- runner property calls (`str_kind()` etc. were properties) → attribute access.
- MockBackend "tok" prefix tripped the degeneration filter → varied pseudo-words.
- `build_branch` "d2 must be multiple of seq_len" → `d2=(d2//seq_len)*seq_len`.
- treatment shortfall check used the wrong length attr → `len(treatment)`.
- missing eval fixtures (social_iqa/hellaswag) → generated all primary-task fixtures.
- `gate_from_index` returns a GateResult (not tuple) → fixed unpacking.
- `tstar` needed `t_params_nominal` → added the alias.
- scaling `_mean_curve` binning too coarse for micro → relative binning.
- `build_context` referenced before defined → added to pipeline.py.
- install.sh reinstalling torch could clobber the image's CUDA build → install torch only if missing
  (`--system-site-packages` venv).
- fleet claim not NFS-safe → **hardlink lock**.

---

## 20. Verified vs. unverified

**Verified (offline):** 68 unit tests pass; the whole pipeline runs end-to-end on CPU (`make micro`)
including the data index and novelty analyses; the fleet drains its DAG in dependency order and
produces `index.csv` + `analysis/novelty_report.json` (via `run_all.sh micro --fleet` and
`scripts/fleet_local.sh micro`, with the hardlink lock); the cost planner runs; every launch script
is syntax-clean and its JSON parsers were unit-checked against fake payloads; the code tarball
(`git archive HEAD`) includes the scripts.

**Not yet verified (needs the GPU run / a live account):** the code has **never run on a real GPU** —
that is what the smoke gate is for; the RunPod `runpodctl` provisioning path and whether the pod
start-command runs the bootstrap cleanly (console fallback covers it); real H100 availability/pricing;
`vastai copy` traversal; and real teacher-decode throughput (all cost/time estimates are ±a factor).

---

## 21. Commit trail (branch `claude/admiring-fermi-btrw8q`)

```
1d2b5ef  Rework into C4 (two-family, mechanism-targeted study)
1349979  Fleet orchestration + single-card int4 serving
8aa78e2  Vast.ai launch/monitor toolkit + on-box multi-GPU fleet
d30eab0  Document the vast.ai egress requirement
7e3cf34  Add HANDOFF.md (message-by-message handoff)
d2b6329  Switch recommended provider to RunPod; provider cost comparison; NFS-safe lock; docs/28
(this)   Add EVERYTHING.md (complete dossier)
```

---

## 22. Honest caveats

- **No mentor, high-school team.** Main-track NeurIPS is genuinely hard without a senior co-author;
  a strong workshop or arXiv paper is the realistic and still-valuable target. The design is built so
  a clean negative is publishable.
- **Estimates are ±a factor** until the pilot measures real throughput; treat all costs/timings as
  planning figures, not guarantees.
- **Reliability costs money.** RunPod is ~$2,000 CAD more than Vast for C4; that buys durability and
  availability. Choose per your risk tolerance — the code runs on either.
- **Results are not guaranteed to be "shocking."** The value is a clean, pre-registered mechanism
  result (screening) with a validated predictive law — novel regardless of which way it lands.

---

## 23. Immediate next step

1. Start a **fresh session** on `claude/admiring-fermi-btrw8q` (for the All-domains egress).
2. Have ready: **funded RunPod balance + API key + a ~600 GB network volume + HF token (Llama access)
   + read-only GitHub PAT**.
3. Say go. Order: confirm `api.runpod.io` reachable → `runpodctl config --apiKey` →
   `runpod/preflight.sh` → **smoke on one cheap GPU (must pass)** → create the network volume →
   `runpod/launch.sh c4 --volume <id> --gpus 8 --gh-token <pat> --hf <tok>` → `runpod/monitor.sh`
   (scheduled check-ins) → `runpod/teardown.sh` the moment `/workspace/DONE` appears.
   (Vast.ai remains a fully-automated cheaper alternative via `scripts/vast/`.)

**Do not skip the smoke gate.** It is the difference between a few dollars to find a bug and hundreds
into a broken run.
