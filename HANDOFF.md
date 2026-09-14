# Session Handoff — Synthetic Scaling Laws (eb1386/syntheticscaling)

A complete, message-by-message reconstruction of the working session that produced the current
state of this repository, plus the technical state, decisions, file inventory, cost numbers, and
exactly what to do next. Written so a fresh session (or a person) can pick up with zero context
loss.

- **Repo:** `eb1386/syntheticscaling` (private)
- **Working branch:** `claude/admiring-fermi-btrw8q` (all work pushed here)
- **As of:** 2026-09-14, commit `d30eab0`
- **Tests:** 68 passing (CPU; torch/vllm tests skip if unavailable)
- **Runnable today:** whole pipeline end to end on CPU via `make micro`; GPU run pending a vast.ai launch
- **One live blocker:** see "The vast.ai network blocker" below (resolved by starting a fresh session)

---

## 1. What this project is

A machine-learning research repository studying **synthetic-data scaling laws for small language
models**. The scientific question the current design (called **C4**) answers is a *mechanism*
question, not an incremental one:

> When a small model is trained on synthetic data from a larger teacher, is the scaling governed by
> **teacher size**, by **teacher family**, or by **measurable properties of the data itself**
> (measured relative to the student)?

The headline construct is a **student-referenced data-quality index `q`** built from three measured
properties of each teacher's data — learnability (NLL of the teacher's data under the student's own
base checkpoint), diversity (distinct-n), and correctness. The central test ("screening") asks
whether teacher identity (size + family) is *screened off* by `q`.

Primary outcome is **held-out loss** (not accuracy): `L(D) = E + A·D^(−α)`, fit per cell, with the
data axis `D` read for free from the training curve (~20 points per run). Five falsifiable
sub-questions (N1–N5): rate-vs-level, screening, a predictive out-of-sample law, compute-optimal
allocation, and cross-family multiplier invariance.

The intended contributor context: **high-school students, no mentor**, targeting a NeurIPS
main-track-plausible result (honest framing: novelty is proven by *results*, not by ambition; a
clean negative is a real finding; workshop is the realistic floor).

---

## 2. Message-by-message history

Messages 1–18 are reconstructed from the pre-compaction summary (the earlier, long portion of the
conversation whose full detail was compacted). Messages 19+ are from the current session and are
first-hand. Each entry is: **what the user asked → what was done.**

### Part A — before context compaction (reconstructed from summary)

**1. "so once I'm ready to run it on a data cluster, is all the code there?"**
→ Reviewed the repo's completeness for cluster execution; confirmed/served the state of the code
base (the project already had a large experimental spec and pipeline from prior work).

**2. "could i run the whole thing locally? i have a 5080"**
→ Assessed feasibility on a single RTX 5080 (16 GB). Began reshaping toward a single-GPU-runnable
study.

**3. "Finish the task, have it ready to run on my rtx 5080 ... one command to install ... another to
run ... enough data for NeurIPS ... estimate how long on my 5080."**
→ Built the two-command flow (`install.sh` / `run_all.sh`), a Makefile, profiles for the 5080, and
time estimates. Established the loss-based, matched-checkpoint, WSD-branching methodology.

**4. "use smaller models ... needs depth and a simple but effective readme and simple code comments
... broad enough that it actually investigates if synthetic data scaling laws [exist]."**
→ Re-derived smaller LLaMA-style student configs at 16k vocab (25M→1B), broadened the teacher/
student grid, tightened README and comments.

**5. Long reviewer message — five lock decisions + additions.** Locked: (a) seed-level fits,
(b) call `E` an *asymptote/`L_inf`* not "irreducible loss", (c) teacher-independent evaluation,
(d) 16k byte-level BPE, (e) `fast` profile = pilot. Add the **synthetic-data multiplier**
(`D_real/D_syn` at matched loss). Do **not** let the README claim "NeurIPS-scale." Stop redesigning
the core experiment; finish implementation.
→ Implemented all five locks, added the multiplier to the analysis, renamed `E`→`L_inf_hat`,
dropped the "NeurIPS-scale" claim.

**6. "fix readme so no em dash less about set up 5080 etc most of it should be the study."**
→ Rewrote the README around the science ("What Governs Synthetic-Data Scaling"), removed em dashes,
minimized setup prose.

**7. "verify it's all done and ready and perfect ... then share the private repo with
Dylanduancanada@gmail.com."**
→ Verified tests/pipeline. **Could not add the collaborator** (no add-collaborator tool available);
reported this limitation honestly.

**8. "does it help small model trainers determine which model and what size synthetic data is best
... [cost]?"**
→ Explained yes: the compute-optimal allocation frontier + the multiplier answer exactly the
"which teacher size and how much synthetic data per dollar" question.

**9. "do you think the results will be shocking?"** and
**10. "do things for NeurIPS need to be shocking — be brutally honest, is this complex enough for
NeurIPS?"**
→ Brutally honest answer: results need not be shocking, but they must be *clean and novel*; a
mechanism/screening finding is more novel than "which teacher is best"; main-track without a mentor
is hard; workshop is realistic.

**11. "i'll run it on runpod, then figure out what model size makes this truly novel, estimate cost/
performance across 5 configs, decide setup, you forget the 5080, optimize + add novelty ... no
mentor, just the high school kids."**
→ Built `synscale/analysis/config_planner.py` with five candidate configs (C1–C5) across GPUs, each
with cost/wall-clock and a blunt NeurIPS read.

**12. "is there a site cheaper than runpod?"**
→ Compared RunPod vs marketplace/spot options (Vast, Spheron); embedded spot vs on-demand pricing.

**13/14. "be honest c2 or c4" → "c3 or c4 i mean."**
→ Honest recommendation: C4 (two families, breaks the size-vs-family confound) or C3 + partial
family; explained the tradeoffs.

**15. "find best way to do c4 cost efficient, ultra-optimize for that setup, change readme, rework
ALL the code to make it a new main-track-NeurIPS-worthy experiment without a mentor, targeting
things never done before."**
→ Reworked into the **C4 study**: two teacher families (Qwen2.5-Instruct full 0.5–72B across all
students + Llama-3.x 3/8/70B as a *partial* cross-family screen at 100M & 1B). Added the novelty
analyses (`analysis/novelty.py`: screening, predictive law, allocation frontier), the data-quality
index (`analysis/data_properties.py`), and `docs/24_novel_experiment.md`. This was committed as the
"Rework into C4" commit (`1d2b5ef`).

**16. "which site do i do, and how do i load it on?"**
→ Guidance toward rented H100s; began the cloud runbook thread.

**17. "estimate for the main training the cost in CAD."**
→ Produced CAD estimates from the planner.

**18. "I WANT IT ULTRA OPTIMIZED HYBRID APPROACH also why not do all spot h100."**
→ Began the **fleet/ultra-optimization** work: single-card int4 serving so no teacher needs
tensor-parallel=2 (everything becomes spot-safe), a file-locked job queue, generation sharding,
SIGTERM-checkpoint preemption handling. (Completed at the start of the current session.)

### Part B — current session (first-hand)

**19. [Auto-continuation after compaction].** Finished and committed the fleet work.
→ Added `scripts/worker.py` (+`--build-only` later) and `scripts/worker.sh`; ran the full suite
(**68 passing**); reconciled the cost planner with reality (single-card int4 + C4's *partial*
second family, which the planner had been over-counting as a full 2× grid); added the spot / hybrid
/ on-demand cost table in USD + CAD; wrote `docs/26_fleet_orchestration.md`; fixed stale FP8/
two-GPU/188-run claims in README and `docs/24`; added `--fleet` mode to `run_all.sh`. Verified the
whole fleet path end to end on the micro profile via `run_all.sh micro --fleet`. Committed as
`1349979`. Delivered the ultra-optimized-hybrid explanation and the honest "why not all spot"
answer: with single-card int4, all-spot ≈ hybrid cost (only the ~76 GPU-h 1B base warrants an
on-demand anchor; the anchor buys availability + a home for the one long serial job, not price).

**20. "please run `npx skills add vast-ai/vast-cli` ... go over everything so it'll work on vast.ai
... just do hybrid ... 100% ready ... get the cli and everything so once I load a balance and log
you in you can do it ... monitor it ... I don't want to waste money if it doesn't work."**
→ Installed the vastai skill (`npx skills add`) and the `vastai` CLI (v1.7.0, in an isolated venv
so it didn't fight Debian's `cryptography`). **Discovered vast.ai's key architectural difference:
no network volume shared across separate instances** (unlike RunPod). Re-mapped the fleet to the
robust vast.ai arrangement: **one multi-GPU box, local disk = shared volume, one GPU-pinned worker
per card** (`scripts/fleet_local.sh`, verified on CPU). Built the money-safe `scripts/vast/`
toolkit (preflight, smoke gate, onstart, launch, monitor, teardown, common). Made `install.sh`
vast-safe (don't clobber the image's CUDA torch; pass HF token for gated Llama). Wrote
`docs/27_vast_runbook.md`. Committed `8aa78e2`. **Then found the live blocker:** this environment's
egress policy blocked vast.ai (403 CONNECT). Documented it (`d30eab0`) and reported it honestly.

**21. [Screenshot of settings: "Allow network egress" ON, "Domain allowlist: All domains"] "will it
work now do i really need new session."**
→ Tested live: vast.ai still `000` (blocked) in this session. Explained that the network policy is
bound at session start, so the (correct) "All domains" change applies only to a **new session**.

**22. "write a comprehensive md file message by message your entire context window as an MD
handoff."**
→ This document.

---

## 3. The current experiment (C4) at a glance

| Axis | Values |
|---|---|
| Students `S` | 25M, 50M, 100M, 250M, 500M, 1B (one LLaMA-style family, shared 16k BPE) |
| Teacher family A | Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B (all students) |
| Teacher family B | Llama-3.x 3B, 8B, 70B at 100M & 1B only (partial cross-family screen) |
| Data `D` | ~20 points 25M→400M per cell, read from the training curve |
| Controls | C1 matched real tokens, C1b human Q&A, C0 base |
| Seeds | 3 per cell (base seeds: 3 at 25M, 1 elsewhere) |
| Serving | **every teacher single-card int4 (AWQ/awq_marlin)** — nothing needs 2 GPUs → spot-safe |
| Scale | ~198 training runs; C4 DAG = 295 fleet jobs at gen_shards=16 |

**Students (16k vocab, re-derived params):** s025m L12 d384 (25.2M), s050m L12 d576 (51.9M),
s100m L14 d768 (100.7M), s250m L24 d960 (251.7M), s500m L25 d1344 (503.8M), s1b L39 d1536 (1.006B).
LLaMA-style: RMSNorm, RoPE, SwiGLU, GQA 3:1, head_dim 64, tied embeddings.

---

## 4. The ultra-optimized fleet (how it runs cheaply)

The enabling decision: **serve every teacher (up to 72B) single-card at int4.** A 70B int4 ≈ 35 GB
fits one 80 GB H100, so there are **no two-GPU jobs**, so **every job fits one spot GPU**.

- `synscale/fleet.py` — file-locked job queue on the shared volume (no server). Atomic claim via
  `os.open(O_CREAT|O_EXCL)`; dependency DAG `prepare → pool → gen_shard[sharded] → base → branch →
  finalize`; `requeue_stale()` reclaims dead jobs after `CLAIM_TTL=3600`s; only `base:s1b:1` is
  tagged `tier="ondemand"`. `ondemand`-tier workers also run spot jobs.
- Generation for 72B/70B is **sharded** across workers (contiguous prompt-pool slices, per-worker
  shard files + manifests).
- Training catches **SIGTERM/SIGUSR1**, writes `resume.pt`, exits **75** (`EX_TEMPFAIL`) → worker
  requeues → another worker resumes. A reclaim costs minutes.
- `scripts/fleet_local.sh` — on ONE multi-GPU box, launches one GPU-pinned worker per card
  (`CUDA_VISIBLE_DEVICES`), all draining the same local queue. This is the vast.ai execution model.
- `scripts/worker.py` / `scripts/worker.sh` — one-job-per-process worker (frees VRAM between jobs)
  and a loop; `--build-only` seeds the queue and prints a summary.

---

## 5. Cost (from `python -m synscale.analysis.config_planner`)

C4 ≈ **1,230 GPU-hours** (single-card int4). Transparent FLOP+bandwidth estimate, ±a factor until
the pilot measures real throughput.

| C4 plan | Cost (USD) | Cost (CAD) |
|---|---|---|
| **All spot** (~$1.49/GPU-h) | ~$1,840 | ~$2,550 |
| **Hybrid** (1B base on-demand, rest spot) | ~$1,910 | ~$2,660 |
| All on-demand (~$2.50/GPU-h) | ~$3,080 | ~$4,280 |

Add ~25% for preemption re-runs → **budget ~$2,500–$3,300 CAD**. All-spot ≈ hybrid because only the
~76 GPU-h 1B base is long enough to warrant an on-demand anchor (which buys *availability*, not
price). On an 8×H100 box the wall-clock is ~6–7 days; on 4×H100 ~13 days (same total dollars).

---

## 6. Running on vast.ai (the current target)

Vast.ai has **no cross-instance shared volume**, so the study runs on **one multi-GPU box** (local
disk = shared volume; the file-lock fleet works within one filesystem). Full runbook:
`docs/27_vast_runbook.md`. Money-safe staged sequence:

```bash
vastai set api-key <KEY>
./scripts/vast/preflight.sh                       # auth + balance + ssh key + offers
./scripts/vast/smoke.sh                            # ~$1-5 real-GPU gate; auto-destroys
./scripts/vast/launch.sh c4 --gpus 8 --hf <TOK>   # only if smoke passes (on-demand)
#   or interruptible: ...launch.sh c4 --gpus 8 --hf <TOK> --bid 12.0
./scripts/vast/monitor.sh                          # one snapshot: RUNNING|DONE|STOPPED|GONE
./scripts/vast/teardown.sh                         # fetch results, destroy, confirm billing stopped
```

**Money-safety rails built in:** every `create` is paired with a timed wait that auto-destroys on
any terminal status; smoke self-destructs (EXIT trap) pass or fail; the funded balance is a hard
cap (vast stops the box, disk preserved, if credit runs out — resumable); teardown lists anything
still billing. The onstart hook installs and runs the fleet detached so the study **survives the
driver session dying**; the code never needs a GitHub token on the third-party box (code is
uploaded via `vastai copy` + a GO marker).

---

## 7. The vast.ai network blocker (current live issue)

**This session cannot reach vast.ai** — its egress policy was "trusted" (GitHub + package
registries only), which returns `403 CONNECT` for `vast.ai`, `console.vast.ai`, `huggingface.co`.
The user then set **Domain allowlist → All domains** (correct fix), but a running session's egress
proxy is **bound at session start** and does not pick up the change live (re-tested: still `000`).

**Resolution: start a fresh Claude Code session on branch `claude/admiring-fermi-btrw8q`.** The new
session boots under "All domains" and vast.ai becomes reachable. First check in the new session:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://console.vast.ai/   # want not-000
```
Everything is committed, so a new session has the full toolkit immediately.

---

## 8. What the user must provide (for the GPU run)

1. **Funded vast.ai balance** — ~$2,400–2,800 USD for the full hybrid run + preemption buffer
   (the balance is a hard cap; the box auto-stops if it's hit).
2. **vast.ai API key** — https://console.vast.ai/manage-keys/
3. **Hugging Face token with Llama-3.x access** — the gated second teacher family; request access
   on the Llama model pages. Qwen teachers need none. Pass as `--hf <TOKEN>`.

---

## 9. File inventory (new/changed in the fleet + vast work)

```
synscale/fleet.py                     NEW  file-locked queue, DAG, sharding, preemption requeue
synscale/analysis/config_planner.py   UPD  single-card int4 + partial family B; spot/hybrid/on-demand USD+CAD
synscale/pipeline.py                  UPD  build_context() so any stage runs standalone from a worker
synscale/generation/runner.py         UPD  shard_index/shard_count → per-worker shard files/manifests
synscale/training/trainer.py          UPD  SIGTERM/SIGUSR1 handler → resume.pt, exit 75 (requeue)
scripts/fleet_local.sh                NEW  on-box: one GPU-pinned worker per GPU (vast execution model)
scripts/worker.py / worker.sh         NEW  one-job-per-process worker (+--build-only) and loop
run_all.sh                            UPD  --fleet mode
install.sh                            UPD  keep image CUDA torch; HF token for gated Llama; header
scripts/vast/common.sh                NEW  helpers; every create pairs with a timed auto-destroy wait
scripts/vast/preflight.sh             NEW  auth + balance + ssh key + offer availability
scripts/vast/smoke.sh                 NEW  the $1-5 real-GPU money gate (auto-destroys)
scripts/vast/onstart.sh               NEW  unattended box setup + detached fleet launch
scripts/vast/launch.sh                NEW  provision one multi-GPU box + release the run
scripts/vast/monitor.sh               NEW  one status snapshot (RUNNING|DONE|STOPPED|GONE)
scripts/vast/teardown.sh              NEW  fetch results, destroy, list anything still billing
docs/26_fleet_orchestration.md        NEW  the fleet design + "why not all spot"
docs/27_vast_runbook.md               NEW  the vast.ai procedure + network requirement + money safety
README.md                             UPD  reconciled to single-card int4; points at docs/26 + 27
.gitignore                            UPD  excludes session tooling (.agents/, .claude/, logs/, etc.)
```

Earlier C4 files already committed in `1d2b5ef`: `analysis/novelty.py`, `analysis/data_properties.py`,
`analysis/scaling.py` (per-seed fits, `L_inf_hat`, multiplier), `docs/24_novel_experiment.md`,
students/teachers/training configs, `scripts/prepare_data.py` (16k BPE), tests.

**Commit trail on the branch:**
`1d2b5ef` Rework into C4 → `1349979` fleet + single-card int4 → `8aa78e2` vast toolkit + on-box
fleet → `d30eab0` document vast egress requirement.

---

## 10. Verified vs. unverified

**Verified (offline):** 68 unit tests pass; the whole pipeline runs end to end on CPU (`make micro`)
including the data index and novelty analyses; the fleet path drains its DAG in dependency order and
produces `index.csv` + `analysis/novelty_report.json` (via both `run_all.sh micro --fleet` and
`scripts/fleet_local.sh micro`); the cost planner runs; every vast script is syntax-clean and its
JSON parsers were unit-checked against fake payloads; the code tarball (`git archive HEAD`) includes
the scripts.

**Not yet verified (needs the GPU run):** the code has **never run on a real GPU** — that is exactly
what `smoke.sh` is for (spends a few dollars to catch any CUDA/vLLM issue before the H100 budget).
Also unverified until an account is connected: vast.ai offer availability/pricing at run time, that
`vastai copy` traverses cleanly, and real teacher-decode throughput (all estimates are ±a factor).

---

## 11. Immediate next step

1. Start a **fresh session** on `claude/admiring-fermi-btrw8q` (for the "All domains" policy).
2. Have the **funded balance + vast API key + HF token** ready.
3. Say go. Order of operations: confirm vast.ai reachable → `vastai set api-key` → `preflight` →
   **`smoke` (must pass)** → `launch c4 --gpus 8 --hf` → `monitor` (scheduled check-ins) →
   `teardown` the moment it's DONE.

Do **not** skip the smoke gate. It is the difference between "a few dollars to find a bug" and
"hundreds of dollars into a broken run."
