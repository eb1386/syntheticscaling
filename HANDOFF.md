# Session Handoff — Synthetic Scaling Laws (eb1386/syntheticscaling)

A complete, message-by-message reconstruction of the working session that produced the current
state of this repository, plus the technical state, decisions, file inventory, cost numbers, and
exactly what to do next. Written so a fresh session (or a person) can pick up with zero context
loss.

- **Repo:** `eb1386/syntheticscaling` (private)
- **Working branch:** `claude/admiring-fermi-btrw8q` (all work pushed here)
- **As of:** 2026-09-14, latest fleet/RunPod work on this branch
- **Tests:** 68 passing (CPU; torch/vllm tests skip if unavailable)
- **Recommended provider:** **RunPod** (reliability + network volumes); vast.ai is the cheaper alternative
- **Runnable today:** whole pipeline end to end on CPU via `make micro`; GPU run pending a RunPod launch
- **One live blocker:** see "The egress blocker" below (resolved by starting a fresh all-domains session)

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
→ Created this document (`HANDOFF.md`), committed `7e3cf34`.

**23. "I heard vast.ai is really unreliable, maybe RunPod. Check ACTUAL current costs, change code,
update handoff so the new one is the best/most reliable option: Koyeb / Shadeform / TensorDock /
Lambda Labs / Modal or RunPod."**
→ Researched current (Sep 2026) H100 pricing + reliability via web search. **Switched the
recommended provider to RunPod** and documented why: RunPod Secure (99% SLA, RunPod-owned) or
Community, both with **network volumes that survive a pod crash** so work is never lost — which,
with the fleet's checkpoint+requeue, absorbs RunPod's occasional pod failures. Lambda has the best
SLA but is 8-GPU-only, no spot, and frequently sold out (capacity is the risk); Vast is cheapest
but no SLA / no shared volume; TensorDock is the cost-conscious middle; Modal/Koyeb are serverless
mismatches. Installed `runpodctl` (2.14.0). Hardened `synscale/fleet.py`'s claim to a **hardlink
lock** (atomic on RunPod's NFS network volumes, where plain `O_EXCL` is not enough). Rewrote the
cost model (`config_planner.py`) with real per-provider rates and a ranked comparison. Wrote the
`scripts/runpod/` toolkit (bootstrap, preflight, launch, monitor, teardown) and
`docs/28_runpod_runbook.md`. Updated README + this handoff. (This turn.)

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
  (`CUDA_VISIBLE_DEVICES`), all draining the same queue. This is the single-box execution model used
  on **both** RunPod (pod + network volume) and vast.ai; only provisioning differs.
- `synscale/fleet.py` claim is a **hardlink lock**, atomic on RunPod's NFS network volumes (plain
  `O_EXCL` is not reliably atomic there) and on local disk.
- `scripts/worker.py` / `scripts/worker.sh` — one-job-per-process worker (frees VRAM between jobs)
  and a loop; `--build-only` seeds the queue and prints a summary.

---

## 5. Cost + provider choice (from `python -m synscale.analysis.config_planner`)

C4 ≈ **1,230 GPU-hours** (single-card int4). Verified H100-80GB rates (Sep 2026), cheapest-first:

| Provider | $/GPU-h | C4 total | Reliability |
|---|---|---|---|
| Vast.ai spot | ~$1.49 | ~$1,840 / ~$2,550 CAD | No SLA, no shared volume. Least reliable. |
| TensorDock | ~$2.25 | ~$2,770 / ~$3,850 CAD | Curated marketplace; cost-conscious middle. |
| **RunPod Community** | ~$2.69 | **~$3,310 / ~$4,610 CAD** | **Recommended default.** Network volumes (durable). |
| **RunPod Secure** | ~$2.99 | ~$3,680 / ~$5,120 CAD | Most reliable. RunPod-owned, 99% SLA. |
| Modal / Lambda | ~$3.95–3.99 | ~$4,870–4,920 / ~$6,800 CAD | Serverless mismatch / sold out. |

**RunPod is the recommendation** (message 23): network volumes survive a pod crash so work is never
lost, which the fleet's checkpoint+requeue turns into a non-event; real availability (Lambda sells
out). Budget ~**$4,600–5,100 CAD** (Community→Secure) + ~25% re-runs → ~$5,500–6,000 CAD to finish
comfortably. Vast (~$2,550 CAD) is ~$2,000 cheaper but unreliable; TensorDock is the middle. On an
8×H100 box wall-clock is ~6–7 days; on 4×H100 ~13 days (same total dollars).

---

## 6. Running on RunPod (recommended target)

One multi-GPU pod + a persistent **network volume** at `/workspace` (survives pod death) + one
GPU-pinned worker per card. Full runbook + provider comparison + console fallback:
`docs/28_runpod_runbook.md`. The queue claim uses a **hardlink lock** (atomic on RunPod's NFS
network volumes; plain `O_EXCL` is not enough there). Sequence (run in the new all-domains session):

```bash
runpodctl config --apiKey <RUNPOD_KEY>
./scripts/runpod/preflight.sh                      # verify auth + reminders
# create a ~600GB network volume in the console (Storage -> Network Volumes), note its id, then:
./scripts/runpod/launch.sh c4 --volume <VOL_ID> --gpus 8 --gh-token <GH_PAT> --hf <HF_TOKEN>
#   add --secure for RunPod Secure Cloud
./scripts/runpod/monitor.sh                        # pod state + where to read live progress
./scripts/runpod/teardown.sh                       # remove pod (billing stops); volume persists
```

The pod self-runs `scripts/runpod/bootstrap.sh` (clones with a read-only PAT, installs, launches the
fleet detached), so the study survives the driver session dying. RunPod's remote exec is SSH-based,
so live progress is read from the console Logs tab or over SSH; monitoring may need an occasional
pasted log snippet. A **console fallback** and a **no-token manual-upload path** are in the runbook.
`runpodctl` 2.14.0 is installed; `api.runpod.io` is egress-blocked in *this* build session, so the
provisioning calls are **untested here** and run in the new all-domains session.

### vast.ai alternative (single-box, fully automated)

Cheaper, less reliable. The `scripts/vast/` toolkit is fully automated over the vast HTTPS API
(`preflight → smoke → launch → monitor → teardown`), with a self-destructing ~$1–5 smoke gate and
auto-destroy-on-failure rails; runbook `docs/27_vast_runbook.md`. Both providers use the same on-box
engine (`scripts/fleet_local.sh`); only provisioning differs.

**Money-safety rails (both providers):** validate on real CUDA with a cheap smoke run before the
H100 budget; the funded balance is a hard cap; tear the box/pod down the moment `/workspace/DONE`
appears; `--cost`/on-demand ceilings prevent taking a pricier host than intended.

---

## 7. The egress blocker (current live issue)

**This build session cannot reach the provider APIs** — its egress policy was "trusted" (GitHub +
package registries only), returning `403 CONNECT` for `vast.ai`, `console.vast.ai`,
`api.runpod.io`, `huggingface.co`. The user then set **Domain allowlist → All domains** (correct
fix), but a running session's egress proxy is **bound at session start** and does not pick up the
change live (re-tested: still `000`).

**Resolution: start a fresh Claude Code session on branch `claude/admiring-fermi-btrw8q`.** The new
session boots under "All domains" and the provider APIs become reachable. First check in the new
session:
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://api.runpod.io/    # want not-000
```
Everything is committed, so a new session has the full toolkit immediately.

---

## 8. What the user must provide (for the RunPod run)

1. **Funded RunPod balance** — ~$4,600–5,100 CAD for C4 (Community→Secure) + ~25% re-runs, so
   ~$5,500–6,000 CAD to finish comfortably. (Vast alternative: ~$2,400–2,800 USD.)
2. **RunPod API key** — https://www.runpod.io/console/user/settings → `runpodctl config --apiKey`.
3. **A network volume** (~600 GB) created in the RunPod console, in a region with H100 SXM stock.
4. **Hugging Face token with Llama-3.x access** — the gated second teacher family (Qwen needs none).
5. **A read-only GitHub PAT** scoped to `eb1386/syntheticscaling` (so the pod can clone; dies with
   the pod). Optional if using the manual-upload path.

---

## 9. File inventory (new/changed across the fleet + vast + RunPod work)

```
synscale/fleet.py                     NEW  file-locked queue (HARDLINK lock, NFS-safe), DAG, sharding, requeue
synscale/analysis/config_planner.py   UPD  single-card int4 + partial family B; PROVIDERS table + ranked comparison
synscale/pipeline.py                  UPD  build_context() so any stage runs standalone from a worker
synscale/generation/runner.py         UPD  shard_index/shard_count → per-worker shard files/manifests
synscale/training/trainer.py          UPD  SIGTERM/SIGUSR1 handler → resume.pt, exit 75 (requeue)
scripts/fleet_local.sh                NEW  on-box: one GPU-pinned worker per GPU (RunPod + vast engine)
scripts/worker.py / worker.sh         NEW  one-job-per-process worker (+--build-only) and loop
run_all.sh / install.sh               UPD  --fleet mode; install keeps image CUDA torch + HF token
scripts/runpod/bootstrap.sh           NEW  pod-side: clone (PAT) or wait-for-upload, install, run fleet
scripts/runpod/preflight.sh           NEW  runpodctl auth check + reminders
scripts/runpod/launch.sh              NEW  create pod on a network volume; pod self-runs bootstrap
scripts/runpod/monitor.sh             NEW  pod state via runpodctl; where to read live progress
scripts/runpod/teardown.sh            NEW  remove pod (billing stops); network volume persists
scripts/vast/*.sh                     NEW  vast alternative (common/preflight/smoke/onstart/launch/monitor/teardown)
docs/26_fleet_orchestration.md        NEW  fleet design + "why not all spot"
docs/27_vast_runbook.md               NEW  vast.ai procedure (alternative)
docs/28_runpod_runbook.md             NEW  RunPod procedure + provider comparison + reliability rationale
README.md                             UPD  RunPod primary; provider cost/reliability table
.gitignore                            UPD  excludes session tooling (.agents/, .claude/, logs/, etc.)
```

Tooling installed in this environment: `runpodctl` 2.14.0, `vastai` 1.7.0 (both need the new
all-domains session to reach their APIs).

Earlier C4 files already committed in `1d2b5ef`: `analysis/novelty.py`, `analysis/data_properties.py`,
`analysis/scaling.py` (per-seed fits, `L_inf_hat`, multiplier), `docs/24_novel_experiment.md`,
students/teachers/training configs, `scripts/prepare_data.py` (16k BPE), tests.

**Commit trail on the branch:**
`1d2b5ef` Rework into C4 → `1349979` fleet + single-card int4 → `8aa78e2` vast toolkit + on-box
fleet → `d30eab0` vast egress requirement → `7e3cf34` HANDOFF → (this turn) RunPod provider switch:
NFS-safe lock, provider cost comparison, `scripts/runpod/` + `docs/28`.

---

## 10. Verified vs. unverified

**Verified (offline):** 68 unit tests pass; the whole pipeline runs end to end on CPU (`make micro`)
including the data index and novelty analyses; the fleet path drains its DAG in dependency order and
produces `index.csv` + `analysis/novelty_report.json` (via both `run_all.sh micro --fleet` and
`scripts/fleet_local.sh micro`); the cost planner runs; every vast script is syntax-clean and its
JSON parsers were unit-checked against fake payloads; the code tarball (`git archive HEAD`) includes
the scripts.

**Not yet verified (needs the GPU run):** the code has **never run on a real GPU** — that is exactly
what the smoke gate is for (spends a few dollars to catch any CUDA/vLLM issue before the H100
budget). Also unverified until an account is connected and reachable in the new session: the RunPod
`runpodctl` provisioning path (API blocked in the build session), whether the pod start-command runs
the bootstrap cleanly (console fallback covers it), real H100 availability/pricing, and real
teacher-decode throughput (all estimates ±a factor).

---

## 11. Immediate next step

1. Start a **fresh session** on `claude/admiring-fermi-btrw8q` (for the "All domains" policy).
2. Have ready: **funded RunPod balance + RunPod API key + a ~600GB network volume (console) + HF
   token (Llama access) + read-only GitHub PAT**.
3. Say go. Order of operations: confirm `api.runpod.io` reachable → `runpodctl config --apiKey` →
   `runpod/preflight.sh` → **smoke on one cheap GPU (must pass)** → create the network volume →
   `runpod/launch.sh c4 --volume <id> --gpus 8 --gh-token <pat> --hf <tok>` → `runpod/monitor.sh`
   (scheduled check-ins) → `runpod/teardown.sh` the moment `/workspace/DONE` appears.
   (Vast.ai remains a fully-automated cheaper alternative via `scripts/vast/` if preferred.)

Do **not** skip the smoke gate. It is the difference between "a few dollars to find a bug" and
"hundreds of dollars into a broken run."
