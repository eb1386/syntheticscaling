# 28. RunPod Runbook (recommended provider)

RunPod is the recommended provider for the main run. This document explains why (with current,
verified prices and a reliability read across all the providers you asked about), the exact
architecture, and the step-by-step procedure.

## Which provider, and why

You were right to be wary of Vast.ai. For a multi-day training run that needs a multi-GPU box,
persistent storage, and programmatic control, here is the honest picture (H100-80GB, Sep 2026):

| Provider | $/GPU-h | C4 (~1,230 GPU-h) | Reliability read |
|---|---|---|---|
| Vast.ai spot | ~$1.49 | ~$1,840 / ~$2,550 CAD | Cheapest. **No uptime SLA**, peer-to-peer, no shared volume. Least reliable. |
| TensorDock | ~$2.25 | ~$2,770 / ~$3,850 CAD | Curated marketplace, cheaper than RunPod. Less proven; storage model varies. |
| **RunPod Community** | **~$2.69** | **~$3,310 / ~$4,610 CAD** | **Recommended default.** RunPod marketplace **+ network volumes** (durable). Host-dependent uptime, but work is never lost. |
| **RunPod Secure** | **~$2.99** | **~$3,680 / ~$5,120 CAD** | **Most reliable tier.** RunPod-owned hardware, **99% SLA**, network volumes. |
| Modal | ~$3.95 | ~$4,870 / ~$6,760 CAD | Serverless, per-second, preempts by default. Paradigm mismatch for multi-day SSH training. |
| Lambda Cloud | ~$3.99 | ~$4,920 / ~$6,830 CAD | Best SLA (99.9%) **but** H100 SXM is 8-GPU-only, no spot, and **frequently sold out**. Capacity is the real risk. |

Numbers from `python -m synscale.analysis.config_planner`; ±a factor until the pilot measures real
throughput. Add ~25% for preemption re-runs.

**Why RunPod wins for this run:**

1. **Network volumes that survive a pod crash.** A network volume is persistent storage that lives
   independently of the pod. If a pod dies (RunPod's own reviews note occasional failures), the
   volume keeps every checkpoint, the queue, and all results; you relaunch a pod, remount the same
   volume, and the fleet resumes exactly where it left off. Vast.ai has no equivalent, so a dead
   host there means a lost run.
2. **The fleet already absorbs pod failures.** The file-locked queue requeues any job whose worker
   dies, and training checkpoints on SIGTERM. So RunPod's imperfect per-pod reliability is exactly
   the failure mode the architecture is built for, and the network volume makes recovery lossless.
3. **Real availability.** Lambda has the best SLA but you often cannot get H100 capacity at all
   (community reports of checking for months). RunPod generally has H100s available now.
4. **Reliability tiers you can choose.** Secure Cloud (RunPod-owned, 99% SLA) for maximum
   reliability, or Community Cloud (cheaper) — and because the network volume protects the work,
   Community is a sound default, with the one long 1B base run optionally pinned to Secure.
5. **Programmatic control.** `runpodctl` (installed) drives create/get/remove over the HTTPS API.

The cost of that reliability is real: RunPod runs roughly $2,000 CAD more than Vast spot for C4. If
you want to trade some reliability for cost, **TensorDock (~$2.25/GPU-h)** is the sensible
middle: cheaper than RunPod, more curated than Vast. Vast remains the cheapest if you accept the
risk. The code runs on any of them (the on-box fleet is provider-agnostic); only provisioning
differs.

## Architecture on RunPod

One multi-GPU pod, a network volume mounted at `/workspace`, one GPU-pinned fleet worker per card.

```
   RunPod pod (8x H100 SXM)                 Network volume (persistent, ~600 GB)
   ┌───────────────────────────┐            ┌──────────────────────────────────┐
   │ scripts/fleet_local.sh     │  mounts →  │ /workspace/synscale/  data+ckpts │
   │  worker GPU0 ... worker GPU7│            │ /workspace/.../queue  job queue  │
   │  (file-lock queue on volume)│           │ /workspace/{status,run.log,DONE} │
   └───────────────────────────┘            └──────────────────────────────────┘
        pod can die and be replaced;  the volume (and all work) persists
```

The queue's claim uses a hardlink lock (`synscale/fleet.py`), which is atomic even on RunPod's NFS
network volumes (they provide NFS **without** advisory `flock`, so plain `O_EXCL` is not enough;
hardlink creation is).

## What you provide

1. **Funded RunPod balance.** Budget ~**$4,600–5,100 CAD** for C4 on RunPod (Community→Secure),
   plus ~25% for re-runs, so ~$5,500–6,000 CAD to finish comfortably. Network storage is tiny
   (~$0.07/GB/mo → ~$10 for 600 GB over the run).
2. **RunPod API key** — https://www.runpod.io/console/user/settings → `runpodctl config --apiKey <KEY>`.
3. **A network volume** created in the console (Storage → Network Volumes), ~600 GB, in a region
   that has H100 SXM availability. Note its volume id.
4. **Hugging Face token** with Llama-3.x access (for C4's second family). Pass as `--hf`.
5. **A fine-grained GitHub PAT**, read-only, scoped to just `eb1386/syntheticscaling` (for the pod
   to clone). It is used only on the pod and dies with it. (Alternative: the manual-upload path
   below, no token.)

## The procedure

```bash
# in the NEW session (all-domains egress), one time:
runpodctl config --apiKey <RUNPOD_KEY>
./scripts/runpod/preflight.sh                      # verify auth; reminders

# create the network volume in the console first (get its id), then:
./scripts/runpod/launch.sh c4 --volume <VOL_ID> --gpus 8 --gh-token <GH_PAT> --hf <HF_TOKEN>
#   add --secure for RunPod Secure Cloud (more reliable, pricier)

./scripts/runpod/monitor.sh                        # pod state + where to read live progress
./scripts/runpod/teardown.sh                       # removes the pod; the volume + results persist
```

**Validate cheaply first (same as any provider): run the smoke profile on one cheap GPU before
committing the H100 budget.** The quickest smoke on RunPod is a single-GPU pod with
`PROFILE=smoke`; or run the Vast smoke gate (`scripts/vast/smoke.sh`) which is already fully
automated and costs ~$1–5. The point is to prove the code on real CUDA once; the result transfers.

### Console fallback (most reliable, no CLI provisioning)

RunPod's remote exec is SSH-based, so if `launch.sh`'s scripted pod-create + start-command does not
run cleanly, use the console, which is bulletproof:

1. Console → Deploy → pick H100 SXM, GPU count 8, attach your network volume at `/workspace`, image
   `runpod/pytorch:2.4.0-...`, enable SSH.
2. Set the pod's **Container Start Command** to the one `launch.sh` prints (it pulls
   `scripts/runpod/bootstrap.sh` from the repo with your PAT and runs it), and add env vars
   `GH_TOKEN`, `HF_TOKEN`, `PROFILE=c4`.
3. Deploy. The pod self-installs and runs the fleet; watch the Logs tab, or the files under
   `/workspace/` on the volume.

### Manual-upload path (no GitHub token)

If you prefer not to mint a PAT: deploy the pod without `GH_TOKEN`; the bootstrap then waits for
`/workspace/repo.tgz` + a `GO` marker. Upload the code with `runpodctl send` (or over SSH) and touch
`GO`. Details in the bootstrap script header.

## Monitoring and teardown

- `./scripts/runpod/monitor.sh` shows pod state and cost from the API. Live progress
  (`/workspace/status.txt`, `run.log`, `DONE`) is on the volume — read it from the console Logs tab
  or over SSH (`runpodctl ssh info <pod>`).
- The assistant schedules check-ins; because RunPod exec is SSH-based, expect to occasionally paste
  a console log snippet if SSH egress is not available from the assistant's session.
- **Teardown removes the pod (billing stops); the network volume persists** so results are safe.
  Pull results (attach the volume to a cheap pod and copy `/workspace/index.csv` + `analysis/`),
  then delete the volume in the console when you are done with it.

## Money safety

- Balance is a hard cap: if credit runs out, RunPod stops the pod; the volume persists; resume by
  relaunching. `launch.sh` also sets a `--cost` per-hour ceiling so a pricier host is never taken.
- Always run the smoke gate before the H100 budget.
- Tear the pod down the moment `/workspace/DONE` appears. The network volume is cheap to keep only
  as long as you need the results off it.
