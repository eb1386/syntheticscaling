# 26. Fleet Orchestration: the ultra-optimized spot path

This document explains how the C4 study is run cheaply and quickly on a fleet of preemptible
(spot) GPU pods instead of one reliable, expensive machine. It covers the four design decisions
that make all-spot viable, how the job queue works, how to launch a fleet, and the honest
cost picture including the "why not all spot" question.

## The one decision everything hangs on: single-card int4 teachers

The most expensive and most preemption-sensitive part of a synthetic-data study is serving the
largest teachers. A 70B or 72B model in bf16 needs about 140 GB of weights, which forces a
two-GPU tensor-parallel pod. Two-GPU spot pods are the worst thing you can rent: you pay for two
cards, and a preemption of *either* card kills the whole server, so the effective preemption rate
roughly doubles.

C4 serves **every** teacher, up to 72B, single-card at int4 (AWQ, `awq_marlin` kernel). A 70B
model at int4 is about 35 GB of weights, which fits one 80 GB H100 with room for the KV cache.
The consequences cascade:

* No teacher needs tensor-parallel=2, so there are **no two-GPU jobs at all**.
* Every job in the study fits on a single spot GPU, so every job is spot-eligible.
* Int4 decode is memory-bandwidth-bound and actually *faster* than bf16 for these large models,
  because it moves a quarter of the bytes per token.

Quality cost is small and measured, not assumed: AWQ int4 on instruct models loses a fraction of
a point on standard benchmarks, and the study's outcome is the *student's* held-out loss, which
is far downstream of any small teacher-perplexity change. The pilot confirms the teacher decode
quality before the main run commits budget.

## The other three decisions

2. **A file-locked job queue on the shared volume.** There is no coordinator server. The queue is
   a single JSON file plus one lock file per job on the persistent volume that every pod mounts.
   Workers claim jobs atomically with `os.open(..., O_CREAT | O_EXCL)`, which is the one filesystem
   primitive guaranteed to have exactly one winner. This means N pods coordinate with zero
   infrastructure, and adding a pod is just starting another worker.

3. **Generation sharding.** The 72B and 70B generation jobs are the longest single pieces of work
   (order-of-hundreds of GPU-hours each). Each is split into `gen_shards` contiguous slices of the
   prompt pool, each an independent queue job writing its own shard file and manifest. Ten pods can
   therefore chew through one giant teacher's corpus in parallel, so the critical path is a few
   hours instead of days, and a preempted shard re-runs alone.

4. **Preemption-safe training.** Spot providers send SIGTERM (typically with a short grace period)
   before reclaiming a pod. The trainer installs a handler that, on the next safe step boundary,
   writes `resume.pt` to the persistent volume and exits with code 75 (`EX_TEMPFAIL`). The worker
   sees 75, requeues the job, and another pod picks it up from the checkpoint. A reclaim costs
   minutes of recompute, not the whole run.

## How the queue works

`synscale/fleet.py` is the whole mechanism, and it is small on purpose.

* `build_queue(profile, gen_shards=8)` walks the profile and emits the DAG as a list of `Job`s:
  `prepare -> pool -> {gen_shard...} -> {base...} -> {branch...} -> finalize`. Dependencies are job
  ids; a job becomes runnable only when all its deps are `done`. Each big generation job expands
  into `gen_shards` shard jobs. Only the single longest base run (the 1B student) is tagged
  `tier="ondemand"`; everything else is `tier="spot"`.
* `Queue` stores state in `<queue_dir>/queue.json` and locks in `<queue_dir>/locks/`. Key methods:
  `claim_next(worker, tier)` returns the next runnable job whose tier the worker may run, taking an
  exclusive lock; `heartbeat(job_id)` touches the lock so long jobs are not seen as dead;
  `complete(job_id, ok)` marks state and releases the lock; `requeue_stale()` reclaims any job whose
  lock has not been touched within `CLAIM_TTL` (one hour), which is how a silently-killed pod's job
  returns to the pool.
* A worker loops: `requeue_stale()`, `claim_next()`, run, `complete()`. If nothing is runnable and
  nothing is in flight, the queue is drained and the worker exits.

Tiers are a soft preference, not a wall: an on-demand pod runs spot jobs too when no on-demand job
is waiting, so the reliable pod is never idle.

## Launching a fleet

One command on the pod that holds the persistent volume:

```bash
./install.sh c4                      # once, onto the persistent volume
./run_all.sh c4 --fleet              # builds the queue and drains it with one local worker
```

Then start as many additional spot pods as you want throughput, each mounting the same volume:

```bash
# on every extra spot pod, after install.sh:
./scripts/worker.sh c4 spot          # loops: claim, run, repeat, until the queue drains
```

Optionally keep one on-demand pod as an availability anchor and to own the 1B base run:

```bash
./scripts/worker.sh c4 ondemand      # runs the 1B base, then helps with spot jobs
```

`scripts/worker.py --one` (what `worker.sh` calls in a loop) claims and runs exactly one job then
exits, so VRAM is fully released between jobs of different sizes. Set `QUEUE_DIR` and
`SYNSCALE_STORE` to paths on the shared volume (the defaults point at `/workspace/synscale`).

Recommended shape for C4: 6 to 8 spot pods plus one on-demand anchor finishes the study in about a
week. Fewer pods cost the same total dollars and just take longer.

## Cost, and "why not all spot"

Numbers come from `python -m synscale.analysis.config_planner` (transparent FLOP-and-bandwidth
estimates, plus or minus a factor until the pilot measures real throughput). For C4:

| Plan | GPU-hours | USD | CAD |
|---|---|---|---|
| All spot (~$1.49/hr) | ~1,230 | ~$1,840 | ~$2,550 |
| Hybrid (1B base on-demand, rest spot) | ~1,230 | ~$1,910 | ~$2,660 |
| All on-demand (~$2.50/hr) | ~1,230 | ~$3,080 | ~$4,280 |

Add roughly 25 percent for preemption re-runs, so budget about **$2,500 to $3,300 CAD**.

The honest answer to "why not do it all on spot H100": you basically can, and it is the cheapest
plan. Because single-card int4 made every job spot-safe and the trainer checkpoints on SIGTERM,
there is no job that *needs* an on-demand pod. All-spot and hybrid differ by under five percent,
because the only job long enough to be worth protecting is the single 1B base run (about 76
GPU-hours); a preemption late in that one run is the only place spot pricing meaningfully hurts,
and even there the checkpoint caps the loss.

So the on-demand pod is optional insurance, not a requirement, and it buys two specific things:

1. **Availability, not just price.** Spot H100 capacity comes and goes. One on-demand anchor pod
   guarantees the study always makes forward progress even when spot inventory dries up for a
   while, so wall-clock does not stall.
2. **A clean home for the one long serial job.** The 1B base training is the one piece that does
   not shard and is long enough that repeated late preemptions would waste real time. Pinning it to
   the anchor removes that tail risk for a few dollars.

If you are optimizing purely for dollars and can tolerate a variable finish time, run all spot. If
you want a predictable finish and near-zero babysitting, run the hybrid: one on-demand anchor plus
a swarm of spot workers. The code supports both with the same queue; the only difference is whether
you start a `worker.sh c4 ondemand` pod.
