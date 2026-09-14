"""Fleet orchestration for a cheap, spot-safe, parallel run (docs/26).

The whole study is a set of jobs with dependencies. On a persistent shared volume we keep a
queue; any number of spot pods run `worker()`, each of which atomically CLAIMS the next runnable
job (dependencies satisfied, not already claimed), runs it, and marks it done. Because claiming
is a filesystem lock on the shared volume, N pods coordinate with no server. Long jobs (the 72B
and 70B generation) are split into shard-range jobs, so ten spot pods finish them ten times
faster for the same dollars. Preempted jobs are resumable (generation skips finished shards,
training resumes from a checkpoint) and their stale claim is requeued after a timeout.

Design goals:
  * spot-safe: every job is single-card and resumable; a reclaim loses at most one shard / one
    checkpoint interval.
  * cost-neutral parallelism: more pods cut wall-clock, not total GPU-hours.
  * no external services: coordination is atomic file operations on the shared volume.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

CLAIM_TTL = 3600.0        # a claim older than this (no heartbeat) is considered dead and requeued


@dataclass
class Job:
    id: str
    kind: str                       # prepare | pool | gen_shard | base | branch | finalize
    tier: str = "spot"              # spot | ondemand (ondemand only for long critical jobs)
    deps: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)
    est_hours: float = 0.0
    state: str = "pending"          # pending | claimed | done | failed
    claimed_by: Optional[str] = None
    claimed_at: float = 0.0
    attempts: int = 0


class Queue:
    """A queue on the shared volume: one JSON file plus per-job claim lock files."""

    def __init__(self, root: str | Path):
        self.dir = Path(root); self.dir.mkdir(parents=True, exist_ok=True)
        self.qpath = self.dir / "queue.json"
        self.locks = self.dir / "locks"; self.locks.mkdir(exist_ok=True)

    # ---- persistence -------------------------------------------------------------------
    def load(self) -> dict[str, Job]:
        if not self.qpath.exists():
            return {}
        raw = json.loads(self.qpath.read_text())
        return {jid: Job(**j) for jid, j in raw.items()}

    def save(self, jobs: dict[str, Job]) -> None:
        tmp = self.qpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({jid: asdict(j) for jid, j in jobs.items()}, indent=2))
        os.replace(tmp, self.qpath)

    def init(self, jobs: list[Job]) -> None:
        if self.qpath.exists():
            return  # never clobber an in-progress run
        self.save({j.id: j for j in jobs})

    # ---- claiming (atomic via O_EXCL lock files) ---------------------------------------
    def _try_lock(self, jid: str, worker: str) -> bool:
        # Hardlink-based claim: create a unique temp file, then hardlink it to the lock name. Link
        # creation is atomic even on NFS *without* advisory locking (RunPod network volumes), so this
        # is safe both on a single box's local disk and across pods sharing a network volume. O_EXCL
        # alone is not reliably atomic on all NFS setups; hardlink is.
        import uuid
        lock = self.locks / f"{jid}.lock"
        tmp = self.locks / f".{jid}.{uuid.uuid4().hex}.tmp"
        def _link_now() -> bool:
            try:
                with open(tmp, "w") as f:
                    f.write(worker)
                os.link(tmp, lock)     # raises FileExistsError if the lock is already held
                return True
            except FileExistsError:
                return False
            finally:
                try: os.unlink(tmp)
                except FileNotFoundError: pass
        if _link_now():
            return True
        # held: steal only if the holder is stale (no heartbeat within TTL)
        try:
            if time.time() - lock.stat().st_mtime > CLAIM_TTL:
                lock.unlink(missing_ok=True)
                return _link_now()
        except FileNotFoundError:
            return _link_now()
        return False

    def _unlock(self, jid: str) -> None:
        (self.locks / f"{jid}.lock").unlink(missing_ok=True)

    def heartbeat(self, jid: str) -> None:
        lock = self.locks / f"{jid}.lock"
        if lock.exists():
            os.utime(lock, None)

    def claim_next(self, worker: str, tier: Optional[str] = None) -> Optional[Job]:
        jobs = self.load()
        done = {jid for jid, j in jobs.items() if j.state == "done"}
        for jid, j in jobs.items():
            if j.state not in ("pending",):
                continue
            if tier and j.tier != tier and not (tier == "ondemand"):  # ondemand pods may run spot jobs too
                continue
            if any(d not in done for d in j.deps):
                continue
            if not self._try_lock(jid, worker):
                continue
            # re-read and commit the claim
            jobs = self.load()
            j = jobs[jid]
            if j.state != "pending":
                self._unlock(jid); continue
            j.state = "claimed"; j.claimed_by = worker; j.claimed_at = time.time(); j.attempts += 1
            self.save(jobs)
            return j
        return None

    def complete(self, jid: str, ok: bool = True) -> None:
        jobs = self.load()
        if jid in jobs:
            jobs[jid].state = "done" if ok else "failed"
            self.save(jobs)
        self._unlock(jid)

    def requeue_stale(self) -> int:
        """Return claimed jobs whose worker died (lock older than TTL) to pending."""
        jobs = self.load(); n = 0
        for jid, j in jobs.items():
            if j.state == "claimed":
                lock = self.locks / f"{jid}.lock"
                if (not lock.exists()) or (time.time() - lock.stat().st_mtime > CLAIM_TTL):
                    j.state = "pending"; j.claimed_by = None; self._unlock(jid); n += 1
        if n:
            self.save(jobs)
        return n

    def summary(self) -> dict[str, int]:
        jobs = self.load()
        out: dict[str, int] = {}
        for j in jobs.values():
            out[j.state] = out.get(j.state, 0) + 1
        return out


def build_queue(profile, *, gen_shards: int = 8) -> list[Job]:
    """Build the job DAG for a profile. Long teacher generation is split into `gen_shards`
    shard-range jobs so many pods share it. Tiers: the 1B base is 'ondemand' (long critical
    path, small); everything else is 'spot'."""
    from synscale.pipeline import enumerate_branches
    from synscale.tracking.ids import experiment_id
    jobs: list[Job] = []
    jobs.append(Job("prepare", "prepare", tier="spot", deps=[]))
    jobs.append(Job("pool", "pool", tier="spot", deps=["prepare"]))
    # generation: shard-range jobs per teacher
    gen_ids = []
    for t in profile.teachers:
        big = t.nominal_params >= 30e9
        shards = gen_shards if big else max(2, gen_shards // 2)
        for k in range(shards):
            jid = f"gen:{t.name}:{k}"
            jobs.append(Job(jid, "gen_shard", tier="spot", deps=["pool"],
                            args={"teacher": t.name, "shard_index": k, "shard_count": shards},
                            est_hours=(256.0 / shards if big else 2.0)))
            gen_ids.append((t.name, jid))
    # base runs per student/seed
    base_ids = {}
    for s in profile.students:
        for b in range(1, profile.base_seeds[s] + 1):
            jid = f"base:{s}:{b}"
            tier = "ondemand" if s == "s1b" else "spot"
            jobs.append(Job(jid, "base", tier=tier, deps=["prepare"], args={"student": s, "seed_base": b}))
            if b == 1:
                base_ids[s] = jid
    # branches depend on their teacher's generation shards and their student base
    for j in enumerate_branches(profile):
        tdeps = [jid for (tn, jid) in gen_ids if tn == j.cond] if j.arm == "synthetic" else []
        jid = f"branch:{j.student}:{j.cond}:{j.d_syn}:{j.seed_phase}"
        jobs.append(Job(jid, "branch", tier="spot", deps=[base_ids[j.student]] + tdeps,
                        args={"student": j.student, "arm": j.arm, "teacher": j.teacher.name if j.teacher else None,
                              "seed_phase": j.seed_phase, "d_syn": j.d_syn, "variant": j.variant, "cond": j.cond}))
    branch_ids = [j.id for j in jobs if j.kind == "branch"]
    jobs.append(Job("finalize", "finalize", tier="spot", deps=branch_ids))
    return jobs


# ==========================================================================================
# Worker: claim -> run -> mark done, until the queue drains
# ==========================================================================================
def worker(profile_name: str, *, queue_dir: str | Path, tier: str = "spot",
           dispatch: Optional[Callable[[Job], None]] = None, max_jobs: Optional[int] = None,
           poll_seconds: float = 20.0) -> dict[str, int]:
    """Run jobs of `tier` (an ondemand pod also runs spot jobs) until none remain runnable.

    `dispatch(job)` executes the job; the default dispatches to the pipeline stage functions.
    Any exception marks the job failed (so it is visible), except SystemExit(75) from a
    preemption, which requeues the job.
    """
    from synscale.config.profiles import get_profile
    profile = get_profile(profile_name)
    q = Queue(queue_dir)
    if not q.qpath.exists():
        q.init(build_queue(profile))
    dispatch = dispatch or (lambda job: _default_dispatch(profile, job, Path(queue_dir).parent))
    me = f"{socket.gethostname()}:{os.getpid()}"
    ran = 0
    while True:
        q.requeue_stale()
        job = q.claim_next(me, tier=tier)
        if job is None:
            s = q.summary()
            if s.get("pending", 0) == 0 and s.get("claimed", 0) == 0:
                break                      # queue drained
            time.sleep(poll_seconds); continue
        try:
            dispatch(job)
            q.complete(job.id, ok=True)
        except SystemExit as e:
            if getattr(e, "code", 1) == 75:   # preemption -> requeue
                jobs = q.load(); jobs[job.id].state = "pending"; jobs[job.id].claimed_by = None
                q.save(jobs); q._unlock(job.id)
                break
            q.complete(job.id, ok=False); raise
        except Exception as ex:
            q.complete(job.id, ok=False)
            print(f"[worker] job {job.id} FAILED: {ex}")
        ran += 1
        if max_jobs and ran >= max_jobs:
            break
    return q.summary()


def run_one_job(profile_name: str, job: Job, *, data_dir="data", results_dir=None, store_dir=None,
                backend_kind="auto", device="auto") -> None:
    """Execute a single claimed job via the pipeline stage functions. One job per process is the
    intended use (scripts/worker.sh restarts between jobs) so VRAM is freed every time."""
    import os as _os
    from pathlib import Path as _P
    from synscale import pipeline as pl
    from synscale.config.profiles import get_profile
    profile = get_profile(profile_name)
    ctx = pl.build_context(profile, data_dir=_P(data_dir), results_dir=results_dir, store_dir=store_dir,
                           backend_kind=backend_kind, device=device)
    k = job.kind
    if k == "prepare":
        pl.prepare_data(profile, ctx["data_dir"], ctx["tokenizer"], ctx["bos_id"], ctx["eos_id"],
                        synthetic=ctx["is_toy"])
    elif k == "pool":
        pl.build_pool_stage(profile, ctx["data_dir"])
    elif k == "gen_shard":
        t = next(tt for tt in profile.teachers if tt.name == job.args["teacher"])
        pl.generate_stage(profile, t, ctx["pool_path"], ctx["counter"], ctx["data_dir"],
                          backend_kind=ctx["backend_kind"], device=ctx["device"], sampling=ctx["sampling"],
                          shard_index=job.args["shard_index"], shard_count=job.args["shard_count"])
    elif k == "base":
        cfg, ckpt = pl.pretrain_base_stage(profile, job.args["student"], job.args["seed_base"], ctx["corpora"],
                                           exp_dir=ctx["exp_dir"], results_dir=ctx["results_dir"], store_dir=ctx["store_dir"],
                                           device=ctx["device"], micro_batch_seqs=ctx["micro_batch_seqs"], peak_tflops=ctx["peak"])
        if job.args["seed_base"] == 1:
            pl.evaluate_stage(cfg, ckpt, ctx["tokenizer"], ctx["bos_id"], cond="C0", results_dir=ctx["results_dir"],
                              use_fixtures=ctx["is_toy"], device=ctx["device"], tasks=ctx["tasks"], nll_stems={}, limit=ctx["eval_limit"])
    elif k == "branch":
        from synscale.pipeline import BranchJob
        a = job.args
        t = next((tt for tt in profile.teachers if tt.name == a["teacher"]), None) if a.get("teacher") else None
        bj = BranchJob(a["student"], a["arm"], t, 1, a["seed_phase"], a["d_syn"], a["variant"], a["cond"])
        gen_dirs = {tt.name: str(ctx["data_dir"] / "processed" / profile.name / "gen" / tt.name) for tt in profile.teachers}
        base_ckpt = ctx["store_dir"] / f"base_{a['student']}_b1.pt"
        cfg, ckpt, cond = pl.branch_stage(profile, bj, corpora=ctx["corpora"], gen_dirs=gen_dirs, union_drop=set(),
                                          tokenizer=ctx["tokenizer"], counter=ctx["counter"], bos_id=ctx["bos_id"], eos_id=ctx["eos_id"],
                                          base_ckpt=base_ckpt, exp_dir=ctx["exp_dir"], results_dir=ctx["results_dir"],
                                          store_dir=ctx["store_dir"], proc_dir=ctx["data_dir"] / "processed" / profile.name,
                                          device=ctx["device"], micro_batch_seqs=ctx["micro_batch_seqs"], peak_tflops=ctx["peak"])
        pl.evaluate_stage(cfg, ckpt, ctx["tokenizer"], ctx["bos_id"], cond=cond, results_dir=ctx["results_dir"],
                          use_fixtures=ctx["is_toy"], device=ctx["device"], tasks=ctx["tasks"], nll_stems={}, limit=ctx["eval_limit"])
    elif k == "finalize":
        from synscale.tracking.index import build_index
        from synscale.analysis.run import analyze_index
        idx = build_index(ctx["results_dir"], out_path=ctx["results_dir"] / "index.csv")
        try:
            analyze_index(idx, out_dir=ctx["results_dir"] / "analysis")
        except Exception:
            pass
    else:
        raise ValueError(f"unknown job kind {k}")


def _default_dispatch(profile, job: Job, data_root: Path) -> None:
    run_one_job(profile.name, job, data_dir=str(data_root))
