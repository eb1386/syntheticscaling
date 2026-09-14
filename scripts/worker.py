#!/usr/bin/env python3
"""Fleet worker: claim and run jobs from the shared queue on the persistent volume.

One job per process (default) so VRAM is freed between jobs; scripts/worker.sh restarts it in a
loop until the queue drains. Run many of these across cheap spot pods to parallelise the study.

    python scripts/worker.py --profile c4 --tier spot --queue-dir /workspace/synscale/results/c4/queue

Exit codes: 0 = ran a job or none currently runnable (retry); 3 = queue drained (stop the loop).
"""
from __future__ import annotations
import argparse, socket, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synscale.config.profiles import get_profile
from synscale.fleet import Queue, build_queue, run_one_job


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="c4")
    ap.add_argument("--tier", default="spot", choices=["spot", "ondemand"])
    ap.add_argument("--queue-dir", default=None, help="shared queue dir (default results/<profile>/queue)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--gen-shards", type=int, default=8, help="split each big generation job this many ways")
    ap.add_argument("--loop", action="store_true", help="drain the queue in-process instead of one job")
    ap.add_argument("--build-only", action="store_true", help="create the queue.json (if absent) and exit")
    a = ap.parse_args()
    profile = get_profile(a.profile)
    qdir = a.queue_dir or f"results/{profile.name}/queue"
    q = Queue(qdir)
    if not q.qpath.exists():
        q.init(build_queue(profile, gen_shards=a.gen_shards))
    if a.build_only:
        print(f"[worker] queue ready at {q.qpath}: {q.summary()}")
        return
    me = f"{socket.gethostname()}:{os.getpid()}"

    def disp(job):
        run_one_job(a.profile, job, data_dir=a.data_dir or profile.persistent_dir,
                    store_dir=a.store_dir, backend_kind=a.backend, device=a.device)

    if a.loop:
        from synscale.fleet import worker
        print(worker(a.profile, queue_dir=qdir, tier=a.tier, dispatch=disp))
        return
    q.requeue_stale()
    job = q.claim_next(me, tier=a.tier)
    if job is None:
        s = q.summary()
        drained = s.get("pending", 0) == 0 and s.get("claimed", 0) == 0
        print(f"[worker] no runnable job; {s}")
        sys.exit(3 if drained else 0)
    print(f"[worker] running {job.id} ({job.kind})")
    try:
        disp(job); q.complete(job.id, ok=True); print(f"[worker] done {job.id}")
    except SystemExit as e:
        if getattr(e, "code", 1) == 75:   # preemption: requeue and exit 0 to retry
            jobs = q.load(); jobs[job.id].state = "pending"; jobs[job.id].claimed_by = None
            q.save(jobs); q._unlock(job.id); print(f"[worker] preempted {job.id}, requeued"); sys.exit(0)
        q.complete(job.id, ok=False); raise
    except Exception as ex:
        q.complete(job.id, ok=False); print(f"[worker] FAILED {job.id}: {ex}"); sys.exit(1)


if __name__ == "__main__":
    main()
