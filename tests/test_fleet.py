"""Fleet queue coordination: dependency order, atomic claiming, stale requeue, drain."""
import time
from synscale.fleet import Queue, Job, worker


def test_dependency_order_and_drain(tmp_path):
    q = Queue(tmp_path / "q")
    q.init([
        Job("a", "x", deps=[]),
        Job("b", "x", deps=["a"]),
        Job("c", "x", deps=["a"]),
        Job("d", "x", deps=["b", "c"]),
    ])
    order = []
    # single worker via claim/complete loop
    while True:
        j = q.claim_next("w1")
        if j is None:
            if q.summary().get("pending", 0) == 0:
                break
            continue
        order.append(j.id)
        q.complete(j.id, ok=True)
    assert order[0] == "a"                      # a before its dependents
    assert order.index("d") > order.index("b") and order.index("d") > order.index("c")
    assert q.summary().get("done") == 4


def test_atomic_claim_no_double(tmp_path):
    q = Queue(tmp_path / "q")
    q.init([Job("only", "x", deps=[])])
    j1 = q.claim_next("w1")
    j2 = q.claim_next("w2")     # second worker must not get the same job
    assert j1 is not None and j2 is None


def test_stale_claim_requeues(tmp_path):
    import synscale.fleet as fleet
    old = fleet.CLAIM_TTL
    fleet.CLAIM_TTL = 0.2
    try:
        q = Queue(tmp_path / "q")
        q.init([Job("j", "x", deps=[])])
        j = q.claim_next("w1")            # w1 claims then "dies"
        assert j is not None
        time.sleep(0.3)
        n = q.requeue_stale()
        assert n == 1 and q.summary().get("pending") == 1
        j2 = q.claim_next("w2")           # another worker can now take it
        assert j2 is not None
    finally:
        fleet.CLAIM_TTL = old


def test_worker_runs_with_fake_dispatch(tmp_path):
    q = Queue(tmp_path / "q")
    q.init([Job("a", "x", deps=[]), Job("b", "x", deps=["a"])])
    seen = []
    # patch build_queue path: worker() re-inits only if queue absent; ours exists, so it runs.
    from synscale import fleet
    def fake(job):
        seen.append(job.id)
    # call worker with a profile that exists; dispatch overrides the default
    s = fleet.worker("micro", queue_dir=tmp_path / "q", dispatch=fake, poll_seconds=0.01)
    assert seen == ["a", "b"] and s.get("done") == 2
