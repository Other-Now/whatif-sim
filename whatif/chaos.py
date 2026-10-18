"""Kill workers for real, mid-batch, and check the answer doesn't change.

Runs the same model + config twice in Postgres:
  1. reference: one worker, never killed;
  2. chaos: N workers, and every ``kill_every`` seconds a random worker is
     hard-killed (TerminateProcess / SIGKILL) and replaced.
Then compares the two summaries' SHA-256 over every sample of every output.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

from . import store
from .mc import RunConfig
from .model import ModelSpec


def spawn(rid: int, lease_s: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "whatif", "worker", str(rid), "--lease", str(lease_s)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parent.parent,
    )


def run_workers(rid: int, workers: int, lease_s: float, kill_every: float = 0.0, max_kills: int = 0,
                seed: int = 1, timeout_s: float = 900) -> dict:
    rng = random.Random(seed)
    procs = [spawn(rid, lease_s) for _ in range(workers)]
    kills = respawns = 0
    t0 = time.monotonic()
    next_kill = t0 + kill_every if kill_every else float("inf")
    conn = store.connect()
    while True:
        status = conn.execute("select status from runs where id=%s", (rid,)).fetchone()[0]
        if status in ("converged", "max_batches"):
            break
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"run {rid} did not finish")
        now = time.monotonic()
        if now >= next_kill and kills < max_kills:
            victim = rng.randrange(len(procs))
            procs[victim].kill()
            procs[victim].wait()
            procs[victim] = spawn(rid, lease_s)
            kills += 1
            next_kill = now + kill_every * rng.uniform(0.5, 1.5)
        for i, p in enumerate(procs):          # a worker that exited early (idle) gets replaced
            if p.poll() is not None and status in ("running", "finishing", "summarizing"):
                respawns += 1
                if respawns > 50:
                    raise RuntimeError("workers keep exiting on their own; run `python -m whatif worker` to see why")
                procs[i] = spawn(rid, lease_s)
        time.sleep(0.02)
    wall = time.monotonic() - t0
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    st = store.run_status(conn, rid)
    conn.close()
    return {"run": rid, "wall_s": round(wall, 2), "kills": kills, **{k: st[k] for k in
            ("status", "stop_batch", "retried_claims", "fenced_writes", "wasted_batches")},
            "sha256": st["summary"]["sha256"], "samples": st["summary"]["samples"]}


def chaos(spec: ModelSpec, cfg: RunConfig, workers: int = 4, kills: int = 20, kill_every: float = 0.4,
          lease_s: float = 2.0) -> dict:
    conn = store.connect()
    store.init_schema(conn)
    ref_id = store.create_run(conn, spec, cfg)
    chaos_id = store.create_run(conn, spec, cfg)
    conn.close()
    ref = run_workers(ref_id, 1, lease_s)
    ch = run_workers(chaos_id, workers, lease_s, kill_every=kill_every, max_kills=kills)
    return {"reference": ref, "chaos": ch, "identical": ref["sha256"] == ch["sha256"]}


if __name__ == "__main__":
    print(json.dumps(chaos(ModelSpec.model_validate(json.load(open(sys.argv[1]))),
                           RunConfig(scenario=sys.argv[2] if len(sys.argv) > 2 else None)), indent=2))
