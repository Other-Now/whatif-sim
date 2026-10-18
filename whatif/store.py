"""Postgres-backed Monte Carlo runs: checkpoint per batch, resume after a kill.

A run is a row in ``runs`` plus ``max_batches`` rows in ``batches``. Workers
claim the lowest-numbered claimable batch with ``FOR UPDATE SKIP LOCKED``,
compute it, and write the result in one UPDATE. Killing a worker at any point
loses at most the batch it was holding: its lease expires and someone else
claims it, and since a batch is a pure function of its index the recomputed
result is bit-identical.

A claim is dead when its lease has expired OR the claiming worker's Postgres
session is gone (``backend_pid`` no longer in ``pg_stat_activity``). The second
check matters: a hard-killed process drops its connection immediately, so its
batch is reclaimed in milliseconds instead of after the lease. Before it, the
unfinished batch blocked the finished prefix for a whole lease, the stopping
rule couldn't fire, and live workers computed 77-103 batches past the stop
point per run; with it, 1 (results/chaos.json, 5 runs each).

``attempt`` is a fencing token. A worker that stalls past its lease and then
wakes up can't overwrite the batch, because its completion UPDATE requires
the attempt number it claimed with.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Optional

import numpy as np
import psycopg

from .mc import Batch, RunConfig, convergence, first_converged, run_batch, summarize
from .model import ModelSpec
from .sim import Compiled

SCHEMA = """
create table if not exists runs (
    id           bigserial primary key,
    created_at   timestamptz not null default now(),
    finished_at  timestamptz,
    model        jsonb not null,
    config       jsonb not null,
    status       text not null default 'running',   -- running | finishing | summarizing | converged | max_batches
    stop_batch   int,
    checked_upto int not null default 0,                -- prefixes shorter than this didn't converge
    finalizer_pid int,                                   -- backend of the worker building the summary
    finalize_until timestamptz,
    summary      jsonb
);
create table if not exists batches (
    run_id      bigint not null references runs(id) on delete cascade,
    idx         int not null,
    status      text not null default 'pending',     -- pending | claimed | done
    attempt     int not null default 0,
    worker      text,
    backend_pid int,
    lease_until timestamptz,
    stats       jsonb,
    base        bytea,
    scen        bytea,
    steps       int,
    elapsed_ms  real,
    finished_at timestamptz,
    primary key (run_id, idx)
);
create table if not exists fenced (
    run_id  bigint not null,
    idx     int not null,
    attempt int not null,
    worker  text,
    at      timestamptz not null default now()
);
"""


def dsn() -> str:
    return os.environ.get("WHATIF_DB", "postgresql://whatif:whatif@localhost:5433/whatif")


def connect(url: Optional[str] = None) -> psycopg.Connection:
    return psycopg.connect(url or dsn(), autocommit=True)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def _pack(a: Optional[np.ndarray]) -> Optional[bytes]:
    if a is None:
        return None
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def _unpack(b: Optional[bytes]) -> Optional[np.ndarray]:
    return None if b is None else np.load(io.BytesIO(b), allow_pickle=False)


def create_run(conn, spec: ModelSpec, cfg: RunConfig) -> int:
    with conn.transaction():
        rid = conn.execute(
            "insert into runs (model, config) values (%s, %s) returning id",
            (json.dumps(spec.dump()), json.dumps(cfg.to_dict())),
        ).fetchone()[0]
        conn.execute(
            "insert into batches (run_id, idx) select %s, g from generate_series(0, %s) g",
            (rid, cfg.max_batches - 1),
        )
    return rid


def load_run(conn, rid: int) -> tuple[ModelSpec, RunConfig, str]:
    m, c, status = conn.execute("select model, config, status from runs where id=%s", (rid,)).fetchone()
    return ModelSpec.model_validate(m), RunConfig(**c), status


# Ablation switch for the benchmark in results/: WHATIF_SESSION_CHECK=0 falls
# back to lease expiry only.
_SESSION_CHECK = os.environ.get("WHATIF_SESSION_CHECK", "1") != "0"
_DEAD = "or backend_pid not in (select pid from pg_stat_activity)" if _SESSION_CHECK else ""


def claim(conn, rid: int, worker: str, lease_s: float) -> Optional[tuple[int, int]]:
    row = conn.execute(
        f"""
        update batches b set status='claimed', attempt=b.attempt+1, worker=%(w)s,
               backend_pid = pg_backend_pid(),
               lease_until = now() + make_interval(secs => %(lease)s)
        where (b.run_id, b.idx) = (
            select run_id, idx from batches
            where run_id=%(r)s
              and (status='pending'
                   or (status='claimed' and (lease_until < now() {_DEAD})))
              and idx < coalesce((select stop_batch from runs where id=%(r)s), 2147483647)
            order by idx limit 1
            for update skip locked)
        returning b.idx, b.attempt
        """,
        {"r": rid, "w": worker, "lease": lease_s},
    ).fetchone()
    return None if row is None else (row[0], row[1])


def complete(conn, rid: int, b: Batch, attempt: int, worker: str, elapsed_ms: float) -> bool:
    cur = conn.execute(
        """
        update batches set status='done', stats=%s, base=%s, scen=%s, steps=%s,
               elapsed_ms=%s, finished_at=now(), lease_until=null
        where run_id=%s and idx=%s and attempt=%s and status='claimed'
        """,
        (json.dumps(b.stats), _pack(b.base), _pack(b.scen), b.steps, elapsed_ms, rid, b.idx, attempt),
    )
    if cur.rowcount == 1:
        return True
    conn.execute("insert into fenced (run_id, idx, attempt, worker) values (%s,%s,%s,%s)",
                 (rid, b.idx, attempt, worker))
    return False


def done_prefix_stats(conn, rid: int) -> list[dict]:
    rows = conn.execute(
        "select idx, status, stats from batches where run_id=%s order by idx", (rid,)
    ).fetchall()
    out = []
    for idx, status, stats in rows:
        if status != "done":
            break
        out.append(stats)
    return out


def check_stop(conn, rid: int, cfg: RunConfig) -> Optional[str]:
    """Set the run's stop point if the finished prefix has converged.

    Uses the *smallest* converging prefix, so the answer is the same however
    the batches were scheduled. Only one caller wins the conditional UPDATE.
    """
    checked = conn.execute("select checked_upto from runs where id=%s", (rid,)).fetchone()[0]
    stats = done_prefix_stats(conn, rid)
    if len(stats) < checked and len(stats) < cfg.max_batches:
        return None
    m = first_converged(stats, cfg, start=checked)
    if m is None:
        # Incremental: the next check starts after this prefix. greatest()
        # because another worker may have checked a longer prefix already.
        conn.execute("update runs set checked_upto=greatest(checked_upto, %s) where id=%s",
                     (len(stats) + 1, rid))
    status = None
    if m is not None:
        status = "converged"
    elif len(stats) == cfg.max_batches:
        m, status = cfg.max_batches, "max_batches"
    if status is None:
        return None
    conn.execute(
        "update runs set stop_batch=%s, status='finishing' where id=%s and status='running'",
        (m, rid),
    )
    return status


def claim_finalize(conn, rid: int, lease_s: float = 120.0) -> bool:
    """Exactly one worker builds the summary (it loads every batch's
    trajectories, ~115 MB for 384 batches). When every idle worker did it,
    they fought over the CPU and the summary step took 6 s instead of 3.
    The claim is taken over the same way as a batch claim: lease expiry or
    the holder's Postgres session disappearing."""
    row = conn.execute(
        f"""
        update runs set status='summarizing', finalizer_pid=pg_backend_pid(),
               finalize_until = now() + make_interval(secs => %s)
        where id=%s and (status='finishing'
              or (status='summarizing' and (finalize_until < now() {_DEAD.replace('backend_pid', 'finalizer_pid')})))
        returning id
        """,
        (lease_s, rid),
    ).fetchone()
    return row is not None


def finalize(conn, rid: int) -> Optional[dict]:
    """Build the summary once every batch below the stop point is done."""
    spec, cfg, status = load_run(conn, rid)
    stop = conn.execute("select stop_batch from runs where id=%s", (rid,)).fetchone()[0]
    if stop is None:
        return None
    rows = conn.execute(
        "select idx, stats, base, scen from batches where run_id=%s and idx < %s and status='done' order by idx",
        (rid, stop),
    ).fetchall()
    if len(rows) < stop:
        return None
    batches = [Batch(i, _unpack(b), _unpack(s), st) for i, st, b, s in rows]
    res = summarize(spec, cfg, batches)
    res["stopped"] = "converged" if first_converged([b.stats for b in batches], cfg) else "max_batches"
    conn.execute(
        "update runs set summary=%s, status=%s, finished_at=now() where id=%s and summary is null",
        (json.dumps(res), res["stopped"], rid),
    )
    return res


def work(rid: int, worker: Optional[str] = None, lease_s: float = 5.0, url: Optional[str] = None) -> dict:
    """Worker loop. Safe to run any number of these, and to kill them anytime."""
    worker = worker or f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
    conn = connect(url)
    spec, cfg, _ = load_run(conn, rid)
    comp = Compiled(spec)
    done = fenced = 0
    while True:
        status = conn.execute("select status from runs where id=%s", (rid,)).fetchone()[0]
        if status in ("converged", "max_batches", "failed"):
            break
        got = claim(conn, rid, worker, lease_s)
        if got is None:
            # Nothing claimable. Either others hold the remaining leases
            # (wait; they may die), or the run is finishing: one worker wins
            # the summary, the rest wait in case it dies and needs replacing.
            if status in ("finishing", "summarizing") and claim_finalize(conn, rid):
                finalize(conn, rid)
                break
            time.sleep(0.05 if status == "running" else 0.2)
            continue
        idx, attempt = got
        t0 = time.perf_counter()
        b = run_batch(comp, cfg, idx)
        ok = complete(conn, rid, b, attempt, worker, (time.perf_counter() - t0) * 1e3)
        done += ok
        fenced += not ok
        if ok:
            check_stop(conn, rid, cfg)
    conn.close()
    return {"worker": worker, "batches": done, "fenced": fenced}


def run_status(conn, rid: int) -> dict:
    spec, cfg, status = load_run(conn, rid)
    counts = dict(conn.execute(
        "select status, count(*) from batches where run_id=%s group by status", (rid,)).fetchall())
    prefix = done_prefix_stats(conn, rid)
    conv = convergence(prefix, cfg.tol) if len(prefix) >= 2 else None
    stop, summary = conn.execute("select stop_batch, summary from runs where id=%s", (rid,)).fetchone()
    retries = conn.execute(
        "select coalesce(sum(attempt-1),0) from batches where run_id=%s and attempt>1", (rid,)).fetchone()[0]
    nfenced = conn.execute("select count(*) from fenced where run_id=%s", (rid,)).fetchone()[0]
    wasted = conn.execute(
        "select count(*) from batches where run_id=%s and status='done' and idx >= coalesce(%s, 2147483647)",
        (rid, stop)).fetchone()[0]
    return {
        "id": rid, "status": status, "model": spec.name, "scenario": cfg.scenario,
        "batches": counts, "prefix_done": len(prefix), "stop_batch": stop,
        "convergence": conv, "retried_claims": int(retries), "fenced_writes": int(nfenced),
        "wasted_batches": int(wasted), "summary": summary,
    }
