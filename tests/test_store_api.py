"""Postgres checkpointing and the HTTP API. Postgres tests need WHATIF_TEST_DB."""
import os
import time

import psycopg
import pytest
from fastapi.testclient import TestClient

from whatif import store
from whatif.mc import RunConfig, run_local


@pytest.fixture
def conn(db_url, monkeypatch):
    monkeypatch.setenv("WHATIF_DB", db_url)
    c = store.connect(db_url)
    c.execute("drop table if exists fenced, batches, runs cascade")
    store.init_schema(c)
    yield c
    c.close()


def test_checkpointed_run_matches_local(conn, labor, db_url):
    cfg = RunConfig(scenario="find_rate_down_20", batch_size=64, tol=3e-4)
    rid = store.create_run(conn, labor, cfg)
    store.work(rid, url=db_url)
    st = store.run_status(conn, rid)
    assert st["status"] == "converged"
    assert st["summary"]["sha256"] == run_local(labor, cfg)["sha256"]


def test_dead_session_claim_is_reclaimed_immediately(conn, labor, db_url):
    cfg = RunConfig(batch_size=16, max_batches=3)
    rid = store.create_run(conn, labor, cfg)
    a = store.connect(db_url)
    assert store.claim(a, rid, "a", lease_s=3600) == (0, 1)
    b = store.connect(db_url)
    assert store.claim(b, rid, "b", lease_s=3600) == (1, 1)   # a's claim is alive
    a.close()                                                # "kill" a
    for _ in range(50):                                      # backend exit is async
        got = store.claim(b, rid, "b", lease_s=3600)
        if got and got[0] == 0:
            break
        if got:                                              # took batch 2; put it back
            b.execute("update batches set status='pending' where run_id=%s and idx=%s", (rid, got[0]))
        time.sleep(0.05)
    assert got == (0, 2)
    b.close()


def test_stale_worker_is_fenced(conn, labor, db_url):
    from whatif.mc import run_batch
    from whatif.sim import Compiled
    cfg = RunConfig(batch_size=16, max_batches=2)
    rid = store.create_run(conn, labor, cfg)
    idx, att = store.claim(conn, rid, "slow", lease_s=0.01)
    time.sleep(0.05)
    other = store.connect(db_url)
    assert store.claim(other, rid, "fast", lease_s=60) == (idx, att + 1)
    b = run_batch(Compiled(labor), cfg, idx)
    assert store.complete(other, rid, b, att + 1, "fast", 1.0)
    assert not store.complete(conn, rid, b, att, "slow", 1.0)   # late write rejected
    assert conn.execute("select count(*) from fenced where run_id=%s", (rid,)).fetchone()[0] == 1
    other.close()


def test_api_offline_endpoints(labor_json):
    from whatif.api import app
    c = TestClient(app)
    assert any(e["name"] == "labor_market" for e in c.get("/api/examples").json())
    r = c.post("/api/validate", json={"model": labor_json}).json()
    assert r["ok"] and r["loops"]
    labor_json["flows"][0]["rate"] = "find_rate * unemployed + 1"
    r = c.post("/api/validate", json={"model": labor_json}).json()
    assert not r["ok"] and r["issues"][0]["code"] == "UNIT_MISMATCH"
    assert c.post("/api/runs", json={"model": labor_json}).status_code == 422
    os.environ.pop("ANTHROPIC_API_KEY", None)
    assert c.post("/api/build", json={"question": "q"}).status_code == 503


def test_api_run_end_to_end(conn, labor_json):
    from whatif.api import app
    c = TestClient(app)
    rid = c.post("/api/runs", json={"model": labor_json, "scenario": "find_rate_down_20",
                                    "tol": 3e-4, "workers": 2}).json()["id"]
    for _ in range(600):
        st = c.get(f"/api/runs/{rid}").json()
        if st["status"] in ("converged", "max_batches"):
            break
        time.sleep(0.1)
    assert st["status"] == "converged"
    res = c.get(f"/api/runs/{rid}/result").json()
    assert len(res["effect"]["unemployment_rate"]["p50"]) == 25
    md = c.post("/api/brief", json={"run_id": rid}).text
    assert "## Inputs and where they came from" in md
