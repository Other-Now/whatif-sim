"""FastAPI app: validate, run (checkpointed in Postgres), sensitivity, brief, build."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .brief import render
from .chaos import spawn
from .mc import RunConfig
from .model import ModelSpec
from .sobol import model_sensitivity
from .validate import validate, validate_json

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

app = FastAPI(title="whatif-sim")
_procs: dict[int, list] = {}


class ModelIn(BaseModel):
    model: dict


class RunIn(BaseModel):
    model: dict
    scenario: Optional[str] = None
    tol: float = 1e-4
    workers: int = 2
    max_batches: int = 200


class SobolIn(BaseModel):
    model: dict
    scenario: Optional[str] = None
    n: int = 1024


class BriefIn(BaseModel):
    run_id: int


class BuildIn(BaseModel):
    question: str


def _spec(data: dict) -> ModelSpec:
    spec, rep = validate_json(data)
    if not rep.ok:
        raise HTTPException(422, rep.to_dict())
    return spec


@app.get("/api/examples")
def examples() -> list[dict]:
    out = []
    for p in sorted(EXAMPLES.glob("*.json")):
        if p.name.endswith(".template.json") and (EXAMPLES / p.name.replace(".template", "")).exists():
            continue
        m = json.loads(p.read_text())
        out.append({"id": p.stem, "name": m["name"], "question": m.get("question", "")})
    return out


@app.get("/api/examples/{eid}")
def example(eid: str) -> dict:
    p = EXAMPLES / f"{eid}.json"
    if not p.exists() or p.parent != EXAMPLES:
        raise HTTPException(404)
    return json.loads(p.read_text())


@app.post("/api/validate")
def validate_model(body: ModelIn) -> dict:
    return validate_json(body.model)[1].to_dict()


@app.post("/api/runs")
def create_run(body: RunIn) -> dict:
    spec = _spec(body.model)
    if body.scenario:
        spec.scenario(body.scenario)
    cfg = RunConfig(scenario=body.scenario, tol=body.tol, max_batches=body.max_batches)
    conn = store.connect()
    store.init_schema(conn)
    rid = store.create_run(conn, spec, cfg)
    conn.close()
    _procs[rid] = [spawn(rid, 5.0) for _ in range(max(1, min(body.workers, os.cpu_count() or 1)))]
    return {"id": rid}


@app.get("/api/runs/{rid}")
def run_status(rid: int) -> dict:
    with store.connect() as conn:
        st = store.run_status(conn, rid)
    st.pop("summary", None)
    if st["status"] in ("converged", "max_batches"):
        for p in _procs.pop(rid, []):
            p.poll()
    return st


@app.get("/api/runs/{rid}/result")
def run_result(rid: int) -> dict:
    with store.connect() as conn:
        row = conn.execute("select summary from runs where id=%s", (rid,)).fetchone()
    if row is None:
        raise HTTPException(404)
    if row[0] is None:
        raise HTTPException(409, "run not finished")
    return row[0]


@app.post("/api/sobol")
def sobol(body: SobolIn) -> dict:
    return model_sensitivity(_spec(body.model), body.scenario, n=min(body.n, 8192))


@app.post("/api/brief", response_class=PlainTextResponse)
def brief(body: BriefIn) -> str:
    with store.connect() as conn:
        spec, cfg, _ = store.load_run(conn, body.run_id)
        summary = conn.execute("select summary from runs where id=%s", (body.run_id,)).fetchone()[0]
    if summary is None:
        raise HTTPException(409, "run not finished")
    return render(spec, summary, validate(spec), model_sensitivity(spec, cfg.scenario, n=1024), cfg.scenario)


@app.post("/api/build")
def build(body: BuildIn) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "LLM model builder needs ANTHROPIC_API_KEY on the server")
    from .llm import AnthropicProvider, BuildFailed
    from .llm import build as do_build
    try:
        spec, report, log = do_build(body.question, AnthropicProvider())
    except BuildFailed as e:
        raise HTTPException(422, {"stage": e.stage, "log": e.log})
    return {"model": spec.dump(), "report": report.to_dict(), "log": log}


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
