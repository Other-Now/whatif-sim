"""Command line: python -m whatif <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .mc import RunConfig, run_local
from .model import ModelSpec


def _load(path: str) -> ModelSpec:
    return ModelSpec.model_validate(json.loads(Path(path).read_text()))


def _cfg(a) -> RunConfig:
    return RunConfig(scenario=a.scenario, batch_size=a.batch_size, max_batches=a.max_batches,
                     min_batches=a.min_batches, seed=a.seed, tol=a.tol)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="whatif")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def run_args(p):
        p.add_argument("model")
        p.add_argument("--scenario")
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--max-batches", type=int, default=200)
        p.add_argument("--min-batches", type=int, default=10)
        p.add_argument("--seed", type=int, default=20261018)
        p.add_argument("--tol", type=float, default=1e-4)

    p = sub.add_parser("validate", help="validate a model JSON and print issues + loops")
    p.add_argument("model")
    p = sub.add_parser("local", help="Monte Carlo in-process, no database")
    run_args(p)
    p = sub.add_parser("run", help="checkpointed Monte Carlo in Postgres with N worker processes")
    run_args(p)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("worker", help="work on a run until it finishes (safe to kill)")
    p.add_argument("run_id", type=int)
    p.add_argument("--lease", type=float, default=5.0)
    p = sub.add_parser("status")
    p.add_argument("run_id", type=int)
    p = sub.add_parser("chaos", help="kill workers mid-run; compare to an unkilled reference run")
    run_args(p)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--kills", type=int, default=20)
    p.add_argument("--kill-every", type=float, default=0.4)
    p = sub.add_parser("sobol", help="Sobol sensitivity of the tracked output")
    p.add_argument("model")
    p.add_argument("--scenario")
    p.add_argument("-n", type=int, default=2048)
    p = sub.add_parser("brief", help="Markdown brief for a model (runs MC + Sobol locally)")
    run_args(p)
    p.add_argument("-o", "--out")
    p = sub.add_parser("fred-fetch", help="snapshot FRED series into data/fred")
    p.add_argument("series", nargs="*")
    p = sub.add_parser("fred-fill", help="fill the labor template's parameters from the snapshot")
    p.add_argument("template")
    p.add_argument("out")
    p = sub.add_parser("build", help="LLM: question -> validated model (needs ANTHROPIC_API_KEY)")
    p.add_argument("question")
    p.add_argument("-o", "--out", default="built_model.json")
    p.add_argument("--log", default="build_log.jsonl")
    p.add_argument("--script", help="replay a scripted provider file instead of calling the API")
    p = sub.add_parser("serve", help="FastAPI app + built web UI")
    p.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        from .validate import validate_json
        _, r = validate_json(json.loads(Path(a.model).read_text()))
        print(json.dumps(r.to_dict(), indent=2))
        sys.exit(0 if r.ok else 1)
    elif a.cmd == "local":
        t = time.perf_counter()
        res = run_local(_load(a.model), _cfg(a))
        res = {k: v for k, v in res.items() if k not in ("baseline", "scenario", "effect", "t")}
        print(json.dumps(res | {"wall_s": round(time.perf_counter() - t, 2)}, indent=2))
    elif a.cmd == "run":
        from . import store
        from .chaos import run_workers
        conn = store.connect()
        store.init_schema(conn)
        rid = store.create_run(conn, _load(a.model), _cfg(a))
        conn.close()
        print(json.dumps(run_workers(rid, a.workers, lease_s=5.0), indent=2))
    elif a.cmd == "worker":
        from . import store
        print(json.dumps(store.work(a.run_id, lease_s=a.lease)))
    elif a.cmd == "status":
        from . import store
        st = store.run_status(store.connect(), a.run_id)
        st.pop("summary", None)
        print(json.dumps(st, indent=2, default=str))
    elif a.cmd == "chaos":
        from .chaos import chaos
        print(json.dumps(chaos(_load(a.model), _cfg(a), a.workers, a.kills, a.kill_every), indent=2))
    elif a.cmd == "sobol":
        from .sobol import model_sensitivity
        print(json.dumps(model_sensitivity(_load(a.model), a.scenario, a.n), indent=2))
    elif a.cmd == "brief":
        from .brief import render
        from .sobol import model_sensitivity
        from .validate import validate
        spec = _load(a.model)
        md = render(spec, run_local(spec, _cfg(a)), validate(spec),
                    model_sensitivity(spec, a.scenario), a.scenario)
        if a.out:
            Path(a.out).write_text(md, encoding="utf-8")
        else:
            print(md)
    elif a.cmd == "fred-fetch":
        from .evidence import CATALOG, snapshot
        print(json.dumps(snapshot(a.series or list(CATALOG)), indent=2))
    elif a.cmd == "fred-fill":
        from .evidence import LABOR_MAPPING, fill
        spec, notes = fill(_load(a.template), LABOR_MAPPING)
        Path(a.out).write_text(json.dumps(spec.dump(), indent=2) + "\n")
        print(json.dumps(notes, indent=2))
    elif a.cmd == "build":
        from .llm import AnthropicProvider, BuildFailed, ScriptedProvider, build
        prov = ScriptedProvider(json.loads(Path(a.script).read_text())) if a.script else AnthropicProvider()
        try:
            spec, report, log = build(a.question, prov, log_path=Path(a.log))
        except BuildFailed as e:
            print(f"build failed at {e.stage}; see {a.log}", file=sys.stderr)
            sys.exit(1)
        Path(a.out).write_text(json.dumps(spec.dump(), indent=2) + "\n")
        for e in log:
            print(f"{e['stage']:<18} attempt {e['attempt']}  {e['verdict']:<8} "
                  + ", ".join(i["code"] for i in e["issues"] if i["severity"] == "error")
                  + (f"  usage={e['usage']}" if e["usage"] else ""))
        print(f"wrote {a.out}; loops: " + "; ".join(f"{l.polarity}:{'->'.join(l.nodes)}" for l in report.loops))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("whatif.api:app", host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
