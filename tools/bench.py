"""Benchmarks behind every number in the README. Each writes results/<name>.json.

    python tools/bench.py solver     # error vs tolerance on problems with exact solutions
    python tools/bench.py order      # fixed-step convergence order
    python tools/bench.py crn        # common random numbers: variance of the effect
    python tools/bench.py sobol      # Ishigami (exact indices known) + the labor model
    python tools/bench.py stopping   # batches needed vs tolerance; batch-means vs naive SE
    python tools/bench.py workers    # parallel speed-up, 1..N worker processes (Postgres)
    python tools/bench.py chaos      # kill workers mid-run, with and without the session check
    python tools/bench.py validator  # seeded model bugs: how many does the validator catch
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from whatif.mc import RunConfig, run_local  # noqa: E402
from whatif.model import ModelSpec  # noqa: E402
from whatif.solver import solve  # noqa: E402

RESULTS = ROOT / "results"
MODEL = ROOT / "examples" / "labor_market.json"
if not MODEL.exists():
    MODEL = ROOT / "examples" / "labor_market.template.json"
SCEN = "find_rate_down_20"


def spec() -> ModelSpec:
    return ModelSpec.model_validate(json.loads(MODEL.read_text()))


def save(name: str, data: dict) -> None:
    data = {"model_file": MODEL.name, "machine": platform.processor() or platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, **data}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


# ---- problems with exact solutions -------------------------------------------
def problems():
    k = 0.7
    yield ("decay", lambda t, y: -k * y, np.array([[1.0]]), lambda t: np.array([np.exp(-k * t)]), 10.0)
    r, K = 0.9, 10.0
    yield ("logistic", lambda t, y: r * y * (1 - y / K), np.array([[0.1]]),
           lambda t: np.array([K / (1 + (K / 0.1 - 1) * np.exp(-r * t))]), 15.0)
    w = 2.0
    yield ("oscillator", lambda t, y: np.stack([y[:, 1], -w * w * y[:, 0]], axis=1), np.array([[1.0, 0.0]]),
           lambda t: np.array([np.cos(w * t), -w * np.sin(w * t)]), 20.0)


def bench_solver():
    rows = []
    for name, f, y0, exact, T in problems():
        for tol in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
            ys, st = solve(f, 0.0, y0, np.array([T]), rtol=tol, atol=tol * 1e-3)
            err = float(np.max(np.abs(ys[0, -1] - exact(T))))
            rows.append({"problem": name, "rtol": tol, "max_abs_err": err, "steps": st.steps,
                         "rejected": st.rejected, "fevals": st.fevals})
    save("solver", {"rows": rows})


def bench_order():
    rows = []
    for name, f, y0, exact, T in problems():
        errs = []
        for n in (20, 40, 80, 160, 320):
            ys, _ = solve(f, 0.0, y0, np.array([T]), h_fixed=T / n)
            errs.append(float(np.max(np.abs(ys[0, -1] - exact(T)))))
        orders = [float(np.log2(a / b)) for a, b in zip(errs, errs[1:])]
        rows.append({"problem": name, "steps": [20, 40, 80, 160, 320], "errors": errs, "observed_order": orders})
    save("order", {"rows": rows, "expected_order": 5})


def bench_crn():
    from whatif.sim import Compiled
    s = spec()
    comp = Compiled(s)
    sc = s.scenario(SCEN)
    n = 20000
    rng = np.random.default_rng(1)
    P1 = comp.sample_params(rng, n)
    P2 = comp.sample_params(rng, n)
    b1 = comp.simulate(P1, None).outputs[:, -1, 0]
    s1 = comp.simulate(P1, sc).outputs[:, -1, 0]
    s2 = comp.simulate(P2, sc).outputs[:, -1, 0]
    v_crn = float(np.var(s1 - b1, ddof=1))
    v_ind = float(np.var(s2 - b1, ddof=1))
    save("crn", {"samples": n, "effect_mean_crn": float(np.mean(s1 - b1)),
                 "effect_mean_independent": float(np.mean(s2 - b1)),
                 "var_effect_crn": v_crn, "var_effect_independent": v_ind,
                 "variance_reduction": v_ind / v_crn,
                 "note": "samples needed for the same CI shrink by the variance ratio"})


def bench_sobol():
    from whatif.sobol import indices, model_sensitivity
    a, b = 7.0, 0.1
    f = lambda X: np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
    exact = {"S1": [0.3139, 0.4424, 0.0], "ST": [0.5576, 0.4424, 0.2437]}
    ish = {}
    for n in (1000, 10000, 100000):
        r = indices(lambda g, m: g.uniform(-np.pi, np.pi, (m, 3)), f, 3, n, n_boot=100)
        ish[n] = {"S1": r["S1"], "ST": r["ST"],
                  "max_abs_err": max(max(abs(x - y) for x, y in zip(r["S1"], exact["S1"])),
                                     max(abs(x - y) for x, y in zip(r["ST"], exact["ST"])))}
    t = time.perf_counter()
    lab = model_sensitivity(spec(), SCEN, n=4096)
    lab["wall_s"] = round(time.perf_counter() - t, 2)
    save("sobol", {"ishigami_exact": exact, "ishigami": ish, "labor": lab})


def bench_stopping():
    rows = []
    for tol in (4e-4, 2e-4, 1e-4, 5e-5):
        t = time.perf_counter()
        r = run_local(spec(), RunConfig(scenario=SCEN, tol=tol, max_batches=400))
        c = r["convergence"]
        rows.append({"tol": tol, "batches": r["batches"], "samples": r["samples"], "stopped": r["stopped"],
                     "mean_halfwidth_batch_means": c["mean"]["halfwidth"],
                     "mean_halfwidth_naive": c["mean"]["naive_halfwidth"],
                     "p95_halfwidth": c["p95"]["halfwidth"], "wall_s": round(time.perf_counter() - t, 2)})
    save("stopping", {"batch_size": 256, "rows": rows})


def _cli(*args, env=None) -> dict:
    out = subprocess.run([sys.executable, "-m", "whatif", *args], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, **(env or {})}, check=True)
    return json.loads(out.stdout)


def bench_workers():
    rows = []
    for w in (1, 2, 4, 6):
        r = _cli("run", str(MODEL), "--scenario", SCEN, "--tol", "0", "--max-batches", "96",
                 "--workers", str(w))
        rows.append({"workers": w, "batches": 96, "wall_s": r["wall_s"]})
    for row in rows:
        row["speedup"] = round(rows[0]["wall_s"] / row["wall_s"], 2)
    save("workers", {"cpu_count": os.cpu_count(), "note": "tol=0 so every run computes exactly 96 batches",
                     "rows": rows})


def bench_chaos(reps: int = 5):
    out = {}
    for label, env in (("lease_only", {"WHATIF_SESSION_CHECK": "0"}), ("lease_or_dead_session", {})):
        runs = []
        for i in range(reps):
            r = _cli("chaos", str(MODEL), "--scenario", SCEN, "--tol", "5e-5", "--max-batches", "400",
                     "--kills", "20", "--kill-every", "0.25", "--workers", "4", env=env)
            c = r["chaos"]
            runs.append({"identical": r["identical"], "kills": c["kills"], "wasted_batches": c["wasted_batches"],
                         "retried_claims": c["retried_claims"], "fenced_writes": c["fenced_writes"],
                         "stop_batch": c["stop_batch"], "wall_s": c["wall_s"],
                         "reference_wall_s": r["reference"]["wall_s"]})
        out[label] = runs
    save("chaos", out)


def bench_validator():
    from tests.mutations import MUTATIONS, apply_mutation
    from whatif.validate import validate_json
    base = json.loads(MODEL.read_text())
    rows = []
    for m in MUTATIONS:
        _, rep = validate_json(apply_mutation(base, m))
        codes = sorted({i.code for i in rep.errors})
        rows.append({"mutation": m["name"], "expected": m["expect"], "got": codes, "caught": m["expect"] in codes})
    save("validator", {"caught": sum(r["caught"] for r in rows), "total": len(rows), "rows": rows})


if __name__ == "__main__":
    {"solver": bench_solver, "order": bench_order, "crn": bench_crn, "sobol": bench_sobol,
     "stopping": bench_stopping, "workers": bench_workers, "chaos": bench_chaos,
     "validator": bench_validator}[sys.argv[1]]()
