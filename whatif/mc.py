"""Monte Carlo over parameter uncertainty, in deterministic batches.

Three ideas carry the design:

* **Batch i is a pure function of (model, config, i).** Its random stream is
  ``SeedSequence(seed, spawn_key=(i,))``, so any worker can compute any batch,
  a batch recomputed after a crash is bit-identical, and the result doesn't
  depend on how many workers ran or in what order.
* **Common random numbers.** Baseline and scenario use the *same* parameter
  draws, so the what-if effect (scenario - baseline) is paired per sample and
  its variance is much smaller than the two runs' variances added.
* **Convergence-based stopping on a prefix.** After each batch we look at the
  contiguous prefix of finished batches and stop at the *smallest* prefix
  length whose batch-means confidence intervals are all within tolerance.
  Using the smallest qualifying prefix (not "whenever a worker happened to
  check") makes the stop point independent of scheduling too.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .model import ModelSpec
from .sim import Compiled

QUANTILES = (5, 25, 50, 75, 95)


@dataclass
class RunConfig:
    scenario: Optional[str] = None
    batch_size: int = 256
    min_batches: int = 10
    max_batches: int = 200
    seed: int = 20261018
    tol: float = 5e-4            # absolute CI half-width target, in output units
    rtol: float = 1e-6           # ODE tolerances
    atol: float = 1e-9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Batch:
    idx: int
    base: np.ndarray              # (b, T, n_outputs)
    scen: Optional[np.ndarray]
    stats: dict                   # tracked statistics for this batch
    steps: int = 0


def batch_rng(seed: int, idx: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(idx,)))


def tracked_series(base: np.ndarray, scen: Optional[np.ndarray]) -> np.ndarray:
    """The scalar the stopping rule watches: primary output at the horizon,
    as an effect (scenario - baseline) when there is a scenario."""
    x = base[:, -1, 0]
    return x if scen is None else scen[:, -1, 0] - x


def batch_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "var": float(np.var(x, ddof=1)),
        "n": int(x.size),
    }


def run_batch(comp: Compiled, cfg: RunConfig, idx: int) -> Batch:
    rng = batch_rng(cfg.seed, idx)
    P = comp.sample_params(rng, cfg.batch_size)
    sc = comp.spec.scenario(cfg.scenario)
    with np.errstate(all="ignore"):
        b = comp.simulate(P, None, cfg.rtol, cfg.atol)
        s = comp.simulate(P, sc, cfg.rtol, cfg.atol) if sc else None
    steps = b.stats.steps + (s.stats.steps if s else 0)
    out_s = s.outputs if s else None
    return Batch(idx, b.outputs, out_s, batch_stats(tracked_series(b.outputs, out_s)), steps)


# ---- stopping rule -----------------------------------------------------------

def t_quantile_975(df: int) -> float:
    """Student t 97.5% quantile via the Cornish-Fisher expansion (no scipy).
    Error 3e-4 at df=9 (the smallest the default min_batches allows) and
    under 1e-4 from df=29; 3e-3 at df=5."""
    z = 1.959963984540054
    v = float(df)
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


STOP_STATS = ("mean", "p05", "p95")


def convergence(stats: list[dict], tol: float) -> dict:
    """Batch-means estimate and 95% half-width for each tracked statistic."""
    nb = len(stats)
    t = t_quantile_975(nb - 1)
    out = {}
    for k in STOP_STATS:
        v = np.array([s[k] for s in stats])
        hw = t * float(np.std(v, ddof=1)) / math.sqrt(nb)
        out[k] = {"est": float(v.mean()), "halfwidth": hw}
    # the i.i.d. standard error of the mean, to compare with batch means
    n = sum(s["n"] for s in stats)
    pooled_var = sum((s["n"] - 1) * s["var"] for s in stats) / max(1, n - nb)
    out["mean"]["naive_halfwidth"] = 1.959963984540054 * math.sqrt(pooled_var / n)
    out["converged"] = all(out[k]["halfwidth"] <= tol for k in STOP_STATS)
    out["batches"] = nb
    return out


def first_converged(stats: list[dict], cfg: RunConfig, start: int = 0) -> Optional[int]:
    """Smallest prefix length m >= min_batches that meets the tolerance.

    ``start``: prefixes shorter than this are already known not to converge
    (stats are immutable once written, so that can't change)."""
    for m in range(max(cfg.min_batches, start), len(stats) + 1):
        if convergence(stats[:m], cfg.tol)["converged"]:
            return m
    return None


# ---- results -----------------------------------------------------------------

def summarize(spec: ModelSpec, cfg: RunConfig, batches: list[Batch]) -> dict:
    base = np.concatenate([b.base for b in batches])
    scen = np.concatenate([b.scen for b in batches]) if batches[0].scen is not None else None
    comp_t = np.round(np.arange(0.0, spec.horizon + 1e-9, spec.dt_out), 12)

    def fan(a: np.ndarray) -> dict:
        q = np.percentile(a, QUANTILES, axis=0)        # (Q, T, n_outputs)
        return {name: {f"p{qq:02d}": q[i, :, j].tolist() for i, qq in enumerate(QUANTILES)}
                | {"mean": a[:, :, j].mean(axis=0).tolist()}
                for j, name in enumerate(spec.outputs)}

    h = hashlib.sha256(base.tobytes())
    if scen is not None:
        h.update(scen.tobytes())
    out = {
        "samples": int(base.shape[0]),
        "batches": len(batches),
        "t": comp_t.tolist(),
        "baseline": fan(base),
        "convergence": convergence([b.stats for b in batches], cfg.tol),
        "sha256": h.hexdigest(),
    }
    if scen is not None:
        out["scenario"] = fan(scen)
        out["effect"] = fan(scen - base)
    x = tracked_series(base, scen)
    out["tracked"] = {"output": spec.outputs[0], "t": float(comp_t[-1]),
                      "kind": "effect" if scen is not None else "level",
                      "mean": float(x.mean()), "p05": float(np.percentile(x, 5)),
                      "p95": float(np.percentile(x, 95))}
    return out


def run_local(spec: ModelSpec, cfg: RunConfig, progress=None) -> dict:
    """Single-process run, no checkpointing. Same answer as the Postgres runner."""
    comp = Compiled(spec)
    batches: list[Batch] = []
    stop = None
    for i in range(cfg.max_batches):
        batches.append(run_batch(comp, cfg, i))
        if progress:
            progress(i)
        if len(batches) >= cfg.min_batches and convergence([b.stats for b in batches], cfg.tol)["converged"]:
            stop = len(batches)
            break
    res = summarize(spec, cfg, batches)
    res["stopped"] = "converged" if stop else "max_batches"
    return res
