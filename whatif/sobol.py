"""Variance-based (Sobol) sensitivity indices, Saltelli/Jansen estimators.

For k random inputs we draw two independent N x k matrices A and B, and for
each input i the matrix AB_i (A with column i taken from B). With f the model
output:

    first order  S_i  = mean(f(B) * (f(AB_i) - f(A))) / Var(f)   (Saltelli 2010)
    total        ST_i = mean((f(A) - f(AB_i))**2) / (2 Var(f))   (Jansen 1999)

Cost is N * (k + 2) model evaluations. Confidence intervals by bootstrap over
the N rows. Inputs only need to be independent, not uniform, so we sample each
parameter from its own distribution directly.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .mc import batch_rng
from .model import ModelSpec
from .sim import Compiled


def indices(
    sample: Callable[[np.random.Generator, int], np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    k: int,
    n: int,
    seed: int = 1,
    n_boot: int = 200,
) -> dict:
    rng = batch_rng(seed, 0)
    A, B = sample(rng, n), sample(rng, n)
    fA, fB = f(A), f(B)
    fAB = np.empty((k, n))
    for i in range(k):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB[i] = f(ABi)

    def est(rows: np.ndarray):
        # Centre first: the estimators are shift-invariant in expectation, but
        # a large mean (an effect of ~0.011 with sd ~0.001) inflates their
        # variance by mean^2/var. Without this, S1 for find_rate came out -0.19.
        a, b, ab = fA[rows], fB[rows], fAB[:, rows]
        c = np.mean(np.concatenate([a, b]))
        a, b, ab = a - c, b - c, ab - c
        var = np.var(np.concatenate([a, b]), ddof=1)
        s1 = np.mean(b * (ab - a), axis=1) / var
        st = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
        return s1, st

    s1, st = est(np.arange(n))
    brng = batch_rng(seed, 1)
    boots = [est(brng.integers(0, n, n)) for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boots])
    bt = np.array([b[1] for b in boots])
    return {
        "S1": s1.tolist(), "ST": st.tolist(),
        "S1_ci": np.percentile(b1, [2.5, 97.5], axis=0).T.tolist(),
        "ST_ci": np.percentile(bt, [2.5, 97.5], axis=0).T.tolist(),
        "evaluations": n * (k + 2),
    }


def model_sensitivity(spec: ModelSpec, scenario: Optional[str], n: int = 2048, seed: int = 7,
                      chunk: int = 512) -> dict:
    """Sobol indices of the tracked quantity (primary output at the horizon,
    as a scenario effect if a scenario is given) w.r.t. the random parameters."""
    comp = Compiled(spec)
    rand = spec.random_params()
    names = [p.name for p in rand]
    fixed = {p.name: p.dist.center() for p in spec.params if p.dist.kind == "fixed"}
    sc = spec.scenario(scenario)

    def sample(rng, m):
        return np.stack([p.dist.sample(rng, m) for p in rand], axis=1)

    def f(X: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(X), chunk):
            xs = X[s:s + chunk]
            P = {k: np.full(len(xs), v) for k, v in fixed.items()}
            P.update({nm: xs[:, i] for i, nm in enumerate(names)})
            b = comp.simulate(P, None).outputs[:, -1, 0]
            if sc:
                b = comp.simulate(P, sc).outputs[:, -1, 0] - b
            out.append(b)
        return np.concatenate(out)

    res = indices(sample, f, len(names), n, seed)
    res["params"] = names
    res["target"] = {"output": spec.outputs[0], "t": spec.horizon,
                     "kind": "effect" if sc else "level"}
    return res
