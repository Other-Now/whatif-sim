"""Dormand-Prince 5(4) with adaptive step-size control, vectorized over a batch.

The state is an ``(n_samples, n_states)`` array, so one call integrates a whole
Monte Carlo batch at once. All samples share a step size chosen by the worst
sample's error estimate: a little wasted work on easy samples, in exchange for
numpy doing the arithmetic instead of a Python loop per sample.

The right-hand side must be smooth on the interval. Discontinuities (a
scenario switching on at t=6) are handled by the caller splitting the
integration into segments at the breakpoints; see ``sim.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Butcher tableau (Dormand & Prince 1980)
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
E = B5 - B4

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0

Rhs = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class SolveStats:
    steps: int = 0
    rejected: int = 0
    fevals: int = 0
    h_min: float = field(default=np.inf)

    def add(self, o: "SolveStats") -> None:
        self.steps += o.steps
        self.rejected += o.rejected
        self.fevals += o.fevals
        self.h_min = min(self.h_min, o.h_min)


class SolverError(RuntimeError):
    pass


def _initial_step(f: Rhs, t0: float, y0: np.ndarray, f0: np.ndarray, rtol: float, atol: float) -> float:
    # Hairer, Norsett & Wanner, "Solving ODEs I", II.4
    sc = atol + rtol * np.abs(y0)
    d0 = np.sqrt(np.mean((y0 / sc) ** 2))
    d1 = np.sqrt(np.mean((f0 / sc) ** 2))
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    f1 = f(t0 + h0, y0 + h0 * f0)
    d2 = np.sqrt(np.mean(((f1 - f0) / sc) ** 2)) / h0
    h1 = max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15 else (0.01 / max(d1, d2)) ** (1 / 5)
    return min(100 * h0, h1)


def _stages(f: Rhs, t: float, y: np.ndarray, h: float, k1: np.ndarray):
    k = [k1]
    for i in range(1, 7):
        yi = y + h * sum(a * k[j] for j, a in enumerate(A[i]) if a != 0)
        k.append(f(t + C[i] * h, yi))
    y5 = y + h * sum(b * k[j] for j, b in enumerate(B5) if b != 0)
    err = h * sum(e * k[j] for j, e in enumerate(E) if e != 0)
    return y5, err, k[6]


def solve(
    f: Rhs,
    t0: float,
    y0: np.ndarray,
    t_out: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_fixed: Optional[float] = None,
    max_steps: int = 100_000,
) -> tuple[np.ndarray, SolveStats]:
    """Integrate y' = f(t, y) from t0 and return y at each time in ``t_out``.

    Returns an array of shape ``(n_samples, len(t_out), n_states)``. The solver
    lands exactly on every output time instead of interpolating, so outputs
    carry the full 5th-order accuracy. With ``h_fixed`` it takes fixed steps
    (used only by the convergence-order test).
    """
    y = np.array(y0, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    t_out = np.asarray(t_out, dtype=float)
    out = np.empty((y.shape[0], len(t_out), y.shape[1]))
    st = SolveStats()
    t = float(t0)
    k1 = f(t, y)
    st.fevals += 1
    h = h_fixed if h_fixed else _initial_step(f, t, y, k1, rtol, atol)
    st.fevals += 0 if h_fixed else 1

    for i, target in enumerate(t_out):
        while target - t > 1e-12 * max(1.0, abs(target)):
            if st.steps + st.rejected >= max_steps:
                raise SolverError(f"exceeded {max_steps} steps at t={t}")
            hh = min(h, target - t)
            y5, err, k7 = _stages(f, t, y, hh, k1)
            st.fevals += 6
            if h_fixed:
                accept, enorm = True, 0.0
            else:
                sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
                per_sample = np.sqrt(np.mean((err / sc) ** 2, axis=1))
                enorm = float(np.max(per_sample)) if np.all(np.isfinite(per_sample)) else np.inf
                accept = enorm <= 1.0
            if accept:
                t += hh
                y, k1 = y5, k7      # FSAL: last stage is the next step's first
                st.steps += 1
                st.h_min = min(st.h_min, hh)
                if not h_fixed:
                    fac = MAX_FACTOR if enorm == 0 else min(MAX_FACTOR, SAFETY * enorm ** -0.2)
                    # don't let a short landing step shrink the next step
                    h = max(h, hh * fac) if hh < h else hh * fac
            else:
                st.rejected += 1
                if not np.isfinite(enorm):
                    h = hh * MIN_FACTOR
                else:
                    h = hh * max(MIN_FACTOR, SAFETY * enorm ** -0.2)
                if h < 1e-12 * max(1.0, abs(t)):
                    raise SolverError(f"step size underflow at t={t}")
        out[:, i, :] = y
    return out, st
