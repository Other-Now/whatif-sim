"""Turn a validated ModelSpec into something the solver can integrate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .expr import Expr
from .model import ModelSpec, Scenario
from .solver import SolveStats, solve


def topo_order(deps: dict[str, set[str]]) -> list[str]:
    """Order names so each comes after its dependencies. Raises on a cycle."""
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str, path: list[str]) -> None:
        s = state.get(n, 0)
        if s == 2:
            return
        if s == 1:
            cyc = path[path.index(n):] + [n]
            raise ValueError(" -> ".join(cyc))
        state[n] = 1
        for d in sorted(deps[n]):
            if d in deps:
                visit(d, path + [n])
        state[n] = 2
        order.append(n)

    for n in sorted(deps):
        visit(n, [])
    return order


@dataclass
class Simulation:
    t: np.ndarray                 # output times, shape (T,)
    outputs: np.ndarray           # (n_samples, T, n_outputs)
    states: np.ndarray            # (n_samples, T, n_stocks)
    stats: SolveStats


class Compiled:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.stock_names = [s.name for s in spec.stocks]
        self.stock_idx = {n: i for i, n in enumerate(self.stock_names)}
        aux_exprs = {a.name: Expr.parse(a.expr) for a in spec.aux}
        self.aux_order = topo_order({n: set(e.names) for n, e in aux_exprs.items()})
        self.aux_fns = {n: aux_exprs[n].compile() for n in self.aux_order}
        self.flow_fns = [
            (Expr.parse(f.rate).compile(),
             self.stock_idx.get(f.from_) if f.from_ else None,
             self.stock_idx.get(f.to) if f.to else None)
            for f in spec.flows
        ]
        self.init_fns = [Expr.parse(s.initial).compile() for s in spec.stocks]
        self.t_out = np.round(np.arange(0.0, spec.horizon + 1e-9, spec.dt_out), 12)

    # ---- parameters -------------------------------------------------------
    def sample_params(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        # Fixed iteration order (spec order) so a seed always means the same draws.
        return {p.name: p.dist.sample(rng, n) for p in self.spec.params}

    def center_params(self, n: int = 1) -> dict[str, np.ndarray]:
        return {p.name: np.full(n, p.dist.center()) for p in self.spec.params}

    @staticmethod
    def apply_scenario(P: dict[str, np.ndarray], sc: Optional[Scenario], t: float) -> dict[str, np.ndarray]:
        """Parameter values in force at time t (changes are right-continuous)."""
        if sc is None:
            return P
        Q = dict(P)
        for c in sc.changes:
            if t >= c.from_t:
                Q[c.param] = Q[c.param] * c.multiply
        return Q

    # ---- evaluation -------------------------------------------------------
    def env(self, P: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
        e = dict(P)
        for name, j in self.stock_idx.items():
            e[name] = y[:, j]
        for name in self.aux_order:
            e[name] = self.aux_fns[name](e)
        return e

    def rhs_for(self, P: dict[str, np.ndarray]):
        def f(t: float, y: np.ndarray) -> np.ndarray:
            e = self.env(P, y)
            dy = np.zeros_like(y)
            for fn, src, dst in self.flow_fns:
                r = np.broadcast_to(fn(e), (y.shape[0],))
                if src is not None:
                    dy[:, src] -= r
                if dst is not None:
                    dy[:, dst] += r
            return dy

        return f

    def initial_state(self, P: dict[str, np.ndarray], n: int) -> np.ndarray:
        return np.stack([np.broadcast_to(fn(P), (n,)).astype(float) for fn in self.init_fns], axis=1)

    def simulate(
        self,
        P: dict[str, np.ndarray],
        scenario: Optional[Scenario] = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> Simulation:
        n = len(next(iter(P.values()))) if P else 1
        y = self.initial_state(P, n)
        T = self.t_out
        horizon = T[-1]
        bps = sorted({c.from_t for c in (scenario.changes if scenario else []) if 0 < c.from_t < horizon})
        edges = [0.0, *bps, horizon]
        states = np.empty((n, len(T), y.shape[1]))
        states[:, 0, :] = y
        stats = SolveStats()
        for a, b in zip(edges[:-1], edges[1:]):
            Pseg = self.apply_scenario(P, scenario, a)
            seg_t = sorted({float(x) for x in T if a < x <= b} | {b})
            ys, st = solve(self.rhs_for(Pseg), a, y, np.array(seg_t), rtol=rtol, atol=atol)
            stats.add(st)
            for k, tt in enumerate(seg_t):
                hit = np.nonzero(np.isclose(T, tt))[0]
                if hit.size:
                    states[:, hit[0], :] = ys[:, k, :]
            y = ys[:, -1, :]
        return Simulation(T, self.outputs_from_states(P, scenario, states), states, stats)

    def outputs_from_states(self, P, scenario, states: np.ndarray) -> np.ndarray:
        n, nt, _ = states.shape
        out = np.empty((n, nt, len(self.spec.outputs)))
        for k, t in enumerate(self.t_out):
            e = self.env(self.apply_scenario(P, scenario, t), states[:, k, :])
            for j, name in enumerate(self.spec.outputs):
                out[:, k, j] = np.broadcast_to(e[name], (n,))
        return out
