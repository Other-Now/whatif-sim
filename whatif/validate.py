"""Deterministic validators. The LLM proposes; this module decides.

Every check returns ``Issue`` records with a stable code, so the repair loop
can quote the exact failure back to the model and the decision log can be
grepped. Errors block a model; warnings are reported but allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from pydantic import ValidationError

from .expr import Expr, ExprError
from .model import ModelSpec
from .sim import Compiled, topo_order
from .units import Unit, UnitError


@dataclass
class Issue:
    code: str
    where: str
    message: str
    severity: str = "error"


@dataclass
class Loop:
    nodes: list[str]
    polarity: str              # "R" reinforcing, "B" balancing, "?" unknown


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
            "loops": [asdict(l) for l in self.loops],
            "edges": self.edges,
        }


def validate_json(data: dict, dynamic: bool = True) -> tuple[Optional[ModelSpec], Report]:
    try:
        spec = ModelSpec.model_validate(data)
    except ValidationError as e:
        r = Report()
        for err in e.errors():
            r.issues.append(Issue("SCHEMA", ".".join(map(str, err["loc"])), err["msg"]))
        return None, r
    return spec, validate(spec, dynamic=dynamic)


def validate(spec: ModelSpec, dynamic: bool = True) -> Report:
    r = Report()
    err = lambda code, where, msg: r.issues.append(Issue(code, where, msg))
    warn = lambda code, where, msg: r.issues.append(Issue(code, where, msg, "warning"))

    # ---- names --------------------------------------------------------------
    seen: dict[str, str] = {}
    for kind, items in (("stock", spec.stocks), ("flow", spec.flows), ("aux", spec.aux), ("param", spec.params)):
        for it in items:
            if not it.name.isidentifier() or it.name in ("t", "min", "max", "exp", "log", "abs"):
                err("BAD_NAME", it.name, f"{it.name!r} is not a usable identifier")
            if it.name in seen:
                err("DUPLICATE_NAME", it.name, f"{it.name!r} defined as both {seen[it.name]} and {kind}")
            seen[it.name] = kind

    # ---- units --------------------------------------------------------------
    try:
        time = Unit.parse(spec.time_unit)
    except UnitError as e:
        err("UNIT_PARSE", "time_unit", str(e))
        return r
    units: dict[str, Unit] = {}
    for it in [*spec.stocks, *spec.aux, *spec.params]:
        try:
            units[it.name] = Unit.parse(it.unit)
        except UnitError as e:
            err("UNIT_PARSE", it.name, str(e))
    if r.errors:
        return r

    # ---- parameters ---------------------------------------------------------
    for p in spec.params:
        d = p.dist
        need = {"fixed": ["value"], "uniform": ["low", "high"], "triangular": ["low", "mode", "high"],
                "lognormal": ["median", "gsd"]}[d.kind]
        missing = [k for k in need if getattr(d, k) is None]
        if missing:
            err("PARAM_RANGE", p.name, f"{d.kind} distribution needs {missing}")
            continue
        if d.kind in ("uniform", "triangular") and not d.low < d.high:
            err("PARAM_RANGE", p.name, f"low {d.low} must be < high {d.high}")
        if d.kind == "triangular" and not d.low <= d.mode <= d.high:
            err("PARAM_RANGE", p.name, f"mode {d.mode} outside [{d.low}, {d.high}]")
        if d.kind == "lognormal" and not (d.median > 0 and d.gsd >= 1):
            err("PARAM_RANGE", p.name, "lognormal needs median > 0 and gsd >= 1")
        if p.source.kind == "fred" and not p.source.series:
            err("PROVENANCE", p.name, "source is 'fred' but names no series")
        if p.source.kind == "assumption" and d.kind != "fixed" and not p.source.note:
            warn("PROVENANCE", p.name, "assumed range has no justification note")

    # ---- expressions --------------------------------------------------------
    exprs: dict[str, Expr] = {}

    def parse(where: str, text: str) -> Optional[Expr]:
        try:
            e = Expr.parse(text)
        except ExprError as x:
            err(x.code, where, str(x))
            return None
        exprs[where] = e
        return e

    param_units = {p.name: units[p.name] for p in spec.params}
    for s in spec.stocks:
        e = parse(s.name + ".initial", s.initial)
        if e is None:
            continue
        bad = [n for n in e.names if n not in param_units]
        if bad:
            err("INITIAL_NOT_CONST", s.name, f"initial value may use parameters only, not {bad}")
            continue
        _check_unit(r, s.name + ".initial", e, param_units, units[s.name])

    aux_parsed = {a.name: parse(a.name, a.expr) for a in spec.aux}
    flow_parsed = {f.name: parse(f.name, f.rate) for f in spec.flows}
    if r.errors:
        return r

    # aux may depend on each other; a cycle there has no solution order
    try:
        aux_order = topo_order({n: {d for d in e.names if d in aux_parsed} for n, e in aux_parsed.items()})
    except ValueError as cyc:
        err("ALGEBRAIC_LOOP", "aux", f"auxiliaries depend on each other in a cycle: {cyc}")
        return r
    for name in aux_order:
        a = next(x for x in spec.aux if x.name == name)
        _check_unit(r, name, aux_parsed[name], units, units[name])

    flow_units: dict[str, Unit] = {}
    for f in spec.flows:
        if f.from_ is None and f.to is None:
            err("FLOW_ENDPOINTS", f.name, "flow connects nothing (both ends outside the model)")
            continue
        ends = [x for x in (f.from_, f.to) if x is not None]
        unknown = [x for x in ends if x not in {s.name for s in spec.stocks}]
        if unknown:
            err("FLOW_ENDPOINTS", f.name, f"flow endpoint(s) {unknown} are not stocks")
            continue
        if f.from_ == f.to:
            err("FLOW_ENDPOINTS", f.name, "flow goes from a stock to itself")
            continue
        if len(ends) == 2 and units[ends[0]] != units[ends[1]]:
            err("FLOW_ENDPOINTS", f.name, f"connects stocks with different units [{units[ends[0]]}] and [{units[ends[1]]}]")
            continue
        expected = units[ends[0]] / time
        got = _check_unit(r, f.name, flow_parsed[f.name], units, None)
        if got is not None and got != expected:
            err("FLOW_UNIT", f.name, f"rate has units [{got}] but stock/time is [{expected}]")
        flow_units[f.name] = expected

    # ---- references and coverage -------------------------------------------
    for o in spec.outputs:
        if o not in units:
            err("OUTPUT_UNKNOWN", o, f"output {o!r} is not a stock or auxiliary")
        elif o in param_units:
            err("OUTPUT_UNKNOWN", o, f"output {o!r} is a parameter, not a model result")
    for sc in spec.scenarios:
        for c in sc.changes:
            if c.param not in param_units:
                err("SCENARIO_UNKNOWN_PARAM", sc.name, f"scenario changes unknown parameter {c.param!r}")
            if not (0 <= c.from_t <= spec.horizon):
                err("SCENARIO_TIME", sc.name, f"from_t {c.from_t} outside [0, {spec.horizon}]")
            if c.multiply < 0:
                err("SCENARIO_TIME", sc.name, "negative multiplier")
    used = set().union(*[e.names for e in exprs.values()]) if exprs else set()
    for p in spec.params:
        if p.name not in used:
            warn("PARAM_UNUSED", p.name, "parameter is never referenced")
    connected = {x for f in spec.flows for x in (f.from_, f.to) if x}
    for s in spec.stocks:
        if s.name not in connected:
            warn("STOCK_NO_FLOW", s.name, "stock has no flows, so it never changes")
    if spec.horizon <= 0 or spec.dt_out <= 0 or spec.dt_out > spec.horizon:
        err("SCHEMA", "horizon", "need 0 < dt_out <= horizon")
    if r.errors:
        return r

    # ---- structure: feedback loops -----------------------------------------
    comp = Compiled(spec)
    with np.errstate(all="ignore"):
        _loops(r, spec, comp, aux_parsed, flow_parsed)
    if not r.loops:
        warn("NO_FEEDBACK", "model", "no feedback loops; a what-if on this model is just arithmetic")

    # ---- dynamics: run it once at the parameter centres --------------------
    if dynamic:
        _dynamic_checks(r, spec, comp)
    return r


def _check_unit(r: Report, where: str, e: Expr, env, expected: Optional[Unit]) -> Optional[Unit]:
    try:
        u = e.units(env)
    except ExprError as x:
        r.issues.append(Issue(x.code, where, str(x)))
        return None
    if expected is not None and u != expected:
        r.issues.append(Issue("UNIT_MISMATCH", where, f"{e.text!r} has units [{u}] but declared [{expected}]"))
    return u


def _loops(r: Report, spec: ModelSpec, comp: Compiled, aux_parsed, flow_parsed) -> None:
    """Find feedback loops in the influence graph and sign each link.

    Nodes are stocks, auxiliaries and flows (parameters are constants, so they
    can't be in a loop). The sign of a link a -> b is the sign of db/da at the
    parameter centres, by central finite difference; flow -> stock is +1 for
    an inflow and -1 for an outflow by definition.
    """
    P = comp.center_params(1)
    y0 = comp.initial_state(P, 1)
    base = comp.env(P, y0)
    graph: dict[str, set[str]] = {n: set() for n in [*comp.stock_names, *aux_parsed, *flow_parsed]}
    sign: dict[tuple[str, str], int] = {}

    def perturbed(src: str, delta: float):
        y = y0.copy()
        P2 = dict(P)
        if src in comp.stock_idx:
            y[0, comp.stock_idx[src]] += delta
            return comp.env(P2, y)
        # perturb an aux: recompute downstream aux with this one overridden
        e = dict(P2)
        for name, j in comp.stock_idx.items():
            e[name] = y[:, j]
        for name in comp.aux_order:
            e[name] = comp.aux_fns[name](e)
            if name == src:
                e[name] = e[name] + delta
        return e

    flow_fns = {f.name: Expr.parse(f.rate).compile() for f in spec.flows}
    for tgt, e in [*aux_parsed.items(), *flow_parsed.items()]:
        for src in e.names:
            if src not in graph:
                continue
            graph[src].add(tgt)
            v = float(np.abs(base[src]).max())
            d = 1e-6 * (v if v > 0 else 1.0)
            up, dn = perturbed(src, d), perturbed(src, -d)
            fu = flow_fns[tgt](up) if tgt in flow_fns else up[tgt]
            fd = flow_fns[tgt](dn) if tgt in flow_fns else dn[tgt]
            deriv = float(np.ravel(fu - fd)[0])
            scale = 1e-9 * max(1.0, float(np.abs(np.ravel(fu)[0])))
            sign[(src, tgt)] = 0 if abs(deriv) < scale else (1 if deriv > 0 else -1)
    for f in spec.flows:
        if f.to:
            graph[f.name].add(f.to)
            sign[(f.name, f.to)] = 1
        if f.from_:
            graph[f.name].add(f.from_)
            sign[(f.name, f.from_)] = -1
    r.edges = [{"from": a, "to": b, "sign": s} for (a, b), s in sorted(sign.items())]

    for cyc in _cycles(graph):
        signs = [sign[(cyc[i], cyc[(i + 1) % len(cyc)])] for i in range(len(cyc))]
        pol = "?" if 0 in signs else ("R" if np.prod(signs) > 0 else "B")
        r.loops.append(Loop(cyc, pol))


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """All elementary cycles. Models are small (tens of nodes), so a plain DFS
    that only starts each cycle from its smallest node is enough."""
    nodes = sorted(graph)
    rank = {n: i for i, n in enumerate(nodes)}
    out: list[list[str]] = []

    def dfs(start: str, cur: str, path: list[str], onpath: set[str]) -> None:
        for nxt in sorted(graph[cur]):
            if nxt == start:
                out.append(list(path))
            elif rank[nxt] > rank[start] and nxt not in onpath:
                onpath.add(nxt)
                path.append(nxt)
                dfs(start, nxt, path, onpath)
                path.pop()
                onpath.discard(nxt)

    for s in nodes:
        dfs(s, s, [s], {s})
    return out


def _dynamic_checks(r: Report, spec: ModelSpec, comp: Compiled) -> None:
    for sc in [None, *spec.scenarios]:
        label = sc.name if sc else "baseline"
        P = comp.center_params(1)
        try:
            with np.errstate(all="ignore"):
                sim = comp.simulate(P, sc, rtol=1e-6, atol=1e-9)
        except Exception as x:  # solver blew up: report, don't crash validation
            r.issues.append(Issue("NONFINITE", label, f"simulation failed: {x}"))
            continue
        if not np.all(np.isfinite(sim.outputs)):
            r.issues.append(Issue("NONFINITE", label, "outputs contain NaN/inf at parameter centres"))
            continue
        for s in spec.stocks:
            if s.nonnegative:
                col = sim.states[..., comp.stock_idx[s.name]]
                if col.min() < -1e-9 * max(1.0, float(np.abs(col).max())):
                    r.issues.append(Issue("NEGATIVE_STOCK", f"{label}:{s.name}", "stock goes negative"))
