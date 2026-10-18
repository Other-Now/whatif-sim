"""Question -> validated ModelSpec, in three stages with a repair loop.

    stage 1  propose_structure   -> structural checks (names, units, endpoints)
    stage 2  write_equations     -> full validation incl. units, loops, a test simulation
    stage 3  map_evidence        -> allowlist + unit checks, then CODE fills the numbers
                                    from the FRED snapshot and validates again

A rejected submission goes back to the model as an ``is_error`` tool_result
carrying the validator's codes and messages, up to ``max_repairs`` times per
stage. Every decision is appended to a JSONL log.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from ..evidence import DATA, EvidenceError, fill, manifest, transform_unit
from ..model import ModelSpec
from ..units import Unit, UnitError
from ..validate import Issue, Report, validate
from .providers import Provider

STAGES = ("propose_structure", "write_equations", "map_evidence")


class BuildFailed(RuntimeError):
    def __init__(self, stage: str, log: list[dict]):
        super().__init__(f"stage {stage} still failing after repairs")
        self.stage = stage
        self.log = log


def _structure_issues(s: dict) -> list[Issue]:
    out: list[Issue] = []
    seen: set[str] = set()
    units: dict[str, Unit] = {}
    try:
        Unit.parse(s.get("time_unit", ""))
    except UnitError as e:
        out.append(Issue("UNIT_PARSE", "time_unit", str(e)))
    if not s.get("horizon", 0) > 0:
        out.append(Issue("SCHEMA", "horizon", "horizon must be > 0"))
    for kind in ("stocks", "flows"):
        for it in s.get(kind, []):
            n = it.get("name", "")
            if not n.isidentifier():
                out.append(Issue("BAD_NAME", n, f"{n!r} is not an identifier"))
            if n in seen:
                out.append(Issue("DUPLICATE_NAME", n, f"{n!r} used twice"))
            seen.add(n)
            if kind == "stocks":
                try:
                    units[n] = Unit.parse(it.get("unit", ""))
                except UnitError as e:
                    out.append(Issue("UNIT_PARSE", n, str(e)))
    if not s.get("stocks"):
        out.append(Issue("SCHEMA", "stocks", "need at least one stock"))
    for f in s.get("flows", []):
        ends = [f.get("from"), f.get("to")]
        if not any(ends):
            out.append(Issue("FLOW_ENDPOINTS", f["name"], "flow connects nothing"))
        for e in ends:
            if e and e not in units:
                out.append(Issue("FLOW_ENDPOINTS", f["name"], f"{e!r} is not a stock"))
        if all(e in units for e in ends) and units[ends[0]] != units[ends[1]]:
            out.append(Issue("FLOW_ENDPOINTS", f["name"], "connects stocks with different units"))
    if not s.get("outputs"):
        out.append(Issue("SCHEMA", "outputs", "need at least one output"))
    return out


def _assemble(question: str, s: dict, eq: dict) -> tuple[Optional[ModelSpec], list[Issue]]:
    issues = []
    for st in s["stocks"]:
        if st["name"] not in eq.get("initial", {}):
            issues.append(Issue("SCHEMA", st["name"], "no initial value given"))
    for f in s["flows"]:
        if f["name"] not in eq.get("rates", {}):
            issues.append(Issue("SCHEMA", f["name"], "no rate expression given"))
    if issues:
        return None, issues
    data = {
        "name": s["name"], "question": question, "time_unit": s["time_unit"],
        "horizon": s["horizon"], "dt_out": s.get("dt_out", 1),
        "stocks": [{**st, "initial": eq["initial"][st["name"]]} for st in s["stocks"]],
        "flows": [{**f, "rate": eq["rates"][f["name"]]} for f in s["flows"]],
        "aux": eq.get("aux", []),
        "params": [{**p, "source": {"kind": "assumption", "note": "stage-2 range, not yet reviewed"}}
                   for p in eq.get("params", [])],
        "outputs": s["outputs"],
        "scenarios": [eq["scenario"]],
    }
    try:
        return ModelSpec.model_validate(data), []
    except ValidationError as e:
        return None, [Issue("SCHEMA", ".".join(map(str, x["loc"])), x["msg"]) for x in e.errors()]


def _apply_evidence(spec: ModelSpec, ev: dict, base: Path) -> tuple[Optional[ModelSpec], list[Issue], dict]:
    issues: list[Issue] = []
    params = {p.name: p for p in spec.params}
    maps = {m["param"]: m for m in ev.get("mappings", [])}
    for n in params:
        if n not in maps:
            issues.append(Issue("SCHEMA", n, "parameter not mapped (map to fred or keep as assumption)"))
    for n in maps:
        if n not in params:
            issues.append(Issue("UNKNOWN_NAME", n, "mapping for a parameter that doesn't exist"))
    have = manifest(base)
    fred: dict[str, tuple[str, list[str]]] = {}
    spec = spec.model_copy(deep=True)
    for p in spec.params:
        m = maps.get(p.name)
        if m is None:
            continue
        if m["source"] == "fred":
            try:
                u = transform_unit(m.get("transform", ""), m.get("series", []))
            except EvidenceError as e:
                issues.append(Issue(e.code, p.name, str(e)))
                continue
            if Unit.parse(u) != Unit.parse(p.unit):
                issues.append(Issue("EVIDENCE_UNIT", p.name,
                                    f"{m['transform']} gives [{u}] but the parameter is [{p.unit}]"))
                continue
            missing = [s for s in m["series"] if s not in have]
            if missing:
                issues.append(Issue("EVIDENCE_MISSING", p.name, f"no snapshot for {missing}"))
                continue
            fred[p.name] = (m["transform"], m["series"])
            # Whatever number the model put in "dist" is ignored: fill() computes it.
        else:
            if not m.get("note"):
                issues.append(Issue("PROVENANCE", p.name, "assumption needs a one-sentence justification note"))
            if m.get("dist"):
                p.dist = p.dist.model_validate(m["dist"])
            p.source.kind = "assumption"
            p.source.note = m.get("note", "")
    if issues:
        return None, issues, {}
    spec, notes = fill(spec, fred, base=base)
    return spec, [], notes


def build(question: str, provider: Provider, max_repairs: int = 2, log_path: Optional[Path] = None,
          evidence_base: Path = DATA) -> tuple[ModelSpec, Report, list[dict]]:
    log: list[dict] = []
    messages: list[dict] = [{"role": "user", "content":
                             f"Question: {question}\n\nBegin with stage 1: call propose_structure."}]
    state: dict = {}
    report: Optional[Report] = None

    def record(entry: dict) -> None:
        entry["ts"] = time.time()
        log.append(entry)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    for si, tool in enumerate(STAGES):
        for attempt in range(max_repairs + 1):
            call = provider.call(messages, tool)
            messages.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}]})
            if tool == "propose_structure":
                issues = _structure_issues(call.input)
                product = call.input
            elif tool == "write_equations":
                spec, issues = _assemble(question, state["structure"], call.input)
                if spec is not None:
                    report = validate(spec)
                    issues = report.issues
                product = spec
            else:
                spec, issues, notes = _apply_evidence(state["equations"], call.input, evidence_base)
                if spec is not None:
                    report = validate(spec)
                    issues = report.issues
                product = spec
            errors = [i for i in issues if i.severity == "error"]
            record({"stage": tool, "attempt": attempt, "verdict": "rejected" if errors else "accepted",
                    "issues": [asdict(i) for i in issues], "usage": call.usage,
                    **({"loops": [asdict(l) for l in report.loops]} if report and not errors and tool != "propose_structure" else {})})
            if not errors:
                state[{"propose_structure": "structure", "write_equations": "equations",
                       "map_evidence": "final"}[tool]] = product
                if si + 1 < len(STAGES):
                    msg = "Accepted."
                    if tool == "write_equations":
                        msg += " Loops found: " + "; ".join(
                            f"{l.polarity}: {' -> '.join(l.nodes)}" for l in report.loops) + "."
                    warns = [i for i in issues if i.severity == "warning"]
                    if warns:
                        msg += " Warnings: " + "; ".join(f"{w.code} at {w.where}: {w.message}" for w in warns) + "."
                    msg += f" Now stage {si + 2}: call {STAGES[si + 1]}."
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": call.id, "content": msg}]})
                break
            body = "Rejected by the validator. Fix these and resubmit " + tool + ":\n" + "\n".join(
                f"- {e.code} at {e.where}: {e.message}" for e in errors)
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call.id, "is_error": True, "content": body}]})
        else:
            raise BuildFailed(tool, log)
    return state["final"], report, log
