"""The decision brief: a Markdown page built from the run, by code.

Every number in it is traced: model results come from the Monte Carlo summary,
and every input says whether it is FRED data (series, window, fetch time) or
an assumption (with its stated justification).
"""
from __future__ import annotations

from typing import Optional

from .model import ModelSpec
from .validate import Report


def _fmt(x: float, unit: str) -> str:
    if unit == "1" and abs(x) < 1:
        return f"{100 * x:.2f}%"
    return f"{x:,.4g}"


def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def render(spec: ModelSpec, summary: dict, report: Report, sobol: Optional[dict] = None,
           scenario: Optional[str] = None) -> str:
    out_name = spec.outputs[0]
    unit = next((a.unit for a in spec.aux if a.name == out_name),
                next((s.unit for s in spec.stocks if s.name == out_name), ""))
    T = summary["t"][-1]
    L = [f"# What-if brief: {spec.name}", "", f"**Question.** {spec.question}", ""]
    sc = spec.scenario(scenario) if scenario else None
    if sc:
        ch = "; ".join(f"`{c.param}` x{c.multiply} from t={c.from_t:g}" for c in sc.changes)
        L += [f"**Scenario.** {sc.description or sc.name} ({ch}).", ""]
    tr = summary["tracked"]
    eff = _pp if unit == "1" else (lambda x: f"{x:+,.4g} {unit}")
    b = summary["baseline"][out_name]
    L += ["## Answer", ""]
    if sc:
        s = summary["scenario"][out_name]
        conv = summary["convergence"]["mean"]
        hw = f"{100 * conv['halfwidth']:.3f} pp" if unit == "1" else f"{conv['halfwidth']:.3g} {unit}"
        L += [
            f"At {spec.time_unit} {T:g}, `{out_name}` changes by **{eff(tr['mean'])}** versus baseline "
            f"(90% of runs between {eff(tr['p05'])} and {eff(tr['p95'])}).",
            "",
            f"| | median at t={T:g} | 90% band |",
            "|---|---|---|",
            f"| baseline | {_fmt(b['p50'][-1], unit)} | {_fmt(b['p05'][-1], unit)} - {_fmt(b['p95'][-1], unit)} |",
            f"| scenario | {_fmt(s['p50'][-1], unit)} | {_fmt(s['p05'][-1], unit)} - {_fmt(s['p95'][-1], unit)} |",
            "",
            f"Monte Carlo: {summary['samples']:,} parameter draws in {summary['batches']} batches "
            f"({summary['stopped']}); the mean effect's 95% half-width is {hw}. "
            "Baseline and scenario share the same draws (common random numbers).",
        ]
    else:
        L += [f"At t={T:g}, median `{out_name}` is {_fmt(b['p50'][-1], unit)} "
              f"(90%: {_fmt(b['p05'][-1], unit)} - {_fmt(b['p95'][-1], unit)})."]
    L.append("")

    if sobol:
        L += ["## What drives the uncertainty", "",
              "Sobol indices of the answer above (share of its variance; total includes interactions).", "",
              "| parameter | first-order | total | source |", "|---|---|---|---|"]
        src = {p.name: p.source.kind for p in spec.params}
        rows = sorted(zip(sobol["params"], sobol["S1"], sobol["ST"]), key=lambda r: -r[2])
        for n, s1, st in rows:
            L.append(f"| `{n}` | {s1:.2f} | {st:.2f} | {src[n]} |")
        top = rows[0]
        if src[top[0]] == "assumption":
            L += ["", f"> The largest driver, `{top[0]}`, is an **assumption**, not data. "
                  "Narrowing it is worth more than refining anything else."]
        L.append("")

    L += ["## Structure", ""]
    for lp in report.loops:
        kind = {"R": "reinforcing", "B": "balancing", "?": "unsigned"}[lp.polarity]
        L.append(f"- {kind}: " + " -> ".join(lp.nodes + [lp.nodes[0]]))
    L.append("")

    L += ["## Inputs and where they came from", "", "| parameter | value | source |", "|---|---|---|"]
    for p in spec.params:
        d = p.dist
        val = {"fixed": lambda: f"{d.value:g}",
               "uniform": lambda: f"uniform({d.low:g}, {d.high:g})",
               "triangular": lambda: f"triangular({d.low:g}, {d.mode:g}, {d.high:g})",
               "lognormal": lambda: f"lognormal(median {d.median:.4g}, gsd {d.gsd:.3f})"}[d.kind]()
        if p.source.kind == "fred":
            ids = ", ".join(p.source.series)
            who = (f"BLS CPS {ids} (same ids on FRED)" if "BLS" in p.source.note else f"FRED {ids}")
            src = (f"{who}; "
                   f"{p.source.transform}; window {p.source.window or '?'}; fetched {p.source.fetched_at or '?'}")
        else:
            src = f"**assumption**: {p.source.note or '(no justification given)'}"
        L.append(f"| `{p.name}` [{p.unit}] | {val} | {src} |")
    warns = [i for i in report.issues if i.severity == "warning"]
    if warns:
        L += ["", "Validator warnings: " + "; ".join(f"{w.code} ({w.where})" for w in warns)]
    L += ["", f"Run fingerprint: `{summary['sha256'][:16]}`", ""]
    return "\n".join(L)
