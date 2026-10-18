"""Seeded bugs for the validator: each mutation breaks the labor model in one
way an LLM plausibly would, and names the error code that must catch it."""
from __future__ import annotations

import copy


def _find(lst, name):
    return next(x for x in lst if x["name"] == name)


MUTATIONS = [
    {"name": "rate per year in a monthly model", "expect": "FLOW_UNIT",
     "fn": lambda m: _find(m["params"], "sep_rate").update(unit="1/year")},
    # caught first by the initial-value check (employed0 is kpeople), which
    # stops validation before the flow-endpoint check would also flag it
    {"name": "people vs kpeople", "expect": "UNIT_MISMATCH",
     "fn": lambda m: _find(m["stocks"], "employed").update(unit="people")},
    {"name": "add a rate to a stock", "expect": "UNIT_MISMATCH",
     "fn": lambda m: _find(m["aux"], "unemployment_rate").update(expr="unemployed / (employed + sep_rate)")},
    {"name": "bare number added to a flow", "expect": "UNIT_MISMATCH",
     "fn": lambda m: _find(m["flows"], "hires").update(rate="find_rate * demand * unemployed + 50")},
    {"name": "undefined name", "expect": "UNKNOWN_NAME",
     "fn": lambda m: _find(m["flows"], "hires").update(rate="job_finding * unemployed")},
    {"name": "code injection", "expect": "BAD_EXPR",
     "fn": lambda m: _find(m["aux"], "demand").update(expr="__import__('os').system('calc')")},
    {"name": "attribute access", "expect": "BAD_EXPR",
     "fn": lambda m: _find(m["aux"], "demand").update(expr="demand_floor.real")},
    {"name": "function not in whitelist", "expect": "BAD_EXPR",
     "fn": lambda m: _find(m["aux"], "demand").update(expr="sqrt(demand_floor)")},
    {"name": "algebraic loop between auxiliaries", "expect": "ALGEBRAIC_LOOP",
     "fn": lambda m: _find(m["aux"], "unemployment_rate").update(expr="unemployed / (employed + unemployed) * demand")},
    {"name": "flow from a stock to itself", "expect": "FLOW_ENDPOINTS",
     "fn": lambda m: _find(m["flows"], "hires").update(to="unemployed")},
    {"name": "flow to a stock that doesn't exist", "expect": "FLOW_ENDPOINTS",
     "fn": lambda m: _find(m["flows"], "hires").update(to="jobs")},
    {"name": "initial value uses another stock", "expect": "INITIAL_NOT_CONST",
     "fn": lambda m: _find(m["stocks"], "employed").update(initial="unemployed * 20")},
    {"name": "uniform low > high", "expect": "PARAM_RANGE",
     "fn": lambda m: _find(m["params"], "demand_sensitivity").update(dist={"kind": "uniform", "low": 5, "high": 0})},
    {"name": "lognormal gsd < 1", "expect": "PARAM_RANGE",
     "fn": lambda m: _find(m["params"], "find_rate").update(dist={"kind": "lognormal", "median": 0.25, "gsd": 0.5})},
    {"name": "duplicate name", "expect": "DUPLICATE_NAME",
     "fn": lambda m: m["aux"].append({"name": "employed", "expr": "employed0", "unit": "kpeople"})},
    {"name": "scenario changes unknown parameter", "expect": "SCENARIO_UNKNOWN_PARAM",
     "fn": lambda m: m["scenarios"][0]["changes"][0].update(param="hiring_rate")},
    {"name": "scenario starts after the horizon", "expect": "SCENARIO_TIME",
     "fn": lambda m: m["scenarios"][0]["changes"][0].update(from_t=99)},
    {"name": "output that doesn't exist", "expect": "OUTPUT_UNKNOWN",
     "fn": lambda m: m["outputs"].insert(0, "inflation")},
    {"name": "exp of a dimensioned quantity", "expect": "UNIT_MISMATCH",
     "fn": lambda m: _find(m["aux"], "demand").update(expr="exp(unemployed)")},
    {"name": "rate missing the stock (dimensionless rate)", "expect": "FLOW_UNIT",
     "fn": lambda m: _find(m["flows"], "separations").update(rate="sep_rate")},
    {"name": "unparseable unit", "expect": "UNIT_PARSE",
     "fn": lambda m: _find(m["params"], "sep_rate").update(unit="1/month/")},
    {"name": "'fred' source with no series", "expect": "PROVENANCE",
     "fn": lambda m: _find(m["params"], "sep_rate")["source"].update(series=[])},
    {"name": "constant outflow drains a stock below zero", "expect": "NEGATIVE_STOCK",
     "fn": lambda m: _find(m["flows"], "separations").update(rate="sep_rate * employed0 * 30")},
    {"name": "runaway feedback overflows", "expect": "NONFINITE",
     "fn": lambda m: _find(m["flows"], "hires").update(
         rate="find_rate * unemployed * exp(800 * employed / employed0)")},
]


def apply_mutation(model: dict, mutation: dict) -> dict:
    m = copy.deepcopy(model)
    mutation["fn"](m)
    return m
