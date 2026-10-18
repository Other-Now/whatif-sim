"""System prompt and tool schemas for the model builder.

Everything in here is sent byte-identical on every call (all three stages and
every repair), with the cache breakpoint on the system prompt. The order the
API caches in is tools -> system -> messages, so the *tools list must not
change between stages* either: every call sends all three tools and picks one
with ``tool_choice``. Changing ``tool_choice`` only invalidates the message
part of the cache, not the tools+system prefix.
"""
from __future__ import annotations

from ..evidence import CATALOG, TRANSFORMS
from ..expr import FUNCS

_catalog = "\n".join(f"- {k}: {d} [unit: {u}]" for k, (d, u) in CATALOG.items())
_transforms = "\n".join(f"- {k} ({n} series): {d}" for k, (n, d) in TRANSFORMS.items())

SYSTEM = f"""You build small system-dynamics (stock-and-flow) models that answer a
"what if" question. You work in three stages, each submitted through a tool.
Deterministic validators check every submission. If a submission fails, you
receive the validator's error codes and messages and must resubmit a corrected
version of the SAME stage. Never argue with a validator; fix the model.

## Stage 1: propose_structure
Choose the stocks (things that accumulate), the flows between them, the
outputs that answer the question, and the time unit and horizon.
- Prefer 2 to 5 stocks. A small model you can defend beats a big one.
- A flow moves material from one stock to another. `from` or `to` may be null
  for a flow that enters from, or leaves to, outside the model.
- A flow may only connect stocks that have the same unit.
- Outputs must be stocks or auxiliaries (auxiliaries are defined in stage 2;
  you may name planned ones now).

## Stage 2: write_equations
Write every flow rate, every auxiliary, each stock's initial value, the
parameters, and the scenario.

Expression language. A whitelisted subset of Python arithmetic:
- operators: + - * / ** (the exponent must be an integer literal unless the
  base is dimensionless)
- functions: {", ".join(sorted(FUNCS))}. min/max/abs take arguments of one unit;
  exp/log take one dimensionless argument.
- names: stocks, auxiliaries and parameters. Nothing else. There is no `t`,
  no if/else, no comparisons, no attribute access, no strings.
- numeric literals are DIMENSIONLESS. You may write `1 - x` only if x is
  dimensionless. Any number with a unit must be a named parameter.

Unit rules (checked exactly, there is no automatic conversion):
- A unit is a product of tokens with integer powers: `kpeople`, `1/month`,
  `usd*month^-1`. `1` means dimensionless.
- Tokens are symbols, not scales: `people` and `kpeople` are different units
  and can't be added. Pick one and stay with it.
- Every flow rate must have units [stock unit / time unit].
- Only like units can be added, subtracted, or compared in min/max.
- Each auxiliary declares a unit; the inferred unit must match exactly.

Initial values: an expression of parameters only.
Auxiliaries may depend on stocks, parameters and other auxiliaries, but not in
a cycle (that is an algebraic loop with no solution order).

Parameters: give each a unit and a distribution. In stage 2, give your best
plausible range as `uniform` or `triangular` (or `fixed` for definitional
constants). Stage 3 may replace ranges with data. Do not invent precise
"data" values; a range with an honest justification is better.

Scenario: express the what-if as multipliers on parameters from a time
`from_t` onward, e.g. {{"param": "find_rate", "multiply": 0.8, "from_t": 3}}.

A good model has at least one feedback loop: some stock must affect its own
flows, directly or through auxiliaries. Without one, a what-if is just
arithmetic. The validator reports the loops it finds and their polarity
(reinforcing or balancing), computed from the equations.

## Stage 3: map_evidence
For each parameter, either map it to data or keep it as an assumption.
To map it to data, choose series from this catalog ONLY:
{_catalog}

and one transform that code will compute (you never supply the number):
{_transforms}

The transform's output unit must equal the parameter's unit. For example
`flow_over_lagged_level` on [LNS17100000, LNS13000000] gives 1/month.
For an assumption, give a distribution and a one-sentence note saying why that
range is reasonable. The note appears in the final brief next to the number.

## General
- Names: lower_snake_case identifiers, unique across stocks, flows,
  auxiliaries and parameters.
- Keep descriptions to one short sentence.
- Validator error codes you may see: SCHEMA, BAD_NAME, DUPLICATE_NAME,
  UNIT_PARSE, BAD_EXPR, UNKNOWN_NAME, UNIT_MISMATCH, FLOW_UNIT,
  FLOW_ENDPOINTS, INITIAL_NOT_CONST, ALGEBRAIC_LOOP, PARAM_RANGE, PROVENANCE,
  OUTPUT_UNKNOWN, SCENARIO_UNKNOWN_PARAM, SCENARIO_TIME, NONFINITE,
  NEGATIVE_STOCK, EVIDENCE_UNKNOWN_SERIES, EVIDENCE_TRANSFORM, EVIDENCE_UNIT.
  Each comes with the location and a message; fix exactly what it says.
"""

_dist = {
    "type": "object",
    "properties": {
        "kind": {"enum": ["fixed", "uniform", "triangular", "lognormal"]},
        "value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
        "mode": {"type": "number"}, "median": {"type": "number"}, "gsd": {"type": "number"},
    },
    "required": ["kind"],
}

TOOLS = [
    {
        "name": "propose_structure",
        "description": "Stage 1: stocks, flows, outputs, time unit and horizon.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "time_unit": {"type": "string"},
                "horizon": {"type": "number"},
                "stocks": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "unit": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["name", "unit"]}},
                "flows": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "from": {"type": ["string", "null"]},
                    "to": {"type": ["string", "null"]}, "description": {"type": "string"}},
                    "required": ["name", "from", "to"]}},
                "outputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "time_unit", "horizon", "stocks", "flows", "outputs"],
        },
    },
    {
        "name": "write_equations",
        "description": "Stage 2: equations, parameters with ranges, initial values, scenario.",
        "input_schema": {
            "type": "object",
            "properties": {
                "initial": {"type": "object", "additionalProperties": {"type": "string"},
                            "description": "stock name -> initial-value expression"},
                "rates": {"type": "object", "additionalProperties": {"type": "string"},
                          "description": "flow name -> rate expression"},
                "aux": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "expr": {"type": "string"}, "unit": {"type": "string"},
                    "description": {"type": "string"}}, "required": ["name", "expr", "unit"]}},
                "params": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "unit": {"type": "string"}, "dist": _dist,
                    "description": {"type": "string"}}, "required": ["name", "unit", "dist"]}},
                "scenario": {"type": "object", "properties": {
                    "name": {"type": "string"}, "description": {"type": "string"},
                    "changes": {"type": "array", "items": {"type": "object", "properties": {
                        "param": {"type": "string"}, "multiply": {"type": "number"},
                        "from_t": {"type": "number"}}, "required": ["param", "multiply", "from_t"]}}},
                    "required": ["name", "changes"]},
            },
            "required": ["initial", "rates", "aux", "params", "scenario"],
        },
    },
    {
        "name": "map_evidence",
        "description": "Stage 3: map each parameter to a FRED series + transform, or keep it as a justified assumption.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mappings": {"type": "array", "items": {"type": "object", "properties": {
                    "param": {"type": "string"},
                    "source": {"enum": ["fred", "assumption"]},
                    "series": {"type": "array", "items": {"type": "string"}},
                    "transform": {"type": "string"},
                    "dist": _dist,
                    "note": {"type": "string"}},
                    "required": ["param", "source"]}},
            },
            "required": ["mappings"],
        },
    },
]
