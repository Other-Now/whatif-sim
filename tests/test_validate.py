import pytest

from tests.mutations import MUTATIONS, apply_mutation
from whatif.validate import validate, validate_json


def test_labor_model_is_valid(labor):
    r = validate(labor)
    assert r.ok, r.issues
    assert not [i for i in r.issues if i.severity == "warning"]


def test_loops_and_polarity(labor):
    loops = {tuple(l.nodes): l.polarity for l in validate(labor).loops}
    # unemployed -> hires (+), hires -> unemployed (-): more unemployed, more hires, fewer unemployed
    assert loops[("hires", "unemployed")] == "B"
    assert loops[("employed", "separations")] == "B"
    # unemployment up -> demand down -> fewer hires -> unemployment up
    assert loops[("demand", "hires", "unemployed", "unemployment_rate")] == "R"


def test_edge_signs(labor):
    edges = {(e["from"], e["to"]): e["sign"] for e in validate(labor).edges}
    assert edges[("unemployment_rate", "demand")] == -1
    assert edges[("demand", "hires")] == 1
    assert edges[("hires", "unemployed")] == -1
    assert edges[("hires", "employed")] == 1


@pytest.mark.parametrize("m", MUTATIONS, ids=[m["name"] for m in MUTATIONS])
def test_validator_catches_mutation(labor_json, m):
    _, rep = validate_json(apply_mutation(labor_json, m))
    assert not rep.ok
    assert m["expect"] in {i.code for i in rep.errors}, rep.issues


def test_schema_errors_are_reported_not_raised():
    spec, rep = validate_json({"name": "x"})
    assert spec is None and not rep.ok
    assert {i.code for i in rep.issues} == {"SCHEMA"}


def test_no_feedback_is_a_warning(labor_json):
    m = labor_json
    m["aux"][1]["expr"] = "demand_floor * 5"     # demand no longer depends on the state
    for f in m["flows"]:
        f["rate"] = {"hires": "find_rate * demand * unemployed0", "separations": "sep_rate * employed0"}[f["name"]]
    _, rep = validate_json(m)
    assert rep.ok
    assert "NO_FEEDBACK" in {i.code for i in rep.issues}
