import json
from pathlib import Path

import pytest

from whatif.brief import render
from whatif.evidence import EvidenceError, LABOR_MAPPING, apply, fill, transform_unit
from whatif.llm import BuildFailed, ScriptedProvider, build
from whatif.llm.prompts import SYSTEM
from whatif.mc import RunConfig, run_local
from whatif.validate import validate

SCRIPT = Path(__file__).parent / "fixtures" / "scripted_labor.json"


def script():
    return json.loads(SCRIPT.read_text())


def test_transform_units():
    assert transform_unit("flow_over_lagged_level", ["LNS17100000", "LNS13000000"]) == "1/month"
    assert transform_unit("latest", ["LNS12000000"]) == "kpeople"
    assert transform_unit("share_latest", ["LNS13000000", "LNS12000000"]) == "1"
    with pytest.raises(EvidenceError) as e:
        transform_unit("flow_over_lagged_level", ["LNS12000000", "LNS13000000"])
    assert e.value.code == "EVIDENCE_UNIT"
    with pytest.raises(EvidenceError) as e:
        transform_unit("latest", ["GDP"])
    assert e.value.code == "EVIDENCE_UNKNOWN_SERIES"


def test_flow_rate_from_snapshot(fred_dir):
    dist, window, note = apply("flow_over_lagged_level", ["LNS17100000", "LNS13000000"], 36, fred_dir)
    assert dist.kind == "lognormal" and abs(dist.median - 0.25) < 0.002
    assert note["n"] == 36 and window.endswith("(36 months)")


def test_flow_rate_skips_a_missing_level_month(fred_dir):
    # drop one level month: the next month's rate must be skipped, not
    # computed against the level two months back
    lines = (fred_dir / "LNS13000000.csv").read_text().splitlines()
    gap = lines[30].split(",")[0]
    lines[30] = gap + ",-"
    (fred_dir / "LNS13000000.csv").write_text("\n".join(lines) + "\n")
    _, window, note = apply("flow_over_lagged_level", ["LNS17100000", "LNS13000000"], 60, fred_dir)
    assert note["n"] == 58            # 59 possible rates, one loses its t-1 level
    assert abs(note["median"] - 0.25) < 0.002


def test_fill_stamps_provenance(labor, fred_dir):
    spec, _ = fill(labor, LABOR_MAPPING, base=fred_dir)
    p = {x.name: x for x in spec.params}
    assert p["find_rate"].source.series == ["LNS17100000", "LNS13000000"]
    assert p["find_rate"].source.fetched_at == "2026-01-01T00:00:00+00:00"
    assert p["employed0"].dist.value == 160000 + 50 * 59
    assert validate(spec).ok


def test_build_repairs_and_succeeds(fred_dir, tmp_path):
    prov = ScriptedProvider(script())
    log_path = tmp_path / "log.jsonl"
    spec, report, log = build("What if job finding falls 20%?", prov, log_path=log_path, evidence_base=fred_dir)
    assert [(e["stage"], e["verdict"]) for e in log] == [
        ("propose_structure", "accepted"),
        ("write_equations", "rejected"), ("write_equations", "accepted"),
        ("map_evidence", "rejected"), ("map_evidence", "accepted"),
    ]
    assert {i["code"] for i in log[1]["issues"]} == {"FLOW_UNIT"}
    assert {i["code"] for i in log[3]["issues"] if i["severity"] == "error"} == {"EVIDENCE_UNKNOWN_SERIES", "PROVENANCE"}
    assert len(log_path.read_text().splitlines()) == 5
    assert report.ok and any(l.polarity == "R" for l in report.loops)

    # the repair message quotes the code back to the model as an error tool_result
    repair = prov.requests[2]["messages"][-1]["content"][0]
    assert repair["is_error"] and "FLOW_UNIT" in repair["content"]

    # the made-up 0.99 in the script is ignored: the number comes from the snapshot
    find = next(p for p in spec.params if p.name == "find_rate")
    assert find.dist.kind == "lognormal" and abs(find.dist.median - 0.25) < 0.002


def test_cacheable_prefix_is_identical_on_every_call(fred_dir):
    prov = ScriptedProvider(script())
    build("q", prov, evidence_base=fred_dir)
    first = json.dumps([prov.requests[0]["tools"], prov.requests[0]["system"]], sort_keys=True)
    for r in prov.requests[1:]:
        assert json.dumps([r["tools"], r["system"]], sort_keys=True) == first
    assert prov.requests[0]["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # long enough to be cacheable (minimum is ~1-4k tokens depending on model; ~4 chars/token)
    assert len(SYSTEM) > 4 * 1024


def test_build_gives_up_after_max_repairs(fred_dir):
    s = script()
    bad = s[1]
    prov = ScriptedProvider([s[0], bad, bad, bad])
    with pytest.raises(BuildFailed) as e:
        build("q", prov, max_repairs=2, evidence_base=fred_dir)
    assert e.value.stage == "write_equations" and len(e.value.log) == 4


def test_brief_traces_every_input(labor, fred_dir):
    spec, _ = fill(labor, LABOR_MAPPING, base=fred_dir)
    res = run_local(spec, RunConfig(scenario="find_rate_down_20", batch_size=64, tol=1e-3))
    md = render(spec, res, validate(spec), None, "find_rate_down_20")
    assert "FRED LNS17100000, LNS13000000" in md
    assert "**assumption**: No direct data" in md
    assert "reinforcing" in md and "balancing" in md
    assert res["sha256"][:16] in md
