import json
import math
import os
from pathlib import Path

import pytest

from whatif.model import ModelSpec

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "examples" / "labor_market.template.json"


@pytest.fixture
def labor_json() -> dict:
    return json.loads(TEMPLATE.read_text())


@pytest.fixture
def labor(labor_json) -> ModelSpec:
    return ModelSpec.model_validate(labor_json)


@pytest.fixture
def fred_dir(tmp_path) -> Path:
    """A SYNTHETIC FRED snapshot (smooth made-up series), so tests never need
    the network. Rates are built to be exactly 0.25 and 0.01 per month at the
    end of the window, with a small seasonal wiggle."""
    d = tmp_path / "fred"
    d.mkdir()
    months = [f"{2020 + i // 12}-{i % 12 + 1:02d}-01" for i in range(60)]
    E = [160000 + 50 * i for i in range(60)]
    U = [7000 + 10 * math.sin(i) for i in range(60)]
    UE = [None] + [0.25 * (1 + 0.02 * math.cos(i)) * U[i - 1] for i in range(1, 60)]
    EU = [None] + [0.01 * (1 + 0.02 * math.sin(i)) * E[i - 1] for i in range(1, 60)]
    series = {"LNS12000000": E, "LNS13000000": U, "LNS17100000": UE, "LNS17400000": EU}
    man = {}
    for sid, vals in series.items():
        lines = ["observation_date," + sid] + [f"{m},{'.' if v is None else round(v, 3)}" for m, v in zip(months, vals)]
        (d / f"{sid}.csv").write_text("\n".join(lines) + "\n")
        man[sid] = {"title": "synthetic", "fetched_at": "2026-01-01T00:00:00+00:00", "observations": 60,
                    "first": months[0], "last": months[-1]}
    (d / "manifest.json").write_text(json.dumps(man))
    return d


@pytest.fixture
def db_url() -> str:
    url = os.environ.get("WHATIF_TEST_DB")
    if not url:
        pytest.skip("set WHATIF_TEST_DB to run Postgres tests")
    return url
