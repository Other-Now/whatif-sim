"""FRED evidence: fetch series, keep provenance, turn them into parameters.

Numbers that claim to come from data are computed here, by code, from a
snapshot on disk. The LLM can say *which* series a parameter should come from
(from an allowlist), never what the number is.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import urllib.request
from pathlib import Path

from .model import Dist, ModelSpec

DATA = Path(__file__).resolve().parent.parent / "data" / "fred"

# The allowlist the LLM may map parameters to: series id -> (description, unit).
CATALOG = {
    "LNS12000000": ("Employment Level, thousands of persons, monthly, SA", "kpeople"),
    "LNS13000000": ("Unemployment Level, thousands of persons, monthly, SA", "kpeople"),
    "LNS17100000": ("Labor Force Flows Unemployed to Employed, thousands of persons per month, SA", "kpeople/month"),
    "LNS17400000": ("Labor Force Flows Employed to Unemployed, thousands of persons per month, SA", "kpeople/month"),
    "UNRATE": ("Unemployment Rate, percent, monthly, SA", "percent"),
}

# Transforms code knows how to compute. name -> (number of series, description).
TRANSFORMS = {
    "latest": (1, "latest observation of the series, as a fixed value"),
    "latest_percent_as_fraction": (1, "latest observation / 100, as a fixed value"),
    "share_latest": (2, "latest a / (a + b), as a fixed value"),
    "flow_over_lagged_level": (2, "monthly flow_t / level_(t-1) over the window, lognormal fitted (unit: 1/month)"),
}


# FRED re-publishes these BLS Current Population Survey series under the same
# ids (UNRATE is BLS LNS14000000). fred.stlouisfed.org's keyless CSV endpoint
# timed out both from the dev machine and from GitHub's runners, so the
# snapshot comes from the BLS API unless a FRED API key is configured.
BLS_ID = {"UNRATE": "LNS14000000"}


def _fetch_fred_api(series: str, start: str, key: str) -> tuple[str, str]:
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
           f"&observation_start={start}&file_type=json&api_key={key}")
    with urllib.request.urlopen(url, timeout=60) as r:
        obs = json.load(r)["observations"]
    text = f"observation_date,{series}\n" + "".join(f"{o['date']},{o['value']}\n" for o in obs)
    return text, url.replace(key, "***")


def _fetch_bls(series: str, start: str) -> tuple[str, str]:
    bls = BLS_ID.get(series, series)
    y0 = int(start[:4])
    y1 = dt.date.today().year
    rows: dict[str, str] = {}
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    while y0 <= y1:                       # keyless v1 serves at most 10 years per request
        body = json.dumps({"seriesid": [bls], "startyear": str(y0), "endyear": str(min(y1, y0 + 9))}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.load(r)
        if res.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS {bls}: {res.get('message')}")
        for d in res["Results"]["series"][0]["data"]:
            if d["period"].startswith("M") and d["period"] != "M13":
                rows[f"{d['year']}-{d['period'][1:]}-01"] = d["value"]
        y0 += 10
    text = f"observation_date,{series}\n" + "".join(f"{k},{rows[k]}\n" for k in sorted(rows))
    return text, f"{url}{bls}"


def _fetch_fredgraph(series: str, start: str) -> tuple[str, str]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
    req = urllib.request.Request(url, headers={"User-Agent": "whatif-sim/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode(), url


def fetch(series: str, start: str = "2015-01-01") -> tuple[str, str]:
    """Download one monthly series as CSV. Returns (csv text, source URL)."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return _fetch_fred_api(series, start, key)
    try:
        return _fetch_bls(series, start)
    except Exception:
        return _fetch_fredgraph(series, start)


def snapshot(series: list[str], out: Path = DATA) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for s in series:
        text, url = fetch(s)
        (out / f"{s}.csv").write_text(text, newline="")
        rows = load(s, out)
        manifest[s] = {
            "title": CATALOG.get(s, ("", ""))[0],
            "url": url,
            "publisher": "BLS (CPS)" if "bls.gov" in url else "FRED",
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "observations": len(rows),
            "first": rows[0][0], "last": rows[-1][0],
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(series: str, base: Path = DATA) -> list[tuple[str, float]]:
    rows = []
    with open(base / f"{series}.csv", newline="") as f:
        for rec in csv.reader(f):
            if not rec or rec[0] in ("DATE", "observation_date"):
                continue
            if rec[1] in (".", "", "-"):
                continue          # missing: "." on FRED, "-" on BLS (no October 2025 CPS)
            rows.append((rec[0], float(rec[1])))
    return rows


def manifest(base: Path = DATA) -> dict:
    p = base / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _lognormal_fit(xs: list[float]) -> tuple[float, float]:
    logs = [math.log(x) for x in xs]
    m = sum(logs) / len(logs)
    sd = math.sqrt(sum((l - m) ** 2 for l in logs) / (len(logs) - 1))
    return math.exp(m), math.exp(sd)


def _prev_month(d: str) -> str:
    y, m = int(d[:4]), int(d[5:7])
    y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return f"{y:04d}-{m:02d}-01"


def monthly_rate(flow: str, level: str, window: int, base: Path = DATA) -> tuple[list[float], str, str]:
    """flow_t / level_{t-1} for the last ``window`` months where both exist.

    t-1 is the previous *calendar* month, looked up by date. The first version
    paired adjacent rows, which divides by a two-month-old level across a gap
    (BLS has no October 2025 CPS). On the current snapshot that was latent:
    BLS also withholds the November 2025 flows, since they need matched
    October records. The test below makes the gap case explicit.
    """
    f = dict(load(flow, base))
    lv = dict(load(level, base))
    rates, dates = [], []
    for d1 in sorted(f):
        l0 = lv.get(_prev_month(d1))
        if l0:
            rates.append(f[d1] / l0)
            dates.append(d1)
    rates, dates = rates[-window:], dates[-window:]
    return rates, dates[0], dates[-1]


class EvidenceError(ValueError):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


def transform_unit(transform: str, series: list[str]) -> str:
    """Unit of the value a transform produces, so it can be checked against the
    parameter's declared unit before anything is computed."""
    if transform not in TRANSFORMS:
        raise EvidenceError("EVIDENCE_TRANSFORM", f"unknown transform {transform!r}; allowed: {sorted(TRANSFORMS)}")
    want = TRANSFORMS[transform][0]
    if len(series) != want:
        raise EvidenceError("EVIDENCE_TRANSFORM", f"{transform} takes {want} series, got {len(series)}")
    unknown = [s for s in series if s not in CATALOG]
    if unknown:
        raise EvidenceError("EVIDENCE_UNKNOWN_SERIES", f"series {unknown} not in the catalog {sorted(CATALOG)}")
    units = [CATALOG[s][1] for s in series]
    if transform == "latest":
        return units[0]
    if transform == "latest_percent_as_fraction":
        if units[0] != "percent":
            raise EvidenceError("EVIDENCE_UNIT", f"{series[0]} is not a percent series")
        return "1"
    if transform == "share_latest":
        if units[0] != units[1]:
            raise EvidenceError("EVIDENCE_UNIT", f"share of series with different units {units}")
        return "1"
    flow, level = units
    if flow != f"{level}/month":
        raise EvidenceError("EVIDENCE_UNIT", f"flow [{flow}] is not level [{level}] per month")
    return "1/month"


def apply(transform: str, series: list[str], window: int = 36, base: Path = DATA) -> tuple[Dist, str, dict]:
    """Compute a parameter distribution from the snapshot. Returns (dist, window text, notes)."""
    transform_unit(transform, series)
    man = manifest(base)
    missing = [s for s in series if s not in man]
    if missing:
        raise EvidenceError("EVIDENCE_MISSING", f"no snapshot for {missing}; run `python -m whatif fred-fetch`")
    if transform == "flow_over_lagged_level":
        rates, d0, d1 = monthly_rate(series[0], series[1], window, base)
        med, gsd = _lognormal_fit(rates)
        return (Dist(kind="lognormal", median=round(med, 6), gsd=round(gsd, 6)),
                f"{d0}..{d1} ({len(rates)} months)",
                {"min": min(rates), "max": max(rates), "median": med, "gsd": gsd, "n": len(rates)})
    data = [dict(load(s, base)) for s in series]
    last = min(max(d) for d in data)
    vals = [d[last] for d in data]
    v = {"latest": lambda: vals[0], "latest_percent_as_fraction": lambda: vals[0] / 100,
         "share_latest": lambda: vals[0] / (vals[0] + vals[1])}[transform]()
    return Dist(kind="fixed", value=round(v, 6)), last, {"date": last, "values": vals}


def fill(spec: ModelSpec, mapping: dict[str, tuple[str, list[str]]], window: int = 36,
         base: Path = DATA) -> tuple[ModelSpec, dict]:
    """Replace each mapped parameter's distribution with one computed from FRED."""
    man = manifest(base)
    spec = spec.model_copy(deep=True)
    P = {p.name: p for p in spec.params}
    notes = {}
    for name, (transform, series) in mapping.items():
        dist, win, note = apply(transform, series, window, base)
        p = P[name]
        p.dist = dist
        p.source.kind = "fred"
        p.source.series = list(series)
        p.source.transform = f"{transform}: {TRANSFORMS[transform][1]}"
        p.source.window = win
        p.source.fetched_at = min(man[s]["fetched_at"] for s in series)
        p.source.note = "via " + ", ".join(sorted({man[s].get("publisher", "FRED") for s in series}))
        notes[name] = note
    return spec, notes


LABOR_MAPPING = {
    "find_rate": ("flow_over_lagged_level", ["LNS17100000", "LNS13000000"]),
    "sep_rate": ("flow_over_lagged_level", ["LNS17400000", "LNS12000000"]),
    "employed0": ("latest", ["LNS12000000"]),
    "unemployed0": ("latest", ["LNS13000000"]),
    "u_ref": ("share_latest", ["LNS13000000", "LNS12000000"]),
}
