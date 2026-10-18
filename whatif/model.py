"""Stock-and-flow model spec (what the LLM produces) and its compiled form."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """Where a parameter's numbers came from. Carried through to the brief."""

    kind: Literal["fred", "assumption"]
    series: list[str] = []          # FRED series ids, for kind == "fred"
    transform: str = ""             # how the series became this parameter
    window: str = ""                # observation window used
    fetched_at: str = ""            # when the snapshot was taken
    note: str = ""


class Dist(BaseModel):
    kind: Literal["fixed", "uniform", "triangular", "lognormal"]
    value: Optional[float] = None   # fixed
    low: Optional[float] = None     # uniform / triangular
    high: Optional[float] = None
    mode: Optional[float] = None    # triangular
    median: Optional[float] = None  # lognormal
    gsd: Optional[float] = None     # lognormal geometric sd (>= 1)

    def center(self) -> float:
        if self.kind == "fixed":
            return float(self.value)
        if self.kind == "uniform":
            return 0.5 * (self.low + self.high)
        if self.kind == "triangular":
            return float(self.mode)
        return float(self.median)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.kind == "fixed":
            return np.full(n, float(self.value))
        if self.kind == "uniform":
            return rng.uniform(self.low, self.high, n)
        if self.kind == "triangular":
            return rng.triangular(self.low, self.mode, self.high, n)
        return self.median * np.exp(np.log(self.gsd) * rng.standard_normal(n))


class Param(BaseModel):
    name: str
    unit: str
    dist: Dist
    source: Source = Source(kind="assumption")
    description: str = ""


class Stock(BaseModel):
    name: str
    unit: str
    initial: str                    # expression of parameters only
    nonnegative: bool = True
    description: str = ""


class Flow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    from_: Optional[str] = Field(default=None, alias="from")   # None = outside the model
    to: Optional[str] = None
    rate: str
    description: str = ""


class Aux(BaseModel):
    name: str
    expr: str
    unit: str
    description: str = ""


class Change(BaseModel):
    param: str
    multiply: float
    from_t: float = 0.0


class Scenario(BaseModel):
    name: str
    description: str = ""
    changes: list[Change]


class ModelSpec(BaseModel):
    name: str
    question: str = ""
    time_unit: str
    horizon: float
    dt_out: float = 1.0
    stocks: list[Stock]
    flows: list[Flow] = []
    aux: list[Aux] = []
    params: list[Param] = []
    outputs: list[str]
    scenarios: list[Scenario] = []

    def scenario(self, name: Optional[str]) -> Optional[Scenario]:
        if name is None:
            return None
        for s in self.scenarios:
            if s.name == name:
                return s
        raise KeyError(f"no scenario {name!r}")

    def random_params(self) -> list[Param]:
        return [p for p in self.params if p.dist.kind != "fixed"]

    def dump(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)
