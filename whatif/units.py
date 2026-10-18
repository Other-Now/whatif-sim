"""Dimensional analysis for model units.

A unit is a product of symbolic base tokens with integer exponents, written
like ``kpeople/month`` or ``usd*month^-1``. ``1`` (or an empty string) is
dimensionless. There are no scale prefixes: ``kpeople`` is its own token, so
mixing ``people`` and ``kpeople`` is a unit error, which is what we want.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\^(-?\d+))?$")


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class Unit:
    dims: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def parse(text: str) -> "Unit":
        s = (text or "").replace(" ", "")
        if s in ("", "1", "dimensionless"):
            return DIMENSIONLESS
        dims: dict[str, int] = {}
        # split into numerator / denominators: a*b/c/d^2
        parts = re.split(r"([*/])", s)
        sign = 1
        for p in parts:
            if p == "*":
                sign = 1
                continue
            if p == "/":
                sign = -1
                continue
            if p == "1":
                continue
            m = _TOKEN.match(p)
            if not m:
                raise UnitError(f"cannot parse unit {text!r} (bad token {p!r})")
            exp = int(m.group(2) or 1) * sign
            dims[m.group(1)] = dims.get(m.group(1), 0) + exp
        return Unit._make(dims)

    @staticmethod
    def _make(dims: dict[str, int]) -> "Unit":
        return Unit(tuple(sorted((k, v) for k, v in dims.items() if v != 0)))

    def __mul__(self, other: "Unit") -> "Unit":
        d = dict(self.dims)
        for k, v in other.dims:
            d[k] = d.get(k, 0) + v
        return Unit._make(d)

    def __truediv__(self, other: "Unit") -> "Unit":
        return self * other ** -1

    def __pow__(self, n: int) -> "Unit":
        return Unit._make({k: v * n for k, v in self.dims})

    @property
    def dimensionless(self) -> bool:
        return not self.dims

    def __str__(self) -> str:
        if not self.dims:
            return "1"
        num = [k if v == 1 else f"{k}^{v}" for k, v in self.dims if v > 0]
        den = [k if v == -1 else f"{k}^{-v}" for k, v in self.dims if v < 0]
        out = "*".join(num) or "1"
        for d in den:
            out += "/" + d
        return out


DIMENSIONLESS = Unit()
