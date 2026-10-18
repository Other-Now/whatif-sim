import numpy as np
import pytest

from whatif.expr import Expr, ExprError
from whatif.units import Unit, UnitError


def U(s):
    return Unit.parse(s)


def test_unit_algebra():
    assert U("kpeople/month") == U("kpeople*month^-1")
    assert U("kpeople/month") * U("month") == U("kpeople")
    assert U("usd/kpeople") * U("kpeople") == U("usd")
    assert U("1").dimensionless and U("").dimensionless
    assert U("month/month").dimensionless
    assert str(U("usd*month^-2")) == "usd/month^2"
    assert U("people") != U("kpeople")


@pytest.mark.parametrize("bad", ["kpeople/", "kpeople*", "2kpeople", "a^x", "a-b"])
def test_unit_parse_errors(bad):
    with pytest.raises(UnitError):
        U(bad)


ENV = {"E": U("kpeople"), "U": U("kpeople"), "f": U("1/month"), "k": U("1"), "w": U("usd/month")}


@pytest.mark.parametrize("text,unit", [
    ("f * U", "kpeople/month"),
    ("U / (E + U)", "1"),
    ("max(0.2, 1 - k * (U / (E + U)))", "1"),
    ("exp(-k)", "1"),
    ("E ** 2 / U", "kpeople"),
    ("abs(E - U)", "kpeople"),
    ("-f", "1/month"),
])
def test_unit_inference(text, unit):
    assert Expr.parse(text).units(ENV) == U(unit)


@pytest.mark.parametrize("text,code", [
    ("E + f", "UNIT_MISMATCH"),
    ("E + 1", "UNIT_MISMATCH"),          # literals are dimensionless
    ("exp(E)", "UNIT_MISMATCH"),
    ("min(E, f)", "UNIT_MISMATCH"),
    ("E ** k", "UNIT_MISMATCH"),
    ("Q * 2", "UNKNOWN_NAME"),
])
def test_unit_errors(text, code):
    with pytest.raises(ExprError) as e:
        Expr.parse(text).units(ENV)
    assert e.value.code == code


@pytest.mark.parametrize("text", [
    "__import__('os').system('x')", "E.real", "E if k else U", "E > U", "[E]", "lambda: 1",
    "sqrt(E)", "max(E, key=U)", "'a'", "E[0]",
])
def test_whitelist_rejects(text):
    with pytest.raises(ExprError) as e:
        Expr.parse(text)
    assert e.value.code == "BAD_EXPR"


def test_compile_is_vectorized_and_has_no_builtins():
    fn = Expr.parse("max(a, b) * 2").compile()
    out = fn({"a": np.array([1.0, 5.0]), "b": np.array([3.0, 2.0])})
    assert out.tolist() == [6.0, 10.0]
    # a name that would be a builtin is not one here
    with pytest.raises(NameError):
        Expr.parse("open").compile()({})
