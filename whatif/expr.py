"""The model expression language: a whitelisted subset of Python arithmetic.

Equations come from an LLM, so they are never ``eval``'d as written. They are
parsed with ``ast``, every node is checked against a whitelist, units are
inferred from the tree, and only then is the tree compiled with an empty
``__builtins__`` and numpy functions bound in.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .units import DIMENSIONLESS, Unit

FUNCS: dict[str, Callable] = {
    "min": np.minimum,
    "max": np.maximum,
    "exp": np.exp,
    "log": np.log,
    "abs": np.abs,
}
_SAME_UNIT_FUNCS = {"min", "max", "abs"}      # result has the arguments' unit
_DIMENSIONLESS_FUNCS = {"exp", "log"}          # argument must be dimensionless

_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


class ExprError(ValueError):
    """Raised with a stable ``code`` so the repair loop can quote it back."""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


@dataclass
class Expr:
    text: str
    tree: ast.Expression
    names: frozenset[str]

    @staticmethod
    def parse(text: str) -> "Expr":
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as e:
            raise ExprError("BAD_EXPR", f"syntax error in {text!r}: {e.msg}") from None
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                raise ExprError("BAD_EXPR", f"{type(node).__name__} not allowed in {text!r}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ExprError("BAD_EXPR", f"non-numeric constant in {text!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                    raise ExprError("BAD_EXPR", f"unknown function in {text!r}; allowed: {sorted(FUNCS)}")
                if node.keywords:
                    raise ExprError("BAD_EXPR", f"keyword arguments not allowed in {text!r}")
            if isinstance(node, ast.Name) and not _is_func_name(tree, node):
                names.add(node.id)
        return Expr(text, tree, frozenset(names))

    def units(self, env: Mapping[str, Unit]) -> Unit:
        return _infer(self.tree.body, env, self.text)

    def compile(self) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
        code = compile(self.tree, "<expr>", "eval")
        glb = {"__builtins__": {}, **FUNCS}

        def fn(env: Mapping[str, np.ndarray]) -> np.ndarray:
            return eval(code, glb, env)  # noqa: S307 - tree is whitelisted above

        return fn


def _is_func_name(tree: ast.AST, name: ast.Name) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.func is name:
            return True
    return False


def _infer(node: ast.AST, env: Mapping[str, Unit], text: str) -> Unit:
    if isinstance(node, ast.Constant):
        return DIMENSIONLESS
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExprError("UNKNOWN_NAME", f"{node.id!r} is not defined (in {text!r})")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return _infer(node.operand, env, text)
    if isinstance(node, ast.BinOp):
        a = _infer(node.left, env, text)
        if isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and float(node.right.value).is_integer()):
                if not a.dimensionless:
                    raise ExprError("UNIT_MISMATCH", f"non-integer power of a dimensioned quantity in {text!r}")
                if not _infer(node.right, env, text).dimensionless:
                    raise ExprError("UNIT_MISMATCH", f"exponent must be dimensionless in {text!r}")
                return DIMENSIONLESS
            return a ** int(node.right.value)
        b = _infer(node.right, env, text)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if a != b:
                raise ExprError(
                    "UNIT_MISMATCH",
                    f"cannot add/subtract [{a}] and [{b}] in {text!r}",
                )
            return a
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b
    if isinstance(node, ast.Call):
        fname = node.func.id
        args = [_infer(x, env, text) for x in node.args]
        if not args:
            raise ExprError("BAD_EXPR", f"{fname}() needs arguments in {text!r}")
        if fname in _DIMENSIONLESS_FUNCS:
            if len(args) != 1 or not args[0].dimensionless:
                raise ExprError("UNIT_MISMATCH", f"{fname}() needs one dimensionless argument in {text!r}")
            return DIMENSIONLESS
        if fname == "abs" and len(args) != 1:
            raise ExprError("BAD_EXPR", f"abs() takes one argument in {text!r}")
        if any(u != args[0] for u in args):
            raise ExprError(
                "UNIT_MISMATCH",
                f"{fname}() arguments have different units {[str(u) for u in args]} in {text!r}",
            )
        return args[0]
    raise ExprError("BAD_EXPR", f"unsupported syntax in {text!r}")
