#!/usr/bin/env python3
"""Offline numeric falsification for declared theoretical-claim companions."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping


_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class NumericExpressionError(ValueError):
    """Raised when a falsification expression leaves the safe numeric subset."""


_POW_EXPONENT_LIMIT = 1024


def _finite_arithmetic(value: float | bool) -> float | bool:
    if isinstance(value, bool):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise NumericExpressionError("arithmetic produced a non-finite intermediate")
    return number


def _guarded_pow(base: float, exponent: float) -> float:
    # coerce to float and bound the exponent so a claim-authored inequality
    # like 10**10**10 overflows to inf (caught) instead of hanging on an
    # arbitrary-precision integer computation inside a gate.
    if abs(float(exponent)) > _POW_EXPONENT_LIMIT:
        raise NumericExpressionError("exponent magnitude exceeds the falsification limit")
    return float(base) ** float(exponent)


def _evaluate_node(node: ast.AST, names: Mapping[str, float]) -> float | bool:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise NumericExpressionError("only numeric and boolean constants are allowed")
    if isinstance(node, ast.Name):
        if node.id in names:
            return float(names[node.id])
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise NumericExpressionError(f"unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, names)
        if isinstance(node.op, ast.USub):
            return _finite_arithmetic(-float(value))
        if isinstance(node.op, ast.UAdd):
            return _finite_arithmetic(float(value))
        if isinstance(node.op, ast.Not):
            return not bool(value)
        raise NumericExpressionError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = float(_evaluate_node(node.left, names))
        right = float(_evaluate_node(node.right, names))
        operators = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: _guarded_pow(left, right),
            ast.Mod: lambda: left % right,
        }
        operation = operators.get(type(node.op))
        if operation is None:
            raise NumericExpressionError("unsupported binary operator")
        return _finite_arithmetic(operation())
    if isinstance(node, ast.BoolOp):
        values = [bool(_evaluate_node(value, names)) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise NumericExpressionError("unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, names)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _evaluate_node(comparator, names)
            if isinstance(operator, ast.Lt):
                passed = float(left) < float(right)
            elif isinstance(operator, ast.LtE):
                passed = float(left) <= float(right)
            elif isinstance(operator, ast.Gt):
                passed = float(left) > float(right)
            elif isinstance(operator, ast.GtE):
                passed = float(left) >= float(right)
            elif isinstance(operator, ast.Eq):
                passed = float(left) == float(right)
            elif isinstance(operator, ast.NotEq):
                passed = float(left) != float(right)
            else:
                raise NumericExpressionError("unsupported comparison operator")
            if not passed:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise NumericExpressionError("unsupported function call")
        if node.keywords:
            raise NumericExpressionError("keyword arguments are not allowed")
        arguments = [float(_evaluate_node(argument, names)) for argument in node.args]
        return _finite_arithmetic(float(_FUNCTIONS[node.func.id](*arguments)))
    raise NumericExpressionError(f"unsupported expression node: {type(node).__name__}")


def evaluate_numeric_expression(expression: str, point: Mapping[str, float]) -> float | bool:
    if not isinstance(expression, str) or not expression.strip():
        raise NumericExpressionError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise NumericExpressionError(f"invalid expression syntax: {exc.msg}") from exc
    try:
        value = _evaluate_node(tree, point)
    except (ArithmeticError, OverflowError) as exc:
        raise NumericExpressionError(f"numeric evaluation failed: {exc}") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise NumericExpressionError("expression returned a non-finite value")
    return value


def _expression_from_entry(entry: object, *keys: str) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _sign_holds(value: float, expected: str, tolerance: float) -> bool:
    normalized = expected.strip().lower().replace(" ", "_")
    if normalized in {"positive", ">", ">0"}:
        return value > tolerance
    if normalized in {"nonnegative", "non_negative", ">=", ">=0"}:
        return value >= -tolerance
    if normalized in {"negative", "<", "<0"}:
        return value < -tolerance
    if normalized in {"nonpositive", "non_positive", "<=", "<=0"}:
        return value <= tolerance
    if normalized in {"zero", "=", "==", "==0"}:
        return abs(value) <= tolerance
    raise NumericExpressionError(f"unknown comparative-statics sign: {expected}")


def _kernel_sample_points(
    domain: object,
    seed: object,
    sample_count: object,
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    if not isinstance(domain, dict) or not domain:
        return [], [{"reason": "falsification_domain_invalid"}]
    if not isinstance(seed, int) or isinstance(seed, bool):
        return [], [{"reason": "falsification_seed_invalid"}]
    if sample_count is None:
        count = 9
    elif (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 3
        or sample_count > 128
    ):
        return [], [{"reason": "falsification_sample_count_invalid"}]
    else:
        count = sample_count

    bounds: dict[str, tuple[float, float]] = {}
    for name, raw_bounds in sorted(domain.items()):
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or not isinstance(raw_bounds, list)
            or len(raw_bounds) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in raw_bounds
            )
            or float(raw_bounds[0]) >= float(raw_bounds[1])
        ):
            failures.append(
                {"reason": "falsification_domain_invalid", "name": name, "bounds": raw_bounds}
            )
            continue
        bounds[name] = (float(raw_bounds[0]), float(raw_bounds[1]))
    if failures:
        return [], failures

    rng = random.Random(seed)
    permutations: dict[str, list[int]] = {}
    for name in bounds:
        order = list(range(count))
        rng.shuffle(order)
        permutations[name] = order
    points: list[dict[str, float]] = []
    for index in range(count):
        point: dict[str, float] = {}
        for name, (low, high) in bounds.items():
            stratum = permutations[name][index]
            point[name] = low + ((stratum + 0.5) / count) * (high - low)
        points.append(point)
    points.extend(
        {
            name: low if position == 0 else (low + high) / 2.0 if position == 1 else high
            for name, (low, high) in bounds.items()
        }
        for position in range(3)
    )
    return points, []


def evaluate_falsification_spec(spec: object) -> list[dict[str, object]]:
    """Return one diagnostic per invalid declaration or falsified sampled point."""
    if not isinstance(spec, dict):
        return [{"reason": "falsification_spec_not_object"}]
    inequalities = spec.get("inequalities")
    comparative_statics = spec.get("comparative_statics")
    author_sample_points = spec.get("sample_points", [])
    failures: list[dict[str, object]] = []
    if not isinstance(inequalities, list):
        failures.append({"reason": "inequalities_not_list"})
        inequalities = []
    if not isinstance(comparative_statics, list):
        failures.append({"reason": "comparative_statics_not_list"})
        comparative_statics = []
    if not inequalities and not comparative_statics:
        failures.append({"reason": "falsification_spec_empty"})
    kernel_points, kernel_failures = _kernel_sample_points(
        spec.get("domain"),
        spec.get("seed"),
        spec.get("kernel_sample_count"),
    )
    failures.extend(kernel_failures)
    if not isinstance(author_sample_points, list):
        failures.append({"reason": "sample_points_not_list"})
        author_sample_points = []

    points = list(kernel_points)
    for author_index, raw_point in enumerate(author_sample_points):
        point_index = len(kernel_points) + author_index
        if not isinstance(raw_point, dict) or not raw_point:
            failures.append({"reason": "sample_point_not_object", "point_index": point_index})
            continue
        point: dict[str, float] = {}
        invalid = False
        for name, value in raw_point.items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                failures.append(
                    {
                        "reason": "invalid_sample_coordinate",
                        "point_index": point_index,
                        "name": name,
                        "value": value,
                    }
                )
                invalid = True
                continue
            point[name] = float(value)
        if not invalid:
            points.append(point)

    for inequality_index, entry in enumerate(inequalities):
        expression = _expression_from_entry(entry, "expression", "inequality", "predicate")
        if expression is None:
            failures.append(
                {"reason": "invalid_inequality", "inequality_index": inequality_index}
            )
            continue
        for point_index, point in enumerate(points):
            try:
                result = evaluate_numeric_expression(expression, point)
            except NumericExpressionError as exc:
                failures.append(
                    {
                        "reason": "inequality_evaluation_error",
                        "inequality_index": inequality_index,
                        "point_index": point_index,
                        "error": str(exc),
                    }
                )
                continue
            if not isinstance(result, bool) or not result:
                failures.append(
                    {
                        "reason": "inequality_violated",
                        "inequality_index": inequality_index,
                        "point_index": point_index,
                        "expression": expression,
                        "point": point,
                        "actual": result,
                    }
                )

    for static_index, entry in enumerate(comparative_statics):
        if not isinstance(entry, dict):
            failures.append(
                {"reason": "invalid_comparative_static", "comparative_static_index": static_index}
            )
            continue
        expected = entry.get("sign", entry.get("expected_sign"))
        if not isinstance(expected, str) or not expected.strip():
            failures.append(
                {
                    "reason": "comparative_static_missing_sign",
                    "comparative_static_index": static_index,
                }
            )
            continue
        tolerance_value = entry.get("tolerance", spec.get("tolerance", 1e-7))
        tolerance = (
            float(tolerance_value)
            if isinstance(tolerance_value, (int, float)) and not isinstance(tolerance_value, bool)
            else -1.0
        )
        if tolerance < 0 or not math.isfinite(tolerance):
            failures.append(
                {
                    "reason": "invalid_comparative_static_tolerance",
                    "comparative_static_index": static_index,
                }
            )
            continue
        derivative = _expression_from_entry(entry.get("derivative"), "expression")
        expression = _expression_from_entry(entry, "expression", "function")
        parameter = entry.get("parameter")
        if derivative is None and (
            expression is None or not isinstance(parameter, str) or not parameter.isidentifier()
        ):
            failures.append(
                {
                    "reason": "comparative_static_missing_expression_or_parameter",
                    "comparative_static_index": static_index,
                }
            )
            continue
        for point_index, point in enumerate(points):
            try:
                if derivative is not None:
                    value = float(evaluate_numeric_expression(derivative, point))
                else:
                    assert expression is not None and isinstance(parameter, str)
                    if parameter not in point:
                        raise NumericExpressionError(f"parameter absent from point: {parameter}")
                    base = point[parameter]
                    raw_step = entry.get("step")
                    step = (
                        float(raw_step)
                        if isinstance(raw_step, (int, float))
                        and not isinstance(raw_step, bool)
                        and float(raw_step) > 0
                        else 1e-6 * (1.0 + abs(base))
                    )
                    plus = dict(point)
                    minus = dict(point)
                    plus[parameter] = base + step
                    minus[parameter] = base - step
                    value = (
                        float(evaluate_numeric_expression(expression, plus))
                        - float(evaluate_numeric_expression(expression, minus))
                    ) / (2.0 * step)
                    if not math.isfinite(value):
                        raise NumericExpressionError(
                            "comparative-static calculation produced a non-finite intermediate"
                        )
                passed = _sign_holds(value, expected, tolerance)
            except NumericExpressionError as exc:
                failures.append(
                    {
                        "reason": "comparative_static_evaluation_error",
                        "comparative_static_index": static_index,
                        "point_index": point_index,
                        "error": str(exc),
                    }
                )
                continue
            if not passed:
                failures.append(
                    {
                        "reason": "comparative_static_violated",
                        "comparative_static_index": static_index,
                        "point_index": point_index,
                        "point": point,
                        "expected_sign": expected,
                        "actual": value,
                    }
                )
    return failures


def _load_claims(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load claims: {exc}") from exc
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list) or not all(isinstance(item, dict) for item in claims):
        raise ValueError("claims contract must contain a list of claim objects")
    return claims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Numerically falsify declared theoretical-claim companions offline."
    )
    parser.add_argument("--claims", default="contracts/claims.yaml", help="JSON-compatible claim ledger")
    parser.add_argument("--claim-id", action="append", default=[], help="Restrict to one claim id (repeatable)")
    args = parser.parse_args(argv)

    try:
        claims = _load_claims(Path(args.claims))
    except ValueError as exc:
        parser.error(str(exc))
    selected_ids = set(args.claim_id)
    results: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for claim in claims:
        claim_id = claim.get("claim_id")
        if isinstance(claim_id, str):
            seen_ids.add(claim_id)
        if selected_ids and claim_id not in selected_ids:
            continue
        if claim.get("type") != "theoretical" or "falsification_spec" not in claim:
            continue
        failures = evaluate_falsification_spec(claim.get("falsification_spec"))
        results.append({"claim_id": claim_id, "ok": not failures, "failures": failures})
    missing = sorted(selected_ids - seen_ids)
    for claim_id in missing:
        results.append(
            {"claim_id": claim_id, "ok": False, "failures": [{"reason": "claim_not_found"}]}
        )
    payload = {"ok": all(item["ok"] for item in results), "claim_count": len(results), "results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
