#!/usr/bin/env python3
"""Deterministically enumerate a locked experiment grid without running a solver."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def _values(value: object, *, field: str) -> list[object]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{field} must not be empty")
    for item in values:
        if isinstance(item, (dict, list)):
            raise ValueError(f"{field} values must be scalar")
    return values


def enumerate_cells(experiment_spec: object) -> list[dict[str, object]]:
    """Return canonical grid × seed × budget cells in stable key/value order."""
    if not isinstance(experiment_spec, dict):
        raise ValueError("experiment spec must be an object")
    grid = experiment_spec.get("grid")
    dimensions = grid.get("dimensions") if isinstance(grid, dict) else None
    if not isinstance(dimensions, dict):
        raise ValueError("grid.dimensions must be an object")
    names = sorted(dimensions)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("grid dimension names must be non-empty strings")
    if set(names) & {"seed", "budget"}:
        raise ValueError("grid dimensions may not shadow seed or budget")
    dimension_values = [_values(dimensions[name], field=f"grid.dimensions.{name}") for name in names]
    seeds = _values(experiment_spec.get("seeds"), field="seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must contain integers")
    budgets = _values(experiment_spec.get("budget"), field="budget")
    if any(
        not isinstance(budget, (int, float)) or isinstance(budget, bool) or float(budget) <= 0
        for budget in budgets
    ):
        raise ValueError("budget must contain positive numbers")

    products = itertools.product(*dimension_values) if names else [()]
    cells: list[dict[str, object]] = []
    for values in products:
        base = dict(zip(names, values))
        for seed in seeds:
            for budget in budgets:
                cells.append({**base, "seed": seed, "budget": budget})
    return cells


def load_experiment_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load experiment spec: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment spec must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate the deterministic cells of an experiment spec."
    )
    parser.add_argument("experiment_spec", help="JSON-compatible experiment spec path")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)
    try:
        spec = load_experiment_spec(Path(args.experiment_spec))
        cells = enumerate_cells(spec)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {
        "schema_version": "research_swarm.sweep_cells.v1",
        "cell_count": len(cells),
        "cells": cells,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
