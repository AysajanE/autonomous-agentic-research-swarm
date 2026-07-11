#!/usr/bin/env python3
"""Resolve computed-paper value tokens without requiring Quarto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "reports" / "paper" / "index.qmd"
DEFAULT_VALUES = REPO_ROOT / "reports" / "paper" / "paper_values.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "paper" / "build" / "index.resolved.qmd"
VALUE_TOKEN_RE = re.compile(r"\{\{value:([A-Za-z0-9_]+)\}\}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="render_paper.py")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def resolve_manuscript(source_text: str, payload: dict[str, object]) -> str:
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ValueError("paper_values_missing_values_object")

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        entry = values.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("display"), str):
            missing.add(key)
            return match.group(0)
        return str(entry["display"])

    resolved = VALUE_TOKEN_RE.sub(replace, source_text)
    if missing:
        raise ValueError("paper_value_key_missing:" + ",".join(sorted(missing)))
    if "{{value:" in resolved or VALUE_TOKEN_RE.search(resolved):
        raise ValueError("paper_value_token_unresolved")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.values.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("paper_values_top_level_not_object")
    try:
        resolved = resolve_manuscript(args.source.read_text(encoding="utf-8"), payload)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(resolved, encoding="utf-8")
    print(f"Resolved {args.source.relative_to(REPO_ROOT)} -> {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
