#!/usr/bin/env python3
"""Offline verification for the STR battle-test reference claims.

Reads `contracts/claims.yaml` and, for every registered
`manuscript_numeric_literals` value across ALL reference claims, confirms the
value is reproduced by one of the claim's committed `supporting_artifacts`
(numbers gathered from JSON recursively and from Markdown/CSV/text via regex).
Percentages are matched against both their percent and fractional forms; ETH and
plain decimals against the rounded value; counts against an exact integer.

This is the runnable, network-free `verification_command` the reference claims
cite. Exit 0 when every literal reproduces; exit 1 (with a diff) otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CLAIMS = Path("contracts/claims.yaml")


def _iter_json_numbers(obj: object):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_json_numbers(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_json_numbers(value)


_TEXT_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _artifact_numbers(path: Path) -> list[float]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    numbers: list[float] = []
    if path.suffix == ".json":
        try:
            numbers.extend(_iter_json_numbers(json.loads(raw)))
        except json.JSONDecodeError:
            pass
    for token in _TEXT_NUMBER_RE.findall(raw):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return numbers


def _parse_literal(literal: str) -> tuple[float, bool] | None:
    """Return (value, is_percent) for a registered literal, or None."""
    match = re.match(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%?)", str(literal))
    if match is None:
        return None
    try:
        return float(match.group(1).replace(",", "")), match.group(2) == "%"
    except ValueError:
        return None


def _reproduced(value: float, is_percent: bool, numbers: list[float]) -> bool:
    decimals = len(f"{value}".split(".")[1]) if "." in f"{value}" else 0
    targets = [value]
    if is_percent:
        targets.append(value / 100.0)  # artifact may store a fraction
    for candidate in numbers:
        for target in targets:
            # round the artifact number to the literal's stated precision
            if is_percent and abs(round(candidate * 100.0, decimals) - value) < 10 ** (-decimals) / 2:
                return True
            if abs(round(candidate, decimals) - round(target, decimals)) < 10 ** (-decimals) / 2:
                return True
    return False


def main() -> int:
    if not CLAIMS.is_file():
        print(f"missing ledger: {CLAIMS}", file=sys.stderr)
        return 1
    ledger = json.loads(CLAIMS.read_text(encoding="utf-8"))
    diffs: list[str] = []
    checked = 0
    for claim in ledger.get("claims", []):
        literals = claim.get("manuscript_numeric_literals") or []
        artifacts = claim.get("supporting_artifacts") or []
        numbers: list[float] = []
        for artifact in artifacts:
            numbers.extend(_artifact_numbers(Path(artifact["path"])))
        for literal in literals:
            parsed = _parse_literal(literal)
            if parsed is None:
                diffs.append(f"{claim.get('claim_id')}: unparseable literal {literal!r}")
                continue
            value, is_percent = parsed
            checked += 1
            if not _reproduced(value, is_percent, numbers):
                diffs.append(
                    f"{claim.get('claim_id')}: literal {literal!r} not reproduced by "
                    f"{[a['path'] for a in artifacts]}"
                )
    if diffs:
        print("reference-claim verification FAILED:", file=sys.stderr)
        for diff in diffs:
            print(f"  - {diff}", file=sys.stderr)
        return 1
    print(f"reference-claim verification OK: {checked} registered literals reproduce from cited artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
