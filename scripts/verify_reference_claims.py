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


def _parse_literal(literal: str):
    """Return (value, is_percent, decimals, is_count) for a registered literal,
    retaining LEXICAL precision (decimals counted from the string, not the
    float, so trailing zeros survive) and count-type (no decimal, no %), or None.
    """
    match = re.match(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%?)", str(literal))
    if match is None:
        return None
    digits = match.group(1).replace(",", "")
    is_percent = match.group(2) == "%"
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    is_count = decimals == 0 and not is_percent
    try:
        return float(digits), is_percent, decimals, is_count
    except ValueError:
        return None


def _reproduced(value: float, is_percent: bool, decimals: int, is_count: bool, numbers: list[float]) -> bool:
    for candidate in numbers:
        if is_count:
            # exact integer equality — 14 must NOT be reproduced by 14.04
            if float(candidate).is_integer() and int(candidate) == int(round(value)):
                return True
        elif is_percent:
            # artifact may store the percent (12.37) or the fraction (0.1237);
            # both must reproduce at the literal's stated decimal precision
            if round(candidate, decimals) == round(value, decimals):
                return True
            if round(candidate * 100.0, decimals) == round(value, decimals):
                return True
        elif round(candidate, decimals) == round(value, decimals):
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
            value, is_percent, decimals, is_count = parsed
            checked += 1
            if not _reproduced(value, is_percent, decimals, is_count, numbers):
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
