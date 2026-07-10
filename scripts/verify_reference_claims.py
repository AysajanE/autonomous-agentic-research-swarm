#!/usr/bin/env python3
"""Offline verification for the STR battle-test reference claims.

Re-derives the regime headline numbers registered in ``contracts/claims.yaml``
from the committed ``reports/tables/str_regime_summary.csv`` and asserts each
manuscript literal is reproduced from the table. This is the runnable
``verification_command`` for the reference descriptive claims — a deterministic
consistency check between the manuscript's reported numbers and the committed
table artifact, requiring no network and no deleted raw evidence.

Exit 0 when every expected literal reproduces; exit 1 (with a diff) otherwise.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TABLE = Path("reports/tables/str_regime_summary.csv")

# The manuscript literals each regime row must reproduce (percent to 2 dp;
# ETH/day mean rent to 3 dp) — the exact figures registered in the ledger.
EXPECTED = {
    "full_sample": {"mean_str_pct": "41.24", "mean_rent_paid_eth": "84.886"},
    "pre_dencun": {"mean_str_pct": "69.14", "mean_rent_paid_eth": "147.014"},
    "post_dencun": {"mean_str_pct": "11.68", "mean_rent_paid_eth": "19.064"},
    "post_dencun_blob_floor": {"mean_str_pct": "8.98", "mean_rent_paid_eth": "18.740"},
    "post_dencun_non_floor": {"mean_str_pct": "12.77", "mean_rent_paid_eth": "19.194"},
}
EXPECTED_DAYS = {"full_sample": "1559", "pre_dencun": "802", "post_dencun": "757"}


def _round(value: str, places: int) -> str:
    return f"{float(value):.{places}f}"


def main() -> int:
    if not TABLE.is_file():
        print(f"missing table: {TABLE}", file=sys.stderr)
        return 1
    rows = {r["regime_id"]: r for r in csv.DictReader(TABLE.read_text().splitlines())}
    diffs: list[str] = []
    for regime_id, fields in EXPECTED.items():
        row = rows.get(regime_id)
        if row is None:
            diffs.append(f"{regime_id}: absent from table")
            continue
        got_str = _round(row["mean_str_pct"], 2)
        if got_str != fields["mean_str_pct"]:
            diffs.append(f"{regime_id}.mean_str_pct: table={got_str} claim={fields['mean_str_pct']}")
        got_rent = _round(row["mean_rent_paid_eth"], 3)
        if got_rent != fields["mean_rent_paid_eth"]:
            diffs.append(f"{regime_id}.mean_rent: table={got_rent} claim={fields['mean_rent_paid_eth']}")
    for regime_id, days in EXPECTED_DAYS.items():
        row = rows.get(regime_id)
        if row is not None and row["days"] != days:
            diffs.append(f"{regime_id}.days: table={row['days']} claim={days}")
    if diffs:
        print("reference-claim verification FAILED:", file=sys.stderr)
        for diff in diffs:
            print(f"  - {diff}", file=sys.stderr)
        return 1
    print("reference-claim verification OK: all registered literals reproduce from the committed table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
