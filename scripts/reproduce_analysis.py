#!/usr/bin/env python3
"""Recompute deterministic paper inputs and compare them with the checked-in surface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC_OUTPUTS = (
    Path("reports/tables/str_regime_summary.csv"),
    Path("reports/tables/str_regime_summary.md"),
    Path("reports/paper/paper_values.json"),
    Path("reports/figures/str_ecosystem_timeseries.data.json"),
    Path("reports/figures/str_post_dencun_regimes.data.json"),
)


def main() -> int:
    missing = [path.as_posix() for path in DETERMINISTIC_OUTPUTS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit("reproduce_analysis_baseline_missing:" + ",".join(missing))

    baseline = {path: (REPO_ROOT / path).read_bytes() for path in DETERMINISTIC_OUTPUTS}
    paper_values = json.loads((REPO_ROOT / "reports/paper/paper_values.json").read_text(encoding="utf-8"))
    as_of = paper_values.get("as_of") if isinstance(paper_values, dict) else None
    if not isinstance(as_of, str):
        raise SystemExit("reproduce_analysis_as_of_missing")

    with tempfile.TemporaryDirectory(prefix="research-swarm-mpl-") as mpl_config:
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = mpl_config
        completed = subprocess.run(
            [
                sys.executable,
                "src/analysis/build_str_release_outputs.py",
                "--as-of",
                as_of,
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
    if completed.returncode != 0:
        return completed.returncode

    mismatches = [
        path.as_posix()
        for path, expected in baseline.items()
        if (REPO_ROOT / path).read_bytes() != expected
    ]
    if mismatches:
        raise SystemExit("reproduce_analysis_byte_mismatch:" + ",".join(mismatches))
    for path in DETERMINISTIC_OUTPUTS:
        print(f"byte-identical: {path.as_posix()}")
    print("SVG byte identity intentionally excluded; figure content is checked through .data.json sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
