#!/usr/bin/env python3
"""Recompute deterministic paper inputs and compare them with the checked-in surface.

Two verification bars, never conflated (plan §7.3 "byte identity vs content
equivalence"):

- BYTE identity for the science artifacts (regime tables + paper_values.json). These
  carry display-precision values (3-6 decimals) that are stable across platforms, so a
  single changed digit is a real change and must fail.
- CONTENT equivalence for the figure `.data.json` sidecars. These carry the full plotted
  series at high precision; float64 reductions (means, rolling windows, quantiles) differ
  in their last ULPs across platforms/BLAS/NumPy builds, so byte identity is not
  achievable without a digest-pinned container. They are compared structurally instead:
  every string/bool must match exactly and every number must be numerically close, which
  still catches any genuine drift (a changed data point differs far above float noise)
  while tolerating cross-platform round-off. The SVGs themselves are regenerated but not
  compared (Matplotlib embeds non-determinism); the sidecar is their content reference.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
BYTE_IDENTICAL_OUTPUTS = (
    Path("reports/tables/str_regime_summary.csv"),
    Path("reports/tables/str_regime_summary.md"),
    Path("reports/paper/paper_values.json"),
    Path("reports/exhibits/manifest.json"),
)
CONTENT_EQUIVALENT_OUTPUTS = (
    Path("reports/figures/str_ecosystem_timeseries.data.json"),
    Path("reports/figures/str_post_dencun_regimes.data.json"),
)
ALL_OUTPUTS = BYTE_IDENTICAL_OUTPUTS + CONTENT_EQUIVALENT_OUTPUTS

# Tolerance for the figure-sidecar content check. The sidecars store values rounded to 12
# decimals, so cross-platform float64 round-off (accumulated reductions + parse) is bounded
# at ~1e-12..1e-11. The tolerance is set 1-2 orders above that noise floor — NOT the earlier
# 1e-9, which was ~100x too loose and could hide real drift (e.g. a near-zero blob-fee value
# flipping to 0, or a 1e-7 change on a ~950 magnitude series). A genuine data change differs
# by orders of magnitude more than 1e-10.
_SIDECAR_REL_TOL = 1e-10
_SIDECAR_ABS_TOL = 1e-10


def _content_equivalent(expected: object, actual: object, path: str) -> list[str]:
    """Recursively compare two parsed-JSON values: strings/bools/None must match exactly,
    numbers within tolerance, containers structurally. Returns a list of mismatch paths."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        # bool is a subclass of int — compare exactly and never as a number.
        return [] if expected == actual and type(expected) is type(actual) else [path]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # Non-finite values (NaN/inf) are never valid plotted data and never compare equal;
        # reject them explicitly rather than letting NaN!=NaN silently pass or fail.
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))):
            return [f"{path}(non_finite:{expected}/{actual})"]
        if math.isclose(float(expected), float(actual), rel_tol=_SIDECAR_REL_TOL, abs_tol=_SIDECAR_ABS_TOL):
            return []
        return [f"{path}({expected}!={actual})"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[str] = []
        if set(expected) != set(actual):
            return [f"{path}:keys{sorted(set(expected) ^ set(actual))}"]
        for key in expected:
            mismatches.extend(_content_equivalent(expected[key], actual[key], f"{path}.{key}"))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}:len({len(expected)}!={len(actual)})"]
        mismatches = []
        for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
            mismatches.extend(_content_equivalent(exp_item, act_item, f"{path}[{index}]"))
        return mismatches
    return [] if expected == actual else [f"{path}({expected!r}!={actual!r})"]


def main() -> int:
    missing = [path.as_posix() for path in ALL_OUTPUTS if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit("reproduce_analysis_baseline_missing:" + ",".join(missing))

    baseline_bytes = {path: (REPO_ROOT / path).read_bytes() for path in BYTE_IDENTICAL_OUTPUTS}
    baseline_json = {
        path: json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in CONTENT_EQUIVALENT_OUTPUTS
    }
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

    byte_mismatches = [
        path.as_posix()
        for path, expected in baseline_bytes.items()
        if (REPO_ROOT / path).read_bytes() != expected
    ]
    if byte_mismatches:
        raise SystemExit("reproduce_analysis_byte_mismatch:" + ",".join(byte_mismatches))

    content_mismatches: list[str] = []
    for path, expected in baseline_json.items():
        actual = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        content_mismatches.extend(_content_equivalent(expected, actual, path.as_posix()))
    if content_mismatches:
        raise SystemExit("reproduce_analysis_content_mismatch:" + ",".join(content_mismatches[:20]))

    for path in BYTE_IDENTICAL_OUTPUTS:
        print(f"byte-identical: {path.as_posix()}")
    for path in CONTENT_EQUIVALENT_OUTPUTS:
        print(f"content-equivalent: {path.as_posix()}")
    print("SVG byte identity intentionally excluded; figure content is checked through .data.json sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
