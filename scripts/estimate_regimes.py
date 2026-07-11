#!/usr/bin/env python3
"""Estimate and manifest structural breaks from the committed primary panel.

Input: the active pack's primary panel plus its processed manifest.
Output: reports/analysis/regime_breaks.json.

Run: python scripts/estimate_regimes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
from inference import bai_perron_breaks  # noqa: E402
from spec_curve import statistical_config_from_lock  # noqa: E402
from swarm_taskfile import load_prereg_lock  # noqa: E402
from pack_config import load_pack_config, pack_value  # noqa: E402


PACK = load_pack_config(REPO_ROOT)
DEFAULT_PANEL = Path(pack_value(PACK, "paths.primary_panel"))
DEFAULT_OUTPUT = Path("reports/analysis/regime_breaks.json")


def _panel_columns(pack: dict[str, object] = PACK) -> dict[str, str]:
    columns = pack_value(pack, "analysis.panel_columns", dict)
    return {key: str(columns[key]) for key in ("date", "denominator", "numerator", "metric")}


def _regime_series(pack: dict[str, object] = PACK) -> tuple[str, ...]:
    return tuple(pack_value(pack, "analysis.regime_series", list))


def _sha256_and_bytes(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {path}") from exc


def _matching_processed_manifest(panel_path: Path) -> Path:
    panel_rel = _repo_relative(panel_path)
    actual_sha, actual_bytes = _sha256_and_bytes(panel_path)
    matches: list[Path] = []
    for candidate in sorted((REPO_ROOT / "data" / "processed_manifest").glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for output in payload.get("outputs", []):
            if (
                isinstance(output, dict)
                and output.get("path") == panel_rel
                and output.get("sha256") == actual_sha
                and output.get("bytes") == actual_bytes
            ):
                matches.append(candidate)
    if not matches:
        raise ValueError(f"no processed manifest content-binds {panel_rel}")
    return matches[-1]


def _verify_panel_claim(panel_path: Path, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel_rel = _repo_relative(panel_path)
    actual_sha, actual_bytes = _sha256_and_bytes(panel_path)
    matching = [
        entry
        for entry in payload.get("outputs", [])
        if isinstance(entry, dict) and entry.get("path") == panel_rel
    ]
    if len(matching) != 1:
        raise ValueError(f"processed manifest must contain exactly one claim for {panel_rel}")
    entry = matching[0]
    if entry.get("sha256") != actual_sha or entry.get("bytes") != actual_bytes:
        raise ValueError(f"processed manifest hash/byte claim is stale for {panel_rel}")


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else "unavailable"


def estimate_regime_manifest(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    series_name: str,
    max_breaks: int,
    min_segment: int,
    imposed_date: str | None,
    prereg_lock_sha256: str | None = None,
    parameter_source: str = "explicit_arguments",
    panel_columns: dict[str, str] | None = None,
    allowed_series: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Create a content-bound regime-break manifest without writing it."""

    _verify_panel_claim(panel_path, panel_manifest_path)
    columns = panel_columns or _panel_columns()
    panel = pd.read_csv(panel_path)
    panel[columns["date"]] = pd.to_datetime(panel[columns["date"]], errors="raise")
    for column in (columns["denominator"], columns["numerator"]):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    ecosystem = (
        panel.groupby(columns["date"], as_index=False, sort=True)[
            [columns["denominator"], columns["numerator"]]
        ]
        .sum(min_count=1)
        .sort_values(columns["date"], kind="stable")
    )
    ecosystem[columns["metric"]] = ecosystem[columns["numerator"]] / ecosystem[columns["denominator"]]
    ecosystem.loc[ecosystem[columns["denominator"]] == 0.0, columns["metric"]] = float("nan")
    if series_name not in set(allowed_series or _regime_series()):
        raise ValueError(f"unsupported series: {series_name}")
    complete = ecosystem.loc[
        ecosystem[series_name].notna(),
        [columns["date"], series_name],
    ].reset_index(drop=True)
    breaks, ssr = bai_perron_breaks(
        complete[series_name].to_numpy(dtype=float),
        max_breaks=max_breaks,
        min_segment=min_segment,
    )
    break_dates = [complete.loc[index, columns["date"]].date().isoformat() for index in breaks]
    panel_sha, panel_bytes = _sha256_and_bytes(panel_path)
    comparison: dict[str, object] | None = None
    if imposed_date is not None:
        imposed = pd.Timestamp(imposed_date)
        comparison = {
            "imposed_date": imposed.date().isoformat(),
            "nearest_break_date": None,
            "distance_days": None,
        }
        if break_dates:
            nearest = min(
                (pd.Timestamp(value) for value in break_dates),
                key=lambda value: (abs((value - imposed).days), value),
            )
            comparison["nearest_break_date"] = nearest.date().isoformat()
            comparison["distance_days"] = int((nearest - imposed).days)
    return {
        "schema_version": "research_swarm.regime_breaks.v1",
        "inputs": [
            {
                "path": _repo_relative(panel_path),
                "sha256": panel_sha,
                "bytes": panel_bytes,
            }
        ],
        "source_processed_manifest": {
            "path": _repo_relative(panel_manifest_path),
            "sha256": _sha256_and_bytes(panel_manifest_path)[0],
        },
        "method": "bai_perron_dynamic_programming_piecewise_constant",
        "series": series_name,
        "max_breaks": max_breaks,
        "min_segment": min_segment,
        "break_indices": breaks,
        "break_dates": break_dates,
        "ssr": ssr,
        "imposed_regime_comparison": comparison,
        "parameter_source": parameter_source,
        "prereg_lock_sha256": prereg_lock_sha256,
        "git_sha": _git_sha(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="estimate_regimes.py")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=None,
        help="Processed manifest binding the panel; auto-detected by content hash when omitted.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--series",
        choices=_regime_series(),
        default=None,
    )
    parser.add_argument("--max-breaks", type=int, default=None)
    parser.add_argument("--min-segment", type=int, default=None)
    parser.add_argument("--imposed-date", default=None)
    parser.add_argument("--analysis-lock", type=Path, default=Path("docs/prereg/analysis_plan.lock.md"))
    return parser.parse_args(argv)


def _resolved_parameters(args: argparse.Namespace) -> tuple[str, int, int, str | None, str | None, str]:
    lock_path = (
        (REPO_ROOT / args.analysis_lock).resolve()
        if not args.analysis_lock.is_absolute()
        else args.analysis_lock
    )
    lock, error = load_prereg_lock(lock_path, expected_phase="2b")
    if error is not None or lock is None:
        raise ValueError(f"invalid analysis-plan lock: {error}")
    if lock.get("active") is True:
        config, lock_sha = statistical_config_from_lock(lock_path)
        regime = config["regime_breaks"]
        locked = (
            str(regime["series"]),
            int(regime["max_breaks"]),
            int(regime["min_segment"]),
            str(regime["imposed_date"]) if regime.get("imposed_date") is not None else None,
        )
        requested = (args.series, args.max_breaks, args.min_segment, args.imposed_date)
        for name, requested_value, locked_value in zip(
            ("series", "max_breaks", "min_segment", "imposed_date"),
            requested,
            locked,
            strict=True,
        ):
            if requested_value is not None and requested_value != locked_value:
                raise ValueError(
                    f"--{name.replace('_', '-')}={requested_value} conflicts with locked value {locked_value}"
                )
        return (*locked, lock_sha, "active_analysis_lock")
    defaults = pack_value(PACK, "analysis.regime_defaults", dict)
    return (
        args.series or str(defaults["series"]),
        args.max_breaks if args.max_breaks is not None else int(defaults["max_breaks"]),
        args.min_segment if args.min_segment is not None else int(defaults["min_segment"]),
        args.imposed_date or str(defaults["imposed_date"]),
        None,
        "draft_lock_cli_or_documented_defaults",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    series_name, max_breaks, min_segment, imposed_date, lock_sha, parameter_source = _resolved_parameters(args)
    panel_path = (REPO_ROOT / args.panel).resolve() if not args.panel.is_absolute() else args.panel
    if args.panel_manifest is None:
        panel_manifest_path = _matching_processed_manifest(panel_path)
    else:
        panel_manifest_path = (
            (REPO_ROOT / args.panel_manifest).resolve()
            if not args.panel_manifest.is_absolute()
            else args.panel_manifest
        )
    output_path = (REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    payload = estimate_regime_manifest(
        panel_path=panel_path,
        panel_manifest_path=panel_manifest_path,
        series_name=series_name,
        max_breaks=max_breaks,
        min_segment=min_segment,
        imposed_date=imposed_date,
        prereg_lock_sha256=lock_sha,
        parameter_source=parameter_source,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        display_path = _repo_relative(output_path)
    except ValueError:
        display_path = output_path.as_posix()
    print(f"Wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
