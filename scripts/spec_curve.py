#!/usr/bin/env python3
"""Run the preregistered specification curve, including construction variants.

Inputs: a hash-locked phase-2b analysis plan and a local processed panel.
Output: reports/analysis/spec_curve/<claim_id>.json.

The lock body must contain one fenced JSON object with a
``statistical_reporting`` member.  Its ``specification_curve`` member declares
``construction_variants.keying``, ``construction_variants.missingness``,
``analysis_variants``, and ``survival_threshold``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = REPO_ROOT / "src" / "analysis"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from inference import (  # noqa: E402
    clustered_se,
    driscoll_kraay_se,
    newey_west_se,
    two_way_clustered_se,
)
from swarm_taskfile import load_prereg_lock  # noqa: E402
from pack_config import load_pack_config, pack_value  # noqa: E402


PACK = load_pack_config(REPO_ROOT)
DEFAULT_PANEL = Path(pack_value(PACK, "paths.primary_panel"))
DEFAULT_LOCK = Path("docs/prereg/analysis_plan.lock.md")
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistical_config_from_lock(lock_path: Path) -> tuple[dict[str, Any], str]:
    """Return the frozen statistical config and active lock hash."""

    lock, error = load_prereg_lock(lock_path, expected_phase="2b")
    if error is not None or lock is None:
        raise ValueError(f"invalid analysis-plan lock: {error}")
    if lock.get("active") is not True:
        raise ValueError("analysis-plan lock is not active")
    body = str(lock.get("body", ""))
    candidates: list[dict[str, Any]] = []
    for match in _JSON_FENCE_RE.finditer(body):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in analysis-plan lock: {exc}") from exc
        if isinstance(payload, dict) and isinstance(payload.get("statistical_reporting"), dict):
            candidates.append(payload["statistical_reporting"])
    if len(candidates) != 1:
        raise ValueError("active analysis lock must contain exactly one statistical_reporting JSON object")
    config = candidates[0]
    _validate_config(config)
    return config, str(lock["body_sha256"])


def _validate_config(config: dict[str, Any]) -> None:
    vce = config.get("vce")
    if vce not in {"newey_west", "driscoll_kraay", "clustered", "two_way_clustered"}:
        raise ValueError("statistical_reporting.vce is unsupported")
    regime = config.get("regime_breaks")
    if not isinstance(regime, dict):
        raise ValueError("statistical_reporting.regime_breaks must be an object")
    if regime.get("method") != "bai_perron_dynamic_programming_piecewise_constant":
        raise ValueError("regime_breaks.method is unsupported")
    if regime.get("series") not in {"str", "rent_paid_eth", "l2_fees_eth"}:
        raise ValueError("regime_breaks.series is unsupported")
    for field in ("max_breaks", "min_segment"):
        if not isinstance(regime.get(field), int) or isinstance(regime.get(field), bool) or regime[field] < 1:
            raise ValueError(f"regime_breaks.{field} must be a positive integer")
    curve = config.get("specification_curve")
    if not isinstance(curve, dict):
        raise ValueError("statistical_reporting.specification_curve must be an object")
    construction = curve.get("construction_variants")
    if not isinstance(construction, dict):
        raise ValueError("construction_variants must be an object")
    keying = construction.get("keying")
    missingness = construction.get("missingness")
    if not isinstance(keying, list) or len(keying) < 2 or len(set(keying)) != len(keying):
        raise ValueError("the locked grid requires at least two distinct keying variants")
    if not isinstance(missingness, list) or len(missingness) < 2 or len(set(missingness)) != len(missingness):
        raise ValueError("the locked grid requires at least two distinct missingness variants")
    if not set(keying).issubset({"rollup_day", "date_aggregate"}):
        raise ValueError("keying variants must be rollup_day/date_aggregate")
    if vce in {"clustered", "two_way_clustered"} and "date_aggregate" in keying:
        raise ValueError("vce_incompatible_with_aggregate_variant")
    if not set(missingness).issubset({"drop", "zero_fill"}):
        raise ValueError("missingness variants must be drop/zero_fill")
    variants = curve.get("analysis_variants")
    if not isinstance(variants, list) or not variants or not all(isinstance(item, dict) for item in variants):
        raise ValueError("analysis_variants must be a non-empty list of objects")
    ids = [item.get("id") for item in variants]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("analysis variant ids must be non-empty and unique")
    if any(item.get("estimator") != "regime_difference" for item in variants):
        raise ValueError("only the auditable regime_difference analysis variant is supported")
    threshold = curve.get("survival_threshold")
    total = len(keying) * len(missingness) * len(variants)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 1 <= threshold <= total:
        raise ValueError("survival_threshold must be an integer within the grid size")


def _construct_panel(panel: pd.DataFrame, keying: str, missingness: str) -> pd.DataFrame:
    required = {"date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    frame = panel.loc[:, sorted(required)].copy()
    frame["date_utc"] = pd.to_datetime(frame["date_utc"], errors="raise")
    for column in ("l2_fees_eth", "rent_paid_eth"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if missingness == "drop":
        frame = frame.dropna(subset=["l2_fees_eth", "rent_paid_eth"])
    elif missingness == "zero_fill":
        frame[["l2_fees_eth", "rent_paid_eth"]] = frame[
            ["l2_fees_eth", "rent_paid_eth"]
        ].fillna(0.0)
    else:
        raise ValueError(f"unsupported missingness variant: {missingness}")

    if keying == "rollup_day":
        grouped = (
            frame.groupby(["date_utc", "rollup_id"], as_index=False, sort=True)[
                ["l2_fees_eth", "rent_paid_eth"]
            ]
            .sum(min_count=1)
            .sort_values(["date_utc", "rollup_id"], kind="stable")
        )
    elif keying == "date_aggregate":
        grouped = frame.groupby("date_utc", as_index=False, sort=True)[
            ["l2_fees_eth", "rent_paid_eth"]
        ].sum(min_count=1)
        grouped["rollup_id"] = "ecosystem"
    else:
        raise ValueError(f"unsupported keying variant: {keying}")
    grouped["str"] = grouped["rent_paid_eth"] / grouped["l2_fees_eth"]
    grouped.loc[grouped["l2_fees_eth"] == 0.0, "str"] = np.nan
    return grouped.dropna(subset=["str"]).reset_index(drop=True)


def _run_variant(
    frame: pd.DataFrame,
    variant: dict[str, Any],
    *,
    vce: str,
    default_lags: int,
    dk_min_time_periods: int,
) -> dict[str, object]:
    regime_date = pd.Timestamp(str(variant.get("regime_date")))
    subset = frame.copy()
    if variant.get("start_date") is not None:
        subset = subset.loc[subset["date_utc"] >= pd.Timestamp(str(variant["start_date"]))]
    if variant.get("end_date") is not None:
        subset = subset.loc[subset["date_utc"] <= pd.Timestamp(str(variant["end_date"]))]
    post = (subset["date_utc"] >= regime_date).astype(float).to_numpy()
    if len(subset) < 3 or len(np.unique(post)) != 2:
        raise ValueError(f"analysis variant {variant['id']} must contain both regimes")
    design = np.column_stack([np.ones(len(subset)), post])
    outcome = subset["str"].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    lags = int(variant.get("lags", default_lags))
    if vce == "newey_west":
        standard_errors = newey_west_se(residuals, design, lags)
    elif vce == "driscoll_kraay":
        standard_errors = driscoll_kraay_se(
            residuals,
            design,
            subset["date_utc"].dt.date.to_numpy(dtype=object),
            lags,
            min_time_periods=dk_min_time_periods,
        )
    elif vce == "clustered":
        standard_errors = clustered_se(design, residuals, subset["rollup_id"].to_numpy())
    else:
        standard_errors = two_way_clustered_se(
            design,
            residuals,
            subset["rollup_id"].to_numpy(),
            subset["date_utc"].dt.date.to_numpy(dtype=object),
        )
    estimate = float(coefficients[1])
    se = float(standard_errors[1])
    critical = 1.959963984540054
    return {
        "analysis_variant": str(variant["id"]),
        "estimate": estimate,
        "se": se,
        "ci": [estimate - critical * se, estimate + critical * se],
        "n": int(len(subset)),
        "vce": vce,
        "critical_value_method": "normal_approximation",
    }


def build_spec_curve(
    *,
    panel: pd.DataFrame,
    claim_id: str,
    headline_estimate: float,
    config: dict[str, Any],
    prereg_lock_sha256: str,
    input_entry: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run every Cartesian grid cell and return the manifested curve."""

    _validate_config(config)
    curve = config["specification_curve"]
    construction = curve["construction_variants"]
    specs: list[dict[str, object]] = []
    default_lags = int(config.get("lags", 0))
    dk_min = int(config.get("dk_min_time_periods", 20))
    for keying, missingness, variant in itertools.product(
        construction["keying"],
        construction["missingness"],
        curve["analysis_variants"],
    ):
        constructed = _construct_panel(panel, keying, missingness)
        result = _run_variant(
            constructed,
            variant,
            vce=config["vce"],
            default_lags=default_lags,
            dk_min_time_periods=dk_min,
        )
        result.update(
            {
                "spec_id": f"keying={keying}|missingness={missingness}|analysis={variant['id']}",
                "construction_variant": {"keying": keying, "missingness": missingness},
            }
        )
        specs.append(result)
    sign = np.sign(float(headline_estimate))
    survival_count = sum(
        1
        for spec in specs
        if np.sign(float(spec["estimate"])) == sign
        and (float(spec["ci"][0]) > 0.0 or float(spec["ci"][1]) < 0.0)  # type: ignore[index]
    )
    estimates = [float(spec["estimate"]) for spec in specs]
    payload: dict[str, object] = {
        "schema_version": "research_swarm.spec_curve.v1",
        "claim_id": claim_id,
        "prereg_lock_sha256": prereg_lock_sha256,
        "declared_vce": config["vce"],
        "grid_dimensions": {
            "keying": list(construction["keying"]),
            "missingness": list(construction["missingness"]),
            "analysis_variant": [item["id"] for item in curve["analysis_variants"]],
        },
        "specs": specs,
        "headline_estimate": float(headline_estimate),
        "survival_count": survival_count,
        "survival_threshold": int(curve["survival_threshold"]),
        "headline_within_curve": min(estimates) <= float(headline_estimate) <= max(estimates),
    }
    if input_entry is not None:
        payload["inputs"] = [input_entry]
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="spec_curve.py")
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--headline-estimate", required=True, type=float)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--analysis-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    panel_path = (REPO_ROOT / args.panel).resolve() if not args.panel.is_absolute() else args.panel
    lock_path = (
        (REPO_ROOT / args.analysis_lock).resolve()
        if not args.analysis_lock.is_absolute()
        else args.analysis_lock
    )
    config, lock_sha = statistical_config_from_lock(lock_path)
    panel = pd.read_csv(panel_path)
    panel_bytes = panel_path.stat().st_size
    payload = build_spec_curve(
        panel=panel,
        claim_id=args.claim_id,
        headline_estimate=args.headline_estimate,
        config=config,
        prereg_lock_sha256=lock_sha,
        input_entry={
            "path": panel_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(panel_path),
            "bytes": panel_bytes,
        },
    )
    output = args.output or Path(f"reports/analysis/spec_curve/{args.claim_id}.json")
    output_path = (REPO_ROOT / output).resolve() if not output.is_absolute() else output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
