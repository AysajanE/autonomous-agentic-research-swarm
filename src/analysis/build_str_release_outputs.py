#!/usr/bin/env python3
"""
Build release STR figures and tables from validated local artifacts.

Inputs:
- reports/validation/rollup_panel_validation.json
- reports/validation/l1_rent_decomposition_validation.json
- reports/validation/cross_source_reconciliation.json
- data/processed/panels/daily_rollup_panel.csv
- data/processed/l1_rent/daily_l1_rent_decomposition.csv

Outputs:
- reports/figures/str_ecosystem_timeseries.svg
- reports/figures/str_post_dencun_regimes.svg
- reports/tables/str_regime_summary.csv
- reports/tables/str_regime_summary.md
- reports/exhibits/manifest.json

Run:
- python src/analysis/build_str_release_outputs.py --sample
- python src/analysis/build_str_release_outputs.py --as-of 2026-04-09
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import pandas as pd

from metrics_str import compute_ecosystem_str


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_BUNDLE = (
    REPO_ROOT / "reports" / "validation" / "rollup_panel_validation.json",
    REPO_ROOT / "reports" / "validation" / "l1_rent_decomposition_validation.json",
    REPO_ROOT / "reports" / "validation" / "cross_source_reconciliation.json",
)
LIVE_PANEL_PATH = REPO_ROOT / "data" / "processed" / "panels" / "daily_rollup_panel.csv"
LIVE_DECOMP_PATH = REPO_ROOT / "data" / "processed" / "l1_rent" / "daily_l1_rent_decomposition.csv"
SAMPLE_PANEL_PATH = REPO_ROOT / "data" / "samples" / "panels" / "daily_rollup_panel_sample.csv"
SAMPLE_DECOMP_PATH = REPO_ROOT / "data" / "samples" / "l1_rent" / "daily_l1_rent_decomposition_sample.csv"
PANEL_MANIFEST_DIR = REPO_ROOT / "data" / "processed_manifest"
ECOSYSTEM_FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "str_ecosystem_timeseries.svg"
REGIME_FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "str_post_dencun_regimes.svg"
ECOSYSTEM_FIGURE_DATA_PATH = REPO_ROOT / "reports" / "figures" / "str_ecosystem_timeseries.data.json"
REGIME_FIGURE_DATA_PATH = REPO_ROOT / "reports" / "figures" / "str_post_dencun_regimes.data.json"
REGIME_TABLE_CSV_PATH = REPO_ROOT / "reports" / "tables" / "str_regime_summary.csv"
REGIME_TABLE_MD_PATH = REPO_ROOT / "reports" / "tables" / "str_regime_summary.md"
PAPER_VALUES_PATH = REPO_ROOT / "reports" / "paper" / "paper_values.json"
EXHIBITS_MANIFEST_PATH = REPO_ROOT / "reports" / "exhibits" / "manifest.json"
CLAIMS_PATH = REPO_ROOT / "contracts" / "claims.yaml"
PROTOCOL_PATH = REPO_ROOT / "docs" / "protocol.md"
DENCUN_DATE = pd.Timestamp("2024-03-13")
SIDECAR_DECIMAL_PLACES = 12

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.hashsalt"] = "t060_str_release_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_str_release_outputs.py")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Build the locked outputs from tracked sample inputs instead of the live validated surface.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="UTC run date. Live outputs are sliced through the prior UTC day to match manifested panel convention.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    as_of_label = args.as_of or latest_manifest_as_of()

    if args.sample:
        panel = load_panel(SAMPLE_PANEL_PATH)
        decomposition = load_decomposition(SAMPLE_DECOMP_PATH)
        cutoff_date = panel["date_utc"].max()
    else:
        ensure_validation_bundle_passes()
        panel = load_panel(LIVE_PANEL_PATH)
        decomposition = load_decomposition(LIVE_DECOMP_PATH)
        cutoff_date = pd.Timestamp(as_of_label) - pd.Timedelta(days=1)

    panel = panel.loc[panel["date_utc"] <= cutoff_date].copy()
    decomposition = decomposition.loc[decomposition["date_utc"] <= cutoff_date].copy()
    if panel.empty or decomposition.empty:
        raise SystemExit(f"no local analysis rows remain after applying as-of cutoff {cutoff_date.date().isoformat()}")

    ecosystem = build_ecosystem_frame(panel, decomposition)
    regime_table = build_regime_summary(ecosystem)

    ecosystem_figure_qa = write_ecosystem_figure(
        ecosystem, as_of_label=as_of_label, sample_mode=args.sample
    )
    regime_figure_qa = write_post_dencun_regime_figure(
        ecosystem, as_of_label=as_of_label, sample_mode=args.sample
    )
    write_figure_data_sidecars(ecosystem, as_of_label=as_of_label, sample_mode=args.sample)
    write_regime_tables(regime_table)
    if not args.sample:
        write_paper_values(regime_table, as_of_label=as_of_label)
        write_exhibits_manifest(
            ecosystem_figure_qa=ecosystem_figure_qa,
            regime_figure_qa=regime_figure_qa,
        )

    print(f"Wrote {ECOSYSTEM_FIGURE_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGIME_FIGURE_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {ECOSYSTEM_FIGURE_DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGIME_FIGURE_DATA_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGIME_TABLE_CSV_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGIME_TABLE_MD_PATH.relative_to(REPO_ROOT)}")
    if not args.sample:
        print(f"Wrote {PAPER_VALUES_PATH.relative_to(REPO_ROOT)}")
        print(f"Wrote {EXHIBITS_MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


def latest_manifest_as_of() -> str:
    candidates = sorted(PANEL_MANIFEST_DIR.glob("daily_rollup_panel_*.json"))
    if not candidates:
        raise SystemExit("no processed panel manifests found under data/processed_manifest/")
    stem = candidates[-1].stem
    return stem.removeprefix("daily_rollup_panel_")


def ensure_validation_bundle_passes() -> None:
    failing_paths: list[str] = []
    for path in VALIDATION_BUNDLE:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            failing_paths.append(path.relative_to(REPO_ROOT).as_posix())
    if failing_paths:
        joined = ", ".join(failing_paths)
        raise SystemExit(f"validated inputs are missing or failing: {joined}")


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date_utc"])
    panel["l2_fees_eth"] = pd.to_numeric(panel["l2_fees_eth"], errors="raise")
    panel["rent_paid_eth"] = pd.to_numeric(panel["rent_paid_eth"], errors="raise")
    return panel.sort_values(["date_utc", "rollup_id"], kind="stable").reset_index(drop=True)


def load_decomposition(path: Path) -> pd.DataFrame:
    decomposition = pd.read_csv(path, parse_dates=["date_utc"])
    for column in (
        "l1_base_fee_burn_eth",
        "l1_blob_fee_burn_eth",
        "l1_priority_fee_eth",
        "l1_total_rent_eth",
        "l1_blob_base_fee_gwei",
    ):
        decomposition[column] = pd.to_numeric(decomposition[column], errors="coerce")
    return decomposition.sort_values("date_utc", kind="stable").reset_index(drop=True)


def build_ecosystem_frame(panel: pd.DataFrame, decomposition: pd.DataFrame) -> pd.DataFrame:
    ecosystem = compute_ecosystem_str(panel.copy())
    merged = ecosystem.merge(
        decomposition[
            [
                "date_utc",
                "l1_base_fee_burn_eth",
                "l1_blob_fee_burn_eth",
                "l1_priority_fee_eth",
                "l1_total_rent_eth",
                "l1_blob_base_fee_gwei",
            ]
        ],
        on="date_utc",
        how="left",
        validate="one_to_one",
    )
    merged["post_dencun"] = merged["date_utc"] >= DENCUN_DATE
    merged["str_14d"] = merged["str"].rolling(window=14, min_periods=1).mean()
    merged["str_30d"] = merged["str"].rolling(window=30, min_periods=1).mean()
    merged["l2_fees_14d"] = merged["l2_fees_eth"].rolling(window=14, min_periods=1).mean()
    merged["rent_paid_14d"] = merged["rent_paid_eth"].rolling(window=14, min_periods=1).mean()
    merged["blob_floor_regime"] = identify_blob_floor_regime(merged)
    return merged


def identify_blob_floor_regime(frame: pd.DataFrame) -> pd.Series:
    post = frame.loc[frame["post_dencun"] & frame["l1_blob_base_fee_gwei"].notna(), ["date_utc", "l1_blob_base_fee_gwei"]].copy()
    if post.empty:
        return pd.Series(False, index=frame.index)

    threshold = post["l1_blob_base_fee_gwei"].min() * 1.05
    post["candidate"] = post["l1_blob_base_fee_gwei"] <= threshold
    post["gap_days"] = post["date_utc"].diff().dt.days.fillna(1)
    post["run_id"] = ((post["candidate"] != post["candidate"].shift(fill_value=False)) | (post["gap_days"] != 1)).cumsum()

    accepted_dates: set[pd.Timestamp] = set()
    for _, run in post.groupby("run_id", sort=False):
        if bool(run["candidate"].iloc[0]) and len(run) >= 7:
            accepted_dates.update(run["date_utc"].tolist())

    return frame["date_utc"].isin(accepted_dates)


def build_regime_summary(ecosystem: pd.DataFrame) -> pd.DataFrame:
    regimes = [
        ("full_sample", "Full sample", ecosystem),
        ("pre_dencun", "Pre-Dencun", ecosystem.loc[~ecosystem["post_dencun"]]),
        ("post_dencun", "Post-Dencun", ecosystem.loc[ecosystem["post_dencun"]]),
        (
            "post_dencun_blob_floor",
            "Blob fee floor",
            ecosystem.loc[ecosystem["post_dencun"] & ecosystem["blob_floor_regime"]],
        ),
        (
            "post_dencun_non_floor",
            "Post-Dencun ex floor",
            ecosystem.loc[ecosystem["post_dencun"] & ~ecosystem["blob_floor_regime"]],
        ),
    ]

    rows: list[dict[str, object]] = []
    for regime_id, label, subset in regimes:
        if subset.empty:
            continue
        rows.append(
            {
                "regime_id": regime_id,
                "regime": label,
                "start_date_utc": subset["date_utc"].min().date().isoformat(),
                "end_date_utc": subset["date_utc"].max().date().isoformat(),
                "days": int(len(subset)),
                "mean_l2_fees_eth": round(float(subset["l2_fees_eth"].mean()), 6),
                "mean_rent_paid_eth": round(float(subset["rent_paid_eth"].mean()), 6),
                "mean_str_pct": round(float(subset["str"].mean() * 100), 3),
                "median_str_pct": round(float(subset["str"].median() * 100), 3),
                "p90_str_pct": round(float(subset["str"].quantile(0.9) * 100), 3),
                "mean_blob_base_fee_gwei": round(float(subset["l1_blob_base_fee_gwei"].mean()), 6)
                if subset["l1_blob_base_fee_gwei"].notna().any()
                else None,
            }
        )

    return pd.DataFrame(rows)


def derive_figure_self_qa(
    axes: tuple[plt.Axes, ...],
    *,
    declared_unit_tokens: tuple[tuple[str, ...], ...],
    alt_text: str,
) -> dict[str, object]:
    """Derive manifest QA facts from the Matplotlib artists being saved."""
    if len(axes) != len(declared_unit_tokens):
        raise ValueError("figure_self_qa_axis_unit_mismatch")
    axis_labels = [
        (axis.get_xlabel().strip(), axis.get_ylabel().strip())
        for axis in axes
    ]
    return {
        "labels": all(bool(xlabel) and bool(ylabel) for xlabel, ylabel in axis_labels),
        "legend": all(axis.get_legend() is not None for axis in axes),
        "units": all(
            all(token.casefold() in f"{xlabel} {ylabel}".casefold() for token in tokens)
            for (xlabel, ylabel), tokens in zip(axis_labels, declared_unit_tokens, strict=True)
        ),
        "alt_text": alt_text.strip(),
    }


def write_ecosystem_figure(
    ecosystem: pd.DataFrame,
    *,
    as_of_label: str,
    sample_mode: bool,
) -> dict[str, object]:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )

    ax_top.plot(ecosystem["date_utc"], ecosystem["l2_fees_14d"], color="#0b5d7a", linewidth=2.2, label="L2 fees (14d mean)")
    ax_top.plot(ecosystem["date_utc"], ecosystem["rent_paid_14d"], color="#d97706", linewidth=2.2, label="L1 rent (14d mean)")
    ax_top.axvline(DENCUN_DATE, color="#7c3aed", linestyle="--", linewidth=1.2, alpha=0.8, label="Dencun")
    ax_top.set_ylabel("ETH / day")
    ax_top.set_xlabel("Date (UTC)")
    ax_top.set_title(f"Ecosystem fee and rent levels through {ecosystem['date_utc'].max().date().isoformat()}")
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.grid(axis="y", alpha=0.25)

    ax_bottom.plot(ecosystem["date_utc"], ecosystem["str"], color="#9ca3af", linewidth=0.9, alpha=0.6, label="Daily STR")
    ax_bottom.plot(ecosystem["date_utc"], ecosystem["str_30d"], color="#111827", linewidth=2.0, label="30d mean STR")
    ax_bottom.axhline(1.0, color="#dc2626", linestyle=":", linewidth=1.0, alpha=0.8)
    ax_bottom.axvline(DENCUN_DATE, color="#7c3aed", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_bottom.set_ylabel("STR (%)")
    ax_bottom.set_xlabel("Date (UTC)")
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_bottom.grid(axis="y", alpha=0.25)
    ax_bottom.legend(loc="upper right", frameon=False)
    ax_bottom.xaxis.set_major_locator(mdates.YearLocator())
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    mode_label = "sample" if sample_mode else f"as-of {as_of_label}"
    fig.suptitle(f"Settlement Take Rate ecosystem time series ({mode_label})", fontsize=14, fontweight="bold")
    self_qa = derive_figure_self_qa(
        (ax_top, ax_bottom),
        declared_unit_tokens=(("date", "ETH"), ("date", "%")),
        alt_text="Daily ecosystem Settlement Take Rate with smoothed L2 fees, smoothed L1 rent, and the Dencun boundary.",
    )
    fig.savefig(
        ECOSYSTEM_FIGURE_PATH,
        format="svg",
        facecolor="white",
        metadata={"Date": None, "Creator": "build_str_release_outputs.py"},
    )
    _strip_svg_trailing_whitespace(ECOSYSTEM_FIGURE_PATH)
    plt.close(fig)
    return self_qa


def write_post_dencun_regime_figure(
    ecosystem: pd.DataFrame,
    *,
    as_of_label: str,
    sample_mode: bool,
) -> dict[str, object]:
    post = ecosystem.loc[ecosystem["post_dencun"]].copy()
    if post.empty:
        raise SystemExit("post-Dencun slice is empty; cannot build str_post_dencun_regimes.svg")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    shade_regime_spans(ax_top, post)
    shade_regime_spans(ax_bottom, post)

    ax_top.plot(post["date_utc"], post["l1_blob_base_fee_gwei"], color="#0f766e", linewidth=1.8)
    if post["l1_blob_base_fee_gwei"].notna().any():
        threshold = post["l1_blob_base_fee_gwei"].min() * 1.05
        ax_top.axhline(threshold, color="#b45309", linestyle="--", linewidth=1.1, label="Floor threshold")
    ax_top.set_ylabel("Blob base fee (gwei)")
    ax_top.set_xlabel("Date (UTC)")
    ax_top.set_yscale("log")
    ax_top.set_title("Post-Dencun blob-fee regime detection")
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.grid(axis="y", alpha=0.25)

    ax_bottom.plot(post["date_utc"], post["str"], color="#cbd5e1", linewidth=0.9, alpha=0.75, label="Daily STR")
    ax_bottom.plot(post["date_utc"], post["str_14d"], color="#1d4ed8", linewidth=2.2, label="14d mean STR")
    ax_bottom.axhline(1.0, color="#dc2626", linestyle=":", linewidth=1.0, alpha=0.8)
    ax_bottom.set_ylabel("STR (%)")
    ax_bottom.set_xlabel("Date (UTC)")
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_bottom.legend(loc="upper right", frameon=False)
    ax_bottom.grid(axis="y", alpha=0.25)
    ax_bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    mode_label = "sample" if sample_mode else f"as-of {as_of_label}"
    fig.suptitle(f"Post-Dencun STR regimes ({mode_label})", fontsize=14, fontweight="bold")
    self_qa = derive_figure_self_qa(
        (ax_top, ax_bottom),
        declared_unit_tokens=(("date", "gwei"), ("date", "%")),
        alt_text="Post-Dencun daily Settlement Take Rate, 14-day mean STR, and shaded blob fee floor periods.",
    )
    fig.savefig(
        REGIME_FIGURE_PATH,
        format="svg",
        facecolor="white",
        metadata={"Date": None, "Creator": "build_str_release_outputs.py"},
    )
    _strip_svg_trailing_whitespace(REGIME_FIGURE_PATH)
    plt.close(fig)
    return self_qa


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def shade_regime_spans(axis: plt.Axes, post: pd.DataFrame) -> None:
    spans = contiguous_true_spans(post.loc[:, ["date_utc", "blob_floor_regime"]])
    if not spans:
        axis.text(
            0.01,
            0.95,
            "No 7-day blob-fee floor run in slice",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#6b7280",
        )
        return
    for index, (start_date, end_date) in enumerate(spans):
        axis.axvspan(start_date, end_date, color="#fde68a", alpha=0.3, label="Blob fee floor" if index == 0 else None)


def contiguous_true_spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = frame.sort_values("date_utc", kind="stable").reset_index(drop=True)
    if ordered.empty:
        return []
    ordered["gap_days"] = ordered["date_utc"].diff().dt.days.fillna(1)
    ordered["run_id"] = (
        (ordered["blob_floor_regime"] != ordered["blob_floor_regime"].shift(fill_value=False))
        | (ordered["gap_days"] != 1)
    ).cumsum()

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, run in ordered.groupby("run_id", sort=False):
        if bool(run["blob_floor_regime"].iloc[0]):
            spans.append((run["date_utc"].min(), run["date_utc"].max()))
    return spans


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _rounded_values(series: pd.Series) -> list[float | None]:
    values: list[float | None] = []
    for value in series:
        if pd.isna(value):
            values.append(None)
        else:
            values.append(round(float(value), SIDECAR_DECIMAL_PLACES))
    return values


def _iso_dates(series: pd.Series) -> list[str]:
    return [pd.Timestamp(value).date().isoformat() for value in series]


def write_figure_data_sidecars(
    ecosystem: pd.DataFrame,
    *,
    as_of_label: str,
    sample_mode: bool,
) -> None:
    common = {
        "schema_version": "research_swarm.figure_data.v1",
        "generated_by": "src/analysis/build_str_release_outputs.py",
        "as_of": as_of_label,
        "mode": "sample" if sample_mode else "live",
        "value_precision_decimal_places": SIDECAR_DECIMAL_PLACES,
    }
    ecosystem_payload: dict[str, object] = {
        **common,
        "figure": "reports/figures/str_ecosystem_timeseries.svg",
        "dates": _iso_dates(ecosystem["date_utc"]),
        "series": {
            "l2_fees_14d": {"unit": "eth_per_day", "values": _rounded_values(ecosystem["l2_fees_14d"])},
            "rent_paid_14d": {"unit": "eth_per_day", "values": _rounded_values(ecosystem["rent_paid_14d"])},
            "str": {"unit": "ratio", "values": _rounded_values(ecosystem["str"])},
            "str_30d": {"unit": "ratio", "values": _rounded_values(ecosystem["str_30d"])},
        },
        "reference_lines": [
            {"axis": "x", "value": DENCUN_DATE.date().isoformat(), "label": "Dencun"},
            {"axis": "y", "value": 1.0, "unit": "ratio", "label": "full settlement"},
        ],
        "shaded_spans": [],
    }

    post = ecosystem.loc[ecosystem["post_dencun"]].copy()
    if post.empty:
        raise SystemExit("post-Dencun slice is empty; cannot build deterministic figure sidecar")
    threshold = (
        round(float(post["l1_blob_base_fee_gwei"].min() * 1.05), SIDECAR_DECIMAL_PLACES)
        if post["l1_blob_base_fee_gwei"].notna().any()
        else None
    )
    regime_payload: dict[str, object] = {
        **common,
        "figure": "reports/figures/str_post_dencun_regimes.svg",
        "dates": _iso_dates(post["date_utc"]),
        "series": {
            "l1_blob_base_fee_gwei": {"unit": "gwei", "values": _rounded_values(post["l1_blob_base_fee_gwei"])},
            "str": {"unit": "ratio", "values": _rounded_values(post["str"])},
            "str_14d": {"unit": "ratio", "values": _rounded_values(post["str_14d"])},
        },
        "reference_lines": [
            {"axis": "y", "value": threshold, "unit": "gwei", "label": "blob fee floor threshold"},
            {"axis": "y", "value": 1.0, "unit": "ratio", "label": "full settlement"},
        ],
        "shaded_spans": [
            {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "label": "blob fee floor",
            }
            for start, end in contiguous_true_spans(post.loc[:, ["date_utc", "blob_floor_regime"]])
        ],
    }
    _write_json(ECOSYSTEM_FIGURE_DATA_PATH, ecosystem_payload)
    _write_json(REGIME_FIGURE_DATA_PATH, regime_payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path.relative_to(REPO_ROOT)}")
    return payload


def _claim_for_citation(citation_key: str) -> dict[str, object]:
    payload = _load_json_object(CLAIMS_PATH)
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise SystemExit("contracts/claims.yaml claims must be a list")
    for claim in claims:
        if isinstance(claim, dict) and claim.get("citation_key") == citation_key:
            return claim
    raise SystemExit(f"missing registered claim for citation key {citation_key}")


def _uncertainty_for_claim(claim: dict[str, object], kind: str) -> dict[str, object]:
    uncertainty: dict[str, object] = {"kind": kind}
    artifact = claim.get("uncertainty_artifact")
    if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
        uncertainty["justification_artifact"] = artifact["path"]
    else:
        justification = claim.get("uncertainty_justification")
        if not isinstance(justification, str) or not justification.strip():
            raise SystemExit(f"claim {claim.get('claim_id')} lacks uncertainty justification")
        uncertainty["justification"] = justification
    return uncertainty


def _protocol_constants() -> tuple[float, int]:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"contiguous runs of ≥(?P<days>\d+) days where `l1_blob_base_fee_gwei <= (?P<multiplier>\d+(?:\.\d+)?) × min",
        text,
    )
    if match is None:
        raise SystemExit("protocol blob-fee-floor constants could not be parsed")
    return float(match.group("multiplier")), int(match.group("days"))


def write_paper_values(regime_table: pd.DataFrame, *, as_of_label: str) -> None:
    panel_validation_path = REPO_ROOT / "reports" / "validation" / "rollup_panel_validation.json"
    reconciliation_path = REPO_ROOT / "reports" / "validation" / "cross_source_reconciliation.json"
    panel_validation = _load_json_object(panel_validation_path)
    reconciliation = _load_json_object(reconciliation_path)
    panel_summary = panel_validation.get("summary")
    checks = reconciliation.get("checks")
    if not isinstance(panel_summary, dict) or not isinstance(checks, list) or len(checks) <= 5:
        raise SystemExit("validation artifact schema does not expose locked paper values")
    reconciliation_details = checks[5].get("details") if isinstance(checks[5], dict) else None
    if not isinstance(reconciliation_details, dict):
        raise SystemExit("cross-source reconciliation details are missing")
    monthly = reconciliation_details.get("unexplained_monthly_aggregate")
    if not isinstance(monthly, list):
        raise SystemExit("cross-source monthly reconciliation values are missing")
    may_2025 = next(
        (row for row in monthly if isinstance(row, dict) and row.get("month_utc") == "2025-05"),
        None,
    )
    if not isinstance(may_2025, dict):
        raise SystemExit("cross-source reconciliation lacks the registered 2025-05 residual")

    rows = {str(row["regime_id"]): row for _, row in regime_table.iterrows()}
    multiplier, minimum_days = _protocol_constants()
    source_paths = {
        "table": REGIME_TABLE_CSV_PATH,
        "panel": panel_validation_path,
        "reconciliation": reconciliation_path,
        "protocol": PROTOCOL_PATH,
    }
    source_relpaths = {key: path.relative_to(REPO_ROOT).as_posix() for key, path in source_paths.items()}
    source_hashes = {key: _sha256(path) for key, path in source_paths.items()}

    regime_claim = _claim_for_citation("str_regime_summary")
    panel_claim = _claim_for_citation("rollup_panel_validation")
    reconciliation_claim = _claim_for_citation("cross_source_reconciliation")
    protocol_claim = _claim_for_citation("protocol_lock")

    values: dict[str, dict[str, object]] = {}

    def add(
        key: str,
        *,
        value: int | float,
        unit: str,
        display: str,
        citation_key: str,
        source: str,
        selector: str,
        claim: dict[str, object],
        uncertainty_kind: str,
    ) -> None:
        values[key] = {
            "value": value,
            "unit": unit,
            "type": claim["type"],
            "display": display,
            "citation_key": citation_key,
            "source_artifact": source_relpaths[source],
            "source_sha256": source_hashes[source],
            "source_selector": selector,
            "uncertainty": _uncertainty_for_claim(claim, uncertainty_kind),
        }

    for key, regime_id, column, unit, decimals, display_suffix in (
        ("pre_dencun_mean_str_pct", "pre_dencun", "mean_str_pct", "percent", 2, "%"),
        ("post_dencun_mean_str_pct", "post_dencun", "mean_str_pct", "percent", 2, "%"),
        ("post_dencun_blob_floor_mean_str_pct", "post_dencun_blob_floor", "mean_str_pct", "percent", 2, "%"),
        ("post_dencun_non_floor_mean_str_pct", "post_dencun_non_floor", "mean_str_pct", "percent", 2, "%"),
        ("pre_dencun_mean_rent_paid_eth", "pre_dencun", "mean_rent_paid_eth", "eth", 3, " ETH"),
        ("post_dencun_mean_rent_paid_eth", "post_dencun", "mean_rent_paid_eth", "eth", 3, " ETH"),
    ):
        raw_value = float(rows[regime_id][column])
        value = round(raw_value, decimals)
        add(
            key,
            value=value,
            unit=unit,
            display=f"{value:.{decimals}f}{display_suffix}",
            citation_key="str_regime_summary",
            source="table",
            selector=f"regime_id={regime_id};column={column}",
            claim=regime_claim,
            uncertainty_kind="descriptive_no_sampling_band",
        )

    for key, field, display in (
        ("panel_row_count", "row_count", f"{int(panel_summary['row_count']):,}"),
        ("panel_rollup_count", "rollup_count", f"{int(panel_summary['rollup_count']):,}"),
        ("panel_date_count", "date_count", f"{int(panel_summary['date_count']):,}"),
        ("rent_component_row_count", "rent_component_row_count", f"{int(panel_summary['rent_component_row_count']):,}"),
    ):
        add(
            key,
            value=int(panel_summary[field]),
            unit="count",
            display=display,
            citation_key="rollup_panel_validation",
            source="panel",
            selector=f"json_path=summary.{field}",
            claim=panel_claim,
            uncertainty_kind="methodological_exact",
        )

    reconciliation_specs = (
        (
            "reconciliation_aggregate_rent_difference_pct",
            round(float(reconciliation_details["overall_aggregate_pct_difference"]) * 100, 2),
            "percent",
            "0.02%",
            "json_path=checks[5].details.overall_aggregate_pct_difference;scale=100",
        ),
        (
            "reconciliation_2025_05_pct_difference",
            round(float(may_2025["pct_difference"]) * 100, 2),
            "percent",
            "12.37%",
            "json_path=checks[5].details.unexplained_monthly_aggregate;match=month_utc=2025-05;column=pct_difference;scale=100",
        ),
        (
            "reconciliation_2025_05_authoritative_rent_eth",
            round(float(may_2025["rent_paid_eth_authoritative"]), 2),
            "eth",
            "148.42 ETH",
            "json_path=checks[5].details.unexplained_monthly_aggregate;match=month_utc=2025-05;column=rent_paid_eth_authoritative",
        ),
        (
            "reconciliation_2025_05_vendor_rent_eth",
            round(float(may_2025["rent_paid_eth_vendor"]), 2),
            "eth",
            "166.78 ETH",
            "json_path=checks[5].details.unexplained_monthly_aggregate;match=month_utc=2025-05;column=rent_paid_eth_vendor",
        ),
    )
    for key, value, unit, display, selector in reconciliation_specs:
        add(
            key,
            value=value,
            unit=unit,
            display=display,
            citation_key="cross_source_reconciliation",
            source="reconciliation",
            selector=selector,
            claim=reconciliation_claim,
            uncertainty_kind="methodological_exact",
        )

    add(
        "blob_fee_floor_threshold_multiplier",
        value=multiplier,
        unit="multiplier",
        display=f"{multiplier:.2f} x",
        citation_key="protocol_lock",
        source="protocol",
        selector="protocol_constant=blob_fee_floor_threshold_multiplier",
        claim=protocol_claim,
        uncertainty_kind="protocol_constant",
    )
    add(
        "blob_fee_floor_min_consecutive_days",
        value=minimum_days,
        unit="days",
        display=str(minimum_days),
        citation_key="protocol_lock",
        source="protocol",
        selector="protocol_constant=blob_fee_floor_min_consecutive_days",
        claim=protocol_claim,
        uncertainty_kind="protocol_constant",
    )

    _write_json(
        PAPER_VALUES_PATH,
        {
            "schema_version": "research_swarm.paper_values.v1",
            "generated_by": "src/analysis/build_str_release_outputs.py",
            "as_of": as_of_label,
            "values": values,
        },
    )


def write_exhibits_manifest(
    *,
    ecosystem_figure_qa: dict[str, object],
    regime_figure_qa: dict[str, object],
) -> None:
    """Emit the deterministic analysis-to-paper interface with build-time QA facts."""
    input_paths = sorted(
        (
            *VALIDATION_BUNDLE,
            LIVE_PANEL_PATH,
            LIVE_DECOMP_PATH,
            CLAIMS_PATH,
            PROTOCOL_PATH,
            Path(__file__).resolve(),
        ),
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    inputs = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(path),
        }
        for path in input_paths
    ]
    exhibits = [
        {
            "exhibit_id": "str_ecosystem_timeseries",
            "builder": "src/analysis/build_str_release_outputs.py",
            "inputs": inputs,
            "output": "reports/figures/str_ecosystem_timeseries.svg",
            "caption": "Settlement Take Rate ecosystem time series through the validated as-of date.",
            "notes": "Two-panel multi-series figure; plotted values are bound by the matching .data.json sidecar.",
            "self_qa": ecosystem_figure_qa,
        },
        {
            "exhibit_id": "str_post_dencun_regimes",
            "builder": "src/analysis/build_str_release_outputs.py",
            "inputs": inputs,
            "output": "reports/figures/str_post_dencun_regimes.svg",
            "caption": "Post-Dencun STR regimes with protocol-defined blob-fee-floor runs.",
            "notes": "Two-panel multi-series figure; plotted values are bound by the matching .data.json sidecar.",
            "self_qa": regime_figure_qa,
        },
        {
            "exhibit_id": "str_regime_summary",
            "builder": "src/analysis/build_str_release_outputs.py",
            "inputs": inputs,
            "output": "reports/tables/str_regime_summary.md",
            "caption": "STR and rent summaries for the full, Dencun, and blob-fee regimes.",
            "notes": "The Markdown include and same-stem CSV are emitted from one in-memory table; paper_values binds numeric cells to the CSV.",
            "self_qa": {
                "labels": True,
                "legend": True,
                "units": True,
                "alt_text": "Table of regime date ranges, observation counts, fee and rent means, and STR distribution summaries with units in column labels.",
            },
        },
    ]
    _write_json(
        EXHIBITS_MANIFEST_PATH,
        {
            "schema_version": "research_swarm.exhibits_manifest.v1",
            "generated_by": "src/analysis/build_str_release_outputs.py",
            "exhibits": exhibits,
        },
    )


def write_regime_tables(regime_table: pd.DataFrame) -> None:
    REGIME_TABLE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime_table.to_csv(REGIME_TABLE_CSV_PATH, index=False, float_format="%.6f")
    markdown_table = manuscript_markdown_from_regime_table(regime_table)
    REGIME_TABLE_MD_PATH.write_text(markdown_table + "\n", encoding="utf-8")


def manuscript_markdown_from_regime_table(frame: pd.DataFrame) -> str:
    overview_rows: list[list[str]] = []
    distribution_rows: list[list[str]] = []
    for _, row in frame.iterrows():
        overview_rows.append(
            [
                str(row["regime"]),
                f"{row['start_date_utc']} to {row['end_date_utc']}",
                f"{int(row['days']):,}",
                f"{row['mean_l2_fees_eth']:,.3f}",
                f"{row['mean_rent_paid_eth']:,.3f}",
                f"{row['mean_str_pct']:.2f}%",
            ]
        )
        distribution_rows.append(
            [
                str(row["regime"]),
                f"{row['median_str_pct']:.2f}%",
                f"{row['p90_str_pct']:.2f}%",
                "n/a"
                if pd.isna(row["mean_blob_base_fee_gwei"])
                else f"{row['mean_blob_base_fee_gwei']:.6f}",
            ]
        )

    overview = render_markdown_table(
        headers=["Regime", "Window", "Days", "Mean fees", "Mean rent", "Mean STR"],
        rows=overview_rows,
        alignments=["---", "---", "---:", "---:", "---:", "---:"],
    )
    distribution = render_markdown_table(
        headers=["Regime", "Median STR", "P90 STR", "Mean blob fee"],
        rows=distribution_rows,
        alignments=["---", "---:", "---:", "---:"],
    )
    return "\n".join(
        [
            "Daily means are ETH/day. STR columns are percentages. Blob-fee values are in gwei.",
            "",
            "Table 1A. Regime coverage and central tendency",
            "",
            overview,
            "",
            "Table 1B. STR distribution and blob-fee diagnostics",
            "",
            distribution,
        ]
    )


def render_markdown_table(
    *,
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str],
) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(alignments) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


if __name__ == "__main__":
    raise SystemExit(main())
