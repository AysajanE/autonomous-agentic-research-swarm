#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.analysis.metrics_str import compute_ecosystem_str, compute_rollup_str


REPORT_DIR = REPO_ROOT / "reports" / "validation"

PANEL_REQUIRED_COLUMNS = (
    "date_utc",
    "rollup_id",
    "l2_fees_eth",
    "rent_paid_eth",
)
PANEL_OPTIONAL_COLUMNS = ("profit_eth", "txcount")

DECOMP_REQUIRED_COLUMNS = (
    "date_utc",
    "l1_base_fee_burn_eth",
    "l1_blob_fee_burn_eth",
    "l1_priority_fee_eth",
    "l1_total_rent_eth",
)
DECOMP_OPTIONAL_COLUMNS = (
    "l1_blob_gas_used",
    "l1_calldata_gas_used",
    "l1_blob_base_fee_gwei",
)

VENDOR_REQUIRED_COLUMNS = PANEL_REQUIRED_COLUMNS
VENDOR_OPTIONAL_COLUMNS = PANEL_OPTIONAL_COLUMNS

RECONCILIATION_PASS_THRESHOLD = Decimal("0.10")

REPORT_SPECS = (
    ("rollup_panel_validation", "Canonical rollup panel validation"),
    ("l1_rent_decomposition_validation", "L1 rent decomposition validation"),
    ("cross_source_reconciliation", "Cross-source reconciliation"),
)


@dataclass(frozen=True)
class DataArtifact:
    dataset: str
    path: Path
    manifest_path: Path | None
    as_of_utc_date: str | None


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    details: dict[str, object] = field(default_factory=dict)
    plausible_causes: list[str] = field(default_factory=list)
    next_step: str | None = None


@dataclass(frozen=True)
class ReportPayload:
    report_id: str
    title: str
    status: str
    mode: str
    as_of_utc_date: str | None
    summary: dict[str, object]
    checks: list[CheckResult]
    provenance: dict[str, object]


class ValidationBlocked(RuntimeError):
    def __init__(self, message: str, issues: dict[str, object]) -> None:
        super().__init__(message)
        self.issues = issues


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical STR inputs and emit validation reports.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Validate the tracked sample CSVs under data/samples/.",
    )
    mode.add_argument(
        "--as-of",
        metavar="YYYY-MM-DD",
        help="Validate the canonical manifest-backed artifacts for an as-of date.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    mode = "sample" if args.sample else "canonical"
    as_of = None if args.sample else args.as_of

    try:
        artifacts = resolve_artifacts(mode=mode, as_of_utc_date=as_of)
        reports = build_reports(mode=mode, artifacts=artifacts)
    except ValidationBlocked as exc:
        reports = build_blocked_reports(mode=mode, as_of_utc_date=as_of, issues=exc.issues)
        write_reports(reports)
        return 2

    write_reports(reports)
    if any(report.status != "pass" for report in reports):
        return 1
    return 0


def resolve_artifacts(mode: str, as_of_utc_date: str | None) -> dict[str, DataArtifact]:
    if mode == "sample":
        return {
            "vendor_panel": DataArtifact(
                dataset="vendor_panel",
                path=REPO_ROOT / "data" / "samples" / "growthepie" / "vendor_daily_rollup_panel_sample.csv",
                manifest_path=None,
                as_of_utc_date=None,
            ),
            "l1_decomposition": DataArtifact(
                dataset="l1_decomposition",
                path=REPO_ROOT / "data" / "samples" / "l1_rent" / "daily_l1_rent_decomposition_sample.csv",
                manifest_path=None,
                as_of_utc_date=None,
            ),
            "authoritative_panel": DataArtifact(
                dataset="authoritative_panel",
                path=REPO_ROOT / "data" / "samples" / "panels" / "daily_rollup_panel_sample.csv",
                manifest_path=None,
                as_of_utc_date=None,
            ),
        }

    if as_of_utc_date is None:
        raise ValueError("as_of_utc_date is required in canonical mode")

    manifests = {
        "vendor_panel": (
            REPO_ROOT / "data" / "processed_manifest" / f"vendor_daily_rollup_panel_{as_of_utc_date}.json",
            "vendor_daily_rollup_panel.csv",
        ),
        "l1_decomposition": (
            REPO_ROOT / "data" / "processed_manifest" / f"daily_l1_rent_decomposition_{as_of_utc_date}.json",
            "daily_l1_rent_decomposition.csv",
        ),
        "authoritative_panel": (
            REPO_ROOT / "data" / "processed_manifest" / f"daily_rollup_panel_{as_of_utc_date}.json",
            "daily_rollup_panel.csv",
        ),
    }

    artifacts: dict[str, DataArtifact] = {}
    missing_manifests: list[str] = []
    missing_artifacts: list[dict[str, str]] = []

    for dataset, (manifest_path, expected_output_name) in manifests.items():
        if not manifest_path.exists():
            missing_manifests.append(str(manifest_path.relative_to(REPO_ROOT)))
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_path = select_manifest_output_path(
            manifest=manifest,
            expected_output_name=expected_output_name,
        )
        if artifact_path is None:
            missing_artifacts.append(
                {
                    "dataset": dataset,
                    "reason": f"manifest does not declare {expected_output_name}",
                    "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
                }
            )
            continue

        resolved_path = REPO_ROOT / artifact_path
        if not resolved_path.exists():
            missing_artifacts.append(
                {
                    "dataset": dataset,
                    "reason": "processed artifact is absent in the worktree",
                    "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
                    "expected_artifact_path": artifact_path,
                }
            )
            continue

        artifacts[dataset] = DataArtifact(
            dataset=dataset,
            path=resolved_path,
            manifest_path=manifest_path,
            as_of_utc_date=str(manifest.get("as_of_utc_date") or as_of_utc_date),
        )

    if missing_manifests or missing_artifacts:
        raise ValidationBlocked(
            "Canonical validation inputs are incomplete",
            {
                "missing_manifests": missing_manifests,
                "missing_artifacts": missing_artifacts,
                "requested_as_of_utc_date": as_of_utc_date,
                "next_step": (
                    "Restore the manifest-backed processed CSVs locally or rerun the producing ETL "
                    f"for {as_of_utc_date} before re-running this validator."
                ),
            },
        )

    return artifacts


def select_manifest_output_path(manifest: dict[str, object], expected_output_name: str) -> str | None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        return None

    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        if Path(path).name == expected_output_name:
            return path
    return None


def build_reports(mode: str, artifacts: dict[str, DataArtifact]) -> list[ReportPayload]:
    vendor_panel = read_csv_frame(artifacts["vendor_panel"].path)
    l1_decomposition = read_csv_frame(artifacts["l1_decomposition"].path)
    authoritative_panel = read_csv_frame(artifacts["authoritative_panel"].path)

    rollup_checks = build_rollup_panel_checks(authoritative_panel, vendor_panel)
    decomp_checks = build_l1_decomposition_checks(l1_decomposition, authoritative_panel)
    reconciliation_checks = build_cross_source_checks(vendor_panel, authoritative_panel)

    as_of_values = {
        artifact.as_of_utc_date
        for artifact in artifacts.values()
        if artifact.as_of_utc_date is not None
    }
    as_of_utc_date = next(iter(as_of_values), None)

    provenance = build_provenance(mode=mode, artifacts=artifacts)
    reports = [
        ReportPayload(
            report_id="rollup_panel_validation",
            title="Canonical rollup panel validation",
            status=aggregate_status(rollup_checks),
            mode=mode,
            as_of_utc_date=as_of_utc_date,
            summary=summarize_rollup_panel(authoritative_panel, vendor_panel),
            checks=rollup_checks,
            provenance=provenance,
        ),
        ReportPayload(
            report_id="l1_rent_decomposition_validation",
            title="L1 rent decomposition validation",
            status=aggregate_status(decomp_checks),
            mode=mode,
            as_of_utc_date=as_of_utc_date,
            summary=summarize_l1_decomposition(l1_decomposition, authoritative_panel),
            checks=decomp_checks,
            provenance=provenance,
        ),
        ReportPayload(
            report_id="cross_source_reconciliation",
            title="Cross-source reconciliation",
            status=aggregate_status(reconciliation_checks),
            mode=mode,
            as_of_utc_date=as_of_utc_date,
            summary=summarize_cross_source(vendor_panel, authoritative_panel),
            checks=reconciliation_checks,
            provenance=provenance,
        ),
    ]
    return reports


def build_blocked_reports(
    mode: str,
    as_of_utc_date: str | None,
    issues: dict[str, object],
) -> list[ReportPayload]:
    blocked_check = CheckResult(
        name="canonical_input_resolution",
        status="blocked",
        details=issues,
        plausible_causes=[
            "The processed manifests exist but their referenced CSV outputs were not materialized in this worktree.",
            "The ETL run that produced the manifests may have been executed outside this sandbox and only the manifests were committed.",
        ],
        next_step=str(issues.get("next_step")),
    )
    provenance = {
        "mode": mode,
        "artifacts": [],
    }
    summary = {
        "message": "Canonical validation could not run because manifest-backed inputs are incomplete.",
        "requested_as_of_utc_date": as_of_utc_date,
    }
    return [
        ReportPayload(
            report_id=report_id,
            title=title,
            status="blocked",
            mode=mode,
            as_of_utc_date=as_of_utc_date,
            summary=summary,
            checks=[blocked_check],
            provenance=provenance,
        )
        for report_id, title in REPORT_SPECS
    ]


def read_csv_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return frame


def build_rollup_panel_checks(panel: pd.DataFrame, vendor_panel: pd.DataFrame) -> list[CheckResult]:
    checks = [
        check_required_columns(
            report_name="authoritative_panel_schema",
            frame=panel,
            required_columns=PANEL_REQUIRED_COLUMNS,
        ),
        check_primary_key_uniqueness(
            report_name="authoritative_panel_primary_key_uniqueness",
            frame=panel,
            key_columns=("date_utc", "rollup_id"),
        ),
        check_required_non_null(
            report_name="authoritative_panel_required_non_null",
            frame=panel,
            required_columns=PANEL_REQUIRED_COLUMNS,
        ),
        check_key_coverage(panel=panel, vendor_panel=vendor_panel),
        check_metrics_compatibility(panel=panel),
    ]
    return checks


def build_l1_decomposition_checks(
    decomposition: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
) -> list[CheckResult]:
    checks = [
        check_required_columns(
            report_name="l1_decomposition_schema",
            frame=decomposition,
            required_columns=DECOMP_REQUIRED_COLUMNS,
        ),
        check_primary_key_uniqueness(
            report_name="l1_decomposition_primary_key_uniqueness",
            frame=decomposition,
            key_columns=("date_utc",),
        ),
        check_required_non_null(
            report_name="l1_decomposition_required_non_null",
            frame=decomposition,
            required_columns=DECOMP_REQUIRED_COLUMNS,
        ),
        check_l1_identity(decomposition),
        check_decomposition_covers_panel_dates(decomposition, authoritative_panel),
    ]
    return checks


def build_cross_source_checks(
    vendor_panel: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
) -> list[CheckResult]:
    merged = vendor_panel.merge(
        authoritative_panel[["date_utc", "rollup_id", "rent_paid_eth"]],
        on=["date_utc", "rollup_id"],
        how="outer",
        suffixes=("_vendor", "_authoritative"),
        indicator=True,
    )

    checks = [
        check_required_columns(
            report_name="vendor_panel_schema",
            frame=vendor_panel,
            required_columns=VENDOR_REQUIRED_COLUMNS,
        ),
        check_primary_key_uniqueness(
            report_name="vendor_panel_primary_key_uniqueness",
            frame=vendor_panel,
            key_columns=("date_utc", "rollup_id"),
        ),
        check_required_non_null(
            report_name="vendor_panel_required_non_null",
            frame=vendor_panel,
            required_columns=VENDOR_REQUIRED_COLUMNS,
        ),
        check_vendor_profit_identity(vendor_panel),
        check_monthly_reconciliation(merged),
    ]
    return checks


def check_required_columns(
    report_name: str,
    frame: pd.DataFrame,
    required_columns: Iterable[str],
) -> CheckResult:
    missing = [column for column in required_columns if column not in frame.columns]
    status = "pass" if not missing else "fail"
    return CheckResult(
        name=report_name,
        status=status,
        details={
            "required_columns": list(required_columns),
            "present_columns": list(frame.columns),
            "missing_columns": missing,
        },
        plausible_causes=(
            []
            if not missing
            else [
                "The producing ETL emitted a schema that diverges from the locked contracts.",
                "A downstream artifact was truncated or replaced with the wrong table.",
            ]
        ),
        next_step=(
            None
            if not missing
            else "Compare the produced CSV header to the locked schema contract before advancing downstream work."
        ),
    )


def check_primary_key_uniqueness(
    report_name: str,
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> CheckResult:
    if any(column not in frame.columns for column in key_columns):
        return CheckResult(
            name=report_name,
            status="fail",
            details={
                "key_columns": list(key_columns),
                "reason": "key columns are missing; uniqueness could not be evaluated",
            },
            plausible_causes=["Required key fields are absent from the artifact schema."],
            next_step="Restore the locked key columns, then rerun validation.",
        )

    duplicate_rows = frame.duplicated(list(key_columns), keep=False)
    duplicate_count = int(duplicate_rows.sum())
    sample_duplicates = []
    if duplicate_count:
        sample_duplicates = (
            frame.loc[duplicate_rows, list(key_columns)]
            .head(10)
            .to_dict(orient="records")
        )
    return CheckResult(
        name=report_name,
        status="pass" if duplicate_count == 0 else "fail",
        details={
            "key_columns": list(key_columns),
            "row_count": int(len(frame)),
            "duplicate_row_count": duplicate_count,
            "sample_duplicates": sample_duplicates,
        },
        plausible_causes=(
            []
            if duplicate_count == 0
            else [
                "The ETL joined multiple source rows into the same canonical grain.",
                "The artifact includes rerun residue instead of a deduplicated output.",
            ]
        ),
        next_step=(
            None
            if duplicate_count == 0
            else "Trace the duplicated keys back to the producing ETL join or aggregation step."
        ),
    )


def check_required_non_null(
    report_name: str,
    frame: pd.DataFrame,
    required_columns: Iterable[str],
) -> CheckResult:
    present_columns = [column for column in required_columns if column in frame.columns]
    null_counts = {
        column: int((frame[column] == "").sum())
        for column in present_columns
    }
    violating_columns = {column: count for column, count in null_counts.items() if count}
    return CheckResult(
        name=report_name,
        status="pass" if not violating_columns else "fail",
        details={
            "null_counts": null_counts,
            "violating_columns": violating_columns,
        },
        plausible_causes=(
            []
            if not violating_columns
            else [
                "The artifact encoded missingness with nulls instead of the contract’s row-omission rule.",
                "A numeric conversion step dropped values after the upstream grain was fixed.",
            ]
        ),
        next_step=(
            None
            if not violating_columns
            else "Rebuild the artifact so missing panel metrics are represented by row omission, not empty required fields."
        ),
    )


def check_key_coverage(panel: pd.DataFrame, vendor_panel: pd.DataFrame) -> CheckResult:
    panel_keys = set(zip(panel["date_utc"], panel["rollup_id"]))
    vendor_keys = set(zip(vendor_panel["date_utc"], vendor_panel["rollup_id"]))
    only_in_panel = sorted(panel_keys - vendor_keys)[:10]
    only_in_vendor = sorted(vendor_keys - panel_keys)[:10]
    status = "pass" if panel_keys == vendor_keys else "fail"
    return CheckResult(
        name="authoritative_vs_vendor_key_coverage",
        status=status,
        details={
            "authoritative_panel_key_count": len(panel_keys),
            "vendor_panel_key_count": len(vendor_keys),
            "only_in_authoritative_panel": [
                {"date_utc": date_utc, "rollup_id": rollup_id}
                for date_utc, rollup_id in only_in_panel
            ],
            "only_in_vendor_panel": [
                {"date_utc": date_utc, "rollup_id": rollup_id}
                for date_utc, rollup_id in only_in_vendor
            ],
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The authoritative panel and vendor panel were built from different rollup universes or sample windows.",
                "One pipeline emitted rows with missing paired metrics while the other followed the row-omission rule.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Reconcile the key-level coverage mismatch before trusting STR comparisons or downstream figures."
        ),
    )


def check_metrics_compatibility(panel: pd.DataFrame) -> CheckResult:
    numeric_panel = numeric_frame(panel, columns=("l2_fees_eth", "rent_paid_eth"))
    rollup_str = compute_rollup_str(numeric_panel.copy())
    ecosystem_str = compute_ecosystem_str(numeric_panel.copy())
    distinct_dates = int(panel["date_utc"].nunique()) if "date_utc" in panel.columns else 0

    row_count_match = len(rollup_str) == len(panel)
    ecosystem_count_match = len(ecosystem_str) == distinct_dates
    status = "pass" if row_count_match and ecosystem_count_match else "fail"
    return CheckResult(
        name="metrics_module_compatibility",
        status=status,
        details={
            "authoritative_panel_row_count": int(len(panel)),
            "rollup_str_row_count": int(len(rollup_str)),
            "distinct_date_count": distinct_dates,
            "ecosystem_str_row_count": int(len(ecosystem_str)),
            "ecosystem_str_preview": ecosystem_str.head(5).to_dict(orient="records"),
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The panel does not satisfy the assumptions locked into src.analysis.metrics_str.",
                "Unexpected nulls or zero-denominator days changed the metric-layer row counts.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Inspect the panel rows excluded by the metrics module before using the STR outputs downstream."
        ),
    )


def check_l1_identity(decomposition: pd.DataFrame) -> CheckResult:
    required = [
        "l1_base_fee_burn_eth",
        "l1_blob_fee_burn_eth",
        "l1_priority_fee_eth",
        "l1_total_rent_eth",
    ]
    max_abs_diff = Decimal("0")
    violating_rows: list[dict[str, object]] = []

    for _, row in decomposition.iterrows():
        try:
            base_fee = decimal_from_value(row["l1_base_fee_burn_eth"])
            blob_fee = decimal_from_value(row["l1_blob_fee_burn_eth"])
            priority_fee = decimal_from_value(row["l1_priority_fee_eth"])
            total_rent = decimal_from_value(row["l1_total_rent_eth"])
        except (InvalidOperation, KeyError):
            return CheckResult(
                name="l1_total_rent_identity",
                status="fail",
                details={"required_columns": required},
                plausible_causes=["A decomposition value could not be parsed as a Decimal ETH quantity."],
                next_step="Restore numeric ETH values for the decomposition components and total rent field.",
            )

        diff = abs(total_rent - (base_fee + blob_fee + priority_fee))
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff != 0 and len(violating_rows) < 10:
            violating_rows.append(
                {
                    "date_utc": row["date_utc"],
                    "difference_eth": str(diff),
                }
            )

    status = "pass" if not violating_rows else "fail"
    return CheckResult(
        name="l1_total_rent_identity",
        status=status,
        details={
            "row_count": int(len(decomposition)),
            "max_abs_difference_eth": str(max_abs_diff),
            "violating_rows": violating_rows,
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The authoritative decomposition lost precision or mixed units during aggregation.",
                "One L1 rent component was omitted or double-counted in the total.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Trace the failing days back to the on-chain aggregation that populated the decomposition CSV."
        ),
    )


def check_decomposition_covers_panel_dates(
    decomposition: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
) -> CheckResult:
    decomp_dates = set(decomposition["date_utc"]) if "date_utc" in decomposition.columns else set()
    panel_dates = set(authoritative_panel["date_utc"]) if "date_utc" in authoritative_panel.columns else set()
    missing_dates = sorted(panel_dates - decomp_dates)[:10]
    status = "pass" if not missing_dates else "fail"
    return CheckResult(
        name="decomposition_covers_panel_dates",
        status=status,
        details={
            "decomposition_date_count": len(decomp_dates),
            "panel_date_count": len(panel_dates),
            "missing_panel_dates": missing_dates,
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The canonical panel extends beyond the decomposition sample window.",
                "The decomposition ETL dropped days that the panel retained.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Rebuild the decomposition so every authoritative panel date has an L1 rent row."
        ),
    )


def check_vendor_profit_identity(vendor_panel: pd.DataFrame) -> CheckResult:
    if "profit_eth" not in vendor_panel.columns:
        return CheckResult(
            name="vendor_profit_identity",
            status="pass",
            details={
                "evaluated_row_count": 0,
                "skipped": True,
                "reason": "profit_eth column is absent; the contract marks it optional",
            },
        )

    evaluated = vendor_panel.loc[vendor_panel["profit_eth"] != ""].copy()
    if evaluated.empty:
        return CheckResult(
            name="vendor_profit_identity",
            status="pass",
            details={
                "evaluated_row_count": 0,
                "skipped": True,
                "reason": "profit_eth values are empty; nothing to validate",
            },
        )

    numeric = numeric_frame(
        evaluated,
        columns=("l2_fees_eth", "rent_paid_eth", "profit_eth"),
    )
    numeric["expected_profit_eth"] = numeric["l2_fees_eth"] - numeric["rent_paid_eth"]
    numeric["abs_difference_eth"] = (numeric["profit_eth"] - numeric["expected_profit_eth"]).abs()
    numeric["tolerance_eth"] = numeric.apply(
        lambda row: vendor_identity_tolerance(
            fees=row["l2_fees_eth"],
            rent_paid=row["rent_paid_eth"],
        ),
        axis=1,
    )
    violations = numeric.loc[numeric["abs_difference_eth"] > numeric["tolerance_eth"]].copy()

    status = "pass" if violations.empty else "fail"
    return CheckResult(
        name="vendor_profit_identity",
        status=status,
        details={
            "evaluated_row_count": int(len(numeric)),
            "violation_count": int(len(violations)),
            "max_abs_difference_eth": float(numeric["abs_difference_eth"].max()),
            "max_tolerance_eth": float(numeric["tolerance_eth"].max()),
            "sample_violations": violations.head(10)[
                [
                    "date_utc",
                    "rollup_id",
                    "profit_eth",
                    "expected_profit_eth",
                    "abs_difference_eth",
                    "tolerance_eth",
                ]
            ].to_dict(orient="records"),
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The vendor profit series drifted away from fees minus rent beyond the protocol tolerance.",
                "Source rounding or unit conversion changed one of the vendor series without updating the others.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Inspect the failing vendor rows and confirm whether the source changed its profit definition or units."
        ),
    )


def check_monthly_reconciliation(merged: pd.DataFrame) -> CheckResult:
    missing_keys = merged.loc[merged["_merge"] != "both", ["date_utc", "rollup_id", "_merge"]]
    if not missing_keys.empty:
        details = {
            "mismatched_key_count": int(len(missing_keys)),
            "sample_key_mismatches": missing_keys.head(10).to_dict(orient="records"),
        }
        return CheckResult(
            name="monthly_cross_source_reconciliation",
            status="fail",
            details=details,
            plausible_causes=[
                "The vendor and authoritative panels do not cover the same rollup-day keys.",
                "One input was built from a different sample window or rollup registry snapshot.",
            ],
            next_step="Resolve the key mismatch before interpreting cross-source rent deltas.",
        )

    numeric = numeric_frame(
        merged,
        columns=("rent_paid_eth_vendor", "rent_paid_eth_authoritative"),
    )
    numeric["month_utc"] = numeric["date_utc"].str.slice(0, 7)

    monthly = (
        numeric.groupby(["month_utc", "rollup_id"], as_index=False)[
            ["rent_paid_eth_vendor", "rent_paid_eth_authoritative"]
        ]
        .sum()
        .sort_values(["month_utc", "rent_paid_eth_authoritative", "rollup_id"], ascending=[True, False, True], kind="stable")
        .reset_index(drop=True)
    )
    monthly["pct_difference"] = monthly.apply(
        lambda row: percent_difference(
            row["rent_paid_eth_vendor"],
            row["rent_paid_eth_authoritative"],
        ),
        axis=1,
    )

    aggregate = (
        numeric.groupby("month_utc", as_index=False)[
            ["rent_paid_eth_vendor", "rent_paid_eth_authoritative"]
        ]
        .sum()
        .sort_values("month_utc", kind="stable")
        .reset_index(drop=True)
    )
    aggregate["pct_difference"] = aggregate.apply(
        lambda row: percent_difference(
            row["rent_paid_eth_vendor"],
            row["rent_paid_eth_authoritative"],
        ),
        axis=1,
    )

    rollup_violations = monthly.loc[monthly["pct_difference"] > float(RECONCILIATION_PASS_THRESHOLD)]
    aggregate_violations = aggregate.loc[aggregate["pct_difference"] > float(RECONCILIATION_PASS_THRESHOLD)]
    status = "pass" if rollup_violations.empty and aggregate_violations.empty else "fail"

    daily = numeric.copy()
    daily["pct_difference"] = daily.apply(
        lambda row: percent_difference(
            row["rent_paid_eth_vendor"],
            row["rent_paid_eth_authoritative"],
        ),
        axis=1,
    )
    daily_outliers = daily.loc[daily["pct_difference"] > float(RECONCILIATION_PASS_THRESHOLD)]

    return CheckResult(
        name="monthly_cross_source_reconciliation",
        status=status,
        details={
            "target_tolerance_pct": float(RECONCILIATION_PASS_THRESHOLD * Decimal("100")),
            "monthly_top_rollups": monthly.head(10).to_dict(orient="records"),
            "monthly_aggregate": aggregate.to_dict(orient="records"),
            "rollup_violation_count": int(len(rollup_violations)),
            "aggregate_violation_count": int(len(aggregate_violations)),
            "daily_outlier_count": int(len(daily_outliers)),
            "sample_daily_outliers": daily_outliers.head(10)[
                [
                    "date_utc",
                    "rollup_id",
                    "rent_paid_eth_vendor",
                    "rent_paid_eth_authoritative",
                    "pct_difference",
                ]
            ].to_dict(orient="records"),
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The authoritative on-chain rent series diverged materially from the vendor proxy for a top rollup-month.",
                "A registry or attribution change altered the canonical rollup universe between sources.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Document the dominant rollup-month deltas and isolate whether the gap comes from source coverage or attribution logic."
        ),
    )


def summarize_rollup_panel(panel: pd.DataFrame, vendor_panel: pd.DataFrame) -> dict[str, object]:
    numeric_panel = numeric_frame(panel, columns=("l2_fees_eth", "rent_paid_eth"))
    ecosystem = compute_ecosystem_str(numeric_panel)
    return {
        "row_count": int(len(panel)),
        "date_count": int(panel["date_utc"].nunique()),
        "rollup_count": int(panel["rollup_id"].nunique()),
        "vendor_key_count": int(len(set(zip(vendor_panel["date_utc"], vendor_panel["rollup_id"])))),
        "ecosystem_str_preview": ecosystem.head(5).to_dict(orient="records"),
    }


def summarize_l1_decomposition(
    decomposition: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
) -> dict[str, object]:
    numeric = numeric_frame(
        decomposition,
        columns=(
            "l1_base_fee_burn_eth",
            "l1_blob_fee_burn_eth",
            "l1_priority_fee_eth",
            "l1_total_rent_eth",
        ),
    )
    return {
        "row_count": int(len(decomposition)),
        "panel_date_count": int(authoritative_panel["date_utc"].nunique()),
        "total_l1_rent_eth_sum": float(numeric["l1_total_rent_eth"].sum()),
        "date_range": {
            "min": decomposition["date_utc"].min() if not decomposition.empty else None,
            "max": decomposition["date_utc"].max() if not decomposition.empty else None,
        },
    }


def summarize_cross_source(vendor_panel: pd.DataFrame, authoritative_panel: pd.DataFrame) -> dict[str, object]:
    merged = vendor_panel.merge(
        authoritative_panel[["date_utc", "rollup_id", "rent_paid_eth"]],
        on=["date_utc", "rollup_id"],
        how="inner",
        suffixes=("_vendor", "_authoritative"),
    )
    numeric = numeric_frame(
        merged,
        columns=("rent_paid_eth_vendor", "rent_paid_eth_authoritative"),
    )
    vendor_total = float(numeric["rent_paid_eth_vendor"].sum())
    authoritative_total = float(numeric["rent_paid_eth_authoritative"].sum())
    pct_difference = percent_difference(vendor_total, authoritative_total)
    return {
        "matched_row_count": int(len(merged)),
        "vendor_total_rent_eth": vendor_total,
        "authoritative_total_rent_eth": authoritative_total,
        "aggregate_pct_difference": pct_difference,
    }


def build_provenance(mode: str, artifacts: dict[str, DataArtifact]) -> dict[str, object]:
    return {
        "mode": mode,
        "artifacts": [
            {
                "dataset": artifact.dataset,
                "path": str(artifact.path.relative_to(REPO_ROOT)),
                "manifest_path": (
                    str(artifact.manifest_path.relative_to(REPO_ROOT))
                    if artifact.manifest_path is not None
                    else None
                ),
                "as_of_utc_date": artifact.as_of_utc_date,
            }
            for artifact in artifacts.values()
        ],
        "command_hints": {
            "sample": "python src/validation/validate_str_pipeline.py --sample",
            "canonical": "python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD",
        },
    }


def aggregate_status(checks: Iterable[CheckResult]) -> str:
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "fail"
    if "blocked" in statuses:
        return "blocked"
    return "pass"


def numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for column in columns:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def vendor_identity_tolerance(*, fees: float, rent_paid: float) -> float:
    return max(1e-9, 0.01 * max(abs(fees), abs(rent_paid), 1e-9))


def percent_difference(left: float, right: float) -> float:
    if right == 0:
        return 0.0 if left == 0 else float("inf")
    return abs(left - right) / abs(right)


def decimal_from_value(value: object) -> Decimal:
    if not isinstance(value, str):
        raise InvalidOperation("expected string-backed decimal value")
    return Decimal(value)


def write_reports(reports: list[ReportPayload]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        json_path = REPORT_DIR / f"{report.report_id}.json"
        md_path = REPORT_DIR / f"{report.report_id}.md"
        json_payload = asdict(report)
        json_payload["checks"] = [asdict(check) for check in report.checks]
        json_path.write_text(
            json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: ReportPayload) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Status: `{report.status}`",
        f"- Mode: `{report.mode}`",
        f"- As of: `{report.as_of_utc_date or 'sample'}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report.summary.items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")

    lines.extend(["", "## Checks", ""])
    for check in report.checks:
        lines.append(f"### {check.name}")
        lines.append("")
        lines.append(f"- Status: `{check.status}`")
        if check.plausible_causes:
            lines.append(f"- Plausible causes: `{json.dumps(check.plausible_causes, sort_keys=True)}`")
        if check.next_step is not None:
            lines.append(f"- Next step: `{check.next_step}`")
        lines.append(f"- Details: `{json.dumps(check.details, sort_keys=True)}`")
        lines.append("")

    lines.extend(["## Provenance", ""])
    for artifact in report.provenance.get("artifacts", []):
        lines.append(f"- {artifact['dataset']}: `{json.dumps(artifact, sort_keys=True)}`")
    command_hints = report.provenance.get("command_hints")
    if command_hints:
        lines.append(f"- command_hints: `{json.dumps(command_hints, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
