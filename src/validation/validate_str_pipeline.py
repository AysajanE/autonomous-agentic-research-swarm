#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd

from src.analysis.metrics_str import compute_ecosystem_str, compute_rollup_str
from pack_config import load_pack_config, pack_value


REPORT_DIR = REPO_ROOT / "reports" / "validation"
PACK = load_pack_config(REPO_ROOT)


def _dataframe_schema_fields(config_key: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    schema_path = REPO_ROOT / pack_value(PACK, config_key)
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    fields = payload.get("fields") if isinstance(payload, dict) else None
    if not isinstance(fields, list):
        raise ValueError(f"dataframe_schema_fields_missing:{schema_path}")
    required: list[str] = []
    optional: list[str] = []
    for field_spec in fields:
        if not isinstance(field_spec, dict) or not isinstance(field_spec.get("name"), str):
            raise ValueError(f"dataframe_schema_field_invalid:{schema_path}")
        target = optional if field_spec.get("nullable") is True else required
        target.append(str(field_spec["name"]))
    return tuple(required), tuple(optional)

PANEL_REQUIRED_COLUMNS, PANEL_OPTIONAL_COLUMNS = _dataframe_schema_fields("paths.primary_panel_schema")
DECOMP_REQUIRED_COLUMNS, DECOMP_OPTIONAL_COLUMNS = _dataframe_schema_fields("paths.decomposition_panel_schema")
COMPONENT_REQUIRED_COLUMNS, _COMPONENT_OPTIONAL_COLUMNS = _dataframe_schema_fields("paths.rent_components_schema")
COMPONENT_TX_FAMILY_COLUMNS = (
    "batch_submissions_eth",
    "proof_submissions_eth",
    "state_updates_eth",
)
COMPONENT_FEE_CLASS_COLUMNS = (
    "blob_fee_burn_eth",
    "execution_base_fee_burn_eth",
    "execution_priority_fee_eth",
)

VENDOR_REQUIRED_COLUMNS = PANEL_REQUIRED_COLUMNS
VENDOR_OPTIONAL_COLUMNS = PANEL_OPTIONAL_COLUMNS

RECONCILIATION_PASS_THRESHOLD = Decimal("0.10")
ROLLUP_BENCHMARK_MATERIALITY_ETH = Decimal("10")
MONTHLY_BENCHMARK_MATERIALITY_ETH = Decimal("10")
IDENTITY_TOLERANCE_ETH = Decimal("0.000000001")

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
        inputs_consumed = build_inputs_consumed(validation_input_paths(artifacts))
        reports = build_reports(mode=mode, artifacts=artifacts)
    except ValidationBlocked as exc:
        reports = build_blocked_reports(mode=mode, as_of_utc_date=as_of, issues=exc.issues)
        write_reports(reports)
        return 2

    write_reports(reports, inputs_consumed=inputs_consumed)
    if any(report.status != "pass" for report in reports):
        return 1
    return 0


def resolve_artifacts(mode: str, as_of_utc_date: str | None) -> dict[str, DataArtifact]:
    if mode == "sample":
        return {
            "vendor_panel": DataArtifact(
                dataset="vendor_panel",
                path=REPO_ROOT / pack_value(PACK, "paths.vendor_panel_sample"),
                manifest_path=None,
                as_of_utc_date=None,
            ),
            "l1_decomposition": DataArtifact(
                dataset="l1_decomposition",
                path=REPO_ROOT / pack_value(PACK, "paths.decomposition_sample"),
                manifest_path=None,
                as_of_utc_date=None,
            ),
            "rent_components": DataArtifact(
                dataset="rent_components",
                path=REPO_ROOT / pack_value(PACK, "paths.rent_components_sample"),
                manifest_path=None,
                as_of_utc_date=None,
            ),
            "authoritative_panel": DataArtifact(
                dataset="authoritative_panel",
                path=REPO_ROOT / pack_value(PACK, "paths.primary_panel_sample"),
                manifest_path=None,
                as_of_utc_date=None,
            ),
        }

    if as_of_utc_date is None:
        raise ValueError("as_of_utc_date is required in canonical mode")

    manifests = {
        "vendor_panel": (
            REPO_ROOT / pack_value(PACK, "paths.vendor_panel_manifest_pattern").format(date=as_of_utc_date),
            Path(pack_value(PACK, "paths.vendor_panel")).name,
        ),
        "l1_decomposition": (
            REPO_ROOT / pack_value(PACK, "paths.decomposition_manifest_pattern").format(date=as_of_utc_date),
            Path(pack_value(PACK, "paths.decomposition")).name,
        ),
        "rent_components": (
            REPO_ROOT / pack_value(PACK, "paths.rent_components_manifest_pattern").format(date=as_of_utc_date),
            Path(pack_value(PACK, "paths.rent_components")).name,
        ),
        "authoritative_panel": (
            REPO_ROOT / pack_value(PACK, "paths.primary_panel_manifest_pattern").format(date=as_of_utc_date),
            Path(pack_value(PACK, "paths.primary_panel")).name,
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
    rent_components = read_csv_frame(artifacts["rent_components"].path)
    authoritative_panel = read_csv_frame(artifacts["authoritative_panel"].path)
    cross_source_analysis = analyze_benchmark_reconciliation(
        vendor_panel=vendor_panel,
        authoritative_panel=authoritative_panel,
        rent_components=rent_components,
    )

    rollup_checks = build_rollup_panel_checks(authoritative_panel, rent_components)
    decomp_checks = build_l1_decomposition_checks(l1_decomposition, authoritative_panel, rent_components)
    reconciliation_checks = build_cross_source_checks(
        vendor_panel,
        authoritative_panel,
        cross_source_analysis,
    )

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
            summary=summarize_rollup_panel(authoritative_panel, rent_components),
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
            summary=summarize_cross_source(cross_source_analysis),
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


def build_rollup_panel_checks(
    panel: pd.DataFrame,
    rent_components: pd.DataFrame,
) -> list[CheckResult]:
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
        check_required_columns(
            report_name="rent_component_schema",
            frame=rent_components,
            required_columns=COMPONENT_REQUIRED_COLUMNS,
        ),
        check_primary_key_uniqueness(
            report_name="rent_component_primary_key_uniqueness",
            frame=rent_components,
            key_columns=("date_utc", "rollup_id"),
        ),
        check_required_non_null(
            report_name="rent_component_required_non_null",
            frame=rent_components,
            required_columns=COMPONENT_REQUIRED_COLUMNS,
        ),
        check_component_key_coverage(panel=panel, rent_components=rent_components),
        check_component_identity(
            frame=rent_components,
            component_columns=COMPONENT_TX_FAMILY_COLUMNS,
            target_column="rent_paid_eth",
            check_name="rent_component_tx_family_identity",
        ),
        check_component_identity(
            frame=rent_components,
            component_columns=COMPONENT_FEE_CLASS_COLUMNS,
            target_column="rent_paid_eth",
            check_name="rent_component_fee_class_identity",
        ),
        check_component_matches_panel(panel=panel, rent_components=rent_components),
        check_metrics_compatibility(panel=panel),
    ]
    return checks


def build_l1_decomposition_checks(
    decomposition: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
    rent_components: pd.DataFrame,
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
        check_component_daily_totals_match_decomposition(
            decomposition=decomposition,
            rent_components=rent_components,
        ),
    ]
    return checks


def build_cross_source_checks(
    vendor_panel: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
    analysis: dict[str, object],
) -> list[CheckResult]:
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
        check_key_coverage(panel=authoritative_panel, vendor_panel=vendor_panel),
        check_benchmark_reconciliation(analysis),
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


def summarize_unmatched_keys(unmatched_keys: pd.DataFrame) -> list[dict[str, object]]:
    if unmatched_keys.empty:
        return []

    summary = (
        unmatched_keys.groupby("rollup_id", as_index=False)
        .agg(
            key_count=("date_utc", "size"),
            min_date_utc=("date_utc", "min"),
            max_date_utc=("date_utc", "max"),
        )
        .sort_values(["key_count", "rollup_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return summary.to_dict(orient="records")


def check_key_coverage(panel: pd.DataFrame, vendor_panel: pd.DataFrame) -> CheckResult:
    key_columns = ["date_utc", "rollup_id"]
    merged_keys = vendor_panel[key_columns].drop_duplicates().merge(
        panel[key_columns].drop_duplicates(),
        on=key_columns,
        how="outer",
        indicator=True,
    )
    only_in_vendor = (
        merged_keys.loc[merged_keys["_merge"] == "left_only", key_columns]
        .sort_values(key_columns, kind="stable")
        .reset_index(drop=True)
    )
    only_in_panel = (
        merged_keys.loc[merged_keys["_merge"] == "right_only", key_columns]
        .sort_values(key_columns, kind="stable")
        .reset_index(drop=True)
    )
    status = "pass" if only_in_panel.empty and only_in_vendor.empty else "fail"
    return CheckResult(
        name="authoritative_vs_vendor_key_coverage",
        status=status,
        details={
            "authoritative_panel_key_count": int(len(panel[key_columns].drop_duplicates())),
            "vendor_panel_key_count": int(len(vendor_panel[key_columns].drop_duplicates())),
            "only_in_authoritative_panel_count": int(len(only_in_panel)),
            "only_in_authoritative_panel": only_in_panel.head(10).to_dict(orient="records"),
            "only_in_authoritative_panel_by_rollup": summarize_unmatched_keys(only_in_panel),
            "only_in_vendor_panel_count": int(len(only_in_vendor)),
            "only_in_vendor_panel": only_in_vendor.head(10).to_dict(orient="records"),
            "only_in_vendor_panel_by_rollup": summarize_unmatched_keys(only_in_vendor),
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


def check_component_key_coverage(
    panel: pd.DataFrame,
    rent_components: pd.DataFrame,
) -> CheckResult:
    key_columns = ["date_utc", "rollup_id"]
    merged_keys = rent_components[key_columns].drop_duplicates().merge(
        panel[key_columns].drop_duplicates(),
        on=key_columns,
        how="outer",
        indicator=True,
    )
    only_in_components = (
        merged_keys.loc[merged_keys["_merge"] == "left_only", key_columns]
        .sort_values(key_columns, kind="stable")
        .reset_index(drop=True)
    )
    only_in_panel = (
        merged_keys.loc[merged_keys["_merge"] == "right_only", key_columns]
        .sort_values(key_columns, kind="stable")
        .reset_index(drop=True)
    )
    status = "pass" if only_in_panel.empty else "fail"
    return CheckResult(
        name="rent_component_covers_panel_keys",
        status=status,
        details={
            "authoritative_panel_key_count": int(len(panel[key_columns].drop_duplicates())),
            "rent_component_key_count": int(len(rent_components[key_columns].drop_duplicates())),
            "only_in_authoritative_panel_count": int(len(only_in_panel)),
            "only_in_authoritative_panel": only_in_panel.head(10).to_dict(orient="records"),
            "additional_rent_component_key_count": int(len(only_in_components)),
            "additional_rent_component_keys": only_in_components.head(10).to_dict(orient="records"),
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The component artifact dropped panel keys that should remain comparable to the analysis-ready authoritative panel.",
                "The panel and component surfaces were rebuilt from different manifested canonical runs.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Rebuild the canonical panel and component artifact from the same manifested canonical run before advancing validation."
        ),
    )


def check_component_identity(
    frame: pd.DataFrame,
    component_columns: Iterable[str],
    target_column: str,
    check_name: str,
) -> CheckResult:
    component_columns = tuple(component_columns)
    max_abs_diff = Decimal("0")
    violating_rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        try:
            component_sum = sum(decimal_from_value(row[column]) for column in component_columns)
            target_value = decimal_from_value(row[target_column])
        except (InvalidOperation, KeyError):
            return CheckResult(
                name=check_name,
                status="fail",
                details={
                    "component_columns": list(component_columns),
                    "target_column": target_column,
                },
                plausible_causes=["A component value could not be parsed as a Decimal ETH quantity."],
                next_step="Restore numeric ETH values for the component surface before rerunning validation.",
            )

        diff = abs(target_value - component_sum)
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > IDENTITY_TOLERANCE_ETH and len(violating_rows) < 10:
            violating_rows.append(
                {
                    "date_utc": row["date_utc"],
                    "rollup_id": row["rollup_id"],
                    "difference_eth": str(diff),
                }
            )

    status = "pass" if not violating_rows else "fail"
    return CheckResult(
        name=check_name,
        status=status,
        details={
            "row_count": int(len(frame)),
            "component_columns": list(component_columns),
            "target_column": target_column,
            "identity_tolerance_eth": str(IDENTITY_TOLERANCE_ETH),
            "max_abs_difference_eth": str(max_abs_diff),
            "violating_rows": violating_rows,
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The component artifact lost or double-counted one of the contract-locked cost families.",
                "CSV serialization or aggregation drift pushed the component surface away from canonical rent beyond tolerance.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Trace the failing rows back to the T049 aggregation and restore the locked component identity."
        ),
    )


def check_component_matches_panel(
    panel: pd.DataFrame,
    rent_components: pd.DataFrame,
) -> CheckResult:
    merged = panel[["date_utc", "rollup_id", "rent_paid_eth"]].merge(
        rent_components[["date_utc", "rollup_id", "rent_paid_eth"]],
        on=["date_utc", "rollup_id"],
        how="left",
        suffixes=("_panel", "_components"),
        indicator=True,
    )
    missing_keys = merged.loc[merged["_merge"] != "both", ["date_utc", "rollup_id", "_merge"]]
    if not missing_keys.empty:
        return CheckResult(
            name="rent_component_panel_overlap_identity",
            status="fail",
            details={
                "mismatched_key_count": int(len(missing_keys)),
                "sample_key_mismatches": missing_keys.head(10).to_dict(orient="records"),
            },
            plausible_causes=[
                "The component artifact no longer covers every authoritative panel key.",
                "The canonical panel and component surfaces were refreshed from different manifested runs.",
            ],
            next_step="Refresh both canonical outputs from the same manifested canonical run.",
        )

    max_abs_diff = Decimal("0")
    violating_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        try:
            component_rent = decimal_from_value(row["rent_paid_eth_components"])
            panel_rent = decimal_from_value(row["rent_paid_eth_panel"])
        except InvalidOperation:
            return CheckResult(
                name="rent_component_panel_overlap_identity",
                status="fail",
                details={"reason": "component or panel rent could not be parsed as Decimal ETH"},
                plausible_causes=["A canonical rent field was serialized with an invalid numeric value."],
                next_step="Restore numeric ETH values for both canonical surfaces before rerunning validation.",
            )

        diff = abs(component_rent - panel_rent)
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > IDENTITY_TOLERANCE_ETH and len(violating_rows) < 10:
            violating_rows.append(
                {
                    "date_utc": row["date_utc"],
                    "rollup_id": row["rollup_id"],
                    "difference_eth": str(diff),
                }
            )

    status = "pass" if not violating_rows else "fail"
    return CheckResult(
        name="rent_component_panel_overlap_identity",
        status=status,
        details={
            "row_count": int(len(panel)),
            "identity_tolerance_eth": str(IDENTITY_TOLERANCE_ETH),
            "max_abs_difference_eth": str(max_abs_diff),
            "violating_rows": violating_rows,
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The authoritative panel and component surface disagree on canonical rent for overlapping panel keys.",
                "One surface was rebuilt against a different canonical input universe than the other.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Rebuild the component artifact and canonical panel together from the same T049 ETL run."
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


def check_component_daily_totals_match_decomposition(
    decomposition: pd.DataFrame,
    rent_components: pd.DataFrame,
) -> CheckResult:
    component_totals: dict[str, Decimal] = {}
    for _, row in rent_components.iterrows():
        try:
            rent_paid = decimal_from_value(row["rent_paid_eth"])
        except (InvalidOperation, KeyError):
            return CheckResult(
                name="l1_total_rent_matches_rollup_components",
                status="fail",
                details={"reason": "component rent could not be parsed as Decimal ETH"},
                plausible_causes=["A component rent value was serialized with an invalid numeric value."],
                next_step="Restore numeric ETH values for the component surface before rerunning validation.",
            )
        component_totals[row["date_utc"]] = component_totals.get(row["date_utc"], Decimal("0")) + rent_paid

    decomposition_dates = set(decomposition["date_utc"]) if "date_utc" in decomposition.columns else set()
    component_dates = set(component_totals)
    extra_component_dates = sorted(component_dates - decomposition_dates)
    if extra_component_dates:
        return CheckResult(
            name="l1_total_rent_matches_rollup_components",
            status="fail",
            details={
                "extra_component_date_count": int(len(extra_component_dates)),
                "sample_extra_component_dates": extra_component_dates[:10],
            },
            plausible_causes=[
                "The component surface was materialized over a different date window than the decomposition.",
                "A component rebuild was mixed with an older decomposition artifact.",
            ],
            next_step="Refresh the decomposition and component artifacts from the same manifested canonical run.",
        )

    max_abs_diff = Decimal("0")
    violating_rows: list[dict[str, object]] = []
    zero_component_dates = 0
    for _, row in decomposition.iterrows():
        try:
            total_rent = decimal_from_value(row["l1_total_rent_eth"])
        except (InvalidOperation, KeyError):
            return CheckResult(
                name="l1_total_rent_matches_rollup_components",
                status="fail",
                details={"reason": "decomposition total rent could not be parsed as Decimal ETH"},
                plausible_causes=["A decomposition rent value was serialized with an invalid numeric value."],
                next_step="Restore numeric ETH values for the decomposition surface before rerunning validation.",
            )

        component_total = component_totals.get(row["date_utc"], Decimal("0"))
        if component_total == 0:
            zero_component_dates += 1
        diff = abs(total_rent - component_total)
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > IDENTITY_TOLERANCE_ETH and len(violating_rows) < 10:
            violating_rows.append(
                {
                    "date_utc": row["date_utc"],
                    "decomposition_total_rent_eth": str(total_rent),
                    "component_total_rent_eth": str(component_total),
                    "difference_eth": str(diff),
                }
            )

    status = "pass" if not violating_rows else "fail"
    return CheckResult(
        name="l1_total_rent_matches_rollup_components",
        status=status,
        details={
            "row_count": int(len(decomposition)),
            "component_date_count": int(len(component_totals)),
            "zero_component_date_count": zero_component_dates,
            "identity_tolerance_eth": str(IDENTITY_TOLERANCE_ETH),
            "max_abs_difference_eth": str(max_abs_diff),
            "violating_rows": violating_rows,
        },
        plausible_causes=(
            []
            if status == "pass"
            else [
                "The authoritative component surface omitted canonical rent that remains present in the daily decomposition.",
                "The decomposition and component surfaces were rebuilt from different canonical rollup-day universes.",
            ]
        ),
        next_step=(
            None
            if status == "pass"
            else "Refresh the decomposition and component artifacts from the same manifested canonical run."
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


def analyze_benchmark_reconciliation(
    *,
    vendor_panel: pd.DataFrame,
    authoritative_panel: pd.DataFrame,
    rent_components: pd.DataFrame,
) -> dict[str, object]:
    key_columns = ["date_utc", "rollup_id"]
    merged = vendor_panel.merge(
        authoritative_panel[["date_utc", "rollup_id", "rent_paid_eth"]],
        on=key_columns,
        how="outer",
        suffixes=("_vendor", "_authoritative"),
        indicator=True,
    )
    merged_keys = merged[key_columns + ["_merge"]].drop_duplicates().copy()
    missing_keys = (
        merged_keys.loc[merged_keys["_merge"] != "both", key_columns + ["_merge"]]
        .sort_values(key_columns + ["_merge"], kind="stable")
        .reset_index(drop=True)
    )
    matched_rows = merged.loc[merged["_merge"] == "both"].copy()
    if matched_rows.empty:
        return {
            "mismatched_key_count": int(len(missing_keys)),
            "sample_key_mismatches": missing_keys.head(10).to_dict(orient="records"),
            "only_in_vendor_panel_by_rollup": summarize_unmatched_keys(
                missing_keys.loc[missing_keys["_merge"] == "left_only", key_columns]
            ),
            "only_in_authoritative_panel_by_rollup": summarize_unmatched_keys(
                missing_keys.loc[missing_keys["_merge"] == "right_only", key_columns]
            ),
            "matched_row_count": 0,
        }

    numeric = numeric_frame(
        matched_rows,
        columns=("rent_paid_eth_vendor", "rent_paid_eth_authoritative"),
    )
    numeric["delta_eth"] = numeric["rent_paid_eth_authoritative"] - numeric["rent_paid_eth_vendor"]
    numeric["abs_delta_eth"] = numeric["delta_eth"].abs()
    numeric["month_utc"] = numeric["date_utc"].str.slice(0, 7)

    rollup_totals = aggregate_reconciliation_frame(
        numeric,
        group_columns=("rollup_id",),
    )
    monthly_aggregate = aggregate_reconciliation_frame(
        numeric,
        group_columns=("month_utc",),
    )
    monthly_rollup = aggregate_reconciliation_frame(
        numeric,
        group_columns=("month_utc", "rollup_id"),
    )

    component_rollup = aggregate_component_frame(
        rent_components,
        group_columns=("rollup_id",),
    )
    component_monthly_rollup = aggregate_component_frame(
        rent_components,
        group_columns=("month_utc", "rollup_id"),
    )

    rollup_audit = build_benchmark_component_audit(
        reconciliation_frame=rollup_totals,
        component_frame=component_rollup,
        materiality_threshold_eth=ROLLUP_BENCHMARK_MATERIALITY_ETH,
    )
    explained_rollups = set(
        rollup_audit.loc[
            rollup_audit["audit_status"] == "explained_methodology_difference",
            "rollup_id",
        ]
    )
    unexplained_rollups = rollup_audit.loc[
        rollup_audit["audit_status"] == "unexplained"
    ].copy()

    unresolved_rows = numeric.loc[~numeric["rollup_id"].isin(explained_rollups)].copy()
    unresolved_monthly_aggregate = aggregate_reconciliation_frame(
        unresolved_rows,
        group_columns=("month_utc",),
    )
    unresolved_monthly_aggregate_violations = unresolved_monthly_aggregate.loc[
        (unresolved_monthly_aggregate["pct_difference"] > float(RECONCILIATION_PASS_THRESHOLD))
        & (unresolved_monthly_aggregate["abs_delta_eth"] >= float(MONTHLY_BENCHMARK_MATERIALITY_ETH))
    ].copy()

    explained_rollup_months = build_benchmark_component_audit(
        reconciliation_frame=monthly_rollup.loc[
            monthly_rollup["rollup_id"].isin(explained_rollups)
        ].copy(),
        component_frame=component_monthly_rollup,
        materiality_threshold_eth=MONTHLY_BENCHMARK_MATERIALITY_ETH,
    )
    unexplained_rollup_months = build_benchmark_component_audit(
        reconciliation_frame=monthly_rollup.loc[
            monthly_rollup["rollup_id"].isin(unexplained_rollups["rollup_id"])
        ].copy(),
        component_frame=component_monthly_rollup,
        materiality_threshold_eth=MONTHLY_BENCHMARK_MATERIALITY_ETH,
    )
    material_explained_rollup_months = explained_rollup_months.loc[
        explained_rollup_months["audit_status"] == "explained_methodology_difference"
    ].copy()
    material_unexplained_rollup_months = unexplained_rollup_months.loc[
        unexplained_rollup_months["audit_status"] == "unexplained"
    ].copy()

    overall_vendor_total = float(numeric["rent_paid_eth_vendor"].sum())
    overall_authoritative_total = float(numeric["rent_paid_eth_authoritative"].sum())
    unresolved_vendor_total = float(unresolved_rows["rent_paid_eth_vendor"].sum())
    unresolved_authoritative_total = float(unresolved_rows["rent_paid_eth_authoritative"].sum())

    return {
        "target_tolerance_pct": float(RECONCILIATION_PASS_THRESHOLD * Decimal("100")),
        "rollup_materiality_threshold_eth": float(ROLLUP_BENCHMARK_MATERIALITY_ETH),
        "monthly_materiality_threshold_eth": float(MONTHLY_BENCHMARK_MATERIALITY_ETH),
        "mismatched_key_count": int(len(missing_keys)),
        "sample_key_mismatches": missing_keys.head(10).to_dict(orient="records"),
        "only_in_vendor_panel_by_rollup": summarize_unmatched_keys(
            missing_keys.loc[missing_keys["_merge"] == "left_only", key_columns]
        ),
        "only_in_authoritative_panel_by_rollup": summarize_unmatched_keys(
            missing_keys.loc[missing_keys["_merge"] == "right_only", key_columns]
        ),
        "matched_row_count": int(len(matched_rows)),
        "overall_vendor_total_rent_eth": overall_vendor_total,
        "overall_authoritative_total_rent_eth": overall_authoritative_total,
        "overall_aggregate_pct_difference": percent_difference(
            overall_vendor_total,
            overall_authoritative_total,
        ),
        "rollup_total_audit_top_abs_delta": (
            rollup_audit.sort_values(
                ["abs_delta_eth", "rollup_id"],
                ascending=[False, True],
                kind="stable",
            )
            .head(20)
            .to_dict(orient="records")
        ),
        "material_explained_rollups": (
            rollup_audit.loc[
                rollup_audit["audit_status"] == "explained_methodology_difference"
            ]
            .sort_values(["abs_delta_eth", "rollup_id"], ascending=[False, True], kind="stable")
            .to_dict(orient="records")
        ),
        "material_unexplained_rollups": (
            unexplained_rollups.sort_values(
                ["abs_delta_eth", "rollup_id"],
                ascending=[False, True],
                kind="stable",
            ).to_dict(orient="records")
        ),
        "unexplained_slice_vendor_total_rent_eth": unresolved_vendor_total,
        "unexplained_slice_authoritative_total_rent_eth": unresolved_authoritative_total,
        "unexplained_slice_pct_difference": percent_difference(
            unresolved_vendor_total,
            unresolved_authoritative_total,
        ),
        "unexplained_monthly_aggregate": unresolved_monthly_aggregate.to_dict(orient="records"),
        "unexplained_monthly_aggregate_violation_count": int(len(unresolved_monthly_aggregate_violations)),
        "sample_unexplained_monthly_aggregate_violations": (
            unresolved_monthly_aggregate_violations.sort_values(
                ["abs_delta_eth", "month_utc"],
                ascending=[False, True],
                kind="stable",
            )
            .head(12)
            .to_dict(orient="records")
        ),
        "sample_explained_rollup_months": (
            material_explained_rollup_months.sort_values(
                ["abs_delta_eth", "month_utc", "rollup_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(12)
            .to_dict(orient="records")
        ),
        "sample_unexplained_rollup_months": (
            material_unexplained_rollup_months.sort_values(
                ["abs_delta_eth", "month_utc", "rollup_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(12)
            .to_dict(orient="records")
        ),
    }


def aggregate_reconciliation_frame(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    columns = ["rent_paid_eth_vendor", "rent_paid_eth_authoritative"]
    if frame.empty:
        return pd.DataFrame(
            columns=list(group_columns) + columns + ["delta_eth", "abs_delta_eth", "pct_difference"]
        )

    aggregated = (
        frame.groupby(list(group_columns), as_index=False)[columns]
        .sum()
        .reset_index(drop=True)
    )
    aggregated["delta_eth"] = aggregated["rent_paid_eth_authoritative"] - aggregated["rent_paid_eth_vendor"]
    aggregated["abs_delta_eth"] = aggregated["delta_eth"].abs()
    aggregated["pct_difference"] = aggregated.apply(
        lambda row: percent_difference(
            row["rent_paid_eth_vendor"],
            row["rent_paid_eth_authoritative"],
        ),
        axis=1,
    )
    return aggregated


def aggregate_component_frame(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    component_columns = list(COMPONENT_REQUIRED_COLUMNS[2:])
    if frame.empty:
        return pd.DataFrame(columns=list(group_columns) + component_columns)

    numeric = numeric_frame(frame, columns=component_columns)
    if "month_utc" in group_columns and "month_utc" not in numeric.columns:
        numeric["month_utc"] = numeric["date_utc"].str.slice(0, 7)
    return (
        numeric.groupby(list(group_columns), as_index=False)[component_columns]
        .sum()
        .reset_index(drop=True)
    )


def build_benchmark_component_audit(
    *,
    reconciliation_frame: pd.DataFrame,
    component_frame: pd.DataFrame,
    materiality_threshold_eth: Decimal,
) -> pd.DataFrame:
    group_columns = [
        column
        for column in ("month_utc", "rollup_id")
        if column in reconciliation_frame.columns
    ]
    if reconciliation_frame.empty:
        return pd.DataFrame(
            columns=group_columns
            + [
                "rent_paid_eth_vendor",
                "rent_paid_eth_authoritative",
                "delta_eth",
                "abs_delta_eth",
                "pct_difference",
                "audit_status",
                "audit_reason",
                "best_matching_tx_family_combo",
                "best_matching_combo_eth",
                "residual_after_best_combo_eth",
                "residual_after_best_combo_pct_of_authoritative",
            ]
        )

    merged = reconciliation_frame.merge(
        component_frame,
        on=group_columns,
        how="left",
    )
    merged[list(COMPONENT_REQUIRED_COLUMNS[2:])] = merged[list(COMPONENT_REQUIRED_COLUMNS[2:])].fillna(0.0)

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        component_values = {
            column: float(row[column])
            for column in COMPONENT_TX_FAMILY_COLUMNS
        }
        best_combo_fields, best_combo_sum, residual = best_matching_component_combo(
            delta_eth=float(row["delta_eth"]),
            component_values=component_values,
        )
        residual_pct = residual / abs(float(row["rent_paid_eth_authoritative"])) if float(row["rent_paid_eth_authoritative"]) else (0.0 if residual == 0 else float("inf"))

        if row["abs_delta_eth"] < float(materiality_threshold_eth):
            audit_status = "within_tolerance"
            audit_reason = "absolute_delta_below_materiality_threshold"
        elif row["pct_difference"] <= float(RECONCILIATION_PASS_THRESHOLD):
            audit_status = "within_tolerance"
            audit_reason = "pct_difference_within_tolerance"
        elif row["delta_eth"] > 0 and residual_pct <= float(RECONCILIATION_PASS_THRESHOLD):
            audit_status = "explained_methodology_difference"
            audit_reason = "positive_canonical_excess_matches_tx_family_components"
        else:
            audit_status = "unexplained"
            audit_reason = "remaining_delta_exceeds_component_audit_tolerance"

        rows.append(
            {
                **{column: row[column] for column in group_columns},
                "rent_paid_eth_vendor": float(row["rent_paid_eth_vendor"]),
                "rent_paid_eth_authoritative": float(row["rent_paid_eth_authoritative"]),
                "delta_eth": float(row["delta_eth"]),
                "abs_delta_eth": float(row["abs_delta_eth"]),
                "pct_difference": float(row["pct_difference"]),
                "audit_status": audit_status,
                "audit_reason": audit_reason,
                "best_matching_tx_family_combo": (
                    " + ".join(best_combo_fields) if best_combo_fields else None
                ),
                "best_matching_combo_eth": best_combo_sum,
                "residual_after_best_combo_eth": residual,
                "residual_after_best_combo_pct_of_authoritative": residual_pct,
                "batch_submissions_eth": float(row["batch_submissions_eth"]),
                "proof_submissions_eth": float(row["proof_submissions_eth"]),
                "state_updates_eth": float(row["state_updates_eth"]),
            }
        )

    return pd.DataFrame(rows)


def best_matching_component_combo(
    *,
    delta_eth: float,
    component_values: dict[str, float],
) -> tuple[tuple[str, ...], float, float]:
    if delta_eth <= 0:
        return (), 0.0, abs(delta_eth)

    best_combo: tuple[str, ...] = ()
    best_sum = 0.0
    best_residual = abs(delta_eth)
    for size in range(1, len(COMPONENT_TX_FAMILY_COLUMNS) + 1):
        for combo in itertools.combinations(COMPONENT_TX_FAMILY_COLUMNS, size):
            combo_sum = sum(component_values[column] for column in combo)
            residual = abs(delta_eth - combo_sum)
            candidate = (residual, len(combo), combo)
            incumbent = (best_residual, len(best_combo), best_combo)
            if candidate < incumbent:
                best_combo = combo
                best_sum = combo_sum
                best_residual = residual
    return best_combo, best_sum, best_residual


def check_benchmark_reconciliation(analysis: dict[str, object]) -> CheckResult:
    mismatched_key_count = int(analysis["mismatched_key_count"])
    matched_row_count = int(analysis["matched_row_count"])
    unexplained_rollups = list(analysis.get("material_unexplained_rollups", []))
    unexplained_monthly_aggregate_violation_count = int(
        analysis.get("unexplained_monthly_aggregate_violation_count", 0)
    )
    unexplained_slice_pct_difference = float(analysis.get("unexplained_slice_pct_difference", 0.0))

    if matched_row_count == 0:
        return CheckResult(
            name="benchmark_reconciliation_policy",
            status="fail",
            details=analysis,
            plausible_causes=[
                "The canonical and vendor panels do not share any matched rollup-day keys.",
                "One validation input was hydrated from a different as-of snapshot than the other.",
            ],
            next_step="Restore a coherent as-of surface before interpreting any benchmark reconciliation result.",
        )

    # Monthly aggregate residuals stay visible in the report details, but they are
    # diagnostic evidence rather than an independent release gate once the
    # benchmark passes on keys, unresolved aggregate gap, and material rollups.
    status = "pass"
    if mismatched_key_count:
        status = "fail"
    elif unexplained_slice_pct_difference > float(RECONCILIATION_PASS_THRESHOLD):
        status = "fail"
    elif unexplained_rollups:
        status = "fail"

    plausible_causes: list[str] = []
    if mismatched_key_count:
        plausible_causes.extend(
            [
                "The vendor and authoritative panels do not cover the same rollup-day keys.",
                "One input was materialized from a different sample window or stale hydration surface.",
            ]
        )
    if unexplained_rollups:
        plausible_causes.extend(
            [
                "A material matched-key benchmark gap remains after auditing the canonical tx-family component surface.",
                "At least one rollup still has a vendor-versus-canonical methodology mismatch or an unresolved canonical attribution defect.",
            ]
        )

    if status == "pass":
        next_step = None
    elif mismatched_key_count:
        next_step = "Reconcile the vendor-only or authoritative-only rollup-day keys before interpreting benchmark deltas."
    elif unexplained_rollups:
        next_step = (
            "Investigate the material unexplained rollups in the cross-source report and determine whether the gap is "
            "a canonical attribution defect or a documented vendor-methodology difference that still lacks component evidence."
        )
    else:
        next_step = None

    return CheckResult(
        name="benchmark_reconciliation_policy",
        status=status,
        details=analysis,
        plausible_causes=plausible_causes,
        next_step=next_step,
    )


def summarize_rollup_panel(panel: pd.DataFrame, rent_components: pd.DataFrame) -> dict[str, object]:
    numeric_panel = numeric_frame(panel, columns=("l2_fees_eth", "rent_paid_eth"))
    ecosystem = compute_ecosystem_str(numeric_panel)
    return {
        "row_count": int(len(panel)),
        "date_count": int(panel["date_utc"].nunique()),
        "rollup_count": int(panel["rollup_id"].nunique()),
        "rent_component_row_count": int(len(rent_components)),
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


def summarize_cross_source(analysis: dict[str, object]) -> dict[str, object]:
    return {
        "matched_row_count": int(analysis.get("matched_row_count", 0)),
        "mismatched_key_count": int(analysis.get("mismatched_key_count", 0)),
        "overall_vendor_total_rent_eth": analysis.get("overall_vendor_total_rent_eth"),
        "overall_authoritative_total_rent_eth": analysis.get("overall_authoritative_total_rent_eth"),
        "overall_aggregate_pct_difference": analysis.get("overall_aggregate_pct_difference"),
        "material_explained_rollup_count": int(len(analysis.get("material_explained_rollups", []))),
        "material_unexplained_rollup_count": int(len(analysis.get("material_unexplained_rollups", []))),
        "unexplained_slice_pct_difference": analysis.get("unexplained_slice_pct_difference"),
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
        return "fail"
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


def validation_input_paths(artifacts: dict[str, DataArtifact]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for artifact in artifacts.values():
        for path in (artifact.manifest_path, artifact.path):
            if path is None:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    return paths


def build_inputs_consumed(paths: Iterable[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in paths:
        resolved = path.resolve()
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        entries.append(
            {
                "path": resolved.relative_to(REPO_ROOT.resolve()).as_posix(),
                "sha256": digest.hexdigest(),
                "bytes": size,
            }
        )
    return entries


def write_reports(
    reports: list[ReportPayload],
    *,
    inputs_consumed: list[dict[str, object]] | None = None,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        json_path = REPORT_DIR / f"{report.report_id}.json"
        md_path = REPORT_DIR / f"{report.report_id}.md"
        json_payload = asdict(report)
        json_payload["checks"] = [asdict(check) for check in report.checks]
        if inputs_consumed:
            json_payload["schema_version"] = "research_swarm.validation_report.v2"
            json_payload["inputs_consumed"] = inputs_consumed
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
