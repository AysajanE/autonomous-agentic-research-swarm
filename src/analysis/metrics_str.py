from __future__ import annotations

import pandas as pd

DATE_COLUMN = "date_utc"
ROLLUP_COLUMN = "rollup_id"
L2_FEES_COLUMN = "l2_fees_eth"
RENT_PAID_COLUMN = "rent_paid_eth"
STR_COLUMN = "str"

REQUIRED_COLUMNS = (
    DATE_COLUMN,
    ROLLUP_COLUMN,
    L2_FEES_COLUMN,
    RENT_PAID_COLUMN,
)
METRIC_COLUMNS = (L2_FEES_COLUMN, RENT_PAID_COLUMN)

__all__ = [
    "DATE_COLUMN",
    "ROLLUP_COLUMN",
    "L2_FEES_COLUMN",
    "RENT_PAID_COLUMN",
    "STR_COLUMN",
    "compute_rollup_str",
    "compute_ecosystem_str",
]


def compute_rollup_str(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute rollup-day STR for complete rows in a daily_rollup_panel."""
    complete_rows = _complete_metric_rows(panel)
    result = complete_rows.copy()
    result[STR_COLUMN] = _safe_ratio(result[RENT_PAID_COLUMN], result[L2_FEES_COLUMN])
    return result


def compute_ecosystem_str(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute daily ecosystem STR using the locked row-omission and NaN rules."""
    complete_rows = _complete_metric_rows(panel)
    aggregated = (
        complete_rows.groupby(DATE_COLUMN, as_index=False, sort=True)[
            [RENT_PAID_COLUMN, L2_FEES_COLUMN]
        ]
        .sum()
        .sort_values(DATE_COLUMN, kind="stable")
        .reset_index(drop=True)
    )
    aggregated[STR_COLUMN] = _safe_ratio(
        aggregated[RENT_PAID_COLUMN],
        aggregated[L2_FEES_COLUMN],
    )
    return aggregated


def _complete_metric_rows(panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel)
    return panel.loc[panel[list(METRIC_COLUMNS)].notna().all(axis=1)].copy()


def _require_columns(panel: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in panel.columns]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Panel is missing required STR columns: {missing}")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator
    return ratio.where(denominator != 0)
