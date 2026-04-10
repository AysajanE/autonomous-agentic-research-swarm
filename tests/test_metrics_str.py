from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PANEL_PATH = ROOT / "data" / "samples" / "panels" / "daily_rollup_panel_sample.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.metrics_str import (  # noqa: E402
    DATE_COLUMN,
    L2_FEES_COLUMN,
    RENT_PAID_COLUMN,
    ROLLUP_COLUMN,
    STR_COLUMN,
    compute_ecosystem_str,
    compute_rollup_str,
)

class MetricsStrTest(unittest.TestCase):
    def test_compute_rollup_str_matches_sample_panel_values(self) -> None:
        panel = pd.read_csv(SAMPLE_PANEL_PATH)

        result = compute_rollup_str(panel)

        self.assertEqual(len(result), 9)
        self.assertTrue(result[STR_COLUMN].notna().all())

        arbitrum_day_one = result.loc[
            (result[DATE_COLUMN] == "2024-03-13")
            & (result[ROLLUP_COLUMN] == "arbitrum"),
            STR_COLUMN,
        ].iloc[0]
        optimism_day_three = result.loc[
            (result[DATE_COLUMN] == "2024-03-15")
            & (result[ROLLUP_COLUMN] == "optimism"),
            STR_COLUMN,
        ].iloc[0]

        self.assertAlmostEqual(arbitrum_day_one, 0.878349487034478)
        self.assertAlmostEqual(optimism_day_three, 0.016435122014817855)

    def test_compute_ecosystem_str_matches_sample_panel_aggregates(self) -> None:
        panel = pd.read_csv(SAMPLE_PANEL_PATH)

        result = compute_ecosystem_str(panel)

        self.assertEqual(
            result[DATE_COLUMN].tolist(),
            ["2024-03-13", "2024-03-14", "2024-03-15"],
        )
        self.assertSequenceAlmostEqual(
            result[RENT_PAID_COLUMN].tolist(),
            [299.68625594797317, 85.83637010071874, 5.29198529136078],
        )
        self.assertSequenceAlmostEqual(
            result[L2_FEES_COLUMN].tolist(),
            [363.0395018258075, 193.0488796670926, 126.24674243866971],
        )
        self.assertSequenceAlmostEqual(
            result[STR_COLUMN].tolist(),
            [0.825492141876527, 0.44463542212076634, 0.041917796761620295],
        )

    def test_missing_metric_rows_are_omitted_from_rollup_and_ecosystem_outputs(
        self,
    ) -> None:
        panel = pd.DataFrame(
            [
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "arbitrum",
                    L2_FEES_COLUMN: 10.0,
                    RENT_PAID_COLUMN: 4.0,
                },
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "base",
                    L2_FEES_COLUMN: 6.0,
                    RENT_PAID_COLUMN: None,
                },
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "optimism",
                    L2_FEES_COLUMN: None,
                    RENT_PAID_COLUMN: 3.0,
                },
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "zksync",
                    L2_FEES_COLUMN: 0.0,
                    RENT_PAID_COLUMN: 0.0,
                },
            ]
        )

        rollup_result = compute_rollup_str(panel)
        ecosystem_result = compute_ecosystem_str(panel)

        self.assertEqual(rollup_result[ROLLUP_COLUMN].tolist(), ["arbitrum", "zksync"])
        self.assertAlmostEqual(rollup_result.iloc[0][STR_COLUMN], 0.4)
        self.assertTrue(pd.isna(rollup_result.iloc[1][STR_COLUMN]))
        self.assertEqual(ecosystem_result.loc[0, DATE_COLUMN], "2024-03-13")
        self.assertAlmostEqual(ecosystem_result.loc[0, RENT_PAID_COLUMN], 4.0)
        self.assertAlmostEqual(ecosystem_result.loc[0, L2_FEES_COLUMN], 10.0)
        self.assertAlmostEqual(ecosystem_result.loc[0, STR_COLUMN], 0.4)

    def test_zero_denominator_days_return_nan(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "arbitrum",
                    L2_FEES_COLUMN: 0.0,
                    RENT_PAID_COLUMN: 0.0,
                },
                {
                    DATE_COLUMN: "2024-03-13",
                    ROLLUP_COLUMN: "base",
                    L2_FEES_COLUMN: 0.0,
                    RENT_PAID_COLUMN: 0.0,
                },
                {
                    DATE_COLUMN: "2024-03-14",
                    ROLLUP_COLUMN: "arbitrum",
                    L2_FEES_COLUMN: 5.0,
                    RENT_PAID_COLUMN: 1.0,
                },
            ]
        )

        rollup_result = compute_rollup_str(panel)
        ecosystem_result = compute_ecosystem_str(panel)

        self.assertTrue(pd.isna(rollup_result.loc[0, STR_COLUMN]))
        self.assertTrue(pd.isna(rollup_result.loc[1, STR_COLUMN]))
        self.assertEqual(ecosystem_result.loc[0, DATE_COLUMN], "2024-03-13")
        self.assertTrue(pd.isna(ecosystem_result.loc[0, STR_COLUMN]))
        self.assertAlmostEqual(ecosystem_result.loc[1, STR_COLUMN], 0.2)

    def assertSequenceAlmostEqual(
        self,
        actual: list[float],
        expected: list[float],
        places: int = 7,
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)
