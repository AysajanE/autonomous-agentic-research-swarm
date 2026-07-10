from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "src" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from inference import (  # noqa: E402
    SmallTError,
    bai_perron_breaks,
    block_bootstrap_ci,
    cinelli_hazlett_rv,
    clustered_se,
    driscoll_kraay_se,
    newey_west_se,
    oster_delta,
    two_way_clustered_se,
)
from uncertainty import (  # noqa: E402
    generate_limitations,
    headline_uncertainty_artifact,
    registry_caveats,
)


def _manual_cluster_covariance(
    X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """Direct score-sum oracle, intentionally separate from production helpers."""
    inverse = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for cluster in sorted(set(clusters.tolist()), key=repr):
        indices = [index for index, value in enumerate(clusters.tolist()) if value == cluster]
        score = sum((X[index] * residuals[index] for index in indices), np.zeros(X.shape[1]))
        meat += np.outer(score, score)
    return inverse @ meat @ inverse


class M3cInferenceKnownAnswerTest(unittest.TestCase):
    def test_newey_west_bartlett_known_answer(self) -> None:
        # Intercept-only: B = 10 + 2*(1/2)*(-7) = 3 and bread = 1/4.
        X = np.ones((4, 1))
        residuals = np.array([1.0, -1.0, 2.0, -2.0])
        expected = math.sqrt(3.0) / 4.0
        self.assertAlmostEqual(float(newey_west_se(residuals, X, 1)[0]), expected, places=12)

    def test_driscoll_kraay_single_cross_section_known_answer(self) -> None:
        # With one observation per period, the DK time score is the NW score.
        X = np.ones((4, 1))
        residuals = np.array([1.0, -1.0, 2.0, -2.0])
        result = driscoll_kraay_se(
            residuals,
            X,
            np.array([1, 2, 3, 4]),
            1,
            min_time_periods=4,
        )
        self.assertAlmostEqual(float(result[0]), math.sqrt(3.0) / 4.0, places=12)

    def test_driscoll_kraay_large_t_guard(self) -> None:
        with self.assertRaises(SmallTError):
            driscoll_kraay_se(
                np.array([1.0, -1.0, 1.0, -1.0]),
                np.ones((4, 1)),
                np.array([1, 1, 2, 2]),
                0,
                min_time_periods=3,
            )

    def test_one_way_cluster_known_answer(self) -> None:
        # Cluster scores are +3 and -3: V = (9+9)/(4^2) = 18/16.
        X = np.ones((4, 1))
        residuals = np.array([1.0, 2.0, -1.0, -2.0])
        result = clustered_se(X, residuals, np.array(["a", "a", "b", "b"]))
        self.assertAlmostEqual(float(result[0]), math.sqrt(18.0 / 16.0), places=12)

    def test_two_way_cluster_matches_independent_inclusion_exclusion(self) -> None:
        rng = np.random.default_rng(2)
        X = np.column_stack([np.ones(32), np.tile([0.0, 1.0], 16)])
        cluster_a = np.repeat(np.arange(4), 8)
        cluster_b = np.tile(np.repeat(np.arange(4), 2), 4)
        residuals = (
            rng.normal(size=4)[cluster_a]
            + rng.normal(size=4)[cluster_b]
            + rng.normal(size=32) * 0.1
        )
        residuals -= X @ np.linalg.lstsq(X, residuals, rcond=None)[0]
        intersection = np.empty(len(residuals), dtype=object)
        intersection[:] = list(zip(cluster_a.tolist(), cluster_b.tolist(), strict=True))
        expected_covariance = (
            _manual_cluster_covariance(X, residuals, cluster_a.astype(object))
            + _manual_cluster_covariance(X, residuals, cluster_b.astype(object))
            - _manual_cluster_covariance(X, residuals, intersection)
        )
        expected = np.sqrt(np.diag(expected_covariance))
        np.testing.assert_allclose(
            two_way_clustered_se(X, residuals, cluster_a, cluster_b),
            expected,
            rtol=0,
            atol=1e-12,
        )

    def test_block_bootstrap_full_length_block_is_degenerate_known_answer(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        point, lo, hi = block_bootstrap_ci(
            lambda values: float(np.mean(values)),
            data,
            n_boot=100,
            block_len=4,
            seed=17,
            alpha=0.1,
        )
        self.assertEqual((point, lo, hi), (2.5, 2.5, 2.5))

    def test_bai_perron_piecewise_constant_known_break_and_ssr(self) -> None:
        breaks, ssr = bai_perron_breaks(
            np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0]),
            max_breaks=1,
            min_segment=2,
        )
        self.assertEqual(breaks, [3])
        self.assertAlmostEqual(ssr, 0.0, places=12)

    def test_oster_textbook_proportional_selection_example(self) -> None:
        # ((.4-0)*(.3-.1))/((.5-.4)*(.6-.3)) = 8/3.
        self.assertAlmostEqual(oster_delta(0.4, 0.5, 0.3, 0.1, 0.6), 8.0 / 3.0, places=12)

    def test_cinelli_hazlett_textbook_formula_example(self) -> None:
        # t=2, df=100 -> f=.2 -> (sqrt(.2^4+4*.2^2)-.2^2)/2.
        expected = (math.sqrt(0.2**4 + 4.0 * 0.2**2) - 0.2**2) / 2.0
        self.assertAlmostEqual(cinelli_hazlett_rv(2.0, 1.0, 100), expected, places=12)

    def test_uncertainty_band_uses_registered_gap_bound(self) -> None:
        artifact = headline_uncertainty_artifact(
            claim_id="C1",
            estimate=10.0,
            ci_lower=8.0,
            ci_upper=12.0,
            coverage_gap_fraction=0.25,
            coverage_gap_source="reports/validation/coverage.json",
        )
        band = artifact["measurement_uncertainty"]
        self.assertEqual((band["lower"], band["upper"]), (6.0, 15.0))

    def test_registry_limitations_propagate_arbitrum_and_linea(self) -> None:
        caveats = registry_caveats(ROOT / "registry/rollup_registry_v1.csv")
        ids = {item["rollup_id"] for item in caveats}
        self.assertTrue({"arbitrum", "linea"}.issubset(ids))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "limitations.md"
            generate_limitations(ROOT / "registry/rollup_registry_v1.csv", output)
            text = output.read_text(encoding="utf-8")
        self.assertIn("pre-2023-01-04 sender coverage remains unresolved", text)
        self.assertIn("coverage before 2024-02-13 remains an explicit registry caveat", text)


if __name__ == "__main__":
    unittest.main()
