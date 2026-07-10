from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import calibrate_referee


class M3bGoldenTests(unittest.TestCase):
    def test_gold_set_contains_required_adversarial_floors(self) -> None:
        gold_dir = REPO_ROOT / "tests/gold_set"
        key = json.loads((gold_dir / "verdict_key.json").read_text(encoding="utf-8"))
        cases = key["cases"]
        wrong_artifacts = [item for item in cases if item["kind"] == "artifact" and item["human_verdict"] == "not_supported"]
        wrong_proofs = [item for item in cases if item["kind"] == "proof" and item["human_verdict"] == "not_supported"]
        self.assertGreaterEqual(len(wrong_artifacts), 3)
        self.assertGreaterEqual(len(wrong_proofs), 3)
        self.assertIn("semantic_value_role_swap", {item["defect"] for item in cases})
        self.assertIn("causal_claim_mistyped_descriptive", {item["defect"] for item in cases})
        self.assertIn("fabricated_but_internally_coherent", {item["defect"] for item in cases})
        for case in cases:
            self.assertTrue((gold_dir / case["artifact"]).is_file(), case)

    def test_calibrated_mock_meets_precommitted_bar(self) -> None:
        result = calibrate_referee.calibrate(
            calibration_path=REPO_ROOT / "contracts/rubrics/calibration.yaml",
            gold_dir=REPO_ROOT / "tests/gold_set",
            mock_path=REPO_ROOT / "tests/gold_set/mock_referee.json",
            output_path=None,
        )
        self.assertTrue(result["calibrated"])
        self.assertGreaterEqual(result["agreement"], result["agreement_floor"])
        self.assertLessEqual(result["position_flip_rate"], result["position_flip_ceiling"])

    def test_disagreeing_mock_is_blocked(self) -> None:
        result = calibrate_referee.calibrate(
            calibration_path=REPO_ROOT / "contracts/rubrics/calibration.yaml",
            gold_dir=REPO_ROOT / "tests/gold_set",
            mock_path=REPO_ROOT / "tests/gold_set/mock_referee_disagree.json",
            output_path=None,
        )
        self.assertFalse(result["calibrated"])


if __name__ == "__main__":
    unittest.main()
