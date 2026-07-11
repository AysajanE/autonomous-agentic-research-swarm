from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from runtime_test_utils import chdir, load_quality_gates_module  # noqa: E402


quality_gates = load_quality_gates_module()


class GoldenM4Test(unittest.TestCase):
    def test_GM4_01_source_oracles_match_then_forged_headline_fails(self) -> None:
        with (ROOT / "reports/tables/str_regime_summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["regime_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(round(float(rows["pre_dencun"]["mean_str_pct"]), 2), 69.14)
        self.assertEqual(round(float(rows["post_dencun"]["mean_str_pct"]), 2), 11.68)

        reconciliation = json.loads(
            (ROOT / "reports/validation/cross_source_reconciliation.json").read_text(encoding="utf-8")
        )
        may_2025 = next(
            row
            for row in reconciliation["checks"][5]["details"]["unexplained_monthly_aggregate"]
            if row["month_utc"] == "2025-05"
        )
        self.assertEqual(round(float(may_2025["pct_difference"]) * 100, 2), 12.37)
        self.assertEqual(round(float(may_2025["rent_paid_eth_authoritative"]), 2), 148.42)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relpath in (
                "reports/paper/index.qmd",
                "reports/paper/paper_values.json",
                "reports/tables/str_regime_summary.csv",
                "reports/validation/rollup_panel_validation.json",
                "reports/validation/cross_source_reconciliation.json",
                "contracts/claims.yaml",
                "contracts/pack.json",
                "docs/protocol.md",
            ):
                target = root / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relpath, target)
            values_path = root / "reports/paper/paper_values.json"
            payload = json.loads(values_path.read_text(encoding="utf-8"))
            payload["values"]["pre_dencun_mean_str_pct"]["value"] = 99.99
            payload["values"]["pre_dencun_mean_str_pct"]["display"] = "99.99%"
            values_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("paper_value_mismatch_source" in failure for failure in result.details["failures"]),
                result.details,
            )


if __name__ == "__main__":
    unittest.main()
