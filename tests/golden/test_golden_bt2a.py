from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import bt2a_rehearsal  # noqa: E402
import seeded_drill  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module  # noqa: E402


quality_gates = load_quality_gates_module()


_PROVENANCE = {
    "seeded_defect_catch_rate_min": "firm invariant",
    "human_review_hours_per_artifact_max": "provisional",
    "token_dollar_ceiling_usd": "provisional",
    "unresolved_fabrication_findings_max": "firm invariant",
    "registered_vs_reported_hypothesis_ratio": "firm invariant",
}
_VALID_BAR = {
    "schema_version": "research_swarm.bt2_bar.v1",
    "rehearsal_report": "reports/status/bt2a/rehearsal.json",
    "pass_fail_bar": {
        "seeded_defect_catch_rate_min": 1.0,
        "human_review_hours_per_artifact_max": 8.0,
        "token_dollar_ceiling_usd": 1500.0,
        "unresolved_fabrication_findings_max": 0,
        "registered_vs_reported_hypothesis_ratio": 1,
    },
    "number_provenance": dict(_PROVENANCE),
    "stage_b_abort_clause": {
        "triggers": ["a", "b", "c", "d"],
        "required_artifact": "event-journal closure + negative-result report",
    },
}
_VALID_REPORT = {
    "schema_version": "research_swarm.bt2a_rehearsal.v1",
    "all_known_answers_reproduced": True,
    "release_perimeter": {
        "registry_all_failing": True,
        "release_blocked_as_expected": True,
        "blocked_by_registry": True,
    },
    "drills": [{"drill_id": "D", "defect_class": "x", "blocking_gate": "g", "caught": True}],
    "seeded_defect_catch_rate": 1.0,
    "registered_vs_reported_hypothesis_ratio": 1,
}


def _write_bar_and_report(root: Path, *, bar: dict, report: dict | None) -> None:
    (root / "contracts").mkdir(parents=True, exist_ok=True)
    (root / "contracts/bt2_bar.json").write_text(json.dumps(bar), encoding="utf-8")
    if report is not None:
        (root / "reports/status/bt2a").mkdir(parents=True, exist_ok=True)
        (root / "reports/status/bt2a/rehearsal.json").write_text(json.dumps(report), encoding="utf-8")


def _reasons(result: object) -> set[str]:
    return {
        str(item["reason"])
        for item in getattr(result, "details", {}).get("failures", [])
        if isinstance(item, dict) and isinstance(item.get("reason"), str)
    }


class GoldenBt2aTests(unittest.TestCase):
    def test_bt2_bar_gate_passes_on_committed_reference(self) -> None:
        # The real repo's committed bar + rehearsal report must satisfy the gate
        # on a clean checkout.
        with chdir(ROOT):
            result = quality_gates.gate_bt2_bar()
        self.assertTrue(result.ok, result.details)

    def test_bt2_bar_gate_skips_when_absent(self) -> None:
        # A pack that has not committed a BT2 bar is not forced to have one.
        with tempfile.TemporaryDirectory() as tmp:
            with chdir(Path(tmp)):
                result = quality_gates.gate_bt2_bar()
        self.assertTrue(result.ok)
        self.assertTrue(result.details.get("skipped"))

    def test_bt2_bar_gate_fails_below_catch_threshold(self) -> None:
        # A rehearsal whose measured catch rate is below the committed threshold
        # must fail the gate (the bar has teeth).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # a drill list GENUINELY consistent with a below-threshold rate (4/5)
            drills = [
                {"drill_id": f"D{i}", "defect_class": "x", "blocking_gate": "g", "caught": i < 4}
                for i in range(5)
            ]
            report = {**_VALID_REPORT, "drills": drills, "seeded_defect_catch_rate": 0.8}
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_catch_rate_below_threshold", _reasons(result))

    def test_bt2_bar_gate_fails_on_hypothesis_ratio_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {**_VALID_REPORT, "registered_vs_reported_hypothesis_ratio": 0.5}
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_hypothesis_ratio_violated", _reasons(result))

    def test_bt2_bar_gate_fails_when_release_not_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {
                **_VALID_REPORT,
                "release_perimeter": {"registry_all_failing": False, "release_blocked_as_expected": False},
            }
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_release_perimeter_not_exercised", _reasons(result))

    def test_bt2_bar_gate_fails_when_reference_not_reproduced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {**_VALID_REPORT, "all_known_answers_reproduced": False}
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_rehearsal_reference_not_reproduced", _reasons(result))

    def test_bt2_bar_gate_fails_on_malformed_abort_clause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bar = {**_VALID_BAR, "stage_b_abort_clause": {"triggers": ["only", "two"], "required_artifact": "x"}}
            _write_bar_and_report(root, bar=bar, report=_VALID_REPORT)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_abort_clause_malformed", _reasons(result))

    def test_bt2_bar_gate_reconciles_catch_rate_against_drill_list(self) -> None:
        # A summary claiming 1.0 with a drill marked caught:false must NOT pass —
        # the gate recomputes the rate from the drill list.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {
                **_VALID_REPORT,
                "drills": [{"drill_id": "D", "defect_class": "x", "blocking_gate": "g", "caught": False}],
                "seeded_defect_catch_rate": 1.0,
            }
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_catch_rate_inconsistent_with_drills", _reasons(result))

    def test_bt2_bar_gate_fails_on_out_of_range_bar_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bar = {
                **_VALID_BAR,
                "pass_fail_bar": {**_VALID_BAR["pass_fail_bar"], "registered_vs_reported_hypothesis_ratio": 2},
            }
            _write_bar_and_report(root, bar=bar, report=_VALID_REPORT)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_hypothesis_ratio_invariant_not_one", _reasons(result))

    def test_bt2_bar_gate_fails_when_catch_invariant_silently_lowered(self) -> None:
        # The catch threshold is a FIRM invariant (unforgivable class): a bar that
        # lowers it below 1.0 must fail, not silently weaken the invariant.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bar = {
                **_VALID_BAR,
                "pass_fail_bar": {**_VALID_BAR["pass_fail_bar"], "seeded_defect_catch_rate_min": 0.6},
            }
            _write_bar_and_report(root, bar=bar, report=_VALID_REPORT)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_catch_threshold_not_unforgivable_one", _reasons(result))

    def test_bt2_bar_gate_fails_on_null_abort_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bar = {**_VALID_BAR, "stage_b_abort_clause": {"triggers": [None, None, None, None], "required_artifact": "x"}}
            _write_bar_and_report(root, bar=bar, report=_VALID_REPORT)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_abort_clause_malformed", _reasons(result))

    def test_bt2_bar_gate_fails_on_missing_number_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bar = {**_VALID_BAR, "number_provenance": {}}
            _write_bar_and_report(root, bar=bar, report=_VALID_REPORT)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn(
                "bt2_bar_number_provenance_missing:seeded_defect_catch_rate_min", _reasons(result)
            )

    def test_bt2_bar_gate_fails_when_block_not_attributed_to_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {
                **_VALID_REPORT,
                "release_perimeter": {
                    "registry_all_failing": True,
                    "release_blocked_as_expected": True,
                    "blocked_by_registry": False,
                },
            }
            _write_bar_and_report(root, bar=_VALID_BAR, report=report)
            with chdir(root):
                result = quality_gates.gate_bt2_bar()
            self.assertFalse(result.ok)
            self.assertIn("bt2_bar_release_perimeter_not_exercised", _reasons(result))

    def test_rehearsal_release_perimeter_blocks_all_failing_registry(self) -> None:
        # On the real STR repo the all-failing registry blocks release; the
        # rehearsal records that as the expected known outcome (not a failure).
        perimeter = bt2a_rehearsal.exercise_release_perimeter()
        self.assertTrue(perimeter["release_blocked_as_expected"])
        self.assertTrue(perimeter["registry_all_failing"])

    def test_rehearsal_hypothesis_ratio_is_one_for_reference(self) -> None:
        self.assertEqual(bt2a_rehearsal.hypothesis_ratio(), 1.0)

    def test_rehearsal_missed_injection_fails_red(self) -> None:
        # A drill that is injected but NOT caught must fail the rehearsal RED.
        missed = seeded_drill.DrillSpec(
            "BT2A-MISS",
            "fixture_missed",
            lambda: seeded_drill.Observation(False, "fixture_gate", "gate returned green"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(seeded_drill.MissedInjection):
                seeded_drill.run_rotation((missed,), journal_root=Path(tmp), timestamp="2026-07-11T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
