from __future__ import annotations

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

import seeded_drill  # noqa: E402
import swarm_events  # noqa: E402
from golden.harness import GoldenRepo  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module  # noqa: E402


quality_gates = load_quality_gates_module()


class GoldenM5bTests(unittest.TestCase):
    def test_rotation_records_five_real_catches_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal"
            result = seeded_drill.run_rotation(
                seeded_drill.ROTATION,
                journal_root=journal,
                timestamp="2026-07-11T12:00:00Z",
            )
            self.assertEqual(len(result["events"]), 5)
            self.assertTrue(all(event["injected"] is True for event in result["events"]))
            self.assertTrue(all(event["caught"] is True for event in result["events"]))
            self.assertEqual(result["summary"]["catch_rate"], 1.0)
            events, malformed = swarm_events.read_events(journal)
            self.assertEqual(malformed, 0)
            self.assertEqual(len(events), 6)

    def test_injected_but_not_caught_is_red_and_raises(self) -> None:
        missed = seeded_drill.DrillSpec(
            "M5B-MISS",
            "fixture_missed_defect",
            lambda: seeded_drill.Observation(False, "fixture_gate", "gate returned green"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal"
            with self.assertRaises(seeded_drill.MissedInjection):
                seeded_drill.run_rotation(
                    (missed,),
                    journal_root=journal,
                    timestamp="2026-07-11T12:00:00Z",
                )
            events, malformed = swarm_events.read_events(journal)
            self.assertEqual(malformed, 0)
            self.assertFalse(events[0]["caught"])
            self.assertEqual(events[-1]["status"], "red")
            self.assertEqual(events[-1]["catch_rate"], 0.0)

    def test_runbook_staleness_fails_new_undocumented_registry_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "operator_runbook.md"
            path.write_text("- gate: documented\n- escalation_class: documented\n", encoding="utf-8")
            result = quality_gates.check_runbook_staleness(
                path,
                gate_names=("documented", "new_gate"),
                escalation_classes=("documented", "new_escalation"),
            )
            reasons = {item["reason"] for item in result.details["failures"]}
            self.assertFalse(result.ok)
            self.assertEqual(
                reasons,
                {"runbook_gate_missing", "runbook_escalation_class_missing"},
            )

    def test_task_lint_requires_program_tags_in_template_workstreams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task = repo.write_task(
                "backlog",
                "T995",
                schema="v2",
                task_kind="model",
                workstream="W8",
            )
            with chdir(repo.root):
                red = quality_gates.gate_task_lint()
            red_reasons = {
                item["reason"] for item in red.details["failures"] if item.get("task") == "T995"
            }
            self.assertEqual(red_reasons, {"template_program_tag_required"})
            text = task.read_text(encoding="utf-8").replace(
                'workstream: "W8"',
                'workstream: "W8"\nprogram_id: "theory"\nprogram_node: "formalization"',
            )
            task.write_text(text, encoding="utf-8")
            with chdir(repo.root):
                green = quality_gates.gate_task_lint()
            self.assertTrue(green.ok, green.details)

    def test_effective_scratch_state_never_claims_os_enforcement_here(self) -> None:
        state = seeded_drill.REPO_ROOT  # sanity: harness is repo-bound, not an external service
        self.assertTrue(state.is_dir())
        import integrity_audit

        effective = integrity_audit.effective_confinement()
        self.assertFalse(effective["os_enforced"])
        self.assertEqual(effective["effective_network"], "proxy_environment_only")

    def test_ci_filters_prompt_contract_and_gate_surfaces_into_golden_suite(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("contracts/", workflow)
        self.assertIn("docs/prompts/", workflow)
        self.assertIn("scripts/(quality_gates|swarm|swarm_taskfile)", workflow)
        self.assertIn("discover -s tests/golden", workflow)


if __name__ == "__main__":
    unittest.main()
