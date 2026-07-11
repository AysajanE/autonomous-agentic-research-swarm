from __future__ import annotations

import hashlib
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

import seeded_drill  # noqa: E402
import swarm_events  # noqa: E402
from golden.harness import GoldenRepo  # noqa: E402
from runtime_test_utils import (  # noqa: E402
    chdir,
    load_quality_gates_module,
    pin_pack_to_ledger,
    scaffold_runtime_repo,
)


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
            events, malformed = swarm_events.read_events(
                journal, journal_path=swarm_events.DRILL_JOURNAL_PATH
            )
            self.assertEqual(malformed, 0)
            self.assertEqual(len(events), 6)
            # Drills must NEVER touch the compliance journal the disclosure reads.
            compliance, _ = swarm_events.read_events(journal)
            self.assertEqual(compliance, [])

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
            events, malformed = swarm_events.read_events(
                journal, journal_path=swarm_events.DRILL_JOURNAL_PATH
            )
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

    def test_effective_scratch_state_is_honest_per_backend(self) -> None:
        # F2: the confinement report must describe what is ACTUALLY applied to the
        # launched subprocesses — never claim a scrub/proxy it does not perform.
        import os

        import integrity_audit

        # mock backend (CI default): no auditor egress; offline subprocesses are
        # credential-scrubbed and dead-proxied.
        mock = integrity_audit.effective_confinement("mock")
        self.assertFalse(mock["os_enforced"])
        self.assertEqual(mock["effective_network"], "proxy_environment_only")
        self.assertEqual(mock["credential_isolation"], "environment_scrub_only")

        # live backend: honestly reports retained vendor transport + credential,
        # never a scrub/proxy the live API call cannot honour.
        live = integrity_audit.effective_confinement("live_claude")
        self.assertFalse(live["os_enforced"])
        self.assertEqual(live["effective_network"], "vendor_api_transport_only")
        self.assertEqual(live["credential_isolation"], "vendor_credential_retained")

        # the `environment_scrub_only` label is TRUE: the offline env strips
        # credentials and dead-proxies the network.
        os.environ["FAKE_SECRET_TOKEN"] = "sensitive"
        os.environ["AWS_ACCESS_KEY_ID"] = "sensitive"
        try:
            offline_env = integrity_audit._offline_subprocess_env()
        finally:
            del os.environ["FAKE_SECRET_TOKEN"], os.environ["AWS_ACCESS_KEY_ID"]
        self.assertNotIn("FAKE_SECRET_TOKEN", offline_env)
        self.assertNotIn("AWS_ACCESS_KEY_ID", offline_env)
        self.assertEqual(offline_env["https_proxy"], "http://127.0.0.1:9")

        # the live backend keeps ONLY the vendor credential, scrubbing the rest.
        os.environ["ANTHROPIC_API_KEY"] = "vendor"
        os.environ["AWS_ACCESS_KEY_ID"] = "sensitive"
        try:
            live_env = integrity_audit._live_backend_env()
        finally:
            del os.environ["ANTHROPIC_API_KEY"], os.environ["AWS_ACCESS_KEY_ID"]
        self.assertIn("ANTHROPIC_API_KEY", live_env)
        self.assertNotIn("AWS_ACCESS_KEY_ID", live_env)

    def test_escalate_rejects_unregistered_human_escalation_class(self) -> None:
        # F3: a new *human* escalation cannot ship without a registered class
        # (which runbook_staleness then forces to carry a playbook).  Operational
        # escalations (no class) remain allowed as the broad notable-event tier.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            registered = swarm_events.escalate(
                repo,
                {"event": "plan_awaiting_human_approval", "escalation_class": "blocked_with_human"},
            )
            self.assertEqual(registered["escalation_class"], "blocked_with_human")
            self.assertTrue(registered["escalation"])
            # operational escalation without a class: still allowed
            swarm_events.escalate(repo, {"event": "merge_reverted"})
            # unregistered human-escalation class: rejected before journaling
            with self.assertRaises(ValueError):
                swarm_events.escalate(
                    repo,
                    {"event": "data_loss_detected", "escalation_class": "data_loss_detected"},
                )
            # every class tagged on a real runtime emitter must be registered
            for cls in ("blocked_with_human", "hypothesis_task_retirement"):
                self.assertIn(cls, swarm_events.ESCALATION_CLASSES)

    def test_frozen_exemption_ledger_rejects_silent_new_task_exemption(self) -> None:
        # F4: task_lint honours anything on the exemption ledger, so a silently
        # self-added v1 W6/W7/W8 task exemption would skip mandatory program
        # tagging.  The ledger is frozen by a pack-declared digest; growing it
        # without updating the pin (a deliberate, reviewable act) is rejected.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            ledger = root / "contracts/historical_exemptions.json"
            base = {
                "schema_version": "research_swarm.historical_exemptions.v1",
                "created_at_utc": "2026-07-09T00:00:00Z",
                "rationale": "fixture",
                "run_manifests": [],
                "review_logs": [],
                "tasks": [],
            }
            ledger.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
            pin_pack_to_ledger(root)  # pack pin matches the ledger -> green
            with chdir(root):
                self.assertTrue(quality_gates.gate_historical_exemptions().ok)
            # Silently append a new v1 W8 task exemption WITHOUT updating the pin.
            sneaky = root / ".orchestrator/done/T999_sneaky_w8.md"
            sneaky.parent.mkdir(parents=True, exist_ok=True)
            sneaky.write_text("# sneaky untagged W8 task\n", encoding="utf-8")
            base["tasks"].append(
                {
                    "path": ".orchestrator/done/T999_sneaky_w8.md",
                    "schema_version": "v1",
                    "sha256": hashlib.sha256(sneaky.read_bytes()).hexdigest(),
                }
            )
            ledger.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
            with chdir(root):
                red = quality_gates.gate_historical_exemptions()
            self.assertFalse(red.ok)
            self.assertIn(
                "historical_exemptions_ledger_mutated",
                [str(x) for x in red.details["failures"]],
            )

    def test_ci_filters_prompt_contract_and_gate_surfaces_into_golden_suite(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("contracts/", workflow)
        self.assertIn("docs/prompts/", workflow)
        self.assertIn("discover -s tests/golden", workflow)
        # every gate-behaviour-bearing script that a golden regression can hinge
        # on must trip the change filter (not just the three original surfaces).
        for surface in (
            "quality_gates",
            "swarm",
            "swarm_taskfile",
            "swarm_events",
            "integrity_audit",
            "generate_disclosure",
            "falsify_claims",
            "replication_package",
        ):
            self.assertIn(surface, workflow)


if __name__ == "__main__":
    unittest.main()
