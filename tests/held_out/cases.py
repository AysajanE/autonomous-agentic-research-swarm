from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from golden.test_golden_m4b import _program_fixture, _task  # noqa: E402
from golden.test_golden_m4c import _refresh_member_hash, _venue_fixture  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module  # noqa: E402
from test_m3b_referee import RefereeFixture  # noqa: E402
from test_m4c_replication import replication  # noqa: E402


quality_gates = load_quality_gates_module()


def _reasons(result: object) -> set[str]:
    details = getattr(result, "details", {})
    reasons: set[str] = set()
    for item in details.get("failures", []):
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            reasons.add(str(item["reason"]))
        elif isinstance(item, str):
            reasons.add(item.split(":", 1)[0])
    return reasons


class HeldOutM5Cases(unittest.TestCase):
    def test_computed_paper_rejects_fresh_source_forgery(self) -> None:
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
            values = root / "reports/paper/paper_values.json"
            payload = json.loads(values.read_text(encoding="utf-8"))
            payload["values"]["post_dencun_mean_str_pct"]["value"] = 88.88
            payload["values"]["post_dencun_mean_str_pct"]["display"] = "88.88%"
            values.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertIn("paper_value_mismatch_source", _reasons(result))

    def test_referee_blocks_fresh_wrong_but_coherent_artifact(self) -> None:
        text = (
            "# Analysis\n\nThe task claims a preregistered estimator, but the artifact "
            "quietly substitutes an unregistered complete-case estimator and "
            "then repeats internally consistent values from that substitution.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo", text=text)
            fixture.write_mock(overrides={"ANALYSIS_PROTOCOL_CONFORMANCE": "not_supported"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            self.assertEqual(report["overall"], "not_supported")

    def test_program_conformance_rejects_fresh_cross_mode_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _program_fixture(root, mode="hybrid")
            _task(
                root,
                "T991",
                program_id="w7_writing",
                program_node="section_draft",
                task_kind="writing",
                workstream="W7",
            )
            result = quality_gates.check_program_conformance(root, strict=True)
            self.assertFalse(result.ok)
            self.assertIn("mode_foreign_program_node", _reasons(result))

    def test_replication_audit_rejects_fresh_bridge_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT / "tests/fixtures/m4c_hybrid", package, profile="hybrid")
            generator = package / "bridge/generate_instances.py"
            generator.write_text("print('coherent but does not regenerate the bridge')\n", encoding="utf-8")
            _refresh_member_hash(package, "bridge/generate_instances.py")
            clean_room = package / "bridge/clean_room.json"
            payload = json.loads(clean_room.read_text(encoding="utf-8"))
            payload.update({"traversed_bridge": True, "regenerated_instances": True})
            clean_room.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _refresh_member_hash(package, "bridge/clean_room.json")
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertIn("replication_hybrid_clean_room_bridge_not_traversed", result["failures"])

    def test_compliance_rejects_fresh_release_mode_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            venue_path = root / "contracts/venue.yaml"
            venue = json.loads(venue_path.read_text(encoding="utf-8"))
            venue["release"]["mode"] = "ai_native"
            venue["ai_policy"]["allowed_release_modes"] = ["mainstream"]
            venue_path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertFalse(result.ok)
            self.assertIn("venue_compliance_mode_conflict", _reasons(result))


if __name__ == "__main__":
    unittest.main()
