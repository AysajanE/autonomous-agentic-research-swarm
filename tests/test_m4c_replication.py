from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replication = _load("m4c_replication_package", "scripts/replication_package.py")
quality_gates = _load("m4c_quality_gates", "scripts/quality_gates.py")
disclosure = _load("m4c_generate_disclosure", "scripts/generate_disclosure.py")


def _reasons(result) -> set[str]:
    return {
        str(item.get("reason"))
        for item in result.details.get("failures", [])
        if isinstance(item, dict)
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _venue_fixture(root: Path) -> None:
    for relpath in (
        "contracts/venue.yaml",
        "contracts/authorship.yaml",
        "contracts/manuscript_sections.yaml",
        "reports/paper/index.qmd",
        "reports/paper/registry.json",
    ):
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relpath, target)
    target = root / "reports/paper/disclosure.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(disclosure.generate_disclosure(root), encoding="utf-8")


class M4cReplicationTests(unittest.TestCase):
    def test_live_empirical_and_fixture_profiles_have_truthful_levels(self) -> None:
        result = replication.audit_repo_profiles(ROOT, execute_empirical_master=False)
        self.assertTrue(result["ok"], result)
        profiles = result["profiles"]
        self.assertTrue(profiles["empirical"]["levels"]["Functional"])
        self.assertFalse(profiles["empirical"]["levels"]["Reproduced"])
        self.assertTrue(profiles["modeling"]["levels"]["Reproduced"])
        self.assertTrue(profiles["hybrid"]["levels"]["Reproduced"])

    def test_readme_uses_declared_versions_and_truthfully_marks_unlogged_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT, package, profile="empirical")
            readme = (package / "README.md").read_text(encoding="utf-8")
            self.assertIn("matplotlib>=3.8,<4", readme)
            self.assertIn("pandas>=2.2,<3", readme)
            # Current processed manifests have no environment.dependencies block.
            self.assertIn("exact runtime version unlogged", readme)
            self.assertIn("raw_evidence_unavailable", readme)
            self.assertIn("partial-reproducibility statement", readme.casefold())

    def test_raw_retention_live_repo_is_satisfied_only_by_recorded_amendment(self) -> None:
        result = quality_gates.check_raw_retention(ROOT)
        self.assertTrue(result.ok, result.details)
        self.assertEqual(len(result.details["amendment_satisfied"]), 6)
        self.assertEqual(result.details["archive_satisfied"], [])

    def test_reference_bundle_is_valid_but_null_author_blocks_submission(self) -> None:
        reference = quality_gates.check_venue_compliance(ROOT)
        self.assertTrue(reference.ok, reference.details)
        self.assertFalse(reference.details["submission_eligible"])
        submission = quality_gates.check_venue_compliance(ROOT, submission_declared=True)
        self.assertFalse(submission.ok)
        self.assertIn("venue_compliance_no_consented_author", _reasons(submission))

    def test_disclosure_is_byte_regenerable_from_logs(self) -> None:
        expected = disclosure.generate_disclosure(ROOT)
        actual = (ROOT / "reports/paper/disclosure.md").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)

    def test_submission_mode_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = yaml.safe_load(path.read_text(encoding="utf-8"))
            venue["ai_policy"]["allowed_release_modes"] = ["ai_native"]
            path.write_text(yaml.safe_dump(venue, sort_keys=False), encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertFalse(result.ok)
            self.assertIn("venue_compliance_mode_conflict", _reasons(result))

    def test_consent_incompatible_submission_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = yaml.safe_load(path.read_text(encoding="utf-8"))
            venue["venue_consent"]["consent_compatible"] = False
            path.write_text(yaml.safe_dump(venue, sort_keys=False), encoding="utf-8")
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            self.assertFalse(result.ok)
            self.assertIn("venue_compliance_consent_incompatible", _reasons(result))

    def test_forged_disclosure_is_not_regenerable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            (root / "reports/paper/disclosure.md").write_text("model-memory claim\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertFalse(result.ok)
            self.assertIn("venue_compliance_disclosure_not_regenerable", _reasons(result))

    def test_declared_submission_without_perimeter_or_program_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            reasons = _reasons(result)
            self.assertIn("venue_compliance_submission_manuscript_perimeter_missing", reasons)
            self.assertIn("venue_compliance_submission_program_not_instantiated", reasons)


if __name__ == "__main__":
    unittest.main()
