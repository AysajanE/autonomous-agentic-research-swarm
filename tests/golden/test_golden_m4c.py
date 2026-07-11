from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replication = _load("golden_m4c_replication", "scripts/replication_package.py")
quality_gates = _load("golden_m4c_gates", "scripts/quality_gates.py")
disclosure = _load("golden_m4c_disclosure", "scripts/generate_disclosure.py")


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
    output = root / "reports/paper/disclosure.md"
    output.write_text(disclosure.generate_disclosure(root), encoding="utf-8")


def _reasons(result) -> set[str]:
    return {
        str(item.get("reason"))
        for item in result.details.get("failures", [])
        if isinstance(item, dict)
    }


class GoldenM4cTests(unittest.TestCase):
    def _hybrid_package(self, root: Path) -> Path:
        package = root / "package"
        replication.generate_package(ROOT / "tests/fixtures/m4c_hybrid", package, profile="hybrid")
        return package

    def test_missing_profile_member_fails_closed(self) -> None:
        # Independent expected value: MASTER.sh is in COMMON_REQUIRED, so deletion must fail.
        with tempfile.TemporaryDirectory() as tmp:
            package = self._hybrid_package(Path(tmp))
            (package / "MASTER.sh").unlink()
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertIn("replication_required_member_missing:MASTER.sh", result["failures"])

    def test_broken_cross_layer_hash_link_fails_closed(self) -> None:
        # Independent expected value: output.instance_manifest.sha256 must hash the packaged instance manifest.
        with tempfile.TemporaryDirectory() as tmp:
            package = self._hybrid_package(Path(tmp))
            path = package / "bridge/experiment_output.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["instance_manifest"]["sha256"] = "0" * 64
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertIn("replication_cross_layer_hash_link_broken", result["failures"])

    def test_byte_vs_content_bar_substitution_fails_closed(self) -> None:
        # Independent expected value: every empirical SVG is content-equivalent, never byte-identity.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT, package, profile="empirical")
            path = package / "package_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            svg = next(item for item in payload["reproduction_bars"] if item["path"].endswith(".svg"))
            svg["verification_bar"] = "byte_identity"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertTrue(any(item.startswith("replication_bar_mismatch:") for item in result["failures"]))

    def test_clean_room_that_consumes_instances_without_bridge_traversal_fails(self) -> None:
        # Independent expected value: Reproduced requires traversed_bridge and regenerated_instances true.
        with tempfile.TemporaryDirectory() as tmp:
            package = self._hybrid_package(Path(tmp))
            path = package / "bridge/clean_room.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["traversed_bridge"] = False
            payload["regenerated_instances"] = False
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertFalse(result["levels"]["Reproduced"])
            self.assertIn("replication_hybrid_clean_room_bridge_not_traversed", result["failures"])

    def test_absent_raw_without_archive_or_amendment_fails(self) -> None:
        # Independent expected value: an absent manifested file has neither permitted satisfier here.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data/raw_manifest/source_2026-01-01.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "source": "fixture",
                        "files": [{"path": "data/raw/missing.csv", "sha256": "a" * 64, "bytes": 1}],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result = quality_gates.check_raw_retention(root)
            reasons = {item["reason"] for item in result.details["failures"]}
            self.assertEqual(reasons, {"raw_retention_unresolvable_pointer"})

    def test_release_mode_conflict_is_refused(self) -> None:
        # Independent expected value: mainstream is not a member of the mutated venue allowlist.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = yaml.safe_load(path.read_text(encoding="utf-8"))
            venue["ai_policy"]["allowed_release_modes"] = ["ai_native"]
            path.write_text(yaml.safe_dump(venue, sort_keys=False), encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertIn("venue_compliance_mode_conflict", _reasons(result))

    def test_consent_incompatible_submission_is_refused(self) -> None:
        # Independent expected value: false consent_compatible cannot authorize submission.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = yaml.safe_load(path.read_text(encoding="utf-8"))
            venue["venue_consent"]["consent_compatible"] = False
            path.write_text(yaml.safe_dump(venue, sort_keys=False), encoding="utf-8")
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            self.assertIn("venue_compliance_consent_incompatible", _reasons(result))

    def test_nonregenerable_disclosure_is_refused(self) -> None:
        # Independent expected value: arbitrary prose cannot equal the log projection.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            (root / "reports/paper/disclosure.md").write_text("memory-only claim\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertIn("venue_compliance_disclosure_not_regenerable", _reasons(result))

    def test_declared_submission_requires_manuscript_perimeter_and_program(self) -> None:
        # Independent expected value: the fixture omits references/config/values and all program tasks.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            reasons = _reasons(result)
            self.assertIn("venue_compliance_submission_manuscript_perimeter_missing", reasons)
            self.assertIn("venue_compliance_submission_program_not_instantiated", reasons)


if __name__ == "__main__":
    unittest.main()
