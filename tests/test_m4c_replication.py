from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock



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


def _refresh_member_hash(package: Path, relpath: str) -> None:
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = next(item for item in manifest["members"] if item["path"] == relpath)
    path = package / relpath
    member["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    member["bytes"] = path.stat().st_size
    _write_json(manifest_path, manifest)


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
        self.assertEqual(profiles["empirical"]["master_execution"], "staged_release_perimeter")
        self.assertTrue(profiles["hybrid"]["bridge_traversed"])

    def test_hybrid_generator_writes_deterministic_instance_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            shutil.copytree(ROOT / "tests/fixtures/m4c_hybrid", fixture)
            instance = fixture / "modeling/instance_manifest.json"
            expected = hashlib.sha256(instance.read_bytes()).hexdigest()
            instance.unlink()
            completed = subprocess.run(
                [sys.executable, "bridge/generate_instances.py"],
                cwd=fixture,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(instance.is_file(), "generator must write the instance, not print a digest")
            self.assertEqual(hashlib.sha256(instance.read_bytes()).hexdigest(), expected)

    def test_replication_gate_stages_empirical_master_until_release_perimeter(self) -> None:
        result = {
            "ok": True,
            "profiles": {
                profile: {"ok": True, "profile": profile, "levels": {"Functional": True, "Reproduced": profile != "empirical"}, "failures": []}
                for profile in ("empirical", "modeling", "hybrid")
            },
        }
        with mock.patch.object(quality_gates.replication_package, "audit_repo_profiles", return_value=result) as audit:
            lightweight = quality_gates.check_replication_package_audit(ROOT)
            audit.assert_called_once_with(ROOT, execute_empirical_master=False)
            self.assertEqual(lightweight.details["status"], "active_lightweight")
        with mock.patch.object(quality_gates.replication_package, "audit_repo_profiles", return_value=result) as audit:
            release = quality_gates.check_replication_package_audit(ROOT, release_perimeter=True)
            audit.assert_called_once_with(ROOT, execute_empirical_master=True)
            self.assertEqual(release.details["status"], "release_perimeter")

    def test_fixture_audit_passes_from_successor_tree_tracked_only_export(self) -> None:
        # A temporary index models the uncommitted successor without touching the real index/history.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            staging = tmp_path / "staging"
            staging.mkdir()
            shutil.copyfile(ROOT / ".gitignore", staging / ".gitignore")
            for profile in ("modeling", "hybrid"):
                destination = staging / f"tests/fixtures/m4c_{profile}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(ROOT / f"tests/fixtures/m4c_{profile}", destination)
            subprocess.run(["git", "init", "-q"], cwd=staging, check=True)
            index = tmp_path / "successor.index"
            environment = {**os.environ, "GIT_INDEX_FILE": str(index)}
            subprocess.run(
                ["git", "add", "-A", "--", ".gitignore", "tests/fixtures/m4c_modeling", "tests/fixtures/m4c_hybrid"],
                cwd=staging,
                env=environment,
                check=True,
            )
            tree = subprocess.run(
                ["git", "write-tree"], cwd=staging, env=environment, check=True, capture_output=True, text=True
            ).stdout.strip()
            archive = tmp_path / "fixtures.tar"
            with archive.open("wb") as handle:
                subprocess.run(
                    ["git", "archive", tree, "tests/fixtures/m4c_modeling", "tests/fixtures/m4c_hybrid"],
                    cwd=staging,
                    env=environment,
                    check=True,
                    stdout=handle,
                )
            export = tmp_path / "export"
            export.mkdir()
            with tarfile.open(archive) as handle:
                # The data filter was backported to CPython 3.11.4; CI runs 3.11.0. This tar is a
                # git archive of the repo's own tracked files, so fall back to a plain extract.
                try:
                    handle.extractall(export, filter="data")
                except TypeError:
                    handle.extractall(export)
            for profile in ("modeling", "hybrid"):
                package = tmp_path / f"package-{profile}"
                replication.generate_package(export / f"tests/fixtures/m4c_{profile}", package, profile=profile)
                result = replication.audit_package(package)
                self.assertTrue(result["ok"], result)

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
            venue = json.loads(path.read_text(encoding="utf-8"))
            venue["ai_policy"]["allowed_release_modes"] = ["ai_native"]
            path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertFalse(result.ok)
            self.assertIn("venue_compliance_mode_conflict", _reasons(result))

    def test_consent_incompatible_submission_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = json.loads(path.read_text(encoding="utf-8"))
            venue["venue_consent"]["consent_compatible"] = False
            path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

    def test_remote_archive_url_without_offline_receipt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data/raw_manifest/remote.json",
                {
                    "source": "fixture",
                    "command": "fixture acquisition command",
                    "archive_url": "https://example.invalid/archive.tar.zst",
                    "archive_sha256": "a" * 64,
                    "files": [{"path": "data/raw/missing.csv", "sha256": "b" * 64, "bytes": 1}],
                },
            )
            result = quality_gates.check_raw_retention(root)
            self.assertIn("raw_retention_remote_pointer_unresolved", _reasons(result))

    def test_hash_bound_offline_archive_receipt_satisfies_remote_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_url = "s3://fixture-bucket/snapshot-v1.tar.zst"
            archive_sha = "a" * 64
            receipt_path = root / "data/raw_manifest/receipts/snapshot-v1.json"
            _write_json(
                receipt_path,
                {
                    "archive_url": archive_url,
                    "archive_sha256": archive_sha,
                    "retrieval_metadata": {"verified_at_utc": "2026-01-01T00:00:00Z", "operator": "fixture"},
                },
            )
            _write_json(
                root / "data/raw_manifest/remote.json",
                {
                    "source": "fixture",
                    "command": "fixture acquisition command",
                    "archive_url": archive_url,
                    "archive_sha256": archive_sha,
                    "archive_receipt": {
                        "path": "data/raw_manifest/receipts/snapshot-v1.json",
                        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    },
                    "files": [{"path": "data/raw/missing.csv", "sha256": "b" * 64, "bytes": 1}],
                },
            )
            result = quality_gates.check_raw_retention(root)
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["archive_satisfied"], ["data/raw_manifest/remote.json"])

    def test_empty_raw_inventory_without_retention_coverage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data/raw_manifest/empty.json",
                {"source": "fixture", "command": "fixture acquisition command", "files": []},
            )
            result = quality_gates.check_raw_retention(root)
            self.assertIn("raw_retention_empty_inventory_uncovered", _reasons(result))

    def test_blank_raw_access_command_fails_manifest_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data/raw_manifest/blank.json",
                {
                    "source": "fixture",
                    "fetched_at_utc": "2026-01-01T00:00:00Z",
                    "command": "   ",
                    "files": [],
                },
            )
            with contextlib.chdir(root):
                result = quality_gates.gate_raw_manifest_validity()
            self.assertFalse(result.ok)
            self.assertTrue(any(failure.endswith(":command_blank") for failure in result.details["failures"]))

    def test_claimed_consent_with_dangling_evidence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            authorship_path = root / "contracts/authorship.yaml"
            authorship = json.loads(authorship_path.read_text(encoding="utf-8"))
            authorship["human_author_of_record"] = "Fixture Author"
            authorship["human_author_consent"] = {
                "status": "consented",
                "evidence_pointer": {"path": "evidence/missing-consent.json", "sha256": "a" * 64},
            }
            authorship_path.write_text(json.dumps(authorship, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manuscript = root / "reports/paper/index.qmd"
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8").replace("author: null", 'author: "Fixture Author"'),
                encoding="utf-8",
            )
            venue_path = root / "contracts/venue.yaml"
            venue = json.loads(venue_path.read_text(encoding="utf-8"))
            venue["venue_consent"].update(
                {
                    "real_submission_authorized": True,
                    "evidence_pointer": {"path": "evidence/missing-venue.json", "sha256": "b" * 64},
                }
            )
            venue["ai_policy"]["evidence_pointer"] = {
                "path": "evidence/missing-policy.json",
                "sha256": "c" * 64,
            }
            venue_path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            reasons = _reasons(result)
            self.assertIn("venue_compliance_consent_evidence_unresolved", reasons)
            self.assertIn("venue_compliance_venue_evidence_unresolved", reasons)

    def test_declared_submission_missing_required_statement_binding_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            sections_path = root / "contracts/manuscript_sections.yaml"
            sections = json.loads(sections_path.read_text(encoding="utf-8"))
            del sections["required_statement_bindings"]["data_availability"]
            sections_path.write_text(json.dumps(sections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root, submission_declared=True)
            failures = result.details["failures"]
            self.assertTrue(
                any(
                    item.get("reason") == "venue_compliance_required_statement_missing"
                    and item.get("subject") == "data_availability"
                    for item in failures
                    if isinstance(item, dict)
                )
            )


if __name__ == "__main__":
    unittest.main()
