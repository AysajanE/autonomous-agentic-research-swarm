from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest



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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_member_hash(package: Path, relpath: str) -> None:
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = next(item for item in manifest["members"] if item["path"] == relpath)
    path = package / relpath
    member["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    member["bytes"] = path.stat().st_size
    _write_json(manifest_path, manifest)


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
            _refresh_member_hash(package, "bridge/experiment_output.json")
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

    def test_clean_room_that_consumes_committed_instances_without_regeneration_fails(self) -> None:
        # Independent expected: the audit deletes the committed instance, so a hash-printing stub cannot recreate it.
        with tempfile.TemporaryDirectory() as tmp:
            package = self._hybrid_package(Path(tmp))
            generator = package / "bridge/generate_instances.py"
            generator.write_text(
                "from pathlib import Path\nimport hashlib\n"
                "p=Path('data/processed_manifest/source.json')\nprint(hashlib.sha256(p.read_bytes()).hexdigest())\n",
                encoding="utf-8",
            )
            _refresh_member_hash(package, "bridge/generate_instances.py")
            metadata_path = package / "bridge/clean_room.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["traversed_bridge"] = False
            payload["regenerated_instances"] = False
            metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _refresh_member_hash(package, "bridge/clean_room.json")
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertFalse(result["levels"]["Reproduced"])
            self.assertIn("replication_hybrid_clean_room_bridge_not_traversed", result["failures"])

    def test_self_asserted_true_clean_room_flags_do_not_rescue_stub_generator(self) -> None:
        # Independent expected: advisory booleans cannot substitute for an observed recreated file.
        with tempfile.TemporaryDirectory() as tmp:
            package = self._hybrid_package(Path(tmp))
            generator = package / "bridge/generate_instances.py"
            generator.write_text("print('pretend-regenerated')\n", encoding="utf-8")
            _refresh_member_hash(package, "bridge/generate_instances.py")
            metadata_path = package / "bridge/clean_room.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload.update(
                {
                    "traversed_bridge": True,
                    "regenerated_instances": True,
                    "regenerated_source_sha256": "a" * 64,
                    "master_script_returncode": 0,
                }
            )
            _write_json(metadata_path, payload)
            _refresh_member_hash(package, "bridge/clean_room.json")
            result = replication.audit_package(package)
            self.assertFalse(result["levels"]["Reproduced"])
            self.assertIn("replication_hybrid_clean_room_bridge_not_traversed", result["failures"])

    def test_omitted_expected_bar_fails_complete_coverage(self) -> None:
        # Independent expected: all eight empirical outputs are fixed by EMPIRICAL_EXPECTED_BARS.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT, package, profile="empirical")
            manifest_path = package / "package_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            omitted = "reports/paper/paper_values.json"
            manifest["reproduction_bars"] = [item for item in manifest["reproduction_bars"] if item["path"] != omitted]
            _write_json(manifest_path, manifest)
            result = replication.audit_package(package)
            self.assertFalse(result["ok"])
            self.assertIn(f"replication_bar_coverage_missing:{omitted}", result["failures"])

    def test_master_returncode_zero_but_wrong_output_fails_reproduced(self) -> None:
        # Independent expected: rc=0 cannot override a byte mismatch in a declared output.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT / "tests/fixtures/m4c_modeling", package, profile="modeling")
            script = package / "modeling/derivation_check.py"
            script.write_text(
                "from pathlib import Path\nPath('modeling/convergence.jsonl').write_text('wrong\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            _refresh_member_hash(package, "modeling/derivation_check.py")
            result = replication.audit_package(package)
            self.assertFalse(result["levels"]["Reproduced"])
            self.assertIn(
                "replication_reproduced_byte_mismatch:modeling/convergence.jsonl",
                result["failures"],
            )

    def test_release_perimeter_empirical_master_wrong_bar_blocks(self) -> None:
        # Independent expected: release execution snapshots the table before a zero-rc master corrupts it.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            replication.generate_package(ROOT, package, profile="empirical")
            master = package / "MASTER.sh"
            master.write_text(
                "#!/bin/sh\nset -eu\nprintf 'corrupt\\n' > reports/tables/str_regime_summary.csv\n",
                encoding="utf-8",
            )
            _refresh_member_hash(package, "MASTER.sh")
            result = replication.audit_package(package, execute_master=True)
            self.assertFalse(result["ok"])
            self.assertIn(
                "replication_reproduced_byte_mismatch:reports/tables/str_regime_summary.csv",
                result["failures"],
            )

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
                        "command": "fixture acquisition command",
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

    def test_remote_url_with_unverified_digest_is_not_archive_evidence(self) -> None:
        # Independent expected: offline code cannot resolve a remote URL without a bound receipt.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data/raw_manifest/remote.json",
                {
                    "source": "fixture",
                    "command": "fixture acquisition command",
                    "archive_url": "https://example.invalid/fabricated.tar.zst",
                    "archive_sha256": "a" * 64,
                    "files": [{"path": "data/raw/missing", "sha256": "b" * 64, "bytes": 1}],
                },
            )
            result = quality_gates.check_raw_retention(root)
            self.assertIn("raw_retention_remote_pointer_unresolved", _reasons(result))

    def test_empty_raw_inventory_is_not_vacuously_retained(self) -> None:
        # Independent expected: zero files supplies no evidence of a retained snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "data/raw_manifest/empty.json",
                {"source": "fixture", "command": "fixture acquisition command", "files": []},
            )
            result = quality_gates.check_raw_retention(root)
            self.assertIn("raw_retention_empty_inventory_uncovered", _reasons(result))

    def test_release_mode_conflict_is_refused(self) -> None:
        # Independent expected value: mainstream is not a member of the mutated venue allowlist.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = json.loads(path.read_text(encoding="utf-8"))
            venue["ai_policy"]["allowed_release_modes"] = ["ai_native"]
            path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_venue_compliance(root)
            self.assertIn("venue_compliance_mode_conflict", _reasons(result))

    def test_consent_incompatible_submission_is_refused(self) -> None:
        # Independent expected value: false consent_compatible cannot authorize submission.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _venue_fixture(root)
            path = root / "contracts/venue.yaml"
            venue = json.loads(path.read_text(encoding="utf-8"))
            venue["venue_consent"]["consent_compatible"] = False
            path.write_text(json.dumps(venue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
