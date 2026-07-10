from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    register_historical_exemption,
    REPO_ROOT,
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_tools = _load_module("m0_batch_d_manifest_tools", REPO_ROOT / "scripts" / "manifest_tools.py")
validate_str_pipeline = _load_module(
    "m0_batch_d_validate_str_pipeline",
    REPO_ROOT / "src" / "validation" / "validate_str_pipeline.py",
)


def _sha256_and_bytes(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _processed_payload(output_path: str, output: Path) -> dict[str, object]:
    sha256, size = _sha256_and_bytes(output)
    return {
        "as_of_utc_date": "2026-07-09",
        "inputs": ["data/raw_manifest/source_2026-07-09.json"],
        "transform": {
            "script_path": "src/etl/build.py",
            "git_sha": "0" * 40,
            "command": "python src/etl/build.py",
        },
        "outputs": [{"path": output_path, "sha256": sha256, "bytes": size}],
    }


def _raw_payload(raw_path: str, *, sha256: str = "a" * 64, size: int = 10) -> dict[str, object]:
    return {
        "source": "fixture",
        "fetched_at_utc": "2026-07-09T00:00:00Z",
        "command": "python fetch.py",
        "files": [{"path": raw_path, "sha256": sha256, "bytes": size}],
    }


def _write_rebaseline(
    root: Path,
    manifest_rel: str,
    *,
    mode: str,
    entries: object | None = None,
    superseded_by: str | None = None,
    original_manifest_sha256: str | None = None,
) -> Path:
    manifest_path = root / manifest_rel
    manifest_sha256, _ = _sha256_and_bytes(manifest_path)
    payload: dict[str, object] = {
        "schema_version": "research_swarm.manifest_rebaseline.v1",
        "rebaseline_of": manifest_rel,
        "original_manifest_sha256": original_manifest_sha256 or manifest_sha256,
        "mode": mode,
        "provenance_note": (
            "Honest re-baseline or supersession of surviving artifacts; this is not a regeneration claim."
        ),
        "rebaselined_at_utc": "2026-07-09T00:00:00Z",
    }
    if entries is not None:
        payload["entries"] = entries
    if superseded_by is not None:
        payload["superseded_by"] = superseded_by
    manifest_dir = Path(manifest_rel).parent
    sidecar_rel = manifest_dir / "rebaselines" / f"{Path(manifest_rel).name}.rebaseline.json"
    sidecar_path = write_json(root, sidecar_rel.as_posix(), payload)
    section = "raw_manifests" if manifest_rel.startswith("data/raw_manifest/") else "processed_manifests"
    register_historical_exemption(root, section=section, rel_path=manifest_rel)
    register_historical_exemption(root, section="rebaselines", rel_path=sidecar_rel.as_posix())
    return sidecar_path


class ManifestHashGateTest(unittest.TestCase):
    def _processed_fixture(self, root: Path) -> tuple[str, Path, str]:
        scaffold_runtime_repo(root)
        output_rel = "data/processed/panels/panel.csv"
        output = write_text(root, output_rel, "date,value\n2026-07-09,1\n")
        manifest_rel = "data/processed_manifest/panel_2026-07-09.json"
        write_json(root, manifest_rel, _processed_payload(output_rel, output))
        return output_rel, output, manifest_rel

    def test_processed_manifest_hashes_accept_matching_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._processed_fixture(root)
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["checked_entries"], 1)

    def test_processed_manifest_hashes_detect_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, output, _ = self._processed_fixture(root)
            output.write_text("date,value\n2026-07-09,999\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(result.ok)
            sha_failure = next(
                failure for failure in result.details["failures"] if failure["reason"] == "sha256_mismatch"
            )
            self.assertTrue(sha_failure["expected"])
            self.assertTrue(sha_failure["actual"])
            self.assertNotEqual(sha_failure["expected"], sha_failure["actual"])

    def test_recomputed_rebaseline_restores_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_rel, output, manifest_rel = self._processed_fixture(root)
            output.write_text("date,value\n2026-07-09,2\n", encoding="utf-8")
            sha256, size = _sha256_and_bytes(output)
            _write_rebaseline(
                root,
                manifest_rel,
                mode="recomputed_against_disk",
                entries=[{"path": output_rel, "sha256": sha256, "bytes": size}],
            )
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["annotations"][0]["mode"], "recomputed_against_disk")

    def test_rebaseline_rejects_original_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_rel, output, manifest_rel = self._processed_fixture(root)
            sha256, size = _sha256_and_bytes(output)
            _write_rebaseline(
                root,
                manifest_rel,
                mode="recomputed_against_disk",
                entries=[{"path": output_rel, "sha256": sha256, "bytes": size}],
                original_manifest_sha256="f" * 64,
            )
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(result.ok)
            self.assertIn("invalid_rebaseline", {item["reason"] for item in result.details["failures"]})

    def test_recomputed_rebaseline_can_itself_go_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_rel, output, manifest_rel = self._processed_fixture(root)
            sha256, size = _sha256_and_bytes(output)
            _write_rebaseline(
                root,
                manifest_rel,
                mode="recomputed_against_disk",
                entries=[{"path": output_rel, "sha256": sha256, "bytes": size}],
            )
            output.write_text("changed after rebaseline\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(result.ok)
            self.assertIn("rebaseline_stale", {item["reason"] for item in result.details["failures"]})

    def test_raw_missing_file_requires_unavailable_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            raw_rel = "data/raw/source/2026-07-09/deleted.json"
            manifest_rel = "data/raw_manifest/source_2026-07-09.json"
            write_json(root, manifest_rel, _raw_payload(raw_rel))
            with chdir(root):
                missing = quality_gates.gate_raw_manifest_hashes()
            self.assertFalse(missing.ok)
            self.assertIn("missing_file", {item["reason"] for item in missing.details["failures"]})

            _write_rebaseline(root, manifest_rel, mode="raw_evidence_unavailable", entries="all")
            with chdir(root):
                annotated = quality_gates.gate_raw_manifest_hashes()
            self.assertTrue(annotated.ok, annotated.details)
            self.assertEqual(
                annotated.details["annotations"][0]["annotated"],
                "raw_evidence_unavailable",
            )

    def test_superseded_manifest_accepts_accurate_covering_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_rel, output, old_rel = self._processed_fixture(root)
            old_payload = _processed_payload(output_rel, output)
            old_payload["outputs"][0]["sha256"] = "a" * 64
            write_json(root, old_rel, old_payload)
            new_rel = "data/processed_manifest/panel_2026-07-10.json"
            write_json(root, new_rel, _processed_payload(output_rel, output))
            _write_rebaseline(root, old_rel, mode="superseded", superseded_by=new_rel)
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["annotations"][0]["superseded_by"], new_rel)

    def test_superseded_manifest_must_cover_original_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, output, old_rel = self._processed_fixture(root)
            other_rel = "data/processed/panels/other.csv"
            other = write_text(root, other_rel, "other\n")
            new_rel = "data/processed_manifest/panel_2026-07-10.json"
            write_json(root, new_rel, _processed_payload(other_rel, other))
            _write_rebaseline(root, old_rel, mode="superseded", superseded_by=new_rel)
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(result.ok)
            self.assertIn("invalid_rebaseline", {item["reason"] for item in result.details["failures"]})
            self.assertTrue(output.exists())

    def test_superseding_manifest_hashes_must_match_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_rel, output, old_rel = self._processed_fixture(root)
            new_rel = "data/processed_manifest/panel_2026-07-10.json"
            write_json(root, new_rel, _processed_payload(output_rel, output))
            _write_rebaseline(root, old_rel, mode="superseded", superseded_by=new_rel)
            output.write_text("drifted\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(result.ok)
            self.assertIn(
                "superseding_manifest_stale",
                {item["reason"] for item in result.details["failures"]},
            )


class ManifestWriterTest(unittest.TestCase):
    def test_writer_records_v2_provenance_and_enforces_dirty_tree_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            script = write_text(root, "src/etl/build.py", "print('build')\n")
            output = write_text(root, "data/processed/panel.csv", "value\n1\n")
            write_text(root, "data/raw_manifest/input.json", "{}\n")
            init_git_fixture_repo(root)

            payload = manifest_tools.write_processed_manifest(
                repo=root,
                manifest_path="data/processed_manifest/panel.json",
                as_of_utc_date="2026-07-09",
                inputs=["data/raw_manifest/input.json"],
                script_path=script.relative_to(root),
                command="python src/etl/build.py",
                outputs=[output.relative_to(root)],
            )
            self.assertEqual(payload["schema_version"], "research_swarm.processed_manifest.v2")
            self.assertEqual(payload["transform"]["script_sha256"], _sha256_and_bytes(script)[0])
            self.assertFalse(payload["transform"]["dirty"])
            self.assertIn("dependencies", payload["environment"])

            script.write_text("print('dirty build')\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "^dirty_tree_manifest_refused:"):
                manifest_tools.write_processed_manifest(
                    repo=root,
                    manifest_path="data/processed_manifest/refused.json",
                    as_of_utc_date="2026-07-09",
                    inputs=[],
                    script_path="src/etl/build.py",
                    command="python src/etl/build.py",
                    outputs=["data/processed/panel.csv"],
                )

            dirty_payload = manifest_tools.write_processed_manifest(
                repo=root,
                manifest_path="data/processed_manifest/dirty.json",
                as_of_utc_date="2026-07-09",
                inputs=[],
                script_path="src/etl/build.py",
                command="python src/etl/build.py",
                outputs=["data/processed/panel.csv"],
                allow_dirty_with_diff=True,
            )
            self.assertTrue(dirty_payload["transform"]["dirty"])
            self.assertIn("dirty build", dirty_payload["transform"]["tree_diff"])

    def test_v2_shape_requires_script_hash_while_v1_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            manifest_rel = "data/processed_manifest/panel.json"
            v1_payload = {
                "as_of_utc_date": "2026-07-09",
                "inputs": [],
                "transform": {
                    "script_path": "src/etl/build.py",
                    "git_sha": "0" * 40,
                    "command": "python src/etl/build.py",
                },
                "outputs": [{"path": "data/processed/panel.csv", "sha256": "a" * 64, "bytes": 1}],
            }
            v2_payload = {
                **v1_payload,
                "schema_version": "research_swarm.processed_manifest.v2",
                "transform": {**v1_payload["transform"], "dirty": False},
                "environment": {"python": "3.13.5", "dependencies": {}},
            }
            write_json(root, manifest_rel, v2_payload)
            with chdir(root):
                invalid = quality_gates.gate_processed_manifest_validity()
            self.assertFalse(invalid.ok)
            self.assertTrue(any("missing_key:script_sha256" in item for item in invalid.details["failures"]))

            # a legacy-shaped manifest is a schema downgrade unless it sits on
            # the hash-pinned historical exemption list
            write_json(root, manifest_rel, v1_payload)
            with chdir(root):
                unexempted = quality_gates.gate_processed_manifest_validity()
            self.assertFalse(unexempted.ok)
            self.assertTrue(
                any("unexempted_legacy_processed_manifest" in item for item in unexempted.details["failures"]),
                unexempted.details,
            )

            register_historical_exemption(root, section="processed_manifests", rel_path=manifest_rel)
            with chdir(root):
                legacy = quality_gates.gate_processed_manifest_validity()
            self.assertTrue(legacy.ok, legacy.details)


class ValidationAndProjectionGateTest(unittest.TestCase):
    def test_validation_report_detects_drift_and_counts_legacy_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            input_rel = "data/processed/input.csv"
            input_path = write_text(root, input_rel, "value\n1\n")
            sha256, size = _sha256_and_bytes(input_path)
            write_json(
                root,
                "reports/validation/bound.json",
                {
                    "schema_version": "research_swarm.validation_report.v2",
                    "status": "pass",
                    "inputs_consumed": [{"path": input_rel, "sha256": sha256, "bytes": size}],
                },
            )
            legacy_path = write_json(root, "reports/validation/legacy.json", {"status": "pass"})
            input_path.write_text("value\n2\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_validation_report_content_binding()
            self.assertFalse(result.ok)
            self.assertEqual(result.details["legacy_reports"], 1)
            reasons = {item["reason"] for item in result.details["failures"]}
            self.assertIn("sha256_mismatch", reasons)
            # an unlisted legacy report is a schema downgrade, not a free pass
            self.assertIn("unexempted_legacy_validation_report", reasons)

            register_historical_exemption(
                root,
                section="validation_reports",
                rel_path=legacy_path.relative_to(root).as_posix(),
            )
            with chdir(root):
                exempted = quality_gates.gate_validation_report_content_binding()
            exempted_reasons = {item["reason"] for item in exempted.details["failures"]}
            self.assertNotIn("unexempted_legacy_validation_report", exempted_reasons)
            self.assertIn("sha256_mismatch", exempted_reasons)

    def test_projection_drift_gate_reports_move_then_turns_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T900", state="active")
            with chdir(root):
                drifted = quality_gates.gate_projection_drift()
            self.assertFalse(drifted.ok)
            self.assertEqual(len(drifted.details["moves"]), 1)

            destination = root / ".orchestrator" / "active" / task_path.name
            task_path.rename(destination)
            with chdir(root):
                repaired = quality_gates.gate_projection_drift()
            self.assertTrue(repaired.ok, repaired.details)

    def test_validate_str_report_writer_binds_toy_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = write_text(root, "data/toy.csv", "value\n1\n")
            report_dir = root / "reports" / "validation"
            report = validate_str_pipeline.ReportPayload(
                report_id="toy_validation",
                title="Toy validation",
                status="pass",
                mode="sample",
                as_of_utc_date=None,
                summary={"rows": 1},
                checks=[validate_str_pipeline.CheckResult(name="toy_check", status="pass")],
                provenance={"artifacts": []},
            )
            with (
                mock.patch.object(validate_str_pipeline, "REPO_ROOT", root),
                mock.patch.object(validate_str_pipeline, "REPORT_DIR", report_dir),
            ):
                bindings = validate_str_pipeline.build_inputs_consumed([input_path])
                validate_str_pipeline.write_reports([report], inputs_consumed=bindings)

            payload = json.loads((report_dir / "toy_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "research_swarm.validation_report.v2")
            self.assertEqual(payload["inputs_consumed"], bindings)
            self.assertEqual(payload["inputs_consumed"][0]["path"], "data/toy.csv")


if __name__ == "__main__":
    unittest.main()
