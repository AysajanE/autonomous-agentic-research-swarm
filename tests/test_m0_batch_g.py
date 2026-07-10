from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime_test_utils import (
    SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1,
    chdir,
    load_quality_gates_module,
    scaffold_runtime_repo,
    write_json,
    write_run_manifest,
    write_task,
)


quality_gates = load_quality_gates_module()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exemptions(root: Path, run_entries: list[dict[str, str]]) -> Path:
    return write_json(
        root,
        "contracts/historical_exemptions.json",
        {
            "schema_version": "research_swarm.historical_exemptions.v1",
            "created_at_utc": "2026-07-09T00:00:00Z",
            "rationale": "test fixture",
            "run_manifests": run_entries,
            "review_logs": [],
        },
    )


def _write_annotation(root: Path, manifest_path: Path, *, sha256: str, provenance_class: str = "backfill") -> Path:
    rel = manifest_path.relative_to(root).as_posix()
    return write_json(
        root,
        f"reports/status/swarm_runs/annotations/{manifest_path.name}.provenance.json",
        {
            "schema_version": "research_swarm.provenance_annotation.v1",
            "annotates": rel,
            "annotates_sha256": sha256,
            "provenance_class": provenance_class,
            "rationale": "test fixture",
            "annotated_at_utc": "2026-07-09T00:00:00Z",
        },
    )


class HistoricalExemptionsGateTest(unittest.TestCase):
    def _fixture_with_v1_manifest(self, tmp: str) -> tuple[Path, Path]:
        root = Path(tmp)
        scaffold_runtime_repo(root)
        task_path = write_task(root, "backlog", "T500")
        manifest_path = write_run_manifest(
            root,
            "T500",
            task_path=task_path.relative_to(root).as_posix(),
            schema_version=SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1,
        )
        return root, manifest_path

    def test_no_exemptions_file_and_no_v1_artifacts_is_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T501")
            write_run_manifest(root, "T501", task_path=task_path.relative_to(root).as_posix())
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["v1_run_manifests"], 0)

    def test_unexempted_v1_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertFalse(result.ok)
            rel = manifest_path.relative_to(root).as_posix()
            self.assertIn(f"unexempted_v1_artifact:{rel}", result.details["failures"])

    def test_exempted_and_annotated_v1_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            digest = _sha256_file(manifest_path)
            rel = manifest_path.relative_to(root).as_posix()
            _write_exemptions(root, [{"path": rel, "sha256": digest, "schema_version": "v1"}])
            _write_annotation(root, manifest_path, sha256=digest)
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["annotations_checked"], 1)

    def test_exemption_list_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            rel = manifest_path.relative_to(root).as_posix()
            _write_exemptions(root, [{"path": rel, "sha256": "0" * 64, "schema_version": "v1"}])
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertFalse(result.ok)
            self.assertIn(f"exemption_list_drift:sha256_mismatch:{rel}", result.details["failures"])
            # a drifted entry exempts nothing
            self.assertIn(f"unexempted_v1_artifact:{rel}", result.details["failures"])

    def test_missing_annotation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            digest = _sha256_file(manifest_path)
            rel = manifest_path.relative_to(root).as_posix()
            _write_exemptions(root, [{"path": rel, "sha256": digest, "schema_version": "v1"}])
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertFalse(result.ok)
            self.assertIn(f"provenance_annotation_missing:{rel}", result.details["failures"])

    def test_annotation_sha_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            digest = _sha256_file(manifest_path)
            rel = manifest_path.relative_to(root).as_posix()
            _write_exemptions(root, [{"path": rel, "sha256": digest, "schema_version": "v1"}])
            _write_annotation(root, manifest_path, sha256="f" * 64)
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertFalse(result.ok)
            self.assertIn(
                f"provenance_annotation_invalid:{rel}:annotates_sha256_mismatch",
                result.details["failures"],
            )

    def test_invalid_provenance_class_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path = self._fixture_with_v1_manifest(tmp)
            digest = _sha256_file(manifest_path)
            rel = manifest_path.relative_to(root).as_posix()
            _write_exemptions(root, [{"path": rel, "sha256": digest, "schema_version": "v1"}])
            _write_annotation(root, manifest_path, sha256=digest, provenance_class="invented")
            with chdir(root):
                result = quality_gates.gate_historical_exemptions()
            self.assertFalse(result.ok)
            self.assertIn(
                f"provenance_annotation_invalid:{rel}:provenance_class",
                result.details["failures"],
            )


class RealRepoRemediationTest(unittest.TestCase):
    """The actual battle-test record must satisfy the remediation invariants."""

    REPO = Path(__file__).resolve().parents[1]

    def test_release_amendment_present_with_class_counts(self) -> None:
        release = json.loads(
            (self.REPO / "reports/status/releases/release_2026-04-11.json").read_text(encoding="utf-8")
        )
        notes = [n for n in release["notes"] if isinstance(n, dict) and n.get("type") == "raw_evidence_unavailable"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(
            notes[0]["provenance_class_run_counts"],
            {"executor_run": 18, "manual_operator": 9, "backfill": 5},
        )

    def test_every_historical_run_manifest_is_annotated_and_untouched(self) -> None:
        exemptions = json.loads(
            (self.REPO / "contracts/historical_exemptions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(exemptions["run_manifests"]), 32)
        self.assertEqual(len(exemptions["review_logs"]), 23)
        for entry in exemptions["run_manifests"]:
            manifest_path = self.REPO / entry["path"]
            self.assertEqual(_sha256_file(manifest_path), entry["sha256"], entry["path"])
            annotation_path = (
                manifest_path.parent / "annotations" / f"{manifest_path.name}.provenance.json"
            )
            self.assertTrue(annotation_path.is_file(), annotation_path)

    def test_rebaselines_never_claim_regeneration(self) -> None:
        for rebaseline_dir in (
            self.REPO / "data/processed_manifest/rebaselines",
            self.REPO / "data/raw_manifest/rebaselines",
        ):
            for sidecar in sorted(rebaseline_dir.glob("*.rebaseline.json")):
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                self.assertIn(
                    payload["mode"],
                    {"superseded", "raw_evidence_unavailable"},
                    sidecar.name,
                )
                note = payload["provenance_note"].lower()
                self.assertTrue(note.strip(), sidecar.name)
                # the note must be an honesty statement, not a regeneration claim
                self.assertTrue(
                    "not claim regeneration" in note or "cannot be recovered" in note
                    or "no such claim" in note or "nothing here claims" in note,
                    (sidecar.name, payload["provenance_note"]),
                )


if __name__ == "__main__":
    unittest.main()
