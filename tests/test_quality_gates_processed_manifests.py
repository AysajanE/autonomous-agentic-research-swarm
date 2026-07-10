import sys
from pathlib import Path
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    chdir,
    load_quality_gates_module,
    scaffold_runtime_repo,
    write_json,
    write_run_manifest,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()


class ProcessedManifestGateTest(unittest.TestCase):
    def test_ready_for_review_processed_output_requires_processed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            output_path = "data/processed/panels/daily_rollup_panel.csv"
            write_text(root, output_path, "date_utc,rollup_id,l2_fees_eth,rent_paid_eth\n")

            task_path = write_task(
                root,
                "ready_for_review",
                "T300",
                workstream="W2",
                task_kind="etl",
                role="Worker",
                allowed_paths=["data/processed/"],
                disallowed_paths=["contracts/"],
                outputs=[output_path],
                state="ready_for_review",
                slug="panel",
            )
            write_run_manifest(
                root,
                "T300",
                task_path=task_path.relative_to(root).as_posix(),
                task_role="Worker",
                workstream="W2",
                state_after="ready_for_review",
            )

            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()

            self.assertFalse(result.ok)
            failures = result.details.get("failures") or []
            self.assertTrue(any("missing_declared_processed_manifest_output" in failure for failure in failures))

    def test_valid_processed_manifest_allows_review_bundle_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            output_path = "data/processed/panels/daily_rollup_panel.csv"
            manifest_spec = "data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json"
            manifest_path = "data/processed_manifest/daily_rollup_panel_2026-03-29.json"

            write_text(root, output_path, "date_utc,rollup_id,l2_fees_eth,rent_paid_eth\n")
            write_json(
                root,
                manifest_path,
                {
                    "schema_version": "research_swarm.processed_manifest.v2",
                    "as_of_utc_date": "2026-03-29",
                    "inputs": ["data/raw_manifest/growthepie_2026-03-29.json"],
                    "transform": {
                        "script_path": "src/etl/build_l1_rent_panel.py",
                        "script_sha256": "e" * 64,
                        "git_sha": "0123456789abcdef0123456789abcdef01234567",
                        "command": "python src/etl/build_l1_rent_panel.py --run-date 2026-03-29",
                        "dirty": False,
                    },
                    "environment": {"python": "3.11.0", "dependencies": {"pandas": "2.2.0"}},
                    "outputs": [
                        {
                            "path": output_path,
                            "sha256": "a" * 64,
                            "bytes": 48,
                        }
                    ],
                },
            )

            task_path = write_task(
                root,
                "ready_for_review",
                "T301",
                workstream="W2",
                task_kind="etl",
                role="Worker",
                allowed_paths=["data/processed/", "data/processed_manifest/"],
                disallowed_paths=["contracts/"],
                outputs=[output_path, manifest_spec],
                state="ready_for_review",
                slug="panel",
            )
            write_run_manifest(
                root,
                "T301",
                task_path=task_path.relative_to(root).as_posix(),
                task_role="Worker",
                workstream="W2",
                state_after="ready_for_review",
            )

            with chdir(root):
                manifest_result = quality_gates.gate_processed_manifest_validity()
                bundle_result = quality_gates.gate_review_bundle_integrity()

            self.assertTrue(manifest_result.ok, manifest_result.details)
            self.assertTrue(bundle_result.ok, bundle_result.details)

    def test_manifest_backed_etl_review_bundle_allows_missing_local_raw_and_processed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            code_path = "src/etl/growthepie_fetch.py"
            raw_output_spec = "data/raw/growthepie/YYYY-MM-DD/..."
            raw_manifest_spec = "data/raw_manifest/growthepie_YYYY-MM-DD.json"
            raw_manifest_path = "data/raw_manifest/growthepie_2026-03-31.json"
            processed_output_path = "data/processed/growthepie/vendor_daily_rollup_panel.csv"
            processed_manifest_spec = "data/processed_manifest/vendor_daily_rollup_panel_YYYY-MM-DD.json"
            processed_manifest_path = "data/processed_manifest/vendor_daily_rollup_panel_2026-03-31.json"
            sample_path = "data/samples/growthepie/vendor_daily_rollup_panel_sample.csv"

            write_text(root, code_path, "print('growthepie')\n")
            write_text(root, sample_path, "date_utc,rollup_id,l2_fees_eth,rent_paid_eth\n")
            write_json(
                root,
                raw_manifest_path,
                {
                    "schema_version": "research_swarm.processed_manifest.v2",
                    "as_of_utc_date": "2026-03-31",
                    "command": "python src/etl/growthepie_fetch.py --run-date 2026-03-31",
                    "files": [
                        {
                            "path": "data/raw/growthepie/2026-03-31/master.json",
                            "sha256": "a" * 64,
                            "bytes": 128,
                        }
                    ],
                },
            )
            write_json(
                root,
                processed_manifest_path,
                {
                    "schema_version": "research_swarm.processed_manifest.v2",
                    "as_of_utc_date": "2026-03-31",
                    "inputs": [raw_manifest_path],
                    "transform": {
                        "script_path": code_path,
                        "script_sha256": "e" * 64,
                        "git_sha": "0" * 40,
                        "command": "python src/etl/growthepie_fetch.py --run-date 2026-03-31",
                        "dirty": False,
                    },
                    "environment": {"python": "3.11.0", "dependencies": {"pandas": "2.2.0"}},
                    "outputs": [
                        {
                            "path": processed_output_path,
                            "sha256": "b" * 64,
                            "bytes": 1024,
                        },
                        {
                            "path": sample_path,
                            "sha256": "c" * 64,
                            "bytes": 64,
                        },
                    ],
                },
            )

            task_path = write_task(
                root,
                "ready_for_review",
                "T302",
                workstream="W1",
                task_kind="etl",
                role="Worker",
                allowed_paths=[
                    "src/etl/",
                    "data/raw/",
                    "data/raw_manifest/",
                    "data/processed/",
                    "data/processed_manifest/",
                    "data/samples/",
                ],
                disallowed_paths=["contracts/"],
                outputs=[
                    code_path,
                    raw_output_spec,
                    raw_manifest_spec,
                    processed_output_path,
                    processed_manifest_spec,
                    sample_path,
                ],
                state="ready_for_review",
                slug="growthepie",
            )
            write_run_manifest(
                root,
                "T302",
                task_path=task_path.relative_to(root).as_posix(),
                task_role="Worker",
                workstream="W1",
                state_after="ready_for_review",
            )

            with chdir(root):
                bundle_result = quality_gates.gate_review_bundle_integrity()

            self.assertTrue(bundle_result.ok, bundle_result.details)

    def test_manifest_backed_etl_review_bundle_still_requires_tracked_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            code_path = "src/etl/growthepie_fetch.py"
            raw_manifest_spec = "data/raw_manifest/growthepie_YYYY-MM-DD.json"
            raw_manifest_path = "data/raw_manifest/growthepie_2026-03-31.json"
            processed_output_path = "data/processed/growthepie/vendor_daily_rollup_panel.csv"
            processed_manifest_spec = "data/processed_manifest/vendor_daily_rollup_panel_YYYY-MM-DD.json"
            processed_manifest_path = "data/processed_manifest/vendor_daily_rollup_panel_2026-03-31.json"
            sample_path = "data/samples/growthepie/vendor_daily_rollup_panel_sample.csv"

            write_text(root, code_path, "print('growthepie')\n")
            write_json(root, raw_manifest_path, {"as_of_utc_date": "2026-03-31", "files": []})
            write_json(
                root,
                processed_manifest_path,
                {
                    "schema_version": "research_swarm.processed_manifest.v2",
                    "as_of_utc_date": "2026-03-31",
                    "inputs": [raw_manifest_path],
                    "transform": {
                        "script_path": code_path,
                        "script_sha256": "e" * 64,
                        "git_sha": "0" * 40,
                        "command": "python src/etl/growthepie_fetch.py --run-date 2026-03-31",
                        "dirty": False,
                    },
                    "environment": {"python": "3.11.0", "dependencies": {"pandas": "2.2.0"}},
                    "outputs": [
                        {
                            "path": processed_output_path,
                            "sha256": "b" * 64,
                            "bytes": 1024,
                        }
                    ],
                },
            )

            task_path = write_task(
                root,
                "ready_for_review",
                "T303",
                workstream="W1",
                task_kind="etl",
                role="Worker",
                allowed_paths=[
                    "src/etl/",
                    "data/raw_manifest/",
                    "data/processed/",
                    "data/processed_manifest/",
                    "data/samples/",
                ],
                disallowed_paths=["contracts/"],
                outputs=[
                    code_path,
                    "data/raw/growthepie/YYYY-MM-DD/...",
                    raw_manifest_spec,
                    processed_output_path,
                    processed_manifest_spec,
                    sample_path,
                ],
                state="ready_for_review",
                slug="growthepie-sample",
            )
            write_run_manifest(
                root,
                "T303",
                task_path=task_path.relative_to(root).as_posix(),
                task_role="Worker",
                workstream="W1",
                state_after="ready_for_review",
            )

            with chdir(root):
                bundle_result = quality_gates.gate_review_bundle_integrity()

            self.assertFalse(bundle_result.ok)
            failures = bundle_result.details.get("failures") or []
            self.assertTrue(any(sample_path in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
