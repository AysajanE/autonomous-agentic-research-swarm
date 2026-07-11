from __future__ import annotations

from datetime import date
import importlib.util
from functools import lru_cache
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    instantiate_program_fixture,
    scaffold_runtime_repo,
    write_json,
    write_review_log,
    write_run_manifest,
    write_task,
    write_text,
)


@lru_cache(maxsize=None)
def load_release_assembly_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "release_assembly.py"
    spec = importlib.util.spec_from_file_location("stage5_release_assembly_module_integrity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage5_release_assembly_module_integrity"] = module
    spec.loader.exec_module(module)
    return module


release_assembly = load_release_assembly_module()


def scaffold_release_ready_repo(root: Path) -> None:
    scaffold_runtime_repo(root, mode="empirical")

    shutil.copyfile(
        _TESTS_ROOT.parent / "contracts/schemas/release_manifest_v1.yaml",
        root / "contracts/schemas/release_manifest_v1.yaml",
    )
    write_text(root, "reports/AGENTS.md", "# reports\n")
    write_text(root, "reports/status/README.md", "# status\n")
    write_text(root, "reports/status/releases/README.md", "# releases\n")

    write_json(
        root,
        "data/raw_manifest/growthepie_2026-03-29.json",
        {
            "source": "growthepie",
            "fetched_at_utc": "2026-03-29T00:00:00Z",
            "command": "python src/etl/fetch_growthepie.py --run-date 2026-03-29",
            "files": [
                {
                    "path": "data/raw/growthepie/2026-03-29/vendor_snapshot.csv",
                    "sha256": "a" * 64,
                    "bytes": 123,
                }
            ],
        },
    )
    write_json(
        root,
        "data/processed_manifest/daily_rollup_panel_2026-03-29.json",
        {
            "schema_version": "research_swarm.processed_manifest.v2",
            "as_of_utc_date": "2026-03-29",
            "inputs": ["data/raw_manifest/growthepie_2026-03-29.json"],
            "transform": {
                "script_path": "src/etl/build_panel.py",
                "script_sha256": "e" * 64,
                "git_sha": "b" * 40,
                "command": "python src/etl/build_panel.py --run-date 2026-03-29",
                "dirty": False,
            },
            "environment": {"python": "3.11.0", "dependencies": {"pandas": "2.2.0"}},
            "outputs": [
                {
                    "path": "data/processed/panels/daily_rollup_panel.csv",
                    "sha256": "c" * 64,
                    "bytes": 456,
                }
            ],
        },
    )

    validation_output = "reports/validation/summary.json"
    write_text(root, validation_output, "{}\n")
    write_text(root, "reports/figures/str.svg", "<svg/>\n")
    write_text(root, "reports/tables/str.csv", "date_utc,str\n2026-03-29,0.42\n")

    task_path = write_task(
        root,
        "done",
        "T500",
        workstream="W5",
        task_kind="validation",
        role="Worker",
        allowed_paths=["reports/validation/"],
        disallowed_paths=["contracts/"],
        outputs=[validation_output],
        state="done",
        slug="validation",
    )
    instantiate_program_fixture(
        root,
        task_path,
        task_kind="validation",
        role="Worker",
        workstream="W5",
    )
    run_manifest_path = write_run_manifest(
        root,
        "T500",
        task_path=task_path.relative_to(root).as_posix(),
        task_role="Worker",
        workstream="W5",
        state_before="active",
        state_after="ready_for_review",
    )
    write_review_log(
        root,
        "T500",
        task_path=task_path.relative_to(root).as_posix(),
        run_manifest_path=run_manifest_path.relative_to(root).as_posix(),
        reviewer_role="Judge",
        outcome="approve",
        state_before="ready_for_review",
        state_after="done",
    )


class ReleaseCatalogIntegrityTest(unittest.TestCase):
    def test_empty_release_namespace_can_sync_catalog_and_pass_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root)

            catalog_path = release_assembly.sync_catalog(root)
            self.assertTrue(catalog_path.exists())

            integrity = release_assembly.check_release_integrity(root)
            self.assertTrue(integrity["ok"], integrity)
            self.assertEqual(integrity["release_manifest_count"], 0)

            catalog_text = (root / "reports" / "catalog.yaml").read_text(encoding="utf-8")
            self.assertIn("releases: []", catalog_text)

    def test_check_rejects_noncanonical_release_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root)
            release_assembly.write_release(root, date(2026, 3, 31))

            canonical = root / "reports" / "status" / "releases" / "release_2026-03-31.json"
            invalid = root / "reports" / "status" / "releases" / "manual_release.json"
            shutil.copyfile(canonical, invalid)

            integrity = release_assembly.check_release_integrity(root)
            self.assertFalse(integrity["ok"])
            self.assertTrue(
                any(
                    "noncanonical_release_manifest_filename:reports/status/releases/manual_release.json"
                    in failure
                    for failure in integrity["failures"]
                ),
                integrity,
            )

    def test_check_detects_catalog_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root)
            release_assembly.write_release(root, date(2026, 3, 31))

            catalog_path = root / "reports" / "catalog.yaml"
            catalog_path.write_text(
                catalog_path.read_text(encoding="utf-8") + "# drift\n",
                encoding="utf-8",
            )

            integrity = release_assembly.check_release_integrity(root)
            self.assertFalse(integrity["ok"])
            self.assertTrue(
                any(
                    "catalog_out_of_sync:reports/catalog.yaml" in failure
                    for failure in integrity["failures"]
                ),
                integrity,
            )


if __name__ == "__main__":
    unittest.main()
