from __future__ import annotations

from datetime import date
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import sys
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
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
    spec = importlib.util.spec_from_file_location("stage5_release_assembly_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage5_release_assembly_module"] = module
    spec.loader.exec_module(module)
    return module


release_assembly = load_release_assembly_module()


def scaffold_release_ready_repo(root: Path, *, include_paper: bool = False) -> None:
    scaffold_runtime_repo(root, mode="empirical")

    write_text(
        root,
        "contracts/schemas/release_manifest_v1.yaml",
        "\n".join(
            [
                "version: 1",
                "artifact: release_manifest",
                "schema_version: research_swarm.release_manifest.v1",
                "",
            ]
        ),
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
            "as_of_utc_date": "2026-03-29",
            "inputs": ["data/raw_manifest/growthepie_2026-03-29.json"],
            "transform": {
                "script_path": "src/etl/build_panel.py",
                "git_sha": "b" * 40,
                "command": "python src/etl/build_panel.py --run-date 2026-03-29",
            },
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

    if include_paper:
        write_text(root, "reports/paper/build/index.html", "<html>paper</html>\n")

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


class ReleaseAssemblyTest(unittest.TestCase):
    def test_write_generates_canonical_release_manifest_and_catalog_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=False)

            summary = release_assembly.write_release(root, date(2026, 3, 31))

            manifest_path = root / "reports" / "status" / "releases" / "release_2026-03-31.json"
            self.assertTrue(manifest_path.exists(), summary)

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"],
                "research_swarm.release_manifest.v1",
            )
            self.assertEqual(
                payload["repo"]["release_manifest_path"],
                "reports/status/releases/release_2026-03-31.json",
            )
            self.assertEqual(
                payload["lifecycle"]["release_manifest_pattern"],
                "reports/status/releases/release_<YYYY-MM-DD>.json",
            )
            self.assertEqual(payload["artifacts"]["paper"]["status"], "pending_stage2")
            self.assertEqual(payload["counts"]["runtime_reviews"], 1)
            self.assertEqual(payload["counts"]["validation"], 1)

            catalog_text = (root / "reports" / "catalog.yaml").read_text(encoding="utf-8")
            self.assertIn(
                'manifest_path: "reports/status/releases/release_2026-03-31.json"',
                catalog_text,
            )
            self.assertIn('paper_status: "pending_stage2"', catalog_text)

            integrity = release_assembly.check_release_integrity(root)
            self.assertTrue(integrity["ok"], integrity)

            self.assertTrue(summary["ok"])
            self.assertEqual(
                summary["release_manifest"],
                "reports/status/releases/release_2026-03-31.json",
            )

    def test_write_marks_paper_present_when_stage2_build_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=True)

            release_assembly.write_release(root, date(2026, 3, 31))

            manifest_path = root / "reports" / "status" / "releases" / "release_2026-03-31.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["artifacts"]["paper"]["status"], "present")
            self.assertEqual(payload["counts"]["paper_artifacts"], 1)
            self.assertEqual(
                payload["artifacts"]["paper"]["artifacts"][0]["path"],
                "reports/paper/build/index.html",
            )

            catalog_text = (root / "reports" / "catalog.yaml").read_text(encoding="utf-8")
            self.assertIn('paper_status: "present"', catalog_text)


if __name__ == "__main__":
    unittest.main()