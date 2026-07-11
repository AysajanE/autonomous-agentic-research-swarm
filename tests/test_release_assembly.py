from __future__ import annotations

from datetime import date
import importlib.util
import json
from functools import lru_cache
import hashlib
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
    spec = importlib.util.spec_from_file_location("stage5_release_assembly_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage5_release_assembly_module"] = module
    spec.loader.exec_module(module)
    return module


release_assembly = load_release_assembly_module()


def scaffold_release_ready_repo(
    root: Path,
    *,
    include_paper: bool = False,
    include_legacy_paper: bool = False,
) -> None:
    scaffold_runtime_repo(root, mode="empirical")

    shutil.copyfile(
        _TESTS_ROOT.parent / "contracts/schemas/release_manifest_v1.yaml",
        root / "contracts/schemas/release_manifest_v1.yaml",
    )
    write_text(root, "reports/AGENTS.md", "# reports\n")
    write_text(root, "reports/status/README.md", "# status\n")
    write_text(root, "reports/status/releases/README.md", "# releases\n")
    write_text(root, "reports/paper/index.qmd", "# Paper\n")
    write_text(root, "reports/paper/references.bib", "@misc{paper}\n")
    write_text(root, "reports/paper/_quarto.yml", "project: default\n")
    write_json(root, "reports/paper/paper_values.json", {"values": {}})
    write_text(root, "data/processed/panels/daily_rollup_panel.csv", "date_utc,rollup_id\n")
    write_text(
        root,
        "data/processed/l1_rent/daily_l1_rent_decomposition.csv",
        "date_utc,l1_total_rent_eth\n",
    )

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

    if include_paper:
        write_text(
            root,
            "reports/paper/build/l2_l1_rent_working_paper.html",
            "<html>paper</html>\n",
        )
        write_text(
            root,
            "reports/paper/build/l2_l1_rent_working_paper.pdf",
            "%PDF-1.4\n",
        )
        write_json(
            root,
            "reports/paper/build/render_manifest.json",
            {
                "entrypoint": "reports/paper/index.qmd",
                "outputs": [
                    "reports/paper/build/l2_l1_rent_working_paper.html",
                    "reports/paper/build/l2_l1_rent_working_paper.pdf",
                ],
            },
        )
    if include_legacy_paper:
        write_text(root, "reports/paper/build/index.html", "<html>legacy</html>\n")

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


class ReleaseAssemblyTest(unittest.TestCase):
    def test_allow_gate_failures_cannot_bypass_failing_paper_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=False)
            manuscript = write_text(root, "reports/paper/index.qmd", "## Abstract\n\nText.\n")
            schema = root / "contracts/schemas/paper_registry_v1.json"
            schema.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "contracts/schemas/paper_registry_v1.json",
                schema,
            )
            write_json(
                root,
                "reports/paper/registry.json",
                {
                    "schema_version": "research_swarm.paper_registry.v1",
                    "entries": [
                        {
                            "registry_id": "section_abstract",
                            "kind": "section",
                            "required": True,
                            "status": "failing",
                            "artifact": {
                                "path": "reports/paper/index.qmd",
                                "sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest(),
                            },
                            "referee_report": None,
                            "reason": "awaiting referee",
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(
                SystemExit,
                "release_assembly_blocked:.*paper_registry_required_entry_failing",
            ):
                release_assembly.write_release(
                    root,
                    date(2026, 3, 31),
                    allow_gate_failures=True,
                )

    def test_allow_gate_failures_cannot_bypass_uninstantiated_program(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=False)
            task_path = next((root / ".orchestrator/done").glob("T500*.md"))
            text = task_path.read_text(encoding="utf-8")
            text = text.replace('program_id: "release_fixture"\n', "")
            text = text.replace('program_node: "release_ready"\n', "")
            task_path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(
                SystemExit,
                "release_assembly_blocked:.*program_conformance_not_instantiated_at_release",
            ):
                release_assembly.write_release(
                    root,
                    date(2026, 3, 31),
                    allow_gate_failures=True,
                )

    def test_required_release_perimeter_member_removed_from_inventory_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=False)
            payload = release_assembly.assemble_release_manifest(
                root,
                date(2026, 3, 31),
                allow_gate_failures=True,
            )
            payload["artifacts"]["release_perimeter"] = [
                item
                for item in payload["artifacts"]["release_perimeter"]
                if item["path"] != "reports/paper/paper_values.json"
            ]
            manifest_path = write_json(
                root,
                "reports/status/releases/release_2026-03-31.json",
                payload,
            )
            failures = release_assembly.validate_release_manifest(manifest_path, root)
            self.assertIn(
                "reports/status/releases/release_2026-03-31.json:"
                "release_perimeter_artifact_missing:reports/paper/paper_values.json",
                failures,
            )

    def test_missing_required_release_perimeter_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=False)
            missing = root / "reports/paper/paper_values.json"
            missing.unlink()

            with self.assertRaisesRegex(
                SystemExit,
                "release_perimeter_artifact_missing=reports/paper/paper_values.json",
            ):
                release_assembly.assemble_release_manifest(
                    root,
                    date(2026, 3, 31),
                    allow_gate_failures=True,
                )

    def test_rendered_paper_without_sources_fails_closed(self) -> None:
        # Codex F2 (BLOCKER): if the canonical rendered outputs exist but BOTH manuscript
        # source surfaces are deleted, assembly must fail rather than ship stale renders
        # whose F14 perimeter would otherwise be suppressed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=True)
            (root / "reports/paper/index.qmd").unlink()
            (root / "reports/paper/paper_values.json").unlink()

            with self.assertRaisesRegex(
                SystemExit,
                "release_rendered_paper_without_sources",
            ):
                release_assembly.assemble_release_manifest(
                    root,
                    date(2026, 3, 31),
                    allow_gate_failures=True,
                )

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

    def test_materialized_paper_without_referee_panel_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_paper=True)

            with self.assertRaisesRegex(SystemExit, "referee_release_evidence"):
                release_assembly.assemble_release_manifest(root, date(2026, 3, 31))

    def test_legacy_index_html_does_not_mark_paper_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_release_ready_repo(root, include_legacy_paper=True)

            release_assembly.write_release(root, date(2026, 3, 31))

            manifest_path = root / "reports" / "status" / "releases" / "release_2026-03-31.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["artifacts"]["paper"]["status"], "pending_stage2")
            self.assertEqual(payload["counts"]["paper_artifacts"], 0)
            self.assertEqual(payload["artifacts"]["paper"]["artifacts"], [])


if __name__ == "__main__":
    unittest.main()
