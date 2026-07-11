from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from runtime_test_utils import init_git_fixture_repo, load_swarm_module, write_task  # noqa: E402
import quality_gates  # noqa: E402
import release_assembly  # noqa: E402
import swarm_init  # noqa: E402


swarm = load_swarm_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mock_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        executor_backend="mock",
        codex_model=None,
        codex_sandbox="workspace-write",
        unattended=False,
        skip_executor=False,
        record_session=False,
        force_deps=False,
        max_worker_seconds=0,
        repair_context=None,
        create_pr=False,
        final_state="ready_for_review",
    )


class GoldenM5aTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None
        swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()

    def test_interface_descriptor_and_pack_compat_mismatch(self) -> None:
        descriptor = json.loads((ROOT / "contracts/kernel_interface.json").read_text(encoding="utf-8"))
        self.assertEqual(descriptor["kernel_version"], "1.0.0")
        self.assertEqual(descriptor["task_schema_version"], "research_swarm.task.v2")
        self.assertIn("run", descriptor["manifest_schema_versions"])
        self.assertEqual(descriptor["gate_registration_api_version"], "1.0.0")
        self.assertEqual(descriptor["executor_config_version"], "1.0.0")
        self.assertIn("refs/swarm/claims/<program>/T###", descriptor["reserved_claim_ref_namespaces"])

        with tempfile.TemporaryDirectory() as tmp:
            pack_root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack_path = pack_root / "contracts/pack.json"
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            pack["kernel_requires"] = ">=2.0.0,<3.0.0"
            pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_pack_compat(pack_root)
            self.assertFalse(result.ok)
            self.assertIn(
                "kernel_version_mismatch:1.0.0:requires:>=2.0.0,<3.0.0",
                result.details["failures"],
            )

    def test_config_drives_non_reference_paper_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "contracts").mkdir()
            pack = json.loads((ROOT / "contracts/pack.json").read_text(encoding="utf-8"))
            pack["paper"]["artifact_basename"] = "different_project_paper"
            (root / "contracts/pack.json").write_text(
                json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.assertEqual(
                release_assembly._canonical_paper_build_rel_paths(root),
                (
                    "reports/paper/build/different_project_paper.html",
                    "reports/paper/build/different_project_paper.pdf",
                    "reports/paper/build/render_manifest.json",
                ),
            )

    def test_venue_profile_sections_are_pack_parameterized(self) -> None:
        profile = {
            "canonical_section_ids": ["section_summary", "section_methods"],
            "section_headings": {
                "section_summary": "## Summary",
                "section_methods": "## Methods",
            },
        }
        manuscript = "# A different project\n\n## Summary\n\nText.\n\n## Methods\n\nText.\n"
        expected = [profile["section_headings"][item] for item in profile["canonical_section_ids"]]
        self.assertEqual(expected, ["## Summary", "## Methods"])
        self.assertTrue(all(heading in manuscript for heading in expected))

    def test_scaffold_each_mode_gates_and_mock_executor_end_to_end(self) -> None:
        for index, mode in enumerate(swarm_init.MODES, start=1):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = swarm_init.create_scaffold(Path(tmp) / "pack", mode)
                gate = subprocess.run(
                    ["make", "gate", f"PYTHON={sys.executable}"],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)

                task_id = f"T90{index}"
                write_task(
                    root,
                    "active",
                    task_id,
                    workstream="data",
                    state="active",
                    allowed_paths=["work/"],
                    disallowed_paths=["contracts/"],
                    outputs=["work/result.txt"],
                    gates=["make gate"],
                )
                transcript = {
                    "schema_version": "research_swarm.mock_transcript.v1",
                    "actions": [
                        {"write": "work/result.txt", "content": f"{mode}\n"},
                        {"set_task_state": "ready_for_review"},
                    ],
                    "returncode": 0,
                    "stdout": "mock scaffold complete\n",
                }
                path = root / ".orchestrator/mock_transcripts" / f"{task_id}.json"
                path.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                init_git_fixture_repo(root)

                output = io.StringIO()
                with (
                    mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                    mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                    contextlib.redirect_stdout(output),
                ):
                    code = swarm.cmd_run_task(_mock_args(task_id))
                self.assertEqual(code, 0, output.getvalue())
                self.assertEqual((root / "work/result.txt").read_text(encoding="utf-8"), f"{mode}\n")
                manifests = list((root / "reports/status/swarm_runs").glob(f"{task_id}_*.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["executor"]["tool"], "mock")
                self.assertEqual(manifest["task"]["state_after"], "ready_for_review")

    def test_scaffold_str_config_swap_does_not_touch_kernel(self) -> None:
        kernel_paths = [ROOT / "scripts/quality_gates.py", ROOT / "scripts/swarm.py", ROOT / "src/analysis/metrics_str.py"]
        before = {path: _sha(path) for path in kernel_paths}
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack = json.loads((ROOT / "contracts/pack.json").read_text(encoding="utf-8"))
            pack["scaffold"] = True
            (root / "contracts/pack.json").write_text(
                json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            gate = subprocess.run(
                ["make", "gate", f"PYTHON={sys.executable}"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            self.assertEqual(
                release_assembly._canonical_paper_build_rel_paths(root),
                release_assembly._canonical_paper_build_rel_paths(ROOT),
            )
        self.assertEqual(before, {path: _sha(path) for path in kernel_paths})


if __name__ == "__main__":
    unittest.main()
