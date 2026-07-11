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

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from runtime_test_utils import init_git_fixture_repo, load_swarm_module, write_task  # noqa: E402
import quality_gates  # noqa: E402
import release_assembly  # noqa: E402
import spec_curve  # noqa: E402
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
        self.assertEqual(descriptor["kernel_version"], "1.1.0")
        self.assertEqual(descriptor["task_schema_version"], "research_swarm.task.v2")
        self.assertIn("run", descriptor["manifest_schema_versions"])
        self.assertEqual(descriptor["gate_registration_api_version"], "1.0.0")
        self.assertEqual(descriptor["executor_config_version"], "1.0.0")
        self.assertIn("refs/swarm/claims/<program>/T###", descriptor["reserved_claim_ref_namespaces"])
        self.assertEqual(descriptor["reserved_claim_ref_namespace_policy"]["status"], "documentation_only")
        self.assertFalse(descriptor["reserved_claim_ref_namespace_policy"]["multi_program_runtime_implemented"])

        with tempfile.TemporaryDirectory() as tmp:
            pack_root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            copied_descriptor_path = pack_root / "contracts/kernel_interface.json"
            copied_descriptor = json.loads(copied_descriptor_path.read_text(encoding="utf-8"))
            copied_descriptor["kernel_version"] = "1.0.0"
            copied_descriptor_path.write_text(
                json.dumps(copied_descriptor, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            kernel_root = Path(tmp) / "executing-kernel"
            (kernel_root / "contracts/schemas").mkdir(parents=True)
            for relpath in (
                "contracts/schemas/kernel_interface_v1.json",
                "contracts/schemas/pack_config_v1.json",
            ):
                shutil.copyfile(ROOT / relpath, kernel_root / relpath)
            executing_descriptor = dict(descriptor)
            executing_descriptor["kernel_version"] = "2.0.0"
            (kernel_root / "contracts/kernel_interface.json").write_text(
                json.dumps(executing_descriptor, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            copied_descriptor = json.loads(copied_descriptor_path.read_text(encoding="utf-8"))
            self.assertEqual(copied_descriptor["kernel_version"], "1.0.0")
            result = quality_gates.check_pack_compat(
                pack_root,
                executing_kernel_root=kernel_root,
            )
            self.assertFalse(result.ok)
            self.assertIn(
                "kernel_version_mismatch:2.0.0:requires:>=1.0.0,<2.0.0",
                result.details["failures"],
            )
            self.assertEqual(
                result.details["descriptor_source"],
                (kernel_root / "contracts/kernel_interface.json").resolve().as_posix(),
            )

    def test_pack_schema_rejects_missing_unconditional_empirical_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack_path = root / "contracts/pack.json"
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            del pack["paths"]["primary_panel_schema"]
            pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = quality_gates.check_pack_compat(root)
            self.assertFalse(result.ok)
            rendered = json.dumps(result.details["failures"], sort_keys=True)
            self.assertIn("$.paths.primary_panel_schema", rendered)
            self.assertIn("required", rendered)

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
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack_path = root / "contracts/pack.json"
            pack = json.loads(pack_path.read_text(encoding="utf-8"))
            pack["paper"]["entrypoint"] = "reports/paper/non_str.qmd"
            pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            profile = {
                "schema_version": "research_swarm.manuscript_sections.v1",
                "canonical_section_ids": ["section_overview", "section_design"],
                "section_headings": {
                    "section_overview": "## Project Overview",
                    "section_design": "## Identification Design",
                },
            }
            (root / "contracts/manuscript_sections.yaml").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manuscript_path = root / "reports/paper/non_str.qmd"
            manuscript_path.write_text(
                "# A different project\n\n## Project Overview\n\nText.\n\n## Identification Design\n\nText.\n",
                encoding="utf-8",
            )
            positive = quality_gates.check_manuscript_section_profile(root)
            self.assertTrue(positive.ok, positive.details)
            self.assertEqual(
                positive.details["matched_section_ids"],
                ["section_design", "section_overview"],
            )

            manuscript_path.write_text(
                "# A different project\n\n## Project Overview\n\nText.\n",
                encoding="utf-8",
            )
            negative = quality_gates.check_manuscript_section_profile(root)
            self.assertFalse(negative.ok)
            self.assertEqual(
                [failure["reason"] for failure in negative.details["failures"]],
                ["venue_compliance_required_section_missing"],
            )
            self.assertEqual(negative.details["failures"][0]["subject"], "section_design")

    def test_non_str_columns_and_regime_series_pass_kernel_analysis_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack = json.loads((root / "contracts/pack.json").read_text(encoding="utf-8"))
            columns = {
                "date": "day",
                "entity": "unit",
                "denominator": "gross_value",
                "numerator": "cost_value",
                "metric": "cost_share",
            }
            pack["analysis"]["panel_columns"] = columns
            pack["analysis"]["regime_series"] = ["cost_share", "cost_value", "gross_value"]
            pack["analysis"]["regime_defaults"]["series"] = "cost_share"
            (root / "contracts/pack.json").write_text(
                json.dumps(pack, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(quality_gates.check_pack_compat(root).ok)
            self.assertEqual(
                quality_gates._configured_regime_series(root),
                ("cost_share", "cost_value", "gross_value"),
            )
            config = {
                "vce": "newey_west",
                "regime_breaks": {
                    "method": "bai_perron_dynamic_programming_piecewise_constant",
                    "series": "cost_share",
                    "max_breaks": 1,
                    "min_segment": 1,
                },
                "specification_curve": {
                    "construction_variants": {"keying": ["rollup_day", "date_aggregate"], "missingness": ["drop", "zero_fill"]},
                    "analysis_variants": [{"id": "different-pack", "estimator": "regime_difference", "regime_date": "2024-01-03"}],
                    "survival_threshold": 1,
                },
            }
            body = (
                "# Non-STR analysis plan\n\n```json\n"
                + json.dumps({"statistical_reporting": config}, indent=2, sort_keys=True)
                + "\n```\n"
            )
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            (root / "docs/prereg/analysis_plan.lock.md").write_text(
                "\n".join(
                    [
                        "---",
                        "schema_version: research_swarm.prereg_lock.v1",
                        "phase: 2b",
                        "status: locked",
                        "locked_at_utc: 2026-07-11T12:00:00Z",
                        f"locked_sha256: {digest}",
                        "locked_by: Golden Owner",
                        "lock_version: 1",
                        "---",
                        "",
                    ]
                )
                + body,
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                locked_config, lock_sha, config_failures = quality_gates._active_statistical_config()
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(locked_config, config)
            self.assertEqual(lock_sha, digest)
            self.assertEqual(config_failures, [])
            panel = pd.DataFrame(
                {
                    "day": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                    "unit": ["x", "x", "x", "x"],
                    "gross_value": [10.0, 10.0, 20.0, 20.0],
                    "cost_value": [2.0, 3.0, 8.0, 10.0],
                }
            )
            artifact = spec_curve.build_spec_curve(
                panel=panel,
                claim_id="DIFFERENT-PACK-CLAIM",
                headline_estimate=0.25,
                config=config,
                prereg_lock_sha256="a" * 64,
                panel_columns=columns,
                allowed_regime_series=("cost_share", "cost_value", "gross_value"),
            )
            self.assertEqual(len(artifact["specs"]), 4)
            self.assertTrue(all(item["n"] == 4 for item in artifact["specs"]))

    def test_scaffold_assertion_cannot_rescue_instantiated_failing_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            claims_path = root / "contracts/claims.yaml"
            claims = json.loads(claims_path.read_text(encoding="utf-8"))
            claims["claims"] = [{"id": "deliberately-invalid-live-claim"}]
            claims_path.write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (root / "reports/paper/paper_values.json").write_text("{}\n", encoding="utf-8")
            gate = subprocess.run(
                ["make", "gate", f"PYTHON={sys.executable}"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            self.assertIn("[scaffold_safety] ok=False", gate.stdout)
            self.assertIn("scaffold_asserted_on_instantiated_repo", gate.stdout)
            self.assertIn("[claim_evidence_ledger] ok=False", gate.stdout)

    def test_scaffold_each_mode_gates_and_mock_executor_end_to_end(self) -> None:
        for index, mode in enumerate(swarm_init.MODES, start=1):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = swarm_init.create_scaffold(Path(tmp) / "pack", mode)
                pack = json.loads((root / "contracts/pack.json").read_text(encoding="utf-8"))
                self.assertEqual(pack["project"]["mode"], mode)
                self.assertTrue((root / "contracts/prompts/planner.md").is_file())
                self.assertTrue((root / "contracts/prompts/judge.md").is_file())
                self.assertTrue((root / "src/analysis/project_analysis.py").is_file())
                if mode == "modeling":
                    self.assertNotIn("primary_panel", pack["paths"])
                    self.assertNotIn("panel_columns", pack["analysis"])
                    self.assertIn("model_spec", pack["paths"])
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
                    gates=["make scaffold-task-gate"],
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
        kernel_paths = [ROOT / "scripts/quality_gates.py", ROOT / "scripts/swarm.py"]
        before = {path: _sha(path) for path in kernel_paths}
        with tempfile.TemporaryDirectory() as tmp:
            root = swarm_init.create_scaffold(Path(tmp) / "pack", "empirical")
            pack = json.loads((ROOT / "contracts/pack.json").read_text(encoding="utf-8"))
            (root / "contracts/pack.json").write_text(
                json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.assertFalse(pack["scaffold"])
            self.assertTrue(quality_gates.check_pack_compat(root).ok)
            self.assertFalse(quality_gates.check_scaffold_safety(root).details["scaffold_asserted"])
            self.assertEqual(
                release_assembly._canonical_paper_build_rel_paths(root),
                release_assembly._canonical_paper_build_rel_paths(ROOT),
            )
        self.assertEqual(before, {path: _sha(path) for path in kernel_paths})


if __name__ == "__main__":
    unittest.main()
