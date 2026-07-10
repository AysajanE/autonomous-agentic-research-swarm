from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    REPO_ROOT,
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_run_manifest,
    write_task,
    write_text,
)


swarm = load_swarm_module()
quality_gates = load_quality_gates_module()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import calibrate_referee
from swarm_taskfile import lint_task_files


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        chdir(root),
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


def _args(task_id: str, family: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        task=task_id,
        referee_backend="mock",
        referee_family=family,
        remote="origin",
        base_branch="main",
        timeout_seconds=30,
    )


def _set_executor_tool(path: Path, tool: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["executor"]["tool"] = tool
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RefereeFixture:
    def __init__(self, root: Path, task_id: str = "T850", *, task_kind: str = "analysis", text: str | None = None, workstream: str = "W4") -> None:
        self.root = root
        self.task_id = task_id
        scaffold_runtime_repo(root)
        suffix = ".qmd" if task_kind == "writing" else ".md"
        self.output = f"reports/paper/{task_id}{suffix}" if task_kind == "writing" else f"reports/tables/{task_id}{suffix}"
        write_text(root, self.output, text or "The declared result is supported by the primary artifact.\n")
        self.task_path = write_task(
            root,
            "ready_for_review",
            task_id,
            schema="v2",
            state="ready_for_review",
            task_kind=task_kind,
            workstream=workstream,
            allowed_paths=[self.output],
            disallowed_paths=[],
            outputs=[self.output],
            gates=["python scripts/noop_gate.py"],
        )
        evidence = write_text(root, "reports/validation/referee_sample.txt", "primary evidence\n")
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        write_json(
            root,
            "contracts/claims.yaml",
            {
                "schema_version": "research_swarm.claims.v1",
                "description": "referee fixture ledger",
                "claims": [
                    {
                        "claim_id": "FIXTURE-CLAIM",
                        "citation_key": "fixture",
                        "type": "descriptive",
                        "statement": "The fixture's registered result.",
                        "manuscript_numeric_literals": ["69.14%", "11.68%"],
                        "supporting_artifacts": [
                            {"path": "reports/validation/referee_sample.txt", "sha256": digest}
                        ],
                    }
                ],
            },
        )
        init_git_fixture_repo(root)
        self.manifest = write_run_manifest(
            root,
            task_id,
            task_path=self.task_path.relative_to(root).as_posix(),
            workstream=workstream,
            state_before="active",
            state_after="ready_for_review",
        )
        _set_executor_tool(self.manifest, "codex")
        calibration_sha = hashlib.sha256((root / "contracts/rubrics/calibration.yaml").read_bytes()).hexdigest()
        write_json(
            root,
            "reports/status/referee_calibration.json",
            {
                "schema_version": "research_swarm.referee_calibration.v1",
                "calibration_sha256": calibration_sha,
                "calibrated": True,
            },
        )

    def required_ids(self) -> dict[str, str]:
        contract = swarm.load_framework_contract(self.root)
        tasks, quarantined = swarm.load_tasks_quarantined(contract)
        task = swarm._resolve_runtime_task(tasks, quarantined, self.task_id)
        frontmatter = swarm._task_frontmatter(task)
        rubric, _ = swarm._load_referee_rubric(self.root, task.task_kind)
        rubrics = [rubric]
        if task.task_kind == "writing":
            manuscript, _ = swarm._load_referee_rubric(self.root, task.task_kind, manuscript=True)
            rubrics.append(manuscript)
        assertions = swarm._assertion_candidates(self.root, task) if task.task_kind == "writing" else []
        required = swarm._referee_required_verdicts(
            task=task,
            frontmatter=frontmatter,
            rubrics=rubrics,
            assertions=assertions,
        )
        return {identifier: str(spec["identifier_key"]) for identifier, spec in required.items()}

    def write_mock(
        self,
        *,
        family: str = "claude",
        overrides: dict[str, str] | None = None,
        open_sample: bool = True,
    ) -> Path:
        overrides = overrides or {}
        verdicts: list[dict[str, object]] = []
        for identifier, key in self.required_ids().items():
            verdict = overrides.get(identifier, "supported")
            verdicts.append(
                {
                    key: identifier,
                    "verdict": verdict,
                    "evidence_pointer": f"{self.output}:1",
                    "note": f"fixture verdict for {identifier}",
                }
            )
        sampled = swarm._kernel_sampled_artifacts(self.root, self.task_id)
        return write_json(
            self.root,
            f".orchestrator/mock_referee/{self.task_id}.json",
            {
                "schema_version": "research_swarm.mock_referee.v1",
                "referee_family": family,
                "returncode": 0,
                "stdout": "mock referee complete",
                "verdicts": verdicts,
                "opened_artifacts": [
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in sampled
                ] if open_sample else [],
                "overall": "supported",
            },
        )

    def run(self) -> tuple[int, dict[str, object]]:
        stdout = io.StringIO()
        with _fixture_root(self.root), contextlib.redirect_stdout(stdout):
            code = swarm.cmd_referee_task(_args(self.task_id))
        output = json.loads(stdout.getvalue())
        return code, output

    def latest_report(self) -> tuple[Path, dict[str, object]]:
        paths = sorted((self.root / "reports/status/referee_reports").glob(f"{self.task_id}_*.json"))
        if not paths:
            raise AssertionError("missing referee report")
        return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))


class M3bRefereeTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None

    def test_same_family_referee_is_hard_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock(family="codex")
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertEqual(output["error"], "referee_family_of_author")
            self.assertFalse((fixture.root / "reports/status/referee_reports").exists())

    def test_cannot_verify_blocks_done_and_escalates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock(overrides={"ANALYSIS_STATISTICAL_VALIDITY": "cannot_verify"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            self.assertEqual(report["overall"], "cannot_verify")
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            failures = swarm._referee_review_failures(
                repo=fixture.root, task=task, run_manifest_path=fixture.manifest
            )
            self.assertIn("referee_cannot_verify:ANALYSIS_STATISTICAL_VALIDITY", failures)

    def test_not_supported_major_spawns_lint_passing_revision_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock(overrides={"ANALYSIS_PROTOCOL_CONFORMANCE": "not_supported"})
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertEqual(len(output["revision_tasks"]), 1)
            revision = fixture.root / output["revision_tasks"][0]
            self.assertTrue(revision.is_file())
            diagnostics = lint_task_files(
                sorted((fixture.root / ".orchestrator").glob("*/T*.md")),
                repo_root=fixture.root,
                network_workstreams=("W1", "W2", "W3"),
                v1_exemptions={},
            )
            self.assertFalse([item.as_dict() for item in diagnostics if item.task in revision.name], diagnostics)
            text = revision.read_text(encoding="utf-8")
            self.assertIn('task_kind: "repair"', text)
            self.assertIn(f'  - "{fixture.output}"', text)

    def test_kernel_sample_must_be_opened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock(open_sample=False)
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertTrue(any(item.startswith("referee_did_not_open_sampled") for item in output["failures"]))
            _, report = fixture.latest_report()
            self.assertFalse(report["valid"])

    def test_supported_kernel_report_passes_report_validity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock()
            code, _ = fixture.run()
            self.assertEqual(code, 0)
            with chdir(fixture.root):
                result = quality_gates.gate_referee_report_validity()
            self.assertTrue(result.ok, result.details)

    def test_supervised_output_root_keeps_referee_writes_out_of_artifact_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "worktree")
            fixture.write_mock()
            control_root = Path(tmp) / "control"
            scaffold_runtime_repo(control_root)
            stdout = io.StringIO()
            with (
                _fixture_root(fixture.root),
                mock.patch.dict(
                    os.environ,
                    {"SWARM_REFEREE_OUTPUT_ROOT": str(control_root)},
                    clear=False,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = swarm.cmd_referee_task(_args(fixture.task_id))
            self.assertEqual(code, 0, stdout.getvalue())
            self.assertEqual(
                len(list((control_root / "reports/status/referee_reports").glob("*.json"))),
                1,
            )
            self.assertFalse((fixture.root / "reports/status/referee_reports").exists())

    def test_unregistered_numeric_assertion_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
                text="The otherwise undocumented program returned 999.9 ETH.\n",
            )
            assertion_id = next(identifier for identifier in fixture.required_ids() if identifier.startswith("ASSERTION-"))
            fixture.write_mock(overrides={assertion_id: "not_supported"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            finding = next(item for item in report["verdicts"] if item.get("check_id") == assertion_id)
            self.assertEqual(finding["verdict"], "not_supported")

    def test_semantic_same_unit_value_swap_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
                text="Mean ecosystem STR was 11.68% before Dencun and 11.68% after Dencun.\n",
            )
            fixture.write_mock(overrides={"MANUSCRIPT_SEMANTIC_VALUE_ROLE": "not_supported"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            finding = next(item for item in report["verdicts"] if item.get("check_id") == "MANUSCRIPT_SEMANTIC_VALUE_ROLE")
            self.assertEqual(finding["verdict"], "not_supported")

    def test_paraphrased_causal_claim_is_referee_adjudicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
                text="Teams entering the program earlier went on to outperform later entrants under otherwise similar baselines.\n",
            )
            fixture.write_mock(overrides={"WRITING_CLAIM_TYPING": "not_supported"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            finding = next(item for item in report["verdicts"] if item.get("check_id") == "WRITING_CLAIM_TYPING")
            self.assertEqual(finding["verdict"], "not_supported")

    def test_h055_replica_wrong_but_coherent_artifact_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = (REPO_ROOT / "tests/gold_set/artifacts/wrong_h055_fabricated_artifact.md").read_text(encoding="utf-8")
            fixture = RefereeFixture(Path(tmp) / "repo", text=artifact)
            fixture.write_mock(overrides={"ANALYSIS_PROTOCOL_CONFORMANCE": "not_supported"})
            code, _ = fixture.run()
            self.assertEqual(code, 1)
            _, report = fixture.latest_report()
            self.assertEqual(report["overall"], "not_supported")

    def test_calibration_gate_passes_calibrated_and_blocks_disagreeing_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            output = root / "reports/status/referee_calibration.json"
            calibrated = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=root / "tests/gold_set/mock_referee.json",
                output_path=output,
            )
            self.assertTrue(calibrated["calibrated"])
            with chdir(root):
                self.assertTrue(quality_gates.gate_referee_calibration().ok)
            uncalibrated = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=root / "tests/gold_set/mock_referee_disagree.json",
                output_path=output,
            )
            self.assertFalse(uncalibrated["calibrated"])
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(any("calibrated_false" in item for item in result.details["failures"]))

    def test_calibration_commit_guard_refuses_after_grading_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = write_json(
                root,
                "tests/gold_set/verdict_key.json",
                {"graded_at_utc": "2026-07-10T00:00:00Z", "cases": [{"case_id": "G001"}]},
            )
            with self.assertRaisesRegex(ValueError, "grading_already_started"):
                calibrate_referee.commit_bar(
                    path=root / "contracts/rubrics/calibration.yaml",
                    output_path=root / "reports/status/referee_calibration.json",
                    gold_key_path=gold,
                    agreement_floor=0.8,
                    position_flip_ceiling=0.1,
                    committed_by="grader",
                )

    def test_claude_referee_argv_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            argv = swarm._claude_referee_argv(root)
            self.assertEqual(argv[argv.index("--tools") + 1], "Read,Glob,Grep")
            self.assertNotIn("--allowedTools", argv)
            self.assertIn("--strict-mcp-config", argv)

    def test_manuscript_panel_counts_one_vote_per_family_and_requires_two_non_authors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
                text="The registered result is reported without a new numeric assertion.\n",
            )
            fixture.write_mock()
            code, _ = fixture.run()
            self.assertEqual(code, 0)
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            votes = swarm._referee_family_votes(
                fixture.root,
                fixture.task_id,
                run_manifest_relpath=fixture.manifest.relative_to(fixture.root).as_posix(),
            )
            self.assertEqual(set(votes), {"claude"})
            failures = swarm._referee_review_failures(
                repo=fixture.root,
                task=task,
                run_manifest_path=fixture.manifest,
            )
            self.assertIn("referee_manuscript_panel_family_quorum:1<2", failures)


if __name__ == "__main__":
    unittest.main()
