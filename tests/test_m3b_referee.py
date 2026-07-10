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


def _bind_gold_transcript(root: Path, source_name: str = "mock_referee.json") -> Path:
    transcript = json.loads((root / f"tests/gold_set/{source_name}").read_text(encoding="utf-8"))
    transcript.update(
        {
            "backend": "mock",
            "family": "mock",
            "model": "mock-referee-v1",
            "cli_version": "mock-1",
            "profile": "read-only",
            "prompt_path": "docs/prompts/referee.md",
        }
    )
    return write_json(root, f"tests/gold_set/bound_{source_name}", transcript)


def _configure_mock_panel(root: Path) -> None:
    path = root / "contracts/framework.json"
    framework = json.loads(path.read_text(encoding="utf-8"))
    framework["executors"]["referee_panel"] = [
        {
            "backend": "mock",
            "family": "mock",
            "command": "mock",
            "model": "mock-referee-v1",
            "cli_version": "mock-1",
            "profile": "read-only",
            "prompt_path": "docs/prompts/referee.md",
            "tools": ["Read", "Glob", "Grep"],
        }
    ]
    path.write_text(json.dumps(framework, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_bound_calibration(root: Path) -> dict[str, object]:
    return calibrate_referee.calibrate(
        calibration_path=root / "contracts/rubrics/calibration.yaml",
        gold_dir=root / "tests/gold_set",
        mock_path=_bind_gold_transcript(root),
        output_path=root / "reports/status/referee_calibration.json",
    )


class RefereeFixture:
    def __init__(self, root: Path, task_id: str = "T850", *, task_kind: str = "analysis", text: str | None = None, workstream: str = "W4", complexity_tier: str = "S", mode: str = "empirical") -> None:
        self.root = root
        self.task_id = task_id
        scaffold_runtime_repo(root, mode=mode)
        _configure_mock_panel(root)
        shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
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
            complexity_tier=complexity_tier,
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
                        "task_id": task_id,
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
        _write_bound_calibration(root)

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
        family: str = "mock",
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
            _configure_mock_panel(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            init_git_fixture_repo(root)
            output = root / "reports/status/referee_calibration.json"
            calibrated = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=_bind_gold_transcript(root),
                output_path=output,
            )
            self.assertTrue(calibrated["calibrated"])
            with chdir(root):
                self.assertTrue(quality_gates.gate_referee_calibration().ok)
            uncalibrated = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=_bind_gold_transcript(root, "mock_referee_disagree.json"),
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
            self.assertEqual(set(votes), {"mock"})
            failures = swarm._referee_review_failures(
                repo=fixture.root,
                task=task,
                run_manifest_path=fixture.manifest,
            )
            self.assertIn("referee_manuscript_panel_family_quorum:1<2", failures)

    def test_fail_closed_scope_requires_m_analysis_but_skips_s_and_etl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m_fixture = RefereeFixture(Path(tmp) / "m", complexity_tier="M")
            s_fixture = RefereeFixture(Path(tmp) / "s", complexity_tier="S")
            etl_fixture = RefereeFixture(
                Path(tmp) / "etl", task_kind="etl", complexity_tier="M"
            )
            for fixture, expected in (
                (m_fixture, "referee_required_missing"),
                (s_fixture, None),
                (etl_fixture, None),
            ):
                contract = swarm.load_framework_contract(fixture.root)
                tasks, quarantined = swarm.load_tasks_quarantined(contract)
                task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
                failures = swarm._referee_review_failures(
                    repo=fixture.root,
                    task=task,
                    run_manifest_path=fixture.manifest,
                )
                if expected is None:
                    self.assertEqual(failures, [])
                else:
                    self.assertIn(expected, failures)

    def test_referee_backend_outage_is_a_distinct_hard_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo", complexity_tier="M")
            manifest_sha = hashlib.sha256(fixture.manifest.read_bytes()).hexdigest()
            swarm_events = __import__("swarm_events")
            swarm_events.append_event(
                fixture.root,
                {
                    "event": "referee_invocation_failed",
                    "task_id": fixture.task_id,
                    "run_manifest_sha256": manifest_sha,
                    "reason": "fixture backend outage",
                },
                actor_session="referee-kernel",
            )
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            failures = swarm._referee_review_failures(
                repo=fixture.root, task=task, run_manifest_path=fixture.manifest
            )
            self.assertEqual(failures, ["referee_backend_unavailable"])

    def test_hand_placed_report_without_invocation_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock()
            self.assertEqual(fixture.run()[0], 0)
            (fixture.root / "reports/status/events/events.jsonl").unlink()
            with chdir(fixture.root):
                result = quality_gates.gate_referee_report_validity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("referee_report_unjournaled" in item for item in result.details["failures"])
            )

    def test_minor_cannot_verify_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo", complexity_tier="M")
            fixture.write_mock(overrides={"ANALYSIS_LIMITATIONS": "cannot_verify"})
            self.assertEqual(fixture.run()[0], 1)
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            failures = swarm._referee_review_failures(
                repo=fixture.root, task=task, run_manifest_path=fixture.manifest
            )
            self.assertIn("referee_cannot_verify:ANALYSIS_LIMITATIONS", failures)

    def test_major_not_supported_blocks_analysis_and_ignores_reported_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo", complexity_tier="M")
            fixture.write_mock(overrides={"ANALYSIS_PROTOCOL_CONFORMANCE": "not_supported"})
            self.assertEqual(fixture.run()[0], 1)
            report_path, report = fixture.latest_report()
            finding = next(
                item
                for item in report["verdicts"]
                if item.get("check_id") == "ANALYSIS_PROTOCOL_CONFORMANCE"
            )
            finding["severity"] = "minor"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            failures = swarm._referee_review_failures(
                repo=fixture.root, task=task, run_manifest_path=fixture.manifest
            )
            self.assertIn("referee_not_supported:ANALYSIS_PROTOCOL_CONFORMANCE", failures)

    def test_tampered_sampled_artifact_blocks_disk_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock()
            write_text(fixture.root, "reports/validation/referee_sample.txt", "tampered\n")
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertTrue(
                any(item.startswith("referee_sampled_artifact_tampered") for item in output["failures"])
            )

    def test_mock_backend_calibration_does_not_authorize_live_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            init_git_fixture_repo(root)
            calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=root / "tests/gold_set/mock_referee.json",
                output_path=root / "reports/status/referee_calibration.json",
            )
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(any("backend_binding_not_deployed" in item for item in result.details["failures"]))

    def test_calibration_gate_recomputes_fabricated_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            _configure_mock_panel(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            init_git_fixture_repo(root)
            output = root / "reports/status/referee_calibration.json"
            report = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=_bind_gold_transcript(root, "mock_referee_disagree.json"),
                output_path=output,
            )
            report.update({"agreement": 1.0, "position_flip_rate": 0.0, "calibrated": True})
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(any("recompute_mismatch" in item for item in result.details["failures"]))

    def test_post_grading_calibration_bar_change_is_rejected_by_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            _configure_mock_panel(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            init_git_fixture_repo(root)
            bar_path = root / "contracts/rubrics/calibration.yaml"
            bar = json.loads(bar_path.read_text(encoding="utf-8"))
            bar["agreement_floor"] = 0.1
            bar["committed_at_utc"] = "2020-01-01T00:00:00Z"
            bar_path.write_text(json.dumps(bar, indent=2) + "\n", encoding="utf-8")
            subprocess.run(["git", "add", str(bar_path.relative_to(root))], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "lower bar after grading"], cwd=root, check=True, capture_output=True)
            _write_bound_calibration(root)
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(any("committed_after_grading" in item for item in result.details["failures"]))

    def test_lock_bound_repair_inherits_lock_and_is_referee_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="model",
                complexity_tier="M",
                mode="hybrid",
            )
            fixture.write_mock(overrides={"MODEL_SPEC_CONFORMANCE": "not_supported"})
            code, output = fixture.run()
            self.assertEqual(code, 1)
            revision_path = fixture.root / output["revision_tasks"][0]
            revision_frontmatter = swarm._parse_task_frontmatter(
                revision_path.read_text(encoding="utf-8")
            )
            self.assertIn("lock_b", revision_frontmatter["required_prereg_locks"])
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            repair_id = revision_frontmatter["task_id"]
            repair = swarm._resolve_runtime_task(tasks, quarantined, repair_id)
            self.assertTrue(swarm._repair_is_referee_reviewable(repair))
            self.assertIn("lock_b", swarm._effective_required_active_locks(repair, contract))
            self.assertTrue((fixture.root / ".swarm/plan_approval_pending.json").is_file())

    def test_missing_claim_ledger_is_diagnostic_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            (fixture.root / "contracts/claims.yaml").unlink()
            fixture.write_mock()
            code, _ = fixture.run()
            self.assertEqual(code, 0)
            events = (fixture.root / "reports/status/events/events.jsonl").read_text(encoding="utf-8")
            self.assertIn("referee_claim_ledger_unavailable", events)

    def test_journaled_owner_waiver_allows_one_family_and_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
            )
            fixture.write_mock()
            self.assertEqual(fixture.run()[0], 0)
            waiver = {"human_id": "owner-1", "reason": "documented third-family outage"}
            framework_path = fixture.root / "contracts/framework.json"
            framework = json.loads(framework_path.read_text(encoding="utf-8"))
            framework["referee_panel"]["owner_waiver"] = waiver
            framework_path.write_text(json.dumps(framework, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_sha = hashlib.sha256(fixture.manifest.read_bytes()).hexdigest()
            waiver_sha = hashlib.sha256(
                json.dumps(waiver, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            swarm_events = __import__("swarm_events")
            swarm_events.append_event(
                fixture.root,
                {
                    "event": "referee_owner_waiver",
                    "task_id": fixture.task_id,
                    "run_manifest_sha256": manifest_sha,
                    **waiver,
                    "waiver_sha256": waiver_sha,
                },
                actor_session="owner-session",
            )
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            failures = swarm._referee_review_failures(
                repo=fixture.root, task=task, run_manifest_path=fixture.manifest
            )
            self.assertFalse(any("panel_family_quorum" in item for item in failures), failures)
            recorded = swarm._referee_owner_waiver(
                fixture.root,
                task_id=fixture.task_id,
                run_manifest_sha256=manifest_sha,
            )
            self.assertEqual(recorded["waiver_sha256"], waiver_sha)
            write_text(
                fixture.root,
                "reports/paper/build/l2_l1_rent_working_paper.html",
                "<html>fixture</html>\n",
            )
            write_text(
                fixture.root,
                "reports/paper/build/l2_l1_rent_working_paper.pdf",
                "%PDF-1.4\n",
            )
            write_json(
                fixture.root,
                "reports/paper/build/render_manifest.json",
                {"entrypoint": fixture.output, "outputs": []},
            )
            with chdir(fixture.root):
                release_result = quality_gates.gate_referee_release_evidence()
            self.assertTrue(release_result.ok, release_result.details)
            self.assertEqual(
                release_result.details["evidence"][0]["owner_waiver"]["waiver_sha256"],
                waiver_sha,
            )
            import test_release_assembly

            manifest = test_release_assembly.release_assembly.assemble_release_manifest(
                fixture.root,
                __import__("datetime").date(2026, 7, 10),
                allow_gate_failures=True,
            )
            self.assertEqual(
                manifest["referee_evidence"]["evidence"][0]["owner_waiver"]["waiver_sha256"],
                waiver_sha,
            )
            report_artifact = manifest["referee_evidence"]["evidence"][0]["reports"][0]
            self.assertEqual(
                report_artifact["sha256"],
                hashlib.sha256((fixture.root / report_artifact["path"]).read_bytes()).hexdigest(),
            )
            calibration_artifact = manifest["referee_evidence"]["calibration"]
            self.assertEqual(
                calibration_artifact["sha256"],
                hashlib.sha256((fixture.root / calibration_artifact["path"]).read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
