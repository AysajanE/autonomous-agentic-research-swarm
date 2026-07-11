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
    instantiate_program_fixture,
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


def _run_task_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        executor_backend="codex",
        codex_model=None,
        codex_sandbox="workspace-write",
        i_accept_full_access=False,
        unattended=False,
        skip_executor=False,
        record_session=False,
        force_deps=False,
        max_worker_seconds=0,
        repair_context=None,
        create_pr=False,
        final_state="ready_for_review",
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


def _materialize_release_surface(fixture: "RefereeFixture") -> None:
    # A COMPLETE, coherent release surface: manuscript sources + processed panels (the F14
    # perimeter) AND the rendered build outputs whose render_manifest the referee-release
    # gate reads to locate the paper. Rendered outputs are never materialized WITHOUT their
    # sources — that is the invalid state the release assembler now fails closed on (Codex
    # F2) — and the render_manifest declares all six perimeter inputs with real hashes so
    # the render-perimeter verification passes.
    root = fixture.root
    program_task = write_task(
        root,
        "backlog",
        "T998",
        task_kind="analysis",
        workstream="W6",
        role="Worker",
        state="backlog",
        slug="release_program_fixture",
    )
    instantiate_program_fixture(
        root,
        program_task,
        mode=fixture.mode,
        task_kind="analysis",
        role="Worker",
        workstream="W6",
    )
    write_text(root, "reports/paper/index.qmd", "# Paper\n\nThe validated result.\n")
    write_text(root, "reports/paper/references.bib", "@misc{fixture}\n")
    write_text(root, "reports/paper/_quarto.yml", "project: default\n")
    write_json(root, "reports/paper/paper_values.json", {"values": {}})
    write_text(root, "data/processed/panels/daily_rollup_panel.csv", "date_utc,rollup_id\n")
    write_text(
        root,
        "data/processed/l1_rent/daily_l1_rent_decomposition.csv",
        "date_utc,l1_total_rent_eth\n",
    )
    write_text(root, "reports/paper/build/l2_l1_rent_working_paper.html", "<html>fixture</html>\n")
    write_text(root, "reports/paper/build/l2_l1_rent_working_paper.pdf", "%PDF-1.4\n")
    perimeter_inputs = []
    for rel in (
        "reports/paper/index.qmd",
        "reports/paper/references.bib",
        "reports/paper/_quarto.yml",
        "reports/paper/paper_values.json",
        "data/processed/panels/daily_rollup_panel.csv",
        "data/processed/l1_rent/daily_l1_rent_decomposition.csv",
    ):
        data = (root / rel).read_bytes()
        perimeter_inputs.append(
            {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )
    write_json(
        root,
        "reports/paper/build/render_manifest.json",
        {"entrypoint": fixture.output, "inputs": perimeter_inputs, "outputs": []},
    )


class RefereeFixture:
    def __init__(self, root: Path, task_id: str = "T850", *, task_kind: str = "analysis", text: str | None = None, workstream: str = "W4", complexity_tier: str = "S", mode: str = "empirical") -> None:
        self.root = root
        self.task_id = task_id
        self.task_kind = task_kind
        self.workstream = workstream
        self.mode = mode
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
        quoted_span_override: str | None = None,
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
                    {
                        "path": item["path"],
                        "quoted_span": (
                            quoted_span_override
                            if quoted_span_override is not None
                            else item["expected_quoted_span"]
                        ),
                    }
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

    def test_quote_challenge_selects_non_blank_line_and_signals_none(self) -> None:
        # A blank challenge line is echo-satisfiable; the selector must pick a
        # non-blank line, and signal challenge_line 0 when none exists (R4-B1).
        raw = b"\n\n   \nMean STR was 11.68%\n\n"
        line_no, span = swarm._artifact_quote_challenge(
            raw=raw, seed="s", task_id="T070", claim_id="C1", path="reports/tables/x.md"
        )
        self.assertNotEqual(line_no, 0)
        self.assertTrue(span.strip())
        blank_line, blank_span = swarm._artifact_quote_challenge(
            raw=b"\n   \n\n", seed="s", task_id="T070", claim_id="C1", path="p"
        )
        self.assertEqual(blank_line, 0)
        self.assertEqual(blank_span, "")

    def test_executor_written_swarm_run_forgery_hard_fails_before_kernel_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            task_id = "T849"
            forged_rel = f"reports/status/swarm_runs/{task_id}_evil.json"
            write_task(
                root,
                "active",
                task_id,
                state="active",
                allowed_paths=["README.md", forged_rel],
                outputs=["README.md"],
                gates=["python scripts/noop_gate.py"],
            )
            init_git_fixture_repo(root)

            def forged_executor(**_: object) -> object:
                write_json(root, forged_rel, {"forged": True})
                return swarm.ExecutorOutcome(
                    returncode=0,
                    stdout="forged control-plane file",
                    wall_clock_seconds=0.01,
                    usage=None,
                    transcript_path=None,
                )

            stdout = io.StringIO()
            with (
                _fixture_root(root),
                mock.patch.object(swarm, "_codex_exec_cmd", return_value=["fake-executor"]),
                mock.patch.object(swarm, "_execute_task", side_effect=forged_executor),
                contextlib.redirect_stdout(stdout),
            ):
                code = swarm.cmd_run_task(_run_task_args(task_id))
            self.assertEqual(code, 1, stdout.getvalue())
            result = json.loads(stdout.getvalue())
            self.assertIn(
                f"executor_wrote_control_plane:{forged_rel}",
                result["blocked_reasons"],
            )
            self.assertNotEqual(
                subprocess.run(
                    ["git", "ls-files", "--error-unmatch", "--", forged_rel],
                    cwd=root,
                    check=False,
                    capture_output=True,
                ).returncode,
                0,
            )
            manifests = [
                path
                for path in (root / "reports/status/swarm_runs").glob(f"{task_id}_*.json")
                if path.name != f"{task_id}_evil.json"
            ]
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["result"]["status"], "blocked")
            events = (root / "reports/status/events/events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event":"executor_wrote_control_plane"', events)
            allowed, reason = swarm._path_is_allowed(
                path=forged_rel,
                allowed_paths=[forged_rel],
                disallowed_paths=[],
                task_file_path=f".orchestrator/active/{task_id}_task.md",
                task_id=task_id,
            )
            self.assertFalse(allowed)
            self.assertEqual(reason, "swarm_runs_kernel_only")

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
            fixture = RefereeFixture(Path(tmp) / "repo", complexity_tier="M")
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

    def test_opened_artifact_hash_echo_without_correct_quote_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock(quoted_span_override="not the challenged disk line")
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertTrue(
                any(
                    item.startswith("referee_opened_artifact_quote_mismatch")
                    for item in output["failures"]
                )
            )

    def test_honest_read_quote_passes_and_kernel_supplies_disk_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo")
            fixture.write_mock()
            self.assertEqual(fixture.run()[0], 0)
            _, report = fixture.latest_report()
            opened = report["opened_artifacts"][0]
            self.assertEqual(opened["quoted_span"], "primary evidence")
            self.assertEqual(
                opened["sha256"],
                hashlib.sha256(
                    (fixture.root / "reports/validation/referee_sample.txt").read_bytes()
                ).hexdigest(),
            )

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

    def test_stale_optional_report_does_not_block_out_of_scope_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(Path(tmp) / "repo", complexity_tier="S")
            fixture.write_mock()
            self.assertEqual(fixture.run()[0], 0)
            manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
            manifest["generated_at_utc"] = "2026-07-10T01:00:00Z"
            fixture.manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            contract = swarm.load_framework_contract(fixture.root)
            tasks, quarantined = swarm.load_tasks_quarantined(contract)
            task = swarm._resolve_runtime_task(tasks, quarantined, fixture.task_id)
            self.assertEqual(
                swarm._referee_review_failures(
                    repo=fixture.root,
                    task=task,
                    run_manifest_path=fixture.manifest,
                ),
                [],
            )

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
            code, output = fixture.run()
            self.assertEqual(code, 1)
            self.assertEqual(len(output["revision_tasks"]), 1)
            revision = fixture.root / output["revision_tasks"][0]
            diagnostics = lint_task_files(
                sorted((fixture.root / ".orchestrator").glob("*/T*.md")),
                repo_root=fixture.root,
                network_workstreams=("W1", "W2", "W3"),
                v1_exemptions={},
            )
            self.assertFalse(
                [item.as_dict() for item in diagnostics if item.task in revision.name],
                diagnostics,
            )
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

    def test_live_metadata_predictions_require_kernel_calibration_run_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            init_git_fixture_repo(root)
            source = json.loads(
                (root / "tests/gold_set/mock_referee.json").read_text(encoding="utf-8")
            )
            predictions = []
            for item in source["predictions"]:
                case_id = item["case_id"]
                predictions.append(
                    {
                        **item,
                        "session_id": f"live-session-{case_id}",
                        "argv_sha256": hashlib.sha256(
                            f"fixture-referee --case {case_id}".encode("utf-8")
                        ).hexdigest(),
                    }
                )
            live_predictions = write_json(
                root,
                "tests/gold_set/live_predictions.json",
                {
                    "schema_version": "research_swarm.referee_gold_predictions.v1",
                    "backend": "claude",
                    "family": "claude",
                    "model": "fixture-referee",
                    "cli_version": "fixture-cli-1",
                    "profile": "read-only",
                    "prompt_path": "docs/prompts/referee.md",
                    "predictions": predictions,
                },
            )
            output = root / "reports/status/referee_calibration.json"
            report = calibrate_referee.calibrate(
                calibration_path=root / "contracts/rubrics/calibration.yaml",
                gold_dir=root / "tests/gold_set",
                mock_path=live_predictions,
                output_path=output,
            )
            run_path = root / report["calibration_run_path"]
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(
                run_record["schema_version"],
                "research_swarm.referee_calibration_run.v1",
            )
            self.assertEqual(len(run_record["case_invocations"]), len(predictions))
            run_path.unlink()
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("calibration_run_record" in item for item in result.details["failures"]),
                result.details,
            )

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

    def test_same_commit_calibration_bar_and_grading_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            _configure_mock_panel(root)
            shutil.copytree(REPO_ROOT / "tests/gold_set", root / "tests/gold_set", dirs_exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "swarm-bot"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "swarm-bot@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "bar and grading together"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            _write_bound_calibration(root)
            with chdir(root):
                result = quality_gates.gate_referee_calibration()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("committed_after_grading" in item for item in result.details["failures"]),
                result.details,
            )

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

    def test_release_blocks_ready_for_review_minor_cannot_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
            )
            fixture.write_mock(overrides={"WRITING_CLARITY": "cannot_verify"})
            self.assertEqual(fixture.run()[0], 1)
            _materialize_release_surface(fixture)
            with chdir(fixture.root):
                result = quality_gates.gate_referee_release_evidence()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("referee_release_cannot_verify" in item for item in result.details["failures"]),
                result.details,
            )

    def test_release_waiver_with_zero_calibrated_votes_still_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
            )
            waiver = {"human_id": "owner-1", "reason": "documented third-family outage"}
            with _fixture_root(fixture.root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    swarm.cmd_referee_waiver(
                        argparse.Namespace(
                            task=fixture.task_id,
                            human_id=waiver["human_id"],
                            reason=waiver["reason"],
                        )
                    ),
                    0,
                )
            _materialize_release_surface(fixture)
            with chdir(fixture.root):
                result = quality_gates.gate_referee_release_evidence()
            self.assertFalse(result.ok)
            self.assertIn(
                f"{fixture.task_id}:referee_release_panel_quorum:0<1",
                result.details["failures"],
            )

    def test_referee_waiver_command_refuses_task_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = RefereeFixture(
                Path(tmp) / "repo",
                task_kind="writing",
                workstream="W7",
            )
            subprocess.run(
                ["git", "switch", "-c", f"{fixture.task_id}_task"],
                cwd=fixture.root,
                check=True,
                capture_output=True,
            )
            with _fixture_root(fixture.root), self.assertRaisesRegex(
                SystemExit,
                "referee_waiver_requires_integration_branch",
            ):
                swarm.cmd_referee_waiver(
                    argparse.Namespace(
                        task=fixture.task_id,
                        human_id="owner-1",
                        # a task branch cannot authorize even by passing its own
                        # name as base_branch — the trusted base comes from git
                        reason="task branch must not authorize",
                        base_branch=f"{fixture.task_id}_task",
                    )
                )

    def test_only_referee_waiver_command_authorizes_one_family(self) -> None:
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
            self.assertIsNone(
                swarm._referee_owner_waiver(
                    fixture.root,
                    task_id=fixture.task_id,
                    run_manifest_sha256=manifest_sha,
                )
            )
            waiver_stdout = io.StringIO()
            with _fixture_root(fixture.root), contextlib.redirect_stdout(waiver_stdout):
                waiver_code = swarm.cmd_referee_waiver(
                    argparse.Namespace(
                        task=fixture.task_id,
                        human_id=waiver["human_id"],
                        reason=waiver["reason"],
                    )
                )
            self.assertEqual(waiver_code, 0, waiver_stdout.getvalue())
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
            _materialize_release_surface(fixture)
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
