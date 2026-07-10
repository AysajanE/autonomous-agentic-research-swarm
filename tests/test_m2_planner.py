from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    chdir,
    init_git_fixture_repo,
    load_swarm_module,
    scaffold_runtime_repo,
    write_framework_json,
    write_json,
    write_run_manifest,
    write_task,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import swarm_events


swarm = load_swarm_module()
GREEN_GATE = 'python scripts/noop_gate.py'


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        chdir(root),
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


def _planner_args(*, backend: str = "mock", no_push: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        planner_backend=backend,
        remote="origin",
        base_branch="main",
        unattended=False,
        no_push=no_push,
    )


def _tick_args() -> argparse.Namespace:
    return argparse.Namespace(
        planner="heuristic",
        runner="local",
        tmux_session="swarm",
        max_workers=10,
        worktree_parent=None,
        remote="origin",
        base_branch="main",
        executor_backend="mock",
        codex_model=None,
        codex_sandbox="workspace-write",
        i_accept_full_access=False,
        unattended=False,
        max_worker_seconds=0,
        create_pr=False,
        final_state="ready_for_review",
        dry_run=True,
    )


def _run_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        executor_backend="mock",
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


def _insert_frontmatter(text: str, line: str) -> str:
    lines = text.splitlines(keepends=True)
    end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    lines.insert(end, line.rstrip("\n") + "\n")
    return "".join(lines)


def _with_triage(text: str, note: str = "Planner confirmed bounded scope") -> str:
    escaped = json.dumps(note, ensure_ascii=False)
    return _insert_frontmatter(
        text,
        f"triage: {{status: confirmed, by: planner, note: {escaped}}}",
    )


def _with_recon(text: str) -> str:
    section = "\n".join(
        [
            "## Reconnaissance",
            "",
            "- Scope: implement the declared output only.",
            "- Risks: preserve the pinned task contract.",
            "- Decomposition: the task remains bounded.",
            "",
        ]
    )
    return text.replace("## Status\n", section + "\n## Status\n", 1)


def _render_task(
    root: Path,
    task_id: str,
    *,
    complexity_tier: str = "S",
    task_kind: str = "analysis",
    checkpoint_contract: str = "none",
    recon_required: bool = False,
    inputs: list[dict[str, object]] | None = None,
    outputs: list[str] | None = None,
    budgets: dict[str, object] | None = None,
    triaged: bool = True,
) -> str:
    path = write_task(
        root,
        "backlog",
        task_id,
        schema="v2",
        complexity_tier=complexity_tier,
        task_kind=task_kind,
        checkpoint_contract=checkpoint_contract,
        recon_required=recon_required,
        inputs=inputs,
        outputs=outputs,
        budgets=budgets,
        gates=[GREEN_GATE],
    )
    text = path.read_text(encoding="utf-8")
    path.unlink()
    return _with_triage(text) if triaged else text


def _write_mock_planner(
    root: Path,
    mode: str,
    trigger_id: str,
    proposals: list[dict[str, object]],
    *,
    returncode: int = 0,
) -> Path:
    return write_json(
        root,
        f".orchestrator/mock_planner/{mode}_{trigger_id}.json",
        {
            "schema_version": "research_swarm.mock_planner.v1",
            "proposals": proposals,
            "returncode": returncode,
        },
    )


def _write_mock_executor(root: Path, task_id: str) -> Path:
    return write_json(
        root,
        f".orchestrator/mock_transcripts/{task_id}.json",
        {
            "schema_version": "research_swarm.mock_transcript.v1",
            "actions": [
                {"write": "src/result.txt", "content": "result\n"},
                {"set_task_state": "ready_for_review"},
            ],
            "returncode": 0,
            "stdout": "mock executor complete\n",
        },
    )


def _events(root: Path, event_name: str) -> list[dict[str, object]]:
    events, malformed = swarm_events.read_events(root)
    if malformed:
        raise AssertionError(f"malformed fixture events: {malformed}")
    return [event for event in events if event.get("event") == event_name]


class M2PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None
        swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()

    def _root(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        return root

    def test_bounded_write_refuses_whole_batch_for_each_outside_surface(self) -> None:
        for forbidden in ("src/x.py", ".orchestrator/active/T901_bad.md"):
            with self.subTest(forbidden=forbidden), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                valid = _render_task(root, "T900")
                invalid = _render_task(root, "T901")
                init_git_fixture_repo(root)

                with _fixture_root(root):
                    result = swarm._apply_planner_proposals(
                        mode="launch",
                        proposals=[
                            {
                                "action": "create_task",
                                "path": ".orchestrator/backlog/T900_valid.md",
                                "content": valid,
                            },
                            {"action": "create_task", "path": forbidden, "content": invalid},
                        ],
                        repo=root,
                        args=_planner_args(),
                    )

                self.assertTrue(result["batch_refused"])
                self.assertFalse((root / ".orchestrator/backlog/T900_valid.md").exists())
                events = _events(root, "planner_write_refused")
                self.assertEqual(len(events), 1)
                self.assertTrue(events[0]["escalation"])
                self.assertIn("planner_path_outside_authority", events[0]["violations"][0]["reason"])

    def test_valid_backlog_proposal_is_committed_linted_and_claimable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            content = _render_task(root, "T902")
            init_git_fixture_repo(root)

            with _fixture_root(root):
                result = swarm._apply_planner_proposals(
                    mode="launch",
                    proposals=[
                        {
                            "action": "create_task",
                            "path": ".orchestrator/backlog/T902_valid.md",
                            "content": content,
                        }
                    ],
                    repo=root,
                    args=_planner_args(),
                )
                contract = swarm.load_framework_contract(root)
                tasks = swarm.load_tasks(contract)
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)

            self.assertTrue(result["committed"])
            self.assertEqual(result["outcomes"][0]["status"], "applied")
            self.assertEqual(_git(root, "log", "-1", "--format=%s").stdout.strip(), "planner: launch")
            self.assertEqual([task.task_id for task in ready], ["T902"])

    def test_backlog_symlink_cannot_expand_planner_write_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            target = root / "src/x.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_render_task(root, "T899", triaged=False), encoding="utf-8")
            link = root / ".orchestrator/backlog/T899_escape.md"
            link.symlink_to("../../src/x.py")
            init_git_fixture_repo(root)
            original = target.read_text(encoding="utf-8")

            with _fixture_root(root):
                result = swarm._apply_planner_proposals(
                    mode="triage",
                    proposals=[
                        {
                            "action": "triage_confirm",
                            "task_id": "T899",
                            "note": "This must not follow the symlink.",
                        }
                    ],
                    repo=root,
                    args=_planner_args(),
                )

            self.assertTrue(result["batch_refused"])
            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertEqual(len(_events(root, "planner_write_refused")), 1)

    def test_lint_failure_refuses_only_bad_proposal_and_keeps_valid_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            invalid = _render_task(root, "T903").replace(
                'task_kind: "analysis"', 'task_kind: "not_a_kind"', 1
            )
            valid = _render_task(root, "T904")
            init_git_fixture_repo(root)

            with _fixture_root(root):
                result = swarm._apply_planner_proposals(
                    mode="launch",
                    proposals=[
                        {
                            "action": "create_task",
                            "path": ".orchestrator/backlog/T903_invalid.md",
                            "content": invalid,
                        },
                        {
                            "action": "create_task",
                            "path": ".orchestrator/backlog/T904_valid.md",
                            "content": valid,
                        },
                    ],
                    repo=root,
                    args=_planner_args(),
                )

            self.assertEqual(
                [outcome["status"] for outcome in result["outcomes"]],
                ["lint_failed", "applied"],
            )
            diagnostic = result["outcomes"][0]["diagnostics"][0]
            self.assertEqual(diagnostic["reason"], "invalid_task_kind")
            self.assertFalse((root / ".orchestrator/backlog/T903_invalid.md").exists())
            self.assertTrue((root / ".orchestrator/backlog/T904_valid.md").is_file())
            self.assertEqual(len(_events(root, "planner_proposal_lint_failed")), 1)

    def test_plan_program_requires_approval_and_tick_unblocks_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            first = _render_task(root, "T905")
            second = _render_task(root, "T906")
            planned_workstreams = "# Workstreams\n\nPlanner launch pass.\n"
            _write_mock_planner(
                root,
                "launch",
                "launch",
                [
                    {
                        "action": "create_task",
                        "path": ".orchestrator/backlog/T905_first.md",
                        "content": first,
                    },
                    {
                        "action": "create_task",
                        "path": ".orchestrator/backlog/T906_second.md",
                        "content": second,
                    },
                    {"action": "update_workstreams", "content": planned_workstreams},
                ],
            )
            init_git_fixture_repo(root)
            stdout = io.StringIO()

            with _fixture_root(root), contextlib.redirect_stdout(stdout):
                self.assertEqual(swarm.cmd_plan_program(_planner_args(no_push=False)), 0)
                pending = root / ".swarm/plan_approval_pending.json"
                self.assertTrue(pending.is_file())
                tick_before = io.StringIO()
                with contextlib.redirect_stdout(tick_before):
                    self.assertEqual(swarm.cmd_tick(_tick_args()), 0)
                before = json.loads(tick_before.getvalue())
                self.assertEqual(before["selected"], [])
                self.assertEqual(
                    {item["task_id"] for item in before["skipped"]}, {"T905", "T906"}
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        swarm.cmd_approve_plan(argparse.Namespace(approved_by="fixture-owner")),
                        0,
                    )
                self.assertFalse(pending.exists())
                tick_after = io.StringIO()
                with contextlib.redirect_stdout(tick_after):
                    self.assertEqual(swarm.cmd_tick(_tick_args()), 0)
                after = json.loads(tick_after.getvalue())

            launch = json.loads(stdout.getvalue())
            self.assertTrue(launch["approval_pending"])
            self.assertEqual(launch["planner"]["proposal_count"], 3)
            self.assertEqual(set(after["ready"]), {"T905", "T906"})
            self.assertTrue(after["selected"])
            self.assertEqual(after["skipped"], [])
            self.assertEqual((root / ".orchestrator/workstreams.md").read_text(), planned_workstreams)
            self.assertEqual(len(_events(root, "plan_awaiting_human_approval")), 1)
            self.assertEqual(len(_events(root, "plan_unapproved")), 2)
            self.assertEqual(_events(root, "plan_approved")[0]["approved_by"], "fixture-owner")

    def test_triage_confirm_makes_l_task_claimable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task_path = write_task(
                root,
                "backlog",
                "T907",
                schema="v2",
                task_kind="analysis",
                complexity_tier="L",
                checkpoint_contract="progress_file",
                recon_required=True,
                gates=[GREEN_GATE],
            )
            _write_mock_planner(
                root,
                "triage",
                "T907",
                [
                    {
                        "action": "triage_confirm",
                        "task_id": "T907",
                        "note": "Scope is bounded, with construction excluded.",
                    }
                ],
            )
            init_git_fixture_repo(root)

            triage_args = argparse.Namespace(**vars(_planner_args(no_push=False)), task="T907")
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                tasks = swarm.load_tasks(contract)
                self.assertEqual(swarm.ready_backlog_tasks(tasks, set(), contract), [])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(swarm.cmd_triage(triage_args), 0)

                tasks = swarm.load_tasks(contract)
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)

            self.assertEqual([task.task_id for task in ready], ["T907"])
            frontmatter = swarm._parse_task_frontmatter(task_path.read_text(encoding="utf-8"))
            self.assertEqual(frontmatter["triage"]["status"], "confirmed")
            self.assertEqual(frontmatter["triage"]["by"], "planner")

    def test_triage_split_replaces_parent_and_journals_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            parent = write_task(
                root,
                "backlog",
                "T908",
                schema="v2",
                task_kind="analysis",
                complexity_tier="L",
                checkpoint_contract="progress_file",
                recon_required=True,
                gates=[GREEN_GATE],
            )
            child_one = _render_task(root, "T909")
            child_two = _render_task(root, "T910")
            _write_mock_planner(
                root,
                "triage",
                "T908",
                [
                    {
                        "action": "split_task",
                        "task_id": "T908",
                        "into": [
                            {
                                "path": ".orchestrator/backlog/T909_discovery.md",
                                "content": child_one,
                            },
                            {
                                "path": ".orchestrator/backlog/T910_construction.md",
                                "content": child_two,
                            },
                        ],
                    }
                ],
            )
            init_git_fixture_repo(root)

            triage_args = argparse.Namespace(**vars(_planner_args(no_push=False)), task="T908")
            with _fixture_root(root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(swarm.cmd_triage(triage_args), 0)

            self.assertFalse(parent.exists())
            self.assertTrue((root / ".orchestrator/backlog/T909_discovery.md").is_file())
            self.assertTrue((root / ".orchestrator/backlog/T910_construction.md").is_file())
            split = _events(root, "task_split")
            self.assertEqual(split[0]["parent"], "T908")
            self.assertEqual(split[0]["children"], ["T909", "T910"])

    def test_heuristic_multi_input_etl_and_many_outputs_require_triage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            anchor = write_task(
                root,
                "backlog",
                "T911",
                schema="v2",
                task_kind="analysis",
                gates=[GREEN_GATE],
            )
            anchor.write_text(_with_triage(anchor.read_text()), encoding="utf-8")
            multi = write_task(
                root,
                "backlog",
                "T912",
                schema="v2",
                task_kind="etl",
                inputs=[
                    {"path": "first.json", "sha256": "1" * 64},
                    {"path": "second.json", "sha256": "2" * 64},
                ],
                gates=[GREEN_GATE],
            )
            many = write_task(
                root,
                "backlog",
                "T913",
                schema="v2",
                task_kind="analysis",
                outputs=["src/a.py", "src/b.py", "src/c.py"],
                gates=[GREEN_GATE],
            )
            contract = swarm.load_framework_contract(root)
            tasks = swarm.load_tasks(contract)

            with mock.patch.object(swarm, "_record_swarm_event"):
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)

            self.assertEqual([task.task_id for task in ready], ["T911"])
            self.assertIn("etl_multi_input", swarm._task_triage_reasons(tasks["T912"], tasks))
            self.assertIn("more_than_two_outputs", swarm._task_triage_reasons(tasks["T913"], tasks))
            self.assertTrue(multi.is_file() and many.is_file())

    def test_failed_run_threshold_dispatches_replan_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            parent = write_task(
                root,
                "backlog",
                "T914",
                schema="v2",
                task_kind="analysis",
                state="blocked",
                gates=[GREEN_GATE],
            )
            child_one = _render_task(root, "T915")
            child_two = _render_task(root, "T916")
            init_git_fixture_repo(root)
            first_manifest = write_run_manifest(
                root,
                "T914",
                task_path=parent.relative_to(root).as_posix(),
                state_before="active",
                state_after="blocked",
                result_status="blocked",
            )
            payload = json.loads(first_manifest.read_text(encoding="utf-8"))
            payload["run_id"] = "T914_20260329T000001Z"
            write_json(root, "reports/status/swarm_runs/T914_20260329T000001Z.json", payload)
            _write_mock_planner(
                root,
                "replan",
                "T914_failed_runs",
                [
                    {
                        "action": "split_task",
                        "task_id": "T914",
                        "into": [
                            {
                                "path": ".orchestrator/backlog/T915_recon.md",
                                "content": child_one,
                            },
                            {
                                "path": ".orchestrator/backlog/T916_build.md",
                                "content": child_two,
                            },
                        ],
                    }
                ],
            )

            with _fixture_root(root):
                result = swarm._step_plan(_planner_args())

            self.assertEqual(result["dispatched"][0]["trigger"], "failed_runs")
            self.assertFalse(parent.exists())
            self.assertTrue((root / ".orchestrator/backlog/T915_recon.md").is_file())
            self.assertEqual(_events(root, "replan_dispatched")[0]["task_id"], "T914")
            self.assertEqual(_events(root, "task_split")[0]["children"], ["T915", "T916"])

    def test_planner_marker_dispatches_replan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task = write_task(
                root,
                "backlog",
                "T917",
                schema="v2",
                task_kind="analysis",
                gates=[GREEN_GATE],
            )
            task.write_text(task.read_text() + "- @planner split this task\n", encoding="utf-8")
            created = _render_task(root, "T918")
            init_git_fixture_repo(root)
            _write_mock_planner(
                root,
                "replan",
                "T917_planner_marker",
                [
                    {
                        "action": "create_task",
                        "path": ".orchestrator/backlog/T918_followup.md",
                        "content": created,
                    }
                ],
            )

            with _fixture_root(root):
                result = swarm._step_plan(_planner_args())

            self.assertEqual(result["dispatched"][0]["trigger"], "planner_marker")
            self.assertTrue((root / ".orchestrator/backlog/T918_followup.md").is_file())

    def test_timebox_exceeded_dispatches_replan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task = write_task(
                root,
                "backlog",
                "T919",
                schema="v2",
                task_kind="analysis",
                budgets={"max_wall_clock": "1h", "max_tokens": 1000, "max_cost_usd": 1},
                gates=[GREEN_GATE],
            )
            created = _render_task(root, "T920")
            init_git_fixture_repo(root)
            write_run_manifest(
                root,
                "T919",
                task_path=task.relative_to(root).as_posix(),
                usage={"wall_clock_seconds": 3601},
            )
            _write_mock_planner(
                root,
                "replan",
                "T919_timebox_exceeded",
                [
                    {
                        "action": "create_task",
                        "path": ".orchestrator/backlog/T920_followup.md",
                        "content": created,
                    }
                ],
            )

            with _fixture_root(root):
                result = swarm._step_plan(_planner_args())

            self.assertEqual(result["dispatched"][0]["trigger"], "timebox_exceeded")
            self.assertTrue((root / ".orchestrator/backlog/T920_followup.md").is_file())

    def test_structured_failure_context_stays_valid_and_within_two_kibibytes(self) -> None:
        manifest = {
            "result": {
                "blocked_reasons": ["blocked-" + ("é" * 300) for _ in range(20)]
            },
            "gates": [
                {
                    "command": "python huge_gate.py " + ("x" * 500),
                    "returncode": 1,
                    "timed_out": False,
                    "constraint_violation": None,
                    "output_head": "é" * 2000,
                    "output_tail": "z" * 2000,
                }
                for _ in range(20)
            ],
        }

        payload = swarm._failure_context_from_manifest(manifest)
        rendered = swarm._repair_context_from_manifest(manifest)

        self.assertLessEqual(len(rendered.encode("utf-8")), 2048)
        self.assertEqual(json.loads(rendered), payload)
        self.assertIsInstance(payload["blocked_reasons"], list)
        self.assertIsInstance(payload["gate_diagnostics"], list)

    def test_hypothesis_retirement_is_refused_and_escalated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            parent = write_task(
                root,
                "backlog",
                "T921",
                schema="v2",
                task_kind="analysis",
                gates=[GREEN_GATE],
            )
            text = _insert_frontmatter(parent.read_text(), 'hypothesis_id: "H-001"')
            parent.write_text(text + "- @planner retire this task\n", encoding="utf-8")
            child = _render_task(root, "T922")
            init_git_fixture_repo(root)
            original = parent.read_text(encoding="utf-8")
            _write_mock_planner(
                root,
                "replan",
                "T921_planner_marker",
                [
                    {
                        "action": "split_task",
                        "task_id": "T921",
                        "into": [
                            {
                                "path": ".orchestrator/backlog/T922_replacement.md",
                                "content": child,
                            }
                        ],
                    }
                ],
            )

            with _fixture_root(root):
                result = swarm._step_plan(_planner_args())

            application = result["dispatched"][0]["application"]
            self.assertEqual(
                application["outcomes"][0]["reason"],
                "hypothesis_retirement_requires_human",
            )
            self.assertEqual(parent.read_text(encoding="utf-8"), original)
            self.assertFalse((root / ".orchestrator/backlog/T922_replacement.md").exists())
            escalation = _events(root, "hypothesis_retirement_escalated")[0]
            self.assertEqual(escalation["level"], "L3")
            self.assertTrue(escalation["escalation"])

    def test_reconnaissance_shape_blocks_or_allows_review_promotion(self) -> None:
        for filled, expected_returncode, expected_state in (
            (False, 1, "blocked"),
            (True, 0, "ready_for_review"),
        ):
            with self.subTest(filled=filled), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                task_id = "T923" if not filled else "T924"
                task = write_task(
                    root,
                    "active",
                    task_id,
                    schema="v2",
                    task_kind="analysis",
                    complexity_tier="M",
                    recon_required=True,
                    state="active",
                    allowed_paths=["src/"],
                    outputs=["src/result.txt"],
                    gates=[GREEN_GATE],
                )
                if filled:
                    task.write_text(_with_recon(task.read_text()), encoding="utf-8")
                _write_mock_executor(root, task_id)
                init_git_fixture_repo(root)

                stdout = io.StringIO()
                with _fixture_root(root), contextlib.redirect_stdout(stdout):
                    returncode = swarm.cmd_run_task(_run_args(task_id))

                summary = json.loads(stdout.getvalue())
                self.assertEqual(returncode, expected_returncode)
                self.assertEqual(summary["state_after"], expected_state)
                if filled:
                    self.assertNotIn("recon_missing", summary["blocked_reasons"])
                else:
                    self.assertIn("recon_missing", summary["blocked_reasons"])

    def test_mock_and_claude_planner_invocation_seams_never_run_real_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write_mock_planner(root, "launch", "launch", [])
            outcome = swarm._invoke_planner(
                mode="launch",
                context={"trigger_id": "launch"},
                repo=root,
                args=_planner_args(),
            )
            self.assertEqual(outcome.returncode, 0)
            self.assertEqual(outcome.proposals, [])

            framework = json.loads((root / "contracts/framework.json").read_text())
            framework["executors"] = {
                "planner": {
                    "backend": "claude",
                    # a nonexistent binary: the seam must refuse BEFORE any
                    # process launch — tests never touch a real CLI
                    "command": "claude-nonexistent-fixture",
                    "model": "fixture-top-tier",
                    "profile": "read-only+backlog-write",
                }
            }
            write_json(root, "contracts/framework.json", framework)
            with self.assertRaisesRegex(
                RuntimeError, "^planner_backend_unavailable:missing_cli:"
            ):
                swarm._invoke_planner(
                    mode="launch",
                    context={"trigger_id": "launch"},
                    repo=root,
                    args=_planner_args(backend="claude"),
                )

    def test_claude_planner_argv_and_proposal_extraction(self) -> None:
        # M2-C: the read-only profile is encoded in the argv itself
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            framework = json.loads((root / "contracts/framework.json").read_text())
            framework["executors"] = {
                "planner": {
                    "backend": "claude",
                    "command": "python",  # exists everywhere; never invoked here
                    "model": "fixture-top-tier",
                }
            }
            write_json(root, "contracts/framework.json", framework)
            argv = swarm._claude_planner_argv(root)
            self.assertEqual(argv[1], "-p")
            # --tools RESTRICTS (M2 review C1); --allowedTools only suppresses prompts
            self.assertIn("--tools", argv)
            self.assertEqual(argv[argv.index("--tools") + 1], "Read,Glob,Grep")
            self.assertNotIn("--allowedTools", argv)
            self.assertIn("fixture-top-tier", argv)

        good = 'reasoning...\n```json\n{"proposals": [{"action": "triage_confirm", "task_id": "T001", "note": "ok"}]}\n```\n'
        parsed = swarm._extract_planner_proposals(good)
        self.assertEqual(parsed, [{"action": "triage_confirm", "task_id": "T001", "note": "ok"}])
        # the LAST block wins
        two = good + '```json\n{"proposals": []}\n```'
        self.assertEqual(swarm._extract_planner_proposals(two), [])
        # malformed → None (planner_output_unparseable upstream)
        self.assertIsNone(swarm._extract_planner_proposals("no blocks here"))
        self.assertIsNone(swarm._extract_planner_proposals("```json\n{broken\n```"))
        self.assertIsNone(swarm._extract_planner_proposals('```json\n{"proposals": "not-a-list"}\n```'))

    def test_planner_env_passthrough_is_minimal(self) -> None:
        from unittest import mock as umock
        import os as _os

        with umock.patch.dict(
            _os.environ,
            {
                "ANTHROPIC_API_KEY": "fixture-key",
                "AWS_SECRET_ACCESS_KEY": "must-not-leak",
                "GITHUB_TOKEN": "must-not-leak",
            },
            clear=False,
        ):
            env = swarm._planner_passthrough_env()
        self.assertEqual(env.get("ANTHROPIC_API_KEY"), "fixture-key")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertNotIn("GITHUB_TOKEN", env)


if __name__ == "__main__":
    unittest.main()
