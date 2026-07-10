"""Attack-scenario regressions for the M2 dual-vendor review fixes."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
)

swarm = load_swarm_module()
quality_gates = load_quality_gates_module()
sys.path.insert(0, str(_TESTS_ROOT.parent / "scripts"))
import swarm_taskfile


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


def _v2_task_kwargs() -> dict:
    return dict(
        schema="v2",
        task_kind="etl",
        complexity_tier="S",
        gates=["python scripts/noop_gate.py"],
        outputs=["README.md"],
    )


class GateFormPolicyTest(unittest.TestCase):
    """C3: an autonomously-authored gate can never be an arbitrary-code channel."""

    def test_lint_rejects_python_dash_c_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T820",
                **{**_v2_task_kwargs(), "gates": ["python -c \"import os\" ;"]},
            )
            with _fixture_root(root):
                from runtime_test_utils import chdir

                with chdir(root):
                    result = quality_gates.gate_task_lint()
            reasons = {f["reason"] for f in result.details["failures"]}
            self.assertIn("gate_code_execution_forbidden", reasons)

    def test_validator_allows_real_gates_only(self) -> None:
        self.assertIsNone(swarm_taskfile.gate_command_violation("python scripts/quality_gates.py"))
        self.assertIsNone(swarm_taskfile.gate_command_violation("make gate"))
        self.assertEqual(
            swarm_taskfile.gate_command_violation("python -c code_here ;").split(":")[0],
            "gate_code_execution_forbidden",
        )
        self.assertEqual(
            swarm_taskfile.gate_command_violation("python -m pytest").split(":")[0],
            "gate_code_execution_forbidden",
        )
        self.assertEqual(
            swarm_taskfile.gate_command_violation("python /etc/evil.py").split(":")[0],
            "gate_python_script_outside_repo",
        )
        self.assertEqual(
            swarm_taskfile.gate_command_violation("bash run.sh").split(":")[0],
            "gate_interpreter_not_allowlisted",
        )

    def test_execution_rejects_forbidden_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok, outputs = swarm._run_gates(Path(tmp), ['python -c "print(1)"'])
            self.assertFalse(ok)
            self.assertTrue(
                str(outputs[0]["constraint_violation"]).startswith("gate_form_forbidden:")
            )


class PlanApprovalGateTest(unittest.TestCase):
    """C2 + both reviewers' BLOCKER: the hold binds EVERY dispatch path."""

    def _fixture_with_pending_plan(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        write_task(root, "backlog", "T830", slug="planned", **_v2_task_kwargs())
        init_git_fixture_repo(root)
        write_json(
            root,
            ".swarm/plan_approval_pending.json",
            {"schema_version": "research_swarm.plan_approval_pending.v1", "digest": "x"},
        )
        return root

    def test_ready_backlog_funnel_excludes_unapproved_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_with_pending_plan(tmp)
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                tasks, _ = swarm.load_tasks_quarantined(contract)
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)
            self.assertNotIn("T830", [t.task_id for t in ready])

    def test_supervise_step_tick_honors_pending_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_with_pending_plan(tmp)
            stdout = io.StringIO()
            with _fixture_root(root), contextlib.redirect_stdout(stdout):
                summary = swarm._step_tick(_supervise_args())
            self.assertEqual(summary["selected"], [])
            self.assertEqual(summary["started"], [])

    def test_direct_run_task_refuses_unapproved_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "active", "T831", state="active", slug="planned", **_v2_task_kwargs())
            init_git_fixture_repo(root)
            write_json(
                root,
                ".swarm/plan_approval_pending.json",
                {"schema_version": "research_swarm.plan_approval_pending.v1"},
            )
            with _fixture_root(root):
                with self.assertRaisesRegex(SystemExit, "plan_unapproved:T831"):
                    swarm.cmd_run_task(_run_task_args("T831"))


def _supervise_args() -> argparse.Namespace:
    return argparse.Namespace(
        remote="origin",
        base_branch="main",
        max_workers=1,
        worktree_parent=None,
        unattended=False,
        codex_model=None,
        codex_sandbox="workspace-write",
        max_worker_seconds=0,
        executor_backend="mock",
        planner_backend="mock",
        create_pr=False,
    )


def _run_task_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        codex_model=None,
        codex_sandbox="workspace-write",
        unattended=False,
        skip_executor=True,
        force_deps=False,
        max_worker_seconds=0,
        repair_context=None,
        create_pr=False,
        final_state="ready_for_review",
        executor_backend="codex",
        record_session=False,
        i_accept_full_access=False,
    )


class ReadOnlyPlannerArgvTest(unittest.TestCase):
    def test_argv_restricts_toolset_not_just_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            framework = json.loads((root / "contracts/framework.json").read_text())
            framework["executors"] = {
                "planner": {"backend": "claude", "command": "python", "model": "m"}
            }
            write_json(root, "contracts/framework.json", framework)
            argv = swarm._claude_planner_argv(root)
            self.assertIn("--tools", argv)
            self.assertEqual(argv[argv.index("--tools") + 1], "Read,Glob,Grep")
            self.assertNotIn("--allowedTools", argv)
            self.assertIn("--strict-mcp-config", argv)


if __name__ == "__main__":
    unittest.main()
