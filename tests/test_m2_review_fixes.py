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



class ProposalIntegrityTest(unittest.TestCase):
    """C5/C6/C10 — proposal-level integrity the reviewers demanded."""

    def test_duplicate_task_id_proposal_is_refused(self) -> None:
        # C5: a create_task whose frontmatter task_id collides with an
        # existing task is refused (cannot brick the real task)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "done", "T500", state="done", **_v2_task_kwargs())
            init_git_fixture_repo(root)
            args = _supervise_args()
            proposals = [
                {
                    "action": "create_task",
                    "path": ".orchestrator/backlog/T500_shadow.md",
                    "content": _minimal_v2_task("T500", "shadow"),
                }
            ]
            with _fixture_root(root):
                summary = swarm._apply_planner_proposals(
                    mode="launch", proposals=proposals, repo=root, args=args
                )
            outcome = summary["outcomes"][0]
            self.assertEqual(outcome["status"], "refused")
            self.assertTrue(str(outcome["reason"]).startswith("planner_task_id_not_unique"))

    def test_task_id_filename_mismatch_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            init_git_fixture_repo(root)
            proposals = [
                {
                    "action": "create_task",
                    "path": ".orchestrator/backlog/T901_x.md",
                    "content": _minimal_v2_task("T902", "x"),
                }
            ]
            with _fixture_root(root):
                summary = swarm._apply_planner_proposals(
                    mode="launch", proposals=proposals, repo=root, args=_supervise_args()
                )
            self.assertTrue(
                str(summary["outcomes"][0]["reason"]).startswith("planner_task_id_filename_mismatch")
            )

    def test_update_workstreams_rejects_invalid_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            init_git_fixture_repo(root)
            proposals = [{"action": "update_workstreams", "content": "just prose, no rows"}]
            with _fixture_root(root):
                summary = swarm._apply_planner_proposals(
                    mode="launch", proposals=proposals, repo=root, args=_supervise_args()
                )
            self.assertEqual(summary["outcomes"][0]["reason"], "planner_workstreams_content_invalid")

    def test_validation_same_path_different_hash_rejected(self) -> None:
        # C6: same underlying artifact under two fake hashes is not independence
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root, "backlog", "T100", schema="v2", task_kind="etl", complexity_tier="S",
                gates=["python scripts/noop_gate.py"], outputs=["data/processed/panel.csv"],
                inputs=[{"manifest": "data/raw_manifest/src.json", "sha256": "a" * 64}],
            )
            write_task(
                root, "backlog", "T101", schema="v2", task_kind="validation", complexity_tier="S",
                gates=["python scripts/noop_gate.py"], outputs=["reports/validation/v.json"],
                constructed_by="T100",
                inputs=[{"manifest": "data/raw_manifest/src.json", "sha256": "b" * 64, "comparison_basis": True}],
            )
            from runtime_test_utils import chdir
            with chdir(root):
                result = quality_gates.gate_task_lint()
            reasons = {f["reason"] for f in result.details["failures"]}
            self.assertIn("comparison_basis_path_not_disjoint", reasons)


class HypothesisGuardTest(unittest.TestCase):
    def test_split_of_hypothesis_linked_task_escalates(self) -> None:
        # C11: list-field hypothesis_ids is honored by the retirement guard
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root, "backlog", "T300", schema="v2", task_kind="analysis",
                complexity_tier="S", gates=["python scripts/noop_gate.py"],
                outputs=["reports/tables/t.csv"],
                extra_frontmatter={"hypothesis_ids": ["H1"]},
            )
            init_git_fixture_repo(root)
            proposals = [{
                "action": "split_task", "task_id": "T300",
                "into": [{"path": ".orchestrator/backlog/T301_a.md", "content": _minimal_v2_task("T301", "a")}],
            }]
            with _fixture_root(root):
                summary = swarm._apply_planner_proposals(
                    mode="replan", proposals=proposals, repo=root, args=_supervise_args()
                )
            self.assertEqual(summary["outcomes"][0]["reason"], "hypothesis_retirement_requires_human")
            self.assertTrue((root / ".orchestrator/backlog/T300_task.md").exists())


def _minimal_v2_task(task_id: str, slug: str) -> str:
    return (
        "---\n"
        "task_schema: research_swarm.task.v2\n"
        f"task_id: {task_id}\n"
        f"title: {slug}\n"
        "workstream: W1\n"
        "task_kind: etl\n"
        "complexity_tier: S\n"
        "role: Worker\n"
        "priority: medium\n"
        "allow_network: false\n"
        "recon_required: false\n"
        "recon_waiver: S-tier bounded\n"
        "dependencies: []\n"
        "integration_ready_dependencies: []\n"
        "success_criteria:\n"
        "  - id: SC1\n"
        "    statement: does the thing\n"
        "    verification: python scripts/noop_gate.py\n"
        "budgets:\n"
        "  max_wall_clock: 1h\n"
        "  max_tokens: 100000\n"
        "  max_cost_usd: 5\n"
        "inputs: []\n"
        "allowed_paths:\n"
        "  - src/etl/\n"
        "disallowed_paths:\n"
        "  - contracts/\n"
        "outputs:\n"
        "  - src/etl/x.py\n"
        "gates:\n"
        "  - python scripts/noop_gate.py\n"
        "stop_conditions:\n"
        "  - ambiguous\n"
        "---\n\n"
        f"# Task {task_id}\n\n## Context\nx\n\n## Inputs\nx\n\n## Outputs\nx\n\n"
        "## Success Criteria\n- [ ] SC1\n\n## Review Bundle Requirements\n- [ ] x\n\n"
        "## Validation / Commands\n- python scripts/noop_gate.py\n\n"
        "## Reconnaissance\n- none (S-tier waived)\n\n"
        "## Status\n- State: backlog\n- Last updated: 2026-07-10\n\n"
        "## Notes / Decisions\n- 2026-07-10: created.\n"
    )



class VerificationPassRegressionsTest(unittest.TestCase):
    """Regressions the fix-verification pass demanded (tranche 3)."""

    def test_make_dash_c_gate_is_rejected(self) -> None:
        self.assertEqual(
            swarm_taskfile.gate_command_violation("make -C /outside gate").split(":")[0],
            "gate_make_flag_forbidden",
        )
        self.assertIsNone(swarm_taskfile.gate_command_violation("make gate"))

    def test_force_deps_cannot_bypass_plan_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "active", "T840", state="active", slug="p", **_v2_task_kwargs())
            init_git_fixture_repo(root)
            write_json(root, ".swarm/plan_approval_pending.json", {"x": 1})
            args = _run_task_args("T840")
            args.force_deps = True
            with _fixture_root(root):
                with self.assertRaisesRegex(SystemExit, "plan_unapproved:T840"):
                    swarm.cmd_run_task(args)

    def test_approve_plan_fails_closed_on_missing_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            init_git_fixture_repo(root)
            write_json(root, ".swarm/plan_approval_pending.json", {"schema_version": "x"})
            stdout = io.StringIO()
            with _fixture_root(root), contextlib.redirect_stdout(stdout):
                rc = swarm.cmd_approve_plan(argparse.Namespace(approved_by="owner"))
            self.assertEqual(rc, 1)
            self.assertTrue((root / ".swarm/plan_approval_pending.json").exists())

    def test_first_of_kind_is_per_task_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root, "backlog", "T860", task_kind="analysis",
                **{k: v for k, v in _v2_task_kwargs().items() if k != "task_kind"},
            )
            write_task(
                root, "backlog", "T861", task_kind="proof",
                **{k: v for k, v in _v2_task_kwargs().items() if k != "task_kind"},
            )
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                tasks, _ = swarm.load_tasks_quarantined(contract)
                reasons = swarm._task_triage_reasons(tasks["T861"], tasks)
            self.assertIn("first_of_kind_in_workstream", reasons)

    def test_split_orphaning_validation_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root, "backlog", "T100", schema="v2", task_kind="etl", complexity_tier="S",
                gates=["python scripts/noop_gate.py"], outputs=["data/processed/p.csv"],
                inputs=[{"manifest": "data/raw_manifest/a.json", "sha256": "a" * 64}],
            )
            write_task(
                root, "backlog", "T101", schema="v2", task_kind="validation", complexity_tier="S",
                gates=["python scripts/noop_gate.py"], outputs=["reports/validation/v.json"],
                constructed_by="T100",
                inputs=[{"manifest": "data/raw_manifest/b.json", "sha256": "b" * 64, "comparison_basis": True}],
            )
            init_git_fixture_repo(root)
            proposals = [{
                "action": "split_task", "task_id": "T100",
                "into": [{"path": ".orchestrator/backlog/T110_x.md", "content": _minimal_v2_task("T110", "x")}],
            }]
            with _fixture_root(root):
                summary = swarm._apply_planner_proposals(
                    mode="replan", proposals=proposals, repo=root, args=_supervise_args()
                )
            self.assertEqual(summary["outcomes"][0]["status"], "lint_failed")
            self.assertTrue((root / ".orchestrator/backlog/T100_task.md").exists())

    def test_integration_ready_promotion_enforces_recon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root, "active", "T870", state="active", schema="v2", task_kind="model",
                complexity_tier="M", recon_required=True, workstream="W8",
                gates=["python scripts/noop_gate.py"], outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            args = _run_task_args("T870")
            args.final_state = "integration_ready"
            with _fixture_root(root):
                swarm.cmd_run_task(args)
            manifest = json.loads(
                sorted((root / "reports/status/swarm_runs").glob("T870_*.json"))[0].read_text()
            )
            self.assertIn("recon_missing", manifest["result"]["blocked_reasons"])

    def test_recon_placeholder_suffix_does_not_count(self) -> None:
        placeholder = (
            "## Reconnaissance\n"
            "- Scope understanding: TBD\n"
            "- Risks and unknowns: ???\n"
            "- Decomposition pressure assessment: N/A\n"
            "## Status\n"
        )
        self.assertLess(swarm._reconnaissance_line_count(placeholder), 3)
        substantive = (
            "## Reconnaissance\n"
            "- Scope understanding: two vendor sources, keyed by rollup_id\n"
            "- Risks and unknowns: scroll attribution unresolved pre-Dencun\n"
            "- Decomposition pressure assessment: split discovery from build\n"
            "## Status\n"
        )
        self.assertGreaterEqual(swarm._reconnaissance_line_count(substantive), 3)


if __name__ == "__main__":
    unittest.main()
