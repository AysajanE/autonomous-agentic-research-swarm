"""Regression tests for the M0 dual-vendor adversarial review fixes.

Each test is a concrete attack scenario from the review round: the Codex
critique (F1..F11) and the Claude adversary's judge-time tamper variants.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_review_log,
    write_run_manifest,
    write_task,
    write_text,
)


swarm = load_swarm_module()
quality_gates = load_quality_gates_module()
GREEN_GATE = 'python scripts/noop_gate.py'
WEAK_GATE = 'python -c "pass";'


def _judge_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        unattended=False,
        on_fail="blocked",
        note="",
    )


def _run_args(task_id: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
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
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class JudgePinnedBindingTest(unittest.TestCase):
    """F4 + the Claude adversary's judge-time tamper scenarios."""

    def _fixture(self, tmp: str, task_id: str) -> tuple[Path, Path]:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        task_path = write_task(
            root,
            "ready_for_review",
            task_id,
            state="ready_for_review",
            gates=[GREEN_GATE],
            outputs=["README.md"],
        )
        init_git_fixture_repo(root)
        return root, task_path

    def _judge(self, root: Path, task_id: str) -> tuple[int, dict[str, object]]:
        stdout = io.StringIO()
        with (
            mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            result = swarm.cmd_judge_task(_judge_args(task_id))
        review_paths = sorted((root / "reports/status/reviews").glob(f"{task_id}_*.json"))
        assert len(review_paths) == 1
        return result, json.loads(review_paths[0].read_text(encoding="utf-8"))

    def test_working_tree_frontmatter_tamper_blocks_judge(self) -> None:
        # Claude scenario: an UNCOMMITTED edit to the task file widens
        # allowed_paths and neuters the gate; no new commit exists.
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T601")
            write_run_manifest(root, "T601", task_path=task_path.relative_to(root).as_posix())

            text = task_path.read_text(encoding="utf-8")
            tampered = text.replace(GREEN_GATE.rstrip(";"), WEAK_GATE.rstrip(";"))
            self.assertNotEqual(text, tampered)
            task_path.write_text(tampered, encoding="utf-8")

            result, review = self._judge(root, "T601")

            self.assertEqual(result, 1)
            self.assertIn("post_run_frontmatter_tamper", review["checks"]["failures"])

    def test_judge_executes_pinned_gates_not_live_ones(self) -> None:
        # even with a matching frontmatter, the executed commands must come
        # from the manifest's pinned copy
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T602")
            write_run_manifest(root, "T602", task_path=task_path.relative_to(root).as_posix())

            executed: list[list[str]] = []
            original = swarm._run_gates

            def spying_run_gates(repo, gates, **kwargs):
                executed.append(list(gates))
                return original(repo, gates, **kwargs)

            stdout = io.StringIO()
            with (
                mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                mock.patch.object(swarm, "_run_gates", side_effect=spying_run_gates),
                contextlib.redirect_stdout(stdout),
            ):
                swarm.cmd_judge_task(_judge_args("T602"))

            manifest = json.loads(
                sorted((root / "reports/status/swarm_runs").glob("T602_*.json"))[0].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(executed[0], manifest["commands"]["gates"])

    def test_drifted_executor_log_blocks_judge(self) -> None:
        # F2 partial hardening: the durable log must still match its hash
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T603")
            write_run_manifest(root, "T603", task_path=task_path.relative_to(root).as_posix())
            log_path = root / "reports/status/swarm_runs/logs/T603_20260329T000000Z.log"
            log_path.write_text("rewritten after the fact\n", encoding="utf-8")

            result, review = self._judge(root, "T603")

            self.assertEqual(result, 1)
            self.assertTrue(
                any(
                    item.startswith("executor_log_binding_failed:sha256_mismatch:")
                    for item in review["checks"]["failures"]
                ),
                review["checks"]["failures"],
            )

    def test_manifest_without_timestamp_cannot_qualify(self) -> None:
        # Claude finding 2: the separation window must fail closed
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T604")
            manifest_path = write_run_manifest(
                root, "T604", task_path=task_path.relative_to(root).as_posix()
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["generated_at_utc"] = "not-a-timestamp"
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            self.assertFalse(swarm._is_valid_run_manifest(manifest_path, "T604"))


class RunTaskDependencyStateTest(unittest.TestCase):
    """F7: dependency enforcement covers every runnable state."""

    def test_blocked_state_cannot_bypass_dependency_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "backlog", "T610", slug="upstream")
            write_task(
                root,
                "blocked",
                "T611",
                state="blocked",
                dependencies=["T610"],
                slug="downstream",
            )
            init_git_fixture_repo(root)

            with (
                mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            ):
                with self.assertRaisesRegex(SystemExit, "dependencies_unsatisfied:T611:T610"):
                    swarm.cmd_run_task(_run_args("T611"))


class RunTaskQuarantineTest(unittest.TestCase):
    """F8: one malformed sibling must not wedge run-task/judge-task."""

    def test_run_task_proceeds_past_malformed_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_text(root, ".orchestrator/backlog/T698_broken.md", "---\ntask_id: T698\n")
            task_path = write_task(
                root,
                "active",
                "T699",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            stdout = io.StringIO()
            with (
                mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(stdout),
            ):
                result = swarm.cmd_run_task(_run_args("T699"))

            self.assertEqual(result, 0)
            manifest = json.loads(
                sorted((root / "reports/status/swarm_runs").glob("T699_*.json"))[0].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(manifest["quarantined_tasks"]), 1)

    def test_quarantined_target_is_a_precise_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_text(root, ".orchestrator/backlog/T700_broken.md", "---\ntask_id: T700\n")
            init_git_fixture_repo(root)

            with (
                mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            ):
                with self.assertRaisesRegex(SystemExit, "task_quarantined:T700:"):
                    swarm.cmd_run_task(_run_args("T700"))


class DoneBundleGateTest(unittest.TestCase):
    """F1: a durable done state needs an approving review bound to a passing
    executor_run manifest."""

    def _fixture_done(self, tmp: str, **manifest_overrides: object) -> tuple[Path, Path, Path]:
        root = Path(tmp)
        scaffold_runtime_repo(root)
        task_path = write_task(
            root,
            "done",
            "T620",
            state="done",
            gates=[GREEN_GATE],
            outputs=["README.md"],
        )
        init_git_fixture_repo(root)
        manifest_path = write_run_manifest(
            root,
            "T620",
            task_path=task_path.relative_to(root).as_posix(),
            **manifest_overrides,
        )
        return root, task_path, manifest_path

    def test_blocking_review_cannot_satisfy_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path, manifest_path = self._fixture_done(tmp)
            write_review_log(
                root,
                "T620",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=manifest_path.relative_to(root).as_posix(),
                outcome="block",
                state_after="blocked",
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("missing_approving_review_log" in item for item in result.details["failures"]),
                result.details,
            )

    def test_approving_review_linked_to_backfill_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path, manifest_path = self._fixture_done(
                tmp, provenance_class="backfill"
            )
            write_review_log(
                root,
                "T620",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=manifest_path.relative_to(root).as_posix(),
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("approving_review_manifest_provenance" in item for item in result.details["failures"]),
                result.details,
            )

    def test_approving_review_linked_to_blocked_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path, manifest_path = self._fixture_done(tmp, result_status="blocked")
            write_review_log(
                root,
                "T620",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=manifest_path.relative_to(root).as_posix(),
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("approving_review_manifest_not_passing" in item for item in result.details["failures"]),
                result.details,
            )

    def test_clean_v2_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path, manifest_path = self._fixture_done(tmp)
            write_review_log(
                root,
                "T620",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=manifest_path.relative_to(root).as_posix(),
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertTrue(result.ok, result.details)


class SchemaDowngradeTest(unittest.TestCase):
    """F5: dropping schema_version no longer produces an unchecked artifact."""

    def test_unexempted_v1_run_manifest_fails_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T630")
            write_run_manifest(
                root,
                "T630",
                task_path=task_path.relative_to(root).as_posix(),
                schema_version="research_swarm.runtime_run_manifest.v1",
            )
            with chdir(root):
                result = quality_gates.gate_swarm_run_manifest_validity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("unexempted_v1_schema" in item for item in result.details["failures"]),
                result.details,
            )


if __name__ == "__main__":
    unittest.main()
