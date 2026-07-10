from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import init_git_fixture_repo
from runtime_test_utils import load_quality_gates_module
from runtime_test_utils import load_swarm_module
from runtime_test_utils import scaffold_runtime_repo
from runtime_test_utils import write_framework_json
from runtime_test_utils import write_review_log
from runtime_test_utils import write_run_manifest
from runtime_test_utils import write_task

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import swarm_claims
import swarm_events


swarm = load_swarm_module()
quality_gates = load_quality_gates_module()
GREEN_GATE = 'python -c "raise SystemExit(0)";'
REVIEW_BUNDLE_IMMEDIATE = {
    "run_manifest_dir": "reports/status/swarm_runs",
    "judge_review_dir": "reports/status/reviews",
    "min_separation_seconds": 0,
}


def _supervise_args(worktree_parent: Path, *, max_workers: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        once=True,
        interval_seconds=5,
        runner="local",
        max_workers=max_workers,
        worktree_parent=str(worktree_parent),
        remote="origin",
        base_branch="main",
        codex_model=None,
        codex_sandbox="workspace-write",
        unattended=False,
        max_worker_seconds=0,
    )


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _event_names(root: Path, task_id: str | None = None) -> list[str]:
    events, _ = swarm_events.read_events(root)
    return [
        str(event.get("event"))
        for event in events
        if task_id is None or event.get("task_id") == task_id
    ]


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


def _fake_executor(*, returncode: int = 0, write_output: bool = False):
    def invoke(**kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = Path(str(kwargs["cwd"]))
        if write_output:
            output = cwd / "src" / "result.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("result\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            ["fake-executor"], returncode, f"executor rc={returncode}\n", ""
        )

    return invoke


class M1SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None
        swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()

    def _scaffold_task_repo(
        self,
        root: Path,
        task_id: str,
        *,
        review_immediately: bool = True,
        framework_overrides: dict[str, object] | None = None,
        state: str = "backlog",
        folder: str | None = None,
        role: str = "Worker",
        outputs: list[str] | None = None,
        gates: list[str] | None = None,
        allowed_paths: list[str] | None = None,
    ) -> Path:
        scaffold_runtime_repo(root)
        overrides: dict[str, object] = dict(framework_overrides or {})
        if review_immediately:
            overrides["review_bundle"] = dict(REVIEW_BUNDLE_IMMEDIATE)
        if overrides:
            write_framework_json(root, overrides=overrides)
        folder = folder or state
        task_path = write_task(
            root,
            folder,
            task_id,
            state=state,
            role=role,
            outputs=outputs or ["src/result.txt"],
            gates=gates or [GREEN_GATE],
            allowed_paths=allowed_paths or ["src/"],
        )
        init_git_fixture_repo(root)
        return task_path

    def _run_cycle(
        self,
        root: Path,
        args: argparse.Namespace,
        *,
        executor: object | None = None,
    ) -> dict[str, object]:
        with _fixture_root(root):
            if executor is None:
                return swarm._supervise_cycle(args)
            with (
                mock.patch.object(swarm, "_codex_exec_cmd", return_value=["fake-executor"]),
                mock.patch.object(swarm, "_invoke_executor", side_effect=executor),
            ):
                return swarm._supervise_cycle(args)

    def _prepare_approved_branch(
        self,
        root: Path,
        worktree_parent: Path,
        task_id: str,
        *,
        gate: str = GREEN_GATE,
        role: str = "Worker",
        changed_path: str | None = None,
        changed_text: str = "changed\n",
        before_review=None,
    ) -> tuple[Path, Path, Path]:
        task_path = self._scaffold_task_repo(
            root,
            task_id,
            state="active",
            folder="active",
            role=role,
            outputs=["README.md"],
            gates=[gate],
            allowed_paths=["README.md", "break.marker", "reports/catalog.yaml"],
        )
        with _fixture_root(root):
            contract = swarm.load_framework_contract(root)
            task = swarm.load_tasks(contract)[task_id]
            worktree, _ = swarm.ensure_worktree(
                repo=root,
                task=task,
                worktree_parent=worktree_parent,
                base_ref="main",
            )
        worktree_task = worktree / task_path.relative_to(root)
        swarm._update_task_status_and_notes(
            task_path=worktree_task,
            new_state="ready_for_review",
            note_line="Fixture ready for supervisor merge.",
        )
        if changed_path is not None:
            changed = worktree / changed_path
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text(changed_text, encoding="utf-8")
        manifest_path = write_run_manifest(
            worktree,
            task_id,
            task_path=worktree_task.relative_to(worktree).as_posix(),
            generated_at_utc="2026-03-29T00:00:00Z",
        )
        if before_review is not None:
            before_review(worktree, manifest_path)
        # mirror the real judge flow: the work (incl. manifest) is committed
        # first; the review is a SEPARATE commit atop the reviewed tip and
        # records the binding fields the merge queue verifies.
        _git(worktree, "add", "-A")
        _git(worktree, "commit", "-m", f"{task_id}: ready_for_review")
        reviewed_sha = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        import hashlib as _hashlib

        write_review_log(
            worktree,
            task_id,
            task_path=worktree_task.relative_to(worktree).as_posix(),
            run_manifest_path=manifest_path.relative_to(worktree).as_posix(),
            manifest_sha256=_hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            reviewed_branch_sha=reviewed_sha,
        )
        _git(worktree, "add", "-A")
        _git(worktree, "commit", "-m", f"{task_id}: approved_pending_merge")
        return task_path, worktree, manifest_path

    def test_full_happy_path_claims_runs_judges_merges_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T901"
            self._scaffold_task_repo(root, task_id)
            args = _supervise_args(worktrees)

            first = self._run_cycle(
                root,
                args,
                executor=_fake_executor(returncode=0, write_output=True),
            )

            self.assertEqual([item["task_id"] for item in first["tick"]["started"]], [task_id])
            claim = swarm_claims.read_claims(root, "origin")[task_id]
            worktree = worktrees / f"wt-{task_id}"
            manifests = sorted(
                (worktree / "reports/status/swarm_runs").glob(f"{task_id}_*.json")
            )
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["claim"]["lease_id"], claim.lease_id)
            self.assertEqual(manifest["claim"]["sha"], claim.sha)
            self.assertEqual(manifest["claim"]["transport"], "remote")
            self.assertEqual(
                quality_gates._validate_swarm_run_manifest(
                    manifests[0], quality_gates.load_framework_contract(worktree)
                ),
                [],
            )

            second = self._run_cycle(root, args)

            self.assertEqual(second["merge"]["merged"], [task_id])
            self.assertTrue((root / ".orchestrator/done/T901_task.md").is_file())
            self.assertNotIn(task_id, swarm_claims.read_claims(root, "origin"))
            self.assertFalse(worktree.exists())
            self.assertNotIn(
                f"{task_id}_task",
                _git(root, "branch", "--format=%(refname:short)").stdout.splitlines(),
            )

            ordered = [
                name
                for name in _event_names(root, task_id)
                if name
                in {
                    "claim_created",
                    "run_started",
                    "run_finished",
                    "review_recorded",
                    "task_done",
                }
            ]
            self.assertEqual(
                ordered,
                [
                    "claim_created",
                    "run_started",
                    "run_finished",
                    "review_recorded",
                    "task_done",
                ],
            )

    def test_review_window_defers_fresh_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T902"
            self._scaffold_task_repo(root, task_id, review_immediately=False)
            args = _supervise_args(worktrees)
            self._run_cycle(
                root,
                args,
                executor=_fake_executor(returncode=0, write_output=True),
            )

            second = self._run_cycle(root, args)

            self.assertEqual([item["task_id"] for item in second["judge"]["deferred"]], [task_id])
            self.assertIn("review_deferred", _event_names(root, task_id))
            task_path = worktrees / f"wt-{task_id}" / ".orchestrator/active/T902_task.md"
            self.assertEqual(
                swarm._parse_status_value(task_path.read_text(encoding="utf-8"), "State"),
                "ready_for_review",
            )

    def test_review_backpressure_prevents_tick_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            scaffold_runtime_repo(root)
            write_framework_json(
                root,
                overrides={
                    "review_bundle": dict(REVIEW_BUNDLE_IMMEDIATE),
                    "wip": {"max_ready_for_review": 2},
                },
            )
            for task_id in ("T903", "T904"):
                write_task(
                    root,
                    "ready_for_review",
                    task_id,
                    state="ready_for_review",
                    outputs=["README.md"],
                    gates=[GREEN_GATE],
                )
            write_task(
                root,
                "backlog",
                "T905",
                state="backlog",
                outputs=["README.md"],
                gates=[GREEN_GATE],
            )
            init_git_fixture_repo(root)
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                tasks = swarm.load_tasks(contract)
                for task_id in ("T903", "T904"):
                    worktree, _ = swarm.ensure_worktree(
                        repo=root,
                        task=tasks[task_id],
                        worktree_parent=worktrees,
                        base_ref="main",
                    )
                    task_rel = tasks[task_id].path.resolve().relative_to(root.resolve()).as_posix()
                    write_run_manifest(worktree, task_id, task_path=task_rel)
                summary = swarm._step_tick(_supervise_args(worktrees))

            self.assertTrue(summary["backpressure"])
            self.assertEqual(summary["started"], [])
            self.assertNotIn("T905", swarm_claims.read_claims(root, "origin"))
            self.assertIn("review_backpressure", _event_names(root))

    def test_reap_reopens_expired_claim_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            task_id = "T906"
            self._scaffold_task_repo(
                root,
                task_id,
                state="active",
                folder="active",
                outputs=["README.md"],
            )
            old = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
            claim = swarm_claims.claim_task(
                root,
                "origin",
                task_id,
                session_id=swarm._ACTOR_SESSION_ID,
                branch=f"{task_id}_task",
                ttl_seconds=0,
                now=old,
            )
            self.assertTrue(claim.ok)

            with _fixture_root(root):
                summary = swarm._step_reap(_supervise_args(Path(tmp) / "worktrees"))

            self.assertEqual(summary["reopened"], [task_id])
            self.assertNotIn(task_id, swarm_claims.read_claims(root, "origin"))
            reopened = root / ".orchestrator/backlog/T906_task.md"
            self.assertIn("orphaned: lease expired", reopened.read_text(encoding="utf-8"))
            self.assertEqual(
                swarm._parse_status_value(reopened.read_text(encoding="utf-8"), "State"),
                "backlog",
            )
            names = _event_names(root, task_id)
            self.assertIn("task_orphaned", names)
            self.assertNotIn("judge_block", names)

    def test_merge_failure_reverts_task_commit_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T907"
            failing_gate = (
                'python -c "from pathlib import Path; '
                "raise SystemExit(Path('break.marker').exists())\";"
            )
            self._prepare_approved_branch(
                root,
                worktrees,
                task_id,
                gate=failing_gate,
                changed_path="break.marker",
            )
            pre_merge_sha = _git(root, "rev-parse", "HEAD").stdout.strip()

            with _fixture_root(root):
                summary = swarm._step_merge(_supervise_args(worktrees))

            self.assertEqual(summary["reverted"], [task_id])
            self.assertFalse((root / "break.marker").exists())
            self.assertEqual(
                _git(root, "rev-parse", "HEAD^").stdout.strip(),
                pre_merge_sha,
            )
            blocked = root / ".orchestrator/blocked/T907_task.md"
            self.assertTrue(blocked.is_file())
            self.assertIn("merge_reverted", _event_names(root, task_id))
            events, _ = swarm_events.read_events(root)
            reverted = [event for event in events if event.get("event") == "merge_reverted"]
            self.assertTrue(reverted[-1].get("escalation"))

    def test_merge_refuses_worker_operator_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T908"
            self._prepare_approved_branch(
                root,
                worktrees,
                task_id,
                role="Worker",
                changed_path="reports/catalog.yaml",
                changed_text="reports: []\n",
            )

            with _fixture_root(root):
                summary = swarm._step_merge(_supervise_args(worktrees))

            self.assertEqual(summary["refused"], [{"task_id": task_id, "reason": "operator_surface"}])
            self.assertFalse((root / "reports/catalog.yaml").exists())
            self.assertTrue((root / ".orchestrator/blocked/T908_task.md").is_file())
            self.assertIn("merge_refused_operator_surface", _event_names(root, task_id))

    def test_merge_refuses_superseded_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T909"
            claim_holder: dict[str, object] = {}

            def stamp_claim(worktree: Path, manifest_path: Path) -> None:
                claim = swarm_claims.claim_task(
                    root,
                    "origin",
                    task_id,
                    session_id=swarm._ACTOR_SESSION_ID,
                    branch=f"{task_id}_task",
                )
                assert claim.ok
                claim_holder["claim"] = claim
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["claim"] = {
                    "lease_id": claim.lease_id,
                    "sha": claim.sha,
                    "transport": claim.transport,
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

            _, worktree, manifest_path = self._prepare_approved_branch(
                root,
                worktrees,
                task_id,
                before_review=stamp_claim,
            )
            claim = claim_holder["claim"]

            # renewals advance the SAME chain — a renewed claim still merges,
            # so first prove the epoch attack is what fails: release + reclaim
            released = swarm_claims.release_claim(
                root,
                "origin",
                task_id,
                expected_sha=str(claim.sha),
                reason="test_reap",
            )
            self.assertTrue(released.ok)
            reclaimed = swarm_claims.claim_task(
                root,
                "origin",
                task_id,
                session_id=swarm._ACTOR_SESSION_ID,
                branch=f"{task_id}_task",
            )
            self.assertTrue(reclaimed.ok)
            self.assertEqual(reclaimed.lease_id, 1)  # recycled number, new root

            with _fixture_root(root):
                summary = swarm._step_merge(_supervise_args(worktrees))

            self.assertEqual(
                summary["refused"],
                [{"task_id": task_id, "reason": "stale_lease_chain"}],
            )
            self.assertIn("merge_refused_stale_lease", _event_names(root, task_id))

    def test_repair_retries_with_context_then_exhausts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T910"
            self._scaffold_task_repo(
                root,
                task_id,
                framework_overrides={"repair": {"max_attempts": 2}},
                outputs=["README.md"],
            )
            args = _supervise_args(worktrees)
            executor = _fake_executor(returncode=1)

            self._run_cycle(root, args, executor=executor)
            second = self._run_cycle(root, args, executor=executor)
            with mock.patch.object(
                swarm,
                "_invoke_executor",
                side_effect=AssertionError("repair should be exhausted"),
            ):
                third = self._run_cycle(root, args)

            self.assertEqual([item["task_id"] for item in second["repair"]["repaired"]], [task_id])
            self.assertEqual(third["repair"]["exhausted"], [task_id])
            worktree = worktrees / f"wt-{task_id}"
            manifests = sorted(
                (worktree / "reports/status/swarm_runs").glob(f"{task_id}_*.json")
            )
            self.assertEqual(len(manifests), 2)
            second_manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
            self.assertTrue(second_manifest["executor"]["repair_context"])
            self.assertEqual(_event_names(root, task_id).count("repair_exhausted"), 1)

    def test_fresh_cycle_after_completion_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T911"
            self._scaffold_task_repo(root, task_id)
            args = _supervise_args(worktrees)
            self._run_cycle(
                root,
                args,
                executor=_fake_executor(returncode=0, write_output=True),
            )
            self._run_cycle(root, args)
            before = _event_names(root, task_id)

            fresh = self._run_cycle(root, args)
            after = _event_names(root, task_id)

            self.assertEqual(fresh["tick"]["selected"], [])
            self.assertEqual(fresh["merge"]["merged"], [])
            self.assertEqual(after.count("claim_created"), before.count("claim_created"))
            self.assertEqual(after.count("task_done"), before.count("task_done"))
            self.assertNotIn(task_id, swarm_claims.read_claims(root, "origin"))

    def test_repair_escalates_integrity_block_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T912"
            task_path = self._scaffold_task_repo(
                root,
                task_id,
                state="active",
                folder="active",
                outputs=["README.md"],
            )
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                task = swarm.load_tasks(contract)[task_id]
                worktree, _ = swarm.ensure_worktree(
                    repo=root,
                    task=task,
                    worktree_parent=worktrees,
                    base_ref="main",
                )
            worktree_task = worktree / task_path.relative_to(root)
            swarm._update_task_status_and_notes(
                task_path=worktree_task,
                new_state="blocked",
                note_line="Fixture integrity block.",
            )
            manifest_path = write_run_manifest(
                worktree,
                task_id,
                task_path=worktree_task.relative_to(worktree).as_posix(),
                result_status="blocked",
                state_after="blocked",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["result"]["blocked_reasons"] = ["frontmatter_tampered"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _git(worktree, "add", "-A")
            _git(worktree, "commit", "-m", f"{task_id}: blocked")

            with _fixture_root(root), mock.patch.object(
                swarm,
                "_run_task_in_process",
                side_effect=AssertionError("integrity failures are not repairable"),
            ):
                summary = swarm._step_repair(
                    _supervise_args(worktrees),
                    candidate_ids={task_id},
                )

            self.assertEqual(summary["integrity_blocks"], [task_id])
            self.assertEqual(summary["repaired"], [])
            self.assertIn("integrity_block", _event_names(root, task_id))
            events, _ = swarm_events.read_events(root)
            integrity = [event for event in events if event.get("event") == "integrity_block"]
            self.assertTrue(integrity[-1].get("escalation"))


if __name__ == "__main__":
    unittest.main()
