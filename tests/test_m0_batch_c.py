from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import subprocess
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
# Trailing ';' keeps the closing quote out of the frontmatter parser's
# strip("'\"") — a known wart the task-lint gate will reject at M2.
GREEN_GATE = 'python -c "raise SystemExit(0)";'


def _git(root: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


def _judge_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        unattended=False,
        on_fail="blocked",
        note="",
    )


def _commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", message],
        check=True,
        capture_output=True,
        text=True,
    )


class GuardedBaseSyncTest(unittest.TestCase):
    def _fixture(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        init_git_fixture_repo(root)
        return root

    def test_sync_refuses_when_local_base_is_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            write_text(root, "local_only.txt", "local\n")
            _commit_all(root, "local-only commit")

            with self.assertRaisesRegex(SystemExit, "base_sync_refused_local_ahead:main:1"):
                swarm._supervisor_sync_to_remote_base(repo=root, remote="origin", base_branch="main")

            self.assertEqual(_git(root, "log", "-1", "--pretty=%s"), "local-only commit")

    def test_sync_fast_forwards_when_local_base_is_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            write_text(root, "second.txt", "second\n")
            _commit_all(root, "second commit")
            subprocess.run(
                ["git", "-C", str(root), "push", "origin", "main"],
                check=True,
                capture_output=True,
                text=True,
            )
            remote_tip = _git(root, "rev-parse", "HEAD")
            subprocess.run(
                ["git", "-C", str(root), "reset", "--hard", "HEAD~1"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(_git(root, "rev-parse", "HEAD"), remote_tip)

            swarm._supervisor_sync_to_remote_base(repo=root, remote="origin", base_branch="main")

            self.assertEqual(_git(root, "rev-parse", "HEAD"), remote_tip)

    def test_sync_from_other_branch_checks_out_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            subprocess.run(
                ["git", "-C", str(root), "checkout", "-b", "T999_side"],
                check=True,
                capture_output=True,
                text=True,
            )

            swarm._supervisor_sync_to_remote_base(repo=root, remote="origin", base_branch="main")

            self.assertEqual(_git(root, "rev-parse", "--abbrev-ref", "HEAD"), "main")


class JudgeIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None

    def _judge(self, root: Path, task_id: str) -> tuple[int, dict[str, object]]:
        stdout = io.StringIO()
        with (
            mock.patch.dict("os.environ", {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            result = swarm.cmd_judge_task(_judge_args(task_id))
        review_paths = sorted((root / "reports/status/reviews").glob(f"{task_id}_*.json"))
        assert len(review_paths) == 1, review_paths
        return result, json.loads(review_paths[0].read_text(encoding="utf-8"))

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

    def test_clean_bundle_is_approved_with_v2_review_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T210")
            write_run_manifest(root, "T210", task_path=task_path.relative_to(root).as_posix())

            result, review = self._judge(root, "T210")

            self.assertEqual(result, 0)
            self.assertEqual(review["decision"]["outcome"], "approve")
            self.assertEqual(review["schema_version"], swarm.JUDGE_REVIEW_LOG_SCHEMA_VERSION)
            self.assertTrue(review["reviewer"]["session_id"])
            self.assertIsNone(review["operator_attestation"])
            committed = _git(root, "log", "-1", "--name-only", "--pretty=format:").splitlines()
            self.assertIn(task_path.relative_to(root).as_posix(), committed)

    def test_stray_unowned_file_blocks_and_stays_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T217")
            write_run_manifest(root, "T217", task_path=task_path.relative_to(root).as_posix())
            write_text(root, "stray_leftover.txt", "leftover violation\n")

            result, review = self._judge(root, "T217")

            self.assertEqual(result, 1)
            self.assertTrue(
                any(
                    item.startswith("ownership_violation:stray_leftover.txt:")
                    for item in review["checks"]["failures"]
                ),
                review["checks"]["failures"],
            )
            status = _git(root, "status", "--porcelain")
            self.assertIn("?? stray_leftover.txt", status)
            committed = _git(root, "log", "-1", "--name-only", "--pretty=format:").splitlines()
            self.assertNotIn("stray_leftover.txt", committed)

    def test_post_manifest_commit_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T211")
            write_run_manifest(root, "T211", task_path=task_path.relative_to(root).as_posix())
            write_text(root, "README.md", "tampered after manifest\n")
            _commit_all(root, "unrelated tamper commit")

            result, review = self._judge(root, "T211")

            self.assertEqual(result, 1)
            failures = review["checks"]["failures"]
            self.assertTrue(
                any(item.startswith("post_manifest_commit_message:") for item in failures)
                or any(item.startswith("post_manifest_tamper:") for item in failures),
                failures,
            )

    def test_multiple_post_manifest_commits_block_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T212")
            write_run_manifest(root, "T212", task_path=task_path.relative_to(root).as_posix())
            write_text(root, "README.md", "first tamper\n")
            _commit_all(root, "tamper one")
            write_text(root, "README.md", "second tamper\n")
            _commit_all(root, "tamper two")

            result, review = self._judge(root, "T212")

            self.assertEqual(result, 1)
            self.assertIn("post_manifest_commits:2", review["checks"]["failures"])

    def test_manifest_branch_mismatch_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T213")
            write_run_manifest(
                root,
                "T213",
                task_path=task_path.relative_to(root).as_posix(),
                branch="T213_other_branch",
            )

            result, review = self._judge(root, "T213")

            self.assertEqual(result, 1)
            self.assertIn(
                "manifest_branch_mismatch:T213_other_branch:main",
                review["checks"]["failures"],
            )

    def test_same_actor_session_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T214")
            write_run_manifest(
                root,
                "T214",
                task_path=task_path.relative_to(root).as_posix(),
                actor_session_id=swarm._ACTOR_SESSION_ID,
            )

            result, review = self._judge(root, "T214")

            self.assertEqual(result, 1)
            self.assertIn("actor_separation_same_session", review["checks"]["failures"])

    def test_fresh_manifest_inside_separation_window_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T215")
            now_iso = (
                dt.datetime.now(tz=dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            write_run_manifest(
                root,
                "T215",
                task_path=task_path.relative_to(root).as_posix(),
                generated_at_utc=now_iso,
            )

            result, review = self._judge(root, "T215")

            self.assertEqual(result, 1)
            self.assertTrue(
                any(item.startswith("actor_separation_window:") for item in review["checks"]["failures"]),
                review["checks"]["failures"],
            )

    def test_committed_ownership_violation_blocks_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, task_path = self._fixture(tmp, "T216")
            write_text(root, "unowned_path.txt", "committed violation\n")
            _commit_all(root, "T216: ready_for_review")
            write_run_manifest(root, "T216", task_path=task_path.relative_to(root).as_posix())

            result, review = self._judge(root, "T216")

            self.assertEqual(result, 1)
            self.assertTrue(
                any(
                    item.startswith("ownership_violation:unowned_path.txt:")
                    for item in review["checks"]["failures"]
                ),
                review["checks"]["failures"],
            )


class ReviewBundleActorSeparationGateTest(unittest.TestCase):
    def _fixture_with_done_bundle(
        self,
        tmp: str,
        *,
        run_session: str,
        review_session: str,
        review_generated_at: str = "2026-03-29T01:00:00Z",
    ) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        task_path = write_task(
            root,
            "done",
            "T220",
            state="done",
            gates=[GREEN_GATE],
            outputs=["README.md"],
        )
        init_git_fixture_repo(root)
        manifest_path = write_run_manifest(
            root,
            "T220",
            task_path=task_path.relative_to(root).as_posix(),
            actor_session_id=run_session,
        )
        write_review_log(
            root,
            "T220",
            task_path=task_path.relative_to(root).as_posix(),
            run_manifest_path=manifest_path.relative_to(root).as_posix(),
            reviewer_session_id=review_session,
            generated_at_utc=review_generated_at,
        )
        return root

    def test_same_session_review_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_with_done_bundle(
                tmp, run_session="shared-session", review_session="shared-session"
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("actor_separation_same_session" in item for item in result.details["failures"]),
                result.details,
            )

    def test_review_inside_window_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_with_done_bundle(
                tmp,
                run_session="worker-session",
                review_session="judge-session",
                review_generated_at="2026-03-29T00:00:30Z",
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("actor_separation_window" in item for item in result.details["failures"]),
                result.details,
            )

    def test_separated_distinct_sessions_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_with_done_bundle(
                tmp, run_session="worker-session", review_session="judge-session"
            )
            with chdir(root):
                result = quality_gates.gate_review_bundle_integrity()
            self.assertTrue(result.ok, result.details)


if __name__ == "__main__":
    unittest.main()
