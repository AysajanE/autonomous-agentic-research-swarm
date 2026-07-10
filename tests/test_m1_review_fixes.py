"""Attack-scenario regressions for the M1 dual-vendor review fix rounds."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "m1sup_fixture", _TESTS_ROOT / "test_m1_supervisor.py"
)
m1sup = importlib.util.module_from_spec(_spec)
sys.modules["m1sup_fixture"] = m1sup
_spec.loader.exec_module(m1sup)

swarm = m1sup.swarm
swarm_claims = m1sup.swarm_claims
swarm_events = sys.modules[swarm.swarm_events.__name__] if hasattr(swarm, "swarm_events") else None
import swarm_events as swarm_events_mod  # noqa: E402  (scripts on sys.path via fixtures)

from runtime_test_utils import write_text  # noqa: E402


def _git(repo: Path, *args: str):
    return m1sup._git(repo, *args)


class ForgedReviewTest(unittest.TestCase):
    """Codex F1: a worker-committed approving review must never merge."""

    def test_forged_review_without_journal_backing_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T971"
            _, worktree, manifest_path = m1sup.M1SupervisorTests._prepare_approved_branch(
                m1sup.M1SupervisorTests("test_full_happy_path_claims_runs_judges_merges_and_cleans"),
                root,
                worktrees,
                task_id,
            )
            # erase the journal backing: the review now looks worker-forged
            journal = root / "reports/status/events/events.jsonl"
            kept = [
                line
                for line in journal.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("event") != "review_recorded"
            ]
            journal.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

            with m1sup._fixture_root(root):
                summary = swarm._step_merge(m1sup._supervise_args(worktrees))

            self.assertEqual(summary["merged"], [])
            self.assertEqual(
                summary["refused"],
                [{"task_id": task_id, "reason": "review_not_journal_backed"}],
            )

    def test_dirty_worktree_bytes_cannot_satisfy_binding(self) -> None:
        # new-BLOCKER from the verification pass: binding reads committed
        # blobs; a dirty manifest matching the review hash earns nothing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T972"
            _, worktree, manifest_path = m1sup.M1SupervisorTests._prepare_approved_branch(
                m1sup.M1SupervisorTests("test_full_happy_path_claims_runs_judges_merges_and_cleans"),
                root,
                worktrees,
                task_id,
            )
            # tamper the COMMITTED manifest via a new commit, then restore the
            # worktree copy so worktree bytes match the review hash again
            original = manifest_path.read_text(encoding="utf-8")
            payload = json.loads(original)
            payload["result"]["blocked_reasons"] = ["tampered"]
            manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _git(worktree, "add", "-A")
            _git(worktree, "commit", "-m", f"{task_id}: post-approval tamper")
            manifest_path.write_text(original, encoding="utf-8")

            with m1sup._fixture_root(root):
                summary = swarm._step_merge(m1sup._supervise_args(worktrees))

            self.assertEqual(summary["merged"], [])
            reasons = {item["reason"] for item in summary["refused"]}
            self.assertTrue(
                reasons & {"manifest_content_changed_after_review", "post_review_commits_present"},
                summary,
            )


class ReapStatePreservationTest(unittest.TestCase):
    """Claude #2: blocked/@human and approved states survive lease expiry."""

    def _expire_claim_on_state(self, tmp: str, state: str, folder: str) -> tuple[Path, dict]:
        root = Path(tmp) / "repo"
        task_id = "T973"
        m1sup.M1SupervisorTests._scaffold_task_repo(
            m1sup.M1SupervisorTests("test_reap_reopens_expired_claim_without_blocking"),
            root,
            task_id,
            state=state,
            folder=folder,
        )
        claim = swarm_claims.claim_task(
            root,
            "origin",
            task_id,
            session_id="worker-x",
            branch=f"{task_id}_task",
            ttl_seconds=0,
        )
        assert claim.ok
        with m1sup._fixture_root(root):
            summary = swarm._step_reap(m1sup._supervise_args(Path(tmp) / "worktrees"))
        return root, summary

    def test_blocked_task_keeps_human_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, summary = self._expire_claim_on_state(tmp, "blocked", "blocked")
            self.assertEqual(summary["reopened"], [])
            self.assertEqual(summary["preserved"], [{"task_id": "T973", "state": "blocked"}])
            self.assertNotIn("T973", swarm_claims.read_claims(root, "origin"))
            text = (root / ".orchestrator/blocked/T973_task.md").read_text(encoding="utf-8")
            self.assertIn("- State: blocked", text)

    def test_ready_for_review_keeps_approval_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, summary = self._expire_claim_on_state(
                tmp, "ready_for_review", "ready_for_review"
            )
            self.assertEqual(summary["reopened"], [])
            self.assertEqual(
                summary["preserved"], [{"task_id": "T973", "state": "ready_for_review"}]
            )
            text = (root / ".orchestrator/ready_for_review/T973_task.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- State: ready_for_review", text)


class MockContainmentTest(unittest.TestCase):
    """Codex F11: mock writes obey task ownership and the hard denylist."""

    def _run_mock(self, tmp: str, target: str) -> tuple[int, Path]:
        import argparse
        import contextlib
        import io
        import os
        from unittest import mock as umock

        root = Path(tmp) / "repo"
        m1sup.M1SupervisorTests._scaffold_task_repo(
            m1sup.M1SupervisorTests("test_reap_reopens_expired_claim_without_blocking"),
            root,
            "T974",
            state="active",
            folder="active",
        )
        write_text(
            root,
            ".orchestrator/mock_transcripts/T974.json",
            json.dumps(
                {
                    "schema_version": "research_swarm.mock_transcript.v1",
                    "actions": [{"write": target, "content": "x"}],
                    "returncode": 0,
                    "stdout": "",
                }
            ),
        )
        args = argparse.Namespace(
            task_id="T974",
            remote="origin",
            base_branch="main",
            codex_model=None,
            codex_sandbox="workspace-write",
            unattended=False,
            skip_executor=False,
            force_deps=False,
            max_worker_seconds=0,
            repair_context=None,
            create_pr=False,
            final_state="ready_for_review",
            executor_backend="mock",
            record_session=False,
            i_accept_full_access=False,
        )
        stdout = io.StringIO()
        with (
            umock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
            umock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            code = swarm.cmd_run_task(args)
        return code, root

    def test_mock_cannot_write_outside_allowed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run_mock(tmp, "contracts/framework.json")
            self.assertEqual(code, 1)
            self.assertEqual(
                json.loads((root / "contracts/framework.json").read_text(encoding="utf-8"))
                .get("framework_version"),
                "v1",
            )

    def test_mock_cannot_write_control_plane_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, root = self._run_mock(
                tmp, "reports/status/swarm_runs/T974_forged.json"
            )
            self.assertEqual(code, 1)
            self.assertFalse(
                (root / "reports/status/swarm_runs/T974_forged.json").exists()
            )


class ReconcileRegressionTest(unittest.TestCase):
    """Verification-pass BLOCKER: verified must bind to the LATEST intent."""

    def test_unverified_intent_after_verified_one_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            task_id = "T975"
            _, worktree, _ = m1sup.M1SupervisorTests._prepare_approved_branch(
                m1sup.M1SupervisorTests("test_full_happy_path_claims_runs_judges_merges_and_cleans"),
                root,
                worktrees,
                task_id,
            )
            branch_tip = _git(worktree, "rev-parse", "HEAD").stdout.strip()
            base_tip = _git(root, "rev-parse", "HEAD").stdout.strip()
            # verified first attempt, then a LATER intent that merged but
            # never verified (simulated: branch already reachable from base
            # after we ff-merge manually)
            for event in (
                {"event": "merge_started", "task_id": task_id, "branch": f"{task_id}_task", "pre_merge_sha": base_tip},
                {"event": "merge_verified", "task_id": task_id, "pre_merge_sha": base_tip, "post_merge_sha": branch_tip},
                {"event": "merge_started", "task_id": task_id, "branch": f"{task_id}_task", "pre_merge_sha": base_tip},
            ):
                swarm_events_mod.append_event(root, event, actor_session="fixture")
            _git(root, "merge", "--ff-only", f"{task_id}_task")

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
            import swarm_reconcile

            payload = swarm_reconcile.reconcile(root)
            classes = {item["class"] for item in payload["findings"]}
            self.assertIn("unverified_merge_in_base", classes, payload["findings"])


class HeartbeatTest(unittest.TestCase):
    def test_heartbeat_renews_during_blocking_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            m1sup.M1SupervisorTests._scaffold_task_repo(
                m1sup.M1SupervisorTests("test_reap_reopens_expired_claim_without_blocking"),
                root,
                "T976",
                state="active",
                folder="active",
            )
            claim = swarm_claims.claim_task(
                root,
                "origin",
                "T976",
                session_id=swarm._ACTOR_SESSION_ID,
                branch="T976_task",
                ttl_seconds=4,  # interval = max(1, min(1, 1)) = 1s
            )
            self.assertTrue(claim.ok)
            with swarm._lease_heartbeat(repo=root, remote="origin", task_id="T976"):
                time.sleep(2.5)
            state = swarm_claims.read_claims(root, "origin")["T976"]
            self.assertGreater(state.lease_id, 1)
            self.assertFalse(
                state.expired(now=dt.datetime.now(tz=dt.timezone.utc))
            )


if __name__ == "__main__":
    unittest.main()
