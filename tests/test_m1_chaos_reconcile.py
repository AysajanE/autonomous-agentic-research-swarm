from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import SWARM_PATH
from runtime_test_utils import attest_containment_fixture
from runtime_test_utils import init_git_fixture_repo
from runtime_test_utils import load_swarm_module
from runtime_test_utils import scaffold_runtime_repo
from runtime_test_utils import write_framework_json
from runtime_test_utils import write_json
from runtime_test_utils import write_task


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
RECONCILE_PATH = SCRIPTS_DIR / "swarm_reconcile.py"
CHAOS_PATH = SCRIPTS_DIR / "swarm_chaos.py"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import swarm_claims
import swarm_events
import swarm_reconcile


swarm = load_swarm_module()
GREEN_GATE = 'python -c "raise SystemExit(0)";'
SLOW_GATE = 'python -c "import time; time.sleep(2)";'
REVIEW_BUNDLE_IMMEDIATE = {
    "run_manifest_dir": "reports/status/swarm_runs",
    "judge_review_dir": "reports/status/reviews",
    "min_separation_seconds": 0,
}


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


class M1ChaosReconcileTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None
        swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()

    def _scaffold_queue(
        self,
        root: Path,
        task_ids: tuple[str, ...],
        *,
        sleeping_task: str | None = None,
        slow_gate_task: str | None = None,
        lease_ttl_seconds: int | None = None,
        review_min_separation_seconds: int = 0,
    ) -> None:
        scaffold_runtime_repo(root)
        overrides: dict[str, object] = {
            "review_bundle": {
                **REVIEW_BUNDLE_IMMEDIATE,
                "min_separation_seconds": review_min_separation_seconds,
            },
        }
        if lease_ttl_seconds is not None:
            overrides["claims"] = {"lease_ttl_seconds": lease_ttl_seconds}
        write_framework_json(root, overrides=overrides)
        for task_id in task_ids:
            output = f"src/{task_id.lower()}_result.txt"
            gate = SLOW_GATE if task_id == slow_gate_task else GREEN_GATE
            write_task(
                root,
                "backlog",
                task_id,
                outputs=[output],
                gates=[gate],
                allowed_paths=["src/"],
            )
            actions: list[dict[str, object]] = []
            if task_id == sleeping_task:
                actions.append({"sleep_seconds": 10})
            actions.append({"write": output, "content": f"{task_id} result\n"})
            write_json(
                root,
                f".orchestrator/mock_transcripts/{task_id}.json",
                {
                    "schema_version": "research_swarm.mock_transcript.v1",
                    "actions": actions,
                    "returncode": 0,
                    "stdout": f"{task_id} mock complete\n",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        init_git_fixture_repo(root)
        attest_containment_fixture(root)

    def _run_supervisor(self, root: Path, worktrees: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["SWARM_REPO_ROOT"] = str(root)
        env["SWARM_EVENT_REPO_ROOT"] = str(root)
        return subprocess.run(
            [
                sys.executable,
                str(SWARM_PATH),
                "supervise",
                "--once",
                "--runner",
                "local",
                "--max-workers",
                "1",
                "--worktree-parent",
                str(worktrees),
                "--remote",
                "origin",
                "--base-branch",
                "main",
                "--executor-backend",
                "mock",
            ],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def _complete_one_task(self, root: Path, worktrees: Path, task_id: str = "T951") -> None:
        self._scaffold_queue(root, (task_id,))
        first = self._run_supervisor(root, worktrees)
        second = self._run_supervisor(root, worktrees)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertTrue((root / f".orchestrator/done/{task_id}_task.md").is_file())

    def _run_chaos(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHAOS_PATH),
                "--repo",
                str(root),
                *extra,
                "--json",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_reconcile_clean_after_supervised_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._complete_one_task(root, Path(tmp) / "worktrees")

            payload = swarm_reconcile.reconcile(root)

            self.assertTrue(payload["ok"], payload["findings"])
            self.assertEqual(payload["findings_count"], 0)
            self.assertEqual(payload["checks"]["task_done_bundle"]["checked"], 1)

    def test_reconcile_detects_done_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._complete_one_task(root, Path(tmp) / "worktrees", "T952")
            for path in (root / "reports/status/reviews").glob("T952_*.json"):
                path.unlink()

            payload = swarm_reconcile.reconcile(root)
            classes = {item["class"] for item in payload["findings"]}

            self.assertFalse(payload["ok"])
            self.assertIn("done_without_approval", classes)

    def test_reconcile_detects_dangling_live_claim_on_done_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._complete_one_task(root, Path(tmp) / "worktrees", "T953")
            claim = swarm_claims.claim_task(
                root,
                "origin",
                "T953",
                session_id="impossible-state-session",
                branch="T953_task",
                journal=lambda event: swarm_events.append_event(root, event),
            )
            self.assertTrue(claim.ok)

            payload = swarm_reconcile.reconcile(root)
            classes = {item["class"] for item in payload["findings"]}

            self.assertFalse(payload["ok"])
            self.assertIn("done_with_live_claim", classes)

    def test_reconcile_detects_deleted_run_finished_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._complete_one_task(root, Path(tmp) / "worktrees", "T954")
            manifest = next((root / "reports/status/swarm_runs").glob("T954_*.json"))
            manifest.unlink()

            payload = swarm_reconcile.reconcile(root)
            classes = {item["class"] for item in payload["findings"]}

            self.assertFalse(payload["ok"])
            self.assertIn("missing_run_manifest", classes)

    def test_chaos_supervisor_kill_recovers_without_duplicate_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._scaffold_queue(
                root,
                ("T955", "T956"),
                slow_gate_task="T955",
                review_min_separation_seconds=60,
            )

            cp = self._run_chaos(
                root,
                "--cycles",
                "3",
                "--kill-supervisor-at-cycle",
                "2",
                "--seed",
                "1",
            )
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            events, malformed = swarm_events.read_events(root)

            self.assertTrue(payload["ok"], payload["errors"])
            self.assertEqual(payload["kills_injected"]["supervisor"], 1)
            self.assertNotIn("blocked", payload["final_task_states"].values(), payload)
            self.assertTrue(payload["serial_merges_ok"])
            self.assertTrue(payload["reconciliation"]["ok"])
            self.assertEqual(malformed, 0)
            claim_created = [
                event["task_id"]
                for event in events
                if event.get("event") == "claim_created"
            ]
            self.assertEqual(claim_created.count("T955"), 1)
            self.assertEqual(claim_created.count("T956"), 1)
            run_finished = {
                event["task_id"]
                for event in events
                if event.get("event") == "run_finished"
            }
            self.assertEqual(run_finished, {"T955", "T956"})

    def test_chaos_worker_kill_reaps_to_backlog_without_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            task_id = "T957"
            self._scaffold_queue(
                root,
                (task_id,),
                sleeping_task=task_id,
                lease_ttl_seconds=1,
            )
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                task = swarm.load_tasks(contract)[task_id]
                claim = swarm._claim_for_dispatch(
                    repo=root,
                    remote="origin",
                    task=task,
                    ttl_seconds=contract.claim_lease_ttl_seconds,
                )
                self.assertTrue(claim.ok)
                swarm._update_task_status_and_notes(
                    task_path=task.path,
                    new_state="active",
                    note_line="Prepared for worker-kill fixture.",
                )
                active_task = swarm.load_task(task.path, contract)
                swarm._move_task_to_state_projection(root, active_task)
                swarm._persist_projection_changes(
                    repo=root,
                    remote="origin",
                    base_branch="main",
                    filenames=[task.path.name],
                    message="fixture: prepare claimed worker",
                    strict=False,
                )
                active_task = swarm.load_tasks(contract)[task_id]
                worktree, _ = swarm.ensure_worktree(
                    repo=root,
                    task=active_task,
                    worktree_parent=Path(tmp) / "prepared-worktrees",
                    base_ref="main",
                )
            live_claim = swarm_claims.read_claims(root, "origin", fetch=False)[task_id]
            self.assertEqual(live_claim.payload["lease_ttl_seconds"], 1)
            self.assertTrue(worktree.is_dir())

            cp = self._run_chaos(
                root,
                "--cycles",
                "1",
                "--kill-worker",
                "--seed",
                "1",
            )
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            payload = json.loads(cp.stdout)
            events, malformed = swarm_events.read_events(root)

            self.assertTrue(payload["worker_kill"]["injected"])
            self.assertTrue(payload["worker_orphan_reaped"])
            self.assertEqual(payload["final_task_states"][task_id], "backlog")
            self.assertEqual(payload["live_claim_count"], 0)
            self.assertEqual(malformed, 0)
            trail = [event for event in events if event.get("task_id") == task_id]
            self.assertTrue(any(event.get("event") == "task_orphaned" for event in trail))
            self.assertFalse(
                any(
                    event.get("status") == "blocked"
                    or event.get("state_before") == "blocked"
                    or "blocked" in event.get("blocked_reasons", [])
                    for event in trail
                )
            )

    def test_mock_sleep_action_rejects_values_over_30_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._scaffold_queue(root, ("T958",))
            write_json(
                root,
                ".orchestrator/mock_transcripts/T958.json",
                {
                    "schema_version": "research_swarm.mock_transcript.v1",
                    "actions": [{"sleep_seconds": 31}],
                    "returncode": 0,
                    "stdout": "must not run\n",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            )
            with _fixture_root(root):
                contract = swarm.load_framework_contract(root)
                task = swarm.load_tasks(contract)["T958"]
                returncode, stdout, _, _ = swarm._run_mock_transcript(repo=root, task=task)

            self.assertEqual(returncode, 1)
            self.assertIn("mock_transcript_sleep_too_long", stdout)

    def test_reconcile_spend_ledger_requires_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._complete_one_task(root, Path(tmp) / "worktrees", "T959")
            manifest = next((root / "reports/status/swarm_runs").glob("T959_*.json"))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(swarm_reconcile.reconcile(root)["ok"])
            self.assertIn("usage", payload)
            payload.pop("usage")
            manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            reconciliation = swarm_reconcile.reconcile(root)
            classes = {item["class"] for item in reconciliation["findings"]}

            self.assertFalse(reconciliation["ok"])
            self.assertIn("missing_usage", classes)


if __name__ == "__main__":
    unittest.main()
