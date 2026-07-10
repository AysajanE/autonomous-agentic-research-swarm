from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import init_git_fixture_repo
from runtime_test_utils import load_swarm_module
from runtime_test_utils import scaffold_runtime_repo
from runtime_test_utils import write_framework_json
from runtime_test_utils import write_run_manifest
from runtime_test_utils import write_task


swarm = load_swarm_module()
swarm_events = swarm.swarm_events
GREEN_GATE = 'python scripts/noop_gate.py'
RED_GATE = 'python -c "raise SystemExit(7)";'


def _run_args(task_id: str) -> argparse.Namespace:
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
    )


def _judge_args(task_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        unattended=False,
        on_fail="blocked",
        note="",
    )


class M1EventTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None

    def _run_task(self, root: Path, task_id: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = swarm.cmd_run_task(_run_args(task_id))
        return result, stdout.getvalue(), stderr.getvalue()

    def _judge_task(self, root: Path, task_id: str) -> int:
        with (
            mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return swarm.cmd_judge_task(_judge_args(task_id))

    def test_append_read_roundtrip_enriches_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = [
                swarm_events.append_event(
                    root,
                    {"event": f"event_{index}", "sequence": index},
                    actor_session="fixture-session",
                )
                for index in range(3)
            ]

            events, malformed_count = swarm_events.read_events(root)

            self.assertEqual(events, written)
            self.assertEqual([event["sequence"] for event in events], [0, 1, 2])
            self.assertEqual(malformed_count, 0)
            for event in events:
                self.assertEqual(event["schema_version"], swarm_events.EVENT_SCHEMA_VERSION)
                self.assertEqual(event["actor_session"], "fixture-session")
                self.assertRegex(event["ts_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_read_events_skips_torn_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            swarm_events.append_event(root, {"event": "complete"})
            journal = root / swarm_events.EVENT_JOURNAL_PATH
            with open(journal, "a", encoding="utf-8") as handle:
                handle.write('{"event":"torn"')

            events, malformed_count = swarm_events.read_events(root)

            self.assertEqual([event["event"] for event in events], ["complete"])
            self.assertEqual(malformed_count, 1)

    def test_escalate_default_file_sink_writes_main_and_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            returned = swarm_events.escalate(
                root,
                {"event": "human_question", "task_id": "T301"},
                actor_session="fixture-session",
            )

            events, malformed_count = swarm_events.read_events(root)
            sink_path = root / "reports/status/events/escalations.jsonl"
            sink_records = [json.loads(line) for line in sink_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(malformed_count, 0)
            self.assertEqual(len(events), 1)
            self.assertEqual(len(sink_records), 1)
            self.assertTrue(events[0]["escalation"])
            self.assertTrue(sink_records[0]["escalation"])
            self.assertEqual(events[0]["event"], "human_question")
            self.assertEqual(returned["delivery"], "delivered:file")

    def test_escalate_command_sink_delivers_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            marker = root / "command-marker.json"
            script = root / "copy_stdin.py"
            script.write_text(
                "import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
                encoding="utf-8",
            )
            target = " ".join(
                [shlex.quote(sys.executable), shlex.quote(str(script)), shlex.quote(str(marker))]
            )
            write_framework_json(
                root,
                overrides={"escalation_sink": {"type": "command", "target": target}},
            )

            delivered = swarm_events.escalate(root, {"event": "judge_block"})

            self.assertEqual(delivered["delivery"], "delivered:command")
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["event"], "judge_block")

            failing_target = " ".join(
                [shlex.quote(sys.executable), "-c", shlex.quote("raise SystemExit(3)")]
            )
            write_framework_json(
                root,
                overrides={
                    "escalation_sink": {"type": "command", "target": failing_target}
                },
            )
            failed = swarm_events.escalate(root, {"event": "loop_iteration_failed"})

            self.assertIn("command_exit:3", failed["delivery_error"])

    def test_run_task_emits_started_finished_and_human_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T310",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            result, _, _ = self._run_task(root, "T310")
            events, _ = swarm_events.read_events(root)

            self.assertEqual(result, 0)
            run_events = [event for event in events if event["task_id"] == "T310"]
            self.assertEqual([event["event"] for event in run_events], ["run_started", "run_finished"])
            self.assertEqual(run_events[1]["status"], "ok")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            task_path = write_task(
                root,
                "active",
                "T311",
                state="active",
                gates=[RED_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            result, _, _ = self._run_task(root, "T311")
            escalation_path = root / "reports/status/events/escalations.jsonl"
            escalations = [
                json.loads(line)
                for line in escalation_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result, 1)
            self.assertEqual(escalations[-1]["event"], "human_question")
            self.assertEqual(escalations[-1]["task_id"], "T311")
            self.assertIn("@human", escalations[-1]["note"])
            self.assertIn("@human", task_path.read_text(encoding="utf-8").splitlines()[-1])

    def test_judge_emits_review_recorded_and_block_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            task_path = write_task(
                root,
                "ready_for_review",
                "T320",
                state="ready_for_review",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            write_run_manifest(root, "T320", task_path=task_path.relative_to(root).as_posix())

            result = self._judge_task(root, "T320")
            events, _ = swarm_events.read_events(root)

            self.assertEqual(result, 0)
            review = [event for event in events if event["event"] == "review_recorded"][-1]
            self.assertEqual(review["task_id"], "T320")
            self.assertEqual(review["outcome"], "approve")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "ready_for_review",
                "T321",
                state="ready_for_review",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            result = self._judge_task(root, "T321")
            escalation_path = root / "reports/status/events/escalations.jsonl"
            escalations = [
                json.loads(line)
                for line in escalation_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(result, 1)
            self.assertEqual(escalations[-1]["event"], "judge_block")
            self.assertEqual(escalations[-1]["outcome"], "block")

    def test_status_json_reports_states_claims_quarantine_and_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "backlog", "T350")
            write_task(root, "active", "T351", state="active")
            write_task(root, "backlog", "T352")
            blocked_path = write_task(root, "blocked", "T353", state="blocked")
            blocked_path.write_text(
                blocked_path.read_text(encoding="utf-8")
                + "- 2026-07-09: @human choose the canonical source\n",
                encoding="utf-8",
            )
            write_task(root, "backlog", "T354", state="invalid")
            init_git_fixture_repo(root)
            write_run_manifest(
                root,
                "T353",
                task_path=blocked_path.relative_to(root).as_posix(),
                state_before="blocked",
                state_after="blocked",
                result_status="blocked",
            )
            now = dt.datetime.now(tz=dt.timezone.utc)
            active_claim = swarm.swarm_claims.claim_task(
                root,
                "origin",
                "T351",
                session_id="active-session",
                branch="T351_task",
                ttl_seconds=60,
                now=now,
            )
            expired_claim = swarm.swarm_claims.claim_task(
                root,
                "origin",
                "T352",
                session_id="expired-session",
                branch="T352_task",
                ttl_seconds=0,
                now=now,
            )
            self.assertTrue(active_claim.ok)
            self.assertTrue(expired_claim.ok)

            stdout = io.StringIO()
            args = argparse.Namespace(remote="origin", no_fetch=True, json=True)
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(stdout),
            ):
                result = swarm.cmd_status(args)
            payload = json.loads(stdout.getvalue())

            self.assertEqual(result, 0)
            self.assertEqual(payload["states"]["active"], ["T351"])
            self.assertEqual(payload["states"]["blocked"], ["T353"])
            self.assertEqual(payload["quarantine_count"], 1)
            self.assertEqual(payload["human_questions"][0]["task_id"], "T353")
            self.assertEqual(payload["non_done_tasks"]["T353"]["last_run_status"], "blocked")
            self.assertEqual(
                payload["non_done_tasks"]["T353"]["blocked_reasons"],
                ["fixture_blocked"],
            )
            self.assertEqual(payload["spend"], "unknown")
            leases = {lease["task_id"]: lease for lease in payload["leases"]}
            self.assertFalse(leases["T351"]["expired"])
            self.assertFalse(leases["T351"]["orphaned"])
            self.assertTrue(leases["T352"]["expired"])
            self.assertTrue(leases["T352"]["orphaned"])

    def test_run_task_tolerates_journal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T360",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            with mock.patch.object(
                swarm_events,
                "append_event",
                side_effect=OSError("fixture journal unavailable"),
            ):
                result, _, stderr = self._run_task(root, "T360")

            self.assertEqual(result, 0)
            self.assertIn("[warn] event journal failed", stderr)
            self.assertIn("fixture journal unavailable", stderr)


if __name__ == "__main__":
    unittest.main()
