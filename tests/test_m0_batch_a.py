from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import load_swarm_module, load_sweep_module, scaffold_runtime_repo, write_task, write_text


swarm = load_swarm_module()
sweep = load_sweep_module()

import swarm_taskfile as taskfile


class M0BatchATest(unittest.TestCase):
    def test_branch_task_id_round_trip_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T035", slug="l1_rent_panel")

            slug = swarm._slug_from_task_path(task_path, "T035")
            branch = f"T035_{slug}"

            self.assertEqual(branch, "T035_l1_rent_panel")
            self.assertEqual(taskfile.parse_task_id_from_branch(branch), "T035")

        expected = {
            "T035_l1_rent_panel": "T035",
            "T100-fix": "T100",
            "T035": "T035",
            "T03_x": None,
            "T0355_x": None,
            "feature/T035_x": None,
        }
        for branch_name, task_id in expected.items():
            with self.subTest(branch_name=branch_name):
                self.assertEqual(taskfile.parse_task_id_from_branch(branch_name), task_id)

    def test_status_parsing_is_scoped_to_status_section(self) -> None:
        text = "\n".join(
            [
                "# Task T100",
                "",
                "## Status",
                "",
                "- State: backlog",
                "- Last updated: 2026-07-09",
                "",
                "## Notes / Decisions",
                "",
                "- State: done",
                "",
            ]
        )
        no_status = "# Task T100\n\n## Notes / Decisions\n\n- State: done\n"

        self.assertEqual(taskfile.parse_status_value(text, "State"), "backlog")
        self.assertEqual(taskfile.parse_status_value(text, "Last updated"), "2026-07-09")
        self.assertIsNone(taskfile.parse_status_value(no_status, "State"))

    def test_status_update_places_note_inside_notes_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T200")
            text = task_path.read_text(encoding="utf-8")
            text = text.replace(
                "## Notes / Decisions\n\n- 2026-03-29: note\n",
                "\n".join(
                    [
                        "## Notes / Decisions",
                        "",
                        "- State: done",
                        "- Last updated: 1999-01-01",
                        "- 2026-03-29: note",
                        "",
                        "## Follow-up",
                        "",
                        "- State: blocked",
                        "- Last updated: 1998-01-01",
                        "",
                    ]
                ),
            )
            task_path.write_text(text, encoding="utf-8")

            taskfile.update_task_status_and_notes(
                task_path=task_path,
                new_state="ready_for_review",
                note_line="Regression note.",
                allowed_states=swarm.DEFAULT_ALLOWED_STATES,
            )

            updated = task_path.read_text(encoding="utf-8")
            status = taskfile.extract_section(updated, "Status")
            notes = taskfile.extract_section(updated, "Notes / Decisions")
            follow_up = taskfile.extract_section(updated, "Follow-up")
            today = dt.datetime.now(tz=dt.timezone.utc).date().isoformat()

            self.assertIsNotNone(status)
            self.assertIsNotNone(notes)
            self.assertIsNotNone(follow_up)
            self.assertIn("- State: ready_for_review", status)
            self.assertIn(f"- Last updated: {today}", status)
            self.assertIn("- State: done", notes)
            self.assertIn("- Last updated: 1999-01-01", notes)
            self.assertIn(f"- {today}: Regression note.", notes)
            self.assertIn("- State: blocked", follow_up)
            self.assertIn("- Last updated: 1998-01-01", follow_up)
            self.assertLess(updated.index("Regression note."), updated.index("## Follow-up"))

    def test_malformed_task_is_quarantined_and_plan_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            malformed_path = write_text(
                root,
                ".orchestrator/backlog/T300_malformed.md",
                '---\ntask_id: "T300"\n',
            )
            write_task(root, "backlog", "T301", slug="valid")
            contract = swarm.load_framework_contract(root)

            tasks, quarantined = swarm.load_tasks_quarantined(contract)

            self.assertEqual(set(tasks), {"T301"})
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0]["path"], ".orchestrator/backlog/T300_malformed.md")
            self.assertEqual(quarantined[0]["error"], f"missing_yaml_frontmatter:{malformed_path}")
            with self.assertRaisesRegex(ValueError, "missing_yaml_frontmatter"):
                swarm.load_tasks(contract)

            args = argparse.Namespace(remote="origin", base_branch="main")
            stdout = io.StringIO()
            with (
                mock.patch.object(swarm, "_repo_root", return_value=root),
                mock.patch.object(swarm, "claimed_task_ids", return_value=set()),
                contextlib.redirect_stdout(stdout),
            ):
                result = swarm.cmd_plan(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual([item["task_id"] for item in payload["ready"]], ["T301"])
            self.assertEqual(payload["quarantined"], quarantined)

    def test_later_duplicate_task_id_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            first_path = write_task(root, "backlog", "T400", slug="first")
            later_path = write_task(root, "active", "T400", state="active", slug="later")
            contract = swarm.load_framework_contract(root)

            tasks, quarantined = swarm.load_tasks_quarantined(contract)

            self.assertEqual(tasks["T400"].path, first_path)
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0]["path"], ".orchestrator/active/T400_later.md")
            self.assertEqual(
                quarantined[0]["error"],
                f"duplicate_task_id:T400:{first_path}:{later_path}",
            )

    def test_existing_worktree_path_raises_collision_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T500")
            contract = swarm.load_framework_contract(root)
            task = swarm.load_task(task_path, contract)
            worktree_parent = root / "worktrees"
            collision_path = worktree_parent / "wt-T500"
            collision_path.mkdir(parents=True)

            with self.assertRaises(swarm.WorktreeCollisionError) as raised:
                swarm.ensure_worktree(
                    repo=root,
                    task=task,
                    worktree_parent=worktree_parent,
                    base_ref="main",
                )

            self.assertEqual(raised.exception.worktree_path, collision_path)
            self.assertEqual(str(raised.exception), f"worktree_path_already_exists:{collision_path}")

    def test_loop_iteration_system_exit_continues_with_backoff(self) -> None:
        args = argparse.Namespace()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                swarm,
                "_loop_iteration",
                side_effect=SystemExit("worktree_path_already_exists:x"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            failures, backoff = swarm._attempt_loop_iteration(
                args,
                interval_seconds=7,
                consecutive_failures=0,
            )

        self.assertEqual((failures, backoff), (1, 7))
        self.assertIn(
            "[loop] escalation iteration_failed kind=SystemExit detail=worktree_path_already_exists:x",
            stderr.getvalue(),
        )

        with mock.patch.object(swarm, "_loop_iteration", return_value=0):
            self.assertEqual(
                swarm._attempt_loop_iteration(args, interval_seconds=7, consecutive_failures=3),
                (0, 0),
            )
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                swarm._handle_loop_failure(RuntimeError("x"), interval_seconds=5, consecutive_failures=20),
                3600,
            )

    def test_sweep_reports_state_outside_status_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            scaffold_runtime_repo(root)
            task_path = write_task(root, "backlog", "T600")
            text = task_path.read_text(encoding="utf-8")
            text = text.replace("- State: backlog\n", "", 1)
            text = text.replace("## Notes / Decisions\n", "## Notes / Decisions\n\n- State: done\n", 1)
            task_path.write_text(text, encoding="utf-8")

            moves, problems = sweep.plan_sweep(root)

            self.assertEqual(moves, [])
            self.assertEqual(problems, [f"{task_path}:missing_state"])


if __name__ == "__main__":
    unittest.main()
