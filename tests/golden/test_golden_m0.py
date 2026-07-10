from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from golden.harness import GoldenRepo
from runtime_test_utils import (
    attest_containment_fixture,
    register_historical_exemption,
    chdir,
    load_quality_gates_module,
    load_swarm_module,
    write_json,
    write_text,
)


swarm = load_swarm_module()
quality_gates = load_quality_gates_module()

import swarm_taskfile


GREEN_GATE = 'python scripts/noop_gate.py'


def _only_json(root: Path, rel_dir: str, pattern: str) -> tuple[Path, dict[str, object]]:
    paths = sorted((root / rel_dir).glob(pattern))
    if len(paths) != 1:
        raise AssertionError(f"expected one JSON artifact, found {paths}")
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _sha256_and_bytes(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _processed_payload(output_rel: str, output: Path, *, as_of: str) -> dict[str, object]:
    sha256, size = _sha256_and_bytes(output)
    return {
        "as_of_utc_date": as_of,
        "inputs": ["data/raw_manifest/source_2026-07-09.json"],
        "transform": {
            "script_path": "src/etl/build.py",
            "git_sha": "0" * 40,
            "command": "python src/etl/build.py",
        },
        "outputs": [{"path": output_rel, "sha256": sha256, "bytes": size}],
    }


class GoldenM0Test(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None

    def test_G01_claim_regex_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task("backlog", "T035", slug="l1_rent_panel")
            repo.write_task("backlog", "T100", slug="fix")
            slugless = repo.write_task("backlog", "T101", slug="placeholder")
            slugless.rename(slugless.with_name("T101.md"))

            expected = (
                ("T035", "l1_rent_panel", "T035_l1_rent_panel"),
                ("T100", "fix", "T100_fix"),
                ("T101", "", "T101_T101"),
            )
            for task_id, slug, expected_branch in expected:
                with self.subTest(task_id=task_id, slug=slug):
                    _, branch = repo.claim_branch_and_worktree(task_id, slug)
                    self.assertEqual(branch, expected_branch)
                    self.assertEqual(swarm_taskfile.parse_task_id_from_branch(branch), task_id)

        for branch in ("T03_x", "T0355_x", "feature/T035_x"):
            with self.subTest(branch=branch):
                self.assertIsNone(swarm_taskfile.parse_task_id_from_branch(branch))

    def test_G02_two_ticks_no_double_claim_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task("backlog", "T201", slug="inflight")
            repo.claim_branch_and_worktree("T201", "inflight")

            for _ in range(2):
                exit_code, summary = repo.run_tick(max_workers=2, dry_run=False)
                self.assertEqual(exit_code, 0)
                self.assertIn("T201", summary["claimed"])
                self.assertEqual(summary["selected"], [])
                self.assertEqual(summary.get("started", []), [])

            branches = repo.git("branch", "--list", "T201_*").splitlines()
            task_worktrees = [
                line
                for line in repo.git("worktree", "list", "--porcelain").splitlines()
                if line == "branch refs/heads/T201_inflight"
            ]
            self.assertEqual(len(branches), 1)
            self.assertEqual(len(task_worktrees), 1)

            repo.write_task("backlog", "T202", slug="next", workstream="W8")
            exit_code, summary = repo.run_tick(max_workers=2, dry_run=False)
            self.assertEqual(exit_code, 0)
            self.assertIn("T201", summary["claimed"])
            self.assertEqual(summary["selected"], ["T202"])
            self.assertEqual([item["task_id"] for item in summary["started"]], ["T202"])

    def test_G03_judge_rejects_fabricated_backfill_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task_path = repo.write_task(
                "ready_for_review",
                "T203",
                state="ready_for_review",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            repo.write_run_manifest(
                "T203",
                task_path=task_path.relative_to(repo.root).as_posix(),
                provenance_class="backfill",
                result_status="ok",
            )

            exit_code, summary = repo.judge("T203")

            self.assertEqual(exit_code, 1)
            self.assertFalse(summary["approved"])
            self.assertEqual(summary["state_after"], "blocked")
            _, review = _only_json(repo.root, "reports/status/reviews", "T203_*.json")
            self.assertIn(
                "provenance_requires_independent_reverification",
                review["checks"]["failures"],
            )
            self.assertEqual(review["decision"]["outcome"], "block")
            self.assertEqual(
                swarm_taskfile.parse_status_value(task_path.read_text(encoding="utf-8"), "State"),
                "blocked",
            )

    def test_G04_judge_rejects_blocked_status_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task_path = repo.write_task(
                "ready_for_review",
                "T204",
                state="ready_for_review",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            repo.write_run_manifest(
                "T204",
                task_path=task_path.relative_to(repo.root).as_posix(),
                provenance_class="executor_run",
                result_status="blocked",
            )

            exit_code, summary = repo.judge("T204")

            self.assertEqual(exit_code, 1)
            self.assertFalse(summary["approved"])
            _, review = _only_json(repo.root, "reports/status/reviews", "T204_*.json")
            self.assertIn("no_passing_run_manifest", review["checks"]["failures"])
            self.assertNotEqual(review["decision"]["outcome"], "approve")

    def test_G05_malformed_task_quarantined_runtime_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            malformed = write_text(
                repo.root,
                ".orchestrator/backlog/T205_malformed.md",
                '---\ntask_id: "T205"\n',
            )
            repo.write_task("backlog", "T206", slug="valid")

            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(repo.root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(stdout),
            ):
                plan_exit = swarm.cmd_plan(argparse.Namespace(remote="origin", base_branch="main"))
            plan = json.loads(stdout.getvalue())
            tick_exit, tick = repo.run_tick(dry_run=True)

            self.assertEqual(plan_exit, 0)
            self.assertEqual(tick_exit, 0)
            for summary in (plan, tick):
                self.assertEqual(len(summary["quarantined"]), 1)
                self.assertEqual(
                    summary["quarantined"][0]["path"],
                    malformed.relative_to(repo.root).as_posix(),
                )
            self.assertEqual([item["task_id"] for item in plan["ready"]], ["T206"])
            self.assertEqual(tick["selected"], ["T206"])

            with chdir(repo.root):
                hygiene = quality_gates.gate_task_hygiene()
            self.assertFalse(hygiene.ok)
            self.assertTrue(
                any("T205_malformed.md:missing_yaml_frontmatter" in item for item in hygiene.details["failures"]),
                hygiene.details,
            )

    def test_G06_frontmatter_tamper_blocked_with_pinned_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task_path = repo.write_task(
                "active",
                "T207",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            weaker_gate = 'python -c "pass";'

            def tamper(**_: object) -> subprocess.CompletedProcess[str]:
                text = task_path.read_text(encoding="utf-8")
                task_path.write_text(text.replace(GREEN_GATE, weaker_gate), encoding="utf-8")
                return subprocess.CompletedProcess(["fake-executor"], 0, "tampered\n", "")

            with (
                mock.patch.object(swarm, "_codex_exec_cmd", return_value=["fake-executor"]),
                mock.patch.object(swarm, "_invoke_executor", side_effect=tamper),
            ):
                exit_code, summary = repo.run_task("T207")

            self.assertEqual(exit_code, 1)
            self.assertIn("frontmatter_tampered", summary["blocked_reasons"])
            _, manifest = _only_json(repo.root, "reports/status/swarm_runs", "T207_*.json")
            self.assertIn("gates", manifest["frontmatter"]["tampered_keys"])
            self.assertEqual(manifest["commands"]["gates"], [GREEN_GATE])
            self.assertEqual(manifest["gates"][0]["command"], GREEN_GATE)

    def test_G07_stale_hash_red_then_green_after_annotated_rebaseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            output_rel = "data/processed/panels/panel.csv"
            output = write_text(repo.root, output_rel, "date,value\n2026-07-09,1\n")
            stale_rel = "data/processed_manifest/panel_2026-07-09.json"
            stale_path = write_json(
                repo.root,
                stale_rel,
                _processed_payload(output_rel, output, as_of="2026-07-09"),
            )
            stale_bytes = stale_path.read_bytes()
            stale_sha256 = hashlib.sha256(stale_bytes).hexdigest()
            output.write_text("date,value\n2026-07-09,2\n", encoding="utf-8")

            with chdir(repo.root):
                stale_result = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(stale_result.ok)
            mismatch = next(
                item for item in stale_result.details["failures"] if item["reason"] == "sha256_mismatch"
            )
            self.assertNotEqual(mismatch["expected"], mismatch["actual"])

            newer_rel = "data/processed_manifest/panel_2026-07-10.json"
            write_json(
                repo.root,
                newer_rel,
                _processed_payload(output_rel, output, as_of="2026-07-10"),
            )
            sidecar_rel = "data/processed_manifest/rebaselines/panel_2026-07-09.json.rebaseline.json"
            write_json(
                repo.root,
                sidecar_rel,
                {
                    "schema_version": "research_swarm.manifest_rebaseline.v1",
                    "rebaseline_of": stale_rel,
                    "original_manifest_sha256": stale_sha256,
                    "mode": "superseded",
                    "provenance_note": "Stale claim superseded by a newer manifest recomputed from disk.",
                    "rebaselined_at_utc": "2026-07-10T00:00:00Z",
                    "superseded_by": newer_rel,
                },
            )

            # a sidecar for a manifest that is NOT on the hash-pinned
            # historical exemption list is refused — remediation is a
            # one-time act, not a general bypass
            with chdir(repo.root):
                unexempted = quality_gates.gate_processed_manifest_hashes()
            self.assertFalse(unexempted.ok)
            self.assertTrue(
                any(
                    item.get("reason") == "invalid_rebaseline"
                    and item.get("actual") == "rebaseline_not_exempted"
                    for item in unexempted.details["failures"]
                ),
                unexempted.details,
            )

            register_historical_exemption(
                repo.root, section="processed_manifests", rel_path=stale_rel
            )
            register_historical_exemption(
                repo.root, section="rebaselines", rel_path=sidecar_rel
            )
            with chdir(repo.root):
                rebaselined = quality_gates.gate_processed_manifest_hashes()
            self.assertTrue(rebaselined.ok, rebaselined.details)
            self.assertIn(
                {"manifest": stale_rel, "mode": "superseded", "superseded_by": newer_rel},
                rebaselined.details["annotations"],
            )
            self.assertEqual(stale_path.read_bytes(), stale_bytes)

    def test_G08_ownership_violation_left_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task(
                "active",
                "T208",
                state="active",
                allowed_paths=["src/"],
                disallowed_paths=["contracts/"],
                outputs=["src/allowed.txt"],
                gates=[GREEN_GATE],
            )

            def write_paths(**_: object) -> subprocess.CompletedProcess[str]:
                write_text(repo.root, "src/allowed.txt", "allowed\n")
                write_text(repo.root, "violating.txt", "violation\n")
                return subprocess.CompletedProcess(["fake-executor"], 0, "wrote paths\n", "")

            with (
                mock.patch.object(swarm, "_codex_exec_cmd", return_value=["fake-executor"]),
                mock.patch.object(swarm, "_invoke_executor", side_effect=write_paths),
            ):
                exit_code, summary = repo.run_task("T208")

            self.assertEqual(exit_code, 1)
            self.assertIn("path_ownership_violation", summary["blocked_reasons"])
            _, manifest = _only_json(repo.root, "reports/status/swarm_runs", "T208_*.json")
            self.assertIn("violating.txt", manifest["ownership"]["uncommitted_violations"])
            committed = repo.git("log", "-1", "--name-only", "--pretty=format:").splitlines()
            self.assertNotIn("violating.txt", committed)
            self.assertIn("?? violating.txt", repo.git("status", "--porcelain").splitlines())

    def test_G09_gate_command_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task(
                "active",
                "T209",
                state="active",
                gates=["curl http://example.invalid"],
                outputs=["README.md"],
            )
            exit_code, summary = repo.run_task("T209", skip_executor=True)
            self.assertEqual(exit_code, 1)
            self.assertIn("gates_failed", summary["blocked_reasons"])
            _, manifest = _only_json(repo.root, "reports/status/swarm_runs", "T209_*.json")
            gate_record = manifest["gates"][0]
            self.assertEqual(
                gate_record["constraint_violation"],
                "gate_interpreter_not_allowlisted:curl",
            )
            self.assertIsNone(gate_record["returncode"])

        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            marker = repo.root / "chained-command-ran"
            chained_gate = (
                f'python -c "raise SystemExit(0)" && '
                f'python -c "open(\'{marker}\', \'w\')";'
            )
            repo.write_task(
                "active",
                "T210",
                state="active",
                gates=[chained_gate],
                outputs=["README.md"],
            )
            exit_code, summary = repo.run_task("T210", skip_executor=True)
            # form policy now rejects the chained gate outright — the chained
            # command never runs, an even stronger guarantee than shlex inertness
            self.assertEqual(exit_code, 1)
            self.assertIn("gates_failed", summary["blocked_reasons"])
            _, manifest = _only_json(repo.root, "reports/status/swarm_runs", "T210_*.json")
            self.assertFalse(marker.exists())
            self.assertIn("&&", manifest["gates"][0]["argv"])
            self.assertTrue(
                str(manifest["gates"][0]["constraint_violation"]).startswith("gate_form_forbidden:")
            )

    def test_G10_loop_survives_collision_and_systemexit(self) -> None:
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
        self.assertEqual(failures, 1)
        self.assertGreater(backoff, 0)
        self.assertIn("[loop] escalation iteration_failed", stderr.getvalue())

        with mock.patch.object(swarm, "_loop_iteration", return_value=0):
            self.assertEqual(
                swarm._attempt_loop_iteration(
                    args,
                    interval_seconds=7,
                    consecutive_failures=failures,
                ),
                (0, 0),
            )

        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task("backlog", "T211", slug="inflight")
            repo.claim_branch_and_worktree("T211", "inflight")
            loop_args = repo.tick_args(max_workers=2, unattended=True, dry_run=False)
            stdout = io.StringIO()
            swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()
            attest_containment_fixture(repo.root)
            with tempfile.TemporaryDirectory() as clean_home:
                with (
                    mock.patch.dict(
                        os.environ,
                        {
                            "SWARM_REPO_ROOT": str(repo.root),
                            "SWARM_UNATTENDED_I_UNDERSTAND": "1",
                            "HOME": clean_home,
                        },
                        clear=False,
                    ),
                    mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                    contextlib.redirect_stdout(stdout),
                ):
                    exit_code = swarm._loop_iteration(loop_args)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn("T211", summary["claimed"])
            self.assertEqual(summary["selected"], [])


if __name__ == "__main__":
    unittest.main()
