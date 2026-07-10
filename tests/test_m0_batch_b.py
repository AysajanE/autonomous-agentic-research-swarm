from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import init_git_fixture_repo
from runtime_test_utils import load_swarm_module
from runtime_test_utils import scaffold_runtime_repo
from runtime_test_utils import write_run_manifest
from runtime_test_utils import write_task


swarm = load_swarm_module()
GREEN_GATE = 'python -c "raise SystemExit(0)";'


def _run_args(task_id: str, *, skip_executor: bool = False, force_deps: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        codex_model=None,
        codex_sandbox="workspace-write",
        unattended=False,
        skip_executor=skip_executor,
        force_deps=force_deps,
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


def _load_only_run_manifest(root: Path, task_id: str) -> tuple[Path, dict[str, object]]:
    paths = sorted((root / "reports/status/swarm_runs").glob(f"{task_id}_*.json"))
    if len(paths) != 1:
        raise AssertionError(f"expected one run manifest for {task_id}, found {paths}")
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


class M0BatchBTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None

    def _run_task(
        self,
        root: Path,
        args: argparse.Namespace,
        *,
        invoke_executor: object | None = None,
    ) -> int:
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            if invoke_executor is None:
                return swarm.cmd_run_task(args)
            invoke_patch = (
                mock.patch.object(swarm, "_invoke_executor", side_effect=invoke_executor)
                if callable(invoke_executor) or isinstance(invoke_executor, BaseException)
                else mock.patch.object(swarm, "_invoke_executor", return_value=invoke_executor)
            )
            with (
                mock.patch.object(swarm, "_codex_exec_cmd", return_value=["fake-executor"]),
                invoke_patch,
            ):
                return swarm.cmd_run_task(args)

    def test_run_task_rejects_unsatisfied_dependencies_before_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "active", "T101", state="active", gates=[GREEN_GATE], outputs=["README.md"])
            write_task(
                root,
                "backlog",
                "T102",
                dependencies=["T101"],
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            head_before = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()

            with self.assertRaisesRegex(SystemExit, "dependencies_unsatisfied:T102:T101"):
                self._run_task(root, _run_args("T102", skip_executor=True))

            head_after = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            self.assertEqual(head_after, head_before)
            self.assertFalse((root / "reports/status/swarm_runs").exists())
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout
            self.assertEqual(status, "")

    def test_run_task_force_deps_records_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(root, "active", "T111", state="active", gates=[GREEN_GATE], outputs=["README.md"])
            write_task(
                root,
                "backlog",
                "T112",
                dependencies=["T111"],
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T112", skip_executor=True, force_deps=True))

            self.assertEqual(result, 0)
            _, manifest = _load_only_run_manifest(root, "T112")
            self.assertEqual(
                manifest["overrides"],
                {"force_deps": True, "unsatisfied_dependencies": ["T111"]},
            )

    def test_skip_executor_writes_manual_operator_v2_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T121",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T121", skip_executor=True))

            self.assertEqual(result, 0)
            _, manifest = _load_only_run_manifest(root, "T121")
            self.assertEqual(manifest["schema_version"], "research_swarm.runtime_run_manifest.v2")
            self.assertEqual(manifest["provenance_class"], "manual_operator")
            self.assertIsNone(manifest["commands"]["executor_log_path"])
            self.assertIsNone(manifest["commands"]["executor_log_sha256"])

    def test_executor_run_writes_durable_hashed_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T131",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            completed = subprocess.CompletedProcess(["fake-executor"], 0, "executor output\n", "")

            result = self._run_task(root, _run_args("T131"), invoke_executor=completed)

            self.assertEqual(result, 0)
            _, manifest = _load_only_run_manifest(root, "T131")
            self.assertEqual(manifest["provenance_class"], "executor_run")
            log_path = root / manifest["commands"]["executor_log_path"]
            self.assertEqual(log_path, root / "reports/status/swarm_runs/logs" / f"{manifest['run_id']}.log")
            self.assertEqual(log_path.read_text(encoding="utf-8"), "executor output\n")
            self.assertEqual(
                manifest["commands"]["executor_log_sha256"],
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
            )

    def test_executor_log_is_size_capped_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T141",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            completed = subprocess.CompletedProcess(["fake-executor"], 0, "x" * (300 * 1024), "")

            result = self._run_task(root, _run_args("T141"), invoke_executor=completed)

            self.assertEqual(result, 0)
            _, manifest = _load_only_run_manifest(root, "T141")
            log_path = root / manifest["commands"]["executor_log_path"]
            self.assertLessEqual(log_path.stat().st_size, 132 * 1024)
            self.assertIn(b"...[truncated 176128 bytes]...", log_path.read_bytes())
            self.assertEqual(
                manifest["commands"]["executor_log_sha256"],
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
            )

    def test_frontmatter_tampering_blocks_and_executes_pinned_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            task_path = write_task(
                root,
                "active",
                "T151",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            tampered_gate = 'python -c "raise SystemExit(9)";'

            def tamper_frontmatter(**_: object) -> subprocess.CompletedProcess[str]:
                text = task_path.read_text(encoding="utf-8")
                task_path.write_text(text.replace(GREEN_GATE, tampered_gate), encoding="utf-8")
                return subprocess.CompletedProcess(["fake-executor"], 0, "changed frontmatter\n", "")

            result = self._run_task(root, _run_args("T151"), invoke_executor=tamper_frontmatter)

            self.assertEqual(result, 1)
            _, manifest = _load_only_run_manifest(root, "T151")
            self.assertIn("frontmatter_tampered", manifest["result"]["blocked_reasons"])
            self.assertTrue(manifest["frontmatter"]["tampered"])
            self.assertIn("gates", manifest["frontmatter"]["tampered_keys"])
            self.assertEqual(manifest["commands"]["gates"], [GREEN_GATE])
            self.assertEqual(manifest["gates"][0]["command"], GREEN_GATE)
            self.assertEqual(manifest["gates"][0]["returncode"], 0)

    def test_ownership_violation_is_left_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T161",
                state="active",
                allowed_paths=["src/"],
                disallowed_paths=["contracts/"],
                outputs=["src/allowed.txt"],
                gates=[GREEN_GATE],
            )
            init_git_fixture_repo(root)

            def write_owned_and_violating(**_: object) -> subprocess.CompletedProcess[str]:
                (root / "src").mkdir(parents=True, exist_ok=True)
                (root / "src/allowed.txt").write_text("allowed\n", encoding="utf-8")
                (root / "violating.txt").write_text("violation\n", encoding="utf-8")
                subprocess.run(["git", "add", "--", "violating.txt"], cwd=root, check=True)
                return subprocess.CompletedProcess(["fake-executor"], 0, "wrote files\n", "")

            result = self._run_task(root, _run_args("T161"), invoke_executor=write_owned_and_violating)

            self.assertEqual(result, 1)
            manifest_path, manifest = _load_only_run_manifest(root, "T161")
            self.assertIn("path_ownership_violation", manifest["result"]["blocked_reasons"])
            self.assertEqual(manifest["ownership"]["uncommitted_violations"], ["violating.txt"])

            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout
            self.assertIn("?? violating.txt", status)
            committed_paths = subprocess.run(
                ["git", "log", "-1", "--name-only", "--pretty=format:"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertIn("src/allowed.txt", committed_paths)
            self.assertIn(manifest_path.relative_to(root).as_posix(), committed_paths)
            self.assertTrue(any(path.startswith("reports/status/swarm_runs/logs/T161_") for path in committed_paths))
            self.assertNotIn("violating.txt", committed_paths)

    def test_judge_requires_passing_executor_provenance(self) -> None:
        cases = (
            ("backfill", "ok", "provenance_requires_independent_reverification", False),
            ("executor_run", "blocked", "no_passing_run_manifest", False),
            ("executor_run", "ok", None, True),
        )
        for index, (provenance, status, expected_failure, approved) in enumerate(cases, start=1):
            with self.subTest(provenance=provenance, status=status):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "repo"
                    scaffold_runtime_repo(root)
                    task_id = f"T17{index}"
                    task_path = write_task(
                        root,
                        "ready_for_review",
                        task_id,
                        state="ready_for_review",
                        gates=[GREEN_GATE],
                        outputs=["README.md"],
                    )
                    init_git_fixture_repo(root)
                    write_run_manifest(
                        root,
                        task_id,
                        task_path=task_path.relative_to(root).as_posix(),
                        provenance_class=provenance,
                        result_status=status,
                    )

                    stdout = io.StringIO()
                    with (
                        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                        contextlib.redirect_stdout(stdout),
                    ):
                        result = swarm.cmd_judge_task(_judge_args(task_id))

                    self.assertEqual(result, 0 if approved else 1)
                    review_paths = sorted((root / "reports/status/reviews").glob(f"{task_id}_*.json"))
                    self.assertEqual(len(review_paths), 1)
                    review = json.loads(review_paths[0].read_text(encoding="utf-8"))
                    self.assertEqual(review["decision"]["outcome"] == "approve", approved)
                    if expected_failure is None:
                        self.assertEqual(review["checks"]["failures"], [])
                        self.assertEqual(review["task"]["state_after"], "done")
                    else:
                        self.assertIn(expected_failure, review["checks"]["failures"])

    def test_timeout_persists_partial_executor_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T181",
                state="active",
                gates=[GREEN_GATE],
                outputs=["README.md"],
            )
            init_git_fixture_repo(root)
            timeout = subprocess.TimeoutExpired(["fake-executor"], 1)
            timeout.stdout = "partial output"

            result = self._run_task(root, _run_args("T181"), invoke_executor=timeout)

            self.assertEqual(result, 1)
            _, manifest = _load_only_run_manifest(root, "T181")
            self.assertIn("executor_timeout", manifest["result"]["blocked_reasons"])
            log_path = root / manifest["commands"]["executor_log_path"]
            self.assertEqual(log_path.read_text(encoding="utf-8"), "partial output")
            self.assertEqual(
                manifest["commands"]["executor_log_sha256"],
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
