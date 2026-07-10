from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable
from unittest import mock

from runtime_test_utils import (
    init_git_fixture_repo,
    load_swarm_module,
    scaffold_runtime_repo,
    write_run_manifest,
    write_task,
)


swarm = load_swarm_module()


class GoldenRepo:
    def __init__(self, root: Path):
        self._root = root

    @classmethod
    def create(cls, tmp_dir: str | Path) -> "GoldenRepo":
        root = Path(tmp_dir).resolve() / "repo"
        scaffold_runtime_repo(root)
        init_git_fixture_repo(root)
        return cls(root)

    @property
    def root(self) -> Path:
        return self._root

    def write_task(self, folder: str, task_id: str, **kwargs: Any) -> Path:
        return write_task(self.root, folder, task_id, **kwargs)

    def write_run_manifest(self, task_id: str, **kwargs: Any) -> Path:
        return write_run_manifest(self.root, task_id, **kwargs)

    def tick_args(self, **overrides: Any) -> argparse.Namespace:
        values = {
            "planner": "heuristic",
            "runner": "local",
            "tmux_session": "golden",
            "max_workers": 1,
            "worktree_parent": str(self.root.parent),
            "remote": "origin",
            "base_branch": "main",
            "codex_model": None,
            "codex_sandbox": "workspace-write",
            "unattended": False,
            "max_worker_seconds": 0,
            "create_pr": False,
            "final_state": "ready_for_review",
            "dry_run": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def run_task_args(self, task_id: str, **overrides: Any) -> argparse.Namespace:
        values = {
            "task_id": task_id,
            "remote": "origin",
            "base_branch": "main",
            "codex_model": None,
            "codex_sandbox": "workspace-write",
            "unattended": False,
            "skip_executor": False,
            "force_deps": False,
            "max_worker_seconds": 0,
            "repair_context": None,
            "create_pr": False,
            "final_state": "ready_for_review",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def judge_args(self, task_id: str, **overrides: Any) -> argparse.Namespace:
        values = {
            "task_id": task_id,
            "remote": "origin",
            "base_branch": "main",
            "unattended": False,
            "on_fail": "blocked",
            "note": "",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run_command(
        self,
        command: Callable[[argparse.Namespace], int],
        args: argparse.Namespace,
    ) -> tuple[int, dict[str, Any]]:
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(self.root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = command(args)
        summary = json.loads(stdout.getvalue())
        return exit_code, summary

    def run_tick(self, **overrides: Any) -> tuple[int, dict[str, Any]]:
        return self._run_command(swarm.cmd_tick, self.tick_args(**overrides))

    def run_task(self, task_id: str, **overrides: Any) -> tuple[int, dict[str, Any]]:
        return self._run_command(swarm.cmd_run_task, self.run_task_args(task_id, **overrides))

    def judge(self, task_id: str, **overrides: Any) -> tuple[int, dict[str, Any]]:
        return self._run_command(swarm.cmd_judge_task, self.judge_args(task_id, **overrides))

    def git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def claim_branch_and_worktree(self, task_id: str, slug: str) -> tuple[Path, str]:
        filename = f"{task_id}_{slug}.md" if slug else f"{task_id}.md"
        paths = sorted((self.root / ".orchestrator").glob(f"*/{filename}"))
        if len(paths) != 1:
            raise ValueError(f"expected_one_task_path:{task_id}:{slug}:{paths}")
        contract = swarm.load_framework_contract(self.root)
        task = swarm.load_task(paths[0], contract)
        return swarm.ensure_worktree(
            repo=self.root,
            task=task,
            worktree_parent=self.root.parent,
            base_ref="main",
        )
