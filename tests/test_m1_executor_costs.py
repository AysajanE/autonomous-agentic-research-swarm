from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import attest_containment_fixture
from runtime_test_utils import init_git_fixture_repo
from runtime_test_utils import load_swarm_module
from runtime_test_utils import scaffold_runtime_repo
from runtime_test_utils import write_framework_json
from runtime_test_utils import write_run_manifest
from runtime_test_utils import write_task


swarm = load_swarm_module()
GREEN_GATE = 'python scripts/noop_gate.py'


def _run_args(
    task_id: str,
    *,
    model: str | None = None,
    record_session: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote="origin",
        base_branch="main",
        executor_backend="mock",
        codex_model=model,
        codex_sandbox="workspace-write",
        unattended=False,
        skip_executor=False,
        record_session=record_session,
        force_deps=False,
        max_worker_seconds=0,
        repair_context=None,
        create_pr=False,
        final_state="ready_for_review",
    )


def _write_transcript(
    root: Path,
    task_id: str,
    *,
    actions: list[dict[str, object]],
    returncode: int = 0,
    stdout: str = "mock executor output\n",
    usage: dict[str, int] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": "research_swarm.mock_transcript.v1",
        "actions": actions,
        "returncode": returncode,
        "stdout": stdout,
    }
    if usage is not None:
        payload["usage"] = usage
    path = root / ".orchestrator" / "mock_transcripts" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_manifest(root: Path, task_id: str) -> tuple[Path, dict[str, object]]:
    paths = sorted((root / "reports/status/swarm_runs").glob(f"{task_id}_*.json"))
    if len(paths) != 1:
        raise AssertionError(f"expected one manifest for {task_id}, found {paths}")
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


@contextlib.contextmanager
def _fixture_root(root: Path):
    with (
        mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
        mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
    ):
        yield


class M1ExecutorCostsTests(unittest.TestCase):
    def setUp(self) -> None:
        swarm._REPO_ROOT_CACHE = None
        swarm._PREFLIGHT_STRICT_SYNC_CACHE.clear()

    def _run_task(self, root: Path, args: argparse.Namespace) -> int:
        with _fixture_root(root), contextlib.redirect_stdout(io.StringIO()):
            return swarm.cmd_run_task(args)

    def test_mock_backend_end_to_end_reaches_ready_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T971",
                state="active",
                allowed_paths=["src/"],
                outputs=["src/result.txt"],
                gates=[GREEN_GATE],
            )
            _write_transcript(
                root,
                "T971",
                actions=[
                    {"write": "src/result.txt", "content": "result\n"},
                    {"set_task_state": "ready_for_review"},
                ],
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T971"))

            self.assertEqual(result, 0)
            _, manifest = _load_manifest(root, "T971")
            self.assertEqual(manifest["task"]["state_after"], "ready_for_review")
            self.assertEqual(manifest["executor"]["tool"], "mock")
            self.assertEqual(manifest["provenance_class"], "executor_run")
            self.assertIsInstance(manifest["usage"]["wall_clock_seconds"], (int, float))
            self.assertEqual(manifest["usage"]["source"], "unavailable")
            self.assertEqual((root / "src/result.txt").read_text(encoding="utf-8"), "result\n")

    def test_mock_transcript_missing_blocks_and_logs_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T972",
                state="active",
                allowed_paths=["README.md"],
                outputs=["README.md"],
                gates=[GREEN_GATE],
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T972"))

            self.assertEqual(result, 1)
            _, manifest = _load_manifest(root, "T972")
            self.assertIn("executor_failed", manifest["result"]["blocked_reasons"])
            log = root / manifest["commands"]["executor_log_path"]
            self.assertIn(
                "mock_transcript_missing:.orchestrator/mock_transcripts/T972.json",
                log.read_text(encoding="utf-8"),
            )

    def test_mock_rejects_relative_and_absolute_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            cases = (
                ("T973", "../evil", temp_root / "evil"),
                ("T974", str(temp_root / "absolute-evil"), temp_root / "absolute-evil"),
            )
            for task_id, action_path, forbidden_path in cases:
                with self.subTest(action_path=action_path):
                    root = temp_root / f"repo-{task_id}"
                    scaffold_runtime_repo(root)
                    write_task(
                        root,
                        "active",
                        task_id,
                        state="active",
                        allowed_paths=["README.md"],
                        outputs=["README.md"],
                        gates=[GREEN_GATE],
                    )
                    _write_transcript(
                        root,
                        task_id,
                        actions=[{"write": action_path, "content": "forbidden\n"}],
                    )
                    init_git_fixture_repo(root)

                    result = self._run_task(root, _run_args(task_id))

                    self.assertEqual(result, 1)
                    self.assertFalse(forbidden_path.exists())
                    _, manifest = _load_manifest(root, task_id)
                    self.assertIn("executor_failed", manifest["result"]["blocked_reasons"])
                    log = root / manifest["commands"]["executor_log_path"]
                    self.assertIn("mock_action_path_forbidden", log.read_text(encoding="utf-8"))

    def test_mock_set_task_state_and_usage_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T975",
                state="active",
                allowed_paths=["README.md"],
                outputs=["README.md"],
                gates=[GREEN_GATE],
            )
            _write_transcript(
                root,
                "T975",
                actions=[{"set_task_state": "ready_for_review"}],
                usage={"input_tokens": 1200, "output_tokens": 345},
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T975"))

            self.assertEqual(result, 0)
            _, manifest = _load_manifest(root, "T975")
            self.assertEqual(manifest["task"]["state_after"], "ready_for_review")
            self.assertEqual(manifest["usage"]["input_tokens"], 1200)
            self.assertEqual(manifest["usage"]["output_tokens"], 345)
            self.assertEqual(manifest["usage"]["source"], "mock_transcript")

    def test_record_session_hash_and_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            stdout = "session output\nwith a second line\n"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "active",
                "T976",
                state="active",
                allowed_paths=["README.md"],
                outputs=["README.md"],
                gates=[GREEN_GATE],
            )
            _write_transcript(
                root,
                "T976",
                actions=[{"set_task_state": "ready_for_review"}],
                stdout=stdout,
                usage={"input_tokens": 10, "output_tokens": 4},
            )
            init_git_fixture_repo(root)

            result = self._run_task(root, _run_args("T976", record_session=True))

            self.assertEqual(result, 0)
            _, manifest = _load_manifest(root, "T976")
            session_path = root / manifest["commands"]["session_path"]
            session = json.loads(session_path.read_text(encoding="utf-8"))
            self.assertEqual(session["schema_version"], "research_swarm.executor_session.v1")
            self.assertEqual(session["backend"], "mock")
            self.assertEqual(session["run_id"], manifest["run_id"])
            self.assertEqual(session["returncode"], 0)
            self.assertEqual(session["stdout_sha256"], hashlib.sha256(stdout.encode()).hexdigest())
            self.assertEqual(session["argv"], ["mock", ".orchestrator/mock_transcripts/T976.json"])

    def test_parse_codex_usage_synthetic_samples(self) -> None:
        # Synthetic parser fixtures; these strings are not claims about a vendor format.
        samples = (
            (
                "analysis complete\ninput tokens: 1,234\noutput tokens: 567\ntokens used: 1,801\n",
                {"input_tokens": 1234, "output_tokens": 567, "source": "parsed"},
            ),
            (
                "tokens used — input: 80_000 output: 9_001\n",
                {"input_tokens": 80000, "output_tokens": 9001, "source": "parsed"},
            ),
        )
        for stdout, expected in samples:
            with self.subTest(stdout=stdout):
                self.assertEqual(swarm._parse_codex_usage(stdout), expected)
        self.assertIsNone(swarm._parse_codex_usage("garbage output\n"))
        self.assertIsNone(swarm._parse_codex_usage("tokens used: 1,000\n"))

    def test_pricing_estimate_and_missing_pricing(self) -> None:
        cases = ((True, 4.0), (False, None))
        for with_pricing, expected_cost in cases:
            with self.subTest(with_pricing=with_pricing):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "repo"
                    scaffold_runtime_repo(root)
                    if with_pricing:
                        write_framework_json(
                            root,
                            overrides={
                                "pricing": {
                                    "fixture-model": {
                                        "input_per_mtok_usd": 2,
                                        "output_per_mtok_usd": 4,
                                    }
                                }
                            },
                        )
                    write_task(
                        root,
                        "active",
                        "T977",
                        state="active",
                        allowed_paths=["README.md"],
                        outputs=["README.md"],
                        gates=[GREEN_GATE],
                    )
                    _write_transcript(
                        root,
                        "T977",
                        actions=[{"set_task_state": "ready_for_review"}],
                        usage={"input_tokens": 1_000_000, "output_tokens": 500_000},
                    )
                    init_git_fixture_repo(root)

                    result = self._run_task(root, _run_args("T977", model="fixture-model"))

                    self.assertEqual(result, 0)
                    _, manifest = _load_manifest(root, "T977")
                    if expected_cost is None:
                        self.assertNotIn("estimated_cost_usd", manifest["usage"])
                        self.assertNotIn("pricing_source", manifest["usage"])
                    else:
                        self.assertEqual(manifest["usage"]["estimated_cost_usd"], expected_cost)
                        self.assertEqual(
                            manifest["usage"]["pricing_source"], "framework_contract"
                        )

    def test_costs_json_aggregates_usage_and_missing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            first_task = write_task(root, "active", "T978", state="active", gates=[GREEN_GATE])
            second_task = write_task(root, "active", "T979", state="active", gates=[GREEN_GATE])
            first_manifest = write_run_manifest(
                root,
                "T978",
                task_path=first_task.relative_to(root).as_posix(),
                workstream="W1",
                usage={
                    "wall_clock_seconds": 1.25,
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "estimated_cost_usd": 0.125,
                    "source": "mock_transcript",
                    "pricing_source": "framework_contract",
                },
            )
            second_manifest = write_run_manifest(
                root,
                "T979",
                task_path=second_task.relative_to(root).as_posix(),
                workstream="W2",
            )
            for manifest_path in (first_manifest, second_manifest):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["executor"]["tool"] = "mock"
                manifest["executor"]["model"] = "fixture-model"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

            stdout = io.StringIO()
            with _fixture_root(root), contextlib.redirect_stdout(stdout):
                result = swarm.cmd_costs(argparse.Namespace(json=True))
            payload = json.loads(stdout.getvalue())

            self.assertEqual(result, 0)
            self.assertEqual(payload["total"]["run_count"], 2)
            self.assertEqual(payload["total"]["wall_clock_seconds"], 1.25)
            self.assertEqual(payload["total"]["input_tokens"], 100)
            self.assertEqual(payload["total"]["output_tokens"], 25)
            self.assertEqual(payload["total"]["estimated_cost_usd"], 0.125)
            self.assertEqual(payload["total"]["runs_without_usage"], 1)
            self.assertEqual(payload["by_task_id"]["T978"]["run_count"], 1)
            self.assertEqual(payload["by_workstream"]["W2"]["runs_without_usage"], 1)
            self.assertEqual(payload["by_model"]["fixture-model"]["run_count"], 2)

    @unittest.skipUnless(hasattr(swarm, "cmd_supervise"), "M1 supervise is not landed")
    def test_supervise_cycle_dispatches_mock_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            worktrees = Path(tmp) / "worktrees"
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T990",
                state="backlog",
                allowed_paths=["src/"],
                outputs=["src/result.txt"],
                gates=[GREEN_GATE],
            )
            _write_transcript(
                root,
                "T990",
                actions=[
                    {"write": "src/result.txt", "content": "supervised\n"},
                    {"set_task_state": "ready_for_review"},
                ],
                usage={"input_tokens": 20, "output_tokens": 5},
            )
            init_git_fixture_repo(root)
            attest_containment_fixture(root)
            args = argparse.Namespace(
                once=True,
                interval_seconds=5,
                runner="local",
                max_workers=1,
                worktree_parent=str(worktrees),
                remote="origin",
                base_branch="main",
                executor_backend="mock",
                codex_model="fixture-model",
                codex_sandbox="workspace-write",
                unattended=False,
                max_worker_seconds=0,
            )

            with _fixture_root(root), contextlib.redirect_stdout(io.StringIO()):
                result = swarm.cmd_supervise(args)

            self.assertEqual(result, 0)
            worktree = worktrees / "wt-T990"
            _, manifest = _load_manifest(worktree, "T990")
            self.assertEqual(manifest["executor"]["tool"], "mock")
            self.assertEqual(manifest["usage"]["source"], "mock_transcript")


if __name__ == "__main__":
    unittest.main()
