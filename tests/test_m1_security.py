"""M1 batch E — §9.4 security preflight + §4.1 handoff namespacing."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    attest_containment_fixture,
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
    write_text,
)


swarm = load_swarm_module()
quality_gates = load_quality_gates_module()
GREEN_GATE = 'python -c "raise SystemExit(0)";'


def _run_args(task_id: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = dict(
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
        executor_backend="codex",
        record_session=False,
        i_accept_full_access=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@contextlib.contextmanager
def _clean_home():
    with tempfile.TemporaryDirectory() as home:
        with mock.patch.dict(os.environ, {"HOME": home}, clear=False):
            yield Path(home)


class ContainmentPreflightTest(unittest.TestCase):
    def _fixture(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        write_task(root, "backlog", "T901", gates=[GREEN_GATE], outputs=["README.md"])
        init_git_fixture_repo(root)
        return root

    def test_unattended_requires_containment_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _clean_home():
            root = self._fixture(tmp)
            with mock.patch.dict(
                os.environ,
                {"SWARM_UNATTENDED_I_UNDERSTAND": "1", "SWARM_REPO_ROOT": str(root)},
                clear=False,
            ), mock.patch.object(swarm, "_REPO_ROOT_CACHE", None):
                with self.assertRaisesRegex(SystemExit, "containment_marker_missing:"):
                    swarm._require_unattended_ack(root)

    def test_readable_credentials_disprove_containment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _clean_home() as home:
            root = self._fixture(tmp)
            attest_containment_fixture(root)
            aws = home / ".aws" / "credentials"
            aws.parent.mkdir(parents=True)
            aws.write_text("[default]\naws_access_key_id=AKIAFIXTURE\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ, {"SWARM_UNATTENDED_I_UNDERSTAND": "1"}, clear=False
            ):
                with self.assertRaisesRegex(
                    SystemExit, "containment_credentials_readable:aws_credentials"
                ):
                    swarm._require_unattended_ack(root)

    def test_vendor_ack_required_and_recordable_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _clean_home():
            root = self._fixture(tmp)
            write_json(
                root,
                ".swarm/containment.json",
                {
                    "schema_version": "research_swarm.containment_marker.v1",
                    "contained": True,
                    "attested_by": "fixture",
                    "attested_at_utc": "2026-07-10T00:00:00Z",
                },
            )
            with mock.patch.dict(
                os.environ, {"SWARM_UNATTENDED_I_UNDERSTAND": "1"}, clear=False
            ):
                with self.assertRaisesRegex(SystemExit, "vendor_policy_ack_missing:"):
                    swarm._require_unattended_ack(root)

            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(stdout),
            ):
                result = swarm.cmd_ack_vendor_policy(
                    argparse.Namespace(
                        vendor="codex",
                        note="unattended use reviewed against vendor policy",
                        acked_by="owner",
                    )
                )
            self.assertEqual(result, 0)
            with mock.patch.dict(
                os.environ, {"SWARM_UNATTENDED_I_UNDERSTAND": "1"}, clear=False
            ):
                swarm._require_unattended_ack(root)  # now passes

    def test_attest_containment_cli_writes_marker_and_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, _clean_home():
            root = self._fixture(tmp)
            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(stdout),
            ):
                result = swarm.cmd_attest_containment(
                    argparse.Namespace(attested_by="owner", note="sandboxed host")
                )
            self.assertEqual(result, 0)
            marker = json.loads((root / ".swarm/containment.json").read_text(encoding="utf-8"))
            self.assertIs(marker["contained"], True)
            self.assertEqual(marker["attested_by"], "owner")
            events_path = root / "reports/status/events/events.jsonl"
            self.assertIn("containment_attested", events_path.read_text(encoding="utf-8"))


class FullAccessOptInTest(unittest.TestCase):
    def _fixture(self, tmp: str, *, allow_network: bool) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        write_task(
            root,
            "active",
            "T902",
            state="active",
            workstream="W1" if allow_network else "W8",
            task_kind="etl" if allow_network else "bridge",
            allow_network=allow_network,
            gates=[GREEN_GATE],
            outputs=["README.md"],
        )
        init_git_fixture_repo(root)
        return root

    def _run(self, root: Path, args: argparse.Namespace) -> int:
        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
            mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
            contextlib.redirect_stdout(stdout),
        ):
            return swarm.cmd_run_task(args)

    def test_full_access_without_flag_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp, allow_network=True)
            with self.assertRaisesRegex(SystemExit, "full_access_requires_double_opt_in"):
                self._run(root, _run_args("T902", codex_sandbox="danger-full-access"))

    def test_full_access_without_task_allow_network_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp, allow_network=False)
            with self.assertRaisesRegex(SystemExit, "full_access_requires_double_opt_in"):
                self._run(
                    root,
                    _run_args(
                        "T902",
                        codex_sandbox="danger-full-access",
                        i_accept_full_access=True,
                    ),
                )

    def test_double_opt_in_proceeds_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp, allow_network=True)
            result = self._run(
                root,
                _run_args(
                    "T902",
                    codex_sandbox="danger-full-access",
                    i_accept_full_access=True,
                ),
            )
            self.assertEqual(result, 0)
            manifest_path = sorted(
                (root / "reports/status/swarm_runs").glob("T902_*.json")
            )[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIs(manifest["executor"]["full_access_opt_in"], True)
            effective = manifest["executor"]["effective_network"]
            self.assertIs(effective["declared_allow_network"], True)
            self.assertEqual(effective["sandbox"], "danger-full-access")

    def test_ordinary_run_records_effective_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp, allow_network=False)
            result = self._run(root, _run_args("T902"))
            self.assertEqual(result, 0)
            manifest_path = sorted(
                (root / "reports/status/swarm_runs").glob("T902_*.json")
            )[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            effective = manifest["executor"]["effective_network"]
            self.assertIs(effective["declared_allow_network"], False)
            self.assertEqual(effective["enforcement"], "codex_sandbox")
            self.assertIs(manifest["executor"]["full_access_opt_in"], False)


class HandoffNamespaceTest(unittest.TestCase):
    def test_handoff_writes_are_namespaced_to_the_task(self) -> None:
        ok, reason = swarm._path_is_allowed(
            path=".orchestrator/handoff/H903_notes.md",
            allowed_paths=["src/"],
            disallowed_paths=[],
            task_file_path=".orchestrator/active/T903_task.md",
            task_id="T903",
        )
        self.assertTrue(ok, reason)

        ok, reason = swarm._path_is_allowed(
            path=".orchestrator/handoff/H999_other_task.md",
            allowed_paths=["src/"],
            disallowed_paths=[],
            task_file_path=".orchestrator/active/T903_task.md",
            task_id="T903",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "handoff_namespace_violation")


class NetworkStringsGateTest(unittest.TestCase):
    def test_curl_gate_in_non_network_workstream_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T904",
                workstream="W8",
                task_kind="bridge",
                gates=["curl https://example.invalid/health"],
            )
            with chdir(root):
                result = quality_gates.gate_network_strings()
            self.assertFalse(result.ok)
            self.assertTrue(
                any("network_string_in_gate" in item for item in result.details["failures"]),
                result.details,
            )

    def test_network_workstream_may_declare_network_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T905",
                workstream="W1",
                allow_network=True,
                gates=["curl https://example.invalid/health"],
            )
            with chdir(root):
                result = quality_gates.gate_network_strings()
            self.assertTrue(result.ok, result.details)

    def test_clean_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(root, "backlog", "T906", gates=[GREEN_GATE])
            with chdir(root):
                result = quality_gates.gate_network_strings()
            self.assertTrue(result.ok, result.details)


if __name__ == "__main__":
    unittest.main()
