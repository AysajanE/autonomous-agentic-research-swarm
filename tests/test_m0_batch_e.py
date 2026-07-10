from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import load_swarm_module


swarm = load_swarm_module()


class ConstrainedGateExecutionTest(unittest.TestCase):
    def _run(self, gates: list[str], **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return swarm._run_gates(Path(tmp), gates, **kwargs)

    def test_green_python_gate_passes(self) -> None:
        ok, outputs = self._run(['python -c "raise SystemExit(0)";'])
        self.assertTrue(ok, outputs)
        self.assertEqual(outputs[0]["returncode"], 0)
        self.assertIsNone(outputs[0]["constraint_violation"])

    def test_non_allowlisted_interpreter_is_refused(self) -> None:
        ok, outputs = self._run(["curl http://example.invalid"])
        self.assertFalse(ok)
        self.assertEqual(
            outputs[0]["constraint_violation"],
            "gate_interpreter_not_allowlisted:curl",
        )
        self.assertIsNone(outputs[0]["returncode"])

    def test_path_qualified_interpreter_is_refused(self) -> None:
        for gate in ("/usr/bin/nonexistent-shell -c evil", "./python -c evil", "/tmp/python -c evil"):
            ok, outputs = self._run([gate])
            self.assertFalse(ok)
            self.assertTrue(
                str(outputs[0]["constraint_violation"]).startswith(
                    "gate_interpreter_path_qualified:"
                ),
                outputs[0],
            )

    def test_shell_metacharacters_have_no_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "pwned"
            gate = f'python -c "raise SystemExit(0)" && python -c "open(\'{marker}\', \'w\')"'
            ok, outputs = self._run([gate])
            self.assertFalse(marker.exists(), "shell chaining must not execute")
            self.assertEqual(outputs[0]["argv"][0], "python")
            self.assertIn("&&", outputs[0]["argv"])  # inert argument, not an operator

    def test_credentials_are_stripped_from_gate_env(self) -> None:
        gate = (
            "python -c \"import os, sys;"
            " sys.exit(1 if os.environ.get('SECRET_TOKEN') else 0)\";"
        )
        with mock.patch.dict(os.environ, {"SECRET_TOKEN": "leaked"}, clear=False):
            ok, outputs = self._run([gate])
        self.assertTrue(ok, outputs)

    def test_gate_timeout_is_enforced_and_recorded(self) -> None:
        gate = 'python -c "import time; time.sleep(30)";'
        ok, outputs = self._run([gate], timeout_seconds=1)
        self.assertFalse(ok)
        self.assertTrue(outputs[0]["timed_out"])
        self.assertIsNone(outputs[0]["returncode"])

    def test_effective_network_state_is_recorded_honestly(self) -> None:
        gate = (
            "python -c \"import socket, sys\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "    sys.exit(7)\n"
            "except OSError:\n"
            "    sys.exit(0)\";"
        )
        ok, outputs = self._run([gate])
        record = outputs[0]
        if record["network_disabled"]:
            # Enforcement claimed -> it must be real: the connect attempt fails.
            self.assertTrue(ok, record)
            self.assertNotEqual(record["network_disable_method"], "none")
        else:
            # No enforcement available -> the record must say so.
            self.assertEqual(record["network_disable_method"], "none")

    def test_output_is_capped_head_and_tail(self) -> None:
        gate = 'python -c "print(\'x\' * 100000)";'
        ok, outputs = self._run([gate])
        self.assertTrue(ok)
        head = outputs[0]["output_head"]
        tail = outputs[0]["output_tail"]
        self.assertLessEqual(len(head), swarm.GATE_OUTPUT_SEGMENT_BYTES)
        self.assertLessEqual(len(tail), swarm.GATE_OUTPUT_SEGMENT_BYTES)

    def test_custom_allowlist_from_contract_is_honored(self) -> None:
        ok, outputs = self._run(
            ['python -c "raise SystemExit(0)";'],
            interpreter_allowlist=("make",),
        )
        self.assertFalse(ok)
        self.assertEqual(
            outputs[0]["constraint_violation"],
            "gate_interpreter_not_allowlisted:python",
        )


if __name__ == "__main__":
    unittest.main()
