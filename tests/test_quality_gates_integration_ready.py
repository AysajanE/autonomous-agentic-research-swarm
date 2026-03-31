import sys
from pathlib import Path
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import chdir, load_quality_gates_module, scaffold_runtime_repo, write_task


quality_gates = load_quality_gates_module()


class IntegrationReadyGateTest(unittest.TestCase):
    def test_gate_rejects_integration_ready_without_downstream_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "integration_ready",
                "T200",
                workstream="W8",
                task_kind="bridge",
                role="Worker",
                allowed_paths=["contracts/instances/"],
                disallowed_paths=["data/raw/"],
                outputs=["contracts/instances/demo/manifest.yaml"],
                state="integration_ready",
                slug="bridge",
            )

            with chdir(root):
                result = quality_gates.gate_integration_ready_policy()

            self.assertFalse(result.ok)
            failures = result.details.get("failures") or []
            self.assertTrue(any("integration_ready_missing_downstream_allowlist" in failure for failure in failures))

    def test_gate_rejects_allowlist_not_present_in_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "integration_ready",
                "T210",
                workstream="W8",
                task_kind="bridge",
                role="Worker",
                allowed_paths=["contracts/instances/"],
                disallowed_paths=["data/raw/"],
                outputs=["contracts/instances/demo/manifest.yaml"],
                state="integration_ready",
                slug="bridge",
            )
            write_task(
                root,
                "backlog",
                "T211",
                workstream="W8",
                task_kind="model",
                role="Worker",
                dependencies=[],
                integration_ready_dependencies=["T210"],
                allowed_paths=["reports/models/"],
                disallowed_paths=["data/raw/"],
                outputs=["reports/models/demo/results.json"],
                state="backlog",
                slug="model",
            )

            with chdir(root):
                result = quality_gates.gate_integration_ready_policy()

            self.assertFalse(result.ok)
            failures = result.details.get("failures") or []
            self.assertTrue(any("integration_ready_dependency_not_in_dependencies:T210" in failure for failure in failures))

    def test_gate_accepts_allowlisted_bridge_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "integration_ready",
                "T220",
                workstream="W8",
                task_kind="bridge",
                role="Worker",
                allowed_paths=["contracts/instances/"],
                disallowed_paths=["data/raw/"],
                outputs=["contracts/instances/demo/manifest.yaml"],
                state="integration_ready",
                slug="bridge",
            )
            write_task(
                root,
                "backlog",
                "T221",
                workstream="W8",
                task_kind="model",
                role="Worker",
                dependencies=["T220"],
                integration_ready_dependencies=["T220"],
                allowed_paths=["reports/models/"],
                disallowed_paths=["data/raw/"],
                outputs=["reports/models/demo/results.json"],
                state="backlog",
                slug="model",
            )

            with chdir(root):
                result = quality_gates.gate_integration_ready_policy()

            self.assertTrue(result.ok, result.details)


if __name__ == "__main__":
    unittest.main()
