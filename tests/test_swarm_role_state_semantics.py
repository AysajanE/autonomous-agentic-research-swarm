import sys
from pathlib import Path
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import load_swarm_module, load_sweep_module, scaffold_runtime_repo, write_task


swarm = load_swarm_module()
sweep = load_sweep_module()


class SwarmRoleStateSemanticsTest(unittest.TestCase):
    def test_operator_role_and_integration_ready_state_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            task_path = write_task(
                root,
                "integration_ready",
                "T900",
                workstream="W9",
                task_kind="ops",
                role="Operator",
                allowed_paths=["reports/status/"],
                disallowed_paths=["src/"],
                outputs=["reports/status/swarm_runs/T900_20260329T000000Z.json"],
                state="integration_ready",
            )

            contract = swarm.load_framework_contract(root)
            task = swarm.load_task(task_path, contract)

            self.assertEqual(task.role, "Operator")
            self.assertEqual(task.state, "integration_ready")
            self.assertTrue(swarm.task_is_integration_ready_eligible(task, contract))

    def test_dependency_requires_done_or_allowlisted_integration_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "integration_ready",
                "T100",
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
                "T110",
                workstream="W8",
                task_kind="model",
                role="Worker",
                dependencies=["T100"],
                integration_ready_dependencies=["T100"],
                allowed_paths=["reports/models/"],
                disallowed_paths=["data/raw/"],
                outputs=["reports/models/demo/run_manifest.json"],
                state="backlog",
                slug="model",
            )

            write_task(
                root,
                "integration_ready",
                "T120",
                workstream="W1",
                task_kind="etl",
                role="Worker",
                allowed_paths=["data/processed/"],
                disallowed_paths=["contracts/"],
                outputs=[
                    "data/processed/growthepie/vendor_daily_rollup_panel.csv",
                    "data/processed_manifest/vendor_daily_rollup_panel_YYYY-MM-DD.json",
                ],
                state="integration_ready",
                slug="etl",
            )
            write_task(
                root,
                "backlog",
                "T130",
                workstream="W4",
                task_kind="metrics",
                role="Worker",
                dependencies=["T120"],
                integration_ready_dependencies=["T120"],
                allowed_paths=["src/analysis/"],
                disallowed_paths=["contracts/"],
                outputs=["src/analysis/metrics_str.py"],
                state="backlog",
                slug="metrics",
            )

            contract = swarm.load_framework_contract(root)
            tasks = swarm.load_tasks(contract)

            self.assertTrue(swarm.dependency_is_satisfied("T100", tasks["T110"], tasks, contract))
            self.assertFalse(swarm.dependency_is_satisfied("T120", tasks["T130"], tasks, contract))

    def test_sweep_plans_projection_move_from_backlog_to_ready_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            task_path = write_task(
                root,
                "backlog",
                "T140",
                workstream="W4",
                task_kind="metrics",
                role="Worker",
                allowed_paths=["src/analysis/"],
                disallowed_paths=["contracts/"],
                outputs=["src/analysis/metrics_str.py"],
                state="ready_for_review",
                slug="projection",
            )

            moves, problems = sweep.plan_sweep(root)

            self.assertEqual(problems, [])
            self.assertEqual(len(moves), 1)
            source, target = moves[0]
            self.assertEqual(source, task_path)
            self.assertEqual(target.parent.name, "ready_for_review")


if __name__ == "__main__":
    unittest.main()
