import sys
from pathlib import Path
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import chdir, load_quality_gates_module, scaffold_runtime_repo


quality_gates = load_quality_gates_module()


class GateRepoStructureRuntimeTest(unittest.TestCase):
    def test_min_runtime_repo_passes_without_stage5_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root, mode="empirical")

            with chdir(root):
                result = quality_gates.gate_repo_structure()

            self.assertTrue(result.ok, result.details)
            deferred = set(result.details.get("deferred_paths") or [])
            self.assertIn("reports/paper/", deferred)
            self.assertIn("reports/status/", deferred)

    def test_missing_operator_prompt_and_runtime_schema_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root, mode="empirical")

            operator_prompt = root / "docs" / "prompts" / "operator.md"
            run_schema = root / "contracts" / "schemas" / "swarm_run_manifest_v1.yaml"
            if operator_prompt.exists():
                operator_prompt.unlink()
            if run_schema.exists():
                run_schema.unlink()

            with chdir(root):
                result = quality_gates.gate_repo_structure()

            self.assertFalse(result.ok)
            missing = set(result.details.get("missing") or [])
            self.assertIn("docs/prompts/operator.md", missing)
            self.assertIn("contracts/schemas/swarm_run_manifest_v1.yaml", missing)

    def test_empirical_runtime_requires_registry_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root, mode="empirical")

            registry_csv = root / "registry" / "rollup_registry_v1.csv"
            if registry_csv.exists():
                registry_csv.unlink()

            with chdir(root):
                result = quality_gates.gate_repo_structure()

            self.assertFalse(result.ok)
            missing = set(result.details.get("missing") or [])
            self.assertIn("registry/rollup_registry_v1.csv", missing)


if __name__ == "__main__":
    unittest.main()
