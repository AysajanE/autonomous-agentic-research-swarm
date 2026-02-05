import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUALITY_GATES_PATH = _REPO_ROOT / "scripts" / "quality_gates.py"


def _load_quality_gates_module():
    spec = importlib.util.spec_from_file_location("quality_gates", _QUALITY_GATES_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


quality_gates = _load_quality_gates_module()


@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _write_text(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mkdir(root: Path, rel: str) -> None:
    (root / rel).mkdir(parents=True, exist_ok=True)


def _write_framework_json(
    root: Path,
    *,
    mode: str,
    features: dict[str, bool] | None = None,
    required_paths: object | None = None,
) -> None:
    data: dict[str, object] = {"mode": mode}
    if features is not None:
        data["features"] = features
    if required_paths is not None:
        data["required_paths"] = required_paths
    _write_text(root, "contracts/framework.json", json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_project_yaml(root: Path, *, mode: str) -> None:
    _write_text(root, "contracts/project.yaml", f"project_name: test\nmode: {mode}\n")


def _scaffold_min_repo(root: Path, *, mode: str) -> None:
    # Base required paths (see gate_repo_structure).
    _write_text(root, "AGENTS.md", "# test\n")
    _write_text(root, "CLAUDE.md", "# test\n")

    _mkdir(root, "contracts")
    _write_text(root, "contracts/AGENTS.md", "# test\n")
    _write_text(root, "contracts/CHANGELOG.md", "# test\n")
    _write_text(root, "contracts/assumptions.md", "# test\n")
    _write_text(root, "contracts/decisions.md", "# test\n")
    _mkdir(root, "contracts/schemas")

    _mkdir(root, "docs")
    _mkdir(root, ".orchestrator")
    _write_text(root, ".orchestrator/AGENTS.md", "# test\n")
    _mkdir(root, ".orchestrator/ready_for_review")
    _write_text(root, ".orchestrator/workstreams.md", "# test\n")

    _mkdir(root, "data")
    _write_text(root, "data/AGENTS.md", "# test\n")
    _mkdir(root, "data/samples")
    _mkdir(root, "data/processed_manifest")

    _mkdir(root, "reports")
    _write_text(root, "reports/AGENTS.md", "# test\n")
    _write_text(root, "reports/catalog.yaml", "reports: []\n")

    _mkdir(root, "scripts")
    _write_text(root, "scripts/AGENTS.md", "# test\n")
    _write_text(root, "scripts/quality_gates.py", "# placeholder\n")

    _mkdir(root, "src")
    _write_text(root, "src/AGENTS.md", "# test\n")
    _mkdir(root, "tests")

    # Mode-specific required paths.
    if mode in {"modeling", "hybrid"}:
        _mkdir(root, "contracts/instances")
        _mkdir(root, "contracts/instances/benchmark_small")
        _mkdir(root, "contracts/experiments")
        _mkdir(root, "src/model")

    if mode in {"empirical", "hybrid"}:
        _write_text(root, "docs/protocol.md", "# protocol\n")
        _write_text(root, "contracts/schemas/panel_schema.yaml", "# schema\n")
        _write_text(root, "contracts/schemas/panel_schema_str_v1.yaml", "panel:\n  fields: []\n")
        _write_text(root, "contracts/schemas/panel_schema_decomp_v1.yaml", "# schema\n")


class GateRepoStructureConfigTest(unittest.TestCase):
    def test_modeling_mode_registry_disabled_allows_no_registry_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold_min_repo(root, mode="modeling")
            _write_project_yaml(root, mode="modeling")
            _write_framework_json(root, mode="modeling", features={"registry": False}, required_paths={"common": [], "modeling": []})

            with _chdir(root):
                result = quality_gates.gate_repo_structure()

            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details.get("registry_enabled"), False)

    def test_registry_enabled_requires_registry_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold_min_repo(root, mode="modeling")
            _write_project_yaml(root, mode="modeling")
            _write_framework_json(root, mode="modeling", features={"registry": True}, required_paths={"common": [], "modeling": []})

            with _chdir(root):
                result = quality_gates.gate_repo_structure()

            self.assertFalse(result.ok)
            missing = set(result.details.get("missing") or [])
            self.assertIn("registry/README.md", missing)
            self.assertIn("registry/AGENTS.md", missing)

    def test_registry_files_only_required_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _scaffold_min_repo(root, mode="modeling")
            _write_project_yaml(root, mode="modeling")
            _mkdir(root, "registry")
            _write_text(root, "registry/README.md", "# test\n")
            _write_text(root, "registry/AGENTS.md", "# test\n")
            _write_text(root, "registry/CHANGELOG.md", "# test\n")

            # Registry enabled, but no specific registry file required.
            _write_framework_json(root, mode="modeling", features={"registry": True}, required_paths={"common": [], "modeling": []})
            with _chdir(root):
                result1 = quality_gates.gate_repo_structure()
            self.assertTrue(result1.ok, result1.details)

            # Now declare a required registry artifact.
            _write_framework_json(
                root,
                mode="modeling",
                features={"registry": True},
                required_paths={"common": ["registry/rollup_registry_v1.csv"], "modeling": []},
            )
            with _chdir(root):
                result2 = quality_gates.gate_repo_structure()
            self.assertFalse(result2.ok)
            missing2 = set(result2.details.get("missing") or [])
            self.assertIn("registry/rollup_registry_v1.csv", missing2)

            _write_text(root, "registry/rollup_registry_v1.csv", "id\n")
            with _chdir(root):
                result3 = quality_gates.gate_repo_structure()
            self.assertTrue(result3.ok, result3.details)


if __name__ == "__main__":
    unittest.main()
