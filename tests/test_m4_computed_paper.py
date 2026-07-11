from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from runtime_test_utils import chdir, load_quality_gates_module  # noqa: E402


quality_gates = load_quality_gates_module()


def _load_render_paper_module():
    path = ROOT / "scripts" / "render_paper.py"
    spec = importlib.util.spec_from_file_location("m4_render_paper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_paper = _load_render_paper_module()


FIXTURE_PATHS = (
    "reports/paper/index.qmd",
    "reports/paper/paper_values.json",
    "reports/tables/str_regime_summary.csv",
    "reports/validation/rollup_panel_validation.json",
    "reports/validation/cross_source_reconciliation.json",
    "contracts/claims.yaml",
    "docs/protocol.md",
)


def _copy_computed_paper_fixture(root: Path) -> None:
    for relpath in FIXTURE_PATHS:
        source = ROOT / relpath
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _failure_text(result) -> str:
    return "\n".join(str(item) for item in result.details.get("failures", []))


class M4ComputedPaperTest(unittest.TestCase):
    def test_committed_computed_paper_gate_is_active_and_green(self) -> None:
        result = quality_gates.gate_manuscript_computed_paper()
        self.assertTrue(result.ok, result.details)
        self.assertEqual(result.details["status"], "active")
        self.assertEqual(result.details["value_count"], 16)

    def test_bare_69_14_percent_reintroduced_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            path = root / "reports/paper/index.qmd"
            text = path.read_text(encoding="utf-8")
            text = text.replace("{{value:pre_dencun_mean_str_pct}}", "69.14%", 1)
            path.write_text(text, encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertIn("manuscript_bare_numeric_literal", _failure_text(result))
            self.assertIn("69.14%", _failure_text(result))

    def test_paper_value_70_disagrees_with_independent_69_14_table_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            path = root / "reports/paper/paper_values.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["values"]["pre_dencun_mean_str_pct"]["value"] = 70.0
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertIn("paper_value_mismatch_source", _failure_text(result))
            table_rows = (root / "reports/tables/str_regime_summary.csv").read_text(encoding="utf-8")
            self.assertIn("pre_dencun,Pre-Dencun", table_rows)
            self.assertIn(",69.143000,", table_rows)

    def test_unresolved_bogus_key_fails_gate_and_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            manuscript = root / "reports/paper/index.qmd"
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8") + "\nBogus `{{value:BOGUS}}`.\n",
                encoding="utf-8",
            )
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertIn("manuscript_unresolved_value_key", _failure_text(result))
            payload = json.loads((root / "reports/paper/paper_values.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "paper_value_key_missing:BOGUS"):
                render_paper.resolve_manuscript(manuscript.read_text(encoding="utf-8"), payload)

    def test_claim_literal_99_99_percent_without_paper_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            path = root / "contracts/claims.yaml"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["claims"][0]["manuscript_numeric_literals"].append("99.99%")
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok)
            self.assertIn("claims_paper_values_divergence", _failure_text(result))
            self.assertIn("99.99%", _failure_text(result))


if __name__ == "__main__":
    unittest.main()
