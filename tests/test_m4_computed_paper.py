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


def _load_reproduce_analysis_module():
    path = ROOT / "scripts" / "reproduce_analysis.py"
    spec = importlib.util.spec_from_file_location("m4_reproduce_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reproduce_analysis = _load_reproduce_analysis_module()


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

    def test_no_manuscript_surface_skips_gate(self) -> None:
        # Finding 2 (activation signal): with NEITHER computed-paper surface on disk
        # (a modeling-only / not-yet-written project), the gate is inactive, not failing.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertTrue(result.ok, result.details)
            self.assertEqual(result.details["status"], "inactive_no_manuscript")
            self.assertTrue(result.details["skipped"])

    def test_partial_manuscript_deletion_keeps_gate_active_and_fails(self) -> None:
        # Finding 2 (consistency): paper_values.json present but index.qmd deleted is a
        # partial-deletion state. The gate must stay ACTIVE (either surface present) and
        # FAIL closed — matching the release perimeter, which uses the same activation
        # signal (_manuscript_surface_present). It must NOT silently skip.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            (root / "reports/paper/index.qmd").unlink()
            with chdir(root):
                result = quality_gates.gate_manuscript_computed_paper()
            self.assertFalse(result.ok, result.details)
            self.assertEqual(result.details["status"], "active")
            self.assertIn("manuscript_computed_paper_missing", _failure_text(result))

    def test_recompute_matches_generator_rounding_on_boundary(self) -> None:
        # Finding 4 (rounding-mode): the gate recomputes with the generator's exact
        # operation, round(float(source), precision). On a rounding boundary where
        # Decimal ROUND_HALF_UP would disagree with Python round(), the gate must ACCEPT
        # the value the generator actually produced (round semantics), not spuriously
        # reject the repo's own output. 2.675 -> round(2.675, 2) == 2.67 (float), whereas
        # Decimal('2.675').quantize(.01, HALF_UP) == 2.68. The generator writes 2.67, so
        # the gate must pass 2.67 and reject the HALF_UP artifact 2.68.
        self.assertEqual(round(2.675, 2), 2.67)  # independent statement of the boundary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy_computed_paper_fixture(root)
            probe_csv = root / "reports/tables/boundary_probe.csv"
            probe_csv.write_text("regime_id,val\nprobe,2.675000\n", encoding="utf-8")
            import hashlib

            probe_sha = hashlib.sha256(probe_csv.read_bytes()).hexdigest()
            values_path = root / "reports/paper/paper_values.json"
            payload = json.loads(values_path.read_text(encoding="utf-8"))
            template = payload["values"]["pre_dencun_mean_str_pct"]
            base_entry = {
                "unit": "ratio",
                "type": template["type"],
                "citation_key": template["citation_key"],
                "source_artifact": "reports/tables/boundary_probe.csv",
                "source_sha256": probe_sha,
                "source_selector": "regime_id=probe;column=val",
                "uncertainty": template["uncertainty"],
            }

            def _run(probe_value: float, probe_display: str):
                payload["values"]["boundary_probe"] = {
                    **base_entry,
                    "value": probe_value,
                    "display": probe_display,
                }
                values_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                with chdir(root):
                    return quality_gates.gate_manuscript_computed_paper()

            # Generator's round() result is accepted…
            ok_result = _run(2.67, "2.67")
            self.assertTrue(ok_result.ok, ok_result.details)
            # …and the HALF_UP artifact (2.68) is rejected as a source mismatch.
            bad_result = _run(2.68, "2.68")
            self.assertFalse(bad_result.ok)
            self.assertIn("paper_value_mismatch_source", _failure_text(bad_result))


class ReproduceAnalysisContentCheckTest(unittest.TestCase):
    # The figure sidecars are content-equivalence-checked (not byte-identity) because
    # float64 reductions differ in their last ULPs across platforms. The comparator must
    # tolerate that round-off while still catching genuine drift and structural changes.
    def _eq(self, a, b):
        return reproduce_analysis._content_equivalent(a, b, "root")

    def test_cross_platform_float_noise_is_tolerated(self) -> None:
        base = {"series": [69.143000000001, 147.014432999998], "label": "STR"}
        noisy = {"series": [69.143000000002, 147.014433000001], "label": "STR"}
        self.assertEqual(self._eq(base, noisy), [])

    def test_identical_payloads_match(self) -> None:
        payload = {"series": [1.0, 2.5, 3.25], "dates": ["2024-03-13"], "n": 3}
        self.assertEqual(self._eq(payload, json.loads(json.dumps(payload))), [])

    def test_real_value_drift_is_caught(self) -> None:
        base = {"series": [69.143, 11.68]}
        drifted = {"series": [69.143, 11.69]}  # 0.01 change — far above float noise
        self.assertTrue(self._eq(base, drifted))

    def test_structural_and_string_differences_are_caught(self) -> None:
        self.assertTrue(self._eq({"series": [1.0, 2.0]}, {"series": [1.0]}))  # length
        self.assertTrue(self._eq({"a": 1.0}, {"b": 1.0}))  # keys
        self.assertTrue(self._eq({"label": "STR"}, {"label": "str"}))  # string exact
        self.assertTrue(self._eq({"post_dencun": True}, {"post_dencun": 1}))  # bool vs int


if __name__ == "__main__":
    unittest.main()
