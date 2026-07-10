"""Regression tests for the M3a fix-round-2 bindings (verification-pass findings).

Each test pins a specific hole the dual-vendor fix-verification pass surfaced:
falsify math-domain crash, computational->experiment-lock, program-wide
amendment cap + rollback, author-supplied --task-kind, self-referential
verification commands, interpretation self-scope, modeling proposition->outcome,
and the claim-artifact git-tracked purity guard.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import sys
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (  # noqa: E402
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_text,
)
from test_m3a_science_gates import _valid_claim, _write_claims, _write_lock, _reasons  # noqa: E402
from test_m3a_modeling_battery import _active_lock  # noqa: E402

quality_gates = load_quality_gates_module()
swarm = load_swarm_module()

_SCRIPTS = str((_TESTS_ROOT.parent / "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import falsify_claims  # noqa: E402
import swarm_taskfile  # noqa: E402
import swarm_events  # noqa: E402


class FalsifyMathDomainTest(unittest.TestCase):
    def test_math_domain_and_complex_degrade_to_error_not_crash(self) -> None:
        for expr, point in [
            ("sqrt(x)", {"x": -1.0}),
            ("log(x)", {"x": 0.0}),
            ("x ** 0.5", {"x": -2.0}),
            ("pow(x, 0.5)", {"x": -2.0}),
        ]:
            with self.subTest(expr=expr):
                with self.assertRaises(falsify_claims.NumericExpressionError):
                    falsify_claims.evaluate_numeric_expression(expr, point)

    def test_spec_over_ill_posed_domain_records_evaluation_error(self) -> None:
        spec = {
            "domain": {"x": [-1.0, 1.0]},
            "seed": 7,
            "inequalities": ["sqrt(x) >= 0"],
            "comparative_statics": [],
        }
        failures = falsify_claims.evaluate_falsification_spec(spec)
        reasons = {f.get("reason") for f in failures}
        self.assertIn("inequality_evaluation_error", reasons)


class VerificationCommandSemanticsTest(unittest.TestCase):
    def test_self_referential_gate_runner_with_flags_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            claim = _valid_claim(root, claim_type="methodological")
            claim["verification_command"] = "python scripts/quality_gates.py --json"
            _write_claims(root, [claim])
            with chdir(root):
                result = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("verification_command_self_referential", _reasons(result))


class InterpretationScopeTest(unittest.TestCase):
    def test_interpretation_cannot_cite_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            base = _valid_claim(root, claim_type="descriptive")
            base["claim_id"] = "C-BASE"
            interp = _valid_claim(root, claim_type="interpretation")
            interp["claim_id"] = "C-INT"
            interp["uncertainty_artifact"] = None
            interp["evidence_scope"] = ["C-INT"]
            _write_claims(root, [base, interp])
            with chdir(root):
                result = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("interpretation_evidence_scope_self_referential", _reasons(result))
            interp["evidence_scope"] = ["C-BASE"]
            _write_claims(root, [base, interp])
            with chdir(root):
                self.assertTrue(quality_gates.gate_claim_evidence_ledger().ok)


class GateCommandTaskKindTest(unittest.TestCase):
    def test_author_supplied_task_kind_is_rejected(self) -> None:
        self.assertEqual(
            swarm_taskfile.gate_command_violation(
                "python scripts/quality_gates.py --task-kind lit_review"
            ),
            "gate_task_kind_author_supplied",
        )

    def test_run_gates_strips_and_overrides_author_task_kind(self) -> None:
        stripped = swarm._strip_cli_option(
            ["python", "scripts/quality_gates.py", "--task-kind", "lit_review"],
            "--task-kind",
        )
        self.assertNotIn("--task-kind", stripped)
        self.assertNotIn("lit_review", stripped)


class ClaimArtifactTrackedTest(unittest.TestCase):
    def test_untracked_claim_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            init_git_fixture_repo(root)
            claim = _valid_claim(root, claim_type="methodological")
            # reports/evidence.txt was written after the fixture commit -> untracked.
            _write_claims(root, [claim])
            with chdir(root):
                red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("artifact_not_tracked_regular_file", _reasons(red))


class ModelingPropositionOutcomeTest(unittest.TestCase):
    def _modeling_repo(self, root: Path) -> None:
        scaffold_runtime_repo(root, mode="modeling")

    def test_locked_proposition_requires_reported_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            self._modeling_repo(root)
            _active_lock(root, "lock_a", "# Experiment lock\n\n- H1: The bound is tight.\n")
            with chdir(root):
                red = quality_gates.gate_prereg_conformance(form="modeling")
            self.assertIn("missing_proposition_outcome", _reasons(red))
            write_text(
                root,
                "reports/paper/index.qmd",
                "# Results\n\nH1 holds on the locked grid.\n",
            )
            write_json(
                root,
                "docs/prereg/outcomes.yaml",
                {
                    "schema_version": "research_swarm.prereg_outcomes.v1",
                    "outcomes": [
                        {"hypothesis_id": "H1", "outcome": "supported", "reported_in": "reports/paper/index.qmd"}
                    ],
                },
            )
            with chdir(root):
                green = quality_gates.gate_prereg_conformance(form="modeling")
            self.assertNotIn("missing_proposition_outcome", _reasons(green))


class AmendmentCapProgramWideTest(unittest.TestCase):
    def _amendment_event(self, phase: str, version: int) -> dict[str, object]:
        return {
            "event": "prereg_amendment",
            "phase": phase,
            "lock_version": version,
            "status": "locked",
        }

    def test_gate_flags_program_over_cap_and_non_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            # Two amendments to 2a and one to 2b == three program amendments (>2).
            for event in [
                self._amendment_event("2a", 2),
                self._amendment_event("2a", 3),
                self._amendment_event("2b", 2),
            ]:
                swarm_events.append_event(root, event)
            with chdir(root):
                over = quality_gates.gate_prereg_lock_coverage()
            self.assertIn("amendment_cap_exceeded_program", _reasons(over))

    def test_gate_flags_version_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            for event in [
                self._amendment_event("2a", 2),
                self._amendment_event("2a", 3),
                self._amendment_event("2a", 2),
            ]:
                swarm_events.append_event(root, event)
            with chdir(root):
                rolled = quality_gates.gate_prereg_lock_coverage()
            self.assertIn("amendment_version_non_monotonic", _reasons(rolled))


if __name__ == "__main__":
    unittest.main()
