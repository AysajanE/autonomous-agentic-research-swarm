from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    chdir,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()
swarm = load_swarm_module()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_lock(root: Path, phase: str, body: str, *, locked: bool = True) -> tuple[Path, str]:
    name = "data_construction.lock.md" if phase == "2a" else "analysis_plan.lock.md"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    status = "locked" if locked else "draft"
    recorded = digest if locked else "null"
    version = 1 if locked else 0
    return (
        write_text(
            root,
            f"docs/prereg/{name}",
            "\n".join(
                [
                    "---",
                    "schema_version: research_swarm.prereg_lock.v1",
                    f"phase: {phase}",
                    f"status: {status}",
                    "locked_at_utc: 2026-07-10T12:00:00Z" if locked else "locked_at_utc: null",
                    f"locked_sha256: {recorded}",
                    "locked_by: Test Owner" if locked else "locked_by: null",
                    f"lock_version: {version}",
                    "---",
                    "",
                ]
            )
            + body,
        ),
        digest,
    )


def _valid_claim(root: Path, *, claim_type: str = "methodological") -> dict[str, object]:
    evidence = write_text(root, "reports/evidence.txt", "evidence\n")
    uncertainty = write_text(root, "reports/uncertainty.txt", "uncertainty\n")
    claim: dict[str, object] = {
        "claim_id": "C1",
        "statement": "A registered statement.",
        "type": claim_type,
        "supporting_artifacts": [{"path": "reports/evidence.txt", "sha256": _hash(evidence)}],
        "verification_command": "make verify-claim",
        "uncertainty_artifact": None,
        "uncertainty_justification": "No numeric uncertainty applies to this claim type.",
    }
    if claim_type in {"descriptive", "associational", "causal", "computational", "counterfactual"}:
        claim["uncertainty_artifact"] = {
            "path": "reports/uncertainty.txt",
            "sha256": _hash(uncertainty),
        }
        claim.pop("uncertainty_justification", None)
    if claim_type == "causal":
        sensitivity = write_text(root, "reports/sensitivity.txt", "sensitivity\n")
        claim["sensitivity_artifact"] = {
            "path": "reports/sensitivity.txt",
            "sha256": _hash(sensitivity),
        }
        claim["identification_strategy"] = "docs/prereg/analysis_plan.lock.md#identification"
    if claim_type == "theoretical":
        claim["assumption_scope"] = "The proposition holds on the declared domain."
    return claim


def _write_claims(root: Path, claims: list[dict[str, object]]) -> Path:
    return write_json(
        root,
        "contracts/claims.yaml",
        {"schema_version": "research_swarm.claims.v1", "claims": claims},
    )


def _reasons(result) -> set[str]:
    return {
        item["reason"]
        for item in result.details.get("failures", [])
        if isinstance(item, dict) and isinstance(item.get("reason"), str)
    }


def _snapshot(citekey: str, **overrides: object) -> dict[str, object]:
    retrieval_payload = {"citekey": citekey, "provider": "fixture"}
    payload: dict[str, object] = {
        "schema_version": "research_swarm.citation_snapshot.v1",
        "citekey": citekey,
        "title": "Verified",
        "source": "crossref",
        "retrieved_at_utc": "2026-07-10T12:00:00Z",
        "retrieval_sha256": hashlib.sha256(
            json.dumps(retrieval_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "retrieval_payload": retrieval_payload,
        "resolved": True,
        "retraction_status": "none",
        "url_resolves": True,
    }
    payload.update(overrides)
    return payload


class M3aScienceGateTest(unittest.TestCase):
    def _root(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        return root

    def test_claim_evidence_ledger_valid_is_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write_claims(root, [_valid_claim(root)])
            with chdir(root):
                result = quality_gates.gate_claim_evidence_ledger()
            self.assertTrue(result.ok, result.details)

    def test_claim_evidence_ledger_failure_classes_are_rule_level(self) -> None:
        cases = {
            "hash_drift": "artifact_sha256_mismatch",
            "missing_uncertainty": "uncertainty_artifact_required",
            "missing_justification": "uncertainty_na_justification_required",
            "bad_command": "verification_command_policy_violation",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                claim = _valid_claim(root, claim_type="causal" if case == "missing_uncertainty" else "methodological")
                if case == "hash_drift":
                    write_text(root, "reports/evidence.txt", "drifted\n")
                elif case == "missing_uncertainty":
                    claim["uncertainty_artifact"] = None
                elif case == "missing_justification":
                    claim.pop("uncertainty_justification")
                elif case == "bad_command":
                    claim["verification_command"] = "python -c pass"
                _write_claims(root, [claim])
                with chdir(root):
                    result = quality_gates.gate_claim_evidence_ledger()
                self.assertFalse(result.ok)
                self.assertIn(expected, _reasons(result), result.details)

    def test_prereg_conformance_active_lock_and_outcome_is_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _, digest = _write_lock(root, "2b", "# Plan\n\n- H1: Primary hypothesis\n")
            claim = _valid_claim(root, claim_type="causal")
            claim.update({"prereg_lock_sha256": digest, "hypothesis_id": "H1"})
            _write_claims(root, [claim])
            write_json(
                root,
                "docs/prereg/outcomes.yaml",
                {"schema_version": "research_swarm.prereg_outcomes.v1", "outcomes": [{"hypothesis_id": "H1", "outcome": "supported"}]},
            )
            with chdir(root):
                result = quality_gates.gate_prereg_conformance()
            self.assertTrue(result.ok, result.details)

    def test_prereg_conformance_rejects_prelock_hash_mismatch_and_missing_outcome(self) -> None:
        for case, expected in (
            ("prelock", "confirmatory_claim_without_active_lock"),
            ("hash", "confirmatory_claim_prereg_hash_mismatch"),
            ("outcome", "missing_hypothesis_outcome"),
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                claim = _valid_claim(root, claim_type="causal")
                if case != "prelock":
                    _, digest = _write_lock(root, "2b", "# Plan\n\n- H1: Primary hypothesis\n")
                    claim["prereg_lock_sha256"] = "0" * 64 if case == "hash" else digest
                    write_json(
                        root,
                        "docs/prereg/outcomes.yaml",
                        {
                            "schema_version": "research_swarm.prereg_outcomes.v1",
                            "outcomes": [] if case == "outcome" else [{"hypothesis_id": "H1", "outcome": "inconclusive"}],
                        },
                    )
                _write_claims(root, [claim])
                with chdir(root):
                    result = quality_gates.gate_prereg_conformance()
                self.assertFalse(result.ok)
                self.assertIn(expected, _reasons(result), result.details)

    def test_citation_integrity_clean_snapshot_is_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "references.bib", "@article{paper1, title={Verified}}\n")
            write_text(root, "data/citations/AS_OF", "2026-07-10\n")
            write_json(root, "data/citations/2026-07-10/paper1.json", _snapshot("paper1"))
            with chdir(root):
                result = quality_gates.gate_citation_integrity()
            self.assertTrue(result.ok, result.details)

    def test_citation_integrity_failure_classes_are_rule_level(self) -> None:
        cases = {
            "missing": "missing_citation_snapshot",
            "extra": "extra_citation_snapshot",
            "stale": "citation_snapshot_stale",
            "unresolved": "citation_unresolved",
            "retracted": "citation_retraction_status_not_clean",
            "url": "citation_url_unresolved",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                write_text(root, "references.bib", "@article{paper1, title={Verified}}\n")
                write_text(root, "data/citations/AS_OF", "2026-07-10\n")
                if case != "missing":
                    overrides: dict[str, object] = {}
                    if case == "stale":
                        overrides["retrieved_at_utc"] = "2025-01-01T00:00:00Z"
                    elif case == "unresolved":
                        overrides["resolved"] = False
                    elif case == "retracted":
                        overrides["retraction_status"] = "retracted"
                    elif case == "url":
                        overrides["url_resolves"] = False
                    write_json(root, "data/citations/2026-07-10/paper1.json", _snapshot("paper1", **overrides))
                if case == "extra":
                    write_json(root, "data/citations/2026-07-10/extra.json", _snapshot("extra"))
                with chdir(root):
                    result = quality_gates.gate_citation_integrity()
                self.assertFalse(result.ok)
                self.assertIn(expected, _reasons(result), result.details)

    def test_citation_integrity_local_key_checks_repo_path_without_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/local.txt", "local\n")
            write_text(root, "references.bib", "@misc{local:reports/local.txt, title={Local}}\n")
            with chdir(root):
                result = quality_gates.gate_citation_integrity()
            self.assertTrue(result.ok, result.details)

    def test_etl_decision_log_valid_clause_and_coverage_is_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write_lock(root, "2a", "# Construction\n\n### missingness_1\n\nPolicy.\n")
            write_json(
                root,
                "data/processed_manifest/example.json",
                {
                    "decision_log": [{"clause_id": "missingness_1", "choice": "Keep null", "rationale": "Preserve coverage."}],
                    "zero_fill_columns": ["rent"],
                    "coverage_flag_columns": ["rent"],
                },
            )
            with chdir(root):
                result = quality_gates.gate_etl_decision_log()
            self.assertTrue(result.ok, result.details)
            self.assertFalse(result.details["completeness_audited"])

    def test_etl_decision_log_rejects_unknown_clause_and_unflagged_zero_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write_lock(root, "2a", "# Construction\n\n### known\n\nPolicy.\n")
            write_json(
                root,
                "data/processed_manifest/example.json",
                {
                    "decision_log": [{"clause_id": "unknown", "choice": "Fill", "rationale": "Test."}],
                    "zero_fill_columns": ["rent"],
                    "coverage_flag_columns": [],
                },
            )
            with chdir(root):
                result = quality_gates.gate_etl_decision_log()
            self.assertFalse(result.ok)
            self.assertEqual(
                {"unknown_locked_protocol_clause", "zero_fill_without_coverage_flag"} - _reasons(result),
                set(),
                result.details,
            )

    def test_rigor_sections_valid_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task = write_task(root, "backlog", "T901", schema="v2", task_kind="analysis")
            with chdir(root):
                red = quality_gates.gate_rigor_sections()
            self.assertFalse(red.ok)
            self.assertIn("missing_or_empty_rigor_section", _reasons(red))
            task.write_text(
                task.read_text(encoding="utf-8")
                + "\n## Evidence table\n\nEvidence.\n"
                + "\n## Alternative explanations considered\n\nAlternative.\n"
                + "\n## Uncertainty statement\n\nUncertainty.\n",
                encoding="utf-8",
            )
            with chdir(root):
                green = quality_gates.gate_rigor_sections()
            self.assertTrue(green.ok, green.details)

    def test_refresh_stub_is_byte_stable(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "m3a_refresh_citations", Path(__file__).resolve().parents[1] / "scripts/refresh_citations.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = write_json(
                root,
                "fixtures/paper1.json",
                {
                    "schema_version": "research_swarm.citation_fixture.v1",
                    "citekey": "paper1",
                    "source": "crossref",
                    "retrieved_at_utc": "2026-07-10T12:00:00Z",
                    "raw_response": {"z": 1, "a": [2, 3]},
                    "normalized": {
                        "doi": "10.1234/example",
                        "title": "Verified",
                        "resolved": True,
                        "retraction_status": "none",
                        "url_resolves": True,
                    },
                },
            )
            self.assertTrue(fixture.is_file())
            outputs = module.refresh_from_fixtures(root / "fixtures", root / "out")
            first = {path.relative_to(root / "out").as_posix(): path.read_bytes() for path in outputs}
            first["AS_OF"] = (root / "out/AS_OF").read_bytes()
            outputs = module.refresh_from_fixtures(root / "fixtures", root / "out")
            second = {path.relative_to(root / "out").as_posix(): path.read_bytes() for path in outputs}
            second["AS_OF"] = (root / "out/AS_OF").read_bytes()
            self.assertEqual(first, second)

    def test_lock_command_stamps_hash_refuses_relock_and_amends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            args = argparse.Namespace(phase="2b", locked_by="Test Owner", amend=False)
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                mock.patch.object(swarm, "_utc_now_iso", return_value="2026-07-10T12:00:00Z"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 0)
                self.assertEqual(swarm.cmd_lock_prereg(args), 1)
                lock_path = root / "docs/prereg/analysis_plan.lock.md"
                lock_path.write_text(
                    lock_path.read_text(encoding="utf-8") + "\n- H1: Amended hypothesis\n",
                    encoding="utf-8",
                )
                args.amend = True
                write_json(
                    root,
                    "docs/prereg/amendments/2b_v2.md",
                    {
                        "schema_version": "research_swarm.prereg_amendment.v1",
                        "phase": "2b",
                        "from_version": 1,
                        "to_version": 2,
                        "dual_definition_rerun": {
                            "old_artifact": "reports/old.json",
                            "new_artifact": "reports/new.json",
                            "sensitivity_delta_artifact": "reports/delta.json",
                        },
                        "human_reviewer": "Independent Reviewer",
                        "justification": "New evidence requires a definition amendment.",
                        "effective_date": "2026-07-10",
                    },
                )
                self.assertEqual(swarm.cmd_lock_prereg(args), 0)
            import swarm_taskfile

            lock, error = swarm_taskfile.load_prereg_lock(
                root / "docs/prereg/analysis_plan.lock.md", expected_phase="2b"
            )
            self.assertIsNone(error)
            self.assertTrue(lock["active"])
            self.assertEqual(lock["lock_version"], 2)
            self.assertIn("H1", {item["hypothesis_id"] for item in lock["hypotheses"]})
            events = (root / "reports/status/events/events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event":"prereg_locked"', events)
            self.assertIn('"event":"prereg_amendment"', events)

    def test_task_lint_enforces_tier_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T902",
                schema="v2",
                budgets={"max_wall_clock": "1h", "max_tokens": 250001, "max_cost_usd": 10},
            )
            with chdir(root):
                result = quality_gates.gate_task_lint()
            reasons = {item["reason"] for item in result.details["failures"]}
            self.assertIn("budget_exceeds_tier_ceiling", reasons)

    def test_task_lint_links_hypothesis_ids_only_when_lock_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task = write_task(
                root,
                "backlog",
                "T903",
                schema="v2",
                extra_frontmatter={"hypothesis_ids": ["H2"]},
            )
            with chdir(root):
                draft = quality_gates.gate_task_lint()
            self.assertTrue(draft.ok, draft.details)
            _write_lock(root, "2b", "# Plan\n\n- H1: Registered hypothesis\n")
            with chdir(root):
                red = quality_gates.gate_task_lint()
            self.assertIn(
                "hypothesis_id_not_in_prereg",
                {item["reason"] for item in red.details["failures"]},
            )
            task.write_text(task.read_text(encoding="utf-8").replace("  - H2", "  - H1"), encoding="utf-8")
            with chdir(root):
                green = quality_gates.gate_task_lint()
            self.assertTrue(green.ok, green.details)

    def test_manuscript_numeric_must_be_registered_or_computed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/paper/index.qmd", "# Results\n\nMean STR was 12.5%.\n")
            with chdir(root):
                red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("unregistered_manuscript_numeric", _reasons(red))
            claim = _valid_claim(root, claim_type="descriptive")
            claim["statement"] = "Mean STR was 12.5%."
            _write_claims(root, [claim])
            with chdir(root):
                registered = quality_gates.gate_claim_evidence_ledger()
            self.assertTrue(registered.ok, registered.details)
            _write_claims(root, [])
            write_text(
                root,
                "reports/paper/index.qmd",
                "# Results\n\nMean STR was {{ paper_values.mean_str }}.\n",
            )
            with chdir(root):
                computed = quality_gates.gate_claim_evidence_ledger()
            self.assertTrue(computed.ok, computed.details)

    def test_citation_payload_hash_and_bibliography_identity_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/paper/index.qmd", "# Paper\n")
            write_text(root, "reports/paper/references.bib", "@article{paper1, title={Verified}}\n")
            write_text(root, "data/citations/AS_OF", "2026-07-10\n")
            snapshot = _snapshot("paper1")
            snapshot["retrieval_payload"] = {"tampered": True}
            write_json(root, "data/citations/2026-07-10/paper1.json", snapshot)
            with chdir(root):
                hash_red = quality_gates.gate_citation_integrity()
            self.assertIn("retrieval_sha256_mismatch", _reasons(hash_red))
            snapshot = _snapshot("paper1", title="Different title")
            write_json(root, "data/citations/2026-07-10/paper1.json", snapshot)
            with chdir(root):
                identity_red = quality_gates.gate_citation_integrity()
            self.assertIn("citation_snapshot_bib_identity_mismatch", _reasons(identity_red))

    def test_falsification_rejects_nonfinite_empty_and_author_avoidance(self) -> None:
        overflow = quality_gates.evaluate_falsification_spec(
            {
                "domain": {"x": [0, 1]},
                "seed": 1,
                "inequalities": ["1e308 * 1e308 == 1e307 * 1e307"],
                "comparative_statics": [],
            }
        )
        self.assertIn("inequality_evaluation_error", {item["reason"] for item in overflow})
        empty = quality_gates.evaluate_falsification_spec(
            {
                "domain": {"x": [0, 1]},
                "seed": 1,
                "inequalities": [],
                "comparative_statics": [],
            }
        )
        self.assertIn("falsification_spec_empty", {item["reason"] for item in empty})
        avoided = quality_gates.evaluate_falsification_spec(
            {
                "domain": {"x": [0, 1]},
                "seed": 1,
                "inequalities": ["x < 0.5"],
                "comparative_statics": [],
                "sample_points": [{"x": 0.1}],
            }
        )
        self.assertIn("inequality_violated", {item["reason"] for item in avoided})

    def test_claim_type_uncertainty_table_is_structural(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            causal = _valid_claim(root, claim_type="causal")
            causal.pop("sensitivity_artifact")
            _write_claims(root, [causal])
            with chdir(root):
                causal_red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("causal_sensitivity_artifact_required", _reasons(causal_red))

            interpretation = _valid_claim(root, claim_type="interpretation")
            interpretation["uncertainty_artifact"] = None
            interpretation.pop("uncertainty_justification", None)
            _write_claims(root, [interpretation])
            with chdir(root):
                interpretation_red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("interpretation_evidence_scope_required", _reasons(interpretation_red))

            theoretical = _valid_claim(root, claim_type="theoretical")
            theoretical.pop("assumption_scope")
            _write_claims(root, [theoretical])
            with chdir(root):
                theoretical_red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("theoretical_assumption_scope_required", _reasons(theoretical_red))

            counterfactual = _valid_claim(root, claim_type="counterfactual")
            _write_claims(root, [counterfactual])
            with chdir(root):
                counterfactual_red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn(
                "counterfactual_union_artifact_required", _reasons(counterfactual_red)
            )

            methodological = _valid_claim(root)
            methodological["verification_command"] = "make gate"
            _write_claims(root, [methodological])
            with chdir(root):
                command_red = quality_gates.gate_claim_evidence_ledger()
            self.assertIn("verification_command_self_referential", _reasons(command_red))

    def test_checklist_human_attestation_cannot_mask_red_machine_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/paper/index.qmd", "Figure Figure ??\n")
            write_json(
                root,
                "contracts/venue.yaml",
                {
                    "checklist": [
                        {
                            "question": "Does the manuscript render?",
                            "answer": "yes",
                            "derived_from": ["gate:render_qa", "human_attested"],
                        }
                    ]
                },
            )
            with chdir(root):
                result = quality_gates.gate_checklist_derivation()
            self.assertFalse(result.ok)
            self.assertIn("checklist_answer_not_supported", _reasons(result))

    def test_task_lint_rejects_nonfinite_budgets_and_ceilings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T904",
                schema="v2",
                budgets={"max_wall_clock": "1h", "max_tokens": float("nan"), "max_cost_usd": 1},
            )
            with chdir(root):
                budget_red = quality_gates.gate_task_lint()
            self.assertIn(
                "invalid_budget_value",
                {item["reason"] for item in budget_red.details["failures"]},
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            framework = json.loads((root / "contracts/framework.json").read_text())
            framework["complexity_tier_ceilings"]["S"]["max_tokens"] = float("inf")
            write_json(root, "contracts/framework.json", framework)
            write_task(root, "backlog", "T905", schema="v2")
            with chdir(root):
                ceiling_red = quality_gates.gate_task_lint()
            self.assertIn(
                "invalid_tier_ceiling",
                {item["reason"] for item in ceiling_red.details["failures"]},
            )

    def test_lock_matrix_and_task_kind_gate_threading(self) -> None:
        cases = (
            ("empirical", "etl", ["data/processed/panel.csv"], "2a"),
            ("empirical", "analysis", ["reports/tables/result.md"], "2b"),
            ("modeling", "proof", ["reports/proofs/p.md"], None),
            ("modeling", "model", ["reports/models/result.json"], "lock_a"),
            ("hybrid", "bridge", ["contracts/instances/i.json"], "lock_a"),
            ("hybrid", "model", ["reports/models/result.json"], "lock_b"),
        )
        for mode, task_kind, outputs, expected in cases:
            with self.subTest(mode=mode, task_kind=task_kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "repo"
                scaffold_runtime_repo(root, mode=mode)
                write_task(root, "backlog", "T906", schema="v2", task_kind=task_kind, outputs=outputs)
                contract = swarm.load_framework_contract(root)
                task = swarm.load_tasks(contract)["T906"]
                self.assertEqual(swarm._required_active_lock(task, mode), expected)

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            ok, outputs = swarm._run_gates(
                root,
                ["python scripts/quality_gates.py"],
                task_kind="analysis",
            )
            self.assertTrue(ok, outputs)
            self.assertEqual(outputs[0]["argv"][-2:], ["--task-kind", "analysis"])

    def test_ready_funnel_blocks_until_required_lock_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T908",
                schema="v2",
                task_kind="analysis",
                outputs=["reports/tables/result.md"],
            )
            contract = swarm.load_framework_contract(root)
            tasks = swarm.load_tasks(contract)
            with (
                mock.patch.object(swarm, "_task_has_planner_triage", return_value=True),
                mock.patch.object(swarm, "_record_swarm_event") as record_event,
            ):
                self.assertEqual(swarm.ready_backlog_tasks(tasks, set(), contract), [])
            self.assertTrue(
                any(
                    call.args[1].get("event") == "blocked_on_prereg_lock"
                    for call in record_event.call_args_list
                )
            )
            _write_lock(root, "2b", "# Plan\n")
            tasks = swarm.load_tasks(contract)
            with mock.patch.object(swarm, "_task_has_planner_triage", return_value=True):
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)
            self.assertEqual([task.task_id for task in ready], ["T908"])

    def test_amendment_record_cap_and_lock_header_journal_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            args = argparse.Namespace(phase="2b", locked_by="Test Owner", amend=False)
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                mock.patch.object(swarm, "_utc_now_iso", return_value="2026-07-10T12:00:00Z"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 0)
                lock_path = root / "docs/prereg/analysis_plan.lock.md"
                lock_path.write_text(lock_path.read_text() + "\nAmended body.\n")
                args.amend = True
                self.assertEqual(swarm.cmd_lock_prereg(args), 1)
            self.assertIn("amendment_record_required", stderr.getvalue())

            lock, error = swarm.load_prereg_lock(lock_path, expected_phase="2b")
            self.assertIsNone(error)
            body = str(lock["body"])
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            lock_path.write_text(
                "\n".join(
                    [
                        "---",
                        "schema_version: research_swarm.prereg_lock.v1",
                        "phase: 2b",
                        "status: locked",
                        "locked_at_utc: 2026-07-10T12:00:00Z",
                        f"locked_sha256: {body_hash}",
                        "locked_by: Test Owner",
                        "lock_version: 3",
                        "---",
                        "",
                    ]
                )
                + body
            )
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 1)
            self.assertIn("amendment_cap_exceeded:L3_required", stderr.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            args = argparse.Namespace(phase="2b", locked_by="Test Owner", amend=False)
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                mock.patch.object(swarm, "_utc_now_iso", return_value="2026-07-10T12:00:00Z"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 0)
            with chdir(root):
                self.assertTrue(quality_gates.gate_prereg_lock_coverage().ok)
            lock_path = root / "docs/prereg/analysis_plan.lock.md"
            lock_path.write_text(
                lock_path.read_text().replace(
                    "locked_at_utc: 2026-07-10T12:00:00Z",
                    "locked_at_utc: 2026-07-11T12:00:00Z",
                )
            )
            with chdir(root):
                tampered = quality_gates.gate_prereg_lock_coverage()
            self.assertIn("prereg_lock_header_journal_mismatch", _reasons(tampered))

    def test_prereg_lock_coverage_rejects_historical_bypass(self) -> None:
        for done_before_lock, expected_ok in ((True, False), (False, True)):
            with self.subTest(done_before_lock=done_before_lock), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp)
                write_task(
                    root,
                    "done",
                    "T907",
                    schema="v2",
                    state="done",
                    task_kind="analysis",
                    outputs=["reports/tables/result.md"],
                )
                if done_before_lock:
                    swarm.swarm_events.append_event(root, {"event": "task_done", "task_id": "T907"})
                args = argparse.Namespace(phase="2b", locked_by="Test Owner", amend=False)
                with (
                    mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(root)}, clear=False),
                    mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                    mock.patch.object(swarm, "_utc_now_iso", return_value="2026-07-10T12:00:00Z"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(swarm.cmd_lock_prereg(args), 0)
                if not done_before_lock:
                    swarm.swarm_events.append_event(root, {"event": "task_done", "task_id": "T907"})
                with chdir(root):
                    result = quality_gates.gate_prereg_lock_coverage()
                self.assertEqual(result.ok, expected_ok, result.details)
                if not expected_ok:
                    self.assertIn(
                        "task_completed_before_required_prereg_lock", _reasons(result)
                    )

    def test_science_gate_registry_order_follows_task_lint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            with chdir(root):
                names = list(quality_gates._collect_gate_results())
            self.assertEqual(names, list(quality_gates._ALL_GATE_NAMES))
            self.assertEqual(
                names[-3:],
                ["render_qa", "text_overlap", "checklist_derivation"],
            )


if __name__ == "__main__":
    unittest.main()
