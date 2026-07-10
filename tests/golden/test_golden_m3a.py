from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from golden.harness import GoldenRepo
from runtime_test_utils import chdir, load_quality_gates_module, load_swarm_module, write_json, write_project_yaml, write_text
from test_m3a_modeling_battery import (
    _bridge_instance,
    _claim as _model_claim,
    _claims as _model_claims,
    _experiment_manifest,
    _lock_a,
    _lock_b,
    _output_entry,
    _synthetic_instance,
)


quality_gates = load_quality_gates_module()
swarm = load_swarm_module()


def _set_mode(repo: GoldenRepo, mode: str) -> None:
    write_project_yaml(repo.root, mode=mode)
    protocol = repo.root / "docs/protocol.md"
    protocol.write_text(
        protocol.read_text(encoding="utf-8").replace("- Mode: empirical", f"- Mode: {mode}"),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock(repo: GoldenRepo, phase: str, body: str, *, version: int = 1) -> str:
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    filename = "data_construction.lock.md" if phase == "2a" else "analysis_plan.lock.md"
    write_text(
        repo.root,
        f"docs/prereg/{filename}",
        "\n".join(
            [
                "---",
                "schema_version: research_swarm.prereg_lock.v1",
                f"phase: {phase}",
                "status: locked",
                "locked_at_utc: 2026-07-10T12:00:00Z",
                f"locked_sha256: {digest}",
                "locked_by: Golden Owner",
                f"lock_version: {version}",
                "---",
                "",
            ]
        )
        + body,
    )
    return digest


def _claim(repo: GoldenRepo, *, claim_type: str = "causal") -> dict[str, object]:
    evidence = write_text(repo.root, "reports/golden_evidence.txt", "evidence\n")
    uncertainty = write_text(repo.root, "reports/golden_uncertainty.txt", "uncertainty\n")
    claim: dict[str, object] = {
        "claim_id": "C-GOLD",
        "statement": "Golden registered claim.",
        "type": claim_type,
        "supporting_artifacts": [
            {"path": "reports/golden_evidence.txt", "sha256": _sha(evidence)}
        ],
        "verification_command": "make verify-claim",
        "uncertainty_artifact": {
            "path": "reports/golden_uncertainty.txt",
            "sha256": _sha(uncertainty),
        },
    }
    if claim_type == "causal":
        sensitivity = write_text(repo.root, "reports/golden_sensitivity.txt", "sensitivity\n")
        claim["sensitivity_artifact"] = {
            "path": "reports/golden_sensitivity.txt",
            "sha256": _sha(sensitivity),
        }
        claim["identification_strategy"] = "docs/prereg/analysis_plan.lock.md#identification"
    return claim


def _claims(repo: GoldenRepo, claims: list[dict[str, object]]) -> None:
    write_json(
        repo.root,
        "contracts/claims.yaml",
        {"schema_version": "research_swarm.claims.v1", "claims": claims},
    )
    # Claim evidence must be git-tracked for the ledger's purity check (§6.5);
    # stage everything written into the fixture so the gate sees real tracking.
    repo.git("add", "-A")


def _snapshot(*, retraction_status: str = "none") -> dict[str, object]:
    retrieval_payload = {"citekey": "poisoned", "provider": "fixture"}
    return {
        "schema_version": "research_swarm.citation_snapshot.v1",
        "citekey": "poisoned",
        "title": "Citation",
        "source": "crossref",
        "retrieved_at_utc": "2026-07-10T12:00:00Z",
        "retrieval_sha256": hashlib.sha256(
            json.dumps(retrieval_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "retrieval_payload": retrieval_payload,
        "resolved": True,
        "retraction_status": retraction_status,
        "url_resolves": True,
    }


def _track(repo: GoldenRepo) -> None:
    repo.git("add", "-A")


def _bound_sweep_cells(repo: GoldenRepo, cells: list[dict[str, object]]) -> list[dict[str, object]]:
    bound: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        path = write_json(
            repo.root,
            f"reports/models/experiment_C1_{index}.json",
            {
                "schema_version": "research_swarm.experiment_manifest.v1",
                "experiment_id": f"C1-{index}",
                "instance_id": "toy",
                "seed": cell["seed"],
                "budget": cell["budget"],
                "solver": "toy",
                "solver_version": "1",
                "optimality_gap": 0.0001,
                "converged": True,
                "parameters": {key: value for key, value in cell.items() if key not in {"seed", "budget"}},
                "outputs": {},
            },
        )
        bound.append(
            {
                "cell": cell,
                "experiment_manifest": {
                    "path": path.relative_to(repo.root).as_posix(),
                    "sha256": _sha(path),
                },
            }
        )
    return bound


class GoldenM3aTest(unittest.TestCase):
    def test_GM3A_01_poisoned_bibliography_blocks_then_clean_snapshot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_text(repo.root, "reports/paper/index.qmd", "# Paper\n")
            write_text(repo.root, "reports/paper/references.bib", "@article{poisoned, title={Citation}}\n")
            write_text(repo.root, "data/citations/AS_OF", "2026-07-10\n")
            write_json(
                repo.root,
                "data/citations/2026-07-10/poisoned.json",
                _snapshot(retraction_status="retracted"),
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_citation_integrity().ok)
            write_json(repo.root, "data/citations/2026-07-10/poisoned.json", _snapshot())
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_citation_integrity().ok)

    def test_GM3A_02_missing_hypothesis_outcome_blocks_then_terminal_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _lock(repo, "2b", "# Plan\n\n- H1: Golden hypothesis\n")
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_prereg_conformance().ok)
            # A terminal outcome is only 'reported' when content-bound to
            # committed manuscript/deviations text naming the hypothesis (§6.1).
            write_text(
                repo.root,
                "reports/paper/deviations.md",
                "# Deviations\n\nH1: not supported on the validated surface.\n",
            )
            write_json(
                repo.root,
                "docs/prereg/outcomes.yaml",
                {"schema_version": "research_swarm.prereg_outcomes.v1", "outcomes": [{"hypothesis_id": "H1", "outcome": "not_supported", "reported_in": "reports/paper/deviations.md"}]},
            )
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_prereg_conformance().ok)

    def test_GM3A_03_zero_fill_blocks_then_matching_coverage_flag_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_json(
                repo.root,
                "data/processed_manifest/zero_fill.json",
                {"zero_fill_columns": ["rent"], "coverage_flag_columns": []},
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_etl_decision_log().ok)
            write_json(
                repo.root,
                "data/processed_manifest/zero_fill.json",
                {"zero_fill_columns": ["rent"], "coverage_flag_columns": ["rent"]},
            )
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_etl_decision_log().ok)

    def test_GM3A_04_confirmatory_claim_prelock_refused_then_lock_bound_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            claim = _claim(repo)
            _claims(repo, [claim])
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_prereg_conformance().ok)
            digest = _lock(repo, "2b", "# Plan\n")
            claim["prereg_lock_sha256"] = digest
            _claims(repo, [claim])
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_prereg_conformance().ok)

    def test_GM3A_05_supporting_artifact_hash_drift_blocks_then_rebind_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            claim = _claim(repo, claim_type="descriptive")
            _claims(repo, [claim])
            write_text(repo.root, "reports/golden_evidence.txt", "drift\n")
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_claim_evidence_ledger().ok)
            claim["supporting_artifacts"][0]["sha256"] = _sha(
                repo.root / "reports/golden_evidence.txt"
            )
            _claims(repo, [claim])
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_claim_evidence_ledger().ok)

    def test_GM3A_06_causal_uncertainty_missing_blocks_then_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            claim = _claim(repo)
            uncertainty = claim["uncertainty_artifact"]
            claim["uncertainty_artifact"] = None
            _claims(repo, [claim])
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_claim_evidence_ledger().ok)
            claim["uncertainty_artifact"] = uncertainty
            _claims(repo, [claim])
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_claim_evidence_ledger().ok)

    def test_GM3A_07_analysis_task_rigor_sections_missing_then_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task = repo.write_task("backlog", "T971", schema="v2", task_kind="analysis")
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_rigor_sections().ok)
            task.write_text(
                task.read_text(encoding="utf-8")
                + "\n## Evidence table\n\nEvidence.\n"
                + "\n## Alternative explanations considered\n\nAlternative.\n"
                + "\n## Uncertainty statement\n\nUncertainty.\n",
                encoding="utf-8",
            )
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_rigor_sections().ok)

    def test_GM3A_08_tier_ceiling_exceeded_then_bounded_budget_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            repo.write_task(
                "backlog",
                "T972",
                schema="v2",
                budgets={"max_wall_clock": "1h", "max_tokens": 250001, "max_cost_usd": 10},
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_task_lint().ok)
            task = next((repo.root / ".orchestrator/backlog").glob("T972_*.md"))
            task.write_text(task.read_text(encoding="utf-8").replace("250001", "250000"), encoding="utf-8")
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_task_lint().ok)

    def test_GM3A_09_hypothesis_link_unknown_then_registered_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _lock(repo, "2b", "# Plan\n\n- H1: Golden hypothesis\n")
            repo.write_task(
                "backlog",
                "T973",
                schema="v2",
                extra_frontmatter={"hypothesis_ids": ["H2"]},
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_task_lint().ok)
            task = next((repo.root / ".orchestrator/backlog").glob("T973_*.md"))
            task.write_text(task.read_text(encoding="utf-8").replace("  - H2", "  - H1"), encoding="utf-8")
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_task_lint().ok)

    def test_GM3A_10_false_lemma_blocks_then_true_companion_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "modeling")
            claim = _model_claim(repo.root, "theoretical")
            claim["falsification_spec"] = {
                "inequalities": ["x ** 2 >= 0.251"],
                "comparative_statics": [],
                "domain": {"x": [-0.5, 0.5]},
                "seed": 17,
                "sample_points": [{"x": -0.5}],
            }
            _model_claims(repo.root, [claim])
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_theoretical_falsification().ok)
            claim["falsification_spec"]["inequalities"] = ["x ** 2 >= 0"]
            _model_claims(repo.root, [claim])
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_theoretical_falsification().ok)

    def test_GM3A_11_seed_budget_violation_blocks_then_locked_cell_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "modeling")
            _lock_a(repo.root)
            _experiment_manifest(repo.root, seed=999, budget=100)
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_seed_budget_lock().ok)
            _experiment_manifest(repo.root, seed=11, budget=100)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_seed_budget_lock().ok)

    def test_GM3A_12_instance_conformance_missing_seed_blocks_then_valid_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "modeling")
            output = _output_entry(repo.root, "reports/models/toy_instance.txt")
            write_json(
                repo.root,
                "contracts/instances/toy.json",
                {
                    "schema_version": "research_swarm.instance_manifest.v1",
                    "instance_id": "toy",
                    "generator_command": "python scripts/generate_toy.py",
                    "parameter_ranges": {"alpha": [1, 2]},
                    "git_sha": "abcdef0",
                    "outputs": [output],
                },
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_instance_manifest_conformance().ok)
            _synthetic_instance(repo.root)
            _track(repo)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_instance_manifest_conformance().ok)

    def test_GM3A_13_missing_sweep_blocks_then_locked_grid_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "modeling")
            lock_hash, spec = _lock_a(repo.root)
            claim = _model_claim(repo.root, "computational", headline=True)
            claim["lock_a_sha256"] = lock_hash
            _model_claims(repo.root, [claim])
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_sweep_artifact().ok)
                cells = quality_gates.enumerate_cells(spec)
            write_json(
                repo.root,
                "reports/models/sweeps/C1.json",
                {
                    "schema_version": "research_swarm.sweep_artifact.v1",
                    "claim_id": "C1",
                    "cells": _bound_sweep_cells(repo, cells),
                    "survival_count": len(cells),
                },
            )
            _track(repo)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_sweep_artifact().ok)

    def test_GM3A_14_ambiguous_model_spec_blocks_then_complete_lock_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "modeling")
            write_text(repo.root, "contracts/model_spec.md", "# Model spec\n\nTBD\n")
            _lock_a(repo.root, fill_model=False)
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_prereg_conformance().ok)
            _lock_a(repo.root)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_prereg_conformance().ok)

    def test_GM3A_15_bridge_stale_source_hash_blocks_then_rebind_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "hybrid")
            manifest = _bridge_instance(repo.root, stale_source=True)
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_hybrid_interface_conformance().ok)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["source_processed_manifests"][0]["sha256"] = _sha(
                repo.root / "data/processed_manifest/source.json"
            )
            write_json(repo.root, "contracts/instances/bridge.json", payload)
            _track(repo)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance().ok)

    def test_GM3A_16_model_direct_processed_input_blocks_then_instance_input_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "hybrid")
            task = repo.write_task(
                "backlog",
                "T974",
                schema="v2",
                task_kind="model",
                inputs=[{"path": "data/processed/direct.csv", "sha256": "1" * 64}],
            )
            with chdir(repo.root):
                self.assertFalse(
                    quality_gates.gate_hybrid_interface_conformance(task_kind="model").ok
                )
            task.write_text(
                task.read_text(encoding="utf-8").replace(
                    "data/processed/direct.csv", "contracts/instances/toy.json"
                ),
                encoding="utf-8",
            )
            with chdir(repo.root):
                self.assertTrue(
                    quality_gates.gate_hybrid_interface_conformance(task_kind="model").ok
                )

    def test_GM3A_17_post_generation_edit_amend_without_record_stays_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _set_mode(repo, "hybrid")
            manifest = _bridge_instance(repo.root)
            _lock_b(repo.root, [manifest])
            old_manifest_sha = _sha(manifest)
            _track(repo)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance().ok)
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            lock_path = repo.root / "docs/prereg/lock_b.md"
            lock_path.write_text(
                lock_path.read_text().replace(old_manifest_sha, _sha(manifest)),
                encoding="utf-8",
            )
            with chdir(repo.root):
                self.assertFalse(quality_gates.gate_hybrid_interface_conformance().ok)
            args = argparse.Namespace(phase="lock_b", locked_by="Golden Owner", amend=True)
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(repo.root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 1)
            write_json(
                repo.root,
                "docs/prereg/amendments/lock_b_v2.md",
                {
                    "schema_version": "research_swarm.prereg_amendment.v1",
                    "phase": "lock_b",
                    "from_version": 1,
                    "to_version": 2,
                    "dual_definition_rerun": {
                        "old_artifact": "reports/old.json",
                        "new_artifact": "reports/new.json",
                        "sensitivity_delta_artifact": "reports/delta.json",
                    },
                    "human_reviewer": "Independent Reviewer",
                    "justification": "The concrete instance set changed.",
                    "effective_date": "2026-07-10",
                },
            )
            _track(repo)
            with (
                mock.patch.dict(os.environ, {"SWARM_REPO_ROOT": str(repo.root)}, clear=False),
                mock.patch.object(swarm, "_REPO_ROOT_CACHE", None),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(swarm.cmd_lock_prereg(args), 0)
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance().ok)

    def test_GM3A_18_unregistered_numeric_blocks_then_registered_claim_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_text(
                repo.root,
                "reports/paper/index.qmd",
                "# Results\n\nMean STR was 11.68% [@str_mean].\n",
            )
            with chdir(repo.root):
                red = quality_gates.gate_claim_evidence_ledger()
            self.assertFalse(red.ok)
            self.assertIn(
                "unregistered_manuscript_numeric",
                {item["reason"] for item in red.details["failures"]},
            )
            # Occurrence-scoped registration: the numeric binds only because its
            # line cites [@str_mean] AND a claim under that citation key owns it.
            claim = _claim(repo, claim_type="descriptive")
            claim["statement"] = "Mean STR was 11.68%."
            claim["citation_key"] = "str_mean"
            _claims(repo, [claim])
            with chdir(repo.root):
                green = quality_gates.gate_claim_evidence_ledger()
            self.assertTrue(green.ok, green.details)

    def test_GM3A_19_amended_headline_requires_exploratory_release_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            digest = _lock(repo, "2b", "# Amended plan\n", version=2)
            claim = _claim(repo)
            claim.update(
                {
                    "prereg_lock_sha256": digest,
                    "confirmatory": True,
                    "headline": True,
                }
            )
            _claims(repo, [claim])
            write_json(repo.root, "contracts/venue.yaml", {"release_type": "research_article"})
            with chdir(repo.root):
                tagging_red = quality_gates.gate_amendment_exploratory_tagging()
                headline_red = quality_gates.gate_headline_confirmatory()
            self.assertFalse(tagging_red.ok)
            self.assertFalse(headline_red.ok)
            claim["confirmatory"] = False
            _claims(repo, [claim])
            write_json(repo.root, "contracts/venue.yaml", {"release_type": "exploratory_report"})
            with chdir(repo.root):
                self.assertTrue(quality_gates.gate_amendment_exploratory_tagging().ok)
                self.assertTrue(quality_gates.gate_headline_confirmatory().ok)

    def test_GM3A_20_judge_rejects_analysis_task_without_active_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            task = repo.write_task(
                "ready_for_review",
                "T976",
                schema="v2",
                state="ready_for_review",
                task_kind="analysis",
                outputs=["README.md"],
                gates=["python scripts/noop_gate.py"],
            )
            repo.write_run_manifest(
                "T976",
                task_path=task.relative_to(repo.root).as_posix(),
                provenance_class="executor_run",
                result_status="ok",
            )
            exit_code, summary = repo.judge("T976")
            self.assertEqual(exit_code, 1)
            self.assertFalse(summary["approved"])
            review_path = next((repo.root / "reports/status/reviews").glob("T976_*.json"))
            review = json.loads(review_path.read_text())
            self.assertIn("inactive_prereg_lock:2b", review["checks"]["failures"])


if __name__ == "__main__":
    unittest.main()
