from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from golden.harness import GoldenRepo
from runtime_test_utils import chdir, load_quality_gates_module, write_json, write_text


quality_gates = load_quality_gates_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock(repo: GoldenRepo, phase: str, body: str) -> str:
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
                "lock_version: 1",
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
    return {
        "claim_id": "C-GOLD",
        "statement": "Golden registered claim.",
        "type": claim_type,
        "supporting_artifacts": [
            {"path": "reports/golden_evidence.txt", "sha256": _sha(evidence)}
        ],
        "verification_command": "make gate",
        "uncertainty_artifact": {
            "path": "reports/golden_uncertainty.txt",
            "sha256": _sha(uncertainty),
        },
    }


def _claims(repo: GoldenRepo, claims: list[dict[str, object]]) -> None:
    write_json(
        repo.root,
        "contracts/claims.yaml",
        {"schema_version": "research_swarm.claims.v1", "claims": claims},
    )


def _snapshot(*, retraction_status: str = "none") -> dict[str, object]:
    return {
        "schema_version": "research_swarm.citation_snapshot.v1",
        "citekey": "poisoned",
        "title": "Citation",
        "source": "crossref",
        "retrieved_at_utc": "2026-07-10T12:00:00Z",
        "retrieval_sha256": "a" * 64,
        "resolved": True,
        "retraction_status": retraction_status,
        "url_resolves": True,
    }


class GoldenM3aTest(unittest.TestCase):
    def test_GM3A_01_poisoned_bibliography_blocks_then_clean_snapshot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_text(repo.root, "references.bib", "@article{poisoned, title={Citation}}\n")
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
            write_json(
                repo.root,
                "docs/prereg/outcomes.yaml",
                {"schema_version": "research_swarm.prereg_outcomes.v1", "outcomes": [{"hypothesis_id": "H1", "outcome": "not_supported"}]},
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


if __name__ == "__main__":
    unittest.main()
