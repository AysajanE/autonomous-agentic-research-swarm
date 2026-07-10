from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from runtime_test_utils import (
    chdir,
    init_git_fixture_repo,
    load_quality_gates_module,
    load_swarm_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
    write_text,
)


REPO = Path(__file__).resolve().parents[1]
quality_gates = load_quality_gates_module()
swarm = load_swarm_module()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


integrity_audit = _load("m3b_batch2_integrity_audit", REPO / "scripts/integrity_audit.py")
literature = _load("m3b_batch2_literature", REPO / "scripts/literature.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reasons(result) -> set[str]:
    return {
        item.get("reason")
        for item in result.details.get("failures", [])
        if isinstance(item, dict)
    }


def _valid_report(root: Path, *, audited_sha: str = "a" * 40) -> dict[str, object]:
    inventory = write_json(root, "reports/status/releases/release.json", {"artifacts": {}})
    return {
        "schema_version": "research_swarm.integrity_audit.v1",
        "generated_at_utc": "2026-07-10T12:00:00Z",
        "status": "pass",
        "mode": "empirical",
        "audited_git_sha": audited_sha,
        "release_inventory": {
            "path": "reports/status/releases/release.json",
            "sha256": _sha(inventory),
        },
        "executor": {
            "backend": "mock",
            "audit_family": "claude",
            "builder_families": ["codex"],
            "profile": "scratch-worktree",
            "network": "off",
            "commit_push_allowed": False,
        },
        "sampling": {
            "seed_path": "contracts/integrity_audit_seed.txt",
            "non_headline_sample_size": 3,
        },
        "surface_rebuilds": [
            {
                "outputs": [
                    {
                        "path": "data/processed/result.txt",
                        "matches_manifest": True,
                        "matches_release_inventory": True,
                    }
                ]
            }
        ],
        "inventory_hash_checks": [{"path": "data/processed/result.txt", "passed": True}],
        "claim_recomputations": [{"claim_id": "C1", "headline": True, "passed": True}],
        "etl_decision_samples": [
            {"manifest": "data/processed_manifest/result.json", "protocol_clause_id": "P1", "status": "pass"}
        ],
        "experiment_recomputations": [],
        "theoretical_rederivations": [],
        "seam_audits": [],
        "authorized_post_audit_repairs": [],
        "failures": [],
    }


class IntegrityAuditTest(unittest.TestCase):
    def _audit_fixture(self, root: Path, *, family: str = "claude", role_value: str = "69.14%") -> argparse.Namespace:
        write_text(root, "contracts/integrity_audit_seed.txt", "fixture-seed\n")
        write_text(root, "data/raw/source.txt", "69.14%\n")
        generator = write_text(
            root,
            "scripts/generate_surface.py",
            "from pathlib import Path\nPath('data/processed').mkdir(parents=True, exist_ok=True)\nPath('data/processed/result.txt').write_bytes(Path('data/raw/source.txt').read_bytes())\n",
        )
        subprocess.run([sys.executable, str(generator)], cwd=root, check=True)
        output = root / "data/processed/result.txt"
        write_json(
            root,
            "data/processed_manifest/result.json",
            {
                "as_of_utc_date": "2026-07-10",
                "transform": {"command": "python scripts/generate_surface.py"},
                "outputs": [
                    {"path": "data/processed/result.txt", "sha256": _sha(output), "bytes": output.stat().st_size}
                ],
            },
        )
        write_json(
            root,
            "contracts/claims.yaml",
            {
                "schema_version": "research_swarm.claims.v1",
                "claims": [
                    {
                        "claim_id": "C1",
                        "type": "descriptive",
                        "headline": True,
                        "statement": "Pre-Dencun mean STR is 69.14%.",
                        "manuscript_numeric_literals": ["69.14%"],
                        "recomputation_roles": {"pre_dencun_mean_str": "69.14%"},
                    }
                ],
            },
        )
        inventory = write_json(
            root,
            "reports/status/releases/release.json",
            {"artifacts": {"processed": [{"path": "data/processed/result.txt", "sha256": _sha(output)}]}},
        )
        transcript = write_json(
            root,
            "mock_audit.json",
            {
                "schema_version": "research_swarm.mock_integrity_audit.v1",
                "audit_family": family,
                "claim_recomputations": [
                    {
                        "claim_id": "C1",
                        "status": "pass",
                        "numeric_literals": [role_value],
                        "role_values": {"pre_dencun_mean_str": role_value},
                    }
                ],
                "etl_decision_samples": [
                    {"manifest": "data/processed_manifest/result.json", "protocol_clause_id": "P1", "status": "pass"}
                ],
            },
        )
        return argparse.Namespace(
            repo_root=root,
            output=Path("reports/status/integrity_audit/audit.json"),
            release_inventory=inventory.relative_to(root),
            mode="empirical",
            audit_family=family,
            builder_family=["codex"],
            backend="mock",
            mock_transcript=transcript,
            seed_path=Path("contracts/integrity_audit_seed.txt"),
            timeout_seconds=30,
            hermetic_copy=True,
        )

    def test_mock_audit_rebuilds_and_recomputes_headline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = integrity_audit.run_audit(self._audit_fixture(root))
            self.assertEqual(report["status"], "pass", report["failures"])
            self.assertTrue(report["surface_rebuilds"][0]["outputs"][0]["matches_release_inventory"])
            self.assertTrue(report["claim_recomputations"][0]["passed"])

    def test_same_family_audit_hard_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root, family="codex")
            report = integrity_audit.run_audit(args)
            self.assertEqual(report["status"], "block")
            self.assertIn("integrity_audit_family_of_builder", report["failures"])

    def test_wrong_headline_semantic_role_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = integrity_audit.run_audit(self._audit_fixture(root, role_value="11.68%"))
            self.assertEqual(report["status"], "block")
            self.assertIn("claim_semantic_role_mismatch:C1", report["failures"])
            self.assertIn("claim_numeric_recompute_mismatch:C1", report["failures"])

    def test_recompute_surface_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root)
            manifest = json.loads((root / "data/processed_manifest/result.json").read_text())
            manifest["outputs"][0]["sha256"] = "0" * 64
            write_json(root, "data/processed_manifest/result.json", manifest)
            report = integrity_audit.run_audit(args)
            self.assertIn("recompute_mismatch:data/processed/result.txt", report["failures"])

    def test_post_audit_commit_without_numbered_repair_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            init_git_fixture_repo(root)
            audited = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
            ).stdout.strip()
            report = _valid_report(root, audited_sha=audited)
            write_json(root, "reports/status/integrity_audit/audit.json", report)
            write_text(root, "unrelated.txt", "out of band\n")
            subprocess.run(["git", "add", "unrelated.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "out of band"], cwd=root, check=True, capture_output=True)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_post_approval_commit", _reasons(result))

    def test_gate_rejects_same_family_and_tampered_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            report["executor"]["audit_family"] = "codex"
            report["surface_rebuilds"][0]["outputs"][0]["matches_manifest"] = False
            write_json(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_family_of_builder", _reasons(result))
            self.assertIn("integrity_audit_recompute_mismatch", _reasons(result))

    def test_scratch_argv_is_detached_and_has_no_commit_push(self) -> None:
        argv = integrity_audit.scratch_worktree_argv(Path("/repo"), Path("/tmp/audit"))
        self.assertEqual(argv[:4], ["git", "worktree", "add", "--detach"])
        self.assertNotIn("commit", argv)
        self.assertNotIn("push", argv)

    def test_modeling_mode_reruns_locked_seed_with_tolerance_and_rederives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root)
            args.mode = "modeling"
            write_text(
                root,
                "scripts/run_model.py",
                "from pathlib import Path\nimport json\nPath('reports/models').mkdir(parents=True, exist_ok=True)\nPath('reports/models/rerun.json').write_text(json.dumps({'objective': 1.0005}))\n",
            )
            experiment_manifest = write_json(
                root,
                "reports/models/experiment_E1.json",
                {
                    "schema_version": "research_swarm.experiment_manifest.v1",
                    "experiment_id": "E1",
                    "instance_id": "toy",
                    "seed": 11,
                    "budget": 100,
                    "solver": "toy",
                    "solver_version": "1",
                    "converged": True,
                    "outputs": {},
                    "reproduction_command": "python scripts/run_model.py --seed 11",
                    "audit_outputs": [
                        {
                            "path": "reports/models/rerun.json",
                            "comparison": "numeric_json",
                            "json_key": "objective",
                            "expected": 1.0,
                            "tolerance": 0.001,
                        }
                    ],
                },
            )
            write_json(
                root,
                "reports/status/releases/release.json",
                {
                    "artifacts": {
                        "models": [
                            {
                                "path": "reports/models/experiment_E1.json",
                                "sha256": _sha(experiment_manifest),
                            }
                        ]
                    }
                },
            )
            transcript = json.loads(args.mock_transcript.read_text())
            transcript["theoretical_rederivations"] = [{"claim_id": "C1", "status": "pass"}]
            write_json(root, "mock_audit.json", transcript)
            report = integrity_audit.run_audit(args)
            self.assertEqual(report["status"], "pass", report["failures"])
            self.assertEqual(report["experiment_recomputations"][0]["status"], "pass")
            self.assertEqual(report["theoretical_rederivations"][0]["status"], "pass")

    def test_hybrid_gate_requires_union_and_seam_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            report["mode"] = "hybrid"
            report["experiment_recomputations"] = [{"status": "pass"}]
            report["theoretical_rederivations"] = [{"status": "pass"}]
            report["seam_audits"] = []
            write_json(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_seam_failed", _reasons(result))


class LiteratureTest(unittest.TestCase):
    def _corpus(self, root: Path) -> tuple[str, Path]:
        fixture = write_text(root, "fixtures/paper.txt", "The intervention reduced measured latency by 12 percent.\n")
        request = write_json(
            root,
            "request.json",
            {
                "schema_version": "research_swarm.literature_request.v1",
                "acquisition_id": "primary",
                "search_strategy": {
                    "databases": ["OpenAlex"],
                    "queries": ["latency intervention"],
                    "inclusion_criteria": ["empirical study"],
                    "executor_family": "codex",
                },
                "entries": [
                    {
                        "citekey": "smith2025",
                        "title": "Latency Study",
                        "authors": ["Smith, A."],
                        "year": 2025,
                        "doi": "10.0000/example",
                        "url": "https://example.invalid/paper",
                        "format": "txt",
                        "fixture": fixture.name,
                    }
                ],
            },
        )
        path = literature.acquire(
            repo=root,
            request_path=request,
            retrieval_date=__import__("datetime").date(2026, 7, 10),
            fixture_dir=fixture.parent,
            allow_network=False,
        )
        return "The intervention reduced measured latency by 12 percent.", path

    def test_mock_acquisition_and_corpus_generated_bibtex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            _, manifest = self._corpus(root)
            self.assertTrue(manifest.is_file())
            bib = root / "reports/paper/references.bib"
            write_text(
                root,
                "reports/paper/references.bib",
                "@misc{local:protocol,\n  title = {Protocol},\n  note = {Path: docs/protocol.md}\n}\n",
            )
            literature.generate_bib(repo=root, output=bib)
            self.assertIn("Retrieval-Evidence: data/raw/literature/2026-07-10/smith2025.txt#", bib.read_text())
            self.assertIn("@misc{local:protocol", bib.read_text())
            with chdir(root):
                self.assertTrue(quality_gates.gate_literature_corpus().ok)

    def test_literature_claim_evidence_span_must_match_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._corpus(root)
            literature.generate_bib(repo=root, output=root / "reports/paper/references.bib")
            write_json(
                root,
                "contracts/claims.yaml",
                {
                    "schema_version": "research_swarm.claims.v1",
                    "claims": [
                        {
                            "claim_id": "L1",
                            "type": "literature",
                            "citation_key": "smith2025",
                            "evidence_span": "This quotation was fabricated.",
                        }
                    ],
                },
            )
            with chdir(root):
                result = quality_gates.gate_citation_integrity()
            self.assertIn("literature_claim_evidence_span_mismatch", _reasons(result))

    def test_uncovered_recall_cluster_escalates_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._corpus(root)
            search = write_json(
                root,
                "recall.json",
                {
                    "schema_version": "research_swarm.recall_search.v1",
                    "primary_search_strategy": {
                        "databases": ["OpenAlex"],
                        "queries": ["latency intervention"],
                        "inclusion_criteria": ["empirical study"],
                        "executor_family": "codex",
                    },
                    "search_strategy": {
                        "databases": ["Semantic Scholar"],
                        "queries": ["response time experiment"],
                        "inclusion_criteria": ["empirical study"],
                        "executor_family": "claude",
                    },
                    "retrieved": [
                        {"citekey": "gap1", "cluster": "queueing"},
                        {"citekey": "gap2", "cluster": "queueing"},
                    ],
                },
            )
            output = root / "reports/status/recall_audit/audit.json"
            report = literature.recall_audit(repo=root, search_path=search, output=output, cluster_threshold=2)
            self.assertTrue(report["requires_human_escalation"])
            self.assertIn("@human", report["human_escalation"])
            with chdir(root):
                result = quality_gates.gate_recall_audit()
            self.assertIn("recall_audit_uncovered_cluster", _reasons(result))

    def test_lit_task_mini_prisma_and_independence_are_linted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            task = write_task(
                root,
                "backlog",
                "T910",
                schema="v2",
                task_kind="lit_review",
                workstream="W-Lit",
                allow_network=True,
            )
            text = task.read_text()
            text = text.replace("allow_network: true", "allow_network: true\nsearch_phase: acquisition\nsearch_family: codex\nsearch_databases:\n  - OpenAlex\nsearch_queries:\n  - latency\ninclusion_criteria:\n  - empirical")
            task.write_text(text)
            with chdir(root):
                self.assertEqual(quality_gates.gate_task_lint().details["failures"], [])

    def test_claim_alignment_referee_finding_is_advisory_only(self) -> None:
        finding = {
            "check_id": "LIT_CITATION_ALIGNMENT",
            "severity": "major",
            "verdict": "not_supported",
        }
        self.assertTrue(swarm._referee_finding_is_advisory("lit_review", finding))
        self.assertFalse(swarm._referee_finding_is_advisory("writing", finding))


class PromptSurfaceTest(unittest.TestCase):
    def test_prompt_surface_is_hash_pinned_and_mandated(self) -> None:
        with chdir(REPO):
            result = quality_gates.gate_prompt_surface()
        self.assertTrue(result.ok, result.details)

    def test_prompt_tamper_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            scaffold_runtime_repo(root)
            with (root / "contracts/prompts/worker.md").open("a", encoding="utf-8") as handle:
                handle.write("\ntampered\n")
            with chdir(root):
                result = quality_gates.gate_prompt_surface()
            self.assertTrue(any("prompt_surface_hash_mismatch:worker" in item for item in result.details["failures"]))


class ReleaseAuditRequirementTest(unittest.TestCase):
    def test_release_assembly_blocks_when_required_audit_is_absent(self) -> None:
        import test_release_assembly

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            test_release_assembly.scaffold_release_ready_repo(root)
            framework_path = root / "contracts/framework.json"
            framework = json.loads(framework_path.read_text())
            framework["features"]["integrity_audit_required_for_release"] = True
            framework_path.write_text(json.dumps(framework, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(SystemExit, "failed_gates=integrity_audit"):
                test_release_assembly.release_assembly.assemble_release_manifest(
                    root, __import__("datetime").date(2026, 7, 10)
                )


if __name__ == "__main__":
    unittest.main()
