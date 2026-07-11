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
from unittest import mock

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


_swarm_events = _load("m3b_batch2_swarm_events", REPO / "scripts/swarm_events.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audit(root: Path, relpath: str, report: dict) -> None:
    """Write an integrity-audit report AND emit the kernel-authored
    `integrity_audit_recorded` journal event that binds it — the gate accepts
    only journal-bound reports (as a real `integrity_audit.py` run does)."""
    write_json(root, relpath, report)
    _swarm_events.append_event(
        root,
        {
            "event": "integrity_audit_recorded",
            "report_path": relpath,
            "report_sha256": _sha(root / relpath),
            "status": report.get("status"),
            "mode": report.get("mode"),
        },
    )


def _reasons(result) -> set[str]:
    return {
        item.get("reason")
        for item in result.details.get("failures", [])
        if isinstance(item, dict)
    }


def _valid_report(root: Path, *, audited_sha: str = "a" * 40) -> dict[str, object]:
    write_text(
        root,
        "contracts/schemas/integrity_audit_v1.json",
        (REPO / "contracts/schemas/integrity_audit_v1.json").read_text(encoding="utf-8"),
    )
    write_text(root, "contracts/project.yaml", "mode: empirical\n")
    write_json(
        root,
        "contracts/framework.json",
        {
            "executors": {
                "integrity_audit": {
                    "backend": "claude",
                    "family": "claude",
                    "model": "fixture-auditor",
                }
            }
        },
    )
    output = write_text(root, "data/processed/result.txt", "69.14%\n")
    manifest = write_json(
        root,
        "data/processed_manifest/result.json",
        {"outputs": [{"path": "data/processed/result.txt", "sha256": _sha(output)}]},
    )
    run = write_json(
        root,
        "reports/status/swarm_runs/T900_run.json",
        {
            "run_id": "run-builder",
            "executor": {"tool": "codex", "family": "manual"},
            "ownership": {"changed_paths": ["data/processed/result.txt"]},
        },
    )
    inventory = write_json(
        root,
        "reports/status/releases/release.json",
        {
            "artifacts": {
                "processed": [{"path": "data/processed/result.txt", "sha256": _sha(output)}],
                "runs": [{"path": run.relative_to(root).as_posix(), "sha256": _sha(run)}],
            }
        },
    )
    digest = _sha(output)
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
            "backend": "claude",
            "model": "fixture-auditor",
            "audit_family": "claude",
            "builder_families": ["codex"],
            "builder_run_manifest_evidence": [
                {
                    "run_manifest": run.relative_to(root).as_posix(),
                    "family": "codex",
                    "family_source": "executor.tool",
                    "artifacts": [{"path": "data/processed/result.txt", "sha256": digest}],
                }
            ],
            "profile": "scratch-worktree",
            "network": "requested_off",
            "commit_push_allowed": False,
            "scratch_kind": "hermetic_temp_copy",
            "effective_confinement": {
                "capability_probe": "fixture_no_namespace",
                "os_enforced": False,
                # backend is "claude" (live) -> honest live tuple: vendor cred +
                # unrestricted egress, not a mock-style scrub/proxy claim.
                "effective_network": "unrestricted_process_egress",
                "credential_isolation": "vendor_credential_retained",
                "filesystem_isolation": "scratch_worktree_plus_mutation_detection",
            },
        },
        "repo_confinement": {
            "excluded_prefixes": [".git/", "reports/status/integrity_audit/"],
            "changed_paths": [],
            "passed": True,
            "enforcement": "detect-and-reject",
        },
        "sampling": {
            "seed_path": "contracts/integrity_audit_seed.txt",
            "non_headline_sample_size": 3,
        },
        "surface_rebuilds": [
            {
                "manifest": manifest.relative_to(root).as_posix(),
                "command": "python scripts/generate_surface.py",
                "returncode": 0,
                "outputs": [
                    {
                        "path": "data/processed/result.txt",
                        "expected_manifest_sha256": digest,
                        "release_inventory_sha256": digest,
                        "recomputed_sha256": digest,
                        "matches_manifest": True,
                        "matches_release_inventory": True,
                    }
                ]
            }
        ],
        "inventory_hash_checks": [
            {
                "path": "data/processed/result.txt",
                "release_inventory_sha256": digest,
                "scratch_sha256": digest,
                "passed": True,
            }
        ],
        "claim_recomputations": [
            {
                "claim_id": "C1",
                "headline": True,
                "claim_type": "descriptive",
                "expected_role_values": {"pre_dencun_mean_str": "69.14%"},
                "recomputed_role_values": {"pre_dencun_mean_str": "69.14%"},
                "expected_numeric_literals": ["69.14%"],
                "recomputed_numeric_literals": ["69.14%"],
                "source_artifacts": [
                    {
                        "path": "data/processed/result.txt",
                        "asserted_sha256": digest,
                        "release_inventory_sha256": digest,
                    }
                ],
                "passed": True,
            }
        ],
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
        write_text(root, "contracts/project.yaml", "mode: empirical\n")
        write_json(
            root,
            "contracts/framework.json",
            {
                "executors": {
                    "integrity_audit": {
                        "backend": "claude",
                        "family": family,
                        "command": "claude",
                        "model": "fixture-auditor",
                    }
                }
            },
        )
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
        write_json(
            root,
            "reports/status/swarm_runs/T900_run.json",
            {
                "run_id": "run-builder",
                "executor": {"tool": "codex", "family": "manual"},
                "ownership": {"changed_paths": ["data/processed/result.txt"]},
            },
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
            builder_family=[],
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

    def test_cli_builder_family_override_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root)
            args.builder_family = ["third-party-label"]
            report = integrity_audit.run_audit(args)
            self.assertIn("builder_family_override_refused", report["failures"])

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
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
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
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_family_of_builder", _reasons(result))
            self.assertIn("integrity_audit_recompute_mismatch", _reasons(result))

    def test_mock_backend_cannot_authorize_release_but_live_contract_report_can(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                self.assertTrue(quality_gates.gate_integrity_audit().ok)
            report["executor"]["backend"] = "mock"
            report["executor"]["model"] = None
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_executor_contract_mismatch", _reasons(result))

    def test_unknown_backend_is_rejected_by_confinement_binding(self) -> None:
        # An unimplemented/typo'd backend, made consistent across framework +
        # report so the executor-contract match passes, must NOT inherit a live
        # confinement tuple by default — the gate rejects unknown backends.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            fw_path = root / "contracts/framework.json"
            framework = json.loads(fw_path.read_text(encoding="utf-8"))
            framework["executors"]["integrity_audit"]["backend"] = "claude_typo"
            write_json(root, "contracts/framework.json", framework)
            report["executor"]["backend"] = "claude_typo"
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertFalse(result.ok)
            self.assertIn("integrity_audit_unknown_backend", _reasons(result))

    def test_gate_rehashes_bytes_even_when_report_flags_claim_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            write_text(root, "data/processed/result.txt", "tampered\n")
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            reasons = _reasons(result)
            self.assertIn("integrity_audit_recompute_mismatch", reasons)
            self.assertIn("integrity_audit_claim_source_hash_mismatch", reasons)

    def test_mode_downgrade_cannot_authorize_current_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            write_text(root, "contracts/project.yaml", "mode: hybrid\n")
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_mode_mismatch", _reasons(result))

    def test_answer_surfaces_are_scrubbed_before_value_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root)
            write_text(root, "reports/tables/answer.csv", "role,value\npre_dencun_mean_str,11.68%\n")
            write_text(root, "reports/figures/answer.svg", "<svg><text>11.68%</text></svg>\n")
            write_text(root, "reports/validation/answer.txt", "11.68%\n")
            claims = json.loads((root / "contracts/claims.yaml").read_text())
            claims["claims"][0]["statement"] = "Pre-Dencun mean STR is 11.68%."
            claims["claims"][0]["manuscript_numeric_literals"] = ["11.68%"]
            claims["claims"][0]["recomputation_roles"] = {"pre_dencun_mean_str": "11.68%"}
            write_json(root, "contracts/claims.yaml", claims)
            transcript = json.loads(args.mock_transcript.read_text())

            def inspect_scrub(**kwargs):
                scratch = kwargs["scratch"]
                for relpath in ("contracts/claims.yaml", "reports/paper", "reports/tables", "reports/figures", "reports/validation"):
                    self.assertFalse((scratch / relpath).exists(), relpath)
                return transcript

            with mock.patch.object(integrity_audit, "_load_transcript", side_effect=inspect_scrub):
                report = integrity_audit.run_audit(args)
            self.assertIn("claim_semantic_role_mismatch:C1", report["failures"])

    def test_manual_builder_self_label_does_not_supply_verified_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            run_path = root / "reports/status/swarm_runs/T900_run.json"
            run = json.loads(run_path.read_text())
            run["executor"] = {"tool": "manual", "family": "codex"}
            write_json(root, run_path.relative_to(root).as_posix(), run)
            report["executor"]["builder_run_manifest_evidence"][0]["family"] = "codex"
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("builder_family_unverified", _reasons(result))

    def test_omitted_same_family_builder_run_is_derived_from_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            write_json(
                root,
                "reports/status/swarm_runs/T901_run.json",
                {
                    "run_id": "same-family-builder",
                    "executor": {"tool": "claude", "family": "manual"},
                    "ownership": {"changed_paths": ["data/processed/result.txt"]},
                },
            )
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            reasons = _reasons(result)
            self.assertIn("builder_family_evidence_mismatch", reasons)
            self.assertIn("integrity_audit_family_of_builder", reasons)

    def test_main_repo_mutation_during_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            args = self._audit_fixture(root)
            transcript = json.loads(args.mock_transcript.read_text())

            def mutate_repo(**_kwargs):
                write_text(root, "outside-audit.txt", "mutation\n")
                return transcript

            with mock.patch.object(integrity_audit, "_load_transcript", side_effect=mutate_repo):
                report = integrity_audit.run_audit(args)
            self.assertTrue(any(item.startswith("main_repo_mutated_during_audit:") for item in report["failures"]))

    def test_future_timestamp_cannot_shadow_committed_audit_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            report = _valid_report(root)
            _write_audit(root, "reports/status/integrity_audit/real.json", report)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "trusted audit"], cwd=root, check=True, capture_output=True)
            forged = dict(report)
            forged["generated_at_utc"] = "2999-01-01T00:00:00Z"
            forged["executor"] = {**report["executor"], "backend": "mock", "model": None}
            write_json(root, "reports/status/integrity_audit/zzz_forged.json", forged)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertEqual(result.details["report"], "reports/status/integrity_audit/real.json")

    def test_scratch_argv_is_detached_and_has_no_commit_push(self) -> None:
        argv = integrity_audit.scratch_worktree_argv(Path("/repo"), Path("/tmp/audit"))
        self.assertEqual(argv[:4], ["git", "worktree", "add", "--detach"])
        self.assertNotIn("commit", argv)
        self.assertNotIn("push", argv)

    def test_rebuild_allowlist_rejects_forbidden_command_in_referenced_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_text(
                root,
                "scripts/rebuild.py",
                "import subprocess\nsubprocess.run(['git', 'status'], check=False)\n",
            )
            self.assertEqual(
                integrity_audit._command_violation("python scripts/rebuild.py", repo=root),
                "referenced_script_forbidden_token:git",
            )

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
                        "processed": [
                            {
                                "path": "data/processed/result.txt",
                                "sha256": _sha(root / "data/processed/result.txt"),
                            }
                        ],
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
            _write_audit(root, "reports/status/integrity_audit/audit.json", report)
            with chdir(root):
                result = quality_gates.gate_integrity_audit()
            self.assertIn("integrity_audit_seam_failed", _reasons(result))


class LiteratureTest(unittest.TestCase):
    def _corpus(self, root: Path) -> tuple[str, Path]:
        write_json(
            root,
            "contracts/framework.json",
            {
                "literature_policy": {
                    "recall_uncovered_cluster_threshold": 2,
                    "fixture_test_corpus_acquisition_ids": [],
                }
            },
        )
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
            write_json(
                root,
                "reports/status/swarm_runs/primary.json",
                {"run_id": "primary", "executor": {"tool": "codex"}},
            )
            write_json(
                root,
                "reports/status/swarm_runs/recall.json",
                {"run_id": "recall", "executor": {"tool": "claude"}},
            )
            search = write_json(
                root,
                "recall.json",
                {
                    "schema_version": "research_swarm.recall_search.v1",
                    "primary_run_manifest": "reports/status/swarm_runs/primary.json",
                    "recall_run_manifest": "reports/status/swarm_runs/recall.json",
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
            self.assertNotIn("recall_audit_executor_run_unverified", _reasons(result))
            self.assertNotIn("recall_audit_query_manifest_unverified", _reasons(result))

    def test_corpus_or_literature_claim_requires_recall_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._corpus(root)
            with chdir(root):
                result = quality_gates.gate_recall_audit()
            self.assertIn("recall_audit_required", _reasons(result))

    def test_fixture_backed_citation_is_rejected_on_release_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._corpus(root)
            literature.generate_bib(repo=root, output=root / "reports/paper/references.bib")
            with chdir(root):
                result = quality_gates.gate_citation_integrity(require_literature_corpus=True)
            self.assertIn("fixture_backed_literature_release_claim", _reasons(result))

    def test_fixture_corpus_must_be_explicitly_pinned_to_be_release_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._corpus(root)
            framework_path = root / "contracts/framework.json"
            framework = json.loads(framework_path.read_text())
            framework["literature_policy"]["fixture_test_corpus_acquisition_ids"] = ["primary"]
            write_json(root, "contracts/framework.json", framework)
            literature.generate_bib(repo=root, output=root / "reports/paper/references.bib")
            with chdir(root):
                result = quality_gates.gate_citation_integrity(require_literature_corpus=True)
            self.assertNotIn("fixture_backed_literature_release_claim", _reasons(result))

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
    def test_audit_and_recall_surfaces_are_operator_owned(self) -> None:
        for index, surface in enumerate(
            ("reports/status/integrity_audit/", "reports/status/recall_audit/"),
            start=920,
        ):
            with self.subTest(surface=surface), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "repo"
                scaffold_runtime_repo(root)
                write_task(
                    root,
                    "backlog",
                    f"T{index}",
                    role="Worker",
                    allowed_paths=[surface],
                    outputs=[surface + "forged.json"],
                    state="backlog",
                )
                with chdir(root):
                    result = quality_gates.gate_operator_surface_ownership()
                self.assertFalse(result.ok)
                self.assertTrue(any(surface in item for item in result.details["failures"]))

    def test_citation_integrity_is_release_required_and_external_citation_blocks(self) -> None:
        import test_release_assembly

        self.assertIn("citation_integrity", test_release_assembly.release_assembly.REQUIRED_RELEASE_GATE_NAMES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            test_release_assembly.scaffold_release_ready_repo(root)
            write_text(
                root,
                "reports/paper/references.bib",
                "@article{external2026,\n  title = {External}\n}\n",
            )
            with self.assertRaisesRegex(SystemExit, "citation_integrity"):
                test_release_assembly.release_assembly.assemble_release_manifest(
                    root, __import__("datetime").date(2026, 7, 10)
                )

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
