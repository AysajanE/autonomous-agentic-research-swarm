from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    chdir,
    load_quality_gates_module,
    scaffold_runtime_repo,
    write_json,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()
_FIXTURES = _TESTS_ROOT / "fixtures/m3a_modeling"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_lock(root: Path, phase: str, body: str) -> str:
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    write_text(
        root,
        f"docs/prereg/{phase}.md",
        "\n".join(
            [
                "---",
                "schema_version: research_swarm.prereg_lock.v1",
                f"phase: {phase}",
                "status: locked",
                "locked_at_utc: 2026-07-10T12:00:00Z",
                f"locked_sha256: {digest}",
                "locked_by: Test Owner",
                "lock_version: 1",
                "---",
                "",
            ]
        )
        + body,
    )
    return digest


def _filled_model_spec(root: Path) -> Path:
    return write_text(
        root,
        "contracts/model_spec.md",
        (_FIXTURES / "model_spec.md").read_text(encoding="utf-8"),
    )


def _experiment_spec(root: Path) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = json.loads(
        (_FIXTURES / "experiment_spec.json").read_text(encoding="utf-8")
    )
    return write_json(root, "contracts/experiments/toy.json", payload), payload


def _lock_a(root: Path, *, fill_model: bool = True) -> tuple[str, dict[str, object]]:
    model = _filled_model_spec(root) if fill_model else root / "contracts/model_spec.md"
    spec_path, spec = _experiment_spec(root)
    body = (
        "# Lock A\n\n"
        f"- path: contracts/model_spec.md\n  sha256: {_sha(model)}\n"
        f"- path: contracts/experiments/toy.json\n  sha256: {_sha(spec_path)}\n"
    )
    return _active_lock(root, "lock_a", body), spec


def _lock_b(root: Path, manifest_paths: list[Path]) -> str:
    body = "# Lock B\n\n" + "".join(
        f"- path: {path.relative_to(root).as_posix()}\n  sha256: {_sha(path)}\n"
        for path in sorted(manifest_paths)
    )
    return _active_lock(root, "lock_b", body)


def _output_entry(root: Path, rel: str, content: str = "output\n") -> dict[str, object]:
    path = write_text(root, rel, content)
    return {"path": rel, "sha256": _sha(path), "bytes": path.stat().st_size}


def _synthetic_instance(root: Path, *, instance_id: str = "toy") -> Path:
    output = _output_entry(root, f"reports/models/{instance_id}_instance.txt")
    payload = json.loads(
        (_FIXTURES / "synthetic_instance_template.json").read_text(encoding="utf-8")
    )
    payload["instance_id"] = instance_id
    payload["outputs"] = [output]
    return write_json(
        root,
        f"contracts/instances/{instance_id}.json",
        payload,
    )


def _bridge_instance(root: Path, *, stale_source: bool = False) -> Path:
    source = write_json(root, "data/processed_manifest/source.json", {"source": "toy"})
    validation = write_json(root, "reports/validation/bridge.json", {"status": "green"})
    source_sha = "0" * 64 if stale_source else _sha(source)
    return write_json(
        root,
        "contracts/instances/bridge.json",
        {
            "schema_version": "research_swarm.instance_manifest.v1",
            "instance_id": "bridge",
            "source_processed_manifests": [
                {"path": "data/processed_manifest/source.json", "sha256": source_sha}
            ],
            "generator_command": "python scripts/generate_bridge.py",
            "generated_at_utc": "2026-07-10T11:00:00Z",
            "outputs": [_output_entry(root, "reports/models/bridge_instance.txt")],
            "pre_bridge_validation": [
                {
                    "path": "reports/validation/bridge.json",
                    "sha256": _sha(validation),
                    "status": "green",
                }
            ],
        },
    )


def _experiment_manifest(
    root: Path,
    *,
    seed: int = 11,
    budget: int = 100,
    gap: float | None = None,
    dispersion: bool = False,
) -> Path:
    payload: dict[str, object] = json.loads(
        (_FIXTURES / "experiment_manifest_template.json").read_text(encoding="utf-8")
    )
    payload["seed"] = seed
    payload["budget"] = budget
    if gap is not None:
        payload["optimality_gap"] = gap
    if dispersion:
        artifact = write_json(root, "reports/models/dispersion.json", {"variance": 0.25})
        payload["dispersion_artifact"] = {
            "path": "reports/models/dispersion.json",
            "sha256": _sha(artifact),
        }
    return write_json(root, "reports/models/experiment_E1.json", payload)


def _claim(root: Path, claim_type: str, *, experiment: Path | None = None, headline: bool = False) -> dict[str, object]:
    evidence = experiment or write_text(root, "reports/evidence.txt", "evidence\n")
    uncertainty = write_text(root, "reports/uncertainty.txt", "uncertainty\n")
    claim: dict[str, object] = {
        "claim_id": "C1",
        "statement": "Toy claim.",
        "type": claim_type,
        "supporting_artifacts": [
            {"path": evidence.relative_to(root).as_posix(), "sha256": _sha(evidence)}
        ],
        "verification_command": "make gate",
        "uncertainty_artifact": None,
        "uncertainty_justification": "A theorem has no sampling uncertainty.",
    }
    if claim_type in {"computational", "counterfactual"}:
        claim["uncertainty_artifact"] = {
            "path": "reports/uncertainty.txt",
            "sha256": _sha(uncertainty),
        }
    if headline:
        claim["headline"] = True
    return claim


def _claims(root: Path, claims: list[dict[str, object]]) -> None:
    write_json(
        root,
        "contracts/claims.yaml",
        {"schema_version": "research_swarm.claims.v1", "claims": claims},
    )


class M3aModelingBatteryTest(unittest.TestCase):
    def _root(self, tmp: str, *, mode: str = "modeling") -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root, mode=mode)
        return root

    def test_instance_manifest_variants_and_stale_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _synthetic_instance(root)
            with chdir(root):
                self.assertTrue(quality_gates.gate_instance_manifest_conformance().ok)
            _bridge_instance(root, stale_source=True)
            with chdir(root):
                red = quality_gates.gate_instance_manifest_conformance()
            self.assertFalse(red.ok)
            self.assertIn("content_binding_sha256_mismatch", {item["reason"] for item in red.details["failures"]})

    def test_seed_budget_lock_rejects_unlocked_cell_then_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _lock_a(root)
            _experiment_manifest(root, seed=99)
            with chdir(root):
                red = quality_gates.gate_seed_budget_lock()
            self.assertFalse(red.ok)
            self.assertIn("seed_outside_active_lock", {item["reason"] for item in red.details["failures"]})
            _experiment_manifest(root, seed=11, budget=100)
            with chdir(root):
                self.assertTrue(quality_gates.gate_seed_budget_lock().ok)

    def test_gap_convergence_requires_gap_and_dispersion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            manifest = _experiment_manifest(root)
            _claims(root, [_claim(root, "computational", experiment=manifest)])
            with chdir(root):
                red = quality_gates.gate_gap_convergence()
            reasons = {item["reason"] for item in red.details["failures"]}
            self.assertIn("missing_or_invalid_optimality_gap", reasons)
            self.assertIn("missing_per_instance_dispersion_artifact", reasons)
            manifest = _experiment_manifest(root, gap=0.01, dispersion=True)
            _claims(root, [_claim(root, "computational", experiment=manifest)])
            with chdir(root):
                self.assertTrue(quality_gates.gate_gap_convergence().ok)

    def test_numeric_falsification_true_and_subtly_false_lemma(self) -> None:
        specs = json.loads((_FIXTURES / "falsification_specs.json").read_text(encoding="utf-8"))
        true_spec = specs["true"]
        self.assertEqual(quality_gates.evaluate_falsification_spec(true_spec), [])
        false_spec = specs["false"]
        failures = quality_gates.evaluate_falsification_spec(false_spec)
        self.assertIn("inequality_violated", {item["reason"] for item in failures})

    def test_theoretical_falsification_gate_blocks_false_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            claim = _claim(root, "theoretical")
            claim["falsification_spec"] = {
                "inequalities": ["x >= 0"],
                "comparative_statics": [],
                "sample_points": [{"x": -0.01}],
            }
            _claims(root, [claim])
            with chdir(root):
                self.assertFalse(quality_gates.gate_theoretical_falsification().ok)
            claim["falsification_spec"]["sample_points"] = [{"x": 0.01}]
            _claims(root, [claim])
            with chdir(root):
                self.assertTrue(quality_gates.gate_theoretical_falsification().ok)

    def test_sweep_enumeration_is_deterministic(self) -> None:
        spec = {
            "grid": {"dimensions": {"beta": [3, 4], "alpha": [1, 2]}},
            "seeds": [7],
            "budget": [10, 20],
        }
        first = quality_gates.enumerate_cells(spec)
        second = quality_gates.enumerate_cells(spec)
        self.assertEqual(first, second)
        self.assertEqual(first[0], {"alpha": 1, "beta": 3, "seed": 7, "budget": 10})
        self.assertEqual(len(first), 8)

    def test_sweep_artifact_missing_then_full_grid_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            lock_hash, spec = _lock_a(root)
            claim = _claim(root, "computational", headline=True)
            claim["lock_a_sha256"] = lock_hash
            _claims(root, [claim])
            with chdir(root):
                self.assertFalse(quality_gates.gate_sweep_artifact().ok)
                cells = quality_gates.enumerate_cells(spec)
            write_json(
                root,
                "reports/models/sweeps/C1.json",
                {
                    "schema_version": "research_swarm.sweep_artifact.v1",
                    "claim_id": "C1",
                    "cells": cells,
                    "survival_count": len(cells),
                },
            )
            with chdir(root):
                self.assertTrue(quality_gates.gate_sweep_artifact().ok)

    def test_ambiguous_model_spec_blocks_active_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "contracts/model_spec.md", "# Model spec\n\nTBD\n")
            _lock_a(root, fill_model=False)
            with chdir(root):
                red = quality_gates.gate_prereg_conformance()
            self.assertFalse(red.ok)
            self.assertIn("ambiguous_locked_model_spec", {item["reason"] for item in red.details["failures"]})

    def test_hybrid_interface_stale_source_then_green(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, mode="hybrid")
            manifest = _bridge_instance(root, stale_source=True)
            with chdir(root):
                red = quality_gates.gate_hybrid_interface_conformance()
            self.assertFalse(red.ok)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["source_processed_manifests"][0]["sha256"] = _sha(root / "data/processed_manifest/source.json")
            write_json(root, "contracts/instances/bridge.json", payload)
            with chdir(root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance().ok)

    def test_hybrid_modeling_task_direct_processed_input_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, mode="hybrid")
            task = write_task(
                root,
                "backlog",
                "T990",
                schema="v2",
                task_kind="model",
                inputs=[{"path": "data/processed/direct.csv", "sha256": "1" * 64}],
            )
            with chdir(root):
                red = quality_gates.gate_hybrid_interface_conformance(task_kind="model")
            self.assertFalse(red.ok)
            task.write_text(
                task.read_text(encoding="utf-8").replace(
                    "data/processed/direct.csv", "contracts/instances/toy.json"
                ),
                encoding="utf-8",
            )
            with chdir(root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance(task_kind="model").ok)

    def test_lock_b_detects_post_generation_manifest_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, mode="hybrid")
            manifest = _bridge_instance(root)
            _lock_b(root, [manifest])
            with chdir(root):
                self.assertTrue(quality_gates.gate_hybrid_interface_conformance().ok)
            manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with chdir(root):
                red = quality_gates.gate_hybrid_interface_conformance()
            self.assertFalse(red.ok)
            self.assertIn("lock_binding_sha256_mismatch", {item["reason"] for item in red.details["failures"]})

    def test_counterfactual_claim_requires_lock_b_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp, mode="hybrid")
            manifest = _bridge_instance(root)
            lock_hash = _lock_b(root, [manifest])
            claim = _claim(root, "counterfactual")
            _claims(root, [claim])
            with chdir(root):
                self.assertFalse(quality_gates.gate_claim_evidence_ledger().ok)
            claim["lock_b_sha256"] = lock_hash
            claim["registered_at_utc"] = "2026-07-10T12:00:00Z"
            _claims(root, [claim])
            with chdir(root):
                self.assertTrue(quality_gates.gate_claim_evidence_ledger().ok)

    def test_render_qa_and_figure_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/figures/toy.svg", "<svg/>\n")
            write_text(root, "reports/paper/index.qmd", "Figure Figure ??\n\n![](../figures/toy.svg)\n")
            with chdir(root):
                self.assertFalse(quality_gates.gate_render_qa().ok)
            write_text(
                root,
                "reports/paper/index.qmd",
                "Figure @fig-toy.\n\n![Toy caption.](../figures/toy.svg){#fig-toy}\n",
            )
            with chdir(root):
                self.assertTrue(quality_gates.gate_render_qa().ok)

    def test_text_overlap_detects_corpus_copy_and_self_repeat(self) -> None:
        paragraph = " ".join(f"word{index}" for index in range(40))
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_text(root, "reports/paper/index.qmd", f"# Paper\n\n{paragraph}\n\n{paragraph}\n")
            write_text(root, "data/raw/literature/source.txt", paragraph + "\n")
            with chdir(root):
                red = quality_gates.gate_text_overlap()
            reasons = {item["reason"] for item in red.details["failures"]}
            self.assertIn("repeated_manuscript_paragraph", reasons)
            self.assertIn("literature_near_duplicate_span", reasons)

    def test_checklist_answers_are_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_json(
                root,
                "contracts/venue.yaml",
                {
                    "checklist": [
                        {
                            "question": "Is code released?",
                            "answer": "yes",
                            "derived_from": "replication/README.md",
                        }
                    ]
                },
            )
            with chdir(root):
                self.assertFalse(quality_gates.gate_checklist_derivation().ok)
            write_text(root, "replication/README.md", "# replication\n")
            with chdir(root):
                self.assertTrue(quality_gates.gate_checklist_derivation().ok)

    def test_mode_matrix_skips_or_activates_modeling_and_hybrid_gates(self) -> None:
        empirical = set(quality_gates._active_gates("empirical"))
        modeling = set(quality_gates._active_gates("modeling"))
        bridge = set(quality_gates._active_gates("hybrid", "bridge"))
        self.assertNotIn("seed_budget_lock", empirical)
        self.assertIn("seed_budget_lock", modeling)
        self.assertIn("hybrid_interface_conformance", bridge)
        self.assertNotIn("seed_budget_lock", bridge)

    def test_empty_modeling_and_hybrid_fixture_repos_run_green(self) -> None:
        for mode in ("modeling", "hybrid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = self._root(tmp, mode=mode)
                write_task(root, "backlog", "T999", task_kind="etl")
                with chdir(root):
                    results = quality_gates._collect_gate_results()
                self.assertTrue(all(result.ok for result in results.values()), {name: result.details for name, result in results.items() if not result.ok})
                if mode == "modeling":
                    self.assertFalse(results["seed_budget_lock"].details.get("skipped", False))
                else:
                    self.assertFalse(results["hybrid_interface_conformance"].details.get("skipped", False))


if __name__ == "__main__":
    unittest.main()
