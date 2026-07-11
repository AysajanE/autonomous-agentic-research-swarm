from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _load_quality_gates():
    path = ROOT / "scripts" / "quality_gates.py"
    spec = importlib.util.spec_from_file_location("golden_m4b_quality_gates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


quality_gates = _load_quality_gates()


def _load_swarm():
    path = ROOT / "scripts" / "swarm.py"
    spec = importlib.util.spec_from_file_location("golden_m4b_swarm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(root: Path, relpath: str, payload: object) -> Path:
    return _write(root, relpath, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy(root: Path, relpath: str) -> None:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relpath, target)


def _reasons(result) -> set[str]:
    return {
        str(item.get("reason"))
        for item in result.details.get("failures", [])
        if isinstance(item, dict)
    }


def _program_fixture(root: Path, *, mode: str = "empirical") -> None:
    _write(root, "contracts/project.yaml", f"mode: {mode}\n")
    _copy(root, "contracts/schemas/program_template_v1.yaml")
    _copy(root, f"contracts/program_templates/{mode}.yaml")


def _task(
    root: Path,
    task_id: str,
    *,
    program_id: str,
    program_node: str,
    task_kind: str,
    workstream: str,
    role: str = "Worker",
) -> None:
    _write(
        root,
        f".orchestrator/backlog/{task_id}.md",
        "\n".join(
            (
                "---",
                f"task_id: {task_id}",
                f"task_kind: {task_kind}",
                f"workstream: {workstream}",
                f"role: {role}",
                "dependencies: []",
                f"program_id: {program_id}",
                f"program_node: {program_node}",
                "---",
                "## Status",
                "- State: backlog",
                "",
            )
        ),
    )


def _exhibit_fixture(root: Path, *, mode: str = "empirical") -> tuple[Path, dict[str, object]]:
    _write(root, "contracts/project.yaml", f"mode: {mode}\n")
    _copy(root, "contracts/schemas/exhibits_manifest_v1.json")
    builder = _write(root, "src/analysis/build.py", "# fixture builder\n")
    source = _write(root, "data/input.csv", "x\n1\n")
    output = _write(root, "reports/figures/registered.svg", "<svg/>\n")
    entry: dict[str, object] = {
        "exhibit_id": "registered",
        "builder": builder.relative_to(root).as_posix(),
        "inputs": [
            {
                "path": source.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        "output": output.relative_to(root).as_posix(),
        "caption": "Fixture exhibit.",
        "notes": "Golden adversarial fixture.",
        "self_qa": {"labels": True, "legend": True, "units": True, "alt_text": "Fixture alt text."},
    }
    return output, entry


def _write_exhibits(root: Path, entry: dict[str, object]) -> None:
    _write_json(
        root,
        "reports/exhibits/manifest.json",
        {
            "schema_version": "research_swarm.exhibits_manifest.v1",
            "generated_by": "src/analysis/build.py",
            "exhibits": [entry],
        },
    )


class GoldenM4bTests(unittest.TestCase):
    def test_planner_prompt_references_and_embeds_canonical_mode_template(self) -> None:
        # The empirical prompt must name the single canonical file and carry its
        # exact node id, while revision work points to the existing M3b generator.
        swarm = _load_swarm()
        prompt = swarm._render_planner_prompt(mode="empirical", context={})
        self.assertIn("contracts/program_templates/empirical.yaml", prompt)
        self.assertIn('"node_id": "estimation_plan"', prompt)
        self.assertIn("scripts/generate_revision_tasks.py", prompt)

    def test_program_dag_missing_required_node_fails_conformance(self) -> None:
        # The empirical template independently declares 11 required nodes; zero
        # instantiated tasks must therefore produce required-node failures.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _program_fixture(root)
            result = quality_gates.check_program_conformance(root, strict=True)
            self.assertFalse(result.ok)
            self.assertIn("required_program_node_missing", _reasons(result))

    def test_mode_foreign_program_node_fails_conformance(self) -> None:
        # `theory:proof` exists only in the modeling template, so an empirical
        # program fixture must reject it as foreign rather than count it by kind.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _program_fixture(root)
            _task(
                root,
                "T900",
                program_id="theory",
                program_node="proof",
                task_kind="proof",
                workstream="W8",
            )
            result = quality_gates.check_program_conformance(root, strict=True)
            self.assertFalse(result.ok)
            self.assertIn("mode_foreign_program_node", _reasons(result))

    def test_modeling_and_hybrid_templates_are_mode_selected_by_conformance(self) -> None:
        # Independently counted from the frozen shapes: modeling has 5 theory +
        # 4 experiment nodes; hybrid has 5 bridge-campaign nodes.
        for mode, expected_required in (("modeling", 9), ("hybrid", 5)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _program_fixture(root, mode=mode)
                result = quality_gates.check_program_conformance(root, strict=True)
                missing = [
                    item
                    for item in result.details["failures"]
                    if isinstance(item, dict) and item.get("reason") == "required_program_node_missing"
                ]
                self.assertEqual(result.details["mode"], mode)
                self.assertEqual(len(missing), expected_required)

    def test_manuscript_referenced_exhibit_absent_from_manifest_fails(self) -> None:
        # The manuscript target and registered output are distinct paths, so
        # exact/same-stem matching has the independently expected result false.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            _write(root, "reports/paper/index.qmd", "![Missing](../figures/missing.svg)\n")
            _write(root, "reports/figures/missing.svg", "<svg/>\n")
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("manuscript_exhibit_unregistered", _reasons(result))

    def test_manifest_entry_with_bad_input_hash_fails(self) -> None:
        # SHA-256("data/input.csv") is not 64 zeroes, so recomputation must fail.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            entry["inputs"][0]["sha256"] = "0" * 64  # type: ignore[index]
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("exhibit_input_hash_mismatch", _reasons(result))

    def test_figure_missing_units_and_legend_self_qa_fails(self) -> None:
        # Both required boolean assertions are independently false/missing, so
        # the gate must name both fields rather than accepting a partial block.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            entry["self_qa"] = {"labels": True, "alt_text": "Present."}
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            fields = {
                item.get("field")
                for item in result.details["failures"]
                if isinstance(item, dict) and item.get("reason") == "exhibit_self_qa_failed"
            }
            self.assertFalse(result.ok)
            self.assertEqual(fields, {"legend", "units"})

    def test_hand_edited_passing_registry_without_evidence_fails(self) -> None:
        # A null artifact/referee binding cannot authorize `passing`, regardless
        # of the hand-edited status string.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy(root, "contracts/schemas/paper_registry_v1.json")
            _write(root, "reports/paper/index.qmd", "## Abstract\n\nText.\n")
            _write_json(
                root,
                "reports/paper/registry.json",
                {
                    "schema_version": "research_swarm.paper_registry.v1",
                    "entries": [
                        {
                            "registry_id": "section_abstract",
                            "kind": "section",
                            "required": True,
                            "status": "passing",
                            "artifact": None,
                            "referee_report": None,
                            "reason": "hand edited",
                        }
                    ],
                },
            )
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_passing_artifact_missing", _reasons(result))

    def test_failing_required_registry_entry_blocks_release_perimeter(self) -> None:
        # One required entry with status `failing` makes the all-required-passing
        # release predicate false, while remaining truthful in the staged gate.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _copy(root, "contracts/schemas/paper_registry_v1.json")
            manuscript = _write(root, "reports/paper/index.qmd", "## Abstract\n\nText.\n")
            _write_json(
                root,
                "reports/paper/registry.json",
                {
                    "schema_version": "research_swarm.paper_registry.v1",
                    "entries": [
                        {
                            "registry_id": "section_abstract",
                            "kind": "section",
                            "required": True,
                            "status": "failing",
                            "artifact": {
                                "path": "reports/paper/index.qmd",
                                "sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest(),
                            },
                            "referee_report": None,
                            "reason": "awaiting referee",
                        }
                    ],
                },
            )
            staged = quality_gates.check_paper_registry(root)
            release = quality_gates.check_paper_registry(root, release_perimeter=True)
            self.assertTrue(staged.ok)
            self.assertFalse(release.ok)
            self.assertIn("paper_registry_required_entry_failing", _reasons(release))

    def test_modeling_numeric_exhibit_without_paper_value_key_fails(self) -> None:
        # The referenced CSV contains the numeric literal 42 and paper_values has
        # no source binding, so the modeling numeric-registration predicate is false.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root, mode="modeling")
            table = _write(root, "reports/tables/model.csv", "estimate\n42\n")
            entry["exhibit_id"] = "model"
            entry["output"] = table.relative_to(root).as_posix()
            _write(root, "reports/paper/index.qmd", "{{< include ../tables/model.csv >}}\n")
            _write_json(
                root,
                "reports/paper/paper_values.json",
                {"schema_version": "research_swarm.paper_values.v1", "values": {}},
            )
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("modeling_exhibit_numeric_unregistered", _reasons(result))


if __name__ == "__main__":
    unittest.main()
