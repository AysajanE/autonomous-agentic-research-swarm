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


def _load_analysis_builder():
    path = ROOT / "src" / "analysis" / "build_str_release_outputs.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("golden_m4b_analysis_builder", path)
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


def _passing_registry_fixture(
    root: Path,
    *,
    registry_id: str = "section_abstract",
    artifact_relpath: str = "reports/paper/index.qmd",
    journaled: bool = True,
) -> tuple[Path, dict[str, object]]:
    _copy(root, "contracts/schemas/paper_registry_v1.json")
    _copy(root, "contracts/schemas/referee_report_v1.json")
    manuscript = root / "reports/paper/index.qmd"
    if not manuscript.is_file():
        _write(root, "reports/paper/index.qmd", "## Abstract\n\nText.\n")
    artifact = root / artifact_relpath
    if not artifact.is_file():
        _write(root, artifact_relpath, "fixture artifact\n")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "schema_version": "research_swarm.referee_report.v1",
        "task_id": "T900",
        "actor": "Referee",
        "session_id": "fixture-referee-session",
        "referee_family": "independent-fixture",
        "run_manifest_sha256": "a" * 64,
        "rubric_version": "fixture-v1",
        "verdicts": [
            {
                "check_id": "authority-check",
                "verdict": "supported",
                "severity": "minor",
                "evidence_pointer": f"{artifact_relpath}:1",
                "note": "Fixture independently supports the artifact.",
            }
        ],
        "opened_artifacts": [
            {"path": artifact_relpath, "sha256": artifact_sha, "quoted_span": "fixture"}
        ],
        "reviewed_artifacts": [artifact_relpath],
        "overall": "supported",
        "valid": True,
    }
    report_path = _write_json(root, "reports/status/referee_reports/T900_fixture.json", report)
    if journaled:
        _write(
            root,
            "reports/status/events/events.jsonl",
            json.dumps(
                {
                    "event": "referee_invoked",
                    "task_id": "T900",
                    "run_manifest_sha256": "a" * 64,
                    "actor": "Referee",
                    "session_id": "fixture-referee-session",
                    "actor_session": "fixture-referee-session",
                },
                sort_keys=True,
            )
            + "\n",
        )
    _write_json(
        root,
        "reports/paper/registry.json",
        {
            "schema_version": "research_swarm.paper_registry.v1",
            "entries": [
                {
                    "registry_id": registry_id,
                    "kind": registry_id.split("_", 1)[0],
                    "required": True,
                    "status": "passing",
                    "artifact": {"path": artifact_relpath, "sha256": artifact_sha},
                    "referee_report": {
                        "path": report_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    },
                    "reason": "independently supported fixture",
                }
            ],
        },
    )
    return report_path, report


def _rewrite_bound_report(root: Path, report_path: Path, report: dict[str, object]) -> None:
    _write_json(root, report_path.relative_to(root).as_posix(), report)
    registry_path = root / "reports/paper/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["entries"][0]["referee_report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _write_json(root, "reports/paper/registry.json", registry)


class GoldenM4bTests(unittest.TestCase):
    def test_planner_prompt_references_and_embeds_canonical_mode_template(self) -> None:
        # The empirical prompt must name the single canonical file and carry its
        # exact node id, while revision work points to the existing M3b generator.
        swarm = _load_swarm()
        prompt = swarm._render_planner_prompt(mode="empirical", context={})
        self.assertIn("contracts/program_templates/empirical.yaml", prompt)
        self.assertIn('"node_id": "estimation_plan"', prompt)
        self.assertIn("scripts/generate_revision_tasks.py", prompt)

    def test_planner_prompt_rejects_unknown_mode_before_path_lookup(self) -> None:
        swarm = _load_swarm()
        with self.assertRaisesRegex(
            ValueError,
            r"^planner_mode_unsupported:None:expected=empirical,modeling,hybrid$",
        ):
            swarm._render_planner_prompt(mode=None, context={})  # type: ignore[arg-type]

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

    def test_uninstantiated_program_fails_at_release_perimeter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _program_fixture(root)
            result = quality_gates.check_program_conformance(root, release_perimeter=True)
            self.assertFalse(result.ok)
            self.assertEqual(
                _reasons(result),
                {"program_conformance_not_instantiated_at_release"},
            )

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

    def test_raw_html_image_absent_from_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            _write(root, "reports/paper/index.qmd", '<img src="../figures/x.svg" alt="x">\n')
            _write(root, "reports/figures/x.svg", "<svg/>\n")
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("manuscript_exhibit_unregistered", _reasons(result))

    def test_quarto_embed_absent_from_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            _write(root, "reports/paper/index.qmd", "{{< embed ../figures/x.svg >}}\n")
            _write(root, "reports/figures/x.svg", "<svg/>\n")
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("manuscript_exhibit_unregistered", _reasons(result))

    def test_same_stem_different_extension_is_not_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            _write(root, "reports/paper/index.qmd", "![Mismatch](../figures/registered.png)\n")
            _write(root, "reports/figures/registered.png", "png fixture\n")
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("manuscript_exhibit_unregistered", _reasons(result))

    def test_outside_repo_exhibit_reference_is_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            _, entry = _exhibit_fixture(root)
            _write(root, "reports/paper/index.qmd", "![Outside](../../../../outside.svg)\n")
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("manuscript_exhibit_reference_unresolved", _reasons(result))

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

    def test_label_less_figure_derives_false_and_fails_exhibits_gate(self) -> None:
        builder = _load_analysis_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, entry = _exhibit_fixture(root)
            fig, axis = builder.plt.subplots()
            axis.plot([1, 2], [1, 2], label="Series")
            axis.set_ylabel("ETH")
            axis.legend()
            qa = builder.derive_figure_self_qa(
                (axis,),
                declared_unit_tokens=(("ETH",),),
                alt_text="Fixture figure.",
            )
            fig.savefig(output)
            builder.plt.close(fig)
            self.assertFalse(qa["labels"])
            self.assertTrue(qa["legend"])
            self.assertTrue(qa["units"])
            entry["self_qa"] = qa
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("exhibit_self_qa_failed", _reasons(result))

    def test_legendless_figure_derives_false_and_fails_exhibits_gate(self) -> None:
        builder = _load_analysis_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, entry = _exhibit_fixture(root)
            fig, axis = builder.plt.subplots()
            axis.plot([1, 2], [1, 2])
            axis.set_xlabel("Date (UTC)")
            axis.set_ylabel("ETH")
            qa = builder.derive_figure_self_qa(
                (axis,),
                declared_unit_tokens=(("date", "ETH"),),
                alt_text="Fixture figure.",
            )
            fig.savefig(output)
            builder.plt.close(fig)
            self.assertTrue(qa["labels"])
            self.assertFalse(qa["legend"])
            self.assertTrue(qa["units"])
            entry["self_qa"] = qa
            _write_exhibits(root, entry)
            result = quality_gates.check_exhibits_manifest(root)
            self.assertFalse(result.ok)
            self.assertIn("exhibit_self_qa_failed", _reasons(result))

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

    def test_passing_exhibit_registry_entry_must_target_canonical_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, entry = _exhibit_fixture(root)
            _write_exhibits(root, entry)
            _passing_registry_fixture(
                root,
                registry_id="exhibit_registered",
                artifact_relpath="reports/paper/index.qmd",
            )
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_entry_artifact_not_canonical", _reasons(result))

    def test_passing_section_registry_entry_must_target_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, entry = _exhibit_fixture(root)
            _write_exhibits(root, entry)
            _passing_registry_fixture(
                root,
                registry_id="section_abstract",
                artifact_relpath=output.relative_to(root).as_posix(),
            )
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_entry_artifact_not_canonical", _reasons(result))

    def test_passing_registry_requires_journaled_referee_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _passing_registry_fixture(root, journaled=False)
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_passing_referee_unjournaled", _reasons(result))

    def test_passing_registry_requires_supported_overall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path, report = _passing_registry_fixture(root)
            report["overall"] = "not_supported"
            _rewrite_bound_report(root, report_path, report)
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_passing_referee_not_supported", _reasons(result))

    def test_passing_registry_requires_artifact_in_reviewed_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path, report = _passing_registry_fixture(root)
            report["opened_artifacts"] = []
            report["reviewed_artifacts"] = []
            _rewrite_bound_report(root, report_path, report)
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_passing_referee_artifact_unreviewed", _reasons(result))

    def test_passing_registry_rejects_open_major_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path, report = _passing_registry_fixture(root)
            report["verdicts"] = [
                {
                    "check_id": "open-major",
                    "verdict": "not_supported",
                    "severity": "major",
                    "evidence_pointer": "reports/paper/index.qmd:1",
                    "note": "Independent major finding remains open.",
                }
            ]
            _rewrite_bound_report(root, report_path, report)
            result = quality_gates.check_paper_registry(root)
            self.assertFalse(result.ok)
            self.assertIn("paper_registry_passing_referee_open_major", _reasons(result))

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
