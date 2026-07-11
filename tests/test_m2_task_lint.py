from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import (
    chdir,
    load_quality_gates_module,
    load_swarm_module,
    register_historical_exemption,
    scaffold_runtime_repo,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()
swarm = load_swarm_module()

import swarm_taskfile


class TaskLintV2Test(unittest.TestCase):
    REPO = Path(__file__).resolve().parents[1]

    def _root(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        return root

    def _failures(self, root: Path) -> list[dict[str, object]]:
        with chdir(root):
            result = quality_gates.gate_task_lint()
        return result.details["failures"]

    def _assert_reason(
        self,
        root: Path,
        reason: str,
        *,
        field: str | None = None,
    ) -> dict[str, object]:
        failures = self._failures(root)
        self.assertEqual(len(failures), 1, failures)
        diagnostic = failures[0]
        self.assertEqual(diagnostic["reason"], reason)
        self.assertEqual(
            set(diagnostic),
            {"task", "field", "reason", "expected", "actual"},
        )
        if field is not None:
            self.assertEqual(diagnostic["field"], field)
        return diagnostic

    def test_valid_v2_s_m_l_tasks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T800",
                schema="v2",
                budgets={"max_wall_clock": "2h", "max_tokens": "250K", "max_cost_usd": 25},
            )
            write_task(
                root,
                "backlog",
                "T801",
                schema="v2",
                complexity_tier="M",
                recon_required=True,
                budgets={"max_wall_clock": "90m", "max_tokens": 200000, "max_cost_usd": 20},
            )
            write_task(
                root,
                "backlog",
                "T802",
                schema="v2",
                complexity_tier="L",
                checkpoint_contract="progress_file",
                recon_required=True,
                budgets={"max_wall_clock": "3600s", "max_tokens": 300000, "max_cost_usd": 30},
            )

            self.assertEqual(self._failures(root), [])

    def test_v1_parser_keeps_historical_mapping_like_values_as_strings(self) -> None:
        parsed = swarm_taskfile.parse_task_frontmatter(
            """---
task_id: T799
budgets: {max_wall_clock: 1h}
inputs:
  - path: input.json
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
gates:
  - "make gate"
---
"""
        )
        self.assertEqual(
            parsed,
            {
                "task_id": "T799",
                "budgets": "{max_wall_clock: 1h}",
                "inputs": ["path: input.json"],
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "gates": ["make gate"],
            },
        )

    def test_invalid_task_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T803", schema="v2", task_kind="protocol")
            self._assert_reason(root, "invalid_task_kind", field="task_kind")

    def test_invalid_complexity_tier_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T804", schema="v2", complexity_tier="XL")
            self._assert_reason(root, "invalid_complexity_tier", field="complexity_tier")

    def test_invalid_checkpoint_contract_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T805", schema="v2", checkpoint_contract="hourly")
            self._assert_reason(root, "invalid_checkpoint_contract", field="checkpoint_contract")

    def test_l_task_requires_progress_file_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T806",
                schema="v2",
                complexity_tier="L",
                recon_required=True,
            )
            self._assert_reason(root, "checkpoint_required_for_l", field="checkpoint_contract")

    def test_m_task_requires_recon_or_nonempty_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T807",
                schema="v2",
                complexity_tier="M",
                recon_required=False,
            )
            self._assert_reason(root, "recon_required_for_tier", field="recon_required")

    def test_m_task_accepts_explicit_recon_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T808",
                schema="v2",
                complexity_tier="M",
                recon_required=False,
                recon_waiver="Existing reconnaissance is hash-pinned in the task inputs.",
            )
            self.assertEqual(self._failures(root), [])

    def test_duplicate_success_criterion_ids_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T809",
                schema="v2",
                success_criteria=[
                    {"id": "SC1", "statement": "First", "verification": "make gate"},
                    {"id": "SC1", "statement": "Second", "verification": "make test"},
                ],
            )
            self._assert_reason(
                root,
                "duplicate_success_criterion_id",
                field="success_criteria[1].id",
            )

    def test_empty_success_criterion_verification_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T810",
                schema="v2",
                success_criteria=[{"id": "SC1", "statement": "First", "verification": ""}],
            )
            self._assert_reason(
                root,
                "empty_success_criterion_verification",
                field="success_criteria[0].verification",
            )

    def test_gate_ending_in_quote_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T811", schema="v2", gates=['python -c "pass"'])
            self._assert_reason(root, "gate_ends_in_quote", field="gates[0]")

    def test_network_string_gate_rejected_outside_network_workstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T812",
                schema="v2",
                workstream="W8",
                gates=["curl https://example.invalid"],
            )
            reasons = {f["reason"] for f in self._failures(root)}
            self.assertIn("network_string_in_gate", reasons)
            self.assertIn("gate_interpreter_not_allowlisted", reasons)

    def test_bad_wall_clock_budget_form_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T813",
                schema="v2",
                budgets={"max_wall_clock": "soon", "max_tokens": 1000, "max_cost_usd": 1},
            )
            self._assert_reason(root, "invalid_budget_value", field="budgets.max_wall_clock")

    def test_malformed_input_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T814",
                schema="v2",
                inputs=[{"manifest": "data/processed_manifest/input.json", "sha256": "not-a-hash"}],
            )
            self._assert_reason(root, "invalid_input_sha256", field="inputs[0].sha256")

    def test_validation_comparison_basis_disjoint_hash_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T815",
                schema="v2",
                inputs=[{"manifest": "construction.json", "sha256": "a" * 64}],
            )
            write_task(
                root,
                "backlog",
                "T816",
                schema="v2",
                task_kind="validation",
                constructed_by="T815",
                inputs=[
                    {
                        "manifest": "comparison.json",
                        "sha256": "b" * 64,
                        "comparison_basis": True,
                    }
                ],
            )
            self.assertEqual(self._failures(root), [])

    def test_validation_comparison_basis_any_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T817",
                schema="v2",
                inputs=[{"manifest": "construction.json", "sha256": "a" * 64}],
            )
            write_task(
                root,
                "backlog",
                "T818",
                schema="v2",
                task_kind="validation",
                constructed_by="T817",
                inputs=[
                    {
                        "manifest": "overlap.json",
                        "sha256": "a" * 64,
                        "comparison_basis": True,
                    },
                    {
                        "manifest": "disjoint.json",
                        "sha256": "b" * 64,
                        "comparison_basis": True,
                    },
                ],
            )
            self._assert_reason(root, "comparison_basis_not_disjoint", field="inputs")

    def test_validation_missing_constructed_by_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(
                root,
                "backlog",
                "T819",
                schema="v2",
                task_kind="validation",
                inputs=[
                    {
                        "manifest": "comparison.json",
                        "sha256": "b" * 64,
                        "comparison_basis": True,
                    }
                ],
            )
            self._assert_reason(root, "missing_constructed_by", field="constructed_by")

    def test_v1_task_requires_hash_pinned_historical_exemption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            task_path = write_task(root, "active", "T820", schema="v1", state="active")
            self._assert_reason(root, "unexempted_v1_task", field="task_schema")

            register_historical_exemption(
                root,
                section="tasks",
                rel_path=task_path.relative_to(root).as_posix(),
                extra={"schema_version": "v1"},
            )
            self.assertEqual(self._failures(root), [])

    def test_ready_backlog_excludes_lint_failing_v2_task_and_journals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T821", schema="v2", task_kind="not-a-kind")
            contract = swarm.load_framework_contract(root)
            tasks = swarm.load_tasks(contract)

            with mock.patch.object(swarm, "_record_swarm_event") as record_event:
                ready = swarm.ready_backlog_tasks(tasks, set(), contract)

            self.assertEqual(ready, [])
            record_event.assert_called_once()
            event = record_event.call_args.args[1]
            self.assertEqual(event["event"], "task_lint_rejected")
            self.assertEqual(event["task_id"], "T821")
            self.assertEqual(event["diagnostics"][0]["reason"], "invalid_task_kind")

    def test_task_v2_template_passes_lint_after_placeholder_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            template = (self.REPO / ".orchestrator/templates/task_v2.md").read_text(encoding="utf-8")
            replacements = {
                "T___": "T822",
                "<concise task title>": "Template lint fixture",
                "W__": "W8",
                "# program_id: w6_analysis  # required when instantiating a contracts/program_templates node": "program_id: bridge_campaign",
                "# program_node: estimation_plan  # exact node_id from the selected mode template": "program_node: calibration",
                "<observable completion statement>": "The declared output exists",
                "<offline command or artifact pointer>": "make gate",
                "<upstream manifest path>": "data/processed_manifest/input.json",
                "<64-character manifest sha256>": "c" * 64,
                "<path/to/file_or_small_prefix>": "src/example.py",
                "<output path>": "src/example.py",
                "YYYY-MM-DD": "2026-07-10",
            }
            rendered = template
            for old, new in replacements.items():
                rendered = rendered.replace(old, new)
            write_text(root, ".orchestrator/backlog/T822_template.md", rendered)

            self.assertEqual(self._failures(root), [])

    def test_rule_diagnostics_are_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            write_task(root, "backlog", "T823", schema="v2", complexity_tier="XL")
            failures = self._failures(root)
            self.assertEqual(json.loads(json.dumps(failures)), failures)


if __name__ == "__main__":
    unittest.main()
