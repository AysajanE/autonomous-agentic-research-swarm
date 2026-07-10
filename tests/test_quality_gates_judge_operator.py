import sys
from pathlib import Path
import tempfile
import unittest

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from runtime_test_utils import (
    chdir,
    load_quality_gates_module,
    scaffold_runtime_repo,
    write_review_log,
    write_run_manifest,
    write_task,
    write_text,
)


quality_gates = load_quality_gates_module()


class JudgeOperatorGateTest(unittest.TestCase):
    def test_worker_task_cannot_claim_operator_owned_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "backlog",
                "T400",
                workstream="W6",
                task_kind="analysis",
                role="Worker",
                allowed_paths=["reports/catalog.yaml"],
                disallowed_paths=["contracts/"],
                outputs=["reports/catalog.yaml"],
                state="backlog",
                slug="catalog",
            )

            with chdir(root):
                result = quality_gates.gate_operator_surface_ownership()

            self.assertFalse(result.ok)
            failures = result.details.get("failures") or []
            self.assertTrue(any("operator_owned_surface" in failure for failure in failures))

    def test_operator_task_must_stay_inside_ops_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            write_task(
                root,
                "backlog",
                "T401",
                workstream="W6",
                task_kind="analysis",
                role="Operator",
                allowed_paths=["reports/figures/"],
                disallowed_paths=["contracts/"],
                outputs=["reports/figures/str_plot.svg"],
                state="backlog",
                slug="analysis",
            )

            with chdir(root):
                result = quality_gates.gate_operator_surface_ownership()

            self.assertFalse(result.ok)
            failures = result.details.get("failures") or []
            self.assertTrue(any("operator_role_outside_ops_boundary" in failure for failure in failures))

    def test_worker_cannot_claim_referee_report_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T410",
                role="Worker",
                allowed_paths=["reports/status/referee_reports/"],
                outputs=["reports/status/referee_reports/T410_forged.json"],
                state="backlog",
            )
            with chdir(root):
                result = quality_gates.gate_operator_surface_ownership()
            self.assertFalse(result.ok)
            self.assertTrue(any("reports/status/referee_reports/" in item for item in result.details["failures"]))

    def test_worker_cannot_claim_swarm_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)
            write_task(
                root,
                "backlog",
                "T411",
                role="Worker",
                allowed_paths=["reports/status/swarm_runs/T411_forged.json"],
                outputs=["reports/status/swarm_runs/T411_forged.json"],
                state="backlog",
            )
            with chdir(root):
                result = quality_gates.gate_operator_surface_ownership()
            self.assertFalse(result.ok)
            self.assertTrue(any("reports/status/swarm_runs/" in item for item in result.details["failures"]))

    def test_worker_cannot_claim_event_journal_or_calibration_run_surface(self) -> None:
        for index, surface in enumerate(
            (
                "reports/status/events/events.jsonl",
                "reports/status/referee_calibration_runs/fixture.json",
            ),
            start=1,
        ):
            with self.subTest(surface=surface), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scaffold_runtime_repo(root)
                write_task(
                    root,
                    "backlog",
                    f"T41{index + 1}",
                    role="Worker",
                    allowed_paths=[surface],
                    outputs=[surface],
                    state="backlog",
                )
                with chdir(root):
                    result = quality_gates.gate_operator_surface_ownership()
                self.assertFalse(result.ok)
                self.assertTrue(
                    any(surface.rsplit("/", 1)[0] in item for item in result.details["failures"]),
                    result.details,
                )

    def test_done_task_requires_valid_judge_review_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scaffold_runtime_repo(root)

            output_path = "reports/validation/rollup_panel_validation.json"
            write_text(root, output_path, "{}\n")

            task_path = write_task(
                root,
                "done",
                "T402",
                workstream="W5",
                task_kind="validation",
                role="Worker",
                allowed_paths=["reports/validation/"],
                disallowed_paths=["contracts/"],
                outputs=[output_path],
                state="done",
                slug="validation",
            )

            run_manifest_path = write_run_manifest(
                root,
                "T402",
                task_path=task_path.relative_to(root).as_posix(),
                task_role="Worker",
                workstream="W5",
                state_after="ready_for_review",
            )

            with chdir(root):
                missing_result = quality_gates.gate_review_bundle_integrity()

            self.assertFalse(missing_result.ok)
            self.assertTrue(
                any("missing_review_log" in failure for failure in (missing_result.details.get("failures") or []))
            )

            invalid_log = write_review_log(
                root,
                "T402",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=run_manifest_path.relative_to(root).as_posix(),
                reviewer_role="Worker",
                outcome="approve",
            )

            with chdir(root):
                invalid_result = quality_gates.gate_judge_review_log_validity()

            self.assertFalse(invalid_result.ok)
            self.assertTrue(
                any("invalid_reviewer_role" in failure for failure in (invalid_result.details.get("failures") or []))
            )

            if invalid_log.exists():
                invalid_log.unlink()

            write_review_log(
                root,
                "T402",
                task_path=task_path.relative_to(root).as_posix(),
                run_manifest_path=run_manifest_path.relative_to(root).as_posix(),
                reviewer_role="Judge",
                outcome="approve",
            )

            with chdir(root):
                review_result = quality_gates.gate_judge_review_log_validity()
                bundle_result = quality_gates.gate_review_bundle_integrity()

            self.assertTrue(review_result.ok, review_result.details)
            self.assertTrue(bundle_result.ok, bundle_result.details)


if __name__ == "__main__":
    unittest.main()
