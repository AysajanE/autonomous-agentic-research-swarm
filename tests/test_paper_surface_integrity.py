from __future__ import annotations

from pathlib import Path
import unittest


EXPECTED_SECTIONS = [
    "## Abstract",
    "## Research Question And Metric Lock",
    "## Data And Provenance Contracts",
    "## Agentic Runtime And Review Semantics",
    "## Release-Ready Paper Surface",
    "## Scope Limits For This Release Candidate",
    "## From Repository Workflow To Finished Paper",
    "## Conclusion",
]

EXPECTED_BIB_KEYS = [
    "@misc{protocol_lock,",
    "@misc{data_dictionary_contract,",
    "@misc{decision_log_contract,",
    "@misc{registry_rollup_v1,",
    "@misc{raw_manifest_readme,",
    "@misc{processed_manifest_readme,",
    "@misc{validation_readme,",
    "@misc{validation_manifests_readme,",
    "@misc{figures_readme,",
    "@misc{tables_readme,",
]


class PaperSurfaceIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def read(self, relpath: str) -> str:
        return (self.repo_root / relpath).read_text(encoding="utf-8")

    def test_required_paper_files_exist(self) -> None:
        required = [
            "reports/paper/README.md",
            "reports/paper/_quarto.yml",
            "reports/paper/index.qmd",
            "reports/paper/references.bib",
            "reports/paper/build/README.md",
            "reports/paper/build/index.html",
        ]
        for relpath in required:
            with self.subTest(relpath=relpath):
                self.assertTrue((self.repo_root / relpath).is_file(), relpath)

    def test_paper_readme_documents_render_and_release_workflow(self) -> None:
        text = self.read("reports/paper/README.md")
        self.assertIn("quarto render reports/paper/", text)
        self.assertIn("reports/paper/build/index.html", text)
        self.assertIn("pending_stage2", text)
        self.assertIn("present", text)

    def test_quarto_project_targets_self_contained_build_surface(self) -> None:
        text = self.read("reports/paper/_quarto.yml")
        self.assertIn("output-dir: build", text)
        self.assertIn("render:", text)
        self.assertIn("- index.qmd", text)
        self.assertIn("bibliography: references.bib", text)
        self.assertIn("embed-resources: true", text)

    def test_manuscript_contains_locked_sections_and_core_terms(self) -> None:
        text = self.read("reports/paper/index.qmd")
        for heading in EXPECTED_SECTIONS:
            with self.subTest(heading=heading):
                self.assertIn(heading, text)

        expected_terms = [
            "STR_t = (Σ_i RentPaid_{i,t}) / (Σ_i L2Fees_{i,t})",
            "Planner",
            "Worker",
            "Judge",
            "Operator",
            "`backlog`",
            "`active`",
            "`integration_ready`",
            "`ready_for_review`",
            "`blocked`",
            "`done`",
            "`reports/paper/build/index.html`",
            "`reports/catalog.yaml`",
            "`paper.status = pending_stage2`",
            "`paper.status = present`",
            "methods-and-release paper rather than a results-forward paper",
        ]
        for needle in expected_terms:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_bibliography_covers_repo_local_contract_and_provenance_sources(self) -> None:
        text = self.read("reports/paper/references.bib")
        for key in EXPECTED_BIB_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, text)

    def test_build_namespace_is_clean_and_contains_single_html_artifact(self) -> None:
        build_root = self.repo_root / "reports" / "paper" / "build"
        files = sorted(
            path.relative_to(build_root).as_posix()
            for path in build_root.rglob("*")
            if path.is_file()
        )
        self.assertEqual(files, ["README.md", "index.html"])

    def test_rendered_html_contains_expected_release_candidate_content(self) -> None:
        text = self.read("reports/paper/build/index.html")
        normalized_text = " ".join(text.split())
        expected_terms = [
            "Measuring Settlement Take Rate for Ethereum Rollups",
            "Protocol, provenance, and a release-complete working paper surface",
            "STR_t = (Σ_i RentPaid_{i,t}) / (Σ_i L2Fees_{i,t})",
            "Planner",
            "Worker",
            "Judge",
            "Operator",
            "reports/paper/build/index.html",
            "reports/catalog.yaml",
            "pending_stage2",
            "present",
            "methods-and-release paper rather than a results-forward paper",
        ]
        for needle in expected_terms:
            with self.subTest(needle=needle):
                self.assertIn(needle, normalized_text)


if __name__ == "__main__":
    unittest.main()
