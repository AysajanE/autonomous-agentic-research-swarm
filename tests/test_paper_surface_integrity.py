from __future__ import annotations

from pathlib import Path
import unittest


EXPECTED_SECTIONS = [
    "## Abstract",
    "## Research Question",
    "## Data And Protocol",
    "## Validation",
    "## Results",
    "## Provenance And Limitations",
]

EXPECTED_BIB_KEYS = [
    "@misc{protocol_lock,",
    "@misc{project_contract,",
    "@misc{rollup_panel_validation,",
    "@misc{l1_rent_decomposition_validation,",
    "@misc{cross_source_reconciliation,",
    "@misc{str_ecosystem_timeseries,",
    "@misc{str_post_dencun_regimes,",
    "@misc{str_regime_summary,",
    "@misc{release_output_caveats,",
]

CANONICAL_PAPER_BUILD_ARTIFACTS = {
    "l2_l1_rent_working_paper.html",
    "l2_l1_rent_working_paper.pdf",
    "render_manifest.json",
}


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
            "reports/paper/paper_values.json",
            "reports/paper/references.bib",
            "reports/paper/build/README.md",
        ]
        for relpath in required:
            with self.subTest(relpath=relpath):
                self.assertTrue((self.repo_root / relpath).is_file(), relpath)

    def test_paper_readme_documents_render_and_release_workflow(self) -> None:
        text = self.read("reports/paper/README.md")
        self.assertIn("quarto render reports/paper/", text)
        self.assertIn("reports/paper/build/l2_l1_rent_working_paper.html", text)
        self.assertIn("reports/paper/build/l2_l1_rent_working_paper.pdf", text)
        self.assertIn("reports/paper/build/render_manifest.json", text)
        self.assertIn("pending_stage2", text)
        self.assertIn("present", text)

    def test_quarto_project_targets_release_candidate_build_surface(self) -> None:
        text = self.read("reports/paper/_quarto.yml")
        self.assertIn("output-dir: build", text)
        self.assertIn("render:", text)
        self.assertIn("- index.qmd", text)
        self.assertIn("bibliography: references.bib", text)
        self.assertIn("embed-resources: true", text)
        self.assertIn("output-file: l2_l1_rent_working_paper.html", text)
        self.assertIn("output-file: l2_l1_rent_working_paper.pdf", text)

    def test_manuscript_contains_required_sections_and_release_links(self) -> None:
        text = self.read("reports/paper/index.qmd")
        for heading in EXPECTED_SECTIONS:
            with self.subTest(heading=heading):
                self.assertIn(heading, text)

        expected_terms = [
            "STR_t = (sum_i RentPaid_{i,t}) / (sum_i L2Fees_{i,t})",
            "../figures/str_ecosystem_timeseries.svg",
            "../figures/str_post_dencun_regimes.svg",
            "../tables/str_regime_summary.md",
            "`2026-04-09`",
            "`2024-03-13`",
            "Operator-owned T080 surfaces",
        ]
        for needle in expected_terms:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_bibliography_covers_repo_local_contract_and_provenance_sources(self) -> None:
        text = self.read("reports/paper/references.bib")
        for key in EXPECTED_BIB_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, text)

    def test_build_readme_documents_operator_owned_release_surface(self) -> None:
        text = self.read("reports/paper/build/README.md")
        self.assertIn("l2_l1_rent_working_paper.html", text)
        self.assertIn("l2_l1_rent_working_paper.pdf", text)
        self.assertIn("render_manifest.json", text)
        self.assertIn("pending_stage2", text)
        self.assertIn("index.html", text)

    def test_build_namespace_contains_only_canonical_release_artifacts(self) -> None:
        build_root = self.repo_root / "reports" / "paper" / "build"
        files = sorted(
            path.relative_to(build_root).as_posix()
            for path in build_root.rglob("*")
            if path.is_file()
        )
        self.assertNotIn("index.html", files)
        for filename in files:
            with self.subTest(filename=filename):
                self.assertIn(
                    filename,
                    {"README.md", "index.resolved.qmd", *CANONICAL_PAPER_BUILD_ARTIFACTS},
                )


if __name__ == "__main__":
    unittest.main()
