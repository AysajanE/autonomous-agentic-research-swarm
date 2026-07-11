from __future__ import annotations

from pathlib import Path
import json
import re
import unittest


class PaperSurfaceIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.pack = json.loads((cls.repo_root / "contracts/pack.json").read_text(encoding="utf-8"))
        cls.sections = json.loads((cls.repo_root / "contracts/manuscript_sections.yaml").read_text(encoding="utf-8"))

    @property
    def paper_build_artifacts(self) -> set[str]:
        paper = self.pack["paper"]
        return {f"{paper['artifact_basename']}.html", f"{paper['artifact_basename']}.pdf", paper["render_manifest"]}

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
        for filename in self.paper_build_artifacts:
            self.assertIn(f"{self.pack['paper']['build_dir']}{filename}", text)
        self.assertIn("pending_stage2", text)
        self.assertIn("present", text)

    def test_quarto_project_targets_release_candidate_build_surface(self) -> None:
        text = self.read("reports/paper/_quarto.yml")
        self.assertIn("output-dir: build", text)
        self.assertIn("render:", text)
        self.assertIn("- index.qmd", text)
        self.assertIn("bibliography: references.bib", text)
        self.assertIn("embed-resources: true", text)
        basename = self.pack["paper"]["artifact_basename"]
        self.assertIn(f"output-file: {basename}.html", text)
        self.assertIn(f"output-file: {basename}.pdf", text)

    def test_manuscript_contains_required_sections_and_release_links(self) -> None:
        text = self.read("reports/paper/index.qmd")
        for section_id in self.sections["canonical_section_ids"]:
            heading = self.sections["section_headings"][section_id]
            with self.subTest(heading=heading):
                self.assertIn(heading, text)

        expected_terms = [
            "../" + self.pack["analysis"]["outputs"][key].removeprefix("reports/")
            for key in ("ecosystem_figure", "regime_figure", "regime_table_markdown")
        ]
        for needle in expected_terms:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_bibliography_covers_repo_local_contract_and_provenance_sources(self) -> None:
        manuscript = self.read("reports/paper/index.qmd")
        bibliography = self.read("reports/paper/references.bib")
        cited = {
            key
            for key in re.findall(r"@([A-Za-z0-9_:.-]+)", manuscript)
            if not key.startswith("fig-")
        }
        available = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
        self.assertTrue(cited.issubset(available), sorted(cited - available))

    def test_build_readme_documents_operator_owned_release_surface(self) -> None:
        text = self.read("reports/paper/build/README.md")
        for filename in self.paper_build_artifacts:
            self.assertIn(filename, text)
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
                    {"README.md", "index.resolved.qmd", *self.paper_build_artifacts},
                )


if __name__ == "__main__":
    unittest.main()
