from __future__ import annotations

from datetime import date
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
import unittest

CANONICAL_PAPER_ARTIFACTS = [
    "reports/paper/build/l2_l1_rent_working_paper.html",
    "reports/paper/build/l2_l1_rent_working_paper.pdf",
    "reports/paper/build/render_manifest.json",
]


@lru_cache(maxsize=None)
def load_release_assembly_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "release_assembly.py"
    if not path.exists():
        raise RuntimeError(
            "scripts/release_assembly.py must exist from the locked Stage 1 continuation packet."
        )

    spec = importlib.util.spec_from_file_location(
        "stage2_paper_release_assembly_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load release assembly module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["stage2_paper_release_assembly_module"] = module
    spec.loader.exec_module(module)
    return module


class PaperReleaseIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.release_assembly = load_release_assembly_module()
        cls.preview = cls.release_assembly.assemble_release_manifest(
            cls.repo_root,
            date(2026, 3, 31),
            allow_gate_failures=True,
        )

    def test_release_preview_marks_materialized_paper_surface_present(self) -> None:
        paper = self.preview["artifacts"]["paper"]
        self.assertEqual(paper["expected_namespace"], "reports/paper/build/")
        expected_relpaths = [
            relpath
            for relpath in CANONICAL_PAPER_ARTIFACTS
            if (self.repo_root / relpath).is_file()
        ]
        if len(expected_relpaths) == len(CANONICAL_PAPER_ARTIFACTS):
            self.assertEqual(paper["status"], "present")
        else:
            self.assertEqual(paper["status"], "pending_stage2")

    def test_release_preview_records_only_canonical_paper_artifacts(self) -> None:
        paper = self.preview["artifacts"]["paper"]
        relpaths = [artifact["path"] for artifact in paper["artifacts"]]
        expected_relpaths = [
            relpath
            for relpath in CANONICAL_PAPER_ARTIFACTS
            if (self.repo_root / relpath).is_file()
        ]
        self.assertEqual(relpaths, expected_relpaths)
        self.assertEqual(self.preview["counts"]["paper_artifacts"], len(expected_relpaths))


if __name__ == "__main__":
    unittest.main()
