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
        cls.expected_relpaths = [
            relpath for relpath in CANONICAL_PAPER_ARTIFACTS if (cls.repo_root / relpath).is_file()
        ]

    def test_materialized_paper_without_panel_blocks_release_preview(self) -> None:
        if len(self.expected_relpaths) != len(CANONICAL_PAPER_ARTIFACTS):
            self.skipTest("canonical paper is not materialized")
        with self.assertRaisesRegex(SystemExit, "referee_release_evidence"):
            self.release_assembly.assemble_release_manifest(
                self.repo_root,
                date(2026, 3, 31),
                allow_gate_failures=True,
            )

    def test_pending_paper_preview_still_skips_referee_release_gate(self) -> None:
        if len(self.expected_relpaths) == len(CANONICAL_PAPER_ARTIFACTS):
            self.skipTest("canonical paper is materialized")
        preview = self.release_assembly.assemble_release_manifest(
            self.repo_root,
            date(2026, 3, 31),
            allow_gate_failures=True,
        )
        self.assertEqual(preview["artifacts"]["paper"]["status"], "pending_stage2")


if __name__ == "__main__":
    unittest.main()
