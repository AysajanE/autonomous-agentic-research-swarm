from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from golden.harness import GoldenRepo  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module, write_json, write_text  # noqa: E402


quality_gates = load_quality_gates_module()


def _theoretical_claim(root: Path, falsification_spec: dict) -> None:
    """Inline theoretical-claim construction — deliberately does NOT import any
    CI-visible test module (`test_m3a_modeling_battery` is discovered by
    `make test`), so this held-out case shares no code with the visible suite."""
    evidence = write_text(root, "reports/evidence.txt", "evidence\n")
    claim = {
        "claim_id": "C1",
        "statement": "Toy claim.",
        "type": "theoretical",
        "supporting_artifacts": [
            {"path": "reports/evidence.txt", "sha256": _sha256_of(evidence)}
        ],
        "verification_command": "make verify-claim",
        "uncertainty_artifact": None,
        "uncertainty_justification": "A theorem has no sampling uncertainty.",
        "assumption_scope": "The claim holds on the declared parameter domain.",
        "falsification_spec": falsification_spec,
    }
    write_json(
        root,
        "contracts/claims.yaml",
        {"schema_version": "research_swarm.claims.v1", "claims": [claim]},
    )


def _reasons(result: object) -> set[str]:
    details = getattr(result, "details", {})
    reasons: set[str] = set()
    for item in details.get("failures", []):
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            reasons.add(str(item["reason"]))
        elif isinstance(item, str):
            reasons.add(item.split(":", 1)[0])
    return reasons


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HeldOutM5Cases(unittest.TestCase):
    """Held-out Goodhart-control tier (plan §9.3).

    Distinctness contract — verified, not aspirational:
    * every case is constructed *independently* here (generic ``GoldenRepo`` /
      ``runtime_test_utils`` harness plus an inline attack), importing NONE of
      the CI-visible test modules — no ``golden.test_golden_*``,
      ``test_m3a_modeling_battery``, ``test_m3b_referee``, or
      ``test_m4c_replication`` (the theoretical-claim builder is inlined); and
    * every case asserts a failure mode (gate + reason) that is exercised by
      NEITHER the ``make test`` suite NOR the seeded-defect drill rotation.  A
      prompt/contract change over-fit to the visible set is therefore still
      caught here, because a regression these cases detect does not already fail
      the suite that produced the optimiser's signal.

    This tier is deliberately excluded from default ``make test`` discovery
    (filename ``cases.py``, not ``test*.py``) and MUST be refreshed adversarially
    at every milestone with fresh cases aimed at then-current mechanisms.  It is
    an offline deterministic control; live-referee held-out judgement is a
    tier-c/live-calibration concern (on-demand/BT2), not faked here with a
    self-supplied mock verdict.
    """

    def test_falsification_rejects_wrong_monotonicity_comparative_static(self) -> None:
        # Wrong-but-coherent theory: a lemma asserts the model output is
        # monotonically INCREASING in x, but the stated function decreases.
        # Detected via `comparative_static_violated` — a mechanism the drill
        # rotation (which uses `inequality_violated`) and the golden suite never
        # assert.
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _theoretical_claim(
                repo.root,
                {
                    "inequalities": [],
                    "comparative_statics": [
                        {"expression": "10.0 - x", "parameter": "x", "sign": "positive"}
                    ],
                    "domain": {"x": [0.0, 1.0]},
                    "seed": 5,
                    "sample_points": [{"x": 0.25}, {"x": 0.75}],
                },
            )
            with chdir(repo.root):
                result = quality_gates.gate_theoretical_falsification()
            self.assertFalse(result.ok)
            self.assertIn("comparative_static_violated", _reasons(result))

    def test_falsification_rejects_wrong_signed_explicit_derivative(self) -> None:
        # A second, mechanistically different theory fabrication: the claim
        # supplies an EXPLICIT derivative expression (not a finite difference)
        # whose sign contradicts the asserted comparative static.  Exercises the
        # explicit-derivative branch, which no golden/drill case reaches.
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            _theoretical_claim(
                repo.root,
                {
                    "inequalities": [],
                    "comparative_statics": [
                        {
                            "derivative": {"expression": "-2.0 * k"},
                            "parameter": "k",
                            "sign": "nonnegative",
                        }
                    ],
                    "domain": {"k": [1.0, 3.0]},
                    "seed": 9,
                    "sample_points": [{"k": 2.0}],
                },
            )
            with chdir(repo.root):
                result = quality_gates.gate_theoretical_falsification()
            self.assertFalse(result.ok)
            self.assertIn("comparative_static_violated", _reasons(result))

    def test_citation_snapshot_from_after_the_freeze_date(self) -> None:
        # Provenance fabrication: a citation snapshot dated AFTER the AS_OF freeze
        # (future-dated evidence).  Detected via
        # `citation_snapshot_directory_after_as_of` — a reason exercised by
        # neither the golden suite nor the drill rotation.
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_text(repo.root, "reports/paper/index.qmd", "# Paper\n\nSee [@future].\n")
            write_text(repo.root, "reports/paper/references.bib", "@article{future, title={T}}\n")
            write_text(repo.root, "data/citations/AS_OF", "2026-07-01\n")
            # snapshot directory dated AFTER the freeze -> impossible evidence
            write_json(
                repo.root,
                "data/citations/2026-08-15/future.json",
                {
                    "schema_version": "research_swarm.citation_snapshot.v1",
                    "citekey": "future",
                    "title": "T",
                    "source": "crossref",
                    "retrieved_at_utc": "2026-08-15T00:00:00Z",
                    "retrieval_sha256": "0" * 64,
                    "retrieval_payload": {"citekey": "future"},
                    "resolved": True,
                    "retraction_status": "clean",
                    "url_resolves": True,
                },
            )
            with chdir(repo.root):
                result = quality_gates.gate_citation_integrity()
            self.assertFalse(result.ok)
            self.assertIn("citation_snapshot_directory_after_as_of", _reasons(result))

    def test_instance_manifest_rejects_schema_violating_bridge(self) -> None:
        # A bridge instance that is internally plausible but violates the manifest
        # schema (missing the required `instance_id`).  Detected via
        # `instance_manifest_schema_violation`, a reason no golden/drill case
        # asserts.
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            output = write_text(repo.root, "reports/models/bridge_instance.txt", "instance output\n")
            write_json(
                repo.root,
                "contracts/instances/bridge.json",
                {
                    "schema_version": "research_swarm.instance_manifest.v1",
                    # instance_id deliberately omitted -> schema violation
                    "source_processed_manifests": [],
                    "generator_command": "python scripts/generate_bridge.py",
                    "generated_at_utc": "2026-07-10T11:00:00Z",
                    "outputs": [{"path": "reports/models/bridge_instance.txt", "sha256": _sha256_of(output)}],
                },
            )
            repo.git("add", "-A")
            with chdir(repo.root):
                result = quality_gates.gate_instance_manifest_conformance()
            self.assertFalse(result.ok)
            self.assertIn("instance_manifest_schema_violation", _reasons(result))

    def test_citation_snapshot_key_forgery(self) -> None:
        # Provenance fabrication: a resolved, non-retracted citation snapshot
        # whose recorded citekey attests a DIFFERENT work than the one cited.
        # Detected via `citation_snapshot_key_mismatch` — the drill fires
        # unresolved/retraction/url reasons instead, and the golden suite asserts
        # neither.
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            write_text(repo.root, "reports/paper/index.qmd", "# Paper\n\nAs shown [@smith2025].\n")
            write_text(
                repo.root,
                "reports/paper/references.bib",
                "@article{smith2025, title={A Real Result}, doi={10.1/real}}\n",
            )
            write_text(repo.root, "data/citations/AS_OF", "2026-07-11\n")
            retrieval = {"citekey": "smith2025", "provider": "fixture"}
            write_json(
                repo.root,
                "data/citations/2026-07-11/smith2025.json",
                {
                    "schema_version": "research_swarm.citation_snapshot.v1",
                    # attests a different key than the filename/cite subject
                    "citekey": "jones2019",
                    "title": "A Real Result",
                    "doi": "10.1/real",
                    "source": "crossref",
                    "retrieved_at_utc": "2026-07-11T00:00:00Z",
                    "retrieval_sha256": hashlib.sha256(
                        json.dumps(retrieval, separators=(",", ":"), sort_keys=True).encode()
                    ).hexdigest(),
                    "retrieval_payload": retrieval,
                    "resolved": True,
                    "retraction_status": "clean",
                    "url_resolves": True,
                },
            )
            with chdir(repo.root):
                result = quality_gates.gate_citation_integrity()
            self.assertFalse(result.ok)
            self.assertIn("citation_snapshot_key_mismatch", _reasons(result))


if __name__ == "__main__":
    unittest.main()
