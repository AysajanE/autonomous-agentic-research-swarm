#!/usr/bin/env python3
"""BT2 Stage A rehearsal — STR frozen-reference regression + seeded drills.

Deterministic and OFFLINE.  It re-runs the STR pack's whole machinery and asserts
the KNOWN ANSWERS reproduce (this measures the MACHINERY, not new science),
executes the seeded-defect drill rotation (empirical fabrication classes + a
proof defect + a bridge-layer tampered-instance defect) which every case must
CATCH, exercises the release perimeter (the all-failing STR registry correctly
BLOCKS a release — recorded as the EXPECTED known outcome, not a failure), and
writes a committed, deterministic rehearsal report that the `bt2_bar` gate reads.

A drift in any frozen-reference value, a missed drill injection, or a release
perimeter that fails to block is a RED regression.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import quality_gates  # noqa: E402
import release_assembly  # noqa: E402
import render_paper  # noqa: E402
import reproduce_analysis  # noqa: E402
import seeded_drill  # noqa: E402
import swarm_events  # noqa: E402


REHEARSAL_REPORT_PATH = Path("reports/status/bt2a/rehearsal.json")
REHEARSAL_SCHEMA_VERSION = "research_swarm.bt2a_rehearsal.v1"


class RehearsalRed(RuntimeError):
    """Raised when a frozen-reference known answer fails to reproduce."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextlib.contextmanager
def _in_repo_root():
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_returns_zero(callable_, *args) -> str | None:
    """Invoke a `main()`-style entrypoint that returns 0 or raises SystemExit."""
    try:
        code = callable_(*args)
    except SystemExit as exc:  # reproduce_analysis / render_paper signal drift this way
        code = exc.code
    if code in (0, None):
        return None
    return str(code)


def assert_reproduction() -> list[str]:
    """The frozen-reference known answers must reproduce byte/content-identically."""
    coverage: list[str] = []
    drift = _run_returns_zero(reproduce_analysis.main)
    if drift is not None:
        raise RehearsalRed(f"reproduce_analysis_drift:{drift}")
    coverage.append("reproduce_analysis_byte_and_content_identity")
    drift = _run_returns_zero(render_paper.main, [])
    if drift is not None:
        # render_paper resolves the computed-paper {{value:…}} tokens (fails on a
        # missing key); byte/content drift itself is caught by reproduce_analysis.
        raise RehearsalRed(f"computed_paper_key_resolution_failed:{drift}")
    coverage.append("computed_paper_key_resolution")
    return coverage


def _registry_all_failing() -> bool:
    payload = json.loads((REPO_ROOT / "reports/paper/registry.json").read_text(encoding="utf-8"))
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    values = entries.values() if isinstance(entries, dict) else entries
    statuses = [item.get("status") for item in values if isinstance(item, dict)]
    return bool(statuses) and all(status == "failing" for status in statuses)


def exercise_release_perimeter() -> dict[str, object]:
    """The all-failing STR registry must BLOCK a release — the expected known
    outcome the rehearsal RECORDS (not a rehearsal failure).  A release that is
    NOT blocked, or a block NOT causally attributable to the failing registry
    (`paper_registry`), is a RED regression of the release perimeter."""
    payload = release_assembly.check_release_integrity(REPO_ROOT, enforce_required_gates=True)
    blocked = not bool(payload.get("ok", False))
    if not blocked:
        raise RehearsalRed("release_perimeter_did_not_block_all_failing_registry")
    registry_all_failing = _registry_all_failing()
    if not registry_all_failing:
        raise RehearsalRed("release_perimeter_registry_not_all_failing")
    # The block must be CAUSED by the failing registry, not merely by some
    # unrelated required gate — otherwise "the registry blocks release" is unproven.
    failures = payload.get("failures", [])
    blocked_by_registry = any("paper_registry" in str(item) for item in failures)
    if not blocked_by_registry:
        raise RehearsalRed("release_block_not_attributable_to_paper_registry")
    return {
        "registry_all_failing": True,
        "release_blocked_as_expected": True,
        "blocked_by_registry": True,
    }


def run_drills(timestamp: str) -> dict[str, object]:
    """Run the seeded-defect rotation.  It spans the empirical fabrication classes,
    a proof defect (false lemma), AND a bridge-layer defect (tampered instance:
    a stale `source_processed_manifests` hash) — the bridge-layer drill the plan
    requires when Stage B is hybrid.  A missed injection raises and fails RED."""
    result = seeded_drill.run_rotation(
        seeded_drill.ROTATION, journal_root=REPO_ROOT, timestamp=timestamp
    )
    drills = sorted(
        (
            {
                "drill_id": str(event["drill_id"]),
                "defect_class": str(event["defect_class"]),
                "blocking_gate": str(event["blocking_gate"]),
                "caught": bool(event["caught"]),
            }
            for event in result["events"]
        ),
        key=lambda item: item["drill_id"],
    )
    return {"drills": drills, "catch_rate": float(result["summary"]["catch_rate"])}


def hypothesis_ratio() -> float:
    """§6/§9.3 registered-vs-reported invariant: every REGISTERED hypothesis in the
    active analysis-plan lock must have a terminal REPORTED outcome.

    Measured from the PREREGISTRATION (the active lock's `hypotheses` + the
    terminal outcomes in `docs/prereg/outcomes.yaml`), NOT the descriptive/
    methodological claims ledger (whose mandatory `supporting_artifacts` would make
    the ratio tautological), and gated on the authoritative bidirectional
    `gate_prereg_conformance`.  The STR reference's prereg is draft (zero registered
    hypotheses, empty outcomes), so the invariant holds trivially → 1.0.  A real
    prereg-conformance violation fails the rehearsal RED.
    """
    # The kernel gate helpers resolve contracts/docs relative to CWD; run them
    # from REPO_ROOT so the rehearsal is robust to the caller's working directory
    # (the rest of the module already uses REPO_ROOT-absolute paths).
    with _in_repo_root():
        if not quality_gates.gate_prereg_conformance().ok:
            raise RehearsalRed("prereg_conformance_violated")
        lock_path = Path(quality_gates.PREREG_PHASE_FILES["2b"])
        lock, _ = quality_gates.load_prereg_lock(lock_path, expected_phase="2b")
        outcomes, _ = quality_gates._load_prereg_outcomes()
    active_lock = lock if lock is not None and lock.get("active") is True else None
    registered = {
        hypothesis["hypothesis_id"]
        for hypothesis in (active_lock.get("hypotheses", []) if active_lock is not None else [])
        if isinstance(hypothesis, dict) and isinstance(hypothesis.get("hypothesis_id"), str)
    }
    if not registered:
        return 1.0
    reported = sum(
        1
        for hid in registered
        if isinstance(outcomes.get(hid), dict)
        and outcomes[hid].get("outcome") in quality_gates.TERMINAL_HYPOTHESIS_OUTCOMES
    )
    return reported / len(registered)


def build_report(*, coverage: list[str], release_perimeter: dict, drills: dict, ratio: float) -> dict:
    """A DETERMINISTIC report (no timestamp) so `make bt2a-rehearsal` is idempotent
    and the committed reference is byte-stable — the `bt2_bar` gate reads it."""
    return {
        "schema_version": REHEARSAL_SCHEMA_VERSION,
        "kind": "frozen_reference_regression",
        "all_known_answers_reproduced": True,
        "machinery_coverage": sorted(coverage + ["release_perimeter_exercised", "seeded_defect_drill_rotation", "registered_vs_reported_hypothesis_ratio"]),
        "release_perimeter": release_perimeter,
        "drills": drills["drills"],
        "seeded_defect_catch_rate": drills["catch_rate"],
        "registered_vs_reported_hypothesis_ratio": ratio,
    }


def run_rehearsal(*, timestamp: str) -> dict:
    coverage = assert_reproduction()
    release_perimeter = exercise_release_perimeter()
    drills = run_drills(timestamp)
    ratio = hypothesis_ratio()
    report = build_report(
        coverage=coverage, release_perimeter=release_perimeter, drills=drills, ratio=ratio
    )
    report_path = REPO_ROOT / REHEARSAL_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The rehearsal's own KPI event goes to the ephemeral drill journal, never the
    # compliance journal the disclosure reads.
    swarm_events.append_event(
        REPO_ROOT,
        {
            "event": "bt2a_rehearsal",
            "all_known_answers_reproduced": report["all_known_answers_reproduced"],
            "seeded_defect_catch_rate": report["seeded_defect_catch_rate"],
            "registered_vs_reported_hypothesis_ratio": ratio,
            "release_blocked_as_expected": release_perimeter["release_blocked_as_expected"],
            "timestamp": timestamp,
        },
        actor_session="bt2a-rehearsal-kernel",
        journal_path=swarm_events.DRILL_JOURNAL_PATH,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bt2a_rehearsal.py")
    parser.add_argument("--timestamp", default=_utc_now(), help="Timestamp recorded in journal events")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_rehearsal(timestamp=args.timestamp)
    except (RehearsalRed, seeded_drill.MissedInjection) as exc:
        print(f"bt2a_rehearsal_red:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
