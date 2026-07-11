#!/usr/bin/env python3
"""Run M5 seeded-fabrication drills against disposable rehearsal fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import swarm_events  # noqa: E402
from golden.harness import GoldenRepo  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module, write_json, write_text  # noqa: E402
from test_m3a_modeling_battery import (  # noqa: E402
    _bridge_instance,
    _claim as _model_claim,
    _claims as _model_claims,
)


quality_gates = load_quality_gates_module()


@dataclass(frozen=True)
class Observation:
    blocked: bool
    blocking_gate: str
    evidence: str


@dataclass(frozen=True)
class DrillSpec:
    drill_id: str
    defect_class: str
    execute: Callable[[], Observation]


class MissedInjection(RuntimeError):
    """Raised after journaling when a real gate failed to catch the injection."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reasons(result: object) -> set[str]:
    details = getattr(result, "details", {})
    return {
        str(item["reason"])
        for item in details.get("failures", [])
        if isinstance(item, dict) and isinstance(item.get("reason"), str)
    }


def _fabricated_run_manifest() -> Observation:
    with tempfile.TemporaryDirectory(prefix="seeded-drill-manifest-") as tmp:
        repo = GoldenRepo.create(tmp)
        task = repo.write_task(
            "ready_for_review",
            "T901",
            state="ready_for_review",
            gates=["python scripts/noop_gate.py"],
            outputs=["README.md"],
        )
        repo.write_run_manifest(
            "T901",
            task_path=task.relative_to(repo.root).as_posix(),
            provenance_class="backfill",
            result_status="ok",
        )
        exit_code, summary = repo.judge("T901")
        review_path = repo.root / str(summary.get("review_log", ""))
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
        checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
        failures = checks.get("failures", [])
        caught = exit_code != 0 and summary.get("approved") is False and any(
            "provenance_requires_independent_reverification" in str(item)
            for item in failures
        )
        return Observation(caught, "swarm_run_manifest", json.dumps(review, sort_keys=True))


def _fabricated_citation() -> Observation:
    with tempfile.TemporaryDirectory(prefix="seeded-drill-citation-") as tmp:
        repo = GoldenRepo.create(tmp)
        write_text(repo.root, "reports/paper/index.qmd", "# Rehearsal paper\n\nPoisoned [@invented].\n")
        write_text(
            repo.root,
            "reports/paper/references.bib",
            "@article{invented, title={Hallucinated result}}\n",
        )
        write_text(repo.root, "data/citations/AS_OF", "2026-07-11\n")
        retrieval = {"citekey": "invented", "provider": "fixture"}
        write_json(
            repo.root,
            "data/citations/2026-07-11/invented.json",
            {
                "schema_version": "research_swarm.citation_snapshot.v1",
                "citekey": "invented",
                "title": "Hallucinated result",
                "source": "crossref",
                "retrieved_at_utc": "2026-07-11T00:00:00Z",
                "retrieval_sha256": hashlib.sha256(
                    json.dumps(retrieval, separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest(),
                "retrieval_payload": retrieval,
                "resolved": False,
                "retraction_status": "unknown",
                "url_resolves": False,
            },
        )
        with chdir(repo.root):
            result = quality_gates.gate_citation_integrity()
        reasons = _reasons(result)
        caught = not result.ok and bool(
            reasons
            & {
                "citation_unresolved",
                "citation_retraction_status_not_clean",
                "citation_url_unresolved",
            }
        )
        return Observation(caught, "citation_integrity", json.dumps(result.details, sort_keys=True))


def _fabricated_claim_number() -> Observation:
    with tempfile.TemporaryDirectory(prefix="seeded-drill-claim-") as tmp:
        repo = GoldenRepo.create(tmp)
        write_text(
            repo.root,
            "reports/paper/index.qmd",
            "# Results\n\nThe fabricated treatment effect was 73.91% [@fabricated_claim].\n",
        )
        with chdir(repo.root):
            result = quality_gates.gate_claim_evidence_ledger()
        caught = not result.ok and "unregistered_manuscript_numeric" in _reasons(result)
        return Observation(caught, "claim_evidence_ledger", json.dumps(result.details, sort_keys=True))


def _false_lemma() -> Observation:
    with tempfile.TemporaryDirectory(prefix="seeded-drill-lemma-") as tmp:
        repo = GoldenRepo.create(tmp)
        claim = _model_claim(repo.root, "theoretical")
        claim["falsification_spec"] = {
            "inequalities": ["x ** 2 >= 0.251"],
            "comparative_statics": [],
            "domain": {"x": [-0.5, 0.5]},
            "seed": 17,
            "sample_points": [{"x": -0.5}],
        }
        _model_claims(repo.root, [claim])
        with chdir(repo.root):
            result = quality_gates.gate_theoretical_falsification()
        caught = not result.ok and "inequality_violated" in _reasons(result)
        return Observation(caught, "theoretical_falsification", json.dumps(result.details, sort_keys=True))


def _tampered_instance() -> Observation:
    with tempfile.TemporaryDirectory(prefix="seeded-drill-instance-") as tmp:
        repo = GoldenRepo.create(tmp)
        _bridge_instance(repo.root, stale_source=True)
        repo.git("add", "-A")
        with chdir(repo.root):
            result = quality_gates.gate_instance_manifest_conformance()
        caught = not result.ok and "content_binding_sha256_mismatch" in _reasons(result)
        return Observation(caught, "instance_manifest_conformance", json.dumps(result.details, sort_keys=True))


ROTATION = (
    DrillSpec("M5B-D01", "fabricated_run_manifest", _fabricated_run_manifest),
    DrillSpec("M5B-D02", "fabricated_citation", _fabricated_citation),
    DrillSpec("M5B-D03", "fabricated_claim_number", _fabricated_claim_number),
    DrillSpec("M5B-D04", "false_lemma", _false_lemma),
    DrillSpec("M5B-D05", "tampered_instance", _tampered_instance),
)


def run_one(spec: DrillSpec, *, journal_root: Path, timestamp: str) -> dict[str, object]:
    observation = spec.execute()
    event = swarm_events.append_event(
        journal_root,
        {
            "event": "seeded_drill",
            "drill_id": spec.drill_id,
            "defect_class": spec.defect_class,
            "injected": True,
            "caught": observation.blocked,
            "blocking_gate": observation.blocking_gate,
            "timestamp": timestamp,
        },
        actor_session="seeded-drill-kernel",
    )
    if not observation.blocked:
        raise MissedInjection(
            f"seeded_drill_missed:{spec.drill_id}:{spec.defect_class}:"
            f"{observation.blocking_gate}:{observation.evidence}"
        )
    return event


def run_rotation(
    specs: tuple[DrillSpec, ...], *, journal_root: Path, timestamp: str
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    try:
        for spec in specs:
            events.append(run_one(spec, journal_root=journal_root, timestamp=timestamp))
    except MissedInjection:
        caught = sum(event.get("caught") is True for event in events)
        injected = len(events) + 1
        swarm_events.append_event(
            journal_root,
            {
                "event": "seeded_drill_summary",
                "timestamp": timestamp,
                "injected": injected,
                "caught": caught,
                "catch_rate": caught / injected,
                "status": "red",
            },
            actor_session="seeded-drill-kernel",
        )
        raise
    injected = len(events)
    caught = sum(event.get("caught") is True for event in events)
    summary = swarm_events.append_event(
        journal_root,
        {
            "event": "seeded_drill_summary",
            "timestamp": timestamp,
            "injected": injected,
            "caught": caught,
            "catch_rate": caught / injected if injected else 0.0,
            "status": "green" if caught == injected else "red",
        },
        actor_session="seeded-drill-kernel",
    )
    return {"events": events, "summary": summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="seeded_drill.py")
    parser.add_argument("--all", action="store_true", help="Run the complete M5 drill rotation")
    parser.add_argument("--drill", choices=[spec.defect_class for spec in ROTATION])
    parser.add_argument("--timestamp", default=_utc_now(), help="Timestamp recorded in drill events")
    parser.add_argument("--journal-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all == bool(args.drill):
        print("choose_exactly_one_of:--all|--drill", file=sys.stderr)
        return 2
    specs = ROTATION if args.all else tuple(spec for spec in ROTATION if spec.defect_class == args.drill)
    try:
        result = run_rotation(
            specs,
            journal_root=args.journal_root.resolve(),
            timestamp=args.timestamp,
        )
    except MissedInjection as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
