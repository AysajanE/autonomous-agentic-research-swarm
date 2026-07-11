#!/usr/bin/env python3
"""Generate the AI-use disclosure from durable execution evidence.

The disclosure is deliberately a projection of run manifests, review manifests,
provenance annotations, and the kernel event journal.  No caller-supplied prose or
model recollection is accepted as evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("reports/paper/disclosure.md")
SCHEMA_VERSION = "research_swarm.generated_disclosure.v1"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_record(repo: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(repo).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _iter_json_objects(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(paths):
        payload = _load_json(path)
        if payload is not None:
            yield path, payload


def collect_disclosure_evidence(repo: Path) -> dict[str, object]:
    repo = repo.resolve()
    sources: list[dict[str, object]] = []
    model_counts: Counter[tuple[str, str, str, str]] = Counter()
    gate_counts: Counter[tuple[str, int]] = Counter()
    review_counts: Counter[tuple[str, str]] = Counter()
    human_edit_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()

    run_glob = (repo / "reports/status/swarm_runs").glob("*.json")
    for path, payload in _iter_json_objects(run_glob):
        sources.append(_source_record(repo, path))
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        executor = payload.get("executor") if isinstance(payload.get("executor"), dict) else {}
        model_counts[
            (
                str(task.get("role") or "unlogged"),
                str(executor.get("tool") or executor.get("backend") or "unlogged"),
                str(executor.get("model") or "unlogged"),
                str(executor.get("model_version") or executor.get("cli_version") or "unlogged"),
            )
        ] += 1
        gates = payload.get("gates")
        if isinstance(gates, list):
            for gate in gates:
                if not isinstance(gate, dict):
                    continue
                command = gate.get("command")
                returncode = gate.get("returncode")
                if isinstance(command, str) and isinstance(returncode, int):
                    gate_counts[(command, returncode)] += 1

    review_glob = (repo / "reports/status/reviews").glob("*.json")
    for path, payload in _iter_json_objects(review_glob):
        sources.append(_source_record(repo, path))
        reviewer = payload.get("reviewer") if isinstance(payload.get("reviewer"), dict) else {}
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        review_counts[(str(reviewer.get("role") or "unlogged"), str(decision.get("outcome") or "unlogged"))] += 1

    annotation_glob = (repo / "reports/status/swarm_runs/annotations").glob("*.json")
    for path, payload in _iter_json_objects(annotation_glob):
        sources.append(_source_record(repo, path))
        classification = payload.get("classification") or payload.get("provenance_class") or payload.get("type")
        if isinstance(classification, str):
            human_edit_counts[classification] += 1

    events_path = repo / "reports/status/events/events.jsonl"
    if events_path.is_file():
        sources.append(_source_record(repo, events_path))
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event_counts["invalid_event_record"] += 1
                continue
            if isinstance(event, dict):
                event_name = str(event.get("event") or event.get("type") or "untyped_event")
                # Seeded-defect drills write to their own ledger, never here; but
                # if a rehearsal event ever leaks into the compliance journal it
                # must not be counted as released-work provenance.
                if event_name.startswith("seeded_drill") or str(event.get("actor_session") or "") == "seeded-drill-kernel":
                    continue
                event_counts[event_name] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generation_policy": "execution_logs_only",
        "models_by_role": [
            {
                "role": key[0],
                "tool": key[1],
                "model": key[2],
                "version": key[3],
                "run_count": count,
            }
            for key, count in sorted(model_counts.items())
        ],
        "gate_outcomes": [
            {"command": key[0], "returncode": key[1], "observation_count": count}
            for key, count in sorted(gate_counts.items())
        ],
        "reviews": [
            {"reviewer_role": key[0], "outcome": key[1], "review_count": count}
            for key, count in sorted(review_counts.items())
        ],
        "human_edit_history": [
            {"classification": key, "record_count": count}
            for key, count in sorted(human_edit_counts.items())
        ],
        "swarm_events": [
            {"event": key, "event_count": count}
            for key, count in sorted(event_counts.items())
        ],
        "event_journal_status": "present" if events_path.is_file() else "absent_not_fabricated",
        "sources": sorted(sources, key=lambda item: str(item["path"])),
    }


def render_disclosure(evidence: dict[str, object]) -> str:
    lines = [
        "# Generated AI-use disclosure",
        "",
        "This compliance record is generated deterministically from durable execution logs. It is not a recollection supplied by an author or model.",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generation policy: `{evidence['generation_policy']}`",
        f"- Event journal: `{evidence['event_journal_status']}`",
        "",
        "## Models and tools by execution role",
        "",
        "| Role | Tool/backend | Model | Version | Runs |",
        "|---|---|---|---|---:|",
    ]
    models = evidence.get("models_by_role")
    if isinstance(models, list) and models:
        for item in models:
            assert isinstance(item, dict)
            lines.append(
                f"| {item['role']} | {item['tool']} | {item['model']} | {item['version']} | {item['run_count']} |"
            )
    else:
        lines.append("| unlogged | unlogged | unlogged | unlogged | 0 |")

    lines.extend(["", "## Logged gate outcomes", ""])
    gates = evidence.get("gate_outcomes")
    if isinstance(gates, list) and gates:
        for item in gates:
            assert isinstance(item, dict)
            lines.append(
                f"- `{item['command']}`: return code `{item['returncode']}` observed {item['observation_count']} time(s)."
            )
    else:
        lines.append("- No gate outcome records were present; none were inferred.")

    lines.extend(["", "## Review and human-edit history", ""])
    reviews = evidence.get("reviews")
    if isinstance(reviews, list) and reviews:
        for item in reviews:
            assert isinstance(item, dict)
            lines.append(
                f"- Reviewer role `{item['reviewer_role']}` recorded outcome `{item['outcome']}` {item['review_count']} time(s)."
            )
    edits = evidence.get("human_edit_history")
    if isinstance(edits, list) and edits:
        for item in edits:
            assert isinstance(item, dict)
            lines.append(
                f"- Provenance classification `{item['classification']}` appears in {item['record_count']} annotation(s)."
            )
    else:
        lines.append("- No separately classified human-edit annotation was present; none was inferred.")

    lines.extend(
        [
            "",
            "## Disclosure statement",
            "",
            "AI systems executed planning, implementation, analysis, review, and/or operational roles only as recorded above. The systems are disclosed assistants and are not authors. Every model/version value shown is taken from a durable log; `unlogged` means the version was not recorded. Scientific and release claims remain subject to repository gates and human-author consent requirements.",
            "",
            "## Evidence inventory",
            "",
        ]
    )
    for source in evidence.get("sources", []):
        assert isinstance(source, dict)
        lines.append(f"- `{source['path']}` — sha256 `{source['sha256']}`")
    return "\n".join(lines) + "\n"


def generate_disclosure(repo: Path, output: Path | None = None) -> str:
    evidence = collect_disclosure_evidence(repo)
    text = render_disclosure(evidence)
    if output is not None:
        destination = output if output.is_absolute() else repo / output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_disclosure.py")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    expected = generate_disclosure(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            print(f"disclosure_not_regenerable:{output}")
            return 1
        print(f"disclosure_regenerable:{output.relative_to(repo)}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"generated_disclosure:{output.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
