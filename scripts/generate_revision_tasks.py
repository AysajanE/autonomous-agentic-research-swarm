#!/usr/bin/env python3
"""Generate lint-clean, artifact-scoped repair tasks from referee findings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Mapping


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from swarm_taskfile import lint_task_files
from swarm_taskfile import parse_task_frontmatter
from pack_config import load_pack_config, pack_value


BLOCKING_VERDICTS = {"not_supported", "cannot_verify"}
REPORT_SCHEMA_VERSION = "research_swarm.referee_report.v1"
PLAN_APPROVAL_PENDING_PATH = Path(".swarm/plan_approval_pending.json")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{path}")
    return payload


def _task_paths(repo: Path) -> list[Path]:
    return sorted((repo / ".orchestrator").glob("*/T*.md"))


def _task_by_id(repo: Path, task_id: str) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in _task_paths(repo):
        frontmatter = parse_task_frontmatter(path.read_text(encoding="utf-8"))
        if isinstance(frontmatter, dict) and frontmatter.get("task_id") == task_id:
            matches.append((path, frontmatter))
    if len(matches) != 1:
        raise ValueError(f"revision_source_task_count:{task_id}:{len(matches)}")
    return matches[0]


def _quote(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _emit_list(key: str, values: list[str], *, quoted: bool = True) -> list[str]:
    if not values:
        return [f"{key}: []"]
    return [f"{key}:", *[f"  - {_quote(value) if quoted else value}" for value in values]]


def _normalize_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _artifact_paths(report: Mapping[str, object], source: Mapping[str, object], finding: Mapping[str, object]) -> list[str]:
    outputs = [
        _normalize_path(value)
        for value in source.get("outputs", [])
        if isinstance(value, str) and value.strip()
    ] if isinstance(source.get("outputs"), list) else []
    reviewed = [
        _normalize_path(value)
        for value in report.get("reviewed_artifacts", [])
        if isinstance(value, str) and value.strip()
    ] if isinstance(report.get("reviewed_artifacts"), list) else []
    candidates = list(dict.fromkeys(reviewed + outputs))
    pointer = finding.get("evidence_pointer")
    if isinstance(pointer, str):
        pointer_path = pointer.split("#", 1)[0]
        match = re.match(r"^(.*?):\d+(?::\d+)?$", pointer_path)
        if match is not None:
            pointer_path = match.group(1)
        pointer_path = _normalize_path(pointer_path)
        exact = [path for path in candidates if pointer_path == path or pointer_path.startswith(path + "/")]
        if exact:
            return exact[:1]
    return candidates


def _finding_identifier(finding: Mapping[str, object]) -> str:
    for key in ("success_criterion_id", "check_id"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _finding_severities(
    repo: Path,
    report: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, str]:
    severities: dict[str, str] = {}
    criteria = source.get("success_criteria")
    if isinstance(criteria, list):
        for criterion in criteria:
            if isinstance(criterion, dict) and isinstance(criterion.get("id"), str):
                severities[criterion["id"]] = "major"
    rubric_files = [
        value
        for value in report.get("rubric_files", [])
        if isinstance(value, str)
    ] if isinstance(report.get("rubric_files"), list) else []
    if not rubric_files:
        task_kind = source.get("task_kind")
        if isinstance(task_kind, str):
            rubric_name = "proof" if task_kind == "proof" else task_kind
            rubric_files.append(f"contracts/rubrics/{rubric_name}.yaml")
            outputs = source.get("outputs") if isinstance(source.get("outputs"), list) else []
            if task_kind == "writing" or any(
                isinstance(output, str) and output.startswith("reports/paper/")
                for output in outputs
            ):
                rubric_files.append("contracts/rubrics/manuscript.yaml")
    for rubric_rel in rubric_files:
        rubric_path = (repo / rubric_rel).resolve()
        try:
            rubric_path.relative_to(repo.resolve())
            rubric = _read_json(rubric_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        checks = rubric.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if (
                isinstance(check, dict)
                and isinstance(check.get("id"), str)
                and check.get("severity") in {"major", "minor"}
            ):
                severities[check["id"]] = str(check["severity"])
    assertions = report.get("assertion_prefilter_floor")
    if isinstance(assertions, list):
        for assertion in assertions:
            if isinstance(assertion, dict) and isinstance(assertion.get("check_id"), str):
                severities[assertion["check_id"]] = "major"
    return severities


def _fingerprint(report_path: Path, report: Mapping[str, object], finding: Mapping[str, object]) -> str:
    material = {
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "task_id": report.get("task_id"),
        "finding_id": _finding_identifier(finding),
        "verdict": finding.get("verdict"),
        "note": finding.get("note"),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _existing_fingerprints(repo: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in _task_paths(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r'^repair_fingerprint:\s*["\']?([0-9a-f]{64})["\']?\s*$', text, flags=re.MULTILINE)
        if match is not None:
            out[match.group(1)] = path
    return out


def _next_task_number(repo: Path, reserved: set[int]) -> int:
    used = set(reserved)
    for path in _task_paths(repo):
        match = re.match(r"T(\d{3})", path.name)
        if match is not None:
            used.add(int(match.group(1)))
    for number in range(max(used, default=0) + 1, 1000):
        if number not in used:
            return number
    raise ValueError("revision_task_id_space_exhausted")


def _project_mode(repo: Path) -> str:
    try:
        text = (repo / "contracts" / "project.yaml").read_text(encoding="utf-8")
    except OSError:
        return "empirical"
    match = re.search(r"^mode:\s*['\"]?([a-z_]+)", text, flags=re.MULTILINE)
    return match.group(1) if match is not None else "empirical"


def _source_required_locks(repo: Path, source: Mapping[str, object]) -> list[str]:
    explicit = source.get("required_prereg_locks")
    if isinstance(explicit, list):
        return sorted({str(item) for item in explicit if item in {"2a", "2b", "lock_a", "lock_b"}})
    mode = _project_mode(repo)
    kind = source.get("task_kind")
    outputs = source.get("outputs") if isinstance(source.get("outputs"), list) else []
    required: set[str] = set()
    if mode in {"empirical", "hybrid"} and kind == "etl" and any(
        isinstance(output, str) and output.startswith("data/processed/") for output in outputs
    ):
        required.add("2a")
    if mode in {"empirical", "hybrid"} and kind == "analysis":
        required.add("2b")
    if mode == "modeling" and kind in {"model", "analysis"}:
        required.add("lock_a")
    if mode == "hybrid" and kind == "bridge":
        required.add("lock_a")
    if mode == "hybrid" and kind == "model":
        required.add("lock_b")
    return sorted(required, key=("2a", "2b", "lock_a", "lock_b").index)


def _plan_content_digest(repo: Path) -> str:
    parts: list[str] = []
    backlog = repo / ".orchestrator" / "backlog"
    for path in sorted(backlog.glob("*.md")) if backlog.is_dir() else []:
        parts.extend((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    workstreams = repo / ".orchestrator" / "workstreams.md"
    if workstreams.is_file():
        parts.append(hashlib.sha256(workstreams.read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _write_plan_approval_pending(repo: Path, source_task_id: str) -> Path:
    path = repo / PLAN_APPROVAL_PENDING_PATH
    payload = {
        "schema_version": "research_swarm.plan_approval_pending.v1",
        "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "planner_backend": "deterministic_referee_revision_generator",
        "proposal_count": len(list((repo / ".orchestrator" / "backlog").glob("T*_repair_*.md"))),
        "base_sha": None,
        "plan_digest": _plan_content_digest(repo),
        "source_task_id": source_task_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _render_revision_task(
    *,
    task_id: str,
    source_task_id: str,
    source: Mapping[str, object],
    report_relpath: str,
    report_sha256: str,
    finding: Mapping[str, object],
    fingerprint: str,
    artifacts: list[str],
    required_locks: list[str],
    operator_workstream: str,
) -> str:
    finding_id = _finding_identifier(finding)
    verdict = str(finding.get("verdict", "cannot_verify"))
    note = str(finding.get("note", "Referee supplied no note.")).strip()
    workstream = source.get("workstream") if isinstance(source.get("workstream"), str) else operator_workstream
    source_gates = [
        item for item in source.get("gates", []) if isinstance(item, str) and item.strip()
    ] if isinstance(source.get("gates"), list) else []
    gates = source_gates or ["python scripts/quality_gates.py"]
    disallowed_paths = [
        item
        for item in source.get("disallowed_paths", [])
        if isinstance(item, str) and item.strip()
    ] if isinstance(source.get("disallowed_paths"), list) else []
    if not disallowed_paths:
        disallowed_paths = ["contracts/"]
    title = f"Repair {source_task_id} referee finding {finding_id}"
    criterion = f"Resolve {finding_id} ({verdict}) with evidence and obtain a supported referee verdict."
    lines = [
        "---",
        'task_schema: "research_swarm.task.v2"',
        f'task_id: "{task_id}"',
        f"title: {_quote(title)}",
        f"workstream: {_quote(workstream)}",
        'task_kind: "repair"',
        'complexity_tier: "S"',
        "success_criteria:",
        '  - id: "SC1"',
        f"    statement: {_quote(criterion)}",
        f"    verification: {_quote(gates[0])}",
        "budgets: {max_wall_clock: 2h, max_tokens: 250000, max_cost_usd: 25}",
        'checkpoint_contract: "none"',
        "recon_required: false",
        "inputs:",
        f"  - path: {_quote(report_relpath)}",
        f"    sha256: {report_sha256}",
        "    comparison_basis: false",
        "allow_network: false",
        'role: "Worker"',
        'priority: "high"',
        "dependencies: []",
        "integration_ready_dependencies: []",
        *_emit_list("allowed_paths", artifacts),
        *_emit_list("disallowed_paths", disallowed_paths),
        *_emit_list("outputs", artifacts),
        *_emit_list("gates", gates, quoted=False),
        *_emit_list(
            "stop_conditions",
            ["Block with @human if resolving the finding would change protocol or contracts."],
        ),
        f'repair_source_task: "{source_task_id}"',
        f'repair_source_task_kind: {_quote(source.get("task_kind", ""))}',
        f'repair_source_complexity_tier: {_quote(source.get("complexity_tier", "S"))}',
        *_emit_list("required_prereg_locks", required_locks),
        f'repair_finding_id: {_quote(finding_id)}',
        f'repair_fingerprint: "{fingerprint}"',
        "---",
        "",
        f"# Task {task_id} — {title}",
        "",
        "## Context",
        "",
        f"A cross-family referee returned `{verdict}` for `{finding_id}` on `{source_task_id}`.",
        f"Finding: {note}",
        f"Source report: `{report_relpath}`.",
        "",
        "## Inputs",
        "",
        f"- Referee report `{report_relpath}` (sha256 `{report_sha256}`).",
        "",
        "## Outputs",
        "",
        *[f"- Repaired artifact `{path}`." for path in artifacts],
        "",
        "## Success Criteria",
        "",
        f"- [ ] {criterion}",
        "",
        "## Review Bundle Requirements",
        "",
        "- [ ] Run manifest, changed-artifact evidence, and a fresh cross-family referee report.",
        "",
        "## Validation / Commands",
        "",
        *[f"- `{gate}`" for gate in gates],
        "",
        "## Status",
        "",
        "- State: backlog",
        f"- Last updated: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}",
        "",
        "## Notes / Decisions",
        "",
        f"- Generated deterministically from `{report_relpath}` finding `{finding_id}`; no inline repair was applied.",
        "",
    ]
    return "\n".join(lines)


def _v1_exemptions(repo: Path) -> dict[str, dict[str, object]]:
    path = repo / "contracts" / "historical_exemptions.json"
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        item["path"]: item
        for item in payload.get("tasks", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    } if isinstance(payload.get("tasks"), list) else {}


def generate_revision_tasks(
    *,
    repo: Path,
    report_path: Path,
    dry_run: bool = False,
) -> list[Path]:
    repo = repo.resolve()
    pack = load_pack_config(repo)
    workflow = pack_value(pack, "workflow", dict)
    operator_workstream = pack_value(pack, "workflow.operator_workstream")
    network_workstreams = tuple(
        item for item in workflow.get("network_workstreams", []) if isinstance(item, str)
    )
    report_path = report_path.resolve()
    report = _read_json(report_path)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"referee_report_schema_invalid:{report.get('schema_version')}")
    source_task_id = report.get("task_id")
    if not isinstance(source_task_id, str):
        raise ValueError("referee_report_task_id_invalid")
    _, source = _task_by_id(repo, source_task_id)
    try:
        report_relpath = report_path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError("referee_report_outside_repo") from exc

    severities = _finding_severities(repo, report, source)
    findings = []
    if isinstance(report.get("verdicts"), list):
        for item in report["verdicts"]:
            if not isinstance(item, dict) or item.get("verdict") not in BLOCKING_VERDICTS:
                continue
            identifier = _finding_identifier(item)
            severity = severities.get(identifier, item.get("severity"))
            if item.get("verdict") == "cannot_verify" or severity == "major":
                findings.append(item)
    findings.sort(key=lambda item: (_finding_identifier(item), str(item.get("verdict"))))
    existing = _existing_fingerprints(repo)
    reserved: set[int] = set()
    created: list[Path] = []
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    required_locks = _source_required_locks(repo, source)
    wrote_new_task = False

    for finding in findings:
        fingerprint = _fingerprint(report_path, report, finding)
        if fingerprint in existing:
            created.append(existing[fingerprint])
            continue
        artifacts = _artifact_paths(report, source, finding)
        if not artifacts:
            raise ValueError(f"revision_artifact_scope_empty:{_finding_identifier(finding)}")
        number = _next_task_number(repo, reserved)
        reserved.add(number)
        task_id = f"T{number:03d}"
        slug = re.sub(r"[^a-z0-9]+", "_", _finding_identifier(finding).lower()).strip("_")[:48] or "finding"
        path = repo / ".orchestrator" / "backlog" / f"{task_id}_repair_{source_task_id.lower()}_{slug}.md"
        text = _render_revision_task(
            task_id=task_id,
            source_task_id=source_task_id,
            source=source,
            report_relpath=report_relpath,
            report_sha256=report_sha256,
            finding=finding,
            fingerprint=fingerprint,
            artifacts=artifacts,
            required_locks=required_locks,
            operator_workstream=operator_workstream,
        )
        all_paths = _task_paths(repo) + [path]
        diagnostics = lint_task_files(
            all_paths,
            repo_root=repo,
            network_workstreams=network_workstreams,
            v1_exemptions=_v1_exemptions(repo),
            task_texts={path: text},
        )
        candidate_failures = [item.as_dict() for item in diagnostics if item.task == task_id]
        if candidate_failures:
            raise ValueError("generated_revision_task_lint_failed:" + json.dumps(candidate_failures, sort_keys=True))
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            wrote_new_task = True
        created.append(path)
        existing[fingerprint] = path
    if wrote_new_task and not dry_run:
        _write_plan_approval_pending(repo, source_task_id)
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_revision_tasks.py",
        description="Generate scoped v2 repair tasks for blocking referee findings.",
    )
    parser.add_argument("--report", required=True, type=Path, help="Path to a referee report JSON file")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = generate_revision_tasks(
            repo=args.repo_root,
            report_path=args.report,
            dry_run=bool(args.dry_run),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": bool(args.dry_run),
                "tasks": [
                    path.resolve().relative_to(args.repo_root.resolve()).as_posix()
                    for path in paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
