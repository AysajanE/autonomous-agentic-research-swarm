#!/usr/bin/env python3
"""
Release assembly and catalog synchronization for the v1 research operating system.

This script sits downstream of the locked Stage 4 runtime/gate layer. It does
not redefine task/runtime semantics; instead, it snapshots the current repo's
release-relevant state into:

- reports/status/releases/release_<YYYY-MM-DD>.json
- reports/catalog.yaml

The canonical release-candidate paper surface is the Operator-owned T080 build:

- reports/paper/build/l2_l1_rent_working_paper.html
- reports/paper/build/l2_l1_rent_working_paper.pdf
- reports/paper/build/render_manifest.json

Until those three files exist together, the release manifest records
paper.status = "pending_stage2".
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RELEASE_MANIFEST_SCHEMA_VERSION = "research_swarm.release_manifest.v1"
CATALOG_SCHEMA_VERSION = "research_swarm.results_catalog.v1"
CATALOG_MANAGED_BY = "python scripts/release_assembly.py --write"
RELEASE_ASSEMBLY_COMMAND_TEMPLATE = (
    "python scripts/release_assembly.py --release-date <YYYY-MM-DD> --write"
)
CANONICAL_RELEASE_MANIFEST_PATTERN = "reports/status/releases/release_<YYYY-MM-DD>.json"
RELEASE_NAMESPACE = Path("reports/status/releases")
CATALOG_PATH = Path("reports/catalog.yaml")
PAPER_BUILD_NAMESPACE = "reports/paper/build/"
CANONICAL_PAPER_BUILD_REL_PATHS = (
    "reports/paper/build/l2_l1_rent_working_paper.html",
    "reports/paper/build/l2_l1_rent_working_paper.pdf",
    "reports/paper/build/render_manifest.json",
)
RELEASE_MANIFEST_FILENAME_RE = re.compile(r"^release_(\d{4}-\d{2}-\d{2})\.json$")

ALL_STAGE4_GATE_NAMES = (
    "framework_contract",
    "repo_structure",
    "project_contract",
    "protocol_complete",
    "workstreams_complete",
    "task_hygiene",
    "task_dependencies",
    "integration_ready_policy",
    "operator_surface_ownership",
    "raw_manifest_validity",
    "processed_manifest_validity",
    "swarm_run_manifest_validity",
    "judge_review_log_validity",
    "review_bundle_integrity",
    "referee_rubrics",
    "referee_report_validity",
    "referee_calibration",
    "referee_release_evidence",
    "citation_integrity",
    "literature_corpus",
    "recall_audit",
    "prompt_surface",
    "integrity_audit",
)
REQUIRED_RELEASE_GATE_NAMES = ALL_STAGE4_GATE_NAMES

CONTRACT_AND_PROTOCOL_PATHS = (
    "docs/protocol.md",
    "contracts/data_dictionary.md",
    "contracts/decisions.md",
    "contracts/model_spec.md",
    "contracts/schemas/panel_schema.yaml",
    "contracts/schemas/panel_schema_str_v1.yaml",
    "contracts/schemas/panel_schema_decomp_v1.yaml",
    "contracts/schemas/swarm_run_manifest_v1.yaml",
    "contracts/schemas/judge_review_log_v1.yaml",
    "contracts/schemas/release_manifest_v1.yaml",
)
REGISTRY_PATHS = ("registry/rollup_registry_v1.csv",)

EXCLUDED_REPORT_FILENAMES = {"README.md", ".gitkeep"}
ARTIFACT_SECTION_NAMES = (
    "contracts_and_protocol",
    "registry",
    "raw_manifests",
    "processed_manifests",
    "processed_data",
    "models",
    "proofs",
    "instances",
    "runtime_runs",
    "runtime_reviews",
    "validation",
    "figures",
    "tables",
)
COUNT_REQUIRED_FIELDS = (
    "contracts_and_protocol",
    "registry",
    "raw_manifests",
    "processed_manifests",
    "processed_data",
    "models",
    "proofs",
    "instances",
    "runtime_runs",
    "runtime_reviews",
    "validation",
    "figures",
    "tables",
    "paper_artifacts",
    "done_tasks",
    "integration_ready_tasks",
    "ready_for_review_tasks",
    "blocked_tasks",
)

_QUALITY_GATES_MODULE = None


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    try:
        os_path = path.resolve()
        import os

        os.chdir(os_path)
        yield
    finally:
        import os

        os.chdir(previous)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _parse_release_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"invalid_release_date:{value}") from exc


def _repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path.cwd().resolve()


def _release_manifest_relpath(release_date: date) -> Path:
    return RELEASE_NAMESPACE / f"release_{release_date.isoformat()}.json"


def _release_manifest_abspath(repo_root: Path, release_date: date) -> Path:
    return repo_root / _release_manifest_relpath(release_date)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{path.as_posix()}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"top_level_not_object:{path.as_posix()}")
    return payload


def _repo_relpath(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path_outside_repo:{path}") from exc


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_record(repo_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _repo_relpath(repo_root, path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _collect_explicit_artifacts(
    repo_root: Path, relpaths: Iterable[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for relpath in relpaths:
        path = repo_root / relpath
        if not path.exists() or not path.is_file():
            missing.append(relpath)
            continue
        artifacts.append(_artifact_record(repo_root, path))
    artifacts.sort(key=lambda item: str(item["path"]))
    return artifacts, missing


def _collect_dir_artifacts(
    repo_root: Path,
    rel_dir: str,
    *,
    suffixes: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = repo_root / rel_dir
    if not root.exists() or not root.is_dir():
        return []

    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_REPORT_FILENAMES:
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        artifacts.append(_artifact_record(repo_root, path))
    artifacts.sort(key=lambda item: str(item["path"]))
    return artifacts


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _git_head_sha(repo_root: Path) -> str | None:
    cp = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if cp.returncode != 0:
        return None
    value = (cp.stdout or "").strip()
    return value or None


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value), ensure_ascii=False)


def _load_quality_gates_module():
    global _QUALITY_GATES_MODULE
    if _QUALITY_GATES_MODULE is not None:
        return _QUALITY_GATES_MODULE

    path = Path(__file__).with_name("quality_gates.py")
    spec = importlib.util.spec_from_file_location("release_stage_quality_gates", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable_to_load_quality_gates:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_stage_quality_gates"] = module
    spec.loader.exec_module(module)
    _QUALITY_GATES_MODULE = module
    return module


def _load_framework_contract(repo_root: Path):
    quality_gates = _load_quality_gates_module()
    return quality_gates.load_framework_contract(repo_root)


def _collect_stage4_gate_results(repo_root: Path) -> dict[str, dict[str, Any]]:
    quality_gates = _load_quality_gates_module()
    results: dict[str, dict[str, Any]] = {}
    with _pushd(repo_root):
        for gate_name in ALL_STAGE4_GATE_NAMES:
            func = getattr(quality_gates, f"gate_{gate_name}")
            result = (
                func(require_literature_corpus=True)
                if gate_name == "citation_integrity"
                else func()
            )
            results[gate_name] = {
                "ok": bool(result.ok),
                "details": _json_safe(result.details),
            }
    return results


def _iter_task_files(repo_root: Path, projection_dirs: Iterable[str]) -> list[Path]:
    task_files: list[Path] = []
    orchestrator_root = repo_root / ".orchestrator"
    for folder_name in projection_dirs:
        folder = orchestrator_root / folder_name
        if not folder.exists() or not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            task_files.append(path)
    return task_files


def _collect_task_status(repo_root: Path) -> dict[str, Any]:
    quality_gates = _load_quality_gates_module()
    contract = quality_gates.load_framework_contract(repo_root)

    parse_failures: list[str] = []
    tasks: dict[str, Any] = {}

    for path in _iter_task_files(repo_root, contract.projection_dirs):
        try:
            task = quality_gates.load_task(path, contract)
        except Exception as exc:  # noqa: BLE001 - preserve exact parse failure text
            parse_failures.append(f"{_repo_relpath(repo_root, path)}:{exc}")
            continue
        if task.task_id in tasks:
            parse_failures.append(
                "duplicate_task_id:"
                f"{task.task_id}:{_repo_relpath(repo_root, tasks[task.task_id].path)}:"
                f"{_repo_relpath(repo_root, path)}"
            )
            continue
        tasks[task.task_id] = task

    state_counts = {state: 0 for state in contract.allowed_states}
    task_ids_by_state = {state: [] for state in contract.allowed_states}
    for task_id, task in sorted(tasks.items()):
        state_counts.setdefault(task.state, 0)
        task_ids_by_state.setdefault(task.state, [])
        state_counts[task.state] += 1
        task_ids_by_state[task.state].append(task_id)

    return {
        "state_counts": state_counts,
        "task_ids_by_state": task_ids_by_state,
        "parse_failures": parse_failures,
        "total_tasks": len(tasks),
    }


def assemble_release_manifest(
    repo_root: Path,
    release_date: date,
    *,
    allow_gate_failures: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract = _load_framework_contract(repo_root)
    gate_results = _collect_stage4_gate_results(repo_root)
    required_gate_failures = [
        gate_name
        for gate_name in REQUIRED_RELEASE_GATE_NAMES
        if not gate_results.get(gate_name, {}).get("ok", False)
    ]
    if (
        contract.features.get("integrity_audit_required_for_release") is True
        and gate_results.get("integrity_audit", {}).get("details", {}).get("skipped") is True
    ):
        required_gate_failures.append("integrity_audit")
    required_gate_failures = sorted(set(required_gate_failures))

    contracts_and_protocol, missing_contracts = _collect_explicit_artifacts(
        repo_root, CONTRACT_AND_PROTOCOL_PATHS
    )
    registry, missing_registry = _collect_explicit_artifacts(repo_root, REGISTRY_PATHS)
    raw_manifests = _collect_dir_artifacts(
        repo_root, "data/raw_manifest", suffixes={".json"}
    )
    processed_manifests = _collect_dir_artifacts(
        repo_root, "data/processed_manifest", suffixes={".json"}
    )
    processed_data = _collect_dir_artifacts(repo_root, "data/processed")
    models = _collect_dir_artifacts(repo_root, "reports/models")
    proofs = _collect_dir_artifacts(repo_root, "reports/proofs")
    instances = _collect_dir_artifacts(repo_root, "contracts/instances", suffixes={".json"})
    runtime_runs = _collect_dir_artifacts(
        repo_root, "reports/status/swarm_runs", suffixes={".json"}
    )
    runtime_reviews = _collect_dir_artifacts(
        repo_root, "reports/status/reviews", suffixes={".json"}
    )
    validation = _collect_dir_artifacts(repo_root, "reports/validation")
    figures = _collect_dir_artifacts(repo_root, "reports/figures")
    tables = _collect_dir_artifacts(repo_root, "reports/tables")
    paper_artifacts, missing_paper_artifacts = _collect_explicit_artifacts(
        repo_root, CANONICAL_PAPER_BUILD_REL_PATHS
    )

    task_status = _collect_task_status(repo_root)
    paper_status = "present" if not missing_paper_artifacts else "pending_stage2"

    missing_required_inputs = sorted(missing_contracts + missing_registry)
    notes: list[str] = []

    if missing_required_inputs:
        notes.append(
            "Missing required release inputs: " + ", ".join(missing_required_inputs)
        )
    if required_gate_failures:
        notes.append(
            "Stage 4 required gates not all passing: "
            + ", ".join(sorted(required_gate_failures))
        )
    if paper_status == "pending_stage2":
        notes.append(
            "Canonical paper build outputs are missing: "
            + ", ".join(sorted(missing_paper_artifacts))
        )
    if task_status["state_counts"].get("ready_for_review", 0) > 0:
        notes.append(
            "There are tasks still in ready_for_review; release status is a snapshot, not a completion claim."
        )
    if task_status["state_counts"].get("blocked", 0) > 0:
        notes.append(
            "There are blocked tasks in the control plane; release status captures them explicitly."
        )

    notes = _dedupe_preserve(notes)

    referee_hard_failures = sorted(
        set(required_gate_failures)
        & {
            "referee_rubrics",
            "referee_report_validity",
            "referee_calibration",
            "referee_release_evidence",
        }
    )
    if referee_hard_failures or (
        (missing_required_inputs or required_gate_failures) and not allow_gate_failures
    ):
        failure_parts: list[str] = []
        if missing_required_inputs:
            failure_parts.append(
                "missing_release_inputs=" + ",".join(missing_required_inputs)
            )
        if required_gate_failures:
            failure_parts.append(
                "failed_gates=" + ",".join(sorted(required_gate_failures))
            )
        raise SystemExit("release_assembly_blocked:" + ";".join(failure_parts))

    manifest_relpath = _release_manifest_relpath(release_date).as_posix()

    artifacts = {
        "contracts_and_protocol": contracts_and_protocol,
        "registry": registry,
        "raw_manifests": raw_manifests,
        "processed_manifests": processed_manifests,
        "processed_data": processed_data,
        "models": models,
        "proofs": proofs,
        "instances": instances,
        "runtime_runs": runtime_runs,
        "runtime_reviews": runtime_reviews,
        "validation": validation,
        "figures": figures,
        "tables": tables,
        "paper": {
            "status": paper_status,
            "expected_namespace": PAPER_BUILD_NAMESPACE,
            "artifacts": paper_artifacts,
        },
    }

    counts = {
        "contracts_and_protocol": len(contracts_and_protocol),
        "registry": len(registry),
        "raw_manifests": len(raw_manifests),
        "processed_manifests": len(processed_manifests),
        "processed_data": len(processed_data),
        "models": len(models),
        "proofs": len(proofs),
        "instances": len(instances),
        "runtime_runs": len(runtime_runs),
        "runtime_reviews": len(runtime_reviews),
        "validation": len(validation),
        "figures": len(figures),
        "tables": len(tables),
        "paper_artifacts": len(paper_artifacts),
        "done_tasks": int(task_status["state_counts"].get("done", 0)),
        "integration_ready_tasks": int(
            task_status["state_counts"].get("integration_ready", 0)
        ),
        "ready_for_review_tasks": int(
            task_status["state_counts"].get("ready_for_review", 0)
        ),
        "blocked_tasks": int(task_status["state_counts"].get("blocked", 0)),
    }

    return {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "release_id": f"release_{release_date.isoformat()}",
        "release_date_utc": release_date.isoformat(),
        "generated_at_utc": _utc_now_iso(),
        "repo": {
            "git_sha": _git_head_sha(repo_root),
            "project_mode": contract.project_mode,
            "catalog_path": CATALOG_PATH.as_posix(),
            "release_manifest_path": manifest_relpath,
        },
        "lifecycle": {
            "allowed_states": list(contract.allowed_states),
            "scientific_review_role": contract.scientific_review_role,
            "runtime_manifest_dir": contract.run_manifest_dir,
            "judge_review_dir": contract.judge_review_dir,
            "release_manifest_pattern": (
                contract.release_manifest_pattern
                or CANONICAL_RELEASE_MANIFEST_PATTERN
            ),
        },
        "task_status": task_status,
        "quality_gates": {
            "required_gate_names": list(REQUIRED_RELEASE_GATE_NAMES),
            "all_required_ok": len(required_gate_failures) == 0,
            "results": gate_results,
        },
        "referee_evidence": gate_results.get("referee_release_evidence", {}).get("details", {}),
        "artifacts": artifacts,
        "counts": counts,
        "notes": notes,
    }


def _require_keys(
    data: object, required_keys: Iterable[str], prefix: str
) -> list[str]:
    if not isinstance(data, dict):
        return [f"{prefix}:not_object"]
    failures: list[str] = []
    for key in required_keys:
        if key not in data:
            failures.append(f"{prefix}:missing_key:{key}")
    return failures


def _validate_artifact_entry(
    entry: object, repo_root: Path, prefix: str
) -> list[str]:
    failures = _require_keys(entry, ("path", "sha256", "bytes"), prefix)
    if failures:
        return failures

    assert isinstance(entry, dict)
    raw_path = entry.get("path")
    sha256 = entry.get("sha256")
    raw_bytes = entry.get("bytes")

    if not isinstance(raw_path, str) or not raw_path.strip():
        failures.append(f"{prefix}:invalid_path")
        return failures
    path = repo_root / raw_path
    if not path.exists() or not path.is_file():
        failures.append(f"{prefix}:missing_file:{raw_path}")
        return failures

    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        failures.append(f"{prefix}:invalid_sha256")
    elif _sha256_file(path) != sha256:
        failures.append(f"{prefix}:sha256_mismatch:{raw_path}")

    if not isinstance(raw_bytes, int) or raw_bytes < 0:
        failures.append(f"{prefix}:invalid_bytes")
    elif path.stat().st_size != raw_bytes:
        failures.append(f"{prefix}:bytes_mismatch:{raw_path}")

    return failures


def validate_release_manifest(path: Path, repo_root: Path) -> list[str]:
    repo_root = repo_root.resolve()
    failures: list[str] = []

    relpath = _repo_relpath(repo_root, path)

    match = RELEASE_MANIFEST_FILENAME_RE.fullmatch(path.name)
    if match is None:
        failures.append(f"noncanonical_release_manifest_filename:{relpath}")
        return failures

    filename_date = match.group(1)
    try:
        payload = _load_json_object(path)
    except ValueError as exc:
        failures.append(str(exc))
        return failures

    failures.extend(
        _require_keys(
            payload,
            (
                "schema_version",
                "release_id",
                "release_date_utc",
                "generated_at_utc",
                "repo",
                "lifecycle",
                "task_status",
                "quality_gates",
                "artifacts",
                "counts",
                "notes",
            ),
            relpath,
        )
    )
    if failures:
        return failures

    if payload.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        failures.append(
            f"{relpath}:invalid_schema_version:{payload.get('schema_version')}"
        )
    if payload.get("release_id") != f"release_{filename_date}":
        failures.append(f"{relpath}:release_id_mismatch")
    if payload.get("release_date_utc") != filename_date:
        failures.append(f"{relpath}:release_date_filename_mismatch")

    repo = payload.get("repo")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(
            repo,
            ("git_sha", "project_mode", "catalog_path", "release_manifest_path"),
            "repo",
        )
    )
    if isinstance(repo, dict):
        if repo.get("catalog_path") != CATALOG_PATH.as_posix():
            failures.append(f"{relpath}:catalog_path_mismatch")
        if repo.get("release_manifest_path") != relpath:
            failures.append(f"{relpath}:release_manifest_path_mismatch")

    lifecycle = payload.get("lifecycle")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(
            lifecycle,
            (
                "allowed_states",
                "scientific_review_role",
                "runtime_manifest_dir",
                "judge_review_dir",
                "release_manifest_pattern",
            ),
            "lifecycle",
        )
    )
    if isinstance(lifecycle, dict):
        if lifecycle.get("release_manifest_pattern") != CANONICAL_RELEASE_MANIFEST_PATTERN:
            failures.append(f"{relpath}:release_manifest_pattern_mismatch")

    task_status = payload.get("task_status")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(
            task_status,
            ("state_counts", "task_ids_by_state", "parse_failures"),
            "task_status",
        )
    )

    allowed_states: list[str] = []
    if isinstance(lifecycle, dict) and isinstance(lifecycle.get("allowed_states"), list):
        allowed_states = [
            str(item) for item in lifecycle.get("allowed_states", []) if isinstance(item, str)
        ]

    if isinstance(task_status, dict):
        state_counts = task_status.get("state_counts")
        task_ids_by_state = task_status.get("task_ids_by_state")
        if not isinstance(state_counts, dict):
            failures.append(f"{relpath}:task_status_state_counts_not_object")
        if not isinstance(task_ids_by_state, dict):
            failures.append(f"{relpath}:task_status_task_ids_by_state_not_object")
        if isinstance(state_counts, dict) and isinstance(task_ids_by_state, dict):
            for state in allowed_states:
                if state not in state_counts:
                    failures.append(f"{relpath}:missing_state_count:{state}")
                    continue
                if state not in task_ids_by_state:
                    failures.append(f"{relpath}:missing_task_ids_by_state:{state}")
                    continue
                if not isinstance(state_counts[state], int) or state_counts[state] < 0:
                    failures.append(f"{relpath}:invalid_state_count:{state}")
                    continue
                if not isinstance(task_ids_by_state[state], list):
                    failures.append(f"{relpath}:invalid_task_id_list:{state}")
                    continue
                if state_counts[state] != len(task_ids_by_state[state]):
                    failures.append(f"{relpath}:state_count_length_mismatch:{state}")

    quality_gates = payload.get("quality_gates")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(
            quality_gates,
            ("required_gate_names", "all_required_ok", "results"),
            "quality_gates",
        )
    )
    if isinstance(quality_gates, dict):
        required_gate_names = quality_gates.get("required_gate_names")
        all_required_ok = quality_gates.get("all_required_ok")
        results = quality_gates.get("results")
        legacy_without_referee_evidence = "referee_evidence" not in payload
        validation_gate_names = list(REQUIRED_RELEASE_GATE_NAMES)

        if not isinstance(required_gate_names, list):
            failures.append(f"{relpath}:required_gate_names_not_list")
        elif legacy_without_referee_evidence:
            validation_gate_names = [
                name for name in required_gate_names if isinstance(name, str)
            ]
            if not set(validation_gate_names).issubset(set(REQUIRED_RELEASE_GATE_NAMES)):
                failures.append(f"{relpath}:required_gate_names_mismatch")
        elif set(required_gate_names) != set(REQUIRED_RELEASE_GATE_NAMES):
            failures.append(f"{relpath}:required_gate_names_mismatch")

        if not isinstance(all_required_ok, bool):
            failures.append(f"{relpath}:all_required_ok_not_bool")

        required_gate_oks: list[bool] = []
        if not isinstance(results, dict):
            failures.append(f"{relpath}:quality_gate_results_not_object")
        else:
            for gate_name in validation_gate_names:
                gate_result = results.get(gate_name)
                if not isinstance(gate_result, dict):
                    failures.append(f"{relpath}:missing_gate_result:{gate_name}")
                    continue
                gate_ok = gate_result.get("ok")
                if not isinstance(gate_ok, bool):
                    failures.append(f"{relpath}:gate_ok_not_bool:{gate_name}")
                    continue
                required_gate_oks.append(gate_ok)
            if isinstance(all_required_ok, bool) and required_gate_oks:
                if all_required_ok != all(required_gate_oks):
                    failures.append(f"{relpath}:all_required_ok_mismatch")

    referee_evidence = payload.get("referee_evidence")
    if referee_evidence is not None and not isinstance(referee_evidence, dict):
        failures.append(f"{relpath}:referee_evidence_not_object")
    elif isinstance(referee_evidence, dict):
        calibration = referee_evidence.get("calibration")
        if calibration is not None:
            failures.extend(
                f"{relpath}:{failure}"
                for failure in _validate_artifact_entry(
                    calibration,
                    repo_root,
                    "referee_evidence.calibration",
                )
            )
        evidence_items = referee_evidence.get("evidence", [])
        if not isinstance(evidence_items, list):
            failures.append(f"{relpath}:referee_evidence_items_not_list")
        else:
            for item_index, item in enumerate(evidence_items):
                reports = item.get("reports") if isinstance(item, dict) else None
                if not isinstance(reports, list):
                    failures.append(f"{relpath}:referee_evidence_reports_not_list:{item_index}")
                    continue
                for report_index, report in enumerate(reports):
                    failures.extend(
                        f"{relpath}:{failure}"
                        for failure in _validate_artifact_entry(
                            report,
                            repo_root,
                            f"referee_evidence.evidence[{item_index}].reports[{report_index}]",
                        )
                    )

    artifacts = payload.get("artifacts")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(
            artifacts,
            (*ARTIFACT_SECTION_NAMES, "paper"),
            "artifacts",
        )
    )
    if isinstance(artifacts, dict):
        for section_name in ARTIFACT_SECTION_NAMES:
            section = artifacts.get(section_name)
            if not isinstance(section, list):
                failures.append(f"{relpath}:artifact_section_not_list:{section_name}")
                continue
            for index, entry in enumerate(section):
                failures.extend(
                    f"{relpath}:{failure}"
                    for failure in _validate_artifact_entry(
                        entry,
                        repo_root,
                        f"artifacts.{section_name}[{index}]",
                    )
                )

        paper = artifacts.get("paper")
        failures.extend(
            f"{relpath}:{failure}"
            for failure in _require_keys(
                paper,
                ("status", "expected_namespace", "artifacts"),
                "artifacts.paper",
            )
        )
        if isinstance(paper, dict):
            status = paper.get("status")
            expected_namespace = paper.get("expected_namespace")
            paper_artifacts = paper.get("artifacts")

            if status not in {"pending_stage2", "present"}:
                failures.append(f"{relpath}:invalid_paper_status:{status}")
            if expected_namespace != PAPER_BUILD_NAMESPACE:
                failures.append(f"{relpath}:paper_expected_namespace_mismatch")
            if not isinstance(paper_artifacts, list):
                failures.append(f"{relpath}:paper_artifacts_not_list")
            else:
                for index, entry in enumerate(paper_artifacts):
                    failures.extend(
                        f"{relpath}:{failure}"
                        for failure in _validate_artifact_entry(
                            entry,
                            repo_root,
                            f"artifacts.paper.artifacts[{index}]",
                        )
                    )
                if status == "present" and not paper_artifacts:
                    failures.append(f"{relpath}:paper_present_without_artifacts")

    counts = payload.get("counts")
    failures.extend(
        f"{relpath}:{failure}"
        for failure in _require_keys(counts, COUNT_REQUIRED_FIELDS, "counts")
    )
    if isinstance(counts, dict) and isinstance(artifacts, dict):
        expected_counts = {
            "contracts_and_protocol": len(artifacts.get("contracts_and_protocol", []))
            if isinstance(artifacts.get("contracts_and_protocol"), list)
            else None,
            "registry": len(artifacts.get("registry", []))
            if isinstance(artifacts.get("registry"), list)
            else None,
            "raw_manifests": len(artifacts.get("raw_manifests", []))
            if isinstance(artifacts.get("raw_manifests"), list)
            else None,
            "processed_manifests": len(artifacts.get("processed_manifests", []))
            if isinstance(artifacts.get("processed_manifests"), list)
            else None,
            "runtime_runs": len(artifacts.get("runtime_runs", []))
            if isinstance(artifacts.get("runtime_runs"), list)
            else None,
            "runtime_reviews": len(artifacts.get("runtime_reviews", []))
            if isinstance(artifacts.get("runtime_reviews"), list)
            else None,
            "validation": len(artifacts.get("validation", []))
            if isinstance(artifacts.get("validation"), list)
            else None,
            "figures": len(artifacts.get("figures", []))
            if isinstance(artifacts.get("figures"), list)
            else None,
            "tables": len(artifacts.get("tables", []))
            if isinstance(artifacts.get("tables"), list)
            else None,
            "paper_artifacts": (
                len(artifacts.get("paper", {}).get("artifacts", []))
                if isinstance(artifacts.get("paper"), dict)
                and isinstance(artifacts.get("paper", {}).get("artifacts"), list)
                else None
            ),
        }

        for key, expected in expected_counts.items():
            raw_value = counts.get(key)
            if not isinstance(raw_value, int) or raw_value < 0:
                failures.append(f"{relpath}:invalid_count:{key}")
                continue
            if expected is not None and raw_value != expected:
                failures.append(f"{relpath}:count_mismatch:{key}")

    if isinstance(counts, dict) and isinstance(task_status, dict):
        state_counts = task_status.get("state_counts")
        if isinstance(state_counts, dict):
            state_count_mapping = {
                "done_tasks": "done",
                "integration_ready_tasks": "integration_ready",
                "ready_for_review_tasks": "ready_for_review",
                "blocked_tasks": "blocked",
            }
            for count_key, state_key in state_count_mapping.items():
                raw_value = counts.get(count_key)
                state_value = state_counts.get(state_key)
                if not isinstance(raw_value, int) or raw_value < 0:
                    failures.append(f"{relpath}:invalid_count:{count_key}")
                    continue
                if isinstance(state_value, int) and raw_value != state_value:
                    failures.append(f"{relpath}:task_count_mismatch:{count_key}")

    notes = payload.get("notes")
    if not isinstance(notes, list):
        failures.append(f"{relpath}:notes_not_list")

    return failures


def _load_release_manifest_payloads(
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    release_dir = repo_root / RELEASE_NAMESPACE
    if not release_dir.exists() or not release_dir.is_dir():
        return [], []

    payloads: list[dict[str, Any]] = []
    failures: list[str] = []

    for path in sorted(release_dir.glob("*.json")):
        manifest_failures = validate_release_manifest(path, repo_root)
        if manifest_failures:
            failures.extend(manifest_failures)
            continue
        payloads.append(_load_json_object(path))

    payloads.sort(key=lambda item: str(item.get("release_date_utc", "")))
    return payloads, failures


def build_catalog_payload(
    repo_root: Path, manifest_payloads: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    project_mode = "unknown"
    try:
        project_mode = _load_framework_contract(repo_root).project_mode or "unknown"
    except Exception:  # noqa: BLE001 - catalog should still be renderable
        project_mode = "unknown"

    if manifest_payloads is None:
        manifest_payloads, failures = _load_release_manifest_payloads(repo_root)
        if failures:
            raise SystemExit(
                "cannot_build_catalog_with_invalid_release_manifests:"
                + ";".join(failures)
            )

    releases: list[dict[str, Any]] = []
    for payload in sorted(
        manifest_payloads, key=lambda item: str(item.get("release_date_utc", ""))
    ):
        counts = payload.get("counts", {})
        artifacts = payload.get("artifacts", {})
        paper = artifacts.get("paper", {}) if isinstance(artifacts, dict) else {}
        repo = payload.get("repo", {}) if isinstance(payload.get("repo"), dict) else {}
        quality_gates = (
            payload.get("quality_gates", {})
            if isinstance(payload.get("quality_gates"), dict)
            else {}
        )

        releases.append(
            {
                "release_date_utc": payload.get("release_date_utc"),
                "manifest_path": repo.get("release_manifest_path"),
                "git_sha": repo.get("git_sha"),
                "required_gates_ok": quality_gates.get("all_required_ok"),
                "done_task_count": counts.get("done_tasks"),
                "blocked_task_count": counts.get("blocked_tasks"),
                "run_manifest_count": counts.get("runtime_runs"),
                "review_log_count": counts.get("runtime_reviews"),
                "validation_artifact_count": counts.get("validation"),
                "figure_count": counts.get("figures"),
                "table_count": counts.get("tables"),
                "paper_status": paper.get("status"),
            }
        )

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "managed_by": CATALOG_MANAGED_BY,
        "project": {
            "mode": project_mode,
            "primary_metric": "Settlement Take Rate (STR)",
            "protocol_path": "docs/protocol.md",
            "data_dictionary_path": "contracts/data_dictionary.md",
            "decisions_path": "contracts/decisions.md",
            "registry_path": "registry/rollup_registry_v1.csv",
        },
        "release_namespace": {
            "directory": RELEASE_NAMESPACE.as_posix(),
            "filename_pattern": "release_<YYYY-MM-DD>.json",
            "manifest_schema": "contracts/schemas/release_manifest_v1.yaml",
            "assembly_command": RELEASE_ASSEMBLY_COMMAND_TEMPLATE,
        },
        "status_namespaces": {
            "swarm_runs": "reports/status/swarm_runs/",
            "reviews": "reports/status/reviews/",
            "releases": "reports/status/releases/",
        },
        "artifact_roots": {
            "raw_manifests": "data/raw_manifest/",
            "processed_manifests": "data/processed_manifest/",
            "validation": "reports/validation/",
            "figures": "reports/figures/",
            "tables": "reports/tables/",
            "paper_build": PAPER_BUILD_NAMESPACE,
        },
        "releases": releases,
    }


def render_catalog_document(payload: dict[str, Any]) -> str:
    lines = [
        f"schema_version: {_yaml_scalar(payload['schema_version'])}",
        f"managed_by: {_yaml_scalar(payload['managed_by'])}",
        "project:",
        f"  mode: {_yaml_scalar(payload['project']['mode'])}",
        f"  primary_metric: {_yaml_scalar(payload['project']['primary_metric'])}",
        f"  protocol_path: {_yaml_scalar(payload['project']['protocol_path'])}",
        f"  data_dictionary_path: {_yaml_scalar(payload['project']['data_dictionary_path'])}",
        f"  decisions_path: {_yaml_scalar(payload['project']['decisions_path'])}",
        f"  registry_path: {_yaml_scalar(payload['project']['registry_path'])}",
        "release_namespace:",
        f"  directory: {_yaml_scalar(payload['release_namespace']['directory'])}",
        f"  filename_pattern: {_yaml_scalar(payload['release_namespace']['filename_pattern'])}",
        f"  manifest_schema: {_yaml_scalar(payload['release_namespace']['manifest_schema'])}",
        f"  assembly_command: {_yaml_scalar(payload['release_namespace']['assembly_command'])}",
        "status_namespaces:",
        f"  swarm_runs: {_yaml_scalar(payload['status_namespaces']['swarm_runs'])}",
        f"  reviews: {_yaml_scalar(payload['status_namespaces']['reviews'])}",
        f"  releases: {_yaml_scalar(payload['status_namespaces']['releases'])}",
        "artifact_roots:",
        f"  raw_manifests: {_yaml_scalar(payload['artifact_roots']['raw_manifests'])}",
        f"  processed_manifests: {_yaml_scalar(payload['artifact_roots']['processed_manifests'])}",
        f"  validation: {_yaml_scalar(payload['artifact_roots']['validation'])}",
        f"  figures: {_yaml_scalar(payload['artifact_roots']['figures'])}",
        f"  tables: {_yaml_scalar(payload['artifact_roots']['tables'])}",
        f"  paper_build: {_yaml_scalar(payload['artifact_roots']['paper_build'])}",
    ]

    releases = payload.get("releases") or []
    if not releases:
        lines.append("releases: []")
        return "\n".join(lines) + "\n"

    lines.append("releases:")
    for entry in releases:
        lines.extend(
            [
                f"  - release_date_utc: {_yaml_scalar(entry['release_date_utc'])}",
                f"    manifest_path: {_yaml_scalar(entry['manifest_path'])}",
                f"    git_sha: {_yaml_scalar(entry['git_sha'])}",
                f"    required_gates_ok: {_yaml_scalar(entry['required_gates_ok'])}",
                f"    done_task_count: {_yaml_scalar(entry['done_task_count'])}",
                f"    blocked_task_count: {_yaml_scalar(entry['blocked_task_count'])}",
                f"    run_manifest_count: {_yaml_scalar(entry['run_manifest_count'])}",
                f"    review_log_count: {_yaml_scalar(entry['review_log_count'])}",
                f"    validation_artifact_count: {_yaml_scalar(entry['validation_artifact_count'])}",
                f"    figure_count: {_yaml_scalar(entry['figure_count'])}",
                f"    table_count: {_yaml_scalar(entry['table_count'])}",
                f"    paper_status: {_yaml_scalar(entry['paper_status'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def sync_catalog(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    manifest_payloads, failures = _load_release_manifest_payloads(repo_root)
    if failures:
        raise SystemExit(
            "cannot_sync_catalog_with_invalid_release_manifests:" + ";".join(failures)
        )
    payload = build_catalog_payload(repo_root, manifest_payloads)
    text = render_catalog_document(payload)
    catalog_path = repo_root / CATALOG_PATH
    _write_text(catalog_path, text)
    return catalog_path


def check_release_integrity(
    repo_root: Path, *, enforce_required_gates: bool = True
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    failures: list[str] = []

    release_dir = repo_root / RELEASE_NAMESPACE
    if not release_dir.exists() or not release_dir.is_dir():
        failures.append(f"missing_release_namespace:{RELEASE_NAMESPACE.as_posix()}")
    if not (release_dir / "README.md").exists():
        failures.append("missing_release_namespace_readme:reports/status/releases/README.md")

    gate_results = _collect_stage4_gate_results(repo_root)
    if enforce_required_gates:
        for gate_name in REQUIRED_RELEASE_GATE_NAMES:
            if not gate_results.get(gate_name, {}).get("ok", False):
                failures.append(f"required_stage4_gate_failed:{gate_name}")

    manifest_payloads, manifest_failures = _load_release_manifest_payloads(repo_root)
    failures.extend(manifest_failures)

    expected_catalog = render_catalog_document(
        build_catalog_payload(repo_root, manifest_payloads)
    )
    catalog_file = repo_root / CATALOG_PATH
    if not catalog_file.exists():
        failures.append(f"missing_catalog:{CATALOG_PATH.as_posix()}")
    else:
        actual_catalog = _read_text(catalog_file)
        if actual_catalog != expected_catalog:
            failures.append(f"catalog_out_of_sync:{CATALOG_PATH.as_posix()}")

    failures = _dedupe_preserve(failures)
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "release_manifest_count": len(manifest_payloads),
        "catalog_path": CATALOG_PATH.as_posix(),
    }


def write_release(
    repo_root: Path,
    release_date: date,
    *,
    allow_gate_failures: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest = assemble_release_manifest(
        repo_root, release_date, allow_gate_failures=allow_gate_failures
    )
    manifest_path = _release_manifest_abspath(repo_root, release_date)
    _write_json(manifest_path, manifest)

    manifest_failures = validate_release_manifest(manifest_path, repo_root)
    if manifest_failures:
        raise SystemExit(
            "release_manifest_invalid_after_write:" + ";".join(manifest_failures)
        )

    sync_catalog(repo_root)
    integrity = check_release_integrity(
        repo_root, enforce_required_gates=not allow_gate_failures
    )
    if not integrity["ok"]:
        raise SystemExit(
            "release_integrity_failed_after_write:" + ";".join(integrity["failures"])
        )

    return {
        "ok": True,
        "release_manifest": _repo_relpath(repo_root, manifest_path),
        "catalog_path": CATALOG_PATH.as_posix(),
        "required_gates_ok": manifest["quality_gates"]["all_required_ok"],
        "paper_status": manifest["artifacts"]["paper"]["status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_assembly.py")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to inspect/write (default: current working directory).",
    )
    parser.add_argument(
        "--release-date",
        default=_utc_today().isoformat(),
        help="UTC release date in YYYY-MM-DD form (default: today UTC).",
    )
    parser.add_argument(
        "--as-of",
        dest="release_date",
        help="Backward-compatible alias for --release-date.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the canonical release manifest and rebuild reports/catalog.yaml.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify release namespace integrity and catalog synchronization.",
    )
    parser.add_argument(
        "--allow-gate-failures",
        action="store_true",
        help="Allow preview/write even when required Stage 4 gates are failing.",
    )
    args = parser.parse_args(argv)

    if args.write and args.check:
        raise SystemExit("choose_only_one_of:--write,--check")

    repo_root = _repo_root(args.repo_root)
    release_date = _parse_release_date(args.release_date)

    if args.check:
        payload = check_release_integrity(
            repo_root, enforce_required_gates=not args.allow_gate_failures
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["ok"] else 1

    if args.write:
        payload = write_release(
            repo_root, release_date, allow_gate_failures=args.allow_gate_failures
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    payload = assemble_release_manifest(
        repo_root, release_date, allow_gate_failures=args.allow_gate_failures
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
