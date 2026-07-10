#!/usr/bin/env python3
"""
Deterministic runtime and repository quality gates.

Stage 4 scope:
- enforce the Stage 3 role/state/runtime contract
- stay offline and sample-safe
- validate runtime/review JSON artifacts introduced in Stage 4
- defer Stage 5 release/paper/catalog integrity checks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from swarm_taskfile import parse_status_value as _parse_status_value
from swarm_taskfile import parse_task_frontmatter as _parse_task_frontmatter


SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v2"
SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 = "research_swarm.runtime_run_manifest.v1"
JUDGE_REVIEW_LOG_SCHEMA_VERSION = "research_swarm.judge_review_log.v1"

DEFAULT_ALLOWED_STATES = (
    "backlog",
    "active",
    "integration_ready",
    "ready_for_review",
    "blocked",
    "done",
)
DEFAULT_ALLOWED_ROLES = ("Planner", "Worker", "Judge", "Operator")
DEFAULT_TASK_EXECUTION_ROLES = ("Worker", "Operator")
DEFAULT_SCIENTIFIC_REVIEW_ROLE = "Judge"
DEFAULT_NETWORK_WORKSTREAMS = ("W1", "W2", "W3")
DEFAULT_PROMPT_TEMPLATES = {
    "planner": "docs/prompts/planner.md",
    "worker": "docs/prompts/worker.md",
    "judge": "docs/prompts/judge.md",
    "operator": "docs/prompts/operator.md",
}
DEFAULT_INTEGRATION_READY_ELIGIBLE_WORKSTREAMS = ("W0", "W3", "W8", "W9")
DEFAULT_INTEGRATION_READY_ELIGIBLE_TASK_KINDS = (
    "protocol",
    "registry",
    "bridge",
    "model",
    "ops",
)
DEFAULT_OPERATOR_OWNED_SHARED_SURFACES = (
    "reports/catalog.yaml",
    "reports/paper/build/",
    "reports/status/releases/",
)
FORBIDDEN_INTEGRATION_READY_OUTPUT_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "reports/validation/",
    "reports/figures/",
    "reports/tables/",
)
REQUIRED_FRONTMATTER_KEYS = (
    "task_id",
    "title",
    "workstream",
    "role",
    "priority",
    "dependencies",
    "allowed_paths",
    "disallowed_paths",
    "outputs",
    "gates",
    "stop_conditions",
)
REQUIRED_TASK_HEADINGS = (
    "## Context",
    "## Inputs",
    "## Outputs",
    "## Success Criteria",
    "## Review Bundle Requirements",
    "## Validation / Commands",
    "## Status",
    "## Notes / Decisions",
)
VALID_TASK_PRIORITIES = {"low", "medium", "high"}
DEFERRED_REQUIRED_PATH_PREFIXES = (
    "reports/status/",
    "reports/catalog.yaml",
    "reports/paper/",
    "reports/status/releases/",
    "scripts/release_assembly.py",
)
MANIFEST_BACKED_LOCAL_ETL_OUTPUT_PREFIXES = (
    "data/raw/",
    "data/processed/",
)


@dataclass(frozen=True)
class GateResult:
    ok: bool
    details: dict[str, object]


@dataclass(frozen=True)
class FrameworkContract:
    project_mode: str | None
    features: dict[str, bool]
    allowed_roles: tuple[str, ...]
    task_execution_roles: tuple[str, ...]
    scientific_review_role: str
    allowed_states: tuple[str, ...]
    projection_dirs: tuple[str, ...]
    prompt_templates: dict[str, str]
    network_workstreams: tuple[str, ...]
    integration_ready_eligible_workstreams: tuple[str, ...]
    integration_ready_eligible_task_kinds: tuple[str, ...]
    forbid_unvalidated_empirical_data_outputs: bool
    operator_owned_shared_surfaces: tuple[str, ...]
    run_manifest_dir: str
    judge_review_dir: str
    release_manifest_pattern: str | None
    required_paths: tuple[str, ...]


@dataclass(frozen=True)
class Task:
    path: Path
    task_id: str
    title: str
    workstream: str
    task_kind: str | None
    role: str
    priority: str
    dependencies: list[str]
    integration_ready_dependencies: list[str]
    allow_network: bool
    allowed_paths: list[str]
    disallowed_paths: list[str]
    outputs: list[str]
    gates: list[str]
    stop_conditions: list[str]
    state: str
    last_updated: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_repo_relative_path(value: str) -> str:
    out = value.strip().replace("\\", "/")
    while out.startswith("./"):
        out = out[2:]
    return out


def _path_matches_prefix(value: str, prefix: str) -> bool:
    norm_value = _normalize_repo_relative_path(value)
    norm_prefix = _normalize_repo_relative_path(prefix)
    if norm_value == norm_prefix.rstrip("/"):
        return True
    return norm_value.startswith(norm_prefix)


def _git_path_is_ignored(path: str, repo: Path) -> bool:
    if not _repo_has_git_worktree(repo):
        return False
    cp = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=str(repo),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cp.returncode == 0


def _git_path_is_tracked(path: str, repo: Path) -> bool:
    if not _repo_has_git_worktree(repo):
        return True
    cp = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=str(repo),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cp.returncode == 0


def _repo_has_git_worktree(repo: Path) -> bool:
    cp = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(repo),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cp.returncode == 0


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return default


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def _parse_project_mode(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw_line in _read_text(path).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or not line.startswith("mode:"):
            continue
        value = line.split(":", 1)[1].strip().strip("'\"").lower()
        return value or None
    return None


def _parse_feature_flags(value: object) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, bool] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        out[key] = _coerce_bool(item, default=False)
    return out


def _parse_required_paths(value: object, mode: str | None) -> list[str]:
    if isinstance(value, list):
        return _coerce_str_list(value)
    if isinstance(value, dict):
        out: list[str] = []
        out.extend(_coerce_str_list(value.get("common")))
        if mode:
            out.extend(_coerce_str_list(value.get(mode)))
        return out
    return []


def load_framework_contract(repo: Path = Path(".")) -> FrameworkContract:
    framework_path = repo / "contracts" / "framework.json"
    if not framework_path.exists():
        raise ValueError(f"missing_framework_contract:{framework_path}")

    try:
        raw = json.loads(_read_text(framework_path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_framework_json:{exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("framework_top_level_not_object")

    roles = raw.get("roles")
    states = raw.get("states")
    review_bundle = raw.get("review_bundle")
    integration_ready_policy = raw.get("integration_ready_policy")
    release_policy = raw.get("release_policy")

    project_mode = _parse_project_mode(repo / "contracts" / "project.yaml")
    features = _parse_feature_flags(raw.get("features"))

    allowed_roles = tuple(_coerce_str_list(roles.get("allowed") if isinstance(roles, dict) else None) or list(DEFAULT_ALLOWED_ROLES))
    task_execution_roles = tuple(
        _coerce_str_list(roles.get("task_execution_roles") if isinstance(roles, dict) else None)
        or list(DEFAULT_TASK_EXECUTION_ROLES)
    )
    scientific_review_role = (
        str(roles.get("scientific_review_role")).strip()
        if isinstance(roles, dict) and isinstance(roles.get("scientific_review_role"), str)
        else DEFAULT_SCIENTIFIC_REVIEW_ROLE
    )
    allowed_states = tuple(
        _coerce_str_list(states.get("allowed") if isinstance(states, dict) else None) or list(DEFAULT_ALLOWED_STATES)
    )
    projection_dirs = tuple(
        Path(item).name
        for item in (_coerce_str_list(states.get("projection_dirs") if isinstance(states, dict) else None) or list(DEFAULT_ALLOWED_STATES))
    )

    prompt_templates = dict(DEFAULT_PROMPT_TEMPLATES)
    raw_prompts = raw.get("prompt_templates")
    if isinstance(raw_prompts, dict):
        for key, value in raw_prompts.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                prompt_templates[key] = _normalize_repo_relative_path(value)

    network_workstreams = tuple(_coerce_str_list(raw.get("network_workstreams")) or list(DEFAULT_NETWORK_WORKSTREAMS))

    eligible_workstreams = tuple(
        _coerce_str_list(
            integration_ready_policy.get("eligible_workstreams")
            if isinstance(integration_ready_policy, dict)
            else None
        )
        or list(DEFAULT_INTEGRATION_READY_ELIGIBLE_WORKSTREAMS)
    )
    eligible_task_kinds = tuple(
        _coerce_str_list(
            integration_ready_policy.get("eligible_task_kinds")
            if isinstance(integration_ready_policy, dict)
            else None
        )
        or list(DEFAULT_INTEGRATION_READY_ELIGIBLE_TASK_KINDS)
    )
    forbid_unvalidated_empirical_data_outputs = _coerce_bool(
        integration_ready_policy.get("forbid_unvalidated_empirical_data_outputs")
        if isinstance(integration_ready_policy, dict)
        else None,
        default=True,
    )

    operator_owned_shared_surfaces = tuple(
        _coerce_str_list(raw.get("operator_owned_shared_surfaces")) or list(DEFAULT_OPERATOR_OWNED_SHARED_SURFACES)
    )

    run_manifest_dir = (
        _normalize_repo_relative_path(review_bundle.get("run_manifest_dir"))
        if isinstance(review_bundle, dict) and isinstance(review_bundle.get("run_manifest_dir"), str)
        else "reports/status/swarm_runs"
    )
    judge_review_dir = (
        _normalize_repo_relative_path(review_bundle.get("judge_review_dir"))
        if isinstance(review_bundle, dict) and isinstance(review_bundle.get("judge_review_dir"), str)
        else "reports/status/reviews"
    )
    release_manifest_pattern = (
        release_policy.get("release_manifest_pattern")
        if isinstance(release_policy, dict) and isinstance(release_policy.get("release_manifest_pattern"), str)
        else None
    )

    required_paths = tuple(_parse_required_paths(raw.get("required_paths"), project_mode))

    return FrameworkContract(
        project_mode=project_mode,
        features=features,
        allowed_roles=allowed_roles,
        task_execution_roles=task_execution_roles,
        scientific_review_role=scientific_review_role,
        allowed_states=allowed_states,
        projection_dirs=projection_dirs,
        prompt_templates=prompt_templates,
        network_workstreams=network_workstreams,
        integration_ready_eligible_workstreams=eligible_workstreams,
        integration_ready_eligible_task_kinds=eligible_task_kinds,
        forbid_unvalidated_empirical_data_outputs=forbid_unvalidated_empirical_data_outputs,
        operator_owned_shared_surfaces=operator_owned_shared_surfaces,
        run_manifest_dir=run_manifest_dir,
        judge_review_dir=judge_review_dir,
        release_manifest_pattern=release_manifest_pattern,
        required_paths=required_paths,
    )


def load_task(path: Path, contract: FrameworkContract) -> Task:
    text = _read_text(path)
    frontmatter = _parse_task_frontmatter(text)
    if frontmatter is None:
        raise ValueError("missing_yaml_frontmatter")

    for key in REQUIRED_FRONTMATTER_KEYS:
        if key not in frontmatter:
            raise ValueError(f"frontmatter_missing_key:{key}")

    def require_str(key: str) -> str:
        value = frontmatter.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"frontmatter_invalid_string:{key}")
        return value.strip()

    def require_list(key: str) -> list[str]:
        value = frontmatter.get(key)
        if not isinstance(value, list):
            raise ValueError(f"frontmatter_invalid_list:{key}")
        out = _coerce_str_list(value)
        if key in {"allowed_paths", "disallowed_paths", "outputs", "gates", "stop_conditions"} and not out:
            raise ValueError(f"frontmatter_empty_list:{key}")
        return out

    role = require_str("role")
    priority = require_str("priority").lower()
    state = _parse_status_value(text, "State")
    last_updated = _parse_status_value(text, "Last updated")

    if role not in set(contract.allowed_roles):
        raise ValueError(f"invalid_role:{role}")
    if priority not in VALID_TASK_PRIORITIES:
        raise ValueError(f"invalid_priority:{priority}")
    if state is None or state not in set(contract.allowed_states):
        raise ValueError(f"invalid_state:{state}")
    if last_updated is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_updated):
        raise ValueError(f"invalid_last_updated:{last_updated}")

    raw_task_kind = frontmatter.get("task_kind")
    task_kind = raw_task_kind.strip() if isinstance(raw_task_kind, str) and raw_task_kind.strip() else None
    integration_ready_dependencies = (
        require_list("integration_ready_dependencies")
        if isinstance(frontmatter.get("integration_ready_dependencies"), list)
        else []
    )

    return Task(
        path=path,
        task_id=require_str("task_id"),
        title=require_str("title"),
        workstream=require_str("workstream"),
        task_kind=task_kind,
        role=role,
        priority=priority,
        dependencies=require_list("dependencies"),
        integration_ready_dependencies=integration_ready_dependencies,
        allow_network=_coerce_bool(frontmatter.get("allow_network"), default=False),
        allowed_paths=require_list("allowed_paths"),
        disallowed_paths=require_list("disallowed_paths"),
        outputs=require_list("outputs"),
        gates=require_list("gates"),
        stop_conditions=require_list("stop_conditions"),
        state=state,
        last_updated=last_updated,
    )


def _iter_task_files(contract: FrameworkContract) -> list[Path]:
    orchestrator_dir = Path(".orchestrator")
    paths: list[Path] = []
    for folder_name in contract.projection_dirs:
        folder = orchestrator_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            paths.append(path)
    return paths


def _collect_tasks(contract: FrameworkContract) -> tuple[dict[str, Task], list[str]]:
    tasks: dict[str, Task] = {}
    failures: list[str] = []
    for path in _iter_task_files(contract):
        try:
            task = load_task(path, contract)
        except ValueError as exc:
            failures.append(f"{path}:{exc}")
            continue
        if task.task_id in tasks:
            failures.append(f"duplicate_task_id:{task.task_id}:{tasks[task.task_id].path}:{path}")
            continue
        tasks[task.task_id] = task
    return tasks, failures


def task_is_integration_ready_eligible(task: Task, contract: FrameworkContract) -> bool:
    workstream_eligible = task.workstream in set(contract.integration_ready_eligible_workstreams)
    task_kind_eligible = bool(task.task_kind) and task.task_kind in set(contract.integration_ready_eligible_task_kinds)
    if not (workstream_eligible or task_kind_eligible):
        return False
    if not contract.forbid_unvalidated_empirical_data_outputs:
        return True
    for output in task.outputs:
        for prefix in FORBIDDEN_INTEGRATION_READY_OUTPUT_PREFIXES:
            if _path_matches_prefix(output, prefix):
                return False
    return True


def _downstream_allowlist_exists(task_id: str, tasks: dict[str, Task]) -> bool:
    return any(task_id in task.integration_ready_dependencies for task in tasks.values())


def _section_has_content(text: str, heading: str) -> bool:
    match = re.search(rf"^{re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    if match is None:
        return False
    after = text[match.end() :]
    for line in after.splitlines():
        if line.startswith("## "):
            return False
        if line.strip():
            return True
    return False


def _output_spec_is_safe(spec: str) -> tuple[bool, str | None]:
    norm = _normalize_repo_relative_path(spec)
    if not norm:
        return False, "empty_output_spec"
    if norm.startswith("/") or norm.startswith("~"):
        return False, "absolute_output_spec_forbidden"
    if norm == ".." or norm.startswith("../") or "/../" in norm:
        return False, "path_traversal_forbidden"
    return True, None


_OUTPUT_WILDCARD_TOKENS = ("...", "YYYY-MM-DD", "<", ">", "*", "?")


def _has_wildcards(segment: str) -> bool:
    return any(token in segment for token in _OUTPUT_WILDCARD_TOKENS)


def _segment_pattern_to_regex(segment: str) -> re.Pattern[str]:
    rendered = re.sub(r"<[^>]+>", "{WILD}", segment)
    rendered = rendered.replace("YYYY-MM-DD", "{DATE}")
    rendered = rendered.replace("...", "{ELLIPSIS}")
    regex = re.escape(rendered)
    regex = regex.replace(re.escape("{WILD}"), r"[^/]+")
    regex = regex.replace(re.escape("{DATE}"), r"\d{4}-\d{2}-\d{2}")
    regex = regex.replace(re.escape("{ELLIPSIS}"), r".*")
    regex = regex.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile("^" + regex + "$")


def _find_paths_matching_output_spec(spec: str, repo: Path) -> list[Path]:
    norm = _normalize_repo_relative_path(spec)
    segments = [segment for segment in norm.split("/") if segment]
    current: list[Path] = [repo]
    for segment in segments:
        next_paths: list[Path] = []
        if not _has_wildcards(segment):
            for base in current:
                candidate = base / segment
                if candidate.exists():
                    next_paths.append(candidate)
        else:
            regex = _segment_pattern_to_regex(segment)
            for base in current:
                if not base.is_dir():
                    continue
                try:
                    for child in base.iterdir():
                        if regex.match(child.name):
                            next_paths.append(child)
                except FileNotFoundError:
                    continue
        current = next_paths
        if not current:
            break
    return current


def _guess_output_kind(spec: str) -> str:
    norm = _normalize_repo_relative_path(spec)
    if norm.endswith("/...") or norm.endswith("..."):
        return "dir_nonempty"
    if norm.endswith("/"):
        return "dir"
    for ext in (".py", ".md", ".json", ".csv", ".yaml", ".yml", ".svg", ".pdf", ".txt"):
        if norm.lower().endswith(ext):
            return "file"
    return "any"


def _strip_trailing_ellipsis(spec: str) -> str:
    norm = _normalize_repo_relative_path(spec)
    if norm.endswith("/..."):
        return norm[:-4]
    if norm.endswith("..."):
        return norm[:-3].rstrip("/")
    return norm


def _check_declared_outputs_exist(task: Task, repo: Path = Path(".")) -> tuple[bool, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    for output in task.outputs:
        safe, reason = _output_spec_is_safe(output)
        if not safe:
            failures.append({"output": output, "reason": reason or "invalid_output_spec"})
            continue

        kind = _guess_output_kind(output)
        match_spec = _strip_trailing_ellipsis(output) if kind == "dir_nonempty" else output
        matches = _find_paths_matching_output_spec(match_spec, repo)

        if kind == "file":
            if not any(path.is_file() for path in matches):
                failures.append({"output": output, "reason": "missing_file"})
            continue
        if kind == "dir":
            if not any(path.is_dir() for path in matches):
                failures.append({"output": output, "reason": "missing_dir"})
            continue
        if kind == "dir_nonempty":
            nonempty = False
            for path in matches:
                if not path.is_dir():
                    continue
                try:
                    next(path.iterdir())
                    nonempty = True
                    break
                except (StopIteration, FileNotFoundError):
                    continue
            if not nonempty:
                failures.append({"output": output, "reason": "missing_or_empty_dir"})
            continue
        if not matches:
            failures.append({"output": output, "reason": "missing_path"})

    return len(failures) == 0, failures


def _task_uses_manifest_backed_local_etl_outputs(task: Task) -> bool:
    return task.task_kind == "etl" and task.workstream in {"W1", "W2"}


def _output_is_manifest_backed_local_etl_output(task: Task, output: str) -> bool:
    if not _task_uses_manifest_backed_local_etl_outputs(task):
        return False
    return any(_path_matches_prefix(output, prefix) for prefix in MANIFEST_BACKED_LOCAL_ETL_OUTPUT_PREFIXES)


def _check_review_bundle_outputs_exist(task: Task, repo: Path = Path(".")) -> tuple[bool, list[dict[str, str]]]:
    outputs_ok, output_failures = _check_declared_outputs_exist(task, repo)
    if outputs_ok:
        return True, []
    if not _task_uses_manifest_backed_local_etl_outputs(task):
        return False, output_failures

    retained_failures = [
        failure
        for failure in output_failures
        if not _output_is_manifest_backed_local_etl_output(task, failure["output"])
    ]
    return len(retained_failures) == 0, retained_failures


def _check_repo_materialized_processed_outputs(task: Task, repo: Path = Path(".")) -> list[str]:
    failures: list[str] = []
    for output in task.outputs:
        if not _path_matches_prefix(output, "data/processed/"):
            continue
        safe, _ = _output_spec_is_safe(output)
        if not safe or _guess_output_kind(output) != "file":
            continue
        for path in _find_paths_matching_output_spec(output, repo):
            if not path.is_file():
                continue
            relpath = path.relative_to(repo).as_posix()
            if _git_path_is_ignored(relpath, repo):
                failures.append(f"{relpath}:git_ignored")
                continue
            if not _git_path_is_tracked(relpath, repo):
                failures.append(f"{relpath}:not_tracked")
    return failures


def _task_requires_manifest(task: Task, prefix: str) -> bool:
    return any(_path_matches_prefix(output, prefix) for output in task.outputs)


def required_manifest_failures(task: Task, repo: Path = Path(".")) -> list[str]:
    failures: list[str] = []

    if _task_requires_manifest(task, "data/raw/"):
        raw_manifest_specs = [output for output in task.outputs if _path_matches_prefix(output, "data/raw_manifest/")]
        if not raw_manifest_specs:
            failures.append("missing_declared_raw_manifest_output")
        elif not any(_find_paths_matching_output_spec(spec, repo) for spec in raw_manifest_specs):
            failures.append("missing_raw_manifest_file")

    if _task_requires_manifest(task, "data/processed/"):
        processed_manifest_specs = [output for output in task.outputs if _path_matches_prefix(output, "data/processed_manifest/")]
        if not processed_manifest_specs:
            failures.append("missing_declared_processed_manifest_output")
        elif not any(_find_paths_matching_output_spec(spec, repo) for spec in processed_manifest_specs):
            failures.append("missing_processed_manifest_file")

    return failures


def _load_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(payload, dict):
        return None, "top_level_not_object"
    return payload, None


def _validate_required_keys(data: object, required_keys: set[str], prefix: str) -> list[str]:
    if not isinstance(data, dict):
        return [f"{prefix}:not_object"]
    failures: list[str] = []
    for key in sorted(required_keys):
        if key not in data:
            failures.append(f"{prefix}:missing_key:{key}")
    return failures


def _matching_task_jsons(directory: Path, task_id: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob(f"{task_id}_*.json") if path.is_file())


def _validate_swarm_run_manifest(path: Path, contract: FrameworkContract) -> list[str]:
    payload, error = _load_json_file(path)
    if error is not None or payload is None:
        return [f"{path}:{error}"]

    failures: list[str] = []
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            payload,
            {"schema_version", "run_id", "generated_at_utc", "task", "repo", "executor", "commands", "gates", "ownership", "artifacts", "result"},
            "top",
        )
    )

    schema_version = payload.get("schema_version")
    if schema_version not in {
        SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1,
        SWARM_RUN_MANIFEST_SCHEMA_VERSION,
    }:
        failures.append(f"{path}:invalid_schema_version:{payload.get('schema_version')}")

    if schema_version == SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        provenance_class = payload.get("provenance_class")
        if provenance_class not in {"executor_run", "manual_operator", "backfill"}:
            failures.append(f"{path}:invalid_provenance_class:{provenance_class}")

    task = payload.get("task")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            task,
            {"task_id", "task_path", "role", "workstream", "state_before", "state_after"},
            "task",
        )
    )
    if isinstance(task, dict):
        if task.get("role") not in set(contract.allowed_roles):
            failures.append(f"{path}:invalid_task_role:{task.get('role')}")
        if task.get("state_after") not in set(contract.allowed_states):
            failures.append(f"{path}:invalid_task_state_after:{task.get('state_after')}")

    repo = payload.get("repo")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(repo, {"branch", "git_sha", "base_branch", "remote"}, "repo")
    )

    executor = payload.get("executor")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(executor, {"role", "runner", "tool", "allow_network"}, "executor")
    )
    if isinstance(executor, dict):
        if executor.get("role") not in set(contract.task_execution_roles):
            failures.append(f"{path}:invalid_executor_role:{executor.get('role')}")

    commands = payload.get("commands")
    command_keys = {"executor", "gates"}
    if schema_version == SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        command_keys.update({"executor_log_path", "executor_log_sha256"})
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(commands, command_keys, "commands")
    )

    ownership = payload.get("ownership")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(ownership, {"ok", "changed_paths", "violations"}, "ownership")
    )

    artifacts = payload.get("artifacts")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            artifacts,
            {"outputs_ok", "missing_outputs", "required_manifests_ok", "missing_manifests"},
            "artifacts",
        )
    )

    result = payload.get("result")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(result, {"status", "blocked_reasons"}, "result")
    )
    if isinstance(result, dict):
        if result.get("status") not in {"ok", "blocked"}:
            failures.append(f"{path}:invalid_result_status:{result.get('status')}")

    return failures


def _validate_judge_review_log(path: Path, contract: FrameworkContract) -> list[str]:
    payload, error = _load_json_file(path)
    if error is not None or payload is None:
        return [f"{path}:{error}"]

    failures: list[str] = []
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            payload,
            {"schema_version", "review_id", "generated_at_utc", "reviewer", "task", "checks", "decision"},
            "top",
        )
    )

    if payload.get("schema_version") != JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        failures.append(f"{path}:invalid_schema_version:{payload.get('schema_version')}")

    reviewer = payload.get("reviewer")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(reviewer, {"role"}, "reviewer")
    )
    if isinstance(reviewer, dict):
        if reviewer.get("role") != contract.scientific_review_role:
            failures.append(f"{path}:invalid_reviewer_role:{reviewer.get('role')}")

    task = payload.get("task")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            task,
            {"task_id", "task_path", "role", "state_before", "state_after", "run_manifest_path"},
            "task",
        )
    )
    if isinstance(task, dict):
        if task.get("state_after") not in set(contract.allowed_states):
            failures.append(f"{path}:invalid_task_state_after:{task.get('state_after')}")

    checks = payload.get("checks")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(
            checks,
            {"gates_ok", "outputs_ok", "required_manifests_ok", "review_bundle_ok", "failures"},
            "checks",
        )
    )

    decision = payload.get("decision")
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(decision, {"outcome", "note"}, "decision")
    )
    if isinstance(decision, dict):
        if decision.get("outcome") not in {"approve", "revise", "block"}:
            failures.append(f"{path}:invalid_decision_outcome:{decision.get('outcome')}")
        if decision.get("outcome") == "approve" and isinstance(task, dict) and task.get("state_after") != "done":
            failures.append(f"{path}:approve_without_done")

    return failures


def gate_framework_contract() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    failures: list[str] = []

    for role in DEFAULT_ALLOWED_ROLES:
        if role not in set(contract.allowed_roles):
            failures.append(f"missing_role:{role}")

    for role in DEFAULT_TASK_EXECUTION_ROLES:
        if role not in set(contract.task_execution_roles):
            failures.append(f"missing_task_execution_role:{role}")

    for state in DEFAULT_ALLOWED_STATES:
        if state not in set(contract.allowed_states):
            failures.append(f"missing_state:{state}")

    for prompt_key in ("planner", "worker", "judge", "operator"):
        if prompt_key not in contract.prompt_templates:
            failures.append(f"missing_prompt_template:{prompt_key}")

    for workstream in DEFAULT_NETWORK_WORKSTREAMS:
        if workstream not in set(contract.network_workstreams):
            failures.append(f"missing_network_workstream:{workstream}")

    if contract.scientific_review_role != DEFAULT_SCIENTIFIC_REVIEW_ROLE:
        failures.append(f"invalid_scientific_review_role:{contract.scientific_review_role}")

    if contract.run_manifest_dir != "reports/status/swarm_runs":
        failures.append(f"invalid_run_manifest_dir:{contract.run_manifest_dir}")

    if contract.judge_review_dir != "reports/status/reviews":
        failures.append(f"invalid_judge_review_dir:{contract.judge_review_dir}")

    if contract.release_manifest_pattern != "reports/status/releases/release_<YYYY-MM-DD>.json":
        failures.append(f"invalid_release_manifest_pattern:{contract.release_manifest_pattern}")

    return GateResult(
        ok=len(failures) == 0,
        details={
            "project_mode": contract.project_mode,
            "failures": failures,
        },
    )


def gate_repo_structure() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    registry_enabled = _coerce_bool(contract.features.get("registry"), default=(contract.project_mode != "modeling"))
    required_paths = [
        "AGENTS.md",
        "README.md",
        ".orchestrator/README.md",
        ".orchestrator/AGENTS.md",
        ".orchestrator/workstreams.md",
        "contracts/project.yaml",
        "contracts/framework.json",
        "contracts/README.md",
        "contracts/data_dictionary.md",
        "contracts/decisions.md",
        "contracts/model_spec.md",
        "contracts/hybrid_interface_v1.yaml",
        "contracts/schemas/README.md",
        "contracts/schemas/panel_schema.yaml",
        "contracts/schemas/panel_schema_str_v1.yaml",
        "contracts/schemas/panel_schema_decomp_v1.yaml",
        "contracts/schemas/swarm_run_manifest_v1.yaml",
        "contracts/schemas/judge_review_log_v1.yaml",
        "docs/protocol.md",
        "docs/runbook_swarm.md",
        "docs/runbook_swarm_automation.md",
        "data/raw_manifest/README.md",
        "data/processed_manifest/README.md",
        "data/samples/README.md",
        "reports/validation/README.md",
        "reports/validation/manifests/README.md",
        "reports/figures/README.md",
        "reports/tables/README.md",
        "scripts/swarm.py",
        "scripts/sweep_tasks.py",
        "scripts/quality_gates.py",
        "tests/README.md",
    ]

    required_paths.extend(contract.prompt_templates.values())

    if registry_enabled:
        required_paths.extend(
            [
                "registry/README.md",
                "registry/CHANGELOG.md",
                "registry/rollup_registry_v1.csv",
            ]
        )

    deferred_paths: list[str] = []
    for raw_path in contract.required_paths:
        if any(_path_matches_prefix(raw_path, prefix) for prefix in DEFERRED_REQUIRED_PATH_PREFIXES):
            deferred_paths.append(_normalize_repo_relative_path(raw_path))
        else:
            required_paths.append(raw_path)

    deduped_required = []
    seen: set[str] = set()
    for raw_path in required_paths:
        norm = _normalize_repo_relative_path(raw_path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped_required.append(norm)

    missing = [path for path in deduped_required if not Path(path).exists()]

    return GateResult(
        ok=len(missing) == 0,
        details={
            "project_mode": contract.project_mode,
            "registry_enabled": registry_enabled,
            "missing": missing,
            "deferred_paths": sorted(set(deferred_paths)),
        },
    )


def gate_project_contract() -> GateResult:
    path = Path("contracts/project.yaml")
    if not path.exists():
        return GateResult(ok=False, details={"failures": [f"missing_project_contract:{path}"]})
    mode = _parse_project_mode(path)
    if mode is None:
        return GateResult(ok=False, details={"failures": ["missing_mode"]})
    if mode not in {"empirical", "modeling", "hybrid"}:
        return GateResult(ok=False, details={"failures": [f"invalid_mode:{mode}"]})
    return GateResult(ok=True, details={"mode": mode})


def gate_protocol_complete() -> GateResult:
    mode = _parse_project_mode(Path("contracts/project.yaml"))
    if mode == "modeling":
        return GateResult(ok=True, details={"skipped": True, "mode": mode})

    protocol_path = Path("docs/protocol.md")
    if not protocol_path.exists():
        return GateResult(ok=False, details={"failures": [f"missing_protocol:{protocol_path}"]})

    text = _read_text(protocol_path)
    failures: list[str] = []

    mode_match = re.search(r"^\s*-\s*Mode:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    if mode_match is None:
        failures.append("missing_mode_line")
    elif mode is not None and mode_match.group(1).strip().lower() != mode:
        failures.append(f"mode_mismatch:{mode_match.group(1).strip().lower()}!={mode}")

    for heading in (
        "## Rollup inclusion criteria",
        "## Data source priority",
        "## Known regime dates",
        "## Validation tolerances",
    ):
        if heading not in text or not _section_has_content(text, heading):
            failures.append(f"missing_or_empty_section:{heading}")

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_workstreams_complete() -> GateResult:
    path = Path(".orchestrator/workstreams.md")
    if not path.exists():
        return GateResult(ok=False, details={"failures": [f"missing_workstreams:{path}"]})

    failures: list[str] = []
    rows_seen = 0
    for line in _read_text(path).splitlines():
        if not re.match(r"^\|\s*W\d+\s+", line):
            continue
        rows_seen += 1
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            failures.append(f"malformed_row:{line.strip()}")
            continue
        if not cells[1]:
            failures.append(f"blank_purpose:{cells[0]}")
        if not cells[2]:
            failures.append(f"blank_owns_paths:{cells[0]}")
        if not cells[3]:
            failures.append(f"blank_does_not_own:{cells[0]}")
    if rows_seen == 0:
        failures.append("no_workstream_rows_found")
    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_task_hygiene() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    failures: list[str] = []
    task_paths = _iter_task_files(contract)
    if not task_paths:
        failures.append("no_task_files_found")

    for path in task_paths:
        text = _read_text(path)
        frontmatter = _parse_task_frontmatter(text)
        if frontmatter is None:
            failures.append(f"{path}:missing_yaml_frontmatter")
            continue

        for key in REQUIRED_FRONTMATTER_KEYS:
            if key not in frontmatter:
                failures.append(f"{path}:frontmatter_missing_key:{key}")

        for key in ("dependencies", "allowed_paths", "disallowed_paths", "outputs", "gates", "stop_conditions"):
            value = frontmatter.get(key)
            if not isinstance(value, list):
                failures.append(f"{path}:frontmatter_invalid_list:{key}")

        if "integration_ready_dependencies" in frontmatter and not isinstance(frontmatter.get("integration_ready_dependencies"), list):
            failures.append(f"{path}:frontmatter_invalid_list:integration_ready_dependencies")

        task_id = frontmatter.get("task_id")
        if isinstance(task_id, str) and not path.name.startswith(task_id):
            failures.append(f"{path}:task_id_filename_mismatch:{task_id}")

        role = frontmatter.get("role")
        if isinstance(role, str) and role not in set(contract.allowed_roles):
            failures.append(f"{path}:invalid_role:{role}")

        priority = frontmatter.get("priority")
        if isinstance(priority, str) and priority not in VALID_TASK_PRIORITIES:
            failures.append(f"{path}:invalid_priority:{priority}")

        allow_network = _coerce_bool(frontmatter.get("allow_network"), default=False)
        workstream = frontmatter.get("workstream")
        if allow_network and isinstance(workstream, str) and workstream not in set(contract.network_workstreams):
            failures.append(f"{path}:network_workstream_not_allowlisted:{workstream}")

        outputs = _coerce_str_list(frontmatter.get("outputs"))
        if any(_path_matches_prefix(output, "data/raw/") for output in outputs) and not any(
            _path_matches_prefix(output, "data/raw_manifest/") for output in outputs
        ):
            failures.append(f"{path}:raw_outputs_missing_manifest_output")
        if any(_path_matches_prefix(output, "data/processed/") for output in outputs) and not any(
            _path_matches_prefix(output, "data/processed_manifest/") for output in outputs
        ):
            failures.append(f"{path}:processed_outputs_missing_manifest_output")

        for heading in REQUIRED_TASK_HEADINGS:
            if heading not in text:
                failures.append(f"{path}:missing_heading:{heading}")

        state = _parse_status_value(text, "State")
        if state is None:
            failures.append(f"{path}:missing_state")
        elif state not in set(contract.allowed_states):
            failures.append(f"{path}:invalid_state:{state}")

        last_updated = _parse_status_value(text, "Last updated")
        if last_updated is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_updated):
            failures.append(f"{path}:invalid_last_updated:{last_updated}")

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_task_dependencies() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    tasks, parse_failures = _collect_tasks(contract)
    failures = list(parse_failures)

    for task in tasks.values():
        for dep in task.dependencies:
            if not re.fullmatch(r"T\d{3}", dep):
                failures.append(f"{task.path}:invalid_dependency_id:{dep}")
            elif dep == task.task_id:
                failures.append(f"{task.path}:self_dependency:{dep}")
            elif dep not in tasks:
                failures.append(f"{task.path}:missing_dependency:{dep}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(task_id: str, stack: list[str]) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            if task_id in stack:
                cycle = stack[stack.index(task_id) :] + [task_id]
                failures.append(f"dependency_cycle:{'->'.join(cycle)}")
            return
        visiting.add(task_id)
        stack.append(task_id)
        for dep in tasks[task_id].dependencies:
            if dep in tasks:
                dfs(dep, stack)
        stack.pop()
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in sorted(tasks.keys()):
        dfs(task_id, [])

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_integration_ready_policy() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    tasks, parse_failures = _collect_tasks(contract)
    failures = list(parse_failures)

    for task in tasks.values():
        for dep in task.integration_ready_dependencies:
            if dep not in task.dependencies:
                failures.append(f"{task.path}:integration_ready_dependency_not_in_dependencies:{dep}")
            if dep not in tasks:
                failures.append(f"{task.path}:integration_ready_dependency_missing_task:{dep}")

        if task.state != "integration_ready":
            continue

        if not task_is_integration_ready_eligible(task, contract):
            failures.append(f"{task.path}:integration_ready_ineligible")

        if not _downstream_allowlist_exists(task.task_id, tasks):
            failures.append(f"{task.path}:integration_ready_missing_downstream_allowlist")

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_operator_surface_ownership() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    tasks, parse_failures = _collect_tasks(contract)
    failures = list(parse_failures)

    for task in tasks.values():
        if task.allow_network and task.workstream not in set(contract.network_workstreams):
            failures.append(f"{task.path}:network_workstream_not_allowlisted:{task.workstream}")

        if task.role == "Operator":
            if task.workstream != "W9" and task.task_kind != "ops":
                failures.append(f"{task.path}:operator_role_outside_ops_boundary")
            continue

        for surface in contract.operator_owned_shared_surfaces:
            for raw_path in [*task.allowed_paths, *task.outputs]:
                if _path_matches_prefix(raw_path, surface):
                    failures.append(f"{task.path}:operator_owned_surface:{surface}:{raw_path}")

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_raw_manifest_validity() -> GateResult:
    manifest_dir = Path("data/raw_manifest")
    if not manifest_dir.exists():
        return GateResult(ok=False, details={"failures": [f"missing_dir:{manifest_dir}"]})

    failures: list[str] = []
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    for path in manifest_paths:
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            failures.append(f"{path}:{error}")
            continue

        failures.extend(
            f"{path}:{failure}"
            for failure in _validate_required_keys(payload, {"source", "fetched_at_utc", "command", "files"}, "top")
        )

        files = payload.get("files")
        if not isinstance(files, list):
            failures.append(f"{path}:files_not_list")
            continue

        for index, item in enumerate(files):
            failures.extend(
                f"{path}:{failure}"
                for failure in _validate_required_keys(item, {"path", "sha256", "bytes"}, f"files[{index}]")
            )
            if isinstance(item, dict):
                sha = item.get("sha256")
                if isinstance(sha, str) and not re.fullmatch(r"[0-9a-f]{64}", sha):
                    failures.append(f"{path}:files[{index}]:invalid_sha256")

    return GateResult(ok=len(failures) == 0, details={"count": len(manifest_paths), "failures": failures})


def gate_processed_manifest_validity() -> GateResult:
    manifest_dir = Path("data/processed_manifest")
    if not manifest_dir.exists():
        return GateResult(ok=False, details={"failures": [f"missing_dir:{manifest_dir}"]})

    failures: list[str] = []
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    for path in manifest_paths:
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            failures.append(f"{path}:{error}")
            continue

        failures.extend(
            f"{path}:{failure}"
            for failure in _validate_required_keys(payload, {"as_of_utc_date", "inputs", "transform", "outputs"}, "top")
        )

        transform = payload.get("transform")
        failures.extend(
            f"{path}:{failure}"
            for failure in _validate_required_keys(transform, {"script_path", "git_sha", "command"}, "transform")
        )

        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            failures.append(f"{path}:outputs_not_list")
            continue

        for index, item in enumerate(outputs):
            failures.extend(
                f"{path}:{failure}"
                for failure in _validate_required_keys(item, {"path", "sha256", "bytes"}, f"outputs[{index}]")
            )
            if isinstance(item, dict):
                sha = item.get("sha256")
                if isinstance(sha, str) and not re.fullmatch(r"[0-9a-f]{64}", sha):
                    failures.append(f"{path}:outputs[{index}]:invalid_sha256")

    return GateResult(ok=len(failures) == 0, details={"count": len(manifest_paths), "failures": failures})


def gate_swarm_run_manifest_validity() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    run_dir = Path(contract.run_manifest_dir)
    if not run_dir.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "run_manifest_dir_missing"})

    failures: list[str] = []
    manifest_paths = sorted(run_dir.glob("*.json"))
    for path in manifest_paths:
        failures.extend(_validate_swarm_run_manifest(path, contract))

    return GateResult(ok=len(failures) == 0, details={"count": len(manifest_paths), "failures": failures})


def gate_judge_review_log_validity() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    review_dir = Path(contract.judge_review_dir)
    if not review_dir.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "judge_review_dir_missing"})

    failures: list[str] = []
    review_paths = sorted(review_dir.glob("*.json"))
    for path in review_paths:
        failures.extend(_validate_judge_review_log(path, contract))

    return GateResult(ok=len(failures) == 0, details={"count": len(review_paths), "failures": failures})


def gate_review_bundle_integrity() -> GateResult:
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    tasks, parse_failures = _collect_tasks(contract)
    failures = list(parse_failures)
    repo = Path(".")

    for task in tasks.values():
        if task.state not in {"integration_ready", "ready_for_review", "done"}:
            continue

        outputs_ok, output_failures = _check_review_bundle_outputs_exist(task, repo)
        if not outputs_ok:
            failures.append(
                f"{task.path}:missing_outputs:"
                + ";".join(f"{item['output']}={item['reason']}" for item in output_failures)
            )

        for reason in _check_repo_materialized_processed_outputs(task, repo):
            failures.append(f"{task.path}:repo_materialization_failure:{reason}")

        for reason in required_manifest_failures(task, repo):
            failures.append(f"{task.path}:required_manifest_failure:{reason}")

        run_dir = Path(contract.run_manifest_dir)
        matching_run_manifests = _matching_task_jsons(run_dir, task.task_id)
        valid_run_manifests = [
            path for path in matching_run_manifests if not _validate_swarm_run_manifest(path, contract)
        ]
        if not valid_run_manifests:
            failures.append(
                f"{task.path}:"
                + ("invalid_run_manifest" if matching_run_manifests else "missing_run_manifest")
            )

        if task.state == "done":
            review_dir = Path(contract.judge_review_dir)
            matching_review_logs = _matching_task_jsons(review_dir, task.task_id)
            valid_review_logs = [
                path for path in matching_review_logs if not _validate_judge_review_log(path, contract)
            ]
            if not valid_review_logs:
                failures.append(
                    f"{task.path}:"
                    + ("invalid_review_log" if matching_review_logs else "missing_review_log")
                )

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def _collect_gate_results() -> dict[str, GateResult]:
    return {
        "framework_contract": gate_framework_contract(),
        "repo_structure": gate_repo_structure(),
        "project_contract": gate_project_contract(),
        "protocol_complete": gate_protocol_complete(),
        "workstreams_complete": gate_workstreams_complete(),
        "task_hygiene": gate_task_hygiene(),
        "task_dependencies": gate_task_dependencies(),
        "integration_ready_policy": gate_integration_ready_policy(),
        "operator_surface_ownership": gate_operator_surface_ownership(),
        "raw_manifest_validity": gate_raw_manifest_validity(),
        "processed_manifest_validity": gate_processed_manifest_validity(),
        "swarm_run_manifest_validity": gate_swarm_run_manifest_validity(),
        "judge_review_log_validity": gate_judge_review_log_validity(),
        "review_bundle_integrity": gate_review_bundle_integrity(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quality_gates.py")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args(argv)

    results = _collect_gate_results()
    overall_ok = all(result.ok for result in results.values())

    if args.json:
        payload = {
            "ok": overall_ok,
            "results": {
                name: {"ok": result.ok, "details": result.details}
                for name, result in results.items()
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, result in results.items():
            print(f"[{name}] ok={result.ok} details={result.details}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
