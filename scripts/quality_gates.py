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
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import shlex
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from swarm_taskfile import NETWORK_COMMAND_TOKENS
from swarm_taskfile import PREREG_PHASE_FILES
from swarm_taskfile import REQUIRED_FRONTMATTER_KEYS
from swarm_taskfile import TASK_SCHEMA_VERSION
from swarm_taskfile import gate_command_violation
from swarm_taskfile import lint_task_files
from swarm_taskfile import load_prereg_lock
from swarm_taskfile import parse_status_value as _parse_status_value
from swarm_taskfile import parse_task_frontmatter as _parse_task_frontmatter
from sweep_tasks import plan_sweep as _plan_sweep
from falsify_claims import evaluate_falsification_spec
from sweep_harness import enumerate_cells
from swarm_events import read_events as _read_swarm_events
from calibrate_referee import calibration_report_failures


SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v2"
SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 = "research_swarm.runtime_run_manifest.v1"
JUDGE_REVIEW_LOG_SCHEMA_VERSION = "research_swarm.judge_review_log.v2"
JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1 = "research_swarm.judge_review_log.v1"
PROCESSED_MANIFEST_SCHEMA_VERSION = "research_swarm.processed_manifest.v2"
MANIFEST_REBASELINE_SCHEMA_VERSION = "research_swarm.manifest_rebaseline.v1"
VALIDATION_REPORT_SCHEMA_VERSION = "research_swarm.validation_report.v2"
INSTANCE_MANIFEST_SCHEMA_VERSION = "research_swarm.instance_manifest.v1"
EXPERIMENT_SPEC_SCHEMA_VERSION = "research_swarm.experiment_spec.v1"
EXPERIMENT_MANIFEST_SCHEMA_VERSION = "research_swarm.experiment_manifest.v1"
SWEEP_ARTIFACT_SCHEMA_VERSION = "research_swarm.sweep_artifact.v1"
REFEREE_REPORT_SCHEMA_VERSION = "research_swarm.referee_report.v1"
REFEREE_CALIBRATION_SCHEMA_VERSION = "research_swarm.referee_calibration.v1"
REFEREE_RUBRIC_SCHEMA_VERSION = "research_swarm.rubric.v1"
REFEREE_GOLD_KEY_SCHEMA_VERSION = "research_swarm.referee_gold_key.v1"
REFEREE_VERDICTS = {"supported", "not_supported", "cannot_verify"}
REFEREE_REPORT_DIR = Path("reports/status/referee_reports")
REFEREE_CALIBRATION_REPORT = Path("reports/status/referee_calibration.json")
REFEREE_WAIVER_EMITTER = "swarm.py referee-waiver"
REFEREE_RUBRIC_TASK_KINDS = {
    "etl": "etl",
    "analysis": "analysis",
    "writing": "writing",
    "validation": "validation",
    "proof": "proof_review",
    "model": "model",
    "bridge": "bridge",
    "lit_review": "lit_review",
    "manuscript": "manuscript",
}

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
    "reports/status/swarm_runs/",
    "reports/status/referee_reports/",
    "reports/status/referee_calibration.json",
    "reports/status/referee_calibration_runs/",
    "reports/status/events/",
)
FORBIDDEN_INTEGRATION_READY_OUTPUT_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "reports/validation/",
    "reports/figures/",
    "reports/tables/",
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
CLAIMS_SCHEMA_VERSION = "research_swarm.claims.v1"
CITATION_SNAPSHOT_SCHEMA_VERSION = "research_swarm.citation_snapshot.v1"
CLAIM_TYPES = {
    "descriptive",
    "associational",
    "causal",
    "interpretation",
    "methodological",
    "theoretical",
    "computational",
    "counterfactual",
    "literature",
}
CONFIRMATORY_CLAIM_TYPES = {"causal", "computational", "counterfactual"}
UNCERTAINTY_ARTIFACT_REQUIRED_TYPES = {
    "descriptive",
    "associational",
    "causal",
    "computational",
    "counterfactual",
}
UNCERTAINTY_JUSTIFICATION_TYPES = {
    "theoretical",
    "interpretation",
    "methodological",
    "literature",
}
TERMINAL_HYPOTHESIS_OUTCOMES = {
    "supported",
    "not_supported",
    "inconclusive",
    "abandoned",
}
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


def _trusted_integration_branch(repo: Path) -> str:
    """The repository's real default branch from git (never a caller argument),
    used to validate that a control-plane waiver was emitted on the integration
    branch rather than a Worker task branch."""
    cp = subprocess.run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    value = (cp.stdout or "").strip() if cp.returncode == 0 else ""
    if value.startswith("origin/"):
        value = value[len("origin/"):]
    return value or "main"


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
    except OSError as exc:
        return None, f"read_error:{type(exc).__name__}:{exc}"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(payload, dict):
        return None, "top_level_not_object"
    return payload, None


def _json_type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _json_pointer(document: object, pointer: str) -> object:
    if pointer in {"", "#"}:
        return document
    raw = pointer[1:] if pointer.startswith("#") else pointer
    if not raw.startswith("/"):
        raise ValueError(f"invalid_json_pointer:{pointer}")
    current = document
    for token in raw[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"unresolved_json_pointer:{pointer}")
        current = current[key]
    return current


def _load_schema_document(path: Path) -> dict[str, Any]:
    payload, error = _load_json_file(path)
    if error is not None or payload is None:
        raise ValueError(f"schema_load_error:{path.as_posix()}:{error}")
    return payload


def _validate_json_schema(
    value: object,
    schema: object,
    *,
    schema_path: Path,
    document: dict[str, Any] | None = None,
    value_path: str = "$",
) -> list[dict[str, object]]:
    """Validate the dependency-free JSON-Schema subset used by M3a contracts."""
    if not isinstance(schema, dict):
        return [{"path": value_path, "reason": "schema_not_object"}]
    if document is None:
        document = schema
    reference = schema.get("$ref")
    if isinstance(reference, str):
        file_part, separator, fragment = reference.partition("#")
        try:
            if file_part:
                target_path = (schema_path.parent / file_part).resolve()
                target_document = _load_schema_document(target_path)
                target_schema = _json_pointer(target_document, f"#{fragment}" if separator else "#")
                return _validate_json_schema(
                    value,
                    target_schema,
                    schema_path=target_path,
                    document=target_document,
                    value_path=value_path,
                )
            target_schema = _json_pointer(document, f"#{fragment}" if separator else reference)
            return _validate_json_schema(
                value,
                target_schema,
                schema_path=schema_path,
                document=document,
                value_path=value_path,
            )
        except ValueError as exc:
            return [{"path": value_path, "reason": str(exc)}]

    failures: list[dict[str, object]] = []
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for child in all_of:
            failures.extend(
                _validate_json_schema(
                    value,
                    child,
                    schema_path=schema_path,
                    document=document,
                    value_path=value_path,
                )
            )
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        variant_failures = [
            _validate_json_schema(
                value,
                child,
                schema_path=schema_path,
                document=document,
                value_path=value_path,
            )
            for child in variants
        ]
        passing = sum(not item for item in variant_failures)
        valid = passing == 1 if keyword == "oneOf" else passing >= 1
        if not valid:
            best = min(variant_failures, key=len, default=[])
            failures.append(
                {
                    "path": value_path,
                    "reason": f"{keyword}_mismatch",
                    "passing_variants": passing,
                    "best_variant_failures": best,
                }
            )
        return failures

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        types = [expected_type]
    elif isinstance(expected_type, list):
        types = [item for item in expected_type if isinstance(item, str)]
    else:
        types = []
    if types and not any(_json_type_matches(value, item) for item in types):
        failures.append(
            {"path": value_path, "reason": "type", "expected": types, "actual": type(value).__name__}
        )
        return failures
    if "const" in schema and value != schema["const"]:
        failures.append(
            {"path": value_path, "reason": "const", "expected": schema["const"], "actual": value}
        )
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        failures.append({"path": value_path, "reason": "enum", "expected": enum, "actual": value})

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    failures.append({"path": f"{value_path}.{key}", "reason": "required"})
        min_properties = schema.get("minProperties")
        if isinstance(min_properties, int) and len(value) < min_properties:
            failures.append(
                {"path": value_path, "reason": "minProperties", "expected": min_properties, "actual": len(value)}
            )
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        additional = schema.get("additionalProperties", True)
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                if additional is False:
                    failures.append({"path": f"{value_path}.{key}", "reason": "additionalProperty"})
                    continue
                if isinstance(additional, dict):
                    child_schema = additional
            if isinstance(child_schema, dict):
                failures.extend(
                    _validate_json_schema(
                        child_value,
                        child_schema,
                        schema_path=schema_path,
                        document=document,
                        value_path=f"{value_path}.{key}",
                    )
                )

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            failures.append(
                {"path": value_path, "reason": "minItems", "expected": min_items, "actual": len(value)}
            )
        if schema.get("uniqueItems") is True:
            rendered = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            if len(rendered) != len(set(rendered)):
                failures.append({"path": value_path, "reason": "uniqueItems"})
        items = schema.get("items")
        if isinstance(items, dict):
            for index, child_value in enumerate(value):
                failures.extend(
                    _validate_json_schema(
                        child_value,
                        items,
                        schema_path=schema_path,
                        document=document,
                        value_path=f"{value_path}[{index}]",
                    )
                )

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            failures.append({"path": value_path, "reason": "minLength", "expected": min_length})
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            failures.append({"path": value_path, "reason": "pattern", "expected": pattern, "actual": value})
        if schema.get("format") == "date-time" and _parse_utc_z(value) is None:
            failures.append({"path": value_path, "reason": "format", "expected": "UTC date-time", "actual": value})

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            failures.append({"path": value_path, "reason": "minimum", "expected": minimum, "actual": value})
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            failures.append(
                {"path": value_path, "reason": "exclusiveMinimum", "expected": exclusive_minimum, "actual": value}
            )
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            failures.append({"path": value_path, "reason": "maximum", "expected": maximum, "actual": value})
    return failures


def _schema_failures(payload: object, schema_path: Path) -> list[dict[str, object]]:
    try:
        schema = _load_schema_document(schema_path)
    except ValueError as exc:
        return [{"path": "$", "reason": str(exc)}]
    return _validate_json_schema(payload, schema, schema_path=schema_path)


def _safe_repo_relative_path(raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    normalized = _normalize_repo_relative_path(raw_path)
    parts = normalized.split("/")
    if Path(normalized).is_absolute() or normalized.startswith("~") or ".." in parts:
        return None
    return Path(normalized)


def _validate_required_keys(data: object, required_keys: set[str], prefix: str) -> list[str]:
    if not isinstance(data, dict):
        return [f"{prefix}:not_object"]
    failures: list[str] = []
    for key in sorted(required_keys):
        if key not in data:
            failures.append(f"{prefix}:missing_key:{key}")
    return failures


HISTORICAL_EXEMPTIONS_PATH = Path("contracts/historical_exemptions.json")


def _historical_exemption_entries(section: str) -> dict[str, dict[str, object]]:
    """Path → entry map for one section of the hash-pinned historical
    exemption list. Empty when the list (or section) is absent — absence
    exempts nothing."""
    if not HISTORICAL_EXEMPTIONS_PATH.exists():
        return {}
    payload, error = _load_json_file(HISTORICAL_EXEMPTIONS_PATH)
    if error is not None or not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for item in payload.get(section, []):
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            out[item["path"]] = item
    return out


def _historical_exemption_hashes(section: str) -> dict[str, str]:
    return {
        path: entry["sha256"]
        for path, entry in _historical_exemption_entries(section).items()
        if isinstance(entry.get("sha256"), str)
    }


def _sha256_and_bytes(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hash_claim_failure(
    *,
    manifest: Path,
    path: object,
    reason: str,
    expected: object,
    actual: object,
) -> dict[str, object]:
    return {
        "manifest": manifest.as_posix(),
        "path": path,
        "reason": reason,
        "expected": expected,
        "actual": actual,
    }


def _verify_hash_claim(
    *,
    manifest: Path,
    entry: object,
    mismatch_reason: str | None = None,
) -> list[dict[str, object]]:
    if not isinstance(entry, dict):
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=None,
                reason="invalid_manifest",
                expected="entry object with path, sha256, and bytes",
                actual=entry,
            )
        ]

    rel_path = entry.get("path")
    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if (
        not isinstance(rel_path, str)
        or not rel_path
        or not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="invalid_manifest",
                expected="path string, sha256 hex digest, and non-negative integer bytes",
                actual=entry,
            )
        ]

    safe_path = _safe_repo_relative_path(rel_path)
    repo = Path.cwd().resolve()
    if safe_path is None:
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="content_binding_path_outside_repo",
                expected="safe repo-relative tracked regular file",
                actual=rel_path,
            )
        ]
    disk_path = repo / safe_path
    expected = {"sha256": expected_sha, "bytes": expected_bytes}
    try:
        resolved = disk_path.resolve(strict=True)
        resolved.relative_to(repo)
    except (FileNotFoundError, OSError, ValueError):
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason=mismatch_reason or "missing_file",
                expected=expected,
                actual=None,
            )
        ]
    if disk_path.is_symlink() or not resolved.is_file():
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="content_binding_target_not_regular_file",
                expected="non-symlink regular file",
                actual=rel_path,
            )
        ]
    if not _git_path_is_tracked(safe_path.as_posix(), repo):
        return [
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="content_binding_target_not_git_tracked",
                expected="git-tracked regular file",
                actual=rel_path,
            )
        ]

    actual_sha, actual_bytes = _sha256_and_bytes(resolved)
    actual = {"sha256": actual_sha, "bytes": actual_bytes}
    if mismatch_reason is not None:
        if actual != expected:
            return [
                _hash_claim_failure(
                    manifest=manifest,
                    path=rel_path,
                    reason=mismatch_reason,
                    expected=expected,
                    actual=actual,
                )
            ]
        return []

    failures: list[dict[str, object]] = []
    if actual_sha != expected_sha:
        failures.append(
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="sha256_mismatch",
                expected=expected_sha,
                actual=actual_sha,
            )
        )
    if actual_bytes != expected_bytes:
        failures.append(
            _hash_claim_failure(
                manifest=manifest,
                path=rel_path,
                reason="bytes_mismatch",
                expected=expected_bytes,
                actual=actual_bytes,
            )
        )
    return failures


def _invalid_rebaseline_failure(
    *,
    manifest: Path,
    sidecar: Path,
    expected: object,
    actual: object,
) -> dict[str, object]:
    return _hash_claim_failure(
        manifest=manifest,
        path=sidecar.as_posix(),
        reason="invalid_rebaseline",
        expected=expected,
        actual=actual,
    )


def _manifest_hash_gate(
    *,
    manifest_dir: Path,
    entries_key: str,
    allow_raw_evidence_unavailable: bool,
) -> GateResult:
    if not manifest_dir.exists():
        return GateResult(ok=False, details={"failures": [f"missing_dir:{manifest_dir}"]})

    failures: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    checked_entries = 0
    manifest_paths = sorted(manifest_dir.glob("*.json"))

    for manifest_path in manifest_paths:
        payload, error = _load_json_file(manifest_path)
        if error is not None or payload is None:
            failures.append(
                _hash_claim_failure(
                    manifest=manifest_path,
                    path=manifest_path.as_posix(),
                    reason="invalid_manifest",
                    expected="JSON object",
                    actual=error,
                )
            )
            continue

        entries = payload.get(entries_key)
        if not isinstance(entries, list):
            failures.append(
                _hash_claim_failure(
                    manifest=manifest_path,
                    path=manifest_path.as_posix(),
                    reason="invalid_manifest",
                    expected=f"{entries_key} list",
                    actual=entries,
                )
            )
            continue

        checked_entries += len(entries)
        sidecar_path = manifest_dir / "rebaselines" / f"{manifest_path.name}.rebaseline.json"
        if not sidecar_path.exists():
            for entry in entries:
                failures.extend(_verify_hash_claim(manifest=manifest_path, entry=entry))
            continue

        # A rebaseline is a one-time historical remediation, not a general
        # mechanism: the sidecar is honored only when both the manifest and
        # the sidecar itself are hash-pinned on the exemption list. Anything
        # else gets direct verification — a new manifest cannot green itself
        # by shipping its own sidecar.
        exempted_sidecars = _historical_exemption_hashes("rebaselines")
        sidecar_rel = sidecar_path.as_posix()
        pinned_sidecar_sha = exempted_sidecars.get(sidecar_rel)
        if pinned_sidecar_sha is None:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected="sidecar hash-pinned in contracts/historical_exemptions.json",
                    actual="rebaseline_not_exempted",
                )
            )
            for entry in entries:
                failures.extend(_verify_hash_claim(manifest=manifest_path, entry=entry))
            continue
        actual_sidecar_sha, _ = _sha256_and_bytes(sidecar_path)
        if actual_sidecar_sha != pinned_sidecar_sha:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected=pinned_sidecar_sha,
                    actual=f"rebaseline_sidecar_drift:{actual_sidecar_sha}",
                )
            )
            continue

        sidecar, sidecar_error = _load_json_file(sidecar_path)
        if sidecar_error is not None or sidecar is None:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected="valid rebaseline JSON object",
                    actual=sidecar_error,
                )
            )
            continue

        required = {
            "schema_version",
            "rebaseline_of",
            "original_manifest_sha256",
            "mode",
            "provenance_note",
            "rebaselined_at_utc",
        }
        missing = sorted(required - set(sidecar))
        expected_manifest_path = manifest_path.as_posix()
        provenance_note = sidecar.get("provenance_note")
        rebaselined_at = sidecar.get("rebaselined_at_utc")
        common_errors: list[str] = []
        if missing:
            common_errors.append(f"missing_keys:{','.join(missing)}")
        if sidecar.get("schema_version") != MANIFEST_REBASELINE_SCHEMA_VERSION:
            common_errors.append("invalid_schema_version")
        if sidecar.get("rebaseline_of") != expected_manifest_path:
            common_errors.append("rebaseline_of_mismatch")
        if not isinstance(provenance_note, str) or not provenance_note.strip():
            common_errors.append("empty_provenance_note")
        if not isinstance(rebaselined_at, str) or not rebaselined_at.strip():
            common_errors.append("invalid_rebaselined_at_utc")

        actual_manifest_sha, _ = _sha256_and_bytes(manifest_path)
        if sidecar.get("original_manifest_sha256") != actual_manifest_sha:
            common_errors.append("original_manifest_sha256_mismatch")

        mode = sidecar.get("mode")
        if mode not in {"recomputed_against_disk", "superseded", "raw_evidence_unavailable"}:
            common_errors.append("invalid_mode")
        if mode == "raw_evidence_unavailable" and not allow_raw_evidence_unavailable:
            common_errors.append("raw_evidence_unavailable_not_allowed")

        if common_errors:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected={
                        "schema_version": MANIFEST_REBASELINE_SCHEMA_VERSION,
                        "rebaseline_of": expected_manifest_path,
                        "original_manifest_sha256": actual_manifest_sha,
                    },
                    actual={"errors": common_errors, "sidecar": sidecar},
                )
            )
            continue

        original_paths = {
            entry.get("path")
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }

        if mode == "recomputed_against_disk":
            rebaseline_entries = sidecar.get("entries")
            if not isinstance(rebaseline_entries, list):
                failures.append(
                    _invalid_rebaseline_failure(
                        manifest=manifest_path,
                        sidecar=sidecar_path,
                        expected="entries list for recomputed_against_disk",
                        actual=rebaseline_entries,
                    )
                )
                continue
            rebaseline_paths = {
                entry.get("path")
                for entry in rebaseline_entries
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            }
            if not original_paths.issubset(rebaseline_paths):
                failures.append(
                    _invalid_rebaseline_failure(
                        manifest=manifest_path,
                        sidecar=sidecar_path,
                        expected={"paths_covering": sorted(original_paths)},
                        actual={"paths": sorted(rebaseline_paths)},
                    )
                )
                continue
            stale_failures: list[dict[str, object]] = []
            for entry in rebaseline_entries:
                stale_failures.extend(
                    _verify_hash_claim(
                        manifest=manifest_path,
                        entry=entry,
                        mismatch_reason="rebaseline_stale",
                    )
                )
            failures.extend(stale_failures)
            if not stale_failures:
                annotations.append(
                    {
                        "manifest": manifest_path.as_posix(),
                        "mode": mode,
                        "entries": len(rebaseline_entries),
                    }
                )
            continue

        if mode == "raw_evidence_unavailable":
            unavailable_entries = sidecar.get("entries")
            if unavailable_entries == "all":
                unavailable_paths = original_paths
            elif isinstance(unavailable_entries, list) and all(
                isinstance(entry, dict) and isinstance(entry.get("path"), str)
                for entry in unavailable_entries
            ):
                unavailable_paths = {entry["path"] for entry in unavailable_entries}
            else:
                failures.append(
                    _invalid_rebaseline_failure(
                        manifest=manifest_path,
                        sidecar=sidecar_path,
                        expected='entries "all" or list of path objects',
                        actual=unavailable_entries,
                    )
                )
                continue

            annotated_entries = 0
            for entry in entries:
                entry_path = entry.get("path") if isinstance(entry, dict) else None
                if entry_path in unavailable_paths:
                    annotated_entries += 1
                else:
                    failures.extend(_verify_hash_claim(manifest=manifest_path, entry=entry))
            if annotated_entries:
                annotations.append(
                    {
                        "manifest": manifest_path.as_posix(),
                        "mode": mode,
                        "annotated": "raw_evidence_unavailable",
                        "entries": annotated_entries,
                    }
                )
            continue

        superseded_by = sidecar.get("superseded_by")
        if not isinstance(superseded_by, str) or not superseded_by:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected="non-empty superseded_by path",
                    actual=superseded_by,
                )
            )
            continue
        superseding_path = Path(superseded_by)
        superseding_payload, superseding_error = _load_json_file(superseding_path)
        if superseding_error is not None or superseding_payload is None:
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected="existing superseding manifest JSON object",
                    actual={"path": superseded_by, "error": superseding_error},
                )
            )
            continue
        superseding_entries = superseding_payload.get(entries_key)
        if not isinstance(superseding_entries, list):
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected=f"superseding manifest {entries_key} list",
                    actual=superseding_entries,
                )
            )
            continue
        superseding_paths = {
            entry.get("path")
            for entry in superseding_entries
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        if not original_paths.issubset(superseding_paths):
            failures.append(
                _invalid_rebaseline_failure(
                    manifest=manifest_path,
                    sidecar=sidecar_path,
                    expected={"paths_covering": sorted(original_paths)},
                    actual={"paths": sorted(superseding_paths)},
                )
            )
            continue
        stale_failures = []
        for entry in superseding_entries:
            stale_failures.extend(
                _verify_hash_claim(
                    manifest=manifest_path,
                    entry=entry,
                    mismatch_reason="superseding_manifest_stale",
                )
            )
        failures.extend(stale_failures)
        if not stale_failures:
            annotations.append(
                {
                    "manifest": manifest_path.as_posix(),
                    "mode": mode,
                    "superseded_by": superseded_by,
                }
            )

    # Rule-level diagnostics stay, volume stays bounded: a fully deleted raw
    # layer must not turn every gate run into a 135k-entry dump.
    max_failure_details = 50
    return GateResult(
        ok=len(failures) == 0,
        details={
            "count": len(manifest_paths),
            "checked_entries": checked_entries,
            "annotations": annotations,
            "failure_count": len(failures),
            "failures": failures[:max_failure_details],
            "failures_truncated": max(0, len(failures) - max_failure_details),
        },
    )


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
    elif schema_version == SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 and path.as_posix() not in _historical_exemption_entries("run_manifests"):
        failures.append(f"{path}:unexempted_v1_schema")

    if schema_version == SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        provenance_class = payload.get("provenance_class")
        if provenance_class not in {"executor_run", "manual_operator", "backfill"}:
            failures.append(f"{path}:invalid_provenance_class:{provenance_class}")

    if "claim" in payload:
        claim = payload.get("claim")
        failures.extend(
            f"{path}:{failure}"
            for failure in _validate_required_keys(claim, {"lease_id", "sha"}, "claim")
        )
        if isinstance(claim, dict):
            lease_id = claim.get("lease_id")
            if not isinstance(lease_id, int) or isinstance(lease_id, bool):
                failures.append(f"{path}:claim:invalid_lease_id")
            sha = claim.get("sha")
            if not isinstance(sha, str) or not sha.strip():
                failures.append(f"{path}:claim:invalid_sha")

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

    if "usage" in payload:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            failures.append(f"{path}:usage:not_object")
        else:
            wall_clock_seconds = usage.get("wall_clock_seconds")
            if not isinstance(wall_clock_seconds, (int, float)) or isinstance(
                wall_clock_seconds, bool
            ):
                failures.append(f"{path}:usage:invalid_wall_clock_seconds")
            source = usage.get("source")
            if not isinstance(source, str):
                failures.append(f"{path}:usage:invalid_source")

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

    schema_version = payload.get("schema_version")
    if schema_version not in {
        JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1,
        JUDGE_REVIEW_LOG_SCHEMA_VERSION,
    }:
        failures.append(f"{path}:invalid_schema_version:{payload.get('schema_version')}")
    elif schema_version == JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1 and path.as_posix() not in _historical_exemption_entries("review_logs"):
        failures.append(f"{path}:unexempted_v1_schema")

    reviewer = payload.get("reviewer")
    reviewer_keys = {"role"}
    if schema_version == JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        reviewer_keys.update({"session_id", "recorded_at_utc"})
    failures.extend(
        f"{path}:{failure}"
        for failure in _validate_required_keys(reviewer, reviewer_keys, "reviewer")
    )
    if isinstance(reviewer, dict):
        if reviewer.get("role") != contract.scientific_review_role:
            failures.append(f"{path}:invalid_reviewer_role:{reviewer.get('role')}")
        if schema_version == JUDGE_REVIEW_LOG_SCHEMA_VERSION:
            session_id = reviewer.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                failures.append(f"{path}:invalid_reviewer_session_id")

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

    try:
        raw_framework = json.loads(Path("contracts/framework.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"framework_contract_unreadable:{exc}")
        raw_framework = {}
    ceilings = raw_framework.get("complexity_tier_ceilings") if isinstance(raw_framework, dict) else None
    for tier in ("S", "M", "L"):
        tier_values = ceilings.get(tier) if isinstance(ceilings, dict) else None
        if not isinstance(tier_values, dict):
            failures.append(f"missing_complexity_tier_ceiling:{tier}")
            continue
        for field in ("max_wall_clock_seconds", "max_tokens", "max_cost_usd"):
            value = tier_values.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                failures.append(f"invalid_complexity_tier_ceiling:{tier}:{field}:{value}")
    citation_policy = raw_framework.get("citation_policy") if isinstance(raw_framework, dict) else None
    if citation_policy is not None:
        staleness_days = citation_policy.get("staleness_days") if isinstance(citation_policy, dict) else None
        if (
            not isinstance(staleness_days, int)
            or isinstance(staleness_days, bool)
            or staleness_days <= 0
        ):
            failures.append(f"invalid_citation_staleness_days:{staleness_days}")

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
        "contracts/claims.yaml",
        "contracts/README.md",
        "contracts/data_dictionary.md",
        "contracts/decisions.md",
        "contracts/model_spec.md",
        "contracts/hybrid_interface_v1.yaml",
        "contracts/instances/README.md",
        "contracts/experiments/README.md",
        "contracts/schemas/README.md",
        "contracts/schemas/instance_manifest_v1.json",
        "contracts/schemas/experiment_spec_v1.json",
        "contracts/schemas/experiment_manifest_v1.json",
        "contracts/schemas/panel_schema.yaml",
        "contracts/schemas/panel_schema_str_v1.yaml",
        "contracts/schemas/panel_schema_decomp_v1.yaml",
        "contracts/schemas/swarm_run_manifest_v1.yaml",
        "contracts/schemas/judge_review_log_v1.yaml",
        "contracts/schemas/claims_v1.yaml",
        "docs/protocol.md",
        "docs/prereg/data_construction.lock.md",
        "docs/prereg/analysis_plan.lock.md",
        "docs/prereg/lock_a.md",
        "docs/prereg/lock_b.md",
        "docs/prereg/outcomes.yaml",
        "docs/runbook_swarm.md",
        "docs/runbook_swarm_automation.md",
        "data/raw_manifest/README.md",
        "data/processed_manifest/README.md",
        "data/samples/README.md",
        "data/citations/README.md",
        "reports/validation/README.md",
        "reports/validation/manifests/README.md",
        "reports/figures/README.md",
        "reports/tables/README.md",
        "scripts/swarm.py",
        "scripts/sweep_tasks.py",
        "scripts/quality_gates.py",
        "scripts/refresh_citations.py",
        "scripts/falsify_claims.py",
        "scripts/sweep_harness.py",
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

        if payload.get("schema_version") != PROCESSED_MANIFEST_SCHEMA_VERSION and path.as_posix() not in _historical_exemption_entries("processed_manifests"):
            failures.append(f"{path}:unexempted_legacy_processed_manifest")

        if payload.get("schema_version") == PROCESSED_MANIFEST_SCHEMA_VERSION:
            failures.extend(
                f"{path}:{failure}"
                for failure in _validate_required_keys(
                    transform,
                    {"script_sha256", "dirty"},
                    "transform",
                )
            )
            if isinstance(transform, dict):
                script_sha = transform.get("script_sha256")
                if not isinstance(script_sha, str) or re.fullmatch(r"[0-9a-f]{64}", script_sha) is None:
                    failures.append(f"{path}:transform:invalid_script_sha256")
                dirty = transform.get("dirty")
                if not isinstance(dirty, bool):
                    failures.append(f"{path}:transform:dirty_not_boolean")
                elif dirty and not isinstance(transform.get("tree_diff"), str):
                    failures.append(f"{path}:transform:dirty_without_tree_diff")

            environment = payload.get("environment")
            failures.extend(
                f"{path}:{failure}"
                for failure in _validate_required_keys(environment, {"dependencies"}, "environment")
            )
            if isinstance(environment, dict) and not isinstance(environment.get("dependencies"), dict):
                failures.append(f"{path}:environment:dependencies_not_object")

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


def gate_processed_manifest_hashes() -> GateResult:
    return _manifest_hash_gate(
        manifest_dir=Path("data/processed_manifest"),
        entries_key="outputs",
        allow_raw_evidence_unavailable=False,
    )


def gate_raw_manifest_hashes() -> GateResult:
    return _manifest_hash_gate(
        manifest_dir=Path("data/raw_manifest"),
        entries_key="files",
        allow_raw_evidence_unavailable=True,
    )


def gate_validation_report_content_binding() -> GateResult:
    report_dir = Path("reports/validation")
    if not report_dir.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "validation_report_dir_missing"})

    failures: list[dict[str, object]] = []
    legacy_reports = 0
    v2_reports = 0
    checked_inputs = 0
    report_paths = sorted(report_dir.glob("*.json"))
    for report_path in report_paths:
        payload, error = _load_json_file(report_path)
        if error is not None or payload is None:
            failures.append(
                {
                    "report": report_path.as_posix(),
                    "path": report_path.as_posix(),
                    "reason": "invalid_validation_report",
                    "expected": "JSON object",
                    "actual": error,
                    "message": "validation report is date-bound to data that has drifted",
                }
            )
            continue
        if payload.get("schema_version") != VALIDATION_REPORT_SCHEMA_VERSION:
            legacy_reports += 1
            if report_path.as_posix() not in _historical_exemption_entries("validation_reports"):
                failures.append(
                    {
                        "report": report_path.as_posix(),
                        "path": report_path.as_posix(),
                        "reason": "unexempted_legacy_validation_report",
                        "expected": VALIDATION_REPORT_SCHEMA_VERSION,
                        "actual": payload.get("schema_version"),
                        "message": "legacy validation reports are accepted only from the hash-pinned historical exemption list",
                    }
                )
            continue

        v2_reports += 1
        status = payload.get("status")
        inputs_consumed = payload.get("inputs_consumed")
        if status not in {"pass", "fail"}:
            failures.append(
                {
                    "report": report_path.as_posix(),
                    "path": report_path.as_posix(),
                    "reason": "invalid_status",
                    "expected": ["fail", "pass"],
                    "actual": status,
                    "message": "validation report is date-bound to data that has drifted",
                }
            )
        if not isinstance(inputs_consumed, list) or not inputs_consumed:
            failures.append(
                {
                    "report": report_path.as_posix(),
                    "path": report_path.as_posix(),
                    "reason": "invalid_inputs_consumed",
                    "expected": "non-empty list of path, sha256, and bytes objects",
                    "actual": inputs_consumed,
                    "message": "validation report is date-bound to data that has drifted",
                }
            )
            continue

        checked_inputs += len(inputs_consumed)
        for entry in inputs_consumed:
            claim_failures = _verify_hash_claim(manifest=report_path, entry=entry)
            for failure in claim_failures:
                failure["report"] = failure.pop("manifest")
                failure["message"] = "validation report is date-bound to data that has drifted"
            failures.extend(claim_failures)

    return GateResult(
        ok=len(failures) == 0,
        details={
            "count": len(report_paths),
            "v2_reports": v2_reports,
            "legacy_reports": legacy_reports,
            "checked_inputs": checked_inputs,
            "failures": failures,
        },
    )


def gate_projection_drift() -> GateResult:
    moves, problems = _plan_sweep(Path(".").resolve())
    serialized_moves = [
        {"source": source.as_posix(), "target": target.as_posix()}
        for source, target in moves
    ]
    # §4.1: the drift gate also covers claim-ref⇔task-file drift (offline —
    # local refs only; liveness enforcement stays with the runtime).
    claim_problems: list[str] = []
    refs_cp = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/swarm/claims/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if refs_cp.returncode == 0:
        try:
            contract = load_framework_contract()
            tasks, _ = _collect_tasks(contract)
        except Exception:
            tasks = {}
        for line in (refs_cp.stdout or "").splitlines():
            ref = line.strip()
            if not ref:
                continue
            task_id = ref.removeprefix("refs/swarm/claims/")
            task = tasks.get(task_id)
            if task is None:
                claim_problems.append(f"claim_ref_without_task:{task_id}")
            elif task.state == "done":
                claim_problems.append(f"claim_ref_on_done_task:{task_id}")
            elif task.state == "backlog":
                claim_problems.append(f"claim_ref_with_backlog_state:{task_id}")
    problems = list(problems) + claim_problems

    return GateResult(
        ok=not moves and not problems,
        details={"moves": serialized_moves, "problems": problems},
    )


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


def _referee_family(tool: object) -> str | None:
    if not isinstance(tool, str) or not tool.strip():
        return None
    normalized = tool.strip().lower()
    if "claude" in normalized or "anthropic" in normalized:
        return "claude"
    if "codex" in normalized or "openai" in normalized:
        return "codex"
    if "gemini" in normalized or "google" in normalized:
        return "gemini"
    return normalized


def _load_referee_bar(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    payload, error = _load_json_file(path)
    if error is not None or not isinstance(payload, dict):
        return None, [f"{path}:{error or 'not_object'}"]
    failures: list[str] = []
    required = {"agreement_floor", "position_flip_ceiling", "committed_by", "committed_at_utc"}
    if set(payload) != required:
        failures.append(f"{path}:fields:{sorted(payload)}")
    for key in ("agreement_floor", "position_flip_ceiling"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
            failures.append(f"{path}:{key}:invalid")
    if not isinstance(payload.get("committed_by"), str) or not payload["committed_by"].strip():
        failures.append(f"{path}:committed_by:invalid")
    if _parse_utc_iso(payload.get("committed_at_utc")) is None:
        failures.append(f"{path}:committed_at_utc:invalid")
    return payload, failures


def gate_referee_rubrics() -> GateResult:
    failures: list[str] = []
    rubric_dir = Path("contracts/rubrics")
    for filename, expected_kind in REFEREE_RUBRIC_TASK_KINDS.items():
        path = rubric_dir / f"{filename}.yaml"
        payload, error = _load_json_file(path)
        if error is not None or not isinstance(payload, dict):
            failures.append(f"{path}:{error or 'not_object'}")
            continue
        if set(payload) != {"schema_version", "task_kind", "checks"}:
            failures.append(f"{path}:invalid_fields:{sorted(payload)}")
        if payload.get("schema_version") != REFEREE_RUBRIC_SCHEMA_VERSION:
            failures.append(f"{path}:invalid_schema_version")
        if payload.get("task_kind") != expected_kind:
            failures.append(f"{path}:invalid_task_kind:{payload.get('task_kind')}")
        checks = payload.get("checks")
        if not isinstance(checks, list) or not checks:
            failures.append(f"{path}:invalid_checks")
            continue
        seen: set[str] = set()
        for index, check in enumerate(checks):
            if not isinstance(check, dict) or set(check) != {"id", "prompt", "severity", "evidence_required"}:
                failures.append(f"{path}:check_{index}:invalid_fields")
                continue
            check_id = check.get("id")
            if not isinstance(check_id, str) or not check_id.strip() or check_id in seen:
                failures.append(f"{path}:check_{index}:invalid_id:{check_id}")
            else:
                seen.add(check_id)
            if not isinstance(check.get("prompt"), str) or not check["prompt"].strip():
                failures.append(f"{path}:check_{index}:invalid_prompt")
            if check.get("severity") not in {"major", "minor"}:
                failures.append(f"{path}:check_{index}:invalid_severity")
            if not isinstance(check.get("evidence_required"), bool):
                failures.append(f"{path}:check_{index}:invalid_evidence_required")
    _, bar_failures = _load_referee_bar(rubric_dir / "calibration.yaml")
    failures.extend(bar_failures)
    seed_path = rubric_dir / "sampling_seed.txt"
    if not seed_path.is_file() or not seed_path.read_text(encoding="utf-8").strip():
        failures.append(f"{seed_path}:missing_or_empty")

    try:
        framework = json.loads(Path("contracts/framework.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        framework = {}
    executors = framework.get("executors") if isinstance(framework, dict) else None
    panel = executors.get("referee_panel") if isinstance(executors, dict) else None
    if not isinstance(panel, list) or not panel:
        failures.append("contracts/framework.json:missing_referee_panel")
    else:
        families: list[str] = []
        for index, item in enumerate(panel):
            if not isinstance(item, dict):
                failures.append(f"contracts/framework.json:referee_panel_{index}:invalid")
                continue
            family = item.get("family")
            if not isinstance(family, str) or not family.strip():
                failures.append(f"contracts/framework.json:referee_panel_{index}:family_invalid")
            else:
                families.append(family)
            if item.get("profile") != "read-only":
                failures.append(f"contracts/framework.json:referee_panel_{index}:profile_not_read_only")
            if not isinstance(item.get("model"), str) or not item["model"].strip():
                failures.append(f"contracts/framework.json:referee_panel_{index}:model_invalid")
            if not isinstance(item.get("cli_version"), str) or not item["cli_version"].strip():
                failures.append(f"contracts/framework.json:referee_panel_{index}:cli_version_invalid")
            prompt_path = item.get("prompt_path")
            if not isinstance(prompt_path, str) or not Path(prompt_path).is_file():
                failures.append(f"contracts/framework.json:referee_panel_{index}:prompt_path_invalid")
            tools = item.get("tools")
            if tools is not None and tools != ["Read", "Glob", "Grep"]:
                failures.append(f"contracts/framework.json:referee_panel_{index}:tools_not_read_only")
        if len(families) != len(set(families)):
            failures.append("contracts/framework.json:referee_panel_duplicate_family_vote")
    policy = framework.get("referee_panel") if isinstance(framework, dict) else None
    if not isinstance(policy, dict) or policy.get("required_non_authoring_families") != 2:
        failures.append("contracts/framework.json:referee_panel_policy_invalid")
    elif policy.get("owner_waiver") is not None:
        waiver = policy.get("owner_waiver")
        if not isinstance(waiver, dict) or set(waiver) != {"human_id", "reason"}:
            failures.append("contracts/framework.json:referee_panel_owner_waiver_invalid")
    return GateResult(ok=not failures, details={"rubric_count": len(REFEREE_RUBRIC_TASK_KINDS), "failures": failures})


def _referee_task_in_scope(task: Task) -> bool:
    frontmatter = _parse_task_frontmatter(_read_text(task.path)) or {}
    complexity = frontmatter.get("complexity_tier")
    kind = task.task_kind
    if kind == "repair":
        kind = frontmatter.get("repair_source_task_kind", kind)
        complexity = frontmatter.get("repair_source_complexity_tier", complexity)
    return bool(
        (complexity in {"M", "L"} and kind in {"analysis", "model", "bridge", "writing"})
        or task.workstream in {"W6", "W7"}
        or any(_path_matches_prefix(output, "reports/paper/") for output in task.outputs)
    )


def _referee_manuscript_surface(task: Task) -> bool:
    return task.task_kind == "writing" or any(
        _path_matches_prefix(output, "reports/paper/") for output in task.outputs
    )


def _referee_declared_text(task: Task) -> str:
    chunks: list[str] = []
    for raw in task.outputs:
        path = Path(raw)
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {".md", ".qmd", ".txt", ".tex"}:
                chunks.append(candidate.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _referee_claim_bound(claim: dict[str, object], task: Task, manuscript_text: str) -> bool:
    frontmatter = _parse_task_frontmatter(_read_text(task.path)) or {}
    source_task = frontmatter.get("repair_source_task") if task.task_kind == "repair" else None
    owner = next(
        (
            claim.get(key)
            for key in ("task_id", "registered_by_task", "source_task_id")
            if isinstance(claim.get(key), str)
        ),
        None,
    )
    if owner in {task.task_id, source_task}:
        return True
    claim_ids = frontmatter.get("claim_ids")
    if isinstance(claim_ids, list) and claim.get("claim_id") in claim_ids:
        return True
    artifacts = claim.get("supporting_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            path = artifact.get("path") if isinstance(artifact, dict) else None
            if isinstance(path, str) and any(
                _path_matches_prefix(path, output) or _path_matches_prefix(output, path)
                for output in task.outputs
            ):
                return True
    if not _referee_manuscript_surface(task):
        return False
    citation_key = claim.get("citation_key")
    if isinstance(citation_key, str) and re.search(
        rf"(?<![A-Za-z0-9_-])@{re.escape(citation_key)}(?![A-Za-z0-9_-])",
        manuscript_text,
    ):
        return True
    literals = claim.get("manuscript_numeric_literals")
    return isinstance(literals, list) and any(
        isinstance(value, str) and value in manuscript_text for value in literals
    )


def _referee_quote_challenge(
    *,
    raw: bytes,
    seed: str,
    task_id: str,
    claim_id: str,
    path: str,
) -> tuple[int, str]:
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if not lines:
        lines = [""]
    selector = hashlib.sha256(
        f"{seed}\0{task_id}\0{claim_id}\0{path}\0quoted-span".encode("utf-8")
    ).digest()
    index = int.from_bytes(selector[:8], "big") % len(lines)
    return index + 1, lines[index]


def _public_referee_sample(item: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in item.items() if key != "expected_quoted_span"}


def _kernel_referee_sample(task: Task) -> tuple[list[dict[str, object]], int]:
    try:
        seed = Path("contracts/rubrics/sampling_seed.txt").read_text(encoding="utf-8").strip()
        ledger = json.loads(Path("contracts/claims.yaml").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 0
    claims = ledger.get("claims") if isinstance(ledger, dict) else None
    if not isinstance(claims, list):
        return [], 0
    manuscript_text = _referee_declared_text(task) if _referee_manuscript_surface(task) else ""
    scoped = [
        claim
        for claim in claims
        if isinstance(claim, dict) and _referee_claim_bound(claim, task, manuscript_text)
    ]
    ranked: list[tuple[str, dict[str, object]]] = []
    for claim in scoped:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str):
            continue
        artifacts = claim.get("supporting_artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not isinstance(artifact.get("sha256"), str):
                continue
            artifact_path = Path(artifact["path"])
            raw = artifact_path.read_bytes() if artifact_path.is_file() else b""
            disk_sha = hashlib.sha256(raw).hexdigest() if artifact_path.is_file() else ""
            ledger_sha = artifact["sha256"].lower()
            challenge_line, expected_quoted_span = _referee_quote_challenge(
                raw=raw,
                seed=seed,
                task_id=task.task_id,
                claim_id=claim["claim_id"],
                path=artifact["path"],
            )
            item: dict[str, object] = {
                "claim_id": claim["claim_id"],
                "path": artifact["path"],
                "sha256": disk_sha,
                "ledger_sha256": ledger_sha,
                "tampered": disk_sha != ledger_sha,
                "challenge_line": challenge_line,
                "expected_quoted_span": expected_quoted_span,
            }
            score = hashlib.sha256(
                f"{seed}\0{task.task_id}\0{item['claim_id']}\0{item['path']}".encode("utf-8")
            ).hexdigest()
            ranked.append((score, item))
    ranked.sort(key=lambda item: (item[0], str(item[1]["path"])))
    return [item for _, item in ranked[:3]], len(scoped)


def _referee_report_journaled(report: dict[str, object]) -> bool:
    task_id = report.get("task_id")
    run_sha = report.get("run_manifest_sha256")
    actor = report.get("actor")
    session_id = report.get("session_id")
    if (
        not isinstance(task_id, str)
        or not isinstance(run_sha, str)
        or actor != "Referee"
        or not isinstance(session_id, str)
        or not session_id
    ):
        return False
    events, _ = _read_swarm_events(Path.cwd())
    return any(
        event.get("event") == "referee_invoked"
        and event.get("task_id") == task_id
        and event.get("run_manifest_sha256") == run_sha
        and event.get("actor") == actor
        and event.get("session_id") == session_id
        and event.get("actor_session") == session_id
        for event in events
    )


def _referee_owner_waiver(
    *,
    task_id: str,
    run_manifest_sha256: str,
) -> dict[str, object] | None:
    events, _ = _read_swarm_events(Path.cwd())
    trusted_base = _trusted_integration_branch(Path.cwd())
    for event in reversed(events):
        if not (
            event.get("event") == "referee_owner_waiver"
            and event.get("emitted_by") == REFEREE_WAIVER_EMITTER
            and event.get("actor") == "HumanOwner"
            # emitted on the trusted integration branch, not any non-empty string
            and event.get("control_plane_branch") == trusted_base
            and event.get("task_id") == task_id
            and event.get("run_manifest_sha256") == run_manifest_sha256
        ):
            continue
        human_id = event.get("human_id")
        reason = event.get("reason")
        if not isinstance(human_id, str) or not human_id.strip():
            continue
        if not isinstance(reason, str) or not reason.strip():
            continue
        waiver = {"human_id": human_id, "reason": reason}
        waiver_sha = hashlib.sha256(
            json.dumps(waiver, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if event.get("waiver_sha256") == waiver_sha:
            return {**waiver, "waiver_sha256": waiver_sha}
    return None


def gate_referee_report_validity() -> GateResult:
    if not REFEREE_REPORT_DIR.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "referee_report_dir_missing"})
    report_paths = sorted(REFEREE_REPORT_DIR.glob("*.json"))
    if not report_paths:
        return GateResult(ok=True, details={"skipped": True, "reason": "no_referee_reports"})
    failures: list[str] = []
    panel_votes: dict[tuple[str, str, str], set[str]] = {}
    panel_authors: dict[tuple[str, str, str], set[str]] = {}
    try:
        contract = load_framework_contract()
        tasks, _ = _collect_tasks(contract)
    except Exception as exc:
        return GateResult(ok=False, details={"failures": [f"task_load_failed:{exc}"]})

    for path in report_paths:
        report, error = _load_json_file(path)
        prefix = path.as_posix()
        if error is not None or not isinstance(report, dict):
            failures.append(f"{prefix}:{error or 'not_object'}")
            continue
        required_fields = {
            "schema_version", "task_id", "actor", "session_id", "referee_family",
            "rubric_version", "run_manifest_sha256", "verdicts", "opened_artifacts",
            "overall",
        }
        missing = sorted(required_fields - set(report))
        if missing:
            failures.append(f"{prefix}:missing_fields:{','.join(missing)}")
        if report.get("schema_version") != REFEREE_REPORT_SCHEMA_VERSION:
            failures.append(f"{prefix}:invalid_schema_version")
        task_id = report.get("task_id")
        if not isinstance(task_id, str) or task_id not in tasks:
            failures.append(f"{prefix}:unknown_task:{task_id}")
            continue
        task = tasks[task_id]
        if not _referee_report_journaled(report):
            failures.append(f"{prefix}:referee_report_unjournaled")
        if report.get("valid") is not True:
            failures.append(f"{prefix}:report_marked_invalid")
        family = report.get("referee_family")
        authoring = report.get("authoring_family")
        if not isinstance(family, str) or not family.strip():
            failures.append(f"{prefix}:referee_family_invalid")
        if family == authoring:
            failures.append(f"{prefix}:referee_family_of_author:{family}")
        manifest_rel = report.get("run_manifest_path")
        if isinstance(manifest_rel, str):
            report_run_sha = report.get("run_manifest_sha256")
            panel_key = (task_id, manifest_rel, str(report_run_sha))
            if isinstance(family, str) and family.strip():
                panel_votes.setdefault(panel_key, set()).add(family)
            if isinstance(authoring, str) and authoring.strip():
                panel_authors.setdefault(panel_key, set()).add(authoring)
            manifest, manifest_error = _load_json_file(Path(manifest_rel))
            if manifest_error is not None or not isinstance(manifest, dict):
                failures.append(f"{prefix}:run_manifest_unreadable:{manifest_rel}")
            else:
                executor = manifest.get("executor") if isinstance(manifest.get("executor"), dict) else {}
                derived_author = _referee_family(executor.get("tool"))
                if derived_author != authoring:
                    failures.append(f"{prefix}:authoring_family_mismatch:{authoring}!={derived_author}")
                expected_manifest_sha = report.get("run_manifest_sha256")
                if isinstance(expected_manifest_sha, str):
                    actual_manifest_sha, _ = _sha256_and_bytes(Path(manifest_rel))
                    if expected_manifest_sha != actual_manifest_sha:
                        failures.append(f"{prefix}:run_manifest_sha256_mismatch")
                else:
                    failures.append(f"{prefix}:run_manifest_sha256_invalid")
        else:
            failures.append(f"{prefix}:run_manifest_path_invalid")

        sampled, scoped_claim_count = _kernel_referee_sample(task)
        reported_sample = report.get("sampled_artifacts")
        public_sample = [_public_referee_sample(item) for item in sampled]
        if reported_sample != public_sample:
            failures.append(f"{prefix}:kernel_sample_mismatch")
        if scoped_claim_count and not sampled:
            failures.append(f"{prefix}:referee_sample_empty_for_claims")
        for item in sampled:
            if item.get("tampered") is True:
                failures.append(f"{prefix}:referee_sampled_artifact_tampered:{item.get('path')}")
        opened = report.get("opened_artifacts")
        opened_by_path = {
            item.get("path"): item
            for item in opened
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        } if isinstance(opened, list) else {}
        missing_opened = [
            item["path"]
            for item in sampled
            if item["path"] not in opened_by_path
        ]
        if missing_opened:
            failures.append(f"{prefix}:referee_did_not_open_sampled:{','.join(missing_opened)}")
        for item in sampled:
            opened_item = opened_by_path.get(item["path"])
            if not isinstance(opened_item, dict):
                continue
            if opened_item.get("sha256") != item["sha256"]:
                failures.append(f"{prefix}:referee_opened_artifact_disk_sha_mismatch:{item['path']}")
            if opened_item.get("quoted_span") != item.get("expected_quoted_span"):
                failures.append(f"{prefix}:referee_opened_artifact_quote_mismatch:{item['path']}")

        frontmatter = _parse_task_frontmatter(_read_text(task.path)) or {}
        expected_ids: dict[str, tuple[str, str]] = {}
        criteria = frontmatter.get("success_criteria")
        if isinstance(criteria, list):
            for item in criteria:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    expected_ids[item["id"]] = ("success_criterion_id", "major")
        rubric_files = report.get("rubric_files")
        if not isinstance(rubric_files, list) or not rubric_files:
            failures.append(f"{prefix}:rubric_files_invalid")
            rubric_files = []
        for rubric_rel in rubric_files:
            if not isinstance(rubric_rel, str) or not rubric_rel.startswith("contracts/rubrics/"):
                failures.append(f"{prefix}:rubric_path_invalid:{rubric_rel}")
                continue
            rubric, rubric_error = _load_json_file(Path(rubric_rel))
            if rubric_error is not None or not isinstance(rubric, dict):
                failures.append(f"{prefix}:rubric_unreadable:{rubric_rel}")
                continue
            for check in rubric.get("checks", []):
                if isinstance(check, dict) and isinstance(check.get("id"), str):
                    expected_ids[check["id"]] = ("check_id", str(check.get("severity")))
        assertion_floor = report.get("assertion_prefilter_floor")
        if isinstance(assertion_floor, list):
            for assertion in assertion_floor:
                if isinstance(assertion, dict) and isinstance(assertion.get("check_id"), str):
                    expected_ids[assertion["check_id"]] = ("check_id", "major")

        verdicts = report.get("verdicts")
        seen: set[str] = set()
        major: list[dict[str, object]] = []
        all_verdicts: list[dict[str, object]] = []
        if not isinstance(verdicts, list):
            failures.append(f"{prefix}:verdicts_invalid")
            verdicts = []
        for index, verdict in enumerate(verdicts):
            if not isinstance(verdict, dict):
                failures.append(f"{prefix}:verdict_{index}:invalid")
                continue
            identifier_keys = [key for key in ("success_criterion_id", "check_id") if isinstance(verdict.get(key), str)]
            if len(identifier_keys) != 1:
                failures.append(f"{prefix}:verdict_{index}:identifier_invalid")
                continue
            key = identifier_keys[0]
            identifier = verdict[key]
            if identifier not in expected_ids or expected_ids[identifier][0] != key:
                failures.append(f"{prefix}:verdict_unexpected:{identifier}")
                continue
            if identifier in seen:
                failures.append(f"{prefix}:verdict_duplicate:{identifier}")
            seen.add(identifier)
            if verdict.get("verdict") not in REFEREE_VERDICTS:
                failures.append(f"{prefix}:verdict_value_invalid:{identifier}")
            if verdict.get("severity") != expected_ids[identifier][1]:
                failures.append(f"{prefix}:verdict_severity_mismatch:{identifier}")
            if not isinstance(verdict.get("evidence_pointer"), str) or not verdict["evidence_pointer"].strip():
                failures.append(f"{prefix}:evidence_pointer_missing:{identifier}")
            if not isinstance(verdict.get("note"), str) or not verdict["note"].strip():
                failures.append(f"{prefix}:note_missing:{identifier}")
            if expected_ids[identifier][1] == "major":
                major.append(verdict)
            all_verdicts.append(verdict)
        for missing_id in sorted(set(expected_ids) - seen):
            failures.append(f"{prefix}:verdict_missing:{missing_id}")
        computed_overall = (
            "cannot_verify" if any(item.get("verdict") == "cannot_verify" for item in all_verdicts)
            else "not_supported" if any(item.get("verdict") == "not_supported" for item in major)
            else "supported"
        )
        if report.get("overall") != computed_overall:
            failures.append(f"{prefix}:overall_mismatch:{report.get('overall')}!={computed_overall}")
        if task.state == "done":
            for item in all_verdicts:
                identifier = item.get("success_criterion_id", item.get("check_id"))
                if item.get("verdict") == "cannot_verify":
                    failures.append(f"{prefix}:done_with_cannot_verify:{identifier}")
                if (
                    item.get("verdict") == "not_supported"
                    and expected_ids.get(str(identifier), (None, None))[1] == "major"
                    and _referee_task_in_scope(task)
                ):
                    failures.append(f"{prefix}:done_with_not_supported:{identifier}")
                if item.get("verdict") == "not_supported" and isinstance(item.get("check_id"), str) and item["check_id"].startswith("ASSERTION-"):
                    failures.append(f"{prefix}:done_with_unregistered_assertion:{identifier}")
    for (task_id, manifest_rel, manifest_sha), families in sorted(panel_votes.items()):
        task = tasks.get(task_id)
        if task is None or task.state != "done" or not _referee_manuscript_surface(task):
            continue
        non_authoring = families - panel_authors.get((task_id, manifest_rel, manifest_sha), set())
        waiver = _referee_owner_waiver(
            task_id=task_id,
            run_manifest_sha256=manifest_sha,
        )
        required_votes = 1 if waiver is not None else 2
        if len(non_authoring) < required_votes:
            failures.append(
                f"{task_id}:{manifest_rel}:referee_manuscript_panel_family_quorum:"
                f"{len(non_authoring)}<{required_votes}"
            )
    return GateResult(ok=not failures, details={"count": len(report_paths), "failures": failures})


def _referee_release_verdict_severity(
    report: dict[str, object],
    verdict: dict[str, object],
) -> str | None:
    if isinstance(verdict.get("success_criterion_id"), str):
        return "major"
    check_id = verdict.get("check_id")
    if not isinstance(check_id, str):
        return None
    if check_id.startswith("ASSERTION-"):
        return "major"
    rubric_files = report.get("rubric_files")
    if not isinstance(rubric_files, list):
        return None
    for raw in rubric_files:
        if not isinstance(raw, str) or not raw.startswith("contracts/rubrics/"):
            continue
        rubric, error = _load_json_file(Path(raw))
        if error is not None or not isinstance(rubric, dict):
            continue
        checks = rubric.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict) and check.get("id") == check_id:
                severity = check.get("severity")
                return str(severity) if severity in {"major", "minor"} else None
    return None


def gate_referee_release_evidence() -> GateResult:
    paper_paths = (
        Path("reports/paper/build/l2_l1_rent_working_paper.html"),
        Path("reports/paper/build/l2_l1_rent_working_paper.pdf"),
        Path("reports/paper/build/render_manifest.json"),
    )
    if not all(path.is_file() for path in paper_paths):
        return GateResult(ok=True, details={"skipped": True, "reason": "no_materialized_manuscript_release"})
    try:
        contract = load_framework_contract()
        tasks, parse_failures = _collect_tasks(contract)
    except Exception as exc:
        return GateResult(ok=False, details={"failures": [f"task_load_failed:{exc}"]})
    failures = list(parse_failures)
    evidence: list[dict[str, object]] = []
    manuscript_tasks = [
        task
        for task in tasks.values()
        if _referee_manuscript_surface(task) and task.state in {"ready_for_review", "done"}
    ]
    if not manuscript_tasks:
        failures.append("referee_release_manuscript_task_missing")
    for task in sorted(manuscript_tasks, key=lambda item: item.task_id):
        manifests: list[tuple[Path, dict[str, object]]] = []
        for path in sorted(Path(contract.run_manifest_dir).glob(f"{task.task_id}_*.json")):
            payload, error = _load_json_file(path)
            if error is None and isinstance(payload, dict):
                result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
                if result.get("status") == "ok":
                    manifests.append((path, payload))
        if not manifests:
            failures.append(f"{task.task_id}:referee_release_run_manifest_missing")
            continue
        manifest_path, manifest = manifests[-1]
        manifest_sha, _ = _sha256_and_bytes(manifest_path)
        executor = manifest.get("executor") if isinstance(manifest.get("executor"), dict) else {}
        authoring_family = _referee_family(executor.get("tool"))
        reports: list[tuple[Path, dict[str, object]]] = []
        for path in sorted(REFEREE_REPORT_DIR.glob(f"{task.task_id}_*.json")):
            report, error = _load_json_file(path)
            if (
                error is None
                and isinstance(report, dict)
                and report.get("run_manifest_sha256") == manifest_sha
                and report.get("run_manifest_path") == manifest_path.as_posix()
            ):
                reports.append((path, report))
        votes: dict[str, tuple[Path, dict[str, object]]] = {}
        for path, report in reversed(reports):
            family = report.get("referee_family")
            if isinstance(family, str) and family not in votes:
                votes[family] = (path, report)
        non_authoring = {
            family: value
            for family, value in votes.items()
            if family != authoring_family and value[1].get("authoring_family") != family
        }
        waiver = _referee_owner_waiver(
            task_id=task.task_id,
            run_manifest_sha256=manifest_sha,
        )
        identifier_sets: dict[str, set[str]] = {}
        report_artifacts: list[dict[str, object]] = []
        valid_calibrated_non_authoring: set[str] = set()
        for family, (path, report) in sorted(non_authoring.items()):
            report_valid = report.get("valid") is True and _referee_report_journaled(report)
            if not report_valid:
                failures.append(f"{task.task_id}:referee_release_report_invalid:{family}")
            calibration_failures = (
                calibration_report_failures(
                    repo=Path.cwd(),
                    report_path=REFEREE_CALIBRATION_REPORT,
                    required_family=family,
                )
                if REFEREE_CALIBRATION_REPORT.is_file()
                else ["calibration_missing"]
            )
            if calibration_failures:
                failures.append(f"{task.task_id}:referee_release_uncalibrated:{family}")
            if report_valid and not calibration_failures:
                valid_calibrated_non_authoring.add(family)
            verdicts = report.get("verdicts") if isinstance(report.get("verdicts"), list) else []
            identifiers = {
                str(item.get("success_criterion_id", item.get("check_id")))
                for item in verdicts
                if isinstance(item, dict)
                and isinstance(item.get("success_criterion_id", item.get("check_id")), str)
            }
            identifier_sets[family] = identifiers
            report_sha, report_bytes = _sha256_and_bytes(path)
            report_artifacts.append(
                {"path": path.as_posix(), "sha256": report_sha, "bytes": report_bytes, "family": family}
            )
        required_quorum = 1 if waiver is not None else 2
        if len(valid_calibrated_non_authoring) < required_quorum:
            failures.append(
                f"{task.task_id}:referee_release_panel_quorum:"
                f"{len(valid_calibrated_non_authoring)}<{required_quorum}"
            )
        # Release checks every report bound to the current manuscript run,
        # including non-voting author-family comments. A waiver changes only
        # family quorum; it never changes substantive verdict semantics.
        for path, report in reports:
            family = report.get("referee_family")
            verdicts = report.get("verdicts") if isinstance(report.get("verdicts"), list) else []
            for item in verdicts:
                if not isinstance(item, dict):
                    continue
                verdict = item.get("verdict")
                identifier = item.get("success_criterion_id", item.get("check_id", "unknown"))
                if verdict == "cannot_verify":
                    failures.append(
                        f"{task.task_id}:referee_release_cannot_verify:{family}:{identifier}"
                    )
                if (
                    verdict == "not_supported"
                    and _referee_release_verdict_severity(report, item) == "major"
                ):
                    failures.append(
                        f"{task.task_id}:referee_release_not_supported:{family}:{identifier}"
                    )
                if (
                    verdict == "not_supported"
                    and isinstance(item.get("check_id"), str)
                    and item["check_id"].startswith("ASSERTION-")
                ):
                    failures.append(
                        f"{task.task_id}:referee_release_unregistered_assertion:{family}:{identifier}"
                    )
        if identifier_sets:
            union = set().union(*identifier_sets.values())
            for family, identifiers in identifier_sets.items():
                missing = sorted(union - identifiers)
                if missing:
                    failures.append(
                        f"{task.task_id}:referee_release_vote_missing:{family}:{','.join(missing)}"
                    )
        elif waiver is None:
            failures.append(f"{task.task_id}:referee_release_votes_missing")
        evidence.append(
            {
                "task_id": task.task_id,
                "run_manifest_path": manifest_path.as_posix(),
                "run_manifest_sha256": manifest_sha,
                "authoring_family": authoring_family,
                "non_authoring_families": sorted(valid_calibrated_non_authoring),
                "reports": report_artifacts,
                "owner_waiver": waiver,
            }
        )
    calibration_artifact = None
    if REFEREE_CALIBRATION_REPORT.is_file():
        digest, size = _sha256_and_bytes(REFEREE_CALIBRATION_REPORT)
        calibration_artifact = {
            "path": REFEREE_CALIBRATION_REPORT.as_posix(),
            "sha256": digest,
            "bytes": size,
        }
    return GateResult(
        ok=not failures,
        details={
            "failures": sorted(set(failures)),
            "evidence": evidence,
            "calibration": calibration_artifact,
        },
    )


def gate_referee_calibration() -> GateResult:
    report_paths = sorted(REFEREE_REPORT_DIR.glob("*.json")) if REFEREE_REPORT_DIR.exists() else []
    if not report_paths and not REFEREE_CALIBRATION_REPORT.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "no_referee_reports"})
    if not REFEREE_CALIBRATION_REPORT.is_file():
        failures = [f"{REFEREE_CALIBRATION_REPORT}:missing"]
    else:
        failures = [
            f"{REFEREE_CALIBRATION_REPORT}:{failure}"
            for failure in calibration_report_failures(
                repo=Path.cwd(),
                report_path=REFEREE_CALIBRATION_REPORT,
            )
        ]
    return GateResult(ok=not failures, details={"reports_present": len(report_paths), "failures": failures})


HISTORICAL_EXEMPTIONS_SCHEMA_VERSION = "research_swarm.historical_exemptions.v1"
PROVENANCE_ANNOTATION_SCHEMA_VERSION = "research_swarm.provenance_annotation.v1"
VALID_PROVENANCE_CLASSES = {"executor_run", "manual_operator", "backfill"}


def gate_historical_exemptions() -> GateResult:
    """Gate-scoping rule (plan §4.0 remediation): strict checks apply to
    schema_version >= 2 artifacts; v1 artifacts must sit on the checked-in,
    hash-pinned exemption list, and every exempted run manifest must carry a
    provenance annotation that still matches the untouched original."""
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    failures: list[str] = []
    exempted: dict[str, dict[str, str]] = {"run_manifests": {}, "review_logs": {}}
    exemptions_path = Path("contracts/historical_exemptions.json")

    if exemptions_path.exists():
        payload, error = _load_json_file(exemptions_path)
        if error is not None or not isinstance(payload, dict):
            return GateResult(ok=False, details={"failures": [f"{exemptions_path}:{error}"]})
        if payload.get("schema_version") != HISTORICAL_EXEMPTIONS_SCHEMA_VERSION:
            failures.append(f"{exemptions_path}:invalid_schema_version")
        for kind in ("run_manifests", "review_logs"):
            for item in payload.get(kind, []):
                if not isinstance(item, dict):
                    failures.append(f"{exemptions_path}:{kind}:invalid_entry")
                    continue
                rel = item.get("path")
                expected_sha = item.get("sha256")
                if not isinstance(rel, str) or not isinstance(expected_sha, str):
                    failures.append(f"{exemptions_path}:{kind}:invalid_entry:{rel}")
                    continue
                artifact = Path(rel)
                if not artifact.is_file():
                    failures.append(f"exemption_list_drift:missing_file:{rel}")
                    continue
                actual_sha, _ = _sha256_and_bytes(artifact)
                if actual_sha != expected_sha:
                    failures.append(f"exemption_list_drift:sha256_mismatch:{rel}")
                    continue
                exempted[kind][rel] = expected_sha

    def sweep_v1(directory: Path, kind: str, v1_version: str) -> int:
        count = 0
        if not directory.exists():
            return count
        for path in sorted(directory.glob("*.json")):
            payload, error = _load_json_file(path)
            if error is not None or not isinstance(payload, dict):
                continue  # shape gates own malformed artifacts
            if payload.get("schema_version") != v1_version:
                continue
            count += 1
            rel = path.as_posix()
            if rel not in exempted[kind]:
                failures.append(f"unexempted_v1_artifact:{rel}")
        return count

    v1_runs = sweep_v1(Path(contract.run_manifest_dir), "run_manifests", SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1)
    v1_reviews = sweep_v1(Path(contract.judge_review_dir), "review_logs", JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1)

    annotations_checked = 0
    annotation_class_counts: dict[str, int] = {}
    exempted_entries = _historical_exemption_entries("run_manifests")
    for rel in sorted(exempted["run_manifests"]):
        manifest_path = Path(rel)
        annotation_path = manifest_path.parent / "annotations" / f"{manifest_path.name}.provenance.json"
        if not annotation_path.is_file():
            failures.append(f"provenance_annotation_missing:{rel}")
            continue
        annotation, error = _load_json_file(annotation_path)
        if error is not None or not isinstance(annotation, dict):
            failures.append(f"provenance_annotation_invalid:{rel}:{error}")
            continue
        annotations_checked += 1
        if annotation.get("schema_version") != PROVENANCE_ANNOTATION_SCHEMA_VERSION:
            failures.append(f"provenance_annotation_invalid:{rel}:schema_version")
        if annotation.get("annotates") != rel:
            failures.append(f"provenance_annotation_invalid:{rel}:annotates_mismatch")
        annotation_class = annotation.get("provenance_class")
        if annotation_class not in VALID_PROVENANCE_CLASSES:
            failures.append(f"provenance_annotation_invalid:{rel}:provenance_class")
        else:
            annotation_class_counts[annotation_class] = annotation_class_counts.get(annotation_class, 0) + 1
        annotated_sha = annotation.get("annotates_sha256")
        if annotated_sha != exempted["run_manifests"][rel]:
            failures.append(f"provenance_annotation_invalid:{rel}:annotates_sha256_mismatch")

        # the annotation FILE is itself hash-pinned, and its class must agree
        # with both the pinned class and a mechanical re-derivation from the
        # manifest's own executor fields — labels are immutable, not editable.
        entry = exempted_entries.get(rel, {})
        pinned_annotation_sha = entry.get("annotation_sha256")
        if isinstance(pinned_annotation_sha, str):
            actual_annotation_sha, _ = _sha256_and_bytes(annotation_path)
            if actual_annotation_sha != pinned_annotation_sha:
                failures.append(f"provenance_annotation_invalid:{rel}:annotation_file_drift")
        else:
            failures.append(f"provenance_annotation_invalid:{rel}:annotation_not_pinned")
        pinned_class = entry.get("provenance_class")
        if pinned_class != annotation_class:
            failures.append(f"provenance_annotation_invalid:{rel}:class_pin_mismatch")
        manifest_payload, manifest_error = _load_json_file(manifest_path)
        if manifest_error is None and isinstance(manifest_payload, dict):
            executor = manifest_payload.get("executor") if isinstance(manifest_payload.get("executor"), dict) else {}
            runner = executor.get("runner")
            tool = executor.get("tool")
            if runner == "legacy_backfill" or tool == "operator_backfill":
                derived = "backfill"
            elif tool == "codex":
                derived = "executor_run"
            elif tool == "manual":
                derived = "manual_operator"
            else:
                derived = None
            if derived is not None and derived != annotation_class:
                failures.append(f"provenance_annotation_invalid:{rel}:class_derivation_mismatch:{derived}")

    # the release amendment's per-class counts must agree with the annotations
    release_dir = Path("reports/status/releases")
    if annotation_class_counts and release_dir.exists():
        for release_path in sorted(release_dir.glob("release_*.json")):
            release_payload, release_error = _load_json_file(release_path)
            if release_error is not None or not isinstance(release_payload, dict):
                continue
            for note in release_payload.get("notes", []):
                if isinstance(note, dict) and note.get("type") == "raw_evidence_unavailable":
                    recorded = note.get("provenance_class_run_counts")
                    if recorded != annotation_class_counts:
                        failures.append(
                            f"release_amendment_count_mismatch:{release_path.name}:{recorded}!={annotation_class_counts}"
                        )

    return GateResult(
        ok=len(failures) == 0,
        details={
            "exemptions_file": exemptions_path.exists(),
            "exempted_run_manifests": len(exempted["run_manifests"]),
            "exempted_review_logs": len(exempted["review_logs"]),
            "v1_run_manifests": v1_runs,
            "v1_review_logs": v1_reviews,
            "annotations_checked": annotations_checked,
            "failures": failures,
        },
    )


def _parse_utc_iso(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _review_min_separation_seconds(repo: Path) -> int:
    framework_path = repo / "contracts" / "framework.json"
    try:
        payload = json.loads(framework_path.read_text(encoding="utf-8"))
        return int(payload["review_bundle"]["min_separation_seconds"])
    except Exception:
        return 60


def _done_bundle_approval_failures(valid_review_logs: list[Path], repo: Path) -> list[str]:
    approve_logs: list[tuple[Path, dict[str, object]]] = []
    for path in valid_review_logs:
        payload, error = _load_json_file(path)
        if error is not None or not isinstance(payload, dict):
            continue
        decision = payload.get("decision")
        if isinstance(decision, dict) and decision.get("outcome") == "approve":
            approve_logs.append((path, payload))

    if not approve_logs:
        return ["missing_approving_review_log"]

    review_path, review = approve_logs[-1]
    if review.get("schema_version") == JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1:
        # historical bundle: the validator already required exemption-list membership
        return []

    task_block = review.get("task") if isinstance(review.get("task"), dict) else {}
    manifest_rel = task_block.get("run_manifest_path")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        return [f"approving_review_missing_manifest_link:{review_path.name}"]
    manifest, error = _load_json_file(repo / manifest_rel)
    if error is not None or not isinstance(manifest, dict):
        return [f"approving_review_manifest_unreadable:{manifest_rel}"]
    if manifest.get("schema_version") != SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        return [f"approving_review_manifest_not_v2:{manifest_rel}"]
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    if result.get("status") != "ok":
        return [f"approving_review_manifest_not_passing:{manifest_rel}"]
    if manifest.get("provenance_class") != "executor_run":
        return [f"approving_review_manifest_provenance:{manifest.get('provenance_class')}:{manifest_rel}"]
    return []


def _review_log_actor_separation_failures(review_path: Path, repo: Path) -> list[str]:
    """§4.0 #17: a v2 review log is invalid if written by the same actor session
    as the run manifest it reviews, or inside the minimum separation window."""
    payload, error = _load_json_file(review_path)
    if error is not None or payload is None:
        return []
    if payload.get("schema_version") != JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        return []

    reviewer = payload.get("reviewer") if isinstance(payload.get("reviewer"), dict) else {}
    task_block = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    manifest_rel = task_block.get("run_manifest_path")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        return []
    manifest_payload, manifest_error = _load_json_file(repo / manifest_rel)
    if manifest_error is not None or manifest_payload is None:
        return []
    if manifest_payload.get("schema_version") != SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        return []

    failures: list[str] = []
    actor = manifest_payload.get("actor") if isinstance(manifest_payload.get("actor"), dict) else {}
    run_session = actor.get("session_id")
    review_session = reviewer.get("session_id")
    if (
        isinstance(run_session, str)
        and run_session.strip()
        and run_session == review_session
    ):
        failures.append(f"actor_separation_same_session:{review_path.name}")

    run_time = _parse_utc_iso(manifest_payload.get("generated_at_utc"))
    review_time = _parse_utc_iso(payload.get("generated_at_utc"))
    if run_time is None or review_time is None:
        failures.append(f"actor_separation_window_unverifiable:{review_path.name}")
    else:
        separation = (review_time - run_time).total_seconds()
        minimum = _review_min_separation_seconds(repo)
        if separation < minimum:
            failures.append(
                f"actor_separation_window:{review_path.name}:{int(separation)}s<{minimum}s"
            )
    return failures


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
            for review_path in valid_review_logs:
                for reason in _review_log_actor_separation_failures(review_path, repo):
                    failures.append(f"{task.path}:{reason}")

            # done requires an APPROVING review whose linked run manifest is a
            # passing executor_run — a blocking review or a backfill/blocked
            # manifest can never durably satisfy done (§4.0 #6).
            failures.extend(
                f"{task.path}:{reason}"
                for reason in _done_bundle_approval_failures(valid_review_logs, repo)
            )

    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_network_strings() -> GateResult:
    """§9.4 (M1): gate-command strings in non-network workstreams must not
    reference network tools or URLs — deterministic gates stay offline."""
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    tasks, parse_failures = _collect_tasks(contract)
    failures: list[str] = list(parse_failures)
    network_workstreams = set(contract.network_workstreams)
    for task in tasks.values():
        if task.workstream in network_workstreams:
            continue
        for gate in task.gates:
            lowered = gate.lower()
            hits = sorted(token for token in NETWORK_COMMAND_TOKENS if token in lowered)
            if hits:
                failures.append(f"{task.path}:network_string_in_gate:{','.join(hits)}:{gate}")
    return GateResult(ok=len(failures) == 0, details={"failures": failures})


def gate_task_lint() -> GateResult:
    """§4.4 (M2): strict schema-v2 specification and independence checks."""
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    task_paths = _iter_task_files(contract)
    diagnostics = lint_task_files(
        task_paths,
        repo_root=Path.cwd(),
        network_workstreams=contract.network_workstreams,
        v1_exemptions=_historical_exemption_entries("tasks"),
    )
    return GateResult(
        ok=not diagnostics,
        details={
            "count": len(task_paths),
            "failures": [diagnostic.as_dict() for diagnostic in diagnostics],
        },
    )


def _science_failure(
    reason: str,
    *,
    subject: str,
    field: str | None = None,
    expected: object = None,
    actual: object = None,
) -> dict[str, object]:
    failure: dict[str, object] = {"reason": reason, "subject": subject}
    if field is not None:
        failure["field"] = field
    if expected is not None:
        failure["expected"] = expected
    if actual is not None:
        failure["actual"] = actual
    return failure


def _load_claim_ledger() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    path = Path("contracts/claims.yaml")
    payload, error = _load_json_file(path)
    if error is not None or payload is None:
        return [], [
            _science_failure(
                "invalid_claims_contract",
                subject=path.as_posix(),
                expected="JSON-compatible YAML object",
                actual=error,
            )
        ]

    failures: list[dict[str, object]] = []
    if payload.get("schema_version") != CLAIMS_SCHEMA_VERSION:
        failures.append(
            _science_failure(
                "invalid_claims_schema",
                subject=path.as_posix(),
                field="schema_version",
                expected=CLAIMS_SCHEMA_VERSION,
                actual=payload.get("schema_version"),
            )
        )
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        failures.append(
            _science_failure(
                "claims_not_list",
                subject=path.as_posix(),
                field="claims",
                expected="list",
                actual=raw_claims,
            )
        )
        return [], failures

    claims: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw_claim in enumerate(raw_claims):
        subject = f"claims[{index}]"
        if not isinstance(raw_claim, dict):
            failures.append(
                _science_failure(
                    "claim_not_object", subject=subject, expected="object", actual=raw_claim
                )
            )
            continue
        claim = dict(raw_claim)
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            failures.append(
                _science_failure(
                    "invalid_claim_id", subject=subject, field="claim_id", actual=claim_id
                )
            )
        elif claim_id in seen_ids:
            failures.append(
                _science_failure("duplicate_claim_id", subject=claim_id, field="claim_id")
            )
        else:
            seen_ids.add(claim_id)
            subject = claim_id

        for field in ("statement", "verification_command"):
            value = claim.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append(
                    _science_failure(
                        f"invalid_{field}", subject=subject, field=field, actual=value
                    )
                )
        claim_type = claim.get("type")
        if claim_type not in CLAIM_TYPES:
            failures.append(
                _science_failure(
                    "invalid_claim_type",
                    subject=subject,
                    field="type",
                    expected=sorted(CLAIM_TYPES),
                    actual=claim_type,
                )
            )
        artifacts = claim.get("supporting_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            failures.append(
                _science_failure(
                    "invalid_supporting_artifacts",
                    subject=subject,
                    field="supporting_artifacts",
                    expected="non-empty list",
                    actual=artifacts,
                )
            )
        if "uncertainty_artifact" not in claim:
            failures.append(
                _science_failure(
                    "missing_uncertainty_artifact_field",
                    subject=subject,
                    field="uncertainty_artifact",
                )
            )
        hypothesis_id = claim.get("hypothesis_id")
        if hypothesis_id is not None and (
            not isinstance(hypothesis_id, str) or not hypothesis_id.strip()
        ):
            failures.append(
                _science_failure(
                    "invalid_hypothesis_id",
                    subject=subject,
                    field="hypothesis_id",
                    actual=hypothesis_id,
                )
            )
        claims.append(claim)
    return claims, failures


def _verify_claim_artifact(
    entry: object,
    *,
    subject: str,
    field: str,
) -> list[dict[str, object]]:
    if not isinstance(entry, dict):
        return [
            _science_failure(
                "invalid_hashed_artifact",
                subject=subject,
                field=field,
                expected="{path, sha256}",
                actual=entry,
            )
        ]
    raw_path = entry.get("path")
    expected_sha = entry.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
        or Path(raw_path).is_absolute()
        or raw_path.startswith("~")
        or ".." in raw_path.replace("\\", "/").split("/")
    ):
        return [
            _science_failure(
                "invalid_artifact_path",
                subject=subject,
                field=f"{field}.path",
                expected="safe repo-relative path",
                actual=raw_path,
            )
        ]
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        return [
            _science_failure(
                "invalid_artifact_sha256",
                subject=subject,
                field=f"{field}.sha256",
                expected="64 lowercase hex characters",
                actual=expected_sha,
            )
        ]

    path = Path(_normalize_repo_relative_path(raw_path))
    if not path.is_file():
        return [
            _science_failure(
                "missing_artifact",
                subject=subject,
                field=field,
                expected=path.as_posix(),
                actual=None,
            )
        ]
    # §6.5 purity: claim evidence must be a non-symlink, in-repo, git-tracked
    # regular file — otherwise the verdict depends on non-committed state (a
    # symlink to an environment file recomputes differently per machine).
    repo = Path.cwd().resolve()
    resolved = path.resolve()
    if (
        path.is_symlink()
        or repo not in resolved.parents
        or not _git_path_is_tracked(path.as_posix(), repo)
    ):
        return [
            _science_failure(
                "artifact_not_tracked_regular_file",
                subject=subject,
                field=field,
                expected="non-symlink, in-repo, git-tracked regular file",
                actual=raw_path,
            )
        ]
    actual_sha, _ = _sha256_and_bytes(path)
    if actual_sha != expected_sha:
        return [
            _science_failure(
                "artifact_sha256_mismatch",
                subject=subject,
                field=field,
                expected=expected_sha,
                actual=actual_sha,
            )
        ]
    return []


_LOCK_BINDING_RE = re.compile(
    r"^\s*-\s+path:\s*(?P<path>\S+)\s*$\n"
    r"^\s+sha256:\s*(?P<sha256>[0-9a-f]{64}|pending)\s*$",
    flags=re.MULTILINE,
)


def _lock_bindings(lock: dict[str, object] | None) -> list[dict[str, str]]:
    if lock is None:
        return []
    body = lock.get("body")
    if not isinstance(body, str):
        return []
    return [match.groupdict() for match in _LOCK_BINDING_RE.finditer(body)]


def _active_prereg_lock(phase: str) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    path = Path(PREREG_PHASE_FILES[phase])
    lock, error = load_prereg_lock(path, expected_phase=phase)
    failures: list[dict[str, object]] = []
    if error is not None or lock is None:
        failures.append(
            _science_failure(
                f"invalid_{phase}_lock",
                subject=path.as_posix(),
                actual=error,
            )
        )
        return None, failures
    if lock.get("status") == "locked" and lock.get("active") is not True:
        failures.append(
            _science_failure(
                f"{phase}_lock_hash_mismatch",
                subject=path.as_posix(),
                field="locked_sha256",
                expected=lock.get("body_sha256"),
                actual=lock.get("locked_sha256"),
            )
        )
    return (lock if lock.get("active") is True else None), failures


def _verify_lock_bindings(
    lock: dict[str, object],
    *,
    phase: str,
    repo: Path = Path("."),
) -> tuple[dict[str, str], list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    by_path: dict[str, str] = {}
    for binding in _lock_bindings(lock):
        raw_path = binding["path"]
        expected_sha = binding["sha256"]
        subject = f"{PREREG_PHASE_FILES[phase]}:{raw_path}"
        path = _safe_repo_relative_path(raw_path)
        if path is None:
            failures.append(_science_failure("invalid_lock_binding_path", subject=subject))
            continue
        normalized = path.as_posix()
        if normalized in by_path:
            failures.append(_science_failure("duplicate_lock_binding", subject=subject))
            continue
        by_path[normalized] = expected_sha
        if expected_sha == "pending":
            failures.append(_science_failure("pending_lock_binding", subject=subject))
            continue
        disk_path = repo / path
        if not disk_path.is_file():
            failures.append(_science_failure("missing_lock_binding_target", subject=subject))
            continue
        actual_sha, _ = _sha256_and_bytes(disk_path)
        if actual_sha != expected_sha:
            failures.append(
                _science_failure(
                    "lock_binding_sha256_mismatch",
                    subject=subject,
                    expected=expected_sha,
                    actual=actual_sha,
                )
            )
    return by_path, failures


def _active_experiment_spec() -> tuple[
    dict[str, object] | None,
    Path | None,
    dict[str, object] | None,
    list[dict[str, object]],
]:
    lock, failures = _active_prereg_lock("lock_a")
    if lock is None:
        return None, None, None, failures
    bindings, binding_failures = _verify_lock_bindings(lock, phase="lock_a")
    failures.extend(binding_failures)
    model_paths = [path for path in bindings if path == "contracts/model_spec.md"]
    experiment_paths = [
        path
        for path in bindings
        if path.startswith("contracts/experiments/") and Path(path).suffix in {".json", ".yaml"}
    ]
    if len(model_paths) != 1:
        failures.append(
            _science_failure(
                "lock_a_model_spec_binding_required",
                subject=PREREG_PHASE_FILES["lock_a"],
                expected="exactly contracts/model_spec.md",
                actual=model_paths,
            )
        )
    if len(experiment_paths) != 1:
        failures.append(
            _science_failure(
                "lock_a_experiment_spec_binding_required",
                subject=PREREG_PHASE_FILES["lock_a"],
                expected="exactly one contracts/experiments/*.json|yaml",
                actual=experiment_paths,
            )
        )
        return None, None, lock, failures
    spec_path = Path(experiment_paths[0])
    payload, error = _load_json_file(spec_path)
    if error is not None or payload is None:
        failures.append(
            _science_failure(
                "invalid_experiment_spec_json",
                subject=spec_path.as_posix(),
                actual=error,
            )
        )
        return None, spec_path, lock, failures
    for issue in _schema_failures(payload, Path("contracts/schemas/experiment_spec_v1.json")):
        failures.append(
            _science_failure(
                "experiment_spec_schema_violation",
                subject=spec_path.as_posix(),
                field=str(issue.get("path")),
                actual=issue,
            )
        )
    return payload, spec_path, lock, failures


def _model_spec_is_ambiguous(path: Path = Path("contracts/model_spec.md")) -> bool:
    if not path.is_file():
        return True
    text = _read_text(path)
    lowered = text.lower()
    if any(token in lowered for token in ("tbd", "todo", "fill it", "declare-before-lock")):
        return True
    headings = (
        "## Objective / question",
        "## Decision variables",
        "## Constraints",
        "## Objective function",
        "## Assumptions (explicit)",
        "## Solver / method",
    )
    return any(not _section_has_content(text, heading) for heading in headings)


def _modeling_prereg_result(claims: list[dict[str, object]]) -> GateResult:
    spec, spec_path, lock, failures = _active_experiment_spec()
    active_hash = lock.get("body_sha256") if lock is not None else None
    applicable = [
        claim for claim in claims if claim.get("type") in {"computational", "counterfactual"}
    ]
    if lock is not None and _model_spec_is_ambiguous():
        failures.append(
            _science_failure(
                "ambiguous_locked_model_spec",
                subject="contracts/model_spec.md",
            )
        )
    for index, claim in enumerate(applicable):
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        if active_hash is None:
            failures.append(
                _science_failure(
                    "modeling_claim_without_active_lock_a",
                    subject=subject,
                    field="lock_a_sha256",
                )
            )
            continue
        recorded_hash = claim.get("lock_a_sha256", claim.get("prereg_lock_sha256"))
        if recorded_hash != active_hash:
            failures.append(
                _science_failure(
                    "modeling_claim_lock_a_hash_mismatch",
                    subject=subject,
                    field="lock_a_sha256",
                    expected=active_hash,
                    actual=recorded_hash,
                )
            )
    # §6.1 modeling: every locked proposition/conjecture must reach a terminal
    # REPORTED outcome — a failed or negative conjecture cannot be silently
    # dropped any more than an empirical hypothesis can.
    proposition_ids = [
        h["hypothesis_id"]
        for h in (lock.get("hypotheses", []) if lock is not None else [])
        if isinstance(h, dict) and isinstance(h.get("hypothesis_id"), str)
    ]
    if proposition_ids and lock is not None and lock.get("active") is True:
        outcomes_by_id, outcome_failures = _load_prereg_outcomes()
        failures.extend(outcome_failures)
        for pid in sorted(set(proposition_ids)):
            outcome = outcomes_by_id.get(pid)
            if outcome is None:
                failures.append(_science_failure("missing_proposition_outcome", subject=pid))
                continue
            if outcome.get("outcome") not in TERMINAL_HYPOTHESIS_OUTCOMES:
                failures.append(
                    _science_failure(
                        "invalid_terminal_outcome",
                        subject=pid,
                        field="outcome",
                        expected=sorted(TERMINAL_HYPOTHESIS_OUTCOMES),
                        actual=outcome.get("outcome"),
                    )
                )
            anchor_failure = _outcome_reported_anchor_failure(outcome, pid, pid)
            if anchor_failure is not None:
                failures.append(anchor_failure)

    if not applicable and lock is None:
        failures = [
            failure
            for failure in failures
            if failure.get("reason") not in {"invalid_lock_a_lock", "lock_a_lock_hash_mismatch"}
        ]
    return GateResult(
        ok=not failures,
        details={
            "status": "no_modeling_claims_or_lock" if not applicable and lock is None else "ok",
            "active_lock_a_sha256": active_hash,
            "experiment_spec": spec_path.as_posix() if spec_path is not None else None,
            "experiment_spec_valid": spec is not None,
            "modeling_claim_count": len(applicable),
            "failures": failures,
        },
    )


def _outcome_reported_anchor_failure(
    outcome: dict[str, object], identifier: str, subject: str
) -> dict[str, object] | None:
    """§6.1/§6.5: a terminal outcome is not 'reported' until it is content-bound
    to committed manuscript or deviations-appendix text. `reported_in` must
    resolve to a tracked in-repo file whose text contains the hypothesis/
    proposition id (or the explicit #anchor). Prevents a negative or missing
    result being silently dropped while `outcomes.yaml` self-asserts it."""
    reported = outcome.get("reported_in")
    if not isinstance(reported, str) or not reported.strip():
        return _science_failure(
            "outcome_missing_reported_in",
            subject=subject,
            field="reported_in",
            expected="manuscript/deviations path that reports this outcome",
        )
    base, _, fragment = reported.partition("#")
    base = base.strip()
    safe = _safe_repo_relative_path(base)
    # The anchor must be a non-symlink git-tracked file on the MANUSCRIPT surface
    # (reports/paper/ — the paper and its deviations appendix). Pointing at the
    # outcomes registry itself (docs/prereg/outcomes.yaml), a lock, a log, or a
    # status artifact is not "reporting" — those necessarily contain the id and
    # would make the check self-satisfying.
    on_manuscript_surface = base.startswith("reports/paper/")
    if (
        safe is None
        or not safe.is_file()
        or safe.is_symlink()
        or not on_manuscript_surface
        or not _git_path_is_tracked(safe.as_posix(), Path.cwd().resolve())
    ):
        return _science_failure(
            "outcome_reported_in_unresolvable",
            subject=subject,
            field="reported_in",
            expected="tracked reports/ or docs/prereg/ file",
            actual=reported,
        )
    text = _read_text(safe)
    anchor = fragment.strip() or identifier
    if anchor not in text and identifier not in text:
        return _science_failure(
            "outcome_reported_in_missing_anchor",
            subject=subject,
            field="reported_in",
            expected=f"'{anchor}' present in {safe.as_posix()}",
        )
    return None


def _load_prereg_outcomes() -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """id -> outcome map from docs/prereg/outcomes.yaml (light loader for the
    modeling proposition coverage check; the empirical path validates inline)."""
    path = Path("docs/prereg/outcomes.yaml")
    payload, error = _load_json_file(path)
    if error is not None or payload is None:
        return {}, [_science_failure("invalid_prereg_outcomes", subject=path.as_posix(), actual=error)]
    raw = payload.get("outcomes")
    if not isinstance(raw, list):
        return {}, [_science_failure("outcomes_not_list", subject=path.as_posix(), field="outcomes", actual=raw)]
    out: dict[str, dict[str, object]] = {}
    for outcome in raw:
        if isinstance(outcome, dict) and isinstance(outcome.get("hypothesis_id"), str):
            out[outcome["hypothesis_id"]] = outcome
    return out, []


def gate_prereg_conformance(*, form: str | None = None) -> GateResult:
    """Confirmatory claims bind to phase 2b and every locked hypothesis terminates."""
    claims, failures = _load_claim_ledger()
    mode = _parse_project_mode(Path("contracts/project.yaml")) or "empirical"
    selected = form or ("modeling" if mode == "modeling" else "union" if mode == "hybrid" else "empirical")
    if selected == "modeling":
        result = _modeling_prereg_result(claims)
        combined = failures + list(result.details.get("failures", []))
        return GateResult(ok=not combined, details={**result.details, "failures": combined})
    lock_path = Path(PREREG_PHASE_FILES["2b"])
    lock, lock_error = load_prereg_lock(lock_path, expected_phase="2b")
    if lock_error is not None or lock is None:
        failures.append(
            _science_failure(
                "invalid_analysis_prereg_lock",
                subject=lock_path.as_posix(),
                actual=lock_error,
            )
        )
        lock = None
    elif lock.get("status") == "locked" and lock.get("active") is not True:
        failures.append(
            _science_failure(
                "prereg_lock_hash_mismatch",
                subject=lock_path.as_posix(),
                field="locked_sha256",
                expected=lock.get("body_sha256"),
                actual=lock.get("locked_sha256"),
            )
        )

    active_lock = lock if lock is not None and lock.get("active") is True else None
    active_hash = active_lock.get("body_sha256") if active_lock is not None else None
    confirmatory_count = 0
    for index, claim in enumerate(claims):
        empirical_types = {"causal"} if mode == "hybrid" else CONFIRMATORY_CLAIM_TYPES
        if claim.get("type") not in empirical_types:
            continue
        confirmatory_count += 1
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        if active_hash is None:
            failures.append(
                _science_failure(
                    "confirmatory_claim_without_active_lock",
                    subject=subject,
                    field="prereg_lock_sha256",
                )
            )
        elif claim.get("prereg_lock_sha256") != active_hash:
            failures.append(
                _science_failure(
                    "confirmatory_claim_prereg_hash_mismatch",
                    subject=subject,
                    field="prereg_lock_sha256",
                    expected=active_hash,
                    actual=claim.get("prereg_lock_sha256"),
                )
            )
        command = claim.get("verification_command")
        if not isinstance(command, str) or not command.strip():
            failures.append(
                _science_failure(
                    "confirmatory_claim_missing_verification_command",
                    subject=subject,
                    field="verification_command",
                )
            )
        artifacts = claim.get("supporting_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            failures.append(
                _science_failure(
                    "confirmatory_claim_missing_supporting_artifacts",
                    subject=subject,
                    field="supporting_artifacts",
                )
            )
        else:
            for artifact_index, artifact in enumerate(artifacts):
                raw_path = artifact.get("path") if isinstance(artifact, dict) else None
                unsafe_path = (
                    not isinstance(raw_path, str)
                    or not raw_path.strip()
                    or Path(raw_path).is_absolute()
                    or raw_path.startswith("~")
                    or ".." in raw_path.replace("\\", "/").split("/")
                )
                if unsafe_path or not Path(_normalize_repo_relative_path(raw_path)).is_file():
                    failures.append(
                        _science_failure(
                            "confirmatory_supporting_artifact_missing",
                            subject=subject,
                            field=f"supporting_artifacts[{artifact_index}]",
                            actual=raw_path,
                        )
                    )

    hypotheses = active_lock.get("hypotheses", []) if active_lock is not None else []
    hypothesis_ids: list[str] = []
    for hypothesis in hypotheses:
        if isinstance(hypothesis, dict) and isinstance(hypothesis.get("hypothesis_id"), str):
            hypothesis_ids.append(hypothesis["hypothesis_id"])
    duplicate_hypotheses = sorted(
        hypothesis_id for hypothesis_id in set(hypothesis_ids) if hypothesis_ids.count(hypothesis_id) > 1
    )
    for hypothesis_id in duplicate_hypotheses:
        failures.append(_science_failure("duplicate_prereg_hypothesis", subject=hypothesis_id))

    outcomes_by_id: dict[str, dict[str, object]] = {}
    if hypothesis_ids:
        outcomes_path = Path("docs/prereg/outcomes.yaml")
        outcomes_payload, outcomes_error = _load_json_file(outcomes_path)
        if outcomes_error is not None or outcomes_payload is None:
            failures.append(
                _science_failure(
                    "invalid_prereg_outcomes",
                    subject=outcomes_path.as_posix(),
                    actual=outcomes_error,
                )
            )
        else:
            raw_outcomes = outcomes_payload.get("outcomes")
            if not isinstance(raw_outcomes, list):
                failures.append(
                    _science_failure(
                        "outcomes_not_list",
                        subject=outcomes_path.as_posix(),
                        field="outcomes",
                        actual=raw_outcomes,
                    )
                )
            else:
                for index, outcome in enumerate(raw_outcomes):
                    if not isinstance(outcome, dict):
                        failures.append(
                            _science_failure(
                                "outcome_not_object", subject=f"outcomes[{index}]", actual=outcome
                            )
                        )
                        continue
                    hypothesis_id = outcome.get("hypothesis_id")
                    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
                        failures.append(
                            _science_failure(
                                "invalid_outcome_hypothesis_id",
                                subject=f"outcomes[{index}]",
                                actual=hypothesis_id,
                            )
                        )
                        continue
                    if hypothesis_id in outcomes_by_id:
                        failures.append(
                            _science_failure("duplicate_hypothesis_outcome", subject=hypothesis_id)
                        )
                    outcomes_by_id[hypothesis_id] = outcome
                    if hypothesis_id not in set(hypothesis_ids):
                        failures.append(
                            _science_failure("outcome_not_in_prereg", subject=hypothesis_id)
                        )
                    terminal = outcome.get("outcome")
                    if terminal not in TERMINAL_HYPOTHESIS_OUTCOMES:
                        failures.append(
                            _science_failure(
                                "invalid_terminal_outcome",
                                subject=hypothesis_id,
                                field="outcome",
                                expected=sorted(TERMINAL_HYPOTHESIS_OUTCOMES),
                                actual=terminal,
                            )
                        )
                    if terminal == "abandoned" and (
                        not isinstance(outcome.get("reason"), str)
                        or not outcome.get("reason", "").strip()
                    ):
                        failures.append(
                            _science_failure(
                                "abandoned_outcome_missing_reason",
                                subject=hypothesis_id,
                                field="reason",
                            )
                        )
                    anchor_failure = _outcome_reported_anchor_failure(
                        outcome, hypothesis_id, hypothesis_id
                    )
                    if anchor_failure is not None:
                        failures.append(anchor_failure)
        for hypothesis_id in sorted(set(hypothesis_ids)):
            if hypothesis_id not in outcomes_by_id:
                failures.append(
                    _science_failure("missing_hypothesis_outcome", subject=hypothesis_id)
                )

    status = "ok"
    if not claims and not hypothesis_ids and not failures:
        status = "no_claims" if active_lock is not None else "no_active_lock"
    if selected == "union":
        modeling_result = _modeling_prereg_result(claims)
        failures.extend(modeling_result.details.get("failures", []))
    return GateResult(
        ok=not failures,
        details={
            "status": status,
            "active_lock_sha256": active_hash,
            "confirmatory_claim_count": confirmatory_count,
            "hypothesis_count": len(set(hypothesis_ids)),
            "failures": failures,
        },
    )


def _coverage_required_phase(task: Task, mode: str) -> str | None:
    writes_processed = any(
        _path_matches_prefix(output, "data/processed/") for output in task.outputs
    )
    if mode in {"empirical", "hybrid"}:
        if task.task_kind == "etl" and writes_processed:
            return "2a"
        if task.task_kind == "analysis":
            return "2b"
    if mode == "modeling" and task.task_kind in {"model", "analysis"}:
        return "lock_a"
    if mode == "hybrid":
        if task.task_kind == "bridge":
            return "lock_a"
        if task.task_kind == "model":
            return "lock_b"
    return None


def _claim_task_id(claim: dict[str, object]) -> str | None:
    for key in ("task_id", "registered_by_task", "source_task_id"):
        value = claim.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def gate_prereg_lock_coverage() -> GateResult:
    """Audit lock headers against the journal and reject post-hoc task completion."""
    failures: list[dict[str, object]] = []
    events, malformed = _read_swarm_events(Path.cwd())
    latest_lock_events: dict[str, dict[str, object]] = {}
    for event in events:
        if event.get("event") in {"prereg_locked", "prereg_amendment"} and event.get(
            "phase"
        ) in PREREG_PHASE_FILES:
            latest_lock_events[str(event["phase"])] = event

    active_phases: set[str] = set()
    for phase, rel_path in PREREG_PHASE_FILES.items():
        lock, error = load_prereg_lock(Path(rel_path), expected_phase=phase)
        if error is not None or lock is None or lock.get("active") is not True:
            continue
        active_phases.add(phase)
        event = latest_lock_events.get(phase)
        subject = rel_path
        if event is None:
            failures.append(
                _science_failure("active_prereg_lock_missing_journal_event", subject=subject)
            )
            continue
        expected_fields = {
            "phase": lock.get("phase"),
            "status": lock.get("status"),
            "locked_at_utc": lock.get("locked_at_utc"),
            "locked_sha256": lock.get("locked_sha256"),
            "lock_version": lock.get("lock_version"),
        }
        for field, expected in expected_fields.items():
            if event.get(field) != expected:
                failures.append(
                    _science_failure(
                        "prereg_lock_header_journal_mismatch",
                        subject=subject,
                        field=field,
                        expected=expected,
                        actual=event.get(field),
                    )
                )

    # §6.1 amendment discipline (deterministic backstop to the CLI guard): the
    # cap is two amendments PER PROGRAM, and per-phase amendment versions are
    # strictly increasing in journal order — a rolled-back header + re-amend is
    # caught here even if the CLI guard was bypassed or the journal hand-edited.
    # A corrupted (malformed) append-only journal is not auditable — silently
    # skipping a malformed amendment line would lower the program count and let
    # a third amendment through. Fail closed on any malformed event so the
    # amendment cap can't be laundered by hand-corrupting an earlier line.
    if malformed:
        failures.append(
            _science_failure(
                "prereg_journal_malformed",
                subject="reports/status/events",
                expected="0 malformed journal events (append-only integrity)",
                actual=malformed,
            )
        )
    program_amendment_count = 0
    amendment_versions_by_phase: dict[str, list[int]] = {}
    for event in events:
        if event.get("event") != "prereg_amendment":
            continue
        phase = event.get("phase")
        if phase not in PREREG_PHASE_FILES:
            continue
        program_amendment_count += 1
        version = event.get("lock_version")
        if isinstance(version, int) and not isinstance(version, bool):
            amendment_versions_by_phase.setdefault(str(phase), []).append(version)
    if program_amendment_count > 2:
        failures.append(
            _science_failure(
                "amendment_cap_exceeded_program",
                subject="docs/prereg",
                expected="<=2 amendments per program (§6.1)",
                actual=program_amendment_count,
            )
        )
    for phase, versions in amendment_versions_by_phase.items():
        for earlier, later in zip(versions, versions[1:]):
            if later <= earlier:
                failures.append(
                    _science_failure(
                        "amendment_version_non_monotonic",
                        subject=PREREG_PHASE_FILES[phase],
                        expected=f">{earlier}",
                        actual=later,
                    )
                )
                break

    try:
        contract = load_framework_contract()
        tasks, _ = _collect_tasks(contract)
        mode = contract.project_mode or "empirical"
    except ValueError as exc:
        failures.append(
            _science_failure("invalid_framework_contract", subject="contracts/framework.json", actual=str(exc))
        )
        tasks = {}
        mode = "empirical"
    claims, _ = _load_claim_ledger()
    claim_types_by_task: dict[str, set[str]] = {}
    has_counterfactual_claim = False
    for claim in claims:
        task_id = _claim_task_id(claim)
        claim_type = claim.get("type")
        if task_id is not None and isinstance(claim_type, str):
            claim_types_by_task.setdefault(task_id, set()).add(claim_type)
        if claim_type == "counterfactual":
            has_counterfactual_claim = True

    seen_locks: set[str] = set()
    audited_done = 0
    for event in events:
        if event.get("event") in {"prereg_locked", "prereg_amendment"}:
            phase = event.get("phase")
            if phase in PREREG_PHASE_FILES and event.get("status", "locked") == "locked":
                seen_locks.add(str(phase))
            continue
        if event.get("event") != "task_done":
            continue
        task_id = event.get("task_id")
        task = tasks.get(task_id) if isinstance(task_id, str) else None
        if task is None or task.state != "done":
            continue
        required = _coverage_required_phase(task, mode)
        frontmatter = _parse_task_frontmatter(_read_text(task.path))
        declared_claim_types = (
            {
                value
                for value in frontmatter.get("claim_types", [])
                if isinstance(value, str)
            }
            if isinstance(frontmatter, dict)
            and isinstance(frontmatter.get("claim_types"), list)
            else set()
        )
        if "counterfactual" in (
            claim_types_by_task.get(task.task_id, set()) | declared_claim_types
        ) or (
            has_counterfactual_claim
            and any(
                _normalize_repo_relative_path(output) == "contracts/claims.yaml"
                for output in task.outputs
            )
        ):
            required = "lock_b"
        if required is None:
            continue
        audited_done += 1
        if required not in seen_locks:
            failures.append(
                _science_failure(
                    "task_completed_before_required_prereg_lock",
                    subject=task.task_id,
                    field="required_phase",
                    expected=required,
                )
            )
    return GateResult(
        ok=not failures,
        details={
            "status": "no_auditable_lock_requiring_completions" if audited_done == 0 else "ok",
            "active_phases": sorted(active_phases),
            "audited_done_count": audited_done,
            "malformed_event_count": malformed,
            "failures": failures,
        },
    )


def _claim_amended_lock(claim: dict[str, object]) -> tuple[str, dict[str, object]] | None:
    binding_fields = {
        "2b": "prereg_lock_sha256",
        "lock_a": "lock_a_sha256",
        "lock_b": "lock_b_sha256",
    }
    for phase, field in binding_fields.items():
        recorded = claim.get(field)
        if not isinstance(recorded, str):
            continue
        lock, error = load_prereg_lock(Path(PREREG_PHASE_FILES[phase]), expected_phase=phase)
        if error is None and lock is not None and lock.get("active") is True:
            version = lock.get("lock_version")
            if recorded == lock.get("body_sha256") and isinstance(version, int) and version > 1:
                return phase, lock
    return None


def _claim_reconfirmation_failures(
    claim: dict[str, object], *, subject: str
) -> tuple[bool, list[dict[str, object]]]:
    artifact = claim.get("reconfirmation_artifact")
    if artifact is None:
        return False, []
    failures = _verify_claim_artifact(
        artifact,
        subject=subject,
        field="reconfirmation_artifact",
    )
    return not failures, failures


def gate_amendment_exploratory_tagging() -> GateResult:
    claims, failures = _load_claim_ledger()
    amended_claims = 0
    for index, claim in enumerate(claims):
        if claim.get("type") not in CONFIRMATORY_CLAIM_TYPES:
            continue
        amended = _claim_amended_lock(claim)
        if amended is None:
            continue
        amended_claims += 1
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        reconfirmed, reconfirmation_failures = _claim_reconfirmation_failures(
            claim, subject=subject
        )
        failures.extend(reconfirmation_failures)
        if claim.get("confirmatory") is True and not reconfirmed:
            failures.append(
                _science_failure(
                    "amended_lock_claim_must_be_exploratory",
                    subject=subject,
                    field="confirmatory",
                    expected=False,
                    actual=True,
                )
            )
    return GateResult(
        ok=not failures,
        details={"status": "ok", "amended_claim_count": amended_claims, "failures": failures},
    )


def gate_headline_confirmatory() -> GateResult:
    claims, failures = _load_claim_ledger()
    exploratory_headlines: list[str] = []
    for index, claim in enumerate(claims):
        if claim.get("headline") is not True or _claim_amended_lock(claim) is None:
            continue
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        reconfirmed, reconfirmation_failures = _claim_reconfirmation_failures(
            claim, subject=subject
        )
        failures.extend(reconfirmation_failures)
        if not reconfirmed:
            exploratory_headlines.append(subject)
    if exploratory_headlines:
        venue, error = _load_venue_contract()
        release_type = venue.get("release_type") if isinstance(venue, dict) else None
        if error is not None or release_type != "exploratory_report":
            failures.append(
                _science_failure(
                    "exploratory_headline_requires_exploratory_release",
                    subject="contracts/venue.yaml",
                    field="release_type",
                    expected="exploratory_report",
                    actual=release_type if error is None else error,
                )
            )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "exploratory_headline_claims": exploratory_headlines,
            "failures": failures,
        },
    )


def gate_claim_evidence_ledger() -> GateResult:
    """Validate registered-to-evidenced claim mappings and uncertainty policy."""
    claims, failures = _load_claim_ledger()
    claim_ids = {
        claim_id
        for claim in claims
        if isinstance((claim_id := claim.get("claim_id")), str) and claim_id.strip()
    }
    for index, claim in enumerate(claims):
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        artifacts = claim.get("supporting_artifacts")
        if isinstance(artifacts, list):
            for artifact_index, artifact in enumerate(artifacts):
                failures.extend(
                    _verify_claim_artifact(
                        artifact,
                        subject=subject,
                        field=f"supporting_artifacts[{artifact_index}]",
                    )
                )

        command = claim.get("verification_command")
        if isinstance(command, str) and command.strip():
            violation = gate_command_violation(command)
            if violation is not None:
                failures.append(
                    _science_failure(
                        "verification_command_policy_violation",
                        subject=subject,
                        field="verification_command",
                        expected="make <target> or python[3] <repo .py>",
                        actual=violation,
                    )
                )
            normalized_command = " ".join(command.strip().split())
            # shlex so a quoted path (python "scripts/quality_gates.py") is
            # canonicalized like an unquoted one — quotes must not evade the
            # self-reference check.
            try:
                command_tokens = shlex.split(command)
            except ValueError:
                command_tokens = normalized_command.split()
            # Self-referential by parsed semantics, not three literal strings:
            # any `make gate` target or any invocation of the gate runner itself
            # cannot be a claim's independent verification — regardless of flags
            # (--json), path spelling (./ or quoted), OR interpreter options
            # before the script (python -B/-u/-- scripts/quality_gates.py). The
            # script operand is the first non-option token, matching
            # gate_command_violation's own operand rule.
            script_operand: str | None = None
            if command_tokens and command_tokens[0] in {"python", "python3"}:
                for token in command_tokens[1:]:
                    if token.startswith("-"):
                        continue
                    script_operand = os.path.normpath(token)
                    break
            is_self_referential = (
                command_tokens[:2] == ["make", "gate"]
                or script_operand == "scripts/quality_gates.py"
            )
            if is_self_referential:
                failures.append(
                    _science_failure(
                        "verification_command_self_referential",
                        subject=subject,
                        field="verification_command",
                        actual=normalized_command,
                    )
                )
            if re.search(r"\b(?:todo|tbd|not[_ -]?run|manual(?:ly)?)\b", command, re.IGNORECASE):
                failures.append(
                    _science_failure(
                        "verification_command_unexecuted_shaped",
                        subject=subject,
                        field="verification_command",
                        actual=command,
                    )
                )

        claim_type = claim.get("type")
        uncertainty = claim.get("uncertainty_artifact")
        if claim_type in UNCERTAINTY_ARTIFACT_REQUIRED_TYPES:
            if uncertainty is None:
                failures.append(
                    _science_failure(
                        "uncertainty_artifact_required",
                        subject=subject,
                        field="uncertainty_artifact",
                        expected=f"hashed artifact for {claim_type}",
                    )
                )
            else:
                failures.extend(
                    _verify_claim_artifact(
                        uncertainty,
                        subject=subject,
                        field="uncertainty_artifact",
                    )
                )
        elif uncertainty is not None:
            failures.extend(
                _verify_claim_artifact(
                    uncertainty,
                    subject=subject,
                    field="uncertainty_artifact",
                )
            )

        if claim_type == "causal":
            sensitivity = claim.get("sensitivity_artifact")
            if sensitivity is None:
                failures.append(
                    _science_failure(
                        "causal_sensitivity_artifact_required",
                        subject=subject,
                        field="sensitivity_artifact",
                    )
                )
            else:
                failures.extend(
                    _verify_claim_artifact(
                        sensitivity,
                        subject=subject,
                        field="sensitivity_artifact",
                    )
                )
            identification = claim.get("identification_strategy")
            if isinstance(identification, dict):
                failures.extend(
                    _verify_claim_artifact(
                        identification,
                        subject=subject,
                        field="identification_strategy",
                    )
                )
            elif isinstance(identification, str) and identification.strip():
                base_path = identification.split("#", 1)[0]
                safe_path = _safe_repo_relative_path(base_path)
                if safe_path is None or not safe_path.is_file():
                    failures.append(
                        _science_failure(
                            "causal_identification_strategy_pointer_invalid",
                            subject=subject,
                            field="identification_strategy",
                            actual=identification,
                        )
                    )
            else:
                failures.append(
                    _science_failure(
                        "causal_identification_strategy_required",
                        subject=subject,
                        field="identification_strategy",
                    )
                )
        elif claim_type == "interpretation" and uncertainty is None:
            evidence_scope = claim.get("evidence_scope")
            if not isinstance(evidence_scope, list) or not evidence_scope or any(
                not isinstance(claim_id, str) or claim_id not in claim_ids
                for claim_id in evidence_scope
            ):
                failures.append(
                    _science_failure(
                        "interpretation_evidence_scope_required",
                        subject=subject,
                        field="evidence_scope",
                        expected="non-empty list of registered claim ids",
                        actual=evidence_scope,
                    )
                )
            elif claim.get("claim_id") in evidence_scope:
                # An interpretation cannot rest on itself — a circular scope is
                # semantically empty (§6.2 evidence-scope must list OTHER claims).
                failures.append(
                    _science_failure(
                        "interpretation_evidence_scope_self_referential",
                        subject=subject,
                        field="evidence_scope",
                        actual=claim.get("claim_id"),
                    )
                )
        elif claim_type == "theoretical":
            assumption_scope = claim.get("assumption_scope")
            if not isinstance(assumption_scope, str) or not assumption_scope.strip():
                failures.append(
                    _science_failure(
                        "theoretical_assumption_scope_required",
                        subject=subject,
                        field="assumption_scope",
                    )
                )
        elif claim_type in {"methodological", "literature"}:
            justification = claim.get("uncertainty_justification")
            if uncertainty is not None or not isinstance(justification, str) or not justification.strip():
                failures.append(
                    _science_failure(
                        "uncertainty_na_justification_required",
                        subject=subject,
                        field="uncertainty_justification",
                        expected="uncertainty_artifact null with non-empty justification",
                    )
                )

        if claim_type == "counterfactual":
            for field in (
                "empirical_artifact",
                "model_artifact",
                "cross_bridge_uncertainty_artifact",
            ):
                artifact = claim.get(field)
                if artifact is None:
                    failures.append(
                        _science_failure(
                            "counterfactual_union_artifact_required",
                            subject=subject,
                            field=field,
                        )
                    )
                else:
                    failures.extend(
                        _verify_claim_artifact(artifact, subject=subject, field=field)
                    )

        if claim_type == "counterfactual":
            lock_b, lock_failures = _active_prereg_lock("lock_b")
            failures.extend(lock_failures)
            if lock_b is None:
                failures.append(
                    _science_failure(
                        "counterfactual_claim_without_active_lock_b",
                        subject=subject,
                        field="lock_b_sha256",
                    )
                )
            else:
                active_hash = lock_b.get("body_sha256")
                if claim.get("lock_b_sha256") != active_hash:
                    failures.append(
                        _science_failure(
                            "counterfactual_claim_lock_b_hash_mismatch",
                            subject=subject,
                            field="lock_b_sha256",
                            expected=active_hash,
                            actual=claim.get("lock_b_sha256"),
                        )
                    )
                registered = _parse_utc_z(claim.get("registered_at_utc"))
                locked_at = _parse_utc_z(lock_b.get("locked_at_utc"))
                if registered is not None and locked_at is not None and registered < locked_at:
                    failures.append(
                        _science_failure(
                            "counterfactual_claim_registered_before_lock_b",
                            subject=subject,
                            field="registered_at_utc",
                            expected=f">={lock_b.get('locked_at_utc')}",
                            actual=claim.get("registered_at_utc"),
                        )
                    )

    manuscript = Path("reports/paper/index.qmd")
    unregistered_numerics: list[dict[str, object]] = []
    if manuscript.is_file():
        manuscript_text = _read_text(manuscript)
        # Occurrence-scoped registration (§6.2 registered->evidenced floor). A
        # reportable manuscript numeric is registered iff its line cites a
        # [@key] whose claim OWNS that literal (in its statement or
        # manuscript_numeric_literals). Literals are scoped PER citation key —
        # a claim cannot whitelist a number on a line that does not cite it, so
        # one benign claim can no longer launder unrelated headline numbers
        # (the global value-equality hole). The full semantic asserted->
        # registered sweep (paraphrase, claim typing) is the §5.3 M3b referee;
        # numeric recompute-against-artifact is the M4 computed-value-key layer.
        line_citations = _manuscript_line_citation_keys(manuscript_text)
        literals_by_key: dict[str, set[str]] = {}
        for claim in claims:
            declared = _claim_declared_literals(claim)
            for key in _claim_citation_keys(claim):
                literals_by_key.setdefault(key, set()).update(declared)
        for item in _reportable_numeric_literals(manuscript_text):
            cited_keys = line_citations.get(item["line"], set())
            bound = any(
                item["normalized"] in literals_by_key.get(key, set())
                for key in cited_keys
            )
            if not bound:
                unregistered_numerics.append(item)
                failures.append(
                    _science_failure(
                        "unregistered_manuscript_numeric",
                        subject=f"{manuscript.as_posix()}:{item['line']}",
                        field="numeric_literal",
                        actual=item["literal"],
                    )
                )
        if not claims and len(_manuscript_words(manuscript_text)) >= 50:
            failures.append(
                _science_failure(
                    "substantive_manuscript_without_registered_claims",
                    subject=manuscript.as_posix(),
                )
            )

    return GateResult(
        ok=not failures,
        details={
            "status": "no_claims" if not claims and not failures else "ok",
            "claim_count": len(claims),
            "unregistered_manuscript_numeric_count": len(unregistered_numerics),
            "failures": failures,
        },
    )


_REPORTABLE_UNIT_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:"
    r"\s*%(?!\w)"
    r"|\s*-?\s*(?:ETH|USD|EUR|GBP|bps?|basis\s+points?|x)\b"
    # count/time nouns tolerate ONE optional descriptor word ("12,563
    # rent-component rows", "7 consecutive days") and a hyphen boundary
    # ("14-day") so a reported quantity can't dodge the ledger by inserting an
    # adjective or a hyphen between the number and the noun.
    r"|[\s-]+(?:[A-Za-z][\w-]*[\s-]+)?(?:rollup-days?|rollups?|observations?|rows?|dates?|"
    r"days?|weeks?|months?|years?|instances?|seeds?|runs?)\b"
    r")",
    flags=re.IGNORECASE,
)
_REPORTABLE_DECIMAL_RE = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])")


def _normalize_reportable_numeric(value: object) -> str:
    """Normalize a reportable literal to `<value>|<unit-class>` so binding is
    both descriptor-independent AND unit-strict: `1,559 daily observations` and
    `1,559 dates` differ (different head noun), while `14 ETH` can NEVER bind to
    a registered `14 rollups` (different unit) — the numeric-core-only laundering
    hole. The unit class is `%`, a currency, `x`, or the trailing head noun
    (last alphabetic token, so an intervening descriptor is ignored); a bare
    number has an empty class."""
    text = str(value)
    match = re.match(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%?)", text)
    if match is None:
        return re.sub(r"\s+", "", text).replace(",", "").casefold()
    number = match.group(1).replace(",", "")
    if match.group(2) == "%":
        return f"{number}|%"
    tail_words = re.findall(r"[A-Za-z]+", text[match.end():])
    unit = tail_words[-1].casefold() if tail_words else ""
    return f"{number}|{unit}"


_CITATION_KEY_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9_:.\-]*)")


def _manuscript_line_citation_keys(text: str) -> dict[int, set[str]]:
    """Map 1-based line number -> citation keys cited on that line
    (`[@key]`, `[@k1; @k2]`), skipping frontmatter and fenced code."""
    out: dict[int, set[str]] = {}
    in_fence = False
    in_frontmatter = text.startswith("---\n")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if line_number == 1 and in_frontmatter:
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        keys: set[str] = set()
        for bracket in re.findall(r"\[@[^\]]+\]", raw_line):
            keys.update(_CITATION_KEY_RE.findall(bracket))
        if keys:
            out[line_number] = keys
    return out


def _claim_citation_keys(claim: dict[str, object]) -> set[str]:
    keys: set[str] = set()
    single = claim.get("citation_key")
    if isinstance(single, str) and single.strip():
        keys.add(single.strip())
    multiple = claim.get("citation_keys")
    if isinstance(multiple, list):
        keys.update(k.strip() for k in multiple if isinstance(k, str) and k.strip())
    return keys


def _claim_declared_literals(claim: dict[str, object]) -> set[str]:
    """Normalized reportable numerics a claim owns — from its statement and its
    explicit `manuscript_numeric_literals`. These bind ONLY on manuscript lines
    that cite one of the claim's citation keys (occurrence scoping)."""
    literals: set[str] = set()
    statement = claim.get("statement")
    if isinstance(statement, str):
        literals.update(item["normalized"] for item in _reportable_numeric_literals(statement))
    explicit = claim.get("manuscript_numeric_literals")
    if isinstance(explicit, list):
        literals.update(
            _normalize_reportable_numeric(value)
            for value in explicit
            if isinstance(value, (str, int, float)) and not isinstance(value, bool)
        )
    return literals


def _reportable_numeric_literals(text: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    in_fence = False
    in_frontmatter = text.startswith("---\n")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if line_number == 1 and in_frontmatter:
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or re.search(r"\{\{<\s*include\b", raw_line):
            continue
        line = re.sub(r"\{\{.*?\}\}", " ", raw_line)
        line = re.sub(r"\[@[^\]]+\]", " ", line)
        # fig-alt / alt accessibility text describes an image; it is not a
        # reported-statistic channel, so its numerics are not manuscript claims.
        line = re.sub(r"""\b(?:fig-)?alt\s*=\s*(?:"[^"]*"|'[^']*')""", " ", line)
        line = re.sub(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", " ", line)
        line = line.replace("`", "")
        matches = list(_REPORTABLE_UNIT_RE.finditer(line))
        occupied = [(match.start(), match.end()) for match in matches]
        matches.extend(
            match
            for match in _REPORTABLE_DECIMAL_RE.finditer(line)
            if not any(start <= match.start() and match.end() <= end for start, end in occupied)
        )
        for match in sorted(matches, key=lambda item: item.start()):
            literal = match.group(0).strip().strip("`")
            results.append(
                {
                    "literal": literal,
                    "normalized": _normalize_reportable_numeric(literal),
                    "line": line_number,
                    "column": match.start() + 1,
                }
            )
    return results


def _parse_utc_z(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    start_pattern = re.compile(r"@\w+\s*([({])\s*", flags=re.IGNORECASE)
    cursor = 0
    while (match := start_pattern.search(text, cursor)) is not None:
        opener = match.group(1)
        closer = ")" if opener == "(" else "}"
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == opener:
                depth += 1
            elif text[index] == closer:
                depth -= 1
            index += 1
        block = text[match.end() : index - 1 if depth == 0 else len(text)]
        cursor = max(index, match.end())
        if "," not in block:
            continue
        citekey, fields_text = block.split(",", 1)
        citekey = citekey.strip()
        if not citekey:
            continue
        fields: dict[str, str] = {}
        for field in ("title", "doi", "note"):
            field_match = re.search(
                rf"(?ims)^\s*{field}\s*=\s*(\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}}|\"[^\"]*\"|[^,\n]+)",
                fields_text,
            )
            if field_match is None:
                continue
            value = field_match.group(1).strip().strip(',').strip()
            while len(value) >= 2 and (
                (value[0] == "{" and value[-1] == "}")
                or (value[0] == '"' and value[-1] == '"')
            ):
                value = value[1:-1].strip()
            fields[field] = value
        entries[citekey] = fields
    return entries


def _normalized_bib_identity(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _normalized_doi(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().casefold()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    return normalized


def _bib_local_path(citekey: str, fields: dict[str, str]) -> str | None:
    if citekey.startswith("local:"):
        return citekey.removeprefix("local:")
    note = fields.get("note")
    if not isinstance(note, str):
        return None
    match = re.search(r"(?:^|\s)Path:\s*([^\s]+)", note)
    return match.group(1).rstrip(".,;)") if match is not None else None


def gate_citation_integrity() -> GateResult:
    """Check bibliography keys against committed, fresh, clean offline snapshots."""
    manuscript = Path("reports/paper/index.qmd")
    manuscript_bibliography = manuscript.parent / "references.bib"
    bibliography = (
        manuscript_bibliography
        if manuscript.is_file() and manuscript_bibliography.is_file()
        else Path("references.bib")
    )
    if not bibliography.is_file():
        return GateResult(
            ok=True,
            details={"status": "no_bibliography", "skipped": True, "failures": []},
        )

    text = _read_text(bibliography)
    entries = _bib_entries(text)
    citekeys = re.findall(r"@\w+\s*[({]\s*([^,\s]+)\s*,", text, flags=re.IGNORECASE)
    failures: list[dict[str, object]] = []
    duplicates = sorted(key for key in set(citekeys) if citekeys.count(key) > 1)
    for citekey in duplicates:
        failures.append(_science_failure("duplicate_bibliography_key", subject=citekey))

    local_paths = {
        key: path
        for key in citekeys
        if (path := _bib_local_path(key, entries.get(key, {}))) is not None
    }
    local_keys = set(local_paths)
    remote_keys = set(citekeys) - local_keys
    for citekey in sorted(local_keys):
        raw_path = local_paths[citekey]
        path = _safe_repo_relative_path(raw_path)
        if path is None or path.is_symlink() or not path.is_file() or not _git_path_is_tracked(
            path.as_posix(), Path.cwd()
        ):
            failures.append(
                _science_failure(
                    "local_citation_missing",
                    subject=citekey,
                    expected="existing tracked repo-relative regular artifact",
                    actual=raw_path,
                )
            )

    citation_root = Path("data/citations")
    dated_dirs: list[Path] = []
    for path in citation_root.glob("????-??-??"):
        if not path.is_dir():
            continue
        try:
            dt.date.fromisoformat(path.name)
        except ValueError:
            failures.append(
                _science_failure("invalid_citation_snapshot_date_dir", subject=path.as_posix())
            )
            continue
        dated_dirs.append(path)
    dated_dirs.sort()
    latest_dir = dated_dirs[-1] if dated_dirs else None
    snapshot_paths = sorted(latest_dir.glob("*.json")) if latest_dir is not None else []
    snapshot_keys = {path.stem for path in snapshot_paths}
    for missing in sorted(remote_keys - snapshot_keys):
        failures.append(_science_failure("missing_citation_snapshot", subject=missing))
    for extra in sorted(snapshot_keys - remote_keys):
        failures.append(_science_failure("extra_citation_snapshot", subject=extra))

    as_of: dt.date | None = None
    if remote_keys:
        as_of_path = citation_root / "AS_OF"
        try:
            as_of = dt.date.fromisoformat(_read_text(as_of_path).strip())
        except (OSError, ValueError):
            failures.append(
                _science_failure(
                    "invalid_citation_as_of",
                    subject=as_of_path.as_posix(),
                    expected="YYYY-MM-DD",
                )
            )
        if latest_dir is None:
            failures.append(
                _science_failure("missing_citation_snapshot_directory", subject=citation_root.as_posix())
            )
        elif as_of is not None and dt.date.fromisoformat(latest_dir.name) > as_of:
            failures.append(
                _science_failure(
                    "citation_snapshot_directory_after_as_of",
                    subject=latest_dir.as_posix(),
                    expected=f"on or before {as_of.isoformat()}",
                )
            )

    staleness_days = 90
    try:
        framework = json.loads(Path("contracts/framework.json").read_text(encoding="utf-8"))
        configured = framework.get("citation_policy", {}).get("staleness_days")
        if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
            staleness_days = configured
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    for snapshot_path in snapshot_paths:
        snapshot, error = _load_json_file(snapshot_path)
        subject = snapshot_path.stem
        if error is not None or snapshot is None:
            failures.append(
                _science_failure(
                    "invalid_citation_snapshot", subject=subject, actual=error
                )
            )
            continue
        required = {
            "schema_version",
            "citekey",
            "title",
            "source",
            "retrieved_at_utc",
            "retrieval_sha256",
            "retrieval_payload",
            "resolved",
            "retraction_status",
            "url_resolves",
        }
        for field in sorted(required - set(snapshot)):
            failures.append(
                _science_failure("citation_snapshot_missing_field", subject=subject, field=field)
            )
        if snapshot.get("schema_version") != CITATION_SNAPSHOT_SCHEMA_VERSION:
            failures.append(
                _science_failure(
                    "invalid_citation_snapshot_schema",
                    subject=subject,
                    field="schema_version",
                    expected=CITATION_SNAPSHOT_SCHEMA_VERSION,
                    actual=snapshot.get("schema_version"),
                )
            )
        if snapshot.get("citekey") != subject:
            failures.append(
                _science_failure(
                    "citation_snapshot_key_mismatch",
                    subject=subject,
                    field="citekey",
                    expected=subject,
                    actual=snapshot.get("citekey"),
                )
            )
        if snapshot.get("source") not in {"crossref", "openalex", "s2"}:
            failures.append(
                _science_failure("invalid_citation_source", subject=subject, field="source")
            )
        if not isinstance(snapshot.get("title"), str) or not snapshot.get("title", "").strip():
            failures.append(
                _science_failure("invalid_citation_title", subject=subject, field="title")
            )
        if not isinstance(snapshot.get("retrieval_sha256"), str) or re.fullmatch(
            r"[0-9a-f]{64}", snapshot.get("retrieval_sha256", "")
        ) is None:
            failures.append(
                _science_failure(
                    "invalid_retrieval_sha256", subject=subject, field="retrieval_sha256"
                )
            )
        elif "retrieval_payload" in snapshot:
            actual_retrieval_sha = hashlib.sha256(
                _canonical_json_bytes(snapshot.get("retrieval_payload"))
            ).hexdigest()
            if snapshot.get("retrieval_sha256") != actual_retrieval_sha:
                failures.append(
                    _science_failure(
                        "retrieval_sha256_mismatch",
                        subject=subject,
                        field="retrieval_sha256",
                        expected=actual_retrieval_sha,
                        actual=snapshot.get("retrieval_sha256"),
                    )
                )
        bib_entry = entries.get(subject, {})
        bib_doi = _normalized_doi(bib_entry.get("doi"))
        snapshot_doi = _normalized_doi(snapshot.get("doi"))
        if bib_doi is not None or snapshot_doi is not None:
            if bib_doi != snapshot_doi:
                failures.append(
                    _science_failure(
                        "citation_snapshot_bib_identity_mismatch",
                        subject=subject,
                        field="doi",
                        expected=bib_doi,
                        actual=snapshot_doi,
                    )
                )
        elif _normalized_bib_identity(bib_entry.get("title")) != _normalized_bib_identity(
            snapshot.get("title")
        ):
            failures.append(
                _science_failure(
                    "citation_snapshot_bib_identity_mismatch",
                    subject=subject,
                    field="title",
                    expected=bib_entry.get("title"),
                    actual=snapshot.get("title"),
                )
            )
        retrieved_at = _parse_utc_z(snapshot.get("retrieved_at_utc"))
        if retrieved_at is None:
            failures.append(
                _science_failure(
                    "invalid_citation_retrieved_at", subject=subject, field="retrieved_at_utc"
                )
            )
        elif as_of is not None:
            age = (as_of - retrieved_at.date()).days
            if age < 0 or age > staleness_days:
                failures.append(
                    _science_failure(
                        "citation_snapshot_stale",
                        subject=subject,
                        field="retrieved_at_utc",
                        expected=f"0..{staleness_days} days before AS_OF {as_of.isoformat()}",
                        actual=age,
                    )
                )
        if snapshot.get("resolved") is not True:
            failures.append(_science_failure("citation_unresolved", subject=subject, field="resolved"))
        if snapshot.get("retraction_status") != "none":
            failures.append(
                _science_failure(
                    "citation_retraction_status_not_clean",
                    subject=subject,
                    field="retraction_status",
                    expected="none",
                    actual=snapshot.get("retraction_status"),
                )
            )
        if snapshot.get("url_resolves") is not True:
            failures.append(
                _science_failure("citation_url_unresolved", subject=subject, field="url_resolves")
            )

    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "bibliography_count": len(citekeys),
            "snapshot_count": len(snapshot_paths),
            "snapshot_directory": latest_dir.as_posix() if latest_dir is not None else None,
            "as_of": as_of.isoformat() if as_of is not None else None,
            "staleness_days": staleness_days,
            "failures": failures,
        },
    )


def _gate_empirical_etl_decision_log() -> GateResult:
    """Bind logged ETL discretion to locked 2a clauses and catch declared zero-fill."""
    failures: list[dict[str, object]] = []
    lock_path = Path(PREREG_PHASE_FILES["2a"])
    lock, lock_error = load_prereg_lock(lock_path, expected_phase="2a")
    if lock_error is not None or lock is None:
        failures.append(
            _science_failure(
                "invalid_data_construction_lock",
                subject=lock_path.as_posix(),
                actual=lock_error,
            )
        )
        lock = None
    elif lock.get("status") == "locked" and lock.get("active") is not True:
        failures.append(
            _science_failure(
                "data_construction_lock_hash_mismatch",
                subject=lock_path.as_posix(),
                expected=lock.get("body_sha256"),
                actual=lock.get("locked_sha256"),
            )
        )
    active_lock = lock if lock is not None and lock.get("active") is True else None
    body = str(active_lock.get("body", "")) if active_lock is not None else ""
    clause_ids = set(
        re.findall(r"^###\s+([A-Za-z0-9][A-Za-z0-9._-]*)\s*$", body, flags=re.MULTILINE)
    )

    manifest_paths = sorted(Path("data/processed_manifest").glob("*.json"))
    decision_log_count = 0
    for manifest_path in manifest_paths:
        payload, error = _load_json_file(manifest_path)
        if error is not None or payload is None:
            failures.append(
                _science_failure(
                    "invalid_processed_manifest_for_decision_log",
                    subject=manifest_path.as_posix(),
                    actual=error,
                )
            )
            continue
        decision_log = payload.get("decision_log")
        if decision_log is not None:
            if not isinstance(decision_log, list):
                failures.append(
                    _science_failure(
                        "decision_log_not_list",
                        subject=manifest_path.as_posix(),
                        field="decision_log",
                        actual=decision_log,
                    )
                )
            else:
                decision_log_count += len(decision_log)
                for index, decision in enumerate(decision_log):
                    subject = f"{manifest_path.as_posix()}:decision_log[{index}]"
                    if not isinstance(decision, dict):
                        failures.append(
                            _science_failure("decision_log_entry_not_object", subject=subject)
                        )
                        continue
                    for field in ("clause_id", "choice", "rationale"):
                        value = decision.get(field)
                        if not isinstance(value, str) or not value.strip():
                            failures.append(
                                _science_failure(
                                    "invalid_decision_log_field",
                                    subject=subject,
                                    field=field,
                                    actual=value,
                                )
                            )
                    clause_id = decision.get("clause_id")
                    if isinstance(clause_id, str) and clause_id.strip() not in clause_ids:
                        failures.append(
                            _science_failure(
                                "unknown_locked_protocol_clause",
                                subject=subject,
                                field="clause_id",
                                expected=sorted(clause_ids),
                                actual=clause_id,
                            )
                        )

        transform = payload.get("transform")
        transform = transform if isinstance(transform, dict) else {}
        zero_fill = payload.get("zero_fill_columns", transform.get("zero_fill_columns"))
        coverage = payload.get("coverage_flag_columns", transform.get("coverage_flag_columns"))
        if zero_fill is not None:
            if not isinstance(zero_fill, list) or not all(
                isinstance(item, str) and item.strip() for item in zero_fill
            ):
                failures.append(
                    _science_failure(
                        "invalid_zero_fill_columns",
                        subject=manifest_path.as_posix(),
                        field="zero_fill_columns",
                        actual=zero_fill,
                    )
                )
            else:
                coverage_set = (
                    {item.strip() for item in coverage if isinstance(item, str) and item.strip()}
                    if isinstance(coverage, list)
                    else set()
                )
                for column in zero_fill:
                    if column.strip() not in coverage_set:
                        failures.append(
                            _science_failure(
                                "zero_fill_without_coverage_flag",
                                subject=manifest_path.as_posix(),
                                field="zero_fill_columns",
                                expected=f"matching coverage_flag_columns entry for {column.strip()}",
                                actual=sorted(coverage_set),
                            )
                        )

    return GateResult(
        ok=not failures,
        details={
            "status": "no_decision_logs" if decision_log_count == 0 and not failures else "ok",
            "manifest_count": len(manifest_paths),
            "decision_log_count": decision_log_count,
            "locked_clause_count": len(clause_ids),
            "completeness_audited": False,
            "failures": failures,
        },
    )


def _gate_modeling_etl_decision_log() -> GateResult:
    failures: list[dict[str, object]] = []
    manifests: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(Path("contracts/instances").glob("*.json")):
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            continue
        if payload.get("decision_log"):
            manifests.append((path, payload))
    if not manifests:
        return GateResult(
            ok=True,
            details={
                "status": "no_instance_decision_logs",
                "manifest_count": 0,
                "decision_log_count": 0,
                "failures": [],
            },
        )

    spec, spec_path, _lock, spec_failures = _active_experiment_spec()
    failures.extend(spec_failures)
    dimensions = (
        spec.get("grid", {}).get("dimensions", {})
        if isinstance(spec, dict) and isinstance(spec.get("grid"), dict)
        else {}
    )
    clause_ids = {
        "seeds",
        "solver",
        "budget",
        "convergence_tolerance",
        "sweep_survival_criterion",
    }
    if isinstance(dimensions, dict):
        clause_ids.update(f"grid.dimensions.{name}" for name in dimensions)
    decision_count = 0
    for path, payload in manifests:
        decision_log = payload.get("decision_log")
        if not isinstance(decision_log, list):
            failures.append(
                _science_failure(
                    "instance_decision_log_not_list",
                    subject=path.as_posix(),
                    field="decision_log",
                )
            )
            continue
        decision_count += len(decision_log)
        for index, decision in enumerate(decision_log):
            subject = f"{path.as_posix()}:decision_log[{index}]"
            if not isinstance(decision, dict):
                failures.append(_science_failure("instance_decision_not_object", subject=subject))
                continue
            clause_id = decision.get("clause_id")
            if clause_id not in clause_ids:
                failures.append(
                    _science_failure(
                        "unknown_locked_experiment_clause",
                        subject=subject,
                        field="clause_id",
                        expected=sorted(clause_ids),
                        actual=clause_id,
                    )
                )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "experiment_spec": spec_path.as_posix() if spec_path is not None else None,
            "manifest_count": len(manifests),
            "decision_log_count": decision_count,
            "locked_clause_count": len(clause_ids),
            "completeness_audited": False,
            "failures": failures,
        },
    )


def gate_etl_decision_log(*, form: str | None = None) -> GateResult:
    """Run empirical, modeling, or union decision-log conformance."""
    mode = _parse_project_mode(Path("contracts/project.yaml")) or "empirical"
    selected = form or ("modeling" if mode == "modeling" else "union" if mode == "hybrid" else "empirical")
    if selected == "empirical":
        return _gate_empirical_etl_decision_log()
    if selected == "modeling":
        return _gate_modeling_etl_decision_log()
    empirical = _gate_empirical_etl_decision_log()
    modeling = _gate_modeling_etl_decision_log()
    return GateResult(
        ok=empirical.ok and modeling.ok,
        details={
            "status": "ok",
            "forms": {"empirical": empirical.details, "modeling": modeling.details},
            "failures": list(empirical.details.get("failures", []))
            + list(modeling.details.get("failures", [])),
        },
    )


def gate_rigor_sections() -> GateResult:
    """Require deterministic rigor headings on v2 analysis/writing and manuscript surfaces."""
    required_headings = (
        "## Evidence table",
        "## Alternative explanations considered",
        "## Uncertainty statement",
    )
    failures: list[dict[str, object]] = []
    checked: list[str] = []
    try:
        contract = load_framework_contract()
    except ValueError as exc:
        return GateResult(ok=False, details={"failures": [str(exc)]})

    for path in _iter_task_files(contract):
        text = _read_text(path)
        frontmatter = _parse_task_frontmatter(text)
        if (
            frontmatter is None
            or frontmatter.get("task_schema") != TASK_SCHEMA_VERSION
            or frontmatter.get("task_kind") not in {"analysis", "writing"}
        ):
            continue
        checked.append(path.as_posix())
        for heading in required_headings:
            if not _section_has_content(text, heading):
                failures.append(
                    _science_failure(
                        "missing_or_empty_rigor_section",
                        subject=path.as_posix(),
                        field=heading.removeprefix("## "),
                    )
                )

    manuscript = Path("reports/paper/index.qmd")
    if manuscript.is_file():
        checked.append(manuscript.as_posix())
        text = _read_text(manuscript)
        for heading in required_headings:
            if not _section_has_content(text, heading):
                failures.append(
                    _science_failure(
                        "missing_or_empty_rigor_section",
                        subject=manuscript.as_posix(),
                        field=heading.removeprefix("## "),
                    )
                )

    return GateResult(
        ok=not failures,
        details={
            "status": "no_applicable_surfaces" if not checked else "ok",
            "checked": checked,
            "failures": failures,
        },
    )


def _verify_content_entry(
    entry: object,
    *,
    subject: str,
    field: str,
    repo: Path = Path("."),
    require_bytes: bool = False,
    required_prefix: str | None = None,
) -> list[dict[str, object]]:
    if not isinstance(entry, dict):
        return [_science_failure("invalid_content_binding", subject=subject, field=field, actual=entry)]
    path = _safe_repo_relative_path(entry.get("path"))
    expected_sha = entry.get("sha256")
    if path is None:
        return [
            _science_failure(
                "invalid_content_binding_path",
                subject=subject,
                field=f"{field}.path",
                actual=entry.get("path"),
            )
        ]
    if required_prefix is not None and not _path_matches_prefix(path.as_posix(), required_prefix):
        return [
            _science_failure(
                "content_binding_path_outside_required_prefix",
                subject=subject,
                field=f"{field}.path",
                expected=required_prefix,
                actual=path.as_posix(),
            )
        ]
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        return [
            _science_failure(
                "invalid_content_binding_sha256",
                subject=subject,
                field=f"{field}.sha256",
                actual=expected_sha,
            )
        ]
    expected_bytes = entry.get("bytes")
    if require_bytes and (
        not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0
    ):
        return [
            _science_failure(
                "invalid_content_binding_bytes",
                subject=subject,
                field=f"{field}.bytes",
                actual=expected_bytes,
            )
        ]
    repo = repo.resolve()
    disk_path = repo / path
    try:
        resolved = disk_path.resolve(strict=True)
        resolved.relative_to(repo)
    except (FileNotFoundError, OSError, ValueError):
        return [
            _science_failure(
                "content_binding_target_missing",
                subject=subject,
                field=field,
                expected=path.as_posix(),
            )
        ]
    if disk_path.is_symlink() or not resolved.is_file():
        return [
            _science_failure(
                "content_binding_target_not_regular_file",
                subject=subject,
                field=field,
                actual=path.as_posix(),
            )
        ]
    if not _git_path_is_tracked(path.as_posix(), repo):
        return [
            _science_failure(
                "content_binding_target_not_git_tracked",
                subject=subject,
                field=field,
                actual=path.as_posix(),
            )
        ]
    actual_sha, actual_bytes = _sha256_and_bytes(resolved)
    failures: list[dict[str, object]] = []
    if actual_sha != expected_sha:
        failures.append(
            _science_failure(
                "content_binding_sha256_mismatch",
                subject=subject,
                field=field,
                expected=expected_sha,
                actual=actual_sha,
            )
        )
    if require_bytes and actual_bytes != expected_bytes:
        failures.append(
            _science_failure(
                "content_binding_bytes_mismatch",
                subject=subject,
                field=field,
                expected=expected_bytes,
                actual=actual_bytes,
            )
        )
    return failures


def _validation_artifact_status(path: Path) -> str | None:
    try:
        text = _read_text(path)
    except (OSError, UnicodeDecodeError):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("status", "overall_status", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            if isinstance(value, dict):
                nested = value.get("status")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip().lower()
    match = re.search(
        r"^(?:[-*]\s*)?(?:status|overall_status)\s*:\s*['\"]?([^'\"\s]+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip().lower() if match is not None else None


def gate_instance_manifest_conformance() -> GateResult:
    """Validate both v1 variants and recompute their complete content bindings."""
    paths = sorted(Path("contracts/instances").glob("*.json"))
    failures: list[dict[str, object]] = []
    variants = {"bridge": 0, "synthetic": 0}
    schema_path = Path("contracts/schemas/instance_manifest_v1.json")
    for path in paths:
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            failures.append(
                _science_failure("invalid_instance_manifest_json", subject=path.as_posix(), actual=error)
            )
            continue
        variant = "bridge" if "source_processed_manifests" in payload else "synthetic"
        variants[variant] += 1
        for issue in _schema_failures(payload, schema_path):
            failures.append(
                _science_failure(
                    "instance_manifest_schema_violation",
                    subject=path.as_posix(),
                    field=str(issue.get("path")),
                    actual=issue,
                )
            )
        if variant == "bridge":
            sources = payload.get("source_processed_manifests")
            if isinstance(sources, list):
                for index, entry in enumerate(sources):
                    failures.extend(
                        _verify_content_entry(
                            entry,
                            subject=path.as_posix(),
                            field=f"source_processed_manifests[{index}]",
                            required_prefix="data/processed_manifest/",
                        )
                    )
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            for index, entry in enumerate(outputs):
                failures.extend(
                    _verify_content_entry(
                        entry,
                        subject=path.as_posix(),
                        field=f"outputs[{index}]",
                        require_bytes=True,
                    )
                )
    require_instance, _, requirement_subjects = _modeling_manifest_requirements()
    if require_instance and not paths:
        failures.append(
            _science_failure(
                "required_instance_manifest_missing",
                subject=",".join(requirement_subjects) or "modeling_claims",
            )
        )
    return GateResult(
        ok=not failures,
        details={
            "status": "no_instance_manifests" if not paths else "ok",
            "manifest_count": len(paths),
            "variants": variants,
            "failures": failures,
        },
    )


def _experiment_manifests() -> tuple[list[tuple[Path, dict[str, object]]], list[dict[str, object]]]:
    records: list[tuple[Path, dict[str, object]]] = []
    failures: list[dict[str, object]] = []
    schema_path = Path("contracts/schemas/experiment_manifest_v1.json")
    for path in sorted(Path("reports/models").glob("*.json")):
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            failures.append(
                _science_failure("invalid_experiment_manifest_json", subject=path.as_posix(), actual=error)
            )
            continue
        if payload.get("schema_version") != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            continue
        records.append((path, payload))
        for issue in _schema_failures(payload, schema_path):
            failures.append(
                _science_failure(
                    "experiment_manifest_schema_violation",
                    subject=path.as_posix(),
                    field=str(issue.get("path")),
                    actual=issue,
                )
            )
    return records, failures


def _modeling_manifest_requirements() -> tuple[bool, bool, list[str]]:
    require_instance = False
    require_experiment = False
    subjects: list[str] = []
    try:
        contract = load_framework_contract()
        tasks, _ = _collect_tasks(contract)
    except ValueError:
        tasks = {}
    claims, _ = _load_claim_ledger()
    claims_by_task: dict[str, set[str]] = {}
    unbound_modeling_claim_types: set[str] = set()
    for claim in claims:
        claim_type = claim.get("type")
        if claim_type not in {"computational", "counterfactual"}:
            continue
        task_id = _claim_task_id(claim)
        if task_id is None:
            unbound_modeling_claim_types.add(str(claim_type))
        else:
            claims_by_task.setdefault(task_id, set()).add(str(claim_type))
    if unbound_modeling_claim_types:
        require_experiment = True
    if "counterfactual" in unbound_modeling_claim_types:
        require_instance = True
    for task in tasks.values():
        if task.state not in {"ready_for_review", "done"} or task.task_kind not in {
            "model",
            "bridge",
        }:
            continue
        claim_types = claims_by_task.get(task.task_id, set())
        instance_output = any(
            _path_matches_prefix(output, "contracts/instances/") for output in task.outputs
        )
        experiment_output = any(
            _path_matches_prefix(output, "reports/models/") for output in task.outputs
        )
        if instance_output or "counterfactual" in claim_types:
            require_instance = True
            subjects.append(task.task_id)
        if experiment_output or claim_types & {"computational", "counterfactual"}:
            require_experiment = True
            subjects.append(task.task_id)
    return require_instance, require_experiment, sorted(set(subjects))


def gate_seed_budget_lock() -> GateResult:
    manifests, failures = _experiment_manifests()
    if not manifests:
        return GateResult(
            ok=not failures,
            details={"status": "no_experiment_manifests", "manifest_count": 0, "failures": failures},
        )
    spec, spec_path, lock, spec_failures = _active_experiment_spec()
    failures.extend(spec_failures)
    seeds = spec.get("seeds", []) if isinstance(spec, dict) else []
    raw_budgets = spec.get("budget", []) if isinstance(spec, dict) else []
    budgets = raw_budgets if isinstance(raw_budgets, list) else [raw_budgets]
    for path, manifest in manifests:
        if manifest.get("seed") not in seeds:
            failures.append(
                _science_failure(
                    "seed_outside_active_lock",
                    subject=path.as_posix(),
                    field="seed",
                    expected=seeds,
                    actual=manifest.get("seed"),
                )
            )
        if manifest.get("budget") not in budgets:
            failures.append(
                _science_failure(
                    "budget_outside_active_lock",
                    subject=path.as_posix(),
                    field="budget",
                    expected=budgets,
                    actual=manifest.get("budget"),
                )
            )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "manifest_count": len(manifests),
            "active_lock_a_sha256": lock.get("body_sha256") if lock is not None else None,
            "experiment_spec": spec_path.as_posix() if spec_path is not None else None,
            "failures": failures,
        },
    )


def _claim_experiment_paths(claim: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    direct = claim.get("experiment_manifest")
    if isinstance(direct, str):
        candidate = _safe_repo_relative_path(direct)
        if candidate is not None:
            paths.append(candidate)
    elif isinstance(direct, dict):
        candidate = _safe_repo_relative_path(direct.get("path"))
        if candidate is not None:
            paths.append(candidate)
    model_artifact = claim.get("model_artifact")
    if isinstance(model_artifact, dict):
        candidate = _safe_repo_relative_path(model_artifact.get("path"))
        if candidate is not None and _path_matches_prefix(
            candidate.as_posix(), "reports/models/"
        ):
            paths.append(candidate)
    artifacts = claim.get("supporting_artifacts")
    if isinstance(artifacts, list):
        for entry in artifacts:
            candidate = _safe_repo_relative_path(entry.get("path")) if isinstance(entry, dict) else None
            if candidate is not None and _path_matches_prefix(candidate.as_posix(), "reports/models/"):
                paths.append(candidate)
    return sorted(set(paths))


def _dispersion_failures(path: Path, manifest: dict[str, object]) -> list[dict[str, object]]:
    subject = path.as_posix()
    direct = manifest.get("dispersion_artifact")
    if direct is not None:
        return _verify_content_entry(direct, subject=subject, field="dispersion_artifact")
    outputs = manifest.get("outputs")
    candidates: list[object] = []
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if "dispersion" in str(key).lower():
                candidates.append(value)
    elif isinstance(outputs, list):
        for value in outputs:
            marker = ""
            if isinstance(value, dict):
                marker = " ".join(str(value.get(key, "")) for key in ("role", "kind", "type", "path"))
            elif isinstance(value, str):
                marker = value
            if "dispersion" in marker.lower():
                candidates.append(value)
    if not candidates:
        return [_science_failure("missing_per_instance_dispersion_artifact", subject=subject)]
    failures: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            candidate_path = _safe_repo_relative_path(candidate)
            if candidate_path is None or not candidate_path.is_file():
                failures.append(
                    _science_failure(
                        "missing_per_instance_dispersion_artifact",
                        subject=subject,
                        field=f"outputs[{index}]",
                        actual=candidate,
                    )
                )
        elif isinstance(candidate, dict):
            failures.extend(
                _verify_content_entry(candidate, subject=subject, field=f"outputs[{index}]")
            )
    return failures


def gate_gap_convergence() -> GateResult:
    claims, claim_failures = _load_claim_ledger()
    failures: list[dict[str, object]] = []
    if not claims and claim_failures:
        failures.extend(claim_failures)
    records, manifest_failures = _experiment_manifests()
    failures.extend(manifest_failures)
    _, require_experiment, requirement_subjects = _modeling_manifest_requirements()
    if require_experiment and not records:
        failures.append(
            _science_failure(
                "required_experiment_manifest_missing",
                subject=",".join(requirement_subjects) or "modeling_claims",
            )
        )
    by_path = {path: payload for path, payload in records}
    relevant_claims = [
        claim for claim in claims if claim.get("type") in {"computational", "counterfactual"}
    ]
    checked: set[Path] = set()
    for index, claim in enumerate(relevant_claims):
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        experiment_paths = _claim_experiment_paths(claim)
        if not experiment_paths:
            failures.append(
                _science_failure("computational_claim_missing_experiment_manifest", subject=subject)
            )
        for path in experiment_paths:
            manifest = by_path.get(path)
            if manifest is None:
                failures.append(
                    _science_failure(
                        "claim_experiment_manifest_not_registered",
                        subject=subject,
                        actual=path.as_posix(),
                    )
                )
                continue
            checked.add(path)
    spec: dict[str, object] | None = None
    spec_path: Path | None = None
    if relevant_claims:
        spec, spec_path, _, spec_failures = _active_experiment_spec()
        failures.extend(spec_failures)
    expected_solver = spec.get("solver") if isinstance(spec, dict) else None
    expected_solver_name = (
        expected_solver.get("name")
        if isinstance(expected_solver, dict)
        else expected_solver
    )
    expected_solver_version = (
        expected_solver.get("version") if isinstance(expected_solver, dict) else None
    )
    convergence_tolerance = (
        float(spec["convergence_tolerance"])
        if isinstance(spec, dict)
        and isinstance(spec.get("convergence_tolerance"), (int, float))
        and not isinstance(spec.get("convergence_tolerance"), bool)
        and math.isfinite(spec["convergence_tolerance"])
        else None
    )
    for path in sorted(checked):
        manifest = by_path[path]
        gap = manifest.get("optimality_gap")
        if not isinstance(gap, (int, float)) or isinstance(gap, bool) or not math.isfinite(float(gap)):
            failures.append(
                _science_failure("missing_or_invalid_optimality_gap", subject=path.as_posix(), field="optimality_gap")
            )
        if not isinstance(manifest.get("converged"), bool):
            failures.append(
                _science_failure("missing_or_invalid_converged", subject=path.as_posix(), field="converged")
            )
        elif manifest.get("converged") is not True:
            failures.append(
                _science_failure(
                    "computational_claim_experiment_not_converged",
                    subject=path.as_posix(),
                    field="converged",
                    expected=True,
                    actual=manifest.get("converged"),
                )
            )
        if expected_solver_name is not None and manifest.get("solver") != expected_solver_name:
            failures.append(
                _science_failure(
                    "solver_outside_active_lock",
                    subject=path.as_posix(),
                    field="solver",
                    expected=expected_solver_name,
                    actual=manifest.get("solver"),
                )
            )
        if (
            expected_solver_version is not None
            and manifest.get("solver_version") != expected_solver_version
        ):
            failures.append(
                _science_failure(
                    "solver_version_outside_active_lock",
                    subject=path.as_posix(),
                    field="solver_version",
                    expected=expected_solver_version,
                    actual=manifest.get("solver_version"),
                )
            )
        if (
            isinstance(gap, (int, float))
            and not isinstance(gap, bool)
            and math.isfinite(float(gap))
            and convergence_tolerance is not None
            and float(gap) > convergence_tolerance
        ):
            failures.append(
                _science_failure(
                    "optimality_gap_exceeds_active_lock",
                    subject=path.as_posix(),
                    field="optimality_gap",
                    expected=f"<={convergence_tolerance}",
                    actual=gap,
                )
            )
        failures.extend(_dispersion_failures(path, manifest))
    return GateResult(
        ok=not failures,
        details={
            "status": "no_computational_claims" if not relevant_claims else "ok",
            "claim_count": len(relevant_claims),
            "manifest_count": len(checked),
            "experiment_spec": spec_path.as_posix() if spec_path is not None else None,
            "failures": failures,
        },
    )


def gate_theoretical_falsification() -> GateResult:
    claims, claim_failures = _load_claim_ledger()
    failures: list[dict[str, object]] = []
    if not claims and claim_failures:
        failures.extend(claim_failures)
    checked = 0
    for index, claim in enumerate(claims):
        if claim.get("type") != "theoretical" or "falsification_spec" not in claim:
            continue
        checked += 1
        subject = str(claim.get("claim_id") or f"claims[{index}]")
        for diagnostic in evaluate_falsification_spec(claim.get("falsification_spec")):
            failures.append(
                _science_failure(
                    str(diagnostic.get("reason", "falsification_failed")),
                    subject=subject,
                    actual=diagnostic,
                )
            )
    return GateResult(
        ok=not failures,
        details={
            "status": "no_declared_falsification_specs" if checked == 0 else "ok",
            "claim_count": checked,
            "failures": failures,
        },
    )


def _canonical_cells(cells: object) -> list[str] | None:
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        return None
    return sorted(json.dumps(cell, sort_keys=True, separators=(",", ":")) for cell in cells)


def _sweep_cell_record(
    entry: object,
    *,
    subject: str,
    index: int,
    spec: dict[str, object] | None,
) -> tuple[dict[str, object] | None, bool, list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    if not isinstance(entry, dict) or not isinstance(entry.get("cell"), dict):
        return None, False, [
            _science_failure(
                "sweep_cell_not_content_bound",
                subject=subject,
                field=f"cells[{index}]",
                expected="{cell, experiment_manifest}",
                actual=entry,
            )
        ]
    cell = dict(entry["cell"])
    binding = entry.get("experiment_manifest")
    failures.extend(
        _verify_content_entry(
            binding,
            subject=subject,
            field=f"cells[{index}].experiment_manifest",
            required_prefix="reports/models/",
        )
    )
    manifest_path = (
        _safe_repo_relative_path(binding.get("path")) if isinstance(binding, dict) else None
    )
    manifest: dict[str, object] | None = None
    if manifest_path is not None and manifest_path.is_file():
        manifest, error = _load_json_file(manifest_path)
        if error is not None or manifest is None:
            failures.append(
                _science_failure(
                    "invalid_sweep_experiment_manifest",
                    subject=subject,
                    field=f"cells[{index}].experiment_manifest",
                    actual=error,
                )
            )
    if manifest is None:
        return cell, False, failures
    if manifest.get("schema_version") != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        failures.append(
            _science_failure(
                "invalid_sweep_experiment_manifest_schema",
                subject=subject,
                field=f"cells[{index}].experiment_manifest",
                actual=manifest.get("schema_version"),
            )
        )
    for issue in _schema_failures(manifest, Path("contracts/schemas/experiment_manifest_v1.json")):
        failures.append(
            _science_failure(
                "sweep_experiment_manifest_schema_violation",
                subject=subject,
                field=f"cells[{index}].experiment_manifest",
                actual=issue,
            )
        )
    for field in ("seed", "budget"):
        if manifest.get(field) != cell.get(field):
            failures.append(
                _science_failure(
                    "sweep_cell_manifest_mismatch",
                    subject=subject,
                    field=f"cells[{index}].{field}",
                    expected=cell.get(field),
                    actual=manifest.get(field),
                )
            )
    dimensions = (
        spec.get("grid", {}).get("dimensions", {})
        if isinstance(spec, dict) and isinstance(spec.get("grid"), dict)
        else {}
    )
    parameters = manifest.get("parameters")
    for name in (dimensions if isinstance(dimensions, dict) else {}):
        actual = parameters.get(name) if isinstance(parameters, dict) else None
        if actual != cell.get(name):
            failures.append(
                _science_failure(
                    "sweep_cell_manifest_mismatch",
                    subject=subject,
                    field=f"cells[{index}].{name}",
                    expected=cell.get(name),
                    actual=actual,
                )
            )
    locked_solver = spec.get("solver") if isinstance(spec, dict) else None
    solver_name = locked_solver.get("name") if isinstance(locked_solver, dict) else locked_solver
    solver_version = locked_solver.get("version") if isinstance(locked_solver, dict) else None
    if solver_name is not None and manifest.get("solver") != solver_name:
        failures.append(
            _science_failure(
                "solver_outside_active_lock",
                subject=subject,
                field=f"cells[{index}].solver",
                expected=solver_name,
                actual=manifest.get("solver"),
            )
        )
    if solver_version is not None and manifest.get("solver_version") != solver_version:
        failures.append(
            _science_failure(
                "solver_version_outside_active_lock",
                subject=subject,
                field=f"cells[{index}].solver_version",
                expected=solver_version,
                actual=manifest.get("solver_version"),
            )
        )
    tolerance = spec.get("convergence_tolerance") if isinstance(spec, dict) else None
    gap = manifest.get("optimality_gap")
    gap_is_acceptable = (
        isinstance(gap, (int, float))
        and not isinstance(gap, bool)
        and math.isfinite(float(gap))
        and isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and float(gap) <= float(tolerance)
    )
    if not gap_is_acceptable:
        failures.append(
            _science_failure(
                "sweep_cell_optimality_gap_outside_active_lock",
                subject=subject,
                field=f"cells[{index}].optimality_gap",
                expected=f"finite and <= {tolerance}",
                actual=gap,
            )
        )
    survived = (
        manifest.get("converged") is True
        and gap_is_acceptable
    )
    if manifest.get("converged") is not True:
        failures.append(
            _science_failure(
                "sweep_cell_not_converged",
                subject=subject,
                field=f"cells[{index}].experiment_manifest",
                actual=manifest.get("converged"),
            )
        )
    return cell, survived, failures


def gate_sweep_artifact() -> GateResult:
    claims, claim_failures = _load_claim_ledger()
    failures: list[dict[str, object]] = []
    if not claims and claim_failures:
        failures.extend(claim_failures)
    headline_claims = [
        claim
        for claim in claims
        if claim.get("type") in {"computational", "counterfactual"}
        and claim.get("headline") is True
    ]
    if not headline_claims:
        return GateResult(
            ok=not failures,
            details={"status": "no_headline_modeling_claims", "claim_count": 0, "failures": failures},
        )
    spec, spec_path, lock, spec_failures = _active_experiment_spec()
    failures.extend(spec_failures)
    try:
        expected_cells = enumerate_cells(spec) if spec is not None else []
    except ValueError as exc:
        failures.append(
            _science_failure(
                "locked_grid_not_enumerable",
                subject=spec_path.as_posix() if spec_path is not None else "lock_a",
                actual=str(exc),
            )
        )
        expected_cells = []
    expected_canonical = _canonical_cells(expected_cells)
    criterion = spec.get("sweep_survival_criterion") if isinstance(spec, dict) else None
    for index, claim in enumerate(headline_claims):
        claim_id = str(claim.get("claim_id") or f"claims[{index}]")
        path = Path("reports/models/sweeps") / f"{claim_id}.json"
        payload, error = _load_json_file(path)
        if error is not None or payload is None:
            failures.append(
                _science_failure("missing_sweep_artifact", subject=claim_id, expected=path.as_posix(), actual=error)
            )
            continue
        if payload.get("schema_version") != SWEEP_ARTIFACT_SCHEMA_VERSION:
            failures.append(
                _science_failure(
                    "invalid_sweep_artifact_schema",
                    subject=claim_id,
                    expected=SWEEP_ARTIFACT_SCHEMA_VERSION,
                    actual=payload.get("schema_version"),
                )
            )
        if payload.get("claim_id") != claim_id:
            failures.append(
                _science_failure("sweep_artifact_claim_mismatch", subject=claim_id, actual=payload.get("claim_id"))
            )
        bound_cells = payload.get("cells")
        actual_cells: list[dict[str, object]] = []
        computed_survival = 0
        if isinstance(bound_cells, list):
            for cell_index, entry in enumerate(bound_cells):
                cell, survived, cell_failures = _sweep_cell_record(
                    entry,
                    subject=claim_id,
                    index=cell_index,
                    spec=spec,
                )
                failures.extend(cell_failures)
                if cell is not None:
                    actual_cells.append(cell)
                if survived:
                    computed_survival += 1
        else:
            failures.append(
                _science_failure(
                    "sweep_cells_not_list",
                    subject=claim_id,
                    actual=bound_cells,
                )
            )
        actual_canonical = _canonical_cells(actual_cells)
        if actual_canonical != expected_canonical:
            failures.append(
                _science_failure(
                    "sweep_grid_coverage_mismatch",
                    subject=claim_id,
                    expected=expected_cells,
                    actual=actual_cells,
                )
            )
        asserted_survival = payload.get("survival_count")
        if asserted_survival != computed_survival:
            failures.append(
                _science_failure(
                    "sweep_survival_count_mismatch",
                    subject=claim_id,
                    expected=computed_survival,
                    actual=asserted_survival,
                )
            )
        if isinstance(criterion, (int, float)) and not isinstance(criterion, bool):
            meets = (
                computed_survival / len(expected_cells) >= float(criterion)
                if float(criterion) <= 1 and expected_cells
                else computed_survival >= float(criterion)
            )
            if not meets:
                failures.append(
                    _science_failure(
                        "sweep_survival_criterion_not_met",
                        subject=claim_id,
                        expected=criterion,
                        actual=computed_survival,
                    )
                )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "claim_count": len(headline_claims),
            "locked_cell_count": len(expected_cells),
            "active_lock_a_sha256": lock.get("body_sha256") if lock is not None else None,
            "failures": failures,
        },
    )


def _task_declared_input_paths(frontmatter: dict[str, object]) -> list[str]:
    paths: list[str] = []
    raw_inputs = frontmatter.get("inputs")
    if isinstance(raw_inputs, list):
        for entry in raw_inputs:
            if isinstance(entry, str):
                paths.append(entry)
            elif isinstance(entry, dict):
                for key in ("path", "manifest"):
                    value = entry.get(key)
                    if isinstance(value, str):
                        paths.append(value)
                        break
    instances = frontmatter.get("instances")
    if isinstance(instances, list):
        paths.extend(value for value in instances if isinstance(value, str))
    return paths


def check_hybrid_interface_conformance(
    repo: Path = Path("."),
    *,
    task_kind: str | None = None,
) -> GateResult:
    """Reusable full bridge seam check for task judging and M4 release assembly."""
    repo = repo.resolve()
    failures: list[dict[str, object]] = []
    manifest_paths = sorted((repo / "contracts/instances").glob("*.json"))
    bridge_paths: list[Path] = []
    interface_schema = repo / "contracts/hybrid_interface_v1.yaml"
    for disk_path in manifest_paths:
        payload, error = _load_json_file(disk_path)
        rel = disk_path.relative_to(repo)
        if error is not None or payload is None:
            failures.append(
                _science_failure("invalid_instance_manifest_json", subject=rel.as_posix(), actual=error)
            )
            continue
        if "source_processed_manifests" not in payload:
            continue
        bridge_paths.append(rel)
        for issue in _schema_failures(payload, interface_schema):
            failures.append(
                _science_failure(
                    "hybrid_interface_schema_violation",
                    subject=rel.as_posix(),
                    field=str(issue.get("path")),
                    actual=issue,
                )
            )
        sources = payload.get("source_processed_manifests")
        if isinstance(sources, list):
            for index, entry in enumerate(sources):
                failures.extend(
                    _verify_content_entry(
                        entry,
                        subject=rel.as_posix(),
                        field=f"source_processed_manifests[{index}]",
                        repo=repo,
                        required_prefix="data/processed_manifest/",
                    )
                )
        validations = payload.get("pre_bridge_validation")
        if isinstance(validations, list):
            for index, entry in enumerate(validations):
                failures.extend(
                    _verify_content_entry(
                        entry,
                        subject=rel.as_posix(),
                        field=f"pre_bridge_validation[{index}]",
                        repo=repo,
                    )
                )
                validation_path = (
                    _safe_repo_relative_path(entry.get("path"))
                    if isinstance(entry, dict)
                    else None
                )
                artifact_status = (
                    _validation_artifact_status(repo / validation_path)
                    if validation_path is not None and (repo / validation_path).is_file()
                    else None
                )
                if artifact_status not in {"green", "ok", "passed"}:
                    failures.append(
                        _science_failure(
                            "pre_bridge_validation_not_green",
                            subject=rel.as_posix(),
                            field=f"pre_bridge_validation[{index}]",
                            expected="artifact content status green|ok|passed",
                            actual=artifact_status,
                        )
                    )
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            for index, entry in enumerate(outputs):
                failures.extend(
                    _verify_content_entry(
                        entry,
                        subject=rel.as_posix(),
                        field=f"outputs[{index}]",
                        repo=repo,
                        require_bytes=True,
                    )
                )

    try:
        contract = load_framework_contract(repo)
    except ValueError as exc:
        failures.append(_science_failure("invalid_framework_contract", subject="contracts/framework.json", actual=str(exc)))
        contract = None
    if contract is not None and task_kind in {None, "model", "proof"}:
        for folder_name in contract.projection_dirs:
            folder = repo / ".orchestrator" / folder_name
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.md")):
                if path.name == "README.md":
                    continue
                frontmatter = _parse_task_frontmatter(_read_text(path))
                if not isinstance(frontmatter, dict) or frontmatter.get("task_kind") not in {"model", "proof"}:
                    continue
                for raw_input in _task_declared_input_paths(frontmatter):
                    candidate = _safe_repo_relative_path(raw_input)
                    if candidate is None or not _path_matches_prefix(candidate.as_posix(), "contracts/instances/"):
                        failures.append(
                            _science_failure(
                                "modeling_task_input_outside_instance_contract",
                                subject=path.relative_to(repo).as_posix(),
                                field="inputs",
                                expected="contracts/instances/**",
                                actual=raw_input,
                            )
                        )

    lock_path = repo / PREREG_PHASE_FILES["lock_b"]
    lock_b, error = load_prereg_lock(lock_path, expected_phase="lock_b")
    if error is not None or lock_b is None:
        failures.append(
            _science_failure("invalid_lock_b_lock", subject=PREREG_PHASE_FILES["lock_b"], actual=error)
        )
        lock_b = None
    active_lock_b = lock_b if lock_b is not None and lock_b.get("active") is True else None
    if lock_b is not None and lock_b.get("status") == "locked" and active_lock_b is None:
        failures.append(
            _science_failure(
                "lock_b_lock_hash_mismatch",
                subject=PREREG_PHASE_FILES["lock_b"],
                expected=lock_b.get("body_sha256"),
                actual=lock_b.get("locked_sha256"),
            )
        )
    if active_lock_b is not None:
        bindings, binding_failures = _verify_lock_bindings(active_lock_b, phase="lock_b", repo=repo)
        failures.extend(binding_failures)
        bound_instances = {
            path for path in bindings if _path_matches_prefix(path, "contracts/instances/")
        }
        actual_instances = {path.as_posix() for path in manifest_paths for path in [path.relative_to(repo)]}
        if bound_instances != actual_instances:
            failures.append(
                _science_failure(
                    "lock_b_instance_set_mismatch",
                    subject=PREREG_PHASE_FILES["lock_b"],
                    expected=sorted(actual_instances),
                    actual=sorted(bound_instances),
                )
            )
    elif lock_b is None and not bridge_paths:
        failures = [failure for failure in failures if failure.get("reason") != "invalid_lock_b_lock"]

    return GateResult(
        ok=not failures,
        details={
            "status": "no_bridge_manifests" if not bridge_paths else "ok",
            "bridge_manifest_count": len(bridge_paths),
            "instance_manifest_count": len(manifest_paths),
            "active_lock_b_sha256": active_lock_b.get("body_sha256") if active_lock_b is not None else None,
            "task_kind": task_kind,
            "failures": failures,
        },
    )


def gate_hybrid_interface_conformance(*, task_kind: str | None = None) -> GateResult:
    mode = _parse_project_mode(Path("contracts/project.yaml"))
    if mode != "hybrid":
        return GateResult(
            ok=True,
            details={"status": "mode_not_hybrid", "skipped": True, "mode": mode, "failures": []},
        )
    return check_hybrid_interface_conformance(Path("."), task_kind=task_kind)


def _yaml_scalar(raw: str) -> object:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    value = value.strip("'\"")
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "~"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        return float(value)
    return value


def _parse_simple_yaml(text: str) -> object:
    lines = [
        (len(raw) - len(raw.lstrip(" ")), raw.strip())
        for raw in text.splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]

    def parse_block(index: int, indent: int) -> tuple[object, int]:
        is_list = index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- ")
        container: object = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"unexpected_yaml_indent:{index + 1}")
            if is_list:
                if not content.startswith("- "):
                    break
                remainder = content[2:].strip()
                if not remainder:
                    if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                        value: object = None
                        index += 1
                    else:
                        value, index = parse_block(index + 1, lines[index + 1][0])
                    assert isinstance(container, list)
                    container.append(value)
                    continue
                if ":" in remainder:
                    key, raw_value = remainder.split(":", 1)
                    item: dict[str, object] = {}
                    item[key.strip()] = _yaml_scalar(raw_value) if raw_value.strip() else None
                    index += 1
                    while index < len(lines) and lines[index][0] > indent:
                        child_indent, child = lines[index]
                        if child.startswith("- ") or ":" not in child:
                            break
                        child_key, child_raw = child.split(":", 1)
                        if child_raw.strip():
                            item[child_key.strip()] = _yaml_scalar(child_raw)
                            index += 1
                        elif index + 1 < len(lines) and lines[index + 1][0] > child_indent:
                            nested, index = parse_block(index + 1, lines[index + 1][0])
                            item[child_key.strip()] = nested
                        else:
                            item[child_key.strip()] = None
                            index += 1
                    assert isinstance(container, list)
                    container.append(item)
                    continue
                assert isinstance(container, list)
                container.append(_yaml_scalar(remainder))
                index += 1
                continue
            if content.startswith("- ") or ":" not in content:
                break
            key, raw_value = content.split(":", 1)
            key = key.strip()
            assert isinstance(container, dict)
            if raw_value.strip():
                container[key] = _yaml_scalar(raw_value)
                index += 1
            elif index + 1 < len(lines) and lines[index + 1][0] > indent:
                nested, index = parse_block(index + 1, lines[index + 1][0])
                container[key] = nested
            else:
                container[key] = None
                index += 1
        return container, index

    if not lines:
        return {}
    payload, final_index = parse_block(0, lines[0][0])
    if final_index != len(lines):
        raise ValueError(f"yaml_parse_stopped_at:{final_index + 1}")
    return payload


def _load_venue_contract() -> tuple[dict[str, object] | None, str | None]:
    path = Path("contracts/venue.yaml")
    if not path.is_file():
        return None, None
    text = _read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = _parse_simple_yaml(text)
        except ValueError as exc:
            return None, str(exc)
    if not isinstance(payload, dict):
        return None, "venue_top_level_not_object"
    return payload, None


def _venue_number(venue: dict[str, object] | None, *keys: str) -> float | None:
    if venue is None:
        return None
    candidates: list[object] = [venue.get(key) for key in keys]
    limits = venue.get("limits")
    if isinstance(limits, dict):
        candidates.extend(limits.get(key) for key in keys)
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _manuscript_words(text: str) -> list[str]:
    body = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    body = re.sub(r"\{[^{}]*\}|`([^`]*)`|\[@[^\]]+\]", r" \1 ", body)
    return re.findall(r"\b[\w'-]+\b", body)


def gate_render_qa() -> GateResult:
    manuscript = Path("reports/paper/index.qmd")
    if not manuscript.is_file():
        return GateResult(
            ok=True,
            details={"status": "no_manuscript", "skipped": True, "failures": []},
        )
    text = _read_text(manuscript)
    failures: list[dict[str, object]] = []
    if re.search(r"\bFigure\s+Figure\b", text, flags=re.IGNORECASE):
        failures.append(_science_failure("duplicate_figure_prefix", subject=manuscript.as_posix()))
    if "??" in text:
        failures.append(_science_failure("unresolved_question_mark_reference", subject=manuscript.as_posix()))

    labels = set(re.findall(r"\{#([A-Za-z0-9:_-]+)\b", text))
    bib_keys: set[str] = set()
    for bib_path in (manuscript.parent / "references.bib", Path("references.bib")):
        if bib_path.is_file():
            bib_keys.update(re.findall(r"@\w+\s*[({]\s*([^,\s]+)\s*,", _read_text(bib_path)))
    references = {
        reference.rstrip(".,;:")
        for reference in re.findall(r"(?<![\w@])@([A-Za-z][A-Za-z0-9:_.-]*)", text)
    }
    for reference in sorted(references - labels - bib_keys):
        failures.append(
            _science_failure("unresolved_source_reference", subject=manuscript.as_posix(), actual=reference)
        )

    include_pattern = re.compile(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}")
    for raw_target in include_pattern.findall(text):
        target = (manuscript.parent / raw_target.strip("'\"")).resolve()
        if not target.is_file():
            failures.append(
                _science_failure("missing_include_target", subject=manuscript.as_posix(), actual=raw_target)
            )

    figure_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?")
    figures = figure_pattern.findall(text)
    for index, (caption, raw_target, attributes) in enumerate(figures):
        subject = f"{manuscript.as_posix()}:figure[{index}]"
        if not caption.strip():
            failures.append(_science_failure("figure_caption_missing", subject=subject))
        if re.search(r"#fig-[A-Za-z0-9_-]+", attributes or "") is None:
            failures.append(_science_failure("figure_label_missing", subject=subject))
        target_text = raw_target.split(maxsplit=1)[0].strip("<>'\"")
        if not (manuscript.parent / target_text).is_file():
            failures.append(_science_failure("figure_target_missing", subject=subject, actual=target_text))

    venue, venue_error = _load_venue_contract()
    if venue_error is not None:
        failures.append(_science_failure("invalid_venue_contract", subject="contracts/venue.yaml", actual=venue_error))
        venue = None
    word_count = len(_manuscript_words(text))
    word_limit = _venue_number(venue, "word_limit", "max_words", "word_count")
    if word_limit is not None and word_count > word_limit:
        failures.append(
            _science_failure("venue_word_limit_exceeded", subject=manuscript.as_posix(), expected=word_limit, actual=word_count)
        )
    words_per_page = _venue_number(venue, "words_per_page") or 500.0
    page_count = math.ceil(word_count / words_per_page) if word_count else 0
    page_limit = _venue_number(venue, "page_limit", "max_pages", "page_count")
    if page_limit is not None and page_count > page_limit:
        failures.append(
            _science_failure("venue_page_limit_exceeded", subject=manuscript.as_posix(), expected=page_limit, actual=page_count)
        )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "word_count": word_count,
            "page_count_heuristic": page_count,
            "figure_count": len(figures),
            "failures": failures,
        },
    )


def _overlap_paragraphs(text: str, min_words: int) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        if raw.lstrip().startswith(("#", "---", "```", "|")):
            continue
        normalized = " ".join(re.findall(r"\b[\w'-]+\b", raw.lower()))
        if len(normalized.split()) >= min_words:
            paragraphs.append(normalized)
    return paragraphs


def gate_text_overlap() -> GateResult:
    manuscript = Path("reports/paper/index.qmd")
    if not manuscript.is_file():
        return GateResult(ok=True, details={"status": "no_manuscript", "skipped": True, "failures": []})
    venue, venue_error = _load_venue_contract()
    failures: list[dict[str, object]] = []
    if venue_error is not None:
        failures.append(_science_failure("invalid_venue_contract", subject="contracts/venue.yaml", actual=venue_error))
        venue = None
    overlap = venue.get("text_overlap") if isinstance(venue, dict) else None
    max_ratio_value = overlap.get("max_ratio") if isinstance(overlap, dict) else None
    min_words_value = overlap.get("min_words") if isinstance(overlap, dict) else None
    max_ratio = float(max_ratio_value) if isinstance(max_ratio_value, (int, float)) else 0.85
    min_words = int(min_words_value) if isinstance(min_words_value, int) and min_words_value > 0 else 30
    paragraphs = _overlap_paragraphs(_read_text(manuscript), min_words)
    seen: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        if paragraph in seen:
            failures.append(
                _science_failure(
                    "repeated_manuscript_paragraph",
                    subject=manuscript.as_posix(),
                    expected=seen[paragraph],
                    actual=index,
                )
            )
        else:
            seen[paragraph] = index

    corpus_root = Path("data/raw/literature")
    corpus_paths = [
        path
        for path in sorted(corpus_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".qmd", ".html"}
    ] if corpus_root.is_dir() else []
    max_observed = 0.0
    for corpus_path in corpus_paths:
        try:
            corpus_text = _read_text(corpus_path)
        except UnicodeDecodeError:
            continue
        corpus_paragraphs = _overlap_paragraphs(corpus_text, min_words)
        for manuscript_index, paragraph in enumerate(paragraphs):
            for corpus_index, candidate in enumerate(corpus_paragraphs):
                ratio = difflib.SequenceMatcher(None, paragraph, candidate, autojunk=False).ratio()
                max_observed = max(max_observed, ratio)
                if ratio > max_ratio:
                    failures.append(
                        _science_failure(
                            "literature_near_duplicate_span",
                            subject=f"{manuscript.as_posix()}:paragraph[{manuscript_index}]",
                            field=f"{corpus_path.as_posix()}:paragraph[{corpus_index}]",
                            expected=f"<={max_ratio}",
                            actual=round(ratio, 6),
                        )
                    )
    return GateResult(
        ok=not failures,
        details={
            "status": "no_corpus" if not corpus_paths else "ok",
            "corpus_file_count": len(corpus_paths),
            "manuscript_paragraph_count": len(paragraphs),
            "max_ratio": max_ratio,
            "max_observed_ratio": round(max_observed, 6),
            "failures": failures,
        },
    )


def _derived_source_state(source: object) -> tuple[bool | None, str]:
    if isinstance(source, str):
        if source == "human_attested":
            return None, source
        gate_name = source.removeprefix("gate:")
        gate_functions = {
            "render_qa": gate_render_qa,
            "text_overlap": gate_text_overlap,
            "citation_integrity": gate_citation_integrity,
            "instance_manifest_conformance": gate_instance_manifest_conformance,
            "seed_budget_lock": gate_seed_budget_lock,
            "gap_convergence": gate_gap_convergence,
            "theoretical_falsification": gate_theoretical_falsification,
            "sweep_artifact": gate_sweep_artifact,
        }
        if gate_name in gate_functions:
            return gate_functions[gate_name]().ok, f"gate:{gate_name}"
        path = _safe_repo_relative_path(source)
        return (path.exists() if path is not None else False), source
    if isinstance(source, dict):
        if source.get("human_attested") is True:
            return None, "human_attested"
        if isinstance(source.get("gate"), str):
            return _derived_source_state(f"gate:{source['gate']}")
        path = _safe_repo_relative_path(source.get("path"))
        if path is None or not path.exists():
            return False, str(source.get("path"))
        expected_sha = source.get("sha256")
        if isinstance(expected_sha, str) and path.is_file():
            actual_sha, _ = _sha256_and_bytes(path)
            return actual_sha == expected_sha, path.as_posix()
        return True, path.as_posix()
    return False, repr(source)


def gate_checklist_derivation() -> GateResult:
    venue, error = _load_venue_contract()
    if error is not None:
        return GateResult(
            ok=False,
            details={"status": "invalid_venue", "failures": [_science_failure("invalid_venue_contract", subject="contracts/venue.yaml", actual=error)]},
        )
    if venue is None or "checklist" not in venue:
        return GateResult(ok=True, details={"status": "no_checklist", "skipped": True, "failures": []})
    checklist = venue.get("checklist")
    failures: list[dict[str, object]] = []
    if not isinstance(checklist, list):
        failures.append(_science_failure("venue_checklist_not_list", subject="contracts/venue.yaml"))
        checklist = []
    human_attested = 0
    for index, item in enumerate(checklist):
        subject = f"contracts/venue.yaml:checklist[{index}]"
        if not isinstance(item, dict):
            failures.append(_science_failure("checklist_item_not_object", subject=subject))
            continue
        for field in ("question", "answer", "derived_from"):
            if field not in item:
                failures.append(_science_failure("checklist_field_missing", subject=subject, field=field))
        source = item.get("derived_from")
        sources = source if isinstance(source, list) else [source]
        states = [_derived_source_state(entry) for entry in sources]
        machine_states = [(state, label) for state, label in states if state is not None]
        has_human_attestation = item.get("human_attested") is True or any(
            state is None for state, _ in states
        )
        if has_human_attestation:
            human_attested += 1
        answer = item.get("answer")
        affirmative = answer is True or (isinstance(answer, str) and answer.strip().lower() in {"yes", "true", "pass", "passed"})
        if affirmative and machine_states and not all(
            state is True for state, _ in machine_states
        ):
            failures.append(
                _science_failure(
                    "checklist_answer_not_supported",
                    subject=subject,
                    field="derived_from",
                    expected="all machine-derivable sources present/green",
                    actual=[{"source": label, "state": state} for state, label in states],
                )
            )
        elif affirmative and not machine_states and not has_human_attestation:
            failures.append(
                _science_failure(
                    "checklist_answer_not_supported",
                    subject=subject,
                    field="derived_from",
                    expected="machine derivation or human_attested",
                    actual=[{"source": label, "state": state} for state, label in states],
                )
            )
    return GateResult(
        ok=not failures,
        details={
            "status": "ok",
            "item_count": len(checklist),
            "human_attested_count": human_attested,
            "failures": failures,
        },
    )


_CORE_GATE_NAMES = (
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
    "referee_rubrics",
    "referee_report_validity",
    "referee_calibration",
    "review_bundle_integrity",
    "processed_manifest_hashes",
    "raw_manifest_hashes",
    "validation_report_content_binding",
    "projection_drift",
    "historical_exemptions",
    "network_strings",
    "task_lint",
    "prereg_lock_coverage",
)
_MODE_INDEPENDENT_SCIENCE_GATES = (
    "citation_integrity",
    "rigor_sections",
    "amendment_exploratory_tagging",
    "headline_confirmatory",
    "render_qa",
    "text_overlap",
    "checklist_derivation",
)
_EMPIRICAL_SCIENCE_GATES = (
    "prereg_conformance",
    "claim_evidence_ledger",
    "etl_decision_log",
)
_MODELING_SCIENCE_GATES = (
    "prereg_conformance",
    "claim_evidence_ledger",
    "etl_decision_log",
    "instance_manifest_conformance",
    "seed_budget_lock",
    "gap_convergence",
    "theoretical_falsification",
    "sweep_artifact",
)
_ALL_GATE_NAMES = _CORE_GATE_NAMES + (
    "prereg_conformance",
    "claim_evidence_ledger",
    "citation_integrity",
    "etl_decision_log",
    "rigor_sections",
    "instance_manifest_conformance",
    "seed_budget_lock",
    "gap_convergence",
    "theoretical_falsification",
    "sweep_artifact",
    "hybrid_interface_conformance",
    "amendment_exploratory_tagging",
    "headline_confirmatory",
    "render_qa",
    "text_overlap",
    "checklist_derivation",
)


def _active_gates(mode: str, task_kind: str | None = None) -> tuple[str, ...]:
    """Select the §5.2 gate form for a project mode and optional task kind."""
    active = set(_CORE_GATE_NAMES) | set(_MODE_INDEPENDENT_SCIENCE_GATES)
    if mode == "empirical":
        active.update(_EMPIRICAL_SCIENCE_GATES)
    elif mode == "modeling":
        active.update(_MODELING_SCIENCE_GATES)
    elif mode == "hybrid":
        if task_kind in {"etl", "analysis", "validation"}:
            active.update(_EMPIRICAL_SCIENCE_GATES)
        elif task_kind in {"model", "proof"}:
            active.update(_MODELING_SCIENCE_GATES)
        elif task_kind == "bridge":
            active.update({"etl_decision_log", "hybrid_interface_conformance"})
        elif task_kind in {"lit_review", "ops", "integrity_audit", "repair"}:
            pass
        else:
            active.update(_EMPIRICAL_SCIENCE_GATES)
            active.update(_MODELING_SCIENCE_GATES)
            active.add("hybrid_interface_conformance")
    return tuple(name for name in _ALL_GATE_NAMES if name in active)


def _collect_gate_results(*, task_kind: str | None = None) -> dict[str, GateResult]:
    mode = _parse_project_mode(Path("contracts/project.yaml")) or "empirical"
    active = set(_active_gates(mode, task_kind))
    functions = {
        "framework_contract": gate_framework_contract,
        "repo_structure": gate_repo_structure,
        "project_contract": gate_project_contract,
        "protocol_complete": gate_protocol_complete,
        "workstreams_complete": gate_workstreams_complete,
        "task_hygiene": gate_task_hygiene,
        "task_dependencies": gate_task_dependencies,
        "integration_ready_policy": gate_integration_ready_policy,
        "operator_surface_ownership": gate_operator_surface_ownership,
        "raw_manifest_validity": gate_raw_manifest_validity,
        "processed_manifest_validity": gate_processed_manifest_validity,
        "swarm_run_manifest_validity": gate_swarm_run_manifest_validity,
        "judge_review_log_validity": gate_judge_review_log_validity,
        "referee_rubrics": gate_referee_rubrics,
        "referee_report_validity": gate_referee_report_validity,
        "referee_calibration": gate_referee_calibration,
        "review_bundle_integrity": gate_review_bundle_integrity,
        "processed_manifest_hashes": gate_processed_manifest_hashes,
        "raw_manifest_hashes": gate_raw_manifest_hashes,
        "validation_report_content_binding": gate_validation_report_content_binding,
        "projection_drift": gate_projection_drift,
        "historical_exemptions": gate_historical_exemptions,
        "network_strings": gate_network_strings,
        "task_lint": gate_task_lint,
        "prereg_lock_coverage": gate_prereg_lock_coverage,
        "claim_evidence_ledger": gate_claim_evidence_ledger,
        "citation_integrity": gate_citation_integrity,
        "rigor_sections": gate_rigor_sections,
        "instance_manifest_conformance": gate_instance_manifest_conformance,
        "seed_budget_lock": gate_seed_budget_lock,
        "gap_convergence": gate_gap_convergence,
        "theoretical_falsification": gate_theoretical_falsification,
        "sweep_artifact": gate_sweep_artifact,
        "amendment_exploratory_tagging": gate_amendment_exploratory_tagging,
        "headline_confirmatory": gate_headline_confirmatory,
        "render_qa": gate_render_qa,
        "text_overlap": gate_text_overlap,
        "checklist_derivation": gate_checklist_derivation,
    }
    results: dict[str, GateResult] = {}
    for name in _ALL_GATE_NAMES:
        if name not in active:
            results[name] = GateResult(
                ok=True,
                details={
                    "status": "inactive_for_mode",
                    "skipped": True,
                    "mode": mode,
                    "task_kind": task_kind,
                    "failures": [],
                },
            )
        elif name == "prereg_conformance":
            form = None
            if mode == "hybrid":
                if task_kind in {"etl", "analysis", "validation"}:
                    form = "empirical"
                elif task_kind in {"model", "proof"}:
                    form = "modeling"
            results[name] = gate_prereg_conformance(form=form)
        elif name == "etl_decision_log":
            form = None
            if mode == "hybrid":
                if task_kind in {"etl", "analysis", "validation"}:
                    form = "empirical"
                elif task_kind in {"model", "proof", "bridge"}:
                    form = "modeling"
            results[name] = gate_etl_decision_log(form=form)
        elif name == "hybrid_interface_conformance":
            results[name] = gate_hybrid_interface_conformance(task_kind=task_kind)
        else:
            results[name] = functions[name]()
    return results


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False so a gate command cannot smuggle the kernel-injected
    # --task-kind via a unique prefix like --task-k (verification-pass bypass).
    parser = argparse.ArgumentParser(prog="quality_gates.py", allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    parser.add_argument(
        "--task-kind",
        choices=["etl", "analysis", "validation", "writing", "lit_review", "model", "proof", "bridge", "ops", "integrity_audit", "repair"],
        help="Apply hybrid composition at one task-kind boundary; omitted means writing/release union",
    )
    args = parser.parse_args(argv)

    results = _collect_gate_results(task_kind=args.task_kind)
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
