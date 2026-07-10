#!/usr/bin/env python3
"""
Local swarm supervisor for the v1 research operating system.

This runtime keeps the file-based control plane intact:

- task `State:` is authoritative
- lifecycle folders are only a projection
- dependencies are satisfied by `done`, except explicit
  `integration_ready_dependencies`
- `integration_ready` is allowed only for eligible interface/export tasks
- `ready_for_review` requires outputs, gates, manifests, and a durable run manifest
- `done` requires a deterministic Judge review log

The public operator-facing commands remain:

- `plan`
- `tick`
- `loop`
- `tmux-start`

Internal helper commands used by the supervisor:

- `run-task`
- `judge-task`
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from swarm_taskfile import WorktreeCollisionError
from swarm_taskfile import parse_status_value as _parse_status_value
from swarm_taskfile import parse_task_frontmatter as _parse_task_frontmatter
from swarm_taskfile import parse_task_id_from_branch as _parse_task_id_from_branch
from swarm_taskfile import update_task_status_and_notes as _shared_update_task_status_and_notes


SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v1"
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
VALID_TASK_PRIORITIES = {"low", "medium", "high"}

_PREFLIGHT_STRICT_SYNC_CACHE: set[tuple[str, bool, bool]] = set()
_REPO_ROOT_CACHE: Path | None = None


@dataclasses.dataclass(frozen=True)
class FrameworkContract:
    repo_root: Path
    control_plane_root: Path
    project_mode: str | None
    allowed_roles: tuple[str, ...]
    task_execution_roles: tuple[str, ...]
    scientific_review_role: str
    allowed_states: tuple[str, ...]
    projection_dirs: tuple[str, ...]
    network_workstreams: tuple[str, ...]
    prompt_templates: dict[str, Path]
    integration_ready_eligible_workstreams: tuple[str, ...]
    integration_ready_eligible_task_kinds: tuple[str, ...]
    forbid_unvalidated_empirical_data_outputs: bool
    operator_owned_shared_surfaces: tuple[str, ...]
    run_manifest_dir: Path
    judge_review_dir: Path
    release_manifest_pattern: str | None


@dataclasses.dataclass(frozen=True)
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


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_timestamp_compact() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "check": check,
        "text": True,
        "env": env,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    if timeout_seconds is None:
        return subprocess.run(cmd, timeout=None, **kwargs)

    popen_kwargs = dict(kwargs)
    popen_kwargs.pop("check", None)
    popen_kwargs["start_new_session"] = True
    with subprocess.Popen(cmd, **popen_kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
            exc.stdout = stdout
            exc.stderr = stderr
            raise

    completed = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _which_or_none(name: str) -> str | None:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(item) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _repo_root() -> Path:
    global _REPO_ROOT_CACHE
    if _REPO_ROOT_CACHE is not None:
        return _REPO_ROOT_CACHE

    env_root = os.environ.get("SWARM_REPO_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"SWARM_REPO_ROOT is not a directory: {root}")
        _REPO_ROOT_CACHE = root
        return root

    try:
        cp = _run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            capture=True,
            check=True,
        )
        top = (cp.stdout or "").strip()
        if top:
            root = Path(top).resolve()
            if root.is_dir():
                _REPO_ROOT_CACHE = root
                return root
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    _REPO_ROOT_CACHE = root
    return root


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


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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


def _resolve_repo_relative_path(repo: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo / path).resolve()


def load_framework_contract(repo: Path) -> FrameworkContract:
    framework_path = repo / "contracts" / "framework.json"
    if not framework_path.exists():
        raise SystemExit(f"Missing framework contract: {framework_path}")

    try:
        raw = json.loads(_read_text(framework_path))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {framework_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit(f"Expected a JSON object in {framework_path}")

    roles = raw.get("roles")
    states = raw.get("states")
    review_bundle = raw.get("review_bundle")
    integration_ready_policy = raw.get("integration_ready_policy")
    execution_engines = raw.get("execution_engines")
    release_policy = raw.get("release_policy")

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
    projection_dirs_raw = _coerce_str_list(states.get("projection_dirs") if isinstance(states, dict) else None)
    projection_dirs = tuple(Path(item).name for item in projection_dirs_raw) or tuple(DEFAULT_ALLOWED_STATES)

    routine_repo_tasks = execution_engines.get("routine_repo_tasks") if isinstance(execution_engines, dict) else None
    control_plane_root_raw = (
        routine_repo_tasks.get("control_plane_root")
        if isinstance(routine_repo_tasks, dict) and isinstance(routine_repo_tasks.get("control_plane_root"), str)
        else ".orchestrator"
    )
    control_plane_root = _resolve_repo_relative_path(repo, control_plane_root_raw)

    prompt_templates_raw = raw.get("prompt_templates")
    prompt_templates = dict(DEFAULT_PROMPT_TEMPLATES)
    if isinstance(prompt_templates_raw, dict):
        for key, value in prompt_templates_raw.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                prompt_templates[key] = value.strip()
    resolved_prompt_templates = {
        key: _resolve_repo_relative_path(repo, value) for key, value in prompt_templates.items()
    }

    network_workstreams = tuple(
        _coerce_str_list(raw.get("network_workstreams")) or list(DEFAULT_NETWORK_WORKSTREAMS)
    )

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
    forbid_unvalidated_empirical = _coerce_bool(
        integration_ready_policy.get("forbid_unvalidated_empirical_data_outputs")
        if isinstance(integration_ready_policy, dict)
        else None,
        default=True,
    )

    operator_owned_shared_surfaces = tuple(
        _coerce_str_list(raw.get("operator_owned_shared_surfaces")) or list(DEFAULT_OPERATOR_OWNED_SHARED_SURFACES)
    )

    run_manifest_dir_raw = (
        review_bundle.get("run_manifest_dir")
        if isinstance(review_bundle, dict) and isinstance(review_bundle.get("run_manifest_dir"), str)
        else "reports/status/swarm_runs"
    )
    judge_review_dir_raw = (
        review_bundle.get("judge_review_dir")
        if isinstance(review_bundle, dict) and isinstance(review_bundle.get("judge_review_dir"), str)
        else "reports/status/reviews"
    )

    release_manifest_pattern = (
        release_policy.get("release_manifest_pattern")
        if isinstance(release_policy, dict) and isinstance(release_policy.get("release_manifest_pattern"), str)
        else None
    )

    return FrameworkContract(
        repo_root=repo,
        control_plane_root=control_plane_root,
        project_mode=_parse_project_mode(repo / "contracts" / "project.yaml"),
        allowed_roles=allowed_roles,
        task_execution_roles=task_execution_roles,
        scientific_review_role=scientific_review_role,
        allowed_states=allowed_states,
        projection_dirs=projection_dirs,
        network_workstreams=network_workstreams,
        prompt_templates=resolved_prompt_templates,
        integration_ready_eligible_workstreams=eligible_workstreams,
        integration_ready_eligible_task_kinds=eligible_task_kinds,
        forbid_unvalidated_empirical_data_outputs=forbid_unvalidated_empirical,
        operator_owned_shared_surfaces=operator_owned_shared_surfaces,
        run_manifest_dir=_resolve_repo_relative_path(repo, run_manifest_dir_raw),
        judge_review_dir=_resolve_repo_relative_path(repo, judge_review_dir_raw),
        release_manifest_pattern=release_manifest_pattern,
    )


def load_task(path: Path, contract: FrameworkContract) -> Task:
    text = _read_text(path)
    frontmatter = _parse_task_frontmatter(text)
    if frontmatter is None:
        raise ValueError(f"missing_yaml_frontmatter:{path}")

    for key in REQUIRED_FRONTMATTER_KEYS:
        if key not in frontmatter:
            raise ValueError(f"frontmatter_missing_key:{path}:{key}")

    def require_str(key: str) -> str:
        value = frontmatter.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"frontmatter_invalid_string:{path}:{key}")
        return value.strip()

    def require_list(key: str) -> list[str]:
        value = frontmatter.get(key)
        if not isinstance(value, list):
            raise ValueError(f"frontmatter_invalid_list:{path}:{key}")
        out = _coerce_str_list(value)
        if key in {"allowed_paths", "disallowed_paths", "outputs", "gates", "stop_conditions"} and not out:
            raise ValueError(f"frontmatter_empty_list:{path}:{key}")
        return out

    task_id = require_str("task_id")
    role = require_str("role")
    priority = require_str("priority").lower()
    state = _parse_status_value(text, "State")
    last_updated = _parse_status_value(text, "Last updated")

    if role not in set(contract.allowed_roles):
        raise ValueError(f"invalid_role:{path}:{role}")
    if priority not in VALID_TASK_PRIORITIES:
        raise ValueError(f"invalid_priority:{path}:{priority}")
    if state is None or state not in set(contract.allowed_states):
        raise ValueError(f"invalid_state:{path}:{state}")
    if last_updated is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_updated):
        raise ValueError(f"invalid_last_updated:{path}:{last_updated}")

    raw_task_kind = frontmatter.get("task_kind")
    task_kind = raw_task_kind.strip() if isinstance(raw_task_kind, str) and raw_task_kind.strip() else None

    return Task(
        path=path,
        task_id=task_id,
        title=require_str("title"),
        workstream=require_str("workstream"),
        task_kind=task_kind,
        role=role,
        priority=priority,
        dependencies=require_list("dependencies"),
        integration_ready_dependencies=require_list("integration_ready_dependencies")
        if isinstance(frontmatter.get("integration_ready_dependencies"), list)
        else [],
        allow_network=_coerce_bool(frontmatter.get("allow_network"), default=False),
        allowed_paths=require_list("allowed_paths"),
        disallowed_paths=require_list("disallowed_paths"),
        outputs=require_list("outputs"),
        gates=require_list("gates"),
        stop_conditions=require_list("stop_conditions"),
        state=state,
        last_updated=last_updated,
    )


def _iter_task_files(contract: FrameworkContract) -> Iterable[Path]:
    for folder_name in contract.projection_dirs:
        folder = contract.control_plane_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            yield path


def load_tasks_quarantined(contract: FrameworkContract) -> tuple[dict[str, Task], list[dict[str, str]]]:
    tasks: dict[str, Task] = {}
    quarantined: list[dict[str, str]] = []
    for path in _iter_task_files(contract):
        try:
            task = load_task(path, contract)
        except ValueError as exc:
            quarantined.append(
                {
                    "path": path.resolve().relative_to(contract.repo_root.resolve()).as_posix(),
                    "error": str(exc),
                }
            )
            continue
        if task.task_id in tasks:
            quarantined.append(
                {
                    "path": path.resolve().relative_to(contract.repo_root.resolve()).as_posix(),
                    "error": f"duplicate_task_id:{task.task_id}:{tasks[task.task_id].path}:{path}",
                }
            )
            continue
        tasks[task.task_id] = task
    return tasks, quarantined


def load_tasks(contract: FrameworkContract) -> dict[str, Task]:
    tasks, quarantined = load_tasks_quarantined(contract)
    if quarantined:
        raise ValueError(quarantined[0]["error"])
    return tasks


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


def downstream_allowlist_exists(task_id: str, tasks: dict[str, Task]) -> bool:
    return any(task_id in task.integration_ready_dependencies for task in tasks.values())


def dependency_is_satisfied(dep_id: str, downstream_task: Task, tasks: dict[str, Task], contract: FrameworkContract) -> bool:
    upstream_task = tasks.get(dep_id)
    if upstream_task is None:
        return False
    if upstream_task.state == "done":
        return True
    if upstream_task.state != "integration_ready":
        return False
    if dep_id not in downstream_task.integration_ready_dependencies:
        return False
    if not task_is_integration_ready_eligible(upstream_task, contract):
        return False
    return True


def _dependencies_satisfied(task: Task, tasks: dict[str, Task], contract: FrameworkContract) -> bool:
    return all(dependency_is_satisfied(dep_id, task, tasks, contract) for dep_id in task.dependencies)


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


def _task_summary(task: Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "workstream": task.workstream,
        "task_kind": task.task_kind,
        "role": task.role,
        "priority": task.priority,
        "dependencies": list(task.dependencies),
        "integration_ready_dependencies": list(task.integration_ready_dependencies),
        "allow_network": task.allow_network,
        "state": task.state,
        "task_path": task.path.as_posix(),
    }


def ready_backlog_tasks(tasks: dict[str, Task], claimed_ids: set[str], contract: FrameworkContract) -> list[Task]:
    ready: list[Task] = []
    for task in tasks.values():
        if task.state != "backlog":
            continue
        if task.role not in set(contract.task_execution_roles):
            continue
        if task.task_id in claimed_ids:
            continue
        if _dependencies_satisfied(task, tasks, contract):
            ready.append(task)
    ready.sort(key=lambda item: (_priority_rank(item.priority), item.task_id))
    return ready


def _format_bullets(items: Iterable[str]) -> str:
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not cleaned:
        return "- (none)"
    return "\n".join(f"- {item}" for item in cleaned)


def load_prompt(template_path: Path, context: dict[str, object]) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"missing_prompt_template:{template_path}")
    text = _read_text(template_path)
    rendered = text
    for key, value in sorted(context.items(), key=lambda entry: len(entry[0]), reverse=True):
        if value is None:
            replacement = ""
        elif isinstance(value, (list, tuple, set)):
            replacement = "\n".join(str(item) for item in value)
        else:
            replacement = str(value)
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def _build_prompt_context(task: Task, repo: Path, repair_context: str | None) -> dict[str, object]:
    return {
        "repo_root": repo.as_posix(),
        "task_path": task.path.relative_to(repo).as_posix(),
        "task_id": task.task_id,
        "title": task.title,
        "workstream": task.workstream,
        "task_kind": task.task_kind or "",
        "allow_network": "true" if task.allow_network else "false",
        "allowed_paths": _format_bullets(task.allowed_paths),
        "disallowed_paths": _format_bullets(task.disallowed_paths),
        "outputs": _format_bullets(task.outputs),
        "gates": _format_bullets(task.gates),
        "stop_conditions": _format_bullets(task.stop_conditions),
        "repair_context": repair_context or "",
        "runner_mode": "local_swarm",
        "base_branch": "",
    }


def _git_current_branch(cwd: Path) -> str:
    cp = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture=True, check=True)
    return (cp.stdout or "").strip()


def _git_head_sha(cwd: Path) -> str | None:
    cp = _run(["git", "rev-parse", "HEAD"], cwd=cwd, capture=True, check=False)
    if cp.returncode != 0:
        return None
    value = (cp.stdout or "").strip()
    return value or None


def _git_has_changes(cwd: Path) -> bool:
    cp = _run(["git", "status", "--porcelain"], cwd=cwd, capture=True, check=True)
    return bool((cp.stdout or "").strip())


def _git_ref_exists(cwd: Path, ref: str) -> bool:
    cp = _run(["git", "rev-parse", "--verify", ref], cwd=cwd, capture=True, check=False)
    return cp.returncode == 0


def _resolve_base_ref_for_diff(*, cwd: Path, base_branch: str, remote: str) -> str | None:
    for candidate in (f"{remote}/{base_branch}", base_branch):
        if _git_ref_exists(cwd, candidate):
            return candidate
    return None


def claimed_task_ids(repo: Path, remote: str, base_branch: str) -> set[str]:
    claimed: set[str] = set()

    try:
        cp = _run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo,
            capture=True,
            check=True,
        )
        for line in (cp.stdout or "").splitlines():
            if not line.startswith("branch "):
                continue
            ref = line.split(" ", 1)[1].strip()
            if ref.startswith("refs/heads/"):
                task_id = _parse_task_id_from_branch(ref.removeprefix("refs/heads/"))
                if task_id is not None:
                    claimed.add(task_id)
    except Exception:
        pass

    gh = _which_or_none("gh")
    if gh is not None:
        try:
            cp = _run(
                [gh, "pr", "list", "--state", "open", "--base", base_branch, "--json", "headRefName"],
                cwd=repo,
                capture=True,
                check=True,
            )
            payload = json.loads(cp.stdout or "[]")
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    head = item.get("headRefName")
                    if isinstance(head, str):
                        task_id = _parse_task_id_from_branch(head)
                        if task_id is not None:
                            claimed.add(task_id)
        except Exception:
            pass

    try:
        cp = _run(
            ["git", "ls-remote", "--heads", remote, "T[0-9][0-9][0-9]_*"],
            cwd=repo,
            capture=True,
            check=False,
        )
        if cp.returncode == 0:
            for line in (cp.stdout or "").splitlines():
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                ref = parts[1].strip()
                if ref.startswith("refs/heads/"):
                    task_id = _parse_task_id_from_branch(ref.removeprefix("refs/heads/"))
                    if task_id is not None:
                        claimed.add(task_id)
    except Exception:
        pass

    return claimed


def choose_tasks_heuristic(ready_tasks: list[Task], capacity: int) -> list[Task]:
    selected: list[Task] = []
    used_workstreams: set[str] = set()
    for task in ready_tasks:
        if task.workstream in used_workstreams:
            continue
        selected.append(task)
        used_workstreams.add(task.workstream)
        if len(selected) >= max(0, capacity):
            break
    return selected


def _slug_from_task_path(path: Path, task_id: str) -> str:
    stem = path.stem
    prefix = f"{task_id}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def ensure_worktree(*, repo: Path, task: Task, worktree_parent: Path, base_ref: str) -> tuple[Path, str]:
    slug = _slug_from_task_path(task.path, task.task_id)
    branch = f"{task.task_id}_{slug}"
    worktree_path = worktree_parent / f"wt-{task.task_id}"

    if worktree_path.exists():
        raise WorktreeCollisionError(worktree_path)

    branch_exists = _run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo,
        capture=False,
        check=False,
    ).returncode == 0

    if branch_exists:
        _run(["git", "worktree", "add", str(worktree_path), branch], cwd=repo, check=True)
    else:
        _run(["git", "worktree", "add", str(worktree_path), "-b", branch, base_ref], cwd=repo, check=True)

    return worktree_path, branch


def _tmux(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    tmux = _which_or_none("tmux")
    if tmux is None:
        raise SystemExit("tmux_not_found")
    return _run([tmux, *args], check=check, capture=capture)


def _tmux_ensure_session(session: str, start_dir: Path) -> None:
    cp = _tmux("has-session", "-t", session, check=False, capture=False)
    if cp.returncode == 0:
        return
    _tmux("new-session", "-d", "-s", session, "-c", str(start_dir))


def _tmux_spawn_task_window(*, session: str, window_name: str, workdir: Path, command: list[str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    _tmux(
        "new-window",
        "-t",
        session,
        "-n",
        window_name,
        "-c",
        str(workdir),
        "bash",
        "-lc",
        rendered,
    )


def _git_config_get(cwd: Path, key: str) -> str | None:
    cp = _run(["git", "config", "--get", key], cwd=cwd, capture=True, check=False)
    if cp.returncode != 0:
        return None
    value = (cp.stdout or "").strip()
    return value or None


def _git_remote_exists(cwd: Path, remote: str) -> bool:
    cp = _run(["git", "remote", "get-url", remote], cwd=cwd, capture=True, check=False)
    return cp.returncode == 0


def _require_git_identity(*, cwd: Path, reason: str) -> None:
    name = _git_config_get(cwd, "user.name")
    email = _git_config_get(cwd, "user.email")
    if name and email:
        return
    missing: list[str] = []
    if not name:
        missing.append("user.name")
    if not email:
        missing.append("user.email")
    raise SystemExit(
        "\n".join(
            [
                f"preflight_failed:{reason}:missing_git_identity:{','.join(missing)}",
                'git config user.name "swarm-bot"',
                'git config user.email "swarm-bot@example.invalid"',
            ]
        )
    )


def _require_git_push_access(*, cwd: Path, remote: str, reason: str, timeout_seconds: int = 30) -> None:
    if not _git_remote_exists(cwd, remote):
        raise SystemExit(f"preflight_failed:{reason}:missing_remote:{remote}")
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    cp = _run(
        ["git", "push", "--dry-run", remote, "HEAD"],
        cwd=cwd,
        capture=True,
        check=False,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    if cp.returncode == 0:
        return
    raise SystemExit(f"preflight_failed:{reason}:cannot_push:{remote}")


def _require_gh_auth(*, cwd: Path, reason: str, timeout_seconds: int = 20) -> None:
    gh = _which_or_none("gh")
    if gh is None:
        raise SystemExit(f"preflight_failed:{reason}:gh_not_found")
    cp = _run([gh, "auth", "status"], cwd=cwd, capture=True, check=False, timeout_seconds=timeout_seconds)
    if cp.returncode != 0:
        raise SystemExit(f"preflight_failed:{reason}:gh_not_authenticated")


def _preflight_strict_sync_requirements(*, cwd: Path, remote: str, unattended: bool, create_pr: bool) -> None:
    if not (unattended or create_pr):
        return
    cache_key = (remote, unattended, create_pr)
    if cache_key in _PREFLIGHT_STRICT_SYNC_CACHE:
        return
    reason = "unattended" if unattended else "create_pr"
    _require_git_identity(cwd=cwd, reason=reason)
    _require_git_push_access(cwd=cwd, remote=remote, reason=reason)
    if create_pr:
        _require_gh_auth(cwd=cwd, reason=reason)
    _PREFLIGHT_STRICT_SYNC_CACHE.add(cache_key)


def _git_commit(*, cwd: Path, message: str, strict: bool) -> None:
    cp = _run(["git", "commit", "-m", message], cwd=cwd, capture=True, check=False)
    if cp.returncode == 0:
        return
    if strict:
        raise SystemExit(f"git_commit_failed:{message}")
    print(f"[warn] git commit failed: {message}", file=sys.stderr)


def _git_push(*, cwd: Path, remote: str, ref: str, set_upstream: bool, strict: bool) -> None:
    env = dict(os.environ)
    if strict:
        env["GIT_TERMINAL_PROMPT"] = "0"
    cmd = ["git", "push"]
    if set_upstream:
        cmd.append("-u")
    cmd.extend([remote, ref])
    cp = _run(cmd, cwd=cwd, capture=True, check=False, env=env, timeout_seconds=60)
    if cp.returncode == 0:
        return
    if strict:
        raise SystemExit(f"git_push_failed:{remote}:{ref}")
    print(f"[warn] git push failed: remote={remote} ref={ref}", file=sys.stderr)


def _gh_create_pr_if_missing(*, cwd: Path, base_branch: str, title: str, body: str) -> None:
    gh = _which_or_none("gh")
    if gh is None:
        return

    branch = _git_current_branch(cwd)
    cp = _run(
        [gh, "pr", "list", "--state", "open", "--head", branch, "--json", "number"],
        cwd=cwd,
        capture=True,
        check=False,
    )
    if cp.returncode == 0:
        payload = json.loads(cp.stdout or "[]")
        if isinstance(payload, list) and payload:
            return

    _run(
        [gh, "pr", "create", "--base", base_branch, "--title", title, "--body", body],
        cwd=cwd,
        check=True,
    )


def _require_unattended_ack() -> None:
    if os.environ.get("SWARM_UNATTENDED_I_UNDERSTAND") == "1":
        return
    raise SystemExit("missing_unattended_ack:SWARM_UNATTENDED_I_UNDERSTAND=1")


def _supervisor_sync_to_remote_base(*, repo: Path, remote: str, base_branch: str) -> None:
    _run(["git", "fetch", remote], cwd=repo, check=True)
    _run(["git", "checkout", "-B", base_branch, f"{remote}/{base_branch}"], cwd=repo, check=True)


def _git_diff_name_status_entries(cwd: Path, diff_args: list[str]) -> list[dict[str, str]]:
    cp = _run(
        ["git", "diff", "--name-status", "-M", *diff_args],
        cwd=cwd,
        capture=True,
        check=True,
    )
    entries: list[dict[str, str]] = []
    for raw_line in (cp.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0].strip()
        code = status[:1]
        old_path = ""
        new_path = ""
        if code in {"R", "C"}:
            if len(parts) < 3:
                continue
            old_path = parts[1].strip()
            new_path = parts[2].strip()
        else:
            if len(parts) < 2:
                continue
            new_path = parts[1].strip()
        entries.append(
            {
                "status": status,
                "code": code,
                "path": new_path,
                "old_path": old_path,
            }
        )
    return entries


def _git_untracked_files(cwd: Path) -> list[str]:
    cp = _run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=cwd,
        capture=True,
        check=True,
    )
    return [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]


def _collect_changed_paths_with_sources(*, repo: Path, base_ref: str | None) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    path_sources: dict[str, set[str]] = {}
    ops: list[dict[str, str]] = []

    def add_entries(source: str, entries: list[dict[str, str]]) -> None:
        for entry in entries:
            record = dict(entry)
            record["source"] = source
            ops.append(record)
            for candidate in (entry.get("path", ""), entry.get("old_path", "")):
                if not candidate:
                    continue
                path_sources.setdefault(candidate, set()).add(source)

    if base_ref is not None:
        add_entries("committed", _git_diff_name_status_entries(repo, [f"{base_ref}...HEAD"]))
    add_entries("staged", _git_diff_name_status_entries(repo, ["--cached"]))
    add_entries("unstaged", _git_diff_name_status_entries(repo, []))
    for path in _git_untracked_files(repo):
        path_sources.setdefault(path, set()).add("untracked")
        ops.append(
            {
                "status": "??",
                "code": "?",
                "path": path,
                "old_path": "",
                "source": "untracked",
            }
        )
    return path_sources, ops


def _task_projection_paths(task_file_path: str) -> set[str]:
    filename = Path(task_file_path).name
    return {
        f".orchestrator/{state}/{filename}"
        for state in ("backlog", "active", "integration_ready", "ready_for_review", "blocked", "done")
    }


def _path_is_allowed(
    *,
    path: str,
    allowed_paths: list[str],
    disallowed_paths: list[str],
    task_file_path: str,
    task_id: str,
) -> tuple[bool, str | None]:
    norm = _normalize_repo_relative_path(path)

    if norm == task_file_path:
        return True, None
    if norm in _task_projection_paths(task_file_path):
        return True, None
    if norm.startswith(".orchestrator/handoff/"):
        return True, None
    if norm.startswith("reports/status/swarm_runs/") and Path(norm).name.startswith(f"{task_id}_"):
        return True, None
    if norm.startswith("reports/status/reviews/") and Path(norm).name.startswith(f"{task_id}_"):
        return True, None
    if norm.startswith(".orchestrator/"):
        return False, "orchestrator_write_forbidden"

    for disallowed in disallowed_paths:
        if _path_matches_prefix(norm, disallowed):
            return False, f"disallowed_path:{disallowed}"

    for allowed in allowed_paths:
        if _path_matches_prefix(norm, allowed):
            return True, None

    return False, "outside_allowed_paths"


_OUTPUT_WILDCARD_TOKENS = ("...", "YYYY-MM-DD", "<", ">", "*", "?")


def _output_spec_is_safe(spec: str) -> tuple[bool, str | None]:
    norm = _normalize_repo_relative_path(spec)
    if not norm:
        return False, "empty_output_spec"
    if norm.startswith("/") or norm.startswith("~"):
        return False, "absolute_output_spec_forbidden"
    if norm == ".." or norm.startswith("../") or "/../" in norm:
        return False, "path_traversal_forbidden"
    return True, None


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


def _has_wildcards(segment: str) -> bool:
    return any(token in segment for token in _OUTPUT_WILDCARD_TOKENS)


def _find_paths_matching_output_spec(*, repo: Path, spec: str) -> list[Path]:
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
    for ext in (".py", ".md", ".json", ".csv", ".yml", ".yaml", ".svg", ".pdf", ".txt"):
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


def _check_declared_outputs_exist(*, repo: Path, task: Task) -> tuple[bool, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    for raw_spec in task.outputs:
        ok, reason = _output_spec_is_safe(raw_spec)
        if not ok:
            failures.append({"output": raw_spec, "reason": reason or "invalid_output_spec"})
            continue

        kind = _guess_output_kind(raw_spec)
        match_spec = _strip_trailing_ellipsis(raw_spec) if kind == "dir_nonempty" else raw_spec
        matches = _find_paths_matching_output_spec(repo=repo, spec=match_spec)

        if kind == "file":
            if not any(path.is_file() for path in matches):
                failures.append({"output": raw_spec, "reason": "missing_file"})
            continue
        if kind == "dir":
            if not any(path.is_dir() for path in matches):
                failures.append({"output": raw_spec, "reason": "missing_dir"})
            continue
        if kind == "dir_nonempty":
            found_nonempty = False
            for path in matches:
                if not path.is_dir():
                    continue
                try:
                    next(path.iterdir())
                    found_nonempty = True
                    break
                except (StopIteration, FileNotFoundError):
                    continue
            if not found_nonempty:
                failures.append({"output": raw_spec, "reason": "missing_or_empty_dir"})
            continue
        if not matches:
            failures.append({"output": raw_spec, "reason": "missing_path"})

    return len(failures) == 0, failures


def _task_requires_manifest(task: Task, prefix: str) -> bool:
    return any(_path_matches_prefix(output, prefix) for output in task.outputs)


def required_manifest_failures(repo: Path, task: Task) -> list[str]:
    failures: list[str] = []

    if _task_requires_manifest(task, "data/raw/"):
        raw_manifest_specs = [output for output in task.outputs if _path_matches_prefix(output, "data/raw_manifest/")]
        if not raw_manifest_specs:
            failures.append("missing_declared_raw_manifest_output")
        elif not any(_find_paths_matching_output_spec(repo=repo, spec=spec) for spec in raw_manifest_specs):
            failures.append("missing_raw_manifest_file")

    if _task_requires_manifest(task, "data/processed/"):
        processed_manifest_specs = [
            output for output in task.outputs if _path_matches_prefix(output, "data/processed_manifest/")
        ]
        if not processed_manifest_specs:
            failures.append("missing_declared_processed_manifest_output")
        elif not any(_find_paths_matching_output_spec(repo=repo, spec=spec) for spec in processed_manifest_specs):
            failures.append("missing_processed_manifest_file")

    return failures


def _next_json_artifact_path(directory: Path, task_id: str, timestamp: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{task_id}_{timestamp}.json"
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        retry = directory / f"{task_id}_{timestamp}_{index}.json"
        if not retry.exists():
            return retry
    return candidate


def _matching_task_jsons(directory: Path, task_id: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob(f"{task_id}_*.json") if path.is_file())


def _is_valid_run_manifest(path: Path, task_id: str) -> bool:
    try:
        data = json.loads(_read_text(path))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("schema_version") != SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        return False
    task = data.get("task")
    return isinstance(task, dict) and task.get("task_id") == task_id


def _is_valid_review_log(path: Path, task_id: str, scientific_review_role: str) -> bool:
    try:
        data = json.loads(_read_text(path))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("schema_version") != JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        return False
    reviewer = data.get("reviewer")
    task = data.get("task")
    decision = data.get("decision")
    return (
        isinstance(reviewer, dict)
        and reviewer.get("role") == scientific_review_role
        and isinstance(task, dict)
        and task.get("task_id") == task_id
        and isinstance(decision, dict)
        and decision.get("outcome") == "approve"
        and task.get("state_after") == "done"
    )


def _update_task_status_and_notes(*, task_path: Path, new_state: str, note_line: str) -> None:
    _shared_update_task_status_and_notes(
        task_path=task_path,
        new_state=new_state,
        note_line=note_line,
        allowed_states=DEFAULT_ALLOWED_STATES,
    )


def _codex_exec_cmd(
    *,
    prompt: str,
    model: str | None,
    sandbox: str,
    unattended: bool,
    allow_network: bool,
    workdir: Path,
) -> list[str]:
    codex = _which_or_none("codex")
    if codex is None:
        raise FileNotFoundError("codex_not_found")
    cmd: list[str] = [codex]
    if unattended:
        cmd.extend(["-a", "never"])
    cmd.extend(["exec", "--sandbox", sandbox])
    if model:
        cmd.extend(["-m", model])
    if allow_network:
        cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])
    cmd.extend(["-C", str(workdir), prompt])
    return cmd


def _run_gates(repo: Path, gates: list[str]) -> tuple[bool, list[dict[str, object]]]:
    outputs: list[dict[str, object]] = []
    all_ok = True
    for gate in gates:
        cp = subprocess.run(
            gate,
            cwd=str(repo),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        outputs.append(
            {
                "command": gate,
                "returncode": cp.returncode,
                "output_tail": (cp.stdout or "")[-4000:],
            }
        )
        if cp.returncode != 0:
            all_ok = False
    return all_ok, outputs


def _executor_prompt_path(task: Task, contract: FrameworkContract) -> Path:
    key = "operator" if task.role == "Operator" else "worker"
    return contract.prompt_templates[key]


def cmd_plan(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)

    done_ids = sorted(task_id for task_id, task in tasks.items() if task.state == "done")
    integration_ready_ids = sorted(task_id for task_id, task in tasks.items() if task.state == "integration_ready")
    claimed_ids = sorted(claimed_task_ids(repo, args.remote, args.base_branch))
    ready = ready_backlog_tasks(tasks, set(claimed_ids), contract)

    payload = {
        "done": done_ids,
        "integration_ready": integration_ready_ids,
        "claimed": claimed_ids,
        "quarantined": quarantined,
        "ready": [_task_summary(task) for task in ready],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)

    if args.unattended:
        _require_unattended_ack()
    if not args.dry_run:
        _preflight_strict_sync_requirements(
            cwd=repo,
            remote=args.remote,
            unattended=bool(args.unattended),
            create_pr=bool(args.create_pr),
        )

    tasks, quarantined = load_tasks_quarantined(contract)
    claimed_ids = claimed_task_ids(repo, args.remote, args.base_branch)
    ready = ready_backlog_tasks(tasks, claimed_ids, contract)

    capacity = max(0, int(args.max_workers))
    selected = choose_tasks_heuristic(ready, capacity)

    summary = {
        "done": sorted(task_id for task_id, task in tasks.items() if task.state == "done"),
        "integration_ready": sorted(task_id for task_id, task in tasks.items() if task.state == "integration_ready"),
        "claimed": sorted(claimed_ids),
        "quarantined": quarantined,
        "ready": [task.task_id for task in ready],
        "selected": [task.task_id for task in selected],
        "skipped": [],
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run or not selected:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    worktree_parent = Path(args.worktree_parent).expanduser().resolve() if args.worktree_parent else repo.parent
    worktree_parent.mkdir(parents=True, exist_ok=True)

    started: list[dict[str, str]] = []
    if args.runner == "tmux":
        _tmux_ensure_session(args.tmux_session, repo)
        if args.unattended:
            _tmux("set-environment", "-g", "SWARM_UNATTENDED_I_UNDERSTAND", "1")

    for task in selected:
        try:
            worktree_path, branch = ensure_worktree(
                repo=repo,
                task=task,
                worktree_parent=worktree_parent,
                base_ref=args.base_branch,
            )
        except WorktreeCollisionError as exc:
            summary["skipped"].append(
                {
                    "task_id": task.task_id,
                    "reason": "worktree_collision",
                    "worktree": str(exc.worktree_path),
                }
            )
            continue
        started.append(
            {
                "task_id": task.task_id,
                "branch": branch,
                "worktree": str(worktree_path),
            }
        )

        command = [
            sys.executable,
            "scripts/swarm.py",
            "run-task",
            "--task-id",
            task.task_id,
            "--remote",
            args.remote,
            "--base-branch",
            args.base_branch,
            "--codex-sandbox",
            args.codex_sandbox,
            "--final-state",
            args.final_state,
        ]
        if args.unattended:
            command.append("--unattended")
        if args.codex_model:
            command.extend(["--codex-model", args.codex_model])
        if args.max_worker_seconds:
            command.extend(["--max-worker-seconds", str(args.max_worker_seconds)])
        if args.create_pr:
            command.append("--create-pr")

        if args.runner == "tmux":
            _tmux_spawn_task_window(
                session=args.tmux_session,
                window_name=task.task_id,
                workdir=worktree_path,
                command=command,
            )
        else:
            _run(command, cwd=worktree_path, check=False)

    summary["started"] = started
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _loop_iteration(args: argparse.Namespace) -> int:
    repo = _repo_root()
    _supervisor_sync_to_remote_base(repo=repo, remote=args.remote, base_branch=args.base_branch)
    return cmd_tick(args)


def _handle_loop_failure(exc: BaseException, *, interval_seconds: int, consecutive_failures: int) -> int:
    print(
        f"[loop] escalation iteration_failed kind={type(exc).__name__} detail={exc}",
        file=sys.stderr,
    )
    return min(3600, interval_seconds * (2 ** max(0, consecutive_failures - 1)))


def _attempt_loop_iteration(
    args: argparse.Namespace,
    *,
    interval_seconds: int,
    consecutive_failures: int,
) -> tuple[int, int]:
    try:
        _loop_iteration(args)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        consecutive_failures += 1
        backoff_seconds = _handle_loop_failure(
            exc,
            interval_seconds=interval_seconds,
            consecutive_failures=consecutive_failures,
        )
        return consecutive_failures, backoff_seconds
    return 0, 0


def cmd_loop(args: argparse.Namespace) -> int:
    repo = _repo_root()

    if args.unattended:
        _require_unattended_ack()

    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )

    interval_seconds = max(5, int(args.interval_seconds))
    print(f"swarm_loop_started interval={interval_seconds}s repo={repo}")
    consecutive_failures = 0

    while True:
        try:
            consecutive_failures, backoff_seconds = _attempt_loop_iteration(
                args,
                interval_seconds=interval_seconds,
                consecutive_failures=consecutive_failures,
            )
            remaining = interval_seconds + backoff_seconds
            while remaining > 0:
                sleep_seconds = min(5, remaining)
                time.sleep(sleep_seconds)
                remaining -= sleep_seconds
        except KeyboardInterrupt:
            print("swarm_loop_stopped")
            return 0


def cmd_tmux_start(args: argparse.Namespace) -> int:
    repo = _repo_root()

    if args.unattended:
        _require_unattended_ack()

    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )

    _tmux_ensure_session(args.tmux_session, repo)
    if args.unattended:
        _tmux("set-environment", "-g", "SWARM_UNATTENDED_I_UNDERSTAND", "1")

    command = [
        sys.executable,
        "scripts/swarm.py",
        "loop",
        "--interval-seconds",
        str(args.interval_seconds),
        "--planner",
        args.planner,
        "--runner",
        "tmux",
        "--tmux-session",
        args.tmux_session,
        "--max-workers",
        str(args.max_workers),
        "--remote",
        args.remote,
        "--base-branch",
        args.base_branch,
        "--codex-sandbox",
        args.codex_sandbox,
        "--final-state",
        args.final_state,
    ]
    if args.worktree_parent:
        command.extend(["--worktree-parent", args.worktree_parent])
    if args.unattended:
        command.append("--unattended")
    if args.codex_model:
        command.extend(["--codex-model", args.codex_model])
    if args.max_worker_seconds:
        command.extend(["--max-worker-seconds", str(args.max_worker_seconds)])
    if args.create_pr:
        command.append("--create-pr")

    _tmux_spawn_task_window(
        session=args.tmux_session,
        window_name="supervisor",
        workdir=repo,
        command=command,
    )

    print(f"tmux_session_started:{args.tmux_session}")
    if args.attach:
        _tmux("attach", "-t", args.tmux_session, check=True, capture=False)
    return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks = load_tasks(contract)

    if args.unattended:
        _require_unattended_ack()

    _require_git_identity(cwd=repo, reason="runtime")
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )
    strict_sync = bool(args.unattended or args.create_pr)

    task = tasks.get(args.task_id)
    if task is None:
        raise SystemExit(f"unknown_task_id:{args.task_id}")

    if task.role not in set(contract.task_execution_roles):
        raise SystemExit(f"task_not_runtime_executable:{task.task_id}:{task.role}")

    if task.state not in {"backlog", "active", "blocked", "integration_ready"}:
        raise SystemExit(f"task_not_runnable_from_state:{task.task_id}:{task.state}")

    state_before = task.state
    if task.state == "backlog":
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="active",
            note_line=f"Claimed by local swarm runtime on branch {_git_current_branch(repo)}.",
        )
        _run(["git", "add", str(task.path)], cwd=repo, check=True)
        _git_commit(cwd=repo, message=f"{task.task_id}: claim active", strict=strict_sync)
        _git_push(
            cwd=repo,
            remote=args.remote,
            ref=_git_current_branch(repo),
            set_upstream=True,
            strict=strict_sync,
        )

    blocked_reasons: list[str] = []
    executor_command: list[str] = []
    executor_returncode: int | None = None
    executor_error: str | None = None
    executor_log_relpath: str | None = None

    if task.allow_network and task.workstream not in set(contract.network_workstreams):
        blocked_reasons.append("network_policy_violation")

    if args.final_state == "integration_ready":
        if not task_is_integration_ready_eligible(task, contract):
            blocked_reasons.append("integration_ready_ineligible")
        elif not downstream_allowlist_exists(task.task_id, tasks):
            blocked_reasons.append("integration_ready_missing_downstream_allowlist")

    run_timestamp = _utc_timestamp_compact()
    logs_dir = repo / "data" / "tmp" / "swarm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_executor and not blocked_reasons:
        try:
            prompt_path = _executor_prompt_path(task, contract)
            prompt = load_prompt(
                prompt_path,
                _build_prompt_context(task, repo, args.repair_context),
            )
            executor_command = _codex_exec_cmd(
                prompt=prompt,
                model=args.codex_model,
                sandbox=args.codex_sandbox,
                unattended=args.unattended,
                allow_network=task.allow_network,
                workdir=repo,
            )
            cp = _run(
                executor_command,
                cwd=repo,
                capture=True,
                check=False,
                timeout_seconds=int(args.max_worker_seconds) if args.max_worker_seconds else None,
            )
            executor_returncode = cp.returncode
            executor_log_path = logs_dir / f"{task.task_id}_{run_timestamp}_executor.log"
            executor_log_path.write_text(cp.stdout or "", encoding="utf-8")
            executor_log_relpath = executor_log_path.relative_to(repo).as_posix()
            if cp.returncode != 0:
                blocked_reasons.append("executor_failed")
        except subprocess.TimeoutExpired:
            executor_error = "executor_timeout"
            blocked_reasons.append("executor_timeout")
        except Exception as exc:
            executor_error = str(exc)
            blocked_reasons.append("executor_unavailable")
    elif args.skip_executor:
        executor_error = "executor_skipped"

    gate_ok, gate_outputs = _run_gates(repo, task.gates)
    if not gate_ok:
        blocked_reasons.append("gates_failed")

    base_ref = _resolve_base_ref_for_diff(cwd=repo, base_branch=args.base_branch, remote=args.remote)
    ownership_failures: list[dict[str, str]] = []
    changed_paths: list[str] = []
    if base_ref is None:
        ownership_failures.append(
            {
                "path": args.base_branch,
                "reason": "base_ref_unresolved",
                "sources": "committed",
            }
        )
    else:
        path_sources, ops = _collect_changed_paths_with_sources(repo=repo, base_ref=base_ref)
        changed_paths = sorted(path_sources.keys())
        task_file_rel = task.path.relative_to(repo).as_posix()

        for op in ops:
            if op.get("code") == "R" and op.get("old_path") == task_file_rel and op.get("path") != task_file_rel:
                ownership_failures.append(
                    {
                        "path": f"{op.get('old_path')} -> {op.get('path')}",
                        "reason": "task_file_moved",
                        "sources": str(op.get("source", "unknown")),
                    }
                )
            if op.get("code") == "D" and op.get("path") == task_file_rel:
                ownership_failures.append(
                    {
                        "path": task_file_rel,
                        "reason": "task_file_deleted",
                        "sources": str(op.get("source", "unknown")),
                    }
                )

        seen: set[tuple[str, str]] = set()
        for changed_path in changed_paths:
            ok, reason = _path_is_allowed(
                path=changed_path,
                allowed_paths=task.allowed_paths,
                disallowed_paths=task.disallowed_paths,
                task_file_path=task_file_rel,
                task_id=task.task_id,
            )
            if ok:
                continue
            key = (changed_path, reason or "unknown")
            if key in seen:
                continue
            seen.add(key)
            ownership_failures.append(
                {
                    "path": changed_path,
                    "reason": reason or "unknown",
                    "sources": ",".join(sorted(path_sources.get(changed_path, set()))),
                }
            )

    if ownership_failures:
        blocked_reasons.append("path_ownership_violation")

    outputs_ok, output_failures = _check_declared_outputs_exist(repo=repo, task=task)
    if not outputs_ok:
        blocked_reasons.append("missing_outputs")

    manifest_failures = required_manifest_failures(repo, task)
    if manifest_failures:
        blocked_reasons.append("missing_required_manifests")

    task_state_after_executor = load_task(task.path, contract).state
    if task_state_after_executor == "blocked":
        blocked_reasons.append("task_marked_blocked")

    blocked_reasons = _dedupe_preserve(blocked_reasons)
    if blocked_reasons:
        state_after = "blocked"
    elif task_state_after_executor in {"active", "integration_ready", "ready_for_review"}:
        # Respect the task state the worker left behind. The runtime should not
        # silently promote an active task to ready_for_review just because the
        # default final_state is reviewable.
        state_after = task_state_after_executor
    else:
        state_after = args.final_state

    run_manifest_path = _next_json_artifact_path(contract.run_manifest_dir, task.task_id, run_timestamp)
    run_manifest_relpath = run_manifest_path.relative_to(repo).as_posix()

    run_manifest = {
        "schema_version": SWARM_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": f"{task.task_id}_{run_timestamp}",
        "generated_at_utc": _utc_now_iso(),
        "task": {
            "task_id": task.task_id,
            "task_path": task.path.relative_to(repo).as_posix(),
            "title": task.title,
            "role": task.role,
            "workstream": task.workstream,
            "task_kind": task.task_kind,
            "dependencies": list(task.dependencies),
            "integration_ready_dependencies": list(task.integration_ready_dependencies),
            "state_before": state_before,
            "state_after": state_after,
        },
        "repo": {
            "branch": _git_current_branch(repo),
            "git_sha": _git_head_sha(repo),
            "base_branch": args.base_branch,
            "remote": args.remote,
        },
        "executor": {
            "role": task.role,
            "runner": "local_swarm",
            "tool": "codex" if not args.skip_executor else "manual",
            "model": args.codex_model,
            "sandbox": args.codex_sandbox,
            "allow_network": task.allow_network,
            "repair_context": args.repair_context,
            "returncode": executor_returncode,
            "error": executor_error,
        },
        "commands": {
            "executor": executor_command,
            "executor_log_path": executor_log_relpath,
            "gates": list(task.gates),
        },
        "gates": gate_outputs,
        "ownership": {
            "ok": not ownership_failures,
            "changed_paths": changed_paths,
            "violations": ownership_failures,
        },
        "artifacts": {
            "outputs_ok": outputs_ok,
            "missing_outputs": output_failures,
            "required_manifests_ok": not manifest_failures,
            "missing_manifests": manifest_failures,
            "run_manifest_path": run_manifest_relpath,
        },
        "result": {
            "status": "ok" if state_after != "blocked" else "blocked",
            "blocked_reasons": blocked_reasons,
        },
    }
    _write_json(run_manifest_path, run_manifest)

    if state_after == "integration_ready":
        note = (
            f"Runtime passed: outputs, gates, manifests, and run manifest are present. "
            f"Marked integration_ready for explicitly allowlisted downstream consumers. "
            f"Run manifest: {run_manifest_relpath}"
        )
    elif state_after == "ready_for_review":
        note = (
            f"Runtime passed: outputs, gates, manifests, and run manifest are present. "
            f"Ready for Judge review. Run manifest: {run_manifest_relpath}"
        )
    elif state_after == "active":
        note = (
            f"Runtime completed without promotion; preserving worker state active. "
            f"Run manifest: {run_manifest_relpath}"
        )
    else:
        details: list[str] = []
        if ownership_failures:
            details.append(
                "ownership="
                + "; ".join(f"{item['path']}[{item['sources']}]={item['reason']}" for item in ownership_failures)
            )
        if output_failures:
            details.append(
                "outputs=" + "; ".join(f"{item['output']}={item['reason']}" for item in output_failures)
            )
        if manifest_failures:
            details.append("manifests=" + ",".join(manifest_failures))
        note = (
            f"@human Runtime blocked: {', '.join(blocked_reasons)}. "
            f"Run manifest: {run_manifest_relpath}. "
            + " ".join(details)
        ).strip()

    _update_task_status_and_notes(task_path=task.path, new_state=state_after, note_line=note)

    if _git_has_changes(repo):
        _run(["git", "add", "-A"], cwd=repo, check=True)
        _git_commit(cwd=repo, message=f"{task.task_id}: {state_after}", strict=strict_sync)
        _git_push(
            cwd=repo,
            remote=args.remote,
            ref=_git_current_branch(repo),
            set_upstream=True,
            strict=strict_sync,
        )

    if args.create_pr and state_after in {"integration_ready", "ready_for_review"}:
        _gh_create_pr_if_missing(
            cwd=repo,
            base_branch=args.base_branch,
            title=f"{task.task_id}: {task.title}",
            body="\n".join(
                [
                    f"Task: `{task.path.relative_to(repo).as_posix()}`",
                    f"State: `{state_after}`",
                    f"Run manifest: `{run_manifest_relpath}`",
                    "",
                    "Deterministic gates:",
                    *[f"- `{item['command']}` (rc={item['returncode']})" for item in gate_outputs],
                ]
            ),
        )

    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "state_before": state_before,
                "state_after": state_after,
                "run_manifest": run_manifest_relpath,
                "blocked_reasons": blocked_reasons,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if state_after != "blocked" else 1


def cmd_judge_task(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks = load_tasks(contract)

    if args.unattended:
        _require_unattended_ack()

    _require_git_identity(cwd=repo, reason="judge")
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=False,
    )
    strict_sync = bool(args.unattended)

    task = tasks.get(args.task_id)
    if task is None:
        raise SystemExit(f"unknown_task_id:{args.task_id}")
    if task.state != "ready_for_review":
        raise SystemExit(f"task_not_ready_for_review:{task.task_id}:{task.state}")

    gate_ok, gate_outputs = _run_gates(repo, task.gates)
    outputs_ok, output_failures = _check_declared_outputs_exist(repo=repo, task=task)
    manifest_failures = required_manifest_failures(repo, task)

    valid_run_manifests = [
        path for path in _matching_task_jsons(contract.run_manifest_dir, task.task_id) if _is_valid_run_manifest(path, task.task_id)
    ]
    review_bundle_failures: list[str] = []
    if not valid_run_manifests:
        review_bundle_failures.append("missing_valid_run_manifest")

    approved = gate_ok and outputs_ok and not manifest_failures and not review_bundle_failures
    state_after = "done" if approved else args.on_fail

    review_log_path = _next_json_artifact_path(contract.judge_review_dir, task.task_id, _utc_timestamp_compact())
    review_log_relpath = review_log_path.relative_to(repo).as_posix()
    run_manifest_relpath = (
        valid_run_manifests[-1].relative_to(repo).as_posix() if valid_run_manifests else None
    )

    check_failures: list[str] = []
    if not gate_ok:
        check_failures.append("gates_failed")
    if not outputs_ok:
        check_failures.extend(f"missing_output:{item['output']}:{item['reason']}" for item in output_failures)
    check_failures.extend(f"manifest:{reason}" for reason in manifest_failures)
    check_failures.extend(review_bundle_failures)

    note_prefix = args.note.strip() if isinstance(args.note, str) and args.note.strip() else ""
    decision_note = (
        f"{note_prefix} Judge approved deterministic review."
        if approved
        else f"{note_prefix} Judge returned task with failures: {', '.join(check_failures)}."
    ).strip()

    review_log = {
        "schema_version": JUDGE_REVIEW_LOG_SCHEMA_VERSION,
        "review_id": f"{task.task_id}_{_utc_timestamp_compact()}",
        "generated_at_utc": _utc_now_iso(),
        "reviewer": {
            "role": contract.scientific_review_role,
        },
        "task": {
            "task_id": task.task_id,
            "task_path": task.path.relative_to(repo).as_posix(),
            "role": task.role,
            "state_before": task.state,
            "state_after": state_after,
            "run_manifest_path": run_manifest_relpath,
        },
        "checks": {
            "gates_ok": gate_ok,
            "outputs_ok": outputs_ok,
            "required_manifests_ok": not manifest_failures,
            "review_bundle_ok": not review_bundle_failures,
            "failures": check_failures,
        },
        "decision": {
            "outcome": "approve" if approved else ("block" if args.on_fail == "blocked" else "revise"),
            "note": decision_note,
        },
    }
    _write_json(review_log_path, review_log)

    task_note = (
        f"Judge approved; review log: {review_log_relpath}"
        if approved
        else f"@human Judge returned task; review log: {review_log_relpath}; failures: {', '.join(check_failures)}"
    )
    _update_task_status_and_notes(task_path=task.path, new_state=state_after, note_line=task_note)

    if _git_has_changes(repo):
        _run(["git", "add", "-A"], cwd=repo, check=True)
        _git_commit(cwd=repo, message=f"{task.task_id}: {state_after}", strict=strict_sync)
        _git_push(
            cwd=repo,
            remote=args.remote,
            ref=_git_current_branch(repo),
            set_upstream=True,
            strict=strict_sync,
        )

    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "state_before": task.state,
                "state_after": state_after,
                "review_log": review_log_relpath,
                "approved": approved,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if approved else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarm.py")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan = subparsers.add_parser("plan", help="Print done/claimed/ready task status as JSON")
    plan.add_argument("--remote", default="origin")
    plan.add_argument("--base-branch", default="main")
    plan.set_defaults(func=cmd_plan)

    tick = subparsers.add_parser("tick", help="Start ready tasks")
    tick.add_argument("--planner", choices=["heuristic"], default="heuristic")
    tick.add_argument("--runner", choices=["tmux", "local"], default="tmux")
    tick.add_argument("--tmux-session", default="swarm")
    tick.add_argument("--max-workers", type=int, default=1)
    tick.add_argument("--worktree-parent", default=None)
    tick.add_argument("--remote", default="origin")
    tick.add_argument("--base-branch", default="main")
    tick.add_argument("--codex-model", default=None)
    tick.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    tick.add_argument("--unattended", action="store_true")
    tick.add_argument("--max-worker-seconds", type=int, default=0)
    tick.add_argument("--create-pr", action="store_true")
    tick.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    tick.add_argument("--dry-run", action="store_true")
    tick.set_defaults(func=cmd_tick)

    loop = subparsers.add_parser("loop", help="Run tick repeatedly")
    loop.add_argument("--interval-seconds", type=int, default=300)
    loop.add_argument("--planner", choices=["heuristic"], default="heuristic")
    loop.add_argument("--runner", choices=["tmux", "local"], default="tmux")
    loop.add_argument("--tmux-session", default="swarm")
    loop.add_argument("--max-workers", type=int, default=1)
    loop.add_argument("--worktree-parent", default=None)
    loop.add_argument("--remote", default="origin")
    loop.add_argument("--base-branch", default="main")
    loop.add_argument("--codex-model", default=None)
    loop.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    loop.add_argument("--unattended", action="store_true")
    loop.add_argument("--max-worker-seconds", type=int, default=0)
    loop.add_argument("--create-pr", action="store_true")
    loop.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    loop.add_argument("--dry-run", action="store_true")
    loop.set_defaults(func=cmd_loop)

    tmux_start = subparsers.add_parser("tmux-start", help="Create a tmux session and launch the supervisor loop")
    tmux_start.add_argument("--tmux-session", default="swarm")
    tmux_start.add_argument("--attach", action="store_true")
    tmux_start.add_argument("--interval-seconds", type=int, default=300)
    tmux_start.add_argument("--planner", choices=["heuristic"], default="heuristic")
    tmux_start.add_argument("--max-workers", type=int, default=1)
    tmux_start.add_argument("--worktree-parent", default=None)
    tmux_start.add_argument("--remote", default="origin")
    tmux_start.add_argument("--base-branch", default="main")
    tmux_start.add_argument("--codex-model", default=None)
    tmux_start.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    tmux_start.add_argument("--unattended", action="store_true")
    tmux_start.add_argument("--max-worker-seconds", type=int, default=0)
    tmux_start.add_argument("--create-pr", action="store_true")
    tmux_start.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    tmux_start.set_defaults(func=cmd_tmux_start)

    run_task = subparsers.add_parser("run-task", help="Execute one Worker/Operator task in the current worktree")
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--remote", default="origin")
    run_task.add_argument("--base-branch", default="main")
    run_task.add_argument("--codex-model", default=None)
    run_task.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    run_task.add_argument("--unattended", action="store_true")
    run_task.add_argument("--skip-executor", action="store_true")
    run_task.add_argument("--max-worker-seconds", type=int, default=0)
    run_task.add_argument("--repair-context", default=None)
    run_task.add_argument("--create-pr", action="store_true")
    run_task.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    run_task.set_defaults(func=cmd_run_task)

    judge_task = subparsers.add_parser("judge-task", help="Perform deterministic Judge review for one ready_for_review task")
    judge_task.add_argument("--task-id", required=True)
    judge_task.add_argument("--remote", default="origin")
    judge_task.add_argument("--base-branch", default="main")
    judge_task.add_argument("--unattended", action="store_true")
    judge_task.add_argument("--on-fail", choices=["active", "blocked"], default="blocked")
    judge_task.add_argument("--note", default="")
    judge_task.set_defaults(func=cmd_judge_task)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
