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

The public operator-facing commands are:

- `status`
- `plan`
- `tick`
- `supervise`
- `loop`
- `tmux-start`

Internal helper commands used by the supervisor:

- `run-task`
- `judge-task`
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any, Iterable


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import swarm_claims
import swarm_events
import sweep_tasks
from swarm_taskfile import WorktreeCollisionError
from swarm_taskfile import extract_section as _extract_section
from swarm_taskfile import parse_status_value as _parse_status_value
from swarm_taskfile import parse_task_frontmatter as _parse_task_frontmatter
from swarm_taskfile import parse_task_id_from_branch as _parse_task_id_from_branch
from swarm_taskfile import update_task_status_and_notes as _shared_update_task_status_and_notes


SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v2"
SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 = "research_swarm.runtime_run_manifest.v1"
JUDGE_REVIEW_LOG_SCHEMA_VERSION = "research_swarm.judge_review_log.v2"
MOCK_TRANSCRIPT_SCHEMA_VERSION = "research_swarm.mock_transcript.v1"
EXECUTOR_SESSION_SCHEMA_VERSION = "research_swarm.executor_session.v1"

EXECUTOR_LOG_MAX_BYTES = 128 * 1024
EXECUTOR_LOG_SEGMENT_BYTES = 64 * 1024
EXECUTOR_SESSION_SEGMENT_BYTES = 16 * 1024

DEFAULT_REVIEW_MIN_SEPARATION_SECONDS = 60
DEFAULT_REPAIR_MAX_ATTEMPTS = 2
DEFAULT_MAX_READY_FOR_REVIEW = 4

GATE_OUTPUT_SEGMENT_BYTES = 8 * 1024
DEFAULT_GATE_INTERPRETER_ALLOWLIST = ("python", "python3", "make")
DEFAULT_GATE_TIMEOUT_SECONDS = 600
GATE_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM")

# One actor session id per process: a review written by the same session that
# produced the run manifest it reviews is invalid (§4.0 #17 actor separation).
_ACTOR_SESSION_ID = os.environ.get("SWARM_ACTOR_SESSION", "").strip() or uuid.uuid4().hex

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
    review_min_separation_seconds: int
    gate_interpreter_allowlist: tuple[str, ...]
    gate_timeout_seconds: int
    repair_max_attempts: int
    wip_max_active: int | None
    wip_max_ready_for_review: int
    budget_max_program_usd: float | None
    claim_lease_ttl_seconds: int


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


@dataclasses.dataclass(frozen=True)
class ExecutorOutcome:
    returncode: int
    stdout: str
    wall_clock_seconds: float
    usage: dict[str, object] | None
    transcript_path: str | None


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_timestamp_compact() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_swarm_event(
    repo: Path,
    event: dict[str, object],
    *,
    escalation: bool = False,
) -> dict | None:
    """Best-effort journaling: runtime exit semantics never depend on it."""
    try:
        event_repo_raw = os.environ.get("SWARM_EVENT_REPO_ROOT", "").strip()
        event_repo = Path(event_repo_raw).expanduser().resolve() if event_repo_raw else repo
        writer = swarm_events.escalate if escalation else swarm_events.append_event
        return writer(event_repo, event, actor_session=_ACTOR_SESSION_ID)
    except Exception as exc:
        print(
            f"[warn] event journal failed event={event.get('event')} "
            f"error={type(exc).__name__}:{exc}",
            file=sys.stderr,
        )
        return None


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


def _invoke_executor(
    *,
    command: list[str],
    cwd: Path,
    timeout_seconds: int | None,
) -> subprocess.CompletedProcess[str]:
    return _run(
        command,
        cwd=cwd,
        capture=True,
        check=False,
        timeout_seconds=timeout_seconds,
    )


def _parse_codex_usage(stdout: str) -> dict[str, object] | None:
    """Best-effort parser for token splits printed by the Codex CLI.

    A total-only ``tokens used`` line is deliberately not attributed to either
    input or output: doing so would make cost estimation look more precise than
    the executor output permits.
    """
    if not isinstance(stdout, str) or not stdout:
        return None

    number = r"(\d[\d,_]*)"
    input_matches = re.findall(
        rf"\binput(?:\s+tokens?)?\s*[:=]\s*{number}",
        stdout,
        flags=re.IGNORECASE,
    )
    output_matches = re.findall(
        rf"\boutput(?:\s+tokens?)?\s*[:=]\s*{number}",
        stdout,
        flags=re.IGNORECASE,
    )
    if not input_matches and not output_matches:
        return None

    usage: dict[str, object] = {"source": "parsed"}
    if input_matches:
        usage["input_tokens"] = int(input_matches[-1].replace(",", "").replace("_", ""))
    if output_matches:
        usage["output_tokens"] = int(output_matches[-1].replace(",", "").replace("_", ""))
    return usage


def _mock_transcript_relpath(task_id: str) -> str:
    return f".orchestrator/mock_transcripts/{task_id}.json"


def _safe_mock_action_path(repo: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("mock_action_path_invalid")
    normalized = raw_path.replace("\\", "/")
    if (
        Path(raw_path).is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or any(part == ".." for part in normalized.split("/"))
    ):
        raise ValueError(f"mock_action_path_forbidden:{raw_path}")
    candidate = (repo / raw_path).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"mock_action_path_forbidden:{raw_path}") from exc
    return candidate


def _mock_usage(raw_usage: object) -> dict[str, object] | None:
    if raw_usage is None:
        return None
    if not isinstance(raw_usage, dict):
        raise ValueError("mock_usage_invalid")
    usage: dict[str, object] = {"source": "mock_transcript"}
    for key in ("input_tokens", "output_tokens"):
        value = raw_usage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"mock_usage_invalid:{key}")
        usage[key] = value
    return usage


def _run_mock_transcript(*, repo: Path, task: Task) -> tuple[int, str, dict[str, object] | None, str]:
    transcript_relpath = _mock_transcript_relpath(task.task_id)
    transcript_path = repo / transcript_relpath
    if not transcript_path.is_file():
        return 1, f"mock_transcript_missing:{transcript_relpath}", None, transcript_relpath

    try:
        payload = json.loads(_read_text(transcript_path))
        if not isinstance(payload, dict):
            raise ValueError("mock_transcript_not_object")
        if payload.get("schema_version") != MOCK_TRANSCRIPT_SCHEMA_VERSION:
            raise ValueError(f"mock_transcript_schema_invalid:{payload.get('schema_version')}")
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError("mock_transcript_actions_invalid")

        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"mock_transcript_action_invalid:{index}")
            action_keys = [
                key
                for key in ("write", "append", "set_task_state", "note", "sleep_seconds")
                if key in action
            ]
            if len(action_keys) != 1:
                raise ValueError(f"mock_transcript_action_invalid:{index}")
            action_key = action_keys[0]
            if action_key in {"write", "append"}:
                target = _safe_mock_action_path(repo, action[action_key])
                rel = target.relative_to(repo.resolve()).as_posix()
                denied_prefixes = (".git", "reports/status/", ".orchestrator/mock_transcripts/")
                if rel == ".git" or any(
                    rel == prefix.rstrip("/") or rel.startswith(prefix)
                    for prefix in denied_prefixes
                ):
                    raise ValueError(f"mock_transcript_path_denied:{rel}")
                task_file_rel = task.path.resolve().relative_to(repo.resolve()).as_posix()
                allowed, reason = _path_is_allowed(
                    path=rel,
                    allowed_paths=task.allowed_paths,
                    disallowed_paths=task.disallowed_paths,
                    task_file_path=task_file_rel,
                    task_id=task.task_id,
                )
                if not allowed:
                    raise ValueError(f"mock_transcript_path_denied:{rel}:{reason}")
                content = action.get("content")
                if not isinstance(content, str):
                    raise ValueError(f"mock_transcript_content_invalid:{index}")
                target.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if action_key == "append" else "w"
                with target.open(mode, encoding="utf-8") as handle:
                    handle.write(content)
            elif action_key == "set_task_state":
                new_state = action[action_key]
                if not isinstance(new_state, str):
                    raise ValueError(f"mock_transcript_state_invalid:{index}")
                _update_task_status_and_notes(
                    task_path=task.path,
                    new_state=new_state,
                    note_line=f"Mock transcript set task state to {new_state}.",
                )
            elif action_key == "note":
                note = action[action_key]
                if not isinstance(note, str):
                    raise ValueError(f"mock_transcript_note_invalid:{index}")
                current_state = _parse_status_value(_read_text(task.path), "State")
                if current_state is None:
                    raise ValueError("mock_transcript_task_state_missing")
                _update_task_status_and_notes(
                    task_path=task.path,
                    new_state=current_state,
                    note_line=note,
                )
            else:
                seconds = action[action_key]
                if (
                    not isinstance(seconds, (int, float))
                    or isinstance(seconds, bool)
                    or seconds < 0
                ):
                    raise ValueError(f"mock_transcript_sleep_invalid:{index}")
                if seconds > 30:
                    raise ValueError(f"mock_transcript_sleep_too_long:{index}:{seconds}")
                time.sleep(float(seconds))

        returncode = payload.get("returncode")
        stdout = payload.get("stdout")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise ValueError("mock_transcript_returncode_invalid")
        if not isinstance(stdout, str):
            raise ValueError("mock_transcript_stdout_invalid")
        usage = _mock_usage(payload.get("usage"))
        return returncode, stdout, usage, transcript_relpath
    except Exception as exc:
        detail = str(exc).replace("\n", " ").strip()
        return (
            1,
            f"mock_transcript_error:{type(exc).__name__}:{detail}",
            None,
            transcript_relpath,
        )


def _execute_task(
    *,
    backend: str,
    task: Task,
    prompt: str,
    args: argparse.Namespace,
    repo: Path,
    timeout_seconds: int | None,
) -> ExecutorOutcome:
    started = time.perf_counter()
    if backend == "mock":
        returncode, stdout, usage, transcript_path = _run_mock_transcript(repo=repo, task=task)
    elif backend == "codex":
        prepared_command = getattr(args, "_executor_command", None)
        command = (
            list(prepared_command)
            if isinstance(prepared_command, list)
            else _codex_exec_cmd(
                prompt=prompt,
                model=getattr(args, "codex_model", None),
                sandbox=getattr(args, "codex_sandbox", "workspace-write"),
                unattended=bool(getattr(args, "unattended", False)),
                allow_network=task.allow_network,
                workdir=repo,
            )
        )
        cp = _invoke_executor(command=command, cwd=repo, timeout_seconds=timeout_seconds)
        returncode = cp.returncode
        stdout = cp.stdout or ""
        usage = _parse_codex_usage(stdout)
        transcript_path = None
    else:
        raise ValueError(f"unsupported_executor_backend:{backend}")

    return ExecutorOutcome(
        returncode=returncode,
        stdout=stdout,
        wall_clock_seconds=max(0.0, time.perf_counter() - started),
        usage=usage,
        transcript_path=transcript_path,
    )


def _executor_output_bytes(output: object) -> bytes:
    if isinstance(output, bytes):
        return output
    if isinstance(output, str):
        return output.encode("utf-8")
    return b""


def _write_executor_log(*, repo: Path, run_id: str, output: object) -> tuple[str, str]:
    raw = _executor_output_bytes(output)
    if len(raw) > EXECUTOR_LOG_MAX_BYTES:
        truncated_bytes = len(raw) - (2 * EXECUTOR_LOG_SEGMENT_BYTES)
        marker = f"\n...[truncated {truncated_bytes} bytes]...\n".encode("utf-8")
        raw = raw[:EXECUTOR_LOG_SEGMENT_BYTES] + marker + raw[-EXECUTOR_LOG_SEGMENT_BYTES:]

    log_path = repo / "reports" / "status" / "swarm_runs" / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(raw)
    return log_path.relative_to(repo).as_posix(), hashlib.sha256(raw).hexdigest()


def _redact_argv_env_values(argv: list[str]) -> list[str]:
    sensitive_fragments = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "API_KEY")
    environment = [(key, value) for key, value in os.environ.items() if value]
    redacted: list[str] = []
    for raw_arg in argv:
        arg = str(raw_arg)
        for key, value in environment:
            marker = f"<redacted-env:{key}>"
            if arg == value:
                arg = marker
            elif arg == f"{key}={value}":
                arg = f"{key}={marker}"
            elif any(fragment in key.upper() for fragment in sensitive_fragments) and value in arg:
                arg = arg.replace(value, marker)
        redacted.append(arg)
    return redacted


def _write_executor_session(
    *,
    repo: Path,
    run_id: str,
    backend: str,
    argv: list[str],
    returncode: int | None,
    wall_clock_seconds: float,
    stdout: object,
    usage: dict[str, object],
) -> str:
    raw = _executor_output_bytes(stdout)
    head = raw[:EXECUTOR_SESSION_SEGMENT_BYTES].decode("utf-8", errors="replace")
    tail = raw[-EXECUTOR_SESSION_SEGMENT_BYTES:].decode("utf-8", errors="replace")
    session_path = repo / "reports" / "status" / "swarm_runs" / "sessions" / f"{run_id}.json"
    _write_json(
        session_path,
        {
            "schema_version": EXECUTOR_SESSION_SCHEMA_VERSION,
            "run_id": run_id,
            "backend": backend,
            "argv": _redact_argv_env_values(argv),
            "returncode": returncode,
            "wall_clock_seconds": wall_clock_seconds,
            "stdout_head": head,
            "stdout_tail": tail,
            "stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "usage": usage,
        },
    )
    return session_path.relative_to(repo).as_posix()


def _usage_with_cost_estimate(
    *,
    repo: Path,
    model: object,
    wall_clock_seconds: float,
    captured_usage: dict[str, object] | None,
) -> dict[str, object]:
    usage: dict[str, object] = {
        "wall_clock_seconds": round(max(0.0, wall_clock_seconds), 6),
        "source": "unavailable",
    }
    if captured_usage is not None:
        usage.update(captured_usage)

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not (
        isinstance(model, str)
        and model
        and isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
    ):
        return usage

    try:
        framework = json.loads(_read_text(repo / "contracts" / "framework.json"))
    except (OSError, json.JSONDecodeError):
        return usage
    pricing = framework.get("pricing") if isinstance(framework, dict) else None
    model_pricing = pricing.get(model) if isinstance(pricing, dict) else None
    if not isinstance(model_pricing, dict):
        return usage
    input_rate = model_pricing.get("input_per_mtok_usd")
    output_rate = model_pricing.get("output_per_mtok_usd")
    if not (
        isinstance(input_rate, (int, float))
        and not isinstance(input_rate, bool)
        and isinstance(output_rate, (int, float))
        and not isinstance(output_rate, bool)
    ):
        return usage

    estimated = ((input_tokens * float(input_rate)) + (output_tokens * float(output_rate))) / 1_000_000
    usage["estimated_cost_usd"] = round(estimated, 4)
    usage["pricing_source"] = "framework_contract"
    return usage


def _task_frontmatter_snapshot(path: Path) -> tuple[str, dict[str, object]]:
    text = _read_text(path)
    lines = text.splitlines(keepends=True)
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError(f"missing_yaml_frontmatter:{path}")

    end_idx = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end_idx is None:
        raise ValueError(f"missing_yaml_frontmatter:{path}")

    parsed = _parse_task_frontmatter(text)
    if parsed is None:
        raise ValueError(f"missing_yaml_frontmatter:{path}")
    return "".join(lines[1:end_idx]), parsed


def _frontmatter_tampered_keys(
    pinned: dict[str, object],
    current: dict[str, object] | None,
) -> list[str]:
    current_data = current or {}
    return sorted(
        key
        for key in set(pinned) | set(current_data)
        if pinned.get(key) != current_data.get(key) or (key in pinned) != (key in current_data)
    )


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

    review_min_separation_raw = (
        review_bundle.get("min_separation_seconds") if isinstance(review_bundle, dict) else None
    )
    try:
        review_min_separation_seconds = int(review_min_separation_raw)
    except (TypeError, ValueError):
        review_min_separation_seconds = DEFAULT_REVIEW_MIN_SEPARATION_SECONDS

    gate_execution = raw.get("gate_execution")
    gate_interpreter_allowlist = tuple(
        _coerce_str_list(gate_execution.get("interpreter_allowlist") if isinstance(gate_execution, dict) else None)
        or list(DEFAULT_GATE_INTERPRETER_ALLOWLIST)
    )
    try:
        gate_timeout_seconds = int(gate_execution.get("timeout_seconds")) if isinstance(gate_execution, dict) else DEFAULT_GATE_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        gate_timeout_seconds = DEFAULT_GATE_TIMEOUT_SECONDS

    repair = raw.get("repair")
    try:
        repair_max_attempts = int(repair.get("max_attempts")) if isinstance(repair, dict) else DEFAULT_REPAIR_MAX_ATTEMPTS
    except (TypeError, ValueError):
        repair_max_attempts = DEFAULT_REPAIR_MAX_ATTEMPTS
    repair_max_attempts = max(0, repair_max_attempts)

    wip = raw.get("wip")
    try:
        configured_max_active = int(wip.get("max_active")) if isinstance(wip, dict) and wip.get("max_active") is not None else None
    except (TypeError, ValueError):
        configured_max_active = None
    wip_max_active = max(0, configured_max_active) if configured_max_active is not None else None
    try:
        wip_max_ready_for_review = int(wip.get("max_ready_for_review")) if isinstance(wip, dict) else DEFAULT_MAX_READY_FOR_REVIEW
    except (TypeError, ValueError):
        wip_max_ready_for_review = DEFAULT_MAX_READY_FOR_REVIEW
    wip_max_ready_for_review = max(0, wip_max_ready_for_review)

    budgets = raw.get("budgets")
    budget_raw = budgets.get("max_program_usd") if isinstance(budgets, dict) else None
    budget_max_program_usd = (
        float(budget_raw)
        if isinstance(budget_raw, (int, float)) and not isinstance(budget_raw, bool)
        else None
    )

    claims = raw.get("claims")
    try:
        claim_lease_ttl_seconds = (
            int(claims.get("lease_ttl_seconds"))
            if isinstance(claims, dict)
            else swarm_claims.DEFAULT_LEASE_TTL_SECONDS
        )
    except (TypeError, ValueError):
        claim_lease_ttl_seconds = swarm_claims.DEFAULT_LEASE_TTL_SECONDS
    claim_lease_ttl_seconds = max(0, claim_lease_ttl_seconds)

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
        review_min_separation_seconds=review_min_separation_seconds,
        gate_interpreter_allowlist=gate_interpreter_allowlist,
        gate_timeout_seconds=gate_timeout_seconds,
        repair_max_attempts=repair_max_attempts,
        wip_max_active=wip_max_active,
        wip_max_ready_for_review=wip_max_ready_for_review,
        budget_max_program_usd=budget_max_program_usd,
        claim_lease_ttl_seconds=claim_lease_ttl_seconds,
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
    try:
        claimed: set[str] = set(swarm_claims.read_claims(repo, remote))
    except Exception:
        claimed = set()

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


def _git_commit(*, cwd: Path, message: str, strict: bool, paths: list[str] | None = None) -> None:
    command = ["git", "commit", "-m", message]
    if paths:
        command.extend(["--", *paths])
    cp = _run(command, cwd=cwd, capture=True, check=False)
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


CONTAINMENT_MARKER_RELPATH = ".swarm/containment.json"
VENDOR_ACK_RELPATH = ".swarm/vendor_policy_ack.json"
CONTAINMENT_MARKER_SCHEMA_VERSION = "research_swarm.containment_marker.v1"
VENDOR_ACK_SCHEMA_VERSION = "research_swarm.vendor_policy_ack.v1"

# Credential classes whose readability disproves containment (§9.4): an
# unattended swarm must not run where user-level credentials beyond a scoped
# deploy key are readable.
_SENSITIVE_CREDENTIAL_PATHS = (
    (".aws/credentials", "aws_credentials"),
    (".ssh/id_rsa", "ssh_private_key"),
    (".ssh/id_ecdsa", "ssh_private_key"),
    (".ssh/id_ed25519", "ssh_private_key"),
    (".config/gcloud/application_default_credentials.json", "gcloud_adc"),
    (".netrc", "netrc"),
    (".docker/config.json", "docker_auth"),
)


def _real_home() -> Path:
    """The account's real home (test seam): env HOME is caller-controlled and
    must not be able to hide credentials from the containment scan."""
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path(os.path.expanduser("~"))


def _readable_credential_classes(home: Path) -> list[str]:
    found: set[str] = set()
    for rel, klass in _SENSITIVE_CREDENTIAL_PATHS:
        candidate = home / rel
        try:
            if candidate.is_file() and os.access(candidate, os.R_OK):
                found.add(klass)
        except OSError:
            continue
    return sorted(found)


def _require_containment(repo: Path) -> None:
    """§9.4 (M1): unattended automation refuses to start outside a sandboxed,
    attested environment. The marker is a machine-local human attestation;
    the credential scan is the mechanical disproof."""
    marker_path = repo / CONTAINMENT_MARKER_RELPATH
    if not marker_path.is_file():
        raise SystemExit(
            f"containment_marker_missing:{CONTAINMENT_MARKER_RELPATH}"
            " (attest with: python scripts/swarm.py attest-containment --attested-by <name>)"
        )
    try:
        marker = json.loads(_read_text(marker_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"containment_marker_invalid:{exc}") from exc
    if not isinstance(marker, dict) or marker.get("contained") is not True:
        raise SystemExit("containment_marker_invalid:contained_not_true")

    credentials = set(_readable_credential_classes(_real_home()))
    waived = marker.get("credential_scan_waiver")
    if isinstance(waived, list):
        # §9.4 allows exactly one exception class: credentials the attesting
        # human explicitly names (e.g. a scoped deploy key). The waiver lives
        # in the ATTESTED marker, never in env — env is caller-controlled.
        credentials -= {item for item in waived if isinstance(item, str)}
    if credentials:
        raise SystemExit(
            "containment_credentials_readable:" + ",".join(sorted(credentials))
        )

    ack_path = repo / VENDOR_ACK_RELPATH
    if not ack_path.is_file():
        raise SystemExit(
            f"vendor_policy_ack_missing:{VENDOR_ACK_RELPATH}"
            " (record with: python scripts/swarm.py ack-vendor-policy"
            " --vendor <vendor> --note <policy note> --acked-by <name>)"
        )
    try:
        ack = json.loads(_read_text(ack_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"vendor_policy_ack_invalid:{exc}") from exc
    if not isinstance(ack, dict) or not str(ack.get("vendor", "")).strip():
        raise SystemExit("vendor_policy_ack_invalid:missing_vendor")


def _require_unattended_ack(repo: Path | None = None) -> None:
    if os.environ.get("SWARM_UNATTENDED_I_UNDERSTAND") != "1":
        raise SystemExit("missing_unattended_ack:SWARM_UNATTENDED_I_UNDERSTAND=1")
    _require_containment(repo if repo is not None else _repo_root())


def _local_base_ahead_count(*, repo: Path, remote: str, base_branch: str) -> int:
    if not _git_ref_exists(repo, f"refs/heads/{base_branch}"):
        return 0
    cp = _run(
        ["git", "rev-list", "--count", f"{remote}/{base_branch}..{base_branch}"],
        cwd=repo,
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        return 0
    try:
        return int((cp.stdout or "0").strip())
    except ValueError:
        return 0


def _supervisor_sync_to_remote_base(*, repo: Path, remote: str, base_branch: str) -> None:
    # Guarded sync (§4.0 #9): never discard local base commits with checkout -B.
    # If the local base is ahead of the remote, refuse loudly and escalate.
    _run(["git", "fetch", remote], cwd=repo, check=True)

    ahead = _local_base_ahead_count(repo=repo, remote=remote, base_branch=base_branch)
    if ahead > 0:
        raise SystemExit(f"base_sync_refused_local_ahead:{base_branch}:{ahead}")

    if _git_current_branch(repo) == base_branch:
        _run(["git", "merge", "--ff-only", f"{remote}/{base_branch}"], cwd=repo, check=True)
    else:
        _run(["git", "branch", "-f", base_branch, f"{remote}/{base_branch}"], cwd=repo, check=True)
        _run(["git", "checkout", base_branch], cwd=repo, check=True)


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


def _runtime_event_paths(repo: Path) -> set[str]:
    paths = {swarm_events.EVENT_JOURNAL_PATH.as_posix()}
    sink = swarm_events.escalation_sink_config(repo)
    if sink.get("type") != "file":
        return paths
    try:
        target = (repo / str(sink["target"])).resolve().relative_to(repo.resolve())
    except (KeyError, ValueError):
        return paths
    paths.add(target.as_posix())
    return paths


def _git_unstage_path(cwd: Path, path: str) -> None:
    head_entry = _run(
        ["git", "ls-tree", "HEAD", "--", path],
        cwd=cwd,
        capture=True,
        check=True,
    )
    line = (head_entry.stdout or "").strip()
    if line:
        metadata = line.split("\t", 1)[0].split()
        if len(metadata) == 3:
            mode, _, object_id = metadata
            _run(
                ["git", "update-index", "--add", "--cacheinfo", mode, object_id, path],
                cwd=cwd,
                check=True,
            )
            return
    _run(
        ["git", "rm", "--cached", "-f", "--ignore-unmatch", "--", path],
        cwd=cwd,
        capture=True,
        check=True,
    )


def _collect_changed_paths_with_sources(*, repo: Path, base_ref: str | None) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    path_sources: dict[str, set[str]] = {}
    ops: list[dict[str, str]] = []
    runtime_event_paths = _runtime_event_paths(repo)

    def add_entries(source: str, entries: list[dict[str, str]]) -> None:
        for entry in entries:
            record = dict(entry)
            record["source"] = source
            ops.append(record)
            for candidate in (entry.get("path", ""), entry.get("old_path", "")):
                if not candidate or candidate in runtime_event_paths:
                    continue
                path_sources.setdefault(candidate, set()).add(source)

    if base_ref is not None:
        add_entries("committed", _git_diff_name_status_entries(repo, [f"{base_ref}...HEAD"]))
    add_entries("staged", _git_diff_name_status_entries(repo, ["--cached"]))
    add_entries("unstaged", _git_diff_name_status_entries(repo, []))
    for path in _git_untracked_files(repo):
        if path in runtime_event_paths:
            continue
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
        digits = task_id[1:] if task_id.startswith("T") else task_id
        if Path(norm).name.startswith(f"H{digits}_"):
            return True, None
        return False, "handoff_namespace_violation"
    if norm.startswith("reports/status/swarm_runs/") and Path(norm).name.startswith(f"{task_id}_"):
        return True, None
    # reports/status/reviews/ is deliberately NOT task-writable: review logs
    # are Judge-only artifacts (M1 review fix — forged-approval channel).
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
    result = data.get("result")
    return (
        isinstance(task, dict)
        and task.get("task_id") == task_id
        and isinstance(result, dict)
        and result.get("status") == "ok"
        and data.get("provenance_class") == "executor_run"
        and _parse_utc_iso(data.get("generated_at_utc")) is not None
    )


def _matching_v2_run_manifest_data(paths: list[Path], task_id: str) -> list[tuple[Path, dict[str, object]]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        try:
            data = json.loads(_read_text(path))
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("schema_version") != SWARM_RUN_MANIFEST_SCHEMA_VERSION:
            continue
        task = data.get("task")
        if isinstance(task, dict) and task.get("task_id") == task_id:
            matches.append((path, data))
    return matches


def _parse_utc_iso(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _git_commit_message(cwd: Path, ref: str) -> str:
    cp = _run(["git", "log", "-1", "--pretty=%s", ref], cwd=cwd, capture=True, check=False)
    return (cp.stdout or "").strip()


def _git_commit_paths(cwd: Path, ref: str) -> list[str]:
    cp = _run(
        ["git", "show", "--name-only", "--pretty=format:", ref],
        cwd=cwd,
        capture=True,
        check=False,
    )
    return [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]


def _judge_manifest_integrity_failures(
    *,
    repo: Path,
    task: Task,
    manifest: dict[str, object],
    contract: FrameworkContract,
) -> list[str]:
    """§4.0 #10 + #17: the branch tip must be exactly what the manifest attests,
    and the review actor must be separated from the run actor."""
    failures: list[str] = []

    repo_block = manifest.get("repo") if isinstance(manifest.get("repo"), dict) else {}
    manifest_branch = repo_block.get("branch")
    manifest_sha = repo_block.get("git_sha")

    current_branch = _git_current_branch(repo)
    if isinstance(manifest_branch, str) and manifest_branch and manifest_branch != current_branch:
        failures.append(f"manifest_branch_mismatch:{manifest_branch}:{current_branch}")

    if not isinstance(manifest_sha, str) or not manifest_sha:
        failures.append("manifest_missing_git_sha")
    else:
        is_ancestor = (
            _run(
                ["git", "merge-base", "--is-ancestor", manifest_sha, "HEAD"],
                cwd=repo,
                capture=True,
                check=False,
            ).returncode
            == 0
        )
        if not is_ancestor:
            failures.append(f"manifest_sha_not_ancestor:{manifest_sha}")
        else:
            cp = _run(
                ["git", "rev-list", f"{manifest_sha}..HEAD"],
                cwd=repo,
                capture=True,
                check=False,
            )
            commits_after = [line.strip() for line in (cp.stdout or "").splitlines() if line.strip()]
            if len(commits_after) > 1:
                failures.append(f"post_manifest_commits:{len(commits_after)}")
            elif len(commits_after) == 1:
                run_commit = commits_after[0]
                task_block = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
                expected_message = f"{task.task_id}: {task_block.get('state_after')}"
                if _git_commit_message(repo, run_commit) != expected_message:
                    failures.append(f"post_manifest_commit_message:{run_commit[:12]}")
                ownership_block = manifest.get("ownership") if isinstance(manifest.get("ownership"), dict) else {}
                declared_changed = set(
                    item for item in ownership_block.get("changed_paths", []) if isinstance(item, str)
                )
                commands_block = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
                task_file_rel = task.path.relative_to(repo).as_posix()
                control_plane = {task_file_rel}
                control_plane.update(_task_projection_paths(task_file_rel))
                for key in ("executor_log_path",):
                    value = commands_block.get(key)
                    if isinstance(value, str) and value:
                        control_plane.add(value)
                artifacts_block = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
                manifest_rel = artifacts_block.get("run_manifest_path")
                if isinstance(manifest_rel, str) and manifest_rel:
                    control_plane.add(manifest_rel)
                for committed_path in _git_commit_paths(repo, run_commit):
                    if committed_path in declared_changed or committed_path in control_plane:
                        continue
                    if committed_path.startswith(".orchestrator/handoff/"):
                        continue
                    failures.append(f"post_manifest_tamper:{committed_path}")

    actor_block = manifest.get("actor") if isinstance(manifest.get("actor"), dict) else {}
    run_session = actor_block.get("session_id")
    if isinstance(run_session, str) and run_session and run_session == _ACTOR_SESSION_ID:
        failures.append("actor_separation_same_session")

    generated_at = _parse_utc_iso(manifest.get("generated_at_utc"))
    if generated_at is None:
        # fail closed: a manifest without a parseable timestamp cannot prove
        # it satisfies the separation window
        failures.append("actor_separation_window_unverifiable")
    else:
        elapsed = (dt.datetime.now(tz=dt.timezone.utc) - generated_at).total_seconds()
        if elapsed < contract.review_min_separation_seconds:
            failures.append(
                f"actor_separation_window:{int(elapsed)}s<{contract.review_min_separation_seconds}s"
            )

    # The LIVE task frontmatter must be byte-identical to the copy pinned at
    # run time — an uncommitted or amended edit to gates/allowed_paths after
    # the run is invisible to every other check (§4.0 #10/#13).
    frontmatter_block = manifest.get("frontmatter") if isinstance(manifest.get("frontmatter"), dict) else {}
    pinned_sha = frontmatter_block.get("pinned_sha256")
    if not isinstance(pinned_sha, str) or not pinned_sha:
        failures.append("manifest_missing_pinned_frontmatter")
    else:
        try:
            current_text, _ = _task_frontmatter_snapshot(task.path)
            current_sha = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        except ValueError:
            current_sha = None
        if current_sha != pinned_sha:
            failures.append("post_run_frontmatter_tamper")

    # Executor-log binding: an executor_run manifest must be backed by the
    # durable log it hashes — fabricating provenance requires fabricating a
    # coherent hashed log too, and drift after the run is visible.
    if manifest.get("provenance_class") == "executor_run":
        commands_block = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
        log_rel = commands_block.get("executor_log_path")
        log_sha = commands_block.get("executor_log_sha256")
        if not isinstance(log_rel, str) or not log_rel or not isinstance(log_sha, str) or not log_sha:
            failures.append("executor_log_binding_missing")
        else:
            log_path = repo / log_rel
            if not log_path.is_file():
                failures.append(f"executor_log_binding_failed:missing:{log_rel}")
            else:
                actual = hashlib.sha256(log_path.read_bytes()).hexdigest()
                if actual != log_sha:
                    failures.append(f"executor_log_binding_failed:sha256_mismatch:{log_rel}")

    return failures


def _judge_ownership_failures(
    *,
    repo: Path,
    task: Task,
    base_branch: str,
    remote: str,
) -> list[str]:
    """§4.0 #10: re-run the same merge-base ownership/diff check run-task uses."""
    base_ref = _resolve_base_ref_for_diff(cwd=repo, base_branch=base_branch, remote=remote)
    if base_ref is None:
        return [f"ownership_recheck_base_unresolved:{base_branch}"]

    failures: list[str] = []
    task_file_rel = task.path.relative_to(repo).as_posix()
    path_sources, _ = _collect_changed_paths_with_sources(repo=repo, base_ref=base_ref)
    for changed_path in sorted(path_sources):
        ok, reason = _path_is_allowed(
            path=changed_path,
            allowed_paths=task.allowed_paths,
            disallowed_paths=task.disallowed_paths,
            task_file_path=task_file_rel,
            task_id=task.task_id,
        )
        if not ok:
            failures.append(f"ownership_violation:{changed_path}:{reason}")
    return failures


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
    if not (
        isinstance(reviewer, dict)
        and reviewer.get("role") == scientific_review_role
        and isinstance(task, dict)
        and task.get("task_id") == task_id
        and isinstance(decision, dict)
        and decision.get("outcome") == "approve"
        and task.get("state_after") == "done"
    ):
        return False
    session_id = reviewer.get("session_id")
    return isinstance(session_id, str) and bool(session_id.strip())


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


_SANDBOX_DENY_NETWORK_PROFILE = "(version 1)(allow default)(deny network*)"
_NETWORK_DENY_WRAPPER: tuple[str, ...] | None | bool = False  # False = unprobed


def _network_deny_wrapper() -> tuple[str, ...] | None:
    """Argv prefix that denies network to the child process tree, or None when
    the platform offers no supported mechanism. Probed once per process; the
    EFFECTIVE state is recorded per gate — enforcement is never assumed."""
    global _NETWORK_DENY_WRAPPER
    if _NETWORK_DENY_WRAPPER is not False:
        return _NETWORK_DENY_WRAPPER
    wrapper: tuple[str, ...] | None = None
    if sys.platform == "darwin" and _which_or_none("sandbox-exec"):
        candidate = ("sandbox-exec", "-p", _SANDBOX_DENY_NETWORK_PROFILE)
        if subprocess.run([*candidate, "true"], capture_output=True, text=True).returncode == 0:
            wrapper = candidate
    elif sys.platform.startswith("linux") and _which_or_none("unshare"):
        if subprocess.run(["unshare", "-n", "true"], capture_output=True, text=True).returncode == 0:
            wrapper = ("unshare", "-n")
    _NETWORK_DENY_WRAPPER = wrapper
    return wrapper


def _gate_environment() -> dict[str, str]:
    env = {key: os.environ[key] for key in GATE_ENV_ALLOWLIST if key in os.environ}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _clip_gate_output(output: str) -> tuple[str, str]:
    raw = output or ""
    head = raw[:GATE_OUTPUT_SEGMENT_BYTES]
    tail = raw[-GATE_OUTPUT_SEGMENT_BYTES:] if len(raw) > GATE_OUTPUT_SEGMENT_BYTES else ""
    return head, tail


def _run_gates(
    repo: Path,
    gates: list[str],
    *,
    interpreter_allowlist: tuple[str, ...] = DEFAULT_GATE_INTERPRETER_ALLOWLIST,
    timeout_seconds: int = DEFAULT_GATE_TIMEOUT_SECONDS,
) -> tuple[bool, list[dict[str, object]]]:
    """Constrained gate execution (§4.0 #12 + #18): no shell, interpreter
    allowlist, stripped environment, per-gate timeout, network denied where
    the OS supports it, head+tail output capture."""
    outputs: list[dict[str, object]] = []
    all_ok = True
    deny_wrapper = _network_deny_wrapper()

    for gate in gates:
        started = time.monotonic()
        record: dict[str, object] = {
            "command": gate,
            "argv": None,
            "returncode": None,
            "duration_seconds": None,
            "timed_out": False,
            "network_disabled": deny_wrapper is not None,
            "network_disable_method": deny_wrapper[0] if deny_wrapper else "none",
            "output_head": "",
            "output_tail": "",
            "constraint_violation": None,
        }
        try:
            argv = shlex.split(gate)
        except ValueError as exc:
            record["constraint_violation"] = f"gate_parse_error:{exc}"
            outputs.append(record)
            all_ok = False
            continue
        if not argv:
            record["constraint_violation"] = "gate_empty_command"
            outputs.append(record)
            all_ok = False
            continue
        record["argv"] = argv

        interpreter = argv[0]
        if interpreter != Path(interpreter).name:
            record["constraint_violation"] = f"gate_interpreter_path_qualified:{interpreter}"
            outputs.append(record)
            all_ok = False
            continue
        if interpreter not in set(interpreter_allowlist):
            record["constraint_violation"] = f"gate_interpreter_not_allowlisted:{interpreter}"
            outputs.append(record)
            all_ok = False
            continue

        full_argv = [*(deny_wrapper or ()), *argv]
        try:
            # _run uses start_new_session + killpg on timeout, so a gate's
            # whole process tree dies with it — not just the direct child.
            cp = _run(
                full_argv,
                cwd=repo,
                capture=True,
                check=False,
                env=_gate_environment(),
                timeout_seconds=timeout_seconds,
            )
            record["returncode"] = cp.returncode
            head, tail = _clip_gate_output(cp.stdout or "")
            record["output_head"] = head
            record["output_tail"] = tail
            if cp.returncode != 0:
                all_ok = False
        except subprocess.TimeoutExpired as exc:
            record["timed_out"] = True
            captured = exc.stdout
            if isinstance(captured, bytes):
                captured = captured.decode("utf-8", errors="replace")
            head, tail = _clip_gate_output(captured or "")
            record["output_head"] = head
            record["output_tail"] = tail
            all_ok = False
        except OSError as exc:
            record["constraint_violation"] = f"gate_exec_error:{exc}"
            all_ok = False
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        outputs.append(record)

    return all_ok, outputs


def _executor_prompt_path(task: Task, contract: FrameworkContract) -> Path:
    key = "operator" if task.role == "Operator" else "worker"
    return contract.prompt_templates[key]


def _latest_run_manifest_status(
    *,
    repo: Path,
    directory: Path,
    task_id: str,
) -> dict[str, object]:
    paths = _matching_task_jsons(directory, task_id)
    if not paths:
        return {
            "last_run_manifest": None,
            "last_run_status": None,
            "blocked_reasons": [],
        }

    path = paths[-1]
    status: str | None = None
    blocked_reasons: list[str] = []
    try:
        data = json.loads(_read_text(path))
        task_block = data.get("task") if isinstance(data, dict) else None
        result = data.get("result") if isinstance(data, dict) else None
        if (
            isinstance(task_block, dict)
            and task_block.get("task_id") == task_id
            and isinstance(result, dict)
        ):
            raw_status = result.get("status")
            if isinstance(raw_status, str):
                status = raw_status
            raw_reasons = result.get("blocked_reasons")
            if isinstance(raw_reasons, list):
                blocked_reasons = [item for item in raw_reasons if isinstance(item, str)]
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "last_run_manifest": path.relative_to(repo).as_posix(),
        "last_run_status": status,
        "blocked_reasons": blocked_reasons,
    }


def _last_human_note(task: Task) -> str | None:
    try:
        section = _extract_section(_read_text(task.path), "Notes / Decisions")
    except OSError:
        return None
    if section is None:
        return None
    note_lines = [line.strip() for line in section.splitlines() if line.strip()]
    if not note_lines or "@human" not in note_lines[-1]:
        return None
    return note_lines[-1].removeprefix("- ").strip()


def _run_manifest_spend(directory: Path) -> float | str:
    values: list[float] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(_read_text(path))
            except (OSError, json.JSONDecodeError):
                continue
            usage = data.get("usage") if isinstance(data, dict) else None
            value = usage.get("estimated_cost_usd") if isinstance(usage, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return sum(values) if values else "unknown"


def _render_status_text(payload: dict[str, object]) -> str:
    lines = ["Swarm status"]
    states = payload["states"]
    assert isinstance(states, dict)
    for state in DEFAULT_ALLOWED_STATES:
        task_ids = states.get(state, [])
        rendered = ", ".join(task_ids) if isinstance(task_ids, list) and task_ids else "(none)"
        lines.append(f"{state}: {rendered}")

    lines.append(f"quarantined: {payload['quarantine_count']}")
    lines.append("non-done tasks:")
    non_done = payload["non_done_tasks"]
    assert isinstance(non_done, dict)
    if not non_done:
        lines.append("  (none)")
    for task_id, summary in non_done.items():
        assert isinstance(summary, dict)
        reasons = summary.get("blocked_reasons") or []
        reason_text = ",".join(reasons) if isinstance(reasons, list) and reasons else "(none)"
        lines.append(
            f"  {task_id}: state={summary.get('state')} "
            f"last_run={summary.get('last_run_status') or 'unknown'} "
            f"blocked_reasons={reason_text}"
        )

    lines.append("open @human questions:")
    questions = payload["human_questions"]
    assert isinstance(questions, list)
    if not questions:
        lines.append("  (none)")
    for question in questions:
        lines.append(f"  {question['task_id']}: {question['note']}")

    lines.append("lease health:")
    leases = payload["leases"]
    assert isinstance(leases, list)
    if not leases:
        lines.append("  (none)")
    for lease in leases:
        lines.append(
            f"  {lease['task_id']}: lease_id={lease['lease_id']} "
            f"session={lease['session']} expired={lease['expired']} "
            f"orphaned={lease['orphaned']}"
        )

    journal = payload["journal"]
    assert isinstance(journal, dict)
    lines.append(
        "journal: "
        f"events={journal['total_events']} malformed={journal['malformed_count']} "
        f"escalations={journal['escalation_count']} "
        f"last={journal['last_event_timestamp'] or 'none'}"
    )
    lines.append(f"spend_usd: {payload['spend']}")
    return "\n".join(lines)


def cmd_attest_containment(args: argparse.Namespace) -> int:
    repo = _repo_root()
    marker_path = repo / CONTAINMENT_MARKER_RELPATH
    payload = {
        "schema_version": CONTAINMENT_MARKER_SCHEMA_VERSION,
        "contained": True,
        "attested_by": args.attested_by,
        "attested_at_utc": _utc_now_iso(),
        "note": args.note or "",
        "credential_scan_waiver": sorted(set(getattr(args, "waive_credential_class", []) or [])),
    }
    _write_json(marker_path, payload)
    _record_swarm_event(
        repo,
        {"event": "containment_attested", "attested_by": args.attested_by},
    )
    print(json.dumps({"written": CONTAINMENT_MARKER_RELPATH}, sort_keys=True))
    return 0


def cmd_ack_vendor_policy(args: argparse.Namespace) -> int:
    repo = _repo_root()
    ack_path = repo / VENDOR_ACK_RELPATH
    payload = {
        "schema_version": VENDOR_ACK_SCHEMA_VERSION,
        "vendor": args.vendor,
        "policy_note": args.note,
        "acked_by": args.acked_by,
        "acked_at_utc": _utc_now_iso(),
    }
    _write_json(ack_path, payload)
    _record_swarm_event(
        repo,
        {"event": "vendor_policy_acked", "vendor": args.vendor, "acked_by": args.acked_by},
    )
    print(json.dumps({"written": VENDOR_ACK_RELPATH}, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)

    states = {
        state: sorted(task_id for task_id, task in tasks.items() if task.state == state)
        for state in DEFAULT_ALLOWED_STATES
    }
    non_done_tasks: dict[str, dict[str, object]] = {}
    human_questions: list[dict[str, str]] = []
    for task_id, task in sorted(tasks.items()):
        if task.state != "done":
            non_done_tasks[task_id] = {
                "state": task.state,
                **_latest_run_manifest_status(
                    repo=repo,
                    directory=contract.run_manifest_dir,
                    task_id=task_id,
                ),
            }
        human_note = _last_human_note(task)
        if human_note is not None:
            human_questions.append({"task_id": task_id, "note": human_note})

    now = dt.datetime.now(tz=dt.timezone.utc)
    claims = swarm_claims.read_claims(
        repo,
        args.remote,
        fetch=not bool(args.no_fetch),
    )
    leases = []
    for task_id, claim in sorted(claims.items()):
        task = tasks.get(task_id)
        leases.append(
            {
                "task_id": task_id,
                "lease_id": claim.lease_id,
                "session": claim.session_id,
                "expired": claim.expired(now=now),
                "orphaned": task is None
                or task.state not in {"active", "ready_for_review"},
            }
        )

    events, malformed_count = swarm_events.read_events(repo)
    payload: dict[str, object] = {
        "states": states,
        "quarantined": quarantined,
        "quarantine_count": len(quarantined),
        "non_done_tasks": non_done_tasks,
        "human_questions": human_questions,
        "leases": leases,
        "journal": {
            "total_events": len(events),
            "malformed_count": malformed_count,
            "escalation_count": sum(event.get("escalation") is True for event in events),
            "last_event_timestamp": events[-1].get("ts_utc") if events else None,
        },
        "spend": _run_manifest_spend(contract.run_manifest_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_status_text(payload))
    return 0


def _new_cost_bucket() -> dict[str, object]:
    return {
        "run_count": 0,
        "wall_clock_seconds": 0.0,
        "runs_without_usage": 0,
        "_input_tokens": 0,
        "_input_seen": False,
        "_output_tokens": 0,
        "_output_seen": False,
        "_estimated_cost_usd": 0.0,
        "_cost_seen": False,
    }


def _add_cost_record(bucket: dict[str, object], usage: object) -> None:
    bucket["run_count"] = int(bucket["run_count"]) + 1
    if not isinstance(usage, dict):
        bucket["runs_without_usage"] = int(bucket["runs_without_usage"]) + 1
        return

    wall_clock = usage.get("wall_clock_seconds")
    source = usage.get("source")
    if isinstance(wall_clock, (int, float)) and not isinstance(wall_clock, bool):
        bucket["wall_clock_seconds"] = float(bucket["wall_clock_seconds"]) + float(wall_clock)
    if not isinstance(source, str) or source == "unavailable":
        bucket["runs_without_usage"] = int(bucket["runs_without_usage"]) + 1

    input_tokens = usage.get("input_tokens")
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        bucket["_input_tokens"] = int(bucket["_input_tokens"]) + input_tokens
        bucket["_input_seen"] = True
    output_tokens = usage.get("output_tokens")
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
        bucket["_output_tokens"] = int(bucket["_output_tokens"]) + output_tokens
        bucket["_output_seen"] = True
    estimated_cost = usage.get("estimated_cost_usd")
    if isinstance(estimated_cost, (int, float)) and not isinstance(estimated_cost, bool):
        bucket["_estimated_cost_usd"] = float(bucket["_estimated_cost_usd"]) + float(estimated_cost)
        bucket["_cost_seen"] = True


def _finalize_cost_bucket(bucket: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "run_count": int(bucket["run_count"]),
        "wall_clock_seconds": round(float(bucket["wall_clock_seconds"]), 6),
        "runs_without_usage": int(bucket["runs_without_usage"]),
    }
    if bucket["_input_seen"]:
        result["input_tokens"] = int(bucket["_input_tokens"])
    if bucket["_output_seen"]:
        result["output_tokens"] = int(bucket["_output_tokens"])
    if bucket["_cost_seen"]:
        result["estimated_cost_usd"] = round(float(bucket["_estimated_cost_usd"]), 4)
    return result


def _costs_payload(run_manifest_dir: Path) -> dict[str, object]:
    total = _new_cost_bucket()
    dimensions: dict[str, dict[str, dict[str, object]]] = {
        "by_task_id": {},
        "by_workstream": {},
        "by_model": {},
    }
    if run_manifest_dir.exists():
        for path in sorted(run_manifest_dir.glob("*.json")):
            try:
                manifest = json.loads(_read_text(path))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
                SWARM_RUN_MANIFEST_SCHEMA_VERSION,
                SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1,
            }:
                continue
            task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
            executor = (
                manifest.get("executor") if isinstance(manifest.get("executor"), dict) else {}
            )
            keys = {
                "by_task_id": task.get("task_id"),
                "by_workstream": task.get("workstream"),
                "by_model": executor.get("model"),
            }
            usage = manifest.get("usage")
            _add_cost_record(total, usage)
            for dimension, raw_key in keys.items():
                key = raw_key if isinstance(raw_key, str) and raw_key else "unknown"
                bucket = dimensions[dimension].setdefault(key, _new_cost_bucket())
                _add_cost_record(bucket, usage)

    return {
        dimension: {
            key: _finalize_cost_bucket(bucket)
            for key, bucket in sorted(buckets.items())
        }
        for dimension, buckets in dimensions.items()
    } | {"total": _finalize_cost_bucket(total)}


def _render_costs_text(payload: dict[str, object]) -> str:
    lines = [
        "Swarm costs",
        "dimension        group                 runs   wall_s       input      output     cost_usd  missing",
    ]
    for dimension in ("by_task_id", "by_workstream", "by_model"):
        buckets = payload.get(dimension)
        if not isinstance(buckets, dict):
            continue
        for key, bucket in buckets.items():
            if not isinstance(bucket, dict):
                continue
            lines.append(
                f"{dimension.removeprefix('by_'):<16} {str(key):<21} "
                f"{int(bucket.get('run_count', 0)):>5} "
                f"{float(bucket.get('wall_clock_seconds', 0.0)):>9.3f} "
                f"{str(bucket.get('input_tokens', '-')):>11} "
                f"{str(bucket.get('output_tokens', '-')):>11} "
                f"{str(bucket.get('estimated_cost_usd', '-')):>12} "
                f"{int(bucket.get('runs_without_usage', 0)):>8}"
            )
    total = payload.get("total") if isinstance(payload.get("total"), dict) else {}
    lines.append(
        f"{'total':<16} {'all':<21} "
        f"{int(total.get('run_count', 0)):>5} "
        f"{float(total.get('wall_clock_seconds', 0.0)):>9.3f} "
        f"{str(total.get('input_tokens', '-')):>11} "
        f"{str(total.get('output_tokens', '-')):>11} "
        f"{str(total.get('estimated_cost_usd', '-')):>12} "
        f"{int(total.get('runs_without_usage', 0)):>8}"
    )
    return "\n".join(lines)


def cmd_costs(args: argparse.Namespace) -> int:
    repo = _repo_root()
    try:
        run_manifest_dir = load_framework_contract(repo).run_manifest_dir
    except (OSError, SystemExit):
        run_manifest_dir = repo / "reports" / "status" / "swarm_runs"
    payload = _costs_payload(run_manifest_dir)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_costs_text(payload))
    return 0


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


@contextlib.contextmanager
def _runtime_repo_context(repo: Path, *, event_repo: Path):
    global _REPO_ROOT_CACHE
    previous_cache = _REPO_ROOT_CACHE
    previous_event_root = os.environ.get("SWARM_EVENT_REPO_ROOT")
    _REPO_ROOT_CACHE = repo
    os.environ["SWARM_EVENT_REPO_ROOT"] = str(event_repo)
    try:
        yield
    finally:
        _REPO_ROOT_CACHE = previous_cache
        if previous_event_root is None:
            os.environ.pop("SWARM_EVENT_REPO_ROOT", None)
        else:
            os.environ["SWARM_EVENT_REPO_ROOT"] = previous_event_root


def _worktree_records(repo: Path) -> list[dict[str, str]]:
    cp = _run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture=True,
        check=True,
    )
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in [*(cp.stdout or "").splitlines(), ""]:
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value.strip()
    return records


def _task_branch_contexts(repo: Path) -> dict[str, dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    for record in _worktree_records(repo):
        branch_ref = record.get("branch", "")
        branch = branch_ref.removeprefix("refs/heads/")
        task_id = _parse_task_id_from_branch(branch)
        worktree_raw = record.get("worktree")
        if task_id is None or not worktree_raw:
            continue
        worktree = Path(worktree_raw).resolve()
        try:
            contract = load_framework_contract(worktree)
            tasks, quarantined = load_tasks_quarantined(contract)
            task = _resolve_runtime_task(tasks, quarantined, task_id)
        except (OSError, SystemExit, ValueError):
            continue
        manifest_paths = _matching_task_jsons(contract.run_manifest_dir, task_id)
        matching = _matching_v2_run_manifest_data(manifest_paths, task_id)
        manifest_path: Path | None = None
        manifest: dict[str, object] = {}
        if matching:
            manifest_path, manifest = matching[-1]
        review_paths = _matching_task_jsons(contract.judge_review_dir, task_id)
        contexts[task_id] = {
            "task_id": task_id,
            "branch": branch,
            "worktree": worktree,
            "contract": contract,
            "task": task,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "review_paths": review_paths,
        }
    return contexts


def _ready_for_review_contexts(repo: Path) -> dict[str, dict[str, object]]:
    return {
        task_id: context
        for task_id, context in _task_branch_contexts(repo).items()
        if isinstance(context.get("task"), Task)
        and context["task"].state == "ready_for_review"
        and isinstance(context.get("manifest_path"), Path)
    }


def _projection_paths_for_filename(filename: str) -> list[str]:
    return [f".orchestrator/{state}/{filename}" for state in DEFAULT_ALLOWED_STATES]


def _persist_projection_changes(
    *,
    repo: Path,
    remote: str,
    base_branch: str,
    filenames: Iterable[str],
    message: str,
    strict: bool,
    push: bool = True,
) -> bool:
    owned_paths = sorted(
        {
            path
            for filename in filenames
            for path in _projection_paths_for_filename(filename)
        }
    )
    owned_paths = [
        path
        for path in owned_paths
        if (repo / path).exists()
        or _run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=repo,
            capture=True,
            check=False,
        ).returncode
        == 0
    ]
    if not owned_paths:
        return False
    staged_before_cp = _run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture=True,
        check=True,
    )
    staged_before = {
        line.strip() for line in (staged_before_cp.stdout or "").splitlines() if line.strip()
    }
    unexpected = sorted(staged_before - set(owned_paths))
    if unexpected:
        raise SystemExit(
            "supervisor_refused_preexisting_staged_changes:" + ",".join(unexpected)
        )
    _run(["git", "add", "-A", "--", *owned_paths], cwd=repo, check=True)
    staged_after = _run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture=True,
        check=True,
    )
    if not (staged_after.stdout or "").strip():
        return False
    _git_commit(cwd=repo, message=message, strict=strict)
    if push:
        _git_push(
            cwd=repo,
            remote=remote,
            ref=base_branch,
            set_upstream=False,
            strict=strict,
        )
    return True


def _apply_projection_sweep(repo: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    moves, problems = sweep_tasks.plan_sweep(repo)
    if moves:
        sweep_tasks._apply_moves(repo, moves)
    return moves, problems


def _claim_for_dispatch(
    *,
    repo: Path,
    remote: str,
    task: Task,
    ttl_seconds: int = swarm_claims.DEFAULT_LEASE_TTL_SECONDS,
) -> swarm_claims.ClaimResult:
    branch = f"{task.task_id}_{_slug_from_task_path(task.path, task.task_id)}"
    return swarm_claims.claim_task(
        repo,
        remote,
        task.task_id,
        session_id=_ACTOR_SESSION_ID,
        branch=branch,
        ttl_seconds=ttl_seconds,
        journal=lambda event: _record_swarm_event(repo, event),
    )


def _supervisor_run_namespace(args: argparse.Namespace, task_id: str, repair_context: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        task_id=task_id,
        remote=args.remote,
        base_branch=args.base_branch,
        executor_backend=getattr(args, "executor_backend", "codex"),
        codex_model=getattr(args, "codex_model", None),
        codex_sandbox=getattr(args, "codex_sandbox", "workspace-write"),
        unattended=bool(getattr(args, "unattended", False)),
        skip_executor=False,
        record_session=False,
        force_deps=False,
        max_worker_seconds=int(getattr(args, "max_worker_seconds", 0)),
        repair_context=repair_context,
        create_pr=False,
        final_state="ready_for_review",
        supervisor_managed=True,
    )


def _run_task_in_process(
    *,
    event_repo: Path,
    worktree: Path,
    args: argparse.Namespace,
    task_id: str,
    repair_context: str | None = None,
) -> tuple[int, str]:
    output = io.StringIO()
    with (
        _runtime_repo_context(worktree, event_repo=event_repo),
        contextlib.redirect_stdout(output),
    ):
        result = cmd_run_task(
            _supervisor_run_namespace(args, task_id, repair_context=repair_context)
        )
    return result, output.getvalue()


def _usage_records(repo: Path) -> tuple[float | None, int]:
    manifests_by_run: dict[str, dict[str, object]] = {}
    roots = [repo]
    roots.extend(
        Path(record["worktree"])
        for record in _worktree_records(repo)
        if record.get("worktree") and Path(record["worktree"]).resolve() != repo.resolve()
    )
    for root in roots:
        try:
            contract = load_framework_contract(root)
        except (OSError, SystemExit):
            continue
        if not contract.run_manifest_dir.exists():
            continue
        for path in sorted(contract.run_manifest_dir.glob("*.json")):
            try:
                payload = json.loads(_read_text(path))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("run_id"), str):
                manifests_by_run[payload["run_id"]] = payload
    values: list[float] = []
    for payload in manifests_by_run.values():
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        value = usage.get("estimated_cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return (sum(values), len(values)) if values else (None, 0)


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
        _record_swarm_event(
            repo,
            {
                "event": "tick_completed",
                "selected": len(summary["selected"]),
                "skipped": len(summary["skipped"]),
                "quarantined": len(quarantined),
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    worktree_parent = Path(args.worktree_parent).expanduser().resolve() if args.worktree_parent else repo.parent
    worktree_parent.mkdir(parents=True, exist_ok=True)

    started: list[dict[str, str]] = []
    if args.runner == "tmux":
        _tmux_ensure_session(args.tmux_session, repo)
        _tmux("set-environment", "-g", "SWARM_ACTOR_SESSION", _ACTOR_SESSION_ID)
        if args.unattended:
            _tmux("set-environment", "-g", "SWARM_UNATTENDED_I_UNDERSTAND", "1")

    claimed: list[tuple[Task, swarm_claims.ClaimResult]] = []
    for task in selected:
        result = _claim_for_dispatch(
            repo=repo,
            remote=args.remote,
            task=task,
            ttl_seconds=contract.claim_lease_ttl_seconds,
        )
        if not result.ok:
            _record_swarm_event(
                repo,
                {
                    "event": "claim_lost",
                    "task_id": task.task_id,
                    "reason": result.reason,
                },
            )
            summary["skipped"].append(
                {"task_id": task.task_id, "reason": "claim_lost"}
            )
            continue
        claimed.append((task, result))

    for task, _ in claimed:
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="active",
            note_line=f"Claimed by swarm session {_ACTOR_SESSION_ID}.",
        )
        claimed_task = load_task(task.path, contract)
        _move_task_to_state_projection(repo, claimed_task)
    if claimed:
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=[task.path.name for task, _ in claimed],
            message="swarm: project claimed tasks active",
            strict=bool(args.unattended),
        )
        tasks, _ = load_tasks_quarantined(contract)

    for selected_task, claim in claimed:
        task = tasks[selected_task.task_id]
        try:
            worktree_path, branch = ensure_worktree(
                repo=repo,
                task=task,
                worktree_parent=worktree_parent,
                base_ref=args.base_branch,
            )
        except WorktreeCollisionError as exc:
            if claim.sha is not None:
                swarm_claims.release_claim(
                    repo,
                    args.remote,
                    task.task_id,
                    expected_sha=claim.sha,
                    reason="worktree_collision",
                    journal=lambda event: _record_swarm_event(repo, event),
                )
            _update_task_status_and_notes(
                task_path=task.path,
                new_state="backlog",
                note_line="Dispatch cancelled after worktree collision; claim released.",
            )
            _persist_projection_changes(
                repo=repo,
                remote=args.remote,
                base_branch=args.base_branch,
                filenames=[task.path.name],
                message=f"{task.task_id}: reopen after dispatch collision",
                strict=bool(args.unattended),
            )
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
            "--executor-backend",
            getattr(args, "executor_backend", "codex"),
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
            env = dict(os.environ)
            env["SWARM_ACTOR_SESSION"] = _ACTOR_SESSION_ID
            env["SWARM_EVENT_REPO_ROOT"] = str(repo)
            _run(command, cwd=worktree_path, check=False, env=env)

    summary["started"] = started
    _record_swarm_event(
        repo,
        {
            "event": "tick_completed",
            "selected": len(summary["selected"]),
            "skipped": len(summary["skipped"]),
            "quarantined": len(quarantined),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _merge_journal_records(repo: Path, task_id: str) -> tuple[dict | None, dict | None]:
    """Latest (merge_started, merge_verified) journal records for a task."""
    events, _ = swarm_events.read_events(repo)
    started: dict | None = None
    verified: dict | None = None
    for event in events:
        if event.get("task_id") != task_id:
            continue
        if event.get("event") == "merge_started":
            started = event
            verified = None
        elif event.get("event") == "merge_verified":
            verified = event
    return started, verified


def _merge_inflight_task_ids(repo: Path) -> set[str]:
    events, _ = swarm_events.read_events(repo)
    state: dict[str, bool] = {}
    terminal_events = {
        "task_done",
        "merge_reverted",
        "merge_refused_operator_surface",
        "merge_refused_stale_lease",
        "merge_refused_non_ff",
    }
    for event in events:
        task_id = event.get("task_id")
        if not isinstance(task_id, str):
            continue
        if event.get("event") == "merge_started":
            state[task_id] = True
        elif event.get("event") in terminal_events:
            state[task_id] = False
    return {task_id for task_id, active in state.items() if active}


def _step_sync(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    _run(["git", "fetch", args.remote], cwd=repo, check=True)
    ahead = _local_base_ahead_count(
        repo=repo,
        remote=args.remote,
        base_branch=args.base_branch,
    )
    inflight = sorted(_merge_inflight_task_ids(repo))
    if ahead > 0 and inflight:
        return {
            "synced": False,
            "recovery": "merge_inflight_local_ahead",
            "ahead": ahead,
            "tasks": inflight,
            "base_sha": _git_head_sha(repo),
        }
    _supervisor_sync_to_remote_base(
        repo=repo,
        remote=args.remote,
        base_branch=args.base_branch,
    )
    return {"synced": True, "ahead": 0, "base_sha": _git_head_sha(repo)}


def _move_task_to_state_projection(repo: Path, task: Task) -> Path:
    destination = repo / ".orchestrator" / task.state / task.path.name
    if task.path.resolve() == destination.resolve():
        return task.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    cp = _run(
        ["git", "mv", str(task.path), str(destination)],
        cwd=repo,
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        task.path.rename(destination)
    return destination


def _block_base_task(
    *,
    repo: Path,
    contract: FrameworkContract,
    args: argparse.Namespace,
    task_id: str,
    note: str,
    message: str,
) -> None:
    tasks, quarantined = load_tasks_quarantined(contract)
    task = _resolve_runtime_task(tasks, quarantined, task_id)
    filename = task.path.name
    _update_task_status_and_notes(
        task_path=task.path,
        new_state="blocked",
        note_line=note,
    )
    blocked_task = load_task(task.path, contract)
    _move_task_to_state_projection(repo, blocked_task)
    _persist_projection_changes(
        repo=repo,
        remote=args.remote,
        base_branch=args.base_branch,
        filenames=[filename],
        message=message,
        strict=bool(args.unattended),
    )


def _step_reap(args: argparse.Namespace) -> dict[str, object]:
    """Reap expired leases STATE-AWARE (§4.1): only genuinely orphaned ACTIVE
    work is reopened; a blocked task keeps its @human hold and an approved
    task keeps its approval — their expired claims are released with the
    state preserved. Reopen is written BEFORE the release so a crash between
    the two converges on the next cycle (backlog + expired claim → release
    only)."""
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, _ = load_tasks_quarantined(contract)
    claims = swarm_claims.read_claims(repo, args.remote)
    expired = {
        action.task_id: action
        for action in swarm_claims.reap_expired(repo, args.remote, fetch=False)
    }
    reopened: list[str] = []
    stale_done: list[str] = []
    released: list[str] = []
    preserved: list[dict[str, str]] = []
    candidates: list[str] = []
    filenames: list[str] = []

    def _release(task_id: str, sha: str, lease_id: object, reason: str) -> bool:
        result = swarm_claims.release_claim(
            repo,
            args.remote,
            task_id,
            expected_sha=sha,
            reason=reason,
            journal=lambda event: _record_swarm_event(repo, event),
        )
        if not result.ok:
            _record_swarm_event(
                repo,
                {
                    "event": "reap_release_failed",
                    "task_id": task_id,
                    "lease_id": lease_id,
                    "reason": result.reason,
                },
                escalation=True,
            )
            return False
        released.append(task_id)
        return True

    for task_id, claim in sorted(claims.items()):
        task = tasks.get(task_id)
        action = expired.get(task_id)

        if task is not None and task.state == "done":
            _record_swarm_event(
                repo,
                {"event": "orphan_stale_claim", "task_id": task_id, "lease_id": claim.lease_id},
            )
            _release(task_id, claim.sha, claim.lease_id, "done_task_stale_claim")
            stale_done.append(task_id)
            continue

        if action is None:
            continue

        if task is None or task.state == "backlog":
            # nothing to reopen (missing task, or a prior crash already
            # reopened it) — the expired ref is the only cleanup left
            _record_swarm_event(
                repo,
                {
                    "event": "orphan_claim_released",
                    "task_id": task_id,
                    "lease_id": action.lease_id,
                    "task_state": task.state if task is not None else None,
                },
            )
            _release(task_id, action.sha, action.lease_id, action.reason)
            continue

        if task.state != "active":
            # blocked keeps its @human hold; ready_for_review keeps its
            # approval; integration_ready keeps its interface state — the
            # lease is released, the scientific state is NEVER conflated
            # with orphanhood (§4.1).
            _record_swarm_event(
                repo,
                {
                    "event": "orphan_claim_released",
                    "task_id": task_id,
                    "lease_id": action.lease_id,
                    "task_state": task.state,
                },
            )
            _release(task_id, action.sha, action.lease_id, action.reason)
            preserved.append({"task_id": task_id, "state": task.state})
            continue

        # active + expired lease = orphaned work: reopen FIRST, release second
        _record_swarm_event(
            repo,
            {
                "event": "task_orphaned",
                "task_id": task_id,
                "lease_id": action.lease_id,
                "cause": action.reason,
            },
        )
        filenames.append(task.path.name)
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="backlog",
            note_line=(
                f"orphaned: lease expired (lease {action.lease_id}); "
                "reopened by supervisor"
            ),
        )
        reopened_task = load_task(task.path, contract)
        _move_task_to_state_projection(repo, reopened_task)
        reopened.append(task_id)
        _release(task_id, action.sha, action.lease_id, action.reason)

    # plan precedence: file active + no ref → orphaned claim, reap it —
    # but only when no worktree evidence suggests a live manual run
    worktree_parent = (
        Path(args.worktree_parent).expanduser().resolve()
        if getattr(args, "worktree_parent", None)
        else repo.parent
    )
    for task_id, task in sorted(tasks.items()):
        if task.state != "active" or task_id in claims:
            continue
        worktree_path = worktree_parent / f"wt-{task_id}"
        if worktree_path.exists():
            _record_swarm_event(
                repo,
                {"event": "orphaned_candidate", "task_id": task_id, "worktree": str(worktree_path)},
                escalation=True,
            )
            candidates.append(task_id)
            continue
        _record_swarm_event(
            repo,
            {"event": "task_orphaned", "task_id": task_id, "cause": "active_without_claim"},
        )
        filenames.append(task.path.name)
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="backlog",
            note_line="orphaned: active with no live claim; reopened by supervisor",
        )
        reopened_task = load_task(task.path, contract)
        _move_task_to_state_projection(repo, reopened_task)
        reopened.append(task_id)

    if filenames:
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=filenames,
            message="swarm: reopen orphaned tasks",
            strict=bool(args.unattended),
        )
    return {
        "expired": sorted(expired),
        "reopened": reopened,
        "stale_done": stale_done,
        "released": released,
        "preserved": preserved,
        "candidates": candidates,
    }

def _step_tick(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)
    ready_review_ids = set(_ready_for_review_contexts(repo))
    ready_review_ids.update(
        task_id for task_id, task in tasks.items() if task.state == "ready_for_review"
    )
    ready_review_count = len(ready_review_ids)
    if ready_review_count >= contract.wip_max_ready_for_review:
        _record_swarm_event(
            repo,
            {
                "event": "review_backpressure",
                "ready_for_review": ready_review_count,
                "cap": contract.wip_max_ready_for_review,
            },
        )
        return {
            "selected": [],
            "started": [],
            "skipped": [],
            "backpressure": True,
            "ready_for_review": ready_review_count,
        }

    spend, _ = _usage_records(repo)
    if (
        spend is not None
        and contract.budget_max_program_usd is not None
        and spend > contract.budget_max_program_usd
    ):
        return {
            "selected": [],
            "started": [],
            "skipped": [],
            "budget_blocked": True,
            "spend_usd": spend,
            "max_program_usd": contract.budget_max_program_usd,
        }

    claimed_ids = claimed_task_ids(repo, args.remote, args.base_branch)
    ready = ready_backlog_tasks(tasks, claimed_ids, contract)
    max_active = (
        contract.wip_max_active
        if contract.wip_max_active is not None
        else max(0, int(args.max_workers))
    )
    active_count = sum(task.state == "active" for task in tasks.values())
    capacity = min(
        max(0, int(args.max_workers)),
        max(0, max_active - active_count),
    )
    selected = choose_tasks_heuristic(ready, capacity)
    summary: dict[str, object] = {
        "ready": [task.task_id for task in ready],
        "selected": [task.task_id for task in selected],
        "started": [],
        "skipped": [],
        "quarantined": quarantined,
        "active": active_count,
        "max_active": max_active,
    }
    if not selected:
        return summary

    claimed: list[tuple[Task, swarm_claims.ClaimResult]] = []
    for task in selected:
        result = _claim_for_dispatch(
            repo=repo,
            remote=args.remote,
            task=task,
            ttl_seconds=contract.claim_lease_ttl_seconds,
        )
        if not result.ok:
            _record_swarm_event(
                repo,
                {
                    "event": "claim_lost",
                    "task_id": task.task_id,
                    "reason": result.reason,
                },
            )
            summary["skipped"].append(
                {"task_id": task.task_id, "reason": "claim_lost"}
            )
            continue
        claimed.append((task, result))

    for task, _ in claimed:
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="active",
            note_line=f"Claimed by supervisor session {_ACTOR_SESSION_ID}.",
        )
        claimed_task = load_task(task.path, contract)
        _move_task_to_state_projection(repo, claimed_task)
    if claimed:
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=[task.path.name for task, _ in claimed],
            message="swarm: project claimed tasks active",
            strict=bool(args.unattended),
        )
        tasks, _ = load_tasks_quarantined(contract)

    worktree_parent = (
        Path(args.worktree_parent).expanduser().resolve()
        if getattr(args, "worktree_parent", None)
        else repo.parent
    )
    worktree_parent.mkdir(parents=True, exist_ok=True)
    for selected_task, claim in claimed:
        task = tasks[selected_task.task_id]
        try:
            worktree, branch = ensure_worktree(
                repo=repo,
                task=task,
                worktree_parent=worktree_parent,
                base_ref=args.base_branch,
            )
        except WorktreeCollisionError as exc:
            if claim.sha is not None:
                swarm_claims.release_claim(
                    repo,
                    args.remote,
                    task.task_id,
                    expected_sha=claim.sha,
                    reason="worktree_collision",
                    journal=lambda event: _record_swarm_event(repo, event),
                )
            _update_task_status_and_notes(
                task_path=task.path,
                new_state="backlog",
                note_line="Dispatch cancelled after worktree collision; claim released.",
            )
            _persist_projection_changes(
                repo=repo,
                remote=args.remote,
                base_branch=args.base_branch,
                filenames=[task.path.name],
                message=f"{task.task_id}: reopen after dispatch collision",
                strict=bool(args.unattended),
            )
            summary["skipped"].append(
                {
                    "task_id": task.task_id,
                    "reason": "worktree_collision",
                    "worktree": str(exc.worktree_path),
                }
            )
            continue

        result, output = _run_task_in_process(
            event_repo=repo,
            worktree=worktree,
            args=args,
            task_id=task.task_id,
        )
        summary["started"].append(
            {
                "task_id": task.task_id,
                "branch": branch,
                "worktree": str(worktree),
                "returncode": result,
                "output": output.strip(),
            }
        )
    _record_swarm_event(
        repo,
        {
            "event": "tick_completed",
            "selected": len(selected),
            "started": len(summary["started"]),
            "skipped": len(summary["skipped"]),
            "quarantined": len(quarantined),
        },
    )
    return summary


def _step_judge(
    args: argparse.Namespace,
    *,
    candidate_ids: set[str] | None = None,
) -> dict[str, object]:
    repo = _repo_root()
    contexts = _ready_for_review_contexts(repo)
    if candidate_ids is not None:
        contexts = {
            task_id: context
            for task_id, context in contexts.items()
            if task_id in candidate_ids
        }
    judged: list[dict[str, object]] = []
    deferred: list[dict[str, object]] = []
    now = dt.datetime.now(tz=dt.timezone.utc)

    # Shepherded claims stay alive through review + merge: the supervisor
    # renews the leases it holds for ready_for_review work so the merge
    # queue's live-claim fencing path remains the common case.
    claims = swarm_claims.read_claims(_repo_root(), args.remote, fetch=False)
    for task_id in sorted(contexts):
        claim = claims.get(task_id)
        if claim is not None and claim.session_id == _ACTOR_SESSION_ID:
            try:
                _renew_runtime_claim(repo=_repo_root(), remote=args.remote, task_id=task_id)
            except SystemExit as exc:
                _record_swarm_event(
                    _repo_root(),
                    {"event": "shepherd_renewal_failed", "task_id": task_id, "reason": str(exc)},
                    escalation=True,
                )

    for task_id, context in sorted(contexts.items()):
        contract = context["contract"]
        manifest = context["manifest"]
        assert isinstance(contract, FrameworkContract)
        assert isinstance(manifest, dict)
        generated_at = _parse_utc_iso(manifest.get("generated_at_utc"))
        if generated_at is not None:
            age = (now - generated_at).total_seconds()
            if age < contract.review_min_separation_seconds:
                remaining = max(
                    1,
                    int(contract.review_min_separation_seconds - age + 0.999),
                )
                event = {
                    "event": "review_deferred",
                    "task_id": task_id,
                    "remaining_seconds": remaining,
                }
                _record_swarm_event(repo, event)
                deferred.append(
                    {"task_id": task_id, "remaining_seconds": remaining}
                )
                continue

        worktree = context["worktree"]
        assert isinstance(worktree, Path)
        judge_session = f"judge-{uuid.uuid4().hex}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "judge-task",
            "--task-id",
            task_id,
            "--remote",
            args.remote,
            "--base-branch",
            args.base_branch,
            "--approve-only",
        ]
        if args.unattended:
            command.append("--unattended")
        env = dict(os.environ)
        env["SWARM_ACTOR_SESSION"] = judge_session
        env["SWARM_REPO_ROOT"] = str(worktree)
        env["SWARM_EVENT_REPO_ROOT"] = str(repo)
        cp = _run(
            command,
            cwd=worktree,
            capture=True,
            check=False,
            env=env,
            timeout_seconds=max(30, contract.gate_timeout_seconds * 2),
        )
        judged.append(
            {
                "task_id": task_id,
                "returncode": cp.returncode,
                "judge_session": judge_session,
                "output_tail": (cp.stdout or "")[-2000:],
            }
        )
        if cp.returncode != 0:
            _record_swarm_event(
                repo,
                {
                    "event": "judge_failed",
                    "task_id": task_id,
                    "returncode": cp.returncode,
                },
                escalation=True,
            )
    return {"judged": judged, "deferred": deferred}


def _latest_approving_review(
    context: dict[str, object],
) -> tuple[Path, dict[str, object]] | None:
    task_id = context.get("task_id")
    contract = context.get("contract")
    if not isinstance(task_id, str) or not isinstance(contract, FrameworkContract):
        return None
    reviews = context.get("review_paths")
    if not isinstance(reviews, list):
        return None
    for path in reversed(reviews):
        if not isinstance(path, Path):
            continue
        if not _is_valid_review_log(path, task_id, contract.scientific_review_role):
            continue
        try:
            payload = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return path, payload
    return None


def _release_current_claim(
    *,
    repo: Path,
    args: argparse.Namespace,
    task_id: str,
    reason: str,
) -> bool:
    claim = swarm_claims.read_claims(repo, args.remote).get(task_id)
    if claim is None:
        return True
    result = swarm_claims.release_claim(
        repo,
        args.remote,
        task_id,
        expected_sha=claim.sha,
        reason=reason,
        journal=lambda event: _record_swarm_event(repo, event),
    )
    if not result.ok:
        _record_swarm_event(
            repo,
            {
                "event": "claim_release_failed",
                "task_id": task_id,
                "reason": result.reason,
            },
            escalation=True,
        )
    return result.ok


def _step_merge(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    base_tasks, _ = load_tasks_quarantined(contract)
    inflight = _merge_inflight_task_ids(repo)
    contexts = _task_branch_contexts(repo)
    merged: list[str] = []
    refused: list[dict[str, str]] = []
    reverted: list[str] = []

    for task_id, context in sorted(contexts.items()):
        approval = _latest_approving_review(context)
        if approval is None:
            continue
        base_task = base_tasks.get(task_id)
        if base_task is None or base_task.state == "blocked":
            continue
        if base_task.state == "done":
            if task_id in inflight:
                started_event, verified_event = _merge_journal_records(repo, task_id)
                if verified_event is None:
                    # a crash raced the durable verification record: never
                    # push an unverified base — revert to the RECORDED
                    # pre-merge sha and block (Codex F4b).
                    recorded_pre = (started_event or {}).get("pre_merge_sha")
                    if isinstance(recorded_pre, str) and recorded_pre:
                        _run(["git", "reset", "--hard", recorded_pre], cwd=repo, check=True)
                    _record_swarm_event(
                        repo,
                        {
                            "event": "merge_reverted",
                            "task_id": task_id,
                            "cause": "done_without_merge_verified",
                            "pre_merge_sha": recorded_pre,
                        },
                        escalation=True,
                    )
                    _block_base_task(
                        repo=repo,
                        contract=contract,
                        args=args,
                        task_id=task_id,
                        note="@human Merge recovery: done recorded without a durable verification record; base reverted.",
                        message=f"{task_id}: block unverified merge recovery",
                    )
                    reverted.append(task_id)
                    base_tasks, _ = load_tasks_quarantined(contract)
                    continue
                _git_push(
                    cwd=repo,
                    remote=args.remote,
                    ref=args.base_branch,
                    set_upstream=False,
                    strict=True,
                )
                _record_swarm_event(
                    repo,
                    {"event": "task_done", "task_id": task_id, "recovered": True},
                )
                _release_current_claim(
                    repo=repo,
                    args=args,
                    task_id=task_id,
                    reason="task_done_recovery",
                )
                merged.append(task_id)
            continue

        # Claude adversary #1: a crash BETWEEN ff-merge and verification
        # leaves the branch already merged while the task is still
        # ready_for_review. Detect it (branch tip is an ancestor of base
        # HEAD but differs from it or the journal shows an unverified
        # merge_started) and verify-or-revert from the RECORDED pre-merge
        # sha — never recompute it from the already-moved HEAD.
        branch_name = context.get("branch")
        if isinstance(branch_name, str):
            branch_tip_cp = _run(
                ["git", "rev-parse", "--verify", "--quiet", branch_name],
                cwd=repo,
                capture=True,
                check=False,
            )
            branch_tip = branch_tip_cp.stdout.strip() if branch_tip_cp.returncode == 0 else None
            head_sha = _git_head_sha(repo)
            if (
                branch_tip
                and head_sha
                and task_id in inflight
                and _run(
                    ["git", "merge-base", "--is-ancestor", branch_tip, head_sha],
                    cwd=repo,
                    capture=True,
                    check=False,
                ).returncode
                == 0
            ):
                started_event, verified_event = _merge_journal_records(repo, task_id)
                recorded_pre = (started_event or {}).get("pre_merge_sha")
                if verified_event is None and isinstance(recorded_pre, str) and recorded_pre:
                    quality_cp = _run(
                        [sys.executable, "scripts/quality_gates.py"],
                        cwd=repo,
                        capture=True,
                        check=False,
                        timeout_seconds=max(30, contract.gate_timeout_seconds * 2),
                    )
                    if quality_cp.returncode != 0:
                        _run(["git", "reset", "--hard", recorded_pre], cwd=repo, check=True)
                        _record_swarm_event(
                            repo,
                            {
                                "event": "merge_reverted",
                                "task_id": task_id,
                                "cause": "crash_recovery_verification_failed",
                                "pre_merge_sha": recorded_pre,
                            },
                            escalation=True,
                        )
                        _block_base_task(
                            repo=repo,
                            contract=contract,
                            args=args,
                            task_id=task_id,
                            note="@human Merge recovery: post-merge verification failed after a crash; base reverted to the recorded pre-merge sha.",
                            message=f"{task_id}: block reverted crash merge",
                        )
                        reverted.append(task_id)
                        base_tasks, _ = load_tasks_quarantined(contract)
                        continue
                    swarm_events.append_event(
                        repo,
                        {
                            "event": "merge_verified",
                            "task_id": task_id,
                            "pre_merge_sha": recorded_pre,
                            "post_merge_sha": head_sha,
                            "recovered": True,
                        },
                        actor_session=_ACTOR_SESSION_ID,
                    )
                    # fall through: the normal path's ff-merge is now a no-op
                    # and promotion proceeds under the verified record

        manifest_path = context.get("manifest_path")
        manifest = context.get("manifest")
        branch = context.get("branch")
        branch_task = context.get("task")
        review_path, review = approval
        review_task = review.get("task") if isinstance(review.get("task"), dict) else {}
        manifest_rel = manifest_path.relative_to(context["worktree"]).as_posix()
        if review_task.get("run_manifest_path") != manifest_rel:
            continue
        if (
            not isinstance(manifest_path, Path)
            or not isinstance(manifest, dict)
            or not isinstance(branch, str)
            or not isinstance(branch_task, Task)
            or not _is_valid_run_manifest(manifest_path, task_id)
        ):
            continue
        review_rel = review_path.relative_to(context["worktree"]).as_posix()
        review_in_tip = _run(
            ["git", "cat-file", "-e", f"{branch}:{review_rel}"],
            cwd=repo,
            capture=True,
            check=False,
        ).returncode == 0
        if not review_in_tip:
            continue

        # Lease fencing binds to the claim COMMIT CHAIN, not the recyclable
        # lease number: the manifest's claim sha must be ancestor-or-equal of
        # the live claim tip (renewals advance the same chain; a reap+reclaim
        # is a NEW ROOT and can never pass). A manifest without a claim block
        # is claimless manual work — M0 semantics apply, the review/ownership/
        # integrity checks still gate it. A claim-stamped manifest whose claim
        # vanished needs a journaled orderly release after the run.
        claims = swarm_claims.read_claims(repo, args.remote)
        claim = claims.get(task_id)
        claim_block = manifest.get("claim") if isinstance(manifest.get("claim"), dict) else {}
        manifest_claim_sha = claim_block.get("sha")
        fencing_failure: str | None = None
        if isinstance(manifest_claim_sha, str) and manifest_claim_sha:
            if claim is not None:
                same_chain = (
                    _run(
                        ["git", "merge-base", "--is-ancestor", manifest_claim_sha, claim.sha],
                        cwd=repo,
                        capture=True,
                        check=False,
                    ).returncode
                    == 0
                )
                if not same_chain:
                    fencing_failure = "stale_lease_chain"
            else:
                events, _ = swarm_events.read_events(repo)
                manifest_time = _parse_utc_iso(manifest.get("generated_at_utc"))
                orderly_release = any(
                    event.get("event") in {"claim_released", "orphan_claim_released"}
                    and event.get("task_id") == task_id
                    and (
                        manifest_time is None
                        or (
                            _parse_utc_iso(event.get("ts_utc")) is not None
                            and _parse_utc_iso(event.get("ts_utc")) >= manifest_time
                        )
                    )
                    for event in events
                )
                if not orderly_release:
                    fencing_failure = "missing_claim_without_release_record"
        if fencing_failure is not None:
            _record_swarm_event(
                repo,
                {
                    "event": "merge_refused_stale_lease",
                    "task_id": task_id,
                    "reason": fencing_failure,
                    "manifest_claim_sha": manifest_claim_sha,
                    "current_claim_sha": claim.sha if claim is not None else None,
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note=f"@human Merge refused: claim fencing failed ({fencing_failure}).",
                message=f"{task_id}: block stale lease merge",
            )
            refused.append({"task_id": task_id, "reason": fencing_failure})
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        diff_cp = _run(
            ["git", "diff", "--name-only", f"{args.base_branch}..{branch}"],
            cwd=repo,
            capture=True,
            check=True,
        )
        changed_paths = [
            line.strip() for line in (diff_cp.stdout or "").splitlines() if line.strip()
        ]
        protected = sorted(
            path
            for path in changed_paths
            if any(
                _path_matches_prefix(path, prefix)
                for prefix in contract.operator_owned_shared_surfaces
            )
        )
        if protected and branch_task.role != "Operator":
            _record_swarm_event(
                repo,
                {
                    "event": "merge_refused_operator_surface",
                    "task_id": task_id,
                    "paths": protected,
                    "role": branch_task.role,
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note=(
                    "@human Merge refused: non-Operator task touched Operator-owned "
                    f"surfaces: {', '.join(protected)}"
                ),
                message=f"{task_id}: block operator surface merge",
            )
            refused.append({"task_id": task_id, "reason": "operator_surface"})
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        # F6: the merge queue owns a CLEAN base; anything else is skipped
        # loudly rather than gambled with reset --hard later.
        status_cp = _run(
            ["git", "status", "--porcelain", "-uall"], cwd=repo, capture=True, check=True
        )
        event_paths = _runtime_event_paths(repo)
        dirty = [
            line
            for line in (status_cp.stdout or "").splitlines()
            if line.strip()
            and line[3:].strip() not in event_paths
            # machine-local attestations are operator state, never merge content
            and not line[3:].strip().startswith(".swarm/")
        ]
        if dirty:
            _record_swarm_event(
                repo,
                {"event": "merge_skipped_dirty_base", "task_id": task_id, "dirty": dirty[:10]},
                escalation=True,
            )
            refused.append({"task_id": task_id, "reason": "dirty_base"})
            continue

        # F2 (merge side): the approval binds to CONTENT and TOPOLOGY —
        # the manifest bytes must hash to what the Judge reviewed, and the
        # branch tip must be exactly the review commit atop the reviewed sha
        # (no post-approval commits of any kind).
        branch_tip_sha = _run(
            ["git", "rev-parse", branch], cwd=repo, capture=True, check=True
        ).stdout.strip()
        reviewed_sha = review.get("reviewed_branch_sha")
        review_manifest_sha = review.get("manifest_sha256")
        actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        tip_parent = _run(
            ["git", "rev-parse", f"{branch}^"], cwd=repo, capture=True, check=False
        ).stdout.strip()
        binding_failure: str | None = None
        if not isinstance(review_manifest_sha, str) or review_manifest_sha != actual_manifest_sha:
            binding_failure = "manifest_content_changed_after_review"
        elif not isinstance(reviewed_sha, str) or reviewed_sha != tip_parent:
            binding_failure = "post_review_commits_present"
        if binding_failure is not None:
            _record_swarm_event(
                repo,
                {
                    "event": "merge_refused_review_binding",
                    "task_id": task_id,
                    "reason": binding_failure,
                    "reviewed_branch_sha": reviewed_sha,
                    "branch_tip_parent": tip_parent,
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note=f"@human Merge refused: review binding failed ({binding_failure}).",
                message=f"{task_id}: block review binding merge",
            )
            refused.append({"task_id": task_id, "reason": binding_failure})
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        # re-run the tamper-evident checks against the branch state: a forged
        # approving review earns nothing these checks would not grant. Actor
        # separation was enforced at judge time by a distinct session; here
        # we verify CONTENT: pinned frontmatter, log binding, sha ancestry,
        # and strict path discipline split around the review commit (the
        # pre-review range must pass ownership WITH reviews disallowed; the
        # review commit itself may touch only the review log + task file).
        worktree_path = Path(context.get("worktree"))
        recheck_failures: list[str] = []

        frontmatter_block = manifest.get("frontmatter") if isinstance(manifest.get("frontmatter"), dict) else {}
        pinned_sha = frontmatter_block.get("pinned_sha256")
        try:
            current_text, _ = _task_frontmatter_snapshot(branch_task.path)
            current_fm_sha = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
        except ValueError:
            current_fm_sha = None
        if not isinstance(pinned_sha, str) or current_fm_sha != pinned_sha:
            recheck_failures.append("post_run_frontmatter_tamper")

        commands_block2 = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
        log_rel = commands_block2.get("executor_log_path")
        log_sha = commands_block2.get("executor_log_sha256")
        if manifest.get("provenance_class") == "executor_run":
            if not isinstance(log_rel, str) or not isinstance(log_sha, str):
                recheck_failures.append("executor_log_binding_missing")
            else:
                log_path = worktree_path / log_rel
                if not log_path.is_file() or hashlib.sha256(log_path.read_bytes()).hexdigest() != log_sha:
                    recheck_failures.append("executor_log_binding_failed")

        base_ref = _resolve_base_ref_for_diff(cwd=worktree_path, base_branch=args.base_branch, remote=args.remote)
        task_file_rel = branch_task.path.relative_to(worktree_path).as_posix()
        if base_ref is None:
            recheck_failures.append("merge_recheck_base_unresolved")
        else:
            range_cp = _run(
                ["git", "diff", "--name-only", f"{base_ref}...{tip_parent}"],
                cwd=worktree_path,
                capture=True,
                check=False,
            )
            for changed in [line.strip() for line in (range_cp.stdout or "").splitlines() if line.strip()]:
                ok, reason = _path_is_allowed(
                    path=changed,
                    allowed_paths=branch_task.allowed_paths,
                    disallowed_paths=branch_task.disallowed_paths,
                    task_file_path=task_file_rel,
                    task_id=task_id,
                )
                if not ok:
                    recheck_failures.append(f"ownership_violation:{changed}:{reason}")
            tip_cp = _run(
                ["git", "show", "--name-only", "--pretty=format:", branch],
                cwd=worktree_path,
                capture=True,
                check=False,
            )
            review_prefix = f"reports/status/reviews/{task_id}_"
            for tip_path in [line.strip() for line in (tip_cp.stdout or "").splitlines() if line.strip()]:
                if tip_path == task_file_rel or tip_path.startswith(review_prefix):
                    continue
                if tip_path in _task_projection_paths(task_file_rel):
                    continue
                recheck_failures.append(f"review_commit_touched:{tip_path}")
        if recheck_failures:
            _record_swarm_event(
                repo,
                {
                    "event": "merge_refused_integrity_recheck",
                    "task_id": task_id,
                    "failures": recheck_failures[:10],
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note="@human Merge refused: judge-checklist recheck failed at merge time.",
                message=f"{task_id}: block integrity recheck merge",
            )
            refused.append({"task_id": task_id, "reason": "integrity_recheck"})
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        # F5: cross-supervisor safety — the local base must equal the remote
        # tip before we move it (the CAS push below enforces it again).
        if _git_remote_exists(repo, args.remote):
            _run(["git", "fetch", args.remote, args.base_branch], cwd=repo, check=False)
            remote_tip = _run(
                ["git", "rev-parse", f"{args.remote}/{args.base_branch}"],
                cwd=repo,
                capture=True,
                check=False,
            ).stdout.strip()
            local_tip = _git_head_sha(repo)
            if remote_tip and local_tip and remote_tip != local_tip:
                _record_swarm_event(
                    repo,
                    {
                        "event": "merge_skipped_base_divergence",
                        "task_id": task_id,
                        "local": local_tip,
                        "remote": remote_tip,
                    },
                    escalation=True,
                )
                refused.append({"task_id": task_id, "reason": "base_divergence"})
                continue

        pre_merge_sha = _git_head_sha(repo)
        if pre_merge_sha is None:
            raise SystemExit("merge_precondition_missing_base_sha")
        # F9: the durable intent record FAILS CLOSED — no record, no merge.
        swarm_events.append_event(
            repo,
            {
                "event": "merge_started",
                "task_id": task_id,
                "branch": branch,
                "pre_merge_sha": pre_merge_sha,
            },
            actor_session=_ACTOR_SESSION_ID,
        )
        merge_cp = _run(
            ["git", "merge", "--ff-only", branch],
            cwd=repo,
            capture=True,
            check=False,
        )
        if merge_cp.returncode != 0:
            _record_swarm_event(
                repo,
                {
                    "event": "merge_refused_non_ff",
                    "task_id": task_id,
                    "branch": branch,
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note="@human Merge refused: task branch is not fast-forwardable.",
                message=f"{task_id}: block non-ff merge",
            )
            refused.append({"task_id": task_id, "reason": "non_ff"})
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        post_merge_sha = _git_head_sha(repo)

        merged_contract = load_framework_contract(repo)
        merged_tasks, merged_quarantined = load_tasks_quarantined(merged_contract)
        merged_task = _resolve_runtime_task(merged_tasks, merged_quarantined, task_id)
        filename = merged_task.path.name
        _move_task_to_state_projection(repo, merged_task)

        quality_cp = _run(
            [sys.executable, "scripts/quality_gates.py"],
            cwd=repo,
            capture=True,
            check=False,
            timeout_seconds=max(30, contract.gate_timeout_seconds * 2),
        )
        commands = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
        pinned_gates = [
            gate for gate in commands.get("gates", []) if isinstance(gate, str)
        ]
        pinned_ok, pinned_outputs = _run_gates(
            repo,
            pinned_gates,
            interpreter_allowlist=contract.gate_interpreter_allowlist,
            timeout_seconds=contract.gate_timeout_seconds,
        )
        if quality_cp.returncode != 0 or not pinned_ok:
            # F6: the reset target is the sha THIS step recorded, and the tip
            # must still be the merge this step created — anything else means
            # concurrent movement and demands a human, not a reset.
            current_head = _git_head_sha(repo)
            if current_head != post_merge_sha:
                _record_swarm_event(
                    repo,
                    {
                        "event": "merge_revert_refused_concurrent_movement",
                        "task_id": task_id,
                        "expected": post_merge_sha,
                        "actual": current_head,
                    },
                    escalation=True,
                )
                refused.append({"task_id": task_id, "reason": "concurrent_base_movement"})
                continue
            _run(["git", "reset", "--hard", pre_merge_sha], cwd=repo, check=True)
            _record_swarm_event(
                repo,
                {
                    "event": "merge_reverted",
                    "task_id": task_id,
                    "branch": branch,
                    "pre_merge_sha": pre_merge_sha,
                    "quality_returncode": quality_cp.returncode,
                    "quality_output_tail": (quality_cp.stdout or "")[-2000:],
                    "pinned_gates": pinned_outputs,
                },
                escalation=True,
            )
            _block_base_task(
                repo=repo,
                contract=contract,
                args=args,
                task_id=task_id,
                note="@human Merge reverted: post-merge verification failed.",
                message=f"{task_id}: block reverted merge",
            )
            reverted.append(task_id)
            base_tasks, _ = load_tasks_quarantined(contract)
            continue

        # F9: verification passed — record it durably (fail closed) BEFORE
        # any promotion; crash recovery keys off this record.
        swarm_events.append_event(
            repo,
            {
                "event": "merge_verified",
                "task_id": task_id,
                "pre_merge_sha": pre_merge_sha,
                "post_merge_sha": post_merge_sha,
            },
            actor_session=_ACTOR_SESSION_ID,
        )

        merged_tasks, merged_quarantined = load_tasks_quarantined(merged_contract)
        merged_task = _resolve_runtime_task(merged_tasks, merged_quarantined, task_id)
        _update_task_status_and_notes(
            task_path=merged_task.path,
            new_state="done",
            note_line=(
                "Supervisor merge queue passed quality_gates.py and the pinned "
                "task gates; claim released."
            ),
        )
        done_task = load_task(merged_task.path, merged_contract)
        _move_task_to_state_projection(repo, done_task)
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=[filename],
            message=f"{task_id}: done",
            strict=True,
            push=False,
        )
        # F5: base advance is a CAS against the sha we verified from — a
        # concurrent supervisor's push loses the race loudly, never silently.
        if _git_remote_exists(repo, args.remote):
            cas_cp = _run(
                [
                    "git",
                    "push",
                    args.remote,
                    f"{args.base_branch}:{args.base_branch}",
                    f"--force-with-lease={args.base_branch}:{pre_merge_sha}",
                ],
                cwd=repo,
                capture=True,
                check=False,
            )
            if cas_cp.returncode != 0:
                _run(["git", "reset", "--hard", pre_merge_sha], cwd=repo, check=True)
                _record_swarm_event(
                    repo,
                    {
                        "event": "merge_reverted",
                        "task_id": task_id,
                        "cause": "base_cas_push_lost",
                        "pre_merge_sha": pre_merge_sha,
                        "push_output": (cas_cp.stderr or cas_cp.stdout or "")[-500:],
                    },
                    escalation=True,
                )
                refused.append({"task_id": task_id, "reason": "base_cas_push_lost"})
                base_tasks, _ = load_tasks_quarantined(contract)
                continue
        _record_swarm_event(
            repo,
            {
                "event": "task_done",
                "task_id": task_id,
                "branch": branch,
                "pre_merge_sha": pre_merge_sha,
                "base_sha": _git_head_sha(repo),
            },
        )
        _release_current_claim(
            repo=repo,
            args=args,
            task_id=task_id,
            reason="task_done",
        )
        merged.append(task_id)
        base_tasks, _ = load_tasks_quarantined(contract)

    return {"merged": merged, "refused": refused, "reverted": reverted}


def _latest_manifest_time(contract: FrameworkContract, task_id: str) -> dt.datetime | None:
    latest: dt.datetime | None = None
    for _, payload in _matching_v2_run_manifest_data(
        _matching_task_jsons(contract.run_manifest_dir, task_id),
        task_id,
    ):
        stamp = _parse_utc_iso(payload.get("generated_at_utc"))
        if stamp is not None and (latest is None or stamp > latest):
            latest = stamp
    return latest


def _step_sweep(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    moves, problems = _apply_projection_sweep(repo)
    filenames = {source.name for source, _ in moves}
    tasks, quarantined = load_tasks_quarantined(contract)
    claims = swarm_claims.read_claims(repo, args.remote)
    events, _ = swarm_events.read_events(repo)
    reconciled: list[str] = []
    orphaned_candidates: list[str] = []

    for task_id, claim in sorted(claims.items()):
        task = tasks.get(task_id)
        if task is None or task.state != "backlog":
            continue
        filenames.add(task.path.name)
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="active",
            note_line=(
                f"Claim ref lease {claim.lease_id} is authoritative; reconciled stale "
                "backlog projection."
            ),
        )
        reconciled_task = load_task(task.path, contract)
        _move_task_to_state_projection(repo, reconciled_task)
        reconciled.append(task_id)
        _record_swarm_event(
            repo,
            {
                "event": "claim_projection_reconciled",
                "task_id": task_id,
                "lease_id": claim.lease_id,
            },
        )

    tasks, _ = load_tasks_quarantined(contract)
    for task_id, task in sorted(tasks.items()):
        if task.state != "active" or task_id in claims:
            continue
        claim_times = [
            _parse_utc_iso(event.get("ts_utc"))
            for event in events
            if event.get("task_id") == task_id
            and event.get("event") in {"claim_created", "lease_renewed", "claim_released"}
        ]
        last_claim = max((stamp for stamp in claim_times if stamp is not None), default=None)
        manifest_time = _latest_manifest_time(contract, task_id)
        if last_claim is not None and manifest_time is not None and manifest_time > last_claim:
            continue
        orphaned_candidates.append(task_id)
        _record_swarm_event(
            repo,
            {
                "event": "orphaned_candidate",
                "task_id": task_id,
                "last_claim_at_utc": (
                    last_claim.isoformat().replace("+00:00", "Z")
                    if last_claim is not None
                    else None
                ),
            },
            escalation=True,
        )

    second_moves, second_problems = _apply_projection_sweep(repo)
    filenames.update(source.name for source, _ in second_moves)
    problems.extend(second_problems)
    if filenames:
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=sorted(filenames),
            message="swarm: reconcile lifecycle projections",
            strict=bool(args.unattended),
        )
    for problem in problems:
        _record_swarm_event(
            repo,
            {"event": "sweep_problem", "problem": problem},
            escalation=True,
        )
    return {
        "moves": len(moves) + len(second_moves),
        "problems": problems,
        "reconciled": reconciled,
        "orphaned_candidates": orphaned_candidates,
        "quarantined": len(quarantined),
    }


def _worktree_dirty_paths(worktree: Path) -> list[str]:
    cp = _run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture=True,
        check=True,
    )
    ignored = _runtime_event_paths(worktree)
    dirty: list[str] = []
    for line in (cp.stdout or "").splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path in ignored:
            continue
        dirty.append(line)
    return dirty


def _step_clean(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, _ = load_tasks_quarantined(contract)
    done_ids = {task_id for task_id, task in tasks.items() if task.state == "done"}
    removed_worktrees: list[str] = []
    deleted_branches: list[str] = []
    dirty_worktrees: list[str] = []
    stale_branches: list[str] = []

    for record in _worktree_records(repo):
        branch = record.get("branch", "").removeprefix("refs/heads/")
        task_id = _parse_task_id_from_branch(branch)
        worktree_raw = record.get("worktree")
        if task_id not in done_ids or not worktree_raw:
            continue
        worktree = Path(worktree_raw)
        dirty = _worktree_dirty_paths(worktree)
        if dirty:
            dirty_worktrees.append(task_id)
            _record_swarm_event(
                repo,
                {
                    "event": "worktree_dirty",
                    "task_id": task_id,
                    "worktree": str(worktree),
                    "paths": dirty[:20],
                },
                escalation=True,
            )
            continue
        cp = _run(
            ["git", "worktree", "remove", str(worktree)],
            cwd=repo,
            capture=True,
            check=False,
        )
        if cp.returncode == 0:
            removed_worktrees.append(task_id)

    branches_cp = _run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
        cwd=repo,
        capture=True,
        check=True,
    )
    for branch in sorted(
        line.strip() for line in (branches_cp.stdout or "").splitlines() if line.strip()
    ):
        task_id = _parse_task_id_from_branch(branch)
        if task_id not in done_ids:
            continue
        merged = _run(
            ["git", "merge-base", "--is-ancestor", branch, args.base_branch],
            cwd=repo,
            capture=True,
            check=False,
        ).returncode == 0
        if not merged:
            stale_branches.append(branch)
            _record_swarm_event(
                repo,
                {
                    "event": "stale_task_branch",
                    "task_id": task_id,
                    "branch": branch,
                },
            )
            continue
        cp = _run(
            ["git", "branch", "-d", branch],
            cwd=repo,
            capture=True,
            check=False,
        )
        if cp.returncode == 0:
            deleted_branches.append(branch)
    _run(["git", "worktree", "prune"], cwd=repo, check=True)
    return {
        "removed_worktrees": removed_worktrees,
        "deleted_branches": deleted_branches,
        "dirty_worktrees": dirty_worktrees,
        "stale_branches": stale_branches,
    }


def _repair_context_from_manifest(manifest: dict[str, object]) -> str:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    gates = manifest.get("gates") if isinstance(manifest.get("gates"), list) else []
    diagnostics: list[dict[str, object]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("returncode") in {0, None} and not gate.get("timed_out") and not gate.get("constraint_violation"):
            continue
        diagnostics.append(
            {
                "command": gate.get("command"),
                "returncode": gate.get("returncode"),
                "timed_out": gate.get("timed_out"),
                "constraint_violation": gate.get("constraint_violation"),
                "output_head": str(gate.get("output_head", ""))[:700],
                "output_tail": str(gate.get("output_tail", ""))[-700:],
            }
        )
    payload = {
        "blocked_reasons": [
            reason for reason in result.get("blocked_reasons", []) if isinstance(reason, str)
        ],
        "gate_diagnostics": diagnostics,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)[:2048]


def _step_repair(
    args: argparse.Namespace,
    *,
    candidate_ids: set[str] | None = None,
) -> dict[str, object]:
    repo = _repo_root()
    contexts = _task_branch_contexts(repo)
    events, _ = swarm_events.read_events(repo)
    repaired: list[dict[str, object]] = []
    exhausted: list[str] = []
    integrity_blocks: list[str] = []
    already_exhausted = {
        event.get("task_id")
        for event in events
        if event.get("event") == "repair_exhausted"
    }

    for task_id, context in sorted(contexts.items()):
        if candidate_ids is not None and task_id not in candidate_ids:
            continue
        task = context.get("task")
        manifest = context.get("manifest")
        contract = context.get("contract")
        worktree = context.get("worktree")
        if (
            not isinstance(task, Task)
            or task.state not in {"active", "blocked"}
            or not isinstance(manifest, dict)
            or not isinstance(contract, FrameworkContract)
            or not isinstance(worktree, Path)
        ):
            continue
        result_block = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        reasons = [
            reason for reason in result_block.get("blocked_reasons", []) if isinstance(reason, str)
        ]
        integrity = "frontmatter_tampered" in reasons or any(
            "ownership" in reason for reason in reasons
        )
        if integrity:
            _record_swarm_event(
                repo,
                {
                    "event": "integrity_block",
                    "task_id": task_id,
                    "blocked_reasons": reasons,
                },
                escalation=True,
            )
            integrity_blocks.append(task_id)
            continue
        if not set(reasons).intersection({"executor_failed", "executor_timeout", "gates_failed"}):
            continue

        event_attempts = sum(
            event.get("event") == "run_finished" and event.get("task_id") == task_id
            for event in events
        )
        manifest_attempts = len(
            _matching_v2_run_manifest_data(
                _matching_task_jsons(contract.run_manifest_dir, task_id),
                task_id,
            )
        )
        attempts = max(event_attempts, manifest_attempts)
        if attempts >= contract.repair_max_attempts:
            if task_id not in already_exhausted:
                _record_swarm_event(
                    repo,
                    {
                        "event": "repair_exhausted",
                        "task_id": task_id,
                        "attempts": attempts,
                        "max_attempts": contract.repair_max_attempts,
                    },
                    escalation=True,
                )
            exhausted.append(task_id)
            continue

        repair_context = _repair_context_from_manifest(manifest)
        _update_task_status_and_notes(
            task_path=task.path,
            new_state="active",
            note_line=(
                f"Supervisor repair attempt {attempts + 1}/{contract.repair_max_attempts}; "
                "failure context injected."
            ),
        )
        returncode, output = _run_task_in_process(
            event_repo=repo,
            worktree=worktree,
            args=args,
            task_id=task_id,
            repair_context=repair_context,
        )
        repaired.append(
            {
                "task_id": task_id,
                "returncode": returncode,
                "repair_context": repair_context,
                "output": output.strip(),
            }
        )
        if returncode != 0 and attempts + 1 >= contract.repair_max_attempts:
            _record_swarm_event(
                repo,
                {
                    "event": "repair_exhausted",
                    "task_id": task_id,
                    "attempts": attempts + 1,
                    "max_attempts": contract.repair_max_attempts,
                },
                escalation=True,
            )
            exhausted.append(task_id)
    return {
        "repaired": repaired,
        "exhausted": exhausted,
        "integrity_blocks": integrity_blocks,
    }


def _step_plan(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    _record_swarm_event(repo, {"event": "plan_step_noop"})
    return {"status": "noop", "reason": "planner_runtime_arrives_in_m2"}


def _step_escalate(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)
    counts = {
        state: sum(task.state == state for task in tasks.values())
        for state in DEFAULT_ALLOWED_STATES
    }
    now = dt.datetime.now(tz=dt.timezone.utc)
    claims = swarm_claims.read_claims(repo, args.remote)
    live = sum(not claim.expired(now=now) for claim in claims.values())
    expired = len(claims) - live
    spend, usage_records = _usage_records(repo)
    snapshot = {
        "event": "status_snapshot",
        "state_counts": counts,
        "claims": {"live": live, "expired": expired},
        "ready_for_review": len(_ready_for_review_contexts(repo)),
        "quarantined": len(quarantined),
        "spend_usd": spend,
        "usage_records": usage_records,
    }
    _record_swarm_event(repo, snapshot)
    return {key: value for key, value in snapshot.items() if key != "event"}


def _step_account(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    spend, usage_records = _usage_records(repo)
    if spend is None or contract.budget_max_program_usd is None:
        reason = "usage_unavailable" if spend is None else "budget_unconfigured"
        _record_swarm_event(
            repo,
            {
                "event": "account_no_data",
                "reason": reason,
                "spend_usd": spend,
                "usage_records": usage_records,
            },
        )
        return {
            "status": "no_data",
            "reason": reason,
            "spend_usd": spend,
            "usage_records": usage_records,
        }
    exceeded = spend > contract.budget_max_program_usd
    event = {
        "event": "budget_exceeded" if exceeded else "account_snapshot",
        "spend_usd": spend,
        "max_program_usd": contract.budget_max_program_usd,
        "usage_records": usage_records,
    }
    _record_swarm_event(repo, event, escalation=exceeded)
    return {
        "status": "exceeded" if exceeded else "within_budget",
        "spend_usd": spend,
        "max_program_usd": contract.budget_max_program_usd,
        "usage_records": usage_records,
    }


def _supervise_cycle(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    summary: dict[str, object] = {
        "cycle_started_at_utc": _utc_now_iso(),
        "actor_session": _ACTOR_SESSION_ID,
    }

    def run_step(name: str, func, *func_args, **func_kwargs) -> dict[str, object]:
        result = func(*func_args, **func_kwargs)
        summary[name.lower()] = result
        _record_swarm_event(
            repo,
            {
                "event": "supervisor_step_completed",
                "step": name,
                "summary": result,
            },
        )
        return result

    run_step("SYNC", _step_sync, args)
    run_step("REAP", _step_reap, args)
    judge_candidates = set(_ready_for_review_contexts(repo))
    repair_candidates = {
        task_id
        for task_id, context in _task_branch_contexts(repo).items()
        if isinstance(context.get("task"), Task)
        and context["task"].state in {"active", "blocked"}
        and isinstance(context.get("manifest"), dict)
        and isinstance(context["manifest"].get("result"), dict)
        and context["manifest"]["result"].get("status") == "blocked"
    }
    run_step("TICK", _step_tick, args)
    run_step("JUDGE", _step_judge, args, candidate_ids=judge_candidates)
    run_step("MERGE", _step_merge, args)
    run_step("SWEEP", _step_sweep, args)
    run_step("CLEAN", _step_clean, args)
    run_step("REPAIR", _step_repair, args, candidate_ids=repair_candidates)
    run_step("PLAN", _step_plan, args)
    run_step("ESCALATE", _step_escalate, args)
    run_step("ACCOUNT", _step_account, args)
    summary["cycle_finished_at_utc"] = _utc_now_iso()
    return summary


def _attempt_supervise_iteration(
    args: argparse.Namespace,
    *,
    interval_seconds: int,
    consecutive_failures: int,
    repo: Path,
) -> tuple[int, int]:
    try:
        _supervise_cycle(args)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        consecutive_failures += 1
        backoff_seconds = _handle_loop_failure(
            exc,
            interval_seconds=interval_seconds,
            consecutive_failures=consecutive_failures,
            repo=repo,
        )
        return consecutive_failures, backoff_seconds
    return 0, 0


def cmd_supervise(args: argparse.Namespace) -> int:
    repo = _repo_root()
    if args.runner != "local":
        raise SystemExit("supervise_runner_must_be_local")
    if args.unattended:
        _require_unattended_ack()
    else:
        # inherently unattended-class entrypoint (§9.4): containment is not
        # optional even without the flag
        _require_containment(_repo_root())
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=False,
    )
    if args.once:
        summary = _supervise_cycle(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    interval_seconds = max(5, int(args.interval_seconds))
    print(f"swarm_supervisor_started interval={interval_seconds}s repo={repo}")
    consecutive_failures = 0
    while True:
        try:
            consecutive_failures, backoff_seconds = _attempt_supervise_iteration(
                args,
                interval_seconds=interval_seconds,
                consecutive_failures=consecutive_failures,
                repo=repo,
            )
            remaining = interval_seconds + backoff_seconds
            while remaining > 0:
                sleep_seconds = min(5, remaining)
                time.sleep(sleep_seconds)
                remaining -= sleep_seconds
        except KeyboardInterrupt:
            print("swarm_supervisor_stopped")
            return 0


def _loop_iteration(args: argparse.Namespace) -> int:
    repo = _repo_root()
    _supervisor_sync_to_remote_base(repo=repo, remote=args.remote, base_branch=args.base_branch)
    return cmd_tick(args)


def _handle_loop_failure(
    exc: BaseException,
    *,
    interval_seconds: int,
    consecutive_failures: int,
    repo: Path | None = None,
) -> int:
    print(
        f"[loop] escalation iteration_failed kind={type(exc).__name__} detail={exc}",
        file=sys.stderr,
    )
    backoff_seconds = min(3600, interval_seconds * (2 ** max(0, consecutive_failures - 1)))
    if repo is not None:
        _record_swarm_event(
            repo,
            {
                "event": "loop_iteration_failed",
                "kind": type(exc).__name__,
                "detail": str(exc),
                "backoff_seconds": backoff_seconds,
            },
            escalation=True,
        )
    return backoff_seconds


def _attempt_loop_iteration(
    args: argparse.Namespace,
    *,
    interval_seconds: int,
    consecutive_failures: int,
    repo: Path | None = None,
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
            repo=repo,
        )
        return consecutive_failures, backoff_seconds
    return 0, 0


def cmd_loop(args: argparse.Namespace) -> int:
    repo = _repo_root()

    if args.unattended:
        _require_unattended_ack()
    else:
        # inherently unattended-class entrypoint (§9.4): containment is not
        # optional even without the flag
        _require_containment(_repo_root())

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
                repo=repo,
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
    else:
        # inherently unattended-class entrypoint (§9.4): containment is not
        # optional even without the flag
        _require_containment(_repo_root())

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


def _resolve_runtime_task(tasks: dict[str, Task], quarantined: list[dict[str, str]], task_id: str) -> Task:
    task = tasks.get(task_id)
    if task is not None:
        return task
    for record in quarantined:
        if Path(record.get("path", "")).name.startswith(f"{task_id}_"):
            raise SystemExit(f"task_quarantined:{task_id}:{record.get('error')}")
    raise SystemExit(f"unknown_task_id:{task_id}")


import threading


@contextlib.contextmanager
def _lease_heartbeat(*, repo: Path, remote: str, task_id: str):
    """§4.1 session model: workers heartbeat below 50% of the lease TTL even
    while blocked on the executor. Renewal failures are journaled, never
    raised from the timer thread (the post-run renewal fails loudly)."""
    claims = swarm_claims.read_claims(repo, remote, fetch=False)
    claim = claims.get(task_id)
    if claim is None or claim.session_id != _ACTOR_SESSION_ID:
        yield
        return
    ttl = claim.payload.get("lease_ttl_seconds")
    interval = max(5, int(ttl) // 3) if isinstance(ttl, int) and ttl > 0 else 1200
    stop = threading.Event()

    def _beat() -> None:
        current_sha = claim.sha
        while not stop.wait(interval):
            try:
                renewed = swarm_claims.renew_lease(
                    repo,
                    remote,
                    task_id,
                    expected_sha=current_sha,
                    session_id=_ACTOR_SESSION_ID,
                    journal=lambda event: _record_swarm_event(repo, event),
                )
                if renewed.ok and renewed.sha:
                    current_sha = renewed.sha
                else:
                    _record_swarm_event(
                        repo,
                        {
                            "event": "heartbeat_failed",
                            "task_id": task_id,
                            "reason": renewed.reason,
                        },
                        escalation=True,
                    )
                    return
            except Exception as exc:
                _record_swarm_event(
                    repo,
                    {"event": "heartbeat_failed", "task_id": task_id, "reason": str(exc)},
                    escalation=True,
                )
                return

    thread = threading.Thread(target=_beat, name=f"lease-heartbeat-{task_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=10)


def _renew_runtime_claim(
    *,
    repo: Path,
    remote: str,
    task_id: str,
) -> dict[str, object] | None:
    claims = swarm_claims.read_claims(repo, remote)
    claim = claims.get(task_id)
    if claim is None:
        return None
    if claim.session_id != _ACTOR_SESSION_ID:
        raise SystemExit(
            f"claim_session_mismatch:{task_id}:{claim.session_id}:{_ACTOR_SESSION_ID}"
        )
    renewed = swarm_claims.renew_lease(
        repo,
        remote,
        task_id,
        expected_sha=claim.sha,
        session_id=_ACTOR_SESSION_ID,
        journal=lambda event: _record_swarm_event(repo, event),
    )
    if not renewed.ok or renewed.sha is None or renewed.lease_id is None:
        raise SystemExit(f"claim_heartbeat_failed:{task_id}:{renewed.reason}")
    return {
        "lease_id": renewed.lease_id,
        "sha": renewed.sha,
        "transport": renewed.transport,
    }


def cmd_run_task(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)

    if args.unattended:
        _require_unattended_ack()

    task = _resolve_runtime_task(tasks, quarantined, args.task_id)

    if args.codex_sandbox == "danger-full-access" and not (
        getattr(args, "i_accept_full_access", False) and task.allow_network
    ):
        raise SystemExit(
            "full_access_requires_double_opt_in:"
            "--i-accept-full-access AND task allow_network: true (§9.4)"
        )

    if task.role not in set(contract.task_execution_roles):
        raise SystemExit(f"task_not_runtime_executable:{task.task_id}:{task.role}")

    if task.state not in {"backlog", "active", "blocked", "integration_ready"}:
        raise SystemExit(f"task_not_runnable_from_state:{task.task_id}:{task.state}")

    missing_dependencies = [
        dep_id
        for dep_id in task.dependencies
        if not dependency_is_satisfied(dep_id, task, tasks, contract)
    ]
    dependencies_satisfied = _dependencies_satisfied(task, tasks, contract)
    force_deps = bool(getattr(args, "force_deps", False))
    if not dependencies_satisfied and not force_deps:
        raise SystemExit(f"dependencies_unsatisfied:{task.task_id}:{','.join(missing_dependencies)}")

    _require_git_identity(cwd=repo, reason="runtime")
    _preflight_strict_sync_requirements(
        cwd=repo,
        remote=args.remote,
        unattended=bool(args.unattended),
        create_pr=bool(args.create_pr),
    )
    strict_sync = bool(args.unattended or args.create_pr)
    claim_stamp = _renew_runtime_claim(
        repo=repo,
        remote=args.remote,
        task_id=task.task_id,
    )

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

    _record_swarm_event(
        repo,
        {
            "event": "run_started",
            "task_id": task.task_id,
            "state_before": state_before,
        },
    )

    blocked_reasons: list[str] = []
    executor_command: list[str] = []
    executor_returncode: int | None = None
    executor_error: str | None = None
    executor_log_relpath: str | None = None
    executor_log_sha256: str | None = None
    executor_wall_clock_seconds: float | None = None
    executor_usage: dict[str, object] | None = None
    executor_session_relpath: str | None = None
    executor_backend = getattr(args, "executor_backend", "codex")

    if task.allow_network and task.workstream not in set(contract.network_workstreams):
        blocked_reasons.append("network_policy_violation")

    if args.final_state == "integration_ready":
        if not task_is_integration_ready_eligible(task, contract):
            blocked_reasons.append("integration_ready_ineligible")
        elif not downstream_allowlist_exists(task.task_id, tasks):
            blocked_reasons.append("integration_ready_missing_downstream_allowlist")

    run_timestamp = _utc_timestamp_compact()
    run_id = f"{task.task_id}_{run_timestamp}"
    pinned_frontmatter_text, pinned_frontmatter = _task_frontmatter_snapshot(task.path)
    pinned_frontmatter_sha256 = hashlib.sha256(pinned_frontmatter_text.encode("utf-8")).hexdigest()

    if not args.skip_executor and not blocked_reasons:
        executor_output: object = None
        executor_started: float | None = None
        try:
            prompt_path = _executor_prompt_path(task, contract)
            prompt = load_prompt(
                prompt_path,
                _build_prompt_context(task, repo, args.repair_context),
            )
            if executor_backend == "codex":
                executor_command = _codex_exec_cmd(
                    prompt=prompt,
                    model=args.codex_model,
                    sandbox=args.codex_sandbox,
                    unattended=args.unattended,
                    allow_network=task.allow_network,
                    workdir=repo,
                )
            else:
                executor_command = ["mock", _mock_transcript_relpath(task.task_id)]
            executor_started = time.perf_counter()
            execution_values = dict(vars(args))
            execution_values["_executor_command"] = executor_command
            execution_args = argparse.Namespace(**execution_values)
            with _lease_heartbeat(repo=repo, remote=args.remote, task_id=task.task_id):
                outcome = _execute_task(
                    backend=executor_backend,
                    task=task,
                    prompt=prompt,
                    args=execution_args,
                    repo=repo,
                    timeout_seconds=int(args.max_worker_seconds) if args.max_worker_seconds else None,
                )
            executor_returncode = outcome.returncode
            executor_output = outcome.stdout
            executor_wall_clock_seconds = outcome.wall_clock_seconds
            executor_usage = outcome.usage
            if outcome.returncode != 0:
                blocked_reasons.append("executor_failed")
        except subprocess.TimeoutExpired as exc:
            executor_output = exc.stdout
            executor_error = "executor_timeout"
            blocked_reasons.append("executor_timeout")
        except Exception as exc:
            executor_output = getattr(exc, "stdout", None) or getattr(exc, "output", None)
            if not executor_output:
                detail = str(exc).replace("\n", " ").strip()
                executor_output = f"executor_error:{type(exc).__name__}:{detail}\n"
            executor_error = str(exc)
            blocked_reasons.append("executor_unavailable")
        finally:
            if executor_started is not None and executor_wall_clock_seconds is None:
                executor_wall_clock_seconds = max(0.0, time.perf_counter() - executor_started)
            if executor_wall_clock_seconds is not None:
                executor_usage = _usage_with_cost_estimate(
                    repo=repo,
                    model=args.codex_model,
                    wall_clock_seconds=executor_wall_clock_seconds,
                    captured_usage=executor_usage,
                )
            executor_log_relpath, executor_log_sha256 = _write_executor_log(
                repo=repo,
                run_id=run_id,
                output=executor_output,
            )
            if bool(getattr(args, "record_session", False)) and executor_wall_clock_seconds is not None:
                assert executor_usage is not None
                executor_session_relpath = _write_executor_session(
                    repo=repo,
                    run_id=run_id,
                    backend=executor_backend,
                    argv=executor_command,
                    returncode=executor_returncode,
                    wall_clock_seconds=executor_usage["wall_clock_seconds"],
                    stdout=executor_output,
                    usage=executor_usage,
                )
    elif args.skip_executor:
        executor_error = "executor_skipped"

    if not args.skip_executor and claim_stamp is not None:
        try:
            claim_stamp = _renew_runtime_claim(
                repo=repo,
                remote=args.remote,
                task_id=task.task_id,
            )
        except SystemExit as exc:
            executor_error = str(exc)
            blocked_reasons.append("claim_heartbeat_failed")

    try:
        task_text_after_executor = _read_text(task.path)
        current_frontmatter = _parse_task_frontmatter(task_text_after_executor)
    except Exception:
        task_text_after_executor = ""
        current_frontmatter = None
    tampered_keys = _frontmatter_tampered_keys(pinned_frontmatter, current_frontmatter)
    frontmatter_tampered = bool(tampered_keys)
    if frontmatter_tampered:
        blocked_reasons.append("frontmatter_tampered")

    gate_ok, gate_outputs = _run_gates(
        repo,
        task.gates,
        interpreter_allowlist=contract.gate_interpreter_allowlist,
        timeout_seconds=contract.gate_timeout_seconds,
    )
    if not gate_ok:
        blocked_reasons.append("gates_failed")

    base_ref = _resolve_base_ref_for_diff(cwd=repo, base_branch=args.base_branch, remote=args.remote)
    ownership_failures: list[dict[str, str]] = []
    uncommitted_violations: list[str] = []
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
            uncommitted_violations.append(changed_path)
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

    task_state_after_executor = _parse_status_value(task_text_after_executor, "State")
    if task_state_after_executor == "blocked":
        blocked_reasons.append("task_marked_blocked")

    blocked_reasons = _dedupe_preserve(blocked_reasons)
    if blocked_reasons:
        state_after = "blocked"
    elif (
        task_state_after_executor in {"active", "integration_ready", "ready_for_review"}
        and not bool(getattr(args, "supervisor_managed", False))
    ):
        # Respect the task state the worker left behind. The runtime should not
        # silently promote an active task to ready_for_review just because the
        # default final_state is reviewable.
        state_after = task_state_after_executor
    else:
        state_after = args.final_state

    run_manifest_path = _next_json_artifact_path(contract.run_manifest_dir, task.task_id, run_timestamp)
    run_manifest_relpath = run_manifest_path.relative_to(repo).as_posix()

    commands_block: dict[str, object] = {
        "executor": executor_command,
        "executor_log_path": executor_log_relpath,
        "executor_log_sha256": executor_log_sha256,
        "gates": list(task.gates),
    }
    if executor_session_relpath is not None:
        commands_block["session_path"] = executor_session_relpath

    run_manifest = {
        "schema_version": SWARM_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "provenance_class": "manual_operator" if args.skip_executor else "executor_run",
        "actor": {
            "session_id": _ACTOR_SESSION_ID,
            "recorded_at_utc": _utc_now_iso(),
        },
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
            "tool": executor_backend if not args.skip_executor else "manual",
            "model": args.codex_model,
            "sandbox": args.codex_sandbox,
            "allow_network": task.allow_network,
            "full_access_opt_in": bool(
                args.codex_sandbox == "danger-full-access"
                and getattr(args, "i_accept_full_access", False)
            ),
            "effective_network": {
                "declared_allow_network": task.allow_network,
                "sandbox": args.codex_sandbox,
                "backend": getattr(args, "executor_backend", "codex"),
                "enforcement": (
                    "mock_backend"
                    if getattr(args, "executor_backend", "codex") == "mock"
                    else "codex_sandbox"
                ),
            },
            "repair_context": args.repair_context,
            "returncode": executor_returncode,
            "error": executor_error,
        },
        "commands": commands_block,
        "frontmatter": {
            "pinned_sha256": pinned_frontmatter_sha256,
            "tampered": frontmatter_tampered,
            "tampered_keys": tampered_keys,
        },
        "gates": gate_outputs,
        "ownership": {
            "ok": not ownership_failures,
            "changed_paths": changed_paths,
            "violations": ownership_failures,
            "uncommitted_violations": sorted(set(uncommitted_violations)),
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
    if force_deps:
        run_manifest["overrides"] = {
            "force_deps": True,
            "unsatisfied_dependencies": missing_dependencies,
        }
    if quarantined:
        run_manifest["quarantined_tasks"] = quarantined
    if claim_stamp is not None:
        run_manifest["claim"] = claim_stamp
    if executor_usage is not None:
        run_manifest["usage"] = executor_usage
    _write_json(run_manifest_path, run_manifest)
    _record_swarm_event(
        repo,
        {
            "event": "run_finished",
            "task_id": task.task_id,
            "status": run_manifest["result"]["status"],
            "blocked_reasons": blocked_reasons,
            "run_manifest": run_manifest_relpath,
            "provenance_class": run_manifest["provenance_class"],
        },
    )

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
    if state_after == "blocked" and "@human" in note:
        _record_swarm_event(
            repo,
            {
                "event": "human_question",
                "task_id": task.task_id,
                "blocked_reasons": blocked_reasons,
                "note": note,
            },
            escalation=True,
        )

    if _git_has_changes(repo):
        task_file_rel = task.path.relative_to(repo).as_posix()
        control_plane_paths = {task_file_rel, run_manifest_relpath}
        if executor_log_relpath is not None:
            control_plane_paths.add(executor_log_relpath)
        if executor_session_relpath is not None:
            control_plane_paths.add(executor_session_relpath)

        final_path_sources, _ = _collect_changed_paths_with_sources(repo=repo, base_ref=base_ref)
        for violating_path in sorted(set(uncommitted_violations)):
            _git_unstage_path(repo, violating_path)

        paths_to_commit: list[str] = []
        for changed_path in sorted(set(final_path_sources) | control_plane_paths):
            allowed, _ = _path_is_allowed(
                path=changed_path,
                allowed_paths=task.allowed_paths,
                disallowed_paths=task.disallowed_paths,
                task_file_path=task_file_rel,
                task_id=task.task_id,
            )
            if allowed or changed_path in control_plane_paths:
                command = ["git", "add"]
                if changed_path == executor_log_relpath:
                    command.append("-f")
                command.extend(["--", changed_path])
                _run(command, cwd=repo, check=True)
                paths_to_commit.append(changed_path)

        if paths_to_commit:
            _git_commit(
                cwd=repo,
                message=f"{task.task_id}: {state_after}",
                strict=strict_sync,
                paths=paths_to_commit,
            )
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
    tasks, quarantined = load_tasks_quarantined(contract)

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

    task = _resolve_runtime_task(tasks, quarantined, args.task_id)
    if task.state != "ready_for_review":
        raise SystemExit(f"task_not_ready_for_review:{task.task_id}:{task.state}")

    outputs_ok, output_failures = _check_declared_outputs_exist(repo=repo, task=task)
    manifest_failures = required_manifest_failures(repo, task)

    candidate_run_manifests = _matching_task_jsons(contract.run_manifest_dir, task.task_id)
    matching_v2_manifests = _matching_v2_run_manifest_data(candidate_run_manifests, task.task_id)
    valid_run_manifests = [
        path for path, _ in matching_v2_manifests if _is_valid_run_manifest(path, task.task_id)
    ]
    review_bundle_failures: list[str] = []
    if not valid_run_manifests:
        passing_v2_manifests = [
            (path, data)
            for path, data in matching_v2_manifests
            if isinstance(data.get("result"), dict) and data["result"].get("status") == "ok"
        ]
        if not matching_v2_manifests:
            review_bundle_failures.append("missing_valid_run_manifest")
        elif not passing_v2_manifests:
            review_bundle_failures.append("no_passing_run_manifest")
        elif passing_v2_manifests[-1][1].get("provenance_class") in {"manual_operator", "backfill"}:
            review_bundle_failures.append("provenance_requires_independent_reverification")
        else:
            review_bundle_failures.append("missing_valid_run_manifest")

    selected_run_manifest = (
        valid_run_manifests[-1]
        if valid_run_manifests
        else (matching_v2_manifests[-1][0] if matching_v2_manifests else None)
    )
    run_manifest_relpath = selected_run_manifest.relative_to(repo).as_posix() if selected_run_manifest else None

    selected_manifest_data: dict[str, object] = {}
    for path, data in matching_v2_manifests:
        if path == selected_run_manifest:
            selected_manifest_data = data
            break

    # The Judge executes the PINNED gate commands from the manifest it is
    # judging, never the live (editable) task file's copy (§4.0 #12).
    commands_block = (
        selected_manifest_data.get("commands")
        if isinstance(selected_manifest_data.get("commands"), dict)
        else {}
    )
    pinned_gates = [
        gate for gate in commands_block.get("gates", []) if isinstance(gate, str)
    ] if valid_run_manifests else []
    gate_ok, gate_outputs = _run_gates(
        repo,
        pinned_gates if pinned_gates else task.gates,
        interpreter_allowlist=contract.gate_interpreter_allowlist,
        timeout_seconds=contract.gate_timeout_seconds,
    )

    integrity_failures: list[str] = []
    if valid_run_manifests:
        integrity_failures.extend(
            _judge_manifest_integrity_failures(
                repo=repo,
                task=task,
                manifest=selected_manifest_data,
                contract=contract,
            )
        )
    integrity_failures.extend(
        _judge_ownership_failures(
            repo=repo,
            task=task,
            base_branch=args.base_branch,
            remote=args.remote,
        )
    )
    review_bundle_failures.extend(integrity_failures)

    approved = gate_ok and outputs_ok and not manifest_failures and not review_bundle_failures
    promote_directly = bool(getattr(args, "promote_directly", False))
    approve_only = not promote_directly
    if promote_directly:
        _record_swarm_event(
            repo,
            {
                "event": "judge_promote_directly",
                "task_id": task.task_id,
                "note": "manual override: done promoted without the merge queue's post-merge verification",
            },
            escalation=True,
        )
    intended_state_after = "done" if approved else args.on_fail
    actual_state_after = (
        "ready_for_review" if approved and approve_only else intended_state_after
    )
    outcome = "approve" if approved else ("block" if args.on_fail == "blocked" else "revise")

    review_log_path = _next_json_artifact_path(contract.judge_review_dir, task.task_id, _utc_timestamp_compact())
    review_log_relpath = review_log_path.relative_to(repo).as_posix()

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

    reviewed_branch_sha = _git_head_sha(repo)
    reviewed_manifest_sha256 = (
        hashlib.sha256(selected_run_manifest.read_bytes()).hexdigest()
        if selected_run_manifest is not None and selected_run_manifest.is_file()
        else None
    )
    review_log = {
        "schema_version": JUDGE_REVIEW_LOG_SCHEMA_VERSION,
        "review_id": f"{task.task_id}_{_utc_timestamp_compact()}",
        "generated_at_utc": _utc_now_iso(),
        "reviewed_branch_sha": reviewed_branch_sha,
        "manifest_sha256": reviewed_manifest_sha256,
        "reviewer": {
            "role": contract.scientific_review_role,
            "session_id": _ACTOR_SESSION_ID,
            "recorded_at_utc": _utc_now_iso(),
        },
        "operator_attestation": None,
        "task": {
            "task_id": task.task_id,
            "task_path": task.path.relative_to(repo).as_posix(),
            "role": task.role,
            "state_before": task.state,
            "state_after": intended_state_after,
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
            "outcome": outcome,
            "note": decision_note,
        },
    }
    _write_json(review_log_path, review_log)
    review_event = {
        "task_id": task.task_id,
        "outcome": outcome,
        "failures": check_failures,
        "review_log": review_log_relpath,
    }
    _record_swarm_event(
        repo,
        {"event": "review_recorded", **review_event},
    )
    if not approved and args.on_fail == "blocked":
        _record_swarm_event(
            repo,
            {"event": "judge_block", **review_event},
            escalation=True,
        )

    task_note = (
        (
            f"Judge approved pending supervisor merge; review log: {review_log_relpath}"
            if approve_only
            else f"Judge approved; review log: {review_log_relpath}"
        )
        if approved
        else f"@human Judge returned task; review log: {review_log_relpath}; failures: {', '.join(check_failures)}"
    )
    _update_task_status_and_notes(
        task_path=task.path,
        new_state=actual_state_after,
        note_line=task_note,
    )

    if _git_has_changes(repo):
        # The Judge commits only its own control-plane artifacts; anything else
        # in the tree (e.g. violations a run left uncommitted) stays uncommitted.
        judge_paths = [review_log_relpath, task.path.relative_to(repo).as_posix()]
        for judge_path in judge_paths:
            _run(["git", "add", "--", judge_path], cwd=repo, check=True)
        _git_commit(
            cwd=repo,
            message=f"{task.task_id}: {'approved_pending_merge' if approved and approve_only else actual_state_after}",
            strict=strict_sync,
            paths=judge_paths,
        )
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
                "state_after": (
                    "approved_pending_merge"
                    if approved and approve_only
                    else actual_state_after
                ),
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

    status = subparsers.add_parser("status", help="Render task, lease, journal, and spend status")
    status.add_argument("--remote", default="origin")
    status.add_argument("--no-fetch", action="store_true")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    costs = subparsers.add_parser("costs", help="Aggregate executor usage and estimated spend")
    costs.add_argument("--json", action="store_true")
    costs.set_defaults(func=cmd_costs)

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
    tick.add_argument("--executor-backend", choices=["codex", "mock"], default="codex")
    tick.add_argument("--codex-model", default=None)
    tick.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    tick.add_argument("--i-accept-full-access", action="store_true", dest="i_accept_full_access")
    tick.add_argument("--unattended", action="store_true")
    tick.add_argument("--max-worker-seconds", type=int, default=0)
    tick.add_argument("--create-pr", action="store_true")
    tick.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    tick.add_argument("--dry-run", action="store_true")
    tick.set_defaults(func=cmd_tick)

    supervise = subparsers.add_parser(
        "supervise",
        help="Run the crash-only Operator supervisor state machine",
    )
    supervise.add_argument("--once", action="store_true")
    supervise.add_argument("--interval-seconds", type=int, default=300)
    supervise.add_argument("--runner", choices=["local"], default="local")
    supervise.add_argument("--max-workers", type=int, default=1)
    supervise.add_argument("--worktree-parent", default=None)
    supervise.add_argument("--remote", default="origin")
    supervise.add_argument("--base-branch", default="main")
    supervise.add_argument("--executor-backend", choices=["codex", "mock"], default="codex")
    supervise.add_argument("--codex-model", default=None)
    supervise.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
    )
    supervise.add_argument("--unattended", action="store_true")
    supervise.add_argument("--max-worker-seconds", type=int, default=0)
    supervise.set_defaults(func=cmd_supervise)

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
    loop.add_argument("--i-accept-full-access", action="store_true", dest="i_accept_full_access")
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
    tmux_start.add_argument("--i-accept-full-access", action="store_true", dest="i_accept_full_access")
    tmux_start.add_argument("--unattended", action="store_true")
    tmux_start.add_argument("--max-worker-seconds", type=int, default=0)
    tmux_start.add_argument("--create-pr", action="store_true")
    tmux_start.add_argument("--final-state", choices=["integration_ready", "ready_for_review"], default="ready_for_review")
    tmux_start.set_defaults(func=cmd_tmux_start)

    run_task = subparsers.add_parser("run-task", help="Execute one Worker/Operator task in the current worktree")
    run_task.add_argument("--task-id", required=True)
    run_task.add_argument("--remote", default="origin")
    run_task.add_argument("--base-branch", default="main")
    run_task.add_argument("--executor-backend", choices=["codex", "mock"], default="codex")
    run_task.add_argument("--codex-model", default=None)
    run_task.add_argument("--codex-sandbox", choices=["read-only", "workspace-write", "danger-full-access"], default="workspace-write")
    run_task.add_argument("--i-accept-full-access", action="store_true", dest="i_accept_full_access")
    run_task.add_argument("--unattended", action="store_true")
    run_task.add_argument("--skip-executor", action="store_true")
    run_task.add_argument("--record-session", action="store_true")
    run_task.add_argument("--force-deps", action="store_true")
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
    judge_task.add_argument("--approve-only", action="store_true", help="(default behavior; retained for compatibility)")
    judge_task.add_argument("--promote-directly", action="store_true", dest="promote_directly", help="Manual override: promote done without the merge queue (loudly journaled)")
    judge_task.set_defaults(func=cmd_judge_task)

    attest = subparsers.add_parser("attest-containment", help="Record the machine-local containment attestation required for unattended runs")
    attest.add_argument("--attested-by", required=True)
    attest.add_argument("--note", default="")
    attest.add_argument(
        "--waive-credential-class",
        action="append",
        dest="waive_credential_class",
        default=[],
        help="Explicitly waive one credential class the scan may find (e.g. a scoped deploy key); recorded in the attested marker",
    )
    attest.set_defaults(func=cmd_attest_containment)

    ack = subparsers.add_parser("ack-vendor-policy", help="Record the one-time vendor-policy compatibility acknowledgment for unattended use")
    ack.add_argument("--vendor", required=True)
    ack.add_argument("--note", required=True)
    ack.add_argument("--acked-by", required=True)
    ack.set_defaults(func=cmd_ack_vendor_policy)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
