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
from calibrate_referee import calibration_report_failures
from generate_revision_tasks import generate_revision_tasks
from swarm_taskfile import WorktreeCollisionError
from swarm_taskfile import PREREG_LOCK_SCHEMA_VERSION
from swarm_taskfile import PREREG_PHASE_FILES
from swarm_taskfile import gate_command_violation
from swarm_taskfile import REQUIRED_FRONTMATTER_KEYS
from swarm_taskfile import TASK_SCHEMA_VERSION
from swarm_taskfile import TaskV2Fields
from swarm_taskfile import extract_section as _extract_section
from swarm_taskfile import lint_task_files
from swarm_taskfile import load_prereg_lock
from swarm_taskfile import parse_wall_clock_seconds
from swarm_taskfile import parse_status_value as _parse_status_value
from swarm_taskfile import parse_task_frontmatter as _parse_task_frontmatter
from swarm_taskfile import parse_task_id_from_branch as _parse_task_id_from_branch
from swarm_taskfile import update_task_status_and_notes as _shared_update_task_status_and_notes


SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v2"
SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 = "research_swarm.runtime_run_manifest.v1"
JUDGE_REVIEW_LOG_SCHEMA_VERSION = "research_swarm.judge_review_log.v2"
MOCK_TRANSCRIPT_SCHEMA_VERSION = "research_swarm.mock_transcript.v1"
EXECUTOR_SESSION_SCHEMA_VERSION = "research_swarm.executor_session.v1"
MOCK_PLANNER_SCHEMA_VERSION = "research_swarm.mock_planner.v1"
MOCK_REFEREE_SCHEMA_VERSION = "research_swarm.mock_referee.v1"
REFEREE_REPORT_SCHEMA_VERSION = "research_swarm.referee_report.v1"
REFEREE_CALIBRATION_SCHEMA_VERSION = "research_swarm.referee_calibration.v1"
PLAN_APPROVAL_PENDING_PATH = Path(".swarm/plan_approval_pending.json")
PREREG_AMENDMENT_SCHEMA_VERSION = "research_swarm.prereg_amendment.v1"

EXECUTOR_LOG_MAX_BYTES = 128 * 1024
EXECUTOR_LOG_SEGMENT_BYTES = 64 * 1024
EXECUTOR_SESSION_SEGMENT_BYTES = 16 * 1024

DEFAULT_REVIEW_MIN_SEPARATION_SECONDS = 60
DEFAULT_REPLAN_FAILURE_THRESHOLD = 2
DEFAULT_REPAIR_MAX_ATTEMPTS = 2
DEFAULT_MAX_READY_FOR_REVIEW = 4
REFEREE_SAMPLE_SIZE = 3
REFEREE_REPORT_DIR = Path("reports/status/referee_reports")
REFEREE_CALIBRATION_PATH = Path("reports/status/referee_calibration.json")
REFEREE_VERDICTS = {"supported", "not_supported", "cannot_verify"}
REFEREE_WAIVER_EMITTER = "swarm.py referee-waiver"

EXECUTOR_FORBIDDEN_CONTROL_PLANE_PATHS = (
    "reports/status/swarm_runs/",
    "reports/status/referee_reports/",
    "reports/status/referee_calibration.json",
    "reports/status/referee_calibration_runs/",
    "reports/status/events/",
)

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
    replan_failure_threshold: int
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


@dataclasses.dataclass(frozen=True)
class PlannerOutcome:
    returncode: int
    stdout: str
    proposals: list[dict[str, object]]


@dataclasses.dataclass(frozen=True)
class RefereeOutcome:
    returncode: int
    stdout: str
    referee_family: str
    payload: dict[str, object]


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


def _planner_trigger_id(context: dict[str, object]) -> str:
    value = context.get("trigger_id")
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("planner_trigger_id_invalid")
    return value


def _invoke_planner(
    *,
    mode: str,
    context: dict[str, object],
    repo: Path,
    args: argparse.Namespace,
) -> PlannerOutcome:
    """Invoke a configured Planner backend without granting it filesystem writes."""
    backend = getattr(args, "planner_backend", "mock")
    trigger_id = _planner_trigger_id(context)
    if backend == "mock":
        relpath = f".orchestrator/mock_planner/{mode}_{trigger_id}.json"
        path = repo / relpath
        if not path.is_file():
            return PlannerOutcome(
                returncode=1,
                stdout=f"mock_planner_missing:{relpath}",
                proposals=[],
            )
        try:
            payload = json.loads(_read_text(path))
            if not isinstance(payload, dict):
                raise ValueError("mock_planner_not_object")
            if payload.get("schema_version") != MOCK_PLANNER_SCHEMA_VERSION:
                raise ValueError(
                    f"mock_planner_schema_invalid:{payload.get('schema_version')}"
                )
            proposals = payload.get("proposals")
            returncode = payload.get("returncode")
            stdout = payload.get("stdout", "")
            if not isinstance(proposals, list) or not all(
                isinstance(proposal, dict) for proposal in proposals
            ):
                raise ValueError("mock_planner_proposals_invalid")
            if not isinstance(returncode, int) or isinstance(returncode, bool):
                raise ValueError("mock_planner_returncode_invalid")
            if not isinstance(stdout, str):
                raise ValueError("mock_planner_stdout_invalid")
            return PlannerOutcome(
                returncode=returncode,
                stdout=stdout,
                proposals=[dict(proposal) for proposal in proposals],
            )
        except Exception as exc:
            detail = str(exc).replace("\n", " ").strip()
            return PlannerOutcome(
                returncode=1,
                stdout=f"mock_planner_error:{type(exc).__name__}:{detail}",
                proposals=[],
            )

    if backend == "claude":
        argv = _claude_planner_argv(repo)
        prompt = _render_planner_prompt(mode=mode, context=context)
        started = time.perf_counter()
        planner_env = {**_gate_environment(), **_planner_passthrough_env()}
        try:
            cp = subprocess.run(
                argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                env=planner_env,
                timeout=int(getattr(args, "planner_timeout_seconds", 0) or 900),
                input=prompt,
            )
        except subprocess.TimeoutExpired:
            return PlannerOutcome(returncode=1, stdout="planner_timeout", proposals=[])
        wall_clock = round(time.perf_counter() - started, 3)
        stdout = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
        if cp.returncode != 0:
            return PlannerOutcome(
                returncode=cp.returncode,
                stdout=f"planner_cli_failed({wall_clock}s):{stdout[-2000:]}",
                proposals=[],
            )
        proposals = _extract_planner_proposals(cp.stdout or "")
        if proposals is None:
            return PlannerOutcome(
                returncode=1,
                stdout=f"planner_output_unparseable({wall_clock}s):{stdout[-2000:]}",
                proposals=[],
            )
        return PlannerOutcome(returncode=0, stdout=stdout[-4000:], proposals=proposals)

    raise ValueError(f"unsupported_planner_backend:{backend}")


def _claude_planner_argv(repo: Path) -> list[str]:
    """Read-only Claude planner profile (§4.3): headless print mode with the
    toolset restricted to Read/Glob/Grep — the model can inspect the repo but
    every write happens through the kernel's bounded proposal application."""
    try:
        framework = json.loads(_read_text(repo / "contracts" / "framework.json"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("planner_backend_unavailable") from exc
    executors = framework.get("executors") if isinstance(framework, dict) else None
    config = executors.get("planner") if isinstance(executors, dict) else None
    if not isinstance(config, dict) or config.get("backend") != "claude":
        raise RuntimeError("planner_backend_unavailable")
    executable = config.get("command", "claude")
    model = config.get("model")
    if (
        not isinstance(executable, str)
        or not executable.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise RuntimeError("planner_backend_unavailable")
    if _which_or_none(executable.strip()) is None:
        raise RuntimeError(f"planner_backend_unavailable:missing_cli:{executable.strip()}")
    return [
        executable.strip(),
        "-p",
        "--model",
        model.strip(),
        # --tools RESTRICTS the toolset (read-only); --allowedTools alone only
        # suppresses permission prompts and would leave Write/Bash reachable
        # via project settings (M2 review C1). Defense-in-depth only: the
        # authoritative boundary is the kernel's bounded proposal application,
        # which never trusts the planner to write anything itself.
        "--tools",
        "Read,Glob,Grep",
        "--strict-mcp-config",
        "--mcp-config",
        "{}",
        "--output-format",
        "text",
    ]


def _planner_passthrough_env() -> dict[str, str]:
    """The Claude CLI needs its auth/config channels and a HOME; nothing else
    from the caller's environment leaks into the planner."""
    passthrough: dict[str, str] = {}
    for key in (
        "HOME",
        "USER",
        "LOGNAME",  # keychain-backed CLI auth resolves the user from these
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "XDG_CONFIG_HOME",
    ):
        value = os.environ.get(key)
        if value:
            passthrough[key] = value
    return passthrough


def _render_planner_prompt(*, mode: str, context: dict[str, object]) -> str:
    lines = [
        "You are the PLANNER of a repo-native research swarm (read-only profile).",
        "You may inspect the repository with your tools, but you MUST NOT edit",
        "anything: every change you want happens exclusively through the",
        "proposals JSON you emit, which the kernel lint-checks and applies to",
        ".orchestrator/backlog/ and workstreams.md only.",
        "",
        f"MODE: {mode}",
        "",
        "The T035 rule: no task may combine discovery and construction — if",
        "reconnaissance shows unclear ground truth, propose a decomposition,",
        "never a bigger task. Every task you propose must satisfy the v2 task",
        "schema (see .orchestrator/templates/task_v2.md) and the task-lint",
        "gate. Blocking with a precise question outperforms guessing.",
        "",
        "v2 schema essentials the lint enforces:",
        "- task_schema: research_swarm.task.v2 (required marker)",
        "- task_kind ∈ etl|analysis|validation|writing|lit_review|model|proof|bridge|ops|integrity_audit|repair",
        "- complexity_tier ∈ S|M|L; L requires checkpoint_contract: progress_file",
        "  (S/M may omit checkpoint_contract; it defaults to none)",
        "- recon_required: true for M/L; an explicit false REQUIRES a non-empty",
        "  recon_waiver: <reason> frontmatter field",
        "- budgets with max_wall_clock (4h/90m forms), max_tokens, max_cost_usd",
        "  (inline {a: b} or an indented block both parse)",
        "- success_criteria: list of {id, statement, verification}; unique ids",
        "- inputs: list of {manifest|path, sha256}; validation tasks need a",
        "  comparison_basis: true input disjoint from constructed_by's inputs",
        "- gates must not end in a quote character; no curl/wget/http outside",
        "  network workstreams",
        "",
        "CONTEXT:",
        json.dumps(context, indent=2, sort_keys=True, default=str),
        "",
        "Reply with your reasoning, then END your reply with exactly one",
        "fenced JSON block:",
        "```json",
        '{"proposals": [{"action": "create_task", "path": ".orchestrator/backlog/T###_slug.md", "content": "..."}',
        '            | {"action": "update_workstreams", "content": "..."}',
        '            | {"action": "split_task", "task_id": "T###", "into": [{"path": "...", "content": "..."}]}',
        '            | {"action": "triage_confirm", "task_id": "T###", "note": "..."}]}',
        "```",
    ]
    return "\n".join(lines)


def _extract_planner_proposals(stdout: str) -> list[dict[str, object]] | None:
    """Parse the LAST fenced json block; malformed output proposes nothing."""
    blocks = re.findall(r"```json\s*(.*?)```", stdout, flags=re.DOTALL)
    if not blocks:
        return None
    try:
        payload = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or not all(
        isinstance(item, dict) for item in proposals
    ):
        return None
    return [dict(item) for item in proposals]


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
    if normalized == "mock":
        return "mock"
    return normalized


def _load_referee_rubric(repo: Path, task_kind: str, *, manuscript: bool = False) -> tuple[dict[str, object], Path]:
    rubric_name = "manuscript" if manuscript else ("proof" if task_kind == "proof" else task_kind)
    path = repo / "contracts" / "rubrics" / f"{rubric_name}.yaml"
    try:
        payload = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"referee_rubric_unreadable:{rubric_name}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "research_swarm.rubric.v1":
        raise ValueError(f"referee_rubric_schema_invalid:{rubric_name}")
    expected_kind = "proof_review" if task_kind == "proof" and not manuscript else rubric_name
    if payload.get("task_kind") != expected_kind:
        raise ValueError(f"referee_rubric_task_kind_invalid:{rubric_name}:{payload.get('task_kind')}")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"referee_rubric_checks_invalid:{rubric_name}")
    seen: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError(f"referee_rubric_check_invalid:{rubric_name}")
        check_id = check.get("id")
        if not isinstance(check_id, str) or not check_id.strip() or check_id in seen:
            raise ValueError(f"referee_rubric_check_id_invalid:{rubric_name}:{check_id}")
        seen.add(check_id)
        if check.get("severity") not in {"major", "minor"}:
            raise ValueError(f"referee_rubric_severity_invalid:{rubric_name}:{check_id}")
        if not isinstance(check.get("prompt"), str) or not check["prompt"].strip():
            raise ValueError(f"referee_rubric_prompt_invalid:{rubric_name}:{check_id}")
        if not isinstance(check.get("evidence_required"), bool):
            raise ValueError(f"referee_rubric_evidence_invalid:{rubric_name}:{check_id}")
    return payload, path


def _claim_ledger(repo: Path) -> list[dict[str, object]]:
    path = repo / "contracts" / "claims.yaml"
    try:
        payload = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("referee_claim_ledger_unreadable") from exc
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        raise ValueError("referee_claim_ledger_invalid")
    return [dict(item) for item in claims if isinstance(item, dict)]


def _safe_claim_ledger(repo: Path) -> tuple[list[dict[str, object]], str | None]:
    try:
        return _claim_ledger(repo), None
    except ValueError as exc:
        return [], str(exc)


def _task_is_manuscript_surface(task: Task) -> bool:
    return task.task_kind == "writing" or any(
        _path_matches_prefix(output, "reports/paper/") for output in task.outputs
    )


def _referee_task_in_scope(task: Task) -> bool:
    frontmatter = _task_frontmatter(task)
    complexity = frontmatter.get("complexity_tier")
    kind = task.task_kind
    if kind == "repair":
        source_kind = frontmatter.get("repair_source_task_kind")
        source_complexity = frontmatter.get("repair_source_complexity_tier")
        if isinstance(source_kind, str):
            kind = source_kind
        if isinstance(source_complexity, str):
            complexity = source_complexity
    tiered_scientific = complexity in {"M", "L"} and kind in {
        "analysis",
        "model",
        "bridge",
        "writing",
    }
    return bool(
        tiered_scientific
        or task.workstream in {"W6", "W7"}
        or any(_path_matches_prefix(output, "reports/paper/") for output in task.outputs)
    )


def _repair_is_referee_reviewable(task: Task) -> bool:
    if task.task_kind != "repair":
        return False
    frontmatter = _task_frontmatter(task)
    return (
        isinstance(frontmatter.get("repair_source_task"), str)
        and frontmatter.get("repair_source_task_kind")
        in {"analysis", "model", "bridge", "writing", "proof", "validation", "etl", "lit_review"}
    )


def _claim_is_bound_to_task(repo: Path, claim: dict[str, object], task: Task) -> bool:
    owner = next(
        (
            claim.get(key)
            for key in ("task_id", "registered_by_task", "source_task_id")
            if isinstance(claim.get(key), str)
        ),
        None,
    )
    frontmatter = _task_frontmatter(task)
    source_task = frontmatter.get("repair_source_task") if task.task_kind == "repair" else None
    if owner in {task.task_id, source_task}:
        return True
    claim_ids = frontmatter.get("claim_ids")
    if isinstance(claim_ids, list) and claim.get("claim_id") in claim_ids:
        return True
    artifacts = claim.get("supporting_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            path = artifact.get("path") if isinstance(artifact, dict) else None
            if not isinstance(path, str):
                continue
            if any(
                _path_matches_prefix(path, output) or _path_matches_prefix(output, path)
                for output in task.outputs
            ):
                return True
    if not _task_is_manuscript_surface(task):
        return False
    manuscript_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in _declared_output_files(repo, task)
        if path.suffix.lower() in _MANUSCRIPT_SUFFIXES
    )
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


def _task_scoped_claims(repo: Path, task: Task) -> tuple[list[dict[str, object]], str | None]:
    claims, diagnostic = _safe_claim_ledger(repo)
    if diagnostic is not None:
        return [], diagnostic
    return [claim for claim in claims if _claim_is_bound_to_task(repo, claim, task)], None


def _referee_claim_context(claims: list[dict[str, object]]) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for claim in claims:
        item = dict(claim)
        artifacts = claim.get("supporting_artifacts")
        if isinstance(artifacts, list):
            item["supporting_artifacts"] = [
                {key: value for key, value in artifact.items() if key != "sha256"}
                for artifact in artifacts
                if isinstance(artifact, dict)
            ]
        sanitized.append(item)
    return sanitized


def _redact_sample_hashes(value: object, target_hashes: set[str]) -> object:
    if isinstance(value, dict):
        return {key: _redact_sample_hashes(item, target_hashes) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sample_hashes(item, target_hashes) for item in value]
    if isinstance(value, str):
        redacted = value
        for digest in target_hashes:
            if digest:
                redacted = redacted.replace(digest, "<kernel-redacted-sampled-sha256>")
        return redacted
    return value


def _artifact_quote_challenge(
    *,
    raw: bytes,
    seed: str,
    task_id: str,
    claim_id: str,
    path: str,
) -> tuple[int, str]:
    """Choose a deterministic NON-BLANK line without exposing its contents. A
    blank challenge is echo-satisfiable (a referee returns "" without reading),
    so only lines with non-whitespace content are eligible; challenge_line 0
    signals no challengeable line exists (binary/blank artifact) and the caller
    fails closed rather than accepting an empty span."""
    lines = raw.decode("utf-8", errors="replace").splitlines()
    candidates = [(number, text) for number, text in enumerate(lines, start=1) if text.strip()]
    if not candidates:
        return 0, ""
    selector = hashlib.sha256(
        f"{seed}\0{task_id}\0{claim_id}\0{path}\0quoted-span".encode("utf-8")
    ).digest()
    index = int.from_bytes(selector[:8], "big") % len(candidates)
    return candidates[index]


def _public_sampled_artifact(item: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in item.items()
        if key != "expected_quoted_span"
    }


def _kernel_sampled_artifacts(
    repo: Path,
    task_id: str,
    *,
    task: Task | None = None,
) -> list[dict[str, object]]:
    seed_path = repo / "contracts" / "rubrics" / "sampling_seed.txt"
    try:
        seed = seed_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("referee_sampling_seed_missing") from exc
    if not seed:
        raise ValueError("referee_sampling_seed_empty")
    if task is None:
        try:
            contract = load_framework_contract(repo)
            tasks, quarantined = load_tasks_quarantined(contract)
            task = _resolve_runtime_task(tasks, quarantined, task_id)
        except (OSError, ValueError):
            return []
    scoped_claims, _ = _task_scoped_claims(repo, task)
    candidates: list[tuple[str, dict[str, object]]] = []
    for claim in scoped_claims:
        claim_id = claim.get("claim_id")
        artifacts = claim.get("supporting_artifacts")
        if not isinstance(claim_id, str) or not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            digest = artifact.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                continue
            artifact_path = (repo / path).resolve()
            try:
                artifact_path.relative_to(repo.resolve())
            except ValueError:
                raw = b""
                disk_digest = ""
            else:
                raw = artifact_path.read_bytes() if artifact_path.is_file() else b""
                disk_digest = hashlib.sha256(raw).hexdigest() if artifact_path.is_file() else ""
            ledger_digest = digest.lower()
            challenge_line, expected_quoted_span = _artifact_quote_challenge(
                raw=raw,
                seed=seed,
                task_id=task_id,
                claim_id=claim_id,
                path=path,
            )
            item: dict[str, object] = {
                "claim_id": claim_id,
                "path": path,
                "sha256": disk_digest,
                "ledger_sha256": ledger_digest,
                "tampered": disk_digest != ledger_digest,
                "challenge_line": challenge_line,
                "expected_quoted_span": expected_quoted_span,
            }
            score = hashlib.sha256(
                f"{seed}\0{task_id}\0{claim_id}\0{path}".encode("utf-8")
            ).hexdigest()
            candidates.append((score, item))
    candidates.sort(key=lambda item: (item[0], str(item[1]["path"])))
    return [item for _, item in candidates[:REFEREE_SAMPLE_SIZE]]


_ASSERTION_LEXICON = re.compile(
    r"\b(?:caus\w*|because|therefore|due to|driv\w*|lead\w*|result\w* in|"
    r"effect\w*|impact\w*|increase\w*|decrease\w*|higher|lower|greater|less|"
    r"more than|fewer|improv\w*|wors\w*|outperform\w*|associated|correlat\w*)\b",
    flags=re.IGNORECASE,
)
_MANUSCRIPT_SUFFIXES = {".md", ".qmd", ".txt", ".tex"}


def _declared_output_files(repo: Path, task: Task) -> list[Path]:
    files: list[Path] = []
    for raw in task.outputs:
        path = (repo / raw).resolve()
        try:
            path.relative_to(repo.resolve())
        except ValueError:
            continue
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in sorted(path.rglob("*")) if candidate.is_file())
    return sorted(set(files))


def _assertion_candidates(repo: Path, task: Task) -> list[dict[str, object]]:
    claims, _ = _safe_claim_ledger(repo)
    candidates: list[dict[str, object]] = []
    for path in _declared_output_files(repo, task):
        if path.suffix.lower() not in _MANUSCRIPT_SUFFIXES:
            continue
        text = _read_text(path)
        # Sentence boundaries are deliberately an over-inclusive deterministic
        # floor. The referee must still enumerate qualitative assertions that
        # this lexical pre-filter misses.
        for match in re.finditer(r"(?ms)(?P<sentence>[^.!?\n][^.!?]*[.!?]|[^\n]+$)", text):
            sentence = re.sub(r"\s+", " ", match.group("sentence")).strip()
            if not sentence or (re.search(r"\d", sentence) is None and _ASSERTION_LEXICON.search(sentence) is None):
                continue
            line = text.count("\n", 0, match.start()) + 1
            literal_claim_ids: list[str] = []
            for claim in claims:
                literals = claim.get("manuscript_numeric_literals")
                if not isinstance(literals, list):
                    continue
                if any(isinstance(value, str) and value in sentence for value in literals):
                    claim_id = claim.get("claim_id")
                    if isinstance(claim_id, str):
                        literal_claim_ids.append(claim_id)
            candidates.append(
                {
                    "check_id": f"ASSERTION-{len(candidates) + 1:03d}",
                    "path": path.relative_to(repo.resolve()).as_posix(),
                    "line": line,
                    "sentence": sentence,
                    "numeric_literal_claim_candidates": sorted(set(literal_claim_ids)),
                    "instruction": (
                        "Map this sentence to the specific claim whose statement and semantic role support it; "
                        "not_supported if unmatched or if a registered value is assigned to the wrong role."
                    ),
                }
            )
    return candidates


def _declared_output_context(repo: Path, task: Task) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for path in _declared_output_files(repo, task):
        raw = path.read_bytes()
        item: dict[str, object] = {
            "path": path.relative_to(repo.resolve()).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        if path.suffix.lower() in _MANUSCRIPT_SUFFIXES or len(raw) <= 64 * 1024:
            item["content"] = raw[:64 * 1024].decode("utf-8", errors="replace")
            item["content_truncated"] = len(raw) > 64 * 1024
        outputs.append(item)
    return outputs


def _referee_diff(repo: Path, base_branch: str, remote: str) -> str:
    base_ref = _resolve_base_ref_for_diff(cwd=repo, base_branch=base_branch, remote=remote)
    if base_ref is None:
        return ""
    cp = _run(
        ["git", "diff", "--no-ext-diff", f"{base_ref}...HEAD", "--"],
        cwd=repo,
        capture=True,
        check=False,
    )
    return (cp.stdout or "")[-512 * 1024 :]


def _latest_referee_run_manifest(contract: FrameworkContract, task_id: str) -> tuple[Path, dict[str, object]]:
    matches = _matching_v2_run_manifest_data(
        _matching_task_jsons(contract.run_manifest_dir, task_id), task_id
    )
    passing = [
        (path, payload)
        for path, payload in matches
        if isinstance(payload.get("result"), dict) and payload["result"].get("status") == "ok"
    ]
    if not passing:
        raise ValueError(f"referee_missing_passing_run_manifest:{task_id}")
    return passing[-1]


def _referee_required_verdicts(
    *,
    task: Task,
    frontmatter: dict[str, object],
    rubrics: list[dict[str, object]],
    assertions: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    required: dict[str, dict[str, object]] = {}
    fields = TaskV2Fields(frontmatter)
    for criterion in fields.success_criteria or ():
        criterion_id = criterion.get("id")
        if isinstance(criterion_id, str):
            required[criterion_id] = {
                "identifier_key": "success_criterion_id",
                "severity": "major",
                "evidence_required": True,
                "prompt": criterion.get("statement"),
            }
    for rubric in rubrics:
        for check in rubric.get("checks", []):
            if isinstance(check, dict) and isinstance(check.get("id"), str):
                required[check["id"]] = {
                    "identifier_key": "check_id",
                    "severity": check.get("severity"),
                    "evidence_required": check.get("evidence_required"),
                    "prompt": check.get("prompt"),
                }
    for assertion in assertions:
        check_id = assertion.get("check_id")
        if isinstance(check_id, str):
            required[check_id] = {
                "identifier_key": "check_id",
                "severity": "major",
                "evidence_required": True,
                "prompt": assertion.get("instruction"),
            }
    return required


def _render_referee_prompt(repo: Path, context: dict[str, object]) -> str:
    prompt_path = repo / "docs" / "prompts" / "referee.md"
    try:
        base_prompt = _read_text(prompt_path).strip()
    except OSError as exc:
        raise ValueError("referee_prompt_unreadable") from exc
    return "\n".join(
        [
            base_prompt,
            "",
            "CONTEXT:",
            json.dumps(context, indent=2, sort_keys=True, default=str),
            "",
            "Return no edits. End with exactly one fenced JSON object containing:",
            "```json",
            '{"referee_family":"...","verdicts":[{"success_criterion_id":"SC1","verdict":"supported|not_supported|cannot_verify","evidence_pointer":"path:line","note":"..."},{"check_id":"...","verdict":"...","evidence_pointer":"...","note":"..."}],"opened_artifacts":[{"path":"...","quoted_span":"exact contents of the challenged line"}],"overall":"supported|not_supported|cannot_verify"}',
            "```",
        ]
    )


def _extract_referee_payload(stdout: str) -> dict[str, object] | None:
    blocks = re.findall(r"```json\s*(.*?)```", stdout, flags=re.DOTALL)
    if not blocks:
        return None
    try:
        payload = json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _claude_referee_argv(repo: Path, referee_family: str = "claude") -> list[str]:
    try:
        framework = json.loads(_read_text(repo / "contracts" / "framework.json"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("referee_backend_unavailable") from exc
    executors = framework.get("executors") if isinstance(framework, dict) else None
    panel = executors.get("referee_panel") if isinstance(executors, dict) else None
    configs = panel if isinstance(panel, list) else []
    config = next(
        (
            item
            for item in configs
            if isinstance(item, dict)
            and item.get("backend") == "claude"
            and item.get("family", "claude") == referee_family
        ),
        None,
    )
    if not isinstance(config, dict):
        raise RuntimeError("referee_backend_unavailable")
    executable = config.get("command", "claude")
    model = config.get("model")
    if not isinstance(executable, str) or not executable.strip() or not isinstance(model, str) or not model.strip():
        raise RuntimeError("referee_backend_unavailable")
    # NB: CLI-presence is checked at INVOCATION (_invoke_referee), not here —
    # argv construction is pure so it (and the read-only-profile assertion) is
    # testable on a runner without the referee CLI installed.
    return [
        executable.strip(),
        "-p",
        "--model",
        model.strip(),
        "--tools",
        "Read,Glob,Grep",
        "--strict-mcp-config",
        "--mcp-config",
        "{}",
        "--output-format",
        "text",
    ]


def _invoke_referee(
    *,
    context: dict[str, object],
    repo: Path,
    task_id: str,
    backend: str,
    referee_family: str | None,
    timeout_seconds: int,
) -> RefereeOutcome:
    if backend == "mock":
        path = repo / ".orchestrator" / "mock_referee" / f"{task_id}.json"
        if not path.is_file():
            return RefereeOutcome(1, f"mock_referee_missing:{path.relative_to(repo)}", referee_family or "", {})
        try:
            payload = json.loads(_read_text(path))
            if not isinstance(payload, dict) or payload.get("schema_version") != MOCK_REFEREE_SCHEMA_VERSION:
                raise ValueError(f"mock_referee_schema_invalid:{payload.get('schema_version') if isinstance(payload, dict) else None}")
            family = payload.get("referee_family")
            if not isinstance(family, str) or not family.strip():
                raise ValueError("mock_referee_family_invalid")
            if referee_family is not None and family != referee_family:
                raise ValueError(f"mock_referee_family_mismatch:{family}!={referee_family}")
            returncode = payload.get("returncode", 0)
            if not isinstance(returncode, int) or isinstance(returncode, bool):
                raise ValueError("mock_referee_returncode_invalid")
            report_payload = {
                key: payload[key]
                for key in ("verdicts", "opened_artifacts", "overall")
                if key in payload
            }
            return RefereeOutcome(
                returncode,
                str(payload.get("stdout", "mock referee complete")),
                family,
                report_payload,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return RefereeOutcome(1, f"mock_referee_error:{type(exc).__name__}:{exc}", referee_family or "", {})
    if backend == "claude":
        family = referee_family or "claude"
        argv = _claude_referee_argv(repo, family)
        # Fail-closed at the invocation boundary if the referee CLI is absent
        # (§4.3: unavailable → hard-block, never a silent waive).
        if not argv or _which_or_none(argv[0]) is None:
            return RefereeOutcome(
                1, f"referee_backend_unavailable:missing_cli:{argv[0] if argv else 'claude'}", family, {}
            )
        prompt = _render_referee_prompt(repo, context)
        try:
            cp = subprocess.run(
                argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                env={**_gate_environment(), **_planner_passthrough_env()},
                timeout=timeout_seconds,
                input=prompt,
            )
        except subprocess.TimeoutExpired:
            return RefereeOutcome(1, "referee_timeout", family, {})
        stdout = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
        if cp.returncode != 0:
            return RefereeOutcome(cp.returncode, f"referee_cli_failed:{stdout[-2000:]}", family, {})
        payload = _extract_referee_payload(cp.stdout or "")
        if payload is None:
            return RefereeOutcome(1, f"referee_output_unparseable:{stdout[-2000:]}", family, {})
        returned_family = payload.get("referee_family")
        if returned_family != family:
            return RefereeOutcome(1, f"referee_family_mismatch:{returned_family}!={family}", family, payload)
        return RefereeOutcome(0, stdout[-4000:], family, payload)
    raise ValueError(f"unsupported_referee_backend:{backend}")


def _referee_calibrated(repo: Path) -> bool:
    path = repo / REFEREE_CALIBRATION_PATH
    if not path.is_file():
        return False
    return not calibration_report_failures(repo=repo, report_path=path)


def _referee_family_calibrated(repo: Path, family: str) -> bool:
    path = repo / REFEREE_CALIBRATION_PATH
    return path.is_file() and not calibration_report_failures(
        repo=repo,
        report_path=path,
        required_family=family,
    )


def _normalize_referee_report(
    *,
    payload: dict[str, object],
    required: dict[str, dict[str, object]],
    sampled_artifacts: list[dict[str, object]],
    sample_required: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, str]], str, list[str]]:
    failures: list[str] = []
    raw_verdicts = payload.get("verdicts")
    raw_opened = payload.get("opened_artifacts")
    if not isinstance(raw_verdicts, list):
        raw_verdicts = []
        failures.append("referee_verdicts_invalid")
    if not isinstance(raw_opened, list):
        raw_opened = []
        failures.append("referee_opened_artifacts_invalid")
    verdicts: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_verdicts):
        if not isinstance(item, dict):
            failures.append(f"referee_verdict_invalid:{index}")
            continue
        present = [key for key in ("success_criterion_id", "check_id") if isinstance(item.get(key), str)]
        if len(present) != 1:
            failures.append(f"referee_verdict_identifier_invalid:{index}")
            continue
        key = present[0]
        identifier = str(item[key]).strip()
        if identifier not in required or required[identifier]["identifier_key"] != key:
            failures.append(f"referee_verdict_unexpected:{identifier}")
            continue
        if identifier in seen:
            failures.append(f"referee_verdict_duplicate:{identifier}")
            continue
        seen.add(identifier)
        verdict = item.get("verdict")
        if verdict not in REFEREE_VERDICTS:
            failures.append(f"referee_verdict_value_invalid:{identifier}:{verdict}")
            continue
        pointer = item.get("evidence_pointer")
        note = item.get("note")
        if required[identifier]["evidence_required"] and (
            not isinstance(pointer, str) or not pointer.strip()
        ):
            failures.append(f"referee_evidence_pointer_missing:{identifier}")
        if not isinstance(note, str) or not note.strip():
            failures.append(f"referee_note_missing:{identifier}")
        verdicts.append(
            {
                key: identifier,
                "verdict": verdict,
                "severity": required[identifier]["severity"],
                "evidence_pointer": pointer if isinstance(pointer, str) else "",
                "note": note if isinstance(note, str) else "",
            }
        )
    for identifier in sorted(set(required) - seen):
        failures.append(f"referee_verdict_missing:{identifier}")

    opened_by_path: dict[str, dict[str, str]] = {}
    for index, item in enumerate(raw_opened):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("quoted_span"), str)
        ):
            failures.append(f"referee_opened_artifact_invalid:{index}")
            continue
        path = str(item["path"])
        if path in opened_by_path:
            failures.append(f"referee_opened_artifact_duplicate:{path}")
            continue
        opened_by_path[path] = {"path": path, "quoted_span": str(item["quoted_span"])}
    if sample_required and not sampled_artifacts:
        failures.append("referee_sample_empty_for_claims")
    for item in sampled_artifacts:
        if item.get("tampered") is True:
            failures.append(f"referee_sampled_artifact_tampered:{item.get('path')}")
        # A non-challengeable sample (blank/binary artifact, challenge_line 0)
        # cannot prove opening — fail closed rather than accept an empty span.
        if not item.get("challenge_line"):
            failures.append(f"referee_sample_not_challengeable:{item.get('path')}")
    missing_sample: list[str] = []
    opened: list[dict[str, str]] = []
    for sampled in sampled_artifacts:
        path = str(sampled["path"])
        supplied = opened_by_path.get(path)
        if supplied is None:
            missing_sample.append(path)
            continue
        if supplied["quoted_span"] != sampled.get("expected_quoted_span"):
            failures.append(f"referee_opened_artifact_quote_mismatch:{path}")
        opened.append(
            {
                "path": path,
                "sha256": str(sampled.get("sha256", "")).lower(),
                "quoted_span": supplied["quoted_span"],
            }
        )
    if missing_sample:
        failures.append("referee_did_not_open_sampled:" + ",".join(sorted(missing_sample)))

    major = [item for item in verdicts if item.get("severity") == "major"]
    if any(item.get("verdict") == "cannot_verify" for item in verdicts):
        overall = "cannot_verify"
    elif any(item.get("verdict") == "not_supported" for item in major):
        overall = "not_supported"
    else:
        overall = "supported"
    return verdicts, opened, overall, failures


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
    replan = raw.get("replan") if isinstance(raw.get("replan"), dict) else {}
    try:
        replan_failure_threshold = int(replan.get("failure_threshold"))
    except (TypeError, ValueError):
        replan_failure_threshold = DEFAULT_REPLAN_FAILURE_THRESHOLD
    replan_failure_threshold = max(1, replan_failure_threshold)

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
        if isinstance(budget_raw, (int, float))
        and not isinstance(budget_raw, bool)
        and math.isfinite(budget_raw)
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
        replan_failure_threshold=replan_failure_threshold,
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


def _load_v1_task_exemptions(repo: Path) -> dict[str, dict[str, object]]:
    exemptions_path = repo / "contracts" / "historical_exemptions.json"
    exemptions: dict[str, dict[str, object]] = {}
    try:
        payload = json.loads(_read_text(exemptions_path))
    except (OSError, json.JSONDecodeError):
        return exemptions
    if not isinstance(payload, dict):
        return exemptions
    for entry in payload.get("tasks", []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            exemptions[entry["path"]] = entry
    return exemptions


def _task_frontmatter(task: Task) -> dict[str, object]:
    frontmatter = _parse_task_frontmatter(_read_text(task.path))
    return frontmatter if isinstance(frontmatter, dict) else {}


def _task_triage_reasons(task: Task, tasks: dict[str, Task]) -> list[str]:
    frontmatter = _task_frontmatter(task)
    fields = TaskV2Fields(frontmatter)
    if fields.task_schema != TASK_SCHEMA_VERSION:
        return []

    reasons: list[str] = []
    if fields.complexity_tier == "L":
        reasons.append("complexity_l")
    if fields.task_kind == "etl" and len(fields.inputs or ()) > 1:
        reasons.append("etl_multi_input")
    if len(task.outputs) > 2:
        reasons.append("more_than_two_outputs")
    backlog_peers = sorted(
        candidate.task_id
        for candidate in tasks.values()
        if candidate.state == "backlog"
        and candidate.workstream == task.workstream
        and candidate.task_kind == task.task_kind
    )
    if backlog_peers and backlog_peers[0] == task.task_id:
        reasons.append("first_of_kind_in_workstream")
    return reasons


def _task_has_planner_triage(task: Task) -> bool:
    triage = TaskV2Fields(_task_frontmatter(task)).triage
    if triage is None:
        return False
    status = triage.get("status")
    by = triage.get("by")
    note = triage.get("note")
    # Only CONFIRMED satisfies claimability: a task marked `split` is a
    # decomposition that must be APPLIED (parent removed, children created)
    # — a lingering split-labelled task is unresolved, never claimable (C8).
    return (
        status == "confirmed"
        and by == "planner"
        and isinstance(note, str)
        and bool(note.strip())
    )


def _required_active_lock(task: Task, mode: str) -> str | None:
    """Return the most advanced prereg phase that must gate this task claim."""
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


def _task_registered_claim_types(repo: Path, task: Task) -> set[str]:
    path = repo / "contracts/claims.yaml"
    try:
        payload = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError):
        return set()
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        return set()
    frontmatter = _task_frontmatter(task)
    declared_claim_ids = set(
        value
        for value in frontmatter.get("claim_ids", [])
        if isinstance(value, str)
    ) if isinstance(frontmatter.get("claim_ids"), list) else set()
    declared_claim_types = set(
        value
        for value in frontmatter.get("claim_types", [])
        if isinstance(value, str)
    ) if isinstance(frontmatter.get("claim_types"), list) else set()
    writes_ledger = any(
        _normalize_repo_relative_path(output) == "contracts/claims.yaml"
        for output in task.outputs
    )
    claim_types: set[str] = set(declared_claim_types)
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        owner = next(
            (
                claim.get(key)
                for key in ("task_id", "registered_by_task", "source_task_id")
                if isinstance(claim.get(key), str) and claim.get(key).strip()
            ),
            None,
        )
        claim_id = claim.get("claim_id")
        if owner == task.task_id or claim_id in declared_claim_ids or writes_ledger:
            claim_type = claim.get("type")
            if isinstance(claim_type, str):
                claim_types.add(claim_type)
    return claim_types


_PREREG_PHASE_ORDER = {"2a": 0, "2b": 1, "lock_a": 2, "lock_b": 3}


def _effective_required_active_locks(
    task: Task, contract: FrameworkContract
) -> tuple[str, ...]:
    """ALL prereg phases that must be ACTIVE before this task may claim (§6.1).
    Claim-type requirements are ADDITIVE to the base phase, never a same-rank
    substitution — a hybrid `analysis` task (base 2b) registering a
    `counterfactual` claim requires BOTH 2b AND lock_b, so lock_b can never be
    collapsed onto 2b by a rank tie. A registered `computational` claim requires
    the experiment lock (Lock A) regardless of task_kind, so a modeling `proof`
    task cannot register one pre-experiment-lock."""
    mode = contract.project_mode or "empirical"
    required: list[str] = []
    frontmatter = _task_frontmatter(task)
    explicit_required = frontmatter.get("required_prereg_locks")
    if isinstance(explicit_required, list):
        required.extend(
            phase
            for phase in explicit_required
            if isinstance(phase, str) and phase in _PREREG_PHASE_ORDER
        )
    base = _required_active_lock(task, mode)
    if base is not None:
        required.append(base)
    claim_types = _task_registered_claim_types(contract.repo_root, task)
    if mode == "hybrid" and "counterfactual" in claim_types and "lock_b" not in required:
        required.append("lock_b")
    if mode in {"modeling", "hybrid"} and "computational" in claim_types and "lock_a" not in required:
        required.append("lock_a")
    return tuple(sorted(set(required), key=lambda phase: _PREREG_PHASE_ORDER.get(phase, 99)))


def _effective_required_active_lock(
    task: Task, contract: FrameworkContract
) -> str | None:
    """Back-compat single-phase view: the FIRST (lowest-order) required-active
    phase, or None. Callers that must enforce every requirement use
    `_effective_required_active_locks`."""
    phases = _effective_required_active_locks(task, contract)
    return phases[0] if phases else None


def _prereg_phase_is_active(repo: Path, phase: str) -> bool:
    lock, error = load_prereg_lock(repo / PREREG_PHASE_FILES[phase], expected_phase=phase)
    return error is None and lock is not None and lock.get("active") is True


def _plan_approval_pending(repo: Path) -> bool:
    return (repo / PLAN_APPROVAL_PENDING_PATH).is_file()


def ready_backlog_tasks(tasks: dict[str, Task], claimed_ids: set[str], contract: FrameworkContract) -> list[Task]:
    ready: list[Task] = []
    v1_exemptions = _load_v1_task_exemptions(contract.repo_root)
    # The plan-approval hold is enforced HERE — the single funnel every
    # dispatch path (cmd_tick, supervise _step_tick) shares — so no v2 task
    # can be selected while a plan awaits human approval, regardless of the
    # entrypoint (§4.2 mandatory gate).
    plan_pending = _plan_approval_pending(contract.repo_root)

    diagnostics = lint_task_files(
        [task.path for task in tasks.values()],
        repo_root=contract.repo_root,
        network_workstreams=contract.network_workstreams,
        v1_exemptions=v1_exemptions,
    )
    diagnostics_by_task: dict[str, list[dict[str, object]]] = {}
    for diagnostic in diagnostics:
        diagnostics_by_task.setdefault(diagnostic.task, []).append(diagnostic.as_dict())

    for task in tasks.values():
        if task.state != "backlog":
            continue
        if task.role not in set(contract.task_execution_roles):
            continue
        if task.task_id in claimed_ids:
            continue
        inactive_required = [
            phase
            for phase in _effective_required_active_locks(task, contract)
            if not _prereg_phase_is_active(contract.repo_root, phase)
        ]
        if inactive_required:
            _record_swarm_event(
                contract.repo_root,
                {
                    "event": "blocked_on_prereg_lock",
                    "task_id": task.task_id,
                    "required_phase": inactive_required[0],
                    "inactive_required_phases": inactive_required,
                },
            )
            continue
        if plan_pending and TaskV2Fields(_task_frontmatter(task)).task_schema == TASK_SCHEMA_VERSION:
            _record_swarm_event(
                contract.repo_root,
                {
                    "event": "plan_unapproved",
                    "task_id": task.task_id,
                    "pending_path": PLAN_APPROVAL_PENDING_PATH.as_posix(),
                },
            )
            continue
        task_diagnostics = diagnostics_by_task.get(task.task_id, [])
        if task_diagnostics:
            _record_swarm_event(
                contract.repo_root,
                {
                    "event": "task_lint_rejected",
                    "task_id": task.task_id,
                    "task_path": task.path.as_posix(),
                    "diagnostics": task_diagnostics,
                },
            )
            continue
        triage_reasons = _task_triage_reasons(task, tasks)
        if triage_reasons and not _task_has_planner_triage(task):
            _record_swarm_event(
                contract.repo_root,
                {
                    "event": "task_triage_required",
                    "task_id": task.task_id,
                    "task_path": task.path.as_posix(),
                    "reasons": triage_reasons,
                },
            )
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
    fields = TaskV2Fields(_task_frontmatter(task))
    return {
        "repo_root": repo.as_posix(),
        "task_path": task.path.relative_to(repo).as_posix(),
        "task_id": task.task_id,
        "title": task.title,
        "workstream": task.workstream,
        "task_kind": task.task_kind or "",
        "allow_network": "true" if task.allow_network else "false",
        "recon_required": "true" if fields.recon_required is True else "false",
        "allowed_paths": _format_bullets(task.allowed_paths),
        "disallowed_paths": _format_bullets(task.disallowed_paths),
        "outputs": _format_bullets(task.outputs),
        "gates": _format_bullets(task.gates),
        "stop_conditions": _format_bullets(task.stop_conditions),
        "repair_context": repair_context or "",
        "runner_mode": "local_swarm",
        "base_branch": "",
    }


_RECON_PLACEHOLDER_PREFIXES = (
    "- Scope understanding:",
    "- Risks and unknowns:",
    "- Decomposition pressure assessment:",
    "- Proposed bounded approach:",
)


def _reconnaissance_line_count(task_text: str) -> int:
    section = _extract_section(task_text, "Reconnaissance")
    if section is None:
        return 0
    count = 0
    for raw in section.splitlines():
        line = raw.strip()
        if not line or line in {"-", "*"} or line.startswith("<!--"):
            continue
        # a template label with only a trivial placeholder after the colon is
        # not reconnaissance. A substantive note is a phrase — >=2 words and
        # >=8 non-space characters (C7, reworked after the verification pass
        # found single-word placeholders like TBC/pending/unknown slipping
        # through an explicit blocklist).
        matched_prefix = next(
            (prefix for prefix in _RECON_PLACEHOLDER_PREFIXES if line.startswith(prefix[:-1])),
            None,
        )
        if matched_prefix is not None:
            after = line.split(":", 1)[1].strip() if ":" in line else ""
            def _norm_token(token: str) -> str:
                # strip markdown/wrapping punctuation so `TBD`, **pending**,
                # (unknown), _todo_ normalize to the bare placeholder word
                return re.sub(r"[^a-z0-9/]+", "", token.lower())

            words = [w for w in re.split(r"\s+", after) if _norm_token(w)]
            normalized = [_norm_token(w) for w in words]
            placeholder_vocab = {"tbd", "tbc", "todo", "pending", "unknown", "na", "none", "wip", "fixme", "xxx"}
            non_space = len(re.sub(r"[^a-z0-9]", "", after.lower()))
            distinct = {w for w in normalized if w}
            if (
                len(words) < 2
                or non_space < 8
                or len(distinct) < 2  # repeated single token ("pending pending")
                or all(w in placeholder_vocab for w in normalized if w)  # all-placeholder
            ):
                continue
        count += 1
    return count


def _git_current_branch(cwd: Path) -> str:
    cp = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture=True, check=True)
    return (cp.stdout or "").strip()


def _trusted_integration_branch(cwd: Path) -> str:
    """The repository's real default/integration branch, derived from git — NOT
    from a caller-supplied argument (a Worker on a task branch must not be able
    to name its own branch as the 'base' to mint a control-plane waiver)."""
    cp = _run(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        cwd=cwd,
        capture=True,
        check=False,
    )
    value = (cp.stdout or "").strip() if cp.returncode == 0 else ""
    if value.startswith("origin/"):
        value = value[len("origin/"):]
    return value or "main"


def _is_linked_worktree(cwd: Path) -> bool:
    """True inside a linked git worktree (where Workers run) — the main
    checkout's git dir equals its common dir; a linked worktree's does not."""
    git_dir = _run(["git", "rev-parse", "--git-dir"], cwd=cwd, capture=True, check=False)
    common = _run(["git", "rev-parse", "--git-common-dir"], cwd=cwd, capture=True, check=False)
    gd = (git_dir.stdout or "").strip()
    cd = (common.stdout or "").strip()
    if not gd or not cd:
        return False
    return Path(gd).resolve() != Path(cd).resolve()


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


def _executor_control_plane_snapshot(repo: Path) -> dict[str, bytes]:
    """Capture task-visible control-plane bytes immediately around execution."""
    snapshot: dict[str, bytes] = {}
    for raw in EXECUTOR_FORBIDDEN_CONTROL_PLANE_PATHS:
        normalized = _normalize_repo_relative_path(raw)
        path = repo / normalized
        if raw.endswith("/"):
            if not path.is_dir():
                continue
            candidates = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
        else:
            candidates = [path] if path.exists() or path.is_symlink() else []
        for candidate in candidates:
            rel = candidate.relative_to(repo).as_posix()
            try:
                snapshot[rel] = (
                    f"symlink:{os.readlink(candidate)}".encode("utf-8")
                    if candidate.is_symlink()
                    else candidate.read_bytes()
                )
            except OSError:
                snapshot[rel] = b"<unreadable>"
    return snapshot


def _kernel_heartbeat_only_journal_append(
    before: bytes,
    after: bytes,
    allowed_events: list[dict[str, object]],
) -> bool:
    """Permit journal bytes concurrently appended by the lease heartbeat only."""
    if not after.startswith(before):
        return False
    appended = after[len(before):]
    if not appended:
        return True
    for raw_line in appended.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        if event not in allowed_events:
            return False
    return True


def _executor_control_plane_changes(
    *,
    repo: Path,
    before: dict[str, bytes],
    allowed_kernel_events: list[dict[str, object]],
) -> list[str]:
    after = _executor_control_plane_snapshot(repo)
    changed: list[str] = []
    event_paths = _runtime_event_paths(repo)
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        if (
            path in event_paths
            and path in after
            and _kernel_heartbeat_only_journal_append(
                before.get(path, b""),
                after[path],
                allowed_kernel_events,
            )
        ):
            continue
        changed.append(path)
    return changed


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

    # Runtime/review/referee evidence is written only by bounded kernel
    # entrypoints. A task cannot make these paths writable by listing them in
    # allowed_paths; legitimate control-plane commits bypass this task-output
    # predicate explicitly at their call sites.
    if norm.startswith("reports/status/swarm_runs/"):
        return False, "swarm_runs_kernel_only"
    if norm.startswith("reports/status/reviews/"):
        return False, "reviews_kernel_only"
    if norm.startswith("reports/status/referee_reports/"):
        return False, "referee_reports_kernel_only"
    if norm == "reports/status/referee_calibration.json":
        return False, "referee_calibration_kernel_only"
    if norm.startswith("reports/status/referee_calibration_runs/"):
        return False, "referee_calibration_runs_kernel_only"
    if norm.startswith("reports/status/events/"):
        return False, "events_kernel_only"

    if norm == task_file_path:
        return True, None
    if norm in _task_projection_paths(task_file_path):
        return True, None
    if norm.startswith(".orchestrator/handoff/"):
        digits = task_id[1:] if task_id.startswith("T") else task_id
        if Path(norm).name.startswith(f"H{digits}_"):
            return True, None
        return False, "handoff_namespace_violation"
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
        if _kernel_namespaced_run_path(task.task_id, changed_path):
            continue
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


def _strip_cli_option(argv: list[str], option: str) -> list[str]:
    """Drop `option value` and `option=value` occurrences from an argv list.
    Used to remove kernel-injected-only options an author tried to supply."""
    out: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            skip_next = True
            continue
        if token.startswith(option + "="):
            continue
        out.append(token)
    return out


def _run_gates(
    repo: Path,
    gates: list[str],
    *,
    interpreter_allowlist: tuple[str, ...] = DEFAULT_GATE_INTERPRETER_ALLOWLIST,
    timeout_seconds: int = DEFAULT_GATE_TIMEOUT_SECONDS,
    enforce_form: bool = True,
    task_kind: str | None = None,
) -> tuple[bool, list[dict[str, object]]]:
    """Constrained gate execution (§4.0 #12 + #18, hardened for M2): no shell,
    interpreter allowlist, gate-FORM policy (make <target> / python <repo .py>
    — no inline -c/-m code, so an autonomously-authored gate is never an
    arbitrary-code channel), stripped environment, per-gate timeout, network
    denied where the OS supports it, head+tail capture. Every production
    caller keeps enforce_form=True; only the lower-layer sandbox-mechanics
    tests set it False to exercise raw execution with synthetic commands."""
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
        if (
            len(argv) >= 2
            and argv[0] in {"python", "python3", sys.executable}
            and os.path.normpath(argv[1]) == "scripts/quality_gates.py"
        ):
            # --task-kind is kernel-injected AUTHORITY, never author-supplied:
            # strip any value the task frontmatter smuggled in (which would let
            # a task suppress its own science-gate form, e.g. a model task
            # pinning --task-kind lit_review) and force the authoritative
            # frontmatter kind. task_kind=None (no authoritative kind) leaves it
            # unset → mode-default union, which is strictly safer than any
            # author-narrowed kind.
            argv = _strip_cli_option(argv, "--task-kind")
            if task_kind is not None:
                argv.extend(["--task-kind", task_kind])
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

        form_violation = gate_command_violation(gate) if enforce_form else None
        if form_violation is not None:
            record["constraint_violation"] = f"gate_form_forbidden:{form_violation}"
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


def _amendment_scalar(value: str) -> object:
    stripped = value.strip().strip("'\"")
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return stripped


def _parse_amendment_record(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced is not None:
            try:
                payload = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None
    if isinstance(payload, dict):
        return payload
    if not stripped.startswith("---"):
        return None
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return None
    record: dict[str, object] = {}
    nested: dict[str, object] | None = None
    for raw_line in parts[1].splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            return None
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if indent == 0:
            if raw_value.strip():
                record[key] = _amendment_scalar(raw_value)
                nested = None
            else:
                child: dict[str, object] = {}
                record[key] = child
                nested = child
        elif nested is not None:
            nested[key] = _amendment_scalar(raw_value)
        else:
            return None
    return record


def _amendment_pointer_present(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        path = value.get("path")
        return isinstance(path, str) and bool(path.strip())
    return False


def _validate_amendment_record(
    *,
    repo: Path,
    phase: str,
    from_version: int,
    to_version: int,
) -> tuple[Path, dict[str, object] | None, list[str]]:
    rel_path = Path("docs/prereg/amendments") / f"{phase}_v{to_version}.md"
    path = repo / rel_path
    failures: list[str] = []
    if not path.is_file() or path.is_symlink():
        return rel_path, None, ["amendment_record_missing_or_not_regular"]
    git_probe = _run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture=True,
        check=False,
    )
    if git_probe.returncode == 0 and _run(
        ["git", "ls-files", "--error-unmatch", "--", rel_path.as_posix()],
        cwd=repo,
        capture=True,
        check=False,
    ).returncode != 0:
        failures.append("amendment_record_not_git_tracked")
    try:
        record = _parse_amendment_record(_read_text(path))
    except OSError:
        record = None
    if record is None:
        failures.append("amendment_record_invalid")
        return rel_path, None, failures
    expected = {
        "schema_version": PREREG_AMENDMENT_SCHEMA_VERSION,
        "phase": phase,
        "from_version": from_version,
        "to_version": to_version,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            failures.append(f"amendment_record_field_mismatch:{field}")
    rerun = record.get("dual_definition_rerun")
    if not isinstance(rerun, dict):
        failures.append("amendment_record_missing_dual_definition_rerun")
    else:
        for field in ("old_artifact", "new_artifact", "sensitivity_delta_artifact"):
            if not _amendment_pointer_present(rerun.get(field)):
                failures.append(f"amendment_record_missing_pointer:{field}")
    for field in ("human_reviewer", "justification"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            failures.append(f"amendment_record_missing_field:{field}")
    effective_date = record.get("effective_date")
    try:
        dt.date.fromisoformat(effective_date if isinstance(effective_date, str) else "")
    except ValueError:
        failures.append("amendment_record_invalid_effective_date")
    return rel_path, record, failures


def cmd_lock_prereg(args: argparse.Namespace) -> int:
    """Activate or explicitly amend one phased preregistration lock."""
    if not isinstance(args.locked_by, str) or not args.locked_by.strip():
        print("--locked-by must be a non-empty name", file=sys.stderr)
        return 1
    args.locked_by = args.locked_by.strip()
    repo = _repo_root()
    rel_path = Path(PREREG_PHASE_FILES[args.phase])
    path = repo / rel_path
    lock, error = load_prereg_lock(path, expected_phase=args.phase)
    if error is not None or lock is None:
        print(f"cannot lock preregistration: {error}", file=sys.stderr)
        return 1

    already_locked = lock.get("status") == "locked"
    if already_locked and not args.amend:
        print(
            "preregistration phase is already locked; pass --amend for an explicit amendment",
            file=sys.stderr,
        )
        return 1
    if args.amend and not already_locked:
        print("--amend requires an already-locked preregistration phase", file=sys.stderr)
        return 1

    current_version = lock.get("lock_version")
    if not isinstance(current_version, int) or isinstance(current_version, bool) or current_version < 0:
        print("preregistration lock_version must be a non-negative integer", file=sys.stderr)
        return 1
    if args.amend:
        # §6.1 amendment discipline: the cap is TWO PER PROGRAM (not per phase),
        # counted from the append-only journal, and versions are monotonic — a
        # hand-edited header rolled back below the journal's recorded version
        # cannot launder a fresh amendment past the cap.
        events, _ = swarm_events.read_events(repo)
        program_amendment_count = 0
        phase_journal_versions: list[int] = []
        for event in events:
            if not isinstance(event, dict) or event.get("event") != "prereg_amendment":
                continue
            program_amendment_count += 1
            ev_version = event.get("lock_version")
            if event.get("phase") == args.phase and isinstance(ev_version, int) and not isinstance(ev_version, bool):
                phase_journal_versions.append(ev_version)
        journal_max = max(phase_journal_versions, default=1)
        if current_version < journal_max:
            _record_swarm_event(
                repo,
                {
                    "event": "amendment_header_rollback",
                    "phase": args.phase,
                    "header_version": current_version,
                    "journal_max_version": journal_max,
                    "required_gate": "L3",
                },
                escalation=True,
            )
            print(
                f"amendment_header_rollback:{args.phase}:{current_version}<{journal_max}",
                file=sys.stderr,
            )
            return 1
        # Per-phase header ceiling (fast local defense) AND the program-wide
        # journal cap (the authoritative §6.1 "two per program" limit).
        if current_version >= 3 or program_amendment_count >= 2:
            _record_swarm_event(
                repo,
                {
                    "event": "amendment_cap_exceeded",
                    "phase": args.phase,
                    "lock_version": current_version,
                    "program_amendment_count": program_amendment_count,
                    "required_gate": "L3",
                },
                escalation=True,
            )
            print("amendment_cap_exceeded:L3_required", file=sys.stderr)
            return 1
    body = lock.get("body")
    body_sha256 = lock.get("body_sha256")
    if not isinstance(body, str) or not isinstance(body_sha256, str):
        print("preregistration body could not be hashed", file=sys.stderr)
        return 1

    version = current_version + 1
    amendment_record_path: Path | None = None
    amendment_record: dict[str, object] | None = None
    if args.amend:
        amendment_record_path, amendment_record, record_failures = _validate_amendment_record(
            repo=repo,
            phase=args.phase,
            from_version=current_version,
            to_version=version,
        )
        if record_failures:
            print(
                "amendment_record_required:" + ",".join(record_failures),
                file=sys.stderr,
            )
            return 1
    locked_at_utc = _utc_now_iso()
    header = "\n".join(
        [
            "---",
            f"schema_version: {PREREG_LOCK_SCHEMA_VERSION}",
            f"phase: {args.phase}",
            "status: locked",
            f"locked_at_utc: {locked_at_utc}",
            f"locked_sha256: {body_sha256}",
            f"locked_by: {args.locked_by}",
            f"lock_version: {version}",
            "---",
            "",
        ]
    )
    path.write_text(header + body, encoding="utf-8")

    event_name = "prereg_amendment" if args.amend else "prereg_locked"
    _record_swarm_event(
        repo,
        {
            "event": event_name,
            "phase": args.phase,
            "lock_path": rel_path.as_posix(),
            "locked_by": args.locked_by,
            "locked_sha256": body_sha256,
            "lock_version": version,
            "status": "locked",
            "locked_at_utc": locked_at_utc,
            **(
                {
                    "amendment_record": amendment_record_path.as_posix(),
                    "human_reviewer": amendment_record.get("human_reviewer"),
                }
                if amendment_record_path is not None and amendment_record is not None
                else {}
            ),
        },
        escalation=True,
    )
    print(
        json.dumps(
            {
                "event": event_name,
                "phase": args.phase,
                "lock_path": rel_path.as_posix(),
                "locked_sha256": body_sha256,
                "lock_version": version,
            },
            indent=2,
            sort_keys=True,
        )
    )
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


def cmd_referee_task(args: argparse.Namespace) -> int:
    repo = _repo_root()
    output_root_raw = os.environ.get("SWARM_REFEREE_OUTPUT_ROOT", "").strip()
    output_repo = (
        Path(output_root_raw).expanduser().resolve()
        if output_root_raw
        else repo
    )
    if not (output_repo / ".orchestrator").is_dir():
        raise SystemExit(f"referee_output_root_invalid:{output_repo}")
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)
    task = _resolve_runtime_task(tasks, quarantined, args.task)
    if (
        task.task_kind not in {"etl", "analysis", "writing", "validation", "proof", "model", "bridge", "lit_review"}
        and not _repair_is_referee_reviewable(task)
        and not _task_is_manuscript_surface(task)
    ):
        raise SystemExit(f"referee_task_kind_not_substantive:{task.task_id}:{task.task_kind}")

    run_manifest_path, run_manifest = _latest_referee_run_manifest(contract, task.task_id)
    executor = run_manifest.get("executor") if isinstance(run_manifest.get("executor"), dict) else {}
    authoring_family = _referee_family(executor.get("tool"))
    if authoring_family is None:
        _record_swarm_event(
            repo,
            {"event": "referee_family_of_author", "task_id": task.task_id, "reason": "authoring_family_unknown"},
            escalation=True,
        )
        raise SystemExit(f"referee_authoring_family_unknown:{task.task_id}")

    frontmatter = _task_frontmatter(task)
    rubric_task_kind = task.task_kind
    if task.task_kind == "repair":
        source_kind = frontmatter.get("repair_source_task_kind")
        rubric_task_kind = str(source_kind) if isinstance(source_kind, str) else task.task_kind
    elif _task_is_manuscript_surface(task) and task.task_kind not in {
        "etl", "analysis", "writing", "validation", "proof", "model", "bridge", "lit_review"
    }:
        rubric_task_kind = "writing"
    rubric, rubric_path = _load_referee_rubric(repo, rubric_task_kind)
    is_manuscript = task.task_kind == "writing" or any(
        _path_matches_prefix(output, "reports/paper/") for output in task.outputs
    )
    rubrics = [rubric]
    rubric_paths = [rubric_path]
    if is_manuscript:
        manuscript_rubric, manuscript_path = _load_referee_rubric(repo, rubric_task_kind, manuscript=True)
        rubrics.append(manuscript_rubric)
        rubric_paths.append(manuscript_path)
    scoped_claims, ledger_diagnostic = _task_scoped_claims(repo, task)
    sampled_artifacts = _kernel_sampled_artifacts(repo, task.task_id, task=task)
    if ledger_diagnostic is not None:
        _record_swarm_event(
            repo,
            {
                "event": "referee_claim_ledger_unavailable",
                "task_id": task.task_id,
                "diagnostic": ledger_diagnostic,
            },
            escalation=True,
        )
    assertions = _assertion_candidates(repo, task) if is_manuscript else []
    required = _referee_required_verdicts(
        task=task,
        frontmatter=frontmatter,
        rubrics=rubrics,
        assertions=assertions,
    )
    if not required:
        raise SystemExit(f"referee_no_required_verdicts:{task.task_id}")
    run_manifest_relpath = run_manifest_path.relative_to(repo).as_posix()
    rubric_digest = hashlib.sha256(
        b"\0".join(path.read_bytes() for path in rubric_paths)
    ).hexdigest()
    declared_outputs = [
        {key: value for key, value in item.items() if key != "sha256"}
        for item in _declared_output_context(repo, task)
    ]
    context: dict[str, object] = {
        "task": {
            "path": task.path.relative_to(repo).as_posix(),
            "frontmatter": frontmatter,
            "contract_text": _read_text(task.path),
        },
        "diff_base_to_branch": _referee_diff(repo, args.base_branch, args.remote),
        "run_manifest": run_manifest,
        "run_manifest_path": run_manifest_relpath,
        "declared_outputs": declared_outputs,
        "rubrics": rubrics,
        "rubric_version": f"research_swarm.rubric.v1:{rubric_digest[:16]}",
        "required_verdicts": required,
        # The line number is an independent open instruction; its contents
        # are never supplied. The referee proves opening by quoting it, while
        # the kernel alone computes and records the disk digest.
        "kernel_sampled_artifacts": [
            {
                "claim_id": item.get("claim_id"),
                "path": item.get("path"),
                "challenge_line": item.get("challenge_line"),
            }
            for item in sampled_artifacts
        ],
        "assertion_prefilter_floor": assertions,
        "claim_ledger": _referee_claim_context(scoped_claims) if is_manuscript else [],
    }
    target_hashes = {
        str(item.get(key, ""))
        for item in sampled_artifacts
        for key in ("sha256", "ledger_sha256")
        if isinstance(item.get(key), str) and item.get(key)
    }
    context = dict(_redact_sample_hashes(context, target_hashes))
    run_manifest_sha256 = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
    invocation_event = _record_swarm_event(
        repo,
        {
            "event": "referee_invoked",
            "task_id": task.task_id,
            "run_manifest_sha256": run_manifest_sha256,
            "actor": "Referee",
            "session_id": _ACTOR_SESSION_ID,
            "backend": args.referee_backend,
            "requested_family": args.referee_family,
        },
    )
    outcome = _invoke_referee(
        context=context,
        repo=repo,
        task_id=task.task_id,
        backend=args.referee_backend,
        referee_family=args.referee_family,
        timeout_seconds=max(1, int(args.timeout_seconds)),
    )
    if outcome.returncode != 0:
        _record_swarm_event(
            repo,
            {
                "event": "referee_invocation_failed",
                "task_id": task.task_id,
                "backend": args.referee_backend,
                "run_manifest_sha256": run_manifest_sha256,
                "reason": outcome.stdout[-1000:],
            },
            escalation=True,
        )
        print(json.dumps({"task_id": task.task_id, "ok": False, "error": outcome.stdout}, indent=2, sort_keys=True))
        return 1
    if outcome.referee_family == authoring_family:
        _record_swarm_event(
            repo,
            {
                "event": "referee_family_of_author",
                "task_id": task.task_id,
                "authoring_family": authoring_family,
                "referee_family": outcome.referee_family,
            },
            escalation=True,
        )
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "ok": False,
                    "error": "referee_family_of_author",
                    "authoring_family": authoring_family,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    verdicts, opened, overall, validation_failures = _normalize_referee_report(
        payload=outcome.payload,
        required=required,
        sampled_artifacts=sampled_artifacts,
        sample_required=bool(scoped_claims),
    )
    timestamp = _utc_timestamp_compact()
    report_path = _next_json_artifact_path(output_repo / REFEREE_REPORT_DIR, task.task_id, timestamp)
    report_relpath = report_path.relative_to(output_repo).as_posix()
    report: dict[str, object] = {
        "schema_version": REFEREE_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "task_id": task.task_id,
        "actor": "Referee",
        "session_id": _ACTOR_SESSION_ID,
        "referee_family": outcome.referee_family,
        "authoring_family": authoring_family,
        "run_manifest_path": run_manifest_relpath,
        "run_manifest_sha256": run_manifest_sha256,
        "rubric_version": context["rubric_version"],
        "rubric_files": [path.relative_to(repo).as_posix() for path in rubric_paths],
        "verdicts": verdicts,
        "opened_artifacts": opened,
        "sampled_artifacts": [
            _public_sampled_artifact(item) for item in sampled_artifacts
        ],
        "assertion_prefilter_floor": assertions,
        "reviewed_artifacts": [item["path"] for item in context["declared_outputs"] if isinstance(item, dict)],
        "overall": overall,
        "valid": not validation_failures,
        "validation_failures": validation_failures,
        "calibrated": _referee_family_calibrated(output_repo, outcome.referee_family),
        "invocation_journaled": isinstance(invocation_event, dict),
    }
    _write_json(report_path, report)

    if validation_failures:
        for failure in validation_failures:
            if failure.startswith("referee_did_not_open_sampled"):
                _record_swarm_event(
                    repo,
                    {"event": "referee_did_not_open_sampled", "task_id": task.task_id, "report": report_relpath, "failure": failure},
                    escalation=True,
                )
        _record_swarm_event(
            repo,
            {"event": "referee_report_invalid", "task_id": task.task_id, "report": report_relpath, "failures": validation_failures},
            escalation=True,
        )
        print(json.dumps({"task_id": task.task_id, "ok": False, "report": report_relpath, "failures": validation_failures}, indent=2, sort_keys=True))
        return 1

    major_not_supported = [
        item
        for item in verdicts
        if item.get("severity") == "major" and item.get("verdict") == "not_supported"
    ]
    cannot_verify = [item for item in verdicts if item.get("verdict") == "cannot_verify"]
    blocking = [*major_not_supported, *cannot_verify]
    unregistered = [
        item
        for item in major_not_supported
        if isinstance(item.get("check_id"), str)
        and item["check_id"].startswith("ASSERTION-")
        and item.get("verdict") == "not_supported"
    ]
    if cannot_verify:
        _record_swarm_event(
            repo,
            {
                "event": "referee_cannot_verify",
                "task_id": task.task_id,
                "report": report_relpath,
                "finding_ids": [item.get("success_criterion_id", item.get("check_id")) for item in cannot_verify],
            },
            escalation=True,
        )
    if unregistered:
        _record_swarm_event(
            repo,
            {
                "event": "unregistered_assertion",
                "task_id": task.task_id,
                "report": report_relpath,
                "finding_ids": [item.get("check_id") for item in unregistered],
            },
            escalation=True,
        )

    revision_paths: list[Path] = []
    if blocking:
        try:
            revision_paths = generate_revision_tasks(repo=output_repo, report_path=report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _record_swarm_event(
                repo,
                {"event": "revision_task_generation_failed", "task_id": task.task_id, "report": report_relpath, "reason": str(exc)},
                escalation=True,
            )
            print(json.dumps({"task_id": task.task_id, "ok": False, "report": report_relpath, "error": str(exc)}, indent=2, sort_keys=True))
            return 1
    _record_swarm_event(
        repo,
        {
            "event": "referee_reported",
            "task_id": task.task_id,
            "report": report_relpath,
            "referee_family": outcome.referee_family,
            "overall": overall,
            "revision_tasks": [path.relative_to(output_repo).as_posix() for path in revision_paths],
        },
    )
    result = {
        "task_id": task.task_id,
        "ok": not blocking,
        "report": report_relpath,
        "overall": overall,
        "calibrated": report["calibrated"],
        "revision_tasks": [path.relative_to(output_repo).as_posix() for path in revision_paths],
        "plan_approval_pending": (
            PLAN_APPROVAL_PENDING_PATH.as_posix()
            if revision_paths and (output_repo / PLAN_APPROVAL_PENDING_PATH).is_file()
            else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if blocking else 0


def _latest_referee_report(repo: Path, task_id: str) -> tuple[Path, dict[str, object]] | None:
    paths = _matching_task_jsons(repo / REFEREE_REPORT_DIR, task_id)
    for path in reversed(paths):
        try:
            payload = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == REFEREE_REPORT_SCHEMA_VERSION:
            return path, payload
    return None


def _referee_family_votes(
    repo: Path,
    task_id: str,
    *,
    run_manifest_relpath: str | None = None,
    run_manifest_sha256: str | None = None,
) -> dict[str, tuple[Path, dict[str, object]]]:
    """Return at most one (the latest) vote per family for one artifact run."""
    votes: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in reversed(_matching_task_jsons(repo / REFEREE_REPORT_DIR, task_id)):
        try:
            payload = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("schema_version") != REFEREE_REPORT_SCHEMA_VERSION:
            continue
        if run_manifest_relpath is not None and payload.get("run_manifest_path") != run_manifest_relpath:
            continue
        if run_manifest_sha256 is not None and payload.get("run_manifest_sha256") != run_manifest_sha256:
            continue
        family = payload.get("referee_family")
        if isinstance(family, str) and family.strip() and family not in votes:
            votes[family] = (path, payload)
    return votes


def _referee_report_journaled(
    repo: Path,
    *,
    task_id: str,
    run_manifest_sha256: str,
    actor: object,
    session_id: object,
) -> bool:
    if actor != "Referee" or not isinstance(session_id, str) or not session_id:
        return False
    events, _ = swarm_events.read_events(repo)
    return any(
        event.get("event") == "referee_invoked"
        and event.get("task_id") == task_id
        and event.get("run_manifest_sha256") == run_manifest_sha256
        and event.get("actor") == actor
        and event.get("session_id") == session_id
        and event.get("actor_session") == session_id
        for event in events
    )


def _referee_backend_failed(
    repo: Path,
    *,
    task_id: str,
    run_manifest_sha256: str,
) -> bool:
    events, _ = swarm_events.read_events(repo)
    return any(
        event.get("event") in {"referee_invocation_failed", "referee_panel_unavailable"}
        and event.get("task_id") == task_id
        and event.get("run_manifest_sha256") == run_manifest_sha256
        for event in events
    )


def _referee_report_severity(
    repo: Path,
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
        path = (repo / raw).resolve()
        try:
            path.relative_to(repo.resolve())
            rubric = json.loads(_read_text(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        checks = rubric.get("checks") if isinstance(rubric, dict) else None
        if not isinstance(checks, list):
            continue
        for check in checks:
            if isinstance(check, dict) and check.get("id") == check_id:
                severity = check.get("severity")
                return str(severity) if severity in {"major", "minor"} else None
    return None


def _referee_owner_waiver(
    repo: Path,
    *,
    task_id: str,
    run_manifest_sha256: str,
) -> dict[str, object] | None:
    events, _ = swarm_events.read_events(repo)
    trusted_base = _trusted_integration_branch(repo)
    for event in reversed(events):
        if not (
            event.get("event") == "referee_owner_waiver"
            and event.get("emitted_by") == REFEREE_WAIVER_EMITTER
            and event.get("actor") == "HumanOwner"
            # the waiver must have been emitted on the trusted integration branch,
            # not merely carry a non-empty branch string
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


def cmd_referee_waiver(args: argparse.Namespace) -> int:
    """Append one human manuscript-quorum waiver as kernel evidence."""
    repo = _repo_root()
    # The protected integration branch comes from git, NOT from --base-branch — a
    # Worker on a task branch cannot name its own branch to authorize a waiver —
    # and the waiver must run on the main checkout, never a linked worktree.
    trusted_base = _trusted_integration_branch(repo)
    current_branch = _git_current_branch(repo)
    if _is_linked_worktree(repo) or current_branch != trusted_base:
        raise SystemExit(
            f"referee_waiver_requires_integration_branch:{current_branch}!={trusted_base}"
        )
    human_id = str(args.human_id).strip()
    reason = str(args.reason).strip()
    if not human_id or not reason:
        raise SystemExit("referee_waiver_identity_and_reason_required")
    contract = load_framework_contract(repo)
    tasks, quarantined = load_tasks_quarantined(contract)
    task = _resolve_runtime_task(tasks, quarantined, args.task)
    if not _task_is_manuscript_surface(task):
        raise SystemExit(f"referee_waiver_not_manuscript_surface:{task.task_id}")
    waiver = {"human_id": human_id, "reason": reason}
    manifest_path, manifest = _latest_referee_run_manifest(contract, task.task_id)
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    if result.get("status") != "ok":
        raise SystemExit(f"referee_waiver_run_not_passing:{task.task_id}")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    waiver_sha = hashlib.sha256(
        json.dumps(waiver, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    event = _record_swarm_event(
        repo,
        {
            "event": "referee_owner_waiver",
            "emitted_by": REFEREE_WAIVER_EMITTER,
            "actor": "HumanOwner",
            "control_plane_branch": current_branch,
            "task_id": task.task_id,
            "run_manifest_path": manifest_path.relative_to(repo).as_posix(),
            "run_manifest_sha256": manifest_sha,
            **waiver,
            "waiver_sha256": waiver_sha,
        },
        escalation=True,
    )
    if not isinstance(event, dict):
        raise SystemExit("referee_waiver_journal_write_failed")
    print(
        json.dumps(
            {
                "ok": True,
                "task_id": task.task_id,
                "run_manifest_sha256": manifest_sha,
                "waiver_sha256": waiver_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _referee_review_failures(
    *,
    repo: Path,
    task: Task,
    run_manifest_path: Path | None,
) -> list[str]:
    # Optional reports on S-tier / otherwise out-of-scope tasks never become
    # latent review requirements after a rerun. Scope is the first predicate.
    if not _referee_task_in_scope(task):
        return []
    evidence_root_raw = (
        os.environ.get("SWARM_REFEREE_OUTPUT_ROOT", "").strip()
        or os.environ.get("SWARM_EVENT_REPO_ROOT", "").strip()
    )
    evidence_repo = (
        Path(evidence_root_raw).expanduser().resolve()
        if evidence_root_raw
        else repo
    )
    run_manifest_relpath = (
        run_manifest_path.relative_to(repo).as_posix()
        if run_manifest_path is not None
        else None
    )
    run_manifest_sha256 = (
        hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
        if run_manifest_path is not None and run_manifest_path.is_file()
        else None
    )
    votes = _referee_family_votes(
        evidence_repo,
        task.task_id,
        run_manifest_relpath=run_manifest_relpath,
        run_manifest_sha256=run_manifest_sha256,
    )
    if not votes:
        if _latest_referee_report(evidence_repo, task.task_id) is not None:
            return ["referee_report_stale_run_manifest"]
        if run_manifest_sha256 is not None and _referee_backend_failed(
            evidence_repo,
            task_id=task.task_id,
            run_manifest_sha256=run_manifest_sha256,
        ):
            return ["referee_backend_unavailable"]
        return ["referee_required_missing"]
    failures: list[str] = []
    authoring_families: set[str] = set()
    for family, (_, report) in sorted(votes.items()):
        if not _referee_family_calibrated(evidence_repo, family):
            failures.append(f"referee_panel_uncalibrated:{family}")
        if report.get("valid") is not True:
            failures.append(f"referee_report_invalid:{family}")
        report_run_sha = report.get("run_manifest_sha256")
        if not isinstance(report_run_sha, str) or not _referee_report_journaled(
            evidence_repo,
            task_id=task.task_id,
            run_manifest_sha256=report_run_sha,
            actor=report.get("actor"),
            session_id=report.get("session_id"),
        ):
            failures.append(f"referee_report_unjournaled:{family}")
        authoring = report.get("authoring_family")
        if isinstance(authoring, str):
            authoring_families.add(authoring)
        if authoring == family:
            failures.append(f"referee_family_of_author:{family}")
        verdicts = report.get("verdicts") if isinstance(report.get("verdicts"), list) else []
        for item in verdicts:
            if not isinstance(item, dict):
                continue
            verdict = item.get("verdict")
            identifier = item.get("success_criterion_id", item.get("check_id", "unknown"))
            if verdict == "cannot_verify":
                failures.append(f"referee_cannot_verify:{identifier}")
            if (
                verdict == "not_supported"
                and _referee_report_severity(evidence_repo, report, item) == "major"
            ):
                failures.append(f"referee_not_supported:{identifier}")
            if (
                verdict == "not_supported"
                and isinstance(item.get("check_id"), str)
                and item["check_id"].startswith("ASSERTION-")
            ):
                failures.append(f"unregistered_assertion:{identifier}")
    is_manuscript_release_surface = _task_is_manuscript_surface(task)
    non_authoring_votes = set(votes) - authoring_families
    if is_manuscript_release_surface and len(non_authoring_votes) < 2:
        waiver = (
            _referee_owner_waiver(
                evidence_repo,
                task_id=task.task_id,
                run_manifest_sha256=run_manifest_sha256,
            )
            if run_manifest_sha256 is not None
            else None
        )
        required_votes = 1 if waiver is not None else 2
        if len(non_authoring_votes) < required_votes:
            failures.append(
                f"referee_manuscript_panel_family_quorum:{len(non_authoring_votes)}<{required_votes}"
            )
    return sorted(set(failures))


def _planner_backlog_path(repo: Path, raw_path: object) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "planner_task_path_invalid"
    normalized = _normalize_repo_relative_path(raw_path)
    path = Path(normalized)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parent.as_posix() != ".orchestrator/backlog"
        or path.suffix != ".md"
        or re.fullmatch(r"T\d{3}(?:[_-][A-Za-z0-9_-]+)?\.md", path.name) is None
    ):
        return None, f"planner_path_outside_authority:{normalized}"
    resolved = (repo / path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        return None, f"planner_path_outside_authority:{normalized}"
    expected_parent = repo.resolve() / ".orchestrator" / "backlog"
    if resolved.parent != expected_parent:
        return None, f"planner_path_outside_authority:{normalized}"
    return resolved, None


def _planner_repo_relpath(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _planner_workstreams_path(repo: Path) -> tuple[Path | None, str | None]:
    path = repo / ".orchestrator" / "workstreams.md"
    expected = repo.resolve() / ".orchestrator" / "workstreams.md"
    resolved = path.resolve()
    if resolved != expected:
        return None, "planner_path_outside_authority:.orchestrator/workstreams.md"
    return path, None


def _planner_task_paths(repo: Path, task_id: str) -> list[Path]:
    matches: list[Path] = []
    orchestrator = repo / ".orchestrator"
    if not orchestrator.exists():
        return matches
    for path in sorted(orchestrator.glob(f"*/{task_id}*.md")):
        try:
            frontmatter = _parse_task_frontmatter(_read_text(path))
        except OSError:
            continue
        if isinstance(frontmatter, dict) and frontmatter.get("task_id") == task_id:
            matches.append(path)
    return matches


def _planner_backlog_task_path(repo: Path, task_id: object) -> tuple[Path | None, str | None]:
    if not isinstance(task_id, str) or re.fullmatch(r"T\d{3}", task_id) is None:
        return None, "planner_task_id_invalid"
    matches = _planner_task_paths(repo, task_id)
    backlog = [path for path in matches if path.parent == repo / ".orchestrator" / "backlog"]
    if len(backlog) == 1 and len(matches) == 1:
        relpath = backlog[0].relative_to(repo).as_posix()
        return _planner_backlog_path(repo, relpath)
    if matches and not backlog:
        relpaths = ",".join(_planner_repo_relpath(repo, path) for path in matches)
        return None, f"planner_path_outside_authority:{relpaths}"
    if len(backlog) > 1 or len(matches) > 1:
        return None, f"planner_task_ambiguous:{task_id}"
    return None, f"planner_task_not_found:{task_id}"


def _planner_task_payload(payload: object) -> tuple[Path | None, str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None, "planner_task_payload_invalid"
    allowed_keys = {"path", "content"}
    if payload.get("action") == "create_task":
        allowed_keys.add("action")
    if set(payload) - allowed_keys:
        return None, None, "planner_task_payload_invalid"
    content = payload.get("content")
    if not isinstance(content, str):
        return None, None, "planner_task_content_invalid"
    return Path(str(payload.get("path", ""))), content, None


def _planner_proposal_authority_violations(
    *, repo: Path, proposals: list[dict[str, object]]
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    allowed_keys = {
        "create_task": {"action", "path", "content"},
        "update_workstreams": {"action", "content"},
        "split_task": {"action", "task_id", "into"},
        "triage_confirm": {"action", "task_id", "note"},
    }
    for index, proposal in enumerate(proposals):
        action = proposal.get("action")
        if action not in allowed_keys:
            violations.append(
                {"proposal_index": index, "reason": "planner_action_invalid", "action": action}
            )
            continue
        extra_keys = sorted(set(proposal) - allowed_keys[str(action)])
        if extra_keys:
            violations.append(
                {
                    "proposal_index": index,
                    "reason": "planner_proposal_schema_invalid",
                    "fields": extra_keys,
                }
            )
            continue
        if action == "create_task":
            _, reason = _planner_backlog_path(repo, proposal.get("path"))
            if reason is not None:
                violations.append({"proposal_index": index, "reason": reason})
        elif action == "update_workstreams":
            _, reason = _planner_workstreams_path(repo)
            if reason is not None:
                violations.append({"proposal_index": index, "reason": reason})
        elif action in {"split_task", "triage_confirm"}:
            _, reason = _planner_backlog_task_path(repo, proposal.get("task_id"))
            if reason is not None and reason.startswith("planner_path_outside_authority"):
                violations.append({"proposal_index": index, "reason": reason})
            if action == "split_task":
                children = proposal.get("into")
                if not isinstance(children, list):
                    continue
                for child_index, child in enumerate(children):
                    if not isinstance(child, dict):
                        continue
                    _, child_reason = _planner_backlog_path(repo, child.get("path"))
                    if child_reason is not None:
                        violations.append(
                            {
                                "proposal_index": index,
                                "child_index": child_index,
                                "reason": child_reason,
                            }
                        )
    return violations


def _set_task_triage(text: str, *, note: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    end_index = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if end_index is None:
        return text
    rendered = (
        "triage: {status: confirmed, by: planner, note: "
        + json.dumps(note.strip(), ensure_ascii=False)
        + "}\n"
    )
    for index in range(1, end_index):
        if re.match(r"^triage\s*:", lines[index]):
            lines[index] = rendered
            return "".join(lines)
    lines.insert(end_index, rendered)
    return "".join(lines)


def _planner_lint_diagnostics(
    *,
    repo: Path,
    contract: FrameworkContract,
    task_texts: dict[Path, str],
    deleted_paths: set[Path],
    changed_paths: set[Path],
) -> list[dict[str, object]]:
    deleted_resolved = {path.resolve() for path in deleted_paths}
    paths = [
        path for path in _iter_task_files(contract) if path.resolve() not in deleted_resolved
    ]
    known_paths = {path.resolve() for path in paths}
    for path in task_texts:
        if path.resolve() not in known_paths and path.resolve() not in deleted_resolved:
            paths.append(path)
            known_paths.add(path.resolve())
    diagnostics = lint_task_files(
        sorted(paths),
        repo_root=repo,
        network_workstreams=contract.network_workstreams,
        v1_exemptions=_load_v1_task_exemptions(repo),
        task_texts=task_texts,
    )
    # baseline diagnostics with NO proposed changes: any diagnostic present in
    # the proposed set but absent from baseline is a regression this batch
    # introduced (e.g. a split orphaning a validation task's constructed_by
    # link) — reject the whole proposal, not just its changed children (C9).
    baseline = lint_task_files(
        sorted(path for path in _iter_task_files(contract)),
        repo_root=repo,
        network_workstreams=contract.network_workstreams,
        v1_exemptions=_load_v1_task_exemptions(repo),
    )
    def _key(d) -> tuple:
        as_dict = d.as_dict()
        return (d.task, d.field, d.reason, str(as_dict.get("expected")), str(as_dict.get("actual")))

    baseline_keys = {_key(d) for d in baseline}
    return [
        diagnostic.as_dict()
        for diagnostic in diagnostics
        if _key(diagnostic) not in baseline_keys
    ]


def _persist_planner_changes(
    *,
    repo: Path,
    remote: str,
    base_branch: str,
    paths: Iterable[Path],
    message: str,
    strict: bool,
    push: bool = True,
) -> bool:
    candidate_paths = {_planner_repo_relpath(repo, path) for path in paths}
    owned_paths = sorted(
        path
        for path in candidate_paths
        if path == ".orchestrator/workstreams.md"
        or Path(path).parent.as_posix() == ".orchestrator/backlog"
    )
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
        raise SystemExit("planner_refused_preexisting_staged_changes:" + ",".join(unexpected))
    _run(["git", "add", "-A", "--", *owned_paths], cwd=repo, check=True)
    staged_after = _run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture=True,
        check=True,
    )
    if not (staged_after.stdout or "").strip():
        return False
    _git_commit(cwd=repo, message=message, strict=strict, paths=owned_paths)
    if push:
        _git_push(
            cwd=repo,
            remote=remote,
            ref=base_branch,
            set_upstream=False,
            strict=strict,
        )
    return True


def _task_hypothesis_links(frontmatter: object) -> list[str]:
    """Resolve hypothesis links from a task's frontmatter — list field
    hypothesis_ids (canonical, linted) plus the legacy scalar hypothesis_id
    (C11). Preregistration-artifact linkage is M3a; recorded forward."""
    if not isinstance(frontmatter, dict):
        return []
    links: list[str] = []
    raw_list = frontmatter.get("hypothesis_ids")
    if isinstance(raw_list, list):
        links.extend(str(item).strip() for item in raw_list if str(item).strip())
    scalar = frontmatter.get("hypothesis_id")
    if isinstance(scalar, str) and scalar.strip():
        links.append(scalar.strip())
    return links


_ROW_PLACEHOLDER_CELLS = {"-", "--", "n/a", "na", "tbd", "todo", "none", "...", "???", "xxx", "wip"}


def _workstream_row_ids(content: str) -> tuple[set[str], bool]:
    """(set of well-formed workstream ids, all-rows-well-formed). A row is a
    table line whose FIRST cell LEADS with a workstream id (production format
    is `W0 Protocol/Contracts`, id + label in one cell). The purpose/owns/
    not-owns cells must be non-blank and not placeholder-only. Prose lines are
    ignored; a malformed W#-leading row fails the batch."""
    ids: set[str] = set()
    ok = True
    for raw in content.splitlines():
        line = raw.strip()
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        first = cells[0] if cells else ""
        id_match = re.match(r"^(W\d+)(?:\s|$)", first)
        if id_match is None:
            continue
        wid = id_match.group(1)
        data_cells = cells[1:4]
        cells_ok = (
            len(cells) >= 4
            and all(cell for cell in data_cells)
            and all(cell.strip("`").lower() not in _ROW_PLACEHOLDER_CELLS for cell in data_cells)
        )
        if not cells_ok:
            ok = False
        if wid in ids:
            ok = False
        ids.add(wid)
    return ids, ok


def _workstreams_update_valid(repo: Path, content: str) -> bool:
    """A proposed workstreams.md is valid when it still parses to >=1
    well-formed unique workstream row AND PRESERVES every workstream id the
    current document defines — the planner may add ownership, never silently
    drop it (C10, reworked after the verification pass found the line-by-line
    check rejected legitimate prose while accepting a lone fake row)."""
    proposed_ids, rows_ok = _workstream_row_ids(content)
    if not proposed_ids or not rows_ok:
        return False
    existing_path = repo / ".orchestrator" / "workstreams.md"
    if existing_path.is_file():
        existing_ids, _ = _workstream_row_ids(existing_path.read_text(encoding="utf-8"))
        if not existing_ids <= proposed_ids:
            return False
    return True


def _existing_task_ids(contract: FrameworkContract) -> set[str]:
    ids: set[str] = set()
    for path in _iter_task_files(contract):
        frontmatter = _parse_task_frontmatter(_read_text(path))
        if isinstance(frontmatter, dict) and isinstance(frontmatter.get("task_id"), str):
            ids.add(frontmatter["task_id"])
    return ids


def _proposed_task_id_error(
    *, path: Path, content: str, existing_ids: set[str], batch_ids: set[str]
) -> str | None:
    frontmatter = _parse_task_frontmatter(content)
    declared = frontmatter.get("task_id") if isinstance(frontmatter, dict) else None
    filename_id = _parse_task_id_from_branch(path.stem)
    if not isinstance(declared, str) or re.fullmatch(r"T\d{3}", declared) is None:
        return "planner_task_id_invalid"
    if filename_id != declared:
        return f"planner_task_id_filename_mismatch:{filename_id}:{declared}"
    if declared in existing_ids or declared in batch_ids:
        return f"planner_task_id_not_unique:{declared}"
    return None


def _apply_planner_proposals(
    *,
    mode: str,
    proposals: list[dict[str, object]],
    repo: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Validate, apply, and persist one bounded Planner proposal batch."""
    authority_violations = _planner_proposal_authority_violations(
        repo=repo, proposals=proposals
    )
    if authority_violations:
        outcomes = [
            {
                "proposal_index": index,
                "action": proposal.get("action"),
                "status": "refused_batch",
            }
            for index, proposal in enumerate(proposals)
        ]
        _record_swarm_event(
            repo,
            {
                "event": "planner_write_refused",
                "mode": mode,
                "violations": authority_violations,
            },
            escalation=True,
        )
        summary = {
            "mode": mode,
            "batch_refused": True,
            "committed": False,
            "outcomes": outcomes,
        }
        _record_swarm_event(repo, {"event": "planner_applied", **summary})
        return summary

    contract = load_framework_contract(repo)
    task_texts: dict[Path, str] = {}
    deleted_paths: set[Path] = set()
    workstreams_text: str | None = None
    outcomes: list[dict[str, object]] = []
    split_events: list[dict[str, object]] = []
    existing_task_ids = _existing_task_ids(contract)
    # pre-scan: every task id this batch DECLARES (create paths + split
    # children). An id declared more than once anywhere in the batch is a
    # duplicate and is rejected wherever it appears (verification pass).
    declared_ids: list[str] = []
    for proposal in proposals:
        action = str(proposal.get("action"))
        if action == "create_task":
            claimed = _parse_task_id_from_branch(Path(str(proposal.get("path") or "")).stem)
            if claimed:
                declared_ids.append(claimed)
        elif action == "split_task":
            into = proposal.get("into")
            for child in (into if isinstance(into, list) else []):
                if isinstance(child, dict):
                    child_id = _parse_task_id_from_branch(Path(str(child.get("path") or "")).stem)
                    if child_id:
                        declared_ids.append(child_id)
    batch_duplicate_ids = {tid for tid in declared_ids if declared_ids.count(tid) > 1}

    for index, proposal in enumerate(proposals):
        task_texts_before = dict(task_texts)
        deleted_paths_before = set(deleted_paths)
        split_event_count_before = len(split_events)
        action = str(proposal.get("action"))
        outcome: dict[str, object] = {
            "proposal_index": index,
            "action": action,
            "status": "refused",
        }
        changed_paths: set[Path] = set()

        if action == "create_task":
            path, path_reason = _planner_backlog_path(repo, proposal.get("path"))
            content = proposal.get("content")
            if path_reason is not None or path is None or not isinstance(content, str):
                outcome["reason"] = path_reason or "planner_task_content_invalid"
                outcomes.append(outcome)
                continue
            if path.exists() or path in task_texts:
                outcome["reason"] = "planner_task_already_exists"
                outcomes.append(outcome)
                continue
            # unique across existing tasks AND not a batch-duplicated id
            id_error = _proposed_task_id_error(
                path=path,
                content=content,
                existing_ids=existing_task_ids,
                batch_ids=batch_duplicate_ids,
            )
            if id_error is not None:
                outcome["reason"] = id_error
                outcomes.append(outcome)
                continue
            task_texts[path] = content
            changed_paths.add(path)

        elif action == "update_workstreams":
            content = proposal.get("content")
            if not isinstance(content, str) or not content.strip() or not _workstreams_update_valid(repo, content):
                outcome["reason"] = "planner_workstreams_content_invalid"
                outcomes.append(outcome)
                continue
            if workstreams_text is not None:
                outcome["reason"] = "planner_workstreams_update_conflict"
                outcomes.append(outcome)
                continue
            workstreams_text = content
            outcome["status"] = "applied"
            outcome["path"] = ".orchestrator/workstreams.md"
            outcomes.append(outcome)
            continue

        elif action == "triage_confirm":
            path, reason = _planner_backlog_task_path(repo, proposal.get("task_id"))
            note = proposal.get("note")
            if path is None or reason is not None:
                outcome["reason"] = reason
                outcomes.append(outcome)
                continue
            if not isinstance(note, str) or not note.strip():
                outcome["reason"] = "planner_triage_note_invalid"
                outcomes.append(outcome)
                continue
            source = task_texts.get(path, _read_text(path))
            task_texts[path] = _set_task_triage(source, note=note)
            changed_paths.add(path)

        elif action == "split_task":
            parent, reason = _planner_backlog_task_path(repo, proposal.get("task_id"))
            children = proposal.get("into")
            if parent is None or reason is not None:
                outcome["reason"] = reason
                outcomes.append(outcome)
                continue
            parent_frontmatter = _parse_task_frontmatter(
                task_texts.get(parent, _read_text(parent))
            )
            hypothesis_links = _task_hypothesis_links(parent_frontmatter)
            hypothesis_id = hypothesis_links[0] if hypothesis_links else None
            if hypothesis_links:
                outcome["reason"] = "hypothesis_retirement_requires_human"
                outcomes.append(outcome)
                _record_swarm_event(
                    repo,
                    {
                        "event": "hypothesis_retirement_escalated",
                        "level": "L3",
                        "task_id": proposal.get("task_id"),
                        "hypothesis_id": hypothesis_id,
                        "mode": mode,
                    },
                    escalation=True,
                )
                continue
            if not isinstance(children, list) or not children:
                outcome["reason"] = "planner_split_children_invalid"
                outcomes.append(outcome)
                continue
            child_ids: list[str] = []
            child_paths: list[Path] = []
            child_error: str | None = None
            for child in children:
                raw_path, content, payload_reason = _planner_task_payload(child)
                path, path_reason = _planner_backlog_path(
                    repo, raw_path.as_posix() if raw_path is not None else None
                )
                if payload_reason is not None or path_reason is not None or path is None or content is None:
                    child_error = payload_reason or path_reason or "planner_task_payload_invalid"
                    break
                if path.exists() or path in task_texts or path in child_paths:
                    child_error = "planner_task_already_exists"
                    break
                parent_id = _parse_task_id_from_branch(parent.stem)
                child_id = _parse_task_id_from_branch(path.stem)
                id_error = _proposed_task_id_error(
                    path=path,
                    content=content,
                    existing_ids=existing_task_ids,
                    batch_ids=batch_duplicate_ids | set(child_ids),
                )
                if id_error is not None or child_id == parent_id:
                    child_error = id_error or "planner_split_child_reuses_parent_id"
                    break
                child_paths.append(path)
                child_ids.append(child_id)
                task_texts[path] = content
                changed_paths.add(path)
            if child_error is not None:
                for path in child_paths:
                    task_texts.pop(path, None)
                outcome["reason"] = child_error
                outcomes.append(outcome)
                continue
            deleted_paths.add(parent)
            changed_paths.update(child_paths)
            outcome["children"] = child_ids
            split_events.append(
                {
                    "event": "task_split",
                    "parent": proposal.get("task_id"),
                    "children": child_ids,
                    "mode": mode,
                }
            )

        diagnostics = _planner_lint_diagnostics(
            repo=repo,
            contract=contract,
            task_texts=task_texts,
            deleted_paths=deleted_paths,
            changed_paths=changed_paths,
        )
        if diagnostics:
            task_texts = task_texts_before
            deleted_paths = deleted_paths_before
            del split_events[split_event_count_before:]
            outcome["status"] = "lint_failed"
            outcome["diagnostics"] = diagnostics
            outcomes.append(outcome)
            _record_swarm_event(
                repo,
                {
                    "event": "planner_proposal_lint_failed",
                    "mode": mode,
                    "proposal_index": index,
                    "action": action,
                    "diagnostics": diagnostics,
                },
            )
            continue

        outcome["status"] = "applied"
        outcome["paths"] = sorted(_planner_repo_relpath(repo, path) for path in changed_paths)
        outcomes.append(outcome)

    applied_task_relpaths = {
        raw_path
        for outcome in outcomes
        if outcome.get("status") == "applied"
        for raw_path in outcome.get("paths", [])
        if isinstance(raw_path, str)
    }
    task_texts_by_relpath = {
        _planner_repo_relpath(repo, path): text for path, text in task_texts.items()
    }
    applied_task_paths = {repo / relpath for relpath in applied_task_relpaths}
    for path in sorted(applied_task_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            task_texts_by_relpath[_planner_repo_relpath(repo, path)],
            encoding="utf-8",
        )
    applied_deletes = {
        path
        for path in deleted_paths
        if any(
            outcome.get("status") == "applied"
            and outcome.get("action") == "split_task"
            for outcome in outcomes
        )
    }
    for path in sorted(applied_deletes):
        path.unlink()
    touched_paths = set(applied_task_paths) | applied_deletes
    if workstreams_text is not None and any(
        outcome.get("status") == "applied"
        and outcome.get("action") == "update_workstreams"
        for outcome in outcomes
    ):
        workstreams_path, workstreams_reason = _planner_workstreams_path(repo)
        if workstreams_path is None:
            raise RuntimeError(workstreams_reason or "planner_workstreams_path_invalid")
        workstreams_path.write_text(workstreams_text, encoding="utf-8")
        touched_paths.add(workstreams_path)

    committed = _persist_planner_changes(
        repo=repo,
        remote=getattr(args, "remote", "origin"),
        base_branch=getattr(args, "base_branch", "main"),
        paths=touched_paths,
        message=f"planner: {mode}",
        strict=bool(getattr(args, "unattended", False)),
        push=not bool(getattr(args, "no_push", False)),
    )
    for event in split_events:
        _record_swarm_event(repo, event)
    summary = {
        "mode": mode,
        "batch_refused": False,
        "committed": committed,
        "outcomes": outcomes,
    }
    _record_swarm_event(repo, {"event": "planner_applied", **summary})
    return summary


def _plan_program_context(repo: Path, contract: FrameworkContract) -> dict[str, object]:
    tasks, quarantined = load_tasks_quarantined(contract)
    backlog = [
        _task_summary(task)
        for task in sorted(tasks.values(), key=lambda item: item.task_id)
        if task.state == "backlog"
    ]
    return {
        "trigger_id": "launch",
        "project_contract": _read_text(repo / "contracts" / "project.yaml"),
        "framework_contract": _read_text(repo / "contracts" / "framework.json"),
        "protocol": _read_text(repo / "docs" / "protocol.md"),
        "workstreams": _read_text(repo / ".orchestrator" / "workstreams.md"),
        "backlog_summary": backlog,
        "quarantined": quarantined,
    }


def _plan_content_digest(repo: Path) -> str:
    """Digest of the exact plan surface an approval binds to: every backlog
    task file plus workstreams.md, content-hashed (C4)."""
    parts: list[str] = []
    backlog_dir = repo / ".orchestrator" / "backlog"
    if backlog_dir.is_dir():
        for path in sorted(backlog_dir.glob("*.md")):
            parts.append(path.name)
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
    workstreams = repo / ".orchestrator" / "workstreams.md"
    if workstreams.is_file():
        parts.append(hashlib.sha256(workstreams.read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def cmd_plan_program(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    outcome = _invoke_planner(
        mode="launch",
        context=_plan_program_context(repo, contract),
        repo=repo,
        args=args,
    )
    application: dict[str, object] | None = None
    approval_pending = False
    if outcome.returncode == 0:
        application = _apply_planner_proposals(
            mode="launch",
            proposals=outcome.proposals,
            repo=repo,
            args=args,
        )
        if not application["batch_refused"]:
            pending_path = repo / PLAN_APPROVAL_PENDING_PATH
            _write_json(
                pending_path,
                {
                    "schema_version": "research_swarm.plan_approval_pending.v1",
                    "created_at_utc": _utc_now_iso(),
                    "planner_backend": getattr(args, "planner_backend", "mock"),
                    "proposal_count": len(outcome.proposals),
                    "base_sha": _git_head_sha(repo),
                    "plan_digest": _plan_content_digest(repo),
                },
            )
            approval_pending = True
            _record_swarm_event(
                repo,
                {
                    "event": "plan_awaiting_human_approval",
                    "pending_path": PLAN_APPROVAL_PENDING_PATH.as_posix(),
                    "proposal_count": len(outcome.proposals),
                },
                escalation=True,
            )
    else:
        _record_swarm_event(
            repo,
            {
                "event": "planner_invocation_failed",
                "mode": "launch",
                "returncode": outcome.returncode,
                "stdout": outcome.stdout[:2048],
            },
            escalation=True,
        )

    print(
        json.dumps(
            {
                "mode": "launch",
                "planner": {
                    "returncode": outcome.returncode,
                    "stdout": outcome.stdout,
                    "proposal_count": len(outcome.proposals),
                },
                "application": application,
                "approval_pending": approval_pending,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.returncode == 0 and application is not None and not application["batch_refused"] else 1


def cmd_approve_plan(args: argparse.Namespace) -> int:
    repo = _repo_root()
    pending_path = repo / PLAN_APPROVAL_PENDING_PATH
    if not pending_path.is_file():
        print(
            json.dumps(
                {"status": "no_pending_plan", "approved_by": args.approved_by},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    try:
        pending = json.loads(_read_text(pending_path))
    except (OSError, json.JSONDecodeError):
        pending = {}
    recorded_digest = pending.get("plan_digest")
    current_digest = _plan_content_digest(repo)
    if not isinstance(recorded_digest, str) or recorded_digest != current_digest:
        # the plan changed after it was proposed — approval of DIFFERENT
        # content must not succeed (C4)
        _record_swarm_event(
            repo,
            {
                "event": "plan_approval_drift_refused",
                "approved_by": args.approved_by,
                "recorded_digest": recorded_digest,
                "current_digest": current_digest,
            },
            escalation=True,
        )
        print(
            json.dumps(
                {"status": "plan_drift_refused", "approved_by": args.approved_by},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    pending_path.unlink()
    _record_swarm_event(
        repo,
        {
            "event": "plan_approved",
            "approved_by": args.approved_by,
            "pending_path": PLAN_APPROVAL_PENDING_PATH.as_posix(),
            "plan_digest": current_digest,
        },
    )
    print(
        json.dumps(
            {"status": "approved", "approved_by": args.approved_by},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _recent_planner_failures(contract: FrameworkContract, task_id: str) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    manifests = _matching_v2_run_manifest_data(
        _matching_task_jsons(contract.run_manifest_dir, task_id),
        task_id,
    )
    for path, manifest in manifests[-2:]:
        result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        if result.get("status") != "blocked":
            continue
        failures.append(
            {
                "manifest": path.relative_to(contract.repo_root).as_posix(),
                "failure_context": _failure_context_from_manifest(manifest),
            }
        )
    return failures


def _workstream_excerpt(repo: Path, workstream: str) -> str:
    text = _read_text(repo / ".orchestrator" / "workstreams.md")
    lines = [line for line in text.splitlines() if workstream in line]
    return "\n".join(lines) or text


def cmd_triage(args: argparse.Namespace) -> int:
    repo = _repo_root()
    contract = load_framework_contract(repo)
    task_path, reason = _planner_backlog_task_path(repo, args.task)
    if task_path is None:
        raise SystemExit(reason or f"planner_task_not_found:{args.task}")
    task = load_task(task_path, contract)
    context = {
        "trigger_id": task.task_id,
        "task_id": task.task_id,
        "task_file": _read_text(task.path),
        "workstream": _workstream_excerpt(repo, task.workstream),
        "recent_failures": _recent_planner_failures(contract, task.task_id),
        "triage_rule": (
            "The T035 rule: discovery and construction never combine. If reconnaissance "
            "shows unclear ground truth, split the task instead of widening its diff."
        ),
    }
    outcome = _invoke_planner(mode="triage", context=context, repo=repo, args=args)
    application: dict[str, object] | None = None
    if outcome.returncode == 0:
        application = _apply_planner_proposals(
            mode="triage",
            proposals=outcome.proposals,
            repo=repo,
            args=args,
        )
    else:
        _record_swarm_event(
            repo,
            {
                "event": "planner_invocation_failed",
                "mode": "triage",
                "task_id": task.task_id,
                "returncode": outcome.returncode,
                "stdout": outcome.stdout[:2048],
            },
            escalation=True,
        )
    print(
        json.dumps(
            {
                "mode": "triage",
                "task_id": task.task_id,
                "planner": {
                    "returncode": outcome.returncode,
                    "stdout": outcome.stdout,
                    "proposal_count": len(outcome.proposals),
                },
                "application": application,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.returncode == 0 and application is not None and not application["batch_refused"] else 1


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
    # ready_backlog_tasks already excludes unapproved v2 tasks (the shared
    # dispatch funnel) — cmd_tick no longer needs its own filter.
    ready = ready_backlog_tasks(tasks, claimed_ids, contract)

    dispatchable = list(ready)
    plan_skipped: list[dict[str, str]] = []
    # reporting only: name the backlog v2 tasks the shared funnel withheld
    # for plan approval (the funnel already enforced the exclusion)
    if _plan_approval_pending(repo):
        for task in tasks.values():
            if (
                task.state == "backlog"
                and task.role in set(contract.task_execution_roles)
                and task.task_id not in claimed_ids
                and TaskV2Fields(_task_frontmatter(task)).task_schema == TASK_SCHEMA_VERSION
            ):
                plan_skipped.append({"task_id": task.task_id, "reason": "plan_unapproved"})

    capacity = max(0, int(args.max_workers))
    selected = choose_tasks_heuristic(dispatchable, capacity)

    summary = {
        "done": sorted(task_id for task_id, task in tasks.items() if task.state == "done"),
        "integration_ready": sorted(task_id for task_id, task in tasks.items() if task.state == "integration_ready"),
        "claimed": sorted(claimed_ids),
        "quarantined": quarantined,
        "ready": [task.task_id for task in ready],
        "selected": [task.task_id for task in selected],
        "skipped": plan_skipped,
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


def _kernel_namespaced_run_path(task_id: str, path: str) -> bool:
    normalized = _normalize_repo_relative_path(path)
    direct = re.fullmatch(
        rf"reports/status/swarm_runs/{re.escape(task_id)}_[A-Za-z0-9_.-]+\.json",
        normalized,
    )
    nested = re.fullmatch(
        rf"reports/status/swarm_runs/(?:logs|sessions)/{re.escape(task_id)}_[A-Za-z0-9_.-]+\.(?:log|json)",
        normalized,
    )
    return direct is not None or nested is not None


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
        # durably persist the reopen BEFORE releasing the ref: a kill between
        # the two leaves backlog+expired-claim, which the next cycle releases
        _persist_projection_changes(
            repo=repo,
            remote=args.remote,
            base_branch=args.base_branch,
            filenames=[task.path.name],
            message=f"{task_id}: reopen orphaned",
            strict=bool(args.unattended),
        )
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


def _referee_output_paths_from_stdout(stdout: str, task_id: str) -> list[str]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict) or payload.get("task_id") != task_id:
        return []
    paths: list[str] = []
    report = payload.get("report")
    if isinstance(report, str):
        normalized = _normalize_repo_relative_path(report)
        if (
            Path(normalized).parent.as_posix() == REFEREE_REPORT_DIR.as_posix()
            and Path(normalized).name.startswith(f"{task_id}_")
            and Path(normalized).suffix == ".json"
        ):
            paths.append(normalized)
    revisions = payload.get("revision_tasks")
    if isinstance(revisions, list):
        for revision in revisions:
            if not isinstance(revision, str):
                continue
            normalized = _normalize_repo_relative_path(revision)
            if (
                Path(normalized).parent.as_posix() == ".orchestrator/backlog"
                and re.fullmatch(r"T\d{3}_[A-Za-z0-9_.-]+\.md", Path(normalized).name)
            ):
                paths.append(normalized)
    pending = payload.get("plan_approval_pending")
    if pending == PLAN_APPROVAL_PENDING_PATH.as_posix():
        paths.append(PLAN_APPROVAL_PENDING_PATH.as_posix())
    return sorted(set(paths))


def _persist_referee_outputs(
    *,
    repo: Path,
    task_id: str,
    paths: list[str],
    remote: str,
    base_branch: str,
    strict: bool,
) -> bool:
    if not paths:
        return False
    if _git_current_branch(repo) != base_branch:
        raise SystemExit(
            f"referee_output_root_wrong_branch:{_git_current_branch(repo)}!={base_branch}"
        )
    missing = [path for path in paths if not (repo / path).is_file()]
    if missing:
        raise SystemExit("referee_output_missing:" + ",".join(missing))
    for raw_path in paths:
        path = repo / raw_path
        if path.parent != repo / REFEREE_REPORT_DIR:
            continue
        try:
            report = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"referee_output_invalid:{raw_path}:{type(exc).__name__}") from exc
        run_sha = report.get("run_manifest_sha256") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("task_id") != task_id
            or not isinstance(run_sha, str)
            or not _referee_report_journaled(
                repo,
                task_id=task_id,
                run_manifest_sha256=run_sha,
                actor=report.get("actor"),
                session_id=report.get("session_id"),
            )
        ):
            raise SystemExit(f"referee_output_unjournaled:{raw_path}")
    staged_before_cp = _run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture=True,
        check=True,
    )
    staged_before = {
        line.strip() for line in (staged_before_cp.stdout or "").splitlines() if line.strip()
    }
    unexpected = sorted(staged_before - set(paths))
    if unexpected:
        raise SystemExit(
            "referee_refused_preexisting_staged_changes:" + ",".join(unexpected)
        )
    _run(["git", "add", "--", *paths], cwd=repo, check=True)
    staged = _run(
        ["git", "diff", "--cached", "--name-only", "--", *paths],
        cwd=repo,
        capture=True,
        check=True,
    )
    if not (staged.stdout or "").strip():
        return False
    _git_commit(
        cwd=repo,
        message=f"{task_id}: referee findings",
        strict=strict,
        paths=paths,
    )
    _git_push(
        cwd=repo,
        remote=remote,
        ref=base_branch,
        set_upstream=False,
        strict=strict,
    )
    _record_swarm_event(
        repo,
        {
            "event": "referee_outputs_persisted",
            "task_id": task_id,
            "paths": paths,
            "actor": "RefereeKernel",
            "session_id": _ACTOR_SESSION_ID,
        },
    )
    return True


def _step_referee(
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
    reported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for task_id, context in sorted(contexts.items()):
        manifest_path = context.get("manifest_path")
        worktree = context.get("worktree")
        task = context.get("task")
        if not isinstance(manifest_path, Path) or not isinstance(worktree, Path) or not isinstance(task, Task):
            skipped.append({"task_id": task_id, "reason": "referee_context_incomplete"})
            continue
        if not _referee_task_in_scope(task):
            skipped.append({"task_id": task_id, "reason": "referee_out_of_scope"})
            continue
        manifest_rel = manifest_path.relative_to(worktree).as_posix()
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        existing = _referee_family_votes(
            repo,
            task_id,
            run_manifest_relpath=manifest_rel,
            run_manifest_sha256=manifest_sha,
        )
        requested_family = getattr(args, "referee_family", None)
        if isinstance(requested_family, str) and requested_family:
            desired_families = [requested_family]
        else:
            try:
                framework = json.loads(_read_text(repo / "contracts" / "framework.json"))
            except (OSError, json.JSONDecodeError):
                framework = {}
            executors = framework.get("executors") if isinstance(framework, dict) else None
            panel = executors.get("referee_panel") if isinstance(executors, dict) else None
            configured = [
                str(item.get("family", item.get("backend")))
                for item in panel
                if isinstance(item, dict) and isinstance(item.get("family", item.get("backend")), str)
            ] if isinstance(panel, list) else []
            manifest = context.get("manifest")
            executor = manifest.get("executor") if isinstance(manifest, dict) and isinstance(manifest.get("executor"), dict) else {}
            authoring_family = _referee_family(executor.get("tool"))
            eligible = [family for family in configured if family != authoring_family]
            if (
                _task_is_manuscript_surface(task)
                and len(set(eligible)) < 2
                and _referee_owner_waiver(
                    repo,
                    task_id=task_id,
                    run_manifest_sha256=manifest_sha,
                )
                is None
            ):
                _record_swarm_event(
                    repo,
                    {
                        "event": "referee_panel_unavailable",
                        "task_id": task_id,
                        "run_manifest_sha256": manifest_sha,
                        "configured_non_authoring_families": sorted(set(eligible)),
                        "required_non_authoring_families": 2,
                        "reason": "configured_family_quorum_unavailable",
                    },
                    escalation=True,
                )
            desired_families = eligible if _task_is_manuscript_surface(task) else eligible[:1]
        missing_families = [family for family in desired_families if family not in existing]
        if not missing_families and existing:
            skipped.append({"task_id": task_id, "reason": "referee_report_current"})
            continue
        if not missing_families:
            _record_swarm_event(
                repo,
                {
                    "event": "referee_panel_unavailable",
                    "task_id": task_id,
                    "run_manifest_sha256": manifest_sha,
                    "reason": "no_non_authoring_family_configured",
                },
                escalation=True,
            )
            reported.append({"task_id": task_id, "returncode": 1, "error": "referee_backend_unavailable"})
            continue
        for referee_family in missing_families:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "referee-task",
                "--task",
                task_id,
                "--referee-backend",
                getattr(args, "referee_backend", "mock"),
                "--referee-family",
                referee_family,
                "--remote",
                args.remote,
                "--base-branch",
                args.base_branch,
            ]
            env = dict(os.environ)
            env["SWARM_REPO_ROOT"] = str(worktree)
            env["SWARM_EVENT_REPO_ROOT"] = str(repo)
            env["SWARM_REFEREE_OUTPUT_ROOT"] = str(repo)
            cp = _run(
                command,
                cwd=worktree,
                capture=True,
                check=False,
                env=env,
                timeout_seconds=max(30, int(getattr(args, "referee_timeout_seconds", 900))),
            )
            output_paths = _referee_output_paths_from_stdout(cp.stdout or "", task_id)
            persisted = _persist_referee_outputs(
                repo=repo,
                task_id=task_id,
                paths=output_paths,
                remote=args.remote,
                base_branch=args.base_branch,
                strict=bool(args.unattended),
            )
            reported.append(
                {
                    "task_id": task_id,
                    "referee_family": referee_family,
                    "returncode": cp.returncode,
                    "output_paths": output_paths,
                    "persisted": persisted,
                    "output_tail": (cp.stdout or "")[-2000:],
                }
            )
    return {"reported": reported, "skipped": skipped}


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
                    quality_command = [sys.executable, "scripts/quality_gates.py"]
                    if base_task.task_kind is not None:
                        quality_command.extend(["--task-kind", base_task.task_kind])
                    quality_cp = _run(
                        quality_command,
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
                release_binds = False
                later_epoch = False
                seen_release = False
                for event in events:
                    if event.get("task_id") != task_id:
                        continue
                    name = event.get("event")
                    if name in {"claim_released", "orphan_claim_released"}:
                        released_sha = event.get("sha")
                        if isinstance(released_sha, str) and (
                            released_sha == manifest_claim_sha
                            or _run(
                                ["git", "merge-base", "--is-ancestor", manifest_claim_sha, released_sha],
                                cwd=repo,
                                capture=True,
                                check=False,
                            ).returncode
                            == 0
                        ):
                            release_binds = True
                            seen_release = True
                            later_epoch = False
                    elif name == "claim_created" and seen_release:
                        later_epoch = True
                if not release_binds:
                    fencing_failure = "missing_claim_without_release_record"
                elif later_epoch:
                    fencing_failure = "newer_claim_epoch_after_release"
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
            and not _kernel_namespaced_run_path(task_id, path)
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
        # every byte the binding trusts comes from the COMMITTED tip, never
        # the mutable worktree
        review_rel2 = review_path.relative_to(context["worktree"]).as_posix()
        committed_review_cp = _run(
            ["git", "show", f"{branch}:{review_rel2}"], cwd=repo, capture=True, check=False
        )
        committed_manifest_cp = _run(
            ["git", "show", f"{branch}:{manifest_rel}"], cwd=repo, capture=True, check=False
        )
        if committed_review_cp.returncode != 0 or committed_manifest_cp.returncode != 0:
            continue
        try:
            review = json.loads(committed_review_cp.stdout)
            manifest = json.loads(committed_manifest_cp.stdout)
        except json.JSONDecodeError:
            continue
        reviewed_sha = review.get("reviewed_branch_sha")
        review_manifest_sha = review.get("manifest_sha256")
        actual_manifest_sha = hashlib.sha256(
            committed_manifest_cp.stdout.encode("utf-8")
        ).hexdigest()
        tip_parent = _run(
            ["git", "rev-parse", f"{branch}^"], cwd=repo, capture=True, check=False
        ).stdout.strip()
        binding_failure: str | None = None
        committed_review_sha256 = hashlib.sha256(
            committed_review_cp.stdout.encode("utf-8")
        ).hexdigest()
        events_all, _ = swarm_events.read_events(repo)
        journal_backed = any(
            event.get("event") == "review_recorded"
            and event.get("task_id") == task_id
            and event.get("review_sha256") == committed_review_sha256
            for event in events_all
        )
        if not isinstance(review_manifest_sha, str) or review_manifest_sha != actual_manifest_sha:
            binding_failure = "manifest_content_changed_after_review"
        elif not isinstance(reviewed_sha, str) or reviewed_sha != tip_parent:
            binding_failure = "post_review_commits_present"
        elif not journal_backed:
            # the base journal is outside the executor sandbox: a review
            # without a matching journal record was not written by any judge
            # this control plane ran
            binding_failure = "review_not_journal_backed"
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
                if _kernel_namespaced_run_path(task_id, changed):
                    continue
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

        quality_command = [sys.executable, "scripts/quality_gates.py"]
        if merged_task.task_kind is not None:
            quality_command.extend(["--task-kind", merged_task.task_kind])
        quality_cp = _run(
            quality_command,
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
            task_kind=merged_task.task_kind,
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
                done_tip = _git_head_sha(repo)
                _record_swarm_event(
                    repo,
                    {
                        "event": "merge_cas_push_lost",
                        "task_id": task_id,
                        "expected_tip": done_tip,
                    },
                    escalation=True,
                )
                if _git_head_sha(repo) == done_tip:
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


def _utf8_prefix(value: object, max_bytes: int) -> str:
    return str(value).encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _failure_context_from_manifest(
    manifest: dict[str, object], *, max_bytes: int = 2048
) -> dict[str, object]:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    gates = manifest.get("gates") if isinstance(manifest.get("gates"), list) else []
    blocked_reasons = [
        _utf8_prefix(reason, 128)
        for reason in result.get("blocked_reasons", [])
        if isinstance(reason, str)
    ][:8]
    diagnostics: list[dict[str, object]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("returncode") in {0, None} and not gate.get("timed_out") and not gate.get("constraint_violation"):
            continue
        diagnostic = {
            "command": _utf8_prefix(gate.get("command", ""), 256),
            "returncode": gate.get("returncode"),
            "timed_out": gate.get("timed_out"),
            "constraint_violation": gate.get("constraint_violation"),
            "output_head": _utf8_prefix(gate.get("output_head", ""), 384),
            "output_tail": _utf8_prefix(gate.get("output_tail", ""), 384),
        }
        candidate = {
            "blocked_reasons": blocked_reasons,
            "gate_diagnostics": [*diagnostics, diagnostic],
        }
        if len(json.dumps(candidate, separators=(",", ":"), sort_keys=True).encode("utf-8")) <= max_bytes:
            diagnostics.append(diagnostic)
            continue
        compact = {
            key: value
            for key, value in diagnostic.items()
            if key not in {"output_head", "output_tail"}
        }
        candidate["gate_diagnostics"] = [*diagnostics, compact]
        if len(json.dumps(candidate, separators=(",", ":"), sort_keys=True).encode("utf-8")) <= max_bytes:
            diagnostics.append(compact)

    payload = {
        "blocked_reasons": blocked_reasons,
        "gate_diagnostics": diagnostics,
    }
    while (
        len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        > max_bytes
        and payload["blocked_reasons"]
    ):
        payload["blocked_reasons"].pop()
    return payload


def _repair_context_from_manifest(manifest: dict[str, object]) -> str:
    return json.dumps(
        _failure_context_from_manifest(manifest),
        separators=(",", ":"),
        sort_keys=True,
    )


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


def _planner_marker_on_last_note(task: Task) -> bool:
    try:
        notes = _extract_section(_read_text(task.path), "Notes / Decisions")
    except OSError:
        return False
    if notes is None:
        return False
    lines = [line.strip() for line in notes.splitlines() if line.strip()]
    return bool(lines and "@planner" in lines[-1])


def _manifest_wall_clock_seconds(manifest: dict[str, object]) -> float:
    usage = manifest.get("usage") if isinstance(manifest.get("usage"), dict) else {}
    value = usage.get("wall_clock_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _planner_replan_contexts(repo: Path) -> dict[str, dict[str, object]]:
    contract = load_framework_contract(repo)
    tasks, _ = load_tasks_quarantined(contract)
    contexts: dict[str, dict[str, object]] = {
        task_id: {
            "task": task,
            "contract": contract,
            "manifests": _matching_v2_run_manifest_data(
                _matching_task_jsons(contract.run_manifest_dir, task_id), task_id
            ),
        }
        for task_id, task in tasks.items()
    }
    for task_id, branch_context in _task_branch_contexts(repo).items():
        task = branch_context.get("task")
        branch_contract = branch_context.get("contract")
        if not isinstance(task, Task) or not isinstance(branch_contract, FrameworkContract):
            continue
        contexts[task_id] = {
            "task": task,
            "contract": branch_contract,
            "manifests": _matching_v2_run_manifest_data(
                _matching_task_jsons(branch_contract.run_manifest_dir, task_id), task_id
            ),
        }
    return contexts


def _replan_fingerprint(*, task_id: str, trigger: str, evidence: str) -> str:
    return hashlib.sha256(f"{task_id}|{trigger}|{evidence}".encode("utf-8")).hexdigest()[:16]


def _fired_replan_fingerprints(repo: Path) -> set[str]:
    events, _ = swarm_events.read_events(repo)
    return {
        event["fingerprint"]
        for event in events
        if event.get("event") == "replan_dispatched" and isinstance(event.get("fingerprint"), str)
    }


def _step_plan(args: argparse.Namespace) -> dict[str, object]:
    repo = _repo_root()
    dispatched: list[dict[str, object]] = []
    for task_id, runtime_context in sorted(_planner_replan_contexts(repo).items()):
        task = runtime_context.get("task")
        contract = runtime_context.get("contract")
        manifests = runtime_context.get("manifests")
        if (
            not isinstance(task, Task)
            or not isinstance(contract, FrameworkContract)
            or not isinstance(manifests, list)
        ):
            continue
        typed_manifests = [
            (path, manifest)
            for path, manifest in manifests
            if isinstance(path, Path) and isinstance(manifest, dict)
        ]
        failed = [
            (path, manifest)
            for path, manifest in typed_manifests
            if isinstance(manifest.get("result"), dict)
            and manifest["result"].get("status") == "blocked"
        ]
        wall_clock_seconds = sum(
            _manifest_wall_clock_seconds(manifest) for _, manifest in typed_manifests
        )
        fields = TaskV2Fields(_task_frontmatter(task))
        budget_seconds = (
            parse_wall_clock_seconds(fields.budgets.get("max_wall_clock"))
            if fields.budgets is not None
            else None
        )

        triggers: list[str] = []
        threshold = contract.replan_failure_threshold
        if task.state == "blocked" and failed and len(failed) >= threshold:
            triggers.append("failed_runs")
        if _planner_marker_on_last_note(task):
            triggers.append("planner_marker")
        if budget_seconds is not None and wall_clock_seconds > budget_seconds:
            triggers.append("timebox_exceeded")

        last_manifest = failed[-1][1] if failed else (typed_manifests[-1][1] if typed_manifests else {})
        failure_context = _failure_context_from_manifest(last_manifest)
        # a trigger fingerprint keyed on the CURRENT evidence — the same
        # standing failure/marker does not re-invoke the planner every cycle
        # (C12); a new failed run or a fresh marker changes the fingerprint.
        already_fired = _fired_replan_fingerprints(repo)
        for trigger in triggers:
            marker_note = ""
            if trigger == "planner_marker":
                notes = _extract_section(_read_text(task.path), "Notes / Decisions") or ""
                marker_lines = [ln.strip() for ln in notes.splitlines() if ln.strip()]
                marker_note = marker_lines[-1] if marker_lines else ""
            fingerprint = _replan_fingerprint(
                task_id=task_id,
                trigger=trigger,
                evidence=f"{len(failed)}:{int(wall_clock_seconds)}:{failure_context['blocked_reasons']}:{marker_note}",
            )
            if fingerprint in already_fired:
                continue
            context = {
                "trigger_id": f"{task_id}_{trigger}",
                "trigger": trigger,
                "task_id": task_id,
                "task_file": _read_text(task.path),
                "blocked_reasons": failure_context["blocked_reasons"],
                "last_gate_diagnostics": failure_context["gate_diagnostics"],
                "failed_runs": len(failed),
                "repair_max_attempts": contract.repair_max_attempts,
                "wall_clock_seconds": wall_clock_seconds,
                "max_wall_clock_seconds": budget_seconds,
            }
            _record_swarm_event(
                repo,
                {
                    "event": "replan_dispatched",
                    "trigger": trigger,
                    "task_id": task_id,
                    "fingerprint": fingerprint,
                },
            )
            already_fired.add(fingerprint)
            outcome = _invoke_planner(
                mode="replan", context=context, repo=repo, args=args
            )
            application: dict[str, object] | None = None
            if outcome.returncode == 0:
                application = _apply_planner_proposals(
                    mode="replan",
                    proposals=outcome.proposals,
                    repo=repo,
                    args=args,
                )
            else:
                _record_swarm_event(
                    repo,
                    {
                        "event": "planner_invocation_failed",
                        "mode": "replan",
                        "trigger": trigger,
                        "task_id": task_id,
                        "returncode": outcome.returncode,
                        "stdout": outcome.stdout[:2048],
                    },
                    escalation=True,
                )
            dispatched.append(
                {
                    "task_id": task_id,
                    "trigger": trigger,
                    "returncode": outcome.returncode,
                    "application": application,
                }
            )
    return {"status": "ok", "dispatched": dispatched}


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
        budget_set_but_unverifiable = (
            spend is None and contract.budget_max_program_usd is not None
        )
        _record_swarm_event(
            repo,
            {
                "event": "budget_unverifiable" if budget_set_but_unverifiable else "account_no_data",
                "reason": reason,
                "spend_usd": spend,
                "usage_records": usage_records,
            },
            escalation=budget_set_but_unverifiable,
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
    run_step("REFEREE", _step_referee, args, candidate_ids=judge_candidates)
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
    kernel_events: list[dict[str, object]] = []

    def _capture_event(event: dict[str, object], *, escalation: bool = False) -> None:
        recorded = _record_swarm_event(repo, event, escalation=escalation)
        if isinstance(recorded, dict):
            normalized = dict(recorded)
            normalized.pop("delivery", None)
            normalized.pop("delivery_error", None)
            kernel_events.append(normalized)

    if claim is None or claim.session_id != _ACTOR_SESSION_ID:
        yield kernel_events
        return
    ttl = claim.payload.get("lease_ttl_seconds")
    if isinstance(ttl, int) and ttl > 0:
        interval = max(1, min(int(ttl) // 3, int(ttl * 0.45)) or 1)
    else:
        interval = 1200
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
                    journal=_capture_event,
                )
                if renewed.ok and renewed.sha:
                    current_sha = renewed.sha
                else:
                    _capture_event(
                        {
                            "event": "heartbeat_failed",
                            "task_id": task_id,
                            "reason": renewed.reason,
                        },
                        escalation=True,
                    )
                    return
            except Exception as exc:
                _capture_event(
                    {"event": "heartbeat_failed", "task_id": task_id, "reason": str(exc)},
                    escalation=True,
                )
                return

    thread = threading.Thread(target=_beat, name=f"lease-heartbeat-{task_id}", daemon=True)
    thread.start()
    try:
        yield kernel_events
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

    # The plan-approval hold binds every entrypoint, including a direct
    # run-task of a freshly-planned v2 task (§4.2 mandatory gate).
    if (
        _plan_approval_pending(repo)
        and TaskV2Fields(_task_frontmatter(task)).task_schema == TASK_SCHEMA_VERSION
    ):
        # force_deps overrides DEPENDENCY ordering, never the human plan gate
        raise SystemExit(f"plan_unapproved:{task.task_id}:{PLAN_APPROVAL_PENDING_PATH.as_posix()}")

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

    # §6.1 preregistration boundary — the SAME predicate the supervise-loop
    # funnel and Judge enforce, applied to the direct run-task entrypoint so a
    # manual claim cannot bypass the lock (force_deps overrides dependency
    # ordering only, never the prereg gate). Fails closed on claim AND resume.
    inactive_required = [
        phase
        for phase in _effective_required_active_locks(task, contract)
        if not _prereg_phase_is_active(repo, phase)
    ]
    if inactive_required:
        _record_swarm_event(
            repo,
            {
                "event": "blocked_on_prereg_lock",
                "task_id": task.task_id,
                "phase": inactive_required[0],
                "inactive_required_phases": inactive_required,
                "entrypoint": "run-task",
            },
        )
        raise SystemExit(
            f"blocked_on_prereg_lock:{task.task_id}:{inactive_required[0]}"
        )

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
    executor_control_plane_before: dict[str, bytes] | None = None
    executor_control_plane_written_paths: list[str] = []
    executor_kernel_events: list[dict[str, object]] = []

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
            executor_control_plane_before = _executor_control_plane_snapshot(repo)
            executor_started = time.perf_counter()
            execution_values = dict(vars(args))
            execution_values["_executor_command"] = executor_command
            execution_args = argparse.Namespace(**execution_values)
            with _lease_heartbeat(
                repo=repo,
                remote=args.remote,
                task_id=task.task_id,
            ) as executor_kernel_events:
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
            if executor_control_plane_before is not None:
                executor_control_plane_written_paths = _executor_control_plane_changes(
                    repo=repo,
                    before=executor_control_plane_before,
                    allowed_kernel_events=executor_kernel_events,
                )
                for path in executor_control_plane_written_paths:
                    blocked_reasons.append(f"executor_wrote_control_plane:{path}")
                    # Quarantine the forged control-plane file immediately so it
                    # cannot survive into a retry's "before" snapshot (where an
                    # unchanged forgery would evade re-detection) or a later merge.
                    try:
                        forged = (repo / path).resolve()
                        if repo.resolve() in forged.parents and forged.is_file():
                            forged.unlink()
                            _run(["git", "rm", "-f", "--ignore-unmatch", "--", path], cwd=repo, check=False)
                    except OSError:
                        pass
                    _record_swarm_event(
                        repo,
                        {
                            "event": "executor_wrote_control_plane",
                            "task_id": task.task_id,
                            "path": path,
                        },
                        escalation=True,
                    )
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

    pinned_fields = TaskV2Fields(pinned_frontmatter)
    if (
        args.final_state in {"ready_for_review", "integration_ready"}
        and pinned_fields.task_schema == TASK_SCHEMA_VERSION
        and pinned_fields.complexity_tier in {"M", "L"}
        and pinned_fields.recon_required is True
        and _reconnaissance_line_count(task_text_after_executor) < 3
    ):
        blocked_reasons.append("recon_missing")

    gate_ok, gate_outputs = _run_gates(
        repo,
        task.gates,
        interpreter_allowlist=contract.gate_interpreter_allowlist,
        timeout_seconds=contract.gate_timeout_seconds,
        task_kind=task.task_kind,
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
        kernel_generated_run_paths = {
            path
            for path in (executor_log_relpath, executor_session_relpath)
            if isinstance(path, str)
        }

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
            if (
                changed_path in kernel_generated_run_paths
                or _kernel_namespaced_run_path(task.task_id, changed_path)
            ):
                continue
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
            "executor_control_plane_changes": executor_control_plane_written_paths,
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
    for required_lock in _effective_required_active_locks(task, contract):
        if not _prereg_phase_is_active(repo, required_lock):
            review_bundle_failures.append(f"inactive_prereg_lock:{required_lock}")
            _record_swarm_event(
                repo,
                {
                    "event": "judge_prereg_lock_rejected",
                    "task_id": task.task_id,
                    "required_phase": required_lock,
                },
            )
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

    referee_failures = _referee_review_failures(
        repo=repo,
        task=task,
        run_manifest_path=selected_run_manifest,
    )
    review_bundle_failures.extend(referee_failures)
    for failure in referee_failures:
        if failure.startswith("referee_cannot_verify"):
            _record_swarm_event(
                repo,
                {"event": "referee_cannot_verify", "task_id": task.task_id, "failure": failure},
                escalation=True,
            )

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
        task_kind=task.task_kind,
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
        # the journal (outside the executor sandbox) anchors this review's
        # exact content — the merge queue refuses reviews without it
        "review_sha256": hashlib.sha256(review_log_path.read_bytes()).hexdigest(),
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

    referee_task = subparsers.add_parser(
        "referee-task",
        help="Run a read-only cross-family referee for one task and emit findings JSON",
    )
    referee_task.add_argument("--task", required=True, metavar="T###")
    referee_task.add_argument("--referee-backend", choices=["mock", "claude"], default="mock")
    referee_task.add_argument("--referee-family", default=None)
    referee_task.add_argument("--remote", default="origin")
    referee_task.add_argument("--base-branch", default="main")
    referee_task.add_argument("--timeout-seconds", type=int, default=900)
    referee_task.set_defaults(func=cmd_referee_task)

    referee_waiver = subparsers.add_parser(
        "referee-waiver",
        help="Journal a human waiver reducing manuscript family quorum to one",
    )
    referee_waiver.add_argument("--task", required=True, metavar="T###")
    referee_waiver.add_argument("--human-id", required=True)
    referee_waiver.add_argument("--reason", required=True)
    referee_waiver.add_argument("--base-branch", default="main")
    referee_waiver.set_defaults(func=cmd_referee_waiver)

    plan = subparsers.add_parser("plan", help="Print done/claimed/ready task status as JSON")
    plan.add_argument("--remote", default="origin")
    plan.add_argument("--base-branch", default="main")
    plan.set_defaults(func=cmd_plan)

    plan_program = subparsers.add_parser(
        "plan-program",
        help="Invoke the bounded Planner launch pass and require human approval",
    )
    plan_program.add_argument("--planner-backend", choices=["mock", "claude"], default="mock")
    plan_program.add_argument("--remote", default="origin")
    plan_program.add_argument("--base-branch", default="main")
    plan_program.add_argument("--unattended", action="store_true")
    plan_program.set_defaults(func=cmd_plan_program)

    triage = subparsers.add_parser(
        "triage", help="Invoke Planner pre-launch triage for one backlog task"
    )
    triage.add_argument("--task", required=True, metavar="T###")
    triage.add_argument("--planner-backend", choices=["mock", "claude"], default="mock")
    triage.add_argument("--remote", default="origin")
    triage.add_argument("--base-branch", default="main")
    triage.add_argument("--unattended", action="store_true")
    triage.set_defaults(func=cmd_triage)

    approve_plan = subparsers.add_parser(
        "approve-plan", help="Record human approval of the current launch plan"
    )
    approve_plan.add_argument("--approved-by", required=True)
    approve_plan.set_defaults(func=cmd_approve_plan)

    lock_prereg = subparsers.add_parser(
        "lock-prereg",
        help="Hash and activate a phased preregistration lock (L3 human gate)",
    )
    lock_prereg.add_argument(
        "--phase",
        choices=["2a", "2b", "lock_a", "lock_b"],
        required=True,
    )
    lock_prereg.add_argument("--locked-by", required=True, help="Human lock approver")
    lock_prereg.add_argument(
        "--amend",
        action="store_true",
        help="Explicitly amend an already-active lock and increment its version",
    )
    lock_prereg.set_defaults(func=cmd_lock_prereg)

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
    supervise.add_argument("--planner-backend", choices=["mock", "claude"], default="mock")
    supervise.add_argument("--referee-backend", choices=["mock", "claude"], default="mock")
    supervise.add_argument("--referee-family", default=None)
    supervise.add_argument("--referee-timeout-seconds", type=int, default=900)
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
