from __future__ import annotations

import contextlib
import copy
import importlib.util
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SWARM_PATH = REPO_ROOT / "scripts" / "swarm.py"
QUALITY_GATES_PATH = REPO_ROOT / "scripts" / "quality_gates.py"
SWEEP_TASKS_PATH = REPO_ROOT / "scripts" / "sweep_tasks.py"

SWARM_RUN_MANIFEST_SCHEMA_VERSION = "research_swarm.runtime_run_manifest.v2"
SWARM_RUN_MANIFEST_SCHEMA_VERSION_V1 = "research_swarm.runtime_run_manifest.v1"
JUDGE_REVIEW_LOG_SCHEMA_VERSION = "research_swarm.judge_review_log.v2"
JUDGE_REVIEW_LOG_SCHEMA_VERSION_V1 = "research_swarm.judge_review_log.v1"


def _git_read_or_default(root: Path, args: list[str], default: str) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        value = cp.stdout.strip()
        return value or default
    except Exception:
        return default


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def load_swarm_module():
    return _load_module("stage4_swarm_module", SWARM_PATH)


@lru_cache(maxsize=None)
def load_quality_gates_module():
    return _load_module("stage4_quality_gates_module", QUALITY_GATES_PATH)


@lru_cache(maxsize=None)
def load_sweep_module():
    return _load_module("stage4_sweep_module", SWEEP_TASKS_PATH)


@contextlib.contextmanager
def chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def write_text(root: Path, rel: str, text: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(root: Path, rel: str, data: object) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def mkdir(root: Path, rel: str) -> Path:
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_project_yaml(root: Path, *, mode: str = "empirical") -> Path:
    return write_text(
        root,
        "contracts/project.yaml",
        "\n".join(
            [
                "project_id: test-project",
                "project_name: test-project",
                f"mode: {mode}",
                "status: test",
                "",
            ]
        ),
    )


def _default_framework_json(mode: str) -> dict[str, Any]:
    return {
        "framework_version": "v1",
        "features": {
            "registry": True,
            "etl": True,
            "analysis": True,
            "validation": True,
            "modeling": True,
            "hybrid": True,
            "paper": True,
            "release": True,
        },
        "roles": {
            "allowed": ["Planner", "Worker", "Judge", "Operator"],
            "task_execution_roles": ["Worker", "Operator"],
            "scientific_review_role": "Judge",
        },
        "states": {
            "allowed": [
                "backlog",
                "active",
                "integration_ready",
                "ready_for_review",
                "blocked",
                "done",
            ],
            "projection_dirs": [
                ".orchestrator/backlog",
                ".orchestrator/active",
                ".orchestrator/integration_ready",
                ".orchestrator/ready_for_review",
                ".orchestrator/blocked",
                ".orchestrator/done",
            ],
        },
        "prompt_templates": {
            "planner": "docs/prompts/planner.md",
            "worker": "docs/prompts/worker.md",
            "judge": "docs/prompts/judge.md",
            "operator": "docs/prompts/operator.md",
        },
        "network_workstreams": ["W1", "W2", "W3"],
        "integration_ready_policy": {
            "eligible_workstreams": ["W0", "W3", "W8", "W9"],
            "eligible_task_kinds": ["protocol", "registry", "bridge", "model", "ops"],
            "forbid_unvalidated_empirical_data_outputs": True,
        },
        "review_bundle": {
            "run_manifest_dir": "reports/status/swarm_runs",
            "judge_review_dir": "reports/status/reviews",
        },
        "operator_owned_shared_surfaces": [
            "reports/catalog.yaml",
            "reports/paper/build/",
            "reports/status/releases/",
        ],
        "release_policy": {
            "release_manifest_pattern": "reports/status/releases/release_<YYYY-MM-DD>.json",
        },
        "required_paths": {
            "common": ["reports/status/"],
            mode: [
                "docs/protocol.md",
                "registry/rollup_registry_v1.csv",
                "data/raw_manifest/",
                "data/processed_manifest/",
                "reports/validation/",
                "reports/figures/",
                "reports/tables/",
                "reports/paper/",
            ],
        },
    }


def write_framework_json(
    root: Path,
    *,
    mode: str = "empirical",
    overrides: dict[str, Any] | None = None,
) -> Path:
    payload = copy.deepcopy(_default_framework_json(mode))
    if overrides:
        for key, value in overrides.items():
            payload[key] = value
    return write_json(root, "contracts/framework.json", payload)


def scaffold_runtime_repo(root: Path, *, mode: str = "empirical") -> None:
    write_text(root, "AGENTS.md", "# root\n")
    write_text(root, "README.md", "# repo\n")

    mkdir(root, ".orchestrator/backlog")
    mkdir(root, ".orchestrator/active")
    mkdir(root, ".orchestrator/integration_ready")
    mkdir(root, ".orchestrator/ready_for_review")
    mkdir(root, ".orchestrator/blocked")
    mkdir(root, ".orchestrator/done")
    mkdir(root, ".orchestrator/handoff")
    mkdir(root, ".orchestrator/templates")
    write_text(root, ".orchestrator/README.md", "# orchestrator\n")
    write_text(root, ".orchestrator/AGENTS.md", "# orchestrator agents\n")
    write_text(
        root,
        ".orchestrator/workstreams.md",
        "\n".join(
            [
                "# Workstreams",
                "",
                "| Workstream | Purpose | Owns paths | Does NOT own | Example outputs | Network | integration_ready eligible |",
                "|---|---|---|---|---|---|---|",
                "| W0 | Protocol/contracts | docs/, contracts/ | src/ | contract edits | no | yes |",
                "| W1 | Off-chain ETL | src/etl/, data/raw/, data/processed/ | reports/paper/ | vendor panel | yes | no |",
                "| W8 | Bridge/model | src/model/, contracts/instances/, reports/models/ | data/raw/ | instance manifests | no | yes |",
                "| W9 | Ops/release | reports/status/, reports/paper/build/ | protocol | run/review logs | no | yes |",
                "",
            ]
        ),
    )

    mkdir(root, "contracts")
    mkdir(root, "contracts/schemas")
    write_text(root, "contracts/README.md", "# contracts\n")
    write_text(root, "contracts/data_dictionary.md", "# dictionary\n")
    write_text(root, "contracts/decisions.md", "# decisions\n")
    write_text(root, "contracts/model_spec.md", "# model spec\n")
    write_text(root, "contracts/hybrid_interface_v1.yaml", "version: 1\n")
    write_text(root, "contracts/schemas/README.md", "# schemas\n")
    write_text(root, "contracts/schemas/panel_schema.yaml", "version: 1\n")
    write_text(root, "contracts/schemas/panel_schema_str_v1.yaml", "version: 1\nfields: []\n")
    write_text(root, "contracts/schemas/panel_schema_decomp_v1.yaml", "version: 1\nfields: []\n")
    write_text(root, "contracts/schemas/swarm_run_manifest_v1.yaml", "version: 1\nartifact: swarm_run_manifest\n")
    write_text(root, "contracts/schemas/judge_review_log_v1.yaml", "version: 1\nartifact: judge_review_log\n")
    write_project_yaml(root, mode=mode)
    write_framework_json(root, mode=mode)

    mkdir(root, "docs/prompts")
    write_text(
        root,
        "docs/protocol.md",
        "\n".join(
            [
                "# Protocol",
                "",
                "## Research mode",
                "",
                "- Mode: empirical",
                "",
                "## Primary metric",
                "",
                "- Name: STR",
                "- Formula: fees / rent",
                "- Units: unitless",
                "",
                "## Rollup inclusion criteria",
                "",
                "Some content.",
                "",
                "## Data source priority",
                "",
                "Some content.",
                "",
                "## Known regime dates",
                "",
                "Some content.",
                "",
                "## Validation tolerances",
                "",
                "Some content.",
                "",
            ]
        ),
    )
    write_text(root, "docs/runbook_swarm.md", "# runbook\n")
    write_text(root, "docs/runbook_swarm_automation.md", "# automation runbook\n")
    write_text(root, "docs/prompts/planner.md", "# planner prompt\n")
    write_text(root, "docs/prompts/worker.md", "# worker prompt\n")
    write_text(root, "docs/prompts/judge.md", "# judge prompt\n")
    write_text(root, "docs/prompts/operator.md", "# operator prompt\n")

    mkdir(root, "data/raw_manifest")
    mkdir(root, "data/processed_manifest")
    mkdir(root, "data/samples")
    write_text(root, "data/raw_manifest/README.md", "# raw manifests\n")
    write_text(root, "data/processed_manifest/README.md", "# processed manifests\n")
    write_text(root, "data/samples/README.md", "# samples\n")

    mkdir(root, "reports/validation/manifests")
    mkdir(root, "reports/figures")
    mkdir(root, "reports/tables")
    write_text(root, "reports/validation/README.md", "# validation\n")
    write_text(root, "reports/validation/manifests/README.md", "# validation manifests\n")
    write_text(root, "reports/figures/README.md", "# figures\n")
    write_text(root, "reports/tables/README.md", "# tables\n")

    mkdir(root, "registry")
    write_text(root, "registry/README.md", "# registry\n")
    write_text(root, "registry/CHANGELOG.md", "# registry changelog\n")
    write_text(
        root,
        "registry/rollup_registry_v1.csv",
        "index,rollup_id,display_name,type,da_posting_method,batcher_addresses_json,evidence_url,verified_utc,status,start_date_utc,end_date_utc,notes\n",
    )

    mkdir(root, "scripts")
    write_text(root, "scripts/swarm.py", "# placeholder\n")
    write_text(root, "scripts/sweep_tasks.py", "# placeholder\n")
    write_text(root, "scripts/quality_gates.py", "# placeholder\n")

    mkdir(root, "tests")
    write_text(root, "tests/README.md", "# tests\n")


def init_git_fixture_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "swarm-bot"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "swarm-bot@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "initial fixture"], cwd=root, check=True, capture_output=True, text=True)

    origin = Path(f"{root}.origin.git")
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=root, check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=root, check=True, capture_output=True, text=True)


def _emit_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []"
    lines = [f"{key}:"]
    lines.extend(f'  - "{value}"' for value in values)
    return "\n".join(lines)


def write_task(
    root: Path,
    folder: str,
    task_id: str,
    *,
    title: str | None = None,
    workstream: str = "W1",
    task_kind: str | None = "etl",
    role: str = "Worker",
    priority: str = "medium",
    dependencies: list[str] | None = None,
    integration_ready_dependencies: list[str] | None = None,
    allow_network: bool = False,
    allowed_paths: list[str] | None = None,
    disallowed_paths: list[str] | None = None,
    outputs: list[str] | None = None,
    gates: list[str] | None = None,
    stop_conditions: list[str] | None = None,
    state: str = "backlog",
    last_updated: str = "2026-03-29",
    slug: str = "task",
) -> Path:
    title = title or f"{task_id} title"
    dependencies = dependencies or []
    integration_ready_dependencies = integration_ready_dependencies or []
    allowed_paths = allowed_paths or ["src/"]
    disallowed_paths = disallowed_paths or ["contracts/"]
    outputs = outputs or ["src/example.py"]
    gates = gates or ["make gate"]
    stop_conditions = stop_conditions or ["Need @human"]

    frontmatter = "\n".join(
        [
            "---",
            f'task_id: "{task_id}"',
            f'title: "{title}"',
            f'workstream: "{workstream}"',
            f'task_kind: "{task_kind or ""}"',
            f"allow_network: {'true' if allow_network else 'false'}",
            f'role: "{role}"',
            f'priority: "{priority}"',
            _emit_list("dependencies", dependencies),
            _emit_list("integration_ready_dependencies", integration_ready_dependencies),
            _emit_list("allowed_paths", allowed_paths),
            _emit_list("disallowed_paths", disallowed_paths),
            _emit_list("outputs", outputs),
            _emit_list("gates", gates),
            _emit_list("stop_conditions", stop_conditions),
            "---",
        ]
    )

    body = "\n".join(
        [
            f"# Task {task_id} — {title}",
            "",
            "## Context",
            "",
            "Context.",
            "",
            "## Inputs",
            "",
            "- input",
            "",
            "## Outputs",
            "",
            "- output",
            "",
            "## Success Criteria",
            "",
            "- [ ] done",
            "",
            "## Review Bundle Requirements",
            "",
            "- [ ] run manifest",
            "",
            "## Validation / Commands",
            "",
            "- `make gate`",
            "",
            "## Status",
            "",
            f"- State: {state}",
            f"- Last updated: {last_updated}",
            "",
            "## Notes / Decisions",
            "",
            "- 2026-03-29: note",
            "",
        ]
    )

    rel = f".orchestrator/{folder}/{task_id}_{slug}.md"
    return write_text(root, rel, frontmatter + "\n" + body)


def write_run_manifest(
    root: Path,
    task_id: str,
    *,
    task_path: str,
    task_role: str = "Worker",
    workstream: str = "W1",
    state_before: str = "active",
    state_after: str = "ready_for_review",
    provenance_class: str = "executor_run",
    result_status: str = "ok",
    schema_version: str = SWARM_RUN_MANIFEST_SCHEMA_VERSION,
    branch: str | None = None,
    git_sha: str | None = None,
    actor_session_id: str = "fixture-worker-session",
    generated_at_utc: str = "2026-03-29T00:00:00Z",
    usage: dict[str, object] | None = None,
) -> Path:
    task_disk_path = root / task_path
    pinned_frontmatter_sha = "0" * 64
    task_gates = ["make gate"]
    if task_disk_path.is_file():
        lines = task_disk_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            for index in range(1, len(lines)):
                if lines[index].strip() == "---":
                    block = "".join(lines[1:index])
                    pinned_frontmatter_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()
                    break
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import swarm_taskfile as _taskfile

        frontmatter = _taskfile.parse_task_frontmatter(task_disk_path.read_text(encoding="utf-8"))
        if isinstance(frontmatter, dict) and isinstance(frontmatter.get("gates"), list):
            parsed_gates = [item for item in frontmatter["gates"] if isinstance(item, str)]
            if parsed_gates:
                task_gates = parsed_gates
    if branch is None:
        branch = _git_read_or_default(root, ["rev-parse", "--abbrev-ref", "HEAD"], f"{task_id}_branch")
    if git_sha is None:
        git_sha = _git_read_or_default(
            root, ["rev-parse", "HEAD"], "0123456789abcdef0123456789abcdef01234567"
        )
    rel = f"reports/status/swarm_runs/{task_id}_20260329T000000Z.json"
    payload = {
        "schema_version": schema_version,
        "run_id": f"{task_id}_20260329T000000Z",
        "generated_at_utc": generated_at_utc,
        "task": {
            "task_id": task_id,
            "task_path": task_path,
            "title": f"{task_id} title",
            "role": task_role,
            "workstream": workstream,
            "task_kind": "etl",
            "dependencies": [],
            "integration_ready_dependencies": [],
            "state_before": state_before,
            "state_after": state_after,
        },
        "repo": {
            "branch": branch,
            "git_sha": git_sha,
            "base_branch": "main",
            "remote": "origin",
        },
        "executor": {
            "role": task_role,
            "runner": "local_swarm",
            "tool": "manual",
            "model": None,
            "sandbox": "workspace-write",
            "allow_network": False,
            "repair_context": None,
            "returncode": 0,
            "error": "executor_skipped",
        },
        "commands": {
            "executor": [],
            "executor_log_path": None,
            "gates": list(task_gates),
        },
        "gates": [
            {
                "command": gate,
                "returncode": 0,
                "output_tail": "",
            }
            for gate in task_gates
        ],
        "ownership": {
            "ok": True,
            "changed_paths": [task_path],
            "violations": [],
        },
        "artifacts": {
            "outputs_ok": True,
            "missing_outputs": [],
            "required_manifests_ok": True,
            "missing_manifests": [],
        },
        "result": {
            "status": result_status,
            "blocked_reasons": [] if result_status == "ok" else ["fixture_blocked"],
        },
    }
    if schema_version == SWARM_RUN_MANIFEST_SCHEMA_VERSION:
        payload["provenance_class"] = provenance_class
        payload["actor"] = {
            "session_id": actor_session_id,
            "recorded_at_utc": "2026-03-29T00:00:00Z",
        }
        if provenance_class == "executor_run":
            log_rel = f"reports/status/swarm_runs/logs/{task_id}_20260329T000000Z.log"
            log_path = write_text(root, log_rel, "fixture executor log\n")
            payload["commands"]["executor_log_path"] = log_rel
            payload["commands"]["executor_log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
        else:
            payload["commands"]["executor_log_sha256"] = None
        payload["frontmatter"] = {
            "pinned_sha256": pinned_frontmatter_sha,
            "tampered": False,
            "tampered_keys": [],
        }
        payload["ownership"]["uncommitted_violations"] = []
        if usage is not None:
            payload["usage"] = copy.deepcopy(usage)
    return write_json(root, rel, payload)


def write_review_log(
    root: Path,
    task_id: str,
    *,
    task_path: str,
    run_manifest_path: str,
    task_role: str = "Worker",
    reviewer_role: str = "Judge",
    state_before: str = "ready_for_review",
    state_after: str = "done",
    outcome: str = "approve",
    schema_version: str = JUDGE_REVIEW_LOG_SCHEMA_VERSION,
    reviewer_session_id: str = "fixture-judge-session",
    generated_at_utc: str = "2026-03-29T01:00:00Z",
    manifest_sha256: str | None = None,
    reviewed_branch_sha: str | None = None,
) -> Path:
    rel = f"reports/status/reviews/{task_id}_20260329T010000Z.json"
    reviewer: dict[str, Any] = {"role": reviewer_role}
    if schema_version == JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        reviewer["session_id"] = reviewer_session_id
        reviewer["recorded_at_utc"] = generated_at_utc
    payload = {
        "schema_version": schema_version,
        "review_id": f"{task_id}_20260329T010000Z",
        "generated_at_utc": generated_at_utc,
        "manifest_sha256": manifest_sha256,
        "reviewed_branch_sha": reviewed_branch_sha,
        "reviewer": reviewer,
        "task": {
            "task_id": task_id,
            "task_path": task_path,
            "role": task_role,
            "state_before": state_before,
            "state_after": state_after,
            "run_manifest_path": run_manifest_path,
        },
        "checks": {
            "gates_ok": True,
            "outputs_ok": True,
            "required_manifests_ok": True,
            "review_bundle_ok": True,
            "failures": [],
        },
        "decision": {
            "outcome": outcome,
            "note": "review note",
        },
    }
    if schema_version == JUDGE_REVIEW_LOG_SCHEMA_VERSION:
        payload["operator_attestation"] = None
    return write_json(root, rel, payload)


def register_historical_exemption(root: Path, *, section: str, rel_path: str, extra: dict[str, Any] | None = None) -> Path:
    """Add (or refresh) a hash-pinned entry in the fixture's historical
    exemption list, creating the file when absent."""
    exemptions_path = root / "contracts/historical_exemptions.json"
    if exemptions_path.exists():
        payload = json.loads(exemptions_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": "research_swarm.historical_exemptions.v1",
            "created_at_utc": "2026-07-09T00:00:00Z",
            "rationale": "test fixture",
            "run_manifests": [],
            "review_logs": [],
            "processed_manifests": [],
            "raw_manifests": [],
            "validation_reports": [],
            "rebaselines": [],
        }
    entries = payload.setdefault(section, [])
    entries[:] = [item for item in entries if item.get("path") != rel_path]
    entry: dict[str, Any] = {
        "path": rel_path,
        "sha256": hashlib.sha256((root / rel_path).read_bytes()).hexdigest(),
        "schema_version": "fixture",
    }
    if extra:
        entry.update(extra)
    entries.append(entry)
    return write_json(root, "contracts/historical_exemptions.json", payload)


def attest_containment_fixture(root: Path) -> None:
    """Write the machine-local containment marker + vendor ack an unattended
    fixture needs (pair with a patched clean HOME so the credential scan is
    hermetic)."""
    write_json(
        root,
        ".swarm/containment.json",
        {
            "schema_version": "research_swarm.containment_marker.v1",
            "contained": True,
            "attested_by": "fixture",
            "attested_at_utc": "2026-07-10T00:00:00Z",
            "note": "test fixture",
            "credential_scan_waiver": [
                "aws_credentials",
                "ssh_private_key",
                "gcloud_adc",
                "netrc",
                "docker_auth",
            ],
        },
    )
    write_json(
        root,
        ".swarm/vendor_policy_ack.json",
        {
            "schema_version": "research_swarm.vendor_policy_ack.v1",
            "vendor": "codex",
            "policy_note": "fixture ack",
            "acked_by": "fixture",
            "acked_at_utc": "2026-07-10T00:00:00Z",
        },
    )
