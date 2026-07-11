from __future__ import annotations

import contextlib
import copy
import importlib.util
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
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


def instantiate_program_fixture(
    root: Path,
    task_path: Path,
    *,
    mode: str = "empirical",
    task_kind: str,
    role: str,
    workstream: str,
) -> None:
    """Bind one existing fixture task to a minimal valid mode program."""
    schema_target = root / "contracts/schemas/program_template_v1.yaml"
    schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "contracts/schemas/program_template_v1.yaml", schema_target)
    write_json(
        root,
        f"contracts/program_templates/{mode}.yaml",
        {
            "program_template": "research_swarm.program_template.v1",
            "mode": mode,
            "programs": [
                {
                    "program_id": "release_fixture",
                    "workstream": workstream,
                    "nodes": [
                        {
                            "node_id": "release_ready",
                            "task_kind": task_kind,
                            "role": role,
                            "produces": "release_fixture_evidence",
                            "depends_on": [],
                            "required": True,
                        }
                    ],
                }
            ],
        },
    )
    text = task_path.read_text(encoding="utf-8")
    marker = "\n---\n"
    if marker not in text:
        raise ValueError(f"program_fixture_task_frontmatter_missing:{task_path}")
    frontmatter, body = text.split(marker, 1)
    frontmatter += '\nprogram_id: "release_fixture"\nprogram_node: "release_ready"'
    task_path.write_text(frontmatter + marker + body, encoding="utf-8")


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
        "citation_policy": {"staleness_days": 90},
        "complexity_tier_ceilings": {
            "S": {
                "max_wall_clock_seconds": 7200,
                "max_tokens": 250000,
                "max_cost_usd": 25,
            },
            "M": {
                "max_wall_clock_seconds": 28800,
                "max_tokens": 1000000,
                "max_cost_usd": 100,
            },
            "L": {
                "max_wall_clock_seconds": 86400,
                "max_tokens": 4000000,
                "max_cost_usd": 400,
            },
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
            "reports/status/swarm_runs/",
            "reports/status/referee_reports/",
            "reports/status/referee_calibration.json",
            "reports/status/referee_calibration_runs/",
            "reports/status/events/",
            "reports/status/integrity_audit/",
            "reports/status/recall_audit/",
        ],
        "literature_policy": {
            "recall_uncovered_cluster_threshold": 2,
            "fixture_test_corpus_acquisition_ids": [],
        },
        "referee_panel": {
            "required_non_authoring_families": 2,
            "owner_waiver": None,
        },
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
        "executors": {
            "integrity_audit": {
                "backend": "claude",
                "family": "claude",
                "command": "claude",
                "model": "fixture-auditor",
                "profile": "scratch-worktree",
            },
            "planner": {
                "backend": "claude",
                "command": "claude",
                "model": "fixture-planner",
                "profile": "read-only+backlog-write",
            },
            "referee_panel": [
                {
                    "backend": "claude",
                    "family": "claude",
                    "command": "claude",
                    "model": "fixture-referee",
                    "cli_version": "fixture-cli-1",
                    "profile": "read-only",
                    "prompt_path": "docs/prompts/referee.md",
                    "tools": ["Read", "Glob", "Grep"],
                }
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
    pack = json.loads((REPO_ROOT / "contracts/pack.json").read_text(encoding="utf-8"))
    pack["project"]["package_name"] = "test-project"
    pack["project"]["primary_metric_label"] = "Fixture metric"
    pack["analysis"]["outputs"] = {
        key: value.replace("str_", "fixture_")
        for key, value in pack["analysis"]["outputs"].items()
    }
    if overrides and isinstance(overrides.get("network_workstreams"), list):
        pack["workflow"]["network_workstreams"] = overrides["network_workstreams"]
    integration = overrides.get("integration_ready_policy") if overrides else None
    if isinstance(integration, dict) and isinstance(integration.get("eligible_workstreams"), list):
        pack["workflow"]["integration_ready_eligible_workstreams"] = integration["eligible_workstreams"]
    write_json(root, "contracts/pack.json", pack)
    for rel in (
        "contracts/kernel_interface.json",
        "contracts/schemas/kernel_interface_v1.json",
        "contracts/schemas/pack_config_v1.json",
    ):
        write_text(root, rel, (REPO_ROOT / rel).read_text(encoding="utf-8"))
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
    mkdir(root, "contracts/prompts")
    mkdir(root, "contracts/instances")
    mkdir(root, "contracts/experiments")
    write_text(root, "contracts/README.md", "# contracts\n")
    write_text(root, "contracts/data_dictionary.md", "# dictionary\n")
    write_text(root, "contracts/decisions.md", "# decisions\n")
    write_text(root, "contracts/model_spec.md", "# model spec\n")
    for rel in (
        "contracts/hybrid_interface_v1.yaml",
        "contracts/schemas/instance_manifest_v1.json",
        "contracts/schemas/experiment_spec_v1.json",
        "contracts/schemas/experiment_manifest_v1.json",
        "contracts/schemas/raw_manifest_v1.json",
        "contracts/schemas/processed_manifest_v2.json",
        "contracts/schemas/integrity_audit_v1.json",
        "contracts/schemas/literature_manifest_v1.json",
    ):
        write_text(root, rel, (REPO_ROOT / rel).read_text(encoding="utf-8"))
    write_text(root, "contracts/instances/README.md", "# instance manifests\n")
    write_text(root, "contracts/experiments/README.md", "# experiment specs\n")
    write_text(root, "contracts/schemas/README.md", "# schemas\n")
    for rel in (
        "contracts/schemas/panel_schema.yaml",
        "contracts/schemas/panel_schema_str_v1.yaml",
        "contracts/schemas/panel_schema_decomp_v1.yaml",
        "contracts/schemas/swarm_run_manifest_v1.yaml",
        "contracts/schemas/swarm_run_manifest_v2.json",
        "contracts/schemas/judge_review_log_v1.yaml",
        "contracts/schemas/judge_review_log_v2.json",
    ):
        write_text(root, rel, (REPO_ROOT / rel).read_text(encoding="utf-8"))
    write_json(
        root,
        "contracts/claims.yaml",
        {
            "schema_version": "research_swarm.claims.v1",
            "description": "Empty test claim ledger.",
            "claims": [],
        },
    )
    write_text(root, "contracts/schemas/claims_v1.yaml", (REPO_ROOT / "contracts/schemas/claims_v1.yaml").read_text(encoding="utf-8"))
    write_text(
        root,
        "contracts/integrity_audit_seed.txt",
        (REPO_ROOT / "contracts/integrity_audit_seed.txt").read_text(encoding="utf-8"),
    )
    for prompt_path in sorted((REPO_ROOT / "contracts/prompts").glob("*")):
        if prompt_path.is_file():
            write_text(
                root,
                f"contracts/prompts/{prompt_path.name}",
                prompt_path.read_text(encoding="utf-8"),
            )
    for rubric_path in sorted((REPO_ROOT / "contracts/rubrics").glob("*")):
        if rubric_path.is_file():
            write_text(
                root,
                f"contracts/rubrics/{rubric_path.name}",
                rubric_path.read_text(encoding="utf-8"),
            )
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
                f"- Mode: {mode}",
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
    write_text(
        root,
        "docs/prompts/referee.md",
        (REPO_ROOT / "docs/prompts/referee.md").read_text(encoding="utf-8"),
    )
    write_text(
        root,
        "docs/prereg/data_construction.lock.md",
        "\n".join(
            [
                "---",
                "schema_version: research_swarm.prereg_lock.v1",
                "phase: 2a",
                "status: draft",
                "locked_at_utc: null",
                "locked_sha256: null",
                "locked_by: null",
                "lock_version: 0",
                "---",
                "",
                "# Data construction",
                "",
            ]
        ),
    )
    write_text(
        root,
        "docs/prereg/analysis_plan.lock.md",
        "\n".join(
            [
                "---",
                "schema_version: research_swarm.prereg_lock.v1",
                "phase: 2b",
                "status: draft",
                "locked_at_utc: null",
                "locked_sha256: null",
                "locked_by: null",
                "lock_version: 0",
                "---",
                "",
                "# Analysis plan",
                "",
            ]
        ),
    )
    for phase in ("lock_a", "lock_b"):
        title = "Lock A" if phase == "lock_a" else "Lock B"
        write_text(
            root,
            f"docs/prereg/{phase}.md",
            "\n".join(
                [
                    "---",
                    "schema_version: research_swarm.prereg_lock.v1",
                    f"phase: {phase}",
                    "status: draft",
                    "locked_at_utc: null",
                    "locked_sha256: null",
                    "locked_by: null",
                    "lock_version: 0",
                    "---",
                    "",
                    f"# {title}",
                    "",
                ]
            ),
        )
    write_json(
        root,
        "docs/prereg/outcomes.yaml",
        {"schema_version": "research_swarm.prereg_outcomes.v1", "outcomes": []},
    )

    mkdir(root, "data/raw_manifest")
    mkdir(root, "data/processed_manifest")
    mkdir(root, "data/samples")
    mkdir(root, "data/citations")
    write_text(root, "data/raw_manifest/README.md", "# raw manifests\n")
    write_text(root, "data/processed_manifest/README.md", "# processed manifests\n")
    write_text(root, "data/samples/README.md", "# samples\n")
    write_text(root, "data/citations/README.md", "# citation snapshots\n")

    mkdir(root, "reports/validation/manifests")
    mkdir(root, "reports/figures")
    mkdir(root, "reports/tables")
    mkdir(root, "reports/analysis/spec_curve")
    mkdir(root, "reports/analysis/sensitivity")
    mkdir(root, "reports/models/sweeps")
    write_text(root, "reports/validation/README.md", "# validation\n")
    write_text(root, "reports/validation/manifests/README.md", "# validation manifests\n")
    write_text(root, "reports/figures/README.md", "# figures\n")
    write_text(root, "reports/tables/README.md", "# tables\n")
    write_text(root, "reports/analysis/README.md", "# analysis artifacts\n")
    write_text(root, "reports/analysis/spec_curve/README.md", "# specification curves\n")
    write_text(root, "reports/analysis/sensitivity/README.md", "# sensitivity artifacts\n")
    write_text(root, "reports/models/README.md", "# models\n")
    write_text(root, "reports/models/sweeps/README.md", "# sweeps\n")

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
    write_text(root, "scripts/integrity_audit.py", "# placeholder\n")
    write_text(root, "scripts/literature.py", "# placeholder\n")
    write_text(root, "scripts/refresh_citations.py", "# placeholder\n")
    write_text(root, "scripts/falsify_claims.py", "# placeholder\n")
    write_text(root, "scripts/sweep_harness.py", "# placeholder\n")
    write_text(root, "scripts/noop_gate.py", "import sys\nif __name__ == '__main__':\n    sys.exit(0)\n")

    mkdir(root, "tests")
    write_text(root, "tests/README.md", "# tests\n")


def init_git_fixture_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "swarm-bot"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "swarm-bot@example.invalid"], cwd=root, check=True)
    # background maintenance detaches and keeps writing .git after tests end,
    # racing TemporaryDirectory teardown on fast runners
    subprocess.run(["git", "config", "gc.auto", "0"], cwd=root, check=True)
    subprocess.run(["git", "config", "gc.autoDetach", "false"], cwd=root, check=True)
    subprocess.run(["git", "config", "maintenance.auto", "false"], cwd=root, check=True)
    gold_key = root / "tests/gold_set/verdict_key.json"
    if gold_key.is_file():
        # Calibration authority requires the bar to be a proper ancestor of
        # the first grading artifact. Keep that topology real in fixtures.
        subprocess.run(
            ["git", "add", "-A", "--", ".", ":(exclude)tests/gold_set/**"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial fixture with calibration bar"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["git", "add", "-A", "--", "tests/gold_set"], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add grading artifacts"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
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


def _emit_unquoted_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []"
    return "\n".join([f"{key}:", *(f"  - {value}" for value in values)])


def _emit_mapping_list(key: str, values: list[dict[str, object]]) -> str:
    if not values:
        return f"{key}: []"
    lines = [f"{key}:"]
    for item in values:
        for index, (item_key, item_value) in enumerate(item.items()):
            prefix = "  - " if index == 0 else "    "
            if isinstance(item_value, bool):
                rendered = "true" if item_value else "false"
            else:
                rendered = f'"{item_value}"'
            lines.append(f"{prefix}{item_key}: {rendered}")
    return "\n".join(lines)


def _emit_inline_mapping(key: str, values: dict[str, object]) -> str:
    rendered = ", ".join(f"{item_key}: {item_value}" for item_key, item_value in values.items())
    return f"{key}: {{{rendered}}}"


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
    schema: str = "v1",
    complexity_tier: str = "S",
    success_criteria: list[dict[str, object]] | None = None,
    budgets: dict[str, object] | None = None,
    checkpoint_contract: str = "none",
    recon_required: bool = False,
    recon_waiver: str | None = None,
    inputs: list[dict[str, object]] | None = None,
    constructed_by: str | None = None,
    extra_frontmatter: dict[str, object] | None = None,
) -> Path:
    if schema not in {"v1", "v2"}:
        raise ValueError(f"unsupported task fixture schema: {schema}")
    title = title or f"{task_id} title"
    dependencies = dependencies or []
    integration_ready_dependencies = integration_ready_dependencies or []
    allowed_paths = allowed_paths or ["src/"]
    disallowed_paths = disallowed_paths or ["contracts/"]
    outputs = outputs or ["src/example.py"]
    gates = gates or ["make gate"]
    stop_conditions = stop_conditions or ["Need @human"]
    success_criteria = success_criteria or [
        {"id": "SC1", "statement": "Declared output exists", "verification": "make gate"}
    ]
    budgets = budgets or {"max_wall_clock": "1h", "max_tokens": 100000, "max_cost_usd": 10}
    inputs = inputs or [
        {
            "path": "data/processed_manifest/input.json",
            "sha256": "1" * 64,
            "comparison_basis": False,
        }
    ]

    frontmatter_lines = ["---"]
    if schema == "v2":
        frontmatter_lines.append('task_schema: "research_swarm.task.v2"')
    frontmatter_lines.extend(
        [
            f'task_id: "{task_id}"',
            f'title: "{title}"',
            f'workstream: "{workstream}"',
            f'task_kind: "{task_kind or ""}"',
        ]
    )
    if schema == "v2":
        frontmatter_lines.extend(
            [
                f'complexity_tier: "{complexity_tier}"',
                _emit_mapping_list("success_criteria", success_criteria),
                _emit_inline_mapping("budgets", budgets),
                f'checkpoint_contract: "{checkpoint_contract}"',
                f"recon_required: {'true' if recon_required else 'false'}",
                *([f'recon_waiver: "{recon_waiver}"'] if recon_waiver is not None else []),
                *([f'constructed_by: "{constructed_by}"'] if constructed_by is not None else []),
                *(
                    [
                        (f"{k}:\n" + "".join(f"  - {item}\n" for item in v)).rstrip()
                        if isinstance(v, list)
                        else f"{k}: {v}"
                        for k, v in (extra_frontmatter or {}).items()
                    ]
                ),
                _emit_mapping_list("inputs", inputs),
            ]
        )
    frontmatter_lines.extend(
        [
            f"allow_network: {'true' if allow_network else 'false'}",
            f'role: "{role}"',
            f'priority: "{priority}"',
            _emit_list("dependencies", dependencies),
            _emit_list("integration_ready_dependencies", integration_ready_dependencies),
            _emit_list("allowed_paths", allowed_paths),
            _emit_list("disallowed_paths", disallowed_paths),
            _emit_list("outputs", outputs),
            _emit_unquoted_list("gates", gates) if schema == "v2" else _emit_list("gates", gates),
            _emit_list("stop_conditions", stop_conditions),
            "---",
        ]
    )
    frontmatter = "\n".join(frontmatter_lines)

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
    path = write_text(root, rel, frontmatter + "\n" + body)
    # Claimability now lints backlog tasks. Register historical v1 backlog
    # fixtures automatically so existing claim-path tests keep exercising v1.
    # Active fixtures bypass claimability and must not dirty this contract file:
    # direct-run ownership tests correctly treat that as an out-of-scope write.
    if schema == "v1" and folder == "backlog" and state == "backlog":
        register_historical_exemption(root, section="tasks", rel_path=rel)
    return path


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
