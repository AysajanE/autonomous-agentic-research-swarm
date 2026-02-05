"""Quality gates for the research pipeline.

Design principles:
- fast (target: <30s locally)
- deterministic (no web calls, no randomness)
- actionable failures (clear next step)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import platform
import sys
import json
import subprocess
import csv
import os


@dataclass
class GateResult:
    ok: bool
    details: dict[str, object]

VALID_TASK_STATES = {"backlog", "active", "blocked", "integration_ready", "ready_for_review", "done"}
VALID_PROJECT_MODES = {"empirical", "modeling", "hybrid"}
VALID_TASK_ROLES = {"Planner", "Worker", "Judge"}
VALID_TASK_PRIORITIES = {"low", "medium", "high"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_framework_config(path: Path) -> tuple[dict[str, object], str | None]:
    if not path.exists():
        return {}, None
    try:
        data_raw = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        return {}, f"invalid_json:{exc}"
    if not isinstance(data_raw, dict):
        return {}, "top_level_not_object"
    return data_raw, None


def _parse_framework_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().strip("'\"").lower()
    return v or None


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s:
            out.append(s)
    return out


def _parse_feature_flags(value: object) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, bool] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip().lower() in {"1", "true", "yes"}
    return out


def _default_features_for_mode(mode: str | None) -> dict[str, bool]:
    if mode == "modeling":
        return {"registry": False, "modeling": True}
    if mode == "hybrid":
        return {"registry": True, "modeling": True}
    if mode == "empirical":
        return {"registry": True, "modeling": False}
    return {"registry": True}


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


def _parse_project_mode(path: Path) -> str | None:
    """Parse a minimal YAML key: `mode: <value>` from contracts/project.yaml.

    We intentionally avoid external YAML dependencies in quality gates.
    """
    if not path.exists():
        return None
    for raw_line in _read_text(path).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not line.startswith("mode:"):
            continue
        value = line.split(":", 1)[1].strip().strip("'\"").lower()
        return value
    return None


def _parse_task_frontmatter(text: str) -> dict[str, object] | None:
    """Parse a minimal YAML frontmatter block (no external YAML dependency).

    Supports:
    - `key: value`
    - `key: [a, b]`
    - `key:` followed by indented `- item` lines
    """
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return None
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None

    data: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        line = raw_line.split("#", 1)[0].rstrip()
        if line.strip() == "":
            continue

        list_item_match = re.match(r"^\s*-\s+(.*)\s*$", line)
        if current_list_key is not None and list_item_match is not None:
            item = list_item_match.group(1).strip().strip("'\"")
            current_list = data.get(current_list_key)
            if isinstance(current_list, list):
                current_list.append(item)
            continue

        current_list_key = None
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if rest == "":
            data[key] = []
            current_list_key = key
            continue

        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            if inner == "":
                data[key] = []
            else:
                items = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                data[key] = items
            continue

        data[key] = rest.strip("'\"")

    return data


def _section_has_content(text: str, heading: str) -> bool:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    if match is None:
        return False
    after = text[match.end() :]
    for line in after.splitlines():
        if line.startswith("## "):
            return False
        if line.strip() == "":
            continue
        if re.search(r"[A-Za-z0-9]", line):
            return True
    return False


def gate_repo_structure() -> GateResult:
    framework_config_path = Path("contracts/framework.json")
    framework_config, framework_config_err = _load_framework_config(framework_config_path)
    if framework_config_err is not None:
        return GateResult(
            ok=False,
            details={
                "config_path": str(framework_config_path),
                "config_error": framework_config_err,
            },
        )

    config_mode = _parse_framework_mode(framework_config.get("mode"))
    project_mode = _parse_project_mode(Path("contracts/project.yaml"))
    mode = config_mode or project_mode

    features = _default_features_for_mode(mode)
    features.update(_parse_feature_flags(framework_config.get("features")))
    registry_enabled = bool(features.get("registry", True))

    required_paths = _parse_required_paths(framework_config.get("required_paths"), mode)
    required = [
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("contracts"),
        Path("contracts/AGENTS.md"),
        Path("contracts/CHANGELOG.md"),
        Path("contracts/assumptions.md"),
        Path("contracts/decisions.md"),
        Path("contracts/framework.json"),
        Path("contracts/project.yaml"),
        Path("contracts/schemas"),
        Path("docs"),
        Path(".orchestrator"),
        Path(".orchestrator/AGENTS.md"),
        Path(".orchestrator/ready_for_review"),
        Path(".orchestrator/workstreams.md"),
        Path("data"),
        Path("data/AGENTS.md"),
        Path("data/samples"),
        Path("data/processed_manifest"),
        Path("reports"),
        Path("reports/AGENTS.md"),
        Path("reports/catalog.yaml"),
        Path("scripts/quality_gates.py"),
        Path("scripts/AGENTS.md"),
        Path("src"),
        Path("src/AGENTS.md"),
        Path("tests"),
    ]

    if registry_enabled:
        required.extend(
            [
                Path("registry"),
                Path("registry/README.md"),
                Path("registry/AGENTS.md"),
                Path("registry/CHANGELOG.md"),
            ]
        )

    if mode in {"empirical", "hybrid"}:
        required.extend(
            [
                Path("docs/protocol.md"),
                Path("contracts/schemas/panel_schema.yaml"),
                Path("contracts/schemas/panel_schema_str_v1.yaml"),
                Path("contracts/schemas/panel_schema_decomp_v1.yaml"),
            ]
        )
    if mode in {"modeling", "hybrid"}:
        required.extend(
            [
                Path("contracts/instances"),
                Path("contracts/instances/benchmark_small"),
                Path("contracts/experiments"),
                Path("src/model"),
            ]
        )
    if mode == "hybrid":
        # Enforce an explicit empirical→modeling interface contract (C3).
        if not (Path("contracts/hybrid_interface_v1.yaml").exists() or Path("contracts/hybrid_interface_v1.json").exists()):
            required.append(Path("contracts/hybrid_interface_v1.yaml"))

    required.extend(Path(p) for p in required_paths)
    # Keep failures readable: avoid duplicate entries in required/missing.
    unique_required: list[Path] = []
    seen: set[str] = set()
    for p in required:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique_required.append(p)
    required = unique_required
    missing = [str(p) for p in required if not p.exists()]
    return GateResult(
        ok=(len(missing) == 0),
        details={
            "mode": mode,
            "project_mode": project_mode,
            "config_mode": config_mode,
            "registry_enabled": registry_enabled,
            "required_paths": required_paths,
            "missing": missing,
        },
    )


def gate_project_contract() -> GateResult:
    path = Path("contracts/project.yaml")
    if not path.exists():
        return GateResult(ok=False, details={"missing": str(path)})
    mode = _parse_project_mode(path)
    if mode is None:
        return GateResult(ok=False, details={"failures": ["missing_mode"]})
    if mode not in VALID_PROJECT_MODES:
        return GateResult(ok=False, details={"failures": [f"invalid_mode:{mode}"]})
    return GateResult(ok=True, details={"mode": mode})


def gate_environment() -> GateResult:
    """Validate that a pinned environment spec exists and report runtime versions."""
    env_spec_candidates = [
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("environment.yml"),
        Path(".python-version"),
        Path(".devcontainer/devcontainer.json"),
    ]
    present = [str(p) for p in env_spec_candidates if p.exists()]
    if len(present) == 0:
        return GateResult(ok=False, details={"missing": [str(p) for p in env_spec_candidates]})

    python_version_file = Path(".python-version")
    pinned_python = None
    if python_version_file.exists():
        pinned_python = _read_text(python_version_file).strip() or None

    return GateResult(
        ok=True,
        details={
            "present_env_specs": present,
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pinned_python": pinned_python,
        },
    )


def gate_protocol_complete() -> GateResult:
    mode = _parse_project_mode(Path("contracts/project.yaml"))
    if mode == "modeling":
        return GateResult(ok=True, details={"skipped": True, "mode": mode})

    path = Path("docs/protocol.md")
    if not path.exists():
        return GateResult(ok=False, details={"missing": str(path)})
    text = _read_text(path)

    failures: list[str] = []

    mode_match = re.search(r"^\s*-\s*Mode:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    if mode_match is None:
        failures.append("missing_mode_line")
    else:
        protocol_mode = mode_match.group(1).strip().lower()
        if mode is not None and protocol_mode != mode:
            failures.append(f"mode_mismatch:{protocol_mode}!={mode}")

    for field in ("Name", "Units"):
        if re.search(rf"^\s*-\s*{field}:\s*$", text, flags=re.MULTILINE):
            failures.append(f"primary_metric_{field.lower()}_blank")

    # Support both `- Formula:` and `- Formula (daily):` styles, but fail if left blank.
    if re.search(r"^\s*-\s*Formula[^:]*:\s*$", text, flags=re.MULTILINE):
        failures.append("primary_metric_formula_blank")

    required_sections = [
        "Rollup inclusion criteria",
        "Data source priority",
        "Known regime dates",
        "Validation tolerances",
    ]
    for heading in required_sections:
        if not _section_has_content(text, heading):
            failures.append(f"missing_or_empty_section:{heading}")

    return GateResult(ok=(len(failures) == 0), details={"failures": failures})


def gate_model_spec_complete() -> GateResult:
    mode = _parse_project_mode(Path("contracts/project.yaml"))
    if mode not in {"modeling", "hybrid"}:
        return GateResult(ok=True, details={"skipped": True, "mode": mode})

    candidates = [
        Path("contracts/model_spec.md"),
        Path("contracts/model_spec.yaml"),
        Path("contracts/model_spec.yml"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return GateResult(ok=False, details={"missing": [str(p) for p in candidates]})

    if path.suffix.lower() in {".yml", ".yaml"}:
        # Minimal check: file exists and is non-empty (structure gates cover existence).
        text = _read_text(path)
        ok = bool(text.strip())
        return GateResult(ok=ok, details={"path": str(path), "empty": (not ok)})

    text = _read_text(path)
    required_sections = [
        "Objective / question",
        "Notation and sets",
        "Decision variables",
        "Constraints",
        "Objective function",
        "Assumptions (explicit)",
        "Baselines / benchmark cases",
        "Solver / method",
        "Outputs (required)",
    ]
    failures = [s for s in required_sections if not _section_has_content(text, s)]
    return GateResult(ok=(len(failures) == 0), details={"path": str(path), "missing_or_empty_sections": failures})


def gate_workstreams_complete() -> GateResult:
    path = Path(".orchestrator/workstreams.md")
    if not path.exists():
        return GateResult(ok=False, details={"missing": str(path)})
    text = _read_text(path)

    failures: list[str] = []
    rows_checked = 0
    for line in text.splitlines():
        if not re.match(r"^\|\s*W\d+\s+", line):
            continue
        rows_checked += 1
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            failures.append(f"malformed_row:{line.strip()}")
            continue

        workstream = cells[0]
        purpose = cells[1]
        owns_paths = cells[2]
        does_not_own = cells[3]

        if purpose == "":
            failures.append(f"blank_purpose:{workstream}")
        if owns_paths == "":
            failures.append(f"blank_owns_paths:{workstream}")
        if does_not_own == "":
            failures.append(f"blank_does_not_own:{workstream}")

    if rows_checked == 0:
        failures.append("no_workstream_rows_found")

    return GateResult(ok=(len(failures) == 0), details={"failures": failures})


def gate_task_hygiene() -> GateResult:
    failures: list[str] = []
    task_dirs = [
        Path(".orchestrator/backlog"),
        Path(".orchestrator/active"),
        Path(".orchestrator/integration_ready"),
        Path(".orchestrator/ready_for_review"),
        Path(".orchestrator/blocked"),
        Path(".orchestrator/done"),
    ]

    task_files: list[Path] = []
    for task_dir in task_dirs:
        if not task_dir.exists():
            failures.append(f"missing_dir:{task_dir}")
            continue
        for p in task_dir.glob("*.md"):
            if p.name == "README.md":
                continue
            task_files.append(p)

    for path in sorted(task_files):
        text = _read_text(path)

        frontmatter = _parse_task_frontmatter(text)
        if frontmatter is None:
            failures.append(f"{path}:missing_yaml_frontmatter")
        else:
            required_keys = [
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
            ]
            for key in required_keys:
                if key not in frontmatter:
                    failures.append(f"{path}:frontmatter_missing_key:{key}")

            list_keys = [
                "dependencies",
                "allowed_paths",
                "disallowed_paths",
                "outputs",
                "gates",
                "stop_conditions",
            ]
            for key in list_keys:
                value = frontmatter.get(key)
                if not isinstance(value, list):
                    failures.append(f"{path}:frontmatter_key_not_list:{key}")

            task_id = frontmatter.get("task_id")
            if isinstance(task_id, str):
                if not path.name.startswith(task_id):
                    failures.append(f"{path}:frontmatter_task_id_mismatch:{task_id}")

            workstream = frontmatter.get("workstream")
            if isinstance(workstream, str) and not re.fullmatch(r"W\d+", workstream):
                failures.append(f"{path}:invalid_workstream:{workstream}")

            role = frontmatter.get("role")
            if isinstance(role, str) and role not in VALID_TASK_ROLES:
                failures.append(f"{path}:invalid_role:{role}")

            priority = frontmatter.get("priority")
            if isinstance(priority, str) and priority not in VALID_TASK_PRIORITIES:
                failures.append(f"{path}:invalid_priority:{priority}")

        required_headings = [
            "## Context",
            "## Inputs",
            "## Outputs",
            "## Success Criteria",
            "## Status",
            "## Notes / Decisions",
        ]
        for heading in required_headings:
            if heading not in text:
                failures.append(f"{path}:missing_heading:{heading}")

        state_match = re.search(r"^\s*-\s*State:\s*(.+)\s*$", text, flags=re.MULTILINE)
        if state_match is None:
            failures.append(f"{path}:missing_state")
        else:
            state = state_match.group(1).strip()
            if state not in VALID_TASK_STATES:
                failures.append(f"{path}:invalid_state:{state}")

        updated_match = re.search(
            r"^\s*-\s*Last updated:\s*(\d{4}-\d{2}-\d{2})\s*$",
            text,
            flags=re.MULTILINE,
        )
        if updated_match is None:
            failures.append(f"{path}:missing_last_updated")

    return GateResult(ok=(len(failures) == 0), details={"failures": failures})


def gate_task_dependencies() -> GateResult:
    failures: list[str] = []
    task_dirs = [
        Path(".orchestrator/backlog"),
        Path(".orchestrator/active"),
        Path(".orchestrator/integration_ready"),
        Path(".orchestrator/ready_for_review"),
        Path(".orchestrator/blocked"),
        Path(".orchestrator/done"),
    ]

    task_files: list[Path] = []
    for task_dir in task_dirs:
        if not task_dir.exists():
            continue
        for p in task_dir.glob("*.md"):
            if p.name == "README.md":
                continue
            task_files.append(p)

    id_to_path: dict[str, Path] = {}
    deps_map: dict[str, list[str]] = {}

    for path in sorted(task_files):
        fm = _parse_task_frontmatter(_read_text(path)) or {}
        task_id = fm.get("task_id")
        deps = fm.get("dependencies")
        if not isinstance(task_id, str):
            continue
        if task_id in id_to_path:
            failures.append(f"duplicate_task_id:{task_id}:{id_to_path[task_id]}:{path}")
            continue
        id_to_path[task_id] = path
        deps_list: list[str] = []
        if isinstance(deps, list):
            deps_list = [d for d in deps if isinstance(d, str)]
        deps_map[task_id] = deps_list

    # Validate dependency IDs exist and are well-formed.
    for task_id, deps in deps_map.items():
        for dep in deps:
            if not re.fullmatch(r"T\d{3}", dep):
                failures.append(f"{id_to_path.get(task_id)}:invalid_dependency_id:{dep}")
                continue
            if dep == task_id:
                failures.append(f"{id_to_path.get(task_id)}:self_dependency:{dep}")
                continue
            if dep not in id_to_path:
                failures.append(f"{id_to_path.get(task_id)}:missing_dependency:{dep}")

    # Detect cycles (simple DFS).
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node: str, stack: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            # cycle: include from first occurrence
            if node in stack:
                i = stack.index(node)
                cycle = stack[i:] + [node]
                failures.append(f"dependency_cycle:{'->'.join(cycle)}")
            return
        visiting.add(node)
        stack.append(node)
        for dep in deps_map.get(node, []):
            if dep in id_to_path:
                _dfs(dep, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for task_id in sorted(id_to_path.keys()):
        _dfs(task_id, [])

    return GateResult(ok=(len(failures) == 0), details={"failures": failures})


def _git_changed_paths_against_base(base_ref: str) -> tuple[list[str], str | None]:
    try:
        cp2 = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if cp2.returncode != 0:
            return [], "git_diff_failed"
        paths = [p.strip() for p in (cp2.stdout or "").splitlines() if p.strip()]
        return paths, None
    except Exception:
        return [], "git_diff_exception"


def _resolve_base_ref(candidate_refs: list[str]) -> str | None:
    for ref in candidate_refs:
        try:
            cp = subprocess.run(
                ["git", "rev-parse", "--verify", ref],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if cp.returncode == 0:
                return ref
        except Exception:
            continue
    return None


def gate_contract_change_discipline() -> GateResult:
    """If contracts/protocol change, require a decision log + contract changelog update.

    Best-effort: compares against a base ref if available. Deterministic and offline.
    """
    env_base = os.environ.get("GATE_BASE_REF")
    base_ref = env_base or _resolve_base_ref(["origin/main", "main"])
    if base_ref is None:
        return GateResult(ok=True, details={"skipped": True, "reason": "base_ref_missing", "candidates": ["origin/main", "main"]})

    changed, err = _git_changed_paths_against_base(base_ref)
    if err is not None:
        return GateResult(ok=True, details={"skipped": True, "reason": err, "base_ref": base_ref})

    def _is_contract_change(p: str) -> bool:
        if p == "docs/protocol.md":
            return True
        if p.startswith("contracts/") and p not in {"contracts/decisions.md", "contracts/CHANGELOG.md"}:
            return True
        return False

    contract_changed = any(_is_contract_change(p) for p in changed)
    if not contract_changed:
        return GateResult(ok=True, details={"base_ref": base_ref, "contract_changed": False})

    required = {"contracts/decisions.md", "contracts/CHANGELOG.md"}
    missing = sorted(r for r in required if r not in changed)
    ok = len(missing) == 0
    return GateResult(
        ok=ok,
        details={
            "base_ref": base_ref,
            "contract_changed": True,
            "missing_required_updates": missing,
        },
    )


def gate_registry_change_discipline() -> GateResult:
    """If registry files change, require registry/CHANGELOG.md update (best-effort diff)."""
    env_base = os.environ.get("GATE_BASE_REF")
    base_ref = env_base or _resolve_base_ref(["origin/main", "main"])
    if base_ref is None:
        return GateResult(ok=True, details={"skipped": True, "reason": "base_ref_missing", "candidates": ["origin/main", "main"]})

    changed, err = _git_changed_paths_against_base(base_ref)
    if err is not None:
        return GateResult(ok=True, details={"skipped": True, "reason": err, "base_ref": base_ref})

    registry_changed = any(p.startswith("registry/") and p != "registry/CHANGELOG.md" for p in changed)
    if not registry_changed:
        return GateResult(ok=True, details={"base_ref": base_ref, "registry_changed": False})

    ok = "registry/CHANGELOG.md" in changed
    return GateResult(
        ok=ok,
        details={"base_ref": base_ref, "registry_changed": True, "missing_registry_changelog": (not ok)},
    )


def gate_raw_manifest_validity() -> GateResult:
    """Validate any tracked raw provenance manifests under data/raw_manifest/.

    This does not require raw snapshots to be present or hashed during gating;
    it only validates that any committed manifest JSON files are well-formed and
    include required keys.
    """
    failures: list[str] = []
    manifest_dir = Path("data/raw_manifest")
    if not manifest_dir.exists():
        return GateResult(ok=False, details={"missing": str(manifest_dir)})

    manifest_paths = sorted(manifest_dir.glob("*.json"))
    required_top_keys = {"source", "fetched_at_utc", "command", "files"}
    required_file_keys = {"path", "sha256", "bytes"}

    for path in manifest_paths:
        try:
            data = json.loads(_read_text(path))
        except json.JSONDecodeError as exc:
            failures.append(f"{path}:invalid_json:{exc}")
            continue

        if not isinstance(data, dict):
            failures.append(f"{path}:top_level_not_object")
            continue

        missing_keys = sorted(k for k in required_top_keys if k not in data)
        if missing_keys:
            failures.append(f"{path}:missing_keys:{','.join(missing_keys)}")
            continue

        files = data.get("files")
        if not isinstance(files, list):
            failures.append(f"{path}:files_not_list")
            continue

        for i, entry in enumerate(files):
            if not isinstance(entry, dict):
                failures.append(f"{path}:files[{i}]:not_object")
                continue
            missing_entry_keys = sorted(k for k in required_file_keys if k not in entry)
            if missing_entry_keys:
                failures.append(
                    f"{path}:files[{i}]:missing_keys:{','.join(missing_entry_keys)}"
                )
                continue
            sha = entry.get("sha256")
            if isinstance(sha, str) and not re.fullmatch(r"[0-9a-f]{64}", sha):
                failures.append(f"{path}:files[{i}]:invalid_sha256")

    return GateResult(
        ok=(len(failures) == 0),
        details={"count": len(manifest_paths), "failures": failures},
    )


def gate_panel_schema_nonempty() -> GateResult:
    mode = _parse_project_mode(Path("contracts/project.yaml"))
    if mode == "modeling":
        return GateResult(ok=True, details={"skipped": True, "mode": mode})

    path = Path("contracts/schemas/panel_schema_str_v1.yaml")
    if not path.exists():
        return GateResult(ok=False, details={"missing": str(path), "mode": mode})

    for raw_line in _read_text(path).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if re.match(r"^[A-Za-z0-9_-]+\s*:", line):
            return GateResult(ok=True, details={"path": str(path), "mode": mode})
    return GateResult(ok=False, details={"path": str(path), "mode": mode, "failure": "comment_only"})


def gate_sample_panel_integrity() -> GateResult:
    """Best-effort integrity checks for committed golden samples (if present)."""
    sample = Path("data/samples/growthepie/vendor_daily_rollup_panel_sample.csv")
    if not sample.exists():
        return GateResult(ok=True, details={"skipped": True, "reason": "sample_missing"})

    required_cols = {"date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth"}
    failures: list[str] = []

    rows: list[dict[str, str]] = []
    with sample.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return GateResult(ok=False, details={"failures": ["missing_header"]})
        cols = set(reader.fieldnames)
        missing = sorted(c for c in required_cols if c not in cols)
        if missing:
            failures.append(f"missing_columns:{','.join(missing)}")
        for row in reader:
            rows.append(row)

    # Non-negativity (where parseable).
    for i, row in enumerate(rows[:2000]):  # safety cap for gates
        for col in ["l2_fees_eth", "rent_paid_eth", "profit_eth"]:
            if col not in row or row[col] in ("", None):
                continue
            try:
                v = float(row[col])
            except ValueError:
                failures.append(f"row{i}:invalid_number:{col}")
                continue
            if v < 0:
                failures.append(f"row{i}:negative:{col}")

    return GateResult(ok=(len(failures) == 0), details={"sample": str(sample), "failures": failures})


def main() -> None:
    results = {
        "repo_structure": gate_repo_structure(),
        "project_contract": gate_project_contract(),
        "environment": gate_environment(),
        "protocol_complete": gate_protocol_complete(),
        "model_spec_complete": gate_model_spec_complete(),
        "panel_schema_nonempty": gate_panel_schema_nonempty(),
        "workstreams_complete": gate_workstreams_complete(),
        "task_hygiene": gate_task_hygiene(),
        "task_dependencies": gate_task_dependencies(),
        "contract_change_discipline": gate_contract_change_discipline(),
        "registry_change_discipline": gate_registry_change_discipline(),
        "raw_manifest_validity": gate_raw_manifest_validity(),
        "sample_panel_integrity": gate_sample_panel_integrity(),
    }
    ok = all(r.ok for r in results.values())
    for name, r in results.items():
        print(f"[{name}] ok={r.ok} details={r.details}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
