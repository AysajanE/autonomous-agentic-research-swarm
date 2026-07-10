"""Shared parsing and linting for research-swarm task files.

The frontmatter parser intentionally implements only the repository's YAML
subset.  In addition to the historical scalar and list forms, task schema v2
accepts inline mappings (used by ``budgets``) and exactly one nested
list-of-mappings level for ``success_criteria`` and ``inputs``::

    success_criteria:
      - id: SC1
        statement: "The output exists"
        verification: "test -f reports/output.json"
    inputs:
      - path: data/processed_manifest/input.json
        sha256: 0123...cdef
        comparison_basis: true

No general-purpose YAML features (anchors, folded scalars, or deeper nesting)
are supported.  Historical task parsing remains unchanged for all other list
keys, including its quote-stripping behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


TASK_SCHEMA_VERSION = "research_swarm.task.v2"
TASK_SCHEMA_MARKER = "task_schema"
TASK_KIND_VALUES = (
    "etl",
    "analysis",
    "validation",
    "writing",
    "lit_review",
    "model",
    "proof",
    "bridge",
    "ops",
    "integrity_audit",
    "repair",
)
COMPLEXITY_TIER_VALUES = ("S", "M", "L")
CHECKPOINT_CONTRACT_VALUES = ("none", "progress_file")
NETWORK_COMMAND_TOKENS = ("curl", "wget", "http://", "https://")

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
TASK_V2_REQUIRED_FRONTMATTER_KEYS = (
    TASK_SCHEMA_MARKER,
    "task_kind",
    "complexity_tier",
    "success_criteria",
    "budgets",
    "checkpoint_contract",
    "recon_required",
    "inputs",
)
SUCCESS_CRITERION_KEYS = ("id", "statement", "verification")
TASK_BUDGET_KEYS = ("max_wall_clock", "max_tokens", "max_cost_usd")
TASK_INPUT_REFERENCE_KEYS = ("path", "manifest")

_NESTED_MAPPING_LIST_KEYS = {"success_criteria", "inputs"}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DURATION_RE = re.compile(r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>[hms])$", re.IGNORECASE)
_TOKEN_COUNT_RE = re.compile(r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>[kmb]?)$", re.IGNORECASE)


TASK_ID_BRANCH_RE = re.compile(r"^(T\d{3})(?=[_-]|$)")


def _parse_new_scalar(value: str) -> object:
    """Parse scalar types only inside v2 mappings.

    Top-level and historical list scalars deliberately retain their old string
    representation so existing task files remain byte-compatible at the parser
    boundary.
    """
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, str):
                return parsed
    stripped = stripped.strip("'\"")
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", stripped):
        try:
            return float(stripped)
        except ValueError:
            pass
    return stripped


def _split_inline_mapping_items(value: str) -> list[str] | None:
    items: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == ",":
            items.append(value[start:index].strip())
            start = index + 1
    if quote is not None:
        return None
    items.append(value[start:].strip())
    return items


def _split_inline_mapping(value: str) -> dict[str, object] | None:
    if not (value.startswith("{") and value.endswith("}")):
        return None
    inner = value[1:-1].strip()
    if not inner:
        return {}
    out: dict[str, object] = {}
    items = _split_inline_mapping_items(inner)
    if items is None:
        return None
    for item in items:
        if ":" not in item:
            return None
        key, raw_value = item.split(":", 1)
        key = key.strip()
        if not key:
            return None
        out[key] = _parse_new_scalar(raw_value)
    return out


def _parse_nested_mapping_scalar(list_key: str, field_key: str, value: str) -> object:
    """Keep hashes and criterion text as strings; type only declared booleans."""
    if list_key == "inputs" and field_key == "comparison_basis":
        return _parse_new_scalar(value)
    return value.strip().strip("'\"")


def parse_task_id_from_branch(branch_name: str) -> str | None:
    match = TASK_ID_BRANCH_RE.match(branch_name)
    return match.group(1) if match else None


def parse_task_frontmatter(text: str) -> dict[str, object] | None:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return None

    end_idx = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_idx = index
            break
    if end_idx is None:
        return None

    has_schema_marker = any(
        re.match(rf"^\s*{re.escape(TASK_SCHEMA_MARKER)}\s*:", raw_line)
        for raw_line in lines[1:end_idx]
    )

    data: dict[str, object] = {}
    current_list_key: str | None = None
    current_mapping: dict[str, object] | None = None
    for raw_line in lines[1:end_idx]:
        line = raw_line.split("#", 1)[0].rstrip()
        if line.strip() == "":
            continue

        list_match = re.match(r"^\s*-\s+(.*)\s*$", line)
        if current_list_key is not None and list_match is not None:
            current = data.get(current_list_key)
            value = list_match.group(1).strip()
            if (
                has_schema_marker
                and current_list_key in _NESTED_MAPPING_LIST_KEYS
                and ":" in value
            ):
                key, rest = value.split(":", 1)
                key = key.strip()
                current_mapping = {
                    key: _parse_nested_mapping_scalar(current_list_key, key, rest)
                }
                if isinstance(current, list):
                    current.append(current_mapping)
            else:
                current_mapping = None
                if isinstance(current, list):
                    current.append(
                        value
                        if has_schema_marker and current_list_key == "gates"
                        else value.strip("'\"")
                    )
            continue

        mapping_field_match = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)\s*$", line)
        if (
            has_schema_marker
            and current_list_key in _NESTED_MAPPING_LIST_KEYS
            and current_mapping is not None
            and mapping_field_match is not None
        ):
            field_key = mapping_field_match.group(1)
            current_mapping[field_key] = _parse_nested_mapping_scalar(
                current_list_key,
                field_key,
                mapping_field_match.group(2),
            )
            continue

        current_list_key = None
        current_mapping = None
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
                data[key] = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
            continue

        if has_schema_marker:
            inline_mapping = _split_inline_mapping(rest)
            if inline_mapping is not None:
                data[key] = inline_mapping
                continue

        data[key] = rest.strip("'\"")

    return data


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if number > 0 else None


def parse_wall_clock_seconds(value: object) -> float | None:
    numeric = _positive_number(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str):
        return None
    match = _DURATION_RE.fullmatch(value.strip())
    if match is None:
        return None
    amount = float(match.group("value"))
    if amount <= 0:
        return None
    multiplier = {"h": 3600.0, "m": 60.0, "s": 1.0}[match.group("unit").lower()]
    return amount * multiplier


def parse_token_count(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    match = _TOKEN_COUNT_RE.fullmatch(value.strip())
    if match is None:
        return None
    amount = float(match.group("value"))
    if amount <= 0:
        return None
    multiplier = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[
        match.group("unit").lower()
    ]
    return amount * multiplier


@dataclass(frozen=True)
class TaskV2Fields:
    """Typed accessors for task-schema-v2 frontmatter fields."""

    frontmatter: Mapping[str, object]

    @property
    def task_schema(self) -> str | None:
        return _string(self.frontmatter.get(TASK_SCHEMA_MARKER))

    @property
    def task_kind(self) -> str | None:
        return _string(self.frontmatter.get("task_kind"))

    @property
    def complexity_tier(self) -> str | None:
        return _string(self.frontmatter.get("complexity_tier"))

    @property
    def checkpoint_contract(self) -> str | None:
        return _string(self.frontmatter.get("checkpoint_contract"))

    @property
    def recon_required(self) -> bool | None:
        return _bool(self.frontmatter.get("recon_required"))

    @property
    def recon_waiver(self) -> str | None:
        return _string(self.frontmatter.get("recon_waiver"))

    @property
    def constructed_by(self) -> str | None:
        return _string(self.frontmatter.get("constructed_by"))

    @property
    def success_criteria(self) -> tuple[Mapping[str, object], ...] | None:
        value = self.frontmatter.get("success_criteria")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            return None
        return tuple(value)

    @property
    def budgets(self) -> Mapping[str, object] | None:
        value = self.frontmatter.get("budgets")
        return value if isinstance(value, dict) else None

    @property
    def inputs(self) -> tuple[Mapping[str, object], ...] | None:
        value = self.frontmatter.get("inputs")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            return None
        return tuple(value)

    @property
    def triage(self) -> Mapping[str, object] | None:
        value = self.frontmatter.get("triage")
        return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class TaskLintDiagnostic:
    task: str
    field: str
    reason: str
    expected: object
    actual: object

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "field": self.field,
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
        }


def _task_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _task_label(path: Path, frontmatter: Mapping[str, object] | None, repo_root: Path) -> str:
    if frontmatter is not None:
        task_id = _string(frontmatter.get("task_id"))
        if task_id is not None:
            return task_id
    return _task_relative_path(path, repo_root)


def _diagnostic(
    diagnostics: list[TaskLintDiagnostic],
    task: str,
    field: str,
    reason: str,
    expected: object,
    actual: object,
) -> None:
    diagnostics.append(
        TaskLintDiagnostic(task=task, field=field, reason=reason, expected=expected, actual=actual)
    )


def lint_task_files(
    task_paths: Sequence[Path],
    *,
    repo_root: Path,
    network_workstreams: Iterable[str],
    v1_exemptions: Mapping[str, Mapping[str, object]],
    task_texts: Mapping[Path, str] | None = None,
) -> list[TaskLintDiagnostic]:
    """Lint a complete task set so validation tasks can resolve construction inputs.

    ``task_texts`` supplies proposed content for paths that must be validated
    before a Planner write reaches disk.
    """
    proposed_texts = {
        Path(path).resolve(): text for path, text in (task_texts or {}).items()
    }
    parsed: dict[Path, dict[str, object] | None] = {}
    tasks_by_id: dict[str, Mapping[str, object]] = {}
    for path in task_paths:
        try:
            text = proposed_texts.get(path.resolve())
            if text is None:
                text = path.read_text(encoding="utf-8")
            frontmatter = parse_task_frontmatter(text)
        except OSError:
            frontmatter = None
        parsed[path] = frontmatter
        if frontmatter is not None:
            task_id = _string(frontmatter.get("task_id"))
            if task_id is not None:
                tasks_by_id[task_id] = frontmatter

    diagnostics: list[TaskLintDiagnostic] = []
    network_workstream_set = set(network_workstreams)
    for path in task_paths:
        frontmatter = parsed[path]
        task = _task_label(path, frontmatter, repo_root)
        rel_path = _task_relative_path(path, repo_root)
        if frontmatter is None:
            _diagnostic(diagnostics, task, "frontmatter", "missing_yaml_frontmatter", "mapping", None)
            continue

        if TASK_SCHEMA_MARKER not in frontmatter:
            if rel_path not in v1_exemptions:
                _diagnostic(
                    diagnostics,
                    task,
                    TASK_SCHEMA_MARKER,
                    "unexempted_v1_task",
                    "path listed under historical_exemptions.tasks",
                    rel_path,
                )
            else:
                exemption = v1_exemptions[rel_path]
                expected_sha = _string(exemption.get("sha256"))
                if exemption.get("schema_version") != "fixture":
                    proposed = proposed_texts.get(path.resolve())
                    actual_bytes = (
                        proposed.encode("utf-8") if proposed is not None else path.read_bytes()
                    )
                    actual_sha = hashlib.sha256(actual_bytes).hexdigest()
                    if expected_sha is None or actual_sha != expected_sha:
                        _diagnostic(
                            diagnostics,
                            task,
                            TASK_SCHEMA_MARKER,
                            "historical_exemption_sha256_mismatch",
                            expected_sha,
                            actual_sha,
                        )
            continue

        fields = TaskV2Fields(frontmatter)
        if fields.task_schema != TASK_SCHEMA_VERSION:
            _diagnostic(
                diagnostics,
                task,
                TASK_SCHEMA_MARKER,
                "invalid_task_schema",
                TASK_SCHEMA_VERSION,
                fields.task_schema,
            )

        for key in TASK_V2_REQUIRED_FRONTMATTER_KEYS:
            if key not in frontmatter:
                _diagnostic(diagnostics, task, key, "missing_required_field", "present", None)

        if fields.task_kind not in TASK_KIND_VALUES:
            _diagnostic(
                diagnostics, task, "task_kind", "invalid_task_kind", list(TASK_KIND_VALUES), fields.task_kind
            )

        if fields.complexity_tier not in COMPLEXITY_TIER_VALUES:
            _diagnostic(
                diagnostics,
                task,
                "complexity_tier",
                "invalid_complexity_tier",
                list(COMPLEXITY_TIER_VALUES),
                fields.complexity_tier,
            )

        if fields.checkpoint_contract not in CHECKPOINT_CONTRACT_VALUES:
            _diagnostic(
                diagnostics,
                task,
                "checkpoint_contract",
                "invalid_checkpoint_contract",
                list(CHECKPOINT_CONTRACT_VALUES),
                fields.checkpoint_contract,
            )
        elif fields.complexity_tier == "L" and fields.checkpoint_contract != "progress_file":
            _diagnostic(
                diagnostics,
                task,
                "checkpoint_contract",
                "checkpoint_required_for_l",
                "progress_file",
                fields.checkpoint_contract,
            )

        if fields.complexity_tier in {"M", "L"} and fields.recon_required is not True:
            if not (fields.recon_required is False and fields.recon_waiver is not None):
                _diagnostic(
                    diagnostics,
                    task,
                    "recon_required",
                    "recon_required_for_tier",
                    "true or explicit false with non-empty recon_waiver",
                    fields.recon_required,
                )
        elif "recon_required" in frontmatter and fields.recon_required is None:
            _diagnostic(
                diagnostics,
                task,
                "recon_required",
                "invalid_boolean",
                "true or false",
                frontmatter.get("recon_required"),
            )

        if "triage" in frontmatter:
            triage = fields.triage
            if triage is None:
                _diagnostic(
                    diagnostics,
                    task,
                    "triage",
                    "invalid_triage",
                    "{status: confirmed|split, by: planner, note: non-empty string}",
                    frontmatter.get("triage"),
                )
            else:
                status = _string(triage.get("status"))
                by = _string(triage.get("by"))
                note = _string(triage.get("note"))
                if status not in {"confirmed", "split"}:
                    _diagnostic(
                        diagnostics,
                        task,
                        "triage.status",
                        "invalid_triage_status",
                        "confirmed or split",
                        status,
                    )
                if by != "planner":
                    _diagnostic(
                        diagnostics,
                        task,
                        "triage.by",
                        "invalid_triage_actor",
                        "planner",
                        by,
                    )
                if note is None:
                    _diagnostic(
                        diagnostics,
                        task,
                        "triage.note",
                        "empty_triage_note",
                        "non-empty string",
                        note,
                    )

        criteria = fields.success_criteria
        if not criteria:
            _diagnostic(
                diagnostics,
                task,
                "success_criteria",
                "invalid_success_criteria",
                "non-empty list of mappings",
                frontmatter.get("success_criteria"),
            )
        else:
            seen_ids: set[str] = set()
            for index, criterion in enumerate(criteria):
                criterion_id = _string(criterion.get("id"))
                if criterion_id is None:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"success_criteria[{index}].id",
                        "empty_success_criterion_id",
                        "non-empty string",
                        criterion.get("id"),
                    )
                elif criterion_id in seen_ids:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"success_criteria[{index}].id",
                        "duplicate_success_criterion_id",
                        "unique id",
                        criterion_id,
                    )
                else:
                    seen_ids.add(criterion_id)
                for key in ("statement", "verification"):
                    if _string(criterion.get(key)) is None:
                        _diagnostic(
                            diagnostics,
                            task,
                            f"success_criteria[{index}].{key}",
                            f"empty_success_criterion_{key}",
                            "non-empty string",
                            criterion.get(key),
                        )

        budgets = fields.budgets
        if budgets is None:
            _diagnostic(
                diagnostics, task, "budgets", "invalid_budgets", "mapping", frontmatter.get("budgets")
            )
        else:
            budget_validators = {
                "max_wall_clock": parse_wall_clock_seconds,
                "max_tokens": parse_token_count,
                "max_cost_usd": _positive_number,
            }
            for key, validator in budget_validators.items():
                value = budgets.get(key)
                if validator(value) is None:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"budgets.{key}",
                        "invalid_budget_value",
                        "positive numeric value" + (" or duration such as 4h/90m/3600s" if key == "max_wall_clock" else ""),
                        value,
                    )

        gates = frontmatter.get("gates")
        if not isinstance(gates, list) or not gates:
            _diagnostic(diagnostics, task, "gates", "invalid_gates", "non-empty list", gates)
        else:
            workstream = _string(frontmatter.get("workstream"))
            for index, gate in enumerate(gates):
                if not isinstance(gate, str) or not gate.strip():
                    _diagnostic(
                        diagnostics, task, f"gates[{index}]", "invalid_gate", "non-empty string", gate
                    )
                    continue
                if gate.endswith(("'", '\"')):
                    _diagnostic(
                        diagnostics,
                        task,
                        f"gates[{index}]",
                        "gate_ends_in_quote",
                        "command not ending in a quote character",
                        gate,
                    )
                if workstream not in network_workstream_set:
                    hits = sorted(token for token in NETWORK_COMMAND_TOKENS if token in gate.lower())
                    if hits:
                        _diagnostic(
                            diagnostics,
                            task,
                            f"gates[{index}]",
                            "network_string_in_gate",
                            "offline command in non-network workstream",
                            hits,
                        )

        inputs = fields.inputs
        comparison_hashes: list[str] = []
        if inputs is None:
            _diagnostic(
                diagnostics,
                task,
                "inputs",
                "invalid_inputs",
                "list of {path (or manifest), sha256} mappings",
                frontmatter.get("inputs"),
            )
        else:
            for index, item in enumerate(inputs):
                references = {
                    key: value
                    for key in TASK_INPUT_REFERENCE_KEYS
                    if (value := _string(item.get(key))) is not None
                }
                if not references:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"inputs[{index}]",
                        "missing_input_reference",
                        "non-empty path or manifest",
                        item,
                    )
                elif len(references) > 1:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"inputs[{index}]",
                        "multiple_input_references",
                        "exactly one of path or manifest",
                        sorted(references),
                    )
                digest = _string(item.get("sha256"))
                if digest is None or _SHA256_RE.fullmatch(digest) is None:
                    _diagnostic(
                        diagnostics,
                        task,
                        f"inputs[{index}].sha256",
                        "invalid_input_sha256",
                        "64 hexadecimal characters",
                        digest,
                    )
                else:
                    normalized_digest = digest.lower()
                    comparison_basis = _bool(item.get("comparison_basis"))
                    if comparison_basis is True:
                        comparison_hashes.append(normalized_digest)
                    elif "comparison_basis" in item and comparison_basis is None:
                        _diagnostic(
                            diagnostics,
                            task,
                            f"inputs[{index}].comparison_basis",
                            "invalid_boolean",
                            "true or false",
                            item.get("comparison_basis"),
                        )

        if fields.task_kind == "validation":
            if fields.constructed_by is None:
                _diagnostic(
                    diagnostics,
                    task,
                    "constructed_by",
                    "missing_constructed_by",
                    "construction task id such as T035",
                    None,
                )
            elif fields.constructed_by not in tasks_by_id:
                _diagnostic(
                    diagnostics,
                    task,
                    "constructed_by",
                    "constructed_by_task_not_found",
                    "existing task id",
                    fields.constructed_by,
                )
            if not comparison_hashes:
                _diagnostic(
                    diagnostics,
                    task,
                    "inputs",
                    "missing_comparison_basis",
                    "at least one valid input with comparison_basis: true",
                    comparison_hashes,
                )
            if fields.constructed_by in tasks_by_id and comparison_hashes:
                construction_fields = TaskV2Fields(tasks_by_id[fields.constructed_by])
                construction_hashes = {
                    digest.lower()
                    for item in (construction_fields.inputs or ())
                    if (digest := _string(item.get("sha256"))) is not None
                    and _SHA256_RE.fullmatch(digest) is not None
                }
                if not construction_hashes:
                    _diagnostic(
                        diagnostics,
                        task,
                        "constructed_by",
                        "construction_inputs_missing",
                        "referenced task with hash-declared inputs",
                        fields.constructed_by,
                    )
                elif overlap := sorted(set(comparison_hashes) & construction_hashes):
                    _diagnostic(
                        diagnostics,
                        task,
                        "inputs",
                        "comparison_basis_not_disjoint",
                        "all comparison-basis sha256 values absent from construction inputs",
                        overlap,
                    )

    return diagnostics


def _section_bounds(text: str, heading: str) -> tuple[int, int] | None:
    match = re.search(rf"^## {re.escape(heading)}[ \t]*\r?$", text, flags=re.MULTILINE)
    if match is None:
        return None

    start = match.end()
    if start < len(text) and text[start] == "\n":
        start += 1

    next_heading = re.search(r"^(?:## |# )", text[start:], flags=re.MULTILINE)
    end = len(text) if next_heading is None else start + next_heading.start()
    return start, end


def extract_section(text: str, heading: str) -> str | None:
    bounds = _section_bounds(text, heading)
    if bounds is None:
        return None
    start, end = bounds
    return text[start:end]


def parse_status_value(text: str, field: str) -> str | None:
    status_section = extract_section(text, "Status")
    if status_section is None:
        return None
    pattern = rf"^[ \t]*-[ \t]*{re.escape(field)}:[ \t]*(.+?)[ \t]*$"
    match = re.search(pattern, status_section, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _utc_today() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).date().isoformat()


def update_task_status_and_notes(
    *,
    task_path: Path,
    new_state: str,
    note_line: str,
    allowed_states: Iterable[str],
) -> None:
    if new_state not in set(allowed_states):
        raise ValueError(f"invalid_state:{new_state}")

    task_path = Path(task_path)
    text = task_path.read_text(encoding="utf-8")
    status_bounds = _section_bounds(text, "Status")
    if status_bounds is None:
        raise SystemExit(f"missing_state_line:{task_path}")

    status_start, status_end = status_bounds
    status_section = text[status_start:status_end]
    updated_status, state_subs = re.subn(
        r"^[ \t]*-[ \t]*State:[ \t]*.+?[ \t]*$",
        f"- State: {new_state}",
        status_section,
        flags=re.MULTILINE,
    )
    if state_subs == 0:
        raise SystemExit(f"missing_state_line:{task_path}")

    today = _utc_today()
    updated_status, last_updated_subs = re.subn(
        r"^[ \t]*-[ \t]*Last updated:[ \t]*\d{4}-\d{2}-\d{2}[ \t]*$",
        f"- Last updated: {today}",
        updated_status,
        flags=re.MULTILINE,
    )
    if last_updated_subs == 0:
        raise SystemExit(f"missing_last_updated_line:{task_path}")

    updated_text = text[:status_start] + updated_status + text[status_end:]
    notes_bounds = _section_bounds(updated_text, "Notes / Decisions")
    if notes_bounds is None:
        raise SystemExit(f"missing_notes_heading:{task_path}")

    notes_start, notes_end = notes_bounds
    notes_section = updated_text[notes_start:notes_end]
    notes_content = notes_section.rstrip()
    trailing = notes_section[len(notes_content) :]
    heading_separator = "" if notes_start == 0 or updated_text[notes_start - 1] == "\n" else "\n"
    if notes_content:
        note_prefix = notes_content + "\n"
    elif trailing:
        note_prefix = "\n"
    else:
        note_prefix = ""
    note_suffix = trailing or "\n"
    updated_notes = f"{heading_separator}{note_prefix}- {today}: {note_line}{note_suffix}"
    updated_text = updated_text[:notes_start] + updated_notes + updated_text[notes_end:]
    task_path.write_text(updated_text, encoding="utf-8")


class WorktreeCollisionError(RuntimeError):
    def __init__(self, worktree_path: Path):
        self.worktree_path = Path(worktree_path)
        self.path = self.worktree_path
        super().__init__(f"worktree_path_already_exists:{self.worktree_path}")
