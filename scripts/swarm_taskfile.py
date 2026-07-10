from __future__ import annotations

import datetime as dt
from pathlib import Path
import re
from typing import Iterable


TASK_ID_BRANCH_RE = re.compile(r"^(T\d{3})(?=[_-]|$)")


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

    data: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        line = raw_line.split("#", 1)[0].rstrip()
        if line.strip() == "":
            continue

        list_match = re.match(r"^\s*-\s+(.*)\s*$", line)
        if current_list_key is not None and list_match is not None:
            value = list_match.group(1).strip().strip("'\"")
            current = data.get(current_list_key)
            if isinstance(current, list):
                current.append(value)
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
                data[key] = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
            continue

        data[key] = rest.strip("'\"")

    return data


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
