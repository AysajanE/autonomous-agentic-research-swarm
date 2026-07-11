"""Durable event journal and offline escalation delivery for the swarm runtime."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess


EVENT_SCHEMA_VERSION = "research_swarm.event.v1"
EVENT_JOURNAL_PATH = Path("reports/status/events/events.jsonl")
# Seeded-defect drills write here, NOT to the compliance journal above.  The
# disclosure/compliance surface consumes only EVENT_JOURNAL_PATH, so rehearsal
# drill events can never pollute a released artifact's provenance.  This path is
# gitignored: running drills must never dirty the committed tree.
DRILL_JOURNAL_PATH = Path("reports/status/drills/drill_events.jsonl")
DEFAULT_ESCALATION_SINK = {
    "type": "file",
    "target": "reports/status/events/escalations.jsonl",
}

# §5.4's standing human-escalation channel.  The operator runbook is checked
# against this registry so adding a class cannot silently omit its playbook.
ESCALATION_CLASSES = (
    "judge_disagreement",
    "budget_breach",
    "blocked_with_human",
    "unsatisfiable_constraints",
    "hypothesis_task_retirement",
)


def _utc_now_iso() -> str:
    return (
        dt.datetime.now(tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _compact_json(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"), sort_keys=True)


def _append_json_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _compact_json(record) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def append_event(
    repo: Path,
    event: dict,
    *,
    actor_session: str | None = None,
    journal_path: Path | None = None,
) -> dict:
    """Append one enriched event to a crash-durable journal.

    Defaults to the compliance journal (``EVENT_JOURNAL_PATH``).  Callers that
    must stay out of the compliance/disclosure surface — the seeded-defect
    drills — pass ``journal_path=DRILL_JOURNAL_PATH`` so their events land in a
    separate ledger the disclosure never reads.
    """
    event_name = event.get("event")
    if not isinstance(event_name, str) or not event_name.strip():
        raise ValueError("event must be a non-empty string")

    record = dict(event)
    record["schema_version"] = EVENT_SCHEMA_VERSION
    record["ts_utc"] = _utc_now_iso()
    record["actor_session"] = (
        actor_session if actor_session is not None else event.get("actor_session")
    )
    _append_json_line(Path(repo) / (journal_path or EVENT_JOURNAL_PATH), record)
    return record


def read_events(repo: Path, *, journal_path: Path | None = None) -> tuple[list[dict], int]:
    """Read valid journal records, skipping malformed/torn lines.

    Defaults to the compliance journal; pass ``journal_path=DRILL_JOURNAL_PATH``
    to read the seeded-defect drill ledger.
    """
    path = Path(repo) / (journal_path or EVENT_JOURNAL_PATH)
    if not path.is_file():
        return [], 0

    events: list[dict] = []
    malformed_count = 0
    with open(path, "rb") as handle:
        for raw_line in handle:
            try:
                line = raw_line.decode("utf-8")
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed_count += 1
                continue
            if not isinstance(record, dict):
                malformed_count += 1
                continue
            events.append(record)
    return events, malformed_count


def escalation_sink_config(repo: Path) -> dict:
    """Return the configured offline escalation sink or its file default."""
    path = Path(repo) / "contracts" / "framework.json"
    try:
        framework = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_ESCALATION_SINK)

    sink = framework.get("escalation_sink") if isinstance(framework, dict) else None
    if not isinstance(sink, dict):
        return dict(DEFAULT_ESCALATION_SINK)
    sink_type = sink.get("type")
    target = sink.get("target")
    if sink_type not in {"file", "command", "webhook"}:
        return dict(DEFAULT_ESCALATION_SINK)
    if not isinstance(target, str) or not target.strip():
        return dict(DEFAULT_ESCALATION_SINK)
    return {"type": sink_type, "target": target}


def escalation_class_of(event: dict) -> str | None:
    """Validate and return an event's human-escalation class, if any.

    Escalations form two tiers.  ``escalation=True`` alone marks an *operational*
    notable event (auto-recovery, refusal, retry) that needs no human playbook.
    An event that additionally carries an ``escalation_class`` is a *human*
    escalation on §5.4's standing channel — every such class MUST be one of the
    registered ``ESCALATION_CLASSES`` (which ``runbook_staleness`` then forces to
    carry a playbook).  A new human-escalation class therefore cannot ship
    without being registered here and documented in the runbook.
    """
    escalation_class = event.get("escalation_class")
    if escalation_class is None:
        return None
    if escalation_class not in ESCALATION_CLASSES:
        raise ValueError(f"unregistered_escalation_class:{escalation_class}")
    return escalation_class


def escalate(
    repo: Path,
    event: dict,
    *,
    actor_session: str | None = None,
) -> dict:
    """Journal an escalation, then deliver it without raising delivery errors."""
    escalation_class_of(event)  # reject an unregistered human-escalation class
    record = append_event(
        Path(repo),
        {**event, "escalation": True},
        actor_session=actor_session,
    )
    sink = escalation_sink_config(Path(repo))
    sink_type = sink["type"]
    target = sink["target"]

    if sink_type == "webhook":
        record["delivery"] = "deferred:webhook"
        return record

    if sink_type == "file":
        try:
            _append_json_line(Path(repo) / target, record)
            record["delivery"] = "delivered:file"
        except OSError as exc:
            record["delivery_error"] = f"{type(exc).__name__}:{exc}"
        return record

    try:
        argv = shlex.split(target)
        if not argv:
            raise ValueError("empty command target")
        completed = subprocess.run(
            argv,
            input=_compact_json(record) + "\n",
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            suffix = f":{detail[:200]}" if detail else ""
            record["delivery_error"] = f"command_exit:{completed.returncode}{suffix}"
        else:
            record["delivery"] = "delivered:command"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        record["delivery_error"] = f"{type(exc).__name__}:{exc}"
    return record
