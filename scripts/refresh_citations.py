#!/usr/bin/env python3
"""Deterministic citation refresh adapter for pre-captured offline fixtures.

This batch deliberately has no live-network implementation. A network-enabled
refresh task must capture one JSON fixture per citation before invoking this
adapter. Fixture shape::

    {
      "schema_version": "research_swarm.citation_fixture.v1",
      "citekey": "smith2024",
      "source": "crossref",
      "retrieved_at_utc": "2026-07-10T12:00:00Z",
      "raw_response": {"the": "captured provider response"},
      "normalized": {
        "doi": "10.1234/example",
        "title": "Example",
        "resolved": true,
        "retraction_status": "none",
        "url_resolves": true
      }
    }

The raw-response value is hashed in canonical JSON form. Output paths and the
AS_OF value are derived entirely from fixture content, so repeated runs are
byte-stable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys


FIXTURE_SCHEMA_VERSION = "research_swarm.citation_fixture.v1"
SNAPSHOT_SCHEMA_VERSION = "research_swarm.citation_snapshot.v1"
_SAFE_CITEKEY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _parse_utc(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("retrieved_at_utc_must_be_utc_z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("retrieved_at_utc_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("retrieved_at_utc_must_be_utc_z")
    return parsed


def _canonical_raw_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fixture_to_snapshot(payload: object, *, fixture_path: Path) -> tuple[str, str, dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{fixture_path}:fixture_must_be_object")
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"{fixture_path}:invalid_fixture_schema")

    citekey = payload.get("citekey")
    if not isinstance(citekey, str) or not _SAFE_CITEKEY_RE.fullmatch(citekey):
        raise ValueError(f"{fixture_path}:invalid_citekey")
    source = payload.get("source")
    if source not in {"crossref", "openalex", "s2"}:
        raise ValueError(f"{fixture_path}:invalid_source")
    retrieved_at_utc = payload.get("retrieved_at_utc")
    retrieved_at = _parse_utc(retrieved_at_utc)
    if "raw_response" not in payload:
        raise ValueError(f"{fixture_path}:missing_raw_response")
    normalized = payload.get("normalized")
    if not isinstance(normalized, dict):
        raise ValueError(f"{fixture_path}:normalized_must_be_object")

    title = normalized.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{fixture_path}:missing_title")
    for key in ("resolved", "url_resolves"):
        if not isinstance(normalized.get(key), bool):
            raise ValueError(f"{fixture_path}:{key}_must_be_boolean")
    if normalized.get("retraction_status") not in {"none", "retracted", "unknown"}:
        raise ValueError(f"{fixture_path}:invalid_retraction_status")

    snapshot: dict[str, object] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "citekey": citekey,
        "title": title.strip(),
        "source": source,
        "retrieved_at_utc": retrieved_at_utc,
        "retrieval_sha256": hashlib.sha256(
            _canonical_raw_bytes(payload["raw_response"])
        ).hexdigest(),
        "resolved": normalized["resolved"],
        "retraction_status": normalized["retraction_status"],
        "url_resolves": normalized["url_resolves"],
    }
    doi = normalized.get("doi")
    if doi is not None:
        if not isinstance(doi, str) or not doi.strip():
            raise ValueError(f"{fixture_path}:invalid_doi")
        snapshot["doi"] = doi.strip()
    return citekey, retrieved_at.date().isoformat(), snapshot


def refresh_from_fixtures(fixture_dir: Path, output_root: Path) -> list[Path]:
    fixture_paths = sorted(fixture_dir.glob("*.json"))
    if not fixture_paths:
        raise ValueError(f"no_fixture_json:{fixture_dir}")

    parsed_fixtures: list[tuple[str, str, dict[str, object]]] = []
    seen: set[str] = set()
    for fixture_path in fixture_paths:
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{fixture_path}:invalid_json:{exc}") from exc
        citekey, date, snapshot = _fixture_to_snapshot(payload, fixture_path=fixture_path)
        if citekey in seen:
            raise ValueError(f"duplicate_fixture:{citekey}")
        seen.add(citekey)
        parsed_fixtures.append((citekey, date, snapshot))

    as_of = max(date for _, date, _ in parsed_fixtures)
    outputs: list[Path] = []
    for citekey, _, snapshot in parsed_fixtures:
        output = output_root / as_of / f"{citekey}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs.append(output)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "AS_OF").write_text(as_of + "\n", encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-fixture",
        type=Path,
        required=True,
        help="Directory containing pre-captured citation fixture JSON files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/citations"),
        help="Snapshot root (default: data/citations)",
    )
    args = parser.parse_args(argv)
    try:
        outputs = refresh_from_fixtures(args.offline_fixture, args.output_root)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"count": len(outputs), "outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
