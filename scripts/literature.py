#!/usr/bin/env python3
"""Provenance-disciplined literature corpus utilities.

Network acquisition is explicit.  Tests and routine gates use ``--fixture-dir``
and never contact the network.  Every acquired payload is immutable once
written, hash-bound by an adjacent acquisition manifest, and is the sole source
from which the bibliography is generated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request
from typing import Any


CORPUS_SCHEMA_VERSION = "research_swarm.literature_manifest.v1"
RECALL_SCHEMA_VERSION = "research_swarm.recall_audit.v1"
REQUEST_SCHEMA_VERSION = "research_swarm.literature_request.v1"
RECALL_REQUEST_SCHEMA_VERSION = "research_swarm.recall_search.v1"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"top_level_not_object:{path}")
    return payload


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists():
        raise ValueError(f"append_only_path_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"invalid_path_component:{value}")
    return cleaned


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = payload.get("search_strategy")
    if not isinstance(strategy, dict):
        raise ValueError("missing_search_strategy")
    for field in ("databases", "queries", "inclusion_criteria"):
        value = strategy.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"invalid_search_strategy:{field}")
    family = strategy.get("executor_family")
    if not isinstance(family, str) or not family.strip():
        raise ValueError("invalid_search_strategy:executor_family")
    return strategy


def _fetch(entry: dict[str, Any], fixture_dir: Path | None, allow_network: bool) -> bytes:
    fixture = entry.get("fixture")
    if fixture_dir is not None:
        if not isinstance(fixture, str) or not fixture.strip():
            raise ValueError(f"missing_fixture:{entry.get('citekey')}")
        path = (fixture_dir / fixture).resolve()
        try:
            path.relative_to(fixture_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"fixture_outside_root:{fixture}") from exc
        return path.read_bytes()
    if not allow_network:
        raise ValueError("network_fetch_requires_explicit_allow_network")
    url = entry.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError(f"invalid_fetch_url:{entry.get('citekey')}")
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - explicit opt-in
        return response.read()


def acquire(
    *,
    repo: Path,
    request_path: Path,
    retrieval_date: dt.date,
    fixture_dir: Path | None,
    allow_network: bool,
) -> Path:
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("invalid_literature_request_schema")
    strategy = _validate_strategy(request)
    acquisition_id = request.get("acquisition_id")
    if not isinstance(acquisition_id, str) or not acquisition_id.strip():
        raise ValueError("invalid_acquisition_id")
    entries = request.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("invalid_literature_entries")

    root = repo / "data" / "raw" / "literature" / retrieval_date.isoformat()
    manifest_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"literature_entry_not_object:{index}")
        citekey = item.get("citekey")
        title = item.get("title")
        if not isinstance(citekey, str) or not citekey.strip() or citekey in seen:
            raise ValueError(f"invalid_or_duplicate_citekey:{citekey}")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"invalid_title:{citekey}")
        if not any(isinstance(item.get(field), str) and item.get(field).strip() for field in ("url", "doi")):
            raise ValueError(f"missing_url_or_doi:{citekey}")
        seen.add(citekey)
        raw = _fetch(item, fixture_dir, allow_network)
        extension = item.get("format", "txt")
        if extension not in {"txt", "pdf", "json", "html"}:
            raise ValueError(f"invalid_snapshot_format:{citekey}:{extension}")
        relpath = Path("data/raw/literature") / retrieval_date.isoformat() / (
            f"{_safe_component(citekey)}.{extension}"
        )
        _write_new(repo / relpath, raw)
        manifest_entries.append(
            {
                "citekey": citekey,
                "title": title,
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "url": item.get("url"),
                "doi": item.get("doi"),
                "retrieved_on": retrieval_date.isoformat(),
                "snapshot_path": relpath.as_posix(),
                "snapshot_sha256": _sha256(raw),
                "snapshot_bytes": len(raw),
                "cluster": item.get("cluster"),
            }
        )
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "acquisition_id": acquisition_id,
        "retrieved_on": retrieval_date.isoformat(),
        "search_strategy": strategy,
        "entries": sorted(manifest_entries, key=lambda item: item["citekey"]),
    }
    manifest_path = root / f"manifest_{_safe_component(acquisition_id)}.json"
    _write_new(manifest_path, _canonical_json(manifest))
    return manifest_path


def load_corpus(repo: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    corpus: dict[str, dict[str, Any]] = {}
    manifests = sorted((repo / "data/raw/literature").glob("????-??-??/manifest_*.json"))
    for path in manifests:
        payload = _read_json(path)
        if payload.get("schema_version") != CORPUS_SCHEMA_VERSION:
            raise ValueError(f"invalid_literature_manifest_schema:{path}")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"invalid_literature_manifest_entries:{path}")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("citekey"), str):
                raise ValueError(f"invalid_literature_manifest_entry:{path}")
            key = entry["citekey"]
            if key in corpus:
                raise ValueError(f"duplicate_corpus_citekey:{key}")
            corpus[key] = dict(entry)
    return corpus, manifests


def _bib_escape(value: object) -> str:
    return str(value).replace("{", "\\{").replace("}", "\\}")


def _local_bib_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    cursor = 0
    opener_for = {"{": "}", "(": ")"}
    while True:
        match = re.search(r"@\w+\s*([({])", text[cursor:], flags=re.IGNORECASE)
        if match is None:
            break
        start = cursor + match.start()
        opener = match.group(1)
        closer = opener_for[opener]
        index = cursor + match.end()
        depth = 1
        while index < len(text) and depth:
            depth += text[index] == opener
            depth -= text[index] == closer
            index += 1
        block = text[start:index].strip()
        if re.search(r"(?im)^\s*note\s*=.*\bPath:\s*", block):
            blocks.append(block)
        cursor = max(index, start + 1)
    return blocks


def generate_bib(*, repo: Path, output: Path) -> None:
    corpus, _ = load_corpus(repo)
    local_blocks = _local_bib_blocks(output.read_text(encoding="utf-8")) if output.is_file() else []
    blocks: list[str] = []
    for citekey, entry in sorted(corpus.items()):
        authors = entry.get("authors")
        author_text = " and ".join(str(author) for author in authors) if isinstance(authors, list) else ""
        fields = [f"  title = {{{_bib_escape(entry.get('title', ''))}}}"]
        if author_text:
            fields.append(f"  author = {{{_bib_escape(author_text)}}}")
        if entry.get("year") is not None:
            fields.append(f"  year = {{{_bib_escape(entry['year'])}}}")
        if isinstance(entry.get("doi"), str) and entry["doi"].strip():
            fields.append(f"  doi = {{{_bib_escape(entry['doi'])}}}")
        if isinstance(entry.get("url"), str) and entry["url"].strip():
            fields.append(f"  url = {{{_bib_escape(entry['url'])}}}")
        evidence = f"Retrieval-Evidence: {entry['snapshot_path']}#{entry['snapshot_sha256']}"
        fields.append(f"  note = {{{_bib_escape(evidence)}}}")
        blocks.append(f"@article{{{citekey},\n" + ",\n".join(fields) + "\n}")
    output.parent.mkdir(parents=True, exist_ok=True)
    all_blocks = blocks + local_blocks
    output.write_text("\n\n".join(all_blocks) + ("\n" if all_blocks else ""), encoding="utf-8")


def recall_audit(
    *, repo: Path, search_path: Path, output: Path, cluster_threshold: int
) -> dict[str, Any]:
    payload = _read_json(search_path)
    if payload.get("schema_version") != RECALL_REQUEST_SCHEMA_VERSION:
        raise ValueError("invalid_recall_search_schema")
    strategy = _validate_strategy(payload)
    primary = payload.get("primary_search_strategy")
    if not isinstance(primary, dict):
        raise ValueError("missing_primary_search_strategy")
    primary_family = primary.get("executor_family")
    independent = (
        strategy.get("executor_family") != primary_family
        and set(strategy.get("queries", [])) != set(primary.get("queries", []))
        and set(strategy.get("databases", [])) != set(primary.get("databases", []))
    )
    corpus, _ = load_corpus(repo)
    retrieved = payload.get("retrieved")
    if not isinstance(retrieved, list):
        raise ValueError("invalid_recall_retrieved")
    uncovered: list[dict[str, Any]] = []
    by_cluster: dict[str, list[str]] = {}
    for item in retrieved:
        if not isinstance(item, dict) or not isinstance(item.get("citekey"), str):
            raise ValueError("invalid_recall_retrieved_entry")
        if item["citekey"] in corpus:
            continue
        cluster = item.get("cluster")
        label = cluster.strip() if isinstance(cluster, str) and cluster.strip() else "unclustered"
        by_cluster.setdefault(label, []).append(item["citekey"])
    for cluster, keys in sorted(by_cluster.items()):
        if len(keys) >= cluster_threshold:
            uncovered.append({"cluster": cluster, "citekeys": sorted(keys), "count": len(keys)})
    report = {
        "schema_version": RECALL_SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "primary_family": primary_family,
        "recall_family": strategy.get("executor_family"),
        "primary_search_strategy": primary,
        "recall_search_strategy": strategy,
        "independent_search": independent,
        "retrieved_count": len(retrieved),
        "corpus_count": len(corpus),
        "uncovered_clusters": uncovered,
        "requires_human_escalation": bool(uncovered),
        "human_escalation": "@human: review uncovered recall clusters before synthesis" if uncovered else None,
        "synthesis_blocked": bool(uncovered) or not independent,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="literature.py")
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_p = sub.add_parser("acquire", help="Snapshot sources into the append-only literature corpus")
    acquire_p.add_argument("--request", required=True, type=Path)
    acquire_p.add_argument("--retrieval-date", required=True)
    acquire_p.add_argument("--fixture-dir", type=Path)
    acquire_p.add_argument("--allow-network", action="store_true")
    acquire_p.add_argument("--repo-root", type=Path, default=Path.cwd())
    bib_p = sub.add_parser("generate-bib", help="Generate BibTeX solely from corpus manifests")
    bib_p.add_argument("--output", type=Path, default=Path("reports/paper/references.bib"))
    bib_p.add_argument("--repo-root", type=Path, default=Path.cwd())
    recall_p = sub.add_parser("recall-audit", help="Diff an independent search against the corpus")
    recall_p.add_argument("--search", required=True, type=Path)
    recall_p.add_argument("--output", required=True, type=Path)
    recall_p.add_argument("--cluster-threshold", type=int, default=2)
    recall_p.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo = args.repo_root.resolve()
        if args.command == "acquire":
            path = acquire(
                repo=repo,
                request_path=args.request.resolve(),
                retrieval_date=dt.date.fromisoformat(args.retrieval_date),
                fixture_dir=args.fixture_dir.resolve() if args.fixture_dir else None,
                allow_network=bool(args.allow_network),
            )
            print(path.relative_to(repo).as_posix())
        elif args.command == "generate-bib":
            output = args.output if args.output.is_absolute() else repo / args.output
            generate_bib(repo=repo, output=output)
            print(output.relative_to(repo).as_posix())
        else:
            output = args.output if args.output.is_absolute() else repo / args.output
            report = recall_audit(
                repo=repo,
                search_path=args.search.resolve(),
                output=output,
                cluster_threshold=args.cluster_threshold,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1 if report["synthesis_blocked"] else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"literature_error:{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
