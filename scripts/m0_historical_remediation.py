#!/usr/bin/env python3
"""Batch G generator — historical-record remediation (plan §4.0, M0).

Owner-run, deliberately NOT part of the kernel: this is a one-time annotation
of the battle-test record. It writes:
  1. provenance annotations for all 32 run manifests (sidecars; originals untouched)
  2. rebaseline sidecars: `superseded` for the 6 stale processed manifests,
     `raw_evidence_unavailable` for the 6 raw manifests
  3. contracts/historical_exemptions.json (per-file sha256, gate-verified)
  4. release_2026-04-11.json amendment (notes entry; per-class run counts)

Run with --check first (prints plan, writes nothing), then --write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]

PROVENANCE_ANNOTATION_SCHEMA = "research_swarm.provenance_annotation.v1"
REBASELINE_SCHEMA = "research_swarm.manifest_rebaseline.v1"
EXEMPTIONS_SCHEMA = "research_swarm.historical_exemptions.v1"

SUPERSEDED = {
    # stale manifest -> accurate successor covering the same output surface
    "daily_l1_rent_decomposition_2026-04-01.json": "daily_l1_rent_decomposition_2026-04-09.json",
    "daily_l1_rent_decomposition_2026-04-08.json": "daily_l1_rent_decomposition_2026-04-09.json",
    "daily_rollup_panel_2026-04-01.json": "daily_rollup_panel_2026-04-09.json",
    "daily_rollup_panel_2026-04-08.json": "daily_rollup_panel_2026-04-09.json",
    "vendor_daily_rollup_panel_2026-04-01.json": "vendor_daily_rollup_panel_2026-04-09.json",
    "vendor_daily_rollup_panel_2026-04-08.json": "vendor_daily_rollup_panel_2026-04-09.json",
}

SUPERSEDED_NOTE = (
    "Honest re-baseline of the battle-test record (plan §4.0 #11): this manifest's "
    "output hash claims refer to an earlier generation of the named output paths; "
    "the outputs were rebuilt during the 2026-04-08/09 repair campaign and the "
    "2026-04-09 manifest for the same surface matches current disk. The original "
    "manifest file is preserved byte-for-byte (it is hash-inventoried by "
    "release_2026-04-11). This annotation supersedes, it does NOT claim regeneration; "
    "the raw evidence layer needed for regeneration was deleted and no such claim is made."
)

RAW_UNAVAILABLE_NOTE = (
    "Honest re-baseline of the battle-test record (plan §4.0 #11 + §7.3): every file "
    "attested by this raw manifest was deleted from disk after the release and exists "
    "in no archive; 135,648 manifested raw files are unavailable across the six raw "
    "manifests. The manifest is preserved as the only surviving attestation of what "
    "was fetched. Nothing here claims the data can be reproduced; the release record "
    "is amended with raw_evidence_unavailable. Future snapshots fall under the "
    "M4 raw-archive retention contract."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def classify_run_manifest(payload: dict) -> str:
    executor = payload.get("executor", {})
    runner = executor.get("runner")
    tool = executor.get("tool")
    if runner == "legacy_backfill" or tool == "operator_backfill":
        return "backfill"
    if tool == "codex":
        return "executor_run"
    if tool == "manual":
        return "manual_operator"
    raise SystemExit(f"unclassifiable manifest: runner={runner} tool={tool}")


def write_json(path: Path, payload: dict, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(("WROTE " if write else "PLAN  ") + str(path.relative_to(REPO)))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    write = bool(args.write)
    stamp = now_iso()

    exemptions: dict[str, list[dict[str, str]]] = {"run_manifests": [], "review_logs": []}
    class_counts: dict[str, int] = {"executor_run": 0, "manual_operator": 0, "backfill": 0}

    # 1. provenance annotations for the 32 run manifests
    runs_dir = REPO / "reports/status/swarm_runs"
    for manifest_path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        klass = classify_run_manifest(payload)
        class_counts[klass] += 1
        rel = manifest_path.relative_to(REPO).as_posix()
        digest = sha256_file(manifest_path)
        annotation = {
            "schema_version": PROVENANCE_ANNOTATION_SCHEMA,
            "annotates": rel,
            "annotates_sha256": digest,
            "provenance_class": klass,
            "rationale": (
                "M0 historical-record remediation: provenance_class backfilled from the "
                "manifest's own executor fields (runner/tool), matching the battle-test "
                "forensics (18 executor runs, 9 manual operator records, 5 legacy "
                "backfills). The original manifest is untouched; it predates schema v2."
            ),
            "annotated_at_utc": stamp,
        }
        write_json(
            runs_dir / "annotations" / f"{manifest_path.name}.provenance.json",
            annotation,
            write,
        )
        exemptions["run_manifests"].append(
            {"path": rel, "sha256": digest, "schema_version": str(payload.get("schema_version"))}
        )

    # 2a. superseded rebaselines for the 6 stale processed manifests
    pm_dir = REPO / "data/processed_manifest"
    for stale_name, successor_name in sorted(SUPERSEDED.items()):
        stale = pm_dir / stale_name
        successor = pm_dir / successor_name
        assert stale.exists() and successor.exists(), (stale, successor)
        rebaseline = {
            "schema_version": REBASELINE_SCHEMA,
            "rebaseline_of": stale.relative_to(REPO).as_posix(),
            "original_manifest_sha256": sha256_file(stale),
            "mode": "superseded",
            "superseded_by": successor.relative_to(REPO).as_posix(),
            "provenance_note": SUPERSEDED_NOTE,
            "rebaselined_at_utc": stamp,
        }
        write_json(pm_dir / "rebaselines" / f"{stale_name}.rebaseline.json", rebaseline, write)

    # 2b. raw_evidence_unavailable rebaselines for the 6 raw manifests
    rm_dir = REPO / "data/raw_manifest"
    for raw_path in sorted(rm_dir.glob("*.json")):
        rebaseline = {
            "schema_version": REBASELINE_SCHEMA,
            "rebaseline_of": raw_path.relative_to(REPO).as_posix(),
            "original_manifest_sha256": sha256_file(raw_path),
            "mode": "raw_evidence_unavailable",
            "entries": "all",
            "provenance_note": RAW_UNAVAILABLE_NOTE,
            "rebaselined_at_utc": stamp,
        }
        write_json(rm_dir / "rebaselines" / f"{raw_path.name}.rebaseline.json", rebaseline, write)

    # 3. historical exemptions (v1-schema artifacts; strict checks apply to v2+)
    reviews_dir = REPO / "reports/status/reviews"
    for review_path in sorted(reviews_dir.glob("*.json")):
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        exemptions["review_logs"].append(
            {
                "path": review_path.relative_to(REPO).as_posix(),
                "sha256": sha256_file(review_path),
                "schema_version": str(payload.get("schema_version")),
            }
        )
    exemptions_payload = {
        "schema_version": EXEMPTIONS_SCHEMA,
        "created_at_utc": stamp,
        "rationale": (
            "Gate-scoping rule (plan §4.0 remediation): strict v2 checks apply to "
            "schema_version >= 2 artifacts; the battle-test's v1 artifacts live on this "
            "checked-in, hash-pinned exemption list. New-schema strictness never "
            "silently rewrites what an old release attested."
        ),
        "run_manifests": exemptions["run_manifests"],
        "review_logs": exemptions["review_logs"],
    }
    write_json(REPO / "contracts/historical_exemptions.json", exemptions_payload, write)

    # 4. release amendment
    release_path = REPO / "reports/status/releases/release_2026-04-11.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    notes = release.get("notes")
    assert isinstance(notes, list)
    already = [n for n in notes if isinstance(n, dict) and n.get("type") == "raw_evidence_unavailable"]
    if not already:
        notes.append(
            {
                "type": "raw_evidence_unavailable",
                "amended_at_utc": stamp,
                "explanation": (
                    "Post-release amendment (M0 remediation): the raw evidence layer this "
                    "release attested (135,648 files across six raw manifests) was deleted "
                    "after release assembly and archived nowhere; it cannot be recovered or "
                    "regenerated. Ten of twenty processed-manifest output hash claims were "
                    "stale at attestation time against the 04-01/04-08 manifest generations; "
                    "the 2026-04-09 generation matches disk and honest superseded-rebaseline "
                    "sidecars now record this without altering the original manifests."
                ),
                "provenance_class_run_counts": class_counts,
                "provenance_annotations_dir": "reports/status/swarm_runs/annotations/",
                "rebaselines_dirs": [
                    "data/processed_manifest/rebaselines/",
                    "data/raw_manifest/rebaselines/",
                ],
            }
        )
        if write:
            release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(("WROTE " if write else "PLAN  ") + str(release_path.relative_to(REPO)) + " (notes amendment)")
    else:
        print("release already amended; skipping")

    print(f"class_counts: {class_counts}")
    expected = {"executor_run": 18, "manual_operator": 9, "backfill": 5}
    if class_counts != expected:
        raise SystemExit(f"classification drifted from forensics: {class_counts} != {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
