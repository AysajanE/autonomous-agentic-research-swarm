#!/usr/bin/env python3
"""Score a referee replay against the pre-committed human gold set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


CALIBRATION_REPORT_SCHEMA_VERSION = "research_swarm.referee_calibration.v1"
GOLD_KEY_SCHEMA_VERSION = "research_swarm.referee_gold_key.v1"
MOCK_GOLD_SCHEMA_VERSION = "research_swarm.mock_referee_gold.v1"
VERDICTS = {"supported", "not_supported", "cannot_verify"}


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bar(path: Path) -> dict[str, object]:
    bar = _read_object(path)
    required = {"agreement_floor", "position_flip_ceiling", "committed_by", "committed_at_utc"}
    if set(bar) != required:
        raise ValueError(f"calibration_bar_fields_invalid:{sorted(bar)}")
    agreement = bar.get("agreement_floor")
    flips = bar.get("position_flip_ceiling")
    if not isinstance(agreement, (int, float)) or isinstance(agreement, bool) or not 0 <= float(agreement) <= 1:
        raise ValueError("calibration_agreement_floor_invalid")
    if not isinstance(flips, (int, float)) or isinstance(flips, bool) or not 0 <= float(flips) <= 1:
        raise ValueError("calibration_position_flip_ceiling_invalid")
    if not isinstance(bar.get("committed_by"), str) or not str(bar["committed_by"]).strip():
        raise ValueError("calibration_committed_by_invalid")
    timestamp = bar.get("committed_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("calibration_committed_at_invalid")
    try:
        dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("calibration_committed_at_invalid") from exc
    return bar


def commit_bar(
    *,
    path: Path,
    output_path: Path,
    gold_key_path: Path | None,
    agreement_floor: float,
    position_flip_ceiling: float,
    committed_by: str,
) -> None:
    if output_path.exists():
        raise ValueError("calibration_bar_locked:grading_already_started")
    if gold_key_path is not None and gold_key_path.exists():
        try:
            gold = _read_object(gold_key_path)
        except (OSError, ValueError, json.JSONDecodeError):
            gold = {}
        if gold.get("graded_at_utc") is not None or gold.get("cases"):
            raise ValueError("calibration_bar_locked:grading_already_started")
    if not committed_by.strip():
        raise ValueError("calibration_committed_by_invalid")
    if not 0 <= agreement_floor <= 1 or not 0 <= position_flip_ceiling <= 1:
        raise ValueError("calibration_bar_out_of_range")
    payload = {
        "agreement_floor": agreement_floor,
        "position_flip_ceiling": position_flip_ceiling,
        "committed_by": committed_by.strip(),
        "committed_at_utc": _utc_now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def calibrate(
    *,
    calibration_path: Path,
    gold_dir: Path,
    mock_path: Path,
    output_path: Path | None,
) -> dict[str, object]:
    bar = _load_bar(calibration_path)
    key_path = gold_dir / "verdict_key.json"
    gold = _read_object(key_path)
    mock = _read_object(mock_path)
    if gold.get("schema_version") != GOLD_KEY_SCHEMA_VERSION:
        raise ValueError(f"gold_key_schema_invalid:{gold.get('schema_version')}")
    if mock.get("schema_version") != MOCK_GOLD_SCHEMA_VERSION:
        raise ValueError(f"mock_gold_schema_invalid:{mock.get('schema_version')}")

    raw_cases = gold.get("cases")
    raw_predictions = mock.get("predictions")
    if not isinstance(raw_cases, list) or not isinstance(raw_predictions, list):
        raise ValueError("gold_cases_or_predictions_invalid")
    cases = {
        item["case_id"]: item
        for item in raw_cases
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    predictions = {
        item["case_id"]: item
        for item in raw_predictions
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(cases) != len(raw_cases) or set(cases) != set(predictions):
        raise ValueError("gold_prediction_case_mismatch")

    wrong_artifacts = 0
    wrong_proofs = 0
    agreements = 0
    flips = 0
    scored: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}
    for case_id in sorted(cases):
        case = cases[case_id]
        prediction = predictions[case_id]
        human = case.get("human_verdict")
        first = prediction.get("position_a")
        second = prediction.get("position_b")
        if human not in VERDICTS or first not in VERDICTS or second not in VERDICTS:
            raise ValueError(f"gold_verdict_invalid:{case_id}")
        artifact_rel = case.get("artifact")
        if not isinstance(artifact_rel, str) or not artifact_rel.strip():
            raise ValueError(f"gold_artifact_invalid:{case_id}")
        artifact_path = (gold_dir / artifact_rel).resolve()
        try:
            artifact_path.relative_to(gold_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"gold_artifact_outside_dir:{case_id}") from exc
        if not artifact_path.is_file():
            raise ValueError(f"gold_artifact_missing:{case_id}:{artifact_rel}")
        artifact_hashes[artifact_rel] = _sha256(artifact_path)
        agreement = first == human
        flipped = first != second
        agreements += int(agreement)
        flips += int(flipped)
        if human == "not_supported" and case.get("kind") == "artifact":
            wrong_artifacts += 1
        if human == "not_supported" and case.get("kind") == "proof":
            wrong_proofs += 1
        scored.append(
            {
                "case_id": case_id,
                "human_verdict": human,
                "position_a": first,
                "position_b": second,
                "agreement": agreement,
                "position_flip": flipped,
            }
        )
    if wrong_artifacts < 3:
        raise ValueError(f"gold_wrong_artifact_floor_not_met:{wrong_artifacts}")
    if wrong_proofs < 3:
        raise ValueError(f"gold_wrong_proof_floor_not_met:{wrong_proofs}")

    count = len(cases)
    agreement_rate = agreements / count if count else 0.0
    flip_rate = flips / count if count else 1.0
    calibrated = (
        agreement_rate >= float(bar["agreement_floor"])
        and flip_rate <= float(bar["position_flip_ceiling"])
    )
    digest_material = {
        "verdict_key_sha256": _sha256(key_path),
        "artifacts": artifact_hashes,
    }
    result: dict[str, object] = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "backend": "mock",
        "profile": mock.get("profile"),
        "calibration_path": calibration_path.as_posix(),
        "calibration_sha256": _sha256(calibration_path),
        "gold_set_sha256": hashlib.sha256(
            json.dumps(digest_material, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "mock_transcript_sha256": _sha256(mock_path),
        "case_count": count,
        "wrong_artifact_count": wrong_artifacts,
        "wrong_proof_count": wrong_proofs,
        "agreement": agreement_rate,
        "position_flip_rate": flip_rate,
        "agreement_floor": float(bar["agreement_floor"]),
        "position_flip_ceiling": float(bar["position_flip_ceiling"]),
        "calibrated": calibrated,
        "cases": scored,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_referee.py",
        description="Commit a calibration bar before grading or score mock referee replays.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--mock", type=Path, default=None, help="Mock gold-set prediction transcript")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--commit", action="store_true", help="Pre-commit the bar; refused after grading artifacts exist")
    parser.add_argument("--committed-by", default=None)
    parser.add_argument("--agreement-floor", type=float, default=0.80)
    parser.add_argument("--position-flip-ceiling", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo_root.resolve()
    calibration_path = (args.calibration or repo / "contracts/rubrics/calibration.yaml").resolve()
    gold_dir = (args.gold_dir or repo / "tests/gold_set").resolve()
    mock_path = (args.mock or gold_dir / "mock_referee.json").resolve()
    output_path = (args.output or repo / "reports/status/referee_calibration.json").resolve()
    try:
        if args.commit:
            if args.no_write:
                raise ValueError("calibration_commit_requires_write")
            if not isinstance(args.committed_by, str):
                raise ValueError("calibration_commit_requires_committed_by")
            commit_bar(
                path=calibration_path,
                output_path=output_path,
                gold_key_path=gold_dir / "verdict_key.json",
                agreement_floor=args.agreement_floor,
                position_flip_ceiling=args.position_flip_ceiling,
                committed_by=args.committed_by,
            )
            print(json.dumps({"ok": True, "committed": calibration_path.as_posix()}, indent=2))
            return 0
        result = calibrate(
            calibration_path=calibration_path,
            gold_dir=gold_dir,
            mock_path=mock_path,
            output_path=None if args.no_write else output_path,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["calibrated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
