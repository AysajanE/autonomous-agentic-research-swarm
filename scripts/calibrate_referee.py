#!/usr/bin/env python3
"""Score a referee replay against the pre-committed human gold set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


CALIBRATION_REPORT_SCHEMA_VERSION = "research_swarm.referee_calibration.v1"
GOLD_KEY_SCHEMA_VERSION = "research_swarm.referee_gold_key.v1"
MOCK_GOLD_SCHEMA_VERSION = "research_swarm.mock_referee_gold.v1"
GOLD_PREDICTIONS_SCHEMA_VERSION = "research_swarm.referee_gold_predictions.v1"
VERDICTS = {"supported", "not_supported", "cannot_verify"}
REFEREE_PROMPT_PATH = "docs/prompts/referee.md"


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repo_root_from_calibration(calibration_path: Path) -> Path:
    resolved = calibration_path.resolve()
    if resolved.parent.name != "rubrics" or resolved.parent.parent.name != "contracts":
        raise ValueError("calibration_path_outside_contract_rubrics")
    return resolved.parents[2]


def _rubric_bundle_sha256(repo: Path) -> str:
    rubric_dir = repo / "contracts" / "rubrics"
    entries = []
    for path in sorted(rubric_dir.glob("*.yaml")):
        if path.name == "calibration.yaml":
            continue
        entries.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": _sha256(path),
            }
        )
    if not entries:
        raise ValueError("calibration_rubric_bundle_empty")
    return _canonical_sha256(entries)


def _panel_members(repo: Path) -> list[dict[str, object]]:
    framework = _read_object(repo / "contracts" / "framework.json")
    executors = framework.get("executors")
    panel = executors.get("referee_panel") if isinstance(executors, dict) else None
    if not isinstance(panel, list):
        return []
    return [dict(item) for item in panel if isinstance(item, dict)]


def _evaluated_binding(
    *,
    repo: Path,
    transcript: dict[str, object],
) -> dict[str, object]:
    backend = transcript.get("backend", "mock")
    family = transcript.get("family", "mock")
    model = transcript.get("model", "mock-referee-v1")
    cli_version = transcript.get("cli_version", "mock-1")
    profile = transcript.get("profile", "read-only")
    prompt_path = transcript.get("prompt_path", REFEREE_PROMPT_PATH)
    for field, value in (
        ("backend", backend),
        ("family", family),
        ("model", model),
        ("cli_version", cli_version),
        ("profile", profile),
        ("prompt_path", prompt_path),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"calibration_binding_{field}_invalid")
    prompt = (repo / str(prompt_path)).resolve()
    try:
        prompt.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError("calibration_prompt_outside_repo") from exc
    if not prompt.is_file():
        raise ValueError(f"calibration_prompt_missing:{prompt_path}")
    return {
        "backend": str(backend),
        "family": str(family),
        "model": str(model),
        "cli_version": str(cli_version),
        "profile": str(profile),
        "prompt_path": str(prompt_path),
        "prompt_sha256": _sha256(prompt),
        "rubric_sha256": _rubric_bundle_sha256(repo),
    }


def _gold_digest(
    *,
    gold_dir: Path,
    key_path: Path,
    cases: dict[str, dict[str, object]],
) -> tuple[str, dict[str, str]]:
    artifact_hashes: dict[str, str] = {}
    for case_id in sorted(cases):
        artifact_rel = cases[case_id].get("artifact")
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
    material = {
        "verdict_key_sha256": _sha256(key_path),
        "artifacts": artifact_hashes,
    }
    return _canonical_sha256(material), artifact_hashes


def _git_output(repo: Path, args: list[str]) -> str | None:
    cp = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if cp.returncode != 0:
        return None
    return (cp.stdout or "").strip()


def _calibration_bar_history_failures(
    *,
    repo: Path,
    calibration_path: Path,
    gold_key_path: Path,
    calibration_sha256: str,
) -> list[str]:
    failures: list[str] = []
    git_root = _git_output(repo, ["rev-parse", "--show-toplevel"])
    if git_root is not None:
        calibration_rel = calibration_path.resolve().relative_to(repo.resolve()).as_posix()
        gold_rel = gold_key_path.resolve().relative_to(repo.resolve()).as_posix()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "--", calibration_rel],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if dirty.returncode != 0:
            failures.append("calibration_bar_uncommitted_after_grading")
            return failures
        bar_commit = _git_output(repo, ["log", "-1", "--format=%H", "--", calibration_rel])
        grading_history = _git_output(
            repo,
            ["log", "--reverse", "--diff-filter=A", "--format=%H", "--", gold_rel],
        )
        grading_commit = grading_history.splitlines()[0] if grading_history else None
        if not bar_commit or not grading_commit:
            failures.append("calibration_bar_history_missing")
            return failures
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", bar_commit, grading_commit],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ancestor.returncode != 0:
            failures.append("calibration_bar_committed_after_grading")
        return failures

    journal = repo / "reports" / "status" / "events" / "events.jsonl"
    if not journal.is_file():
        return ["calibration_bar_history_missing"]
    matched = False
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("event") == "calibration_bar_committed"
            and event.get("calibration_sha256") == calibration_sha256
        ):
            matched = True
            break
    if not matched:
        failures.append("calibration_bar_history_missing")
    return failures


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
    repo = _repo_root_from_calibration(path)
    journal = repo / "reports" / "status" / "events" / "events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": "research_swarm.event.v1",
        "event": "calibration_bar_committed",
        "ts_utc": payload["committed_at_utc"],
        "actor_session": committed_by.strip(),
        "human_id": committed_by.strip(),
        "calibration_path": path.resolve().relative_to(repo.resolve()).as_posix(),
        "calibration_sha256": _sha256(path),
    }
    with open(journal, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def calibrate(
    *,
    calibration_path: Path,
    gold_dir: Path,
    mock_path: Path,
    output_path: Path | None,
) -> dict[str, object]:
    bar = _load_bar(calibration_path)
    repo = _repo_root_from_calibration(calibration_path)
    key_path = gold_dir / "verdict_key.json"
    gold = _read_object(key_path)
    mock = _read_object(mock_path)
    if gold.get("schema_version") != GOLD_KEY_SCHEMA_VERSION:
        raise ValueError(f"gold_key_schema_invalid:{gold.get('schema_version')}")
    if mock.get("schema_version") not in {
        MOCK_GOLD_SCHEMA_VERSION,
        GOLD_PREDICTIONS_SCHEMA_VERSION,
    }:
        raise ValueError(f"mock_gold_schema_invalid:{mock.get('schema_version')}")
    if mock.get("schema_version") == MOCK_GOLD_SCHEMA_VERSION and mock.get("backend", "mock") != "mock":
        raise ValueError("mock_calibration_cannot_claim_live_backend")

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
    for case_id in sorted(cases):
        case = cases[case_id]
        prediction = predictions[case_id]
        human = case.get("human_verdict")
        first = prediction.get("position_a")
        second = prediction.get("position_b")
        if human not in VERDICTS or first not in VERDICTS or second not in VERDICTS:
            raise ValueError(f"gold_verdict_invalid:{case_id}")
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
    gold_set_sha256, artifact_hashes = _gold_digest(
        gold_dir=gold_dir,
        key_path=key_path,
        cases=cases,
    )
    predictions_material = [
        {
            "case_id": item["case_id"],
            "position_a": item["position_a"],
            "position_b": item["position_b"],
        }
        for item in scored
    ]
    result: dict[str, object] = {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "backend_binding": _evaluated_binding(repo=repo, transcript=mock),
        "calibration_path": calibration_path.as_posix(),
        "calibration_sha256": _sha256(calibration_path),
        "verdict_key_sha256": _sha256(key_path),
        "gold_set_sha256": gold_set_sha256,
        "artifact_sha256": artifact_hashes,
        "mock_transcript_sha256": _sha256(mock_path),
        "predictions_sha256": _canonical_sha256(predictions_material),
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
        payload: dict[str, object] = result
        if output_path.is_file():
            try:
                existing = _read_object(output_path)
            except (OSError, ValueError, json.JSONDecodeError):
                existing = {}
            raw_evaluations = existing.get("evaluations")
            evaluations = (
                [dict(item) for item in raw_evaluations if isinstance(item, dict)]
                if isinstance(raw_evaluations, list)
                else [existing]
                if isinstance(existing.get("backend_binding"), dict)
                else []
            )
            binding = result["backend_binding"]
            assert isinstance(binding, dict)
            key = (binding.get("backend"), binding.get("family"), binding.get("model"))
            evaluations = [
                item
                for item in evaluations
                if not isinstance(item.get("backend_binding"), dict)
                or (
                    item["backend_binding"].get("backend"),
                    item["backend_binding"].get("family"),
                    item["backend_binding"].get("model"),
                )
                != key
            ]
            evaluations.append(result)
            if len(evaluations) > 1:
                payload = {
                    "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
                    "generated_at_utc": _utc_now_iso(),
                    "calibration_sha256": result["calibration_sha256"],
                    "evaluations": evaluations,
                }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def calibration_report_failures(
    *,
    repo: Path,
    report_path: Path,
    required_family: str | None = None,
    _report: dict[str, object] | None = None,
) -> list[str]:
    """Recompute calibration authority from current hash-bound inputs."""
    repo = repo.resolve()
    failures: list[str] = []
    calibration_path = repo / "contracts" / "rubrics" / "calibration.yaml"
    gold_dir = repo / "tests" / "gold_set"
    key_path = gold_dir / "verdict_key.json"
    try:
        report = _read_object(report_path) if _report is None else _report
        bar = _load_bar(calibration_path)
        gold = _read_object(key_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"calibration_inputs_unreadable:{type(exc).__name__}:{exc}"]
    evaluations = report.get("evaluations")
    if isinstance(evaluations, list):
        candidates = [dict(item) for item in evaluations if isinstance(item, dict)]
        if required_family is not None:
            candidates = [
                item
                for item in candidates
                if isinstance(item.get("backend_binding"), dict)
                and item["backend_binding"].get("family") == required_family
            ]
        if not candidates:
            return [f"backend_binding_not_deployed:{required_family or 'none'}"]
        failures: list[str] = []
        for item in candidates:
            family = item.get("backend_binding", {}).get("family") if isinstance(item.get("backend_binding"), dict) else None
            failures.extend(
                f"{family}:{failure}"
                for failure in calibration_report_failures(
                    repo=repo,
                    report_path=report_path,
                    required_family=str(family) if isinstance(family, str) else required_family,
                    _report=item,
                )
            )
        return failures
    if report.get("schema_version") != CALIBRATION_REPORT_SCHEMA_VERSION:
        failures.append("invalid_schema_version")
    calibration_sha = _sha256(calibration_path)
    if report.get("calibration_sha256") != calibration_sha:
        failures.append("calibration_sha256_mismatch")
    failures.extend(
        _calibration_bar_history_failures(
            repo=repo,
            calibration_path=calibration_path,
            gold_key_path=key_path,
            calibration_sha256=calibration_sha,
        )
    )
    if gold.get("schema_version") != GOLD_KEY_SCHEMA_VERSION:
        failures.append("gold_key_schema_invalid")
        return failures
    raw_gold_cases = gold.get("cases")
    if not isinstance(raw_gold_cases, list):
        failures.append("gold_cases_invalid")
        return failures
    gold_cases = {
        item["case_id"]: item
        for item in raw_gold_cases
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(gold_cases) != len(raw_gold_cases):
        failures.append("gold_cases_invalid")
        return failures
    try:
        gold_set_sha, artifact_hashes = _gold_digest(
            gold_dir=gold_dir,
            key_path=key_path,
            cases=gold_cases,
        )
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
        return failures
    if report.get("verdict_key_sha256") != _sha256(key_path):
        failures.append("verdict_key_sha256_mismatch")
    if report.get("gold_set_sha256") != gold_set_sha:
        failures.append("gold_set_sha256_mismatch")
    if report.get("artifact_sha256") != artifact_hashes:
        failures.append("gold_artifact_sha256_mismatch")

    raw_predictions = report.get("cases")
    if not isinstance(raw_predictions, list):
        failures.append("calibration_predictions_invalid")
        return failures
    predictions = {
        item["case_id"]: item
        for item in raw_predictions
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(predictions) != len(raw_predictions) or set(predictions) != set(gold_cases):
        failures.append("calibration_prediction_case_mismatch")
        return failures
    prediction_material: list[dict[str, object]] = []
    agreements = 0
    flips = 0
    wrong_artifacts = 0
    wrong_proofs = 0
    for case_id in sorted(gold_cases):
        gold_case = gold_cases[case_id]
        prediction = predictions[case_id]
        human = gold_case.get("human_verdict")
        first = prediction.get("position_a")
        second = prediction.get("position_b")
        if human not in VERDICTS or first not in VERDICTS or second not in VERDICTS:
            failures.append(f"calibration_verdict_invalid:{case_id}")
            continue
        agreements += int(first == human)
        flips += int(first != second)
        wrong_artifacts += int(
            human == "not_supported" and gold_case.get("kind") == "artifact"
        )
        wrong_proofs += int(
            human == "not_supported" and gold_case.get("kind") == "proof"
        )
        prediction_material.append(
            {"case_id": case_id, "position_a": first, "position_b": second}
        )
    count = len(gold_cases)
    agreement = agreements / count if count else 0.0
    flip_rate = flips / count if count else 1.0
    if report.get("predictions_sha256") != _canonical_sha256(prediction_material):
        failures.append("predictions_sha256_mismatch")
    asserted_agreement = report.get("agreement")
    if not isinstance(asserted_agreement, (int, float)) or not math.isclose(
        float(asserted_agreement), agreement, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(f"agreement_recompute_mismatch:{asserted_agreement}!={agreement}")
    asserted_flip = report.get("position_flip_rate")
    if not isinstance(asserted_flip, (int, float)) or not math.isclose(
        float(asserted_flip), flip_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(f"position_flip_recompute_mismatch:{asserted_flip}!={flip_rate}")
    if report.get("wrong_artifact_count") != wrong_artifacts or wrong_artifacts < 3:
        failures.append(f"wrong_artifact_floor:{wrong_artifacts}")
    if report.get("wrong_proof_count") != wrong_proofs or wrong_proofs < 3:
        failures.append(f"wrong_proof_floor:{wrong_proofs}")
    calibrated = (
        agreement >= float(bar["agreement_floor"])
        and flip_rate <= float(bar["position_flip_ceiling"])
    )
    if not calibrated or report.get("calibrated") is not calibrated:
        failures.append("calibrated_false")

    binding = report.get("backend_binding")
    if not isinstance(binding, dict):
        failures.append("backend_binding_missing")
        return failures
    family = binding.get("family")
    if required_family is not None and family != required_family:
        failures.append(f"backend_family_mismatch:{family}!={required_family}")
    member = next(
        (
            item
            for item in _panel_members(repo)
            if item.get("family", item.get("backend")) == family
        ),
        None,
    )
    if member is None:
        failures.append(f"backend_binding_not_deployed:{family}")
        return failures
    expected_fields = {
        "backend": member.get("backend"),
        "family": member.get("family", member.get("backend")),
        "model": member.get("model"),
        "cli_version": member.get("cli_version"),
        "profile": member.get("profile"),
        "prompt_path": member.get("prompt_path", REFEREE_PROMPT_PATH),
    }
    for field, expected in expected_fields.items():
        if binding.get(field) != expected:
            failures.append(f"backend_{field}_mismatch:{binding.get(field)}!={expected}")
    prompt_path = repo / str(expected_fields["prompt_path"])
    if not prompt_path.is_file() or binding.get("prompt_sha256") != _sha256(prompt_path):
        failures.append("backend_prompt_sha256_mismatch")
    try:
        expected_rubric_sha = _rubric_bundle_sha256(repo)
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
    else:
        if binding.get("rubric_sha256") != expected_rubric_sha:
            failures.append("backend_rubric_sha256_mismatch")
    if expected_fields["profile"] != "read-only":
        failures.append("backend_profile_not_read_only")
    return failures


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
