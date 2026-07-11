#!/usr/bin/env python3
"""Execute the §6.5 adversarial integrity audit in a disposable worktree.

The kernel, not the auditing model, selects claims and rebuild commands.  The
auditor supplies independently recomputed semantic-role values and sampled
scientific verdicts.  Deterministic code then compares those results to the
claim ledger, processed manifests, and release inventory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swarm_events  # noqa: E402


SCHEMA_VERSION = "research_swarm.integrity_audit.v1"
MOCK_SCHEMA_VERSION = "research_swarm.mock_integrity_audit.v1"
SAMPLE_SIZE = 3
FORBIDDEN_COMMAND_TOKENS = {
    "git",
    "curl",
    "wget",
    "ssh",
    "scp",
    "nc",
    "netcat",
    "ncat",
    "socat",
    "telnet",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Environment variables whose names mark a credential/secret.  These are removed
# from every offline audit subprocess so the reported `environment_scrub_only`
# label is TRUE, not aspirational — a rebuild/recompute subprocess never inherits
# repository, cloud, or API credentials.
_CREDENTIAL_ENV_MARKERS = (
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API_KEY", "APIKEY",
    "ACCESS_KEY", "PRIVATE_KEY", "SESSION_TOKEN", "AUTH",
)
_CREDENTIAL_ENV_PREFIXES = (
    "AWS_", "ANTHROPIC", "OPENAI", "CLAUDE", "GITHUB", "GH_", "SSH_", "GCP_",
    "GOOGLE_", "AZURE_", "NPM_", "PYPI_", "DOCKER_", "SLACK_", "STRIPE_", "HF_",
    "HUGGINGFACE", "VAULT_", "CLOUDFLARE", "DIGITALOCEAN", "HEROKU", "NETLIFY",
    "VERCEL", "SENTRY", "DATADOG", "TWILIO",
)
# Vendor credentials the LIVE audit backend legitimately needs to reach its API.
_VENDOR_CREDENTIAL_PREFIXES = ("ANTHROPIC", "CLAUDE")


def _is_credential_env(name: str) -> bool:
    upper = name.upper()
    if any(upper.startswith(prefix) for prefix in _CREDENTIAL_ENV_PREFIXES):
        return True
    return any(marker in upper for marker in _CREDENTIAL_ENV_MARKERS)


def _is_vendor_credential_env(name: str) -> bool:
    return name.upper().startswith(_VENDOR_CREDENTIAL_PREFIXES)


def _offline_subprocess_env() -> dict[str, str]:
    """Environment for OFFLINE rebuild/recompute subprocesses: credentials
    stripped (so `environment_scrub_only` is true) and every proxy pointed at a
    dead local port (so `proxy_environment_only` is true)."""
    env = {key: value for key, value in os.environ.items() if not _is_credential_env(key)}
    dead_proxy = "http://127.0.0.1:9"
    env.update(
        {
            "http_proxy": dead_proxy,
            "https_proxy": dead_proxy,
            "HTTP_PROXY": dead_proxy,
            "HTTPS_PROXY": dead_proxy,
            "ALL_PROXY": dead_proxy,
            "all_proxy": dead_proxy,
            "NO_PROXY": "",
            "no_proxy": "",
        }
    )
    return env


def _live_backend_env() -> dict[str, str]:
    """Environment for the LIVE audit backend.  It must reach the vendor API, so
    it retains vendor transport and the vendor credential — but every OTHER
    credential (repo, cloud, unrelated API) is still scrubbed.  The confinement
    report labels this honestly (`vendor_credential_retained`), never claiming a
    full scrub it does not apply."""
    return {
        key: value
        for key, value in os.environ.items()
        if _is_vendor_credential_env(key) or not _is_credential_env(key)
    }


def effective_confinement(backend: str = "mock") -> dict[str, object]:
    """Report the confinement ACTUALLY enforced, per backend — never a control
    that isn't applied to the launched subprocesses.

    - ``mock`` (default, CI): the auditor transcript is read from a local file,
      so no auditor egress; the only subprocesses are offline rebuilds/recomputes
      launched with `_offline_subprocess_env()` (credentials stripped, proxies
      dead).  Hence `environment_scrub_only` + `proxy_environment_only`.
    - live: the auditor subprocess reaches the vendor API and therefore retains
      outbound vendor transport and the vendor credential.  Reported as
      `vendor_api_transport_only` + `vendor_credential_retained` — the honest
      weakest link, not a scrub/proxy claim the live call does not honour.
    """
    system = platform.system().lower() or "unknown"
    if system != "linux":
        reason = f"{system}_network_namespace_unavailable"
    elif shutil.which("bwrap") is None:
        reason = "bubblewrap_not_available"
    else:
        # Having a binary is not enforcement.  Until every rebuild/model/audit
        # subprocess is launched through the reviewed profile, remain honest.
        reason = "bubblewrap_profile_not_enabled"
    if backend == "mock":
        effective_network = "proxy_environment_only"
        credential_isolation = "environment_scrub_only"
    else:
        effective_network = "vendor_api_transport_only"
        credential_isolation = "vendor_credential_retained"
    return {
        "capability_probe": reason,
        "os_enforced": False,
        "effective_network": effective_network,
        "credential_isolation": credential_isolation,
        "filesystem_isolation": "scratch_worktree_plus_mutation_detection",
    }


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"top_level_not_object:{path}")
    return payload


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=check
    )


def scratch_worktree_argv(repo: Path, destination: Path, revision: str = "HEAD") -> list[str]:
    """Pure constructor used by tests to prove detached scratch execution."""
    return ["git", "worktree", "add", "--detach", destination.as_posix(), revision]


def _audit_executor_contract(repo: Path) -> dict[str, str]:
    framework = _read_json(repo / "contracts/framework.json")
    executors = framework.get("executors")
    config = executors.get("integrity_audit") if isinstance(executors, dict) else None
    if not isinstance(config, dict):
        raise ValueError("integrity_audit_backend_unavailable")
    backend = config.get("backend")
    family = config.get("family")
    model = config.get("model")
    command = config.get("command", backend)
    if not all(isinstance(value, str) and value.strip() for value in (backend, family, model, command)):
        raise ValueError("integrity_audit_backend_unavailable")
    return {
        "backend": backend.strip().casefold(),
        "family": family.strip().casefold(),
        "model": model.strip(),
        "command": command.strip(),
    }


def claude_audit_argv(repo: Path) -> list[str]:
    config = _audit_executor_contract(repo)
    if config["backend"] != "claude":
        raise ValueError("integrity_audit_backend_unavailable")
    return [
        config["command"],
        "-p",
        "--model",
        config["model"],
        "--tools",
        "Read,Glob,Grep",
        "--strict-mcp-config",
        "--mcp-config",
        "{}",
        "--output-format",
        "text",
    ]


def _make_scratch(repo: Path, destination: Path, *, hermetic_copy: bool) -> str:
    if hermetic_copy or not (repo / ".git").exists():
        shutil.copytree(
            repo,
            destination,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            dirs_exist_ok=False,
        )
        return "hermetic_temp_copy"
    cp = subprocess.run(
        scratch_worktree_argv(repo, destination), cwd=repo, text=True, capture_output=True
    )
    if cp.returncode != 0:
        raise ValueError(f"scratch_worktree_create_failed:{cp.stderr[-1000:]}")
    return "git_detached_worktree"


def _remove_scratch(repo: Path, destination: Path, scratch_kind: str) -> None:
    if scratch_kind == "git_detached_worktree":
        subprocess.run(
            ["git", "worktree", "remove", "--force", destination.as_posix()],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
    else:
        shutil.rmtree(destination, ignore_errors=True)


def _processed_rebuilds(scratch: Path) -> list[dict[str, Any]]:
    latest_by_output: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    for path in sorted((scratch / "data/processed_manifest").rglob("*.json")):
        payload = _read_json(path)
        as_of = str(payload.get("as_of_utc_date", ""))
        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                continue
            key = output["path"]
            if key not in latest_by_output or as_of > latest_by_output[key][0]:
                latest_by_output[key] = (as_of, path, payload)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for output_path, (_as_of, manifest_path, payload) in latest_by_output.items():
        transform = payload.get("transform")
        command = transform.get("command") if isinstance(transform, dict) else None
        if not isinstance(command, str) or not command.strip():
            continue
        key = (manifest_path.as_posix(), command)
        item = grouped.setdefault(
            key,
            {
                "manifest": manifest_path.relative_to(scratch).as_posix(),
                "command": command,
                "outputs": [],
            },
        )
        expected = next(
            (
                candidate
                for candidate in payload.get("outputs", [])
                if isinstance(candidate, dict) and candidate.get("path") == output_path
            ),
            None,
        )
        if isinstance(expected, dict):
            item["outputs"].append(dict(expected))
    return sorted(grouped.values(), key=lambda item: (item["manifest"], item["command"]))


def _command_violation(command: str, *, repo: Path | None = None) -> str | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        return "unparseable"
    if not argv or argv[0] not in {"python", "python3", "make"}:
        return "interpreter_not_allowlisted"
    lowered = {Path(token).name.casefold() for token in argv}
    forbidden = sorted(lowered & FORBIDDEN_COMMAND_TOKENS)
    if forbidden:
        return "forbidden_token:" + ",".join(forbidden)
    if any(token in {"-c", "-m", "-"} for token in argv[1:]):
        return "inline_code_forbidden"
    if repo is not None and argv[0] in {"python", "python3"}:
        script_token = next((token for token in argv[1:] if not token.startswith("-")), None)
        if script_token is not None:
            script = (repo / script_token).resolve()
            try:
                script.relative_to(repo.resolve())
            except ValueError:
                return "script_outside_scratch"
            if not script.is_file() or script.suffix != ".py":
                return "python_script_invalid"
            source = script.read_text(encoding="utf-8", errors="replace")
            for token in sorted(FORBIDDEN_COMMAND_TOKENS):
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", source, re.IGNORECASE):
                    return f"referenced_script_forbidden_token:{token}"
    return None


def _inventory_artifacts(inventory: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            digest = value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                found[path] = digest.lower()
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(inventory.get("artifacts", inventory))
    return found


def _verify_inventory_surfaces(
    scratch: Path, inventory: dict[str, Any], mode: str
) -> tuple[list[dict[str, Any]], list[str]]:
    prefixes = ["data/processed/"] if mode in {"empirical", "hybrid"} else []
    if mode in {"modeling", "hybrid"}:
        prefixes.extend(["reports/models/", "reports/proofs/", "contracts/instances/"])
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for relpath, expected in sorted(_inventory_artifacts(inventory).items()):
        if not any(relpath.startswith(prefix) for prefix in prefixes):
            continue
        path = scratch / relpath
        actual = _sha256(path) if path.is_file() else None
        passed = actual == expected
        checks.append(
            {
                "path": relpath,
                "release_inventory_sha256": expected,
                "scratch_sha256": actual,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"release_inventory_hash_mismatch:{relpath}")
    if not checks:
        failures.append(f"release_inventory_mode_surfaces_missing:{mode}")
    return checks, failures


def _execute_rebuilds(scratch: Path, inventory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    inventory_hashes = _inventory_artifacts(inventory)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for rebuild in _processed_rebuilds(scratch):
        command = rebuild["command"]
        violation = _command_violation(command, repo=scratch)
        if violation is not None:
            failures.append(f"rebuild_command_violation:{rebuild['manifest']}:{violation}")
            continue
        for expected in rebuild["outputs"]:
            candidate = scratch / expected["path"]
            if candidate.is_file():
                candidate.unlink()
        argv = shlex.split(command)
        cp = subprocess.run(
            argv,
            cwd=scratch,
            text=True,
            capture_output=True,
            env=_offline_subprocess_env(),
            timeout=900,
        )
        record: dict[str, Any] = {
            "manifest": rebuild["manifest"],
            "command": command,
            "returncode": cp.returncode,
            "outputs": [],
        }
        if cp.returncode != 0:
            failures.append(f"rebuild_command_failed:{rebuild['manifest']}:{cp.returncode}")
        for expected in rebuild["outputs"]:
            relpath = expected["path"]
            path = scratch / relpath
            actual = _sha256(path) if path.is_file() else None
            expected_hash = expected.get("sha256")
            inventory_hash = inventory_hashes.get(relpath)
            output_record = {
                "path": relpath,
                "expected_manifest_sha256": expected_hash,
                "release_inventory_sha256": inventory_hash,
                "recomputed_sha256": actual,
                "matches_manifest": actual == expected_hash,
                "matches_release_inventory": inventory_hash is not None and actual == inventory_hash,
            }
            record["outputs"].append(output_record)
            if actual != expected_hash:
                failures.append(f"recompute_mismatch:{relpath}")
            if inventory_hash is None:
                failures.append(f"release_inventory_missing_surface:{relpath}")
            elif actual != inventory_hash:
                failures.append(f"release_inventory_hash_mismatch:{relpath}")
        results.append(record)
    if not results:
        failures.append("no_rebuildable_processed_manifests")
    return results, failures


def _json_number(path: Path, dotted_key: str) -> float | None:
    try:
        value: object = _read_json(path)
        for part in dotted_key.split("."):
            value = value[part] if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def _execute_declared_rerun(
    *,
    scratch: Path,
    manifest_path: Path,
    command: object,
    outputs: object,
    inventory: dict[str, Any],
    require_seed: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    rel_manifest = manifest_path.relative_to(scratch).as_posix()
    record: dict[str, Any] = {"manifest": rel_manifest, "command": command, "status": "block", "outputs": []}
    failures: list[str] = []
    if not isinstance(command, str) or (violation := _command_violation(command, repo=scratch)) is not None:
        failures.append(f"model_rerun_command_invalid:{rel_manifest}:{violation if isinstance(command, str) else 'missing'}")
        return record, failures
    try:
        argv = shlex.split(command)
    except ValueError:
        failures.append(f"model_rerun_command_invalid:{rel_manifest}:unparseable")
        return record, failures
    if require_seed is not None and str(require_seed) not in argv:
        failures.append(f"model_rerun_seed_not_locked:{rel_manifest}:{require_seed}")
    if not isinstance(outputs, list) or not outputs:
        failures.append(f"model_rerun_outputs_missing:{rel_manifest}")
        return record, failures
    for output in outputs:
        if isinstance(output, dict) and isinstance(output.get("path"), str):
            path = scratch / output["path"]
            if path.is_file():
                path.unlink()
    cp = subprocess.run(
        argv,
        cwd=scratch,
        text=True,
        capture_output=True,
        env=_offline_subprocess_env(),
        timeout=900,
    )
    record["returncode"] = cp.returncode
    if cp.returncode != 0:
        failures.append(f"model_rerun_command_failed:{rel_manifest}:{cp.returncode}")
    inventory_hashes = _inventory_artifacts(inventory)
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            failures.append(f"model_rerun_output_invalid:{rel_manifest}")
            continue
        relpath = output["path"]
        path = scratch / relpath
        comparison = output.get("comparison", "sha256")
        passed = False
        item: dict[str, Any] = {"path": relpath, "comparison": comparison}
        if comparison == "sha256":
            actual = _sha256(path) if path.is_file() else None
            expected = output.get("sha256")
            inventory_hash = inventory_hashes.get(relpath)
            passed = actual == expected and inventory_hash == actual
            item.update(
                {
                    "expected_sha256": expected,
                    "release_inventory_sha256": inventory_hash,
                    "recomputed_sha256": actual,
                }
            )
        elif comparison == "numeric_json":
            key = output.get("json_key")
            expected = output.get("expected")
            tolerance = output.get("tolerance")
            actual = _json_number(path, key) if isinstance(key, str) else None
            passed = (
                actual is not None
                and isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and isinstance(tolerance, (int, float))
                and not isinstance(tolerance, bool)
                and tolerance >= 0
                and abs(actual - float(expected)) <= float(tolerance)
            )
            item.update({"json_key": key, "expected": expected, "tolerance": tolerance, "recomputed": actual})
        else:
            failures.append(f"model_rerun_comparison_invalid:{rel_manifest}:{comparison}")
        item["passed"] = passed
        record["outputs"].append(item)
        if not passed:
            failures.append(f"model_rerun_mismatch:{relpath}")
    record["status"] = "pass" if cp.returncode == 0 and not failures else "block"
    return record, failures


def _execute_modeling_reruns(
    scratch: Path, inventory: dict[str, Any], seed: str, *, hybrid: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    experiments: list[tuple[str, Path, dict[str, Any]]] = []
    for path in sorted((scratch / "reports/models").rglob("*.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") != "research_swarm.experiment_manifest.v1":
            continue
        score = hashlib.sha256(f"{seed}\0experiment\0{path.relative_to(scratch)}".encode()).hexdigest()
        experiments.append((score, path, payload))
    experiment_results: list[dict[str, Any]] = []
    if not experiments:
        failures.append("experiment_manifests_missing")
    for _score, path, payload in sorted(experiments)[:SAMPLE_SIZE]:
        result, item_failures = _execute_declared_rerun(
            scratch=scratch,
            manifest_path=path,
            command=payload.get("reproduction_command"),
            outputs=payload.get("audit_outputs"),
            inventory=inventory,
            require_seed=payload.get("seed") if isinstance(payload.get("seed"), int) else None,
        )
        experiment_results.append(result)
        failures.extend(item_failures)

    seams: list[dict[str, Any]] = []
    if hybrid:
        instances: list[tuple[str, Path, dict[str, Any]]] = []
        for path in sorted((scratch / "contracts/instances").glob("*.json")):
            try:
                payload = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if payload.get("schema_version") != "research_swarm.instance_manifest.v1" or "source_processed_manifests" not in payload:
                continue
            score = hashlib.sha256(f"{seed}\0instance\0{path.relative_to(scratch)}".encode()).hexdigest()
            instances.append((score, path, payload))
        if not instances:
            failures.append("hybrid_bridge_instance_manifests_missing")
        for _score, path, payload in sorted(instances)[:SAMPLE_SIZE]:
            raw_outputs = payload.get("outputs")
            outputs = [
                {**item, "comparison": item.get("comparison", "sha256")}
                for item in raw_outputs
                if isinstance(item, dict)
            ] if isinstance(raw_outputs, list) else raw_outputs
            result, item_failures = _execute_declared_rerun(
                scratch=scratch,
                manifest_path=path,
                command=payload.get("generator_command"),
                outputs=outputs,
                inventory=inventory,
            )
            result["kind"] = "instance_regeneration"
            seams.append(result)
            failures.extend(item_failures)
        if experiment_results:
            seams.append({"kind": "experiment_cell", "status": experiment_results[0]["status"], "manifest": experiment_results[0]["manifest"]})
    return experiment_results, seams, failures


def _load_claims(scratch: Path) -> list[dict[str, Any]]:
    payload = _read_json(scratch / "contracts/claims.yaml")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claim_ledger_invalid")
    return [dict(item) for item in claims if isinstance(item, dict)]


def _sample_claims(claims: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    headline = [claim for claim in claims if claim.get("headline") is True]
    non_headline = [claim for claim in claims if claim.get("headline") is not True]
    ranked = sorted(
        non_headline,
        key=lambda claim: hashlib.sha256(
            f"{seed}\0{claim.get('claim_id', '')}".encode("utf-8")
        ).hexdigest(),
    )
    return headline + ranked[:SAMPLE_SIZE]


def _redacted_claim_context(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose semantic roles, never the registered answer, to the auditor."""
    context: list[dict[str, Any]] = []
    for claim in claims:
        statement = str(claim.get("statement", ""))
        redacted_statement = re.sub(
            r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*%|\s+[A-Za-z][A-Za-z0-9._/-]*)?",
            "<RECOMPUTE>",
            statement,
        )
        roles = claim.get("recomputation_roles")
        context.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_type": claim.get("type"),
                "headline": claim.get("headline") is True,
                "redacted_statement": redacted_statement,
                "semantic_roles": sorted(roles) if isinstance(roles, dict) else [],
            }
        )
    return context


def _extract_json_block(stdout: str) -> dict[str, Any]:
    blocks = re.findall(r"```json\s*(.*?)```", stdout, flags=re.DOTALL)
    raw = blocks[-1] if blocks else stdout
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("audit_transcript_not_object")
    return payload


def _load_transcript(
    *, scratch: Path, backend: str, mock_path: Path | None, context: dict[str, Any], timeout: int
) -> dict[str, Any]:
    if backend == "mock":
        if mock_path is None:
            raise ValueError("mock_transcript_required")
        payload = _read_json(mock_path)
        if payload.get("schema_version") != MOCK_SCHEMA_VERSION:
            raise ValueError("invalid_mock_integrity_audit_schema")
        return payload
    argv = claude_audit_argv(scratch)
    prompt_path = scratch / "contracts/prompts/integrity_auditor.md"
    prompt = prompt_path.read_text(encoding="utf-8") + "\n\nAudit context:\n```json\n" + json.dumps(context, indent=2, sort_keys=True) + "\n```\n"
    cp = subprocess.run(
        argv,
        cwd=scratch,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        # Vendor transport + vendor credential retained (the live auditor needs
        # its API); every other credential is scrubbed — matches the honest
        # `vendor_credential_retained` confinement label for the live backend.
        env=_live_backend_env(),
    )
    if cp.returncode != 0:
        raise ValueError(f"audit_backend_failed:{cp.returncode}:{cp.stderr[-1000:]}")
    return _extract_json_block(cp.stdout)


def _semantic_recomputation(
    sampled: list[dict[str, Any]],
    transcript: dict[str, Any],
    mode: str,
    source_artifacts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_results = transcript.get("claim_recomputations")
    indexed = {
        item.get("claim_id"): item
        for item in raw_results
        if isinstance(item, dict) and isinstance(item.get("claim_id"), str)
    } if isinstance(raw_results, list) else {}
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for claim in sampled:
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str):
            failures.append("sampled_claim_missing_id")
            continue
        result = indexed.get(claim_id)
        expected_roles = claim.get("recomputation_roles")
        if result is None:
            failures.append(f"claim_recomputation_missing:{claim_id}")
            continue
        actual_roles = result.get("role_values")
        literal_expected = claim.get("manuscript_numeric_literals", [])
        literal_actual = result.get("numeric_literals", [])
        role_match = isinstance(expected_roles, dict) and actual_roles == expected_roles
        literal_match = isinstance(literal_expected, list) and literal_actual == literal_expected
        passed = result.get("status") == "pass" and role_match and literal_match
        record = {
            "claim_id": claim_id,
            "headline": claim.get("headline") is True,
            "claim_type": claim.get("type"),
            "expected_role_values": expected_roles,
            "recomputed_role_values": actual_roles,
            "expected_numeric_literals": literal_expected,
            "recomputed_numeric_literals": literal_actual,
            "source_artifacts": source_artifacts,
            "passed": passed,
        }
        results.append(record)
        if not isinstance(expected_roles, dict) or not expected_roles:
            failures.append(f"claim_recomputation_roles_unregistered:{claim_id}")
        elif not role_match:
            failures.append(f"claim_semantic_role_mismatch:{claim_id}")
        if not literal_match:
            failures.append(f"claim_numeric_recompute_mismatch:{claim_id}")
        if result.get("status") != "pass":
            failures.append(f"claim_recomputation_failed:{claim_id}")

    if mode in {"modeling", "hybrid"}:
        values = transcript.get("theoretical_rederivations")
        if not isinstance(values, list) or not values:
            failures.append("theoretical_rederivations_missing")
        elif any(not isinstance(item, dict) or item.get("status") != "pass" for item in values):
            failures.append("theoretical_rederivation_failed")
    decisions = transcript.get("etl_decision_samples")
    if not isinstance(decisions, list) or not decisions:
        failures.append("etl_decision_samples_missing")
    elif any(
        not isinstance(item, dict)
        or item.get("status") != "pass"
        or not isinstance(item.get("protocol_clause_id"), str)
        or not item["protocol_clause_id"].strip()
        for item in decisions
    ):
        failures.append("unlogged_etl_decision")
    return results, failures


def _builder_families(
    repo: Path, inventory: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    families: set[str] = set()
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    inventory_artifacts = _inventory_artifacts(inventory)
    for path in sorted((repo / "reports/status/swarm_runs").glob("*.json")):
        relpath = path.relative_to(repo).as_posix()
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        ownership = payload.get("ownership")
        changed_paths = ownership.get("changed_paths") if isinstance(ownership, dict) else None
        produced: list[dict[str, str]] = []
        for changed in changed_paths if isinstance(changed_paths, list) else []:
            if not isinstance(changed, str) or changed.startswith("reports/status/"):
                continue
            expected = inventory_artifacts.get(changed)
            artifact_path = repo / changed
            if expected is None or not artifact_path.is_file():
                continue
            actual = _sha256(artifact_path)
            if actual == expected:
                produced.append({"path": changed, "sha256": actual})
        if not produced:
            continue
        executor = payload.get("executor")
        tool = executor.get("tool") if isinstance(executor, dict) else None
        normalized = tool.strip().casefold() if isinstance(tool, str) and tool.strip() else None
        if normalized in {None, "manual", "operator_backfill", "legacy_backfill"}:
            failures.append(f"builder_family_unverified:{relpath}")
            family = None
        else:
            family = normalized
            families.add(family)
        evidence.append(
            {
                "run_manifest": relpath,
                "family": family,
                "family_source": "executor.tool",
                "artifacts": produced,
            }
        )
    if not evidence:
        failures.append("builder_family_unverified:no_hash_bound_builder_runs")
    return sorted(families), evidence, failures


def _repo_snapshot(repo: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    excluded = (repo / "reports/status/integrity_audit").resolve()
    for path in sorted(repo.rglob("*")):
        if ".git" in path.relative_to(repo).parts:
            continue
        try:
            path.resolve().relative_to(excluded)
            continue
        except ValueError:
            pass
        relpath = path.relative_to(repo).as_posix()
        if path.is_symlink():
            snapshot[relpath] = "symlink:" + os.readlink(path)
        elif path.is_file():
            snapshot[relpath] = _sha256(path)
    return snapshot


def _snapshot_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output_arg = args.output or Path(
        "reports/status/integrity_audit/"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    output = output_arg if output_arg.is_absolute() else repo / output_arg
    output = output.resolve()
    allowed = (repo / "reports/status/integrity_audit").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("audit_output_outside_integrity_surface") from exc
    inventory_path = args.release_inventory if args.release_inventory.is_absolute() else repo / args.release_inventory
    inventory = _read_json(inventory_path)
    executor_contract = _audit_executor_contract(repo)
    audit_family = executor_contract["family"]
    requested_family = args.audit_family.strip().casefold()
    builders, builder_evidence, builder_failures = _builder_families(repo, inventory)
    failures: list[str] = []
    failures.extend(builder_failures)
    if requested_family != audit_family:
        failures.append("integrity_audit_family_override_refused")
    if getattr(args, "builder_family", []):
        failures.append("builder_family_override_refused")
    if not builders:
        failures.append("builder_family_unavailable")
    if audit_family in builders:
        failures.append("integrity_audit_family_of_builder")

    repo_before = _repo_snapshot(repo)

    scratch_parent = Path(tempfile.mkdtemp(prefix="research-swarm-integrity-"))
    scratch = scratch_parent / "worktree"
    scratch_kind = "unknown"
    try:
        scratch_kind = _make_scratch(repo, scratch, hermetic_copy=bool(args.hermetic_copy))
        scratch_inventory_path = scratch / inventory_path.relative_to(repo)
        scratch_inventory = _read_json(scratch_inventory_path)
        inventory_checks, inventory_failures = _verify_inventory_surfaces(
            scratch, scratch_inventory, args.mode
        )
        failures.extend(inventory_failures)
        if args.mode in {"empirical", "hybrid"}:
            rebuilds, rebuild_failures = _execute_rebuilds(scratch, scratch_inventory)
            failures.extend(rebuild_failures)
        else:
            rebuilds = []
        claims = _load_claims(scratch)
        seed = (scratch / args.seed_path).read_text(encoding="utf-8").strip()
        if not seed:
            failures.append("integrity_sampling_seed_empty")
        sampled = _sample_claims(claims, seed)
        if args.mode in {"modeling", "hybrid"}:
            experiment_results, seams, model_failures = _execute_modeling_reruns(
                scratch, scratch_inventory, seed, hybrid=args.mode == "hybrid"
            )
            failures.extend(model_failures)
        else:
            experiment_results, seams = [], []
        source_artifacts = [
            {
                "path": output.get("path"),
                "asserted_sha256": output.get("recomputed_sha256"),
                "release_inventory_sha256": output.get("release_inventory_sha256"),
            }
            for rebuild in rebuilds
            for output in rebuild.get("outputs", [])
            if isinstance(rebuild, dict) and isinstance(output, dict)
        ]
        if not source_artifacts:
            source_artifacts = [
                {
                    "path": item.get("path"),
                    "asserted_sha256": item.get("scratch_sha256"),
                    "release_inventory_sha256": item.get("release_inventory_sha256"),
                }
                for item in inventory_checks
                if isinstance(item, dict)
            ]
        # The adversarial family cannot echo the ledger, manuscript, or any
        # pre-rendered answer surface. Retain registered answers only in kernel
        # memory for the post-response comparison.
        (scratch / "contracts/claims.yaml").unlink(missing_ok=True)
        for answer_surface in (
            "reports/paper",
            "reports/tables",
            "reports/figures",
            "reports/validation",
        ):
            shutil.rmtree(scratch / answer_surface, ignore_errors=True)
        context = {
            "mode": args.mode,
            "audit_family": audit_family,
            "builder_families": builders,
            "sampled_claims": _redacted_claim_context(sampled),
            "rebuild_results": rebuilds,
            "requirements": {
                "empirical": "recompute panel/regime claims and sample ETL decisions",
                "modeling": "rerun sampled experiments and rederive sampled theoretical claims",
                "hybrid": "union plus instance-regeneration and experiment-cell seam audits",
            }[args.mode],
        }
        transcript = _load_transcript(
            scratch=scratch,
            backend=args.backend,
            mock_path=args.mock_transcript.resolve() if args.mock_transcript else None,
            context=context,
            timeout=args.timeout_seconds,
        )
        returned_family = transcript.get("audit_family")
        if not isinstance(returned_family, str) or returned_family.strip().casefold() != audit_family:
            failures.append("integrity_audit_family_transcript_mismatch")
        claim_results, semantic_failures = _semantic_recomputation(
            sampled, transcript, args.mode, source_artifacts
        )
        failures.extend(semantic_failures)
        etl_samples = transcript.get("etl_decision_samples", [])
        derivations = transcript.get("theoretical_rederivations", [])
    finally:
        _remove_scratch(repo, scratch, scratch_kind)
        shutil.rmtree(scratch_parent, ignore_errors=True)

    repo_changes = _snapshot_changes(repo_before, _repo_snapshot(repo))
    if repo_changes:
        failures.append("main_repo_mutated_during_audit:" + ",".join(repo_changes))

    try:
        audited_git_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    except subprocess.CalledProcessError:
        audited_git_sha = "unavailable"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "status": "pass" if not failures else "block",
        "mode": args.mode,
        "audit_task_id": getattr(args, "task_id", None),
        "audited_git_sha": audited_git_sha,
        "release_inventory": {
            "path": inventory_path.relative_to(repo).as_posix(),
            "sha256": _sha256(inventory_path),
        },
        "executor": {
            "backend": args.backend,
            "model": executor_contract["model"] if args.backend == executor_contract["backend"] else None,
            "audit_family": audit_family,
            "builder_families": builders,
            "builder_run_manifest_evidence": builder_evidence,
            "profile": "scratch-worktree",
            "network": "requested_off",
            "commit_push_allowed": False,
            "scratch_kind": scratch_kind,
            "effective_confinement": effective_confinement(args.backend),
        },
        "repo_confinement": {
            "excluded_prefixes": [".git/", "reports/status/integrity_audit/"],
            "changed_paths": repo_changes,
            "passed": not repo_changes,
            "enforcement": "detect-and-reject",
        },
        "sampling": {"seed_path": args.seed_path.as_posix(), "non_headline_sample_size": SAMPLE_SIZE},
        "surface_rebuilds": rebuilds,
        "inventory_hash_checks": inventory_checks,
        "claim_recomputations": claim_results,
        "etl_decision_samples": etl_samples,
        "experiment_recomputations": experiment_results,
        "theoretical_rederivations": derivations,
        "seam_audits": seams,
        "authorized_post_audit_repairs": [],
        "failures": sorted(set(failures)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Emit the kernel-authored journal event that binds this report — the gate
    # accepts ONLY a report bound to such an event (the events journal is
    # kernel-only), so a hand-authored report cannot authorize a release.
    try:
        report_rel = output.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        report_rel = output.as_posix()
    report_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    swarm_events.append_event(
        repo,
        {
            "event": "integrity_audit_recorded",
            "report_path": report_rel,
            "report_sha256": report_sha,
            "status": report.get("status"),
            "mode": report.get("mode"),
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="integrity_audit.py",
        description="Run a different-family scratch-worktree integrity audit",
    )
    parser.add_argument("--release-inventory", required=True, type=Path)
    parser.add_argument("--mode", choices=["empirical", "modeling", "hybrid"], required=True)
    parser.add_argument(
        "--audit-family",
        required=True,
        help="Expected audit family; must match the executor contract and cannot override it",
    )
    parser.add_argument("--task-id", help="Numbered integrity_audit task id used to recognize evidence-persistence commits")
    parser.add_argument(
        "--builder-family",
        action="append",
        default=[],
        help="Deprecated assertion; rejected because builder families are derived from hash-bound runs",
    )
    parser.add_argument("--backend", choices=["mock", "claude"], default="claude")
    parser.add_argument("--mock-transcript", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed-path", type=Path, default=Path("contracts/integrity_audit_seed.txt"))
    parser.add_argument("--output", type=Path, help="Report path under reports/status/integrity_audit/ (default: UTC timestamp)")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--hermetic-copy", action="store_true", help="Use a temporary copy (test fixtures only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_audit(args)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"integrity_audit_error:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
