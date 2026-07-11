#!/usr/bin/env python3
"""Generate and audit an AEA-shaped replication package from repository state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "research_swarm.replication_package.v1"
PROFILES = {"empirical", "modeling", "hybrid"}
BYTE_PATHS = {
    "reports/tables/str_regime_summary.csv",
    "reports/tables/str_regime_summary.md",
    "reports/paper/paper_values.json",
    "reports/exhibits/manifest.json",
}
CONTENT_PATHS = {
    "reports/figures/str_ecosystem_timeseries.svg",
    "reports/figures/str_ecosystem_timeseries.data.json",
    "reports/figures/str_post_dencun_regimes.svg",
    "reports/figures/str_post_dencun_regimes.data.json",
}
COMMON_REQUIRED = {"README.md", "MASTER.sh", "package_manifest.json"}
PROFILE_REQUIRED = {
    "empirical": {
        "data/processed",
        "data/processed_manifest",
        "data/raw_manifest",
        "reports/exhibits/manifest.json",
        "reports/catalog.yaml",
        "reports/paper/paper_values.json",
    },
    "modeling": {
        "modeling/solver_availability.json",
        "modeling/instance_manifest.json",
        "modeling/experiment_design.json",
        "modeling/convergence.log",
        "modeling/derivation_check.py",
        "modeling/clean_room.json",
    },
    "hybrid": {
        "data/processed_manifest",
        "modeling/solver_availability.json",
        "modeling/instance_manifest.json",
        "modeling/experiment_design.json",
        "modeling/convergence.log",
        "modeling/derivation_check.py",
        "bridge/generate_instances.py",
        "bridge/hybrid_interface.yaml",
        "bridge/experiment_output.json",
        "bridge/clean_room.json",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_top_level_not_object:{path}")
    return value


def _project_mode(repo: Path) -> str:
    project = repo / "contracts/project.yaml"
    if project.is_file():
        payload = yaml.safe_load(project.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict) and payload.get("mode") in PROFILES:
            return str(payload["mode"])
    spec = repo / "replication_spec.json"
    if spec.is_file():
        value = _json(spec).get("profile")
        if value in PROFILES:
            return str(value)
    raise ValueError("replication_profile_missing_or_invalid")


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name in {".DS_Store"}:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _declared_dependencies(repo: Path) -> list[dict[str, object]]:
    pyproject = repo / "pyproject.toml"
    dependencies: list[str] = []
    if pyproject.is_file():
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = payload.get("project")
        if isinstance(project, dict) and isinstance(project.get("dependencies"), list):
            dependencies = [item for item in project["dependencies"] if isinstance(item, str)]
    exact: dict[str, set[str]] = {}
    environment_records: list[dict[str, object]] = []
    for manifest_path in sorted((repo / "data/processed_manifest").glob("*.json")):
        payload = _json(manifest_path)
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            continue
        environment_records.append(
            {"manifest": manifest_path.relative_to(repo).as_posix(), "environment": environment}
        )
        versions = environment.get("dependencies")
        if isinstance(versions, dict):
            for name, version in versions.items():
                if isinstance(name, str) and isinstance(version, (str, int, float)):
                    exact.setdefault(name.casefold().replace("_", "-"), set()).add(str(version))
    result: list[dict[str, object]] = []
    for declaration in dependencies:
        match = re.match(r"^([A-Za-z0-9_.-]+)", declaration)
        name = match.group(1) if match else declaration
        versions = sorted(exact.get(name.casefold().replace("_", "-"), set()))
        result.append(
            {
                "package": name,
                "declared": declaration,
                "logged_runtime_versions": versions,
                "status": "exact_runtime_logged" if versions else "declared_exact_runtime_unlogged",
            }
        )
    return result


def _processed_environments(repo: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for manifest_path in sorted((repo / "data/processed_manifest").glob("*.json")):
        environment = _json(manifest_path).get("environment")
        if isinstance(environment, dict):
            records.append(
                {
                    "manifest": manifest_path.relative_to(repo).as_posix(),
                    "environment": environment,
                }
            )
    return records


def _raw_amendment(repo: Path) -> dict[str, object] | None:
    for path in sorted((repo / "reports/status/releases").glob("release_*.json"), reverse=True):
        payload = _json(path)
        notes = payload.get("notes")
        if not isinstance(notes, list):
            continue
        for note in notes:
            if isinstance(note, dict) and note.get("type") == "raw_evidence_unavailable":
                return {"release_manifest": path.relative_to(repo).as_posix(), "note": note}
    return None


def _raw_statements(repo: Path) -> list[dict[str, object]]:
    amendment = _raw_amendment(repo)
    statements: list[dict[str, object]] = []
    for path in sorted((repo / "data/raw_manifest").glob("*.json")):
        payload = _json(path)
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        missing = [
            item.get("path")
            for item in files
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and not (repo / str(item["path"])).is_file()
        ]
        statements.append(
            {
                "manifest": path.relative_to(repo).as_posix(),
                "source": payload.get("source", "unlogged"),
                "as_of": payload.get("as_of_utc_date", "unlogged"),
                "access_instruction": payload.get("command", "acquisition command unlogged"),
                "raw_files_present": len(missing) == 0,
                "missing_file_count": len(missing),
                "retention_satisfier": amendment["release_manifest"] if missing and amendment else None,
            }
        )
    return statements


def _exhibit_mapping(repo: Path) -> list[dict[str, object]]:
    path = repo / "reports/exhibits/manifest.json"
    if not path.is_file():
        return []
    payload = _json(path)
    catalog_path = repo / "reports/catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) if catalog_path.is_file() else {}
    catalog_roots = catalog.get("artifact_roots", {}) if isinstance(catalog, dict) else {}
    mappings: list[dict[str, object]] = []
    for item in payload.get("exhibits", []):
        if isinstance(item, dict):
            mappings.append(
                {
                    "exhibit_id": item.get("exhibit_id"),
                    "output": item.get("output"),
                    "builder": item.get("builder"),
                    "inputs": item.get("inputs", []),
                    "catalog_artifact_roots": catalog_roots,
                }
            )
    return mappings


def _render_readme(profile: str, metadata: dict[str, object]) -> str:
    lines = [
        "# Replication package",
        "",
        f"Profile: **{profile}**",
        "",
        "Run `./MASTER.sh` to execute the package master command: `make reproduce-analysis && make paper`.",
        "",
        "## Reproduction levels",
        "",
        "- **Functional:** package construction and the declared master traversal are machine-audited.",
        "- **Reproduced:** requires a recorded clean-room run by a non-author agent. The live empirical package reports this level as pending; it is never inferred from an authoring run.",
        "",
        "## Verification bars",
        "",
        "- Byte identity: deterministic tables, `paper_values.json`, exhibits manifest, package manifest, and generated disclosure are compared by SHA-256/bytes.",
        "- Content equivalence: SVG figures and their numeric sidecars are compared structurally/numerically because Matplotlib rendering bytes are not cross-platform deterministic.",
        "- These bars are distinct and may not be substituted for one another.",
        "",
        "## Declared and logged package versions",
        "",
    ]
    for item in metadata.get("dependencies", []):
        assert isinstance(item, dict)
        versions = item["logged_runtime_versions"]
        rendered = ", ".join(str(value) for value in versions) if versions else "exact runtime version unlogged"
        lines.append(f"- `{item['package']}`: declared `{item['declared']}`; {rendered}.")
    if not metadata.get("dependencies"):
        lines.append("- No declared Python dependencies were present in `pyproject.toml`.")

    lines.extend(["", "### Logged processed-manifest runtime environments", ""])
    environments = metadata.get("processed_environments")
    if isinstance(environments, list) and environments:
        for item in environments:
            assert isinstance(item, dict)
            lines.append(
                f"- `{item['manifest']}`: `{json.dumps(item['environment'], sort_keys=True, separators=(',', ':'))}`"
            )
    else:
        lines.append("- No `environment` block is logged in the committed processed manifests; no exact runtime value is inferred.")

    lines.extend(["", "## Data availability and access", ""])
    raw = metadata.get("raw_statements", [])
    if isinstance(raw, list):
        for item in raw:
            assert isinstance(item, dict)
            availability = "present" if item["raw_files_present"] else "raw_evidence_unavailable"
            lines.append(
                f"- `{item['manifest']}` ({item['source']}, as of {item['as_of']}): **{availability}**. Access/reacquisition instruction recorded at ingest: `{item['access_instruction']}`."
            )
    if any(isinstance(item, dict) and not item.get("raw_files_present") for item in raw if isinstance(raw, list)):
        lines.extend(
            [
                "",
                "### AEA-style partial-reproducibility statement",
                "",
                "The manifested raw evidence is unavailable and is not included or represented as recoverable. This package supports processed-data-to-results reproduction only. The release amendment named above is the truthful retention satisfier; users must reacquire source data using the recorded ingest commands and provider access conditions for any future raw-to-processed replay.",
            ]
        )

    lines.extend(["", "## Exhibit-to-source mapping", ""])
    for item in metadata.get("exhibits", []):
        assert isinstance(item, dict)
        input_paths = [str(entry.get("path")) for entry in item.get("inputs", []) if isinstance(entry, dict)]
        lines.append(
            f"- `{item['exhibit_id']}` → `{item['output']}`; builder `{item['builder']}`; sources: {', '.join(f'`{value}`' for value in input_paths)}."
        )
    if not metadata.get("exhibits"):
        lines.append("- No empirical exhibit manifest applies to this profile.")
    elif isinstance(metadata.get("exhibits"), list):
        first = metadata["exhibits"][0]
        if isinstance(first, dict):
            lines.append(
                "- Catalog artifact roots: `"
                + json.dumps(first.get("catalog_artifact_roots", {}), sort_keys=True, separators=(",", ":"))
                + "`."
            )

    if profile in {"modeling", "hybrid"}:
        lines.extend(
            [
                "",
                "## Modeling core",
                "",
                "Solver availability, instance manifests, seeds, budgets, convergence logs, derivation checks, and preregistered content-equivalence tolerances are included under `modeling/`.",
            ]
        )
    if profile == "hybrid":
        lines.extend(
            [
                "",
                "## Hybrid bridge",
                "",
                "The bridge inventory binds processed source manifests to regenerated instances and experiment outputs. Reproduced status requires `bridge/clean_room.json` to show a non-author run with `traversed_bridge: true`.",
            ]
        )
    return "\n".join(lines) + "\n"


def _manifest_members(package: Path) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        if path.relative_to(package).as_posix().startswith("reports/replication/"):
            continue
        relpath = path.relative_to(package).as_posix()
        if relpath in CONTENT_PATHS:
            members.append(
                {
                    "path": relpath,
                    "verification_bar": "content_equivalence",
                    "content_reference": (
                        relpath if relpath.endswith(".data.json") else relpath.removesuffix(".svg") + ".data.json"
                    ),
                }
            )
        else:
            members.append(
                {
                    "path": relpath,
                    "sha256": _sha(path),
                    "bytes": path.stat().st_size,
                    "verification_bar": "byte_identity",
                }
            )
    return members


def _copy_empirical(repo: Path, package: Path) -> None:
    for relpath in (
        "Makefile",
        "pyproject.toml",
        "src",
        "scripts",
        "contracts",
        "docs",
        "registry",
        "data/processed",
        "data/samples",
        "data/processed_manifest",
        "data/raw_manifest",
        "reports/catalog.yaml",
        "reports/exhibits",
        "reports/figures",
        "reports/tables",
        "reports/validation",
        "reports/replication",
        "reports/status/swarm_runs",
        "reports/status/reviews",
        "reports/status/events",
    ):
        _copy_tree(repo / relpath, package / relpath)
    for relpath in (
        "reports/paper/_quarto.yml",
        "reports/paper/index.qmd",
        "reports/paper/references.bib",
        "reports/paper/paper_values.json",
        "reports/paper/registry.json",
        "reports/paper/disclosure.md",
    ):
        _copy_tree(repo / relpath, package / relpath)


def _copy_fixture(repo: Path, package: Path, profile: str) -> None:
    spec = _json(repo / "replication_spec.json")
    for relpath in spec.get("package_members", []):
        if isinstance(relpath, str):
            _copy_tree(repo / relpath, package / relpath)
    if (repo / "data/processed_manifest").exists():
        _copy_tree(repo / "data/processed_manifest", package / "data/processed_manifest")
    if (repo / "pyproject.toml").is_file():
        _copy_tree(repo / "pyproject.toml", package / "pyproject.toml")
    commands = ["#!/bin/sh", "set -eu"]
    if profile == "hybrid":
        commands.append("python bridge/generate_instances.py")
    commands.append("python modeling/derivation_check.py")
    (package / "MASTER.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    os.chmod(package / "MASTER.sh", 0o755)


def generate_package(repo: Path, package: Path, *, profile: str | None = None) -> dict[str, object]:
    repo = repo.resolve()
    package = package.resolve()
    profile = profile or _project_mode(repo)
    if profile not in PROFILES:
        raise ValueError(f"replication_profile_invalid:{profile}")
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    fixture = (repo / "replication_spec.json").is_file()
    if fixture:
        _copy_fixture(repo, package, profile)
    else:
        _copy_empirical(repo, package)
        (package / "MASTER.sh").write_text(
            "#!/bin/sh\nset -eu\nmake reproduce-analysis\nmake paper\n",
            encoding="utf-8",
        )
        os.chmod(package / "MASTER.sh", 0o755)

    metadata = {
        "dependencies": _declared_dependencies(repo),
        "processed_environments": _processed_environments(repo),
        "raw_statements": _raw_statements(repo),
        "exhibits": _exhibit_mapping(repo),
    }
    (package / "README.md").write_text(_render_readme(profile, metadata), encoding="utf-8")
    bars = [
        {"path": path, "artifact_class": "deterministic_output", "verification_bar": "byte_identity"}
        for path in sorted(BYTE_PATHS)
        if (package / path).is_file()
    ] + [
        {"path": path, "artifact_class": "figure_or_sidecar", "verification_bar": "content_equivalence"}
        for path in sorted(CONTENT_PATHS)
        if (package / path).is_file()
    ]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "master_command": "make reproduce-analysis && make paper" if not fixture else "python modeling/derivation_check.py",
        "reproduction_bars": bars,
        "levels": {
            "Functional": "declared",
            "Reproduced": "fixture_evidence_present" if fixture else "pending_non_author_clean_room",
        },
        "members": _manifest_members(package),
        "readme_sources": {
            "declared_dependencies": "pyproject.toml",
            "runtime_versions": "data/processed_manifest/*.json#environment.dependencies",
            "exhibits": ["reports/exhibits/manifest.json", "reports/catalog.yaml"],
            "data_statements": "data/raw_manifest/*.json",
        },
    }
    (package / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _required_present(package: Path, relpath: str) -> bool:
    path = package / relpath
    return path.is_dir() and any(item.is_file() for item in path.rglob("*")) if relpath.endswith(("processed", "processed_manifest", "raw_manifest")) else path.exists()


def audit_package(package: Path, *, execute_master: bool = False) -> dict[str, object]:
    package = package.resolve()
    failures: list[str] = []
    manifest_path = package / "package_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "profile": None, "levels": {"Functional": False, "Reproduced": False}, "failures": ["replication_required_member_missing:package_manifest.json"]}
    try:
        manifest = _json(manifest_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "profile": None, "levels": {"Functional": False, "Reproduced": False}, "failures": [f"replication_manifest_invalid:{exc}"]}
    profile = manifest.get("profile")
    if profile not in PROFILES:
        failures.append(f"replication_profile_invalid:{profile}")
        profile = "empirical"
    for relpath in sorted(COMMON_REQUIRED | PROFILE_REQUIRED[str(profile)]):
        if not _required_present(package, relpath):
            failures.append(f"replication_required_member_missing:{relpath}")

    by_member = {
        item.get("path"): item
        for item in manifest.get("members", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for relpath, item in sorted(by_member.items()):
        path = package / str(relpath)
        if relpath in CONTENT_PATHS:
            if (
                not path.is_file()
                or item.get("verification_bar") != "content_equivalence"
                or not isinstance(item.get("content_reference"), str)
                or not (package / str(item["content_reference"])).is_file()
            ):
                failures.append(f"replication_member_content_reference_invalid:{relpath}")
        elif not path.is_file() or item.get("sha256") != _sha(path) or item.get("bytes") != path.stat().st_size:
            failures.append(f"replication_member_hash_mismatch:{relpath}")

    for bar in manifest.get("reproduction_bars", []):
        if not isinstance(bar, dict) or not isinstance(bar.get("path"), str):
            failures.append("replication_bar_invalid")
            continue
        relpath = str(bar["path"])
        expected = "byte_identity" if relpath in BYTE_PATHS else "content_equivalence" if relpath in CONTENT_PATHS else None
        if expected is None or bar.get("verification_bar") != expected:
            failures.append(f"replication_bar_mismatch:{relpath}:expected={expected}:actual={bar.get('verification_bar')}")

    readme = (package / "README.md").read_text(encoding="utf-8") if (package / "README.md").is_file() else ""
    missing_raw = any(
        path.is_file()
        and any(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and not (package / str(item["path"])).is_file()
            for item in _json(path).get("files", [])
        )
        for path in (package / "data/raw_manifest").glob("*.json")
    ) if (package / "data/raw_manifest").is_dir() else False
    if missing_raw:
        if "raw_evidence_unavailable" not in readme or "partial-reproducibility statement" not in readme.casefold():
            failures.append("replication_raw_unavailable_statement_missing")
        if "Access/reacquisition instruction recorded at ingest" not in readme:
            failures.append("replication_access_instruction_missing")

    if profile in {"modeling", "hybrid"}:
        solver = _json(package / "modeling/solver_availability.json") if (package / "modeling/solver_availability.json").is_file() else {}
        if not solver.get("license_class") or not (solver.get("open_solver_fallback") or solver.get("open_instance_subset")):
            failures.append("replication_solver_availability_incomplete")
        design = _json(package / "modeling/experiment_design.json") if (package / "modeling/experiment_design.json").is_file() else {}
        if not isinstance(design.get("content_equivalence_tolerances"), dict):
            failures.append("replication_modeling_tolerances_missing")

    reproduced = False
    if profile == "modeling" and (package / "modeling/clean_room.json").is_file():
        evidence = _json(package / "modeling/clean_room.json")
        reproduced = bool(evidence.get("non_author_agent") and evidence.get("master_script_returncode") == 0)
        if not reproduced:
            failures.append("replication_modeling_clean_room_invalid")
    elif profile == "hybrid" and (package / "bridge/clean_room.json").is_file():
        clean = _json(package / "bridge/clean_room.json")
        instance_path = package / "modeling/instance_manifest.json"
        output_path = package / "bridge/experiment_output.json"
        instance = _json(instance_path) if instance_path.is_file() else {}
        output = _json(output_path) if output_path.is_file() else {}
        source_rel = instance.get("source_manifest", {}).get("path") if isinstance(instance.get("source_manifest"), dict) else None
        source_sha = instance.get("source_manifest", {}).get("sha256") if isinstance(instance.get("source_manifest"), dict) else None
        instance_link = output.get("instance_manifest", {}) if isinstance(output.get("instance_manifest"), dict) else {}
        chain_ok = (
            isinstance(source_rel, str)
            and (package / source_rel).is_file()
            and source_sha == _sha(package / source_rel)
            and instance_link.get("path") == "modeling/instance_manifest.json"
            and instance_link.get("sha256") == _sha(instance_path)
        )
        if not chain_ok:
            failures.append("replication_cross_layer_hash_link_broken")
        traversal = subprocess.run(
            [sys.executable, "bridge/generate_instances.py"],
            cwd=package,
            check=False,
            capture_output=True,
            text=True,
        ) if (package / "bridge/generate_instances.py").is_file() else None
        regenerated_source_sha = traversal.stdout.strip() if traversal is not None and traversal.returncode == 0 else None
        reproduced = bool(
            chain_ok
            and clean.get("non_author_agent")
            and clean.get("master_script_returncode") == 0
            and clean.get("traversed_bridge") is True
            and clean.get("regenerated_instances") is True
            and regenerated_source_sha == source_sha
            and clean.get("regenerated_source_sha256") == source_sha
        )
        if (
            not clean.get("traversed_bridge")
            or not clean.get("regenerated_instances")
            or regenerated_source_sha != source_sha
            or clean.get("regenerated_source_sha256") != source_sha
        ):
            failures.append("replication_hybrid_clean_room_bridge_not_traversed")
    if execute_master and not failures:
        completed = subprocess.run(["sh", "MASTER.sh"], cwd=package, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            failures.append(f"replication_master_failed:{completed.returncode}:{completed.stderr[-500:]}")

    functional_failures = [item for item in failures if not item.startswith(("replication_modeling_clean_room", "replication_hybrid_clean_room"))]
    audit_ok = not functional_failures and (profile == "empirical" or reproduced)
    return {
        "ok": audit_ok,
        "profile": profile,
        "levels": {"Functional": not functional_failures, "Reproduced": reproduced},
        "failures": failures,
    }


def audit_repo_profiles(repo: Path, *, execute_empirical_master: bool = True) -> dict[str, object]:
    repo = repo.resolve()
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="replication-audit-") as tmp:
        root = Path(tmp)
        live = root / "empirical"
        generate_package(repo, live, profile="empirical")
        results["empirical"] = audit_package(live, execute_master=execute_empirical_master)
        for profile in ("modeling", "hybrid"):
            fixture = repo / f"tests/fixtures/m4c_{profile}"
            package = root / profile
            if not fixture.is_dir():
                results[profile] = {"ok": False, "profile": profile, "levels": {"Functional": False, "Reproduced": False}, "failures": [f"replication_fixture_missing:{fixture.relative_to(repo)}"]}
                continue
            generate_package(fixture, package, profile=profile)
            results[profile] = audit_package(package, execute_master=True)
    ok = all(isinstance(value, dict) and value.get("ok") for value in results.values())
    return {"ok": ok, "profiles": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="replication_package.py")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=Path("build/replication_package"))
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--all-profiles", action="store_true")
    parser.add_argument("--execute-master", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=Path("reports/replication"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    if args.all_profiles:
        result = audit_repo_profiles(repo, execute_empirical_master=args.execute_master)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    output = args.output if args.output.is_absolute() else repo / args.output
    manifest = generate_package(repo, output, profile=args.profile)
    evidence_dir = args.evidence_dir if args.evidence_dir.is_absolute() else repo / args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "README.md", evidence_dir / "README.md")
    shutil.copyfile(output / "package_manifest.json", evidence_dir / "package_manifest.json")
    result = audit_package(output, execute_master=args.execute_master) if args.audit else {"ok": True}
    print(json.dumps({"output": output.as_posix(), "manifest": manifest, "audit": result}, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
