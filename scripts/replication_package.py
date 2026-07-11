#!/usr/bin/env python3
"""Generate and audit an AEA-shaped replication package from repository state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Iterable

from pack_config import load_pack_config, pack_value



REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "research_swarm.replication_package.v1"
PROFILES = {"empirical", "modeling", "hybrid"}


def _empirical_expected_bars(repo: Path) -> dict[str, str]:
    pack = load_pack_config(repo)
    byte_keys = ("regime_table_csv", "regime_table_markdown", "paper_values", "exhibits_manifest")
    content_keys = ("ecosystem_figure", "ecosystem_figure_data", "regime_figure", "regime_figure_data")
    return {
        **{pack_value(pack, f"analysis.outputs.{key}"): "byte_identity" for key in byte_keys},
        **{pack_value(pack, f"analysis.outputs.{key}"): "content_equivalence" for key in content_keys},
    }


def _content_paths(repo: Path) -> set[str]:
    return {
        path
        for path, bar in _empirical_expected_bars(repo).items()
        if bar == "content_equivalence"
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
        "modeling/convergence.jsonl",
        "modeling/derivation_check.py",
        "modeling/clean_room.json",
    },
    "hybrid": {
        "data/processed_manifest",
        "modeling/solver_availability.json",
        "modeling/instance_manifest.json",
        "modeling/experiment_design.json",
        "modeling/convergence.jsonl",
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
        # project.yaml is authored YAML; the repo reads only the `mode:` scalar from it
        # with a manual line parse (no pyyaml runtime dependency — see quality_gates
        # _parse_project_mode). framework.json's project_mode is the JSON fallback.
        for raw_line in project.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line.startswith("mode:"):
                mode = line.split(":", 1)[1].strip().strip("'\"").lower()
                if mode in PROFILES:
                    return mode
                break
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


def _repo_file_inventory(repo: Path) -> list[Path]:
    if (repo / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return [repo / value.decode("utf-8") for value in completed.stdout.split(b"\0") if value]
    return [
        path
        for path in sorted(repo.rglob("*"))
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    ]


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
    # reports/catalog.yaml is authored YAML (rendered by release_assembly); the exhibit
    # mapping's authoritative source is the JSON exhibits manifest above, and the catalog's
    # artifact_roots is supplementary. Read it only if it is JSON-in-YAML; otherwise degrade
    # to {} rather than take a pyyaml runtime dependency.
    catalog: dict[str, object] = {}
    if catalog_path.is_file():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                catalog = loaded
        except json.JSONDecodeError:
            catalog = {}
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
        "- **Reproduced:** requires an observed clean-workspace master run whose regenerated outputs satisfy every declared verification bar. The live empirical package reports this level as pending a non-author run; identity is never inferred from an authoring run.",
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
                "The audit removes committed bridge outputs in a clean workspace, runs the master, observes recreation of the instance and experiment output, and verifies the source→instance→output chain on those regenerated bytes. `bridge/clean_room.json` identifies the fixture runner only; its booleans are not reproduction authority.",
            ]
        )
    return "\n".join(lines) + "\n"


def _manifest_members(package: Path) -> list[dict[str, object]]:
    config_root = package if (package / "contracts" / "pack.json").is_file() else REPO_ROOT
    content_paths = _content_paths(config_root)
    members: list[dict[str, object]] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name == "package_manifest.json":
            continue
        if path.relative_to(package).as_posix().startswith("reports/replication/"):
            continue
        relpath = path.relative_to(package).as_posix()
        if relpath in content_paths:
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
    prefixes = (
        "Makefile",
        "pyproject.toml",
        "src/",
        "scripts/",
        "contracts/",
        "docs/",
        "registry/",
        "data/processed/",
        "data/samples/",
        "data/processed_manifest/",
        "data/raw_manifest/",
        "reports/catalog.yaml",
        "reports/exhibits/",
        "reports/figures/",
        "reports/tables/",
        "reports/validation/",
        "reports/replication/",
        "reports/status/swarm_runs/",
        "reports/status/reviews/",
        "reports/status/events/",
    )
    exact_paper = {
        "reports/paper/_quarto.yml",
        "reports/paper/index.qmd",
        "reports/paper/references.bib",
        "reports/paper/paper_values.json",
        "reports/paper/registry.json",
        "reports/paper/disclosure.md",
    }
    for source in _repo_file_inventory(repo):
        if not source.is_file():
            continue
        relpath = source.relative_to(repo).as_posix()
        if relpath not in exact_paper and not any(
            relpath == prefix or (prefix.endswith("/") and relpath.startswith(prefix))
            for prefix in prefixes
        ):
            continue
        target = package / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _copy_fixture(repo: Path, package: Path, profile: str) -> None:
    spec = _json(repo / "replication_spec.json")
    _copy_tree(repo / "replication_spec.json", package / "replication_spec.json")
    for relpath in spec.get("package_members", []):
        if isinstance(relpath, str):
            _copy_tree(repo / relpath, package / relpath)
    if (repo / "data/processed_manifest").exists():
        _copy_tree(repo / "data/processed_manifest", package / "data/processed_manifest")
    if (repo / "pyproject.toml").is_file():
        _copy_tree(repo / "pyproject.toml", package / "pyproject.toml")
    commands = ["#!/bin/sh", "set -eu"]
    if profile == "hybrid":
        commands.append('"${PYTHON:-python3}" bridge/generate_instances.py')
    commands.append('"${PYTHON:-python3}" modeling/derivation_check.py')
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
    expected_bars = _expected_reproduction_bars(package, profile)
    bars = [
        {
            "path": path,
            "artifact_class": (
                "figure_or_sidecar" if verification_bar == "content_equivalence" else "deterministic_output"
            ),
            "verification_bar": verification_bar,
        }
        for path, verification_bar in sorted(expected_bars.items())
        if (package / path).is_file()
    ]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "master_command": "make reproduce-analysis && make paper" if not fixture else "python modeling/derivation_check.py",
        "reproduction_bars": bars,
        "levels": {
            "Functional": "declared",
            "Reproduced": "requires_observed_clean_room" if fixture else "pending_non_author_clean_room",
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


def _expected_reproduction_bars(package: Path, profile: str) -> dict[str, str]:
    if profile == "empirical":
        config_root = package if (package / "contracts" / "pack.json").is_file() else REPO_ROOT
        return _empirical_expected_bars(config_root)
    spec_path = package / "replication_spec.json"
    if not spec_path.is_file():
        return {}
    expected: dict[str, str] = {}
    for item in _json(spec_path).get("reproduction_bars", []):
        if (
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item.get("verification_bar") in {"byte_identity", "content_equivalence"}
        ):
            expected[str(item["path"])] = str(item["verification_bar"])
    return expected


def _declared_reproduction_bars(manifest: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    declared: dict[str, str] = {}
    failures: list[str] = []
    bars = manifest.get("reproduction_bars")
    if not isinstance(bars, list):
        return {}, ["replication_bars_not_list"]
    for item in bars:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            failures.append("replication_bar_invalid")
            continue
        relpath = str(item["path"])
        if relpath in declared:
            failures.append(f"replication_bar_duplicate:{relpath}")
            continue
        declared[relpath] = str(item.get("verification_bar"))
    return declared, failures


def _content_reference(relpath: str) -> str:
    return relpath if relpath.endswith(".data.json") else relpath.removesuffix(".svg") + ".data.json"


def _content_mismatches(expected: object, actual: object, path: str) -> list[str]:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected == actual and type(expected) is type(actual) else [path]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))):
            return [f"{path}:non_finite"]
        return [] if math.isclose(float(expected), float(actual), rel_tol=1e-10, abs_tol=1e-10) else [path]
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}:keys"]
        return [
            mismatch
            for key in expected
            for mismatch in _content_mismatches(expected[key], actual[key], f"{path}.{key}")
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}:length"]
        return [
            mismatch
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual))
            for mismatch in _content_mismatches(expected_item, actual_item, f"{path}[{index}]")
        ]
    return [] if expected == actual else [path]


def _snapshot_declared_outputs(
    package: Path,
    bars: dict[str, str],
) -> tuple[dict[str, object], list[str]]:
    snapshots: dict[str, object] = {}
    failures: list[str] = []
    for relpath, verification_bar in sorted(bars.items()):
        output_path = package / relpath
        if not output_path.is_file():
            failures.append(f"replication_declared_output_missing:{relpath}")
            continue
        if verification_bar == "byte_identity":
            snapshots[relpath] = output_path.read_bytes()
        elif verification_bar == "content_equivalence":
            reference = package / _content_reference(relpath)
            if not reference.is_file():
                failures.append(f"replication_content_reference_missing:{relpath}:{reference.relative_to(package)}")
                continue
            try:
                snapshots[relpath] = json.loads(reference.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                failures.append(f"replication_content_reference_invalid:{relpath}")
        else:
            failures.append(f"replication_bar_invalid_value:{relpath}:{verification_bar}")
    return snapshots, failures


def _verify_regenerated_outputs(
    clean: Path,
    bars: dict[str, str],
    snapshots: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    for relpath, verification_bar in sorted(bars.items()):
        output_path = clean / relpath
        if not output_path.is_file():
            failures.append(f"replication_reproduced_output_missing:{relpath}")
            continue
        if relpath not in snapshots:
            failures.append(f"replication_reproduced_baseline_missing:{relpath}")
            continue
        if verification_bar == "byte_identity":
            if output_path.read_bytes() != snapshots[relpath]:
                failures.append(f"replication_reproduced_byte_mismatch:{relpath}")
        else:
            reference = clean / _content_reference(relpath)
            if not reference.is_file():
                failures.append(f"replication_reproduced_content_reference_missing:{relpath}")
                continue
            try:
                actual = json.loads(reference.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                failures.append(f"replication_reproduced_content_reference_invalid:{relpath}")
                continue
            mismatches = _content_mismatches(snapshots[relpath], actual, relpath)
            if mismatches:
                failures.append(f"replication_reproduced_content_mismatch:{relpath}:{mismatches[0]}")
    return failures


def _hybrid_chain_failures(clean: Path, committed_instance_sha: str) -> list[str]:
    instance_path = clean / "modeling/instance_manifest.json"
    output_path = clean / "bridge/experiment_output.json"
    if not instance_path.is_file() or not output_path.is_file():
        return ["replication_hybrid_clean_room_bridge_not_traversed"]
    try:
        instance = _json(instance_path)
        output = _json(output_path)
    except (ValueError, json.JSONDecodeError):
        return ["replication_cross_layer_hash_link_broken"]
    source_link = instance.get("source_manifest") if isinstance(instance.get("source_manifest"), dict) else {}
    output_link = output.get("instance_manifest") if isinstance(output.get("instance_manifest"), dict) else {}
    source_rel = source_link.get("path")
    source_path = clean / str(source_rel) if isinstance(source_rel, str) else None
    regenerated_instance_sha = _sha(instance_path)
    chain_ok = bool(
        source_path is not None
        and source_path.is_file()
        and source_link.get("sha256") == _sha(source_path)
        and regenerated_instance_sha == committed_instance_sha
        and output_link.get("path") == "modeling/instance_manifest.json"
        and output_link.get("sha256") == regenerated_instance_sha
    )
    return [] if chain_ok else ["replication_cross_layer_hash_link_broken"]


def _run_master_reproduction(
    package: Path,
    profile: str,
    bars: dict[str, str],
) -> tuple[bool, bool, list[str]]:
    snapshots, failures = _snapshot_declared_outputs(package, bars)
    if failures:
        return False, False, failures
    committed_instance_sha = (
        _sha(package / "modeling/instance_manifest.json")
        if profile == "hybrid" and (package / "modeling/instance_manifest.json").is_file()
        else ""
    )
    with tempfile.TemporaryDirectory(prefix=f"replication-clean-{profile}-") as tmp:
        clean = Path(tmp) / "package"
        shutil.copytree(package, clean)
        if profile in {"modeling", "hybrid"}:
            for relpath in bars:
                path = clean / relpath
                if path.is_file():
                    path.unlink()
        environment = dict(os.environ)
        environment["PYTHON"] = sys.executable
        completed = subprocess.run(
            ["sh", "MASTER.sh"],
            cwd=clean,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return False, False, [
                f"replication_master_failed:{completed.returncode}:{completed.stderr[-500:]}"
            ]
        failures.extend(_verify_regenerated_outputs(clean, bars, snapshots))
        bridge_traversed = False
        if profile == "hybrid":
            bridge_failures = _hybrid_chain_failures(clean, committed_instance_sha)
            failures.extend(bridge_failures)
            bridge_traversed = not bridge_failures
        return True, bridge_traversed, failures


def audit_package(package: Path, *, execute_master: bool = False) -> dict[str, object]:
    package = package.resolve()
    structural_failures: list[str] = []
    manifest_path = package / "package_manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "profile": None, "levels": {"Functional": False, "Reproduced": False}, "failures": ["replication_required_member_missing:package_manifest.json"]}
    try:
        manifest = _json(manifest_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "profile": None, "levels": {"Functional": False, "Reproduced": False}, "failures": [f"replication_manifest_invalid:{exc}"]}
    profile = manifest.get("profile")
    if profile not in PROFILES:
        structural_failures.append(f"replication_profile_invalid:{profile}")
        profile = "empirical"
    for relpath in sorted(COMMON_REQUIRED | PROFILE_REQUIRED[str(profile)]):
        if not _required_present(package, relpath):
            structural_failures.append(f"replication_required_member_missing:{relpath}")

    by_member = {
        item.get("path"): item
        for item in manifest.get("members", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    config_root = package if (package / "contracts" / "pack.json").is_file() else REPO_ROOT
    content_paths = _content_paths(config_root)
    for relpath, item in sorted(by_member.items()):
        path = package / str(relpath)
        if relpath in content_paths:
            if (
                not path.is_file()
                or item.get("verification_bar") != "content_equivalence"
                or not isinstance(item.get("content_reference"), str)
                or not (package / str(item["content_reference"])).is_file()
            ):
                structural_failures.append(f"replication_member_content_reference_invalid:{relpath}")
        elif not path.is_file() or item.get("sha256") != _sha(path) or item.get("bytes") != path.stat().st_size:
            structural_failures.append(f"replication_member_hash_mismatch:{relpath}")

    expected_bars = _expected_reproduction_bars(package, str(profile))
    declared_bars, bar_failures = _declared_reproduction_bars(manifest)
    structural_failures.extend(bar_failures)
    for relpath in sorted(set(expected_bars) - set(declared_bars)):
        structural_failures.append(f"replication_bar_coverage_missing:{relpath}")
    for relpath in sorted(set(declared_bars) - set(expected_bars)):
        structural_failures.append(f"replication_bar_unexpected:{relpath}")
    for relpath in sorted(set(expected_bars) & set(declared_bars)):
        if declared_bars[relpath] != expected_bars[relpath]:
            structural_failures.append(
                f"replication_bar_mismatch:{relpath}:expected={expected_bars[relpath]}:actual={declared_bars[relpath]}"
            )

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
            structural_failures.append("replication_raw_unavailable_statement_missing")
        if "Access/reacquisition instruction recorded at ingest" not in readme:
            structural_failures.append("replication_access_instruction_missing")

    if profile in {"modeling", "hybrid"}:
        solver = _json(package / "modeling/solver_availability.json") if (package / "modeling/solver_availability.json").is_file() else {}
        if not solver.get("license_class") or not (solver.get("open_solver_fallback") or solver.get("open_instance_subset")):
            structural_failures.append("replication_solver_availability_incomplete")
        design = _json(package / "modeling/experiment_design.json") if (package / "modeling/experiment_design.json").is_file() else {}
        if not isinstance(design.get("content_equivalence_tolerances"), dict):
            structural_failures.append("replication_modeling_tolerances_missing")
    if profile == "hybrid" and (package / "modeling/instance_manifest.json").is_file():
        structural_failures.extend(
            _hybrid_chain_failures(
                package,
                _sha(package / "modeling/instance_manifest.json"),
            )
        )

    failures = list(structural_failures)
    functional = not structural_failures
    reproduced = False
    output_verified = False
    bridge_traversed = False
    master_executed = bool(profile in {"modeling", "hybrid"} or execute_master)
    if functional and master_executed:
        master_ok, bridge_traversed, reproduction_failures = _run_master_reproduction(
            package,
            str(profile),
            expected_bars,
        )
        failures.extend(reproduction_failures)
        functional = master_ok
        output_verified = master_ok and not reproduction_failures
        if profile in {"modeling", "hybrid"}:
            metadata_path = package / ("bridge/clean_room.json" if profile == "hybrid" else "modeling/clean_room.json")
            metadata = _json(metadata_path) if metadata_path.is_file() else {}
            non_author_recorded = bool(metadata.get("non_author_agent"))
            if not non_author_recorded:
                failures.append(f"replication_{profile}_clean_room_identity_missing")
            reproduced = bool(output_verified and non_author_recorded and (profile != "hybrid" or bridge_traversed))
            if profile == "hybrid" and not bridge_traversed:
                failures.append("replication_hybrid_clean_room_bridge_not_traversed")
    audit_ok = bool(
        functional
        and (
            (profile == "empirical" and (not execute_master or output_verified))
            or (profile in {"modeling", "hybrid"} and reproduced)
        )
    )
    return {
        "ok": audit_ok,
        "profile": profile,
        "levels": {"Functional": functional, "Reproduced": reproduced},
        "master_execution": (
            "passed" if master_executed and output_verified else "failed" if master_executed else "staged_release_perimeter"
        ),
        "output_verification": output_verified,
        "bridge_traversed": bridge_traversed,
        "failures": failures,
    }


def _fixture_trackability_failures(repo: Path, fixture: Path, profile: str) -> list[str]:
    failures: list[str] = []
    required = PROFILE_REQUIRED[profile]
    for relpath in sorted(required):
        source = fixture / relpath
        if not source.exists():
            failures.append(f"replication_fixture_member_missing:{fixture.relative_to(repo)}/{relpath}")
            continue
        if (repo / ".git").exists():
            candidate = source.relative_to(repo).as_posix()
            ignored = subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", "--", candidate],
                check=False,
            ).returncode == 0
            if ignored:
                failures.append(f"replication_fixture_member_gitignored:{candidate}")
    return failures


def audit_repo_profiles(repo: Path, *, execute_empirical_master: bool = False) -> dict[str, object]:
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
            trackability_failures = _fixture_trackability_failures(repo, fixture, profile)
            generate_package(fixture, package, profile=profile)
            results[profile] = audit_package(package, execute_master=True)
            if trackability_failures and isinstance(results[profile], dict):
                results[profile]["ok"] = False
                results[profile]["levels"]["Functional"] = False
                results[profile]["failures"] = trackability_failures + list(results[profile]["failures"])
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
