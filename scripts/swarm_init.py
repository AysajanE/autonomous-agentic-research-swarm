#!/usr/bin/env python3
"""Create a project-pack scaffold for the repo-native research kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


KERNEL_ROOT = Path(__file__).resolve().parents[1]
MODES = ("empirical", "modeling", "hybrid")


def _write_text(root: Path, relpath: str, content: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(root: Path, relpath: str, payload: object) -> None:
    _write_text(root, relpath, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_kernel_contract(root: Path, relpath: str) -> None:
    source = KERNEL_ROOT / relpath
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _scaffold_pack(mode: str) -> dict[str, object]:
    base = json.loads((KERNEL_ROOT / "contracts" / "pack.json").read_text(encoding="utf-8"))
    base["scaffold"] = True
    base["project"] = {
        "package_name": f"new-{mode}-research-pack",
        "primary_metric_label": "Primary research metric",
    }
    base["workflow"] = {
        "network_workstreams": ["data", "literature"],
        "local_etl_workstreams": ["data"],
        "operator_workstream": "operations",
        "integration_ready_eligible_workstreams": ["contracts", "integration", "operations"],
    }
    base["paths"] = {
        "registry": "registry/entities.csv",
        "panel_schema_index": "contracts/schemas/panel_schema.yaml",
        "primary_panel_schema": "contracts/schemas/primary_panel_v1.json",
        "decomposition_panel_schema": "contracts/schemas/decomposition_panel_v1.json",
        "primary_panel": "data/processed/primary_panel.csv",
        "primary_panel_sample": "data/samples/primary_panel_sample.csv",
        "primary_panel_manifest_glob": "data/processed_manifest/primary_panel_*.json",
        "primary_panel_manifest_pattern": "data/processed_manifest/primary_panel_{date}.json",
        "vendor_panel": "data/processed/comparison_panel.csv",
        "vendor_panel_sample": "data/samples/comparison_panel_sample.csv",
        "vendor_panel_manifest_pattern": "data/processed_manifest/comparison_panel_{date}.json",
        "rent_components": "data/processed/components.csv",
        "rent_components_sample": "data/samples/components_sample.csv",
        "rent_components_manifest_pattern": "data/processed_manifest/components_{date}.json",
        "decomposition": "data/processed/decomposition.csv",
        "decomposition_sample": "data/samples/decomposition_sample.csv",
        "decomposition_manifest_pattern": "data/processed_manifest/decomposition_{date}.json",
    }
    base["protocol"] = {"required_headings": ["## Inclusion criteria"]}
    base["paper"] = {
        "entrypoint": "reports/paper/index.qmd",
        "build_dir": "reports/paper/build/",
        "artifact_basename": "working_paper",
        "render_manifest": "render_manifest.json",
        "verified_include_targets": [],
    }
    base["analysis"] = {
        "validation_bundle": [],
        "outputs": {
            "ecosystem_figure": "reports/figures/primary.svg",
            "ecosystem_figure_data": "reports/figures/primary.data.json",
            "regime_figure": "reports/figures/secondary.svg",
            "regime_figure_data": "reports/figures/secondary.data.json",
            "regime_table_csv": "reports/tables/summary.csv",
            "regime_table_markdown": "reports/tables/summary.md",
            "paper_values": "reports/paper/paper_values.json",
            "exhibits_manifest": "reports/exhibits/manifest.json",
        },
        "exhibits": {
            "ecosystem_timeseries": "primary_figure",
            "post_regime_figure": "secondary_figure",
            "regime_summary": "summary_table",
        },
    }
    base["historical_manifest_replacements"] = {}
    return base


def create_scaffold(output: Path, mode: str) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)

    for relpath in (
        "contracts/framework.json",
        "contracts/kernel_interface.json",
        "contracts/schemas/kernel_interface_v1.json",
        "contracts/schemas/pack_config_v1.json",
        "contracts/schemas/swarm_run_manifest_v2.json",
        "contracts/schemas/judge_review_log_v2.json",
        "contracts/prompts/worker.md",
        "contracts/prompts/operator.md",
    ):
        _copy_kernel_contract(output, relpath)
    _write_json(output, "contracts/pack.json", _scaffold_pack(mode))
    _write_text(
        output,
        "contracts/project.yaml",
        f"project_id: new-{mode}-project\nproject_name: new-{mode}-project\nmode: {mode}\nstatus: scaffold\n",
    )
    _write_json(output, "contracts/claims.yaml", {"schema_version": "research_swarm.claims.v1", "description": "Empty scaffold claim ledger.", "claims": []})
    _write_text(output, "contracts/model_spec.md", "# Model Spec Lock\n\nStatus: scaffold; no scientific assumptions are locked.\n")
    _copy_kernel_contract(output, f"contracts/program_templates/{mode}.yaml")
    _write_text(output, "contracts/program_template.yaml", f"reference: contracts/program_templates/{mode}.yaml\nreference_only: true\n")
    _write_json(output, "contracts/venue.yaml", {"schema_version": "research_swarm.venue.v1", "status": "reference_only", "reference_scope": "Replace before submission.", "target_venue": None})
    _write_json(output, "contracts/authorship.yaml", {"schema_version": "research_swarm.authorship.v1", "status": "reference_only", "authors": []})
    _write_json(output, "contracts/manuscript_sections.yaml", {"schema_version": "research_swarm.manuscript_sections.v1", "status": "scaffold", "canonical_section_ids": [], "section_headings": {}})
    for name, phase in (("data_construction.lock.md", "2a"), ("analysis_plan.lock.md", "2b")):
        _write_text(output, f"docs/prereg/{name}", f"# Preregistration lock {phase}\n\nStatus: scaffold-inactive\n")

    _write_text(output, ".orchestrator/README.md", "# Orchestrator\n\nNo tasks are instantiated yet.\n")
    _write_text(output, ".orchestrator/workstreams.md", "# Workstreams\n\nScaffold: define pack-specific workstreams before execution.\n")
    for folder in ("backlog", "active", "integration_ready", "ready_for_review", "blocked", "done", "handoff", "mock_transcripts"):
        _write_text(output, f".orchestrator/{folder}/README.md", f"# {folder}\n")
    for folder in ("status", "figures", "tables", "paper", "validation", "models", "replication", "exhibits"):
        _write_text(output, f"reports/{folder}/README.md", f"# {folder}\n\nScaffold; no built artifacts.\n")

    kernel = KERNEL_ROOT.as_posix()
    _write_text(
        output,
        "Makefile",
        ".PHONY: gate test\n"
        f"KERNEL_ROOT ?= {kernel}\n"
        "PYTHON ?= python3.11\n\n"
        "gate:\n\t$(PYTHON) $(KERNEL_ROOT)/scripts/quality_gates.py --repo $(CURDIR)\n\n"
        "test: gate\n",
    )
    _write_text(
        output,
        "README.md",
        f"# New {mode} research pack\n\n"
        "This is an inactive project-pack scaffold. Replace `contracts/pack.json`, "
        "`contracts/project.yaml`, the referenced program template, venue/authorship references, "
        "and preregistration locks before creating tasks. Kernel code remains under `KERNEL_ROOT`; "
        "starting a new project requires contract swaps, not edits to kernel scripts or source.\n\n"
        "Run `make gate` to verify pack compatibility and scaffold shape.\n",
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swarm_init.py")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or Path(f"swarm-pack-{args.mode}")
    try:
        created = create_scaffold(output, args.mode)
    except ValueError as exc:
        print(str(exc))
        return 2
    print(created)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
