"""Uncertainty artifacts and registry-derived limitation text.

Inputs are local registry rows and numeric estimates; no network access occurs.
The numeric coverage-gap fraction must come from a registered measurement audit
rather than being inferred from prose caveats.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def headline_uncertainty_artifact(
    *,
    claim_id: str,
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    coverage_gap_fraction: float,
    coverage_gap_source: str,
) -> dict[str, object]:
    """Build a CI plus measurement-uncertainty band for a headline estimate.

    If a measurement audit bounds relative error by ``g``, interval arithmetic
    expands the sampling CI to ``[min(CI*(1-g), CI*(1+g)), max(...)]``.  This is
    deliberately symmetric: directional corrections require an explicitly
    registered measurement model rather than an assumption inferred from notes.
    """

    numeric = (float(estimate), float(ci_lower), float(ci_upper))
    if not claim_id.strip():
        raise ValueError("claim_id must be non-empty")
    if numeric[1] > numeric[2]:
        raise ValueError("ci_lower must not exceed ci_upper")
    if not 0.0 <= coverage_gap_fraction <= 1.0:
        raise ValueError("coverage_gap_fraction must be between zero and one")
    if not coverage_gap_source.strip():
        raise ValueError("coverage_gap_source must identify the registered audit")
    candidates = (
        numeric[1] * (1.0 - coverage_gap_fraction),
        numeric[1] * (1.0 + coverage_gap_fraction),
        numeric[2] * (1.0 - coverage_gap_fraction),
        numeric[2] * (1.0 + coverage_gap_fraction),
    )
    return {
        "schema_version": "research_swarm.headline_uncertainty.v1",
        "claim_id": claim_id,
        "estimate": numeric[0],
        "confidence_interval": {"lower": numeric[1], "upper": numeric[2]},
        "measurement_uncertainty": {
            "coverage_gap_fraction": float(coverage_gap_fraction),
            "source": coverage_gap_source,
            "lower": min(candidates),
            "upper": max(candidates),
        },
    }


def registry_caveats(
    registry_path: str | Path,
) -> list[dict[str, str]]:
    """Return active registry caveats in stable rollup-id order.

    Rows are retained when status is not ``active`` or notes contain an explicit
    uncertainty marker.  This captures the known Arbitrum/Linea unresolved
    historical-coverage language without hard-coding project names.
    """

    markers = (
        "caveat",
        "unresolved",
        "contradict",
        "incomplete",
        "missing",
        "blank",
        "unknown",
    )
    path = Path(registry_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"rollup_id", "display_name", "status", "notes"}
    if not rows and not required.issubset(set(rows[0]) if rows else set()):
        # DictReader exposes fieldnames even for an empty registry.
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"registry must contain columns: {sorted(required)}")
    caveats: list[dict[str, str]] = []
    for row in rows:
        status = (row.get("status") or "").strip()
        notes = (row.get("notes") or "").strip()
        if status.lower() != "active" or any(marker in notes.lower() for marker in markers):
            caveats.append(
                {
                    "rollup_id": (row.get("rollup_id") or "").strip(),
                    "display_name": (row.get("display_name") or "").strip(),
                    "status": status,
                    "notes": notes,
                }
            )
    return sorted(caveats, key=lambda item: item["rollup_id"])


def render_limitations(caveats: Iterable[Mapping[str, str]]) -> str:
    """Render registry caveats as a deterministic generated Markdown section."""

    items = sorted(caveats, key=lambda item: str(item.get("rollup_id", "")))
    lines = [
        "## Registry-derived coverage limitations",
        "",
        "This section is generated from `registry/rollup_registry_v1.csv`; edit the registry, not this text.",
        "",
    ]
    if not items:
        lines.append("- No explicit registry coverage caveats were recorded.")
    else:
        for item in items:
            rollup_id = str(item.get("rollup_id", "")).strip()
            display_name = str(item.get("display_name", "")).strip() or rollup_id
            status = str(item.get("status", "")).strip() or "unspecified"
            notes = " ".join(str(item.get("notes", "")).split())
            lines.append(f"- **{display_name} (`{rollup_id}`; status: {status}):** {notes}")
    return "\n".join(lines) + "\n"


def generate_limitations(
    registry_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, str]]:
    """Write generated limitations and return the caveat records used."""

    caveats = registry_caveats(registry_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_limitations(caveats), encoding="utf-8")
    return caveats


__all__ = [
    "generate_limitations",
    "headline_uncertainty_artifact",
    "registry_caveats",
    "render_limitations",
]
