#!/usr/bin/env python3
"""Dependency-free access to the active project pack configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_pack_config(repo: Path | str = Path(".")) -> dict[str, Any]:
    path = Path(repo) / "contracts" / "pack.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"missing_pack_config:{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_pack_config_json:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pack_config_top_level_not_object")
    return payload


def pack_value(config: dict[str, Any], dotted_key: str, expected_type: type = str) -> Any:
    value: object = config
    for component in dotted_key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"pack_config_missing:{dotted_key}")
        value = value[component]
    if not isinstance(value, expected_type):
        raise ValueError(f"pack_config_type:{dotted_key}:{expected_type.__name__}")
    return value


def pack_path(repo: Path, config: dict[str, Any], dotted_key: str) -> Path:
    return repo / pack_value(config, dotted_key)


def kernel_interface(repo: Path | str = Path(".")) -> dict[str, Any]:
    path = Path(repo) / "contracts" / "kernel_interface.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_kernel_interface:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("kernel_interface_top_level_not_object")
    return payload


def manifest_schema_version(name: str, repo: Path | str = Path(".")) -> str:
    descriptor = kernel_interface(repo)
    versions = descriptor.get("manifest_schema_versions")
    if not isinstance(versions, dict) or not isinstance(versions.get(name), str):
        raise ValueError(f"kernel_interface_manifest_schema_missing:{name}")
    return str(versions[name])


def dataframe_schema_field_names(repo: Path | str, pack_key: str) -> tuple[str, ...]:
    root = Path(repo)
    pack = load_pack_config(root)
    schema_path = root / pack_value(pack, pack_key)
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_dataframe_schema:{schema_path}:{exc}") from exc
    fields = payload.get("fields") if isinstance(payload, dict) else None
    if not isinstance(fields, list):
        raise ValueError(f"dataframe_schema_fields_missing:{schema_path}")
    names = tuple(
        str(field["name"])
        for field in fields
        if isinstance(field, dict) and isinstance(field.get("name"), str)
    )
    if len(names) != len(fields) or len(set(names)) != len(names):
        raise ValueError(f"dataframe_schema_fields_invalid:{schema_path}")
    return names
