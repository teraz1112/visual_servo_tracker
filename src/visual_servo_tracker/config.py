from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_root": "../data/samples",
        "outputs_root": "../outputs",
    },
    "experiment": {
        "version": "circle_red_green",
    },
    "jacobian": {
        "range": 2,
    },
    "evaluation": {
        "radius": 2,
    },
    "optimization": {
        "max_offset": 2,
        "iterations": 200,
        "learning_rate": 0.01,
    },
    "tracking": {
        "iterations_per_frame": 10,
        "video_path": "",
        "target_image": "",
        "jacobian_path": "",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(config_file: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((config_file.parent / candidate).resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".json":
        user_config = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        user_config = loaded if isinstance(loaded, dict) else {}

    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    merged["paths"]["data_root"] = _resolve_path(path, merged["paths"]["data_root"])
    merged["paths"]["outputs_root"] = _resolve_path(path, merged["paths"]["outputs_root"])
    tracking = merged.get("tracking", {})
    for key in ("video_path", "target_image", "jacobian_path"):
        value = tracking.get(key)
        if isinstance(value, str) and value.strip():
            tracking[key] = _resolve_path(path, value)
    return merged


def ensure_output_layout(outputs_root: str | Path) -> None:
    root = Path(outputs_root)
    for name in ("jacobian", "graph", "optimized", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)
