#!/usr/bin/env python3
"""Shared TOML configuration helpers for EchoAvatar runtime entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "echoavatar.toml"


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section [{name}] must be a table")
    return value


def get_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_float(config: dict[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))


def get_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(config.get(key, default))


def get_str(config: dict[str, Any], key: str, default: str) -> str:
    return str(config.get(key, default))


def get_float_tuple(
    config: dict[str, Any],
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-value array")
    return float(value[0]), float(value[1]), float(value[2])


def resolve_repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT_DIR / value
