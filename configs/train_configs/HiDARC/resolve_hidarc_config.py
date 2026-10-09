#!/usr/bin/env python3
"""Materialize a HiDARC task config with its shared collaboration profile.

Each ``<model>/<protocol>/collaboration.json`` contains the routing and role
assignment parameters shared by all of that profile's train/eval tasks.  A
task config can override any value locally; local values take precedence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object: {path}")
    return value


def collaboration_path(config_path: Path, config: dict[str, Any]) -> Path | None:
    explicit = config.get("collaboration_config")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("`collaboration_config` must be a non-empty path string")
        path = Path(explicit)
        return path if path.is_absolute() else (config_path.parent / path).resolve()

    # Canonical task files live in <profile>/{train,eval}/task*.json.
    if config_path.parent.name in {"train", "eval"}:
        candidate = config_path.parent.parent / "collaboration.json"
        if candidate.is_file():
            return candidate
    return None


def resolve(config_path: Path) -> tuple[dict[str, Any], Path | None]:
    config = load_json(config_path, "task config")
    profile_path = collaboration_path(config_path, config)
    config.pop("collaboration_config", None)
    if profile_path is None:
        return config, None

    collaboration = load_json(profile_path, "collaboration config")
    if "activation" in config or config_path.parent.name == "eval":
        activation = config.get("activation", {})
        if not isinstance(activation, dict):
            raise ValueError("`activation` must be a JSON object")
        resolved = dict(config)
        resolved["activation"] = {**collaboration, **activation}
        return resolved, profile_path

    # Training parsers consume routing keys at the top level.
    return {**collaboration, **config}, profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-collaboration-path", action="store_true")
    args = parser.parse_args()

    resolved, profile_path = resolve(args.config.resolve())
    if args.print_collaboration_path:
        print("" if profile_path is None else profile_path)
        return
    rendered = json.dumps(resolved, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
