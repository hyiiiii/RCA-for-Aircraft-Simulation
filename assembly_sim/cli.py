"""Shared CLI configuration with the reference project's familiar entry names."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from pipeline_core import TrialConfig

ROOT = Path(__file__).resolve().parents[1]


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="JSON configuration; command-line values override it")
    parser.add_argument("--dataset-name")
    for name in ["n-total", "seed"]:
        parser.add_argument(f"--{name}", type=int)
    for name in ["test-fraction", "process-sigma", "measurement-sigma", "fault-fraction", "fault-magnitude"]:
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--fault-stage", choices=["random", "1", "2", "3", "4"])
    parser.add_argument("--data-root")
    parser.add_argument("--graph-root")
    parser.add_argument("--forward-settings", type=json.loads, help="Initial random pose bounds and process rotation sigma")
    parser.add_argument("--fault-part", choices=["random", "A", "B", "C", "ABC"])
    parser.add_argument("--cad-geometry", type=json.loads, help="FreeCAD dimensions as a JSON object (mm)")
    parser.add_argument("--cad-motion", type=json.loads, help="Fixed interface gaps as a JSON object")
    return parser


def config_from_args(args: argparse.Namespace) -> TrialConfig:
    values = {}
    if args.config is not None:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Config JSON must be an object")
        unknown = set(values) - {field.name for field in fields(TrialConfig)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for field in fields(TrialConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            if field.name == "fault_stage":
                value = None if value == "random" else int(value)
            elif field.name == "fault_part":
                value = None if value.lower() == "random" else value.upper()
            values[field.name] = value
    for key, default in [("data_root", "data"), ("graph_root", "graph")]:
        path = Path(values.get(key, default)).expanduser()
        values[key] = str(path if path.is_absolute() else ROOT / path)
    return TrialConfig(**values)
