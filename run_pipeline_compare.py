"""Repeat the same known-mechanism simulator across seeds; no method discovery."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from assembly_sim.cli import config_from_args, make_parser
from pipeline_core import run_pipeline, write_json


def main():
    parser = make_parser("Repeat simulation/diagnosis across independent seeds")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        base = config_from_args(args)
        configs = [replace(base, dataset_name=f"{base.dataset_name}_{i+1:02d}", seed=base.seed+i) for i in range(args.repeats)]
        for config in configs:
            if (Path(config.data_root)/config.dataset_name).exists() or (Path(config.graph_root)/config.dataset_name).exists():
                raise FileExistsError(f"Existing batch output: {config.dataset_name}; choose a fresh dataset name")
        runs = [run_pipeline(config, progress=lambda line: print(line, flush=True)) for config in configs]
        result=dict(repeats=args.repeats,total_trajectories=sum(r["n_total"] for r in runs),
                    runs=[dict(dataset_dir=r["dataset_dir"],splits=r["splits"]) for r in runs])
        output = Path(base.graph_root) / f"{base.dataset_name}_batch_summary.json"
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
