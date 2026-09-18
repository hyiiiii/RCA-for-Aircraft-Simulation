"""Run the specified simulation and diagnosis; no causal discovery is performed."""
import json
import sys

from assembly_sim.cli import config_from_args, make_parser
from pipeline_core import run_pipeline


def main():
    parser = make_parser("Generate a general CAD train/test dataset")
    try:
        summary = run_pipeline(config_from_args(parser.parse_args()), progress=lambda line: print(line, flush=True))
    except (ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: summary[key] for key in ["dataset_dir", "graph_dir", "n_total", "splits"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
