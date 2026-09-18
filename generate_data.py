"""Generate measured coordinate trajectories and separate evaluator-only truth."""
import sys

from assembly_sim.cli import config_from_args, make_parser
from pipeline_core import create_dataset


def main():
    parser = make_parser("Generate general CAD train/test trajectories with labels")
    try:
        directory = create_dataset(config_from_args(parser.parse_args()), progress=lambda line: print(line, flush=True))
        print(f"dataset_dir: {directory}")
    except (ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
