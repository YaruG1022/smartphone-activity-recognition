"""Command-line entry point for the validation-only supervised comparison."""

import argparse
import json
from pathlib import Path

from .experiment import run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train four supervised baselines; test scoring remains sealed")
    parser.add_argument("--data", type=Path, default=Path("ml/data/prepared/uci-har-v1"))
    parser.add_argument("--config", type=Path, default=Path("ml/configs/supervised-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("ml/outputs/supervised-v1"))
    args = parser.parse_args(argv)
    result = run_comparison(args.data, args.config, args.output)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"],
                      "selection": result["selection"], "test_scores_computed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
