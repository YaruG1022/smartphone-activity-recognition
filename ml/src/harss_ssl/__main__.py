"""Run the masked-reconstruction comparison without opening the official test data."""

import argparse
import json
from pathlib import Path

from .experiment import run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--data-dir", type=Path, default=Path("ml/data/prepared/uci-har-v1"))
    parser.add_argument("--config", type=Path, default=Path("ml/configs/ssl-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("ml/outputs/ssl-v1"))
    parser.add_argument("--workers", type=int, default=1, help="Independent seed processes")
    args = parser.parse_args(argv)
    result = run_comparison(args.data, args.config, args.output, workers=args.workers)
    print(json.dumps({
        "status": result["status"],
        "aggregate": result["aggregate"],
        "test_scores_computed": result["test_scores_computed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
