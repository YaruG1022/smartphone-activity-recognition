"""Run from the repository root after `python -m pip install -e ./ml`."""

import argparse
import json
from pathlib import Path
import sys
import urllib.error

from .prepare import prepare_public, prepare_recording, verify_prepared
from .source import download


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare versioned HARSS data; no model training")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download or verify the pinned UCI source")
    fetch.add_argument("--archive", type=Path, default=Path("ml/data/raw/uci-har-source.zip"))
    prepare = commands.add_parser("prepare", help="Prepare public subject-isolated partitions")
    prepare.add_argument("--archive", type=Path, default=Path("ml/data/raw/uci-har-source.zip"))
    prepare.add_argument("--split", type=Path, default=Path("ml/configs/subjects-v1.json"))
    prepare.add_argument("--output", type=Path, default=Path("ml/data/prepared/uci-har-v1"))
    verify = commands.add_parser("verify", help="Check hashes, partitions, and normalization replay")
    verify.add_argument("--output", type=Path, default=Path("ml/data/prepared/uci-har-v1"))
    recording = commands.add_parser("prepare-recording", help="Prepare one recorded phone session")
    recording.add_argument("--metadata", type=Path, required=True)
    recording.add_argument("--events", type=Path, required=True)
    recording.add_argument("--normalization", type=Path, required=True)
    recording.add_argument("--output", type=Path, required=True)
    recording.add_argument("--allow-final-test", action="store_true",
                           help="Enable held-out session processing after freezing the method")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = {"archive": str(download(args.archive)), "verified": True}
        elif args.command == "prepare":
            result = prepare_public(args.archive, args.split, args.output)
        elif args.command == "verify":
            result = verify_prepared(args.output)
        else:
            result = prepare_recording(args.metadata, args.events, args.normalization,
                                       args.output, args.allow_final_test)
    except (ValueError, OSError, KeyError, AssertionError, ArithmeticError,
            urllib.error.URLError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
