from __future__ import annotations

import argparse

from src.analysis import analyze_all, export_all
from src.clean import clean_all
from src.config import load_config
from src.fetch import fetch_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline: fetch, clean, analyse")
    parser.add_argument("--force", action="store_true", help="re-download raw data even if it already exists")
    args = parser.parse_args()
    config = load_config()
    print("[1/3] Fetching data")
    fetch_all(config, force=args.force)
    print("[2/3] Cleaning data")
    clean_all(config)
    print("[3/3] Running analysis")
    export_all(analyze_all(config), config)
    print("Done.")


if __name__ == "__main__":
    main()
