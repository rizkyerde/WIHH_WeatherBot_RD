"""
Main pipeline entry point.

Usage:
    python -m scripts.run_pipeline           # incremental (skip unchanged)
    python -m scripts.run_pipeline --force   # reparse all files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregator import rebuild_daily_summary
from src.db import init_db
from src.ingest import ingest_all


def main() -> int:
    parser = argparse.ArgumentParser(description="WIHH METAR ingestion pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reparse all files even if unchanged",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip daily_summary rebuild",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log = logging.getLogger("pipeline")

    # Step 1: Init DB
    log.info("=" * 60)
    log.info("STEP 1: Initialize database")
    log.info("=" * 60)
    init_db()

    # Step 2: Ingest raw files
    log.info("=" * 60)
    log.info("STEP 2: Ingest raw METAR files")
    log.info("=" * 60)
    stats = ingest_all(force=args.force)
    log.info(f"Ingestion stats: {stats}")

    # Step 3: Rebuild daily summary
    if not args.skip_aggregate:
        log.info("=" * 60)
        log.info("STEP 3: Rebuild daily_summary")
        log.info("=" * 60)
        n_days = rebuild_daily_summary()
        log.info(f"Summarized {n_days} days")

    log.info("=" * 60)
    log.info("✅ Pipeline complete!")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())