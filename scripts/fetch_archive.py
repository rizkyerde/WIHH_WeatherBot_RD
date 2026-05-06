"""
Fetch historical archive data dari Open-Meteo (ERA5).
Untuk training MOS model.

Usage:
    python -m scripts.fetch_archive --start 2022-01-01 --end 2026-05-04
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.db import (
    get_conn,
    init_db,
    log_forecast_fetch,
    upsert_forecast_hourly,
)
from src.forecast_aggregator import rebuild_forecast_daily
from src.openmeteo_fetcher import fetch_archive

log = logging.getLogger(__name__)


def chunk_date_range(
    start: date, end: date, chunk_months: int = 6
) -> list[tuple[date, date]]:
    """Split date range into chunks (Open-Meteo bisa handle setahun, tapi safer per 6 bulan)."""
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = cur + timedelta(days=chunk_months * 30)
        if chunk_end > end:
            chunk_end = end
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def fetch_and_store(start_date: str, end_date: str) -> int:
    """Fetch archive data, store ke DB."""
    df = fetch_archive(start_date=start_date, end_date=end_date)
    
    if df.empty:
        log.warning("Empty response, skipping")
        return 0
    
    # Drop rows where temp_c is None (incomplete records)
    df = df.dropna(subset=["temp_c"])
    
    # Convert to dict list, handle numpy types
    rows = df.to_dict(orient="records")
    rows = [_clean_for_sqlite(r) for r in rows]
    
    # Filter only columns yang ada di table schema
    valid_cols = {
        "forecast_run_time", "valid_time_utc", "valid_time_wib", "valid_date_wib",
        "lead_hours", "model", "member",
        "temp_c", "dewpoint_c", "rh_pct", "precip_mm",
        "cloud_cover_pct", "cloud_low_pct", "cloud_mid_pct", "cloud_high_pct",
        "shortwave_wm2", "wind_speed_kmh", "wind_dir_deg",
        "pressure_hpa", "cape_jkg",
    }
    rows = [{k: v for k, v in r.items() if k in valid_cols} for r in rows]
    
    with get_conn() as conn:
        n = upsert_forecast_hourly(conn, rows)
        log_forecast_fetch(
            conn,
            api_type="archive",
            start_date=start_date,
            end_date=end_date,
            models="era5_archive",
            n_records=n,
            success=True,
        )
    
    return n


def _clean_for_sqlite(row: dict) -> dict:
    cleaned = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            cleaned[k] = None
        elif hasattr(v, "item"):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    init_db()
    
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    chunks = chunk_date_range(start, end, args.chunk_months)
    log.info(f"Will fetch {len(chunks)} chunks from {start} to {end}")
    
    total = 0
    for i, (s, e) in enumerate(chunks, 1):
        log.info(f"--- Chunk {i}/{len(chunks)}: {s} to {e} ---")
        try:
            n = fetch_and_store(s.isoformat(), e.isoformat())
            total += n
            log.info(f"✅ Inserted {n} records (total: {total:,})")
        except Exception as ex:
            log.error(f"❌ Failed chunk {s}-{e}: {ex}")
            with get_conn() as conn:
                log_forecast_fetch(
                    conn, "archive", s.isoformat(), e.isoformat(),
                    "era5_archive", 0, False, str(ex),
                )
    
    log.info(f"\n🎉 Total records inserted: {total:,}")
    
    if not args.skip_aggregate:
        log.info("Rebuilding forecast_daily...")
        rebuild_forecast_daily()


if __name__ == "__main__":
    main()