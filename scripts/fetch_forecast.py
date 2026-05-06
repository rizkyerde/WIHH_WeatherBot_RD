"""
Fetch real-time forecast (untuk prediksi besok-besok).

Usage:
    python -m scripts.fetch_forecast                    # default models, 7 days
    python -m scripts.fetch_forecast --ensemble         # use ensemble API
"""

from __future__ import annotations

import argparse
import logging
import sys
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
from src.openmeteo_fetcher import fetch_ensemble, fetch_forecast

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble API (multiple members)")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--past-days", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    init_db()
    
    if args.ensemble:
        log.info("Using ENSEMBLE API")
        df = fetch_ensemble(models=args.models, forecast_days=args.days)
        api_type = "ensemble"
    else:
        log.info("Using FORECAST API (deterministic multi-model)")
        df = fetch_forecast(
            models=args.models,
            forecast_days=args.days,
            past_days=args.past_days,
        )
        api_type = "forecast"
    
    if df.empty:
        log.error("Empty response!")
        return 1
    
    df = df.dropna(subset=["temp_c"])
    rows = df.to_dict(orient="records")
    rows = [_clean_for_sqlite(r) for r in rows]
    
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
        models_str = ",".join(df["model"].unique())
        log_forecast_fetch(
            conn, api_type=api_type,
            start_date=None, end_date=None,
            models=models_str, n_records=n, success=True,
        )
    
    log.info(f"✅ Inserted {n} records")
    log.info("Rebuilding forecast_daily...")
    rebuild_forecast_daily()
    return 0


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


if __name__ == "__main__":
    sys.exit(main())