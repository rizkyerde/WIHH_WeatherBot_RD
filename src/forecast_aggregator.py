"""
Aggregate forecast_hourly → forecast_daily.
Mirror dari aggregator.py tapi untuk forecast data.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.db import get_conn, upsert_forecast_daily

log = logging.getLogger(__name__)


def build_forecast_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate forecast_hourly → forecast_daily per (run_time, date, model, member).
    """
    if df_hourly.empty:
        return pd.DataFrame()
    
    # Parse hour from valid_time_wib for time-of-day features
    df = df_hourly.copy()
    df["hour_wib"] = pd.to_datetime(df["valid_time_wib"]).dt.hour
    
    summaries = []
    
    group_cols = ["forecast_run_time", "valid_date_wib", "model", "member"]
    for keys, group in df.groupby(group_cols):
        run_time, date_wib, model, member = keys
        summary = _aggregate_single_day(group, run_time, date_wib, model, member)
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def _aggregate_single_day(
    group: pd.DataFrame,
    run_time: str,
    date_wib: str,
    model: str,
    member: int,
) -> dict:
    """Aggregate single (run, date, model, member) ke daily summary."""
    
    # Lead days = (target date - run date) in days
    run_date = pd.to_datetime(run_time).date()
    target_date = pd.to_datetime(date_wib).date()
    lead_days = (target_date - run_date).days
    
    s = {
        "forecast_run_time": run_time,
        "valid_date_wib": date_wib,
        "lead_days": lead_days,
        "model": model,
        "member": int(member),
    }
    
    # Temperature
    if "temp_c" in group.columns and group["temp_c"].notna().any():
        s["tmax_c"] = float(group["temp_c"].max())
        s["tmin_c"] = float(group["temp_c"].min())
        idx_max = group["temp_c"].idxmax()
        s["tmax_hour_wib"] = int(group.loc[idx_max, "hour_wib"])
    else:
        s["tmax_c"] = None
        s["tmin_c"] = None
        s["tmax_hour_wib"] = None
    
    # Precipitation
    if "precip_mm" in group.columns:
        s["precip_sum_mm"] = float(group["precip_mm"].sum())
        morning = group[(group["hour_wib"] >= 6) & (group["hour_wib"] < 12)]
        afternoon = group[(group["hour_wib"] >= 12) & (group["hour_wib"] < 18)]
        s["precip_morning_mm"] = float(morning["precip_mm"].sum()) if not morning.empty else 0.0
        s["precip_afternoon_mm"] = float(afternoon["precip_mm"].sum()) if not afternoon.empty else 0.0
        
        # First precip > 0.1mm
        precip_hours = group[group["precip_mm"] > 0.1].sort_values("hour_wib")
        s["first_precip_hour"] = (
            float(precip_hours.iloc[0]["hour_wib"]) if not precip_hours.empty else None
        )
    else:
        s["precip_sum_mm"] = None
        s["precip_morning_mm"] = None
        s["precip_afternoon_mm"] = None
        s["first_precip_hour"] = None
    
    # Shortwave radiation (Tmax driver!)
    if "shortwave_wm2" in group.columns and group["shortwave_wm2"].notna().any():
        # Sum dalam MJ/m² (W/m² × 3600s / 1e6)
        s["shortwave_sum_mj"] = float(group["shortwave_wm2"].sum() * 3600 / 1e6)
        # Sunshine: hours with radiation > 120 W/m² (rule of thumb)
        s["sunshine_hours"] = float((group["shortwave_wm2"] > 120).sum())
    else:
        s["shortwave_sum_mj"] = None
        s["sunshine_hours"] = None
    
    # Cloud cover by time of day
    if "cloud_cover_pct" in group.columns and group["cloud_cover_pct"].notna().any():
        morning = group[(group["hour_wib"] >= 6) & (group["hour_wib"] < 12)]
        afternoon = group[(group["hour_wib"] >= 12) & (group["hour_wib"] < 18)]
        s["cloud_mean_morning"] = (
            float(morning["cloud_cover_pct"].mean()) if not morning.empty else None
        )
        s["cloud_mean_afternoon"] = (
            float(afternoon["cloud_cover_pct"].mean()) if not afternoon.empty else None
        )
    else:
        s["cloud_mean_morning"] = None
        s["cloud_mean_afternoon"] = None
    
    return s


def rebuild_forecast_daily() -> int:
    """Full rebuild forecast_daily from forecast_hourly."""
    log.info("Loading forecast_hourly from DB...")
    
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM forecast_hourly", conn)
    
    if df.empty:
        log.warning("No forecast_hourly data to aggregate")
        return 0
    
    log.info(f"Loaded {len(df):,} hourly forecast records")
    df_daily = build_forecast_daily(df)
    
    log.info(f"Aggregated to {len(df_daily)} daily records")
    
    rows = df_daily.to_dict(orient="records")
    rows = [_clean_for_sqlite(r) for r in rows]
    
    with get_conn() as conn:
        conn.execute("DELETE FROM forecast_daily")
        upsert_forecast_daily(conn, rows)
    
    log.info(f"✅ Rebuilt forecast_daily: {len(rows)} rows")
    return len(rows)


def _clean_for_sqlite(row: dict) -> dict:
    """Convert numpy/pandas types to Python native."""
    cleaned = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            cleaned[k] = None
        elif hasattr(v, "item"):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned