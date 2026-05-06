"""
Aggregator: build daily_summary dari metar_observations.

Strategy: full rebuild (simple & fast for ~1500 days).
Group by date_wib, compute 25+ features per day.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.db import get_conn, upsert_daily_summary

log = logging.getLogger(__name__)


def load_observations_df() -> pd.DataFrame:
    """Load semua METAR observations dari DB ke DataFrame."""
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT * FROM metar_observations
            ORDER BY time_utc
            """,
            conn,
        )

    if df.empty:
        return df

    # Parse timestamps
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df["time_wib"] = pd.to_datetime(df["time_wib"])

    return df


def build_daily_summary(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily_summary DataFrame dari raw observations.
    Group by date_wib, compute fitur lengkap.
    """
    if df_obs.empty:
        log.warning("No observations to aggregate")
        return pd.DataFrame()

    summaries = []

    for date_wib, group in df_obs.groupby("date_wib"):
        summary = _build_single_day_summary(date_wib, group)
        summaries.append(summary)

    df_summary = pd.DataFrame(summaries)
    df_summary["last_updated"] = datetime.now(timezone.utc).isoformat()
    return df_summary


def _build_single_day_summary(date_wib: str, group: pd.DataFrame) -> dict:
    """
    Build summary untuk satu hari kalender WIB.

    Catatan tentang "hours_*":
    Kita hitung sebagai (count of obs with flag) / (obs per hour).
    Karena observasi rutin tiap 30 menit (2 obs/jam),
    setiap obs ber-flag = 0.5 hours.
    SPECI di-treat sama, sedikit overcounting tapi acceptable.
    """
    # Default to None untuk semua field
    s: dict = {
        "date_wib": date_wib,
        "station": group["station"].iloc[0],
    }

    # ===== Temperature =====
    temp_valid = group[group["temp_c"].notna()]
    if not temp_valid.empty:
        idx_max = temp_valid["temp_c"].idxmax()
        idx_min = temp_valid["temp_c"].idxmin()
        s["tmax"] = int(temp_valid.loc[idx_max, "temp_c"])
        s["tmax_time_wib"] = temp_valid.loc[idx_max, "time_wib"].isoformat()
        s["tmax_hour_wib"] = int(temp_valid.loc[idx_max, "time_wib"].hour)
        s["tmin"] = int(temp_valid.loc[idx_min, "temp_c"])
        s["tmin_time_wib"] = temp_valid.loc[idx_min, "time_wib"].isoformat()
        s["temp_range"] = s["tmax"] - s["tmin"]

        # Wind & pressure at tmax
        row_max = temp_valid.loc[idx_max]
        s["wind_dir_at_tmax"] = (
            int(row_max["wind_dir_deg"])
            if pd.notna(row_max["wind_dir_deg"])
            else None
        )
        s["qnh_at_tmax"] = (
            int(row_max["qnh_hpa"]) if pd.notna(row_max["qnh_hpa"]) else None
        )

        # Td spread at tmax
        if pd.notna(row_max["dewpoint_c"]):
            s["td_spread_at_tmax"] = int(row_max["temp_c"] - row_max["dewpoint_c"])
        else:
            s["td_spread_at_tmax"] = None
    else:
        for k in [
            "tmax", "tmax_time_wib", "tmax_hour_wib",
            "tmin", "tmin_time_wib", "temp_range",
            "wind_dir_at_tmax", "qnh_at_tmax", "td_spread_at_tmax",
        ]:
            s[k] = None

    # ===== Humidity =====
    dew_valid = group[group["dewpoint_c"].notna()]
    if not dew_valid.empty:
        s["dew_mean"] = float(dew_valid["dewpoint_c"].mean())
        s["dew_max"] = int(dew_valid["dewpoint_c"].max())
    else:
        s["dew_mean"] = None
        s["dew_max"] = None

    # ===== Precipitation timing =====
    # 1 obs = 0.5 hours (asumsi 30-min interval)
    HOURS_PER_OBS = 0.5
    s["hours_rain"] = float(group["has_rain"].sum() * HOURS_PER_OBS)
    s["hours_ts"] = float(group["has_ts"].sum() * HOURS_PER_OBS)

    # Heavy rain: intensity '+' AND rain
    heavy_mask = (
        group["has_rain"] == 1
    ) & group["wx_intensity"].fillna("").str.contains(r"\+", regex=True)
    s["hours_heavy_rain"] = float(heavy_mask.sum() * HOURS_PER_OBS)

    # First/last rain
    rain_obs = group[group["has_rain"] == 1].sort_values("time_wib")
    if not rain_obs.empty:
        first_rain_time = rain_obs.iloc[0]["time_wib"]
        last_rain_time = rain_obs.iloc[-1]["time_wib"]
        s["first_rain_wib"] = first_rain_time.isoformat()
        s["first_rain_hour"] = float(
            first_rain_time.hour + first_rain_time.minute / 60.0
        )
        s["last_rain_wib"] = last_rain_time.isoformat()
    else:
        s["first_rain_wib"] = None
        s["first_rain_hour"] = None
        s["last_rain_wib"] = None

    # ===== Cloud =====
    s["hours_cb"] = float(group["has_cb"].sum() * HOURS_PER_OBS)
    cb_valid = group[group["cloud_base_min_ft"].notna()]
    s["cloud_base_min_ft"] = (
        int(cb_valid["cloud_base_min_ft"].min()) if not cb_valid.empty else None
    )

    # ===== Visibility =====
    vis_valid = group[group["visibility_m"].notna()]
    s["visibility_min_m"] = (
        int(vis_valid["visibility_m"].min()) if not vis_valid.empty else None
    )
    s["hours_fog"] = float(group["has_fog"].sum() * HOURS_PER_OBS)
    s["hours_haze"] = float(group["has_haze"].sum() * HOURS_PER_OBS)

    # ===== Wind =====
    wind_valid = group[group["wind_speed_kt"].notna()]
    s["wind_speed_max_kt"] = (
        int(wind_valid["wind_speed_kt"].max()) if not wind_valid.empty else None
    )

    # ===== Pressure =====
    qnh_valid = group[group["qnh_hpa"].notna()]
    if not qnh_valid.empty:
        s["qnh_min"] = int(qnh_valid["qnh_hpa"].min())
        s["qnh_max"] = int(qnh_valid["qnh_hpa"].max())
    else:
        s["qnh_min"] = None
        s["qnh_max"] = None

    # ===== Data quality =====
    s["n_obs"] = int(len(group))
    s["n_speci"] = int((group["report_type"] == "SPECI").sum())
    s["complete"] = int(s["n_obs"] >= 40)

    return s


def rebuild_daily_summary() -> int:
    """
    Full rebuild daily_summary dari metar_observations.
    Returns number of days summarized.
    """
    log.info("Loading observations from DB...")
    df_obs = load_observations_df()

    if df_obs.empty:
        log.warning("No observations found, skipping aggregation")
        return 0

    log.info(f"Loaded {len(df_obs):,} observations")
    log.info("Building daily summary...")
    df_summary = build_daily_summary(df_obs)

    log.info(f"Built {len(df_summary)} daily summaries, writing to DB...")
    rows = df_summary.to_dict(orient="records")

    # Convert numpy types to Python native (sqlite3 picky)
    rows = [_clean_for_sqlite(r) for r in rows]

    with get_conn() as conn:
        # Truncate and rebuild (full rebuild strategy)
        conn.execute("DELETE FROM daily_summary")
        upsert_daily_summary(conn, rows)

    log.info(f"✅ Rebuilt daily_summary: {len(rows)} rows")
    return len(rows)


def _clean_for_sqlite(row: dict) -> dict:
    """Convert numpy/pandas types to Python native for sqlite3."""
    cleaned = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            cleaned[k] = None
        elif hasattr(v, "item"):  # numpy scalar
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned