"""
Open-Meteo API fetcher untuk forecast & historical data.

Dua mode:
1. fetch_archive(): historical reanalysis (ERA5) — untuk training
2. fetch_forecast(): real-time forecast — untuk prediksi
3. fetch_ensemble(): ensemble forecast (multiple members) — untuk uncertainty
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
import requests_cache
from retry_requests import retry

log = logging.getLogger(__name__)

# Jakarta Halim Perdanakusuma coordinates
WIHH_LAT = -6.2667
WIHH_LON = 106.8907
WIB = timezone(timedelta(hours=7))

# API endpoints
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Variables yang kita ambil
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "rain",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cape",
]

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "sunshine_duration",
]

# Mapping API variable name → DB column name
VAR_TO_DB = {
    "temperature_2m": "temp_c",
    "dewpoint_2m": "dewpoint_c",
    "relative_humidity_2m": "rh_pct",
    "precipitation": "precip_mm",
    "cloud_cover": "cloud_cover_pct",
    "cloud_cover_low": "cloud_low_pct",
    "cloud_cover_mid": "cloud_mid_pct",
    "cloud_cover_high": "cloud_high_pct",
    "shortwave_radiation": "shortwave_wm2",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_direction_10m": "wind_dir_deg",
    "surface_pressure": "pressure_hpa",
    "cape": "cape_jkg",
}


def _make_session() -> requests.Session:
    """HTTP session dengan caching + retry."""
    cached = requests_cache.CachedSession(
        ".cache_openmeteo",
        expire_after=3600,  # 1 hour cache
    )
    session = retry(cached, retries=5, backoff_factor=0.5)
    return session


# ============================================================
# 1. ARCHIVE FETCHER (ERA5 reanalysis)
# ============================================================

def fetch_archive(
    start_date: str,           # 'YYYY-MM-DD'
    end_date: str,             # 'YYYY-MM-DD'
    lat: float = WIHH_LAT,
    lon: float = WIHH_LON,
) -> pd.DataFrame:
    """
    Fetch historical hourly weather dari Open-Meteo Archive (ERA5).
    
    Returns DataFrame dengan kolom:
        valid_time_utc, valid_time_wib, valid_date_wib, 
        plus semua variables (temp_c, dewpoint_c, ...)
    
    Note: ERA5 = reanalysis, bukan forecast. Ini "ground truth" 
    historical, treated as proxy untuk "perfect forecast".
    """
    session = _make_session()
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",  # ambil dalam UTC, convert sendiri
    }
    
    log.info(f"Fetching archive: {start_date} to {end_date}")
    response = session.get(ARCHIVE_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    if "hourly" not in data:
        raise ValueError(f"No hourly data in response: {data}")
    
    df = pd.DataFrame(data["hourly"])
    df = _process_hourly_response(df)
    
    # Set metadata: archive = "perfect forecast", lead time tidak relevan
    df["model"] = "era5_archive"
    df["member"] = 0
    df["forecast_run_time"] = df["valid_time_utc"]  # treat as nowcast
    df["lead_hours"] = 0
    
    log.info(f"Fetched {len(df)} hourly records from archive")
    return df


# ============================================================
# 2. FORECAST FETCHER (real-time, future)
# ============================================================

def fetch_forecast(
    models: list[str] = None,
    forecast_days: int = 7,
    past_days: int = 0,
    lat: float = WIHH_LAT,
    lon: float = WIHH_LON,
) -> pd.DataFrame:
    """
    Fetch real-time forecast dari Open-Meteo.
    
    Args:
        models: list of model names. Default: GFS + ECMWF + ICON + JMA
        forecast_days: berapa hari ke depan (max 16)
        past_days: berapa hari ke belakang (untuk akses model run sebelumnya)
    """
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "jma_seamless"]
    
    session = _make_session()
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": ",".join(models),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "UTC",
    }
    
    log.info(f"Fetching forecast: models={models}, days={forecast_days}")
    response = session.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    # When multiple models, response keys are like "temperature_2m_gfs_seamless"
    df_list = []
    for model in models:
        df_model = _extract_model_data(data["hourly"], model)
        df_model["model"] = model
        df_model["member"] = 0  # deterministic, tidak ada ensemble member
        df_list.append(df_model)
    
    df = pd.concat(df_list, ignore_index=True)
    df = _process_hourly_response(df)
    
    # Forecast run time: gunakan generation time dari API response
    run_time = data.get("generationtime_ms")  # actually gen time, not run time
    # Better: gunakan current UTC time as proxy run time
    # (Open-Meteo tidak expose model run time langsung di API)
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df["forecast_run_time"] = now_utc.isoformat()
    
    # Compute lead hours
    df["lead_hours"] = (
        (pd.to_datetime(df["valid_time_utc"]) - now_utc).dt.total_seconds() / 3600
    ).astype(int)
    
    log.info(f"Fetched {len(df)} forecast records across {len(models)} models")
    return df


# ============================================================
# 3. ENSEMBLE FETCHER (multiple members for uncertainty)
# ============================================================

def fetch_ensemble(
    models: list[str] = None,
    forecast_days: int = 7,
    lat: float = WIHH_LAT,
    lon: float = WIHH_LON,
) -> pd.DataFrame:
    """
    Fetch ensemble forecast dengan multiple members.
    
    Models tersedia di ensemble API:
    - gfs_seamless (31 members)
    - ecmwf_ifs025 (50 members)
    - icon_seamless (40 members)  
    - gem_global (21 members)
    """
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
    
    session = _make_session()
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": ",".join(models),
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    
    log.info(f"Fetching ensemble: models={models}")
    response = session.get(ENSEMBLE_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    # Ensemble response: keys seperti "temperature_2m_member01", "temperature_2m_member02", ...
    # untuk single model. Kalau multi-model: "temperature_2m_gfs_seamless_member01", dst.
    
    df_list = []
    hourly = data["hourly"]
    
    for model in models:
        # Detect members untuk model ini
        members = _detect_ensemble_members(hourly, model, len(models) > 1)
        log.info(f"  {model}: {len(members)} members")
        
        for member_idx in members:
            df_member = _extract_ensemble_member(
                hourly, model, member_idx, multi_model=len(models) > 1
            )
            df_member["model"] = model
            df_member["member"] = member_idx
            df_list.append(df_member)
    
    df = pd.concat(df_list, ignore_index=True)
    df = _process_hourly_response(df)
    
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    df["forecast_run_time"] = now_utc.isoformat()
    df["lead_hours"] = (
        (pd.to_datetime(df["valid_time_utc"]) - now_utc).dt.total_seconds() / 3600
    ).astype(int)
    
    log.info(f"Fetched {len(df)} ensemble records")
    return df


# ============================================================
# Helper functions
# ============================================================

def _process_hourly_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw hourly response:
    - Rename 'time' → 'valid_time_utc'
    - Add valid_time_wib, valid_date_wib
    - Rename API variable names → DB column names
    """
    if "time" in df.columns:
        df = df.rename(columns={"time": "valid_time_utc"})
    
    # Pastikan format ISO dengan timezone
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    df["valid_time_wib"] = df["valid_time_utc"].dt.tz_convert(WIB)
    df["valid_date_wib"] = df["valid_time_wib"].dt.date.astype(str)
    
    # Convert kembali ke ISO string untuk DB
    df["valid_time_utc"] = df["valid_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df["valid_time_wib"] = df["valid_time_wib"].dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    
    # Rename variables
    df = df.rename(columns=VAR_TO_DB)
    
    return df


def _extract_model_data(hourly: dict, model: str) -> pd.DataFrame:
    """
    Extract single-model data dari multi-model response.
    Keys di response: e.g. 'temperature_2m_gfs_seamless'
    """
    df_data = {"time": hourly["time"]}
    for var in HOURLY_VARIABLES:
        key = f"{var}_{model}"
        if key in hourly:
            df_data[var] = hourly[key]
        elif var in hourly:
            # Single model case: tidak ada suffix
            df_data[var] = hourly[var]
        else:
            df_data[var] = [None] * len(hourly["time"])
    return pd.DataFrame(df_data)


def _detect_ensemble_members(
    hourly: dict, model: str, multi_model: bool
) -> list[int]:
    """Detect berapa banyak ensemble members tersedia."""
    members = set()
    base_var = "temperature_2m"
    
    for key in hourly.keys():
        # Pattern: temperature_2m[_model]_memberNN
        if not key.startswith(base_var):
            continue
        
        # Extract member number
        if "_member" in key:
            try:
                member_num = int(key.split("_member")[-1])
                members.add(member_num)
            except ValueError:
                continue
        elif key == base_var or key == f"{base_var}_{model}":
            members.add(0)  # control/deterministic
    
    return sorted(members)


def _extract_ensemble_member(
    hourly: dict, model: str, member_idx: int, multi_model: bool
) -> pd.DataFrame:
    """Extract single member data dari ensemble response."""
    df_data = {"time": hourly["time"]}
    
    suffix_member = f"_member{member_idx:02d}" if member_idx > 0 else ""
    suffix_model = f"_{model}" if multi_model else ""
    
    for var in HOURLY_VARIABLES:
        # Try various key formats
        candidates = [
            f"{var}{suffix_model}{suffix_member}",
            f"{var}{suffix_member}",
            f"{var}{suffix_model}",
            var,
        ]
        for key in candidates:
            if key in hourly:
                df_data[var] = hourly[key]
                break
        else:
            df_data[var] = [None] * len(hourly["time"])
    
    return pd.DataFrame(df_data)