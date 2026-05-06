"""
SQLite database setup, schema, dan helper utilities.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH = Path("data/wihh.db")

SCHEMA_SQL = """
-- =========================================================
-- Tabel 1: Raw METAR observations (semua baris, granular)
-- =========================================================
CREATE TABLE IF NOT EXISTS metar_observations (
    time_utc            TEXT PRIMARY KEY,
    time_wib            TEXT NOT NULL,
    date_wib            TEXT NOT NULL,
    station             TEXT NOT NULL,
    report_type         TEXT,
    is_corrected        INTEGER DEFAULT 0,

    -- Temperature
    temp_c              INTEGER,
    dewpoint_c          INTEGER,

    -- Wind
    wind_dir_deg        INTEGER,
    wind_dir_vrb        INTEGER DEFAULT 0,
    wind_speed_kt       INTEGER,
    wind_gust_kt        INTEGER,

    -- Visibility
    visibility_m        INTEGER,
    cavok               INTEGER DEFAULT 0,

    -- Weather phenomena
    wx_intensity        TEXT,
    wx_descriptor       TEXT,
    wx_phenomena        TEXT,
    has_rain            INTEGER DEFAULT 0,
    has_ts              INTEGER DEFAULT 0,
    has_fog             INTEGER DEFAULT 0,
    has_haze            INTEGER DEFAULT 0,

    -- Clouds
    cloud_layer_1       TEXT,
    cloud_layer_2       TEXT,
    cloud_layer_3       TEXT,
    cloud_base_min_ft   INTEGER,
    has_cb              INTEGER DEFAULT 0,
    has_tcu             INTEGER DEFAULT 0,

    -- Pressure
    qnh_hpa             INTEGER,

    -- Trend
    trend               TEXT,

    -- Metadata
    raw_text            TEXT NOT NULL,
    source_file         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_date_wib ON metar_observations(date_wib);
CREATE INDEX IF NOT EXISTS idx_obs_station ON metar_observations(station);
CREATE INDEX IF NOT EXISTS idx_obs_source ON metar_observations(source_file);

-- =========================================================
-- Tabel 2: Daily summary (1 baris per hari WIB, derived)
-- =========================================================
CREATE TABLE IF NOT EXISTS daily_summary (
    date_wib            TEXT PRIMARY KEY,
    station             TEXT NOT NULL,

    -- Temperature
    tmax                INTEGER,
    tmax_time_wib       TEXT,
    tmax_hour_wib       INTEGER,
    tmin                INTEGER,
    tmin_time_wib       TEXT,
    temp_range          INTEGER,

    -- Humidity proxy
    dew_mean            REAL,
    dew_max             INTEGER,
    td_spread_at_tmax   INTEGER,

    -- Precipitation
    hours_rain          REAL,
    hours_heavy_rain    REAL,
    hours_ts            REAL,
    first_rain_wib      TEXT,
    first_rain_hour     REAL,
    last_rain_wib       TEXT,

    -- Cloud
    hours_cb            REAL,
    cloud_base_min_ft   INTEGER,

    -- Visibility
    visibility_min_m    INTEGER,
    hours_fog           REAL,
    hours_haze          REAL,

    -- Wind
    wind_speed_max_kt   INTEGER,
    wind_dir_at_tmax    INTEGER,

    -- Pressure
    qnh_min             INTEGER,
    qnh_max             INTEGER,
    qnh_at_tmax         INTEGER,

    -- Data quality
    n_obs               INTEGER,
    n_speci             INTEGER,
    complete            INTEGER,

    last_updated        TEXT
);

CREATE INDEX IF NOT EXISTS idx_summary_month
    ON daily_summary(substr(date_wib, 6, 2));

-- =========================================================
-- Tabel 3: File metadata (untuk change detection)
-- =========================================================
CREATE TABLE IF NOT EXISTS _file_meta (
    filename            TEXT PRIMARY KEY,
    file_hash           TEXT NOT NULL,
    file_size           INTEGER,
    n_obs_parsed        INTEGER,
    parsed_at           TEXT NOT NULL
);

-- =========================================================
-- Tabel 4: Forecast hourly
-- =========================================================
CREATE TABLE IF NOT EXISTS forecast_hourly (
    forecast_run_time   TEXT NOT NULL,
    valid_time_utc      TEXT NOT NULL,
    valid_time_wib      TEXT NOT NULL,
    valid_date_wib      TEXT NOT NULL,
    lead_hours          INTEGER NOT NULL,
    model               TEXT NOT NULL,
    member              INTEGER DEFAULT 0,
    
    temp_c              REAL,
    dewpoint_c          REAL,
    rh_pct              REAL,
    precip_mm           REAL,
    cloud_cover_pct     REAL,
    cloud_low_pct       REAL,
    cloud_mid_pct       REAL,
    cloud_high_pct      REAL,
    shortwave_wm2       REAL,
    wind_speed_kmh      REAL,
    wind_dir_deg        REAL,
    pressure_hpa        REAL,
    cape_jkg            REAL,
    
    PRIMARY KEY (forecast_run_time, valid_time_utc, model, member)
);

CREATE INDEX IF NOT EXISTS idx_fc_valid_date ON forecast_hourly(valid_date_wib);
CREATE INDEX IF NOT EXISTS idx_fc_model ON forecast_hourly(model);

-- =========================================================
-- Tabel 5: Forecast daily aggregates
-- =========================================================
CREATE TABLE IF NOT EXISTS forecast_daily (
    forecast_run_time       TEXT NOT NULL,
    valid_date_wib          TEXT NOT NULL,
    lead_days               INTEGER NOT NULL,
    model                   TEXT NOT NULL,
    member                  INTEGER DEFAULT 0,
    
    tmax_c                  REAL,
    tmin_c                  REAL,
    precip_sum_mm           REAL,
    shortwave_sum_mj        REAL,
    sunshine_hours          REAL,
    
    tmax_hour_wib           INTEGER,
    cloud_mean_morning      REAL,
    cloud_mean_afternoon    REAL,
    precip_morning_mm       REAL,
    precip_afternoon_mm     REAL,
    first_precip_hour       REAL,
    
    PRIMARY KEY (forecast_run_time, valid_date_wib, model, member)
);

CREATE INDEX IF NOT EXISTS idx_fcd_date ON forecast_daily(valid_date_wib);
CREATE INDEX IF NOT EXISTS idx_fcd_lead ON forecast_daily(lead_days);

-- =========================================================
-- Tabel 6: Forecast fetch log
-- =========================================================
CREATE TABLE IF NOT EXISTS forecast_fetch_log (
    fetch_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at          TEXT NOT NULL,
    api_type            TEXT NOT NULL,
    start_date          TEXT,
    end_date            TEXT,
    models              TEXT,
    n_records           INTEGER,
    success             INTEGER,
    error_msg           TEXT
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize database with schema. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager for database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Performance pragmas
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_observations(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> int:
    """Bulk UPSERT ke metar_observations. Return number of rows affected."""
    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ",".join(f":{c}" for c in columns)
    col_list = ",".join(columns)

    sql = f"""
        INSERT INTO metar_observations ({col_list})
        VALUES ({placeholders})
        ON CONFLICT(time_utc) DO UPDATE SET
            {",".join(f"{c}=excluded.{c}" for c in columns if c != "time_utc")}
    """
    conn.executemany(sql, rows)
    return len(rows)


def upsert_daily_summary(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> int:
    """Bulk UPSERT ke daily_summary."""
    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ",".join(f":{c}" for c in columns)
    col_list = ",".join(columns)

    sql = f"""
        INSERT INTO daily_summary ({col_list})
        VALUES ({placeholders})
        ON CONFLICT(date_wib) DO UPDATE SET
            {",".join(f"{c}=excluded.{c}" for c in columns if c != "date_wib")}
    """
    conn.executemany(sql, rows)
    return len(rows)


def upsert_file_meta(
    conn: sqlite3.Connection,
    filename: str,
    file_hash: str,
    file_size: int,
    n_obs_parsed: int,
    parsed_at: str,
) -> None:
    """Track which files have been processed."""
    conn.execute(
        """
        INSERT INTO _file_meta (filename, file_hash, file_size, n_obs_parsed, parsed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            file_hash = excluded.file_hash,
            file_size = excluded.file_size,
            n_obs_parsed = excluded.n_obs_parsed,
            parsed_at = excluded.parsed_at
        """,
        (filename, file_hash, file_size, n_obs_parsed, parsed_at),
    )


def get_file_hash_db(conn: sqlite3.Connection, filename: str) -> str | None:
    """Get stored hash for a file, or None if not yet processed."""
    row = conn.execute(
        "SELECT file_hash FROM _file_meta WHERE filename = ?", (filename,)
    ).fetchone()
    return row["file_hash"] if row else None


def delete_observations_by_source(
    conn: sqlite3.Connection, source_file: str
) -> int:
    """Hapus observasi dari file tertentu (untuk reparse)."""
    cur = conn.execute(
        "DELETE FROM metar_observations WHERE source_file = ?", (source_file,)
    )
    return cur.rowcount

def upsert_forecast_hourly(
    conn: sqlite3.Connection, rows: list[dict]
) -> int:
    """Bulk UPSERT ke forecast_hourly."""
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ",".join(f":{c}" for c in columns)
    col_list = ",".join(columns)
    pk_cols = {"forecast_run_time", "valid_time_utc", "model", "member"}
    
    sql = f"""
        INSERT INTO forecast_hourly ({col_list})
        VALUES ({placeholders})
        ON CONFLICT(forecast_run_time, valid_time_utc, model, member) DO UPDATE SET
            {",".join(f"{c}=excluded.{c}" for c in columns if c not in pk_cols)}
    """
    conn.executemany(sql, rows)
    return len(rows)


def upsert_forecast_daily(
    conn: sqlite3.Connection, rows: list[dict]
) -> int:
    """Bulk UPSERT ke forecast_daily."""
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ",".join(f":{c}" for c in columns)
    col_list = ",".join(columns)
    pk_cols = {"forecast_run_time", "valid_date_wib", "model", "member"}
    
    sql = f"""
        INSERT INTO forecast_daily ({col_list})
        VALUES ({placeholders})
        ON CONFLICT(forecast_run_time, valid_date_wib, model, member) DO UPDATE SET
            {",".join(f"{c}=excluded.{c}" for c in columns if c not in pk_cols)}
    """
    conn.executemany(sql, rows)
    return len(rows)


def log_forecast_fetch(
    conn: sqlite3.Connection,
    api_type: str,
    start_date: str | None,
    end_date: str | None,
    models: str,
    n_records: int,
    success: bool,
    error_msg: str | None = None,
) -> None:
    """Log API fetch operation."""
    from datetime import datetime, timezone
    conn.execute(
        """
        INSERT INTO forecast_fetch_log 
        (fetched_at, api_type, start_date, end_date, models, n_records, success, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            api_type, start_date, end_date, models,
            n_records, int(success), error_msg,
        ),
    )