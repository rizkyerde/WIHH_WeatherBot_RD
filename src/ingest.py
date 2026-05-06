"""
File scanner & ingestion logic.
- Scan ./raw_data/ for WIHH_*.txt files
- Detect changes via MD5 hash
- Reparse only changed/new files
- UPSERT to metar_observations
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.db import (
    delete_observations_by_source,
    get_conn,
    get_file_hash_db,
    upsert_file_meta,
    upsert_observations,
)
from src.parser import parse_ogimet_text

log = logging.getLogger(__name__)

RAW_DATA_DIR = Path("raw_data")
FILENAME_PATTERN = re.compile(r"^WIHH_(\d{4})_(\d{2})\.txt$", re.IGNORECASE)


def compute_file_hash(filepath: Path) -> str:
    """MD5 hash of file content."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_raw_data(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """List all valid WIHH_YYYY_MM.txt files, sorted chronologically."""
    if not raw_dir.exists():
        log.warning(f"Raw data directory does not exist: {raw_dir}")
        return []

    files = []
    for p in raw_dir.iterdir():
        if p.is_file() and FILENAME_PATTERN.match(p.name):
            files.append(p)

    files.sort(key=lambda p: p.name)
    return files


def ingest_file(filepath: Path, force: bool = False) -> tuple[bool, int]:
    """
    Ingest single file ke database.
    Skip jika hash sama dengan yang tersimpan (kecuali force=True).

    Returns:
        (was_processed, n_obs_inserted)
    """
    filename = filepath.name
    current_hash = compute_file_hash(filepath)
    file_size = filepath.stat().st_size

    with get_conn() as conn:
        stored_hash = get_file_hash_db(conn, filename)

        if stored_hash == current_hash and not force:
            log.info(f"⏭️  {filename} unchanged, skipping")
            return False, 0

        # Read file (try utf-8, fall back to latin-1 for Windows)
        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning(f"UTF-8 decode failed for {filename}, trying latin-1")
            text = filepath.read_text(encoding="latin-1")

        # Parse
        observations = parse_ogimet_text(text, source_file=filename)

        # Delete old observations from this source (for clean reparse)
        if stored_hash is not None:
            n_deleted = delete_observations_by_source(conn, filename)
            log.info(f"🗑️  Deleted {n_deleted} old obs from {filename}")

        # Insert new
        n_inserted = upsert_observations(conn, observations)

        # Update file metadata
        upsert_file_meta(
            conn,
            filename=filename,
            file_hash=current_hash,
            file_size=file_size,
            n_obs_parsed=n_inserted,
            parsed_at=datetime.now(timezone.utc).isoformat(),
        )

        log.info(f"✅ {filename}: {n_inserted} obs ingested")
        return True, n_inserted


def ingest_all(
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, int]:
    """
    Scan raw_dir dan ingest semua file (atau hanya yang berubah).

    Returns dict dengan summary stats.
    """
    files = scan_raw_data(raw_dir)
    log.info(f"Found {len(files)} files in {raw_dir}")

    stats = {
        "files_total": len(files),
        "files_processed": 0,
        "files_skipped": 0,
        "obs_inserted": 0,
    }

    for filepath in files:
        was_processed, n_obs = ingest_file(filepath, force=force)
        if was_processed:
            stats["files_processed"] += 1
            stats["obs_inserted"] += n_obs
        else:
            stats["files_skipped"] += 1

    return stats