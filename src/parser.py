"""
METAR parser: Ogimet text format → list of dicts ready for DB insert.

Strategy:
- Use `python-metar` library sebagai engine utama (battle-tested)
- Custom regex untuk extract Ogimet timestamp prefix (YYYYMMDDHHMM)
- Custom logic untuk derive flag fields (has_rain, has_cb, etc.)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from metar import Metar

log = logging.getLogger(__name__)

WIB = timezone(timedelta(hours=7))

# Format dari Ogimet: "202605040430 METAR WIHH 040430Z ... ="
OGIMET_LINE_RE = re.compile(
    r"^(?P<ts>\d{12})\s+(?P<rest>(?:METAR|SPECI)\s+.+?)=?\s*$"
)

# Untuk detect COR (corrected) reports
COR_RE = re.compile(r"\b(METAR|SPECI)\s+COR\b")

# Weather phenomena codes (subset relevan untuk Jakarta)
RAIN_CODES = {"RA", "DZ", "SHRA", "TSRA"}
TS_CODES = {"TS", "TSRA", "TSGR", "TSGS"}
FOG_CODES = {"FG", "BCFG", "MIFG"}
HAZE_CODES = {"HZ", "BR"}  # BR = mist, HZ = haze; sering muncul di Jakarta


def parse_ogimet_text(
    text: str,
    source_file: str,
) -> list[dict[str, Any]]:
    """
    Parse full Ogimet output text → list of observation dicts.

    Args:
        text: Raw text dari file Ogimet (.txt)
        source_file: Filename untuk tracking (e.g. 'WIHH_2026_05.txt')

    Returns:
        List of dicts, satu per METAR/SPECI yang berhasil di-parse.
    """
    observations = []
    lines_skipped = 0
    lines_errored = 0

    for line in text.splitlines():
        line = line.strip()
        # Skip comments, empty lines, TAF section, etc.
        if not line or line.startswith("#"):
            continue
        if line.startswith(("TAF ", "AMD ")) or " TAF " in line[:20]:
            continue

        m = OGIMET_LINE_RE.match(line)
        if not m:
            lines_skipped += 1
            continue

        try:
            obs = _parse_single_metar(
                ogimet_ts=m.group("ts"),
                metar_body=m.group("rest"),
                full_line=line,
                source_file=source_file,
            )
            if obs is not None:
                observations.append(obs)
        except Exception as e:
            lines_errored += 1
            log.debug(f"Parse error for line: {line[:80]}... | {e}")
            continue

    log.info(
        f"Parsed {source_file}: {len(observations)} obs, "
        f"{lines_skipped} skipped, {lines_errored} errored"
    )
    return observations


def _parse_single_metar(
    ogimet_ts: str,
    metar_body: str,
    full_line: str,
    source_file: str,
) -> dict[str, Any] | None:
    """
    Parse single METAR/SPECI line ke dict.
    Return None jika tidak parseable atau missing critical fields.
    """
    # Parse timestamp dari Ogimet prefix (YYYYMMDDHHMM in UTC)
    try:
        t_utc = datetime.strptime(ogimet_ts, "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None

    t_wib = t_utc.astimezone(WIB)

    # Detect report type & corrected
    is_corrected = bool(COR_RE.search(metar_body))
    report_type = "SPECI" if metar_body.startswith("SPECI") else "METAR"

    # Strip Ogimet artifacts: "METAR " or "SPECI " prefix → biar python-metar happy
    # python-metar expect format like "WIHH 041330Z 13004KT ..."
    metar_clean = re.sub(
        r"^(METAR|SPECI)(\s+COR)?\s+", "", metar_body
    ).rstrip("=").strip()

    # python-metar butuh month & year context untuk parse "041330Z" dengan benar
    try:
        m_obj = Metar.Metar(
            metar_clean,
            month=t_utc.month,
            year=t_utc.year,
            strict=False,
        )
    except Metar.ParserError as e:
        log.debug(f"python-metar failed: {metar_clean[:60]}... | {e}")
        return None

    # ===== Extract fields =====
    station = m_obj.station_id or "UNKN"

    # Temperature & dewpoint (integer °C)
    temp_c = int(round(m_obj.temp.value("C"))) if m_obj.temp else None
    dew_c = int(round(m_obj.dewpt.value("C"))) if m_obj.dewpt else None

    # Skip records with no temperature (tidak berguna untuk modeling Tmax)
    if temp_c is None:
        return None

    # Sanity check
    if not (-10 <= temp_c <= 50):
        return None

    # Wind
    wind_dir_deg = None
    wind_dir_vrb = 0
    wind_speed_kt = None
    wind_gust_kt = None
    if m_obj.wind_dir is not None:
        if m_obj.wind_dir.compass() == "VRB":
            wind_dir_vrb = 1
        else:
            wind_dir_deg = int(m_obj.wind_dir.value())
    if m_obj.wind_speed is not None:
        wind_speed_kt = int(round(m_obj.wind_speed.value("KT")))
    if m_obj.wind_gust is not None:
        wind_gust_kt = int(round(m_obj.wind_gust.value("KT")))

    # Visibility (meters)
    visibility_m = None
    cavok = 0
    if "CAVOK" in metar_clean:
        cavok = 1
        visibility_m = 9999
    elif m_obj.vis is not None:
        try:
            vis_m = m_obj.vis.value("M")
            visibility_m = int(round(vis_m))
        except Exception:
            pass

    # Weather phenomena
    wx_intensity, wx_descriptor, wx_phenomena = _extract_weather(m_obj)
    has_rain = int(any(c in (wx_phenomena or "") for c in ["RA", "DZ"]))
    has_ts = int("TS" in (wx_descriptor or "") or "TS" in (wx_phenomena or ""))
    has_fog = int("FG" in (wx_phenomena or ""))
    has_haze = int("HZ" in (wx_phenomena or "") or "BR" in (wx_phenomena or ""))

    # Cloud layers
    cloud_layers_str, cloud_base_min_ft, has_cb, has_tcu = _extract_clouds(m_obj)
    cloud_layer_1 = cloud_layers_str[0] if len(cloud_layers_str) > 0 else None
    cloud_layer_2 = cloud_layers_str[1] if len(cloud_layers_str) > 1 else None
    cloud_layer_3 = cloud_layers_str[2] if len(cloud_layers_str) > 2 else None

    # Pressure (QNH in hPa)
    qnh_hpa = None
    if m_obj.press is not None:
        try:
            qnh_hpa = int(round(m_obj.press.value("HPA")))
        except Exception:
            pass

    # Trend
    trend = _extract_trend(metar_clean)

    return {
        "time_utc": t_utc.isoformat(),
        "time_wib": t_wib.isoformat(),
        "date_wib": t_wib.date().isoformat(),
        "station": station,
        "report_type": report_type,
        "is_corrected": int(is_corrected),
        "temp_c": temp_c,
        "dewpoint_c": dew_c,
        "wind_dir_deg": wind_dir_deg,
        "wind_dir_vrb": wind_dir_vrb,
        "wind_speed_kt": wind_speed_kt,
        "wind_gust_kt": wind_gust_kt,
        "visibility_m": visibility_m,
        "cavok": cavok,
        "wx_intensity": wx_intensity,
        "wx_descriptor": wx_descriptor,
        "wx_phenomena": wx_phenomena,
        "has_rain": has_rain,
        "has_ts": has_ts,
        "has_fog": has_fog,
        "has_haze": has_haze,
        "cloud_layer_1": cloud_layer_1,
        "cloud_layer_2": cloud_layer_2,
        "cloud_layer_3": cloud_layer_3,
        "cloud_base_min_ft": cloud_base_min_ft,
        "has_cb": has_cb,
        "has_tcu": has_tcu,
        "qnh_hpa": qnh_hpa,
        "trend": trend,
        "raw_text": full_line,
        "source_file": source_file,
    }


def _extract_weather(m_obj: Metar.Metar) -> tuple[str | None, str | None, str | None]:
    """
    Extract weather phenomena dari python-metar object.
    python-metar simpan weather sebagai list of tuples:
        (intensity, descriptor, precipitation, obscuration, other)
    """
    if not m_obj.weather:
        return None, None, None

    intensities = []
    descriptors = []
    phenomena = []

    for w in m_obj.weather:
        # w adalah tuple of 5 elements: (intensity, desc, precip, obsc, other)
        intensity, descriptor, precip, obscur, other = w
        if intensity:
            intensities.append(intensity)
        if descriptor:
            descriptors.append(descriptor)
        for p in (precip, obscur, other):
            if p:
                phenomena.append(p)

    return (
        ",".join(intensities) if intensities else None,
        ",".join(descriptors) if descriptors else None,
        ",".join(phenomena) if phenomena else None,
    )


def _extract_clouds(
    m_obj: Metar.Metar,
) -> tuple[list[str], int | None, int, int]:
    """
    Extract cloud layers, find lowest base, detect CB/TCU.

    Returns:
        (layer_strings, min_base_ft, has_cb, has_tcu)
    """
    if not m_obj.sky:
        return [], None, 0, 0

    layer_strs = []
    bases_ft = []
    has_cb = 0
    has_tcu = 0

    for layer in m_obj.sky:
        # layer = (cover, height, cloud_type)
        cover = layer[0] if len(layer) > 0 else None
        height = layer[1] if len(layer) > 1 else None
        cloud_type = layer[2] if len(layer) > 2 else None

        if cover is None:
            continue

        # Build string representation: "SCT016" atau "FEW018CB"
        height_str = ""
        height_ft = None
        if height is not None:
            try:
                height_ft = int(round(height.value("FT")))
                height_str = f"{height_ft // 100:03d}"
                bases_ft.append(height_ft)
            except Exception:
                pass

        type_str = cloud_type if cloud_type else ""
        layer_strs.append(f"{cover}{height_str}{type_str}")

        if cloud_type == "CB":
            has_cb = 1
        elif cloud_type == "TCU":
            has_tcu = 1

    min_base = min(bases_ft) if bases_ft else None
    return layer_strs, min_base, has_cb, has_tcu


def _extract_trend(metar_clean: str) -> str | None:
    """Extract trend forecast (NOSIG, BECMG, TEMPO)."""
    for trend in ("NOSIG", "BECMG", "TEMPO"):
        if f" {trend}" in metar_clean or metar_clean.endswith(trend):
            return trend
    return None