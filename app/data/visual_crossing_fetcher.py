"""
app/data/visual_crossing_fetcher.py
=====================================
Fetches daily climate data (precipitation, temperature, humidity) from the
Visual Crossing Timeline API and caches results in timeseries.climate_daily.

Cache strategy
--------------
  1. Check timeseries.climate_daily for existing rows (grid_id + date range).
  2. On cache hit: return from DB without any API call.
  3. On cache miss: call Visual Crossing API, write to climate_daily, return.

grid_id format: f"{round(lat, 2)}_{round(lon, 2)}"
  - ~1–2km grid snap (0.02° resolution, configurable via CLIMATE_GRID_RESOLUTION)
  - Must match whatever the climate ingestion pipeline uses for grid_id.

RainfallRecord.rainfall_mm ← precip field (mm/day, same unit as legacy CHIRPS)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional

import psycopg2
import psycopg2.extras
import requests

from app.core.config import get_settings
from app.core.models import RainfallRecord

logger = logging.getLogger(__name__)

# SQL
_CHECK_CACHE_SQL = """
SELECT observation_date, precip_mm
FROM   timeseries.climate_daily
WHERE  grid_id = %s
  AND  observation_date BETWEEN %s AND %s
ORDER  BY observation_date
"""

_UPSERT_CLIMATE_SQL = """
INSERT INTO timeseries.climate_daily
    (grid_id, observation_date, max_temp_c, min_temp_c, precip_mm, humidity_pct)
VALUES %s
ON CONFLICT (grid_id, observation_date) DO UPDATE SET
    max_temp_c   = EXCLUDED.max_temp_c,
    min_temp_c   = EXCLUDED.min_temp_c,
    precip_mm    = EXCLUDED.precip_mm,
    humidity_pct = EXCLUDED.humidity_pct
"""


class VisualCrossingFetcher:
    """
    Cache-first Visual Crossing climate fetcher.

    Usage (from DatacubeClient):
        vc = VisualCrossingFetcher(conn)
        records = vc.fetch(centroid_lat, centroid_lon, start, end)
    """

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self.conn     = conn
        self.settings = get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        lat: float,
        lon: float,
        start: date,
        end: date,
    ) -> List[RainfallRecord]:
        """
        Return daily rainfall records for (lat, lon) over [start, end].
        Checks DB cache first; falls back to Visual Crossing API on miss.
        """
        grid_id = self._snap_grid_id(lat, lon)

        # ── 1. Try cache ────────────────────────────────────────────────────────
        cached = self._read_from_cache(grid_id, start, end)
        if cached is not None:
            return cached

        # ── 2. Fetch from Visual Crossing ───────────────────────────────────────
        api_key = self.settings.visual_crossing_api_key
        if not api_key:
            logger.warning(
                "VISUAL_CROSSING_API_KEY not set — returning empty rainfall records. "
                "Rainfall signal will fall back to climatological onset."
            )
            return []

        rows = self._call_api(lat, lon, start, end, api_key)
        if not rows:
            return []

        # ── 3. Write to cache ───────────────────────────────────────────────────
        self._write_to_cache(grid_id, rows)

        # Return as RainfallRecord list (gap-filled)
        return self._to_rainfall_records(rows, start, end)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _snap_grid_id(self, lat: float, lon: float) -> str:
        """Snap centroid to CLIMATE_GRID_RESOLUTION grid cell."""
        res = self.settings.climate_grid_resolution
        snapped_lat = round(round(lat / res) * res, 4)
        snapped_lon = round(round(lon / res) * res, 4)
        return f"{snapped_lat}_{snapped_lon}"

    def _read_from_cache(
        self, grid_id: str, start: date, end: date
    ) -> Optional[List[RainfallRecord]]:
        """
        Returns list of RainfallRecord if the full date range is cached,
        otherwise returns None (triggering an API fetch).
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(_CHECK_CACHE_SQL, (grid_id, start, end))
                rows = cur.fetchall()

            if not rows:
                return None

            # Consider cache valid if we have ≥ 80% of expected days
            expected_days = (end - start).days + 1
            if len(rows) < int(expected_days * 0.80):
                logger.debug(
                    "Climate cache partial for grid_id=%s (%d/%d days) — refetching",
                    grid_id, len(rows), expected_days,
                )
                return None

            rmap = {r[0]: r[1] for r in rows}
            return self._gap_fill(rmap, start, end)

        except Exception as exc:
            logger.warning("Climate cache read error: %s", exc)
            return None

    def _call_api(
        self,
        lat: float, lon: float,
        start: date, end: date,
        api_key: str,
    ) -> list:
        """
        Call Visual Crossing Timeline API.
        Returns list of dicts: [{date, precip, tempmax, tempmin, humidity}, ...]
        """
        base_url = self.settings.visual_crossing_base_url
        location = f"{lat},{lon}"
        url = f"{base_url}/{location}/{start.isoformat()}/{end.isoformat()}"

        params = {
            "key":       api_key,
            "include":   "days",
            "elements":  "datetime,precip,tempmax,tempmin,humidity",
            "unitGroup": "metric",
            "contentType": "json",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as exc:
            logger.error("Visual Crossing API HTTP error: %s — %s", exc, exc.response.text[:200])
            return []
        except Exception as exc:
            logger.error("Visual Crossing API error: %s", exc)
            return []

        days = data.get("days", [])
        if not days:
            logger.warning("Visual Crossing returned no days for %s %s → %s", location, start, end)
            return []

        logger.info("Visual Crossing: fetched %d days for grid (%s, %s → %s)",
                    len(days), location, start, end)
        return days

    def _write_to_cache(self, grid_id: str, days: list) -> None:
        """Write API response rows to timeseries.climate_daily."""
        rows = []
        for day in days:
            try:
                obs_date    = date.fromisoformat(day["datetime"])
                precip_mm   = float(day.get("precip")    or 0.0)
                max_temp_c  = float(day.get("tempmax")   or 0.0)
                min_temp_c  = float(day.get("tempmin")   or 0.0)
                humidity_pct= float(day.get("humidity")  or 0.0)
                rows.append((grid_id, obs_date, max_temp_c, min_temp_c, precip_mm, humidity_pct))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("Skipping malformed day record: %s — %s", day, exc)

        if not rows:
            return

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, _UPSERT_CLIMATE_SQL, rows,
                    template="(%s, %s::date, %s::float, %s::float, %s::float, %s::float)",
                    page_size=500,
                )
            self.conn.commit()
            logger.info("Climate cache: wrote %d days for grid_id=%s", len(rows), grid_id)
        except Exception as exc:
            self.conn.rollback()
            logger.error("Climate cache write failed: %s", exc, exc_info=True)

    def _to_rainfall_records(self, days: list, start: date, end: date) -> List[RainfallRecord]:
        """Convert API day-dicts to RainfallRecord list with gap-fill."""
        rmap: dict[date, float] = {}
        for day in days:
            try:
                obs_date = date.fromisoformat(day["datetime"])
                precip   = max(0.0, float(day.get("precip") or 0.0))
                rmap[obs_date] = precip
            except Exception:
                pass
        return self._gap_fill(rmap, start, end)

    def _gap_fill(self, rmap: dict[date, float], start: date, end: date) -> List[RainfallRecord]:
        """Fill any missing dates with 0.0 mm (same as CHIRPS gap-fill)."""
        out, cur = [], start
        while cur <= end:
            out.append(RainfallRecord(
                record_date  = cur,
                rainfall_mm  = round(max(0.0, rmap.get(cur, 0.0)), 2),
                source       = "VisualCrossing" if cur in rmap else "VisualCrossing_gap",
            ))
            cur += timedelta(days=1)
        return out
