"""
app/data/datacube_client.py
============================
Database client for the Nuru Datacube planting engine (Stage 4).

Responsibilities
----------------
1.  Connection pool management — mirrors Stage 1 (postgis_export.py) and
    Stage 3 (db_client.py) patterns exactly.
2.  get_eligible_farms()  — SELECT farms where crop_type is resolved but
    planting_date is NULL (or stale), grouped by tile_id.
3.  get_ndvi_series()     — READ from timeseries.farm_indices (cache).
4.  get_sar_series()      — READ from timeseries.farm_indices (cache).
5.  get_climate_series()  — Cache-first: climate_daily → VisualCrossing API.
6.  upsert_planting_result() — Bulk UPSERT back to spatial.farm_intelligence.

Data flow
---------
The batch script (run_datacube_batch.py) controls the fetch flow:
  1. Call stac_fetcher.Sentinel2STACFetcher.fetch_for_tile() → writes to farm_indices.
  2. Call stac_fetcher.Sentinel1STACFetcher.fetch_for_tile() → writes to farm_indices.
  3. Call get_ndvi_series() / get_sar_series() to read from that cache.
  4. Call get_climate_series() (VisualCrossing fetches and caches automatically).
  5. Run PlantingDateEnsemble.
  6. Call upsert_planting_result().
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from app.core.config import get_settings
from app.core.models import (
    FarmRow, NDVIObservation, SARObservation, RainfallRecord, SeasonResult,
)
from app.data.visual_crossing_fetcher import VisualCrossingFetcher

logger = logging.getLogger(__name__)


# ── SQL ─────────────────────────────────────────────────────────────────────────

_ELIGIBLE_FARMS_SQL = """
SELECT
    fi.farm_uuid::text                              AS farm_uuid,
    ST_AsText(f.geom)                               AS geom_wkt,
    fi.tile_id,
    f.aez_code                                      AS aez_code,
    f.county                                        AS county_code,
    f.ward                                          AS ward_code,
    fi.crop_type,
    ST_X(ST_Centroid(f.geom))                       AS centroid_lon,
    ST_Y(ST_Centroid(f.geom))                       AS centroid_lat,
    f.area_ha                                       AS area_ha
FROM   spatial.farms f
JOIN   spatial.farm_intelligence fi ON f.uid = fi.farm_uuid
WHERE  fi.crop_type IS NOT NULL
  AND  fi.crop_type != 'insufficient_data'
  AND  fi.tile_id IS NOT NULL
  AND  (
      fi.planting_date IS NULL
      OR fi.planting_processed_at < NOW() - INTERVAL '30 days'
  )
ORDER  BY fi.tile_id, fi.farm_uuid
LIMIT  %s
"""



_NDVI_CACHE_SQL = """
SELECT observation_date, index_name, mean_value, cloud_fraction
FROM   timeseries.farm_indices
WHERE  farm_uuid = %s::uuid
  AND  index_name IN ('ndvi', 'evi', 'ndwi')
  AND  observation_date BETWEEN %s AND %s
ORDER  BY observation_date
"""

_SAR_CACHE_SQL = """
SELECT observation_date, index_name, mean_value
FROM   timeseries.farm_indices
WHERE  farm_uuid = %s::uuid
  AND  index_name IN ('vv', 'vh', 'cross_pol')
  AND  observation_date BETWEEN %s AND %s
ORDER  BY observation_date
"""

_UPSERT_PLANTING_SQL = """
INSERT INTO spatial.farm_intelligence (
    farm_uuid,
    planting_date,
    planting_season,
    planting_year,
    planting_confidence,
    planting_confidence_level,
    planting_method,
    peak_ndvi,
    peak_ndvi_date,
    senescence_date,
    season_length_days,
    ndvi_integral,
    ndvi_rise_rate,
    total_rainfall_mm,
    planting_processed_at,
    updated_at
)
VALUES %s
ON CONFLICT (farm_uuid) DO UPDATE SET
    planting_date             = EXCLUDED.planting_date,
    planting_season           = EXCLUDED.planting_season,
    planting_year             = EXCLUDED.planting_year,
    planting_confidence       = EXCLUDED.planting_confidence,
    planting_confidence_level = EXCLUDED.planting_confidence_level,
    planting_method           = EXCLUDED.planting_method,
    peak_ndvi                 = EXCLUDED.peak_ndvi,
    peak_ndvi_date            = EXCLUDED.peak_ndvi_date,
    senescence_date           = EXCLUDED.senescence_date,
    season_length_days        = EXCLUDED.season_length_days,
    ndvi_integral             = EXCLUDED.ndvi_integral,
    ndvi_rise_rate            = EXCLUDED.ndvi_rise_rate,
    total_rainfall_mm         = EXCLUDED.total_rainfall_mm,
    planting_processed_at     = EXCLUDED.planting_processed_at,
    updated_at                = NOW()
"""


class DatacubeClient:
    """
    Thread-safe database client using psycopg2.ThreadedConnectionPool.
    Mirrors Stage 1 (postgis_export.py) pool and host-resolution pattern.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pool    = self._build_pool()

    # ── Connection pool ────────────────────────────────────────────────────────

    def _build_pool(self) -> pg_pool.ThreadedConnectionPool:
        s    = self.settings
        host = self._resolve_host()
        port = s.postgis_port

        logger.info(
            "PostGIS pool: host=%s port=%d db=%s user=%s min=%d max=%d",
            host, port, s.postgis_db, s.postgis_user,
            s.postgis_pool_min, s.postgis_pool_max,
        )
        pool = pg_pool.ThreadedConnectionPool(
            minconn  = s.postgis_pool_min,
            maxconn  = s.postgis_pool_max,
            host     = host,
            port     = port,
            dbname   = s.postgis_db,
            user     = s.postgis_user,
            password = s.postgis_password,
            connect_timeout = 15,
            options  = "-c search_path=spatial,timeseries,public",
        )
        # Connectivity test
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            ver = cur.fetchone()[0]
            logger.info("PostGIS OK — %s", ver[:60])
        pool.putconn(conn)
        return pool

    def _resolve_host(self) -> str:
        """Use private VPC host inside AWS, public/tunnel host otherwise."""
        in_aws = bool(
            os.getenv("AWS_EXECUTION_ENV")
            or os.getenv("AWS_BATCH_JOB_ID")
            or os.getenv("ECS_CONTAINER_METADATA_URI")
        )
        s = self.settings
        if in_aws and s.postgis_host_private:
            return s.postgis_host_private
        return s.postgis_host

    def _conn(self) -> psycopg2.extensions.connection:
        return self._pool.getconn()

    def _release(self, conn) -> None:
        self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()
        logger.info("PostGIS pool closed.")

    # ── Farm selection ─────────────────────────────────────────────────────────

    def get_eligible_farms(self, batch_size: int = 500) -> List[FarmRow]:
        """
        SELECT farms where crop_type is resolved but planting_date is missing/stale.
        Returns FarmRow objects pre-sorted by tile_id (for tile-group processing).
        """
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(_ELIGIBLE_FARMS_SQL, (batch_size,))
                rows = cur.fetchall()
            logger.info("Found %d eligible farms for planting date detection", len(rows))
            return [FarmRow(**dict(r)) for r in rows]
        finally:
            self._release(conn)

    def group_by_tile(self, farms: List[FarmRow]) -> Dict[str, List[FarmRow]]:
        """Group farm list by tile_id (mirrors Stage 3 tile_groups pattern)."""
        groups: Dict[str, List[FarmRow]] = defaultdict(list)
        for farm in farms:
            groups[farm.tile_id].append(farm)
        return dict(groups)

    # ── Cache reads ────────────────────────────────────────────────────────────

    def get_ndvi_series(
        self,
        farm_uuid: str,
        start: date,
        end: date,
    ) -> List[NDVIObservation]:
        """
        Read NDVI/EVI/NDWI time series from timeseries.farm_indices (cache).
        Called AFTER Sentinel2STACFetcher has populated the cache for this tile.
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(_NDVI_CACHE_SQL, (farm_uuid, start, end))
                rows = cur.fetchall()
        finally:
            self._release(conn)

        if not rows:
            return []

        # Pivot rows → NDVIObservation objects
        date_map: Dict[date, Dict[str, float]] = defaultdict(dict)
        cloud_map: Dict[date, float] = {}
        for obs_date, index_name, mean_value, cloud_fraction in rows:
            date_map[obs_date][index_name] = mean_value
            if cloud_fraction is not None:
                cloud_map[obs_date] = cloud_fraction

        observations = []
        for obs_date in sorted(date_map.keys()):
            idxs = date_map[obs_date]
            if "ndvi" not in idxs:
                continue
            cf = cloud_map.get(obs_date, 0.0)
            observations.append(NDVIObservation(
                obs_date        = obs_date,
                ndvi            = round(idxs["ndvi"], 4),
                evi             = round(idxs["evi"],  4) if "evi"  in idxs else None,
                ndwi            = round(idxs["ndwi"], 4) if "ndwi" in idxs else None,
                cloud_cover_pct = round(cf * 100.0, 1),
                pixel_count     = 10,   # Approximate — exact count not stored
            ))
        return observations

    def get_sar_series(
        self,
        farm_uuid: str,
        start: date,
        end: date,
    ) -> List[SARObservation]:
        """
        Read SAR VV/VH/cross_pol from timeseries.farm_indices (cache).
        Returns empty list gracefully if SAR data has not been ingested.
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(_SAR_CACHE_SQL, (farm_uuid, start, end))
                rows = cur.fetchall()
        finally:
            self._release(conn)

        if not rows:
            return []

        date_map: Dict[date, Dict[str, float]] = defaultdict(dict)
        for obs_date, index_name, mean_value in rows:
            date_map[obs_date][index_name] = mean_value

        observations = []
        for obs_date in sorted(date_map.keys()):
            idxs = date_map[obs_date]
            if "vv" not in idxs:
                continue
            observations.append(SARObservation(
                obs_date        = obs_date,
                vv_db           = round(idxs["vv"], 3),
                vh_db           = round(idxs["vh"], 3) if "vh" in idxs else None,
                cross_pol_ratio = round(idxs["cross_pol"], 3) if "cross_pol" in idxs else None,
            ))
        return observations

    def get_climate_series(
        self,
        centroid_lat: float,
        centroid_lon: float,
        start: date,
        end: date,
    ) -> List[RainfallRecord]:
        """
        Cache-first climate fetch (Visual Crossing).
        Checks timeseries.climate_daily first; calls API + writes cache on miss.
        """
        conn = self._conn()
        try:
            vc = VisualCrossingFetcher(conn)
            return vc.fetch(centroid_lat, centroid_lon, start, end)
        finally:
            self._release(conn)

    # ── Result writer ──────────────────────────────────────────────────────────

    def upsert_planting_results(
        self,
        results: List[tuple],   # list of (farm_uuid, SeasonResult) pairs
        dry_run: bool = False,
    ) -> int:
        """
        Bulk UPSERT planting date + phenology into spatial.farm_intelligence.

        Parameters
        ----------
        results   : list of (farm_uuid, SeasonResult)
        dry_run   : if True, log rows but skip DB write

        Returns
        -------
        Number of rows upserted.
        """
        if not results:
            return 0

        from datetime import datetime as dt
        rows = []
        for farm_uuid, sr in results:
            n = sr.ndvi_signal
            r = sr.rainfall_signal
            rows.append((
                farm_uuid,
                sr.estimated_planting_date,       # planting_date
                sr.season,                         # planting_season
                sr.year,                           # planting_year
                sr.confidence,                     # planting_confidence
                sr.confidence_level,               # planting_confidence_level
                sr.method_used,                    # planting_method
                n.peak_ndvi,                       # peak_ndvi
                n.peak_date,                       # peak_ndvi_date
                n.senescence_date,                 # senescence_date
                n.season_length_days,              # season_length_days
                n.ndvi_integral,                   # ndvi_integral
                n.ndvi_rise_rate,                  # ndvi_rise_rate
                r.total_seasonal_rainfall_mm,      # total_rainfall_mm
                dt.utcnow(),                       # planting_processed_at
            ))

        if dry_run:
            logger.info("[DRY RUN] Would upsert %d planting result rows — skipping DB write", len(rows))
            return len(rows)

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    _UPSERT_PLANTING_SQL,
                    rows,
                    template=(
                        "(%s::uuid, %s::date, %s, %s::smallint, %s::float, %s, %s, "
                        "%s::float, %s::date, %s::date, %s::smallint, %s::float, %s::float, "
                        "%s::float, %s::timestamptz)"
                    ),
                    page_size=500,
                )
            conn.commit()
            logger.info("Upserted %d planting results into spatial.farm_intelligence", len(rows))
            return len(rows)
        except Exception as exc:
            conn.rollback()
            logger.error("Planting result upsert failed: %s", exc, exc_info=True)
            raise
        finally:
            self._release(conn)
