"""
app/data/stac_fetcher.py
========================
Fetches Sentinel-2 (NDVI/EVI/NDWI) and Sentinel-1 (VV/VH SAR) time series
from the Element84 Earth Search STAC API using rasterio HTTP range requests.

Architecture
------------
Both fetchers operate at TILE LEVEL (mirrors Stage 3 stac_patch_fetcher.py):
  - One STAC search per tile per season (not per farm).
  - Per-farm pixel means extracted from the shared tile COGs via rasterio windows.
  - Results written to timeseries.farm_indices as the authoritative cache.

SCL Cloud Masking (Sentinel-2 only)
------------------------------------
Follows the same rules as Stage 3 crop classifier:
  SCL_VALID_CLASSES = {4=Vegetation, 5=Bare soil, 6=Water, 7=Unclassified}
  MIN_VALID_PIXEL_FRACTION = 0.50  (configurable)
  A timestep is skipped if < 50% of pixels over the farm are valid.

SAR Backscatter (Sentinel-1)
-----------------------------
  - Collection: sentinel-1-grd (IW mode, VV+VH)
  - COG values are stored as uint16 amplitude; we apply:
      sigma0_linear = (amplitude / 10000)^2
      sigma0_dB     = 10 * log10(sigma0_linear + 1e-10)
  - cross_pol = VH_dB - VV_dB  (volume scattering proxy)
  - No SCL masking (radar is cloud-penetrating).

Cache layer
-----------
Both fetchers write to timeseries.farm_indices using ON CONFLICT DO UPDATE,
so re-running a season is fully idempotent. The DatacubeClient reads from
this table directly (it never calls fetchers; the batch script manages flow).
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pystac_client import Client
from shapely import wkt as shapely_wkt

from app.core.config import get_settings
from app.core.models import NDVIObservation, SARObservation

logger = logging.getLogger(__name__)

# ── SCL masking constants (identical to Stage 3 stac_patch_fetcher.py) ─────────
# 4=Vegetation, 5=Bare soil, 6=Water, 7=Unclassified
# Excluded: 0=No data, 1=Saturated, 2=Dark (cautiously excluded for time series),
#           3=Cloud shadow, 8=Cloud medium, 9=Cloud high, 10=Cirrus, 11=Snow
SCL_VALID_CLASSES = {4, 5, 6, 7}

# Sentinel-2 bands needed for NDVI/EVI/NDWI (Element84 asset keys)
S2_BAND_ASSETS = {
    "B02": "blue",     # 490nm
    "B03": "green",    # 560nm
    "B04": "red",      # 665nm
    "B08": "nir",      # 842nm
}

# rasterio GDAL environment for COG HTTP range requests
RASTERIO_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="YES",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
    # Requester Pays is required to read Sentinel-1 GRD assets from AWS S3
    AWS_REQUEST_PAYER="requester",
)

# SQL for caching results into TimescaleDB
_UPSERT_FARM_INDICES = """
INSERT INTO timeseries.farm_indices
    (farm_uuid, observation_date, index_name, mean_value, cloud_fraction)
VALUES %s
ON CONFLICT (farm_uuid, observation_date, index_name) DO UPDATE SET
    mean_value     = EXCLUDED.mean_value,
    cloud_fraction = EXCLUDED.cloud_fraction
"""


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _resolve_db_host(settings) -> str:
    """Mirror Stage 1 postgis_export.py host resolution logic."""
    in_aws = bool(
        os.getenv("AWS_EXECUTION_ENV")
        or os.getenv("AWS_BATCH_JOB_ID")
        or os.getenv("ECS_CONTAINER_METADATA_URI")
    )
    host = (settings.postgis_host_private or settings.postgis_host) if in_aws else settings.postgis_host
    return host


def _get_tile_items(stac_client: Client, collection: str, tile_id: str,
                    start: date, end: date, max_cloud: int, max_items: int,
                    bbox: Optional[list[float]] = None) -> list:
    """
    Query STAC for items covering a tile within the date range.
    Filters by grid:code / s2:mgrs_tile / item ID (mirrors Stage 3).
    Returns items sorted by datetime ascending.
    """
    datetime_range = f"{start.isoformat()}/{end.isoformat()}"
    query_params = {}
    if collection.startswith("sentinel-2"):
        query_params["eo:cloud_cover"] = {"lt": max_cloud}

    search = stac_client.search(
        collections=[collection],
        datetime=datetime_range,
        bbox=bbox,
        query=query_params or None,
        limit=200,
    )
    items = list(search.items())

    # Filter by tile_id (MGRS code) — same logic as Stage 3
    clean_tile = tile_id.upper().replace("MGRS-", "").strip()
    tile_items = []
    for item in items:
        grid_code = (item.properties.get("grid:code")
                     or item.properties.get("s2:mgrs_tile")
                     or item.properties.get("s1:mgrs_tile", ""))
        if grid_code and clean_tile in grid_code.upper():
            tile_items.append(item)
        elif clean_tile in item.id.upper():
            tile_items.append(item)

    if not tile_items and items:
        logger.warning("Could not filter by tile_id '%s' (clean: '%s') — using all %d items", tile_id, clean_tile, len(items))
        tile_items = items

    tile_items.sort(key=lambda x: x.datetime)
    logger.info("STAC [%s] tile=%s: found %d items (%s → %s)",
                collection, tile_id, len(tile_items[:max_items]), start, end)
    return tile_items[:max_items]




def _read_scl_mask(item, minx: float, miny: float, maxx: float, maxy: float,
                   out_h: int = 32, out_w: int = 32) -> Optional[np.ndarray]:
    """
    Read SCL band and return boolean mask (True = valid pixel).
    Returns None if SCL unavailable → caller treats all pixels as valid.
    Mirrors Stage 3 read_scl_mask() exactly.
    """
    scl_asset = item.assets.get("scl")
    if not scl_asset:
        return None
    try:
        with rasterio.open(scl_asset.href) as src:
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            scl_data = src.read(
                1, window=window,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
                boundless=True,
            )
        return np.isin(scl_data, list(SCL_VALID_CLASSES))
    except Exception as exc:
        logger.warning("SCL read failed for item %s: %s — treating all pixels as valid", item.id, exc)
        return None


def _normalize_s1_l1c_href_date(href: str) -> str:
    """
    Normalize Sentinel-1 L1C S3 key date segments to zero-padded MM/DD.

    Some STAC items expose hrefs like .../GRD/2026/1/4/... while object keys are
    stored as .../GRD/2026/01/04/... . This normalizes the known variant.
    """
    prefix = "s3://sentinel-s1-l1c/GRD/"
    if not href.startswith(prefix):
        return href

    parts = href.split("/")
    try:
        grd_idx = parts.index("GRD")
    except ValueError:
        return href

    if len(parts) <= grd_idx + 3:
        return href

    month = parts[grd_idx + 2]
    day = parts[grd_idx + 3]
    if month.isdigit() and day.isdigit():
        parts[grd_idx + 2] = month.zfill(2)
        parts[grd_idx + 3] = day.zfill(2)
        return "/".join(parts)
    return href


def _s1_asset_href_candidates(asset_href: str) -> List[str]:
    """Return candidate hrefs to try for Sentinel-1 assets."""
    candidates = [asset_href]
    normalized = _normalize_s1_l1c_href_date(asset_href)
    if normalized != asset_href:
        candidates.append(normalized)
    return candidates


# ── Sentinel-2 Fetcher ──────────────────────────────────────────────────────────

class Sentinel2STACFetcher:
    """
    Fetches Sentinel-2 L2A NDVI/EVI/NDWI time series for a group of farms
    sharing the same Sentinel-2 MGRS tile, then caches results to
    timeseries.farm_indices.

    Call pattern (from run_datacube_batch.py):
        fetcher = Sentinel2STACFetcher(conn)
        results = fetcher.fetch_for_tile(tile_id, tile_farms, fetch_start, win_end)
        # results: {farm_uuid: list[NDVIObservation]}
    """

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self.conn     = conn
        self.settings = get_settings()
        self._client  = Client.open(self.settings.stac_api_url)

    def fetch_for_tile(
        self,
        tile_id: str,
        tile_farms: List[Tuple[str, str]],   # [(farm_uuid, geom_wkt), ...]
        start: date,
        end: date,
    ) -> Dict[str, List[NDVIObservation]]:
        """
        Fetch NDVI/EVI/NDWI time series for all farms on a single tile.
        Writes each observation to timeseries.farm_indices (cache).

        Parameters
        ----------
        tile_id     : MGRS tile code, e.g. '37MBU'
        tile_farms  : list of (farm_uuid, geom_wkt) for farms on this tile
        start / end : date range to fetch

        Returns
        -------
        dict mapping farm_uuid → list[NDVIObservation]
        """
        if not tile_farms:
            return {}

        settings   = self.settings
        min_frac   = settings.scl_min_valid_fraction
        results: Dict[str, List[NDVIObservation]] = {farm_uuid: [] for farm_uuid, _ in tile_farms}
        cache_rows: List[tuple] = []

        # Compute union bounding box of all farms in the tile group to restrict search
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        for _, geom_wkt in tile_farms:
            try:
                geom = shapely_wkt.loads(geom_wkt)
                b = geom.bounds
                min_x = min(min_x, b[0])
                min_y = min(min_y, b[1])
                max_x = max(max_x, b[2])
                max_y = max(max_y, b[3])
            except Exception:
                pass
        bbox = [min_x, min_y, max_x, max_y] if min_x != float("inf") else None

        # ── 1. Fetch STAC items for this tile once ─────────────────────────────
        items = _get_tile_items(
            self._client,
            settings.stac_s2_collection,
            tile_id,
            start, end,
            max_cloud=settings.stac_max_cloud_cover_scene,
            max_items=settings.stac_max_items_per_tile,
            bbox=bbox,
        )
        if not items:
            logger.warning("S2: No STAC items found for tile %s — skipping", tile_id)
            return results

        with rasterio.Env(**RASTERIO_ENV):
            for farm_uuid, geom_wkt in tile_farms:
                try:
                    geom = shapely_wkt.loads(geom_wkt)
                    minx, miny, maxx, maxy = geom.bounds

                    for item in items:
                        obs = self._process_s2_item(
                            item, farm_uuid, minx, miny, maxx, maxy, min_frac
                        )
                        if obs is not None:
                            results[farm_uuid].append(obs)
                            cache_rows.append((
                                farm_uuid,
                                obs.obs_date,
                                "ndvi",
                                obs.ndvi,
                                obs.cloud_cover_pct / 100.0,
                            ))
                            if obs.evi is not None:
                                cache_rows.append((
                                    farm_uuid, obs.obs_date, "evi",
                                    obs.evi, obs.cloud_cover_pct / 100.0,
                                ))
                            if obs.ndwi is not None:
                                cache_rows.append((
                                    farm_uuid, obs.obs_date, "ndwi",
                                    obs.ndwi, obs.cloud_cover_pct / 100.0,
                                ))

                except Exception as exc:
                    logger.error("S2 fetch failed for farm %s: %s", farm_uuid, exc, exc_info=True)

        # ── 2. Bulk write to timeseries.farm_indices ───────────────────────────
        if cache_rows:
            self._write_to_cache(cache_rows)

        for farm_uuid in results:
            results[farm_uuid].sort(key=lambda x: x.obs_date)
            logger.debug("S2 tile=%s farm=%s: %d observations cached",
                         tile_id, farm_uuid[:8], len(results[farm_uuid]))

        return results

    def _process_s2_item(
        self,
        item,
        farm_uuid: str,
        minx: float, miny: float, maxx: float, maxy: float,
        min_valid_fraction: float,
    ) -> Optional[NDVIObservation]:
        """
        For one STAC item × one farm: read SCL + spectral bands, compute
        NDVI/EVI/NDWI. Returns None if too cloudy or read error.
        """
        # ── SCL mask ──────────────────────────────────────────────────────────
        scl_mask = _read_scl_mask(item, minx, miny, maxx, maxy)
        if scl_mask is None:
            scl_mask = np.ones((32, 32), dtype=bool)  # no SCL → accept all

        valid_fraction = float(scl_mask.mean())
        if valid_fraction < min_valid_fraction:
            return None  # Too cloudy / shadowy over this farm

        cloud_pct = round((1.0 - valid_fraction) * 100.0, 1)

        # ── Spectral bands ────────────────────────────────────────────────────
        bands: Dict[str, np.ndarray] = {}
        for band_name, asset_key in S2_BAND_ASSETS.items():
            asset = item.assets.get(asset_key)
            if not asset:
                return None
            try:
                with rasterio.open(asset.href) as src:
                    window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                    raw = src.read(
                        1, window=window,
                        out_shape=(32, 32),
                        resampling=Resampling.bilinear,
                        boundless=True,
                    )
                    bands[band_name] = raw.astype(np.float32) / 10000.0
            except Exception as exc:
                logger.debug("Band %s read error for item %s: %s", band_name, item.id, exc)
                return None

        # ── Apply SCL mask (zero-out invalid pixels) ──────────────────────────
        for arr in bands.values():
            arr[~scl_mask] = np.nan

        eps = 1e-8
        B02 = bands["B02"]
        B03 = bands["B03"]
        B04 = bands["B04"]
        B08 = bands["B08"]

        # Compute indices using only valid (non-NaN) pixels
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi_arr  = (B08 - B04) / (B08 + B04 + eps)
            evi_arr   = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1 + eps)
            ndwi_arr  = (B03 - B08) / (B03 + B08 + eps)

        valid_mask = scl_mask & np.isfinite(ndvi_arr)
        n_valid    = int(valid_mask.sum())
        if n_valid == 0:
            return None

        ndvi_mean = float(np.nanmean(ndvi_arr[valid_mask]))
        evi_mean  = float(np.nanmean(evi_arr[valid_mask]))
        ndwi_mean = float(np.nanmean(ndwi_arr[valid_mask]))

        obs_date = item.datetime.date() if item.datetime else None
        if obs_date is None:
            return None

        return NDVIObservation(
            obs_date       = obs_date,
            ndvi           = round(ndvi_mean, 4),
            evi            = round(evi_mean,  4),
            ndwi           = round(ndwi_mean, 4),
            cloud_cover_pct= cloud_pct,
            pixel_count    = n_valid,
        )

    def _write_to_cache(self, rows: List[tuple]) -> None:
        """Bulk upsert index rows to timeseries.farm_indices."""
        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, _UPSERT_FARM_INDICES, rows,
                    template="(%s::uuid, %s::date, %s, %s::float, %s::float)",
                    page_size=1000,
                )
            self.conn.commit()
            logger.info("S2 cache: wrote %d index rows to timeseries.farm_indices", len(rows))
        except Exception as exc:
            self.conn.rollback()
            logger.error("S2 cache write failed: %s", exc, exc_info=True)


# ── Sentinel-1 SAR Fetcher ──────────────────────────────────────────────────────

class Sentinel1STACFetcher:
    """
    Fetches Sentinel-1 GRD VV/VH backscatter time series for a group of farms
    sharing the same Sentinel-2 MGRS tile zone (Sentinel-1 tiles roughly align),
    then caches results to timeseries.farm_indices.

    Backscatter convention
    ----------------------
    Element84 sentinel-1-grd COGs store uint16 amplitude (DN) values.
    Conversion:
      amplitude_f  = DN / 10000.0               (normalised float)
      power_linear = amplitude_f ** 2           (sigma0 linear scale)
      sigma0_dB    = 10 * log10(power + 1e-10)  (dB, matches GEE S1 convention)

    cross_pol = VH_dB - VV_dB (stored as index_name='cross_pol')

    Stored index_names in farm_indices: 'vv', 'vh', 'cross_pol'
    cloud_fraction = NULL for SAR (radar is cloud-penetrating).
    """

    def __init__(self, conn: psycopg2.extensions.connection) -> None:
        self.conn     = conn
        self.settings = get_settings()
        self._client  = Client.open(self.settings.stac_api_url)

    def fetch_for_tile(
        self,
        tile_id: str,
        tile_farms: List[Tuple[str, str]],   # [(farm_uuid, geom_wkt), ...]
        start: date,
        end: date,
    ) -> Dict[str, List[SARObservation]]:
        """
        Fetch Sentinel-1 VV/VH time series for all farms on a tile zone.
        Writes to timeseries.farm_indices and returns results.
        """
        if not tile_farms:
            return {}

        settings = self.settings
        results: Dict[str, List[SARObservation]] = {farm_uuid: [] for farm_uuid, _ in tile_farms}
        cache_rows: List[tuple] = []

        # Compute union bounding box of all farms in the tile group to restrict search
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        for _, geom_wkt in tile_farms:
            try:
                geom = shapely_wkt.loads(geom_wkt)
                b = geom.bounds
                min_x = min(min_x, b[0])
                min_y = min(min_y, b[1])
                max_x = max(max_x, b[2])
                max_y = max(max_y, b[3])
            except Exception:
                pass
        bbox = [min_x, min_y, max_x, max_y] if min_x != float("inf") else None

        # ── 1. Fetch STAC items for this tile ─────────────────────────────────
        items = _get_tile_items(
            self._client,
            settings.stac_s1_collection,
            tile_id,
            start, end,
            max_cloud=100,          # no cloud filter for SAR
            max_items=settings.stac_max_items_per_tile,
            bbox=bbox,
        )
        if not items:
            logger.info("S1: No STAC items for tile %s — SAR signal will be absent (graceful)", tile_id)
            return results

        with rasterio.Env(**RASTERIO_ENV):
            for farm_uuid, geom_wkt in tile_farms:
                try:
                    geom = shapely_wkt.loads(geom_wkt)
                    minx, miny, maxx, maxy = geom.bounds

                    for item in items:
                        obs = self._process_s1_item(item, farm_uuid, minx, miny, maxx, maxy)
                        if obs is not None:
                            results[farm_uuid].append(obs)
                            # VV
                            cache_rows.append((
                                farm_uuid, obs.obs_date, "vv", obs.vv_db, None,
                            ))
                            # VH
                            if obs.vh_db is not None:
                                cache_rows.append((
                                    farm_uuid, obs.obs_date, "vh", obs.vh_db, None,
                                ))
                            # cross_pol
                            if obs.cross_pol_ratio is not None:
                                cache_rows.append((
                                    farm_uuid, obs.obs_date, "cross_pol",
                                    obs.cross_pol_ratio, None,
                                ))

                except Exception as exc:
                    logger.error("S1 fetch failed for farm %s: %s", farm_uuid, exc, exc_info=True)

        # ── 2. Bulk write to timeseries.farm_indices ───────────────────────────
        if cache_rows:
            self._write_to_cache(cache_rows)

        for farm_uuid in results:
            results[farm_uuid].sort(key=lambda x: x.obs_date)
            logger.debug("S1 tile=%s farm=%s: %d SAR observations cached",
                         tile_id, farm_uuid[:8], len(results[farm_uuid]))

        return results

    def _process_s1_item(
        self,
        item,
        farm_uuid: str,
        minx: float, miny: float, maxx: float, maxy: float,
    ) -> Optional[SARObservation]:
        """
        Read VV and VH bands for one farm bounding box, compute dB backscatter.
        Returns None if assets missing or read fails.
        """
        vv_asset = item.assets.get("vv") or item.assets.get("VV")
        vh_asset = item.assets.get("vh") or item.assets.get("VH")
        if not vv_asset:
            logger.debug("No VV asset in S1 item %s", item.id)
            return None

        obs_date = item.datetime.date() if item.datetime else None
        if obs_date is None:
            return None

        # Detect pass direction from properties
        pass_dir = (
            item.properties.get("sat:orbit_state", "").upper()
            or item.properties.get("s1:orbit_state", "ASCENDING").upper()
        )

        def _read_band_mean(asset_href: str) -> Optional[float]:
            """Read band, convert DN amplitude → sigma0 dB."""
            for href in _s1_asset_href_candidates(asset_href):
                try:
                    with rasterio.open(href) as src:
                        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                        raw = src.read(
                            1, window=window,
                            out_shape=(32, 32),
                            resampling=Resampling.bilinear,
                            boundless=True,
                        )
                    arr = raw.astype(np.float32)
                    # Remove nodata (zero or negative values)
                    valid = arr[arr > 0]
                    if valid.size == 0:
                        return None
                    # DN amplitude → linear power → dB
                    amplitude_f  = valid / 10000.0
                    power_linear = amplitude_f ** 2
                    sigma0_db    = 10.0 * np.log10(power_linear + 1e-10)
                    return float(np.mean(sigma0_db))
                except Exception as exc:
                    logger.debug("S1 band read error item %s href=%s: %s", item.id, href, exc)
            return None

        vv_db = _read_band_mean(vv_asset.href)
        if vv_db is None:
            return None

        vh_db = _read_band_mean(vh_asset.href) if vh_asset else None
        cross_pol = round(vh_db - vv_db, 3) if vh_db is not None else None

        return SARObservation(
            obs_date       = obs_date,
            vv_db          = round(vv_db, 3),
            vh_db          = round(vh_db, 3) if vh_db is not None else None,
            cross_pol_ratio= cross_pol,
            pass_direction = pass_dir or "ASCENDING",
        )

    def _write_to_cache(self, rows: List[tuple]) -> None:
        """Bulk upsert SAR rows to timeseries.farm_indices."""
        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, _UPSERT_FARM_INDICES, rows,
                    template="(%s::uuid, %s::date, %s, %s::float, %s)",
                    page_size=1000,
                )
            self.conn.commit()
            logger.info("S1 cache: wrote %d SAR rows to timeseries.farm_indices", len(rows))
        except Exception as exc:
            self.conn.rollback()
            logger.error("S1 cache write failed: %s", exc, exc_info=True)
