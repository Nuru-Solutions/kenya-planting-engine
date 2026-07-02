"""
app/core/pipeline.py
Core single-polygon processor and GeoJSON parser.
"""
from __future__ import annotations
import json
import logging
from datetime import date, timedelta
from typing import Optional
from uuid import uuid4

from app.core.config import (
    get_aez_config, Season, AEZConfig, SeasonWindow, get_settings,
    get_crop_config, CropConfig,
)
from app.core.models import (
    FarmPolygon, SeasonResult,
    NDVIObservation, SARObservation, RainfallRecord,
    RainfallOnsetSignal, NDVIGreenupSignal, SARTillageSignal,
    DataQuality,
)
from app.algorithms.detector import (
    RainfallOnsetDetector, NDVIGreenupDetector,
    SARTillageDetector, PlantingDateEnsemble,
)

logger = logging.getLogger(__name__)


# ── GeoJSON Parser ─────────────────────────────────────────────────────────────

def parse_geojson(geojson: dict) -> list[FarmPolygon]:
    polygons = []
    for feat in geojson.get("features", []):
        props    = feat.get("properties", {})
        geometry = feat.get("geometry", {})
        gtype    = geometry.get("type", "")
        coords   = geometry.get("coordinates") or []

        # Extract outer ring — handle Polygon vs MultiPolygon nesting
        if gtype == "MultiPolygon" and coords:
            ring = coords[0][0] if coords[0] else []
        elif gtype == "Polygon" and coords:
            ring = coords[0]
        else:
            ring = coords[0] if coords else []
            # Detect extra nesting (unknown type safety)
            if ring and isinstance(ring[0], list) and ring[0] and isinstance(ring[0][0], list):
                ring = ring[0]

        if ring:
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            cx, cy = sum(lons) / len(lons), sum(lats) / len(lats)
        else:
            cx = cy = 0.0

        pid = str(props.get("ID") or props.get("fid") or uuid4())
        aez_code = props.get("Aez_Code")
        polygons.append(FarmPolygon(
            polygon_id=pid,
            fid=props.get("fid"),
            county=props.get("County"),
            ward=props.get("Ward"),
            aez_code=str(aez_code) if aez_code is not None else None,
            geometry=geometry,
            centroid_lat=cy,
            centroid_lon=cx,
            area_ha=_area_ha(ring),
        ))
    return polygons


def _area_ha(ring: list) -> float:
    if len(ring) < 3:
        return 0.0
    n    = len(ring)
    area = sum(
        ring[i][0] * ring[(i+1)%n][1] - ring[(i+1)%n][0] * ring[i][1]
        for i in range(n)
    )
    return round(abs(area) / 2.0 * (111_000 ** 2) / 10_000, 4)


def resolve_season(season_str: str, year: int, aez: AEZConfig) -> tuple[Season, SeasonWindow]:
    try:
        season = Season(season_str)
    except ValueError:
        season = Season.LONG_RAINS

    window = aez.get_season_window(season)
    if window is None:
        window = aez.get_season_window(Season.LONG_RAINS)
        season = Season.LONG_RAINS
        logger.warning(f"Season {season_str} not in AEZ {aez.aez_code}; falling back to long_rains")

    return season, window


# ── Single Polygon Processor ───────────────────────────────────────────────────

class PolygonProcessor:
    """
    Fetches all data sources and runs the detection ensemble for
    one polygon × one season. Returns a SeasonResult.

    Modes
    -----
    GEE mode (default, --mock / legacy):
        datacube_client=None → imports GEE fetchers (Sentinel2Fetcher, CHIRPSFetcher).

    Datacube mode (production):
        datacube_client=<DatacubeClient> → reads from TimescaleDB cache that
        was populated by Sentinel2STACFetcher + Sentinel1STACFetcher before
        this processor is called. No GEE imports required.
    """

    def __init__(
        self,
        use_rainfall: bool = True,
        use_ndvi: bool = True,
        use_sar: bool = True,
        fallback_to_climatology: bool = True,
        datacube_client=None,          # DatacubeClient | None
    ):
        self.use_rainfall = use_rainfall
        self.use_ndvi     = use_ndvi
        self.use_sar      = use_sar
        self.fallback     = fallback_to_climatology
        self.datacube     = datacube_client   # None → GEE path
        self.settings     = get_settings()

        self.rain_det  = RainfallOnsetDetector()
        self.ndvi_det  = NDVIGreenupDetector()
        self.sar_det   = SARTillageDetector()
        self.ensemble  = PlantingDateEnsemble()

    def process(
        self,
        polygon: FarmPolygon,
        season: Season,
        season_window: SeasonWindow,
        aez: AEZConfig,
        year: int,
    ) -> SeasonResult:
        try:
            return self._run(polygon, season, season_window, aez, year)
        except Exception as e:
            logger.error(f"FAILED {polygon.polygon_id} {season.value}/{year}: {e}", exc_info=True)
            return SeasonResult(
                polygon_id=polygon.polygon_id, fid=polygon.fid,
                county=polygon.county, ward=polygon.ward,
                aez_code=polygon.aez_code, season=season.value, year=year,
                geometry=polygon.geometry, centroid_lat=polygon.centroid_lat,
                centroid_lon=polygon.centroid_lon, area_ha=polygon.area_ha,
                error=str(e),
            )

    def _run(
        self,
        polygon: FarmPolygon,
        season: Season,
        season_window: SeasonWindow,
        aez: AEZConfig,
        year: int,
    ) -> SeasonResult:
        win_start, win_end = season_window.get_window(year)
        search_start       = season_window.get_search_start(year)
        fetch_start        = search_start - timedelta(days=60)  # extra for baseline

        # Crop-specific parameters (used by NDVI detector for planting offset)
        crop_config: CropConfig = get_crop_config(polygon.crop_type or "maize")

        # ── Fetch ──────────────────────────────────────────────────────────────
        ndvi_obs: list[NDVIObservation] = []
        sar_obs:  list[SARObservation]  = []
        rainfall: list[RainfallRecord]  = []

        if self.datacube is not None:
            # ── Datacube path: read from TimescaleDB cache ─────────────────────
            # STAC fetchers already wrote to farm_indices before process() was called.
            if self.use_ndvi:
                try:
                    ndvi_obs = self.datacube.get_ndvi_series(
                        polygon.farm_uuid, fetch_start, win_end
                    )
                except Exception as e:
                    logger.warning("NDVI cache read failed %s: %s", polygon.polygon_id, e)

            if self.use_sar:
                try:
                    sar_obs = self.datacube.get_sar_series(
                        polygon.farm_uuid, fetch_start, win_end
                    )
                except Exception as e:
                    logger.warning("SAR cache read failed %s: %s", polygon.polygon_id, e)

            if self.use_rainfall:
                try:
                    rainfall = self.datacube.get_climate_series(
                        polygon.centroid_lat, polygon.centroid_lon, fetch_start, win_end
                    )
                except Exception as e:
                    logger.warning("Climate fetch failed %s: %s", polygon.polygon_id, e)

        else:
            # ── GEE path: legacy / mock / validation ───────────────────────────
            if self.use_ndvi:
                try:
                    from app.data.sentinel2 import Sentinel2Fetcher
                    ndvi_obs = Sentinel2Fetcher().fetch(polygon, fetch_start, win_end)
                except Exception as e:
                    logger.warning("S2 GEE fetch failed %s: %s", polygon.polygon_id, e)

            if self.use_sar:
                try:
                    from app.data.sentinel1 import Sentinel1Fetcher
                    sar_obs = Sentinel1Fetcher().fetch(polygon, fetch_start, win_end)
                except Exception as e:
                    logger.warning("SAR GEE fetch failed %s: %s", polygon.polygon_id, e)

            if self.use_rainfall:
                try:
                    from app.data.chirps import CHIRPSFetcher
                    rainfall = CHIRPSFetcher().fetch(polygon, fetch_start, win_end)
                except Exception as e:
                    logger.warning("CHIRPS GEE fetch failed %s: %s", polygon.polygon_id, e)

        # ── Detect ─────────────────────────────────────────────────────────────
        rain_sig = (
            self.rain_det.detect(rainfall, season_window, year)
            if self.use_rainfall and rainfall else RainfallOnsetSignal(available=False)
        )
        ndvi_sig = (
            self.ndvi_det.detect(ndvi_obs, season_window, year, crop_config=crop_config)
            if self.use_ndvi and ndvi_obs else NDVIGreenupSignal(available=False)
        )
        # Pass NDVI peak date to SAR detector so it can extract SAR phenology at crop peak
        sar_sig = (
            self.sar_det.detect(sar_obs, season_window, year, peak_ndvi_date=ndvi_sig.peak_date)
            if self.use_sar and sar_obs else SARTillageSignal(available=False)
        )

        # ── Ensemble ───────────────────────────────────────────────────────────
        est_date, confidence, method = self.ensemble.combine(
            rain_sig, ndvi_sig, sar_sig, season_window, year, self.fallback,
            aez_weights=aez.signal_weights,
        )
        conf_level = self.ensemble.confidence_level(confidence)

        # ── Data quality metadata ──────────────────────────────────────────────
        expected_days    = (win_end - fetch_start).days + 1
        climate_complete = min(1.0, len(rainfall) / expected_days) if rainfall else 0.0
        max_ndvi_gap     = self.ndvi_det._max_gap(ndvi_obs, search_start, win_end) if ndvi_obs else 0
        avg_cloud        = sum(o.cloud_cover_pct for o in ndvi_obs) / len(ndvi_obs) if ndvi_obs else 0.0

        warnings = []
        if max_ndvi_gap > 15:      warnings.append(f"Large NDVI gap: {max_ndvi_gap}d")
        if avg_cloud > 40:         warnings.append(f"High avg cloud: {avg_cloud:.0f}%")
        if climate_complete < 0.8: warnings.append(f"Incomplete climate data: {climate_complete:.0%}")
        if conf_level == "UNCERTAIN": warnings.append("Very low confidence — field-verify")

        return SeasonResult(
            polygon_id=polygon.polygon_id,
            fid=polygon.fid,
            county=polygon.county,
            ward=polygon.ward,
            aez_code=polygon.aez_code,
            aez_zone_name=aez.zone_name,
            season=season.value,
            year=year,
            estimated_planting_date=est_date,
            planting_window_start=win_start,
            planting_window_end=win_end,
            climatological_onset=season_window.get_climatological_onset(year),
            confidence=confidence,
            confidence_level=conf_level,
            method_used=method,
            rainfall_signal=rain_sig,
            ndvi_signal=ndvi_sig,
            sar_signal=sar_sig,
            data_quality=DataQuality(
                cloud_cover_pct=round(avg_cloud, 1),
                ndvi_observations=len(ndvi_obs),
                sar_observations=len(sar_obs),
                chirps_completeness=round(climate_complete, 3),
                max_ndvi_gap_days=max_ndvi_gap,
                data_warnings=warnings,
            ),
            geometry=polygon.geometry,
            centroid_lat=polygon.centroid_lat,
            centroid_lon=polygon.centroid_lon,
            area_ha=polygon.area_ha,
        )
