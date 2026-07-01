"""
app/core/models.py

Key design decision: NDVIGreenupSignal and SARTillageSignal carry
FULL phenological profiles + raw time series so your crop ID
pipeline can consume them directly without re-fetching any data.

Fields marked # [CROP ID] are specifically for the downstream pipeline.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    PARTIAL   = "partial"


# ── Internal data models ───────────────────────────────────────────────────────

class FarmPolygon(BaseModel):
    """
    Represents a single farm polygon being processed.

    In GEE mode: polygon_id / fid / county / ward come from a GeoJSON file.
    In Datacube mode: farm_uuid, tile_id, crop_type are populated from the DB;
    polygon_id is set to farm_uuid for downstream compatibility.
    """
    polygon_id: str                            # Primary key used throughout pipeline
    fid: Optional[int] = None

    # Location / admin
    county: Optional[str] = None
    ward: Optional[str] = None
    aez_code: Optional[str] = None

    # Geometry
    geometry: Dict[str, Any]                   # GeoJSON geometry dict
    geom_wkt: Optional[str] = None             # WKT string (from DB ST_AsText)
    centroid_lat: float
    centroid_lon: float
    area_ha: Optional[float] = None

    # ── Datacube fields (populated when reading from farm_intelligence) ────────
    farm_uuid: Optional[str] = None            # UUID PK from spatial.farms
    tile_id: Optional[str] = None             # Sentinel-2 MGRS tile (e.g. '37MBU')
    crop_type: Optional[str] = None           # Set by Stage 3 classifier


class FarmRow(BaseModel):
    """
    Output of DatacubeClient.get_eligible_farms().
    One row per farm eligible for planting date detection.
    Direct mapping from the JOIN of spatial.farms + spatial.farm_intelligence.
    """
    farm_uuid: str
    geom_wkt: str                              # ST_AsText(f.geom)
    tile_id: str
    aez_code: Optional[str] = None        # DB text code e.g. LH 3
    county_code: Optional[str] = None
    ward_code: Optional[str] = None
    crop_type: str
    centroid_lat: float
    centroid_lon: float
    area_ha: Optional[float] = None

    def to_farm_polygon(self) -> FarmPolygon:
        """Convert DB row into the FarmPolygon the pipeline expects."""
        import json
        from shapely import wkt as shapely_wkt
        geom = shapely_wkt.loads(self.geom_wkt)
        geojson_geom = {
            "type": geom.geom_type,
            "coordinates": list(geom.__geo_interface__["coordinates"]),
        }
        return FarmPolygon(
            polygon_id=self.farm_uuid,
            farm_uuid=self.farm_uuid,
            tile_id=self.tile_id,
            crop_type=self.crop_type,
            county=self.county_code,
            ward=self.ward_code,
            aez_code=self.aez_code,
            geometry=geojson_geom,
            geom_wkt=self.geom_wkt,
            centroid_lat=self.centroid_lat,
            centroid_lon=self.centroid_lon,
            area_ha=self.area_ha,
        )


class NDVIObservation(BaseModel):
    obs_date: date
    ndvi: float
    evi: Optional[float] = None
    ndwi: Optional[float] = None
    cloud_cover_pct: float = 0.0
    pixel_count: int = 0


class SARObservation(BaseModel):
    obs_date: date
    vv_db: float
    vh_db: Optional[float] = None
    cross_pol_ratio: Optional[float] = None   # VH - VV in dB
    pass_direction: str = "ASCENDING"


class RainfallRecord(BaseModel):
    record_date: date
    rainfall_mm: float
    source: str = "VisualCrossing"


# ── Detection signal outputs ───────────────────────────────────────────────────

class RainfallOnsetSignal(BaseModel):
    onset_date: Optional[date] = None
    cumulative_3day_mm: Optional[float] = None
    is_false_start: bool = False
    dry_spell_within_14d: int = 0
    total_seasonal_rainfall_mm: Optional[float] = None   # [CROP ID] water availability
    confidence: float = 0.0
    available: bool = False


class NDVIGreenupSignal(BaseModel):
    # ── Planting date estimate ─────────────────────────────────────────────────
    greenup_date: Optional[date] = None
    estimated_planting_date: Optional[date] = None
    planting_offset_days: int = 12

    # ── Phenological profile — [CROP ID] ──────────────────────────────────────
    baseline_ndvi: Optional[float] = None        # Pre-season bare soil
    peak_ndvi: Optional[float] = None            # Max NDVI during season
    peak_date: Optional[date] = None             # Date of peak
    ndvi_change: Optional[float] = None          # peak - baseline (crop vigour)
    senescence_date: Optional[date] = None       # NDVI drops to 70% of peak
    season_length_days: Optional[int] = None     # greenup → senescence (crop cycle)
    ndvi_at_harvest: Optional[float] = None      # NDVI 90 days after planting
    ndvi_integral: Optional[float] = None        # Area under curve (photosynthetic activity)
    ndvi_rise_rate: Optional[float] = None       # NDVI units/day from planting to peak

    # ── Full time series [CROP ID] — plug directly into your ML model ─────────
    ndvi_timeseries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{date, ndvi, evi, ndwi, cloud_pct}] clean observations"
    )

    # ── Quality ────────────────────────────────────────────────────────────────
    cloud_gap_days: int = 0
    n_observations: int = 0
    confidence: float = 0.0
    available: bool = False


class SARTillageSignal(BaseModel):
    # ── Planting date estimate ─────────────────────────────────────────────────
    onset_date: Optional[date] = None
    vv_change_db: Optional[float] = None         # Change from baseline at tillage

    # ── SAR phenological features — [CROP ID] ─────────────────────────────────
    vv_baseline: Optional[float] = None          # Pre-season VV (bare/fallow)
    vv_at_peak_ndvi: Optional[float] = None      # VV at crop maturity (structure proxy)
    vh_at_peak_ndvi: Optional[float] = None      # VH at crop maturity
    cross_pol_at_peak: Optional[float] = None    # VH-VV ratio at maturity (volume scatter)

    # ── Full SAR time series — [CROP ID] ──────────────────────────────────────
    vv_timeseries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{date, vv_db, vh_db, cross_pol, pass}] full SAR series"
    )

    moisture_increase_detected: bool = False
    tillage_detected: bool = False
    confidence: float = 0.0
    available: bool = False


class DataQuality(BaseModel):
    cloud_cover_pct: float = 0.0
    ndvi_observations: int = 0
    sar_observations: int = 0
    chirps_completeness: float = 0.0
    max_ndvi_gap_days: int = 0
    data_warnings: List[str] = Field(default_factory=list)


# ── Primary output models ──────────────────────────────────────────────────────

class SeasonResult(BaseModel):
    """
    Single polygon × single season result.

    Contains planting date + full phenological profile.
    Signals are structured so your crop ID pipeline can consume
    them directly — no re-fetching needed.
    """
    polygon_id: str
    fid: Optional[int] = None
    county: Optional[str] = None
    ward: Optional[str] = None
    aez_code: Optional[str] = None
    aez_zone_name: Optional[str] = None
    season: str
    year: int

    # Planting date output
    estimated_planting_date: Optional[date] = None
    planting_window_start: Optional[date] = None
    planting_window_end: Optional[date] = None
    climatological_onset: Optional[date] = None
    confidence: float = 0.0
    confidence_level: str = "UNCERTAIN"
    method_used: str = "unknown"

    # Full signals (rich data for crop ID pipeline)
    rainfall_signal: RainfallOnsetSignal = Field(default_factory=RainfallOnsetSignal)
    ndvi_signal: NDVIGreenupSignal = Field(default_factory=NDVIGreenupSignal)
    sar_signal: SARTillageSignal = Field(default_factory=SARTillageSignal)

    data_quality: DataQuality = Field(default_factory=DataQuality)

    geometry: Optional[Dict[str, Any]] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    area_ha: Optional[float] = None
    error: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class FarmSeasonHistory(BaseModel):
    """
    All processed seasons for one farm polygon.
    This is the top-level unit passed to the crop ID pipeline.

    seasons list is ordered: [LR-2024, SR-2024, LR-2025, SR-2025, ...]
    """
    polygon_id: str
    fid: Optional[int] = None
    county: Optional[str] = None
    ward: Optional[str] = None
    aez_code: Optional[str] = None
    aez_zone_name: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    area_ha: Optional[float] = None
    seasons: List[SeasonResult] = Field(default_factory=list)


class MultiSeasonJobResult(BaseModel):
    """Top-level job result returned by run_multiseasonal.py."""
    job_id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.PENDING
    years_processed: List[int] = Field(default_factory=list)
    seasons_processed: List[str] = Field(default_factory=list)
    total_polygons: int = 0
    total_tasks: int = 0          # polygons × seasons
    completed_tasks: int = 0
    succeeded: int = 0
    failed: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    farm_histories: List[FarmSeasonHistory] = Field(default_factory=list)
    error: Optional[str] = None
