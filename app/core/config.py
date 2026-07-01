"""
app/core/config.py
Kenya AEZ season calendar, crop registry, and application settings.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ── Season & AEZ enums/dataclasses ─────────────────────────────────────────────

class Season(str, Enum):
    LONG_RAINS   = "long_rains"
    SHORT_RAINS  = "short_rains"
    THIRD_SEASON = "third_season"


class ConfidenceLevel(str, Enum):
    HIGH      = "HIGH"
    MEDIUM    = "MEDIUM"
    LOW       = "LOW"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SeasonWindow:
    season: Season
    window_start_month: int
    window_start_day: int
    window_end_month: int
    window_end_day: int
    climatological_onset_month: int
    climatological_onset_day: int
    description: str = ""

    def get_window(self, year: int) -> tuple[date, date]:
        start = date(year, self.window_start_month, self.window_start_day)
        end_year = year + 1 if self.window_end_month < self.window_start_month else year
        return start, date(end_year, self.window_end_month, self.window_end_day)

    def get_climatological_onset(self, year: int) -> date:
        return date(year, self.climatological_onset_month, self.climatological_onset_day)

    def get_search_start(self, year: int) -> date:
        """Start searching 14 days before window opens."""
        return self.get_window(year)[0] - timedelta(days=14)


@dataclass
class AEZConfig:
    aez_code: float
    zone_name: str
    zone_type: str
    seasons: list
    has_third_season: bool = False
    typical_crops: list[str] = field(default_factory=list)
    notes: str = ""
    # Per-AEZ signal weights: (rainfall, ndvi, sar) — must sum to 1.0
    signal_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)

    def get_season_window(self, season: Season) -> Optional[SeasonWindow]:
        for s in self.seasons:
            if s.season == season:
                return s
        return None

    def get_active_season(self, ref: date) -> Optional[Season]:
        for sw in self.seasons:
            s, e = sw.get_window(ref.year)
            if s <= ref <= e:
                return sw.season
            s2, e2 = sw.get_window(ref.year - 1)
            if s2 <= ref <= e2:
                return sw.season
        return None


# ── Kenya AEZ Registry ─────────────────────────────────────────────────────────

KENYA_AEZ_REGISTRY: dict[float, AEZConfig] = {

    33.0: AEZConfig(
        aez_code=33.0,
        zone_name="Upper Midland 3",
        zone_type="upper_midland",
        seasons=[
            SeasonWindow(Season.LONG_RAINS,  2, 15, 5, 31, 3, 1, "Long rains: Feb–May"),
            SeasonWindow(Season.SHORT_RAINS, 10, 1, 12, 20, 10, 15, "Short rains: Oct–Dec"),
        ],
        typical_crops=["Maize", "Wheat", "Barley", "Beans"],
        notes="Nakuru highlands. Reliable bimodal. 1500–2000mm/yr.",
        signal_weights=(0.30, 0.50, 0.20),
    ),

    44.0: AEZConfig(
        aez_code=44.0,
        zone_name="Lower Midland 4",
        zone_type="lower_midland",
        seasons=[
            SeasonWindow(Season.LONG_RAINS,  3, 15, 6, 15, 4,  1, "Long rains: Mar–Jun (variable)"),
            SeasonWindow(Season.SHORT_RAINS, 10, 15, 1, 10, 11,  1, "Short rains: Oct–Jan"),
        ],
        typical_crops=["Maize", "Sorghum", "Beans", "Green Gram"],
        notes="Kajiado semi-arid. 500–900mm/yr.",
        signal_weights=(0.35, 0.45, 0.20),
    ),

    46.0: AEZConfig(
        aez_code=46.0,
        zone_name="Lower Midland 6",
        zone_type="semi_arid",
        seasons=[
            SeasonWindow(Season.LONG_RAINS,  4,  1, 6, 30, 4, 20, "Long rains: Apr–Jun (delayed)"),
            SeasonWindow(Season.SHORT_RAINS, 11,  1, 1, 20, 11, 15, "Short rains: Nov–Jan"),
        ],
        typical_crops=["Sorghum", "Cowpea", "Green Gram"],
        notes="300–600mm/yr. SAR most reliable here.",
        signal_weights=(0.35, 0.35, 0.30),
    ),

    99.0: AEZConfig(
        aez_code=99.0,
        zone_name="Highland",
        zone_type="highland",
        seasons=[
            SeasonWindow(Season.LONG_RAINS,  2, 15, 5,  1, 3,  1, "Long rains: Feb–May"),
            SeasonWindow(Season.SHORT_RAINS,  9, 15, 12, 1, 10,  1, "Short rains: Sep–Dec"),
            SeasonWindow(Season.THIRD_SEASON, 6, 15, 8, 31, 7,  1, "Third season: Jun–Aug"),
        ],
        has_third_season=True,
        typical_crops=["Maize", "Irish Potato", "Wheat", "Pyrethrum"],
        notes=">2000mm/yr. Cool highlands. Third season possible.",
        signal_weights=(0.25, 0.45, 0.30),
    ),
}

DEFAULT_AEZ = AEZConfig(
    aez_code=0.0, zone_name="Unknown", zone_type="upper_midland",
    seasons=[
        SeasonWindow(Season.LONG_RAINS,  2, 15, 5, 31, 3, 1, "Default LR (Feb–May)"),
        SeasonWindow(Season.SHORT_RAINS, 10, 1, 12, 31, 10, 20, "Default SR"),
    ],
    typical_crops=["Maize"],
)


# ── AEZ string code → numeric registry key ────────────────────────────────────
# DB stores text codes like "LH 3", "UH 2"; GeoJSON stores numeric like 33.0.
# This mapper bridges both. Coverage follows Kenya AEZ classification.
# LH = Lower Highlands, UH = Upper Highlands, UM = Upper Midland, LM = Lower Midland
_AEZ_TEXT_TO_CODE: dict[str, float] = {
    # Upper & Lower Highlands → Highland config (99.0)
    "UH 1": 99.0, "UH 2": 99.0, "UH 3": 99.0,
    "LH 1": 99.0, "LH 2": 99.0, "LH 3": 99.0,
    "LH 4": 99.0, "LH 5": 99.0,
    # Upper Midland 1-3 → UM3 config (33.0)
    "UM 1": 33.0, "UM 2": 33.0, "UM 3": 33.0,
    # Upper Midland 4-6 → semi-arid configs
    "UM 4": 44.0, "UM 5": 44.0, "UM 6": 46.0,
    # Lower Midland → LM configs
    "LM 1": 44.0, "LM 2": 44.0, "LM 3": 44.0,
    "LM 4": 46.0, "LM 5": 46.0,
    # Inland / Coastal Lowlands → semi-arid
    "IL 1": 46.0, "IL 2": 46.0, "IL 3": 46.0,
    "IL 4": 46.0, "IL 5": 46.0, "IL 6": 46.0,
    "CL 1": 46.0, "CL 2": 46.0, "CL 3": 46.0,
    "CL 4": 46.0, "CL 5": 46.0,
}


def get_aez_config(aez_code) -> AEZConfig:
    """
    Accepts either:
      - float  : 33.0, 44.0, 46.0, 99.0  (GeoJSON / legacy path)
      - str    : 'LH 3', 'UH 2', 'UM 6'  (DB path, spatial.farms.aez_code)
      - None   : falls back to DEFAULT_AEZ

    Falls back to DEFAULT_AEZ if the code is unknown in either format.
    """
    if aez_code is None:
        return DEFAULT_AEZ

    # String code path (DB)
    if isinstance(aez_code, str):
        numeric = _AEZ_TEXT_TO_CODE.get(aez_code.strip().upper())
        if numeric is None:
            logger.warning("Unknown AEZ text code '%s' — falling back to DEFAULT_AEZ", aez_code)
            return DEFAULT_AEZ
        return KENYA_AEZ_REGISTRY.get(numeric, DEFAULT_AEZ)

    # Numeric path (GeoJSON / legacy)
    try:
        return KENYA_AEZ_REGISTRY.get(float(aez_code), DEFAULT_AEZ)
    except (TypeError, ValueError):
        logger.warning("Cannot parse aez_code '%s' — falling back to DEFAULT_AEZ", aez_code)
        return DEFAULT_AEZ


def build_season_run_list(years: list[int], include_third_season: bool = False) -> list[dict]:
    """
    Build the ordered list of {season, year} pairs to process.
    For 2024+2025 this produces:
        [LR-2024, SR-2024, LR-2025, SR-2025]
    """
    runs = []
    for year in sorted(years):
        runs.append({"season": Season.LONG_RAINS.value,  "year": year})
        runs.append({"season": Season.SHORT_RAINS.value, "year": year})
        if include_third_season:
            runs.append({"season": Season.THIRD_SEASON.value, "year": year})
    return runs


# ── Crop Registry ──────────────────────────────────────────────────────────────
# Maize-first. Adding a new crop = one config entry, zero algorithm changes.
# The planting engine reads crop_type from spatial.farm_intelligence (set by
# Stage 3 classifier) and looks up parameters here.

@dataclass
class CropConfig:
    crop_type: str
    planting_offset_days: int          # days from NDVI greenup to planting
    min_season_length_days: int        # shortest viable crop cycle
    max_season_length_days: int        # longest viable crop cycle
    signal_weight_override: Optional[tuple] = None  # (rain, ndvi, sar) — None = AEZ default
    peak_ndvi_expected_range: tuple = (0.30, 0.90)  # (min, max) quality guard
    scl_min_valid_fraction: float = 0.50            # mirrors Stage 3 MIN_VALID_PIXEL_FRACTION
    notes: str = ""


CROP_REGISTRY: dict[str, CropConfig] = {
    "maize": CropConfig(
        crop_type="maize",
        planting_offset_days=12,
        min_season_length_days=75,
        max_season_length_days=130,
        signal_weight_override=None,
        peak_ndvi_expected_range=(0.45, 0.85),
        notes="Primary crop. Bimodal LR + SR. NDVI-dominant in highlands.",
    ),
    # ── Future crops — uncomment + tune when Stage 3 classifier supports them ──
    # "wheat": CropConfig(
    #     "wheat", planting_offset_days=10, min_season_length_days=90,
    #     max_season_length_days=150, peak_ndvi_expected_range=(0.40, 0.80),
    # ),
    # "sorghum": CropConfig(
    #     "sorghum", planting_offset_days=14, min_season_length_days=90,
    #     max_season_length_days=160, signal_weight_override=(0.40, 0.35, 0.25),
    #     peak_ndvi_expected_range=(0.30, 0.70),
    #     notes="SAR more reliable in semi-arid zones.",
    # ),
    # "beans": CropConfig(
    #     "beans", planting_offset_days=8, min_season_length_days=55,
    #     max_season_length_days=90, peak_ndvi_expected_range=(0.30, 0.65),
    # ),
    # "potato": CropConfig(
    #     "potato", planting_offset_days=7, min_season_length_days=80,
    #     max_season_length_days=120, peak_ndvi_expected_range=(0.40, 0.75),
    # ),
    # "cowpea": CropConfig(
    #     "cowpea", planting_offset_days=10, min_season_length_days=60,
    #     max_season_length_days=100, signal_weight_override=(0.35, 0.35, 0.30),
    #     peak_ndvi_expected_range=(0.25, 0.60),
    # ),
}


def get_crop_config(crop_type: str) -> CropConfig:
    """
    Case-insensitive lookup. Falls back to maize defaults if crop unknown.
    To add a new crop: uncomment its entry in CROP_REGISTRY above — zero code changes.
    """
    ct = (crop_type or "maize").strip().lower()
    if ct not in CROP_REGISTRY:
        logger.warning("Unknown crop_type '%s' — falling back to maize CropConfig", ct)
    return CROP_REGISTRY.get(ct, CROP_REGISTRY["maize"])


# ── Application Settings ───────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── API ────────────────────────────────────────────────────────────────────
    api_key: str = "precisionfarms-dev-key"
    api_port: int = 8000

    # ── Detection thresholds ───────────────────────────────────────────────────
    max_cloud_cover: int = 80                # scene-level STAC filter; SCL handles per-pixel
    rainfall_onset_threshold_mm: float = 25.0
    ndvi_greenup_threshold: float = 0.08
    sar_tillage_threshold_db: float = -1.5
    high_confidence_score: float = 0.60
    min_confidence_score: float = 0.20
    ndvi_smoothing_window: int = 5
    rainfall_false_start_days: int = 14

    # ── Signal weights (global defaults; AEZ + crop configs can override) ──────
    weight_rainfall: float = 0.40
    weight_ndvi: float = 0.35
    weight_sar: float = 0.25

    # ── PostGIS / TimescaleDB ──────────────────────────────────────────────────
    # Mirrors Stage 1 (postgis_export.py) and Stage 3 (db_client.py) env var names.
    postgis_host: str = "localhost"
    postgis_host_private: str = ""          # Used inside AWS Batch/ECS (private VPC IP)
    postgis_port: int = 5433                # 5433 = SSH tunnel (dev); 5432 = VPC (prod)
    postgis_db: str = "nuru_datacube"
    postgis_user: str = "postgres"
    postgis_password: str = ""
    postgis_pool_min: int = 2
    postgis_pool_max: int = 10

    # ── STAC (Element84 Earth Search) ──────────────────────────────────────────
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"
    stac_s2_collection: str = "sentinel-2-l2a"
    stac_s1_collection: str = "sentinel-1-grd"
    stac_max_cloud_cover_scene: int = 80    # loose scene filter; SCL does per-pixel masking
    stac_max_items_per_tile: int = 20

    # ── SCL cloud masking (mirrors Stage 3 stac_patch_fetcher.py) ─────────────
    scl_min_valid_fraction: float = 0.50    # min fraction of valid SCL pixels to use a timestep
    scl_min_valid_timesteps: int = 4        # min clear scenes to compute planting date

    # ── Visual Crossing climate ────────────────────────────────────────────────
    visual_crossing_api_key: str = ""
    visual_crossing_base_url: str = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices"
        "/rest/services/timeline"
    )
    climate_grid_resolution: float = 0.02  # degrees (~2 km snap for climate_daily grid_id)

    # ── AWS / S3 ───────────────────────────────────────────────────────────────
    s3_bucket: str = ""
    aws_region: str = "eu-north-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()
