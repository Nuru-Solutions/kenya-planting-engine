"""
app/core/config.py
Kenya AEZ season calendar and application settings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


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
# Covers AEZ codes in your digifarms data: 33, 44, 46, 99

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
        signal_weights=(0.30, 0.50, 0.20),  # NDVI-dominant ensemble
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
        signal_weights=(0.35, 0.45, 0.20),  # still rainfall-biased but higher NDVI weight
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
        signal_weights=(0.35, 0.35, 0.30),  # keep SAR important but not dominant
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
        signal_weights=(0.25, 0.45, 0.30),  # favor NDVI but retain SAR support
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


def get_aez_config(aez_code: float) -> AEZConfig:
    return KENYA_AEZ_REGISTRY.get(aez_code, DEFAULT_AEZ)


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


# ── Application Settings ───────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GEE — pre-configured for your service account
    gee_service_account: str = "precisionfarms@serene-bastion-406504.iam.gserviceaccount.com"
    gee_credentials_path: str = "secrets/gee-credentials.json"

    # API
    api_key: str = "precisionfarms-dev-key"
    api_port: int = 8000

    # Detection thresholds
    max_cloud_cover: int = 70
    rainfall_onset_threshold_mm: float = 25.0
    ndvi_greenup_threshold: float = 0.08
    sar_tillage_threshold_db: float = -1.5
    # Confidence thresholds — calibrated for small tropical farm polygons.
    # Small EAK farms in cloudy seasons rarely exceed 0.65 via RS alone;
    # 0.60 HIGH / 0.20 UNCERTAIN floor reflects achievable ground truth.
    high_confidence_score: float = 0.60   # was 0.70
    min_confidence_score: float = 0.20    # was 0.30 (UNCERTAIN floor)
    ndvi_smoothing_window: int = 5
    rainfall_false_start_days: int = 14

    # Signal weights
    weight_rainfall: float = 0.40
    weight_ndvi: float = 0.35
    weight_sar: float = 0.25

    # AWS / S3
    s3_bucket: str = ""
    aws_region: str = "eu-north-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()
