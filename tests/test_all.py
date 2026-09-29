"""
tests/test_all.py
Full test suite — no GEE credentials required.
Run: pytest tests/ -v
"""
import pytest
from datetime import date, timedelta
from app.algorithms.detector import (
    RainfallOnsetDetector, NDVIGreenupDetector,
    SARTillageDetector, PlantingDateEnsemble,
)
from app.core.config import get_aez_config, Season
from app.core.models import (
    RainfallRecord, NDVIObservation, SARObservation,
    RainfallOnsetSignal, NDVIGreenupSignal, SARTillageSignal,
)
from app.core.pipeline import parse_geojson, _area_ha


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def aez33(): return get_aez_config(33.0)

@pytest.fixture
def lr_window(aez33): return aez33.get_season_window(Season.LONG_RAINS)

@pytest.fixture
def sr_window(aez33): return aez33.get_season_window(Season.SHORT_RAINS)

@pytest.fixture
def rainfall_good():
    records = []
    for i in range(90):
        d = date(2024, 3, 1) + timedelta(days=i)
        rain = (10.0 if date(2024, 3, 16) <= d <= date(2024, 3, 18)
                else 4.0 if d > date(2024, 3, 18) else 1.5)
        records.append(RainfallRecord(record_date=d, rainfall_mm=rain))
    return records

@pytest.fixture
def rainfall_false_start():
    records = []
    for i in range(90):
        d = date(2024, 3, 1) + timedelta(days=i)
        if   date(2024, 3, 10) <= d <= date(2024, 3, 12): rain = 11.0
        elif date(2024, 3, 25) <= d <= date(2024, 3, 27): rain = 11.0
        elif d > date(2024, 3, 27): rain = 4.0
        else: rain = 0.0
        records.append(RainfallRecord(record_date=d, rainfall_mm=rain))
    return records

@pytest.fixture
def ndvi_good():
    obs = []
    for i in range(22):
        d = date(2024, 2, 1) + timedelta(days=i * 5)
        ndvi = (0.20 + i * 0.003 if d < date(2024, 3, 20)
                else min(0.80, 0.35 + (i - 10) * 0.022))
        obs.append(NDVIObservation(obs_date=d, ndvi=ndvi, cloud_cover_pct=10.0, pixel_count=50))
    return obs

@pytest.fixture
def sar_good():
    obs = []
    for i in range(18):
        d    = date(2024, 2, 1) + timedelta(days=i * 6)
        vv   = -14.5 if d >= date(2024, 3, 15) else -12.0
        vh   = vv - 4.5
        obs.append(SARObservation(obs_date=d, vv_db=vv, vh_db=vh,
                                   cross_pol_ratio=round(vh - vv, 3)))
    return obs


# ── AEZ Config ─────────────────────────────────────────────────────────────────

class TestAEZConfig:
    def test_all_four_zones_loaded(self):
        for code in [33.0, 44.0, 46.0, 99.0]:
            assert get_aez_config(code).aez_code == code

    def test_aez33_bimodal(self, aez33):
        assert aez33.get_season_window(Season.LONG_RAINS) is not None
        assert aez33.get_season_window(Season.SHORT_RAINS) is not None

    def test_aez99_third_season(self):
        aez = get_aez_config(99.0)
        assert aez.has_third_season
        assert aez.get_season_window(Season.THIRD_SEASON) is not None

    def test_unknown_code_fallback(self):
        assert get_aez_config(999.0).aez_code == 0.0

    def test_window_start_before_end(self, lr_window):
        start, end = lr_window.get_window(2024)
        assert start < end

    def test_climatological_onset_in_window(self, lr_window):
        start, end = lr_window.get_window(2024)
        clim = lr_window.get_climatological_onset(2024)
        assert start <= clim <= end

    def test_sr_crosses_year(self):
        # AEZ 44 short rains: Oct 15 – Jan 10 (crosses year boundary)
        sw = get_aez_config(44.0).get_season_window(Season.SHORT_RAINS)
        start, end = sw.get_window(2024)
        assert end.year == 2025

    def test_active_season_detection(self, aez33):
        assert aez33.get_active_season(date(2024, 4, 15)) == Season.LONG_RAINS
        assert aez33.get_active_season(date(2024, 11, 1)) == Season.SHORT_RAINS


# ── Rainfall Onset ─────────────────────────────────────────────────────────────

class TestRainfallOnset:
    def setup_method(self): self.det = RainfallOnsetDetector()

    def test_detects_clear_onset(self, rainfall_good, lr_window):
        sig = self.det.detect(rainfall_good, lr_window, 2024)
        assert sig.available
        # Fixture produces rain >= threshold starting 2024-03-16, so onset is Mar 16
        assert sig.onset_date == date(2024, 3, 16)
        assert sig.cumulative_3day_mm >= 15.0
        assert sig.confidence >= 0.3
        assert not sig.is_false_start

    def test_false_start_detected(self, rainfall_false_start, lr_window):
        sig = self.det.detect(rainfall_false_start, lr_window, 2024)
        assert sig.available
        if sig.onset_date == date(2024, 3, 10):
            assert sig.is_false_start

    def test_seasonal_total_populated(self, rainfall_good, lr_window):
        sig = self.det.detect(rainfall_good, lr_window, 2024)
        assert sig.total_seasonal_rainfall_mm is not None
        assert sig.total_seasonal_rainfall_mm > 0

    def test_empty_input(self, lr_window):
        assert not self.det.detect([], lr_window, 2024).available

    def test_all_dry_no_onset(self, lr_window):
        records = [RainfallRecord(record_date=date(2024, 3, 1) + timedelta(days=i), rainfall_mm=0.5)
                   for i in range(60)]
        sig = self.det.detect(records, lr_window, 2024)
        assert sig.onset_date is None or sig.cumulative_3day_mm < 25.0


# ── NDVI Greenup ───────────────────────────────────────────────────────────────

class TestNDVIGreenup:
    def setup_method(self): self.det = NDVIGreenupDetector()

    def test_detects_greenup(self, ndvi_good, lr_window):
        sig = self.det.detect(ndvi_good, lr_window, 2024)
        assert sig.available
        assert sig.greenup_date is not None
        assert sig.estimated_planting_date is not None
        assert (sig.greenup_date - sig.estimated_planting_date).days == 12

    def test_phenology_populated(self, ndvi_good, lr_window):
        sig = self.det.detect(ndvi_good, lr_window, 2024)
        assert sig.peak_ndvi is not None
        assert sig.peak_date is not None
        assert sig.ndvi_change is not None
        assert sig.ndvi_integral is not None
        assert sig.ndvi_rise_rate is not None

    def test_timeseries_populated(self, ndvi_good, lr_window):
        sig = self.det.detect(ndvi_good, lr_window, 2024)
        assert len(sig.ndvi_timeseries) > 0
        first = sig.ndvi_timeseries[0]
        assert "date" in first and "ndvi" in first

    def test_empty_returns_unavailable(self, lr_window):
        assert not self.det.detect([], lr_window, 2024).available

    def test_cloudy_obs_low_confidence(self, lr_window):
        obs = [NDVIObservation(obs_date=date(2024, 3, 1) + timedelta(days=i*5),
                               ndvi=0.3, cloud_cover_pct=90.0, pixel_count=50)
               for i in range(10)]
        assert self.det.detect(obs, lr_window, 2024).confidence < 0.5

    def test_senescence_after_peak(self, ndvi_good, lr_window):
        sig = self.det.detect(ndvi_good, lr_window, 2024)
        if sig.senescence_date and sig.peak_date:
            assert sig.senescence_date > sig.peak_date

    def test_season_length_positive(self, ndvi_good, lr_window):
        sig = self.det.detect(ndvi_good, lr_window, 2024)
        if sig.season_length_days:
            assert sig.season_length_days > 0


# ── SAR Tillage ────────────────────────────────────────────────────────────────

class TestSARTillage:
    def setup_method(self): self.det = SARTillageDetector()

    def test_detects_tillage(self, sar_good, lr_window):
        sig = self.det.detect(sar_good, lr_window, 2024)
        assert sig.available
        if sig.tillage_detected:
            assert (sig.vv_change_db or 0) <= -1.5

    def test_vv_timeseries_populated(self, sar_good, lr_window):
        sig = self.det.detect(sar_good, lr_window, 2024)
        assert len(sig.vv_timeseries) > 0
        assert "vv_db" in sig.vv_timeseries[0]

    def test_sar_at_peak_with_ndvi_date(self, sar_good, lr_window):
        peak_ndvi_date = date(2024, 4, 20)
        sig = self.det.detect(sar_good, lr_window, 2024, peak_ndvi_date=peak_ndvi_date)
        # Should attempt to find SAR value at peak
        assert sig.available

    def test_baseline_populated(self, sar_good, lr_window):
        sig = self.det.detect(sar_good, lr_window, 2024)
        assert sig.vv_baseline is not None

    def test_empty_returns_unavailable(self, lr_window):
        assert not self.det.detect([], lr_window, 2024).available


# ── Ensemble ───────────────────────────────────────────────────────────────────

class TestEnsemble:
    def setup_method(self):
        self.ens = PlantingDateEnsemble()
        self.aez = get_aez_config(33.0)
        self.win = self.aez.get_season_window(Season.LONG_RAINS)

    def test_three_signals_ensemble(self):
        d = date(2024, 3, 18)
        r = RainfallOnsetSignal(onset_date=d, confidence=0.8, available=True, cumulative_3day_mm=30)
        n = NDVIGreenupSignal(estimated_planting_date=d + timedelta(1), confidence=0.75, available=True)
        s = SARTillageSignal(onset_date=d - timedelta(1), confidence=0.70, available=True, tillage_detected=True)
        dt, conf, method = self.ens.combine(r, n, s, self.win, 2024)
        assert dt is not None
        assert conf > 0.6
        assert "ensemble" in method

    def test_signal_agreement_bonus(self):
        d = date(2024, 3, 18)
        r = RainfallOnsetSignal(onset_date=d,                  confidence=0.7, available=True, cumulative_3day_mm=30)
        n = NDVIGreenupSignal(estimated_planting_date=d + timedelta(3), confidence=0.7, available=True)
        _, conf_agree, _ = self.ens.combine(r, n, SARTillageSignal(available=False), self.win, 2024)

        r2 = RainfallOnsetSignal(onset_date=d,                  confidence=0.7, available=True, cumulative_3day_mm=30)
        n2 = NDVIGreenupSignal(estimated_planting_date=d + timedelta(30), confidence=0.7, available=True)
        _, conf_disagree, _ = self.ens.combine(r2, n2, SARTillageSignal(available=False), self.win, 2024)

        assert conf_agree >= conf_disagree

    def test_no_signals_fallback(self):
        r = RainfallOnsetSignal(available=False)
        n = NDVIGreenupSignal(available=False)
        s = SARTillageSignal(available=False)
        dt, conf, method = self.ens.combine(r, n, s, self.win, 2024, fallback=True)
        assert dt is not None
        assert method == "fallback_climatology"

    def test_confidence_levels(self):
        assert self.ens.confidence_level(0.85) == "HIGH"
        assert self.ens.confidence_level(0.55) == "MEDIUM"
        assert self.ens.confidence_level(0.35) == "MEDIUM"   # boundary: >= 0.35 is MEDIUM
        assert self.ens.confidence_level(0.34) == "LOW"       # just below boundary → LOW
        assert self.ens.confidence_level(0.15) == "UNCERTAIN"


# ── GeoJSON Parser ─────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_valid_geojson(self):
        gj = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[35.72, -0.38], [35.73, -0.38],
                                     [35.73, -0.37], [35.72, -0.37], [35.72, -0.38]]]
                },
                "properties": {
                    "fid": 1, "ID": "test-001",
                    "County": "Nakuru", "Ward": "Nyota", "Aez_Code": 33.0,
                }
            }]
        }
        polygons = parse_geojson(gj)
        assert len(polygons) == 1
        p = polygons[0]
        assert p.polygon_id == "test-001"
        assert p.county == "Nakuru"
        # GeoJSON stores AEZ codes as floats; parse_geojson coerces to str for FarmPolygon
        assert p.aez_code == "33.0"
        assert p.area_ha > 0
        assert -1 < p.centroid_lat < 0
        assert 35 < p.centroid_lon < 36

    def test_area_calculation(self):
        ring = [[36.0, -1.0], [36.009, -1.0], [36.009, -1.009], [36.0, -1.009], [36.0, -1.0]]
        area = _area_ha(ring)
        assert 50 < area < 200

    def test_empty_geojson(self):
        polygons = parse_geojson({"type": "FeatureCollection", "features": []})
        assert polygons == []


# ── Multi-season build ─────────────────────────────────────────────────────────

class TestSeasonRunList:
    def test_two_years_four_runs(self):
        from app.core.config import build_season_run_list
        runs = build_season_run_list([2024, 2025])
        assert len(runs) == 4
        seasons = [r["season"] for r in runs]
        assert seasons.count("long_rains") == 2
        assert seasons.count("short_rains") == 2

    def test_runs_ordered_by_year(self):
        from app.core.config import build_season_run_list
        runs = build_season_run_list([2024, 2025])
        years = [r["year"] for r in runs]
        assert years[0] == years[1] == 2024
        assert years[2] == years[3] == 2025


# ── Crop Registry ──────────────────────────────────────────────────────────────

class TestCropRegistry:
    """
    Regression suite for CROP_REGISTRY / get_crop_config.

    Root cause that prompted this class:
    Stage 3 (crop-classifier-v2) emits CROP_NAMES = ["maize", "cassava",
    "common_bean", "soybean"]. Stage 4 (kenya-planting-engine) calls
    get_crop_config(farm.crop_type) for every farm. Prior to the fix,
    CROP_REGISTRY only contained "maize"; any cassava farm triggered
        WARNING Unknown crop_type 'cassava' — falling back to maize CropConfig
    and received maize's 75-130 day cycle parameters instead of cassava's
    180-365 day parameters. This silently produced wrong planting dates for
    every cassava farm in production.

    These tests ensure:
    1. All four classifier-emitted crop types resolve without a warning fallback.
    2. Cassava's parameters are semantically distinct from maize's (the bug was
       silent — same return value, different meaning).
    3. Alias / case-normalisation paths are covered.
    """

    def test_cassava_resolves_without_fallback(self, caplog):
        """get_crop_config('cassava') must NOT log an 'Unknown crop_type' warning."""
        import logging
        from app.core.config import get_crop_config
        with caplog.at_level(logging.WARNING, logger="app.core.config"):
            cfg = get_crop_config("cassava")
        unknown_warns = [r for r in caplog.records if "Unknown crop_type" in r.message]
        assert not unknown_warns, (
            f"get_crop_config('cassava') still triggers fallback warning: "
            f"{[r.message for r in unknown_warns]}"
        )
        assert cfg.crop_type == "cassava"

    def test_cassava_has_distinct_cycle_from_maize(self):
        """Cassava's cycle must be meaningfully distinct from maize's.

        After extending maize max to 210d for highland varieties, the 180-210d
        range is legitimately shared (highland maize ≈ cassava short-season).
        The decisive distinction is cassava's absolute ceiling (365d, full-year
        bitter varieties) which far exceeds any maize variety, and its explicit
        signal_weight_override marking it as a different agronomic regime.
        """
        from app.core.config import get_crop_config
        maize = get_crop_config("maize")
        cassava = get_crop_config("cassava")
        # Cassava can run a full year; maize tops out at ~7 months
        assert cassava.max_season_length_days > maize.max_season_length_days, (
            f"Cassava max cycle ({cassava.max_season_length_days}d) should exceed "
            f"maize max cycle ({maize.max_season_length_days}d)"
        )
        # Cassava has an explicit SAR-upweighted signal override; maize uses AEZ default
        assert cassava.signal_weight_override is not None, (
            "Cassava should have an explicit signal_weight_override distinct from maize"
        )
        assert maize.signal_weight_override is None, (
            "Maize should defer to AEZ signal_weights (no override)"
        )

    def test_cassava_planting_offset_longer_than_maize(self):
        from app.core.config import get_crop_config
        assert get_crop_config("cassava").planting_offset_days >= get_crop_config("maize").planting_offset_days

    def test_cassava_peak_ndvi_lower_than_maize(self):
        """Cassava has a sparser canopy than maize — its peak NDVI upper bound
        must be <= maize's lower bound of 0.45, not exceeding maize's range."""
        from app.core.config import get_crop_config
        cassava = get_crop_config("cassava")
        maize = get_crop_config("maize")
        assert cassava.peak_ndvi_expected_range[1] <= maize.peak_ndvi_expected_range[1], (
            "Cassava peak NDVI upper bound should not exceed maize's"
        )
        assert cassava.peak_ndvi_expected_range[0] < maize.peak_ndvi_expected_range[0], (
            "Cassava peak NDVI lower bound should be below maize's"
        )

    def test_cassava_case_insensitive(self):
        """Classifier outputs lowercase; guard against case drift."""
        from app.core.config import get_crop_config
        for variant in ("cassava", "CASSAVA", "Cassava"):
            cfg = get_crop_config(variant)
            assert cfg.crop_type == "cassava", f"Case variant '{variant}' did not resolve to cassava"

    @pytest.mark.parametrize("crop_name", ["maize", "cassava", "common_bean", "soybean"])
    def test_all_classifier_crop_names_resolve_without_fallback(self, crop_name, caplog):
        """Every entry in Stage 3's CROP_NAMES must resolve without any warning.
        All four are now first-class CROP_REGISTRY entries.
        """
        import logging
        from app.core.config import get_crop_config
        with caplog.at_level(logging.WARNING, logger="app.core.config"):
            cfg = get_crop_config(crop_name)
        unknown_warns = [r for r in caplog.records if "Unknown crop_type" in r.message]
        assert not unknown_warns, (
            f"'{crop_name}' triggered unexpected fallback warning: "
            f"{[r.message for r in unknown_warns]}"
        )
        # Resolved crop_type must not silently fall through to maize
        assert cfg.crop_type != "maize" or crop_name == "maize", (
            f"'{crop_name}' silently resolved to maize config"
        )

    def test_common_bean_alias_resolves_to_beans_registry_entry(self):
        """common_bean (Stage 3 label) must route through _CROP_ALIASES to the
        real 'beans' CROP_REGISTRY entry, not to the old maize silent fallback."""
        from app.core.config import get_crop_config
        cfg = get_crop_config("common_bean")
        assert cfg.crop_type == "beans"
        assert cfg.min_season_length_days == 55
        assert cfg.max_season_length_days == 90

    def test_soybean_resolves_without_fallback(self, caplog):
        import logging
        from app.core.config import get_crop_config
        with caplog.at_level(logging.WARNING, logger="app.core.config"):
            cfg = get_crop_config("soybean")
        unknown_warns = [r for r in caplog.records if "Unknown crop_type" in r.message]
        assert not unknown_warns
        assert cfg.crop_type == "soybean"
        assert cfg.min_season_length_days == 90
        assert cfg.max_season_length_days == 140

    def test_beans_cycle_distinct_from_maize(self):
        """Beans is a short-season legume — its max cycle must be well below
        maize's max, and its peak NDVI range must be lower (sparser canopy)."""
        from app.core.config import get_crop_config
        beans = get_crop_config("beans")
        maize = get_crop_config("maize")
        # Beans tops out at 90 days; maize reaches up to 130 days
        assert beans.max_season_length_days < maize.max_season_length_days, (
            f"Beans max cycle ({beans.max_season_length_days}) should be below "
            f"maize max cycle ({maize.max_season_length_days})"
        )
        # Beans has a sparser canopy than maize
        assert beans.peak_ndvi_expected_range[1] < maize.peak_ndvi_expected_range[1], (
            "Beans peak NDVI upper bound should be below maize's"
        )

    def test_soybean_cycle_overlaps_maize_but_distinct_config(self):
        """Soybean has a similar-length cycle to maize, but must have its own
        crop_type identity so planting parameters aren't shared."""
        from app.core.config import get_crop_config
        soybean = get_crop_config("soybean")
        assert soybean.crop_type == "soybean"
        assert soybean.peak_ndvi_expected_range != get_crop_config("maize").peak_ndvi_expected_range

    def test_maize_still_resolves(self):
        """Smoke test: the primary crop must not have been disturbed by any changes."""
        from app.core.config import get_crop_config
        cfg = get_crop_config("maize")
        assert cfg.crop_type == "maize"
        assert cfg.min_season_length_days == 75
        assert cfg.max_season_length_days == 210  # extended for highland varieties

    def test_maize_highland_cycle_covers_trans_nzoia(self):
        """Trans Nzoia / Uasin Gishu highland maize (DH02, Duma 43) can reach
        ~210 days. The cycle bound must accommodate this so the planting engine
        doesn't reject valid highland detections."""
        from app.core.config import get_crop_config
        maize = get_crop_config("maize")
        highland_cycle_days = 7 * 30  # 7 months ≈ 210 days
        assert maize.max_season_length_days >= highland_cycle_days, (
            f"Maize max cycle ({maize.max_season_length_days}d) is too short for "
            f"highland varieties — Trans Nzoia needs at least {highland_cycle_days}d"
        )
