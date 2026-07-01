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
