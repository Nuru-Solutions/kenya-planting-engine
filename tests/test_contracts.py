"""
tests/test_contracts.py
========================
Pre-push contract tests for Stage 4 (Planting Date Engine).

These guard the specific "pipeline flows as planned" invariants that the
existing algorithm-level tests (test_all.py) don't cover, because the bugs
they catch live in orchestration code (season selection, what gets
persisted, tile-level fetch windows, downstream chaining) rather than in
the NDVI/rainfall/SAR maths.

Every test here is meant to be RED against the code as of commit
7086bd1 (webhook chaining) and GREEN after the fixes in this branch.
Run: pytest tests/test_contracts.py -v
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from unittest import mock

import pytest

from app.core.config import (
    KENYA_AEZ_REGISTRY, DEFAULT_AEZ, get_aez_config, Season,
)
from app.core.models import FarmRow, SeasonResult
from app.core.pipeline import resolve_season
from scripts.run_datacube_batch import (
    _auto_detect_season,
    _persistable_results,
    _tile_fetch_window,
    _trigger_downstream,
)
from app.data.stac_fetcher import _sample_items_evenly


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _mk_result(method_used="ensemble", planting_date=date(2026, 3, 1), error=None) -> SeasonResult:
    return SeasonResult(
        polygon_id="p1", season="long_rains", year=2026,
        estimated_planting_date=planting_date,
        method_used=method_used,
        error=error,
    )


def _mk_farm(aez_code: str, tile_id: str = "37MBU") -> FarmRow:
    return FarmRow(
        farm_uuid="00000000-0000-0000-0000-000000000001",
        geom_wkt="POINT(36.8 -1.3)",
        tile_id=tile_id,
        aez_code=aez_code,
        crop_type="maize",
        centroid_lat=-1.3,
        centroid_lon=36.8,
    )


# ── Season auto-detect: never point at a season that hasn't started ────────────

class TestAutoDetectSeason:
    def test_january_short_rains_uses_previous_year(self):
        """
        Regression guard for the incident that motivated this file: in
        January, the just-finished Short Rains season started last
        October, so it must be paired with year-1 — not the current year
        (which would compute an Oct-Dec window still ~9 months away).
        """
        season, year = _auto_detect_season(date(2026, 1, 15))
        assert season == "short_rains"
        assert year == 2025

    def test_october_short_rains_uses_current_year(self):
        season, year = _auto_detect_season(date(2026, 10, 15))
        assert season == "short_rains"
        assert year == 2026

    @pytest.mark.parametrize("month", [2, 3, 4, 5, 6, 7, 8, 9])
    def test_long_rains_months_use_current_year(self, month):
        season, year = _auto_detect_season(date(2026, month, 15))
        assert season == "long_rains"
        assert year == 2026

    def test_never_selects_a_window_that_has_not_opened(self):
        """
        For every day of a full year, the auto-detected (season, year) must
        resolve, against DEFAULT_AEZ, to a window whose search_start has
        already arrived. Picking a not-yet-open window means the batch run
        has zero real satellite/rainfall data available and every farm
        falls back to the calendar guess (see _persistable_results).
        """
        start = date(2026, 1, 1)
        for offset in range(0, 365, 3):  # sample every 3rd day — full-year coverage, fast
            today = start + timedelta(days=offset)
            season_str, year = _auto_detect_season(today)
            season, window = resolve_season(season_str, year, DEFAULT_AEZ)
            search_start = window.get_search_start(year)
            assert search_start <= today, (
                f"{today}: auto-detected {season_str} {year} has search_start="
                f"{search_start}, which is in the future"
            )

    def test_requesting_third_season_on_non_highland_falls_back_to_long_rains(self):
        """
        Documents existing (intentional) behavior: third_season is only
        defined for the Highland AEZ (99.0). Elsewhere resolve_season()
        relabels the result as long_rains rather than silently tagging a
        long-rains detection as "third_season". If this ever changes it
        should be a deliberate decision, not a silent regression.
        """
        season, _ = resolve_season("third_season", 2026, get_aez_config("UM 3"))
        assert season == Season.LONG_RAINS

        season, _ = resolve_season("third_season", 2026, get_aez_config("LH 2"))
        assert season == Season.THIRD_SEASON


# ── Persistence filter: calendar-only guesses must never reach the DB ──────────

class TestPersistableResults:
    def test_pure_fallback_climatology_is_excluded(self):
        results = [("f1", _mk_result(method_used="fallback_climatology"))]
        ok, skipped, clim_only = _persistable_results(results)
        assert ok == []
        assert skipped == 1
        assert clim_only == 1

    def test_real_detection_is_included(self):
        for method in ["ensemble", "ensemble_2signal", "rainfall_only", "ndvi_only", "sar_only"]:
            results = [("f1", _mk_result(method_used=method))]
            ok, skipped, clim_only = _persistable_results(results)
            assert len(ok) == 1, f"method={method} should be persisted"
            assert skipped == 0
            assert clim_only == 0

    def test_clim_blend_with_real_signal_is_still_included(self):
        """
        A *_clim_blend result still has at least one real signal behind it
        (unlike pure fallback_climatology) — it's kept, just flagged
        separately in the summary log, not silently dropped.
        """
        results = [("f1", _mk_result(method_used="rainfall_only_clim_blend"))]
        ok, skipped, clim_only = _persistable_results(results)
        assert len(ok) == 1
        assert clim_only == 0

    def test_errored_result_is_excluded(self):
        results = [("f1", _mk_result(error="boom"))]
        ok, skipped, clim_only = _persistable_results(results)
        assert ok == []
        assert skipped == 1
        assert clim_only == 0

    def test_empty_date_is_excluded(self):
        results = [("f1", _mk_result(planting_date=None, method_used="no_data"))]
        ok, skipped, clim_only = _persistable_results(results)
        assert ok == []

    def test_mixed_batch_only_keeps_real_detections(self):
        results = [
            ("f1", _mk_result(method_used="ensemble")),
            ("f2", _mk_result(method_used="fallback_climatology")),
            ("f3", _mk_result(error="boom")),
            ("f4", _mk_result(method_used="ndvi_only")),
        ]
        ok, skipped, clim_only = _persistable_results(results)
        assert {fuid for fuid, _ in ok} == {"f1", "f4"}
        assert skipped == 2
        assert clim_only == 1


# ── Tile fetch window: must cover every AEZ present, not just farm[0] ──────────

class TestTileFetchWindow:
    def test_single_aez_matches_its_own_window(self):
        farms = [_mk_farm("UM 3")]
        fetch_start, win_end = _tile_fetch_window(farms, "long_rains", 2026)
        _, expected_window = resolve_season("long_rains", 2026, get_aez_config("UM 3"))
        assert fetch_start == expected_window.get_search_start(2026) - timedelta(days=60)
        assert win_end == expected_window.get_window(2026)[1]

    def test_mixed_aez_tile_covers_the_union(self):
        """
        UM3 long rains ends 2026-05-31; LM4 long rains ends 2026-06-15.
        A tile mixing both AEZs must fetch through the LATER end date so
        LM4 farms aren't starved of their in-season NDVI/SAR observations.
        """
        farms = [_mk_farm("UM 3"), _mk_farm("LM 4")]
        fetch_start, win_end = _tile_fetch_window(farms, "long_rains", 2026)

        _, um3_window = resolve_season("long_rains", 2026, get_aez_config("UM 3"))
        _, lm4_window = resolve_season("long_rains", 2026, get_aez_config("LM 4"))

        assert fetch_start <= um3_window.get_search_start(2026) - timedelta(days=60)
        assert fetch_start <= lm4_window.get_search_start(2026) - timedelta(days=60)
        assert win_end >= um3_window.get_window(2026)[1]
        assert win_end >= lm4_window.get_window(2026)[1]
        # LM4 ends later — confirm the union actually picked it up, not UM3's.
        assert win_end == lm4_window.get_window(2026)[1]


# ── STAC item sampling: truncation must not drop the back half of a season ─────

class TestStacSampling:
    class _FakeItem:
        def __init__(self, i):
            self.i = i
        def __repr__(self):
            return f"Item({self.i})"

    def test_under_the_cap_is_unchanged(self):
        items = [self._FakeItem(i) for i in range(10)]
        assert _sample_items_evenly(items, 20) == items

    def test_over_the_cap_spans_the_full_range(self):
        items = [self._FakeItem(i) for i in range(90)]  # ~150-day season @ 5-day revisit
        sampled = _sample_items_evenly(items, 20)
        assert len(sampled) <= 20
        assert sampled[0].i == 0        # earliest scene kept
        assert sampled[-1].i == 89      # LATEST scene kept — this is what [:20] used to drop
        # Roughly even spread: no gap much larger than n/max_items
        idxs = [it.i for it in sampled]
        max_gap = max(b - a for a, b in zip(idxs, idxs[1:]))
        assert max_gap <= (90 / 20) * 2

    def test_never_exceeds_max_items(self):
        items = [self._FakeItem(i) for i in range(1000)]
        assert len(_sample_items_evenly(items, 20)) <= 20

    def test_never_fabricates_items(self):
        items = [self._FakeItem(i) for i in range(37)]
        sampled = _sample_items_evenly(items, 20)
        assert all(it in items for it in sampled)


# ── Downstream chaining: no duplicate webhook/Stage-5 fires, no weak secret ────

class TestTriggerDownstream:
    @mock.patch("urllib.request.urlopen")
    @mock.patch("boto3.client")
    def test_skips_on_batch_retry_attempt(self, mock_boto_client, mock_urlopen, monkeypatch):
        monkeypatch.setenv("AWS_BATCH_JOB_ATTEMPT", "2")
        monkeypatch.setenv("UPLOAD_ID", "abc123")
        monkeypatch.setenv("GPS_RESULT_WEBHOOK_SECRET", "real-secret")

        _trigger_downstream(dry_run=False)

        mock_urlopen.assert_not_called()
        mock_boto_client.assert_not_called()

    @mock.patch("urllib.request.urlopen")
    @mock.patch("boto3.client")
    def test_skips_webhook_without_secret_but_does_not_use_a_hardcoded_default(
        self, mock_boto_client, mock_urlopen, monkeypatch
    ):
        monkeypatch.delenv("AWS_BATCH_JOB_ATTEMPT", raising=False)
        monkeypatch.setenv("UPLOAD_ID", "abc123")
        monkeypatch.delenv("GPS_RESULT_WEBHOOK_SECRET", raising=False)
        mock_boto_client.return_value.invoke.return_value = {}

        _trigger_downstream(dry_run=False)

        mock_urlopen.assert_not_called()  # never falls back to a known/committed secret

    @mock.patch("urllib.request.urlopen")
    @mock.patch("boto3.client")
    def test_fires_both_on_first_attempt_with_secret_set(
        self, mock_boto_client, mock_urlopen, monkeypatch
    ):
        monkeypatch.setenv("AWS_BATCH_JOB_ATTEMPT", "1")
        monkeypatch.setenv("UPLOAD_ID", "abc123")
        monkeypatch.setenv("GPS_RESULT_WEBHOOK_SECRET", "real-secret")
        mock_urlopen.return_value.__enter__.return_value.status = 200
        mock_boto_client.return_value.invoke.return_value = {}

        _trigger_downstream(dry_run=False)

        mock_urlopen.assert_called_once()
        mock_boto_client.return_value.invoke.assert_called_once()

    @mock.patch("urllib.request.urlopen")
    @mock.patch("boto3.client")
    def test_dry_run_never_calls_either(self, mock_boto_client, mock_urlopen, monkeypatch):
        monkeypatch.delenv("AWS_BATCH_JOB_ATTEMPT", raising=False)
        monkeypatch.setenv("UPLOAD_ID", "abc123")
        monkeypatch.setenv("GPS_RESULT_WEBHOOK_SECRET", "real-secret")

        _trigger_downstream(dry_run=True)

        mock_urlopen.assert_not_called()
        mock_boto_client.assert_not_called()
