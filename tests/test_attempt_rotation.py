"""
Stage 4 must attempt EVERY eligible farm (2026-11-11 fix).

Before: one batch of 500 per run, `ORDER BY tile_id, farm_uuid`, and farms
with no usable signal were never recorded as tried — so the same 500 came
back first on every run and every farm sorting after them (e.g. Machakos,
tiles 37M*) was never processed.

These tests drive run_batch_pipeline() against an in-memory stand-in for
DatacubeClient that applies the same selection rules as _ELIGIBLE_FARMS_SQL
(attempt cut-off, upload priority, never-attempted first). The SQL itself is
exercised against real Postgres separately; here we check the loop.
"""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import scripts.run_datacube_batch as batch
from app.core.models import FarmRow, SeasonResult
from scripts.run_datacube_batch import _attempt_outcomes


def _farm(i: int, tile: str) -> FarmRow:
    return FarmRow(
        farm_uuid=f"00000000-0000-0000-0000-{i:012d}",
        geom_wkt="POINT(36.8 -1.3)",
        tile_id=tile,
        aez_code="LM 4",
        crop_type="maize",
        centroid_lat=-1.3,
        centroid_lon=36.8,
    )


def _result(method="ensemble", planting=date(2026, 10, 25), error=None) -> SeasonResult:
    return SeasonResult(
        polygon_id="p", season="short_rains", year=2026,
        estimated_planting_date=planting, method_used=method, error=error, confidence=0.7,
    )


class FakeDC:
    """In-memory DatacubeClient: same eligibility semantics as the SQL."""

    def __init__(self, farms, upload_of=None):
        self.farms = farms
        self.upload_of = upload_of or {}
        self.attempted_at = {}   # farm_uuid -> datetime
        self.detected = set()
        self.clock = datetime(2026, 11, 11, tzinfo=timezone.utc)
        self.selected_batches = []
        self.outcomes = {}
        self._pool = SimpleNamespace(getconn=lambda: None, putconn=lambda c: None)

    def _tick(self):
        self.clock += timedelta(seconds=1)
        return self.clock

    def get_db_now(self):
        return self._tick()

    def get_eligible_farms(self, season, year, batch_size, retry_before, priority_upload_id=None):
        pool = [
            f for f in self.farms
            if f.farm_uuid not in self.detected
            and (f.farm_uuid not in self.attempted_at or self.attempted_at[f.farm_uuid] < retry_before)
        ]
        pool.sort(key=lambda f: (
            0 if priority_upload_id and self.upload_of.get(f.farm_uuid) == priority_upload_id else 1,
            (0, datetime.min.replace(tzinfo=timezone.utc)) if f.farm_uuid not in self.attempted_at
            else (1, self.attempted_at[f.farm_uuid]),
            f.tile_id, f.farm_uuid,
        ))
        out = pool[:batch_size]
        self.selected_batches.append([f.farm_uuid for f in out])
        return out

    def upsert_planting_results(self, results, dry_run=False):
        if dry_run:
            return len(results)
        now = self._tick()
        for fuid, _ in results:
            self.detected.add(fuid)
            self.attempted_at[fuid] = now
            self.outcomes[fuid] = "detected"
        return len(results)

    def record_attempts(self, attempts, season, year, dry_run=False):
        if dry_run:
            return len(attempts)
        now = self._tick()
        for fuid, outcome in attempts:
            self.attempted_at[fuid] = now
            self.outcomes[fuid] = outcome
        return len(attempts)

    def close(self):
        pass


@pytest.fixture
def wire(monkeypatch):
    """Replace STAC fetchers, per-farm detection and downstream chaining."""
    monkeypatch.setattr(batch, "Sentinel2STACFetcher", lambda conn: SimpleNamespace(fetch_for_tile=lambda *a: None))
    monkeypatch.setattr(batch, "Sentinel1STACFetcher", lambda conn: SimpleNamespace(fetch_for_tile=lambda *a: None))
    monkeypatch.setattr(batch, "_tile_fetch_window", lambda farms, s, y: (date(2026, 8, 1), date(2027, 1, 31)))
    monkeypatch.setattr(batch, "_trigger_downstream", lambda dry_run: None)
    monkeypatch.delenv("UPLOAD_ID", raising=False)

    def install(dc, detect=lambda farm: _result(method="fallback_climatology")):
        monkeypatch.setattr(batch, "DatacubeClient", lambda: dc)
        monkeypatch.setattr(batch, "_process_farm", lambda farm, s, y, d: (farm.farm_uuid, detect(farm)))
        return dc

    return install


def _run(**kw):
    args = dict(batch_size=500, season_str="short_rains", year=2026, workers=2, dry_run=False)
    args.update(kw)
    return batch.run_batch_pipeline(**args)


class TestEveryFarmIsAttempted:
    def test_no_signal_farms_do_not_starve_the_rest(self, wire):
        # 500 no-signal farms that sort first (the real 11-10/11-11 batch),
        # plus 100 Machakos farms on tiles that sort after them.
        blockers = [_farm(i, "MGRS-36NYF") for i in range(500)]
        machakos = [_farm(1000 + i, "MGRS-37MBU") for i in range(100)]
        dc = wire(FakeDC(blockers + machakos))

        _run(batch_size=500)

        assert len(dc.selected_batches) >= 2, "must keep going after the first 500"
        attempted = set(dc.attempted_at)
        assert {f.farm_uuid for f in machakos} <= attempted, "Machakos farms were never attempted"
        assert len(attempted) == 600

    def test_each_farm_attempted_exactly_once_per_run(self, wire):
        farms = [_farm(i, f"T{i % 7}") for i in range(1234)]
        dc = wire(FakeDC(farms))

        _run(batch_size=100)

        flat = [u for b in dc.selected_batches for u in b]
        assert len(flat) == len(set(flat)) == 1234
        assert dc.selected_batches[-1] == [], "loop ends on an empty batch"

    def test_triggering_upload_goes_first(self, wire, monkeypatch):
        others = [_farm(i, "MGRS-11SKA") for i in range(50)]
        mine = [_farm(900 + i, "MGRS-37MBU") for i in range(10)]
        upload = "1eddae1a-17df-4bf3-955f-89b5b0a4c792"
        dc = wire(FakeDC(others + mine, upload_of={f.farm_uuid: upload for f in mine}))
        monkeypatch.setenv("UPLOAD_ID", upload)

        _run(batch_size=10)

        assert set(dc.selected_batches[0]) == {f.farm_uuid for f in mine}


class TestCooldownAndRotation:
    def test_second_run_within_cooldown_rescans_nothing(self, wire):
        farms = [_farm(i, "T") for i in range(30)]
        dc = wire(FakeDC(farms))
        _run(batch_size=10)
        dc.selected_batches.clear()

        _run(batch_size=10, retry_cooldown_hours=24)

        assert dc.selected_batches == [[]]

    def test_new_farm_is_processed_even_within_cooldown(self, wire):
        farms = [_farm(i, "T") for i in range(30)]
        dc = wire(FakeDC(farms))
        _run(batch_size=10)
        new = _farm(5000, "MGRS-37MCU")
        dc.farms.append(new)
        dc.selected_batches.clear()

        _run(batch_size=10, retry_cooldown_hours=24)

        assert dc.selected_batches[0] == [new.farm_uuid]

    def test_max_farms_caps_a_run(self, wire):
        dc = wire(FakeDC([_farm(i, "T") for i in range(50)]))
        _run(batch_size=20, max_farms=30)
        assert sum(len(b) for b in dc.selected_batches) == 30


class TestOutcomesAndFailures:
    def test_detections_are_not_re_recorded_as_attempts(self, wire):
        farms = [_farm(i, "T") for i in range(4)]
        good = {farms[0].farm_uuid, farms[1].farm_uuid}
        dc = wire(FakeDC(farms), detect=lambda f: _result() if f.farm_uuid in good else _result(method="fallback_climatology"))

        _run(batch_size=10)

        assert dc.outcomes[farms[0].farm_uuid] == "detected"
        assert dc.outcomes[farms[2].farm_uuid] == "no_signal"
        assert dc.detected == good

    def test_failing_tile_does_not_stop_the_run(self, wire, monkeypatch):
        bad = [_farm(i, "BAD") for i in range(5)]
        ok = [_farm(100 + i, "OK") for i in range(5)]
        dc = wire(FakeDC(bad + ok))

        def fetcher(conn):
            def fetch(tile_id, *a):
                if tile_id == "BAD":
                    raise RuntimeError("STAC 503")
            return SimpleNamespace(fetch_for_tile=fetch)

        monkeypatch.setattr(batch, "Sentinel2STACFetcher", fetcher)

        code = _run(batch_size=100)

        assert code == 1, "partial failure is reported"
        assert all(dc.outcomes[f.farm_uuid] == "error" for f in bad)
        assert all(dc.outcomes[f.farm_uuid] == "no_signal" for f in ok)

    def test_dry_run_stops_after_one_batch(self, wire):
        dc = wire(FakeDC([_farm(i, "T") for i in range(30)]))
        _run(batch_size=10, dry_run=True)
        assert len(dc.selected_batches) == 1
        assert dc.attempted_at == {}


class TestAttemptOutcomes:
    def test_classification(self):
        uuids = ["a", "b", "c", "d", "e"]
        results = [
            ("a", _result()),
            ("b", _result(method="fallback_climatology")),
            ("c", _result(planting=None)),
            ("d", _result(error="boom")),
            # "e" raised in its worker: no result at all
        ]
        ok = [("a", results[0][1])]
        assert _attempt_outcomes(uuids, results, ok) == [
            ("b", "no_signal"), ("c", "empty"), ("d", "error"), ("e", "error"),
        ]
