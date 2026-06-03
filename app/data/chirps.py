"""
app/data/chirps.py
CHIRPS v2.0 daily rainfall fetcher.
Best rainfall product for East Africa — blends IR + gauge data.
Resolution: ~5.5km  |  Latency: ~2-3 weeks after observation date
"""
from __future__ import annotations
import logging
from datetime import date, timedelta

from app.data.gee_client import init_gee
from app.core.models import FarmPolygon, RainfallRecord

logger = logging.getLogger(__name__)


class CHIRPSFetcher:

    COLLECTION = "UCSB-CHG/CHIRPS/DAILY"

    def __init__(self):
        init_gee()

    def fetch(
        self,
        polygon: FarmPolygon,
        start_date: date,
        end_date: date,
    ) -> list[RainfallRecord]:
        from app.data.cache import get as cache_get, put as cache_put

        # ── Cache check ────────────────────────────────────────────────────────
        cached = cache_get("chirps", polygon.polygon_id, start_date, end_date)
        if cached is not None:
            records = [RainfallRecord.model_validate(d) for d in cached]
            return self._fill_gaps(records, start_date, end_date)

        # ── Fetch from GEE ─────────────────────────────────────────────────────
        import ee

        geometry = ee.Geometry(polygon.geometry)
        chirps = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .select("precipitation")
        )

        def reduce(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry.centroid(), scale=5566, maxPixels=1e6, bestEffort=True,
            )
            return ee.Feature(None, stats.set("date", img.date().format("YYYY-MM-dd")))

        features = chirps.map(reduce).getInfo()["features"]

        records = []
        for f in features:
            p = f["properties"]
            try:
                records.append(RainfallRecord(
                    record_date = date.fromisoformat(p["date"]),
                    rainfall_mm = round(max(0.0, float(p.get("precipitation") or 0.0)), 2),
                    source      = "CHIRPS_v2",
                ))
            except (TypeError, ValueError) as e:
                logger.warning(f"CHIRPS skip: {e}")

        records.sort(key=lambda x: x.record_date)

        # ── Write to cache (raw records before gap-fill so cache stays clean) ───
        cache_put("chirps", polygon.polygon_id, start_date, end_date, records)

        records = self._fill_gaps(records, start_date, end_date)
        logger.debug(f"{polygon.polygon_id}: {len(records)} CHIRPS records")
        return records

    def _fill_gaps(self, records: list[RainfallRecord], start: date, end: date) -> list[RainfallRecord]:
        rmap = {r.record_date: r for r in records}
        out, cur = [], start
        while cur <= end:
            out.append(rmap.get(cur, RainfallRecord(
                record_date=cur, rainfall_mm=0.0, source="CHIRPS_gap"
            )))
            cur += timedelta(days=1)
        return out

    def seasonal_total(self, records: list[RainfallRecord]) -> float:
        return round(sum(r.rainfall_mm for r in records), 1)
