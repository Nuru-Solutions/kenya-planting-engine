"""
app/data/sentinel1.py
Sentinel-1 GRD IW VV+VH backscatter fetcher.

Processing:
  - Both ASCENDING and DESCENDING passes
  - 3×3 focal mean speckle filter
  - Cross-pol ratio: VH − VV (dB) — sensitive to crop volume scattering
  - Values in dB (GEE S1 collection is already log-scaled)
"""
from __future__ import annotations
import logging
from datetime import date

from app.data.gee_client import init_gee
from app.core.models import FarmPolygon, SARObservation

logger = logging.getLogger(__name__)


class Sentinel1Fetcher:

    COLLECTION = "COPERNICUS/S1_GRD"

    def __init__(self):
        init_gee()

    def fetch(
        self,
        polygon: FarmPolygon,
        start_date: date,
        end_date: date,
    ) -> list[SARObservation]:
        from app.data.cache import get as cache_get, put as cache_put

        # ── Cache check ────────────────────────────────────────────────────────
        cached = cache_get("s1", polygon.polygon_id, start_date, end_date)
        if cached is not None:
            return [SARObservation.model_validate(d) for d in cached]

        # ── Fetch from GEE ─────────────────────────────────────────────────────
        import ee

        geometry   = ee.Geometry(polygon.geometry)
        start_str  = start_date.isoformat()
        end_str    = end_date.isoformat()

        def load_pass(direction: str) -> ee.ImageCollection:
            return (
                ee.ImageCollection(self.COLLECTION)
                .filterBounds(geometry)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("orbitProperties_pass", direction))
                .select(["VV", "VH"])
            )

        def filter_and_reduce(img):
            vv    = img.select("VV").focal_mean(3, "square", "pixels").rename("VV_f")
            vh    = img.select("VH").focal_mean(3, "square", "pixels").rename("VH_f")
            ratio = vh.subtract(vv).rename("cross_pol")
            stats = img.addBands([vv, vh, ratio]).select(["VV_f", "VH_f", "cross_pol"]).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry, scale=10, maxPixels=1e9, bestEffort=True,
            )
            return ee.Feature(None, stats.set("date", img.date().format("YYYY-MM-dd")))

        obs = []
        for direction in ["ASCENDING", "DESCENDING"]:
            try:
                features = load_pass(direction).map(filter_and_reduce).getInfo()["features"]
            except Exception as e:
                logger.warning(f"SAR {direction} failed {polygon.polygon_id}: {e}")
                continue
            for f in features:
                p = f["properties"]
                if p.get("VV_f") is None:
                    continue
                try:
                    obs.append(SARObservation(
                        obs_date       = date.fromisoformat(p["date"]),
                        vv_db          = round(float(p["VV_f"]),    3),
                        vh_db          = round(float(p["VH_f"]),    3) if p.get("VH_f")    is not None else None,
                        cross_pol_ratio= round(float(p["cross_pol"]),3) if p.get("cross_pol") is not None else None,
                        pass_direction = direction,
                    ))
                except (TypeError, ValueError) as e:
                    logger.warning(f"SAR skip: {e}")

        obs.sort(key=lambda x: x.obs_date)
        logger.debug(f"{polygon.polygon_id}: {len(obs)} SAR obs")

        # ── Write to cache ─────────────────────────────────────────────────────
        cache_put("s1", polygon.polygon_id, start_date, end_date, obs)
        return obs
