"""
app/data/sentinel2.py
Sentinel-2 L2A NDVI/EVI/NDWI time series fetcher.

Cloud masking:
  SCL bands (cloud, shadow, cirrus) + s2cloudless probability < 35%

Indices:
  NDVI = (B8 - B4) / (B8 + B4)
  EVI  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
  NDWI = (B3 - B8) / (B3 + B8)   ← soil moisture / water body proxy
"""
from __future__ import annotations
import logging
from datetime import date

from app.data.gee_client import init_gee
from app.core.config import get_settings
from app.core.models import FarmPolygon, NDVIObservation

logger = logging.getLogger(__name__)


class Sentinel2Fetcher:

    COLLECTION       = "COPERNICUS/S2_SR_HARMONIZED"
    CLOUD_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"

    def __init__(self):
        init_gee()
        self.settings = get_settings()

    def fetch(
        self,
        polygon: FarmPolygon,
        start_date: date,
        end_date: date,
    ) -> list[NDVIObservation]:
        from app.data.cache import get as cache_get, put as cache_put

        # ── Cache check ────────────────────────────────────────────────────────
        cached = cache_get("s2", polygon.polygon_id, start_date, end_date)
        if cached is not None:
            return [NDVIObservation.model_validate(d) for d in cached]

        # ── Fetch from GEE ─────────────────────────────────────────────────────
        import ee

        geometry = ee.Geometry(polygon.geometry)

        s2 = (
            ee.ImageCollection(self.COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.settings.max_cloud_cover))
        )
        cp = (
            ee.ImageCollection(self.CLOUD_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date.isoformat(), end_date.isoformat())
        )

        joined = ee.ImageCollection(
            ee.Join.saveFirst("cp").apply(
                primary=s2, secondary=cp,
                condition=ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            )
        )

        def mask_and_index(img):
            scl = img.select("SCL")
            scl_mask = (
                scl.neq(1).And(scl.neq(3)).And(scl.neq(8))
                   .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            )
            cp_mask = ee.Image(img.get("cp")).select("probability").lt(35)
            mask = scl_mask.And(cp_mask)

            b2 = img.select("B2").divide(10000)
            b3 = img.select("B3").divide(10000)
            b4 = img.select("B4").divide(10000)
            b8 = img.select("B8").divide(10000)

            ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
            evi  = (b8.subtract(b4)
                      .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
                      .multiply(2.5).rename("EVI"))
            ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("NDWI")

            return (img.addBands([ndvi, evi, ndwi])
                       .updateMask(mask)
                       .copyProperties(img, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]))

        def reduce(img):
            stats = img.select(["NDVI", "EVI", "NDWI"]).reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                geometry=geometry, scale=10, maxPixels=1e9, bestEffort=True,
            )
            return ee.Feature(None, stats.set(
                "date", img.date().format("YYYY-MM-dd")
            ).set(
                "cloud_pct", img.get("CLOUDY_PIXEL_PERCENTAGE")
            ))

        features = joined.map(mask_and_index).map(reduce).getInfo()["features"]

        obs = []
        for f in features:
            p = f["properties"]
            if p.get("NDVI_mean") is None:
                continue
            try:
                obs.append(NDVIObservation(
                    obs_date      = date.fromisoformat(p["date"]),
                    ndvi          = round(float(p["NDVI_mean"]), 4),
                    evi           = round(float(p["EVI_mean"]),  4) if p.get("EVI_mean")  is not None else None,
                    ndwi          = round(float(p["NDWI_mean"]), 4) if p.get("NDWI_mean") is not None else None,
                    cloud_cover_pct = round(float(p.get("cloud_pct") or 0), 1),
                    pixel_count   = int(p.get("NDVI_count") or 0),
                ))
            except (TypeError, ValueError) as e:
                logger.warning(f"S2 skip: {e}")

        obs.sort(key=lambda x: x.obs_date)
        logger.debug(f"{polygon.polygon_id}: {len(obs)} S2 obs")

        # ── Write to cache ─────────────────────────────────────────────────────
        cache_put("s2", polygon.polygon_id, start_date, end_date, obs)
        return obs
