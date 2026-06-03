"""
app/core/exporter.py
Converts MultiSeasonJobResult into GeoJSON, CSV, and JSON outputs.

The JSON output (farm_histories) is the primary feed for the crop ID pipeline —
it contains full phenological profiles per farm per season.
"""
from __future__ import annotations
import csv
import io
import json
from datetime import date
from typing import Any

from app.core.models import MultiSeasonJobResult, FarmSeasonHistory, SeasonResult


def _str(v) -> str:
    return str(v) if v is not None else ""


# ── GeoJSON ────────────────────────────────────────────────────────────────────

def to_geojson_flat(job: MultiSeasonJobResult) -> dict:
    """
    One Feature per polygon × season (flat).
    Each feature has all planting date fields + key phenology fields.
    """
    features = []
    for hist in job.farm_histories:
        for sr in hist.seasons:
            if not sr.geometry:
                continue
            features.append({
                "type": "Feature",
                "geometry": sr.geometry,
                "properties": _season_flat_props(sr),
            })

    return {
        "type": "FeatureCollection",
        "properties": {
            "job_id":          str(job.job_id),
            "years_processed": job.years_processed,
            "total_polygons":  job.total_polygons,
            "succeeded":       job.succeeded,
            "failed":          job.failed,
            "generated_at":    str(job.completed_at),
        },
        "features": features,
    }


def to_geojson_per_polygon(job: MultiSeasonJobResult) -> dict:
    """
    One Feature per polygon; all seasons embedded as a nested 'seasons' array.
    Most useful for GIS visualisation by farm.
    """
    features = []
    for hist in job.farm_histories:
        if not hist.geometry:
            continue
        features.append({
            "type": "Feature",
            "geometry": hist.geometry,
            "properties": {
                "polygon_id": hist.polygon_id,
                "fid":        hist.fid,
                "county":     hist.county,
                "ward":       hist.ward,
                "aez_code":   hist.aez_code,
                "aez_zone":   hist.aez_zone_name,
                "area_ha":    hist.area_ha,
                "centroid_lat": hist.centroid_lat,
                "centroid_lon": hist.centroid_lon,
                "n_seasons":  len(hist.seasons),
                "seasons": [_season_flat_props(sr) for sr in hist.seasons],
            },
        })
    return {
        "type": "FeatureCollection",
        "properties": {
            "job_id": str(job.job_id),
            "years_processed": job.years_processed,
        },
        "features": features,
    }


# ── CSV ────────────────────────────────────────────────────────────────────────

def to_csv_string(job: MultiSeasonJobResult) -> str:
    """
    One row per polygon × season. All key planting date + phenology fields.
    """
    rows = []
    for hist in job.farm_histories:
        for sr in hist.seasons:
            rows.append(_season_flat_props(sr))

    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ── JSON (crop ID pipeline feed) ──────────────────────────────────────────────

def to_crop_pipeline_json(job: MultiSeasonJobResult) -> list[dict]:
    """
    Primary output for the crop ID pipeline.

    Structure:
    [
      {
        "polygon_id": "...",
        "fid": ...,
        "county": "...",
        "aez_code": ...,
        "geometry": {...},
        "centroid_lat": ..., "centroid_lon": ..., "area_ha": ...,
        "seasons": [
          {
            "season": "long_rains", "year": 2024,
            "planting_date": "2024-03-18",
            "confidence": 0.82,
            "confidence_level": "HIGH",
            "method": "ensemble",
            "phenology": {
              "baseline_ndvi": ...,
              "peak_ndvi": ...,
              "peak_date": ...,
              "senescence_date": ...,
              "season_length_days": ...,
              "ndvi_integral": ...,
              "ndvi_rise_rate": ...,
              "ndvi_at_harvest": ...,
              "ndvi_change": ...,
              "total_rainfall_mm": ...,
              "vv_baseline": ...,
              "vv_at_peak": ...,
              "vh_at_peak": ...,
              "cross_pol_at_peak": ...
            },
            "ndvi_timeseries": [...],
            "vv_timeseries": [...]
          },
          ...
        ]
      },
      ...
    ]
    """
    result = []
    for hist in job.farm_histories:
        seasons_out = []
        for sr in hist.seasons:
            n = sr.ndvi_signal
            s = sr.sar_signal
            r = sr.rainfall_signal
            seasons_out.append({
                "season":           sr.season,
                "year":             sr.year,
                "planting_date":    _str(sr.estimated_planting_date),
                "planting_window_start": _str(sr.planting_window_start),
                "planting_window_end":   _str(sr.planting_window_end),
                "climatological_onset":  _str(sr.climatological_onset),
                "confidence":       sr.confidence,
                "confidence_level": sr.confidence_level,
                "method":           sr.method_used,
                "error":            sr.error,

                # Phenological features — direct inputs to crop ID model
                "phenology": {
                    # NDVI
                    "baseline_ndvi":      n.baseline_ndvi,
                    "peak_ndvi":          n.peak_ndvi,
                    "peak_date":          _str(n.peak_date),
                    "ndvi_change":        n.ndvi_change,
                    "senescence_date":    _str(n.senescence_date),
                    "season_length_days": n.season_length_days,
                    "ndvi_integral":      n.ndvi_integral,
                    "ndvi_rise_rate":     n.ndvi_rise_rate,
                    "ndvi_at_harvest":    n.ndvi_at_harvest,
                    # Rainfall
                    "total_rainfall_mm":  r.total_seasonal_rainfall_mm,
                    "onset_rainfall_3d":  r.cumulative_3day_mm,
                    # SAR
                    "vv_baseline":        s.vv_baseline,
                    "vv_at_peak_ndvi":    s.vv_at_peak_ndvi,
                    "vh_at_peak_ndvi":    s.vh_at_peak_ndvi,
                    "cross_pol_at_peak":  s.cross_pol_at_peak,
                },

                # Full time series for ML models
                "ndvi_timeseries": n.ndvi_timeseries,
                "vv_timeseries":   s.vv_timeseries,

                # Data quality
                "data_quality": {
                    "cloud_cover_pct":    sr.data_quality.cloud_cover_pct,
                    "ndvi_observations":  sr.data_quality.ndvi_observations,
                    "sar_observations":   sr.data_quality.sar_observations,
                    "chirps_completeness": sr.data_quality.chirps_completeness,
                    "max_ndvi_gap_days":  sr.data_quality.max_ndvi_gap_days,
                    "warnings":           sr.data_quality.data_warnings,
                },
            })

        result.append({
            "polygon_id":   hist.polygon_id,
            "fid":          hist.fid,
            "county":       hist.county,
            "ward":         hist.ward,
            "aez_code":     hist.aez_code,
            "aez_zone":     hist.aez_zone_name,
            "geometry":     hist.geometry,
            "centroid_lat": hist.centroid_lat,
            "centroid_lon": hist.centroid_lon,
            "area_ha":      hist.area_ha,
            "seasons":      seasons_out,
        })
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _season_flat_props(sr: SeasonResult) -> dict:
    n = sr.ndvi_signal
    s = sr.sar_signal
    r = sr.rainfall_signal
    return {
        "polygon_id":            sr.polygon_id,
        "fid":                   sr.fid,
        "county":                sr.county,
        "ward":                  sr.ward,
        "aez_code":              sr.aez_code,
        "aez_zone":              sr.aez_zone_name,
        "season":                sr.season,
        "year":                  sr.year,
        "estimated_planting_date": _str(sr.estimated_planting_date),
        "planting_window_start": _str(sr.planting_window_start),
        "planting_window_end":   _str(sr.planting_window_end),
        "climatological_onset":  _str(sr.climatological_onset),
        "confidence":            sr.confidence,
        "confidence_level":      sr.confidence_level,
        "method_used":           sr.method_used,
        # Phenology
        "peak_ndvi":             n.peak_ndvi,
        "peak_date":             _str(n.peak_date),
        "ndvi_change":           n.ndvi_change,
        "senescence_date":       _str(n.senescence_date),
        "season_length_days":    n.season_length_days,
        "ndvi_integral":         n.ndvi_integral,
        "ndvi_rise_rate":        n.ndvi_rise_rate,
        "ndvi_at_harvest":       n.ndvi_at_harvest,
        "total_rainfall_mm":     r.total_seasonal_rainfall_mm,
        "vv_baseline":           s.vv_baseline,
        "vv_at_peak_ndvi":       s.vv_at_peak_ndvi,
        "cross_pol_at_peak":     s.cross_pol_at_peak,
        # Quality
        "cloud_cover_pct":       sr.data_quality.cloud_cover_pct,
        "ndvi_observations":     sr.data_quality.ndvi_observations,
        "sar_observations":      sr.data_quality.sar_observations,
        "chirps_completeness":   sr.data_quality.chirps_completeness,
        "max_ndvi_gap_days":     sr.data_quality.max_ndvi_gap_days,
        "data_warnings":         "; ".join(sr.data_quality.data_warnings),
        "centroid_lat":          sr.centroid_lat,
        "centroid_lon":          sr.centroid_lon,
        "area_ha":               sr.area_ha,
        "error":                 sr.error or "",
    }
