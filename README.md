# Kenya Planting Date Engine

Detects planting dates for all 547 farm polygons across **multiple seasons and years**,
outputting rich phenological profiles ready for a downstream crop identification pipeline.

## What it produces

For each farm × season (e.g. LR-2024, SR-2024, LR-2025, SR-2025):

| Field | Source | Description |
|-------|--------|-------------|
| `estimated_planting_date` | Ensemble | Weighted combination of 3 signals |
| `confidence` / `confidence_level` | Ensemble | 0–1 score + HIGH/MEDIUM/LOW/UNCERTAIN |
| `peak_ndvi` | Sentinel-2 | Max NDVI during season |
| `peak_date` | Sentinel-2 | Date of peak greenness |
| `senescence_date` | Sentinel-2 | When NDVI drops to 70% of peak |
| `season_length_days` | Sentinel-2 | Greenup → senescence (crop cycle) |
| `ndvi_integral` | Sentinel-2 | Area under NDVI curve (photosynthetic activity) |
| `ndvi_rise_rate` | Sentinel-2 | NDVI/day from planting to peak (growth rate) |
| `ndvi_at_harvest` | Sentinel-2 | NDVI 90 days post-planting |
| `total_rainfall_mm` | CHIRPS | Total seasonal rainfall |
| `vv_baseline` | Sentinel-1 | Pre-season SAR backscatter |
| `vv_at_peak_ndvi` | Sentinel-1 | SAR at crop maturity |
| `cross_pol_at_peak` | Sentinel-1 | Volume scattering at maturity |
| `ndvi_timeseries` | Sentinel-2 | Full clean NDVI/EVI/NDWI series |
| `vv_timeseries` | Sentinel-1 | Full VV/VH SAR series |

All of the above are included in `*_crop_pipeline_feed.json` — the primary
output for your crop identification pipeline.

## Project structure

```
kenya_planting_engine/
├── app/
│   ├── core/
│   │   ├── config.py       # AEZ season calendar + settings
│   │   ├── models.py       # Pydantic models
│   │   ├── pipeline.py     # Single-polygon processor
│   │   └── exporter.py     # GeoJSON / CSV / JSON outputs
│   ├── algorithms/
│   │   └── detector.py     # Rainfall + NDVI + SAR signals + ensemble
│   └── data/
│       ├── gee_client.py   # GEE authentication (service account)
│       ├── sentinel2.py    # Sentinel-2 NDVI fetcher
│       ├── sentinel1.py    # Sentinel-1 SAR fetcher
│       └── chirps.py       # CHIRPS rainfall fetcher
├── scripts/
│   └── run_multiseasonal.py  ← MAIN ENTRYPOINT
├── tests/
│   └── test_all.py           # 30 unit tests (no GEE needed)
├── secrets/
│   └── gee-credentials.json  ← your service account (already included)
├── digifarms_with_aez.geojson
├── .env                      ← pre-configured
└── requirements.txt
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run tests (no GEE needed)
pytest tests/ -v

# 3. Test with mock data (no GEE — fast, realistic output)
python scripts/run_multiseasonal.py --mock

# 4. Test 5 polygons with mock
python scripts/run_multiseasonal.py --mock --limit 5

# 5. Full live run (uses your GEE service account)
python scripts/run_multiseasonal.py

# 6. Custom years
python scripts/run_multiseasonal.py --years 2023 2024 2025
```

## Output files (saved to `outputs/`)

| File | Description |
|------|-------------|
| `*_flat.geojson` | One feature per polygon × season |
| `*_per_polygon.geojson` | One feature per polygon, seasons nested |
| `*.csv` | Flat CSV, one row per polygon × season |
| `*_crop_pipeline_feed.json` | **Primary output** — full phenology per farm, feed to crop ID pipeline |

## Seasons processed (default)

| Season | Year | AEZ 33 window | AEZ 44 window | AEZ 46 window | AEZ 99 window |
|--------|------|---------------|---------------|---------------|---------------|
| Long Rains | 2024 | Mar 1–May 20 | Mar 15–Jun 15 | Apr 1–Jun 30 | Feb 15–May 1 |
| Short Rains | 2024 | Oct 1–Dec 20 | Oct 15–Jan 10 | Nov 1–Jan 20 | Sep 15–Dec 1 |
| Long Rains | 2025 | Mar 1–May 20 | Mar 15–Jun 15 | Apr 1–Jun 30 | Feb 15–May 1 |
| Short Rains | 2025 | Oct 1–Dec 20 | Oct 15–Jan 10 | Nov 1–Jan 20 | Sep 15–Dec 1 |

AEZ 99 (highland) also supports a third season (Jun–Aug) via `--seasons third_season`.

## Detection signals

| Signal | Weight | Source | Algorithm |
|--------|--------|--------|-----------|
| Rainfall onset | 40% | CHIRPS v2 (5.5km) | Modified Stern: first 3-day ≥ 25mm |
| NDVI greenup | 35% | Sentinel-2 L2A (10m) | SG-smoothed NDVI rise > 0.08 baseline |
| SAR tillage | 25% | Sentinel-1 GRD (10m) | VV drop ≥ -1.5 dB from baseline |

## GEE Service account

Pre-configured in `.env` and `secrets/gee-credentials.json`:
```
precisionfarms@serene-bastion-406504.iam.gserviceaccount.com
```

Make sure this account has **Earth Engine access** enabled in the
[GEE console](https://code.earthengine.google.com/) for project `serene-bastion-406504`.
