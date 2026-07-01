#!/usr/bin/env python3
"""
scripts/run_multiseasonal.py
===========================
Main CLI — processes all farm polygons across multiple seasons and years.

Default behaviour (no args):
  - Input:   digifarms_with_aez.geojson
  - Years:   previous year (2024) + current year (2025)
  - Seasons: Long Rains + Short Rains for each year
  - Output:  outputs/ folder (GeoJSON flat, GeoJSON per-polygon, CSV, JSON)

Usage
-----
# Full run (requires GEE access)
python scripts/run_multiseasonal.py

# Custom years
python scripts/run_multiseasonal.py --years 2023 2024 2025

# Mock mode — no GEE, uses climatological estimates (great for testing)
python scripts/run_multiseasonal.py --mock

# Specific polygons only
python scripts/run_multiseasonal.py --polygon-ids abc123 def456

# Only certain seasons
python scripts/run_multiseasonal.py --years 2024 2025 --no-sar
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Mock mode ──────────────────────────────────────────────────────────────────

def run_mock(
    geojson: dict,
    season_runs: list[dict],
    polygon_ids: list[str] | None,
) -> "MultiSeasonJobResult":
    from app.core.config import get_aez_config
    from app.core.pipeline import parse_geojson, resolve_season
    from app.core.models import (
        MultiSeasonJobResult, FarmSeasonHistory, SeasonResult, JobStatus,
        RainfallOnsetSignal, NDVIGreenupSignal, SARTillageSignal, DataQuality,
    )

    print("  [MOCK] Using climatological estimates — no GEE API calls.\n")
    polygons = parse_geojson(geojson)
    if polygon_ids:
        polygons = [p for p in polygons if p.polygon_id in set(polygon_ids)]

    farm_histories = {}
    total_tasks = len(polygons) * len(season_runs)
    completed = succeeded = failed = 0

    for polygon in polygons:
        aez = get_aez_config(polygon.aez_code or 0.0)
        hist = FarmSeasonHistory(
            polygon_id=polygon.polygon_id, fid=polygon.fid,
            county=polygon.county, ward=polygon.ward,
            aez_code=polygon.aez_code, aez_zone_name=aez.zone_name,
            geometry=polygon.geometry, centroid_lat=polygon.centroid_lat,
            centroid_lon=polygon.centroid_lon, area_ha=polygon.area_ha,
        )

        for run in season_runs:
            season_str = run["season"]
            year       = run["year"]
            rsn, sw    = resolve_season(season_str, year, aez)
            win_start, win_end = sw.get_window(year)
            clim = sw.get_climatological_onset(year)

            jitter   = random.randint(-15, 15)
            est_date = max(win_start, min(win_end, clim + timedelta(days=jitter)))
            conf     = round(random.uniform(0.45, 0.85), 3)
            clvl     = "HIGH" if conf >= 0.7 else "MEDIUM" if conf >= 0.4 else "LOW"

            # Mock phenology (realistic ranges)
            peak_ndvi  = round(random.uniform(0.50, 0.82), 4)
            baseline   = round(random.uniform(0.12, 0.25), 4)
            slen       = random.randint(80, 130)
            greenup    = est_date + timedelta(days=12)
            peak_date  = greenup + timedelta(days=random.randint(30, 55))
            sen_date   = greenup + timedelta(days=slen)
            ndvi_ts    = _mock_ndvi_ts(greenup, peak_date, sen_date, peak_ndvi, baseline)
            vv_ts      = _mock_vv_ts(est_date, win_end)

            sr = SeasonResult(
                polygon_id=polygon.polygon_id, fid=polygon.fid,
                county=polygon.county, ward=polygon.ward,
                aez_code=polygon.aez_code, aez_zone_name=aez.zone_name,
                season=rsn.value, year=year,
                estimated_planting_date=est_date,
                planting_window_start=win_start,
                planting_window_end=win_end,
                climatological_onset=clim,
                confidence=conf, confidence_level=clvl, method_used="mock_climatology",
                rainfall_signal=RainfallOnsetSignal(
                    onset_date=est_date + timedelta(days=random.randint(-3, 3)),
                    cumulative_3day_mm=round(random.uniform(25, 60), 1),
                    total_seasonal_rainfall_mm=round(random.uniform(150, 600), 1),
                    confidence=round(random.uniform(0.5, 0.9), 3), available=True,
                ),
                ndvi_signal=NDVIGreenupSignal(
                    greenup_date=greenup,
                    estimated_planting_date=est_date,
                    baseline_ndvi=baseline, peak_ndvi=peak_ndvi,
                    peak_date=peak_date, ndvi_change=round(peak_ndvi - baseline, 4),
                    senescence_date=sen_date, season_length_days=slen,
                    ndvi_at_harvest=round(random.uniform(0.25, 0.50), 4),
                    ndvi_integral=round(slen * (peak_ndvi + baseline) / 2, 2),
                    ndvi_rise_rate=round((peak_ndvi - baseline) / (peak_date - greenup).days, 5),
                    ndvi_timeseries=ndvi_ts,
                    cloud_gap_days=random.randint(5, 18),
                    n_observations=random.randint(6, 12),
                    confidence=round(random.uniform(0.5, 0.85), 3), available=True,
                ),
                sar_signal=SARTillageSignal(
                    onset_date=est_date + timedelta(days=random.randint(-5, 2)),
                    vv_change_db=round(random.uniform(-3.5, -1.5), 3),
                    vv_baseline=round(random.uniform(-13.0, -11.0), 3),
                    vv_at_peak_ndvi=round(random.uniform(-15.0, -12.0), 3),
                    vh_at_peak_ndvi=round(random.uniform(-19.0, -16.0), 3),
                    cross_pol_at_peak=round(random.uniform(-5.0, -3.0), 3),
                    vv_timeseries=vv_ts,
                    tillage_detected=True,
                    confidence=round(random.uniform(0.4, 0.75), 3), available=True,
                ),
                data_quality=DataQuality(
                    cloud_cover_pct=round(random.uniform(5, 35), 1),
                    ndvi_observations=random.randint(5, 12),
                    sar_observations=random.randint(4, 8),
                    chirps_completeness=round(random.uniform(0.9, 1.0), 3),
                ),
                geometry=polygon.geometry,
                centroid_lat=polygon.centroid_lat, centroid_lon=polygon.centroid_lon,
                area_ha=polygon.area_ha,
            )
            hist.seasons.append(sr)
            completed += 1; succeeded += 1
            print(f"  [{completed:4d}/{total_tasks}] {polygon.polygon_id[:30]:32s} "
                  f"{rsn.value:12s} {year}  {str(est_date):12s}  "
                  f"conf={conf:.2f} {clvl}")

        farm_histories[polygon.polygon_id] = hist

    years = sorted(set(r["year"] for r in season_runs))
    seasons = sorted(set(r["season"] for r in season_runs))
    return MultiSeasonJobResult(
        status=JobStatus.COMPLETED,
        years_processed=years,
        seasons_processed=seasons,
        total_polygons=len(polygons),
        total_tasks=total_tasks,
        completed_tasks=completed,
        succeeded=succeeded,
        failed=failed,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        farm_histories=list(farm_histories.values()),
    )


def _mock_ndvi_ts(greenup, peak_date, sen_date, peak_ndvi, baseline):
    ts = []
    cur = greenup - timedelta(days=10)
    while cur <= sen_date + timedelta(days=20):
        if cur < greenup:
            ndvi = round(baseline + random.uniform(-0.02, 0.02), 4)
        elif cur <= peak_date:
            progress = (cur - greenup).days / max(1, (peak_date - greenup).days)
            ndvi = round(baseline + (peak_ndvi - baseline) * progress + random.uniform(-0.02, 0.02), 4)
        elif cur <= sen_date:
            progress = (cur - peak_date).days / max(1, (sen_date - peak_date).days)
            ndvi = round(peak_ndvi - (peak_ndvi - baseline * 1.2) * progress + random.uniform(-0.02, 0.02), 4)
        else:
            ndvi = round(baseline * 1.1 + random.uniform(-0.02, 0.02), 4)
        ts.append({"date": str(cur), "ndvi": max(0.05, ndvi), "evi": round(ndvi * 0.85, 4),
                   "ndwi": round(-0.3 + ndvi * 0.2, 4), "cloud_pct": round(random.uniform(0, 30), 1)})
        cur += timedelta(days=5)
    return ts


def _mock_vv_ts(start, end):
    ts = []
    cur = start - timedelta(days=20)
    baseline_vv = round(random.uniform(-13.0, -11.0), 3)
    while cur <= end:
        vv = round(baseline_vv + random.uniform(-1.5, 0.5), 3)
        ts.append({"date": str(cur), "vv_db": vv, "vh_db": round(vv - 4.5, 3),
                   "cross_pol": round(-4.5 + random.uniform(-0.5, 0.5), 3), "pass": "ASCENDING"})
        cur += timedelta(days=6)
    return ts


# ── Live run ───────────────────────────────────────────────────────────────────

def run_live(
    geojson: dict,
    season_runs: list[dict],
    polygon_ids: list[str] | None,
    use_rainfall: bool,
    use_ndvi: bool,
    use_sar: bool,
    workers: int = 8,
) -> "MultiSeasonJobResult":
    from app.core.config import get_aez_config
    from app.core.pipeline import parse_geojson, resolve_season, PolygonProcessor
    from app.core.models import MultiSeasonJobResult, FarmSeasonHistory, JobStatus

    # One shared processor — PolygonProcessor is stateless so it is thread-safe.
    processor = PolygonProcessor(use_rainfall, use_ndvi, use_sar)
    polygons  = parse_geojson(geojson)
    if polygon_ids:
        polygons = [p for p in polygons if p.polygon_id in set(polygon_ids)]

    total_tasks = len(polygons) * len(season_runs)

    # Pre-build an ordered list of (polygon, aez, run) work items so we can
    # preserve insertion order when collecting results.
    work_items = []
    for polygon in polygons:
        aez = get_aez_config(polygon.aez_code or 0.0)
        for run in season_runs:
            work_items.append((polygon, aez, run))

    # Thread-safety for progress counter + print
    _lock     = threading.Lock()
    _counter  = [0]  # mutable box so inner closure can mutate it
    _succeeded = [0]
    _failed    = [0]

    # Map (polygon_id, season, year) -> SeasonResult so we can reconstruct order
    result_map: dict[tuple, object] = {}

    def process_one(item):
        polygon, aez, run = item
        rsn, sw = resolve_season(run["season"], run["year"], aez)
        sr = processor.process(polygon, rsn, sw, aez, run["year"])
        key = (polygon.polygon_id, run["season"], run["year"])

        with _lock:
            _counter[0] += 1
            n = _counter[0]
            if sr.error:
                _failed[0] += 1
                print(f"  [{n:4d}/{total_tasks}] FAIL {polygon.polygon_id[:30]:32s} "
                      f"{rsn.value} {run['year']}: {sr.error}")
            else:
                _succeeded[0] += 1
                print(f"  [{n:4d}/{total_tasks}] OK   {polygon.polygon_id[:30]:32s} "
                      f"{rsn.value:12s} {run['year']}  "
                      f"{str(sr.estimated_planting_date):12s}  "
                      f"conf={sr.confidence:.2f} {sr.confidence_level}")

        return key, sr, aez

    print(f"  [Threads: {workers}]  Submitting {total_tasks} tasks …\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, item): item for item in work_items}
        for fut in as_completed(futures):
            try:
                key, sr, aez = fut.result()
                result_map[key] = (sr, aez)
            except Exception as exc:
                # Catch any unexpected thread-level exception so the pool keeps running
                polygon, _, run = futures[fut]
                logging.error(f"Unhandled thread error {polygon.polygon_id} "
                              f"{run['season']}/{run['year']}: {exc}", exc_info=True)
                with _lock:
                    _failed[0] += 1

    # Reconstruct FarmSeasonHistory objects in original polygon/season order
    farm_histories_ordered = []
    for polygon in polygons:
        aez = get_aez_config(polygon.aez_code or 0.0)
        hist = FarmSeasonHistory(
            polygon_id=polygon.polygon_id, fid=polygon.fid,
            county=polygon.county, ward=polygon.ward,
            aez_code=polygon.aez_code, aez_zone_name=aez.zone_name,
            geometry=polygon.geometry, centroid_lat=polygon.centroid_lat,
            centroid_lon=polygon.centroid_lon, area_ha=polygon.area_ha,
        )
        for run in season_runs:
            key = (polygon.polygon_id, run["season"], run["year"])
            if key in result_map:
                hist.seasons.append(result_map[key][0])
        farm_histories_ordered.append(hist)

    years   = sorted(set(r["year"]    for r in season_runs))
    seasons = sorted(set(r["season"]  for r in season_runs))
    succeeded = _succeeded[0]
    failed    = _failed[0]
    status  = (JobStatus.COMPLETED if failed == 0
               else JobStatus.PARTIAL if succeeded > 0
               else JobStatus.FAILED)

    return MultiSeasonJobResult(
        status=status,
        years_processed=years,
        seasons_processed=seasons,
        total_polygons=len(polygons),
        total_tasks=total_tasks,
        completed_tasks=_counter[0],
        succeeded=succeeded,
        failed=failed,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        farm_histories=farm_histories_ordered,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    current_year  = datetime.now().year
    previous_year = current_year - 1

    parser = argparse.ArgumentParser(
        description="Kenya Planting Date Engine — multi-season processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with default years (2024 + 2025)  [GEE path]
  python scripts/run_multiseasonal.py

  # Datacube path (STAC + Visual Crossing — no GEE)
  python scripts/run_multiseasonal.py --datacube --years 2025 --seasons long_rains

  # Side-by-side accuracy validation
  python scripts/run_multiseasonal.py --datacube --limit 50 --years 2025 --seasons long_rains
  python scripts/run_multiseasonal.py          --limit 50 --years 2025 --seasons long_rains

  # Test without GEE (mock data)
  python scripts/run_multiseasonal.py --mock

  # Custom years
  python scripts/run_multiseasonal.py --years 2023 2024 2025
        """,
    )
    parser.add_argument("--datacube", action="store_true",
                        help="Use Datacube path (STAC + Visual Crossing + TimescaleDB cache). "
                             "Loads farms from spatial.farm_intelligence instead of --input GeoJSON.")
    parser.add_argument("--input",   "-i", default="digifarms_with_aez.geojson")
    parser.add_argument("--outdir",  "-o", default="outputs")
    parser.add_argument("--years",   "-y", nargs="+", type=int,
                        default=[previous_year, current_year])
    parser.add_argument("--seasons", "-s", nargs="+",
                        choices=["long_rains", "short_rains", "third_season"],
                        default=["long_rains", "short_rains"])
    parser.add_argument("--polygon-ids", nargs="+", dest="polygon_ids")
    parser.add_argument("--limit",  "-n", type=int, help="Process first N polygons (for testing)")
    parser.add_argument("--mock",   action="store_true", help="Use mock data -- no GEE required")
    parser.add_argument("--include-future", action="store_true",
                        help="Include seasons whose window hasn't started yet")
    parser.add_argument("--no-s3",  action="store_true", help="Skip S3 upload")
    parser.add_argument("--no-rainfall", action="store_true")
    parser.add_argument("--no-ndvi",     action="store_true")
    parser.add_argument("--no-sar",      action="store_true")
    parser.add_argument("--workers",  "-w", type=int, default=8,
                        help="Parallel threads (default: 8)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass disk cache — always fetch fresh from GEE (GEE path only)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached GEE observations before running (GEE path only)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # ── Cache management ────────────────────────────────────────────────────────
    from app.data import cache as _cache
    if args.clear_cache:
        deleted = _cache.clear()
        print(f"  [CACHE] Cleared {deleted} cached GEE files.")
    if args.no_cache:
        # Monkey-patch get() to always return None — puts() still work so the
        # current run populates the cache for next time.
        _cache.get = lambda *a, **kw: None
        print("  [CACHE] Disabled (--no-cache). Fetching all data fresh from GEE.")
    else:
        cs = _cache.stats()
        total = sum(cs.values())
        if total:
            print(f"  [CACHE] {total} cached entries (S2:{cs['s2']}  SAR:{cs['s1']}  CHIRPS:{cs['chirps']}) "
                  f"— cache hits skip GEE entirely.")
        else:
            print("  [CACHE] Empty — all data will be fetched from GEE and cached for next run.")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        geojson = json.load(f)

    n_features = len(geojson.get("features", []))
    if args.limit:
        geojson["features"] = geojson["features"][: args.limit]
        n_features = args.limit

    # Build season/year run list
    from app.core.config import build_season_run_list, get_aez_config, Season
    all_runs = []
    for yr in sorted(args.years):
        for s in args.seasons:
            all_runs.append({"season": s, "year": yr})

    # Filter out seasons where there is genuinely not enough data yet.
    #
    # Rule: skip if today < climatological_onset + 45 days
    #   → planting hasn't happened yet, no signal to detect.
    # Allow ongoing seasons where planting has already occurred (e.g. LR-2026:
    #   onset ~Mar 1, by May 8 we have Feb-Apr CHIRPS+S2 = sufficient for detection).
    # Skip truly future seasons (search_start > today) as before.
    if not args.include_future:
        today = date.today()
        filtered = []
        default_aez = get_aez_config(0.0)
        for run in all_runs:
            try:
                season_enum = Season(run["season"])
                sw = default_aez.get_season_window(season_enum)
                if sw:
                    search_start  = sw.get_search_start(run["year"])
                    clim_onset    = sw.get_climatological_onset(run["year"])
                    data_ready_by = clim_onset + timedelta(days=45)  # enough signal visible
                    if search_start > today:
                        print(f"  [SKIP] {run['season']} {run['year']} "
                              f"-- search window starts {search_start}, not yet")
                    elif today < data_ready_by:
                        print(f"  [SKIP] {run['season']} {run['year']} "
                              f"-- too early for planting signal (ready ~{data_ready_by}, use --include-future to force)")
                    else:
                        filtered.append(run)
                else:
                    filtered.append(run)
            except ValueError:
                filtered.append(run)
        all_runs = filtered

    print("\n" + "=" * 60)
    print("  Kenya Planting Date Engine — Multi-Season Run")
    print("=" * 60)
    print(f"  Input:    {args.input} ({n_features} polygons)")
    print(f"  Years:    {sorted(args.years)}")
    print(f"  Seasons:  {args.seasons}")
    print(f"  Tasks:    {n_features} × {len(all_runs)} = {n_features * len(all_runs)}")
    print(f"  Mode:     {'MOCK (no GEE)' if args.mock else f'LIVE (GEE, {args.workers} threads)'}")
    print("=" * 60 + "\n")

    t0 = datetime.now()

    if args.mock:
        job = run_mock(geojson, all_runs, args.polygon_ids)
    elif getattr(args, "datacube", False):
        # ── Datacube path (STAC + Visual Crossing + TimescaleDB) ────────────────
        from app.data.datacube_client import DatacubeClient
        from app.data.stac_fetcher import Sentinel2STACFetcher, Sentinel1STACFetcher
        from app.core.pipeline import resolve_season, PolygonProcessor
        from app.core.config import get_aez_config
        from collections import defaultdict
        import threading

        dc = DatacubeClient()
        limit = args.limit or 10000
        farms = dc.get_eligible_farms(batch_size=limit)
        if args.polygon_ids:
            farms = [f for f in farms if f.farm_uuid in args.polygon_ids]

        tile_groups = dc.group_by_tile(farms)
        logger = logging.getLogger("run_multiseasonal")
        logger.info("Datacube mode: %d farms in %d tiles", len(farms), len(tile_groups))

        from app.core.models import FarmSeasonHistory, MultiSeasonJobResult, JobStatus
        all_histories = []

        for tile_id, tile_farms in tile_groups.items():
            tile_pairs = [(f.farm_uuid, f.geom_wkt) for f in tile_farms]
            for run in all_runs:
                yr, sname = run["year"], run["season"]
                aez_s = get_aez_config(tile_farms[0].aez_code)
                _, w = resolve_season(sname, yr, aez_s)
                from datetime import timedelta
                fetch_start = w.get_search_start(yr) - timedelta(days=60)
                _, win_end  = w.get_window(yr)

                c_s2 = dc._pool.getconn()
                try:
                    Sentinel2STACFetcher(c_s2).fetch_for_tile(tile_id, tile_pairs, fetch_start, win_end)
                finally:
                    dc._pool.putconn(c_s2)

                c_s1 = dc._pool.getconn()
                try:
                    Sentinel1STACFetcher(c_s1).fetch_for_tile(tile_id, tile_pairs, fetch_start, win_end)
                finally:
                    dc._pool.putconn(c_s1)

            for farm in tile_farms:
                hist = FarmSeasonHistory(
                    polygon_id=farm.farm_uuid,
                    county=farm.county_code, ward=farm.ward_code,
                    aez_code=farm.aez_code,
                    centroid_lat=farm.centroid_lat, centroid_lon=farm.centroid_lon,
                    area_ha=farm.area_ha,
                )
                poly = farm.to_farm_polygon()
                aez  = get_aez_config(farm.aez_code)
                proc = PolygonProcessor(
                    use_rainfall=not args.no_rainfall,
                    use_ndvi=not args.no_ndvi,
                    use_sar=not args.no_sar,
                    fallback_to_climatology=True,
                    datacube_client=dc,
                )
                for run in all_runs:
                    s, w = resolve_season(run["season"], run["year"], aez)
                    hist.seasons.append(proc.process(poly, s, w, aez, run["year"]))
                all_histories.append(hist)

        ok  = sum(1 for h in all_histories for s in h.seasons if not s.error)
        err = sum(1 for h in all_histories for s in h.seasons if s.error)
        job = MultiSeasonJobResult(
            status=JobStatus.COMPLETED if err == 0 else JobStatus.PARTIAL,
            years_processed=sorted(set(r["year"]   for r in all_runs)),
            seasons_processed=sorted(set(r["season"] for r in all_runs)),
            total_polygons=len(all_histories),
            total_tasks=len(all_histories) * len(all_runs),
            succeeded=ok, failed=err,
            farm_histories=all_histories,
        )
        dc.close()
    else:
        job = run_live(
            geojson, all_runs, args.polygon_ids,
            use_rainfall=not args.no_rainfall,
            use_ndvi=not args.no_ndvi,
            use_sar=not args.no_sar,
            workers=args.workers,
        )

    elapsed = (datetime.now() - t0).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Polygons:  {job.total_polygons}")
    print(f"  Tasks:     {job.succeeded} OK / {job.failed} failed")
    if job.farm_histories:
        from collections import Counter
        levels  = Counter(sr.confidence_level
                          for h in job.farm_histories for sr in h.seasons if not sr.error)
        methods = Counter(sr.method_used
                          for h in job.farm_histories for sr in h.seasons if not sr.error)
        print(f"  Confidence: {dict(levels)}")
        print(f"  Methods:    {dict(methods)}")
    print("=" * 60 + "\n")

    # ── Save all outputs ───────────────────────────────────────────────────────
    from app.core.exporter import (
        to_geojson_flat, to_geojson_per_polygon,
        to_csv_string, to_crop_pipeline_json,
    )

    os.makedirs(args.outdir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    yrs  = "_".join(str(y) for y in sorted(args.years))
    base = f"{args.outdir}/planting_dates_{yrs}_{ts}"

    # 1. Flat GeoJSON (one feature per polygon×season)
    path1 = f"{base}_flat.geojson"
    with open(path1, "w") as f:
        json.dump(to_geojson_flat(job), f, indent=2, default=str)
    print(f"  Saved: {path1}")

    # 2. Per-polygon GeoJSON (all seasons nested)
    path2 = f"{base}_per_polygon.geojson"
    with open(path2, "w") as f:
        json.dump(to_geojson_per_polygon(job), f, indent=2, default=str)
    print(f"  Saved: {path2}")

    # 3. CSV (flat rows)
    path3 = f"{base}.csv"
    with open(path3, "w") as f:
        f.write(to_csv_string(job))
    print(f"  Saved: {path3}")

    # 4. JSON — crop ID pipeline feed (PRIMARY output)
    path4 = f"{base}_crop_pipeline_feed.json"
    with open(path4, "w") as f:
        json.dump(to_crop_pipeline_json(job), f, indent=2, default=str)
    print(f"  Saved: {path4}  <-- feed this to your crop ID pipeline")

    # ── Upload to S3 ───────────────────────────────────────────────────────────
    if not args.no_s3:
        try:
            from app.core.s3_uploader import upload_outputs
            s3_prefix = f"planting-dates/{ts}"
            print(f"\n  Uploading to S3: {s3_prefix}/")
            uploaded = upload_outputs([path1, path2, path3, path4], s3_prefix)
            for uri in uploaded:
                print(f"    -> {uri}")
            if not uploaded:
                print("    (no files uploaded -- check S3 config in .env)")
        except Exception as e:
            print(f"  S3 upload failed: {e}")

    print()


if __name__ == "__main__":
    main()
