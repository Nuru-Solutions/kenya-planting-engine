"""
scripts/run_datacube_batch.py
==============================
Stage 4: Planting Date Engine — AWS Batch Production Entrypoint

Architecture (mirrors Stage 3 run_crop_classifier.py structure):
----------------------------------------------------------------
  1. Connect to PostGIS / TimescaleDB (DatacubeClient)
  2. SELECT eligible farms (crop_type set, planting_date NULL/stale)
  3. Group farms by tile_id  [same pattern as Stage 3]
  4. For each tile group:
       a. Sentinel2STACFetcher.fetch_for_tile()  → cache to farm_indices
       b. Sentinel1STACFetcher.fetch_for_tile()  → cache to farm_indices (VV/VH/cross_pol)
       c. For each farm in tile:
             - Determine season/window from AEZ code + CLI args
             - PolygonProcessor(datacube_client=dc).process()
               └─ reads NDVI/SAR from farm_indices cache
               └─ fetches climate from Visual Crossing → climate_daily cache
       d. Bulk upsert SeasonResults → spatial.farm_intelligence
  5. Log summary  +  exit code 0/1 for Batch retry logic

AWS Batch trigger
-----------------
  EventBridge Rule: rate(6 hours)
  Job Definition:   kenya-planting-engine
  Container CMD:    python scripts/run_datacube_batch.py

CLI
---
  python scripts/run_datacube_batch.py
      [--batch-size  500]       # farms per run (default 500)
      [--season      long_rains] # long_rains | short_rains | third_season
      [--year        2025]       # crop calendar year
      [--workers     4]          # ThreadPoolExecutor workers (bounded by DB pool)
      [--dry-run]               # log what would be written, no DB writes
      [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from app.core.config import get_aez_config, Season, get_settings
from app.core.models import FarmRow, SeasonResult
from app.core.pipeline import PolygonProcessor, resolve_season
from app.data.datacube_client import DatacubeClient
from app.data.stac_fetcher import Sentinel2STACFetcher, Sentinel1STACFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Season auto-detect ─────────────────────────────────────────────────────────

def _auto_detect_season(year: int) -> str:
    """
    Infer the most recent completable season based on today's date.
    Long Rains ends ~May, Short Rains ends ~Dec.
    """
    today = date.today()
    if 6 <= today.month <= 9:
        # After LR harvest window — process Short Rains
        return "long_rains"
    if today.month >= 10 or today.month <= 1:
        return "short_rains"
    return "long_rains"


# ── Per-farm worker ────────────────────────────────────────────────────────────

def _process_farm(
    farm: FarmRow,
    season_str: str,
    year: int,
    datacube_client: DatacubeClient,
) -> tuple[str, SeasonResult]:
    """
    Called from ThreadPoolExecutor. Reads NDVI/SAR from cache (already populated),
    fetches climate (cache-first), runs ensemble, returns (farm_uuid, SeasonResult).
    """
    aez    = get_aez_config(farm.aez_code)
    season, window = resolve_season(season_str, year, aez)
    polygon = farm.to_farm_polygon()

    processor = PolygonProcessor(
        use_rainfall=True,
        use_ndvi=True,
        use_sar=True,
        fallback_to_climatology=True,
        datacube_client=datacube_client,
    )
    result = processor.process(polygon, season, window, aez, year)
    return farm.farm_uuid, result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_batch_pipeline(
    batch_size: int,
    season_str: str,
    year: int,
    workers: int,
    dry_run: bool,
) -> int:
    """
    Execute the planting date detection batch.
    Returns exit code: 0 = success, 1 = partial failures.
    """
    logger.info("=" * 65)
    logger.info("🌱  STAGE 4: Planting Date Engine — Datacube Batch")
    logger.info("   Season     : %s %d", season_str, year)
    logger.info("   Batch size : %d farms", batch_size)
    logger.info("   Workers    : %d threads", workers)
    logger.info("   Dry run    : %s", dry_run)
    logger.info("=" * 65)

    # ── 1. Connect ─────────────────────────────────────────────────────────────
    dc = DatacubeClient()

    try:
        # ── 2. Get eligible farms ───────────────────────────────────────────────
        farms = dc.get_eligible_farms(batch_size=batch_size)
        if not farms:
            logger.info("✅ No eligible farms found. Pipeline complete.")
            return 0

        # ── 3. Group by tile_id (mirrors Stage 3) ──────────────────────────────
        tile_groups: dict[str, list[FarmRow]] = defaultdict(list)
        for farm in farms:
            tile_groups[farm.tile_id].append(farm)

        logger.info("Grouped %d farms into %d Sentinel-2 tiles", len(farms), len(tile_groups))

        # ── 4. Process tile-by-tile ─────────────────────────────────────────────
        total_succeeded = 0
        total_failed    = 0
        method_counts: dict[str, int] = defaultdict(int)
        confidence_sum  = 0.0
        all_upsert_rows: list[tuple] = []

        for tile_idx, (tile_id, tile_farms) in enumerate(tile_groups.items(), start=1):
            n = len(tile_farms)
            logger.info("[%d/%d] Tile %s — %d farms", tile_idx, len(tile_groups), tile_id, n)

            # ── 4a. Pre-fetch STAC data for this tile → writes to farm_indices ──
            tile_pairs = [(f.farm_uuid, f.geom_wkt) for f in tile_farms]
            aez_sample = get_aez_config(tile_farms[0].aez_code)
            _, sample_window = resolve_season(season_str, year, aez_sample)
            from datetime import timedelta
            fetch_start = sample_window.get_search_start(year) - timedelta(days=60)
            _, win_end  = sample_window.get_window(year)

            # Sentinel-2 NDVI/EVI/NDWI
            conn_s2 = dc._pool.getconn()
            try:
                s2 = Sentinel2STACFetcher(conn_s2)
                s2.fetch_for_tile(tile_id, tile_pairs, fetch_start, win_end)
            finally:
                dc._pool.putconn(conn_s2)

            # Sentinel-1 SAR VV/VH/cross_pol
            conn_s1 = dc._pool.getconn()
            try:
                s1 = Sentinel1STACFetcher(conn_s1)
                s1.fetch_for_tile(tile_id, tile_pairs, fetch_start, win_end)
            finally:
                dc._pool.putconn(conn_s1)

            # ── 4b. Detect planting date per farm (threaded) ────────────────────
            tile_results: list[tuple[str, SeasonResult]] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_farm, farm, season_str, year, dc): farm.farm_uuid
                    for farm in tile_farms
                }
                for future in as_completed(futures):
                    farm_uuid = futures[future]
                    try:
                        fuid, result = future.result()
                        tile_results.append((fuid, result))
                        if result.error:
                            total_failed += 1
                            logger.warning("Farm %s failed: %s", fuid[:8], result.error)
                        else:
                            total_succeeded += 1
                            method_counts[result.method_used] += 1
                            confidence_sum += result.confidence
                    except Exception as exc:
                        total_failed += 1
                        logger.error("Farm %s raised: %s", farm_uuid[:8], exc)

            # ── 4c. Bulk upsert results for this tile ───────────────────────────
            if tile_results:
                dc.upsert_planting_results(tile_results, dry_run=dry_run)
                all_upsert_rows.extend(tile_results)

        # ── 5. Summary ──────────────────────────────────────────────────────────
        total = total_succeeded + total_failed
        avg_conf = round(confidence_sum / max(total_succeeded, 1), 3)

        logger.info("")
        logger.info("=" * 65)
        logger.info("📊  STAGE 4 COMPLETE — Planting Date Engine")
        logger.info("=" * 65)
        logger.info("   Farms processed  : %d", total)
        logger.info("   Succeeded        : %d  (%.0f%%)", total_succeeded,
                    100 * total_succeeded / max(total, 1))
        logger.info("   Failed           : %d", total_failed)
        logger.info("   Avg confidence   : %.3f", avg_conf)
        logger.info("   Methods used     :")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            logger.info("     %-30s : %d", method, count)
        if dry_run:
            logger.info("   ⚠️  DRY RUN — no rows written to farm_intelligence")
        logger.info("=" * 65)

        return 0 if total_failed == 0 else 1

    finally:
        dc.close()


def main() -> None:
    settings = get_settings()
    default_year   = date.today().year
    default_season = _auto_detect_season(default_year)

    parser = argparse.ArgumentParser(
        description="Stage 4: Planting Date Engine — Datacube Batch Runner"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Max number of farms to process per run (default: 500)",
    )
    parser.add_argument(
        "--season", type=str, default=default_season,
        choices=["long_rains", "short_rains", "third_season"],
        help=f"Season to process (auto-detected: {default_season})",
    )
    parser.add_argument(
        "--year", type=int, default=default_year,
        help=f"Crop calendar year (default: {default_year})",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Thread pool size for per-farm processing (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect planting dates but do NOT write results to DB",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Set log level to DEBUG",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate workers vs DB pool
    max_pool = settings.postgis_pool_max
    if args.workers > max_pool - 1:
        logger.warning(
            "workers=%d exceeds pool_max=%d — capping workers to %d",
            args.workers, max_pool, max_pool - 1,
        )
        args.workers = max_pool - 1

    exit_code = run_batch_pipeline(
        batch_size  = args.batch_size,
        season_str  = args.season,
        year        = args.year,
        workers     = args.workers,
        dry_run     = args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
