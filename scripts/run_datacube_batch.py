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

def _auto_detect_season(today: date | None = None) -> tuple[str, int]:
    """
    Infer the most recently completed (or currently in-progress) season
    AND its crop-calendar year, based on today's date.

    Returns (season, year) rather than season alone. Short Rains windows
    are anchored to the October they *start* in — SeasonWindow.get_window()
    wraps the end date into Jan/Feb of year+1 for AEZs like LM4 — so in
    January the SR season that just finished belongs to *last* year, not
    this one. Pairing "short_rains" with today.year there points
    resolve_season() at an Oct-Dec window that is still ~9 months in the
    future: every farm gets zero satellite/rainfall observations and falls
    back to the hardcoded climatological date (see PlantingDateEnsemble's
    fallback_climatology path), which then gets upserted as if it were a
    real detection.
    """
    today = today or date.today()
    year  = today.year

    if 6 <= today.month <= 9:
        # After LR harvest, before SR opens — most recently completed
        # season is this year's Long Rains.
        return "long_rains", year
    if today.month == 1:
        # This year's Short Rains hasn't opened yet (starts October);
        # the SR season currently wrapping up started *last* October.
        return "short_rains", year - 1
    if today.month >= 10:
        return "short_rains", year
    return "long_rains", year


# ── Downstream chaining (Django webhook + Stage 5 trigger) ────────────────────

def _trigger_downstream(dry_run: bool) -> None:
    """
    1. PATCH Django FarmUploadPipelineStatusView: planting_date=completed
       (CSV-upload runs only — UPLOAD_ID unset on scheduled cron runs).
    2. Invoke nuru-start-crop-health-pipeline (fire-and-forget) to chain
       into Stage 5, forwarding UPLOAD_ID so it can report its own
       completion back to Django.

    Both calls are non-fatal and never affect exit code.

    Skipped on AWS Batch retry attempts (AWS_BATCH_JOB_ATTEMPT != "1"):
    Batch retries the WHOLE job on a non-zero exit code, and this function
    runs at the very end of a successful pass — so if attempt 1 reached
    here, it already fired both calls. Re-running on attempt 2+ would
    report "completed" and invoke Stage 5 a second time for the same
    UPLOAD_ID. (If attempt 1 crashed *before* reaching here, attempt 2 will
    correctly fire once it completes — attempt 2's own AWS_BATCH_JOB_ATTEMPT
    is "2", but it is the first attempt to actually reach this function.)
    """
    import os as _os, json as _json, urllib.request as _urllib_req

    attempt = _os.environ.get("AWS_BATCH_JOB_ATTEMPT")
    if attempt and attempt != "1":
        logger.info(
            "AWS_BATCH_JOB_ATTEMPT=%s (retry) — skipping downstream chaining to "
            "avoid double-firing the Django webhook / Stage 5 trigger",
            attempt,
        )
        return

    upload_id        = _os.environ.get("UPLOAD_ID")
    webhook_template = _os.environ.get(
        "GPS_RESULT_WEBHOOK_URL_TEMPLATE",
        "https://api.nuru.solutions/organization/upload-status/{upload_id}/pipeline/",
    )
    webhook_secret = _os.environ.get("GPS_RESULT_WEBHOOK_SECRET")

    # 1. Report planting_date = "completed" to Django (CSV upload runs only)
    if not upload_id or dry_run:
        logger.info("No UPLOAD_ID set or dry-run — skipping planting_date webhook")
    elif not webhook_secret:
        logger.error(
            "GPS_RESULT_WEBHOOK_SECRET not set — skipping planting_date webhook "
            "for upload_id=%s (refusing to fall back to a hardcoded default secret)",
            upload_id,
        )
    else:
        try:
            url     = webhook_template.format(upload_id=upload_id)
            payload = _json.dumps({"pipeline_status": {"planting_date": "completed"}}).encode()
            req = _urllib_req.Request(
                url, data=payload,
                headers={"Content-Type": "application/json",
                         "X-Pipeline-Key": webhook_secret},
                method="PATCH",
            )
            with _urllib_req.urlopen(req, timeout=10) as resp:
                logger.info("📡 Planting date completion reported to Django (HTTP %s)", resp.status)
        except Exception as wh_err:
            logger.error("Failed to report planting_date completion to Django: %s", wh_err)

    # 2. Trigger Stage 5: Crop Health pipeline.
    #    Always triggered (scheduled cron runs also need to chain here).
    #    UPLOAD_ID is forwarded so crop health can report its own completion
    #    to Django for the correct upload.
    if not dry_run:
        import boto3 as _boto3
        crop_health_lambda = _os.environ.get("CROP_HEALTH_LAMBDA_NAME", "nuru-start-crop-health-pipeline")
        try:
            lc = _boto3.client("lambda", region_name=_os.environ.get("AWS_REGION", "eu-north-1"))
            lc.invoke(
                FunctionName=crop_health_lambda,
                InvocationType="Event",
                Payload=_json.dumps({
                    "source": "planting-engine",
                    "upload_id": upload_id,
                }),
            )
            logger.info("🚀 Triggered %s to start Crop Health pipeline (Stage 5)", crop_health_lambda)
        except Exception as ch_err:
            logger.error("Failed to trigger Crop Health pipeline: %s", ch_err)


# ── Pure helpers (unit-testable without a DB/STAC connection) ─────────────────

def _tile_fetch_window(
    tile_farms: list[FarmRow],
    season_str: str,
    year: int,
) -> tuple:
    """
    Compute the (fetch_start, win_end) STAC search range that covers every
    distinct AEZ window present in a tile group.

    A single Sentinel-2 tile can contain farms from more than one AEZ, and
    each AEZ has its own season window (see AEZConfig.seasons). Returns the
    UNION of all distinct AEZ windows in this tile_farms group, plus the
    usual 60-day pre-season baseline padding on the start.
    """
    from datetime import timedelta

    distinct_aez_codes = {f.aez_code for f in tile_farms}
    windows = [
        resolve_season(season_str, year, get_aez_config(code))[1]
        for code in distinct_aez_codes
    ]
    fetch_start = min(w.get_search_start(year) for w in windows) - timedelta(days=60)
    win_end     = max(w.get_window(year)[1] for w in windows)
    return fetch_start, win_end


def _persistable_results(
    tile_results: list[tuple[str, SeasonResult]],
) -> tuple[list[tuple[str, SeasonResult]], int, int]:
    """
    Filter detection results down to the ones safe to upsert.

    Excludes:
      - errored results
      - empty results (estimated_planting_date is None)
      - "fallback_climatology": ALL THREE signals were unavailable, so the
        date is just the AEZ's hardcoded calendar default, not a detection
        from this farm's actual data. Persisting it would stamp
        planting_processed_at (30-day retry cooldown) and hide the farm
        from Stage 4 eligibility even though nothing was actually observed.

    Returns (ok_results, n_skipped, n_calendar_only).
    """
    ok_results = [
        (fuid, r) for fuid, r in tile_results
        if not r.error
        and r.estimated_planting_date is not None
        and r.method_used != "fallback_climatology"
    ]
    n_skipped = len(tile_results) - len(ok_results)
    n_clim_only = sum(
        1 for _, r in tile_results
        if not r.error and r.method_used == "fallback_climatology"
    )
    return ok_results, n_skipped, n_clim_only


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
            # Still chain forward — a CSV-upload run with zero eligible farms
            # (e.g. crop classifier found nothing plantable) must not leave
            # its upload_id stuck "in progress" on the Django side forever.
            _trigger_downstream(dry_run)
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
            # Fetch the UNION of all distinct AEZ windows present in this
            # tile (see _tile_fetch_window docstring) so every farm's window
            # is fully covered by the shared cache, not just farm[0]'s.
            tile_pairs = [(f.farm_uuid, f.geom_wkt) for f in tile_farms]
            fetch_start, win_end = _tile_fetch_window(tile_farms, season_str, year)

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
            # See _persistable_results() docstring: excludes failures, empty
            # detections, and pure calendar-fallback guesses.
            ok_results, skipped, clim_only = _persistable_results(tile_results)
            if skipped:
                logger.info(
                    "Skipping %d failed/empty/calendar-only detections (not written)"
                    " — %d were pure calendar fallback (no real signal)",
                    skipped, clim_only,
                )
            if ok_results:
                dc.upsert_planting_results(ok_results, dry_run=dry_run)
                all_upsert_rows.extend(ok_results)

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

        # ── Downstream chaining ───────────────────────────────────────────
        _trigger_downstream(dry_run)

        return 0 if total_failed == 0 else 1

    finally:
        dc.close()


def main() -> None:
    settings = get_settings()
    default_season, default_year = _auto_detect_season()

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
