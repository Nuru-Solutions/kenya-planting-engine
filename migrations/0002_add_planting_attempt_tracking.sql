-- migrations/0002_add_planting_attempt_tracking.sql
-- =================================================================
-- Stage 4: record every planting-date ATTEMPT, not just successes.
--
-- THE BUG THIS FIXES
-- -------------------
-- Stage 4 selected `ORDER BY tile_id, farm_uuid LIMIT 500` among farms with
-- no planting date for the season. Farms with no usable satellite signal
-- (pure calendar fallback) were deliberately not written -- so nothing
-- recorded that they had been tried, and the very same 500 farms came back
-- first on every run. Confirmed 2026-11-10 and 2026-11-11: identical
-- 500-farm batches, 0 written, and every farm sorting after them (e.g. all
-- of tiles 37M*, Machakos) was never processed at all.
--
-- WHAT CHANGES
-- -------------
-- * Every farm Stage 4 processes now gets a farm_season_results row for the
--   season, recording when it was attempted and the outcome -- including
--   "no_signal" and "error" outcomes, which keep planting_date NULL.
--   A NULL planting_date row therefore means "tried, nothing detected yet";
--   it never overwrites a real planting date.
-- * Eligibility orders by planting_attempted_at NULLS FIRST, so never-tried
--   farms always go first and the rest rotate oldest-attempt-first.
-- * The batch runner loops until every eligible farm has been attempted
--   in the current run (see scripts/run_datacube_batch.py).
--
-- APPLY BEFORE DEPLOYING THE CODE THAT USES THESE COLUMNS
-- --------------------------------------------------------
-- DatacubeClient._verify_schema() fails fast with a pointer back to this
-- file if planting_attempted_at is missing.
--
-- Additive and idempotent: ADD COLUMN IF NOT EXISTS with constant/NULL
-- defaults is a metadata-only change on PostgreSQL 11+ (no table rewrite).
-- =================================================================

-- Lock timeout: ADD COLUMN needs a brief ACCESS EXCLUSIVE lock. If a
-- long-running reader holds the table, fail fast instead of queueing behind
-- it and blocking every other query on the table meanwhile. Just re-run.
SET lock_timeout = '5s';

ALTER TABLE spatial.farm_season_results
    ADD COLUMN IF NOT EXISTS planting_attempted_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS planting_attempt_count  INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS planting_last_outcome   TEXT;

COMMENT ON COLUMN spatial.farm_season_results.planting_attempted_at IS
    'When Stage 4 last tried to detect a planting date for this farm+season, '
    'whatever the outcome. Drives fair rotation of the eligibility queue.';
COMMENT ON COLUMN spatial.farm_season_results.planting_last_outcome IS
    'detected | no_signal (calendar fallback only) | empty | error';

-- Backfill: rows that already hold a detection count as attempted then.
UPDATE spatial.farm_season_results
SET    planting_attempted_at  = COALESCE(planting_attempted_at, planting_processed_at),
       planting_attempt_count = GREATEST(planting_attempt_count, 1),
       planting_last_outcome  = COALESCE(planting_last_outcome, 'detected')
WHERE  planting_date IS NOT NULL
  AND  planting_attempted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_farm_season_results_attempted
    ON spatial.farm_season_results (planting_year, planting_season, planting_attempted_at);

RESET lock_timeout;

-- ── Rollback ─────────────────────────────────────────────────────────────
-- DROP INDEX IF EXISTS spatial.idx_farm_season_results_attempted;
-- DELETE FROM spatial.farm_season_results WHERE planting_date IS NULL;
-- ALTER TABLE spatial.farm_season_results
--     DROP COLUMN IF EXISTS planting_attempted_at,
--     DROP COLUMN IF EXISTS planting_attempt_count,
--     DROP COLUMN IF EXISTS planting_last_outcome;
