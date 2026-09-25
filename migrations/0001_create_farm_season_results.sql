-- migrations/0001_create_farm_season_results.sql
-- =================================================================
-- Stage 4: per-(farm, season, year) planting date history.
--
-- WHY THIS TABLE EXISTS
-- ----------------------
-- spatial.farm_intelligence is a "current state" cache keyed on farm_uuid
-- alone (one row per farm) — that's its original, intentional design: a
-- fast read table for the Django API to show "this farm's current
-- planting date". It was never meant to hold history, but Stage 4's batch
-- runner has only ever written to it, so every re-run overwrites whatever
-- season was last processed: Short Rains 2026 silently replaces Long
-- Rains 2026 for the same farm, and there is no record that LR-2026 ever
-- happened.
--
-- This table adds the missing history: one row per (farm_uuid,
-- planting_season, planting_year), upserted alongside farm_intelligence
-- (same transaction — see DatacubeClient.upsert_planting_results). This
-- is the *only* source of truth for "what was farm X's planting date in
-- season Y of year Z" — spatial.farm_intelligence continues to reflect
-- only the season last written and should not be used for anything that
-- needs season-over-season history.
--
-- APPLY BEFORE DEPLOYING THE CODE THAT WRITES TO THIS TABLE
-- ------------------------------------------------------------
-- app/data/datacube_client.py checks for this table's existence at
-- startup (DatacubeClient._build_pool) and fails fast with a pointer back
-- to this file if it's missing — but that means Stage 4 will hard-fail on
-- every run until this migration has been applied. Run this against the
-- production DB (and any staging DB) BEFORE merging/deploying the
-- accompanying code change, not after.
--
-- Idempotent: safe to re-run (IF NOT EXISTS throughout).
-- =================================================================

CREATE TABLE IF NOT EXISTS spatial.farm_season_results (
    farm_uuid                  UUID          NOT NULL
        REFERENCES spatial.farms (uid) ON DELETE CASCADE,
    planting_season             TEXT          NOT NULL
        CHECK (planting_season IN ('long_rains', 'short_rains', 'third_season')),
    planting_year                SMALLINT      NOT NULL,

    -- Denormalized convenience column, kept in sync with
    -- (planting_year, planting_season) by the application at write time —
    -- mirrors spatial.farm_intelligence.crop_season for easy cross-table
    -- queries/joins without reconstructing the string.
    crop_season                  TEXT,

    planting_date                 DATE,
    planting_confidence            FLOAT,
    planting_confidence_level      TEXT,
    planting_method                 TEXT,

    -- Phenological profile (same fields as farm_intelligence — this table
    -- is the historical counterpart, not a different shape)
    peak_ndvi                       FLOAT,
    peak_ndvi_date                   DATE,
    senescence_date                  DATE,
    season_length_days               SMALLINT,
    ndvi_integral                     FLOAT,
    ndvi_rise_rate                     FLOAT,
    total_rainfall_mm                  FLOAT,

    -- When Stage 4 last (re)computed this farm+season. Re-running the
    -- same season as more satellite data accumulates through it is
    -- expected and should just update this row in place, not create a
    -- duplicate — hence the composite PK below.
    planting_processed_at               TIMESTAMPTZ,

    -- Set once on first insert, never touched again (see upsert SQL's
    -- ON CONFLICT ... DO UPDATE, which omits created_at from SET).
    created_at                           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (farm_uuid, planting_season, planting_year)
);

-- QA / analytics access patterns: method mix and coverage per season,
-- confidence distribution, etc. (the kind of query a data scientist
-- auditing pipeline health runs regularly).
CREATE INDEX IF NOT EXISTS idx_farm_season_results_season
    ON spatial.farm_season_results (planting_year, planting_season);

CREATE INDEX IF NOT EXISTS idx_farm_season_results_method
    ON spatial.farm_season_results (planting_method);

COMMENT ON TABLE spatial.farm_season_results IS
    'One row per (farm, season, year) — full planting-date detection '
    'history. spatial.farm_intelligence only ever reflects the most '
    'recently processed season for a farm; this table is the source of '
    'truth for anything season-over-season (yield trend analysis, '
    'QA/coverage dashboards, audits).';

-- ── Rollback ─────────────────────────────────────────────────────────────
-- DROP TABLE IF EXISTS spatial.farm_season_results;
-- (No other object depends on this table, so a straight DROP is safe to
-- run standalone if this migration needs to be reverted.)
