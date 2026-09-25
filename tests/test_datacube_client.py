"""
tests/test_datacube_client.py
===============================
Unit tests for app/data/datacube_client.py's DB-write path, specifically
the spatial.farm_season_results addition (see migrations/0001_*.sql):

  - _build_planting_result_row: column count/order regression guard. This
    repo has shipped a row/template column-count mismatch here before
    (commit "Fix planting upsert value count") — these tests fail loudly
    if the row tuple ever drifts out of sync with either SQL statement's
    `incoming (...)` column list.
  - DatacubeClient._verify_schema: fails fast with a migration pointer
    when spatial.farm_season_results hasn't been created yet, instead of
    failing deep inside a threaded upsert call.
  - DatacubeClient.upsert_planting_results: writes farm_intelligence AND
    farm_season_results in the SAME transaction — one commit, and a
    failure in either write rolls back both.

No live PostGIS connection is used anywhere in this file — psycopg2
connections/cursors are mocked throughout.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from unittest import mock

import pytest

from app.core.models import SeasonResult
from app.data.datacube_client import (
    DatacubeClient,
    _build_planting_result_row,
    _UPSERT_PLANTING_SQL,
    _UPSERT_SEASON_RESULT_SQL,
)


def _mk_result(**overrides) -> SeasonResult:
    defaults = dict(
        polygon_id="p1", season="long_rains", year=2026,
        estimated_planting_date=date(2026, 3, 10),
        confidence=0.72, confidence_level="HIGH", method_used="ensemble",
    )
    defaults.update(overrides)
    return SeasonResult(**defaults)


def _incoming_column_count(sql: str) -> int:
    """Count columns declared in a `WITH incoming (...) AS (` CTE header."""
    match = re.search(r"incoming\s*\((.*?)\)\s*AS\s*\(", sql, re.S)
    assert match, "could not find `incoming (...)` CTE header in SQL"
    cols = [c.strip() for c in match.group(1).split(",") if c.strip()]
    return len(cols)


# ── Row builder: shape must match both SQL statements exactly ─────────────────

class TestBuildPlantingResultRow:
    def test_row_length_matches_upsert_template_placeholder_count(self):
        row = _build_planting_result_row("farm-1", _mk_result())
        n_placeholders = DatacubeClient._UPSERT_VALUES_TEMPLATE.count("%s")
        assert len(row) == n_placeholders

    def test_row_length_matches_farm_intelligence_incoming_columns(self):
        row = _build_planting_result_row("farm-1", _mk_result())
        assert len(row) == _incoming_column_count(_UPSERT_PLANTING_SQL)

    def test_row_length_matches_season_results_incoming_columns(self):
        row = _build_planting_result_row("farm-1", _mk_result())
        assert len(row) == _incoming_column_count(_UPSERT_SEASON_RESULT_SQL)

    def test_farm_uuid_and_season_fields_in_expected_positions(self):
        row = _build_planting_result_row(
            "farm-1", _mk_result(season="short_rains", year=2025)
        )
        assert row[0] == "farm-1"
        assert row[2] == "short_rains"   # planting_season
        assert row[3] == 2025            # planting_year

    def test_confidence_and_method_carried_through(self):
        row = _build_planting_result_row(
            "farm-1", _mk_result(confidence=0.55, method_used="rainfall_only")
        )
        assert row[4] == 0.55            # planting_confidence
        assert row[6] == "rainfall_only" # planting_method

    def test_processed_at_and_updated_at_are_equal_and_recent(self):
        before = datetime.utcnow()
        row = _build_planting_result_row("farm-1", _mk_result())
        after = datetime.utcnow()
        processed_at, updated_at = row[14], row[15]
        assert processed_at == updated_at
        assert before <= processed_at <= after


# ── Schema pre-flight check ─────────────────────────────────────────────────

class TestVerifySchema:
    def _client_with_mock_cursor(self, to_regclass_result):
        client = DatacubeClient.__new__(DatacubeClient)  # skip __init__ — no real DB
        conn = mock.MagicMock()
        conn.cursor.return_value.__exit__.return_value = False
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = (to_regclass_result,)
        return client, conn

    def test_raises_with_migration_pointer_when_table_missing(self):
        client, conn = self._client_with_mock_cursor(None)
        with pytest.raises(RuntimeError) as exc_info:
            client._verify_schema(conn)
        msg = str(exc_info.value)
        assert "farm_season_results" in msg
        assert "migrations/0001_create_farm_season_results.sql" in msg

    def test_passes_silently_when_table_exists(self):
        client, conn = self._client_with_mock_cursor("farm_season_results")
        client._verify_schema(conn)  # must not raise


# ── Atomic dual-table upsert ────────────────────────────────────────────────

class TestUpsertWritesBothTablesAtomically:
    def _client_with_mock_conn(self):
        client = DatacubeClient.__new__(DatacubeClient)
        conn = mock.MagicMock()
        conn.cursor.return_value.__exit__.return_value = False
        client._conn = mock.Mock(return_value=conn)
        client._release = mock.Mock()
        return client, conn

    def test_dry_run_never_opens_a_connection(self):
        client = DatacubeClient.__new__(DatacubeClient)
        client._conn = mock.Mock(side_effect=AssertionError("should not connect on dry_run"))
        n = client.upsert_planting_results([("f1", _mk_result())], dry_run=True)
        assert n == 1

    def test_empty_results_is_a_noop(self):
        client = DatacubeClient.__new__(DatacubeClient)
        client._conn = mock.Mock(side_effect=AssertionError("should not connect for empty results"))
        assert client.upsert_planting_results([], dry_run=False) == 0

    @mock.patch("app.data.datacube_client.psycopg2.extras.execute_values")
    def test_writes_both_tables_then_commits_once(self, mock_execute_values):
        client, conn = self._client_with_mock_conn()
        results = [("f1", _mk_result()), ("f2", _mk_result())]

        n = client.upsert_planting_results(results, dry_run=False)

        assert n == 2
        assert mock_execute_values.call_count == 2
        called_sqls = [call.args[1] for call in mock_execute_values.call_args_list]
        assert _UPSERT_PLANTING_SQL in called_sqls
        assert _UPSERT_SEASON_RESULT_SQL in called_sqls
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()
        client._release.assert_called_once_with(conn)

    @mock.patch("app.data.datacube_client.psycopg2.extras.execute_values")
    def test_failure_in_either_write_rolls_back_both(self, mock_execute_values):
        client, conn = self._client_with_mock_conn()
        # First call (farm_intelligence) succeeds, second (farm_season_results) fails.
        mock_execute_values.side_effect = [None, Exception("season table write failed")]

        with pytest.raises(Exception, match="season table write failed"):
            client.upsert_planting_results([("f1", _mk_result())], dry_run=False)

        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()
        client._release.assert_called_once_with(conn)  # connection still released
