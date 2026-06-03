"""
app/data/cache.py
=================
Thread-safe disk cache for raw GEE observations.

Caches NDVIObservation / SARObservation / RainfallRecord lists keyed by
(polygon_id, data_source, start_date, end_date).

Benefit: if a multi-hour run fails at farm #500, restarting skips
all already-fetched GEE calls and resumes from where it left off.
Scoring is always re-run from raw observations, so changing scoring
parameters never requires clearing the cache.

Layout:
    .cache/gee/
        s2/    {key}.json   # NDVIObservation list
        s1/    {key}.json   # SARObservation list
        chirps/{key}.json   # RainfallRecord list

Cache key: first 20 chars of polygon_id + 8-char MD5 of full id+dates
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Absolute path anchored to project root (cache.py is at app/data/cache.py,
# parents[0]=app/data, parents[1]=app, parents[2]=project root).
# Relative paths cause [Errno 22] on Windows when opened from worker threads.
_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache" / "gee"
_write_lock = threading.Lock()   # prevent concurrent writes to the same file


# ── Key generation ─────────────────────────────────────────────────────────────

def _key(polygon_id: str, start: date, end: date) -> str:
    """Short, filesystem-safe, collision-resistant cache key."""
    raw = f"{polygon_id}|{start}|{end}".encode()
    h   = hashlib.md5(raw).hexdigest()[:8]
    # Sanitise polygon_id prefix (remove chars that break Windows filenames)
    prefix = "".join(c for c in polygon_id[:20] if c.isalnum() or c in "-_")
    return f"{prefix}_{h}"


# ── Public API ─────────────────────────────────────────────────────────────────

def get(source: str, polygon_id: str, start: date, end: date) -> list[dict] | None:
    """
    Return cached raw observation dicts, or None on cache miss / read error.

    source: one of 's2', 's1', 'chirps'
    """
    path = _CACHE_ROOT / source / f"{_key(polygon_id, start, end)}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        logger.debug(f"Cache HIT  {source}/{polygon_id[:12]}…  {start}→{end}  ({len(data)} records)")
        return data
    except Exception as e:
        logger.warning(f"Cache read error {path}: {e} — will re-fetch from GEE")
        return None


def put(
    source: str,
    polygon_id: str,
    start: date,
    end: date,
    records: list[Any],       # list of Pydantic models OR plain dicts
) -> None:
    """Serialize and write observation list to disk cache (thread-safe)."""
    path = _CACHE_ROOT / source / f"{_key(polygon_id, start, end)}.json"
    try:
        # Serialize Pydantic models → dicts if needed
        if records and hasattr(records[0], "model_dump"):
            payload = [r.model_dump(mode="json") for r in records]
        else:
            payload = records

        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, default=str)
        logger.debug(
            f"Cache WRITE {source}/{polygon_id[:12]}…  {start}→{end}  ({len(records)} records)"
        )
    except Exception as e:
        logger.warning(f"Cache write error {path}: {e}")


def clear(source: str | None = None) -> int:
    """
    Delete cached files.
    source=None  → clear everything
    source='s2'  → clear only Sentinel-2 cache
    Returns number of files deleted.
    """
    root = _CACHE_ROOT / source if source else _CACHE_ROOT
    deleted = 0
    for p in root.rglob("*.json"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def stats() -> dict[str, int]:
    """Return per-source file counts for a quick cache size check."""
    result = {}
    for src in ["s2", "s1", "chirps"]:
        d = _CACHE_ROOT / src
        result[src] = len(list(d.glob("*.json"))) if d.exists() else 0
    return result
