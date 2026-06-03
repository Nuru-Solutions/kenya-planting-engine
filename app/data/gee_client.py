"""
app/data/gee_client.py
GEE authentication singleton.
Service account: precisionfarms@serene-bastion-406504.iam.gserviceaccount.com
"""
from __future__ import annotations
import logging
import os
import threading

logger = logging.getLogger(__name__)
_GEE_INITIALIZED = False
_GEE_LOCK = threading.Lock()   # guard against race in thread pool startup

# Suppress urllib3 "connection pool full" spam — cosmetic when workers > pool size;
# requests still succeed, they just open a new connection instead of reusing one.
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def init_gee() -> None:
    """Initialize GEE once, thread-safe. Safe to call from any worker thread."""
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return

    with _GEE_LOCK:
        # Double-checked locking — another thread may have initialized while we waited
        if _GEE_INITIALIZED:
            return

        try:
            import ee
        except ImportError:
            raise RuntimeError("Run: pip install earthengine-api")

        from app.core.config import get_settings
        s = get_settings()

        if not os.path.exists(s.gee_credentials_path):
            raise FileNotFoundError(
                f"GEE credentials not found: {s.gee_credentials_path}\n"
                f"Expected: secrets/gee-credentials.json"
            )

        creds = ee.ServiceAccountCredentials(
            email=s.gee_service_account,
            key_file=s.gee_credentials_path,
        )
        ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")
        _GEE_INITIALIZED = True
        logger.info(f"GEE ready. Account: {s.gee_service_account}")
