"""
tests/test_stac_fetcher.py
============================
Regression guard for a bug that silently broke every Sentinel-1 (SAR) read
in production while leaving Sentinel-2 (NDVI) untouched.

RASTERIO_ENV's CPL_VSIL_CURL_ALLOWED_EXTENSIONS controls which file
extensions GDAL's vsicurl layer will recognize as valid datasets at all.
Sentinel-2 COG assets use ".tif" (single-f); Sentinel-1 L1C assets use
".tiff" (double-f, legacy SAFE-format naming, e.g. "iw-vv.tiff"). The
allowlist only had ".tif"/".TIF" — so GDAL rejected every single Sentinel-1
asset with "does not exist in the file system, and is not recognized as a
supported dataset name", an error message that looks exactly like an IAM,
region, or path-format problem and sent debugging in all three of those
directions before the actual one-line cause was found.

Reproduced directly (2026-09-28) against the real production Docker image
and a real S3 object: identical credentials, identical href — fails
without ".tiff"/".TIFF" in this list, succeeds with them.
"""
from __future__ import annotations

from app.data.stac_fetcher import RASTERIO_ENV


class TestRasterioEnvExtensionAllowlist:
    def test_includes_single_f_tif_variants(self):
        """Sentinel-2 COG assets (e.g. B08.tif)."""
        allowed = RASTERIO_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]
        assert ".tif" in allowed.split(",")
        assert ".TIF" in allowed.split(",")

    def test_includes_double_f_tiff_variants(self):
        """
        Sentinel-1 L1C assets (e.g. iw-vv.tiff). This is the specific
        regression: these were missing, and GDAL treats this as an exact
        string allowlist — ".tif" does NOT also match ".tiff".
        """
        allowed = RASTERIO_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]
        assert ".tiff" in allowed.split(",")
        assert ".TIFF" in allowed.split(",")

    def test_requester_pays_still_set(self):
        """Sentinel-1 GRD assets on sentinel-s1-l1c require requester-pays."""
        assert RASTERIO_ENV["AWS_REQUEST_PAYER"] == "requester"
