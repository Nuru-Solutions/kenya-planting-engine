"""
app/core/s3_uploader.py
Upload pipeline outputs to S3.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_outputs(local_paths: list[str], s3_prefix: str) -> list[str]:
    """
    Upload a list of local files to S3 under the given prefix.
    Returns a list of S3 URIs for the uploaded files.

    Requires AWS credentials in .env (read via Settings).
    """
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 upload")
        return []

    from app.core.config import get_settings
    s = get_settings()

    if not getattr(s, "s3_bucket", None):
        logger.warning("S3_BUCKET not configured — skipping upload")
        return []

    client = boto3.client(
        "s3",
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
    )

    uploaded = []
    for path in local_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue
        key = f"{s3_prefix}/{Path(path).name}"
        try:
            client.upload_file(path, s.s3_bucket, key)
            uri = f"s3://{s.s3_bucket}/{key}"
            uploaded.append(uri)
            logger.info(f"Uploaded: {uri}")
        except Exception as e:
            logger.error(f"S3 upload failed for {path}: {e}")

    return uploaded
