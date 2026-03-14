"""
S3-compatible uploader for LoRA checkpoint files.

Supports DigitalOcean Spaces (and any S3-compatible storage) via S3_ENDPOINT_URL.

Environment variables:
    AWS_ACCESS_KEY_ID      - S3 access key
    AWS_SECRET_ACCESS_KEY  - S3 secret key
    S3_BUCKET_NAME         - Bucket name (e.g. "ai-creation")
    S3_ENDPOINT_URL        - Custom endpoint (e.g. "https://fra1.digitaloceanspaces.com")
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


def _get_s3_client():
    """Create boto3 S3 client with optional custom endpoint for DO Spaces."""
    kwargs = {}
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def upload_checkpoint_dir(checkpoint_dir, model_type, prefix=None):
    """
    Upload all LoRA checkpoint files from a directory to S3.

    Args:
        checkpoint_dir: Path to directory containing .safetensors files
        model_type: Model type string for S3 key prefix
        prefix: Optional custom S3 key prefix (default: lora-outputs/<model_type>/<date>)

    Returns:
        List of dicts: [{"filename", "s3_key", "url", "file_size"}, ...]
    """
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        raise ValueError("S3_BUCKET_NAME environment variable is required")

    checkpoint_dir = Path(checkpoint_dir)
    s3 = _get_s3_client()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")

    if prefix is None:
        prefix = f"lora-outputs/{model_type}/{timestamp}"

    uploaded = []

    for f in sorted(checkpoint_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in LORA_EXTENSIONS:
            continue

        s3_key = f"{prefix}/{f.name}"
        file_size = f.stat().st_size

        logger.info(f"Uploading {f.name} ({file_size / 1024 / 1024:.1f} MB) to s3://{bucket}/{s3_key}")
        s3.upload_file(str(f), bucket, s3_key)

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=7 * 24 * 3600,
        )

        uploaded.append({
            "filename": f.name,
            "s3_key": s3_key,
            "url": url,
            "file_size": file_size,
        })
        logger.info(f"  Uploaded: {s3_key}")

    if not uploaded:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")

    return uploaded
