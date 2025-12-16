from __future__ import annotations

import boto3

from py_core.config.settings import settings


def s3_is_configured() -> bool:
    return bool(
        settings.s3_endpoint
        and settings.s3_access_key
        and settings.s3_secret_key
        and settings.s3_bucket_name
    )


def get_s3_client():
    if not s3_is_configured():
        raise RuntimeError(
            "S3 is not configured; set S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET_NAME"
        )

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        use_ssl=settings.s3_secure,
    )


def upload_file(local_path: str, key: str) -> str:
    client = get_s3_client()
    client.upload_file(local_path, settings.s3_bucket_name, key)
    return key


def download_file(key: str, local_path: str) -> str:
    client = get_s3_client()
    client.download_file(settings.s3_bucket_name, key, local_path)
    return local_path
