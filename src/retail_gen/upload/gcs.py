"""Upload generated Parquet files to Google Cloud Storage."""

from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import storage

from retail_gen.config import Config

logger = logging.getLogger(__name__)


def upload_to_gcs(cfg: Config) -> None:
    """Upload all Parquet files from output/ to GCS bucket."""
    if not cfg.gcs_bucket:
        logger.warning("GCS_BUCKET not set, skipping upload")
        return

    client = storage.Client(project=cfg.bq_project)
    bucket = client.bucket(cfg.gcs_bucket)

    output = cfg.output_dir
    for entity_dir in sorted(output.iterdir()):
        if not entity_dir.is_dir():
            continue
        entity_name = entity_dir.name
        for parquet_file in sorted(entity_dir.glob("*.parquet")):
            blob_name = f"retail_raw/{entity_name}/{parquet_file.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(parquet_file))
            logger.info("Uploaded gs://%s/%s", cfg.gcs_bucket, blob_name)
