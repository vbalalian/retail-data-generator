"""Load Parquet files from GCS into BigQuery."""

from __future__ import annotations

import logging

from google.cloud import bigquery

from retail_gen.config import Config

logger = logging.getLogger(__name__)


def load_to_bigquery(cfg: Config) -> None:
    """Load all entity Parquet files from GCS into BigQuery tables."""
    if not cfg.gcs_bucket:
        logger.warning("GCS_BUCKET not set, skipping BQ load")
        return

    client = bigquery.Client(project=cfg.bq_project)

    # Ensure dataset exists
    dataset_ref = f"{cfg.bq_project}.{cfg.bq_dataset}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)
        logger.info("Created dataset %s", dataset_ref)

    # Load each entity
    output = cfg.output_dir
    for entity_dir in sorted(output.iterdir()):
        if not entity_dir.is_dir():
            continue
        entity_name = entity_dir.name
        table_id = f"{dataset_ref}.{entity_name}"
        source_uri = f"gs://{cfg.gcs_bucket}/retail_raw/{entity_name}/*.parquet"

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        load_job = client.load_table_from_uri(
            source_uri, table_id, job_config=job_config
        )
        load_job.result()  # Wait for completion
        table = client.get_table(table_id)
        logger.info("Loaded %s: %d rows", entity_name, table.num_rows)
