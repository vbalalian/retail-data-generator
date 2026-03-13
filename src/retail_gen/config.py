"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """All generation parameters derived from environment variables."""

    scale: float = field(default_factory=lambda: float(os.getenv("SCALE", "0.01")))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    history_start: date = field(
        default_factory=lambda: date.fromisoformat(
            os.getenv("HISTORY_START", "2023-01-01")
        )
    )
    generation_date: date = field(default_factory=lambda: date.today())
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output"))
    )

    # GCP settings
    bq_project: str = field(
        default_factory=lambda: os.getenv("BQ_PROJECT", "mkt-analytics-project")
    )
    bq_dataset: str = field(
        default_factory=lambda: os.getenv("BQ_DATASET", "retail_raw")
    )
    gcs_bucket: str = field(
        default_factory=lambda: os.getenv("GCS_BUCKET", "")
    )

    # --- Fixed dimensions (do not scale) ---
    n_stores: int = 150
    n_products: int = 5000

    # --- Scaled counts ---
    @property
    def n_customers(self) -> int:
        return int(2_000_000 * self.scale)

    @property
    def n_pos_transactions(self) -> int:
        return int(40_000_000 * self.scale)

    @property
    def n_ecommerce_orders(self) -> int:
        return int(25_000_000 * self.scale)

    @property
    def inventory_frequency(self) -> str:
        """Monthly snapshots at SCALE >= 0.1, quarterly below."""
        return "monthly" if self.scale >= 0.1 else "quarterly"
