"""BatchGenerator base class for high-volume fact entities.

Subclasses define a PyArrow schema and implement generate_chunk().
Generates data in configurable chunks, writing each to a Parquet file.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class BatchGenerator(ABC):
    """Base class for chunked fact table generation."""

    name: str  # Entity name, used for output directory
    schema: pa.Schema  # PyArrow schema for the output

    def __init__(
        self,
        total_rows: int,
        chunk_size: int = 100_000,
        output_dir: Path | str = "output",
    ) -> None:
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir) / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_chunk(
        self, chunk_idx: int, start_id: int, n_rows: int
    ) -> pd.DataFrame:
        """Generate a single chunk of rows. Must return a DataFrame matching self.schema."""
        ...

    def generate(self) -> None:
        """Generate all chunks, writing each to a Parquet file."""
        remaining = self.total_rows
        start_id = 1
        chunk_idx = 0

        while remaining > 0:
            n_rows = min(self.chunk_size, remaining)
            logger.info(
                "%s: generating chunk %d (%d rows, ids %d-%d)",
                self.name, chunk_idx, n_rows, start_id, start_id + n_rows - 1,
            )

            df = self.generate_chunk(chunk_idx, start_id, n_rows)
            table = pa.Table.from_pandas(df, schema=self.schema)

            out_path = self.output_dir / f"chunk_{chunk_idx:03d}.parquet"
            pq.write_table(table, out_path)

            start_id += n_rows
            remaining -= n_rows
            chunk_idx += 1

        logger.info("%s: done — %d rows in %d chunks", self.name, self.total_rows, chunk_idx)
