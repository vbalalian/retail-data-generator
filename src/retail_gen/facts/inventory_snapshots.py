"""Inventory Snapshots (V1 — simple, plausible levels).

Monthly or quarterly snapshots. One row per product x store for stores that carry the product.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Category stock level params: (base_min, base_max)
STOCK_PARAMS = {
    "Apparel": (50, 200),
    "Home & Kitchen": (20, 100),
    "Electronics": (10, 50),
    "Beauty & Personal Care": (30, 150),
    "Sports & Outdoors": (15, 80),
    "Food & Grocery": (100, 300),
    "Toys & Games": (20, 100),
}

SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("store_id", pa.int64()),
    ("product_id", pa.int64()),
    ("snapshot_date", pa.date32()),
    ("quantity_on_hand", pa.int32()),
    ("reorder_point", pa.int32()),
    ("quantity_on_order", pa.int32()),
])


class InventorySnapshotGenerator:
    def __init__(
        self,
        store_instances: list,
        product_instances: list,
        start_date: date,
        end_date: date,
        frequency: str = "quarterly",
        seed: int = 42,
        chunk_size: int = 100_000,
        output_dir: str | Path = "output",
    ) -> None:
        self._stores = store_instances
        self._products = [p for p in product_instances if p.status == "Active"]
        self._start_date = start_date
        self._end_date = end_date
        self._frequency = frequency
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._chunk_size = chunk_size
        self._output_dir = Path(output_dir) / "inventory_snapshots"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_dates(self) -> list[date]:
        """Generate first-of-month dates at the configured frequency."""
        dates = []
        d = self._start_date.replace(day=1)
        step = 1 if self._frequency == "monthly" else 3
        while d <= self._end_date:
            dates.append(d)
            month = d.month + step
            year = d.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            d = date(year, month, 1)
        return dates

    def generate(self) -> None:
        rng = self._rng
        snap_dates = self._snapshot_dates()
        n_products = len(self._products)

        # Determine which products each store carries
        # Flagship: 90% carry rate, others: 60%
        carry_masks = {}
        for store in self._stores:
            carry_rate = 0.90 if store.store_type == "Flagship" else 0.60
            mask = rng.random(size=n_products) < carry_rate
            carry_masks[store.id] = mask

        # Product category lookup
        product_categories = [p.category for p in self._products]
        product_ids = [p.id for p in self._products]

        # Precompute per-category stock parameters as arrays aligned with product index
        base_means = np.array([(STOCK_PARAMS.get(c, (20, 100))[0] + STOCK_PARAMS.get(c, (20, 100))[1]) / 2
                               for c in product_categories])
        base_stds = np.array([(STOCK_PARAMS.get(c, (20, 100))[1] - STOCK_PARAMS.get(c, (20, 100))[0]) / 4
                              for c in product_categories])
        reorder_points = (base_means * 0.25).astype(np.int32)

        row_id = 1
        chunk_idx = 0

        for snap_date in snap_dates:
            for store in self._stores:
                mask = carry_masks[store.id]
                carried_idx = np.where(mask)[0]
                n_carried = len(carried_idx)
                if n_carried == 0:
                    continue

                # Vectorized stock generation for all carried products in this store
                on_hand = np.maximum(0, rng.normal(
                    base_means[carried_idx], base_stds[carried_idx]
                ).astype(np.int32))
                rp = reorder_points[carried_idx]
                needs_reorder = on_hand < rp
                on_order = np.where(
                    needs_reorder,
                    rng.integers(rp, np.maximum(rp * 3, rp + 1), dtype=np.int32),
                    0,
                )

                ids = np.arange(row_id, row_id + n_carried, dtype=np.int64)
                chunk_df = pd.DataFrame({
                    "id": ids,
                    "store_id": np.full(n_carried, store.id, dtype=np.int64),
                    "product_id": np.array(product_ids)[carried_idx],
                    "snapshot_date": snap_date,
                    "quantity_on_hand": on_hand,
                    "reorder_point": rp,
                    "quantity_on_order": on_order,
                })
                pq.write_table(
                    pa.Table.from_pandas(chunk_df, schema=SCHEMA),
                    self._output_dir / f"chunk_{chunk_idx:03d}.parquet",
                )
                row_id += n_carried
                chunk_idx += 1

        logger.info("inventory_snapshots: %d records across %d snapshots", row_id - 1, len(snap_dates))
