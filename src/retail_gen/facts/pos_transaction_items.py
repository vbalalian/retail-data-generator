"""POS Transaction Items fact table generator (~1.2M at dev scale).

Items per transaction: Poisson(λ=2.8), clipped [1, 15].
Product selection: two-level weighting (category revenue share, then hero 10x boost).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retail_gen.dimensions.products import CATEGORY_CONFIG
from retail_gen.dimensions.customers import LOYALTY_DISCOUNT_PROB

import logging

logger = logging.getLogger(__name__)


class PosTransactionItems:
    """Generates POS line items for an existing set of transactions.

    Not a BatchGenerator subclass — generates all items from a transaction DataFrame,
    writing in chunks. Total row count is determined by Poisson draws, not a fixed target.
    """

    name = "pos_transaction_items"
    schema = pa.schema([
        ("id", pa.int64()),
        ("transaction_id", pa.int64()),
        ("product_id", pa.int64()),
        ("quantity", pa.int32()),
        ("unit_price", pa.float64()),
        ("discount_amount", pa.float64()),
    ])

    def __init__(
        self,
        transaction_df: pd.DataFrame,
        product_instances: list,
        customer_instances: list,
        seed: int = 42,
        chunk_size: int = 100_000,
        output_dir: str | None = None,
    ) -> None:
        self._txn_df = transaction_df
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._chunk_size = chunk_size
        self._output_dir = (
            Path(output_dir or "output") / self.name
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Build product weight arrays (two-level: category then product)
        self._product_ids = np.array(sorted(p.id for p in product_instances))
        self._product_prices = {p.id: p.retail_price for p in product_instances}
        self._product_price_arr = np.array([self._product_prices[pid] for pid in self._product_ids])
        self._product_categories = {p.id: p.category for p in product_instances}

        # Two-level product weights: (cat_weight / n_active_in_cat) * hero_boost
        from collections import Counter
        active_products = [p for p in product_instances if p.status == "Active"]
        cat_counts = Counter(p.category for p in active_products)
        weights = []
        for p in product_instances:
            if p.status != "Active":
                weights.append(0.0)
            else:
                cat_weight = CATEGORY_CONFIG[p.category]["revenue_weight"]
                n_in_cat = cat_counts[p.category]
                hero_boost = 10.0 if p.is_hero_sku else 1.0
                weights.append((cat_weight / n_in_cat) * hero_boost)
        w = np.array(weights)
        self._product_weights = w / w.sum()

        # Customer loyalty tier lookup for discount probability
        self._customer_tiers = {c.id: c.loyalty_tier for c in customer_instances}

    def generate(self) -> None:
        rng = self._rng
        txn_ids = self._txn_df["id"].values
        txn_customer_ids = self._txn_df["customer_id"].values
        n_txns = len(txn_ids)

        # Items per transaction: Poisson(λ=2.8), clipped [1, 15]
        items_per_txn = np.clip(rng.poisson(2.8, size=n_txns), 1, 15)
        total_items = int(items_per_txn.sum())

        # Expand transaction IDs: repeat each txn_id by its item count
        item_txn_ids = np.repeat(txn_ids, items_per_txn)
        item_cust_ids = np.repeat(txn_customer_ids, items_per_txn)

        # Product selection (vectorized)
        product_choices = rng.choice(
            self._product_ids, size=total_items, p=self._product_weights
        )

        # Quantity: Poisson(λ=1.5), min 1
        quantities = np.clip(rng.poisson(1.5, size=total_items), 1, None).astype(np.int32)

        # Unit prices: vectorized via aligned price array + searchsorted
        price_idx = np.searchsorted(self._product_ids, product_choices)
        unit_prices = self._product_price_arr[price_idx]

        # Discount: vectorized by loyalty tier
        subtotals = unit_prices * quantities
        # Build per-item discount probability from customer tier
        tier_arr = np.array([
            self._customer_tiers.get(int(cid), "None") if not pd.isna(cid) else "None"
            for cid in item_cust_ids
        ])
        disc_probs = np.array([LOYALTY_DISCOUNT_PROB.get(t, 0.30) for t in tier_arr])
        # Vectorized: roll once, apply mask
        rolls = rng.random(size=total_items)
        gets_discount = rolls < disc_probs
        pcts = rng.uniform(0.05, 0.30, size=total_items)
        discount_amounts = np.where(gets_discount, np.round(subtotals * pcts, 2), 0.0)

        # Build full DataFrame and write in chunks
        all_ids = np.arange(1, total_items + 1, dtype=np.int64)

        chunk_idx = 0
        for start in range(0, total_items, self._chunk_size):
            end = min(start + self._chunk_size, total_items)
            chunk_df = pd.DataFrame({
                "id": all_ids[start:end],
                "transaction_id": item_txn_ids[start:end],
                "product_id": product_choices[start:end],
                "quantity": quantities[start:end],
                "unit_price": unit_prices[start:end],
                "discount_amount": discount_amounts[start:end],
            })
            table = pa.Table.from_pandas(chunk_df, schema=self.schema)
            out_path = self._output_dir / f"chunk_{chunk_idx:03d}.parquet"
            pq.write_table(table, out_path)
            chunk_idx += 1

        logger.info("%s: done — %d items across %d transactions", self.name, total_items, n_txns)
