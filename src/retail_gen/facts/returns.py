"""Returns fact table generator. Samples from existing transaction/order items."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CATEGORY_RETURN_RATES = {
    "Apparel": 0.12,
    "Home & Kitchen": 0.05,
    "Electronics": 0.06,
    "Beauty & Personal Care": 0.04,
    "Sports & Outdoors": 0.05,
    "Food & Grocery": 0.0,
    "Toys & Games": 0.07,
}

REASONS_DEFAULT = ["Defective", "Changed mind", "Wrong item", "Other"]
REASONS_DEFAULT_WEIGHTS = np.array([0.20, 0.25, 0.10, 0.15])
REASONS_DEFAULT_WEIGHTS = REASONS_DEFAULT_WEIGHTS / REASONS_DEFAULT_WEIGHTS.sum()
# "Didn't fit" only for Apparel — added on top, weights renormalized
REASONS_APPAREL = ["Didn't fit", "Defective", "Changed mind", "Wrong item", "Other"]
REASONS_APPAREL_WEIGHTS = np.array([0.30, 0.20, 0.25, 0.10, 0.15])

RETURN_CHANNELS = ["In-store", "Mail"]
RETURN_CHANNEL_WEIGHTS = np.array([0.60, 0.40])

CONDITIONS = ["New", "Opened", "Damaged", "Used"]
CONDITION_WEIGHTS = np.array([0.40, 0.35, 0.15, 0.10])

SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("original_transaction_type", pa.string()),
    ("original_transaction_id", pa.int64()),
    ("product_id", pa.int64()),
    ("customer_id", pa.int64()),  # nullable
    ("return_date", pa.date32()),
    ("reason", pa.string()),
    ("refund_amount", pa.float64()),
    ("return_channel", pa.string()),
    ("condition", pa.string()),
])


class ReturnsGenerator:
    def __init__(
        self,
        pos_items_df: pd.DataFrame,
        pos_txns_df: pd.DataFrame,
        ecom_items_df: pd.DataFrame,
        ecom_orders_df: pd.DataFrame,
        product_instances: list,
        seed: int = 42,
        output_dir: str | Path = "output",
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._output_dir = Path(output_dir) / "returns"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._product_categories = {p.id: p.category for p in product_instances}
        self._product_prices = {p.id: p.retail_price for p in product_instances}

        self._pos_items = pos_items_df
        self._pos_txns = pos_txns_df
        self._ecom_items = ecom_items_df
        self._ecom_orders = ecom_orders_df

    def generate(self) -> None:
        rng = self._rng
        returns_rows = []
        return_id = 1

        # Process POS items then ecommerce items
        for source, items_df, parent_df, txn_type, date_col in [
            ("pos", self._pos_items, self._pos_txns, "pos", "transaction_date"),
            ("ecom", self._ecom_items, self._ecom_orders, "ecommerce", "order_date"),
        ]:
            # Join items with parent to get date and customer_id
            id_col = "transaction_id" if source == "pos" else "order_id"
            parent_id_col = "id"
            merged = items_df.merge(
                parent_df[[parent_id_col, date_col, "customer_id"]],
                left_on=id_col, right_on=parent_id_col, suffixes=("", "_parent"),
            )

            # Determine return probability per item based on category
            cats = merged["product_id"].map(self._product_categories)
            return_probs = cats.map(CATEGORY_RETURN_RATES).fillna(0.0).values

            # Sample which items get returned
            rolls = rng.random(size=len(merged))
            returned_mask = rolls < return_probs
            returned = merged[returned_mask]

            n_ret = len(returned)
            if n_ret == 0:
                continue

            ret_product_ids = returned["product_id"].values.astype(int)
            ret_categories = np.array([self._product_categories.get(pid, "") for pid in ret_product_ids])
            ret_parent_dates = pd.to_datetime(returned[date_col].values)
            ret_offsets = rng.integers(1, 31, size=n_ret)
            ret_dates = (ret_parent_dates + pd.to_timedelta(ret_offsets, unit="D")).date

            # Vectorized reason assignment
            is_apparel = ret_categories == "Apparel"
            reasons = np.where(
                is_apparel,
                rng.choice(REASONS_APPAREL, size=n_ret, p=REASONS_APPAREL_WEIGHTS),
                rng.choice(REASONS_DEFAULT, size=n_ret, p=REASONS_DEFAULT_WEIGHTS),
            )

            ret_prices = np.array([self._product_prices.get(pid, 0.0) for pid in ret_product_ids])
            refunds = np.round(ret_prices * returned["quantity"].values.astype(float), 2)
            channels = rng.choice(RETURN_CHANNELS, size=n_ret, p=RETURN_CHANNEL_WEIGHTS)
            conditions = rng.choice(CONDITIONS, size=n_ret, p=CONDITION_WEIGHTS)

            cust_ids = returned["customer_id"].values
            cust_id_vals = [int(c) if pd.notna(c) else None for c in cust_ids]

            for i in range(n_ret):
                returns_rows.append({
                    "id": return_id,
                    "original_transaction_type": txn_type,
                    "original_transaction_id": int(returned[id_col].values[i]),
                    "product_id": int(ret_product_ids[i]),
                    "customer_id": cust_id_vals[i],
                    "return_date": ret_dates[i],
                    "reason": reasons[i],
                    "refund_amount": float(refunds[i]),
                    "return_channel": channels[i],
                    "condition": conditions[i],
                })
                return_id += 1

        df = pd.DataFrame(returns_rows)
        if len(df) > 0:
            df["customer_id"] = pd.array(df["customer_id"], dtype=pd.Int64Dtype())
            pq.write_table(
                pa.Table.from_pandas(df, schema=SCHEMA),
                self._output_dir / "chunk_000.parquet",
            )
        logger.info("returns: %d records written", len(df))
