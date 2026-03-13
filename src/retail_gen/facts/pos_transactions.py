"""POS Transactions fact table generator (~400K at dev scale)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa

from retail_gen.lib.batch import BatchGenerator
from retail_gen.lib.providers.temporal import generate_timestamps
from retail_gen.dimensions.customers import LOYALTY_FREQUENCY

PAYMENT_METHODS = ["Credit", "Debit", "Cash", "Mobile"]
PAYMENT_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10])


class PosTransactions(BatchGenerator):
    name = "pos_transactions"
    schema = pa.schema([
        ("id", pa.int64()),
        ("store_id", pa.int64()),
        ("customer_id", pa.int64()),  # nullable
        ("transaction_date", pa.timestamp("us")),
        ("payment_method", pa.string()),
        ("employee_id", pa.int32()),
    ])

    def __init__(
        self,
        total_rows: int,
        store_instances: list,
        customer_instances: list,
        start_date: date,
        end_date: date,
        seed: int = 42,
        chunk_size: int = 100_000,
        output_dir: str | None = None,
    ) -> None:
        super().__init__(total_rows=total_rows, chunk_size=chunk_size, output_dir=output_dir or "output")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Precompute store weights (by square_footage)
        self._store_ids = np.array([s.id for s in store_instances])
        sqft = np.array([s.square_footage for s in store_instances], dtype=np.float64)
        self._store_weights = sqft / sqft.sum()
        # Store opening dates for filtering (no txn before a store opens)
        self._store_open_dates = {s.id: np.datetime64(s.opening_date) for s in store_instances}

        # Precompute eligible customer pool (non-churned, in-store/both preference)
        eligible = [
            c for c in customer_instances
            if c.status != "Churned" and c.channel_preference in ("In-store", "Both")
        ]
        self._customer_ids = np.array([c.id for c in eligible])
        # Weight by loyalty tier frequency multiplier
        freq = np.array([LOYALTY_FREQUENCY.get(c.loyalty_tier, 1.0) for c in eligible])
        self._customer_weights = freq / freq.sum()

        # Pre-generate all timestamps
        self._timestamps = generate_timestamps(
            n=total_rows, start_date=start_date, end_date=end_date,
            yoy_growth=0.08, seed=seed,
        )

    def generate_chunk(self, chunk_idx: int, start_id: int, n_rows: int) -> pd.DataFrame:
        rng = self._rng

        ids = np.arange(start_id, start_id + n_rows, dtype=np.int64)

        # Timestamps for this chunk (MUST come before store filtering)
        ts_start = start_id - 1
        ts_end = ts_start + n_rows
        timestamps = self._timestamps[ts_start:ts_end]

        # Store assignment (weighted by square footage)
        store_ids = rng.choice(self._store_ids, size=n_rows, p=self._store_weights)

        # Re-roll stores that weren't open yet at the transaction timestamp
        ts_dates = timestamps.astype("datetime64[D]")
        for _ in range(5):  # bounded retry
            bad = np.array([ts_dates[i] < self._store_open_dates[store_ids[i]] for i in range(n_rows)])
            if not bad.any():
                break
            store_ids[bad] = rng.choice(self._store_ids, size=int(bad.sum()), p=self._store_weights)

        # Customer assignment: 70% identified, 30% null
        is_identified = rng.random(size=n_rows) < 0.70
        cust_choices = rng.choice(self._customer_ids, size=n_rows, p=self._customer_weights)
        customer_ids = np.where(is_identified, cust_choices, -1)  # -1 = null placeholder

        # Payment method
        payment_idx = rng.choice(len(PAYMENT_METHODS), size=n_rows, p=PAYMENT_WEIGHTS)
        payments = np.array(PAYMENT_METHODS)[payment_idx]

        # Employee IDs
        employee_ids = rng.integers(1, 2001, size=n_rows, dtype=np.int32)

        df = pd.DataFrame({
            "id": ids,
            "store_id": store_ids,
            "customer_id": pd.array(
                [int(c) if c != -1 else pd.NA for c in customer_ids],
                dtype=pd.Int64Dtype(),
            ),
            "transaction_date": pd.to_datetime(timestamps),
            "payment_method": payments,
            "employee_id": employee_ids,
        })
        return df
