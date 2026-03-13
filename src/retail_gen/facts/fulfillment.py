"""Fulfillment fact table generator. 1:1 with non-cancelled ecommerce orders."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
CARRIER_WEIGHTS = np.array([0.35, 0.30, 0.25, 0.10])

SHIPPING_OFFSETS = {
    "Standard": (5, 7),
    "Express": (2, 3),
    "Next-day": (1, 1),
}

SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("order_id", pa.int64()),
    ("ship_date", pa.timestamp("us")),
    ("estimated_delivery_date", pa.timestamp("us")),
    ("actual_delivery_date", pa.timestamp("us")),  # nullable
    ("carrier", pa.string()),
    ("tracking_number", pa.string()),
    ("status", pa.string()),
    ("is_on_time", pa.bool_()),
])


class FulfillmentGenerator:
    def __init__(
        self,
        orders_df: pd.DataFrame,
        generation_date: date,
        seed: int = 42,
        output_dir: str | Path = "output",
    ) -> None:
        # Filter to non-cancelled orders
        self._orders = orders_df[orders_df["status"] != "Cancelled"].copy()
        self._generation_date = pd.Timestamp(generation_date)
        self._rng = np.random.default_rng(seed)
        self._output_dir = Path(output_dir) / "fulfillment"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> None:
        rng = self._rng
        orders = self._orders
        n = len(orders)

        order_ids = orders["id"].to_numpy()
        order_dates = pd.Series(pd.to_datetime(orders["order_date"].to_numpy()), name="order_date")
        ship_methods = orders["shipping_method"].to_numpy()

        # Ship date: order_date + Poisson(λ=1) days
        ship_offsets = pd.to_timedelta(rng.poisson(1, size=n), unit="D")
        ship_dates = pd.Series(order_dates.values + ship_offsets.values, dtype="datetime64[us]")

        # Estimated delivery: ship_date + method offset
        est_offsets = np.array([
            rng.integers(SHIPPING_OFFSETS[m][0], SHIPPING_OFFSETS[m][1] + 1)
            for m in ship_methods
        ])
        est_delivery = pd.Series(
            ship_dates.values + pd.to_timedelta(est_offsets, unit="D").values,
            dtype="datetime64[us]",
        )

        # Actual delivery: estimated + Gaussian(μ=-1.5, σ=1.0) + right-skew
        # Right-skew: add exponential(1.5) to ~8% of deliveries
        # Calibrated to yield ~92% on-time rate
        delta_days = rng.normal(-1.5, 1.0, size=n)
        is_late_skew = rng.random(size=n) < 0.08
        delta_days = np.where(
            is_late_skew,
            delta_days + np.abs(rng.exponential(1.5, size=n)),
            delta_days,
        )
        actual_delivery = pd.Series(
            est_delivery.values + pd.to_timedelta(np.round(delta_days).astype(int), unit="D").values,
            dtype="datetime64[us]",
        )

        # Determine which orders are not yet delivered (Pending/Shipped status)
        order_status = orders["status"].to_numpy()
        not_delivered = np.isin(order_status, ["Pending", "Shipped"])

        # is_on_time: null for undelivered orders (not False — they haven't failed yet)
        is_on_time_raw = (actual_delivery <= est_delivery).to_numpy()
        is_on_time_series = pd.array(is_on_time_raw, dtype=pd.BooleanDtype())
        is_on_time_series[not_delivered] = pd.NA

        # Null out actual_delivery for not-yet-delivered orders
        actual_delivery_series = actual_delivery.copy()
        actual_delivery_series[not_delivered] = pd.NaT

        # Carrier
        carrier_idx = rng.choice(len(CARRIERS), size=n, p=CARRIER_WEIGHTS)
        carriers = np.array(CARRIERS)[carrier_idx]

        # Tracking numbers (UUID4-style from two 64-bit halves)
        hi = rng.integers(0, 2**63, size=n, dtype=np.int64)
        lo = rng.integers(0, 2**63, size=n, dtype=np.int64)
        tracking = [
            str(uuid.UUID(int=(int(h) << 64) | int(l)))
            for h, l in zip(hi, lo)
        ]

        # Fulfillment status derived from dates
        ff_statuses = []
        for i in range(n):
            if not_delivered[i]:
                ff_statuses.append(order_status[i])  # Pending or Shipped
            else:
                ff_statuses.append("Delivered")

        df = pd.DataFrame({
            "id": np.arange(1, n + 1, dtype=np.int64),
            "order_id": order_ids,
            "ship_date": ship_dates,
            "estimated_delivery_date": est_delivery,
            "actual_delivery_date": actual_delivery_series,
            "carrier": carriers,
            "tracking_number": tracking,
            "status": ff_statuses,
            "is_on_time": is_on_time_series,
        })

        pq.write_table(
            pa.Table.from_pandas(df, schema=SCHEMA),
            self._output_dir / "chunk_000.parquet",
        )
        logger.info("fulfillment: %d records written", n)
