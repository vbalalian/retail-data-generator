"""Ecommerce Orders + Items two-pass generator.

Does NOT extend BatchGenerator. Orders and items are co-generated because
shipping_cost depends on order totals (free shipping over $75).

Pass 1: Generate orders (without shipping_cost) and items in memory.
Pass 2: Compute order totals, backfill shipping_cost, write both to Parquet.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from retail_gen.dimensions.customers import LOYALTY_DISCOUNT_PROB, LOYALTY_FREQUENCY
from retail_gen.dimensions.products import CATEGORY_CONFIG
from retail_gen.lib.providers.address import AddressProvider
from retail_gen.lib.providers.temporal import generate_timestamps

logger = logging.getLogger(__name__)

CHANNELS = ["Web", "Mobile App", "Mobile Web"]
CHANNEL_WEIGHTS = np.array([0.55, 0.30, 0.15])

SHIPPING_METHODS = ["Standard", "Express", "Next-day"]
SHIPPING_WEIGHTS = np.array([0.60, 0.25, 0.15])
SHIPPING_COSTS = {"Standard": 5.99, "Express": 12.99, "Next-day": 19.99}

# Order status by age bucket
STATUS_BUCKETS = {
    "recent": (["Pending", "Shipped", "Delivered"], [0.40, 0.50, 0.10]),       # < 2 days
    "mid": (["Pending", "Shipped", "Delivered", "Cancelled", "Returned"],
            [0.05, 0.25, 0.60, 0.05, 0.05]),                                   # 2-7 days
    "old": (["Delivered", "Cancelled", "Returned"], [0.85, 0.05, 0.10]),        # > 7 days
}

ORDERS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("customer_id", pa.int64()),
    ("order_date", pa.timestamp("us")),
    ("status", pa.string()),
    ("shipping_address_city", pa.string()),
    ("shipping_address_state", pa.string()),
    ("channel", pa.string()),
    ("shipping_method", pa.string()),
    ("shipping_cost", pa.float64()),
])

ITEMS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("order_id", pa.int64()),
    ("product_id", pa.int64()),
    ("quantity", pa.int32()),
    ("unit_price", pa.float64()),
    ("discount_amount", pa.float64()),
])


class EcommerceGenerator:
    """Two-pass generator for ecommerce orders and items."""

    def __init__(
        self,
        total_orders: int,
        customer_instances: list,
        product_instances: list,
        start_date: date,
        end_date: date,
        seed: int = 42,
        output_dir: str | Path = "output",
    ) -> None:
        self._total_orders = total_orders
        self._start_date = start_date
        self._end_date = end_date
        self._generation_date = end_date
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._output_dir = Path(output_dir)

        # Eligible customers: non-churned, online/both pref
        eligible = [
            c for c in customer_instances
            if c.status != "Churned" and c.channel_preference in ("Online", "Both")
        ]
        self._customer_ids = np.array([c.id for c in eligible])
        self._customer_cities = {c.id: c.city for c in eligible}
        self._customer_states = {c.id: c.state for c in eligible}
        self._customer_tiers = {c.id: c.loyalty_tier for c in eligible}
        freq = np.array([LOYALTY_FREQUENCY.get(c.loyalty_tier, 1.0) for c in eligible])
        self._customer_weights = freq / freq.sum()

        # Two-level product weights: (cat_weight / n_active_in_cat) * hero_boost
        from collections import Counter
        self._product_ids = np.array(sorted(p.id for p in product_instances))
        self._product_prices = {p.id: p.retail_price for p in product_instances}
        self._product_price_arr = np.array([self._product_prices[pid] for pid in self._product_ids])
        active_products = [p for p in product_instances if p.status == "Active"]
        cat_counts = Counter(p.category for p in active_products)
        weights = []
        for p in product_instances:
            if p.status != "Active":
                weights.append(0.0)
            else:
                cat_w = CATEGORY_CONFIG[p.category]["revenue_weight"]
                n_in_cat = cat_counts[p.category]
                hero = 10.0 if p.is_hero_sku else 1.0
                weights.append((cat_w / n_in_cat) * hero)
        w = np.array(weights)
        self._product_weights = w / w.sum()

        self._addr_provider = AddressProvider(seed=seed)

    def generate(self) -> None:
        rng = self._rng
        n = self._total_orders

        # --- PASS 1: Generate orders + items in memory ---

        # Timestamps
        timestamps = generate_timestamps(
            n=n, start_date=self._start_date, end_date=self._end_date,
            yoy_growth=0.12, seed=self._seed + 1,  # offset seed from POS
        )

        # Customer assignment
        customer_ids = rng.choice(self._customer_ids, size=n, p=self._customer_weights)

        # Shipping address: 80% same as customer, 20% different
        use_customer_addr = rng.random(size=n) < 0.80
        alt_addrs = self._addr_provider.sample(n)
        ship_cities = [
            self._customer_cities[cid] if use_customer_addr[i] else alt_addrs["city"][i]
            for i, cid in enumerate(customer_ids)
        ]
        ship_states = [
            self._customer_states[cid] if use_customer_addr[i] else alt_addrs["state"][i]
            for i, cid in enumerate(customer_ids)
        ]

        # Channel, shipping method
        channels = np.array(CHANNELS)[rng.choice(len(CHANNELS), size=n, p=CHANNEL_WEIGHTS)]
        ship_methods = np.array(SHIPPING_METHODS)[rng.choice(len(SHIPPING_METHODS), size=n, p=SHIPPING_WEIGHTS)]

        # Status by age
        order_dates_dt = pd.to_datetime(timestamps)
        gen_dt = pd.Timestamp(self._generation_date)
        age_days = (gen_dt - order_dates_dt).days
        statuses = []
        for age in age_days:
            if age < 2:
                bucket = STATUS_BUCKETS["recent"]
            elif age <= 7:
                bucket = STATUS_BUCKETS["mid"]
            else:
                bucket = STATUS_BUCKETS["old"]
            statuses.append(rng.choice(bucket[0], p=bucket[1]))

        # --- Generate items ---
        items_per_order = np.clip(rng.poisson(2.2, size=n), 1, 10)
        total_items = int(items_per_order.sum())

        item_order_ids = np.repeat(np.arange(1, n + 1, dtype=np.int64), items_per_order)
        item_cust_ids = np.repeat(customer_ids, items_per_order)

        product_choices = rng.choice(self._product_ids, size=total_items, p=self._product_weights)
        quantities = np.clip(rng.poisson(1.5, size=total_items), 1, None).astype(np.int32)
        # Vectorized price lookup via sorted product_ids + searchsorted
        price_idx = np.searchsorted(self._product_ids, product_choices)
        unit_prices = self._product_price_arr[price_idx]

        # Discounts: vectorized, loyalty-tier based
        # Ecommerce has ~35% discount rate (vs POS ~30%), so scale up by 0.35/0.30
        subtotals = unit_prices * quantities
        tier_arr = np.array([self._customer_tiers.get(int(cid), "None") for cid in item_cust_ids])
        disc_probs = np.array([LOYALTY_DISCOUNT_PROB.get(t, 0.30) for t in tier_arr])
        disc_probs = disc_probs * (0.35 / 0.30)  # scale POS base rate to ecommerce rate
        rolls = rng.random(size=total_items)
        gets_discount = rolls < disc_probs
        pcts = rng.uniform(0.05, 0.25, size=total_items)
        discount_amounts = np.where(gets_discount, np.round(subtotals * pcts, 2), 0.0)

        items_df = pd.DataFrame({
            "id": np.arange(1, total_items + 1, dtype=np.int64),
            "order_id": item_order_ids,
            "product_id": product_choices,
            "quantity": quantities,
            "unit_price": unit_prices,
            "discount_amount": discount_amounts,
        })

        # --- PASS 2: Compute order totals, backfill shipping_cost ---
        items_df["line_total"] = items_df["unit_price"] * items_df["quantity"] - items_df["discount_amount"]
        order_totals = items_df.groupby("order_id")["line_total"].sum()

        shipping_costs = np.array([SHIPPING_COSTS[m] for m in ship_methods])
        for i in range(n):
            order_id = i + 1
            if order_totals.get(order_id, 0) > 75:
                shipping_costs[i] = 0.0

        # Build orders DataFrame
        orders_df = pd.DataFrame({
            "id": np.arange(1, n + 1, dtype=np.int64),
            "customer_id": customer_ids,
            "order_date": order_dates_dt,
            "status": statuses,
            "shipping_address_city": ship_cities,
            "shipping_address_state": ship_states,
            "channel": channels,
            "shipping_method": ship_methods,
            "shipping_cost": shipping_costs,
        })

        # Drop helper column from items
        items_df = items_df.drop(columns=["line_total"])

        # --- Write to Parquet ---
        orders_dir = self._output_dir / "ecommerce_orders"
        orders_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(orders_df, schema=ORDERS_SCHEMA),
            orders_dir / "chunk_000.parquet",
        )

        items_dir = self._output_dir / "ecommerce_order_items"
        items_dir.mkdir(parents=True, exist_ok=True)
        # Write items in chunks
        chunk_size = 100_000
        for idx, start in enumerate(range(0, len(items_df), chunk_size)):
            chunk = items_df.iloc[start : start + chunk_size]
            pq.write_table(
                pa.Table.from_pandas(chunk, schema=ITEMS_SCHEMA),
                items_dir / f"chunk_{idx:03d}.parquet",
            )

        logger.info(
            "ecommerce: %d orders, %d items written", len(orders_df), len(items_df)
        )
