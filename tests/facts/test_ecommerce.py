from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _setup_dimensions():
    from retail_gen.dimensions.stores import Store
    from retail_gen.dimensions.products import Product
    from retail_gen.dimensions.customers import Customer

    Store.generate(10, seed=42)
    Product.generate(50, seed=42)
    Customer.generate(100, store_instances=Store.instances, seed=42)
    return Store, Product, Customer


def test_ecommerce_generates_orders_and_items(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.ecommerce import EcommerceGenerator

    gen = EcommerceGenerator(
        total_orders=200,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    orders = pd.read_parquet(tmp_path / "ecommerce_orders")
    items = pd.read_parquet(tmp_path / "ecommerce_order_items")
    assert len(orders) == 200
    assert len(items) > 200  # at least 1 item per order


def test_ecommerce_orders_columns(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.ecommerce import EcommerceGenerator

    gen = EcommerceGenerator(
        total_orders=50,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    orders = pd.read_parquet(tmp_path / "ecommerce_orders")
    expected = {"id", "customer_id", "order_date", "status", "shipping_address_city",
                "shipping_address_state", "channel", "shipping_method", "shipping_cost"}
    assert set(orders.columns) == expected


def test_ecommerce_free_shipping_over_75(tmp_path):
    """Orders with total > $75 should have shipping_cost = 0."""
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.ecommerce import EcommerceGenerator

    gen = EcommerceGenerator(
        total_orders=500,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    orders = pd.read_parquet(tmp_path / "ecommerce_orders")
    items = pd.read_parquet(tmp_path / "ecommerce_order_items")

    # Compute order totals
    items["line_total"] = items["unit_price"] * items["quantity"] - items["discount_amount"]
    order_totals = items.groupby("order_id")["line_total"].sum().reset_index()
    merged = orders.merge(order_totals, left_on="id", right_on="order_id")

    # All orders > $75 should have free shipping
    expensive = merged[merged["line_total"] > 75]
    if len(expensive) > 0:
        assert (expensive["shipping_cost"] == 0.0).all()


def test_ecommerce_status_by_age(tmp_path):
    """Orders > 7 days old should mostly be Delivered."""
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.ecommerce import EcommerceGenerator

    gen = EcommerceGenerator(
        total_orders=1000,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    orders = pd.read_parquet(tmp_path / "ecommerce_orders")
    orders["age_days"] = (pd.Timestamp(date(2024, 1, 1)) - orders["order_date"]).dt.days
    old_orders = orders[orders["age_days"] > 7]
    if len(old_orders) > 0:
        delivered_pct = (old_orders["status"] == "Delivered").mean()
        assert delivered_pct > 0.70  # spec says 85%, allow margin


def test_ecommerce_customer_ids_always_present(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.ecommerce import EcommerceGenerator

    gen = EcommerceGenerator(
        total_orders=200,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    orders = pd.read_parquet(tmp_path / "ecommerce_orders")
    assert orders["customer_id"].notna().all()
