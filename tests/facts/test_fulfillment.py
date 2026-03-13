from datetime import date
from pathlib import Path

import pandas as pd


def _generate_ecommerce(tmp_path):
    from retail_gen.dimensions.stores import Store
    from retail_gen.dimensions.products import Product
    from retail_gen.dimensions.customers import Customer
    from retail_gen.facts.ecommerce import EcommerceGenerator

    Store.generate(10, seed=42)
    Product.generate(50, seed=42)
    Customer.generate(100, store_instances=Store.instances, seed=42)
    gen = EcommerceGenerator(
        total_orders=300,
        customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()
    return pd.read_parquet(tmp_path / "ecommerce_orders")


def test_fulfillment_one_per_non_cancelled(tmp_path):
    orders_df = _generate_ecommerce(tmp_path)
    from retail_gen.facts.fulfillment import FulfillmentGenerator

    gen = FulfillmentGenerator(
        orders_df=orders_df,
        generation_date=date(2023, 12, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    ff = pd.read_parquet(tmp_path / "fulfillment")
    non_cancelled = orders_df[orders_df["status"] != "Cancelled"]
    assert len(ff) == len(non_cancelled)


def test_fulfillment_columns(tmp_path):
    orders_df = _generate_ecommerce(tmp_path)
    from retail_gen.facts.fulfillment import FulfillmentGenerator

    gen = FulfillmentGenerator(
        orders_df=orders_df,
        generation_date=date(2023, 12, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    ff = pd.read_parquet(tmp_path / "fulfillment")
    expected = {"id", "order_id", "ship_date", "estimated_delivery_date",
                "actual_delivery_date", "carrier", "tracking_number", "status", "is_on_time"}
    assert set(ff.columns) == expected


def test_fulfillment_on_time_rate(tmp_path):
    orders_df = _generate_ecommerce(tmp_path)
    from retail_gen.facts.fulfillment import FulfillmentGenerator

    gen = FulfillmentGenerator(
        orders_df=orders_df,
        generation_date=date(2023, 12, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    ff = pd.read_parquet(tmp_path / "fulfillment")
    delivered = ff[ff["actual_delivery_date"].notna()]
    if len(delivered) > 50:
        on_time_rate = delivered["is_on_time"].mean()
        assert 0.85 < on_time_rate < 0.97  # target ~92%
