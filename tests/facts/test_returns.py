from datetime import date
from pathlib import Path

import pandas as pd


def _generate_upstream(tmp_path):
    from retail_gen.dimensions.stores import Store
    from retail_gen.dimensions.products import Product
    from retail_gen.dimensions.customers import Customer
    from retail_gen.facts.pos_transactions import PosTransactions
    from retail_gen.facts.pos_transaction_items import PosTransactionItems
    from retail_gen.facts.ecommerce import EcommerceGenerator

    Store.generate(10, seed=42)
    Product.generate(50, seed=42)
    Customer.generate(100, store_instances=Store.instances, seed=42)

    pos_gen = PosTransactions(
        total_rows=200, store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1), end_date=date(2023, 12, 1),
        seed=42, output_dir=tmp_path,
    )
    pos_gen.generate()
    pos_txn = pd.read_parquet(tmp_path / "pos_transactions")
    pos_items_gen = PosTransactionItems(
        transaction_df=pos_txn, product_instances=Product.instances,
        customer_instances=Customer.instances, seed=42, output_dir=tmp_path,
    )
    pos_items_gen.generate()

    ecom_gen = EcommerceGenerator(
        total_orders=200, customer_instances=Customer.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1), end_date=date(2023, 12, 1),
        seed=42, output_dir=tmp_path,
    )
    ecom_gen.generate()

    return {
        "pos_items": pd.read_parquet(tmp_path / "pos_transaction_items"),
        "pos_txns": pos_txn,
        "ecom_items": pd.read_parquet(tmp_path / "ecommerce_order_items"),
        "ecom_orders": pd.read_parquet(tmp_path / "ecommerce_orders"),
        "products": Product.instances,
    }


def test_returns_generate(tmp_path):
    data = _generate_upstream(tmp_path)
    from retail_gen.facts.returns import ReturnsGenerator

    gen = ReturnsGenerator(
        pos_items_df=data["pos_items"],
        pos_txns_df=data["pos_txns"],
        ecom_items_df=data["ecom_items"],
        ecom_orders_df=data["ecom_orders"],
        product_instances=data["products"],
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    returns = pd.read_parquet(tmp_path / "returns")
    assert len(returns) > 0


def test_returns_columns(tmp_path):
    data = _generate_upstream(tmp_path)
    from retail_gen.facts.returns import ReturnsGenerator

    gen = ReturnsGenerator(
        pos_items_df=data["pos_items"],
        pos_txns_df=data["pos_txns"],
        ecom_items_df=data["ecom_items"],
        ecom_orders_df=data["ecom_orders"],
        product_instances=data["products"],
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    returns = pd.read_parquet(tmp_path / "returns")
    expected = {"id", "original_transaction_type", "original_transaction_id",
                "product_id", "customer_id", "return_date", "reason",
                "refund_amount", "return_channel", "condition"}
    assert set(returns.columns) == expected


def test_returns_no_food_returns(tmp_path):
    data = _generate_upstream(tmp_path)
    from retail_gen.facts.returns import ReturnsGenerator

    gen = ReturnsGenerator(
        pos_items_df=data["pos_items"],
        pos_txns_df=data["pos_txns"],
        ecom_items_df=data["ecom_items"],
        ecom_orders_df=data["ecom_orders"],
        product_instances=data["products"],
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    returns = pd.read_parquet(tmp_path / "returns")
    food_products = {p.id for p in data["products"] if p.category == "Food & Grocery"}
    returned_products = set(returns["product_id"].unique())
    assert len(returned_products & food_products) == 0
