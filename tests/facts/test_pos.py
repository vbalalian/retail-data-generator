from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _setup_dimensions():
    """Generate tiny dimensions for testing."""
    from retail_gen.dimensions.stores import Store
    from retail_gen.dimensions.products import Product
    from retail_gen.dimensions.customers import Customer

    Store.generate(10, seed=42)
    Product.generate(50, seed=42)
    Customer.generate(100, store_instances=Store.instances, seed=42)
    return Store, Product, Customer


def test_pos_transactions_generate(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions

    gen = PosTransactions(
        total_rows=500,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    files = sorted((tmp_path / "pos_transactions").glob("*.parquet"))
    assert len(files) >= 1
    total = sum(pq.read_table(f).num_rows for f in files)
    assert total == 500


def test_pos_transactions_columns(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions

    gen = PosTransactions(
        total_rows=100,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "pos_transactions")
    assert set(df.columns) == {"id", "store_id", "customer_id", "transaction_date", "payment_method", "employee_id"}


def test_pos_transactions_store_ids_valid(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions

    gen = PosTransactions(
        total_rows=200,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "pos_transactions")
    valid_ids = {s.id for s in Store.instances}
    assert set(df["store_id"].unique()).issubset(valid_ids)


def test_pos_transactions_customer_nullable(tmp_path):
    """~30% of transactions should have null customer_id."""
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions

    gen = PosTransactions(
        total_rows=1000,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "pos_transactions")
    null_pct = df["customer_id"].isna().mean()
    assert 0.20 < null_pct < 0.40  # ~30%


def test_pos_items_generate(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions
    from retail_gen.facts.pos_transaction_items import PosTransactionItems

    # Generate parent transactions first
    txn_gen = PosTransactions(
        total_rows=100,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    txn_gen.generate()
    txn_df = pd.read_parquet(tmp_path / "pos_transactions")

    item_gen = PosTransactionItems(
        transaction_df=txn_df,
        product_instances=Product.instances,
        customer_instances=Customer.instances,
        seed=42,
        output_dir=tmp_path,
    )
    item_gen.generate()

    item_df = pd.read_parquet(tmp_path / "pos_transaction_items")
    # Every item must reference a valid transaction
    assert set(item_df["transaction_id"].unique()).issubset(set(txn_df["id"]))


def test_pos_items_columns(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions
    from retail_gen.facts.pos_transaction_items import PosTransactionItems

    txn_gen = PosTransactions(
        total_rows=50,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    txn_gen.generate()
    txn_df = pd.read_parquet(tmp_path / "pos_transactions")

    item_gen = PosTransactionItems(
        transaction_df=txn_df,
        product_instances=Product.instances,
        customer_instances=Customer.instances,
        seed=42,
        output_dir=tmp_path,
    )
    item_gen.generate()

    item_df = pd.read_parquet(tmp_path / "pos_transaction_items")
    expected_cols = {"id", "transaction_id", "product_id", "quantity", "unit_price", "discount_amount"}
    assert set(item_df.columns) == expected_cols


def test_pos_items_quantity_positive(tmp_path):
    Store, Product, Customer = _setup_dimensions()
    from retail_gen.facts.pos_transactions import PosTransactions
    from retail_gen.facts.pos_transaction_items import PosTransactionItems

    txn_gen = PosTransactions(
        total_rows=100,
        store_instances=Store.instances,
        customer_instances=Customer.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        seed=42,
        output_dir=tmp_path,
    )
    txn_gen.generate()
    txn_df = pd.read_parquet(tmp_path / "pos_transactions")

    item_gen = PosTransactionItems(
        transaction_df=txn_df,
        product_instances=Product.instances,
        customer_instances=Customer.instances,
        seed=42,
        output_dir=tmp_path,
    )
    item_gen.generate()

    item_df = pd.read_parquet(tmp_path / "pos_transaction_items")
    assert (item_df["quantity"] >= 1).all()
    assert (item_df["unit_price"] > 0).all()
    assert (item_df["discount_amount"] >= 0).all()
