from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _setup_dims():
    from retail_gen.dimensions.stores import Store
    from retail_gen.dimensions.products import Product

    Store.generate(5, seed=42)
    Product.generate(20, seed=42)
    return Store, Product


def test_inventory_generates_snapshots(tmp_path):
    Store, Product = _setup_dims()
    from retail_gen.facts.inventory_snapshots import InventorySnapshotGenerator

    gen = InventorySnapshotGenerator(
        store_instances=Store.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 7, 1),
        frequency="quarterly",
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "inventory_snapshots")
    assert len(df) > 0


def test_inventory_columns(tmp_path):
    Store, Product = _setup_dims()
    from retail_gen.facts.inventory_snapshots import InventorySnapshotGenerator

    gen = InventorySnapshotGenerator(
        store_instances=Store.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 7, 1),
        frequency="quarterly",
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "inventory_snapshots")
    expected = {"id", "store_id", "product_id", "snapshot_date",
                "quantity_on_hand", "reorder_point", "quantity_on_order"}
    assert set(df.columns) == expected


def test_inventory_reorder_triggers_on_order(tmp_path):
    Store, Product = _setup_dims()
    from retail_gen.facts.inventory_snapshots import InventorySnapshotGenerator

    gen = InventorySnapshotGenerator(
        store_instances=Store.instances,
        product_instances=Product.instances,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        frequency="quarterly",
        seed=42,
        output_dir=tmp_path,
    )
    gen.generate()

    df = pd.read_parquet(tmp_path / "inventory_snapshots")
    # When on_hand < reorder_point, quantity_on_order should be > 0
    below_reorder = df[df["quantity_on_hand"] < df["reorder_point"]]
    if len(below_reorder) > 0:
        assert (below_reorder["quantity_on_order"] > 0).all()
