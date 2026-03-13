"""CLI entry point for retail data generation.

Usage:
    uv run retail-gen                         # Generate all at SCALE
    uv run retail-gen --only stores,products  # Specific entities
    uv run retail-gen --upload                # Generate + upload
    uv run retail-gen --upload-only           # Upload existing output/
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

from retail_gen.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retail_gen")

ALL_ENTITIES = [
    "stores", "products", "customers",
    "pos_transactions", "pos_transaction_items",
    "ecommerce",
    "fulfillment", "returns", "inventory_snapshots",
]

# Dependency map: entity -> set of entities it requires
DEPENDENCIES: dict[str, set[str]] = {
    "stores": set(),
    "products": set(),
    "customers": {"stores"},
    "pos_transactions": {"stores", "customers"},
    "pos_transaction_items": {"products", "customers", "pos_transactions"},
    "ecommerce": {"products", "customers"},
    "fulfillment": {"ecommerce"},
    "returns": {"products", "pos_transactions", "pos_transaction_items", "ecommerce"},
    "inventory_snapshots": {"stores", "products"},
}


def _resolve_deps(requested: list[str]) -> list[str]:
    """Resolve transitive dependencies and return entities in generation order."""
    needed: set[str] = set()
    def _add(entity: str) -> None:
        if entity in needed:
            return
        for dep in DEPENDENCIES.get(entity, set()):
            _add(dep)
        needed.add(entity)
    for e in requested:
        _add(e)
    # Return in canonical generation order
    return [e for e in ALL_ENTITIES if e in needed]


def generate_all(cfg: Config, only: list[str] | None = None) -> None:
    """Run the generation pipeline in dependency order."""
    import numpy as np
    import pandas as pd

    if only:
        entities = _resolve_deps(only)
        added = set(entities) - set(only)
        if added:
            logger.info("Auto-added dependencies: %s", ", ".join(sorted(added)))
    else:
        entities = ALL_ENTITIES
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed

    # --- Dimensions ---
    if "stores" in entities:
        t0 = time.time()
        from retail_gen.dimensions.stores import Store
        Store.generate(cfg.n_stores, seed=seed)
        Store.write(output / "stores" / "chunk_000.parquet")
        logger.info("stores: %d rows (%.1fs)", len(Store.instances), time.time() - t0)

    if "products" in entities:
        t0 = time.time()
        from retail_gen.dimensions.products import Product
        Product.generate(cfg.n_products, seed=seed)
        Product.write(output / "products" / "chunk_000.parquet")
        logger.info("products: %d rows (%.1fs)", len(Product.instances), time.time() - t0)

    if "customers" in entities:
        t0 = time.time()
        from retail_gen.dimensions.stores import Store
        from retail_gen.dimensions.customers import Customer
        Customer.generate(cfg.n_customers, seed=seed, store_instances=Store.instances, generation_date=cfg.generation_date)
        Customer.write(output / "customers" / "chunk_000.parquet")
        logger.info("customers: %d rows (%.1fs)", len(Customer.instances), time.time() - t0)

    # --- POS Facts ---
    if "pos_transactions" in entities:
        t0 = time.time()
        from retail_gen.dimensions.stores import Store
        from retail_gen.dimensions.customers import Customer
        from retail_gen.facts.pos_transactions import PosTransactions

        gen = PosTransactions(
            total_rows=cfg.n_pos_transactions,
            store_instances=Store.instances,
            customer_instances=Customer.instances,
            start_date=cfg.history_start,
            end_date=cfg.generation_date,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("pos_transactions: %d rows (%.1fs)", cfg.n_pos_transactions, time.time() - t0)

    if "pos_transaction_items" in entities:
        t0 = time.time()
        from retail_gen.dimensions.products import Product
        from retail_gen.dimensions.customers import Customer
        from retail_gen.facts.pos_transaction_items import PosTransactionItems

        txn_df = pd.read_parquet(output / "pos_transactions")
        gen = PosTransactionItems(
            transaction_df=txn_df,
            product_instances=Product.instances,
            customer_instances=Customer.instances,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("pos_transaction_items: done (%.1fs)", time.time() - t0)

    # --- Ecommerce Facts ---
    if "ecommerce" in entities:
        t0 = time.time()
        from retail_gen.dimensions.customers import Customer
        from retail_gen.dimensions.products import Product
        from retail_gen.facts.ecommerce import EcommerceGenerator

        gen = EcommerceGenerator(
            total_orders=cfg.n_ecommerce_orders,
            customer_instances=Customer.instances,
            product_instances=Product.instances,
            start_date=cfg.history_start,
            end_date=cfg.generation_date,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("ecommerce: %d orders (%.1fs)", cfg.n_ecommerce_orders, time.time() - t0)

    # --- Downstream Facts ---
    if "fulfillment" in entities:
        t0 = time.time()
        from retail_gen.facts.fulfillment import FulfillmentGenerator

        orders_df = pd.read_parquet(output / "ecommerce_orders")
        gen = FulfillmentGenerator(
            orders_df=orders_df,
            generation_date=cfg.generation_date,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("fulfillment: done (%.1fs)", time.time() - t0)

    if "returns" in entities:
        t0 = time.time()
        from retail_gen.dimensions.products import Product
        from retail_gen.facts.returns import ReturnsGenerator

        gen = ReturnsGenerator(
            pos_items_df=pd.read_parquet(output / "pos_transaction_items"),
            pos_txns_df=pd.read_parquet(output / "pos_transactions"),
            ecom_items_df=pd.read_parquet(output / "ecommerce_order_items"),
            ecom_orders_df=pd.read_parquet(output / "ecommerce_orders"),
            product_instances=Product.instances,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("returns: done (%.1fs)", time.time() - t0)

    if "inventory_snapshots" in entities:
        t0 = time.time()
        from retail_gen.dimensions.stores import Store
        from retail_gen.dimensions.products import Product
        from retail_gen.facts.inventory_snapshots import InventorySnapshotGenerator

        gen = InventorySnapshotGenerator(
            store_instances=Store.instances,
            product_instances=Product.instances,
            start_date=cfg.history_start,
            end_date=cfg.generation_date,
            frequency=cfg.inventory_frequency,
            seed=seed,
            output_dir=output,
        )
        gen.generate()
        logger.info("inventory_snapshots: done (%.1fs)", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail data generator")
    parser.add_argument("--only", type=str, help="Comma-separated entity names")
    parser.add_argument("--upload", action="store_true", help="Upload to GCS + BQ after generation")
    parser.add_argument("--upload-only", action="store_true", help="Skip generation, upload existing output/")
    args = parser.parse_args()

    cfg = Config()
    logger.info("Config: SCALE=%.4f, SEED=%d, output=%s", cfg.scale, cfg.seed, cfg.output_dir)

    only = args.only.split(",") if args.only else None

    if not args.upload_only:
        generate_all(cfg, only=only)

    if args.upload or args.upload_only:
        from retail_gen.upload.gcs import upload_to_gcs
        from retail_gen.upload.bigquery import load_to_bigquery
        upload_to_gcs(cfg)
        load_to_bigquery(cfg)


if __name__ == "__main__":
    main()
