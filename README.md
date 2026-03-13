# retail-data-generator

Synthetic retail data generator that produces multi-channel retail datasets (POS + ecommerce) as Parquet files, loadable into BigQuery.

10 entity tables across dimensions and facts:

**Dimensions** (fixed counts, Table metaclass):
- **Stores** (~150): population-weighted US locations, Flagship/Standard/Outlet types
- **Products** (~5,000): 7 categories, 25 curated hero SKUs, two-level revenue weighting
- **Customers** (scaled): loyalty tiers, channel preferences, churn modeling

**Facts** (scaled, numpy-vectorized batch generation):
- **POS Transactions + Items**: in-store sales with seasonal/day-of-week/intraday patterns
- **Ecommerce Orders + Items**: two-pass generation (free shipping over $75 backfill)
- **Fulfillment**: 1:1 with non-cancelled orders, tuned for ~92% on-time delivery
- **Returns**: category-based return rates (Apparel 12%, Food 0%, etc.)
- **Inventory Snapshots**: monthly/quarterly by product x store

## Quick start

```bash
git clone https://github.com/vbalalian/retail-data-generator.git
cd retail-data-generator
uv sync
```

Generate at dev scale (takes ~30 seconds):

```bash
SCALE=0.01 uv run retail-gen
```

Output lands in `output/<entity>/chunk_NNN.parquet`.

## Configuration

All config via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALE` | `0.01` | Volume multiplier. 0.01 = dev (~400K POS txns), 1.0 = production (~40M) |
| `SEED` | `42` | Random seed for reproducibility |
| `HISTORY_START` | `2023-01-01` | Earliest transaction date |
| `BQ_PROJECT` | `mkt-analytics-project` | BigQuery project ID |
| `BQ_DATASET` | `retail_raw` | BigQuery dataset name |
| `GCS_BUCKET` | (none) | GCS bucket for upload |

## Usage

```bash
# Generate all entities at dev scale
SCALE=0.01 uv run retail-gen

# Generate specific entities (auto-resolves dependencies)
uv run retail-gen --only pos_transactions,returns

# Generate and upload to GCS + BigQuery
uv run retail-gen --upload

# Upload existing output without regenerating
uv run retail-gen --upload-only
```

## Scale targets

| Entity | SCALE=0.01 | SCALE=1.0 |
|--------|-----------|-----------|
| Stores | 150 | 150 |
| Products | 5,000 | 5,000 |
| Customers | 20,000 | 2,000,000 |
| POS Transactions | 400,000 | 40,000,000 |
| POS Items | ~1,200,000 | ~120,000,000 |
| Ecommerce Orders | 250,000 | 25,000,000 |
| Ecommerce Items | ~625,000 | ~62,500,000 |

Dimensions stay fixed regardless of scale. Facts scale linearly.

## Tests

```bash
uv run pytest tests/ -v
```

66 tests covering all entities, distributions, and infrastructure.

## Stack

Python 3.11+, numpy, pandas, pyarrow, faker, uv

## License

MIT
