import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def test_batch_generator_creates_chunked_parquet(tmp_path):
    from retail_gen.lib.batch import BatchGenerator

    class TestBatch(BatchGenerator):
        name = "test_entity"
        schema = pa.schema([
            ("id", pa.int64()),
            ("value", pa.float64()),
        ])

        def generate_chunk(self, chunk_idx: int, start_id: int, n_rows: int) -> pd.DataFrame:
            return pd.DataFrame({
                "id": range(start_id, start_id + n_rows),
                "value": [1.0] * n_rows,
            })

    gen = TestBatch(total_rows=250, chunk_size=100, output_dir=tmp_path)
    gen.generate()

    # Should create 3 chunk files: 100, 100, 50
    files = sorted((tmp_path / "test_entity").glob("*.parquet"))
    assert len(files) == 3

    total = sum(pq.read_table(f).num_rows for f in files)
    assert total == 250


def test_batch_generator_ids_are_sequential(tmp_path):
    from retail_gen.lib.batch import BatchGenerator

    class TestBatch(BatchGenerator):
        name = "test_entity"
        schema = pa.schema([("id", pa.int64())])

        def generate_chunk(self, chunk_idx: int, start_id: int, n_rows: int) -> pd.DataFrame:
            return pd.DataFrame({"id": range(start_id, start_id + n_rows)})

    gen = TestBatch(total_rows=250, chunk_size=100, output_dir=tmp_path)
    gen.generate()

    files = sorted((tmp_path / "test_entity").glob("*.parquet"))
    all_ids = []
    for f in files:
        all_ids.extend(pq.read_table(f).column("id").to_pylist())
    assert all_ids == list(range(1, 251))


def test_batch_generator_enforces_schema(tmp_path):
    from retail_gen.lib.batch import BatchGenerator

    class TestBatch(BatchGenerator):
        name = "test_entity"
        schema = pa.schema([
            ("id", pa.int64()),
            ("value", pa.float64()),
        ])

        def generate_chunk(self, chunk_idx: int, start_id: int, n_rows: int) -> pd.DataFrame:
            return pd.DataFrame({
                "id": range(start_id, start_id + n_rows),
                "value": [1.0] * n_rows,
            })

    gen = TestBatch(total_rows=10, chunk_size=10, output_dir=tmp_path)
    gen.generate()

    table = pq.read_table(tmp_path / "test_entity" / "chunk_000.parquet")
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("value").type == pa.float64()
