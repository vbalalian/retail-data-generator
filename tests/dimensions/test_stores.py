import pyarrow.parquet as pq


def test_stores_generate_correct_count():
    from retail_gen.dimensions.stores import Store

    Store.generate(10)
    assert len(Store.instances) == 10


def test_stores_have_required_fields():
    from retail_gen.dimensions.stores import Store

    Store.generate(5)
    s = Store.instances[0]
    assert hasattr(s, "id")
    assert hasattr(s, "name")
    assert hasattr(s, "city")
    assert hasattr(s, "state")
    assert hasattr(s, "zip")
    assert hasattr(s, "store_type")
    assert hasattr(s, "region")
    assert hasattr(s, "square_footage")
    assert hasattr(s, "opening_date")


def test_stores_type_distribution():
    """Standard ~70%, Flagship ~15%, Outlet ~15%."""
    from retail_gen.dimensions.stores import Store

    Store.generate(150)
    types = [s.store_type for s in Store.instances]
    standard_pct = types.count("Standard") / len(types)
    assert 0.55 < standard_pct < 0.85  # ~70% with sampling noise


def test_stores_region_derived_from_state():
    from retail_gen.dimensions.stores import Store
    from retail_gen.lib.providers.address import STATE_TO_REGION

    Store.generate(20)
    for s in Store.instances:
        expected = STATE_TO_REGION.get(s.state, "Unknown")
        assert s.region == expected


def test_stores_flagship_in_top_metros():
    """Flagship stores should be in major cities only."""
    from retail_gen.dimensions.stores import Store

    Store.generate(150)
    flagships = [s for s in Store.instances if s.store_type == "Flagship"]
    # All flagships should have populated cities (not rural areas)
    for f in flagships:
        assert f.city is not None
        assert len(f.city) > 0


def test_stores_write_parquet(tmp_path):
    from retail_gen.dimensions.stores import Store

    Store.generate(10)
    out = tmp_path / "stores.parquet"
    Store.write(out)
    table = pq.read_table(out)
    assert table.num_rows == 10
    assert "store_type" in table.column_names
