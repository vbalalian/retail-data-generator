import numpy as np


def test_address_provider_returns_correct_fields():
    from retail_gen.lib.providers.address import AddressProvider

    provider = AddressProvider(seed=42)
    result = provider.sample(5)
    assert len(result["zip"]) == 5
    assert all(k in result for k in ["zip", "city", "state", "latitude", "longitude"])


def test_address_provider_population_weighted():
    """Higher-population zips should appear more often in large samples."""
    from retail_gen.lib.providers.address import AddressProvider

    provider = AddressProvider(seed=42)
    result = provider.sample(10_000)
    # New York City zips (10001, 10002, etc.) should appear more than rural zips
    ny_count = sum(1 for s in result["state"] if s == "NY")
    wy_count = sum(1 for s in result["state"] if s == "WY")
    assert ny_count > wy_count  # NY has ~34x Wyoming's population


def test_address_provider_reproducible_with_seed():
    from retail_gen.lib.providers.address import AddressProvider

    p1 = AddressProvider(seed=99)
    p2 = AddressProvider(seed=99)
    r1 = p1.sample(100)
    r2 = p2.sample(100)
    assert r1["zip"] == r2["zip"]


def test_address_provider_top_metros():
    """top_n_metros limits to highest-population cities."""
    from retail_gen.lib.providers.address import AddressProvider

    provider = AddressProvider(seed=42)
    result = provider.sample(100, top_n_metros=20)
    # All results should be from top 20 (city, state) pairs
    unique_metros = set(zip(result["city"], result["state"]))
    assert len(unique_metros) <= 20
