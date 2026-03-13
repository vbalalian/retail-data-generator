from datetime import date


def test_customers_generate_correct_count():
    from retail_gen.dimensions.customers import Customer

    Customer.generate(100)
    assert len(Customer.instances) == 100


def test_customers_have_required_fields():
    from retail_gen.dimensions.customers import Customer

    Customer.generate(10)
    c = Customer.instances[0]
    for field in ["id", "first_name", "last_name", "email", "signup_date",
                  "gender", "age", "address", "city", "state", "zip",
                  "channel_preference", "loyalty_tier", "status",
                  "traffic_source", "preferred_store_id"]:
        assert hasattr(c, field), f"Missing field: {field}"


def test_customers_emails_unique():
    from retail_gen.dimensions.customers import Customer

    Customer.generate(500)
    emails = [c.email for c in Customer.instances]
    assert len(set(emails)) == len(emails)


def test_customers_age_distribution():
    from retail_gen.dimensions.customers import Customer
    import numpy as np

    Customer.generate(5000)
    ages = np.array([c.age for c in Customer.instances])
    assert ages.min() >= 18
    assert ages.max() <= 85
    assert 34 < ages.mean() < 42  # μ=38


def test_customers_loyalty_tier_distribution():
    from retail_gen.dimensions.customers import Customer
    from collections import Counter

    Customer.generate(5000)
    tiers = Counter(c.loyalty_tier for c in Customer.instances)
    total = sum(tiers.values())
    # None ~60%, Silver ~25%, Gold ~10%, Platinum ~5%
    assert 0.50 < tiers["None"] / total < 0.70
    assert tiers["Platinum"] / total < 0.10


def test_customers_preferred_store_nullable():
    """~30% should have no preferred store."""
    from retail_gen.dimensions.customers import Customer
    from retail_gen.dimensions.stores import Store

    Store.generate(10)
    Customer.generate(500, store_instances=Store.instances)
    null_count = sum(1 for c in Customer.instances if c.preferred_store_id is None)
    pct = null_count / len(Customer.instances)
    assert 0.20 < pct < 0.40  # ~30%


def test_customers_signup_date_skews_recent():
    """More signups in recent years than early years."""
    from retail_gen.dimensions.customers import Customer

    Customer.generate(5000)
    recent = sum(1 for c in Customer.instances if c.signup_date.year >= 2024)
    early = sum(1 for c in Customer.instances if c.signup_date.year == 2023)
    assert recent > early  # exponential decay toward recent
