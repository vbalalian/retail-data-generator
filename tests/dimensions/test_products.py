def test_products_generate_correct_count():
    from retail_gen.dimensions.products import Product

    Product.generate(100)
    assert len(Product.instances) == 100


def test_products_have_required_fields():
    from retail_gen.dimensions.products import Product

    Product.generate(10)
    p = Product.instances[0]
    for field in ["id", "name", "sku", "category", "subcategory", "brand",
                  "cost", "retail_price", "weight_oz", "is_hero_sku", "status", "launch_date"]:
        assert hasattr(p, field), f"Missing field: {field}"


def test_products_hero_skus_exist():
    from retail_gen.dimensions.products import Product

    Product.generate(5000)
    heroes = [p for p in Product.instances if p.is_hero_sku]
    assert 20 <= len(heroes) <= 30
    # Hero names should be realistic, not faker gibberish
    for h in heroes:
        assert len(h.name.split()) >= 2  # at least two words


def test_products_category_distribution():
    from retail_gen.dimensions.products import Product, CATEGORY_CONFIG

    Product.generate(5000)
    cats = [p.category for p in Product.instances]
    # Apparel should be the most common
    from collections import Counter
    counts = Counter(cats)
    assert counts["Apparel"] > counts["Toys & Games"]


def test_products_sku_format():
    from retail_gen.dimensions.products import Product

    Product.generate(100)
    for p in Product.instances:
        parts = p.sku.split("-")
        assert len(parts) == 3, f"Bad SKU format: {p.sku}"
        assert len(parts[2]) == 5  # 5-digit suffix


def test_products_price_positive_and_margin_correct():
    from retail_gen.dimensions.products import Product

    Product.generate(100)
    for p in Product.instances:
        assert p.cost > 0
        assert p.retail_price > p.cost  # margin always positive


def test_products_status_distribution():
    from retail_gen.dimensions.products import Product

    Product.generate(5000)
    active = sum(1 for p in Product.instances if p.status == "Active")
    pct = active / len(Product.instances)
    assert 0.85 < pct < 0.95  # ~90%
