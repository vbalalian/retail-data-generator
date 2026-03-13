"""Product dimension entity (~5,000 products, fixed count)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from faker import Faker

from retail_gen.lib.table import Table

# Category config: (revenue_weight, avg_retail, margin, count_share, subcategories)
CATEGORY_CONFIG = {
    "Apparel": {
        "revenue_weight": 0.28, "avg_retail": 35.0, "margin": 0.55,
        "count_share": 0.30,  # 1500/5000
        "subcategories": ["Tops", "Bottoms", "Outerwear", "Activewear", "Accessories"],
        "cat_code": "APP",
        "weight_oz_mean": 8.0, "weight_oz_std": 3.0,
    },
    "Home & Kitchen": {
        "revenue_weight": 0.21, "avg_retail": 55.0, "margin": 0.45,
        "count_share": 0.20,
        "subcategories": ["Cookware", "Bedding", "Storage", "Decor", "Appliances"],
        "cat_code": "HMK",
        "weight_oz_mean": 32.0, "weight_oz_std": 20.0,
    },
    "Electronics": {
        "revenue_weight": 0.18, "avg_retail": 120.0, "margin": 0.30,
        "count_share": 0.12,
        "subcategories": ["Audio", "Accessories", "Wearables", "Chargers", "Smart Home"],
        "cat_code": "ELC",
        "weight_oz_mean": 12.0, "weight_oz_std": 8.0,
    },
    "Beauty & Personal Care": {
        "revenue_weight": 0.12, "avg_retail": 28.0, "margin": 0.60,
        "count_share": 0.14,
        "subcategories": ["Skincare", "Haircare", "Makeup", "Fragrance", "Bath & Body"],
        "cat_code": "BPC",
        "weight_oz_mean": 6.0, "weight_oz_std": 3.0,
    },
    "Sports & Outdoors": {
        "revenue_weight": 0.10, "avg_retail": 65.0, "margin": 0.40,
        "count_share": 0.10,
        "subcategories": ["Fitness", "Outdoor Gear", "Team Sports", "Water Sports", "Camping"],
        "cat_code": "SPO",
        "weight_oz_mean": 24.0, "weight_oz_std": 16.0,
    },
    "Food & Grocery": {
        "revenue_weight": 0.06, "avg_retail": 12.0, "margin": 0.25,
        "count_share": 0.08,
        "subcategories": ["Snacks", "Beverages", "Pantry", "Organic", "Specialty"],
        "cat_code": "FDG",
        "weight_oz_mean": 16.0, "weight_oz_std": 10.0,
    },
    "Toys & Games": {
        "revenue_weight": 0.05, "avg_retail": 30.0, "margin": 0.50,
        "count_share": 0.06,
        "subcategories": ["Board Games", "Action Figures", "Puzzles", "Outdoor Play", "Educational"],
        "cat_code": "TOY",
        "weight_oz_mean": 14.0, "weight_oz_std": 10.0,
    },
}

SUBCAT_CODES = {
    "Tops": "TOP", "Bottoms": "BTM", "Outerwear": "OUT", "Activewear": "ACT", "Accessories": "ACC",
    "Cookware": "CKW", "Bedding": "BED", "Storage": "STR", "Decor": "DCR", "Appliances": "APL",
    "Audio": "AUD", "Wearables": "WRB", "Chargers": "CHR", "Smart Home": "SMH",
    "Skincare": "SKN", "Haircare": "HRC", "Makeup": "MKP", "Fragrance": "FRG", "Bath & Body": "BTH",
    "Fitness": "FIT", "Outdoor Gear": "OGR", "Team Sports": "TMS", "Water Sports": "WTR", "Camping": "CMP",
    "Snacks": "SNK", "Beverages": "BVR", "Pantry": "PNT", "Organic": "ORG", "Specialty": "SPC",
    "Board Games": "BRD", "Action Figures": "ACF", "Puzzles": "PZL", "Outdoor Play": "OPL", "Educational": "EDU",
}

# Curated hero SKUs — must sound like real retail products
HERO_SKUS = [
    ("Apparel", "Tops", "Everyday Crew Tee"),
    ("Apparel", "Bottoms", "Classic Slim Chinos"),
    ("Apparel", "Outerwear", "Lightweight Zip Hoodie"),
    ("Apparel", "Activewear", "Performance Running Short"),
    ("Home & Kitchen", "Cookware", "Nonstick Fry Pan Set"),
    ("Home & Kitchen", "Appliances", "Stainless Steel Knife Block"),
    ("Home & Kitchen", "Decor", "Ceramic Pour-Over Dripper"),
    ("Home & Kitchen", "Bedding", "Cooling Gel Memory Pillow"),
    ("Electronics", "Audio", "Wireless Earbuds Pro"),
    ("Electronics", "Chargers", "USB-C Fast Charger"),
    ("Electronics", "Audio", "Portable Bluetooth Speaker"),
    ("Electronics", "Smart Home", "Smart LED Light Strip"),
    ("Beauty & Personal Care", "Skincare", "Daily Glow Moisturizer"),
    ("Beauty & Personal Care", "Haircare", "Volumizing Hair Serum"),
    ("Beauty & Personal Care", "Fragrance", "Coastal Breeze Eau de Toilette"),
    ("Beauty & Personal Care", "Bath & Body", "Exfoliating Body Scrub"),
    ("Sports & Outdoors", "Fitness", "Trail Running Shoes"),
    ("Sports & Outdoors", "Fitness", "Resistance Band Set"),
    ("Sports & Outdoors", "Camping", "Ultralight Camping Hammock"),
    ("Food & Grocery", "Snacks", "Dark Chocolate Almond Clusters"),
    ("Food & Grocery", "Beverages", "Cold Brew Coffee Concentrate"),
    ("Toys & Games", "Board Games", "Strategy Quest Board Game"),
    ("Toys & Games", "Puzzles", "World Map 1000-Piece Puzzle"),
    ("Apparel", "Accessories", "Merino Wool Beanie"),
    ("Sports & Outdoors", "Outdoor Gear", "Insulated Water Bottle 32oz"),
]

LAUNCH_START = date(2018, 1, 1)
LAUNCH_END = date(2025, 12, 31)


@dataclass
class Product(metaclass=Table):
    id: int
    name: str
    sku: str
    category: str
    subcategory: str
    brand: str
    cost: float
    retail_price: float
    weight_oz: float
    is_hero_sku: bool
    status: str
    launch_date: date

    @classmethod
    def generate(cls, count: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        fake = Faker()
        Faker.seed(seed)
        cls.instances = []

        # Assign categories based on count_share
        categories = list(CATEGORY_CONFIG.keys())
        shares = [CATEGORY_CONFIG[c]["count_share"] for c in categories]
        cat_assignments = rng.choice(categories, size=count, p=shares).tolist()

        # Place hero SKUs first
        hero_set = set()
        for i, (cat, subcat, name) in enumerate(HERO_SKUS):
            if i >= count:
                break
            cat_assignments[i] = cat  # Override category for hero slots
            hero_set.add(i)

        launch_start_ord = LAUNCH_START.toordinal()
        launch_end_ord = LAUNCH_END.toordinal()

        for i in range(count):
            cat = cat_assignments[i]
            cfg = CATEGORY_CONFIG[cat]
            is_hero = i in hero_set

            if is_hero:
                _, subcat, name = HERO_SKUS[i]
            else:
                subcat = rng.choice(cfg["subcategories"])
                name = f"{fake.word().title()} {subcat} {fake.word().title()}"

            # SKU: CAT-SUBCAT-00001
            subcat_code = SUBCAT_CODES.get(subcat, subcat[:3].upper())
            sku = f"{cfg['cat_code']}-{subcat_code}-{i + 1:05d}"

            # Pricing
            avg_retail = cfg["avg_retail"]
            margin = cfg["margin"]
            retail_price = max(1.0, round(rng.normal(avg_retail, avg_retail * 0.3), 2))
            cost = round(retail_price * (1 - margin), 2)

            # Weight
            weight = max(0.5, round(rng.normal(cfg["weight_oz_mean"], cfg["weight_oz_std"]), 1))

            # Status
            status = "Active" if rng.random() < 0.90 else "Discontinued"

            # Launch date
            launch_ord = rng.integers(launch_start_ord, launch_end_ord + 1)

            cls.instances.append(
                Product(
                    id=i + 1,
                    name=name,
                    sku=sku,
                    category=cat,
                    subcategory=subcat,
                    brand=fake.company(),
                    cost=cost,
                    retail_price=retail_price,
                    weight_oz=weight,
                    is_hero_sku=is_hero,
                    status=status,
                    launch_date=date.fromordinal(int(launch_ord)),
                )
            )

    @classmethod
    def _make_one(cls, i: int) -> "Product":
        raise NotImplementedError("Use Product.generate() instead")
