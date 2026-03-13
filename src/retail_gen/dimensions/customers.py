"""Customer dimension entity (~20K dev, ~2M production)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from faker import Faker

from retail_gen.lib.providers.address import AddressProvider
from retail_gen.lib.table import Table

GENDERS = ["M", "F", "Non-binary"]
GENDER_WEIGHTS = [0.48, 0.50, 0.02]

CHANNEL_PREFS = ["Online", "In-store", "Both"]
CHANNEL_WEIGHTS = [0.40, 0.35, 0.25]

LOYALTY_TIERS = ["None", "Silver", "Gold", "Platinum"]
LOYALTY_WEIGHTS = [0.60, 0.25, 0.10, 0.05]

# Frequency multipliers used by downstream fact generators
LOYALTY_FREQUENCY = {"None": 1.0, "Silver": 1.5, "Gold": 2.0, "Platinum": 3.0}
LOYALTY_DISCOUNT_PROB = {"None": 0.30, "Silver": 0.35, "Gold": 0.45, "Platinum": 0.55}

STATUSES = ["Active", "Inactive", "Churned"]
STATUS_WEIGHTS = [0.85, 0.10, 0.05]

TRAFFIC_SOURCES = ["Organic", "Search", "Social", "Email", "Referral", "Direct"]
TRAFFIC_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "hotmail.com"]

# Signup date range: exponential decay toward recent
SIGNUP_START = date(2020, 1, 1)


@dataclass
class Customer(metaclass=Table):
    id: int
    first_name: str
    last_name: str
    email: str
    signup_date: date
    gender: str
    age: int
    address: str
    city: str
    state: str
    zip: str
    channel_preference: str
    loyalty_tier: str
    status: str
    traffic_source: str
    preferred_store_id: int | None

    @classmethod
    def generate(
        cls, count: int, seed: int = 42, store_instances: list | None = None,
        generation_date: date | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        fake = Faker()
        Faker.seed(seed)
        addr = AddressProvider(seed=seed)
        cls.instances = []

        # Pre-generate all categorical fields vectorized
        genders = rng.choice(GENDERS, size=count, p=GENDER_WEIGHTS).tolist()
        ages = np.clip(rng.normal(38, 14, size=count).astype(int), 18, 85)
        channels = rng.choice(CHANNEL_PREFS, size=count, p=CHANNEL_WEIGHTS).tolist()
        tiers = rng.choice(LOYALTY_TIERS, size=count, p=LOYALTY_WEIGHTS).tolist()
        statuses = rng.choice(STATUSES, size=count, p=STATUS_WEIGHTS).tolist()
        sources = rng.choice(TRAFFIC_SOURCES, size=count, p=TRAFFIC_WEIGHTS).tolist()

        # Addresses
        addrs = addr.sample(count)

        # Signup dates: exponential decay toward recent (higher λ = more recent)
        signup_start_ord = SIGNUP_START.toordinal()
        generation_date_ord = (generation_date or date.today()).toordinal()
        date_range = generation_date_ord - signup_start_ord
        # Exponential samples concentrate near 0; flip so most signups are recent
        exp_samples = rng.exponential(scale=0.4, size=count)
        exp_samples = 1.0 - (exp_samples / exp_samples.max())  # flip: 1=recent, 0=oldest
        signup_ords = signup_start_ord + (exp_samples * date_range).astype(int)
        signup_ords = np.clip(signup_ords, signup_start_ord, generation_date_ord)

        # Preferred store: 70% have one, 30% null
        store_ids = None
        if store_instances:
            store_id_pool = [s.id for s in store_instances]
            has_store = rng.random(size=count) < 0.70
            store_choices = rng.choice(store_id_pool, size=count)
            store_ids = [int(store_choices[i]) if has_store[i] else None for i in range(count)]

        # Generate emails (unique)
        seen_emails: set[str] = set()
        for i in range(count):
            first = fake.first_name()
            last = fake.last_name()

            # Build unique email
            domain = rng.choice(EMAIL_DOMAINS)
            suffix = rng.integers(10, 9999)
            email = f"{first.lower()}.{last.lower()}{suffix}@{domain}"
            while email in seen_emails:
                suffix = rng.integers(10, 99999)
                email = f"{first.lower()}.{last.lower()}{suffix}@{domain}"
            seen_emails.add(email)

            cls.instances.append(
                Customer(
                    id=i + 1,
                    first_name=first,
                    last_name=last,
                    email=email,
                    signup_date=date.fromordinal(int(signup_ords[i])),
                    gender=genders[i],
                    age=int(ages[i]),
                    address=fake.street_address(),
                    city=addrs["city"][i],
                    state=addrs["state"][i],
                    zip=addrs["zip"][i],
                    channel_preference=channels[i],
                    loyalty_tier=tiers[i],
                    status=statuses[i],
                    traffic_source=sources[i],
                    preferred_store_id=store_ids[i] if store_ids else None,
                )
            )

    @classmethod
    def _make_one(cls, i: int) -> "Customer":
        raise NotImplementedError("Use Customer.generate() instead")
