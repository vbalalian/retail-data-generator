"""Store dimension entity (~150 stores, fixed count)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import numpy as np
from faker import Faker

from retail_gen.lib.providers.address import AddressProvider, STATE_TO_REGION
from retail_gen.lib.table import Table

# Store type weights
STORE_TYPES = ["Standard", "Flagship", "Outlet"]
STORE_TYPE_WEIGHTS = [0.70, 0.15, 0.15]

# Square footage by type: (mean, std)
SQFT_PARAMS = {
    "Standard": (15_000, 4_000),
    "Flagship": (45_000, 8_000),
    "Outlet": (25_000, 5_000),
}

OPENING_START = date(2015, 1, 1)
OPENING_END = date(2024, 6, 30)


@dataclass
class Store(metaclass=Table):
    id: int
    name: str
    address: str
    city: str
    state: str
    zip: str
    latitude: float
    longitude: float
    store_type: str
    region: str
    square_footage: int
    opening_date: date

    # Class-level providers (not serialized to Parquet)
    _rng: ClassVar[np.random.Generator | None] = None
    _addr_provider: ClassVar[AddressProvider | None] = None
    _faker: ClassVar[Faker | None] = None

    @classmethod
    def _init_providers(cls, seed: int = 42) -> None:
        cls._rng = np.random.default_rng(seed)
        cls._addr_provider = AddressProvider(seed=seed)
        cls._faker = Faker()
        Faker.seed(seed)

    @classmethod
    def generate(cls, count: int, seed: int = 42) -> None:
        cls._init_providers(seed)
        cls.instances = []

        # Pre-determine store types
        types = cls._rng.choice(
            STORE_TYPES, size=count, p=STORE_TYPE_WEIGHTS
        ).tolist()

        # Flagship stores get top-metro addresses, others get general population-weighted
        flagship_indices = [i for i, t in enumerate(types) if t == "Flagship"]
        other_indices = [i for i, t in enumerate(types) if t != "Flagship"]

        # Generate addresses
        flagship_addrs = cls._addr_provider.sample(
            len(flagship_indices), top_n_metros=20
        ) if flagship_indices else {k: [] for k in ["zip", "city", "state", "latitude", "longitude"]}
        other_addrs = cls._addr_provider.sample(len(other_indices)) if other_indices else {k: [] for k in ["zip", "city", "state", "latitude", "longitude"]}

        # Merge addresses back in order
        all_addrs: dict[str, list] = {k: [None] * count for k in ["zip", "city", "state", "latitude", "longitude"]}
        for pos, idx in enumerate(flagship_indices):
            for k in all_addrs:
                all_addrs[k][idx] = flagship_addrs[k][pos]
        for pos, idx in enumerate(other_indices):
            for k in all_addrs:
                all_addrs[k][idx] = other_addrs[k][pos]

        # Generate opening dates (uniform)
        open_start_ord = OPENING_START.toordinal()
        open_end_ord = OPENING_END.toordinal()
        open_ords = cls._rng.integers(open_start_ord, open_end_ord + 1, size=count)

        for i in range(count):
            store_type = types[i]
            city = all_addrs["city"][i]
            state = all_addrs["state"][i]

            mu, sigma = SQFT_PARAMS[store_type]
            sqft = max(5_000, int(cls._rng.normal(mu, sigma)))

            cls.instances.append(
                Store(
                    id=i + 1,
                    name=f"{city} {store_type}",
                    address=cls._faker.street_address(),
                    city=city,
                    state=state,
                    zip=all_addrs["zip"][i],
                    latitude=all_addrs["latitude"][i],
                    longitude=all_addrs["longitude"][i],
                    store_type=store_type,
                    region=STATE_TO_REGION.get(state, "Unknown"),
                    square_footage=sqft,
                    opening_date=date.fromordinal(int(open_ords[i])),
                )
            )

    @classmethod
    def _make_one(cls, i: int) -> "Store":
        raise NotImplementedError("Use Store.generate() instead")
