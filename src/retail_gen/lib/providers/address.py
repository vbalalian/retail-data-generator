"""Population-weighted address generation from zips.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np


STATE_TO_REGION: dict[str, str] = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast", "RI": "Northeast",
    "VT": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    # South
    "AL": "South", "AR": "South", "DE": "South", "FL": "South", "GA": "South",
    "KY": "South", "LA": "South", "MD": "South", "MS": "South", "NC": "South",
    "OK": "South", "SC": "South", "TN": "South", "TX": "South", "VA": "South",
    "WV": "South", "DC": "South",
    # West
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West", "HI": "West",
    "ID": "West", "MT": "West", "NV": "West", "NM": "West", "OR": "West",
    "UT": "West", "WA": "West", "WY": "West",
}


@dataclass
class ZipRecord:
    zip: str
    city: str
    state: str
    latitude: float
    longitude: float
    population: int


class AddressProvider:
    """Samples addresses weighted by population from zips.csv."""

    def __init__(self, seed: int = 42, csv_path: Path | str | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._records = self._load(csv_path)
        populations = np.array([r.population for r in self._records], dtype=np.float64)
        self._weights = populations / populations.sum()

        # Precompute metro-level aggregation for top_n_metros filtering
        city_pop: dict[str, int] = {}
        for r in self._records:
            key = f"{r.city}, {r.state}"
            city_pop[key] = city_pop.get(key, 0) + r.population
        self._city_pop = city_pop

    def _load(self, csv_path: Path | str | None) -> list[ZipRecord]:
        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "data" / "zips.csv"
        csv_path = Path(csv_path)
        records = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pop = int(row["population"])
                if pop > 0:
                    records.append(
                        ZipRecord(
                            zip=row["zip"],
                            city=row["city"],
                            state=row["state"],
                            latitude=float(row["latitude"]),
                            longitude=float(row["longitude"]),
                            population=pop,
                        )
                    )
        return records

    def sample(
        self, n: int, top_n_metros: int | None = None
    ) -> dict[str, list]:
        """Sample n addresses. Returns dict with zip, city, state, latitude, longitude."""
        if top_n_metros is not None:
            return self._sample_top_metros(n, top_n_metros)

        indices = self._rng.choice(len(self._records), size=n, p=self._weights)
        return self._indices_to_dict(indices)

    def _sample_top_metros(self, n: int, top_n: int) -> dict[str, list]:
        sorted_cities = sorted(self._city_pop.items(), key=lambda x: -x[1])[:top_n]
        top_city_names = {name for name, _ in sorted_cities}
        mask = np.array(
            [f"{r.city}, {r.state}" in top_city_names for r in self._records]
        )
        filtered_weights = self._weights * mask
        filtered_weights /= filtered_weights.sum()
        indices = self._rng.choice(len(self._records), size=n, p=filtered_weights)
        return self._indices_to_dict(indices)

    def _indices_to_dict(self, indices: np.ndarray) -> dict[str, list]:
        records = [self._records[i] for i in indices]
        return {
            "zip": [r.zip for r in records],
            "city": [r.city for r in records],
            "state": [r.state for r in records],
            "latitude": [r.latitude for r in records],
            "longitude": [r.longitude for r in records],
        }

    @staticmethod
    def region_for_state(state: str) -> str:
        return STATE_TO_REGION.get(state, "Unknown")
