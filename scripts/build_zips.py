"""One-time script to build src/retail_gen/data/zips.csv from uszipcode.

Run: uv run --with uszipcode python scripts/build_zips.py
"""

import csv
from pathlib import Path
from uszipcode import SearchEngine

out_path = Path("src/retail_gen/data/zips.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

search = SearchEngine()
# Fetch all zip codes with population > 100
results = search.by_population(lower=100, upper=10_000_000, returns=40_000)
print(f"Found {len(results)} zip codes")

count = 0
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["zip", "city", "state", "latitude", "longitude", "population"])
    for r in results:
        if r.population and r.lat and r.lng and r.major_city and r.state:
            w.writerow([r.zipcode, r.major_city, r.state, r.lat, r.lng, r.population])
            count += 1

print(f"Wrote {count} rows to {out_path}")
