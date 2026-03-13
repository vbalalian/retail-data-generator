"""Temporal distribution engine for generating realistic timestamps.

Applies yearly growth, monthly seasonality, day-of-week weighting,
and bimodal intraday distribution. Fully vectorized with numpy.
"""

from __future__ import annotations

from datetime import date

import numpy as np

# Monthly seasonality multipliers (index 0=Jan .. 11=Dec)
MONTHLY_SEASONALITY = np.array([
    0.70,  # Jan
    0.75,  # Feb
    0.85,  # Mar
    0.90,  # Apr
    0.95,  # May
    1.00,  # Jun
    1.05,  # Jul
    1.00,  # Aug
    0.95,  # Sep
    1.00,  # Oct
    1.40,  # Nov
    1.60,  # Dec
])

WEEKEND_MULTIPLIER = 1.4


def generate_timestamps(
    n: int,
    start_date: date,
    end_date: date,
    yoy_growth: float,
    seed: int,
) -> np.ndarray:
    """Generate n timestamps within [start_date, end_date] with realistic distribution.

    All computation is vectorized — no Python loops.

    Args:
        n: Total number of timestamps to generate.
        start_date: First possible date (inclusive).
        end_date: Last possible date (inclusive).
        yoy_growth: Year-over-year compound growth rate (e.g. 0.08 for 8%).
        seed: Random seed for reproducibility.

    Returns:
        Sorted np.ndarray of datetime64[s].
    """
    rng = np.random.default_rng(seed)

    # Build array of all days in range as datetime64[D]
    start_np = np.datetime64(start_date, "D")
    end_np = np.datetime64(end_date, "D")
    all_days = np.arange(start_np, end_np + np.timedelta64(1, "D"))
    num_days = len(all_days)

    # Extract date components vectorized
    # Year offset from start for growth calculation
    years = all_days.astype("datetime64[Y]").astype(int) + 1970
    day_of_year = (all_days - all_days.astype("datetime64[Y]")).astype(int) + 1
    base_year = start_date.year
    year_offsets = (years - base_year) + day_of_year / 365.0
    growth_factors = (1.0 + yoy_growth) ** year_offsets

    # Monthly seasonality
    months = all_days.astype("datetime64[M]").astype(int) % 12  # 0-indexed
    season_factors = MONTHLY_SEASONALITY[months]

    # Day-of-week: numpy weekday 0=Mon .. 6=Sun
    # datetime64 epoch (1970-01-01) was a Thursday (weekday=3)
    dow = (all_days.astype(int) + 3) % 7  # 0=Mon..6=Sun
    dow_factors = np.where(dow >= 5, WEEKEND_MULTIPLIER, 1.0)

    # Combine weights
    weights = growth_factors * season_factors * dow_factors
    weights /= weights.sum()

    # Sample day indices
    day_indices = rng.choice(num_days, size=n, p=weights)
    sampled_days = all_days[day_indices]

    # Bimodal intraday times: 50% peak at 12pm, 50% at 6pm, σ=2h each
    mode = rng.choice(2, size=n)
    hours_float = np.where(
        mode == 0,
        rng.normal(12.0, 2.0, size=n),
        rng.normal(18.0, 2.0, size=n),
    )
    hours_float = np.clip(hours_float, 6.0, 22.999)  # 6am - 10:59pm

    # Convert fractional hours to seconds offset
    total_seconds = (hours_float * 3600).astype("int64")
    time_offsets = total_seconds.astype("timedelta64[s]")

    # Combine date + time
    timestamps = sampled_days.astype("datetime64[s]") + time_offsets
    timestamps.sort()
    return timestamps
