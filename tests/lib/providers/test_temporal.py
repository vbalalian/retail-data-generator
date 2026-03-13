from datetime import date

import numpy as np


def test_temporal_generates_correct_count():
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts = generate_timestamps(
        n=1000,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        yoy_growth=0.08,
        seed=42,
    )
    assert len(ts) == 1000
    assert ts.dtype == np.dtype("datetime64[s]")


def test_temporal_respects_date_range():
    from retail_gen.lib.providers.temporal import generate_timestamps

    start = date(2023, 6, 1)
    end = date(2023, 12, 31)
    ts = generate_timestamps(n=500, start_date=start, end_date=end, yoy_growth=0.0, seed=42)
    start_np = np.datetime64(start)
    end_np = np.datetime64(end) + np.timedelta64(1, "D")  # exclusive upper
    assert np.all(ts >= start_np)
    assert np.all(ts < end_np)


def test_temporal_seasonality_december_higher_than_january():
    """December should have ~2.3x January's volume (1.6/0.7)."""
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts = generate_timestamps(
        n=100_000,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        yoy_growth=0.0,
        seed=42,
    )
    # Extract months from datetime64 array
    months = ts.astype("datetime64[M]").astype(int) % 12 + 1
    jan_count = (months == 1).sum()
    dec_count = (months == 12).sum()
    ratio = dec_count / jan_count
    assert 1.8 < ratio < 2.8  # expect ~2.3x


def test_temporal_weekends_higher_than_weekdays():
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts = generate_timestamps(
        n=100_000,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        yoy_growth=0.0,
        seed=42,
    )
    # day-of-week: 0=Monday .. 6=Sunday for numpy
    dow = ((ts.astype("datetime64[D]").astype(int)) % 7)  # 0=Thu for epoch, adjust
    # Simpler: use pandas for DOW extraction
    import pandas as pd
    dow = pd.DatetimeIndex(ts).weekday  # 0=Mon..6=Sun
    weekday_avg = (dow < 5).sum() / 5
    weekend_avg = (dow >= 5).sum() / 2
    ratio = weekend_avg / weekday_avg
    assert 1.2 < ratio < 1.6  # expect ~1.4x


def test_temporal_reproducible():
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts1 = generate_timestamps(n=100, start_date=date(2023, 1, 1), end_date=date(2023, 12, 31), yoy_growth=0.0, seed=42)
    ts2 = generate_timestamps(n=100, start_date=date(2023, 1, 1), end_date=date(2023, 12, 31), yoy_growth=0.0, seed=42)
    assert np.array_equal(ts1, ts2)


def test_temporal_yoy_growth_later_years_heavier():
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts = generate_timestamps(
        n=100_000,
        start_date=date(2023, 1, 1),
        end_date=date(2025, 12, 31),
        yoy_growth=0.12,
        seed=42,
    )
    years = ts.astype("datetime64[Y]").astype(int) + 1970
    c2023 = (years == 2023).sum()
    c2025 = (years == 2025).sum()
    assert c2025 > c2023  # 2025 should have more due to 12% YoY growth


def test_temporal_is_sorted():
    from retail_gen.lib.providers.temporal import generate_timestamps

    ts = generate_timestamps(n=1000, start_date=date(2023, 1, 1), end_date=date(2023, 12, 31), yoy_growth=0.0, seed=42)
    assert np.all(ts[:-1] <= ts[1:])
