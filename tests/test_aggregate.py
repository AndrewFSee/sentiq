"""Test 12 / 13 helpers (newsfeed.eval.aggregate)."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from newsfeed.eval.aggregate import (
    corr_test,
    month_index,
    monthly_tone,
    monthly_total_returns,
    prior_demeaned,
    tone_market_table,
    yoy_pairs,
)


def _rel(rows):
    return pl.DataFrame(
        [
            {"item_id": i, "cik": c, "t_use": datetime(*t, 14, tzinfo=UTC), "sentiment": s}
            for i, c, t, s in rows
        ],
        schema={"item_id": pl.String, "cik": pl.Int64, "t_use": pl.Datetime("us", "UTC"),
                "sentiment": pl.Float64},
    )  # fmt: skip


def test_yoy_pair_is_same_company_closest_to_a_year_earlier():
    rel = _rel(
        [
            ("a0", 1, (2023, 1, 20), 0.1),  # 400 days before a2: in range, not closest
            ("a1", 1, (2023, 2, 22), 0.3),  # 365 days before a2: chosen
            ("a2", 1, (2024, 2, 22), 0.6),
            ("b1", 2, (2023, 2, 22), -0.5),  # other company, same date: never paired with a2
            ("a3", 1, (2024, 5, 1), 0.2),  # 70 days after a2: no pair (outside 320-410)
        ]
    )
    p = yoy_pairs(rel)
    assert p["item_id"].to_list() == ["a2"]
    assert p["prev_sentiment"][0] == pytest.approx(0.3)
    assert p["diff"][0] == pytest.approx(0.3)


def test_prior_demeaned_uses_only_earlier_releases():
    rel = _rel([("x1", 1, (2024, 1, 5), 0.2), ("x2", 1, (2024, 4, 5), 0.6),
                ("x3", 1, (2024, 7, 5), 0.1)])  # fmt: skip
    d = prior_demeaned(rel).sort("t_use")
    assert d["demeaned"][0] is None
    assert d["demeaned"][1] == pytest.approx(0.4)
    assert d["demeaned"][2] == pytest.approx(0.1 - 0.4)


def test_monthly_tone_drops_months_with_few_pairs():
    rows = [(f"p{i}", i, (2023, 3, 1), 0.0) for i in range(12)]
    rows += [(f"c{i}", i, (2024, 3, 1), 0.5) for i in range(12)]  # 12 pairs in 2024-03
    rows += [(f"d{i}", i, (2024, 4, 1), 0.5) for i in range(3)]  # 0 pairs in 2024-04
    t = monthly_tone(_rel(rows), min_pairs=10).sort("month")
    by = {r["month"]: r for r in t.iter_rows(named=True)}
    assert by[date(2024, 3, 1)]["yoy"] == pytest.approx(0.5)
    assert by[date(2024, 3, 1)]["n_pairs"] == 12
    assert by[date(2024, 4, 1)]["yoy"] is None


def test_monthly_total_returns_compound_with_dividends():
    bars = pl.DataFrame(
        {
            "date": [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29), date(2024, 3, 1)],
            "close": [100.0, 101.0, 99.0, 110.0],
            "div": [0.0, 0.0, 1.0, 0.0],
        }
    )
    r = monthly_total_returns(bars)
    assert r["month"].to_list() == [date(2024, 2, 1), date(2024, 3, 1)]  # January: no prior close
    assert r["ret"][0] == pytest.approx(101 / 100 * (99 + 1) / 101 - 1)
    assert r["ret"][1] == pytest.approx(110 / 99 - 1)


def test_targets_never_reach_into_the_holdout():
    months = [date(2026, m, 1) for m in range(1, 10)]
    tone = pl.DataFrame({"month": months, "yoy": np.linspace(0, 1, 9), "level": 0.0, "raw": 0.0})
    spy = pl.DataFrame({"month": months, "ret": [0.01] * 9})
    t = tone_market_table(tone, spy, date(2026, 7, 1))
    by = {r["month"]: r for r in t.iter_rows(named=True)}
    assert by[date(2026, 5, 1)]["spy_next"] is not None  # June return ends before July
    assert by[date(2026, 6, 1)]["spy_next"] is None  # July return is holdout
    assert by[date(2026, 3, 1)]["spy_next3"] is not None  # Apr..Jun
    assert by[date(2026, 4, 1)]["spy_next3"] is None
    assert by[date(2026, 6, 1)]["spy_same"] is not None
    assert by[date(2026, 7, 1)]["spy_same"] is None
    assert by[date(2026, 3, 1)]["yoy_3m"] == pytest.approx(np.mean(np.linspace(0, 1, 9)[:3]))


def test_corr_test_detects_a_strong_relation_and_not_noise():
    rng = np.random.default_rng(0)
    months = month_index([date(2015, 1, 1) + timedelta(days=31 * i) for i in range(120)])
    x = rng.normal(size=120)
    strong = corr_test(x, x + 0.3 * rng.normal(size=120), months, block=3, reps=500, rng=rng)
    noise = corr_test(x, rng.normal(size=120), months, block=3, reps=500, rng=rng)
    assert strong["corr"] > 0.8 and strong["p"] < 0.01
    assert noise["p"] > 0.01
