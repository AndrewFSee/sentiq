"""Test 14 helpers (newsfeed.eval.media): session tone and market targets."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from newsfeed.eval.media import (
    H4_EARLIEST_LOOK,
    check_h4_look,
    h4_test,
    market_daily,
    market_table,
    session_tone,
)


def _sessions(n: int, start: date = date(2025, 1, 6)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_session_tone_demeans_with_earlier_sessions_only():
    s = _sessions(6)
    docs = pl.DataFrame(
        {"date": [s[0]] * 2 + [s[1]] * 2 + [s[2]] * 2 + [s[3]] * 1 + [s[4]] * 2 + [s[5]] * 2,
         "v": [0.1, 0.3, 0.5, 0.5, 0.0, 0.2, 0.9, 1.0, 0.8, 9.0, 9.0]}
    )  # fmt: skip
    t = session_tone(docs, s, value="v", min_docs=2, demean=3, min_prior=2)
    tone = dict(zip(t["date"], t["tone"], strict=True))
    dm = dict(zip(t["date"], t["tone_dm"], strict=True))
    assert tone[s[0]] == pytest.approx(0.2)
    assert tone[s[3]] is None  # one document < min_docs
    assert dm[s[0]] is None and dm[s[1]] is None  # fewer than 2 prior sessions
    assert dm[s[2]] == pytest.approx(0.1 - (0.2 + 0.5) / 2)
    # s[4]: prior window s[1..3]; s[3] has no tone -> mean of s[1], s[2] only
    assert dm[s[4]] == pytest.approx(0.9 - (0.5 + 0.1) / 2)
    # the future (s[5] = 9.0) never enters an earlier session's value
    assert dm[s[4]] == pytest.approx(
        session_tone(docs.filter(pl.col("date") <= s[4]), s[:5], value="v", min_docs=2,
                     demean=3, min_prior=2)["tone_dm"][-1]
    )  # fmt: skip


def test_market_daily_total_return_and_gap():
    s = _sessions(3)
    bars = pl.DataFrame(
        {"date": s, "open": [99.0, 101.0, 103.0], "close": [100.0, 102.0, 101.0],
         "volume": [1, 2, 3], "div": [0.0, 0.0, 1.0]}
    )  # fmt: skip
    d = market_daily(bars)
    assert d["date"].to_list() == s[1:]
    assert d["r"].to_list() == pytest.approx([0.02, (101 + 1) / 102 - 1])
    assert d["gap"].to_list() == pytest.approx([0.01, 103 / 102 - 1])


def test_market_table_alignment_and_holdout_cut():
    s = _sessions(40)
    rng = np.random.default_rng(1)
    daily = pl.DataFrame(
        {"date": s, "r": rng.normal(0, 0.01, 40), "gap": rng.normal(0, 0.005, 40),
         "volume": rng.uniform(1, 2, 40)}
    )  # fmt: skip
    hold = s[30]
    t = market_table(daily, hold)
    r = daily["r"].to_numpy()
    i = 25
    row = t.row(i, named=True)
    assert row["r0"] == pytest.approx(r[i]) and row["r4"] == pytest.approx(r[i - 4])
    assert row["fwd1"] == pytest.approx(r[i + 1])
    j5 = t.row(24, named=True)  # exits s[29], the last session before the holdout
    assert j5["fwd5"] == pytest.approx(np.prod(1 + r[25:30]) - 1)
    assert j5["rev"] == pytest.approx(np.prod(1 + r[26:30]) - 1)
    v = daily["volume"].to_numpy()
    assert row["vol_next"] == pytest.approx(np.log(v[i + 1] / v[i - 19 : i + 1].mean()))
    assert row["vol_today"] == pytest.approx(np.log(v[i] / v[i - 20 : i].mean()))
    # a target whose last session is on/after the holdout start is null
    assert t.row(28, named=True)["fwd1"] == pytest.approx(r[29])  # exits s[29] < hold
    assert t.row(29, named=True)["fwd1"] is None  # exits s[30] = the holdout start
    for k, col in ((1, "fwd1"), (5, "fwd5"), (5, "rev")):
        for j in range(40):
            val = t.row(j, named=True)[col]
            if j + k >= 40 or s[j + k] >= hold:
                assert val is None, (col, j)


def test_h4_look_is_refused_before_the_pre_registered_date():
    with pytest.raises(ValueError):
        check_h4_look(H4_EARLIEST_LOOK - timedelta(days=1))
    check_h4_look(H4_EARLIEST_LOOK)


def _h4_world(n: int = 160, hold_at: int = 100, seed: int = 3):
    s = _sessions(n)
    rng = np.random.default_rng(seed)
    daily = pl.DataFrame(
        {"date": s, "r": rng.normal(0, 0.01, n), "gap": rng.normal(0, 0.005, n),
         "volume": rng.uniform(1, 2, n)}
    )  # fmt: skip
    docs = pl.DataFrame(
        {"date": [d for d in s for _ in range(12)], "v": rng.normal(0, 1, n * 12)},
        schema={"date": pl.Date, "v": pl.Float64},
    )
    return s, daily, docs, s[hold_at]


def _h4(docs, daily, hold, look):
    return h4_test(docs, daily, holdout=hold, look=look, reps=200,
                   rng=np.random.default_rng(0), min_docs=10, demean=20)  # fmt: skip


def test_h4_pre_holdout_mode_never_sees_holdout_returns_or_documents():
    s, daily, docs, hold = _h4_world()
    base = _h4(docs, daily, hold, None)
    assert base["last"] < hold and base["n"] > 30
    later = pl.col("date") >= hold
    daily2 = daily.with_columns(pl.when(later).then(pl.col("r") * 50).otherwise(pl.col("r")))
    docs2 = docs.with_columns(pl.when(later).then(pl.col("v") + 7).otherwise(pl.col("v")))
    assert _h4(docs2, daily2, hold, None)["partial_ic"] == pytest.approx(base["partial_ic"])
    # the last pre-holdout session's target (close[i] -> close[i+1]) would end in the holdout
    assert base["last"] == s[s.index(hold) - 2]


def test_h4_holdout_look_uses_holdout_sessions_with_targets_before_the_look():
    s, daily, docs, hold = _h4_world()
    look = s[140]
    res = _h4(docs, daily, hold, look)
    assert res["first"] == hold  # the demean reaches back into pre-holdout tone
    assert res["last"] == s[138]  # its target ends at s[139] < look
    after = pl.col("date") >= look
    daily2 = daily.with_columns(pl.when(after).then(pl.col("r") * 50).otherwise(pl.col("r")))
    docs2 = docs.with_columns(pl.when(after).then(pl.col("v") - 5).otherwise(pl.col("v")))
    again = _h4(docs2, daily2, hold, look)
    assert again["partial_ic"] == pytest.approx(res["partial_ic"]) and again["n"] == res["n"]
    # a planted signal passes
    fwd = daily.sort("date")["r"].to_numpy()
    sig = (
        pl.DataFrame({"date": s[:-1], "v": fwd[1:] * 100})
        .join(pl.DataFrame({"k": range(12)}), how="cross")
        .select("date", "v")
    )
    assert _h4(sig, daily, hold, look)["pass"]
