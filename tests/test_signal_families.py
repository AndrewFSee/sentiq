"""Tests 6-9 machinery: daily IC / quintile view, partial rank correlation, volatility targets,
event studies and the generalised event-level IC."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from newsfeed.eval.clusters import cluster_ic
from newsfeed.eval.events import abnormal_returns, event_mean_cells
from newsfeed.eval.report import (
    daily_cross_section,
    ic_detail,
    partial_rank_corr,
    run_eval,
    univariate_cells,
)
from newsfeed.eval.targets import volatility_targets
from tests.test_eval import panel, sessions, synthetic_inputs


def test_daily_cross_section_ic_spread_and_min_names():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(30), 20)
    x = rng.normal(size=day.size)
    y = x + 0.1 * rng.normal(size=day.size)
    keep = ~((day == 3) & (np.arange(day.size) % 20 >= 5))  # day 3 keeps only 5 names
    d, ic, sp = daily_cross_section(day[keep], x[keep], y[keep], min_names=10)
    assert 3 not in d and len(d) == 29
    assert (ic > 0.9).all() and (sp > 1).all()  # top quintile of x has the highest y
    # all-tied feature on a day: no IC (the day is skipped), never a crash
    x2 = x.copy()
    x2[day == 0] = 1.0
    d2, _, _ = daily_cross_section(day, x2, y, min_names=10)
    assert 0 not in d2


def test_ic_detail_reports_icir_and_quintile_spread():
    rng = np.random.default_rng(1)
    day = np.repeat(np.arange(200), 15)
    x = rng.normal(size=day.size)
    y = 0.3 * x + rng.normal(size=day.size)
    out = ic_detail(day, x, y, block=2, reps=200, rng=np.random.default_rng(0))
    assert out["days"] == 200
    assert out["ic_daily"] > 0.15 and out["icir"] > 0.5 and out["ic_daily_p"] < 0.01
    assert out["q_spread"] > 0.5 and out["q_spread_ci_lo"] > 0
    few = ic_detail(day[:150], x[:150], y[:150], block=2, reps=50, rng=np.random.default_rng(0))
    assert few["days"] == 10 and np.isnan(few["icir"])  # < 20 usable days: not reported


def test_run_eval_daily_view_does_not_change_tests(cfg):
    """The descriptive columns use their own random stream: pooled-IC p-values and lifts must
    be what they were before the daily view existed (same seed, same draws)."""
    inp, c, _ = synthetic_inputs(cfg, signal=0.4, n_days=150)
    res = run_eval(inp, c, univariate=["1d_llm_sent_decay"])
    u = res["univariate"].to_dicts()[0]
    assert u["d_days"] > 100 and u["d_ic_daily"] > 0.2 and u["d_q_spread"] > 0
    rng = np.random.default_rng(c.eval.seed)
    from newsfeed.eval.report import assemble, block_bootstrap, boot_summary, spearman

    base = assemble(inp, c)
    t = inp.targets.filter(pl.col("entry") == "close")
    df = base.join(t, on=["ticker", "date"]).sort("date", "ticker")
    from newsfeed.eval.cv import pre_holdout_mask

    dec = df["date"].to_numpy().astype("datetime64[D]")
    end = df["label_end"].to_numpy().astype("datetime64[D]")
    df = df.filter(pl.Series(pre_holdout_mask(dec, end, c.eval.holdout_start)))
    idx = {d: i for i, d in enumerate(inp.sessions)}
    s = np.array([idx[d] for d in df["date"].to_list()])
    x, y = df["1d_llm_sent_decay"].to_numpy(), df["ret_market"].to_numpy()
    boots = block_bootstrap(
        s, lambda i: spearman(x[i], y[i]), block=2, reps=c.eval.bootstrap_reps, rng=rng
    )
    assert u["p"] == pytest.approx(boot_summary(spearman(x, y), boots)["p"])
    detail = univariate_cells(inp, c, "1d_llm_sent_decay", [("close", 1, "market")], detail=True)
    assert detail[0]["d_icir"] == pytest.approx(u["d_icir"])


def test_partial_rank_corr_removes_control():
    rng = np.random.default_rng(2)
    n = 4000
    sessions_ = np.repeat(np.arange(400), 10)
    c = rng.normal(size=n)
    x_proxy = c + 0.3 * rng.normal(size=n)  # attention that only reflects trailing vol
    x_new = rng.normal(size=n)  # information the control does not have
    y = c + 0.5 * x_new + 0.5 * rng.normal(size=n)
    kw = {"block": 2, "reps": 200, "rng": np.random.default_rng(0)}
    proxy = partial_rank_corr(x_proxy, y, c[:, None], sessions_, **kw)
    assert proxy["ic"] > 0.5 and abs(proxy["partial_ic"]) < 0.05
    new = partial_rank_corr(x_new, y, c[:, None], sessions_, **kw)
    assert new["partial_ic"] > 0.3 and new["p"] < 0.01
    x_nan = x_new.copy()
    x_nan[:100] = np.nan
    assert partial_rank_corr(x_nan, y, c[:, None], sessions_, **kw)["n"] == n - 100


def test_volatility_targets_use_only_the_following_sessions():
    closes = [100, 101, 99, 102, 100, 105, 104, 103]
    p = panel("AAA", closes)
    v = volatility_targets(p, [3]).sort("date")
    r = np.diff(np.log(closes))
    rv3 = v["rv_3"].to_list()
    assert rv3[0] == pytest.approx(np.std(r[0:3], ddof=1))  # sessions 1..3 after day 0
    assert rv3[2] == pytest.approx(np.std(r[2:5], ddof=1))
    assert rv3[-3] is None and rv3[-1] is None  # window runs past the data
    assert v["absret_1"].to_list()[1] == pytest.approx(abs(99 / 101 - 1))
    # changing a close up to day i leaves day i's target unchanged except through close[i]
    p2 = panel("AAA", [50, *closes[1:]])
    assert volatility_targets(p2, [3]).sort("date")["rv_3"][1] == pytest.approx(rv3[1])


def _targets(days, tickers, ret):
    rows = []
    for d_i, d in enumerate(days):
        for t in tickers:
            r = ret(t, d_i)
            rows.append({"ticker": t, "date": d, "entry": "close", "horizon": 1,
                         "label_end": d + timedelta(days=1), "ret_market": r,
                         "ret_sector": r})  # fmt: skip
    return pl.DataFrame(rows)


def test_abnormal_returns_demean_by_date_over_members():
    days = sessions(3)
    t = _targets(days, ["A", "B", "C"], lambda t, i: {"A": 0.03, "B": 0.01, "C": 0.5}[t])
    members = pl.DataFrame({"ticker": ["A", "B"] * 3, "date": [d for d in days for _ in (0, 1)]})
    ar = abnormal_returns(t, members)
    assert set(ar["ticker"]) == {"A", "B"}  # non-members dropped, not in the mean
    a = ar.filter(pl.col("ticker") == "A")["ar_market"].to_list()
    assert a == pytest.approx([0.01] * 3)


def test_event_mean_cells_detects_drift_and_drops_duplicates(cfg):
    rng = np.random.default_rng(4)
    days = sessions(300)
    tickers = [f"T{i}" for i in range(10)]
    ev_rows = [(t, d) for d in days[::2] for t in tickers[:2]]
    hit = set(ev_rows)
    t = _targets(
        days,
        tickers,
        lambda tk, i: 0.01 * rng.normal() + (0.004 if (tk, days[i]) in hit else 0.0),
    )
    members = t.select("ticker", "date")
    ar = abnormal_returns(t, members)
    ev = pl.DataFrame(
        {
            "ticker": [r[0] for r in ev_rows] * 2,
            "date": [r[1] for r in ev_rows] * 2,
            "group": ["x"] * len(ev_rows) * 2,
        }
    )
    ec = cfg.eval.model_copy(update={"holdout_start": days[-20], "bootstrap_reps": 200})
    out = event_mean_cells(
        ev, ar, days, cfg.model_copy(update={"eval": ec}), [("close", 1, "market")]
    )
    r = out[0]
    assert r["n"] < len(ev_rows)  # duplicates dropped, holdout rows excluded
    assert r["n"] == len([1 for _, d in ev_rows if d + timedelta(days=1) < days[-20]])
    assert r["mean"] > 0.002 and r["p"] < 0.01


def test_cluster_ic_value_and_semi_partial_control(cfg):
    rng = np.random.default_rng(5)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(300)]
    rows, tgt = [], []
    for d in days:
        for j in range(3):
            sent = float(rng.normal())
            surprise = sent + 0.05 * float(rng.normal())  # mostly the same information
            r = 0.01 * sent + 0.01 * rng.normal()
            rows.append({"ticker": f"T{j}", "date": d, "cluster": 0, "sentiment": sent,
                         "surprise": surprise if j else None})  # fmt: skip
            tgt.append({"ticker": f"T{j}", "date": d, "entry": "close", "horizon": 1,
                        "label_end": d + timedelta(days=1), "ret_market": r})  # fmt: skip
    ec = cfg.eval.model_copy(update={"holdout_start": date(2030, 1, 1), "bootstrap_reps": 200})
    c = cfg.model_copy(update={"eval": ec})
    ev, tg = pl.DataFrame(rows), pl.DataFrame(tgt)
    plain = cluster_ic(ev, tg, days, c, [("close", 1, "market")], value="surprise")[0]
    assert plain["n"] == 600 and plain["ic"] > 0.4  # null values dropped
    semi = cluster_ic(
        ev, tg, days, c, [("close", 1, "market")], value="surprise", control="sentiment"
    )[0]
    assert semi["ic_value"] == pytest.approx(plain["ic"]) and semi["ic_control"] > 0.4
    assert abs(semi["ic"]) < 0.1  # nothing beyond sentiment
