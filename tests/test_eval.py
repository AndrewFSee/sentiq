"""Evaluation: targets, controls (PIT), CV purging/holdout, statistics, prices, end-to-end."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import polars as pl
import pytest

from newsfeed.config import CVConfig
from newsfeed.eval.caps import market_caps, shares_history
from newsfeed.eval.cv import pre_holdout_mask, purged_splits
from newsfeed.eval.prices import CachedPrices, TradierPrices, price_panel
from newsfeed.eval.report import (
    EvalInputs,
    assemble,
    benjamini_hochberg,
    block_bootstrap,
    run_eval,
    spearman,
)
from newsfeed.eval.targets import CONTROL_COLS, adjusted_targets, controls, forward_returns
from newsfeed.ingest.http import HttpClient, RateLimiter


def sessions(n: int, start: date = date(2025, 1, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def panel(symbol: str, closes, opens=None, divs=None, vol=1e6, days=None) -> pl.DataFrame:
    days = days or sessions(len(closes))
    n = len(closes)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": days,
            "open": [float(x) for x in (opens if opens is not None else closes)],
            "close": [float(x) for x in closes],
            "volume": [float(vol)] * n,
            "div": [float(x) for x in (divs if divs is not None else [0.0] * n)],
        }
    )


# ------------------------------------------------------------------ targets
def test_forward_returns_entries_and_dividend_timing():
    closes = [100, 101, 102, 103, 104, 105]
    opens = [99, 100.5, 101.5, 102.5, 103.5, 104.5]
    divs = [0, 0, 1.0, 0, 0, 0]  # goes ex on session 2
    p = panel("A", closes, opens, divs)
    f = forward_returns(p, [1, 2], ["close", "next_open"])
    d = sessions(6)

    def r(entry, h, i):
        return f.filter(
            (pl.col("entry") == entry) & (pl.col("horizon") == h) & (pl.col("date") == d[i])
        )["ret"][0]

    # close entry at session 1, exit close of 2; ex-date 2 in (1, 2] -> dividend counts
    assert r("close", 1, 1) == pytest.approx((102 + 1) / 101 - 1)
    # close entry at session 2 (ex-date itself): bought at the ex-date close -> no dividend
    assert r("close", 1, 2) == pytest.approx(103 / 102 - 1)
    # next_open from decision 0: buy open of 1, sell open of 2; ex-date 2 in (1, 2] -> counts
    assert r("next_open", 1, 0) == pytest.approx((101.5 + 1) / 100.5 - 1)
    # next_open from decision 1: buy at open of the ex-date -> not entitled
    assert r("next_open", 1, 1) == pytest.approx(102.5 / 101.5 - 1)
    label = f.filter(
        (pl.col("entry") == "next_open") & (pl.col("horizon") == 2) & (pl.col("date") == d[0])
    )["label_end"][0]
    assert label == d[3]  # exit at open of session 0 + 1 + 2
    # no return when the exit is beyond the data
    assert (
        f.filter(
            (pl.col("entry") == "close") & (pl.col("horizon") == 2) & (pl.col("date") == d[5])
        ).height
        == 0
    )


def test_adjusted_targets():
    a = panel("AAA", [100, 110, 121])
    spy = panel("SPY", [100, 105, 110.25])
    xlk = panel("XLK", [100, 100, 100])
    fwd = forward_returns(pl.concat([a, spy, xlk]), [1], ["close"])
    t = adjusted_targets(fwd, ["AAA"], "SPY", {"AAA": "XLK"})
    row = t.filter(pl.col("date") == sessions(3)[0]).to_dicts()[0]
    assert row["ret_raw"] == pytest.approx(0.10)
    assert row["ret_market"] == pytest.approx(0.05)
    assert row["ret_sector"] == pytest.approx(0.10)


def test_controls_use_only_information_known_at_decision():
    rng = np.random.default_rng(0)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 40)))
    base = pl.concat([panel("A", closes), panel("SPY", closes)])
    i = 30  # decision session
    d = sessions(40)[i]
    c0 = controls(base, "SPY").filter((pl.col("ticker") == "A") & (pl.col("date") == d))
    # change today's close and everything after it: controls at session i must not move
    perturbed = [*closes[:i], *(x * 3 for x in closes[i:])]
    opens = closes[: i + 1] + perturbed[i + 1 :]  # today's open (known at 15:50) unchanged
    p2 = pl.concat([panel("A", perturbed, opens), panel("SPY", closes)])
    c1 = controls(p2, "SPY").filter((pl.col("ticker") == "A") & (pl.col("date") == d))
    assert c0.select(CONTROL_COLS).rows() == c1.select(CONTROL_COLS).rows()
    assert c0["ctl_ret_1d"][0] == pytest.approx(closes[i - 1] / closes[i - 2] - 1)


# ------------------------------------------------------------------ CV and holdout
def test_walk_forward_purges_overlapping_labels():
    s = np.repeat(np.arange(200), 2)
    e = s + 5  # 5-session labels
    folds = purged_splits(s, e, CVConfig(mode="walk_forward", n_folds=4, min_train_sessions=60))
    assert len(folds) == 4
    for f in folds:
        assert e[f.train].max() < s[f.test].min()  # every training label resolved before test
        assert len(np.intersect1d(f.train, f.test)) == 0


def test_blocked_purge_and_embargo():
    s = np.arange(100)
    e = s + 3
    folds = purged_splits(s, e, CVConfig(mode="blocked", n_folds=4, embargo_sessions=5))
    for f in folds:
        a, b = s[f.test].min(), s[f.test].max()
        before, after = f.train[s[f.train] < a], f.train[s[f.train] > b]
        assert (e[before] < a).all()  # purged: no training label overlaps the test block
        assert (s[after] > b + 5).all()  # embargo after the block


def test_pre_holdout_mask_excludes_labels_reaching_into_holdout():
    dec = np.array(["2026-06-25", "2026-06-29", "2026-07-02"], dtype="datetime64[D]")
    end = np.array(["2026-06-26", "2026-07-06", "2026-07-03"], dtype="datetime64[D]")
    assert pre_holdout_mask(dec, end, date(2026, 7, 1)).tolist() == [True, False, False]


# ------------------------------------------------------------------ statistics
def test_spearman_and_bh():
    x = np.arange(20.0)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert np.isnan(spearman(x, np.ones(20)))
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 7


def test_block_bootstrap_matches_analytic_se_for_overlapping_labels():
    """Regression: the bootstrap SE for the mean of overlapping 20-session sums must match the
    analytic ~ 20 sigma / sqrt(n), not the naive sqrt(20) sigma / sqrt(n) (4.5x smaller)."""
    rng = np.random.default_rng(1)
    n, L = 1500, 20
    eps = rng.normal(size=n + L)
    y = np.array([eps[t : t + L].sum() for t in range(n)])  # overlapping labels
    s = np.arange(n)
    boots = block_bootstrap(s, lambda idx: y[idx].mean(), block=L + 1, reps=600, rng=rng)
    analytic = L / np.sqrt(n)
    assert 0.75 * analytic < boots.std() < 1.3 * analytic
    assert boots.std() > 2.5 * np.sqrt(L) / np.sqrt(n)  # clearly above the naive SE


def test_block_bootstrap_uses_calendar_sessions_not_sample_days():
    rng = np.random.default_rng(2)
    s = np.array([0, 1, 50, 51, 99])  # sparse sample over a 100-session span
    sizes = block_bootstrap(s, lambda idx: len(idx), block=10, reps=2000, rng=rng)
    # 10 blocks x 10 sessions cover the span on average once: mean rows ~ 5, edges not starved
    assert 4.0 < np.nanmean(sizes) < 6.0  # (a resample may draw no rows: NaN, by design)
    first = block_bootstrap(s, lambda idx: float(np.sum(s[idx] == 0)), block=10, reps=2000, rng=rng)
    last = block_bootstrap(s, lambda idx: float(np.sum(s[idx] == 99)), block=10, reps=2000, rng=rng)
    assert abs(np.nanmean(first) - np.nanmean(last)) < 0.1  # circular: ends drawn equally


# ------------------------------------------------------------------ caps
def test_shares_history_prefers_cover_page_and_latest_period():
    facts = {
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [{"val": 100, "filed": "2025-02-01", "end": "2025-01-20"}]}
                }
            },
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {"shares": [{"val": 1, "filed": "2025-02-01", "end": "2024-12-31"}]}
                }
            },
        }
    }
    sh = shares_history(facts)
    assert sh.to_dicts() == [{"available_from": date(2025, 2, 2), "shares": 100.0}]
    tenk = {
        "facts": {
            "us-gaap": {
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"val": 90, "filed": "2025-02-01", "end": "2023-12-31"},
                            {"val": 95, "filed": "2025-02-01", "end": "2024-12-31"},
                        ]
                    }
                }
            }
        }
    }
    assert shares_history(tenk)["shares"].to_list() == [95.0]  # most recent period


def test_market_caps_point_in_time_and_buckets():
    days = sessions(5)
    p = pl.concat([panel(t, [10.0] * 5, days=days) for t in ("A", "B", "C", "D")])
    sh = {
        "A": pl.DataFrame({"available_from": [days[3]], "shares": [1.0]}),
        "B": pl.DataFrame({"available_from": [days[0]], "shares": [2.0]}),
        "C": pl.DataFrame({"available_from": [days[0]], "shares": [3.0]}),
        "D": pl.DataFrame(schema={"available_from": pl.Date, "shares": pl.Float64}),
    }
    dec = pl.DataFrame(
        {"ticker": ["A", "A", "B", "C", "D"], "date": [days[2], days[4], days[2], days[2], days[2]]}
    )
    caps = market_caps(dec, p, sh)
    get = {(r["ticker"], r["date"]): r for r in caps.to_dicts()}
    assert get[("A", days[2])]["mcap"] is None  # shares not filed yet
    assert get[("A", days[4])]["mcap"] == pytest.approx(10.0)
    assert get[("D", days[2])]["size_bucket"] == "unknown"
    assert (
        get[("B", days[2])]["size_bucket"] == "mid"
        and get[("C", days[2])]["size_bucket"] == "large"
    )


# ------------------------------------------------------------------ prices
def test_tradier_parsing_and_cache(tmp_path):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        assert req.headers["Authorization"] == "Bearer tok"
        if "history" in req.url.path:
            return httpx.Response(
                200,
                json={
                    "history": {
                        "day": {
                            "date": "2025-01-02",
                            "open": 1,
                            "high": 2,
                            "low": 0.5,
                            "close": 1.5,
                            "volume": 10,
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json=[
                {
                    "results": [
                        {"tables": {"cash_dividends": None}},
                        {
                            "tables": {
                                "cash_dividends": [
                                    {"ex_date": "2025-01-02", "cash_amount": 0.25},
                                    {"ex_date": "2025-01-02", "cash_amount": 0.25},
                                ]
                            }
                        },
                    ]
                }
            ],
        )

    src = TradierPrices("tok", "https://api.tradier.test", 100)
    src.http = HttpClient(
        user_agent="t",
        rate_per_sec=100,
        limiter=RateLimiter(100, clock=lambda: 0.0, sleep=lambda s: None),
        headers={"Authorization": "Bearer tok"},
        transport=httpx.MockTransport(handler),
    )
    assert "BRK.B" in str(
        src.http.get("https://api.tradier.test/v1/markets/history?symbol=BRK.B").request.url
    )
    cached = CachedPrices(src, tmp_path)
    p = price_panel(cached, ["AAA"], date(2025, 1, 1), date(2025, 1, 3))
    assert p.to_dicts() == [
        {
            "symbol": "AAA",
            "date": date(2025, 1, 2),
            "open": 1.0,
            "close": 1.5,
            "volume": 10.0,
            "div": 0.25,
        }
    ]
    n = len(calls)
    price_panel(CachedPrices(None, tmp_path), ["AAA"], date(2025, 1, 1), date(2025, 1, 3))
    assert len(calls) == n  # served from cache, no source needed


# ------------------------------------------------------------------ end to end
def synthetic_inputs(cfg, *, signal: float, n_days: int = 320, seed: int = 0):
    """12 tickers; LLM feature predicts next-day returns with strength ``signal``."""
    rng = np.random.default_rng(seed)
    days = sessions(n_days + 30)
    tickers = [f"T{i}" for i in range(12)]
    llm = rng.normal(size=(len(days), len(tickers)))
    lm = rng.normal(size=llm.shape)
    rets = 0.01 * rng.normal(size=llm.shape)
    rets[1:] += signal * 0.01 * llm[:-1]  # today's feature -> tomorrow's close-to-close
    frames = []
    for j, t in enumerate([*tickers, "SPY"]):
        r = rets[:, j] if j < len(tickers) else np.zeros(len(days))
        closes = 100 * np.cumprod(1 + r)
        frames.append(panel(t, list(closes), days=days))
    p = pl.concat(frames)
    feats = pl.DataFrame(
        {
            "ticker": np.repeat(tickers, len(days)),
            "as_of": pl.Series(
                [datetime(d.year, d.month, d.day, 20, 50, tzinfo=UTC) for d in days] * len(tickers),
                dtype=pl.Datetime("us", "UTC"),
            ),
            "1d_n_stories": 1,
            "1d_llm_sent_decay": llm.T.ravel(),
            "1d_lm_tone_decay": lm.T.ravel(),
        }
    )
    fwd = forward_returns(p, [1], ["close"])
    targets = adjusted_targets(fwd, tickers, "SPY", {t: "SPY" for t in tickers})
    caps = feats.select("ticker", pl.col("as_of").dt.date().alias("date")).with_columns(
        pl.lit(None, pl.Float64).alias("mcap"), pl.lit("unknown").alias("size_bucket")
    )
    ecfg = cfg.eval.model_copy(
        update={
            "horizons": [1],
            "entries": ["close"],
            "adjustments": ["market"],
            "holdout_start": days[-40],
            "bootstrap_reps": 200,
        }
    )
    cfg2 = cfg.model_copy(update={"eval": ecfg})
    return EvalInputs(feats, controls(p, "SPY"), targets, caps, days), cfg2, days


def test_run_eval_detects_planted_signal(cfg):
    inp, c, _ = synthetic_inputs(cfg, signal=0.6)
    res = run_eval(inp, c, univariate=["1d_llm_sent_decay", "1d_lm_tone_decay"])
    u = {r["feature"]: r for r in res["univariate"].to_dicts()}
    assert u["1d_llm_sent_decay"]["ic"] > 0.3 and u["1d_llm_sent_decay"]["bh_discovery"]
    assert abs(u["1d_lm_tone_decay"]["ic"]) < 0.1
    ridge = res["models"].filter(pl.col("model") == "ridge").to_dicts()[0]
    assert ridge["lift"] > 0.2 and ridge["ci_lo"] > 0 and ridge["bh_discovery"]


def test_run_eval_noise_has_no_discoveries_and_respects_holdout(cfg):
    inp, c, _ = synthetic_inputs(cfg, signal=0.0, seed=3)
    res = run_eval(inp, c, univariate=["1d_llm_sent_decay", "1d_lm_tone_decay"])
    assert not res["univariate"]["bh_discovery"].any()
    assert not res["models"]["bh_discovery"].any()
    info = res["sample"]["close/1"]
    assert date.fromisoformat(info["last"]) < c.eval.holdout_start  # holdout untouched
    held = run_eval(inp, c, univariate=["1d_llm_sent_decay"], use_holdout=True)
    assert held["holdout"] and held["univariate"]["n"][0] > 0
    assert json.dumps(held["sample"])  # serialisable summary


def test_sample_event_types_filter(cfg):
    inp, c, _ = synthetic_inputs(cfg, signal=0.0, n_days=20)
    evt = (np.arange(inp.features.height) % 3 == 0).astype(int)
    inp.features = inp.features.with_columns(
        pl.Series("1d_llm_evt_earnings", evt), pl.Series("1d_llm_evt_ma", 0 * evt)
    )
    base = assemble(inp, c)
    c2 = c.model_copy(
        update={"eval": c.eval.model_copy(update={"sample_event_types": ["earnings"]})}
    )
    sub = assemble(inp, c2)
    assert sub.height == int(evt.sum()) < base.height
    assert (sub["1d_llm_evt_earnings"] > 0).all()
    c3 = c.model_copy(update={"eval": c.eval.model_copy(update={"sample_event_types": ["m&a"]})})
    assert assemble(inp, c3).height == 0  # 'm&a' -> column 1d_llm_evt_ma
    c4 = c.model_copy(update={"eval": c.eval.model_copy(update={"sample_event_types": ["x"]})})
    with pytest.raises(ValueError):
        assemble(inp, c4)


def test_sample_size_buckets_filter(cfg):
    inp, c, _ = synthetic_inputs(cfg, signal=0.0, n_days=20)
    b = np.where(np.arange(inp.caps.height) % 2 == 0, "small", "large")
    inp.caps = inp.caps.with_columns(pl.Series("size_bucket", b))
    c2 = c.model_copy(update={"eval": c.eval.model_copy(update={"sample_size_buckets": ["small"]})})
    sub = assemble(inp, c2)
    assert sub.height == int((b == "small").sum()) and set(sub["size_bucket"]) == {"small"}


def test_add_derived_interaction():
    from newsfeed.eval.report import add_derived

    f = pl.DataFrame(
        {"1d_llm_sent_cwmean": [0.5, None, -1.0], "1d_emb_novelty_mean": [0.4, 0.2, None]}
    )
    out = add_derived(f)["1d_llm_sent_x_novelty"].to_list()
    assert out[0] == pytest.approx(0.2) and out[1] is None and out[2] is None
    assert "1d_llm_sent_x_novelty" not in add_derived(f.drop("1d_emb_novelty_mean")).columns


def test_paired_ic_comparison_detects_lost_signal(cfg):
    from newsfeed.eval.report import paired_ic_comparison

    inp, c, _ = synthetic_inputs(cfg, signal=0.6, n_days=250)
    f = inp.features
    rng = np.random.default_rng(5)
    inp.features = f.with_columns(
        pl.col("1d_llm_sent_decay").alias("same"),
        pl.Series("noise", rng.normal(size=f.height)),
    )
    same = paired_ic_comparison(inp, c, "1d_llm_sent_decay", "same").row(0, named=True)
    assert same["diff"] == pytest.approx(0.0) and same["ic_a"] > 0.2
    lost = paired_ic_comparison(inp, c, "1d_llm_sent_decay", "noise").row(0, named=True)
    assert lost["diff"] < -0.2 and lost["ci_hi"] < 0 and lost["p"] < 0.05
