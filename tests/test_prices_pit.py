"""M6 price plumbing: Tiingo parsing, routing by current CIK, fallback, delisting cash padding,
split-consistent market caps, point-in-time sector ETFs."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import polars as pl
import pytest

from newsfeed.eval.caps import market_caps
from newsfeed.eval.prices import (
    BAR_SCHEMA,
    DIV_SCHEMA,
    SPLIT_SCHEMA,
    CachedPrices,
    PriceRoute,
    TiingoPrices,
    TradierPrices,
    build_panel,
    route_prices,
)
from newsfeed.eval.targets import adjusted_targets, forward_returns
from newsfeed.ingest.http import HttpClient, RateLimiter
from newsfeed.universe_pit import PitUniverse


def mock_http(handler) -> HttpClient:
    return HttpClient(
        user_agent="t",
        rate_per_sec=100,
        limiter=RateLimiter(100, clock=lambda: 0.0, sleep=lambda s: None),
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ vendor parsing
def test_tiingo_split_and_dividend_adjustment():
    rows = [
        # unadjusted: $100 stock pays $1 on day 2, splits 2-for-1 on day 3
        {"date": "2020-01-02T00:00:00.000Z", "open": 99, "high": 101, "low": 98, "close": 100,
         "volume": 1000, "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2020-01-03T00:00:00.000Z", "open": 100, "high": 103, "low": 99, "close": 102,
         "volume": 1000, "divCash": 1.0, "splitFactor": 1.0},
        {"date": "2020-01-06T00:00:00.000Z", "open": 51, "high": 52, "low": 50, "close": 51,
         "volume": 2000, "divCash": 0.0, "splitFactor": 2.0},
    ]  # fmt: skip
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url)
        if "/xyz/" not in req.url.path:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json=rows)

    src = TiingoPrices("tok")
    src.http = mock_http(handler)
    bars = src.daily_bars("XYZ", date(2020, 1, 1), date(2020, 1, 10))
    assert bars["close"].to_list() == [50.0, 51.0, 51.0]
    assert bars["volume"].to_list() == [2000.0, 2000.0, 2000.0]
    assert src.dividends("XYZ").rows() == [(date(2020, 1, 3), 0.5)]
    assert src.splits("XYZ").rows() == [(date(2020, 1, 6), 2.0)]
    assert seen[0].params["token"] == "tok" and seen[0].params["startDate"] == "2020-01-01"
    # unknown ticker -> empty frames, not an exception
    assert src.daily_bars("NOPE", date(2020, 1, 1), date(2020, 1, 10)).height == 0
    assert src.dividends("NOPE").height == 0 and src.splits("NOPE").height == 0


def test_tradier_splits_parsing():
    payload = [
        {
            "request": "AAPL",
            "results": [
                {"type": "Company", "tables": {"mergers_and_acquisitions": None}},
                {"type": "Stock", "tables": {"stock_splits": None}},
                {
                    "type": "Stock",
                    "tables": {
                        "stock_splits": {
                            "2014-06-09": {"ex_date": "2014-06-09", "split_from": 1.0,
                                           "split_to": 7.0},
                            "2020-08-31": {"ex_date": "2020-08-31", "split_from": 1.0,
                                           "split_to": 4.0},
                            "2021-01-04": {"ex_date": "2021-01-04", "split_from": 20.0,
                                           "split_to": 1.0},
                        }
                    },
                },
            ],
        }
    ]  # fmt: skip
    src = TradierPrices("tok", "https://api.tradier.test", 100)
    src.http = mock_http(lambda req: httpx.Response(200, json=payload))
    assert src.splits("AAPL").rows() == [
        (date(2014, 6, 9), 7.0),
        (date(2020, 8, 31), 4.0),
        (date(2021, 1, 4), 0.05),
    ]


# ------------------------------------------------------------------ routing
MEMBERS = pl.DataFrame(
    {
        "ticker": ["KEEP", "FB", "META", "GONE", "REUSED"],
        "cik": [1, 2, 2, 3, 4],
        "name": ["Keep", "Facebook", "Meta", "Gone Co", "Old Reused"],
        "start": [
            date(2019, 1, 1),
            date(2019, 1, 1),
            date(2022, 6, 9),
            date(2019, 1, 1),
            date(2019, 1, 1),
        ],
        "end": [None, date(2022, 6, 9), None, date(2020, 3, 2), date(2021, 1, 4)],
    }
)


def test_route_prices():
    current = {"KEEP": 1, "META": 2, "REUSED": 99}  # REUSED now belongs to another company
    routes = {
        r.ticker: r for r in route_prices(MEMBERS, current, date(2019, 6, 1), date(2026, 1, 1))
    }
    assert (routes["KEEP"].source, routes["KEEP"].reason) == ("tradier", "current")
    assert (routes["FB"].fetch_symbol, routes["FB"].reason) == ("META", "renamed")
    assert routes["FB"].end == date(2022, 6, 9) + timedelta(days=45)
    assert (routes["GONE"].source, routes["GONE"].reason) == ("tiingo", "delisted")
    # a reused ticker is never fetched from Tradier under that ticker
    assert (routes["REUSED"].source, routes["REUSED"].fetch_symbol) == ("tiingo", "REUSED")
    assert routes["KEEP"].start == date(2019, 6, 1)
    # windows entirely outside [lo, hi) are skipped
    assert "GONE" not in {r.ticker for r in route_prices(MEMBERS, current, date(2021, 1, 1),
                                                         date(2026, 1, 1))}  # fmt: skip


class MemSource:
    def __init__(
        self, bars: dict[str, pl.DataFrame], splits: dict[str, pl.DataFrame] | None = None
    ):
        self.bars, self._splits, self.calls = bars, splits or {}, []

    def daily_bars(self, symbol, start, end):
        self.calls.append(symbol)
        b = self.bars.get(symbol, pl.DataFrame(schema=BAR_SCHEMA))
        return b.filter(pl.col("date").is_between(start, end))

    def dividends(self, symbol):
        return pl.DataFrame(schema=DIV_SCHEMA)

    def splits(self, symbol):
        return self._splits.get(symbol, pl.DataFrame(schema=SPLIT_SCHEMA))


def bars(days: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": days,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(days),
        }
    ).cast(BAR_SCHEMA)  # type: ignore[arg-type]


def test_build_panel_fallback_and_delisting_padding(tmp_path):
    days = weekdays(date(2021, 1, 4), 40)
    tradier = MemSource({"KEEP": bars(days, [10.0] * 40)})
    tiingo = MemSource(
        {
            "GONE": bars(days[:10], [20.0 + i for i in range(10)]),  # stops trading after day 9
            "THIN": bars(days, [5.0] * 40),
        }
    )
    caches = {
        "tradier": CachedPrices(tradier, tmp_path / "tr"),
        "tiingo": CachedPrices(tiingo, tmp_path / "ti"),
    }
    s, e = days[0], days[-1]
    routes = [
        PriceRoute("KEEP", "KEEP", "tradier", s, e, "Keep", "current"),
        PriceRoute("GONE", "GONE", "tiingo", s, e, "Gone", "delisted"),
        PriceRoute("THIN", "THIN", "tradier", s, e, "Thin", "current"),  # Tradier lacks it
    ]
    panel, report, splits = build_panel(routes, caches, days)
    rep = {r["ticker"]: r for r in report}
    assert rep["THIN"]["source"] == "tiingo" and rep["THIN"]["coverage"] == 1.0
    assert rep["GONE"]["padded"] == 25 and rep["GONE"]["last_real"] == days[9]
    assert set(splits) == {"KEEP", "GONE", "THIN"}
    gone = panel.filter(pl.col("symbol") == "GONE").sort("date")
    assert gone.filter(pl.col("synthetic"))["close"].unique().to_list() == [29.0]
    # a 5-day return from day 7 ends in cash: the move up to the last trade
    fwd = forward_returns(panel.drop("synthetic"), [5], ["close"])
    r = fwd.filter((pl.col("symbol") == "GONE") & (pl.col("date") == days[7]))["ret"][0]
    assert r == pytest.approx(29.0 / 27.0 - 1)
    # a series that trades through its window is never padded
    assert (
        rep["THIN"]["padded"] == 0
        and not panel.filter(pl.col("symbol") == "THIN")["synthetic"].any()
    )


# ------------------------------------------------------------------ market caps x splits
def test_market_caps_scale_reported_shares_by_later_splits():
    days = weekdays(date(2020, 8, 24), 10)  # 4-for-1 split ex 2020-08-31 = days[5]
    # split-adjusted closes: raw $400 before, $100 after -> adjusted 100 throughout
    p = bars(days, [100.0] * 10).with_columns(pl.lit("AAPL").alias("symbol"), div=pl.lit(0.0))
    sh = {
        "AAPL": pl.DataFrame(
            {
                "available_from": [date(2020, 8, 1), date(2020, 9, 2)],
                "shares": [4.0e9, 16.0e9],  # as reported: pre-split, then post-split
            }
        )
    }
    splits = {"AAPL": pl.DataFrame({"ex_date": [date(2020, 8, 31)], "factor": [4.0]})}
    dec = pl.DataFrame({"ticker": ["AAPL"] * 3, "date": [days[2], days[6], days[9]]})
    caps = market_caps(dec, p, sh, splits).sort("date")["mcap"].to_list()
    assert caps == pytest.approx([1.6e12] * 3)  # true cap: 4e9 x $400 = 16e9 x $100
    unadjusted = market_caps(dec, p, sh).sort("date")["mcap"].to_list()
    assert unadjusted[0] == pytest.approx(0.4e12)  # the bug this guards against


# ------------------------------------------------------------------ PIT sectors
def test_pit_sector_etfs_and_targets(caplog):
    sectors = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "valid_from": [date(2021, 1, 1), date(2021, 2, 1), date(2021, 1, 1)],
            "sector": ["Information Technology", "Communication Services", "Martian Mining"],
        }
    )
    u = PitUniverse(MEMBERS, sectors)
    f = u.sector_etf_frame({"Information Technology": "XLK", "Communication Services": "XLC"})
    assert f.rows() == [("AAA", date.min, "XLK"), ("AAA", date(2021, 2, 1), "XLC")]
    assert "Martian Mining" in caplog.text

    def fwd_rows(sym, rets):
        return [
            {"symbol": sym, "date": d, "entry": "close", "horizon": 1, "ret": r, "label_end": d}
            for d, r in zip([date(2020, 12, 1), date(2021, 2, 1)], rets, strict=True)
        ]

    fwd = pl.DataFrame(
        fwd_rows("AAA", [0.10, 0.10])
        + fwd_rows("XLK", [0.01, 0.02])
        + fwd_rows("XLC", [0.03, 0.04])
        + fwd_rows("SPY", [0.0, 0.0])
    )
    t = adjusted_targets(fwd, ["AAA"], "SPY", f).sort("date")
    # before the first snapshot -> first known sector (XLK); from 2021-02-01 -> XLC
    assert t["ret_sector"].to_list() == pytest.approx([0.09, 0.06])
