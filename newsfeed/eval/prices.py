"""Daily prices behind a small interface (Tradier implementation), cached on disk.

Tradier facts (verified 2026-10-10 against the API):
  * /v1/markets/history daily bars are **split-adjusted but not dividend-adjusted**
    (AAPL 2020-01-02 close = 75.0875 = 300.35 / 4).
  * /beta/markets/fundamentals/dividends gives ex-dates and split-adjusted cash amounts
    (NVDA's pre-split $0.04 appears as 0.004), consistent with the bars -> total returns are
    built here by adding dividends on ex-dates.
  * Intraday time & sales only go back 20-40 sessions, so only daily horizons are testable
    historically.
  * Rate limit 120 requests/min.
Caveats: today's split adjustment is applied retroactively (fine for returns); a present-day
ticker list has survivorship bias.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import polars as pl

from newsfeed.ingest.http import HttpClient
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)

BAR_SCHEMA = {
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class PriceSource(Protocol):
    def daily_bars(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Columns of BAR_SCHEMA, one row per trading day in [start, end]."""
        ...

    def dividends(self, symbol: str) -> pl.DataFrame:
        """Columns ex_date (Date), amount (Float64): split-adjusted like the bars."""
        ...


def vendor_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")  # BRK-B -> BRK.B


class TradierPrices:
    def __init__(self, token: str, base_url: str, rate_per_sec: float) -> None:
        self.base = base_url.rstrip("/")
        self.http = HttpClient(
            user_agent="newsfeed-research",
            rate_per_sec=rate_per_sec,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )

    def daily_bars(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        url = (
            f"{self.base}/v1/markets/history?symbol={vendor_symbol(symbol)}&interval=daily"
            f"&start={start.isoformat()}&end={end.isoformat()}"
        )
        hist = (self.http.get(url).json() or {}).get("history") or {}
        days = hist.get("day") or []
        if isinstance(days, dict):  # a single bar comes back as an object
            days = [days]
        if not days:
            return pl.DataFrame(schema=BAR_SCHEMA)
        return pl.DataFrame(days).select(
            pl.col("date").str.to_date(),
            *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")],
        )

    def dividends(self, symbol: str) -> pl.DataFrame:
        url = f"{self.base}/beta/markets/fundamentals/dividends?symbols={vendor_symbol(symbol)}"
        rows = []
        for req in self.http.get(url).json() or []:
            for res in req.get("results") or []:
                for d in (res.get("tables") or {}).get("cash_dividends") or []:
                    if d.get("ex_date") and d.get("cash_amount") is not None:
                        rows.append({"ex_date": d["ex_date"], "amount": float(d["cash_amount"])})
        if not rows:
            return pl.DataFrame(schema={"ex_date": pl.Date, "amount": pl.Float64})
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("ex_date").str.to_date())
            .unique("ex_date", keep="first")  # one entry per share class may repeat
            .sort("ex_date")
        )


class CachedPrices:
    """Disk cache in front of a PriceSource: data/ref/prices/<SYMBOL>.parquet (+ dividends)."""

    def __init__(self, source: PriceSource | None, cache_dir: Path) -> None:
        self.source = source
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _meta(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.json"

    def daily_bars(
        self, symbol: str, start: date, end: date, *, refresh: bool = False
    ) -> pl.DataFrame:
        bars_p = self.dir / f"{symbol}.parquet"
        meta_p = self._meta(symbol)
        if bars_p.exists() and meta_p.exists() and not refresh:
            m = json.loads(meta_p.read_text())
            if date.fromisoformat(m["start"]) <= start and date.fromisoformat(m["end"]) >= end:
                return pl.read_parquet(bars_p).filter(pl.col("date").is_between(start, end))
        if self.source is None:
            raise RuntimeError(f"no cached prices for {symbol} {start}..{end} and no source")
        lo = start - timedelta(days=45)  # room for trailing controls
        bars = self.source.daily_bars(symbol, lo, end)
        bars.write_parquet(bars_p)
        meta_p.write_text(
            json.dumps(
                {
                    "start": lo.isoformat(),
                    "end": end.isoformat(),
                    "fetched_at": utcnow().isoformat(),
                }
            )
        )
        return bars.filter(pl.col("date").is_between(start, end))

    def dividends(self, symbol: str, *, refresh: bool = False) -> pl.DataFrame:
        p = self.dir / f"{symbol}.dividends.parquet"
        if p.exists() and not refresh:
            return pl.read_parquet(p)
        if self.source is None:
            raise RuntimeError(f"no cached dividends for {symbol} and no source")
        d = self.source.dividends(symbol)
        d.write_parquet(p)
        return d


def price_panel(
    prices: CachedPrices, symbols: list[str], start: date, end: date, *, refresh: bool = False
) -> pl.DataFrame:
    """Long panel: symbol, date, open, close, volume, div (cash dividend going ex on date)."""
    frames = []
    for sym in sorted(set(symbols)):
        bars = prices.daily_bars(sym, start, end, refresh=refresh)
        divs = prices.dividends(sym, refresh=refresh).rename({"ex_date": "date", "amount": "div"})
        frames.append(
            bars.join(divs, on="date", how="left")
            .with_columns(pl.col("div").fill_null(0.0), pl.lit(sym).alias("symbol"))
            .select("symbol", "date", "open", "close", "volume", "div")
        )
    return pl.concat(frames).sort("symbol", "date") if frames else pl.DataFrame()
