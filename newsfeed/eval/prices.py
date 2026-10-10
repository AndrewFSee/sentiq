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
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

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

    def splits(self, symbol: str) -> pl.DataFrame:
        """Columns ex_date (Date), factor (Float64; 4.0 for 4-for-1, 0.05 for 1-for-20): the
        splits the bars are adjusted for (adjusted price = raw / prod(factor, ex_date > d))."""
        ...


SPLIT_SCHEMA = {"ex_date": pl.Date, "factor": pl.Float64}
DIV_SCHEMA = {"ex_date": pl.Date, "amount": pl.Float64}


def vendor_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")  # BRK-B -> BRK.B


def _records(tab: object) -> list[dict]:
    """Tradier fundamentals tables come as a list, a single record, or a dict keyed by date."""
    if isinstance(tab, list):
        return [d for d in tab if isinstance(d, dict)]
    if isinstance(tab, dict):
        if "ex_date" in tab:
            return [tab]
        return [d for d in tab.values() if isinstance(d, dict)]
    return []


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
                for d in _records((res.get("tables") or {}).get("cash_dividends")):
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

    def splits(self, symbol: str) -> pl.DataFrame:
        url = f"{self.base}/beta/markets/fundamentals/corporate_actions?symbols="
        rows = []
        for req in self.http.get(url + vendor_symbol(symbol)).json() or []:
            for res in req.get("results") or []:
                for d in _records((res.get("tables") or {}).get("stock_splits")):
                    if not d.get("ex_date"):
                        continue
                    frm, to = float(d.get("split_from") or 0), float(d.get("split_to") or 0)
                    if frm > 0 and to > 0:
                        rows.append({"ex_date": d["ex_date"], "factor": to / frm})
        if not rows:
            return pl.DataFrame(schema=SPLIT_SCHEMA)
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("ex_date").str.to_date())
            .unique("ex_date", keep="first")
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

    def splits(self, symbol: str, *, refresh: bool = False) -> pl.DataFrame:
        p = self.dir / f"{symbol}.splits.parquet"
        if p.exists() and not refresh:
            return pl.read_parquet(p)
        if self.source is None:
            raise RuntimeError(f"no cached splits for {symbol} and no source")
        d = self.source.splits(symbol)
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


class TiingoPrices:
    """Tiingo EOD prices: includes delisted securities (fallback for names Tradier lacks).

    Tiingo returns *unadjusted* OHLC with ``divCash`` and ``splitFactor`` per day. To match the
    Tradier convention (split-adjusted bars, split-adjusted cash dividends, no dividend
    adjustment), prices and dividends are divided by the cumulative split factor of all later
    splits within the fetched range.
    """

    def __init__(self, token: str, rate_per_sec: float = 0.8) -> None:
        self.token = token
        self.http = HttpClient(
            user_agent="newsfeed-research",
            rate_per_sec=rate_per_sec,
            # header auth keeps the token out of URLs (httpx logs URLs at INFO)
            headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
        )
        self._divs: dict[str, pl.DataFrame] = {}
        self._splits: dict[str, pl.DataFrame] = {}

    def daily_bars(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        url = (
            f"https://api.tiingo.com/tiingo/daily/{symbol.lower()}/prices?startDate={start}"
            f"&endDate={end}"
        )
        try:
            rows = self.http.get(url).json() or []
        except Exception as exc:  # unknown ticker -> 404
            log.info("tiingo: no data for %s (%r)", symbol, exc)
            rows = []
        if not rows:
            self._divs[symbol] = pl.DataFrame(schema=DIV_SCHEMA)
            self._splits[symbol] = pl.DataFrame(schema=SPLIT_SCHEMA)
            return pl.DataFrame(schema=BAR_SCHEMA)
        df = (
            pl.DataFrame(rows)
            .select(
                pl.col("date").str.slice(0, 10).str.to_date(),
                *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume")],
                pl.col("divCash").cast(pl.Float64).fill_null(0.0),
                pl.col("splitFactor").cast(pl.Float64).fill_null(1.0),
            )
            .sort("date")
        )
        # cumulative product of splits strictly after each date (reverse cumprod, shifted)
        later = df["splitFactor"].reverse().cum_prod().reverse().shift(-1, fill_value=1.0)
        df = df.with_columns(later.alias("_f"))
        bars = df.select(
            "date",
            *[(pl.col(c) / pl.col("_f")).alias(c) for c in ("open", "high", "low", "close")],
            (pl.col("volume") * pl.col("_f")).alias("volume"),
        )
        self._divs[symbol] = df.filter(pl.col("divCash") > 0).select(
            pl.col("date").alias("ex_date"), (pl.col("divCash") / pl.col("_f")).alias("amount")
        )
        self._splits[symbol] = df.filter(pl.col("splitFactor") != 1.0).select(
            pl.col("date").alias("ex_date"), pl.col("splitFactor").alias("factor")
        )
        return bars

    def search(self, query: str) -> list[dict]:
        """Tiingo ticker search (finds securities filed under a later ticker)."""
        url = f"https://api.tiingo.com/tiingo/utilities/search?query={quote(query)}"
        try:
            return [r for r in self.http.get(url).json() or [] if isinstance(r, dict)]
        except Exception as exc:
            log.info("tiingo: search %r failed (%r)", query, exc)
            return []

    def dividends(self, symbol: str) -> pl.DataFrame:
        if symbol not in self._divs:
            self.daily_bars(symbol, date(2000, 1, 1), utcnow().date())
        return self._divs[symbol]

    def splits(self, symbol: str) -> pl.DataFrame:
        if symbol not in self._splits:
            self.daily_bars(symbol, date(2000, 1, 1), utcnow().date())
        return self._splits[symbol]


@dataclass(frozen=True)
class PriceRoute:
    """Where one universe ticker's prices come from, over its own window."""

    ticker: str  # symbol used in the panel (as in features / membership)
    fetch_symbol: str  # symbol requested from the vendor
    source: str  # "tradier" | "tiingo"
    start: date
    end: date
    name: str  # company name (membership table), for sanity reports
    reason: str  # current | renamed | delisted | override | benchmark
    alts: tuple[str, ...] = ()  # other tickers of the same CIK(s), tried on Tiingo


def route_prices(
    members: pl.DataFrame,
    current_t2c: dict[str, int],
    lo: date,
    hi: date,
    *,
    after_days: int = 45,
    overrides: dict[str, str] | None = None,
) -> list[PriceRoute]:
    """One route per member ticker active in [lo, hi).

    ``overrides`` (ticker -> "source:SYMBOL", reason ``override``) wins over the rules below;
    needed where a vendor files a security under a later ticker (Tiingo: First Republic under
    its OTC ticker FRCB, not FRC).

    * ``current``: SEC's current ticker map still gives this ticker to one of its member CIKs
      -> Tradier under the ticker itself.
    * ``renamed``: the ticker is gone (or reused by another company) but a member CIK has a
      current ticker (FB -> META, DISCA -> WBD) -> Tradier under the current ticker, whose
      history follows the company. With several current share classes the first in sort
      order is used (small class mismatch; reported).
    * ``delisted``: no current ticker for any of its CIKs -> Tiingo under the old ticker.
    A ticker reused by an unrelated company is never fetched from Tradier under that ticker.
    """
    c2t: dict[int, list[str]] = {}
    for t, c in current_t2c.items():
        c2t.setdefault(c, []).append(t)
    sib: dict[int, set[str]] = {}  # CIK -> every member ticker it had
    for c, tk in members.select("cik", "ticker").iter_rows():
        sib.setdefault(int(c), set()).add(tk)
    far = date(9999, 1, 1)
    g = (
        members.with_columns(pl.col("end").fill_null(far))
        .group_by("ticker")
        .agg(
            pl.col("cik").unique(),
            pl.col("start").min(),
            pl.col("end").max(),
            pl.col("name").sort_by("start").last(),
        )
        .sort("ticker")
    )
    routes = []
    for r in g.to_dicts():
        s = max(lo, r["start"])
        e = hi if r["end"] == far else min(hi, r["end"] + timedelta(days=after_days))
        if e <= s:
            continue
        ciks = {int(c) for c in r["cik"]}
        t = r["ticker"]
        if t in (overrides or {}):
            src, sym = (overrides or {})[t].split(":", 1)
            routes.append(PriceRoute(t, sym, src, s, e, r["name"], "override"))
            continue
        if current_t2c.get(t) in ciks:
            routes.append(PriceRoute(t, t, "tradier", s, e, r["name"], "current"))
            continue
        succ = sorted(x for c in ciks for x in c2t.get(c, []))
        alts = tuple(sorted({x for c in ciks for x in sib.get(c, set())} - {t}))
        alts += tuple(x for x in succ if x not in alts)
        if succ:
            routes.append(PriceRoute(t, succ[0], "tradier", s, e, r["name"], "renamed", alts))
        else:
            routes.append(PriceRoute(t, t, "tiingo", s, e, r["name"], "delisted", alts))
    return routes


def build_panel(
    routes: list[PriceRoute],
    caches: dict[str, CachedPrices],
    sessions: list[date],
    *,
    refresh: bool = False,
    min_coverage: float = 0.5,
) -> tuple[pl.DataFrame, list[dict], dict[str, pl.DataFrame]]:
    """Fetch every route (Tiingo as fallback when Tradier has < ``min_coverage`` of the
    window's sessions), pad delisted series with cash, and report what was used per ticker.
    Also returns, per ticker, the splits its bars are adjusted for (for market caps)."""
    frames, report = [], []
    splits: dict[str, pl.DataFrame] = {}
    for rt in routes:
        n_exp = sum(1 for d in sessions if rt.start <= d <= rt.end)
        used, sym = rt.source, rt.fetch_symbol
        p = price_panel(caches[used], [sym], rt.start, rt.end, refresh=refresh)
        if p.height < min_coverage * n_exp and used == "tradier" and "tiingo" in caches:
            alt = price_panel(caches["tiingo"], [rt.ticker], rt.start, rt.end, refresh=refresh)
            if alt.height > p.height:
                p, used, sym = alt, "tiingo", rt.ticker
        if used == "tiingo" and n_exp and not _starts_ok(p, rt, sessions):
            for cand in _tiingo_candidates(rt, caches["tiingo"], exclude={sym}):
                alt = price_panel(caches["tiingo"], [cand], rt.start, rt.end, refresh=refresh)
                if _starts_ok(alt, rt, sessions) and alt.height > p.height:
                    log.info("prices: %s (%s) via Tiingo %s", rt.ticker, rt.name, cand)
                    p, sym = alt, cand
                    break
        p = p.with_columns(pl.lit(rt.ticker).alias("symbol"), pl.lit(False).alias("synthetic"))
        if p.height:
            splits[rt.ticker] = caches[used].splits(sym, refresh=refresh)
        padded = 0
        if p.height and used == "tiingo":
            last = p["date"].max()
            if any(last < d <= rt.end for d in sessions):  # stopped trading inside window
                before = p.height
                p = pad_delisted(p.drop("synthetic"), sessions, {rt.ticker}).filter(
                    pl.col("date") <= rt.end
                )
                padded = p.height - before
        if p.height:
            frames.append(p)
        real = p.filter(~pl.col("synthetic"))
        # density of real bars from the window start to the last real bar (a delisted name
        # is not "missing" after it stopped trading)
        last_real = real["date"].max() if real.height else None
        span = sum(1 for d in sessions if rt.start <= d <= last_real) if last_real else 0
        cov = real.height / span if span else 0.0
        starts = _starts_ok(real, rt, sessions)
        report.append(
            {
                "ticker": rt.ticker,
                "fetch_symbol": sym,
                "source": used,
                "reason": rt.reason,
                "name": rt.name,
                "start": rt.start,
                "end": rt.end,
                "coverage": round(cov, 3),
                "starts_ok": starts,
                "first": p["date"].min() if p.height else None,
                "last_real": p.filter(~pl.col("synthetic"))["date"].max() if p.height else None,
                "padded": padded,
            }
        )
        if n_exp and (not starts or cov < 0.9):
            log.warning(
                "prices: %s (%s) via %s:%s starts_ok=%s density=%.0f%%",
                rt.ticker,
                rt.name,
                used,
                sym,
                starts,
                cov * 100,
            )
    panel = pl.concat(frames).sort("symbol", "date") if frames else pl.DataFrame()
    return panel, report, splits


_NAME_STOP = {"inc", "corp", "corporation", "co", "company", "the", "ltd", "plc", "group"}


def _starts_ok(p: pl.DataFrame, rt: PriceRoute, sessions: list[date], slack: int = 10) -> bool:
    """Bars begin within ``slack`` days of the window's first session."""
    first = next((d for d in sessions if d >= rt.start), None)
    return bool(p.height) and first is not None and p["date"].min() <= first + timedelta(slack)


def _tiingo_candidates(rt: PriceRoute, cache: CachedPrices, exclude: set[str]) -> list[str]:
    """Same-CIK tickers first (same entity), then Tiingo name-search hits whose name contains
    the first significant word of ours (shortest ticker first: common before preferreds)."""
    out = [a for a in rt.alts if a not in exclude]
    src = cache.source
    if isinstance(src, TiingoPrices) and rt.name:
        words = [w for w in re.findall(r"[a-z0-9&]+", rt.name.lower()) if w not in _NAME_STOP]
        if words:
            hits = src.search(" ".join(words[:3]))
            ok = {
                str(h["ticker"]).upper()
                for h in hits
                if h.get("ticker")
                and words[0] in str(h.get("name", "")).lower()
                and h.get("assetType", "Stock") == "Stock"
            }
            out += sorted(ok - exclude - set(out), key=lambda t: (len(t), t))[:3]
    return out


def pad_delisted(
    panel: pl.DataFrame, sessions: list[date], delisted: set[str], max_pad: int = 25
) -> pl.DataFrame:
    """After a delisted symbol's last bar, hold cash: append up to ``max_pad`` sessions with
    open = close = last close, no volume/dividends (``synthetic`` = True). Forward returns whose
    exit falls after delisting then equal the move up to the last trade (e.g. the deal price
    for an acquired company). Caveat: for bankruptcies the last exchange trade may overstate
    the final recovery."""
    out = [panel.with_columns(pl.lit(False).alias("synthetic"))]
    for sym in sorted(delisted):
        bars = panel.filter(pl.col("symbol") == sym)
        if not bars.height:
            continue
        last = bars.sort("date").row(-1, named=True)
        after = [d for d in sessions if d > last["date"]][:max_pad]
        if after:
            out.append(
                pl.DataFrame(
                    {
                        "symbol": sym,
                        "date": after,
                        "open": last["close"],
                        "close": last["close"],
                        "volume": 0.0,
                        "div": 0.0,
                        "synthetic": True,
                    }
                ).select(out[0].columns)
            )
    return pl.concat(out).sort("symbol", "date")
