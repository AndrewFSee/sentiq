"""Warm the price caches for every PIT S&P 500 route (Tradier + Tiingo) and write a coverage
report, so `newsfeed eval` doesn't spend hours inside Tiingo's free-tier rate limit.

    uv run python scripts/prefetch_prices.py [--start 2019-01-02]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import polars as pl

from newsfeed.config import load_config, load_secrets
from newsfeed.eval.prices import (
    CachedPrices,
    PriceRoute,
    TiingoPrices,
    TradierPrices,
    build_panel,
    route_prices,
)
from newsfeed.tickers import load_ticker_map
from newsfeed.timeutil import utcnow
from newsfeed.universe_pit import PitUniverse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-02")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(None)
    ec, sec = cfg.eval, load_secrets()
    caches = {
        "tradier": CachedPrices(
            TradierPrices(sec.tradier_token, ec.tradier_base_url, ec.tradier_requests_per_second),
            cfg.paths.ref_dir / "prices",
        ),
        "tiingo": CachedPrices(
            TiingoPrices(sec.tiingo_token, ec.tiingo_requests_per_hour / 3600),
            cfg.paths.ref_dir / "prices_tiingo",
        ),
    }
    lo, hi = date.fromisoformat(args.start), utcnow().date()
    tmap = load_ticker_map(lambda: None, cfg.sources.edgar.ticker_map_url, cfg.paths.ref_dir)  # type: ignore[arg-type,return-value]
    u = PitUniverse.load(cfg.universe.pit_dir)
    bench = sorted({ec.market_etf, *ec.sector_etfs.values(), *ec.gics_sector_etfs.values()})
    b_routes = [PriceRoute(b, b, "tradier", lo, hi, b, "benchmark") for b in bench]
    spy, _, _ = build_panel([r for r in b_routes if r.ticker == ec.market_etf], caches, [])
    sessions = sorted(spy["date"].to_list())
    routes = route_prices(u.members, tmap.ticker_to_cik, lo, hi, overrides=ec.price_overrides)
    # Tiingo routes first: they are the slow, rate-limited part
    routes.sort(key=lambda r: (r.source != "tiingo", r.ticker))
    _, report, _ = build_panel(routes + b_routes, caches, sessions)
    out = cfg.paths.data_dir / "reports" / "prices-prefetch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(report)
    df.write_csv(out)
    print(df.group_by("source", "reason").agg(pl.len(), pl.col("coverage").mean()))
    print(
        df.filter(pl.col("coverage") < 0.9).select(
            "ticker", "fetch_symbol", "source", "reason", "name", "coverage", "last_real"
        )
    )
    print(f"report: {out}")


if __name__ == "__main__":
    main()
