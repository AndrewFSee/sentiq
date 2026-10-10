"""Forward-return targets and controls, aligned to the feature decision time.

Decision time T = session close - ``features.decision_lag_minutes`` (15:50 ET by default, which
is also the NYSE market-on-close order cutoff). On decision day t (session index i):

entry "close":      buy at close[i],    sell at close[i + h]
entry "next_open":  buy at open[i + 1], sell at open[i + 1 + h]
Total return = (exit + dividends going ex inside the holding period) / entry - 1. Dividend
entitlement: an ex-date d counts if we held at the close before d, i.e. d in (i, i + h] for
close entries and (i + 1, i + 1 + h] for open entries.
Adjusted targets subtract the same-window total return of SPY (market) or the ticker's sector
ETF (sector). ``label_end`` = the exit session, used for purging in CV.

Controls use only information available at T: closes and volumes through session i - 1 plus
today's open (the overnight gap). Today's intraday move up to 15:50 is not observable with
daily bars, so it is not controlled for (it cannot leak into targets, which start at the close).
"""

from __future__ import annotations

import polars as pl

CONTROL_COLS = [
    "ctl_ret_1d",
    "ctl_ret_5d",
    "ctl_ret_20d",
    "ctl_vol_20d",
    "ctl_log_dvol_20d",
    "ctl_volume_ratio",
    "ctl_gap",
    "ctl_mkt_ret_5d",
]


def forward_returns(panel: pl.DataFrame, horizons: list[int], entries: list[str]) -> pl.DataFrame:
    """Long table: symbol, date, entry, horizon, ret (total), label_end (exit date)."""
    p = panel.sort("symbol", "date").with_columns(
        pl.col("div").cum_sum().over("symbol").alias("_cumdiv")
    )
    out = []
    for h in horizons:
        for entry in entries:
            if entry == "close":
                px_in = pl.col("close")
                px_out = pl.col("close").shift(-h).over("symbol")
                divs = pl.col("_cumdiv").shift(-h).over("symbol") - pl.col("_cumdiv")
                end = pl.col("date").shift(-h).over("symbol")
            else:
                px_in = pl.col("open").shift(-1).over("symbol")
                px_out = pl.col("open").shift(-(1 + h)).over("symbol")
                divs = pl.col("_cumdiv").shift(-(1 + h)).over("symbol") - pl.col("_cumdiv").shift(
                    -1
                ).over("symbol")
                end = pl.col("date").shift(-(1 + h)).over("symbol")
            out.append(
                p.select(
                    "symbol",
                    "date",
                    pl.lit(entry).alias("entry"),
                    pl.lit(h).alias("horizon"),
                    ((px_out + divs) / px_in - 1).alias("ret"),
                    end.alias("label_end"),
                )
            )
    return pl.concat(out).drop_nulls(["ret"])


def adjusted_targets(
    fwd: pl.DataFrame,
    tickers: list[str],
    market: str,
    sector_of: dict[str, str],
) -> pl.DataFrame:
    """Per ticker: raw, market-adjusted and sector-adjusted forward returns (long over grid)."""
    bench = fwd.select("date", "entry", "horizon", "symbol", "ret")
    t = fwd.filter(pl.col("symbol").is_in(tickers)).rename({"symbol": "ticker"})
    t = t.with_columns(pl.col("ticker").replace_strict(sector_of, default=None).alias("_sector"))
    mkt = bench.filter(pl.col("symbol") == market).select(
        "date", "entry", "horizon", pl.col("ret").alias("_mkt")
    )
    sec = bench.rename({"symbol": "_sector", "ret": "_sec"})
    t = t.join(mkt, on=["date", "entry", "horizon"], how="left").join(
        sec, on=["date", "entry", "horizon", "_sector"], how="left"
    )
    return t.select(
        "ticker",
        "date",
        "entry",
        "horizon",
        "label_end",
        pl.col("ret").alias("ret_raw"),
        (pl.col("ret") - pl.col("_mkt")).alias("ret_market"),
        (pl.col("ret") - pl.col("_sec")).alias("ret_sector"),
    )


def controls(panel: pl.DataFrame, market: str) -> pl.DataFrame:
    """symbol, date, ctl_*: everything known at 15:50 on ``date``."""
    p = panel.sort("symbol", "date")
    c1 = pl.col("close").shift(1).over("symbol")  # previous close
    logret = (pl.col("close") / pl.col("close").shift(1)).log().over("symbol")
    dvol = pl.col("close") * pl.col("volume")
    out = p.with_columns(
        (c1 / pl.col("close").shift(2).over("symbol") - 1).alias("ctl_ret_1d"),
        (c1 / pl.col("close").shift(6).over("symbol") - 1).alias("ctl_ret_5d"),
        (c1 / pl.col("close").shift(21).over("symbol") - 1).alias("ctl_ret_20d"),
        logret.shift(1).rolling_std(20).over("symbol").alias("ctl_vol_20d"),
        dvol.shift(1).rolling_mean(20).over("symbol").log().alias("ctl_log_dvol_20d"),
        (pl.col("volume").shift(1) / pl.col("volume").shift(1).rolling_mean(20))
        .over("symbol")
        .alias("ctl_volume_ratio"),
        (pl.col("open") / c1 - 1).alias("ctl_gap"),
    )
    mkt = out.filter(pl.col("symbol") == market).select(
        "date", pl.col("ctl_ret_5d").alias("ctl_mkt_ret_5d")
    )
    return out.join(mkt, on="date", how="left").select(
        pl.col("symbol").alias("ticker"), "date", *CONTROL_COLS
    )
