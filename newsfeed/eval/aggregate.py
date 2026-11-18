"""Aggregate (market-level) earnings tone vs the market (test 12) and the materiality label vs
price impact (test 13). Pure functions; the CLI commands `tone-market` / `materiality-impact`
load the data.

Point in time: a release's year-earlier pair and its firm baseline use only *earlier* releases
of the same company; the month of a release is the New York calendar month of t_use (when it
became usable), so month m's measure is known at the last close of month m.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from newsfeed.eval.report import block_bootstrap, boot_summary, spearman

MATERIALITY_RANK = {"low": 0, "medium": 1, "high": 2}


def yoy_pairs(rel: pl.DataFrame, lo_days: int = 320, hi_days: int = 410) -> pl.DataFrame:
    """Each release (item_id, cik, t_use, sentiment) paired with the same CIK's earlier release
    whose t_use is lo_days..hi_days before (closest to 365 days; ties -> the earlier one).
    Returns item_id, cik, t_use, sentiment, prev_sentiment, diff."""
    r = rel.select("item_id", "cik", "t_use", "sentiment")
    prev = r.select(
        "cik",
        pl.col("t_use").alias("t_prev"),
        pl.col("sentiment").alias("prev_sentiment"),
        pl.col("item_id").alias("prev_id"),
    )
    days = (pl.col("t_use") - pl.col("t_prev")).dt.total_seconds() / 86400
    return (
        r.join(prev, on="cik")
        .filter((pl.col("prev_id") != pl.col("item_id")) & days.is_between(lo_days, hi_days))
        .with_columns((days - 365).abs().alias("_dist"))
        .sort("item_id", "_dist", "t_prev")
        .unique("item_id", keep="first", maintain_order=True)
        .select(
            "item_id",
            "cik",
            "t_use",
            "sentiment",
            "prev_sentiment",
            (pl.col("sentiment") - pl.col("prev_sentiment")).alias("diff"),
        )
    )


def prior_demeaned(rel: pl.DataFrame) -> pl.DataFrame:
    """sentiment minus the mean of the same CIK's strictly earlier releases (null if none)."""
    r = rel.sort("cik", "t_use", "item_id")
    cum = pl.col("sentiment").cum_sum().over("cik") - pl.col("sentiment")
    k = pl.col("sentiment").cum_count().over("cik") - 1
    return r.with_columns(
        pl.when(k > 0).then(pl.col("sentiment") - cum / k).otherwise(None).alias("demeaned")
    )


def _month(col: str) -> pl.Expr:
    return pl.col(col).dt.convert_time_zone("America/New_York").dt.truncate("1mo").dt.date()


def monthly_tone(rel: pl.DataFrame, *, min_pairs: int = 10) -> pl.DataFrame:
    """month, yoy (mean same-company change; null if < min_pairs pairs), n_pairs,
    level (mean prior-demeaned sentiment), raw (mean sentiment), n."""
    pairs = (
        yoy_pairs(rel)
        .group_by(_month("t_use").alias("month"))
        .agg(pl.col("diff").mean().alias("yoy"), pl.len().alias("n_pairs"))
    )
    lvl = (
        prior_demeaned(rel)
        .group_by(_month("t_use").alias("month"))
        .agg(
            pl.col("demeaned").mean().alias("level"),
            pl.col("sentiment").mean().alias("raw"),
            pl.len().alias("n"),
        )
    )
    return (
        lvl.join(pairs, on="month", how="left")
        .with_columns(
            pl.when(pl.col("n_pairs").fill_null(0) >= min_pairs).then(pl.col("yoy")).alias("yoy"),
            pl.col("n_pairs").fill_null(0),
        )
        .sort("month")
    )


def monthly_total_returns(bars: pl.DataFrame) -> pl.DataFrame:
    """bars: date, close, div (cash going ex that day, split-adjusted like close) ->
    month, ret = compounded daily total return over the month's sessions (= last close of the
    previous month -> last close of this month, dividends reinvested). The first month, with no
    previous close, is dropped."""
    b = bars.sort("date").with_columns(
        ((pl.col("close") + pl.col("div").fill_null(0)) / pl.col("close").shift(1)).alias("_g")
    )
    first = b["date"][0]
    return (
        b.drop_nulls("_g")
        .group_by(pl.col("date").dt.truncate("1mo").alias("month"))
        .agg((pl.col("_g").product() - 1).alias("ret"))
        .filter(pl.col("month") > first.replace(day=1))
        .sort("month")
    )


def month_index(months: list[date]) -> np.ndarray:
    return np.array([m.year * 12 + m.month - 1 for m in months])


def corr_test(
    x: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    *,
    block: int,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Spearman correlation of two monthly series, circular block bootstrap over months."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, months = x[ok], y[ok], months[ok]
    point = spearman(x, y)
    boots = block_bootstrap(months, lambda i: spearman(x[i], y[i]), block=block, reps=reps, rng=rng)
    return {"n": int(ok.sum()), "corr": point, **boot_summary(point, boots)}


def tone_market_table(tone: pl.DataFrame, spy: pl.DataFrame, holdout: date) -> pl.DataFrame:
    """One row per month m: the tone measures (yoy, level, raw; also their 3-month means ending
    at m) and SPY returns in m (contemporaneous), m + 1, and m + 1..m + 3 (compounded). Only
    months whose longest target ends before ``holdout`` keep targets."""
    s = spy.sort("month")
    ret = s["ret"].to_numpy()
    nxt = np.append(ret[1:], np.nan)
    nxt3 = np.array(
        [
            np.prod(1 + ret[i + 1 : i + 4]) - 1 if i + 3 < len(ret) else np.nan
            for i in range(len(ret))
        ]
    )
    targets = s.with_columns(
        pl.Series("spy_next", nxt), pl.Series("spy_next3", nxt3), pl.col("ret").alias("spy_same")
    ).drop("ret")
    t = tone.sort("month").with_columns(
        *[
            pl.col(c).rolling_mean(3, min_samples=3).alias(f"{c}_3m")
            for c in ("yoy", "level", "raw")
        ]
    )
    out = t.join(targets, on="month", how="left")
    end1 = pl.col("month").dt.offset_by("2mo")  # first day after month m + 1
    end3 = pl.col("month").dt.offset_by("4mo")
    return out.with_columns(
        pl.when(end1 <= holdout).then(pl.col("spy_next")).alias("spy_next"),
        pl.when(end3 <= holdout).then(pl.col("spy_next3")).alias("spy_next3"),
        pl.when(pl.col("month").dt.offset_by("1mo") <= holdout)
        .then(pl.col("spy_same"))
        .alias("spy_same"),
    )
