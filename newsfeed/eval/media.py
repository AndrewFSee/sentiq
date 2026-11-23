"""Market-level media tone vs the next day's market (test 14, after Tetlock 2007). Pure
functions; the CLI command `media-tone` loads the data.

Point in time: a document belongs to session i when its first 15:50 decision at or after
t_use is session i's (so t_use in (decision[i-1], decision[i]]); the demeaned tone subtracts the
mean of *earlier* sessions only. Targets start at close[i]. The return controls r0..r4 end at
close[i] -- 10 minutes after the decision, which daily bars cannot resolve; `ctl_tradable` (r1..
r5 + today's gap) uses only prices known at the decision.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

CONTROLS = ["r0", "r1", "r2", "r3", "r4"]  # SPY returns of sessions i, i-1, .., i-4
CONTROLS_TRADABLE = ["r1", "r2", "r3", "r4", "r5", "gap"]
VOLUME_CONTROLS = [*CONTROLS, "vol_today"]


def session_tone(
    docs: pl.DataFrame,
    sessions: list[date],
    *,
    value: str,
    min_docs: int = 10,
    demean: int = 20,
    min_prior: int = 15,
) -> pl.DataFrame:
    """docs (date = session of the document's first decision, ``value``) -> one row per
    session: n (documents), tone (mean value; null if n < min_docs), tone_dm = tone minus the
    mean tone of the previous ``demean`` sessions (null if fewer than ``min_prior`` of them
    have a tone)."""
    daily = docs.group_by("date").agg(pl.len().alias("n"), pl.col(value).mean().alias("tone"))
    s = (
        pl.DataFrame({"date": sessions}, schema={"date": pl.Date})
        .join(daily, on="date", how="left")
        .sort("date")
        .with_columns(
            pl.col("n").fill_null(0),
            pl.when(pl.col("n").fill_null(0) >= min_docs).then(pl.col("tone")).alias("tone"),
        )
    )
    prior = pl.col("tone").shift(1).rolling_mean(demean, min_samples=min_prior)
    return s.with_columns((pl.col("tone") - prior).alias("tone_dm"))


def market_daily(bars: pl.DataFrame) -> pl.DataFrame:
    """bars (date, open, close, volume, div; one symbol, every session) -> date, r = total
    return close[j-1] -> close[j] (dividend going ex on j added), gap = open[j] / close[j-1] - 1,
    volume. The first session (no previous close) is dropped."""
    b = bars.sort("date")
    prev = pl.col("close").shift(1)
    return (
        b.with_columns(
            ((pl.col("close") + pl.col("div").fill_null(0)) / prev - 1).alias("r"),
            (pl.col("open") / prev - 1).alias("gap"),
        )
        .drop_nulls("r")
        .select("date", "r", "gap", pl.col("volume").cast(pl.Float64))
    )


def _compound(r: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """prod(1 + r[i+lo .. i+hi]) - 1 for each i (NaN when the window leaves the series)."""
    n = len(r)
    out = np.full(n, np.nan)
    for i in range(n):
        if i + hi < n and i + lo >= 0:
            out[i] = np.prod(1 + r[i + lo : i + hi + 1]) - 1
    return out


def market_table(daily: pl.DataFrame, holdout: date) -> pl.DataFrame:
    """One row per session i: return controls r0 (= r_i) .. r5 (= r_{i-5}), gap (today's open
    vs yesterday's close), vol_today = log(volume_i / mean volume of i-20..i-1); targets fwd1 =
    r_{i+1}, fwd5 = close[i] -> close[i+5], rev = close[i+1] -> close[i+5] (days 2-5),
    vol_next = log(volume_{i+1} / mean volume of i-19..i). A target is null unless its last
    session is before ``holdout``."""
    d = daily.sort("date")
    r = d["r"].to_numpy()
    dates = d["date"].to_list()
    n = len(r)
    exit_ok = {k: np.array([i + k < n and dates[i + k] < holdout for i in range(n)])
               for k in (1, 5)}  # fmt: skip
    v = d["volume"].to_numpy().astype(float)
    logv = np.log(np.where(v > 0, v, np.nan))
    base_prev = pl.Series(v).shift(1).rolling_mean(20, min_samples=20).to_numpy()  # i-20..i-1
    base_incl = pl.Series(v).rolling_mean(20, min_samples=20).to_numpy()  # i-19..i
    vol_next = np.append(logv[1:], np.nan) - np.log(base_incl)
    cols = {f"r{k}": pl.Series(r).shift(k).to_numpy() for k in range(6)}
    out = d.select("date", "gap").with_columns(
        *[pl.Series(k, x) for k, x in cols.items()],
        pl.Series("vol_today", logv - np.log(base_prev)),
        pl.Series("fwd1", np.where(exit_ok[1], np.append(r[1:], np.nan), np.nan)),
        pl.Series("fwd5", np.where(exit_ok[5], _compound(r, 1, 5), np.nan)),
        pl.Series("rev", np.where(exit_ok[5], _compound(r, 2, 5), np.nan)),
        pl.Series("vol_next", np.where(exit_ok[1], vol_next, np.nan)),
    )
    return out.with_columns(pl.col(pl.Float64).fill_nan(None))


# --------------------------------------------------------------------- H4 (holdout hypothesis)
H4_EARLIEST_LOOK = date(2027, 8, 31)  # pre-registered 2026-11-22, with H1 / H2
H4_ALPHA_ONE_SIDED = 0.025  # own family


def check_h4_look(today: date) -> None:
    """The single H4 holdout look is allowed on/after ``H4_EARLIEST_LOOK`` only."""
    if today < H4_EARLIEST_LOOK:
        raise ValueError(f"H4 holdout look is pre-registered for on/after {H4_EARLIEST_LOOK}")


def h4_test(
    docs: pl.DataFrame,
    daily: pl.DataFrame,
    *,
    holdout: date,
    look: date | None,
    reps: int,
    rng: np.random.Generator,
    min_docs: int = 10,
    demean: int = 20,
) -> dict:
    """H4 statistic: session 8-K tone (demeaned over the previous ``demean`` sessions) vs SPY
    close[i] -> close[i+1], partial rank correlation given r0..r4, block 2.

    ``look`` None = pre-holdout replication: sessions and targets before ``holdout`` only.
    ``look`` = the holdout look: sessions from ``holdout`` on whose target ends before
    ``look``; the demean and the controls may reach back before the holdout (tone and past
    returns only)."""
    from newsfeed.eval.report import partial_rank_corr

    cut = holdout if look is None else look
    mkt = market_table(daily, cut).filter(pl.col("date") < cut)
    sessions = mkt["date"].to_list()
    tone = session_tone(docs, sessions, value=docs.columns[1], min_docs=min_docs, demean=demean)
    rows = (pl.col("date") < holdout) if look is None else (pl.col("date") >= holdout)
    m = mkt.with_columns(tone["tone_dm"], tone["n"]).filter(rows)

    def a(c: str) -> np.ndarray:
        return m[c].cast(pl.Float64).fill_null(np.nan).to_numpy()

    x, y = a("tone_dm"), a("fwd1")
    r = partial_rank_corr(
        x, y, np.column_stack([a(c) for c in CONTROLS]), np.arange(m.height),
        block=2, reps=reps, rng=rng,
    )  # fmt: skip
    ok = ~(np.isnan(x) | np.isnan(y))
    used = m.filter(pl.Series(ok))
    p_one = r["p"] / 2 if r["partial_ic"] > 0 else 1 - r["p"] / 2
    return {
        **r,
        "p_one_sided": p_one,
        "pass": bool(r["partial_ic"] > 0 and p_one < H4_ALPHA_ONE_SIDED),
        "first": used["date"].min(),
        "last": used["date"].max(),
        "median_docs": used["n"].median(),
    }
