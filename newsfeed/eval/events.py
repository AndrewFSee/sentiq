"""Event studies: mean abnormal return after a class of events (test 8: 8-K item codes).

Each event is dated to the first decision time (session close - lag) at or after it became
usable, as everywhere else, so the measured return is the drift a trader could still capture
after seeing the filing -- not the announcement reaction. Abnormal return = the benchmark-
adjusted return minus the same-date equal-weighted mean adjusted return of all index members:
an equal-weighted S&P 500 trailed SPY over 2019-2026, and at 20 sessions that drift alone
would look like a negative event effect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from newsfeed.config import AppConfig
from newsfeed.eval.cv import pre_holdout_mask
from newsfeed.eval.report import block_bootstrap, boot_summary


def abnormal_returns(targets: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """targets (ticker, date, entry, horizon, label_end, ret_market, ret_sector) restricted to
    ``members`` (ticker, date) -> adds ar_market / ar_sector = return minus that date's
    cross-sectional mean over members (per entry and horizon)."""
    t = targets.join(members.unique(), on=["ticker", "date"], how="semi")
    keys = ["date", "entry", "horizon"]
    return t.with_columns(
        (pl.col("ret_market") - pl.col("ret_market").mean().over(keys)).alias("ar_market"),
        (pl.col("ret_sector") - pl.col("ret_sector").mean().over(keys)).alias("ar_sector"),
    )


def event_mean_cells(
    events: pl.DataFrame,
    ar: pl.DataFrame,
    sessions: list,
    cfg: AppConfig,
    cells: list[tuple[str, int, str]],
    *,
    min_n: int = 30,
) -> list[dict[str, Any]]:
    """events: ticker, date, group (one row per event; duplicates are dropped). Mean abnormal
    return per group and (entry, horizon, adjustment) cell, pre-holdout, with a circular block
    bootstrap over calendar sessions (two-sided p, mean vs 0)."""
    ec = cfg.eval
    rng = np.random.default_rng(ec.seed)
    session_idx = {d: i for i, d in enumerate(sessions)}
    ev = events.select("ticker", "date", "group").unique()
    out = []
    for entry, h, adj in cells:
        t = ar.filter((pl.col("entry") == entry) & (pl.col("horizon") == h))
        df = ev.join(t, on=["ticker", "date"], how="inner")
        if df.height:
            dec = df["date"].to_numpy().astype("datetime64[D]")
            end = df["label_end"].to_numpy().astype("datetime64[D]")
            df = df.filter(pl.Series(pre_holdout_mask(dec, end, ec.holdout_start)))
            df = df.filter(pl.col(f"ar_{adj}").is_not_null())
        for g in sorted(ev["group"].unique().to_list()):
            sub = df.filter(pl.col("group") == g) if df.height else df
            row: dict[str, Any] = {
                "group": g,
                "entry": entry,
                "horizon": h,
                "adjustment": adj,
                "n": sub.height,
                "sessions": sub["date"].n_unique() if sub.height else 0,
            }
            nan = float("nan")
            if sub.height < min_n:
                out.append({**row, "mean": nan, "hit": nan, "ci_lo": nan, "ci_hi": nan, "p": nan})
                continue
            s = np.array([session_idx.get(d, -1) for d in sub["date"].to_list()])
            y = sub[f"ar_{adj}"].to_numpy().astype(float)
            m = float(y.mean())
            boots = block_bootstrap(
                s,
                lambda idx, y=y: float(y[idx].mean()),
                block=h + 1 if entry == "close" else h + 2,
                reps=ec.bootstrap_reps,
                rng=rng,
            )
            out.append({**row, "mean": m, "hit": float(np.mean(y > 0)), **boot_summary(m, boots)})
    return out
