"""Point-in-time market capitalisation from SEC XBRL share counts x previous close.

Shares come from data.sec.gov/api/xbrl/companyfacts and are usable from their ``filed`` date
(+1 day, so a filing made after the close is not used the same day). Preference:
dei:EntityCommonStockSharesOutstanding (cover page) -> us-gaap diluted weighted-average shares
-> basic weighted-average shares. Multi-class companies (Alphabet, Meta) report the cover-page
count per class, which companyfacts omits, so they fall through to the weighted averages.
Berkshire reports Class-A-equivalent shares under a custom tag: its cap is null and it lands in
an 'unknown' bucket rather than being guessed. Buckets are terciles *within the universe* per
date (with 12 mega-caps these are relative sizes, not small/mid/large in the usual sense).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from newsfeed.ingest.http import HttpClient

TAGS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
]


def shares_history(facts: dict) -> pl.DataFrame:
    """available_from (Date), shares -- from the first tag that has data."""
    for ns, tag in TAGS:
        vals = facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).get("shares", [])
        rows = [
            {
                "available_from": date.fromisoformat(v["filed"]) + timedelta(days=1),
                "shares": float(v["val"]),
                "end": v.get("end") or "",
            }
            for v in vals
            if v.get("filed") and v.get("val")
        ]
        if rows:
            return (
                pl.DataFrame(rows)
                .group_by("available_from")
                # a 10-K lists several periods: keep the most recent period's count
                .agg(pl.col("shares").sort_by("end").last())
                .sort("available_from")
            )
    return pl.DataFrame(schema={"available_from": pl.Date, "shares": pl.Float64})


def load_shares(
    http: HttpClient | None, cik: int, cache_dir: Path, *, refresh: bool = False
) -> pl.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"CIK{cik:010d}.json"
    if p.exists() and not refresh:
        facts = json.loads(p.read_text(encoding="utf-8"))
    else:
        if http is None:
            raise RuntimeError(f"no cached SEC facts for CIK {cik}")
        facts = http.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
        p.write_text(json.dumps(facts), encoding="utf-8")
    return shares_history(facts)


def market_caps(
    decisions: pl.DataFrame,
    panel: pl.DataFrame,
    shares: dict[str, pl.DataFrame],
    splits: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """ticker, date, mcap (shares known before date x previous close), size_bucket.

    Closes are split-adjusted to today but filed share counts are as reported, so each count
    is scaled by the product of split factors with ex_date after its filing date:
    mcap = adj_prev_close x shares x prod(factor, ex_date >= available_from). Without the
    ``splits`` of a ticker, pre-split caps would be understated by the split ratio."""
    splits = splits or {}
    prev_close = (
        panel.sort("symbol", "date")
        .with_columns(pl.col("close").shift(1).over("symbol").alias("prev_close"))
        .select(pl.col("symbol").alias("ticker"), "date", "prev_close")
    )
    frames = []
    for tk, sh in shares.items():
        d = decisions.filter(pl.col("ticker") == tk).select("ticker", "date").sort("date")
        if sh.height:
            sp = splits.get(tk)
            if sp is not None and sp.height:
                f = [
                    float(sp.filter(pl.col("ex_date") >= a)["factor"].product())
                    for a in sh["available_from"].to_list()
                ]
                sh = sh.with_columns(pl.col("shares") * pl.Series(f))
            d = d.join_asof(sh, left_on="date", right_on="available_from", strategy="backward")
        else:
            d = d.with_columns(pl.lit(None, pl.Float64).alias("shares"))
        frames.append(d.select("ticker", "date", "shares"))
    caps = (
        pl.concat(frames)
        .join(prev_close, on=["ticker", "date"], how="left")
        .with_columns((pl.col("shares") * pl.col("prev_close")).alias("mcap"))
    )
    rank = pl.col("mcap").rank("average").over("date") / pl.col("mcap").count().over("date")
    return caps.with_columns(
        pl.when(pl.col("mcap").is_null())
        .then(pl.lit("unknown"))
        .when(rank <= 1 / 3)
        .then(pl.lit("small"))
        .when(rank <= 2 / 3)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("large"))
        .alias("size_bucket")
    ).select("ticker", "date", "mcap", "size_bucket")
