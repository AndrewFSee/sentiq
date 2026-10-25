"""Training data for a local model that imitates the LLM's labels (distillation).

One row per scored (story founder, company) pair: the *exact* user turn the LLM saw (re-rendered
with the scoring prompt and checked against the stored ``user_text_hash``) and the LLM's
unmasked labels (sentiment, relevance, event type). Only pre-holdout documents: rows whose
``tradable_at`` is on/after ``eval.holdout_start`` never enter the dataset.

Splits (for the "free scorer" model; walk-forward training for return analyses uses ``year``):
- ``gold``: pairs in a hand-labelling sample (never trained on; scored against hand labels)
- per source, chronological by ``tradable_at``: ``train`` / ``val`` / ``test`` at the
  ``split_fracs`` quantiles of that source's dates (news covers only 2024-10 on, so a global
  calendar cut would leave it without training rows).

Built by ``newsfeed distill-data``; trained by ``scripts/distill_train.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from newsfeed.scoring.prompt import PromptSpec, doc_type_for
from newsfeed.storage import Store

DISTILL_VERSION = 1
RELEVANCE = ["passing", "secondary", "primary"]
EVENT_TYPES = ["earnings", "guidance", "m&a", "legal_regulatory", "analyst_action", "product",
               "macro", "other"]  # fmt: skip
SPLITS = ("train", "val", "test")


def load_pairs(store: Store, sources: Iterable[str], model: str) -> pl.DataFrame:
    """Unmasked first-draw ok scores of each item's current text, with that text (all dates;
    filter later), from ``model`` only (validation runs scored some items with another model).
    One row per (item, ticker): a re-parsed item also has a score of its old text, dropped here."""
    return store.query(
        """
        SELECT u.item_id, u.ticker, u.cik, u.prompt_hash, u.user_text_hash,
               u.sentiment, u.relevance, u.event_type,
               i.source, i.publisher, i.title, i.body_text, i.body_truncated, i.meta,
               i.published_at, i.tradable_at
        FROM scores u JOIN items i USING (item_id)
        WHERE NOT u.masked AND u.status = 'ok' AND u.replicate = 0
          AND u.text_hash = i.text_hash  -- score of the current text (re-parsed items keep both)
          AND u.model = $model AND i.source IN (SELECT unnest($sources))
        """,
        {"sources": list(sources), "model": model},
    )


def pre_holdout(df: pl.DataFrame, holdout_start: date) -> pl.DataFrame:
    """Drop every row tradable on/after the holdout start (the holdout stays unseen)."""
    cut = datetime(holdout_start.year, holdout_start.month, holdout_start.day, tzinfo=UTC)
    return df.filter(pl.col("tradable_at") < cut)


def render_texts(
    df: pl.DataFrame, prompt: PromptSpec, company_names: dict[int, str]
) -> pl.DataFrame:
    """Add ``text`` = the user turn as the scorer rendered it, and ``text_ok`` = its hash matches
    the stored ``user_text_hash`` (False e.g. when a company's display name changed since)."""
    texts, ok = [], []
    for r in df.iter_rows(named=True):
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
        name = company_names.get(r["cik"]) or r["ticker"]
        user = prompt.render_user(
            target=f"{name} (ticker: {r['ticker']})",
            doc_type=doc_type_for(r["source"], meta),
            publisher=r["publisher"],
            title=r["title"] or "",
            body=r["body_text"] or "",
            truncated=bool(r["body_truncated"]),
        )
        texts.append(user)
        ok.append(hashlib.sha256(user.encode()).hexdigest()[:16] == r["user_text_hash"])
    return df.with_columns(pl.Series("text", texts), pl.Series("text_ok", ok))


def assign_splits(
    df: pl.DataFrame,
    gold: set[tuple[str, str]],
    split_fracs: tuple[float, float] = (0.7, 0.8),
) -> pl.DataFrame:
    """``split`` column: gold pairs first, then per-source chronological train/val/test."""
    lo, hi = split_fracs
    if not 0 < lo < hi < 1:
        raise ValueError("split_fracs must satisfy 0 < train < val < 1")
    is_gold = [(i, t) in gold for i, t in zip(df["item_id"], df["ticker"], strict=True)]
    df = df.with_columns(pl.Series("_gold", is_gold, dtype=pl.Boolean))
    # cuts are timestamps compared with <=, so items sharing a tradable_at (a session open)
    # never straddle a cut
    rows = []
    for (src,), g in df.filter(~pl.col("_gold")).group_by("source"):
        ts = g["tradable_at"].sort()
        rows.append({"source": src, "_c1": ts[int(len(ts) * lo)], "_c2": ts[int(len(ts) * hi)]})
    ts_type = df.schema["tradable_at"]
    cuts = pl.DataFrame(rows, schema={"source": pl.String, "_c1": ts_type, "_c2": ts_type})
    df = df.join(cuts, on="source", how="left")
    split = (
        pl.when(pl.col("_gold"))
        .then(pl.lit("gold"))
        .when(pl.col("tradable_at") <= pl.col("_c1"))
        .then(pl.lit("train"))
        .when(pl.col("tradable_at") <= pl.col("_c2"))
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    return df.with_columns(split.alias("split")).drop("_gold", "_c1", "_c2")


def gold_pairs(label_dir: Path) -> set[tuple[str, str]]:
    """(item_id, ticker) of every hand-labelling sample key in ``label_dir``."""
    out: set[tuple[str, str]] = set()
    for p in sorted(label_dir.glob("*_key.csv")):
        k = pl.read_csv(p, columns=["item_id", "ticker"])
        out |= set(zip(k["item_id"], k["ticker"], strict=True))
    return out


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Training columns, label codes and a calendar year (for walk-forward blocks)."""
    rel = {r: i for i, r in enumerate(RELEVANCE)}
    evt = {e: i for i, e in enumerate(EVENT_TYPES)}
    return df.select(
        "item_id",
        "ticker",
        "cik",
        "source",
        "published_at",
        "tradable_at",
        pl.col("tradable_at").dt.year().alias("year"),
        "split",
        "text",
        "text_ok",
        pl.col("sentiment").cast(pl.Float32),
        pl.col("relevance").replace_strict(rel, return_dtype=pl.Int8).alias("relevance_id"),
        pl.col("event_type").replace_strict(evt, return_dtype=pl.Int8).alias("event_type_id"),
        "relevance",
        "event_type",
        "prompt_hash",
    ).sort("source", "tradable_at", "item_id", "ticker")
