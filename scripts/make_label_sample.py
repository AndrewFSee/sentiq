"""Draw a blinded, stratified sample of news (article, company) pairs for hand-labelling.

    uv run python scripts/make_label_sample.py [--per-type 25] [--seed 2026]

Writes data/labeling/news_sample_v1.json (what the labelling page shows: no model labels)
and data/labeling/news_sample_v1_key.csv (the model's unmasked and masked labels, kept local).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

from newsfeed.config import load_config
from newsfeed.storage import Store
from newsfeed.universe_pit import PitUniverse

TYPES = ["earnings", "guidance", "m&a", "legal_regulatory", "analyst_action", "product",
         "macro", "other"]  # fmt: skip
KEY_COLS = ("item_id", "ticker", "relevance", "event_type", "sentiment",
            "m_relevance", "m_event_type", "m_sentiment")  # fmt: skip


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    cfg = load_config(None)
    pit = PitUniverse.load(cfg.universe.pit_dir)
    names = pit.titles()
    t2c = {t: c for c, ts in pit.as_universe().cik_to_tickers.items() for t in ts}
    with Store(cfg.paths) as s:
        pairs = s.query(
            """
            SELECT u.item_id, u.ticker, u.relevance, u.event_type, u.sentiment,
                   m.relevance AS m_relevance, m.event_type AS m_event_type,
                   m.sentiment AS m_sentiment, i.publisher, i.title, i.body_text,
                   i.published_at
            FROM scores u
            JOIN items i USING (item_id)
            LEFT JOIN scores m ON m.item_id = u.item_id AND m.ticker = u.ticker
                 AND m.masked AND m.status = 'ok'
            WHERE i.source = 'news_api' AND NOT u.masked AND u.status = 'ok'
              AND u.replicate = 0 AND i.published_at < TIMESTAMP '2026-07-01'
            """
        )
    rng = random.Random(args.seed)
    picked = []
    for t in TYPES:
        rows = pairs.filter(pl.col("event_type") == t).sort("item_id", "ticker").to_dicts()
        picked += rng.sample(rows, min(args.per_type, len(rows)))
    rng.shuffle(picked)
    items, key = [], []
    for k, r in enumerate(picked, 1):
        sid = f"n{k:03d}"
        ticker = r["ticker"]
        items.append(
            {
                "id": sid,
                "company": names.get(t2c.get(ticker, -1), ticker),
                "ticker": ticker,
                "publisher": r["publisher"],
                "published": r["published_at"].strftime("%Y-%m-%d"),
                "title": r["title"],
                "summary": r["body_text"],
            }
        )
        key.append({"id": sid, **{c: r[c] for c in KEY_COLS}})
    out = Path(cfg.paths.data_dir) / "labeling"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"version": "news_sample_v1", "seed": args.seed, "per_type": args.per_type,
           "items": items}  # fmt: skip
    (out / "news_sample_v1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    pl.DataFrame(key).write_csv(out / "news_sample_v1_key.csv")
    print(
        f"{len(items)} items -> {out}; json {len((out / 'news_sample_v1.json').read_bytes())} bytes"
    )


if __name__ == "__main__":
    main()
