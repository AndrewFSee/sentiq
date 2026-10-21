"""Draw a blinded, stratified sample of scored (document, company) pairs for hand-labelling.

    uv run python scripts/make_label_sample.py [--per-type 25] [--seed 2026]
    uv run python scripts/make_label_sample.py --source edgar --name edgar_8k_sample_v1 \
        --counts guidance=60,earnings=40 --qualifying

Writes data/labeling/<name>.json (what the labelling page shows: no model labels) and
data/labeling/<name>_key.csv (the model's unmasked and masked labels, kept local).
Defaults reproduce news_sample_v1 (Massive news, 25 per model event type, seed 2026).
Pre-holdout documents only (published before 2026-07-01).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import polars as pl

from newsfeed.config import load_config
from newsfeed.scoring.prompt import doc_type_for
from newsfeed.storage import Store
from newsfeed.universe_pit import PitUniverse

TYPES = ["earnings", "guidance", "m&a", "legal_regulatory", "analyst_action", "product",
         "macro", "other"]  # fmt: skip
KEY_COLS = ("item_id", "ticker", "relevance", "event_type", "sentiment",
            "m_relevance", "m_event_type", "m_sentiment")  # fmt: skip
PREFIX = {"news_api": "n", "edgar": "e"}


def parse_counts(spec: str | None, per_type: int) -> list[tuple[str, int]]:
    if not spec:
        return [(t, per_type) for t in TYPES]
    out = []
    for part in spec.split(","):
        t, n = part.split("=")
        if t not in TYPES:
            raise SystemExit(f"unknown event type {t!r}")
        out.append((t, int(n)))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="news_api", choices=sorted(PREFIX))
    ap.add_argument("--name", default="news_sample_v1")
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--counts", default=None, help="e.g. guidance=60,earnings=40")
    ap.add_argument("--qualifying", action="store_true", help="model relevance >= secondary")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    counts = parse_counts(args.counts, args.per_type)
    cfg = load_config(None)
    pit = PitUniverse.load(cfg.universe.pit_dir)
    names = pit.titles()
    t2c = {t: c for c, ts in pit.as_universe().cik_to_tickers.items() for t in ts}
    qualifying = "AND u.relevance IN ('primary', 'secondary')" if args.qualifying else ""
    with Store(cfg.paths) as s:
        pairs = s.query(
            f"""
            SELECT u.item_id, u.ticker, u.relevance, u.event_type, u.sentiment,
                   m.relevance AS m_relevance, m.event_type AS m_event_type,
                   m.sentiment AS m_sentiment, i.publisher, i.title, i.body_text,
                   i.published_at, i.body_truncated, i.meta
            FROM scores u
            JOIN items i USING (item_id)
            LEFT JOIN scores m ON m.item_id = u.item_id AND m.ticker = u.ticker
                 AND m.masked AND m.status = 'ok'
            WHERE i.source = ? AND NOT u.masked AND u.status = 'ok'
              AND u.replicate = 0 AND i.published_at < TIMESTAMP '2026-07-01' {qualifying}
            """,
            [args.source],
        )
    rng = random.Random(args.seed)
    picked = []
    for t, n in counts:
        rows = pairs.filter(pl.col("event_type") == t).sort("item_id", "ticker").to_dicts()
        picked += rng.sample(rows, min(n, len(rows)))
    rng.shuffle(picked)
    items, key = [], []
    for k, r in enumerate(picked, 1):
        sid = f"{PREFIX[args.source]}{k:03d}"
        ticker = r["ticker"]
        item = {
            "id": sid,
            "company": names.get(t2c.get(ticker, -1), ticker),
            "ticker": ticker,
            "publisher": r["publisher"],
            "published": r["published_at"].strftime("%Y-%m-%d"),
            "title": r["title"],
            "summary": r["body_text"],
        }
        if args.source == "edgar":
            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
            item |= {
                "doc_type": doc_type_for(args.source, meta),
                "form": meta.get("form", "8-K"),
                "items": meta.get("items", []),
                "truncated": bool(r["body_truncated"]),
            }
        items.append(item)
        key.append({"id": sid, **{c: r[c] for c in KEY_COLS}})
    out = Path(cfg.paths.data_dir) / "labeling"
    out.mkdir(parents=True, exist_ok=True)
    doc = {"version": args.name, "seed": args.seed, "per_type": args.per_type,
           "items": items}  # fmt: skip
    if args.counts or args.source != "news_api" or args.qualifying:
        doc |= {"source": args.source, "counts": dict(counts), "qualifying": args.qualifying}
    path = out / f"{args.name}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    pl.DataFrame(key).write_csv(out / f"{args.name}_key.csv")
    print(f"{len(items)} items -> {out}; json {len(path.read_bytes())} bytes")


if __name__ == "__main__":
    main()
