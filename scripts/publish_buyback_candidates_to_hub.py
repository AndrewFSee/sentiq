"""Publish candidate share-buyback announcements from stored 8-Ks into the Dashboard Hub.

Read-only over this project's store: items come through ``Store.items_as_of(now)`` (source
``edgar``) and the full text of each filing's primary document and EX-99 exhibits from the raw
store, so nothing is fetched from SEC and nothing here is written back. ``body_text`` is not
used because it is truncated for scoring; a buyback line can sit deep in an earnings release.

A filing is a candidate when a passage mentions a repurchase / buyback of shares together with
an authorization, approval, program, increase, accelerated repurchase or tender offer. The hub
(market-compass ``modules/buybacks/news.py``) has Claude classify each candidate once; this
script only finds them, so it errs towards recall.

The hub's ``run_jobs.py`` runs this with this project's venv and the hub folder on
``PYTHONPATH``. Writes ``<HUB_DATA_DIR>/buyback_news/candidates`` (one row per filing) and
``summary``. Incremental: filings already published are kept; ``--rebuild`` rescans all.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from newsfeed.config import load_config  # noqa: E402
from newsfeed.storage import Store  # noqa: E402
from newsfeed.textutil import html_to_text  # noqa: E402

try:
    from core import store as hub  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "buyback_news"
SINCE = datetime(2019, 1, 1, tzinfo=UTC)
OVERLAP = timedelta(days=3)          # rescan this far back on incremental runs (late items)
WINDOW = 600                         # characters of context on each side of a match
MAX_PASSAGES, MAX_CHARS = 4, 5000

BUYBACK = re.compile(r"\b(?:re-?purchas\w*|buy-?backs?)\b", re.I)
SHARES = re.compile(r"\b(?:shares?|stock|equity|common)\b", re.I)
CONTEXT = re.compile(r"\b(?:authori[sz]\w*|approv\w*|program\w*|plan|increas\w*|additional|new|accelerated|tender|"
                     r"replace\w*|expan\w*)\b", re.I)
REPO = re.compile(r"(?:reverse\s+)?repurchase\s+agreements?|securities\s+sold\s+under", re.I)


def passages(text: str) -> list[str]:
    """Context windows around share-buyback mentions that also mention an authorization etc."""
    spans: list[tuple[int, int]] = []
    for m in BUYBACK.finditer(text):
        near = text[max(0, m.start() - 250): m.end() + 250]
        if REPO.search(text[max(0, m.start() - 30): m.end() + 30]):
            continue
        if not (SHARES.search(near) and CONTEXT.search(near)):
            continue
        a, b = max(0, m.start() - WINDOW), min(len(text), m.end() + WINDOW)
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    out, total = [], 0
    for a, b in spans[:MAX_PASSAGES]:
        chunk = " ".join(text[a:b].split())
        out.append(chunk[: MAX_CHARS - total])
        total += len(out[-1])
        if total >= MAX_CHARS:
            break
    return out


def filing_text(raw, accession: str, meta: dict) -> str:
    parts = []
    names = [meta.get("primary_document")] + [e.get("file") for e in meta.get("exhibits") or []]
    for name in [n for n in names if n]:
        data = raw.get("edgar", accession, name)
        if data is not None:
            parts.append(html_to_text(data))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script reads the local store only")
    parser.add_argument("--rebuild", action="store_true", help="rescan every stored 8-K since 2019")
    args = parser.parse_args()

    import json

    store = Store(load_config().paths)
    old = None if args.rebuild else hub.load_frame(MODULE, "candidates")
    start = SINCE
    if old is not None and not old.empty:
        start = max(SINCE, pd.Timestamp(old["tradable_at"].max()).to_pydatetime() - OVERLAP)
    now = datetime.now(UTC)
    items = store.items_as_of(now, source="edgar", columns="item_id, native_id, tickers, tradable_at, published_at, url, meta")
    import polars as pl
    items = items.filter(pl.col("tradable_at") >= start).to_pandas()
    rows, scanned, missing = [], 0, 0
    for r in items.itertuples():
        meta = json.loads(r.meta) if isinstance(r.meta, str) else (r.meta or {})
        if meta.get("form", "8-K") != "8-K":
            continue
        scanned += 1
        text = filing_text(store.raw, r.native_id, meta)
        if not text:
            missing += 1
            continue
        found = passages(text)
        if found:
            rows.append({"accession": r.native_id, "cik": int(meta.get("cik") or 0), "company": meta.get("company"),
                         "tickers": ",".join(list(r.tickers) if r.tickers is not None else []), "items": ",".join(meta.get("items") or []),
                         "tradable_at": pd.Timestamp(r.tradable_at).tz_convert("UTC"),
                         "published_at": pd.Timestamp(r.published_at).tz_convert("UTC"),
                         "url": r.url, "passages": "\n---\n".join(found)})
    new = pd.DataFrame(rows)
    out = new if old is None or old.empty else pd.concat([old[~old["accession"].isin(new.get("accession", []))], new])
    out = out.sort_values("tradable_at").drop_duplicates("accession", keep="last").reset_index(drop=True)
    hub.save_frame(MODULE, "candidates", out)
    hub.save_json(MODULE, "summary", {
        "as_of": now.isoformat(timespec="seconds"), "candidates": int(len(out)),
        "latest_filing": str(out["tradable_at"].max()) if len(out) else None,
        "scanned_this_run": scanned, "raw_missing_this_run": missing, "since": SINCE.date().isoformat(),
    })
    print(f"Scanned {scanned:,} 8-Ks since {start:%Y-%m-%d} ({missing} without raw text); {len(new):,} candidates "
          f"this run, {len(out):,} in total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
