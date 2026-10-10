"""Derived tables, rebuilt incrementally after ingestion: ticker links, then stories.

Both are deterministic functions of the stored items (+ config), so they can always be
rebuilt from scratch; incremental runs only touch what changed. A lock file serialises
rebuilds across processes (``collect`` loop and a concurrent ``backfill``).
"""

from __future__ import annotations

import logging
from typing import Any

from newsfeed.config import AppConfig
from newsfeed.dedupe import rebuild_stories
from newsfeed.storage import FileLock, Store
from newsfeed.tickers import TickerLinker
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)


def relink(store: Store, linker: TickerLinker, run_id: str) -> int:
    """Link every item without links from the current ``linker_id`` *for its current text*
    (a re-parsed item gets relinked). Returns items linked."""
    pending = store.query(
        """
        SELECT i.item_id, i.title, i.body_text, i.tickers, i.text_hash
        FROM items i
        WHERE NOT EXISTS (
            SELECT 1 FROM item_tickers_all t
            WHERE t.item_id = i.item_id AND t.linker_id = $lid AND t.text_hash = i.text_hash
        )
        """,
        {"lid": linker.linker_id},
    ).to_dicts()
    if not pending:
        return 0
    now = utcnow()
    rows: list[dict[str, Any]] = []
    for r in pending:
        links = linker.link(title=r["title"], body=r["body_text"], source_tickers=r["tickers"])
        base = {
            "item_id": r["item_id"],
            "linker_id": linker.linker_id,
            "text_hash": r["text_hash"],
            "linked_at": now,
        }
        if not links:  # marker so the item is not re-linked every cycle
            rows.append(
                {
                    **base,
                    "ticker": None,
                    "cik": None,
                    "method": "none",
                    "confidence": 0.0,
                    "n_mentions": 0,
                    "in_title": False,
                }
            )
        for lk in links:
            rows.append(
                {
                    **base,
                    "ticker": lk.ticker,
                    "cik": lk.cik,
                    "method": lk.method,
                    "confidence": lk.confidence,
                    "n_mentions": lk.n_mentions,
                    "in_title": lk.in_title,
                }
            )
    store.write_ticker_links(rows, run_id)
    return len(pending)


def derive(
    store: Store,
    cfg: AppConfig,
    linker: TickerLinker,
    run_id: str,
    *,
    full_stories: bool = False,
    wait: bool = False,
) -> dict[str, Any]:
    """Relink + rebuild stories. If another process holds the lock, skip (or raise if
    ``wait``) -- the next cycle will catch up, since both steps are self-healing."""
    lock = FileLock(store.state_dir / "derive.lock")
    if not lock.acquire():
        if wait:
            raise RuntimeError(f"derive lock held: {lock.path}")
        log.info("derive skipped: lock held by another process")
        return {"skipped": True}
    try:
        linked = relink(store, linker, run_id)
        stories = rebuild_stories(store, cfg.dedupe, linker.linker_id, full=full_stories)
        return {"linked": linked, **stories}
    finally:
        lock.release()
