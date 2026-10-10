"""Re-parse stored items with the current parser, preserving point-in-time fields exactly.

A parser fix must never move an item in time: the re-parsed row copies ``published_at``,
``published_at_raw``, ``first_seen_at``, ``ingest_mode``, ``tradable_at`` and ``tradable_rule``
verbatim from the stored row (``tradable_at`` is *not* recomputed, so later config changes such
as a different backfill lag cannot leak in). The acceptance time is re-derived from the same
header and must match, otherwise the item is reported and skipped.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Protocol

from newsfeed.ingest.base import RawItem
from newsfeed.ingest.runner import build_row
from newsfeed.storage import Store
from newsfeed.timeutil import MarketCalendar, utcnow

log = logging.getLogger(__name__)

TIMING_FIELDS = (
    "published_at",
    "published_at_raw",
    "first_seen_at",
    "ingest_mode",
    "tradable_at",
    "tradable_rule",
)


class Reparser(Protocol):
    def reparse(self, stored: dict[str, Any]) -> RawItem: ...


def reparse_source(
    store: Store,
    source: str,
    reparser: Reparser,
    parser_version: str,
    *,
    calendar: MarketCalendar,
    run_id: str,
    everything: bool = False,
) -> dict[str, Any]:
    where = "source = $s" if everything else "source = $s AND parser_version <> $v"
    stored = store.query(f"SELECT * FROM items WHERE {where}", {"s": source, "v": parser_version})
    stats: dict[str, Any] = {
        "candidates": stored.height,
        "rewritten": 0,
        "text_changed": 0,
        "failed": [],
    }
    rows = []
    for old in stored.to_dicts():
        try:
            item = reparser.reparse(old)
        except Exception as exc:
            log.error("reparse failed for %s: %r", old["native_id"], exc)
            stats["failed"].append(old["native_id"])
            continue
        if item.published_at != old["published_at"]:
            log.error(
                "reparse of %s moved published_at %s -> %s; skipped",
                old["native_id"],
                old["published_at"],
                item.published_at,
            )
            stats["failed"].append(old["native_id"])
            continue
        row = build_row(
            item, run_id=run_id, calendar=calendar, backfill_lag=timedelta(0), ingested_at=utcnow()
        )
        row.update({k: old[k] for k in TIMING_FIELDS})
        rows.append(row)
        if row["body_text"] != old["body_text"] or row["title"] != old["title"]:
            stats["text_changed"] += 1
    stats["rewritten"] = store.write_reparsed_items(rows, run_id)
    return stats
