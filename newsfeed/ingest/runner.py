"""Drive an adapter iterator into storage: buffer, derive PIT columns, flush, record failures."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

from newsfeed.config import AppConfig
from newsfeed.dedupe import content_hash, dedupe_text, normalize_url
from newsfeed.ingest.base import RawItem
from newsfeed.storage import Store
from newsfeed.timeutil import MarketCalendar, compute_tradable_at, ensure_utc, utcnow

log = logging.getLogger(__name__)


def build_row(
    item: RawItem,
    *,
    run_id: str,
    calendar: MarketCalendar,
    backfill_lag: timedelta,
    ingested_at: datetime,
) -> dict[str, Any]:
    """Derive storage columns (item_id, tradable_at, hashes) from an adapter's RawItem.

    This is the only place tradable_at is computed for stored items.
    """
    tradable_at = compute_tradable_at(
        published_at=item.published_at,
        first_seen_at=item.first_seen_at,
        ingest_mode=item.ingest_mode,
        backfill_lag=backfill_lag,
        calendar=calendar,
    )
    rule = (
        "live:max(published,first_seen)+session"
        if item.ingest_mode == "live"
        else f"backfill:published+{int(backfill_lag.total_seconds())}s+session"
    )
    return {
        "item_id": item.item_id,
        "source": item.source,
        "native_id": item.native_id,
        "publisher": item.publisher,
        "url": item.url,
        "url_norm": normalize_url(item.url),
        "title": item.title,
        "body_text": item.body_text,
        "body_tokens_est": item.body_tokens_est,
        "body_truncated": item.body_truncated,
        "content_hash": content_hash(dedupe_text(item.title, item.body_text)),
        "tickers": list(item.tickers),
        "published_at": item.published_at,
        "published_at_raw": item.published_at_raw,
        "first_seen_at": item.first_seen_at,
        "tradable_at": tradable_at,
        "ingest_mode": item.ingest_mode,
        "tradable_rule": rule,
        "meta": json.dumps(item.meta, sort_keys=True, default=str),
        "raw_json": item.raw_json,
        "parser_version": item.parser_version,
        "run_id": run_id,
        "ingested_at": ensure_utc(ingested_at),
    }


def make_error_recorder(
    store: Store, run_id: str, source: str, stats: dict[str, Any]
) -> Callable[[str, Exception], None]:
    def on_error(native_id: str, exc: Exception) -> None:
        stats["failed"] += 1
        log.error("failed %s %s: %r", source, native_id, exc)
        store.record_failure(run_id, source, native_id, repr(exc))

    return on_error


def ingest(
    items: Iterable[RawItem],
    *,
    store: Store,
    cfg: AppConfig,
    calendar: MarketCalendar,
    run_id: str,
    flush_every: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Consume ``items`` and append them to the lake. Flushes every ``flush_every`` items so an
    interrupted run keeps its progress (the next run skips stored ids)."""
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        if buf:
            stats["written"] += store.write_items(buf, run_id)
            buf.clear()

    for item in items:
        lag = timedelta(seconds=cfg.timing.lag_for(item.source))
        buf.append(
            build_row(
                item, run_id=run_id, calendar=calendar, backfill_lag=lag, ingested_at=utcnow()
            )
        )
        stats["seen"] += 1
        if item.body_truncated:
            stats["truncated"] += 1
        if len(buf) >= flush_every:
            flush()
            log.info("progress: %s", stats)
    flush()
    return stats


def new_stats() -> dict[str, Any]:
    return {"seen": 0, "written": 0, "failed": 0, "truncated": 0}
