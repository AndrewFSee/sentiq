"""Re-parsing stored items: new text, identical point-in-time fields, derived data refreshed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from newsfeed.derive import relink
from newsfeed.ingest.edgar import PARSER_VERSION
from newsfeed.ingest.reparse import TIMING_FIELDS, reparse_source
from newsfeed.ingest.runner import build_row
from newsfeed.tickers import TickerLinker, Universe

JUL26 = "0000320193-26-000018"


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def stored_old_parse(edgar, store, calendar):
    """Store JUL26 as if an older parser had produced a broken body."""
    (item,) = list(edgar.backfill(utc(2026, 7, 1), utc(2026, 8, 1), skip=store.has_item))
    old = item.model_copy(update={"body_text": "OTHER EVENTS. 3", "parser_version": "edgar-8k/1"})
    row = build_row(
        old,
        run_id="old",
        calendar=calendar,
        backfill_lag=timedelta(seconds=60),
        ingested_at=utc(2026, 10, 8),
    )
    # sentinel timing that no recomputation could produce: proves fields are copied verbatim
    row["tradable_at"] = utc(2026, 8, 3, 13, 30)
    row["tradable_rule"] = "test:sentinel"
    store.write_items([row], "old")
    return row


def test_reparse_replaces_text_and_preserves_timing(edgar, store, calendar):
    old = stored_old_parse(edgar, store, calendar)
    before = store.query("SELECT * FROM items").to_dicts()[0]

    # a later config change (bigger backfill lag) must NOT move the re-parsed item in time
    stats = reparse_source(store, "edgar", edgar, PARSER_VERSION, calendar=calendar, run_id="re")
    assert stats == {"candidates": 1, "rewritten": 1, "text_changed": 1, "failed": []}

    after = store.query("SELECT * FROM items").to_dicts()
    assert len(after) == 1  # still one logical item
    new = after[0]
    assert new["parser_version"] == PARSER_VERSION
    assert "[ITEM 2.02" in new["body_text"]
    for k in TIMING_FIELDS:
        assert new[k] == before[k] == old[k], k
    assert new["text_hash"] != before["text_hash"]

    # idempotent: nothing left to re-parse
    again = reparse_source(store, "edgar", edgar, PARSER_VERSION, calendar=calendar, run_id="x")
    assert again["candidates"] == 0


def test_relink_follows_text_changes(edgar, store, calendar):
    stored_old_parse(edgar, store, calendar)
    linker = TickerLinker(Universe({320193: ["AAPL"]}), {"AAPL": ["Apple"]})
    assert relink(store, linker, "l1") == 1
    assert relink(store, linker, "l2") == 0  # nothing changed
    reparse_source(store, "edgar", edgar, PARSER_VERSION, calendar=calendar, run_id="re")
    assert relink(store, linker, "l3") == 1  # new text -> relinked
    links = store.query("SELECT text_hash FROM item_tickers")
    current = store.query("SELECT text_hash FROM items")["text_hash"][0]
    assert set(links["text_hash"]) == {current}
