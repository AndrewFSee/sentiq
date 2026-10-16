"""Live polling via EDGAR's 'latest filings' Atom feed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from newsfeed.ingest.edgar import EdgarAdapter, parse_current_feed
from newsfeed.ingest.http import RetryableHTTPError
from newsfeed.tickers import Universe

from .conftest import EDGAR_FIXTURES, FixtureTransport, make_http

JUL26 = "0000320193-26-000018"


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def entry(acc: str, cik: int, updated: str, form: str = "8-K", name: str = "Some Co") -> str:
    href = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm"
    return f"""<entry>
<title>{form} - {name} ({cik:010d}) (Filer)</title>
<link rel="alternate" type="text/html" href="{href}"/>
<summary type="html">x</summary>
<updated>{updated}</updated>
<category scheme="https://www.sec.gov/" label="form type" term="{form}"/>
<id>urn:tag:sec.gov,2008:accession-number={acc}</id>
</entry>"""


def feed(*entries: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="ISO-8859-1" ?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom"><title>Latest Filings</title>'
        "<updated>2026-07-30T22:00:00-04:00</updated>" + "".join(entries) + "</feed>"
    ).encode("latin-1")


class FeedTransport(FixtureTransport):
    """Fixture files + the getcurrent feed served page by page (keyed by ?start=)."""

    def __init__(self, pages: dict[int, bytes]):
        super().__init__()
        self.pages = pages

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/cgi-bin/browse-edgar":
            self.requests.append(request)
            start = int(request.url.params["start"])
            return httpx.Response(200, content=self.pages.get(start, feed()))
        return super()._handle(request)


def adapter(cfg, transport, ticker_map, store, clock, **cfg_updates) -> EdgarAdapter:
    ecfg = cfg.sources.edgar.model_copy(update=cfg_updates)
    return EdgarAdapter(
        ecfg, make_http(transport), Universe.build(["AAPL"], ticker_map), store.raw, clock=clock
    )


def test_parse_real_feed():
    entries = parse_current_feed((EDGAR_FIXTURES / "getcurrent_8k_20261008.xml").read_bytes())
    assert len(entries) == 100
    first = entries[0]
    assert first.accession == "0001193125-26-418097"
    assert first.cik == 1786108
    assert first.form == "8-K"
    # explicit offset preserved: 17:30:13 EDT == 21:30:13 UTC (matches the SGML header)
    assert first.updated == utc(2026, 10, 8, 21, 30, 13)
    assert all(e.updated.tzinfo is not None for e in entries)
    assert entries == sorted(entries, key=lambda e: e.updated, reverse=True)  # newest first


def test_feed_poll_fetches_only_universe_filings(cfg, ticker_map, store, clock):
    clock.now = utc(2026, 7, 30, 21, 0)
    page = feed(
        entry("0009999999-26-000001", 9999999, "2026-07-30T16:45:00-04:00"),  # other company
        entry(JUL26, 320193, "2026-07-30T16:30:28-04:00", name="Apple Inc."),
        entry("0009999999-26-000002", 9999999, "2026-07-30T16:10:00-04:00", form="10-Q"),
    )
    t = FeedTransport({0: page})
    a = adapter(cfg, t, ticker_map, store, clock)
    items = list(a.poll(skip=store.has_item))

    assert [i.native_id for i in items] == [JUL26]
    item = items[0]
    assert item.ingest_mode == "live"
    assert item.first_seen_at == clock.now  # stamped when the feed page arrived
    assert item.published_at == utc(2026, 7, 30, 20, 30, 28)  # from the header
    assert item.meta["json_convention"] == "feed"
    assert item.meta["filing_date"] == "2026-07-30"  # backfilled from the SGML header
    assert item.publisher == "SEC EDGAR"
    # activity = newest entry for ANY company/form, not just the universe
    assert a.last_activity_at == utc(2026, 7, 30, 20, 45)
    # only one feed request + the Apple filing documents; no per-CIK submissions sweep
    assert not any("data.sec.gov" in str(r.url) for r in t.requests)


def test_feed_poll_pages_until_previous_high_water(cfg, ticker_map, store, clock):
    clock.now = utc(2026, 7, 30, 21, 0)
    e = [
        entry(f"0009999999-26-{n:06d}", 9999999, f"2026-07-30T16:{59 - n:02d}:00-04:00")
        for n in range(6)
    ]
    t = FeedTransport({0: feed(e[0], e[1]), 2: feed(e[2], e[3]), 4: feed(e[4], e[5])})
    a = adapter(cfg, t, ticker_map, store, clock, current_feed_page_size=2)

    list(a.poll(skip=store.has_item))  # first poll: no high water -> reads all pages
    assert len(t.requests) == 4  # 3 full pages + empty page 6
    assert a.last_activity_at == utc(2026, 7, 30, 20, 59)
    assert a.last_poll_reach == utc(2026, 7, 30, 20, 54)  # oldest entry paged to (16:54 EDT)

    t.requests.clear()
    t.pages = {0: feed(entry("0009999999-26-000100", 9999999, "2026-07-30T17:05:00-04:00"), e[0])}
    list(a.poll(skip=store.has_item))  # overlaps previous newest on page 0 -> stops
    assert len(t.requests) == 1
    assert a.last_activity_at == utc(2026, 7, 30, 21, 5)
    assert a.last_poll_reach == utc(2026, 7, 30, 20, 59)  # reset per poll, not kept from before


def test_feed_poll_ignores_entries_older_than_lookback(cfg, ticker_map, store, clock):
    clock.now = utc(2026, 8, 5, 21, 0)  # earnings 8-K is 6 days old
    t = FeedTransport({0: feed(entry(JUL26, 320193, "2026-07-30T16:30:28-04:00"))})
    a = adapter(cfg, t, ticker_map, store, clock)
    assert list(a.poll(skip=store.has_item)) == []


def test_feed_poll_skips_stored(cfg, ticker_map, store, clock):
    clock.now = utc(2026, 7, 30, 21, 0)
    t = FeedTransport({0: feed(entry(JUL26, 320193, "2026-07-30T16:30:28-04:00"))})
    a = adapter(cfg, t, ticker_map, store, clock)
    assert list(a.poll(skip=lambda _id: True)) == []
    assert not any("/Archives/" in str(r.url) for r in t.requests)


def test_feed_failure_leaves_activity_untouched(cfg, ticker_map, store, clock):
    class Down(FixtureTransport):
        def _handle(self, request):
            return httpx.Response(503)

    a = adapter(cfg, Down(), ticker_map, store, clock)
    with pytest.raises(RetryableHTTPError):
        list(a.poll(skip=store.has_item))
    assert a.last_activity_at is None


def test_catch_up_runs_feed_then_sweep_without_refetching(cfg, ticker_map, store, clock):
    clock.now = utc(2026, 7, 30, 21, 0)
    t = FeedTransport({0: feed(entry(JUL26, 320193, "2026-07-30T16:30:28-04:00"))})
    a = adapter(cfg, t, ticker_map, store, clock)
    items = list(a.poll(skip=store.has_item, on_error=lambda *_: None, catch_up=True))
    assert [i.native_id for i in items].count(JUL26) == 1  # feed found it; sweep skipped it
    assert any(r.url.path == "/cgi-bin/browse-edgar" for r in t.requests)
    assert any("data.sec.gov/submissions" in str(r.url) for r in t.requests)
    hdrs = [str(r.url).rsplit("/", 1)[-1] for r in t.requests if str(r.url).endswith(".hdr.sgml")]
    assert hdrs.count(f"{JUL26}.hdr.sgml") == 1, hdrs  # (fixture listing also has later filings)
    assert a.last_activity_at == utc(2026, 7, 30, 20, 30, 28)  # activity kept fresh


def test_first_seen_is_stamped_after_the_feed_response(cfg, ticker_map, store, clock):
    """The clock advances while the request is in flight; first_seen must be the later time."""
    clock.now = utc(2026, 7, 30, 21, 0)
    page = feed(entry(JUL26, 320193, "2026-07-30T16:30:28-04:00"))

    class Slow(FeedTransport):
        def _handle(self, request):
            if request.url.path == "/cgi-bin/browse-edgar":
                clock.now += timedelta(seconds=30)  # response arrives 30s later
            return super()._handle(request)

    a = adapter(cfg, Slow({0: page}), ticker_map, store, clock)
    (item,) = list(a.poll(skip=store.has_item))
    assert item.first_seen_at == utc(2026, 7, 30, 21, 0, 30)
