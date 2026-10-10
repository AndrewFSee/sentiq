from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from newsfeed.ingest.base import make_item_id
from newsfeed.ingest.news_api import FixtureVendor, NewsApiAdapter, article_from_generic
from newsfeed.tickers import Universe

from .conftest import FIXTURES, FakeClock

ARTICLES = FIXTURES / "news" / "articles.jsonl"


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


@pytest.fixture
def universe(ticker_map) -> Universe:
    # trimmed fixture map has AAPL/MSFT/GOOGL/GOOG/NVDA; add the rest with fake CIKs
    base = Universe.build(["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA"], ticker_map)
    extra = {101: ["AMZN"], 102: ["META"], 103: ["JPM"], 104: ["WMT"], 105: ["JNJ"], 106: ["XOM"]}
    return Universe({**base.cik_to_tickers, **extra})


def make(cfg, universe, now: datetime, **kw):
    clock = FakeClock(now)
    vendor = FixtureVendor(ARTICLES, clock=clock)
    return NewsApiAdapter(cfg.sources.news_api, vendor, universe, clock=clock, **kw), vendor, clock


def test_naive_timestamp_rejected_by_mapping():
    with pytest.raises(ValueError, match="timezone-aware"):
        article_from_generic({"id": "x", "published_at": "2026-07-30T15:00:00"})


def test_backfill_window_filtering_and_errors(cfg, universe):
    a, _, _ = make(cfg, universe, utc(2026, 10, 8))
    errors: list[str] = []
    items = list(
        a.backfill(
            utc(2026, 7, 27),
            utc(2026, 8, 3),
            skip=lambda _: False,
            on_error=lambda nid, e: errors.append(nid),
        )
    )
    ids = {i.native_id for i in items}
    assert errors == ["fixture:a9"]  # naive timestamp -> reported, not guessed
    assert "fixture:a8" not in ids  # outside universe
    assert {"fixture:a1", "fixture:a6", "fixture:a10", "fixture:a14"} <= ids
    assert all(i.ingest_mode == "backfill" for i in items)
    a7 = next(i for i in items if i.native_id == "fixture:a7")
    assert a7.tickers == ["NVDA"]  # TSM dropped (not in universe)
    assert a7.meta["vendor_tickers"] == ["NVDA", "TSM"]
    a14 = next(i for i in items if i.native_id == "fixture:a14")
    assert a14.body_text == "" and a14.title  # headline-only survives


def test_backfill_respects_end_exclusive(cfg, universe):
    a, _, _ = make(cfg, universe, utc(2026, 10, 8))
    items = list(
        a.backfill(
            utc(2026, 7, 30, 20, 35),
            utc(2026, 7, 30, 20, 41),
            skip=lambda _: False,
            on_error=lambda *_: None,
        )
    )
    assert [i.native_id for i in items] == ["fixture:a1"]


def test_poll_first_seen_is_response_time_and_hides_future(cfg, universe):
    now = utc(2026, 7, 30, 20, 50)
    a, _, _ = make(cfg, universe, now)
    items = list(a.poll(skip=lambda _: False, on_error=lambda *_: None))
    ids = [i.native_id for i in items]
    assert "fixture:a1" in ids and "fixture:a2" in ids
    assert "fixture:a3" not in ids  # published 21:10, not yet in existence
    assert all(i.first_seen_at == now and i.ingest_mode == "live" for i in items)
    assert a.last_activity_at == utc(2026, 7, 30, 20, 41)


def test_late_indexed_article_gets_late_first_seen(cfg, universe):
    a, _, clock = make(cfg, universe, utc(2026, 7, 31, 16, 0))
    first = {i.item_id for i in a.poll(skip=lambda _: False, on_error=lambda *_: None)}
    assert make_item_id("news_api", "fixture:a11") not in first  # not indexed until 17:00
    assert a._watermark == utc(2026, 7, 31, 14, 5)  # newest seen: a5
    clock.now = utc(2026, 7, 31, 17, 5)
    second = list(a.poll(skip=lambda iid: iid in first, on_error=lambda *_: None))
    assert [i.native_id for i in second] == ["fixture:a11"]  # since = 14:05 - 15m overlap
    assert second[0].first_seen_at == utc(2026, 7, 31, 17, 5)  # honest, late
    assert second[0].published_at == utc(2026, 7, 31, 15, 0)


def test_backdated_beyond_overlap_needs_catch_up(cfg, universe):
    a, _, clock = make(cfg, universe, utc(2026, 7, 31, 16, 0))
    seen = {i.item_id for i in a.poll(skip=lambda _: False, on_error=lambda *_: None)}
    a._watermark = utc(2026, 7, 31, 16, 0)  # e.g. a newer article moved the watermark
    clock.now = utc(2026, 7, 31, 17, 5)
    plain = [i.native_id for i in a.poll(skip=lambda n: n in seen, on_error=lambda *_: None)]
    assert "fixture:a11" not in plain  # 15:00 < 16:00 - 15m
    caught = [
        i.native_id
        for i in a.poll(skip=lambda n: n in seen, on_error=lambda *_: None, catch_up=True)
    ]
    assert "fixture:a11" in caught


def test_poll_resumes_from_store_watermark(cfg, universe):
    a, _, _ = make(
        cfg, universe, utc(2026, 7, 31, 14, 10), watermark=lambda: utc(2026, 7, 31, 14, 0)
    )
    items = list(a.poll(skip=lambda _: False, on_error=lambda *_: None))
    # since = 14:00 - 15min overlap -> only a5 (14:05)
    assert [i.native_id for i in items] == ["fixture:a5"]


def test_poll_pagination(cfg, universe):
    small = cfg.sources.news_api.model_copy(update={"page_size": 2})
    clock = FakeClock(utc(2026, 10, 8))
    vendor = FixtureVendor(ARTICLES, clock=clock)
    a = NewsApiAdapter(small, vendor, universe, clock=clock)
    items = list(
        a.backfill(utc(2026, 7, 1), utc(2026, 9, 1), skip=lambda _: False, on_error=lambda *_: None)
    )
    assert len(items) == 13  # 15 records - naive (a9) - non-universe (a8)
    assert vendor.requests == 7  # 14 parseable records / 2 per page


def test_filter_to_universe_can_be_disabled(cfg, universe):
    ncfg = cfg.sources.news_api.model_copy(update={"filter_to_universe": False})
    clock = FakeClock(utc(2026, 10, 8))
    a = NewsApiAdapter(ncfg, FixtureVendor(ARTICLES, clock=clock), universe, clock=clock)
    ids = {
        i.native_id
        for i in a.backfill(
            utc(2026, 7, 1), utc(2026, 9, 1), skip=lambda _: False, on_error=lambda *_: None
        )
    }
    assert "fixture:a8" in ids


def test_poll_lookback_caps_first_poll(cfg, universe):
    a, _, _ = make(cfg, universe, utc(2026, 8, 1, 14, 0))
    items = list(a.poll(skip=lambda _: False, on_error=lambda *_: None))
    assert all(i.published_at >= utc(2026, 8, 1, 14, 0) - timedelta(hours=24) for i in items)


def test_first_seen_is_stamped_after_the_vendor_response(cfg, universe):
    clock = FakeClock(utc(2026, 7, 30, 20, 50))

    class SlowVendor(FixtureVendor):
        def fetch(self, **kw):
            page = super().fetch(**kw)
            clock.now += timedelta(seconds=30)  # response arrives 30s after the request
            return page

    a = NewsApiAdapter(
        cfg.sources.news_api, SlowVendor(ARTICLES, clock=clock), universe, clock=clock
    )
    items = list(a.poll(skip=lambda _: False, on_error=lambda *_: None))
    assert items and all(i.first_seen_at == utc(2026, 7, 30, 20, 50, 30) for i in items)


def test_fixture_end_to_end_stories(cfg, universe, store, calendar):
    """Ingest the fixture corpus, derive, and check story grouping + as-of outlet counts."""
    from newsfeed.dedupe import rebuild_stories, stories_as_of
    from newsfeed.derive import relink
    from newsfeed.ingest.runner import ingest, new_stats
    from newsfeed.tickers import TickerLinker

    a, _, _ = make(cfg, universe, utc(2026, 10, 8))
    items = a.backfill(
        utc(2026, 7, 1), utc(2026, 9, 1), skip=store.has_item, on_error=lambda *_: None
    )
    ingest(
        items,
        store=store,
        cfg=cfg,
        calendar=calendar,
        run_id="r",
        flush_every=100,
        stats=new_stats(),
    )
    tick = {t for ts in universe.cik_to_tickers.values() for t in ts}
    linker = TickerLinker(universe, {k: v for k, v in cfg.universe.aliases.items() if k in tick})
    relink(store, linker, "r")
    rebuild_stories(store, cfg.dedupe, linker.linker_id)

    df = store.query(
        "SELECT i.native_id, s.reason, f.native_id AS founder FROM item_story s "
        "JOIN items i USING (item_id) JOIN items f ON f.item_id = s.story_id"
    )
    got = {r["native_id"]: (r["reason"], r["founder"]) for r in df.to_dicts()}
    assert got["fixture:a1"] == ("new", "fixture:a1")
    assert got["fixture:a2"] == ("exact", "fixture:a1")  # syndicated verbatim, new headline
    assert got["fixture:a3"] == ("near", "fixture:a1")  # light rewrite
    assert got["fixture:a4"] == ("url", "fixture:a1")  # same URL, updated text
    assert got["fixture:a5"][0] == "new"  # different Apple story
    assert got["fixture:a14"][0] == got["fixture:a15"][0] == "new"  # headline-only: no collision

    # all four Apple copies are after the close -> tradable at Friday's open (backfill lag)
    friday_open = utc(2026, 7, 31, 13, 30)
    assert stories_as_of(store, friday_open - timedelta(seconds=1), tickers=["AAPL"]).height == 0
    (st,) = stories_as_of(store, friday_open, tickers=["AAPL"]).to_dicts()
    assert (st["n_items"], st["n_outlets"]) == (4, 3)


def test_dedupe_text_ignores_title_only_when_body_is_substantial():
    from newsfeed.dedupe import dedupe_text

    assert dedupe_text("Headline", "x" * 250) == "x" * 250
    assert dedupe_text("Headline", "") == "Headline\n"
    assert dedupe_text("A", "") != dedupe_text("B", "")
