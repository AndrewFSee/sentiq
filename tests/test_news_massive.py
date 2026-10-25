"""Massive (ex-Polygon) news vendor: mapping, pagination, auth, dropped vendor sentiment.

Unit tests use synthetic pages in the documented shape; ``tests/fixtures/news/massive_page.json``
is a real recorded page (2026-07-01, text fields redacted, ``scripts/record_massive_fixture.py``)
that pins the live response layout."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from newsfeed.ingest.http import HttpClient, RateLimiter
from newsfeed.ingest.news_api import (
    MassiveVendor,
    NewsApiAdapter,
    article_from_massive,
)
from newsfeed.tickers import Universe


@pytest.fixture
def universe(ticker_map) -> Universe:
    return Universe.build(["AAPL", "MSFT"], ticker_map)


def rec(i: int, ts: str, tickers: list[str], **kw) -> dict:
    return {
        "id": f"m{i}",
        "published_utc": ts,
        "title": f"Headline {i}",
        "description": f"Summary {i}.",
        "article_url": f"https://news.example/{i}",
        "publisher": {"name": "Example Wire", "homepage_url": "https://news.example"},
        "tickers": tickers,
        "insights": [
            {"ticker": t, "sentiment": "positive", "sentiment_reasoning": "x"} for t in tickers
        ],
        **kw,
    }


PAGE1 = {
    "status": "OK",
    "results": [rec(1, "2026-07-01T13:00:00Z", ["AAPL"]), rec(2, "2026-07-01T14:00:00Z", ["ZZZZ"])],
    "next_url": "https://api.massive.test/v2/reference/news?cursor=abc",
}
PAGE2 = {"status": "OK", "results": [rec(3, "2026-07-01T15:00:00Z", ["MSFT", "AAPL"])]}


def vendor(calls: list[httpx.Request]) -> MassiveVendor:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json=PAGE2 if "cursor=" in str(req.url) else PAGE1)

    http = HttpClient(
        user_agent="t",
        rate_per_sec=100,
        limiter=RateLimiter(100, clock=lambda: 0.0, sleep=lambda s: None),
        headers={"Authorization": "Bearer KEY"},
        transport=httpx.MockTransport(handler),
    )
    return MassiveVendor(http, "https://api.massive.test")


def test_mapping_drops_vendor_insights_and_keeps_utc():
    a = article_from_massive(rec(1, "2026-07-01T13:00:00Z", ["aapl"]))
    assert a.published_at == datetime(2026, 7, 1, 13, tzinfo=UTC)
    assert a.tickers == ["AAPL"] and a.summary == "Summary 1." and a.body is None
    assert a.publisher == "Example Wire"
    assert "insights" not in a.raw and a.raw["had_insights"] is True
    assert "sentiment" not in json.dumps(a.raw)


def test_fetch_query_pagination_and_auth():
    calls: list[httpx.Request] = []
    v = vendor(calls)
    since, until = datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)
    p1 = v.fetch(since=since, until=until, cursor=None, limit=5000)
    q = calls[0].url.params
    assert q["published_utc.gte"] == "2026-07-01T00:00:00Z"
    assert q["published_utc.lt"] == "2026-07-02T00:00:00Z"
    assert (q["order"], q["sort"], q["limit"]) == ("asc", "published_utc", "1000")
    assert calls[0].headers["Authorization"] == "Bearer KEY"
    assert "KEY" not in str(calls[0].url) and "apiKey" not in str(calls[0].url)
    assert [a.vendor_id for a in p1.articles] == ["m1", "m2"] and p1.next_cursor
    p2 = v.fetch(since=since, until=until, cursor=p1.next_cursor, limit=1000)
    assert str(calls[1].url) == PAGE1["next_url"] and p2.next_cursor is None


def test_adapter_backfill_filters_to_universe(cfg, universe):
    calls: list[httpx.Request] = []
    ad = NewsApiAdapter(cfg.sources.news_api, vendor(calls), universe)
    items = list(
        ad.backfill(
            datetime(2026, 7, 1, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
            skip=lambda _id: False,
        )
    )
    assert [i.native_id for i in items] == ["massive:m1", "massive:m3"]  # ZZZZ not in universe
    assert items[1].tickers == ["AAPL", "MSFT"] and items[0].body_text == "Summary 1."
    assert items[0].meta["has_full_body"] is False


def test_recorded_page_maps_cleanly():
    from pathlib import Path

    page = json.loads(
        (Path(__file__).parent / "fixtures" / "news" / "massive_page.json").read_text("utf-8")
    )
    assert page["results"] and all(r.get("insights") for r in page["results"][:1])
    arts = [article_from_massive(r) for r in page["results"]]
    assert all(a.published_at.tzinfo is not None for a in arts)
    assert all(a.tickers and a.publisher for a in arts)
    assert all("insights" not in a.raw and "sentiment" not in json.dumps(a.raw) for a in arts)
    assert [a.published_at for a in arts] == sorted(a.published_at for a in arts)
