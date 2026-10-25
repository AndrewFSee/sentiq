"""Shared fixtures. Network is disabled for all tests (pytest-socket, see pyproject addopts)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from newsfeed.config import AppConfig, load_config
from newsfeed.ingest.edgar import EdgarAdapter
from newsfeed.ingest.http import HttpClient, RateLimiter
from newsfeed.storage import Store
from newsfeed.tickers import TickerMap, Universe
from newsfeed.timeutil import MarketCalendar, get_calendar

FIXTURES = Path(__file__).parent / "fixtures"
EDGAR_FIXTURES = FIXTURES / "edgar"
DEV_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "BRK-B",
]
EMPTY_FEED = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class FixtureTransport(httpx.MockTransport):
    """Serves tests/fixtures/edgar/<host>/<path>; 404 otherwise. Records every request."""

    def __init__(self, root: Path = EDGAR_FIXTURES, overrides: dict[str, Callable] | None = None):
        self.root = root
        self.requests: list[httpx.Request] = []
        self.overrides = overrides or {}
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url in self.overrides:
            return self.overrides[url](request)
        if request.url.path == "/cgi-bin/browse-edgar":  # default: empty "latest filings" feed
            return httpx.Response(200, content=EMPTY_FEED)
        path = self.root / request.url.host / request.url.path.lstrip("/")
        if path.is_file():
            return httpx.Response(200, content=path.read_bytes())
        return httpx.Response(404, text=f"no fixture for {url}")


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    c = load_config(root=tmp_path)  # all data paths land in tmp
    # hermetic: a 34-word excerpt of the LM Master Dictionary instead of the 9 MB download
    # embeddings off by default (no model download in tests); tests opt in with a fake embedder
    feats = c.features.model_copy(
        update={"lm_dictionary": FIXTURES / "lm" / "mini_lm.csv", "embeddings": False}
    )
    # hermetic: tests use the original static dev universe regardless of default.yaml's mode
    uni = c.universe.model_copy(
        update={
            "mode": "static",
            "tickers": DEV_TICKERS,
            "name_linking": "sec_names",
            "extra_members": [],  # default.yaml's Nasdaq-100 extras stay out of tests
        }
    )
    return c.model_copy(update={"features": feats, "universe": uni})


@pytest.fixture
def calendar() -> MarketCalendar:
    return get_calendar("XNYS", "2000-01-01")


@pytest.fixture
def store(cfg: AppConfig):
    s = Store(cfg.paths)
    yield s
    s.close()


@pytest.fixture
def transport() -> FixtureTransport:
    return FixtureTransport()


def make_http(transport: httpx.BaseTransport) -> HttpClient:
    return HttpClient(
        user_agent="test-suite test@example.com",
        rate_per_sec=1000,
        transport=transport,
        limiter=RateLimiter(1000, clock=lambda: 0.0, sleep=lambda s: None),
        max_retries=3,
        retry_wait_initial=0,
        retry_wait_max=0,
    )


@pytest.fixture
def http(transport: FixtureTransport):
    c = make_http(transport)
    yield c
    c.close()


@pytest.fixture
def ticker_map() -> TickerMap:
    payload = json.loads(
        (EDGAR_FIXTURES / "www.sec.gov" / "files" / "company_tickers.json").read_text()
    )
    return TickerMap.from_sec_json(payload)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 8, 18, 0, tzinfo=UTC))


@pytest.fixture
def edgar(cfg, http, ticker_map, store, clock) -> EdgarAdapter:
    universe = Universe.build(["AAPL"], ticker_map)
    return EdgarAdapter(cfg.sources.edgar, http, universe, store.raw, clock=clock)
