"""News-API adapter, vendor-agnostic.

A vendor client implements :class:`NewsVendor` and maps its payload to :class:`VendorArticle`.
No vendor is licensed yet, so :class:`FixtureVendor` serves articles from a JSONL file of
vendor-shaped records; a real client (Polygon/Benzinga/...) is a ~50-line class plus fixtures.

PIT notes:
  * ``published_at`` must be tz-aware in the vendor payload. Naive timestamps are rejected
    (reported via ``on_error``) rather than guessed -- set the zone in the vendor mapping.
  * Live polls re-request ``poll_overlap_minutes`` before the newest timestamp already seen,
    because vendors index late and back-date ``published_at``. Late-indexed articles are caught
    and receive an honest (late) first_seen_at, so tradable_at = max(published, first_seen).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from newsfeed.config import NewsApiConfig
from newsfeed.ingest.base import ErrorFn, RawItem, SkipFn, make_item_id
from newsfeed.textutil import estimate_tokens, truncate_tokens
from newsfeed.tickers import Universe, normalize_ticker
from newsfeed.timeutil import IngestMode, ensure_utc, utcnow

log = logging.getLogger(__name__)

SOURCE = "news_api"
PARSER_VERSION = "news/1"


class VendorArticle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor_id: str
    published_at: datetime
    published_at_raw: str
    url: str | None = None
    title: str | None = None
    body: str | None = None
    summary: str | None = None
    publisher: str | None = None
    tickers: list[str] = []
    raw: dict[str, Any] = {}

    @field_validator("published_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return ensure_utc(v, name="published_at")

    @field_validator("tickers")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        return sorted({normalize_ticker(t) for t in v if t.strip()})


class VendorPage(BaseModel):
    articles: list[VendorArticle]
    next_cursor: str | None = None
    errors: list[tuple[str, str]] = []  # (vendor_id, message) for records that failed mapping


class NewsVendor(Protocol):
    name: str

    def fetch(
        self, *, since: datetime, until: datetime | None, cursor: str | None, limit: int
    ) -> VendorPage:
        """Articles with since <= published_at < until (until=None: up to now), oldest first."""
        ...


def article_from_generic(rec: dict[str, Any]) -> VendorArticle:
    """Map the generic fixture schema (documented in tests/fixtures/news/README.md)."""
    raw_ts = rec["published_at"]
    return VendorArticle(
        vendor_id=str(rec["id"]),
        published_at=datetime.fromisoformat(raw_ts),
        published_at_raw=raw_ts,
        url=rec.get("url"),
        title=rec.get("title"),
        body=rec.get("body"),
        summary=rec.get("summary"),
        publisher=rec.get("publisher"),
        tickers=rec.get("tickers") or [],
        raw=rec,
    )


class FixtureVendor:
    """Serves a JSONL file as if it were a vendor API. ``clock`` hides articles that would not
    yet exist (published after 'now'), and ``visible_at`` (optional per record) simulates
    vendor indexing delay: a record is only returned once clock() >= visible_at."""

    name = "fixture"

    def __init__(self, path: Path, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = path
        self.clock = clock
        self.requests = 0
        self._records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("//")
        ]

    def fetch(
        self, *, since: datetime, until: datetime | None, cursor: str | None, limit: int
    ) -> VendorPage:
        self.requests += 1
        now = self.clock()
        until = min(until, now) if until is not None else now
        good: list[VendorArticle] = []
        errors: list[tuple[str, str]] = []
        for rec in self._records:
            vis = rec.get("visible_at")
            if vis and datetime.fromisoformat(vis) > now:
                continue
            try:
                art = article_from_generic(rec)
            except Exception as exc:
                errors.append((str(rec.get("id")), repr(exc)))
                continue
            if since <= art.published_at < until:
                good.append(art)
        good.sort(key=lambda a: (a.published_at, a.vendor_id))
        start = int(cursor or 0)
        page = good[start : start + limit]
        nxt = str(start + limit) if start + limit < len(good) else None
        return VendorPage(articles=page, next_cursor=nxt, errors=errors if start == 0 else [])


def make_vendor(cfg: NewsApiConfig, clock: Callable[[], datetime] = utcnow) -> NewsVendor:
    if cfg.vendor == "fixture":
        return FixtureVendor(cfg.fixture_path, clock=clock)
    raise ValueError(f"unknown news vendor {cfg.vendor!r}")  # pragma: no cover


class NewsApiAdapter:
    name = SOURCE

    def __init__(
        self,
        cfg: NewsApiConfig,
        vendor: NewsVendor,
        universe: Universe,
        *,
        watermark: Callable[[], datetime | None] = lambda: None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """``watermark`` returns the newest published_at already stored for this source (lets a
        cron-launched process resume without local state)."""
        self.cfg = cfg
        self.vendor = vendor
        self.universe = universe
        self.clock = clock
        self._watermark_fn = watermark
        self._watermark: datetime | None = None
        self.last_activity_at: datetime | None = None
        self.universe_tickers = {t for ts in universe.cik_to_tickers.values() for t in ts}

    # ---- mapping
    def to_item(self, a: VendorArticle, *, first_seen_at: datetime, mode: IngestMode) -> RawItem:
        text = a.body or a.summary or ""
        body, truncated = truncate_tokens(text, self.cfg.max_tokens, self.cfg.chars_per_token)
        return RawItem(
            source=SOURCE,
            native_id=f"{self.vendor.name}:{a.vendor_id}",
            publisher=a.publisher,
            url=a.url,
            title=a.title,
            body_text=body,
            body_tokens_est=estimate_tokens(body, self.cfg.chars_per_token),
            body_truncated=truncated,
            tickers=[t for t in a.tickers if t in self.universe_tickers],
            published_at=a.published_at,
            published_at_raw=a.published_at_raw,
            first_seen_at=first_seen_at,
            ingest_mode=mode,
            meta={
                "vendor": self.vendor.name,
                "vendor_tickers": a.tickers,
                "has_full_body": bool(a.body),
            },
            raw_json=json.dumps(a.raw, sort_keys=True, default=str),
            parser_version=PARSER_VERSION,
        )

    def _wanted(self, a: VendorArticle) -> bool:
        return not self.cfg.filter_to_universe or bool(set(a.tickers) & self.universe_tickers)

    def _pages(
        self,
        since: datetime,
        until: datetime | None,
        skip: SkipFn,
        on_error: ErrorFn | None,
        mode: IngestMode,
    ) -> Iterator[RawItem]:
        cursor: str | None = None
        newest: datetime | None = None
        while True:
            page = self.vendor.fetch(
                since=since, until=until, cursor=cursor, limit=self.cfg.page_size
            )
            seen_at = self.clock()  # stamped after the response arrived
            for vid, msg in page.errors:
                exc = ValueError(msg)
                if on_error is None:
                    raise exc
                on_error(f"{self.vendor.name}:{vid}", exc)
            for a in page.articles:
                newest = a.published_at if newest is None else max(newest, a.published_at)
                if not self._wanted(a):
                    continue
                if skip(make_item_id(SOURCE, f"{self.vendor.name}:{a.vendor_id}")):
                    continue
                first_seen = seen_at if mode == "live" else self.clock()
                yield self.to_item(a, first_seen_at=first_seen, mode=mode)
            cursor = page.next_cursor
            if cursor is None:
                break
        if newest is not None:
            self.last_activity_at = (
                newest if self.last_activity_at is None else max(self.last_activity_at, newest)
            )
            if mode == "live":
                self._watermark = (
                    newest if self._watermark is None else max(self._watermark, newest)
                )

    # ---- adapter interface
    def backfill(
        self, start: datetime, end: datetime, skip: SkipFn, on_error: ErrorFn | None = None
    ) -> Iterator[RawItem]:
        start, end = ensure_utc(start, name="start"), ensure_utc(end, name="end")
        yield from self._pages(start, end, skip, on_error, "backfill")

    def poll(
        self, skip: SkipFn, on_error: ErrorFn | None = None, *, catch_up: bool = False
    ) -> Iterator[RawItem]:
        now = self.clock()
        lookback = timedelta(hours=self.cfg.poll_lookback_hours)
        if self._watermark is None:
            self._watermark = self._watermark_fn()
        if catch_up or self._watermark is None:
            since = now - lookback
        else:
            since = max(
                self._watermark - timedelta(minutes=self.cfg.poll_overlap_minutes), now - lookback
            )
        yield from self._pages(since, None, skip, on_error, "live")
