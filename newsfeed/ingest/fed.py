"""Federal Reserve Board documents (source ``fed``) from federalreserve.gov.

Kinds: ``fomc_statement`` (the "Federal Reserve issues FOMC statement" press release: decision,
assessment, votes), ``fomc_minutes`` (the minutes page linked from the "Minutes of the Federal
Open Market Committee" release), ``speech`` and ``testimony`` (Board members only; regional
Fed presidents publish elsewhere).

Timing. ``published_at`` = the Fed's release time: the JSON listings give New York wall-clock
time ("11/18/2026 2:00:00 PM", converted with zoneinfo, so DST is handled), the RSS feeds the
same instant in GMT (checked 2026-11-18: listing 1:35 PM ET = feed 18:35 GMT). Live items get
``first_seen_at`` stamped after the listing response that revealed them; tradable_at is then
computed by the ingest runner like every other source.

Polls. The cheap live poll reads two RSS feeds (~15 newest entries each); a catch-up (and a
backfill) reads the JSON listings, which go back decades. Document pages are cached verbatim in
the raw store (``data/raw/fed/<native_id>/``), so re-parsing never re-downloads.

These are market-level documents (``config.MARKET_SOURCES``): no tickers, no links, no stories.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from newsfeed.config import FedConfig
from newsfeed.ingest.base import ErrorFn, RawItem, SkipFn
from newsfeed.ingest.http import HttpClient
from newsfeed.storage import RawStore
from newsfeed.textutil import estimate_tokens, truncate_tokens
from newsfeed.timeutil import NY, IngestMode, ensure_utc, utcnow

SOURCE = "fed"
# fed/2: body = longest content column (fed/1 could pick a panel); fed/3: titles HTML-unescaped
PARSER_VERSION = "fed/3"
PUBLISHER = "Board of Governors of the Federal Reserve System"
PSEUDO_TICKER = "FED"  # coverage rows only (the source has no companies)

log = logging.getLogger(__name__)

STATEMENT = re.compile(r"^\s*Federal Reserve issues FOMC statement\s*$", re.I)
MINUTES = re.compile(r"^\s*Minutes of the Federal Open Market Committee\b", re.I)
MINUTES_LINK = re.compile(r"/monetarypolicy/fomcminutes\d{8}\.htm")
KINDS = ("fomc_statement", "fomc_minutes", "speech", "testimony")


class MarketUniverse:
    """Stand-in for a company universe: one pseudo-ticker, so collect / backfill record
    coverage ("was Fed collection running") with the same machinery as company sources."""

    def __init__(self) -> None:
        self.ciks = (0,)
        self.cik_to_tickers = {0: [PSEUDO_TICKER]}

    def tickers_for(self, cik: int) -> list[str]:
        return [PSEUDO_TICKER] if cik == 0 else []


@dataclass(frozen=True)
class FedRef:
    kind: str
    path: str  # site path of the document's page, e.g. /newsevents/speech/waller20261117a.htm
    title: str
    published_at: datetime  # UTC
    published_raw: str
    speaker: str | None = None
    location: str | None = None

    @property
    def native_id(self) -> str:
        """``<kind>_<page name>``: unique, and safe as a raw-store path component."""
        return f"{self.kind}_{self.path.rstrip('/').rsplit('/', 1)[-1].removesuffix('.htm')}"


def classify(title: str, path: str, press_type: str | None = None) -> str | None:
    """Document kind from a listing entry, or None if it is not collected."""
    if "/newsevents/speech/" in path:
        return "speech"
    if "/newsevents/testimony/" in path:
        return "testimony"
    if "/newsevents/pressreleases/" in path and press_type in (None, "Monetary Policy"):
        if STATEMENT.match(title):
            return "fomc_statement"
        if MINUTES.match(title):
            return "fomc_minutes"
    return None


def parse_listing_time(s: str) -> datetime:
    """'11/18/2026 2:00:00 PM' (New York wall clock) -> UTC."""
    s = s.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y"):
        try:
            local = datetime.strptime(s, fmt)  # noqa: DTZ007 - wall clock, zone attached next
        except ValueError:
            continue
        return ensure_utc(local.replace(tzinfo=NY))
    raise ValueError(f"unparseable Fed listing time {s!r}")


def parse_listing(data: bytes, base_kind: str) -> list[FedRef]:
    """Entries of a JSON listing (ne-press / ne-speeches / ne-testimony) that are collected."""
    out = []
    for e in json.loads(data.decode("utf-8-sig")):
        path, title, d = e.get("l"), html.unescape(e.get("t") or "").strip(), e.get("d")
        if not path or not d:
            continue  # the trailing {"updateDate": ...} record
        kind = classify(title, path, e.get("pt") if base_kind == "press" else None)
        if kind is None:
            continue
        out.append(
            FedRef(
                kind=kind,
                path=path,
                title=title,
                published_at=parse_listing_time(d),
                published_raw=f"{d} America/New_York",
                speaker=(e.get("s") or "").strip() or None,
                location=(e.get("lo") or "").strip() or None,
            )
        )
    return out


def parse_feed(data: bytes, base_url: str) -> list[FedRef]:
    """Entries of an RSS feed that are collected (speaker / location come from the page)."""
    out = []
    root = ElementTree.fromstring(data)
    for it in root.iter("item"):
        link = (it.findtext("link") or "").strip()
        title = html.unescape(it.findtext("title") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        if not link or not pub:
            continue
        path = link.removeprefix(base_url).removeprefix("http://www.federalreserve.gov")
        if "/newsevents/speech/" in path or "/newsevents/testimony/" in path:
            title = title.split(",", 1)[1].strip() if "," in title else title  # "Barr, Title"
        kind = classify(title, path)
        if kind is None:
            continue
        out.append(
            FedRef(
                kind=kind,
                path=path,
                title=title,
                published_at=ensure_utc(parsedate_to_datetime(pub)),
                published_raw=pub,
            )
        )
    return out


def _article(html: bytes | str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    art = soup.find(id="article")
    if art is None:
        raise ValueError("no #article element")
    return art


def _clean(el) -> str:
    for junk in el.select("script, style, .shareDL, .dropdown, .watchLive, .hidden-sm"):
        junk.decompose()
    lines = (ln.strip() for ln in el.get_text("\n").splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for ln in lines if ln)).strip()


def page_parts(html: bytes | str) -> dict[str, str | None]:
    """Heading fields (speaker, location) and body text of a press-release / speech page: the
    body is the longest content column that is not the heading or a panel (fed/1 took the last
    column, which on pages with charts is the "View speech charts and figures" attachment
    panel); pages without that layout (minutes) use the whole article."""
    art = _article(html)
    speaker = art.select_one("p.speaker")
    location = art.select_one("p.location")
    cols = [
        d
        for d in art.find_all("div", class_="col-md-8")
        if not {"heading", "panel"} & set(d.get("class") or [])
    ]
    main = max(cols, key=lambda d: len(d.get_text(" ", strip=True))) if cols else art
    body = _clean(main)
    return {
        "speaker": speaker.get_text(" ", strip=True) if speaker else None,
        "location": location.get_text(" ", strip=True) if location else None,
        "body": body,
    }


class FedAdapter:
    name = SOURCE

    def __init__(
        self,
        cfg: FedConfig,
        http: HttpClient,
        raw: RawStore,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.cfg = cfg
        self.http = http
        self.raw = raw
        self.clock = clock
        self.universe = MarketUniverse()
        self.last_activity_at: datetime | None = None
        self.stats: dict[str, int] = {}

    # ---------------------------------------------------------------- listings
    def _get(self, path: str) -> bytes:
        return self.http.get(f"{self.cfg.base_url}{path}").content

    def list_all(self) -> list[FedRef]:
        refs: list[FedRef] = []
        for kind, path in self.cfg.listings.items():
            refs += parse_listing(self._get(path), kind)
        return self._unique(refs)

    def list_feeds(self) -> list[FedRef]:
        refs: list[FedRef] = []
        for path in self.cfg.feeds:
            refs += parse_feed(self._get(path), self.cfg.base_url)
        return self._unique(refs)

    @staticmethod
    def _unique(refs: list[FedRef]) -> list[FedRef]:
        seen: dict[str, FedRef] = {}
        for r in sorted(refs, key=lambda r: (r.published_at, r.native_id)):
            seen.setdefault(r.native_id, r)  # same page in two listings: keep the earliest
        return list(seen.values())

    # ---------------------------------------------------------------- documents
    def _page(self, native_id: str, name: str, path: str) -> bytes:
        cached = self.raw.get(SOURCE, native_id, name)
        if cached is not None:
            return cached
        data = self._get(path)
        self.raw.put(SOURCE, native_id, name, data)
        return data

    def fetch(self, ref: FedRef, *, first_seen_at: datetime, ingest_mode: IngestMode) -> RawItem:
        nid = ref.native_id
        page = self._page(nid, "page.html", ref.path)
        parts = page_parts(page)
        url = f"{self.cfg.base_url}{ref.path}"
        meta: dict[str, object] = {
            "kind": ref.kind,
            "speaker": parts["speaker"] or ref.speaker,
            "location": parts["location"] or ref.location,
        }
        text = parts["body"] or ""
        if ref.kind == "fomc_minutes":
            m = MINUTES_LINK.search(page.decode("utf-8", errors="replace"))
            if m is None:
                raise ValueError(f"{nid}: no link to the minutes page")
            text = page_parts(self._page(nid, "minutes.html", m.group(0)))["body"] or ""
            meta["minutes_url"] = f"{self.cfg.base_url}{m.group(0)}"
            url = meta["minutes_url"]  # type: ignore[assignment]
        if not text.strip():
            raise ValueError(f"{nid}: empty document text")
        body, truncated = truncate_tokens(text, self.cfg.max_tokens, self.cfg.chars_per_token)
        meta["body_tokens_est_full"] = estimate_tokens(text, self.cfg.chars_per_token)
        return RawItem(
            source=SOURCE,
            native_id=nid,
            publisher=PUBLISHER,
            url=str(url),
            title=ref.title,
            body_text=body,
            body_tokens_est=estimate_tokens(body, self.cfg.chars_per_token),
            body_truncated=truncated,
            tickers=[],
            published_at=ref.published_at,
            published_at_raw=ref.published_raw,
            first_seen_at=first_seen_at,
            ingest_mode=ingest_mode,
            meta=meta,
            raw_json=json.dumps(
                {
                    "path": ref.path,
                    "title": ref.title,
                    "published_raw": ref.published_raw,
                    "speaker": ref.speaker,
                    "location": ref.location,
                },
                sort_keys=True,
            ),
            parser_version=PARSER_VERSION,
        )

    def reparse(self, stored: dict[str, Any]) -> RawItem:
        """Rebuild a stored item with the current parser from the raw cache (the caller copies
        the timing fields verbatim)."""
        raw = json.loads(stored["raw_json"])
        meta = json.loads(stored["meta"])
        ref = FedRef(
            kind=meta["kind"],
            path=raw["path"],
            title=html.unescape(raw["title"]),
            published_at=stored["published_at"],
            published_raw=raw["published_raw"],
            speaker=raw.get("speaker"),
            location=raw.get("location"),
        )
        return self.fetch(
            ref, first_seen_at=stored["first_seen_at"], ingest_mode=stored["ingest_mode"]
        )

    def _emit(
        self,
        refs: list[FedRef],
        skip: SkipFn,
        on_error: ErrorFn | None,
        *,
        first_seen_at: datetime | None,
        ingest_mode: IngestMode,
    ) -> Iterator[RawItem]:
        from newsfeed.ingest.base import make_item_id

        for ref in refs:
            if skip(make_item_id(SOURCE, ref.native_id)):
                continue
            try:
                yield self.fetch(
                    ref, first_seen_at=first_seen_at or self.clock(), ingest_mode=ingest_mode
                )
                self.stats[ref.kind] = self.stats.get(ref.kind, 0) + 1
            except Exception as exc:  # one bad page must not stop the run
                if on_error is None:
                    raise
                on_error(ref.native_id, exc)

    # ---------------------------------------------------------------- protocol
    def backfill(
        self, start: datetime, end: datetime, skip: SkipFn, on_error: ErrorFn | None = None
    ) -> Iterator[RawItem]:
        """Documents released in [start, end), oldest first. first_seen_at = when we fetch
        (truthful); tradable_at follows the backfill rule (published + lag)."""
        start, end = ensure_utc(start), ensure_utc(end)
        refs = [r for r in self.list_all() if start <= r.published_at < end]
        yield from self._emit(refs, skip, on_error, first_seen_at=None, ingest_mode="backfill")

    def poll(
        self, skip: SkipFn, on_error: ErrorFn | None = None, *, catch_up: bool = False
    ) -> Iterator[RawItem]:
        """New documents. The feeds (or, on catch-up, the full listings) are read first; their
        arrival time is every new item's first_seen_at -- the moment we could know of it."""
        refs = self.list_all() if catch_up else self.list_feeds()
        seen_at = self.clock()  # stamped after the listing responses arrived
        if refs:
            self.last_activity_at = max(r.published_at for r in refs)
        cutoff = seen_at - timedelta(days=self.cfg.poll_lookback_days)
        # a catch-up re-reads decades of listings; only recent entries can be new live items
        fresh = [r for r in refs if r.published_at >= cutoff]
        yield from self._emit(fresh, skip, on_error, first_seen_at=seen_at, ingest_mode="live")
