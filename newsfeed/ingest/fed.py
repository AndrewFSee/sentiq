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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from newsfeed.config import FedConfig
from newsfeed.ingest.base import ErrorFn, RawItem, SkipFn
from newsfeed.ingest.http import HttpClient
from newsfeed.storage import RawStore
from newsfeed.textutil import estimate_tokens, truncate_tokens
from newsfeed.timeutil import NY, IngestMode, ensure_utc, utcnow

SOURCE = "fed"
# fed/2: body = longest content column (fed/1 could pick a panel); fed/3: titles HTML-unescaped.
# Press-conference PDFs (2026-11-18) are new kinds; stored kinds parse as before, so no bump.
PARSER_VERSION = "fed/3"
PUBLISHER = "Board of Governors of the Federal Reserve System"
PSEUDO_TICKER = "FED"  # coverage rows only (the source has no companies)

log = logging.getLogger(__name__)

STATEMENT = re.compile(r"^\s*Federal Reserve issues FOMC statement\s*$", re.I)
MINUTES = re.compile(r"^\s*Minutes of the Federal Open Market Committee\b", re.I)
MINUTES_LINK = re.compile(r"/monetarypolicy/fomcminutes\d{8}\.htm")
PRESCONF_KINDS = ("presconf_opening", "presconf_transcript")
KINDS = ("fomc_statement", "fomc_minutes", "speech", "testimony", *PRESCONF_KINDS)

# FOMC press conferences: one PDF URL per meeting date, posted twice -- a PRELIMINARY version
# with the Chair's opening statement only, the same afternoon, then a FINAL version with the Q&A
# (1 day to 3 weeks later; later re-uploads of FINAL happen). Each version is its own document.
PRESCONF_PDF = "/mediacenter/files/FOMCpresconf{date}.pdf"
PRESCONF_LINK = re.compile(r"fomcpresconf(\d{8})\.htm")
PRESCONF_VERSION = {"PRELIMINARY": "presconf_opening", "FINAL": "presconf_transcript"}
_PC_HEADER = re.compile(r"Press\s+Conference(?:\s+Call)?\s+(PRELIMINARY|FINAL)\b")
_PC_TITLE = re.compile(
    r"Transcript of (Chair(?:man|woman)?\s+[^\n]+?)['\N{RIGHT SINGLE QUOTATION MARK}]s\s+"
    r"Press Conference[^\n]*"
)
_PC_PAGE_LINES = re.compile(
    r"^.*Press\s+Conference(?:\s+Call)?\s+(?:PRELIMINARY|FINAL)\s*$|^\s*Page \d+ of \d+\s*$",
    re.M,
)


def parse_presconf_pdf(data: bytes) -> dict[str, str]:
    """Version, title, speaker and text of a press-conference PDF (running page headers and
    "Page n of m" lines removed)."""
    import io

    from pypdf import PdfReader

    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
    if not pages:
        raise ValueError("press-conference PDF has no pages")
    head = _PC_HEADER.search(pages[0])
    if head is None:
        raise ValueError("press-conference PDF: no PRELIMINARY / FINAL header")
    text = "\n".join(_PC_PAGE_LINES.sub("", p) for p in pages)
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(ln for ln in lines if ln)).strip()
    m = _PC_TITLE.search(body)
    if m is None:
        raise ValueError("press-conference PDF: no 'Transcript of ...' title")
    speaker = m.group(1).strip()
    return {
        "version": head.group(1),
        "kind": PRESCONF_VERSION[head.group(1)],
        "title": re.sub(r"\s+", " ", m.group(0)).strip(),
        "speaker": speaker,
        "body": body,
    }


def presconf_dates(page: bytes) -> list[str]:
    """Meeting dates (YYYYMMDD) with a press-conference page linked from an FOMC calendar page."""
    return sorted(set(PRESCONF_LINK.findall(page.decode("utf-8", errors="replace"))))


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
        """``<kind>_<page name>`` (press conferences: ``<kind>_<YYYYMMDD>``): unique, and safe as
        a raw-store path component."""
        name = self.path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".htm")
        if name.endswith(".pdf"):
            name = name.removesuffix(".pdf").removeprefix("FOMCpresconf")
        return f"{self.kind}_{name}"


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
        raw = {
            "path": ref.path,
            "title": ref.title,
            "published_raw": ref.published_raw,
            "speaker": ref.speaker,
            "location": ref.location,
        }
        return self._item(
            ref,
            text=text,
            url=str(url),
            meta=meta,
            raw=raw,
            first_seen_at=first_seen_at,
            ingest_mode=ingest_mode,
        )

    def _item(
        self,
        ref: FedRef,
        *,
        text: str,
        url: str,
        meta: dict[str, object],
        raw: dict[str, object],
        first_seen_at: datetime,
        ingest_mode: IngestMode,
    ) -> RawItem:
        if not text.strip():
            raise ValueError(f"{ref.native_id}: empty document text")
        body, truncated = truncate_tokens(text, self.cfg.max_tokens, self.cfg.chars_per_token)
        meta["body_tokens_est_full"] = estimate_tokens(text, self.cfg.chars_per_token)
        return RawItem(
            source=SOURCE,
            native_id=ref.native_id,
            publisher=PUBLISHER,
            url=url,
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
            raw_json=json.dumps(raw, sort_keys=True),
            parser_version=PARSER_VERSION,
        )

    # ---------------------------------------------------------------- press conferences
    def calendar_dates(self, years: Iterable[int] = ()) -> list[str]:
        """Meeting dates with a press conference, from the FOMC calendar page (recent ~5 years)
        and the historical pages of ``years`` (missing pages are skipped)."""
        pages = [self.cfg.presconf_calendar]
        pages += [self.cfg.presconf_historical.format(year=y) for y in years]
        dates: set[str] = set()
        for path in pages:
            try:
                dates.update(presconf_dates(self._get(path)))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
        return sorted(dates)

    def presconf_version(self, date: str) -> tuple[bytes, datetime, str, str] | None:
        """(PDF, Last-Modified in UTC, raw header, raw-store file name) of the version currently
        posted for meeting ``date`` (YYYYMMDD), or None if nothing is posted (404). A HEAD
        decides; each version (by Last-Modified) is downloaded once and kept verbatim in the
        raw store under ``presconf_<date>/``."""
        url = f"{self.cfg.base_url}{PRESCONF_PDF.format(date=date)}"
        try:
            head = self.http.head(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        lm_raw = head.headers.get("last-modified")
        if not lm_raw:
            raise ValueError(f"{url}: no Last-Modified header")
        lm = ensure_utc(parsedate_to_datetime(lm_raw))
        name = f"v{lm:%Y%m%dT%H%M%SZ}.pdf"
        data = self.raw.get(SOURCE, f"presconf_{date}", name)
        if data is None:
            resp = self.http.get(url)
            lm_raw = resp.headers.get("last-modified") or lm_raw  # replaced since the HEAD?
            lm = ensure_utc(parsedate_to_datetime(lm_raw))
            name = f"v{lm:%Y%m%dT%H%M%SZ}.pdf"
            data = resp.content
            self.raw.put(SOURCE, f"presconf_{date}", name, data)
        return data, lm, lm_raw, name

    def presconf_item(
        self,
        date: str,
        data: bytes,
        published_at: datetime,
        published_raw: str,
        raw_file: str,
        *,
        first_seen_at: datetime,
        ingest_mode: IngestMode,
    ) -> RawItem:
        """One version of a press-conference PDF as an item. published_at = the file's HTTP
        Last-Modified (the time this exact text provably existed on the site)."""
        parts = parse_presconf_pdf(data)
        ref = FedRef(
            kind=parts["kind"],
            path=PRESCONF_PDF.format(date=date),
            title=parts["title"],
            published_at=published_at,
            published_raw=published_raw,
            speaker=parts["speaker"],
        )
        meta: dict[str, object] = {
            "kind": ref.kind,
            "speaker": parts["speaker"],
            "location": None,
            "meeting_date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "pdf_version": parts["version"],
            "raw_file": raw_file,
        }
        raw = {
            "path": ref.path,
            "title": ref.title,
            "published_raw": published_raw,
            "meeting_date": date,
            "raw_file": raw_file,
        }
        return self._item(
            ref,
            text=parts["body"],
            url=f"{self.cfg.base_url}{ref.path}",
            meta=meta,
            raw=raw,
            first_seen_at=first_seen_at,
            ingest_mode=ingest_mode,
        )

    def presconf_items(
        self,
        dates: Iterable[str],
        skip: SkipFn,
        on_error: ErrorFn | None,
        *,
        ingest_mode: IngestMode,
    ) -> Iterator[RawItem]:
        from newsfeed.ingest.base import make_item_id

        for date in sorted(set(dates)):
            if skip(make_item_id(SOURCE, f"presconf_transcript_{date}")):
                continue  # the full transcript is stored: nothing later to collect
            try:
                got = self.presconf_version(date)
                if got is None:
                    continue
                seen_at = self.clock()  # after the response that revealed this version
                item = self.presconf_item(
                    date, *got, first_seen_at=seen_at, ingest_mode=ingest_mode
                )
                if skip(make_item_id(SOURCE, item.native_id)):
                    continue  # this version is stored (e.g. the opening, until FINAL appears)
                kind = str(item.meta["kind"])
                self.stats[kind] = self.stats.get(kind, 0) + 1
                yield item
            except Exception as exc:  # one bad PDF must not stop the run
                if on_error is None:
                    raise
                on_error(f"presconf_{date}", exc)

    def reparse(self, stored: dict[str, Any]) -> RawItem:
        """Rebuild a stored item with the current parser from the raw cache (the caller copies
        the timing fields verbatim)."""
        raw = json.loads(stored["raw_json"])
        meta = json.loads(stored["meta"])
        if meta["kind"] in PRESCONF_KINDS:
            date, name = raw["meeting_date"], raw["raw_file"]
            data = self.raw.get(SOURCE, f"presconf_{date}", name)
            if data is None:
                raise ValueError(f"presconf_{date}/{name} missing from the raw store")
            return self.presconf_item(
                date,
                data,
                stored["published_at"],
                raw["published_raw"],
                name,
                first_seen_at=stored["first_seen_at"],
                ingest_mode=stored["ingest_mode"],
            )
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
        # press conferences of meetings in [start, end): the PDF posted *today* (normally FINAL),
        # published_at = its Last-Modified, so the backfill rule times the text we actually hold
        lo, hi = start.astimezone(NY).date(), end.astimezone(NY).date()
        years = range(lo.year, hi.year + 1)
        dates = [
            d
            for d in self.calendar_dates(years)
            if start <= datetime.strptime(d, "%Y%m%d").replace(tzinfo=NY) < end
        ]
        yield from self.presconf_items(dates, skip, on_error, ingest_mode="backfill")

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
        yield from self.presconf_items(
            self.presconf_candidates(refs, seen_at, catch_up=catch_up),
            skip,
            on_error,
            ingest_mode="live",
        )

    def presconf_candidates(
        self, refs: list[FedRef], now: datetime, *, catch_up: bool
    ) -> list[str]:
        """Meeting dates whose press-conference PDF is probed (one HEAD each): every poll, the
        dates of FOMC statements from the last ``presconf_live_days`` (the opening statement
        appears that afternoon); on catch-up, statements and calendar-page meetings from the
        last ``presconf_catch_up_days`` (the FINAL transcript follows within ~3 weeks)."""
        days = self.cfg.presconf_catch_up_days if catch_up else self.cfg.presconf_live_days
        lo = now - timedelta(days=days)
        dates = {
            f"{r.published_at.astimezone(NY):%Y%m%d}"
            for r in refs
            if r.kind == "fomc_statement" and r.published_at >= lo
        }
        if catch_up:
            for d in self.calendar_dates():
                day = datetime.strptime(d, "%Y%m%d").replace(tzinfo=NY)
                if lo <= day <= now:
                    dates.add(d)
        return sorted(dates)
