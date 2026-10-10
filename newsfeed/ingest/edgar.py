"""EDGAR 8-K adapter.

Flow per filing (all requests rate-limited, retried, and raw responses kept in RawStore):
  1. data.sec.gov/submissions/CIK##########.json (+ paginated ``files[]`` for older history)
     -> candidate filings (form in config.forms)
  2. <folder>/<accession>.hdr.sgml  -> AUTHORITATIVE acceptance datetime (Eastern wall clock)
  3. <folder>/<accession>-index.htm -> document list with types (to find EX-99.x exhibits)
  4. primary document + selected exhibits -> visible text -> item sections -> truncated body

Timestamps: the submissions JSON ``acceptanceDateTime`` is NOT trustworthy -- its convention
differs by filer (see ``timeutil.edgar_json_literal``). Read literally it is an upper bound on
the true time, used only for coarse filtering (1-day margins) and recorded as a cross-check.
``filingDate`` is never used for timing (after 17:30 ET it is the next business day).
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from bs4 import BeautifulSoup

from newsfeed.config import EdgarConfig
from newsfeed.ingest.base import ErrorFn, RawItem, SkipFn, make_item_id
from newsfeed.ingest.http import HttpClient
from newsfeed.storage import RawStore
from newsfeed.textutil import estimate_tokens, html_to_text, truncate_tokens
from newsfeed.tickers import Universe
from newsfeed.timeutil import (
    IngestMode,
    classify_edgar_json_convention,
    edgar_json_literal,
    ensure_utc,
    parse_et_wall,
    utcnow,
)

log = logging.getLogger(__name__)

SOURCE = "edgar"
PARSER_VERSION = "edgar-8k/2"  # 2: table-of-contents-aware item splitting
PUBLISHER = "SEC EDGAR"
_ATOM = "{http://www.w3.org/2005/Atom}"
_FEED_TITLE = re.compile(r"^(?P<form>.+?) - (?P<name>.*) \((?P<cik>\d{10})\) \((?P<role>[^)]*)\)$")

ITEM_NAMES: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers; Compensatory Arrangements",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to the Code of Ethics, or Waiver",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

_HDR_FIELD = re.compile(r"^<([A-Z-]+)>(.*)$", re.M)
_ITEM_LINE = re.compile(r"^\s*item\s*(\d{1,2}\.\d{2})\b[\s.:\-\N{EN DASH}\N{EM DASH}]*(.*)$", re.I)
_SIGNATURE_LINE = re.compile(r"^\s*signatures?\s*$", re.I)


# --------------------------------------------------------------------------- parsing helpers
@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession: str
    form: str
    filing_date: str
    acceptance_json_raw: str | None  # None when discovered via the current-filings feed
    items: str
    primary_document: str
    row: dict[str, Any] = field(compare=False, hash=False)
    feed_updated: datetime | None = None  # Atom <updated>: true acceptance time, with offset

    @property
    def accepted_upper_bound(self) -> datetime:
        """An upper bound on the acceptance time, for coarse filtering only. Literal JSON time
        is >= true time by 0-5h; the Atom feed time is exact."""
        if self.feed_updated is not None:
            return self.feed_updated
        if self.acceptance_json_raw is None:
            raise ValueError(f"{self.accession}: no listing timestamp")
        return edgar_json_literal(self.acceptance_json_raw)

    @property
    def item_id(self) -> str:
        return make_item_id(SOURCE, self.accession)


@dataclass(frozen=True)
class IndexDocument:
    seq: str
    description: str
    filename: str
    doc_type: str


@dataclass(frozen=True)
class FeedEntry:
    accession: str
    cik: int
    form: str
    company: str
    updated: datetime
    index_url: str


def parse_current_feed(xml: bytes | str) -> list[FeedEntry]:
    """Parse EDGAR's 'latest filings' Atom feed (``browse-edgar?action=getcurrent``).

    ``<updated>`` carries an explicit UTC offset and equals the SGML-header acceptance time
    (verified 2026-10). Co-registrant filings appear once per filer CIK.
    """
    root = ET.fromstring(xml)
    out: list[FeedEntry] = []
    for e in root.iter(f"{_ATOM}entry"):
        title = (e.findtext(f"{_ATOM}title") or "").strip()
        m = _FEED_TITLE.match(title)
        entry_id = e.findtext(f"{_ATOM}id") or ""
        acc = entry_id.rsplit("accession-number=", 1)[-1].strip()
        updated = e.findtext(f"{_ATOM}updated")
        link = e.find(f"{_ATOM}link")
        if not (m and acc and updated):
            log.warning("unparseable feed entry: %r", title)
            continue
        out.append(
            FeedEntry(
                accession=acc,
                cik=int(m["cik"]),
                form=m["form"].strip(),
                company=m["name"].strip(),
                updated=ensure_utc(datetime.fromisoformat(updated.strip())),
                index_url=link.get("href", "") if link is not None else "",
            )
        )
    return out


def columnar_to_rows(block: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """data.sec.gov returns filings column-wise; turn into a list of row dicts."""
    keys = list(block)
    n = len(block[keys[0]]) if keys else 0
    if any(len(block[k]) != n for k in keys):
        raise ValueError("ragged columnar filings block")
    return [{k: block[k][i] for k in keys} for i in range(n)]


def parse_header(text: str) -> dict[str, Any]:
    """Parse an EDGAR ``.hdr.sgml``. Repeated tags (ITEMS) become lists; first wins otherwise."""
    out: dict[str, Any] = {"ITEMS": []}
    for tag, value in _HDR_FIELD.findall(text):
        value = value.strip()
        if tag == "ITEMS":
            out["ITEMS"].append(value)
        elif value and tag not in out:
            out[tag] = value
    if "ACCEPTANCE-DATETIME" not in out:
        raise ValueError("header has no ACCEPTANCE-DATETIME")
    return out


def parse_index(html: str | bytes) -> tuple[list[IndexDocument], str | None]:
    """Documents from a filing's ``-index.htm`` and its 'Accepted' field (Eastern, as shown)."""
    soup = BeautifulSoup(html, "lxml")
    docs: list[IndexDocument] = []
    for table in soup.find_all("table", class_="tableFile"):
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue
            link = cells[2].find("a")
            if link is None:
                continue
            filename = link.get("href", "").rsplit("/", 1)[-1]
            docs.append(
                IndexDocument(
                    seq=cells[0].get_text(strip=True),
                    description=cells[1].get_text(" ", strip=True),
                    filename=filename,
                    doc_type=cells[3].get_text(strip=True),
                )
            )
    accepted = None
    for head in soup.find_all("div", class_="infoHead"):
        if head.get_text(strip=True) == "Accepted":
            val = head.find_next_sibling("div")
            accepted = val.get_text(strip=True) if val else None
    return docs, accepted


def split_items(text: str) -> dict[str, str]:
    """Split 8-K body text into {item_code: section_text}.

    Many filings open with a table of contents that repeats every item heading *and* a
    "SIGNATURES" entry. So: stop only at the first signature line after the last item heading,
    and when an item heading repeats, the later occurrence (the real section) replaces the
    earlier one (the TOC line).
    """
    lines = text.splitlines()
    heading_rows = [i for i, ln in enumerate(lines) if _ITEM_LINE.match(ln)]
    end = len(lines)
    if heading_rows:
        end = next(
            (i for i in range(heading_rows[-1] + 1, len(lines)) if _SIGNATURE_LINE.match(lines[i])),
            len(lines),
        )
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[:end]:
        m = _ITEM_LINE.match(line)
        if m:
            current = m.group(1)
            sections[current] = [m.group(2)] if m.group(2) else []  # later occurrence wins
            continue
        if current is not None and not _SIGNATURE_LINE.match(line):
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def compose_body(
    items: dict[str, str], primary_text: str, exhibits: list[tuple[IndexDocument, str]]
) -> str:
    parts: list[str] = []
    if items:
        for code, txt in items.items():
            name = ITEM_NAMES.get(code, "")
            parts.append(f"[ITEM {code}{': ' + name if name else ''}]\n{txt}")
    else:  # unrecognised layout (e.g. pre-2004 numbering): keep the whole visible text
        parts.append(f"[8-K TEXT]\n{primary_text}")
    for doc, txt in exhibits:
        label = doc.doc_type + (f" - {doc.description}" if doc.description else "")
        parts.append(f"[EXHIBIT {label}]\n{txt}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- adapter
class EdgarAdapter:
    name = SOURCE

    def __init__(
        self,
        cfg: EdgarConfig,
        http: HttpClient,
        universe: Universe,
        raw: RawStore,
        *,
        poll_lookback: timedelta | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.cfg = cfg
        self.http = http
        self.universe = universe
        self.raw = raw
        self.poll_lookback = poll_lookback or timedelta(hours=cfg.poll_lookback_hours)
        self.clock = clock
        self.forms = set(cfg.forms)
        self.stats: dict[str, int] = {}
        self.last_activity_at: datetime | None = None
        # sp500_pit: per-CIK ingestion windows; None = every CIK over the requested range
        self.cik_windows: dict[int, tuple[datetime, datetime]] | None = None
        self._feed_high_water: datetime | None = None  # newest feed entry seen by last poll

    # ---- listing
    def _folder(self, cik: int, accession: str) -> str:
        return f"{self.cfg.archives_base_url}/{cik}/{accession.replace('-', '')}"

    def _to_refs(self, cik: int, rows: list[dict[str, Any]]) -> list[FilingRef]:
        refs = []
        for r in rows:
            if r.get("form") not in self.forms:
                continue
            refs.append(
                FilingRef(
                    cik=cik,
                    accession=r["accessionNumber"],
                    form=r["form"],
                    filing_date=r.get("filingDate", ""),
                    acceptance_json_raw=r["acceptanceDateTime"],
                    items=r.get("items", "") or "",
                    primary_document=r.get("primaryDocument", "") or "",
                    row=r,
                )
            )
        return refs

    def list_filings(
        self, cik: int, start: datetime | None = None, end: datetime | None = None
    ) -> list[FilingRef]:
        """Candidate filings for one CIK, coarse-filtered (+-1 day margin) on decoded JSON time."""
        data = self.http.get(f"{self.cfg.submissions_base_url}/CIK{cik:010d}.json").json()
        rows = columnar_to_rows(data["filings"]["recent"])
        if start is not None:
            lo = (ensure_utc(start) - timedelta(days=2)).date().isoformat()
            for f in data["filings"].get("files", []):
                if f.get("filingTo", "9999") >= lo:
                    older = self.http.get(f"{self.cfg.submissions_base_url}/{f['name']}").json()
                    rows.extend(columnar_to_rows(older))
        refs = self._to_refs(cik, rows)
        margin = timedelta(days=1)
        if start is not None:
            refs = [r for r in refs if r.accepted_upper_bound >= ensure_utc(start) - margin]
        if end is not None:
            refs = [r for r in refs if r.accepted_upper_bound < ensure_utc(end) + margin]
        return refs

    def window_for(
        self, cik: int, start: datetime | None, end: datetime | None
    ) -> tuple[datetime | None, datetime | None] | None:
        """[start, end) clipped to the CIK's ingestion window (``cik_windows``), or None if
        they don't overlap."""
        if self.cik_windows is None:
            return start, end
        w = self.cik_windows.get(cik)
        if w is None:
            return None
        s = w[0] if start is None else max(w[0], ensure_utc(start))
        e = w[1] if end is None else min(w[1], ensure_utc(end))
        return (s, e) if e > s else None

    def _gather(
        self, start: datetime | None, end: datetime | None
    ) -> list[tuple[FilingRef, list[str], list[tuple[datetime, datetime]]]]:
        """List filings for every universe CIK (inside its window, if any); merge co-registrant
        filings by accession, keeping each filer CIK's window for the exact time filter."""
        by_acc: dict[str, tuple[FilingRef, set[str], list]] = {}
        for cik in self.universe.ciks:
            win = self.window_for(cik, start, end)
            if win is None:
                continue
            for ref in self.list_filings(cik, *win):
                tick = set(self.universe.tickers_for(cik))
                if ref.accession in by_acc:
                    by_acc[ref.accession][1].update(tick)
                    by_acc[ref.accession][2].append(win)
                else:
                    by_acc[ref.accession] = (ref, tick, [win])
        ordered = sorted(by_acc.values(), key=lambda x: x[0].accepted_upper_bound)
        return [(ref, sorted(t), w) for ref, t, w in ordered]

    # ---- fetching
    def _get_raw(self, ref: FilingRef, name: str) -> bytes:
        """Fetch a filing document, using the raw store as a cache (re-parse without re-fetch)."""
        cached = self.raw.get(SOURCE, ref.accession, name)
        if cached is not None:
            return cached
        body = self.http.get(f"{self._folder(ref.cik, ref.accession)}/{name}").content
        self.raw.put(SOURCE, ref.accession, name, body)
        return body

    def fetch_filing(
        self,
        ref: FilingRef,
        tickers: list[str],
        *,
        first_seen_at: datetime,
        ingest_mode: IngestMode,
    ) -> RawItem:
        header_text = self._get_raw(ref, f"{ref.accession}.hdr.sgml").decode("latin-1")
        hdr = parse_header(header_text)
        published_at = parse_et_wall(hdr["ACCEPTANCE-DATETIME"])

        if ref.acceptance_json_raw is None:  # discovered via the Atom feed
            convention = "feed"
            if ref.feed_updated != published_at:
                log.warning(
                    "feed time %s != header time %s for %s",
                    ref.feed_updated,
                    published_at,
                    ref.accession,
                )
        else:
            convention = classify_edgar_json_convention(ref.acceptance_json_raw, published_at)
        key = f"json_convention_{convention}"
        self.stats[key] = self.stats.get(key, 0) + 1
        if convention not in ("utc", "double_offset", "feed"):
            # Would break the upper-bound assumption used for coarse filtering: investigate.
            log.warning(
                "unexpected EDGAR JSON acceptanceDateTime %s json=%s header=%s -> %s",
                ref.accession,
                ref.acceptance_json_raw,
                hdr["ACCEPTANCE-DATETIME"],
                convention,
            )

        docs, index_accepted = parse_index(self._get_raw(ref, f"{ref.accession}-index.htm"))
        primary_name = ref.primary_document or next(
            (d.filename for d in docs if d.doc_type == ref.form), ""
        )
        if not primary_name:
            raise ValueError(f"no primary document for {ref.accession}")
        primary_text = html_to_text(self._get_raw(ref, primary_name))
        items = split_items(primary_text)

        exhibits: list[tuple[IndexDocument, str]] = []
        for d in docs:
            if d.filename == primary_name or not d.filename.lower().endswith(
                (".htm", ".html", ".txt")
            ):
                continue
            if any(d.doc_type.upper().startswith(p) for p in self.cfg.exhibit_type_prefixes):
                exhibits.append((d, html_to_text(self._get_raw(ref, d.filename))))

        body_full = compose_body(items, primary_text, exhibits)
        body, truncated = truncate_tokens(body_full, self.cfg.max_tokens, self.cfg.chars_per_token)
        item_codes = hdr["ITEMS"] or [c for c in ref.items.split(",") if c]
        company = hdr.get("CONFORMED-NAME", "")
        title = f"{company} {ref.form}: " + "; ".join(
            f"Item {c} {ITEM_NAMES.get(c, '')}".strip() for c in item_codes
        )

        filing_date = ref.filing_date
        if not filing_date and len(hdr.get("FILING-DATE", "")) == 8:
            fd = hdr["FILING-DATE"]
            filing_date = f"{fd[:4]}-{fd[4:6]}-{fd[6:]}"

        return RawItem(
            source=SOURCE,
            native_id=ref.accession,
            publisher=PUBLISHER,
            url=f"{self._folder(ref.cik, ref.accession)}/{ref.accession}-index.htm",
            title=title.strip(),
            body_text=body,
            body_tokens_est=estimate_tokens(body, self.cfg.chars_per_token),
            body_truncated=truncated,
            tickers=tickers,
            published_at=published_at,
            published_at_raw=f"{hdr['ACCEPTANCE-DATETIME']} America/New_York",
            first_seen_at=first_seen_at,
            ingest_mode=ingest_mode,
            meta={
                "cik": ref.cik,
                "accession": ref.accession,
                "form": ref.form,
                "company": company,
                "items": item_codes,
                "items_found_in_text": sorted(items),
                "filing_date": filing_date,
                "feed_updated": ref.feed_updated.isoformat() if ref.feed_updated else None,
                "period": hdr.get("PERIOD"),
                "acceptance_header_et": hdr["ACCEPTANCE-DATETIME"],
                "acceptance_index_et": index_accepted,
                "acceptance_json_raw": ref.acceptance_json_raw,
                "json_convention": convention,
                "primary_document": primary_name,
                "exhibits": [{"type": d.doc_type, "file": d.filename} for d, _ in exhibits],
                "body_tokens_est_full": estimate_tokens(body_full, self.cfg.chars_per_token),
            },
            raw_json=json.dumps(
                {
                    "submissions_row": ref.row,
                    "header": header_text,
                    "index_documents": [d.__dict__ for d in docs],
                },
                sort_keys=True,
            ),
            parser_version=PARSER_VERSION,
        )

    # ---- re-parsing stored items
    def reparse(self, stored: dict[str, Any]) -> RawItem:
        """Rebuild a stored item with the current parser from the raw-document cache (network
        only for documents missing from the cache). Timing fields are the caller's to copy."""
        meta = json.loads(stored["meta"])
        raw = json.loads(stored["raw_json"])
        fu = meta.get("feed_updated")
        ref = FilingRef(
            cik=int(meta["cik"]),
            accession=meta["accession"],
            form=meta["form"],
            filing_date=meta.get("filing_date") or "",
            acceptance_json_raw=meta.get("acceptance_json_raw"),
            items=",".join(meta.get("items") or []),
            primary_document=meta.get("primary_document") or "",
            row=raw.get("submissions_row") or {},
            feed_updated=datetime.fromisoformat(fu) if fu else None,
        )
        return self.fetch_filing(
            ref,
            list(stored["tickers"]),
            first_seen_at=stored["first_seen_at"],
            ingest_mode=stored["ingest_mode"],
        )

    # ---- adapter interface
    def backfill(
        self,
        start: datetime,
        end: datetime,
        skip: SkipFn,
        on_error: ErrorFn | None = None,
    ) -> Iterator[RawItem]:
        start, end = ensure_utc(start, name="start"), ensure_utc(end, name="end")
        for ref, tickers, wins in self._gather(start, end):
            if skip(ref.item_id):
                continue
            try:
                item = self.fetch_filing(
                    ref, tickers, first_seen_at=self.clock(), ingest_mode="backfill"
                )
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(ref.accession, exc)
                continue
            # exact filter on the authoritative time, inside a window of one of its filers
            if any(ws <= item.published_at < we for ws, we in wins):
                yield item

    def poll(
        self, skip: SkipFn, on_error: ErrorFn | None = None, *, catch_up: bool = False
    ) -> Iterator[RawItem]:
        """Live poll via the 'latest filings' Atom feed (one request per page, all companies).

        * first_seen_at = arrival time of the feed page that listed the filing (never earlier).
        * Pages until it overlaps the previous poll's newest entry, the lookback cutoff, or
          ``current_feed_max_pages``. Anything beyond that window is recovered by
          ``catch_up=True`` (per-CIK submissions sweep), which the collector runs periodically.
        * ``last_activity_at`` = newest entry for ANY company: 8-Ks arrive market-wide every few
          minutes during the day, so a stale value means the feed (or we) are down.
        """
        yielded: set[str] = set()
        for item in self._poll_feed(skip, on_error):
            yielded.add(item.item_id)
            yield item
        if catch_up:  # feed first (keeps the activity signal fresh), then the wide sweep
            yield from self._poll_sweep(lambda i: i in yielded or skip(i), on_error)

    def _poll_feed(self, skip: SkipFn, on_error: ErrorFn | None) -> Iterator[RawItem]:
        cutoff = self.clock() - self.poll_lookback
        size = self.cfg.current_feed_page_size
        found: dict[str, tuple[FilingRef, set[str], datetime]] = {}
        newest: datetime | None = None
        for page in range(self.cfg.current_feed_max_pages):
            url = f"{self.cfg.current_feed_url}&start={page * size}&count={size}"
            body = self.http.get(url).content
            seen_at = self.clock()  # stamped after the response arrived
            entries = parse_current_feed(body)
            if not entries:
                break
            page_newest = max(e.updated for e in entries)
            newest = page_newest if newest is None else max(newest, page_newest)
            for e in entries:
                if e.form not in self.forms or e.updated < cutoff:
                    continue
                tick = set(self.universe.tickers_for(e.cik))
                if not tick:
                    continue
                ref = FilingRef(
                    cik=e.cik,
                    accession=e.accession,
                    form=e.form,
                    filing_date="",
                    acceptance_json_raw=None,
                    items="",
                    primary_document="",
                    row={"feed_entry": {**e.__dict__, "updated": e.updated.isoformat()}},
                    feed_updated=e.updated,
                )
                if skip(ref.item_id):
                    continue
                if e.accession in found:  # co-registrant: merge tickers, keep earliest sighting
                    found[e.accession][1].update(tick)
                else:
                    found[e.accession] = (ref, tick, seen_at)
            oldest = min(e.updated for e in entries)
            hw = self._feed_high_water
            if oldest < cutoff or (hw is not None and oldest <= hw) or len(entries) < size:
                break
        else:
            if self._feed_high_water is not None:
                log.warning("feed paging hit max_pages; relying on catch-up sweep for gaps")
        for ref, tick, seen_at in sorted(found.values(), key=lambda x: x[0].accepted_upper_bound):
            try:
                yield self.fetch_filing(
                    ref, sorted(tick), first_seen_at=seen_at, ingest_mode="live"
                )
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(ref.accession, exc)
        if newest is not None:
            self.last_activity_at = newest
            self._feed_high_water = newest

    def _poll_sweep(self, skip: SkipFn, on_error: ErrorFn | None) -> Iterator[RawItem]:
        """Catch-up: list every universe CIK's submissions; fetch recent filings not stored.

        first_seen_at is stamped *after* each listing response arrives.
        """
        cutoff = self.clock() - self.poll_lookback
        found: dict[str, tuple[FilingRef, set[str], datetime]] = {}
        for cik in self.universe.ciks:
            refs = self.list_filings(cik)
            seen_at = self.clock()
            for ref in refs:
                if ref.accepted_upper_bound < cutoff or skip(ref.item_id):
                    continue
                tick = set(self.universe.tickers_for(cik))
                if ref.accession in found:
                    found[ref.accession][1].update(tick)
                else:
                    found[ref.accession] = (ref, tick, seen_at)
        for ref, tick, seen_at in found.values():
            try:
                yield self.fetch_filing(
                    ref, sorted(tick), first_seen_at=seen_at, ingest_mode="live"
                )
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(ref.accession, exc)
