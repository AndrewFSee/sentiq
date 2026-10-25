"""EDGAR 10-K / 10-Q adapter (source ``edgar_periodic``): full text of annual and quarterly
reports, for document-change ("Lazy Prices") and narrative tests.

Same listing, header timing and raw-document cache as the 8-K adapter (``EdgarAdapter``); only
the primary document is read (no exhibits), and the body is its whole visible text, not item
excerpts, so filings can be compared word for word. Section boundaries are located and stored
as character offsets in ``meta.sections`` (10-K: items 1, 1A, 7, 7A; 10-Q: Part I item 2 =
MD&A, Part II item 1A = risk-factor updates). Many large filers (banks especially) put their
MD&A outside the item structure and cross-reference it ("see pages 50-120"); such sections are
reported missing rather than guessed.
"""

from __future__ import annotations

import re
from datetime import datetime

from newsfeed.ingest.base import RawItem
from newsfeed.ingest.edgar import EdgarAdapter, FilingRef
from newsfeed.textutil import estimate_tokens, html_to_text, truncate_tokens
from newsfeed.timeutil import IngestMode

SOURCE = "edgar_periodic"
PARSER_VERSION = "edgar-periodic/1"

# "Item 7.", "ITEM 1A. RISK FACTORS", "Item 2 - Management's ..." at the start of a line
_ITEM = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)(?![0-9a-z])", re.I)
_MDA_TITLE = re.compile(r"discussion", re.I)
SECTIONS = {
    "10-K": {
        "1": "Business",
        "1A": "Risk Factors",
        "7": "Management's Discussion and Analysis",
        "7A": "Market Risk",
    },
    "10-Q": {"I-2": "Management's Discussion and Analysis", "II-1A": "Risk Factors"},
}
MIN_SECTION_CHARS = 2000  # shorter = table-of-contents lines (page numbers) or a cross-ref


def locate_sections(text: str, form: str) -> dict[str, tuple[int, int]]:
    """Character spans [start, end) of the wanted sections of a 10-K / 10-Q text.

    Every line that starts with "Item N" is a heading; a section runs to the next heading. The
    table of contents repeats every heading with (almost) nothing in between, so among the
    occurrences of one item the longest span is the real section. 10-Q items are keyed by part
    ("I-2", "II-1A") because both parts have an item 2; "PART I/II" lines are unreliable
    (missing before the MD&A, or split over two lines), so an item 2 is the MD&A (Part I) when
    its title -- the heading line and the next two lines -- mentions "discussion", else Part
    II's item 2; item 1A exists only in Part II. Spans under ``MIN_SECTION_CHARS`` are dropped
    (cross-referenced sections)."""
    wanted = SECTIONS.get(form, {})
    lines = text.splitlines(keepends=True)
    heads: list[tuple[int, int, str]] = []  # (heading line start, body start, key)
    pos = 0
    for i, line in enumerate(lines):
        m = _ITEM.match(line)
        if m:
            code = m.group(1).upper()
            key = code
            if form == "10-Q":
                title = "".join(lines[i : i + 3])[:300]
                if code == "2":
                    key = "I-2" if _MDA_TITLE.search(title) else "II-2"
                else:
                    key = f"II-{code}" if code == "1A" else f"?-{code}"
            heads.append((pos, pos + len(line), key))
        pos += len(line)
    best: dict[str, tuple[int, int]] = {}
    for i, (_, body_start, key) in enumerate(heads):
        if key not in wanted:
            continue
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        if end - body_start > (best[key][1] - best[key][0] if key in best else -1):
            best[key] = (body_start, end)
    return {k: v for k, v in best.items() if v[1] - v[0] >= MIN_SECTION_CHARS}


class EdgarPeriodicAdapter(EdgarAdapter):
    name = SOURCE

    def fetch_filing(
        self,
        ref: FilingRef,
        tickers: list[str],
        *,
        first_seen_at: datetime,
        ingest_mode: IngestMode,
    ) -> RawItem:
        f = self._timing_and_index(ref)
        text_full = html_to_text(self._get_raw(ref, f.primary_name))
        body, truncated = truncate_tokens(text_full, self.cfg.max_tokens, self.cfg.chars_per_token)
        sections = locate_sections(body, ref.form)
        company = f.hdr.get("CONFORMED-NAME", "")
        period = f.hdr.get("PERIOD") or ""
        period_s = f"{period[:4]}-{period[4:6]}-{period[6:]}" if len(period) == 8 else "?"
        return RawItem(
            source=self.name,
            native_id=ref.accession,
            publisher="SEC EDGAR",
            url=f"{self._folder(ref.cik, ref.accession)}/{ref.accession}-index.htm",
            title=f"{company} {ref.form} for the period ended {period_s}".strip(),
            body_text=body,
            body_tokens_est=estimate_tokens(body, self.cfg.chars_per_token),
            body_truncated=truncated,
            tickers=tickers,
            published_at=f.published_at,
            published_at_raw=f"{f.hdr['ACCEPTANCE-DATETIME']} America/New_York",
            first_seen_at=first_seen_at,
            ingest_mode=ingest_mode,
            meta={
                **self._meta(ref, f),
                "items": [],
                "sections": {k: list(v) for k, v in sorted(sections.items())},
                "body_tokens_est_full": estimate_tokens(text_full, self.cfg.chars_per_token),
            },
            raw_json=self._raw_json(ref, f),
            parser_version=PARSER_VERSION,
        )
