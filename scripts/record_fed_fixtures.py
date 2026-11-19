"""Re-record the Fed test fixtures under tests/fixtures/fed/ (needs network).

Saves: the two RSS feeds (verbatim), the three JSON listings trimmed to their newest entries plus
the entries of the recorded pages, and one page of each kind (statement, minutes release +
minutes page, speech, testimony) trimmed to the #article element (the parser only reads that),
plus one press conference (FINAL PDF, the same-day PRELIMINARY from the Wayback Machine, their
Last-Modified headers) and the FOMC calendar page reduced to its press-conference links.

    python scripts/record_fed_fixtures.py [--presconf-only]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "fed"
BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "newsfeed-research/0.1"}
FEEDS = ["/feeds/press_monetary.xml", "/feeds/speeches_and_testimony.xml"]
LISTINGS = {
    "press": "/json/ne-press.json",
    "speech": "/json/ne-speeches.json",
    "testimony": "/json/ne-testimony.json",
}
PAGES = {
    "statement.htm": "/newsevents/pressreleases/monetary20261028a.htm",
    "minutes_release.htm": "/newsevents/pressreleases/monetary20261118a.htm",
    "minutes_page.htm": "/monetarypolicy/fomcminutes20261028.htm",
    "speech.htm": "/newsevents/speech/waller20261117a.htm",
    "testimony.htm": "/newsevents/testimony/warsh20260714a.htm",
}
KEEP_NEWEST = 12
# press conference of 2026-10-28: today's FINAL from the Fed, and the same-day PRELIMINARY
# (opening statement only) from the Wayback Machine's first capture of the same URL
PRESCONF_DATE = "20261028"
PRESCONF_PDF = f"/mediacenter/files/FOMCpresconf{PRESCONF_DATE}.pdf"
PRESCONF_WAYBACK = f"https://web.archive.org/web/20261028200000id_/{BASE}{PRESCONF_PDF}"
CALENDAR = "/monetarypolicy/fomccalendars.htm"


def trim_page(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    art = soup.find(id="article")
    if art is None:
        sys.exit("page without #article -- the site layout changed; check the parser")
    return f'<!doctype html><html><body><div id="content">{art}</div></body></html>\n'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = set(PAGES.values())
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as c:
        for path in FEEDS:
            (OUT / path.rsplit("/", 1)[-1]).write_bytes(c.get(BASE + path).content)
        for kind, path in LISTINGS.items():
            data = json.loads(c.get(BASE + path).content.decode("utf-8-sig"))
            entries = [e for e in data if e.get("l")]
            keep = entries[:KEEP_NEWEST] + [e for e in entries[KEEP_NEWEST:] if e["l"] in pages]
            text = json.dumps(keep + [d for d in data if "updateDate" in d], indent=1)
            (OUT / f"ne-{kind}.json").write_text(text + "\n", encoding="utf-8")
        for name, path in PAGES.items():
            (OUT / name).write_text(trim_page(c.get(BASE + path).content), encoding="utf-8")
        record_presconf(c)
    print(f"recorded {len(FEEDS) + len(LISTINGS) + len(PAGES) + 4} files in {OUT}")


def record_presconf(c: httpx.Client) -> None:
    # calendar page reduced to its press-conference links (all the parser reads)
    cal = c.get(BASE + CALENDAR).content.decode("utf-8", errors="replace")
    links = sorted(set(re.findall(r"/monetarypolicy/fomcpresconf\d{8}\.htm", cal)))
    body = "\n".join(f'<a href="{x}">Press Conference</a>' for x in links)
    (OUT / "fomccalendars.htm").write_text(f"<html><body>\n{body}\n</body></html>\n")
    final = c.get(BASE + PRESCONF_PDF)
    prelim = c.get(PRESCONF_WAYBACK)
    (OUT / "presconf_final.pdf").write_bytes(final.content)
    (OUT / "presconf_prelim.pdf").write_bytes(prelim.content)
    lm = {
        "date": PRESCONF_DATE,
        "final_last_modified": final.headers["last-modified"],
        # the archived response keeps the Fed's header as x-archive-orig-last-modified
        "prelim_last_modified": prelim.headers.get("x-archive-orig-last-modified"),
    }
    (OUT / "presconf.json").write_text(json.dumps(lm, indent=1) + "\n")


if __name__ == "__main__":
    if "--presconf-only" in sys.argv:  # keep the other fixtures (tests pin their contents)
        with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as c:
            record_presconf(c)
    else:
        main()
