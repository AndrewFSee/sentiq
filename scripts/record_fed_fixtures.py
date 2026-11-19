"""Re-record the Fed test fixtures under tests/fixtures/fed/ (needs network).

Saves: the two RSS feeds (verbatim), the three JSON listings trimmed to their newest entries plus
the entries of the recorded pages, and one page of each kind (statement, minutes release +
minutes page, speech, testimony) trimmed to the #article element (the parser only reads that).

    python scripts/record_fed_fixtures.py
"""

from __future__ import annotations

import json
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
    print(f"recorded {len(FEEDS) + len(LISTINGS) + len(PAGES)} files in {OUT}")


if __name__ == "__main__":
    main()
