"""Record real EDGAR responses into tests/fixtures/edgar/ (mirrors URL paths).

Run manually (needs network + SEC_USER_AGENT in .env):
    uv run python scripts/record_edgar_fixtures.py

Tests replay these files through httpx.MockTransport -- no network in tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from newsfeed.config import load_config, load_secrets
from newsfeed.ingest.edgar import parse_index
from newsfeed.ingest.http import HttpClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "edgar"

APPLE = 320193
# Chosen to cover: EDT vs EST, pre-market, after-5:30pm (filingDate != acceptance date),
# earnings with EX-99.1, and a filer-agent accession prefix.
ACCESSIONS = [
    ("0000320193-26-000018", "aapl-20260730.htm"),  # 16:30 EDT earnings, EX-99.1
    ("0000320193-26-000005", "aapl-20260129.htm"),  # 16:30 EST earnings
    ("0001140361-25-025275", "ef20051741_8k.htm"),  # 21:18 EDT; filingDate next day
    ("0001140361-24-040659", "ef20035718_8k.htm"),  # 09:06 EDT pre-market, Item 7.01
]
KEEP_TICKERS = {"AAPL", "MSFT", "GOOGL", "GOOG", "NVDA"}


def save(url: str, body: bytes) -> None:
    p = urlsplit(url)
    dest = OUT / p.netloc / p.path.lstrip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    print(f"saved {dest.relative_to(ROOT)} ({len(body):,} bytes)")


def main() -> None:
    cfg = load_config()
    ecfg = cfg.sources.edgar
    ua = load_secrets().require_sec_user_agent()
    with HttpClient(user_agent=ua, rate_per_sec=4) as http:
        tm = http.get(ecfg.ticker_map_url).json()
        trimmed = {k: v for k, v in tm.items() if v["ticker"] in KEEP_TICKERS}
        save(ecfg.ticker_map_url, json.dumps(trimmed, indent=1).encode())

        sub_url = f"{ecfg.submissions_base_url}/CIK{APPLE:010d}.json"
        save(sub_url, http.get(sub_url).content)

        for acc, primary in ACCESSIONS:
            folder = f"{ecfg.archives_base_url}/{APPLE}/{acc.replace('-', '')}"
            for name in (f"{acc}.hdr.sgml", f"{acc}-index.htm", primary):
                url = f"{folder}/{name}"
                body = http.get(url).content
                save(url, body)
                if name.endswith("-index.htm"):
                    docs, _ = parse_index(body)
            for d in docs:  # exhibits the adapter will request
                if any(d.doc_type.startswith(p) for p in ecfg.exhibit_type_prefixes):
                    url = f"{folder}/{d.filename}"
                    save(url, http.get(url).content)


if __name__ == "__main__":
    main()
