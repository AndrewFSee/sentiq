"""Record one real Massive /v2/reference/news page as a test fixture (needs MASSIVE_API_KEY).

The repo is public and article text is licensed, so titles, summaries, authors, URLs and the
vendor "insights" reasoning are replaced with placeholders; ids, timestamps, tickers,
publishers and the field layout are kept exactly as returned.

    uv run python scripts/record_massive_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from newsfeed.config import load_config, load_secrets
from newsfeed.ingest.http import HttpClient

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "news" / "massive_page.json"
TEXT_FIELDS = ("title", "description", "author", "article_url", "amp_url", "image_url")


def main() -> None:
    cfg = load_config(None).sources.news_api
    http = HttpClient(
        user_agent="sentiq-research",
        rate_per_sec=cfg.massive_requests_per_minute / 60,
        headers={"Authorization": f"Bearer {load_secrets().massive_api_key}"},
    )
    url = (
        f"{cfg.massive_base_url}/v2/reference/news?published_utc.gte=2026-07-01T13:00:00Z"
        "&published_utc.lt=2026-07-01T16:00:00Z&order=asc&sort=published_utc&limit=5"
    )
    page = http.get(url).json()
    for i, r in enumerate(page.get("results") or []):
        for f in TEXT_FIELDS:
            if f in r:
                r[f] = f"<redacted {f} {i}>"
        for ins in r.get("insights") or []:
            ins["sentiment_reasoning"] = "<redacted>"
    page.pop("request_id", None)
    if page.get("next_url"):
        page["next_url"] = "https://api.massive.com/v2/reference/news?cursor=REDACTED"
    OUT.write_text(json.dumps(page, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(page.get('results') or [])} records)")


if __name__ == "__main__":
    main()
