"""Warm the SEC companyfacts share-count cache for every PIT universe CIK (used for size
buckets in eval), so `newsfeed eval` doesn't make ~650 SEC requests inline.

    uv run python scripts/prefetch_shares.py
"""

from __future__ import annotations

import logging

from newsfeed.config import load_config, load_secrets
from newsfeed.eval.caps import load_shares
from newsfeed.ingest.http import HttpClient
from newsfeed.universe_pit import PitUniverse


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(None)
    e = cfg.sources.edgar
    http = HttpClient(
        user_agent=load_secrets().require_sec_user_agent(),
        rate_per_sec=e.max_requests_per_second,
        timeout=e.timeout_seconds,
        max_retries=e.max_retries,
    )
    ciks = sorted({int(c) for c in PitUniverse.load_for(cfg.universe).members["cik"]})
    empty, failed = [], []
    for i, cik in enumerate(ciks, 1):
        try:
            if load_shares(http, cik, cfg.paths.ref_dir / "sec_shares").height == 0:
                empty.append(cik)
        except Exception as exc:  # e.g. 404: no XBRL facts for this CIK
            failed.append((cik, repr(exc)[:120]))
        if i % 100 == 0:
            print(f"{i}/{len(ciks)}", flush=True)
    print(f"done: {len(ciks)} CIKs, {len(empty)} without share counts, {len(failed)} failed")
    print("no share counts:", empty)
    print("failed:", failed)


if __name__ == "__main__":
    main()
