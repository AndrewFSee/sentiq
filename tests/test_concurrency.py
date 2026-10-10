"""Two processes using the same data directory (e.g. `collect` loop + a `backfill`)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta

from newsfeed.ingest.base import RawItem
from newsfeed.ingest.runner import build_row
from newsfeed.storage import FileLock, Store

CHILD = textwrap.dedent(
    """
    import sys
    from datetime import UTC, datetime, timedelta
    from newsfeed.config import load_config
    from newsfeed.ingest.base import RawItem
    from newsfeed.ingest.runner import build_row
    from newsfeed.storage import Store
    from newsfeed.timeutil import get_calendar

    cfg = load_config(root=sys.argv[1])
    cal = get_calendar()
    with Store(cfg.paths) as s:
        run = s.start_run("child", cfg, {})
        t = datetime(2026, 7, 28, 15, tzinfo=UTC)
        rows = []
        for k in range(20):
            it = RawItem(source="test", native_id=f"c{k}", url=None, title="t", body_text="b",
                         body_tokens_est=1, body_truncated=False, tickers=[],
                         published_at=t + timedelta(minutes=k), published_at_raw="x",
                         first_seen_at=t + timedelta(minutes=k), ingest_mode="live", meta={},
                         raw_json="{}", parser_version="t")
            rows.append(build_row(it, run_id=run, calendar=cal, backfill_lag=timedelta(0),
                                  ingested_at=t))
        s.write_items(rows, run)
        s.finish_run(run, "ok", {})
    print("child-ok")
    """
)


def test_second_process_can_write_while_first_store_is_open(cfg, store, calendar, tmp_path):
    # parent holds a Store open (like a running collect loop) and writes too
    parent_run = store.start_run("parent", cfg, {})
    t = datetime(2026, 7, 28, 15, tzinfo=UTC)
    it = RawItem(
        source="test",
        native_id="p0",
        url=None,
        title="t",
        body_text="b",
        body_tokens_est=1,
        body_truncated=False,
        tickers=[],
        published_at=t,
        published_at_raw="x",
        first_seen_at=t,
        ingest_mode="live",
        meta={},
        raw_json="{}",
        parser_version="t",
    )
    store.write_items(
        [
            build_row(
                it, run_id=parent_run, calendar=calendar, backfill_lag=timedelta(0), ingested_at=t
            )
        ],
        parent_run,
    )
    out = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "child-ok" in out.stdout
    store.finish_run(parent_run, "ok", {})
    store.refresh_views()
    assert store.query("SELECT count(*) AS n FROM items")["n"][0] == 21
    assert store.ops_query("SELECT count(*) AS n FROM runs")["n"][0] == 2


def test_file_lock_is_exclusive_and_breaks_stale(tmp_path):
    p = tmp_path / "x.lock"
    a, b = FileLock(p), FileLock(p)
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()
    p.write_text("{}")  # left behind by a killed process
    assert not FileLock(p, stale_after=60).acquire()
    old = time.time() - 120
    os.utime(p, (old, old))
    stale = FileLock(p, stale_after=60)
    assert stale.acquire()  # older than 60s -> broken
    stale.release()


def test_view_prefers_live_row_over_backfill_duplicate(store: Store, calendar):
    """If a race stores both a backfill and a live copy, the truthful live timing wins."""
    t = datetime(2026, 7, 28, 15, tzinfo=UTC)

    def row(mode: str, seen: datetime, run: str):
        it = RawItem(
            source="test",
            native_id="dup",
            url=None,
            title="t",
            body_text="b",
            body_tokens_est=1,
            body_truncated=False,
            tickers=[],
            published_at=t,
            published_at_raw="x",
            first_seen_at=seen,
            ingest_mode=mode,
            meta={},
            raw_json="{}",
            parser_version="t",
        )
        return build_row(
            it, run_id=run, calendar=calendar, backfill_lag=timedelta(seconds=60), ingested_at=seen
        )

    store.write_items([row("backfill", t + timedelta(minutes=1), "r1")], "r1")
    store._known_ids = set()  # simulate the racing writer
    store.write_items([row("live", t + timedelta(minutes=5), "r2")], "r2")
    df = store.query("SELECT ingest_mode, tradable_at FROM items")
    assert df.height == 1
    assert df["ingest_mode"][0] == "live"
    assert df["tradable_at"][0] == t + timedelta(minutes=5)
