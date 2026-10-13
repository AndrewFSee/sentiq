"""Earnings-call transcripts adapter: conservative date-only timing, text cleaning, company
mapping through the current ticker map, membership windows, live polling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from newsfeed.ingest.runner import build_row
from newsfeed.ingest.transcripts import TranscriptsAdapter, end_of_day_ny, transcript_text
from newsfeed.tickers import TickerMap, Universe


def para(n: int, speaker: str, content: str) -> str:
    c = content.replace("'", "''")
    return f"{{'paragraph_number': {n}, 'speaker': '{speaker}', 'content': '{c}'}}"


def write_fixture(path) -> None:
    rows = [
        # Tuesday 2026-07-28 call; current symbol of the company is "NEWT" (renamed from OLDT)
        ("NEWT", 2026, 2, "2026-07-28", 1,
         [para(1, "AI Insights", ""), para(2, "Operator", "Welcome to the call."),
          para(3, "Jane CFO", "Revenue grew 12%."), para(4, "Disclaimer*", "Legal text.")]),
        ("ZZZZ", 2026, 2, "2026-07-28", 2, [para(1, "Operator", "Not in universe.")]),
        ("NEWT", 2026, 1, "2026-04-28", 3, [para(1, "Operator", "Earlier quarter.")]),
        ("NEWT", 2026, 3, "2026-10-08", 4, [para(1, "Operator", "Latest quarter.")]),
    ]  # fmt: skip
    con = duckdb.connect()
    values = ",".join(
        f"('{s}', {y}, {q}, '{d}', {i}, [{', '.join(ps)}])" for s, y, q, d, i, ps in rows
    )
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES "
        + values
        + ") v(symbol, fiscal_year, fiscal_quarter, report_date, transcripts_id, transcripts)"
    )
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")


@pytest.fixture
def adapter(cfg, tmp_path):
    p = tmp_path / "t.parquet"
    write_fixture(p)
    tcfg = cfg.sources.transcripts.model_copy(update={"path": p})
    tmap = TickerMap(ticker_to_cik={"NEWT": 42, "ZZZZ": 99})
    universe = Universe({42: ["NEWT", "OLDT"]})
    return TranscriptsAdapter(
        tcfg, universe, tmap, clock=lambda: datetime(2026, 10, 9, 12, tzinfo=UTC)
    )


def test_text_drops_placeholders_and_keeps_speakers():
    t = transcript_text(
        [{"paragraph_number": 2, "speaker": "B", "content": "two"},
         {"paragraph_number": 1, "speaker": "AI Insights", "content": ""},
         {"paragraph_number": 3, "speaker": "Disclaimer*", "content": "x"}],
        {"AI Insights", "Disclaimer*"},
    )  # fmt: skip
    assert t == "B: two"


def test_backfill_timing_mapping_and_filters(adapter, cfg, calendar):
    start, end = datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)
    items = list(adapter.backfill(start, end, skip=lambda _id: False))
    assert [i.native_id for i in items] == ["defeatbeta:1"]  # ZZZZ not in universe
    it = items[0]
    assert it.tickers == ["NEWT", "OLDT"]  # all tickers of the company (fanned by membership)
    assert "AI Insights" not in it.body_text and "Disclaimer" not in it.body_text
    assert it.body_text.startswith("Operator: Welcome")
    assert it.published_at == end_of_day_ny(datetime(2026, 7, 28).date())
    assert it.published_at_raw == "2026-07-28" and it.meta["timing"] == "date_only_end_of_day"
    row = build_row(
        it,
        run_id="run",
        calendar=calendar,
        backfill_lag=timedelta(seconds=cfg.timing.lag_for("transcripts")),
        ingested_at=datetime(2026, 10, 9, tzinfo=UTC),
    )
    # Tuesday call (any time of day) -> tradable Wednesday 09:30 ET (13:30 UTC in summer)
    assert row["tradable_at"] == datetime(2026, 7, 29, 13, 30, tzinfo=UTC)


def test_membership_windows_and_skip(adapter):
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC)
    adapter.cik_windows = {42: (datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))}
    got = [i.native_id for i in adapter.backfill(start, end, skip=lambda _id: False)]
    assert got == ["defeatbeta:1"]
    assert list(adapter.backfill(start, end, skip=lambda _id: True)) == []


def test_poll_is_live_with_first_seen_now(adapter):
    items = list(adapter.poll(skip=lambda _id: False))
    assert [i.native_id for i in items] == ["defeatbeta:4"]  # within poll_lookback_days
    assert items[0].ingest_mode == "live"
    assert items[0].first_seen_at == datetime(2026, 10, 9, 12, tzinfo=UTC)
    assert adapter.last_activity_at == end_of_day_ny(datetime(2026, 10, 8).date())
