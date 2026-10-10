"""Point-in-time S&P 500 universe: membership spells, CIK history, ingestion windows."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from newsfeed.ingest.edgar import EdgarAdapter
from newsfeed.tickers import Universe
from newsfeed.universe_pit import (
    PitUniverse,
    _cik_segments,
    build_tables,
    membership_intervals,
    parse_wiki_table,
)

FJA = (
    b"date,tickers\n"
    b'2019-01-02,"AAA,FB,OLD,BRK.B"\n'
    b'2020-03-02,"AAA,FB,BRK.B"\n'  # OLD removed
    b'2022-06-09,"AAA,META,BRK.B"\n'  # FB renamed META
)


def wiki_rows(snapshot: date, rows: list[tuple[str, str, str]]) -> list[dict]:
    return [
        {"snapshot": snapshot, "revid": 1, "ticker": t, "name": n, "sector": "IT", "cik": c}
        for t, n, c in rows
    ]


WIKI = pl.DataFrame(
    wiki_rows(
        date(2019, 1, 1),
        [
            ("AAA", "Aaa Inc", "0000000001"),
            ("FB", "Facebook", "0000000002"),
            ("OLD", "Old Co", "0000000003"),
            ("BRK.B", "Berkshire", "0000000004"),
        ],
    )
    + wiki_rows(
        date(2021, 1, 1),
        [
            ("AAA", "Aaa Holdings", "0000000011"),  # new CIK (reorg)
            ("FB", "Facebook", "0000000002"),
            ("BRK.B", "Berkshire", "4"),
        ],
    )
    + wiki_rows(
        date(2022, 7, 1),
        [("AAA", "Aaa Holdings", "11"), ("META", "Meta", "2"), ("BRK.B", "Berkshire", "4")],
    )
    + wiki_rows(
        date(2022, 8, 1),
        [
            ("AAA", "Aaa Holdings", "11"),
            ("META", "Meta", "2"),
            ("BRK.B", "Berkshire", "4"),
            ("NEW", "Newco", "5"),
        ],
    )
)


def test_membership_intervals():
    snaps = [
        (date(2020, 1, 1), {"A", "B"}),
        (date(2020, 2, 1), {"A"}),
        (date(2020, 3, 1), {"A", "B"}),
    ]
    got = membership_intervals(snaps).sort("ticker", "start").rows()
    assert got == [
        ("A", date(2020, 1, 1), None),
        ("B", date(2020, 1, 1), date(2020, 2, 1)),
        ("B", date(2020, 3, 1), None),
    ]


def test_cik_segments_split_at_change():
    hist = [
        {"valid_from": date(2019, 1, 1), "cik": 1, "name": "a"},
        {"valid_from": date(2021, 1, 1), "cik": 11, "name": "b"},
    ]
    segs = _cik_segments("AAA", date(2019, 1, 2), None, hist)
    assert [(s["cik"], s["start"], s["end"]) for s in segs] == [
        (1, date(2019, 1, 2), date(2021, 1, 1)),
        (11, date(2021, 1, 1), None),
    ]
    # spell added between snapshots: falls back to the ticker's first snapshot
    late = [{"valid_from": date(2022, 8, 1), "cik": 5, "name": "n"}]
    assert _cik_segments("NEW", date(2022, 7, 15), None, late)[0]["cik"] == 5


def test_build_tables():
    members, _sectors, stats = build_tables(FJA, WIKI, date(2019, 1, 1))
    assert stats["unmapped_spells"] == 0
    rows = {(r["ticker"], r["cik"]): r for r in members.to_dicts()}
    assert rows[("OLD", 3)]["end"] == date(2020, 3, 2)
    assert rows[("FB", 2)]["end"] == date(2022, 6, 9) and rows[("META", 2)]["end"] is None
    assert rows[("AAA", 1)]["end"] == date(2021, 1, 1) and rows[("AAA", 11)]["start"] == date(
        2021, 1, 1
    )
    assert ("BRK-B", 4) in rows  # ticker normalised
    # after the last fja snapshot, membership comes from Wikipedia snapshots
    assert rows[("NEW", 5)]["start"] == date(2022, 8, 1)


def test_pit_universe_views():
    members, sectors, _ = build_tables(FJA, WIKI, date(2019, 1, 1))
    u = PitUniverse(members, sectors)
    assert u.as_universe().cik_to_tickers[2] == ["FB", "META"]
    df = pl.DataFrame(
        {
            "ticker": ["FB", "META", "OLD", "OLD"],
            "date": [date(2022, 6, 8), date(2022, 6, 8), date(2020, 3, 1), date(2020, 3, 2)],
        }
    )
    assert u.is_member(df).to_list() == [True, False, True, False]
    w = u.ingest_windows(date(2019, 1, 1), date(2026, 1, 1), before_days=90, after_days=30)
    # OLD: left 2020-03-02 -> window ends 30 days later; starts at the global start
    assert w[3] == (date(2019, 1, 1), date(2020, 4, 1))
    # AAA's old CIK covers the *ticker's* whole window, not just its own dated segment
    assert w[1] == w[11] == (date(2019, 1, 1), date(2026, 1, 1))
    # NEW joined 2022-08-01 -> 90 days of history before
    assert w[5][0] == date(2022, 5, 3)


def test_parse_wiki_table_both_header_styles():
    for sym in ("Symbol", "Ticker symbol"):
        html = (
            f"<table class='wikitable'><tr><th>{sym}</th><th>Security</th><th>SEC filings</th>"
            "<th>GICS Sector</th><th>CIK</th></tr><tr><td>MMM</td><td>3M</td><td>reports</td>"
            "<td>Industrials</td><td>0000066740</td></tr></table>"
        )
        assert parse_wiki_table(html) == [
            {"ticker": "MMM", "name": "3M", "sector": "Industrials", "cik": "0000066740"}
        ]


def test_edgar_backfill_respects_cik_windows(cfg, http, ticker_map, store, clock):
    adapter = EdgarAdapter(
        cfg.sources.edgar, http, Universe.build(["AAPL"], ticker_map), store.raw, clock=clock
    )
    start, end = datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)
    adapter.cik_windows = {320193: (datetime(2026, 7, 31, tzinfo=UTC), end)}
    assert list(adapter.backfill(start, end, skip=store.has_item)) == []  # 8-K on 07-30
    adapter.cik_windows = {320193: (datetime(2026, 7, 30, tzinfo=UTC), end)}
    assert [i.native_id for i in adapter.backfill(start, end, skip=store.has_item)] == [
        "0000320193-26-000018"
    ]
    adapter.cik_windows = {999: (start, end)}  # CIK not covered -> nothing listed
    assert list(adapter.backfill(start, end, skip=store.has_item)) == []
