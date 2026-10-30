"""Point-in-time S&P 500 universe: membership spells, CIK history, ingestion windows."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from newsfeed.config import ExtraMember, UniverseConfig
from newsfeed.ingest.edgar import EdgarAdapter
from newsfeed.tickers import Universe
from newsfeed.universe_pit import (
    PitUniverse,
    _cik_segments,
    build_tables,
    freeze_index_members,
    membership_intervals,
    parse_index_table,
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


def test_extra_members_join_from_their_start_and_never_overlap_the_index():
    members, sectors, _ = build_tables(FJA, WIKI, date(2019, 1, 1))
    base = PitUniverse(members, sectors)
    assert base.with_extras([]) is base
    x = ExtraMember(ticker="XTRA", cik=99, name="Extra Inc", start=date(2024, 3, 20),
                    sector="Information Technology")  # fmt: skip
    u = base.with_extras([x])
    assert base.members.height == members.height  # the stored tables are untouched
    assert u.as_universe().cik_to_tickers[99] == ["XTRA"]
    assert u.titles()[99] == "Extra Inc"
    df = pl.DataFrame({"ticker": ["XTRA", "XTRA"], "date": [date(2024, 3, 19), date(2024, 3, 20)]})
    assert u.is_member(df).to_list() == [False, True]
    w = u.ingest_windows(date(2019, 1, 1), date(2026, 1, 1), before_days=90, after_days=30)
    assert w[99] == (date(2023, 12, 21), date(2026, 1, 1))
    etf = u.sector_etf_frame({"Information Technology": "XLK"}).filter(pl.col("ticker") == "XTRA")
    assert etf["etf"].to_list() == ["XLK"]
    with pytest.raises(ValueError, match="overlaps"):
        base.with_extras([x.model_copy(update={"ticker": "META"})])
    with pytest.raises(ValueError, match="overlaps"):
        base.with_extras([x.model_copy(update={"cik": 2})])


SP400_HTML = """<table class="wikitable" id="constituents"><tr><th>Symbol</th><th>Security</th>
<th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters Location</th>
<th>SEC filings</th></tr>
<tr><td>AA</td><td>Alcoa</td><td>Materials</td><td>x</td><td>y</td>
<td><a href="https://www.sec.gov/cgi-bin/browse-edgar?CIK=AA&action=getcompany">r</a></td></tr>
<tr><td>BRK.B</td><td>Num</td><td>Financials</td><td>x</td><td>y</td>
<td><a href="https://www.sec.gov/cgi-bin/browse-edgar?CIK=0000012345&action=getcompany">r</a></td></tr>
</table>"""
SP600_HTML = """<table class="wikitable" id="constituents"><tr><th>Symbol</th><th>Security</th>
<th>GICS Sector</th><th>GICS Sub-Industry</th><th>Headquarters Location</th><th>SEC filings</th>
<th>CIK</th></tr><tr><td>AAMI</td><td>Acadian</td><td>Financials</td><td>x</td><td>y</td>
<td>r</td><td>0001748824</td></tr></table>"""


def test_parse_index_table_takes_cik_from_column_or_numeric_link_only():
    a, b = parse_index_table(SP400_HTML)
    assert (a["ticker"], a["cik"]) == ("AA", None)  # the link names a ticker, not a CIK
    assert (b["ticker"], b["cik"]) == ("BRK-B", 12345)  # normalised ticker, numeric CIK in link
    (c,) = parse_index_table(SP600_HTML)
    assert (c["ticker"], c["cik"], c["sector"]) == ("AAMI", 1748824, "Financials")


def test_freeze_excludes_current_sp500_members_and_ends_at_a_later_promotion():
    day = date(2026, 7, 1)
    sp500 = pl.DataFrame(
        {"ticker": ["IN", "UP", "OLD"], "cik": [1, 2, 3],
         "start": [date(2019, 1, 1), date(2026, 10, 1), date(2019, 1, 1)],
         "end": [None, None, date(2024, 6, 24)]},
        schema={"ticker": pl.String, "cik": pl.Int64, "start": pl.Date, "end": pl.Date},
    )  # fmt: skip
    rows = [
        {"ticker": t, "name": t, "sector": "Industrials", "cik": c}
        for t, c in [("IN", 1), ("UP", 2), ("OLD", 3), ("NEW", None), ("GONE", None), ("NQ", 9)]
    ]
    df, unmapped, excluded = freeze_index_members(
        rows, "sp400", day, {"NEW": 7}, sp500, exclude_ciks={9}
    )
    assert excluded == ["IN", "NQ"] and unmapped == ["GONE"]  # in the S&P 500 / an extra
    got = {r["ticker"]: r for r in df.to_dicts()}
    assert got["UP"]["end"] == date(2026, 10, 1)  # promoted later: stops where that spell starts
    assert got["OLD"]["end"] is None  # left the S&P 500 before the freeze: a plain member
    assert got["NEW"]["cik"] == 7 and all(r["start"] == day for r in got.values())
    # the frozen rows load into the universe without any overlapping spell
    members, sectors, _ = build_tables(FJA, WIKI, date(2019, 1, 1))
    PitUniverse(sp500.with_columns(pl.col("ticker").alias("name")).select(members.columns),
                sectors.head(0)).with_extras(
        [ExtraMember(**{k: v for k, v in r.items() if v is not None}) for r in got.values()]
    )  # fmt: skip


def test_extra_members_may_follow_but_not_overlap_an_sp500_spell():
    members, sectors, _ = build_tables(FJA, WIKI, date(2019, 1, 1))
    base = PitUniverse(members, sectors)
    old = dict(ticker="OLD", cik=3, name="Old", sector="Industrials")  # left 2020-03-02
    base.with_extras([ExtraMember(**old, start=date(2021, 1, 1))])
    with pytest.raises(ValueError, match="overlaps"):
        base.with_extras([ExtraMember(**old, start=date(2020, 1, 1))])
    # a spell that ends before an S&P 500 spell starts is fine
    new = dict(ticker="NEW", cik=5, name="New", sector="Industrials")  # joined 2022-08-01
    base.with_extras([ExtraMember(**new, start=date(2021, 1, 1), end=date(2022, 8, 1))])
    with pytest.raises(ValueError, match="overlaps"):
        base.with_extras([ExtraMember(**new, start=date(2021, 1, 1))])


def test_extra_members_file_loads_and_its_content_is_hashed(tmp_path):
    f = tmp_path / "extras.csv"
    f.write_text(
        "ticker,cik,name,start,end,sector,index,note\n"
        "AAA9,999,A Co,2026-07-01,,Industrials,sp400,x\n"
        "BBB9,998,B Co,2026-07-01,2026-10-01,Financials,sp600,y\n",
        encoding="utf-8",
    )
    u = UniverseConfig(mode="sp500_pit", extra_members_file=f)
    a, b = u.all_extra_members()
    assert (a.cik, a.end, b.end) == (999, None, date(2026, 10, 1))
    h1 = u.hashable()
    assert "extra_members_file" not in h1 and "extra_members_file_sha256" in h1
    f.write_text(f.read_text(encoding="utf-8").replace("A Co", "A Corp"), encoding="utf-8")
    assert u.hashable() != h1  # a changed list changes the ids; a moved file would not


def test_empty_extras_leave_universe_hashes_unchanged():
    u = UniverseConfig(mode="sp500_pit")
    assert "extra_members" not in u.hashable() and "pit_dir" not in u.hashable()
    with_x = u.model_copy(update={"extra_members": [ExtraMember(
        ticker="XTRA", cik=99, name="Extra Inc", start=date(2024, 3, 20), sector="Industrials"
    )]})  # fmt: skip
    assert with_x.hashable()["extra_members"][0]["ticker"] == "XTRA"


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
