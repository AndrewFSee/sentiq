from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from newsfeed.tickers import TickerHolders, TickerLinker, Universe, clean_company_name

UNIVERSE = Universe(
    {
        320193: ["AAPL"],
        789019: ["MSFT"],
        1652044: ["GOOG", "GOOGL"],
        1045810: ["NVDA"],
        19617: ["JPM"],
        104169: ["WMT"],
    }
)
ALIASES = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "GOOGL": ["Alphabet", "Google"],
    "JPM": ["JPMorgan", "JPMorgan Chase"],
    "WMT": ["Walmart"],
}
TITLES = {320193: "Apple Inc.", 1045810: "NVIDIA CORP", 19617: "JPMORGAN CHASE & CO"}


@pytest.fixture
def linker() -> TickerLinker:
    return TickerLinker(UNIVERSE, ALIASES, TITLES)


def by_ticker(links):
    return {lk.ticker: lk for lk in links}


def test_source_tags_are_authoritative(linker):
    got = by_ticker(linker.link(title="x", body="y", source_tickers=["AAPL", "TSM"]))
    assert set(got) == {"AAPL"}  # TSM: not in universe
    assert got["AAPL"].method == "source" and got["AAPL"].confidence == 1.0


def test_exchange_tag_and_cashtag(linker):
    got = by_ticker(
        linker.link(
            title="Chip rally",
            body="Nvidia Corp (NASDAQ: NVDA) rose; $MSFT flat; $TSM up. NYSE:JPM fell.",
            source_tickers=[],
        )
    )
    assert got["NVDA"].method == "exchange_tag"
    assert got["MSFT"].method == "cashtag"
    assert got["JPM"].method == "exchange_tag"
    assert "TSM" not in got


def test_names_are_case_sensitive_and_word_bounded(linker):
    body = "Walmart signed deals with apple orchards; pineapple prices; Applesauce Co."
    got = by_ticker(linker.link(title="Produce", body=body, source_tickers=[]))
    assert set(got) == {"WMT"}
    assert got["WMT"].method == "name_body" and got["WMT"].confidence == 0.6


def test_name_in_title_scores_higher_and_counts_mentions(linker):
    got = by_ticker(
        linker.link(
            title="Apple results", body="Apple said... Apple also said...", source_tickers=[]
        )
    )
    assert got["AAPL"].method == "name_title"
    assert got["AAPL"].in_title and got["AAPL"].n_mentions == 3


def test_company_name_links_all_share_classes(linker):
    got = by_ticker(linker.link(title="Google cloud deal", body="", source_tickers=[]))
    assert set(got) == {"GOOG", "GOOGL"}
    assert got["GOOG"].cik == got["GOOGL"].cik == 1652044


def test_cashtag_links_only_written_share_class(linker):
    got = by_ticker(linker.link(title="", body="$GOOGL up", source_tickers=[]))
    assert set(got) == {"GOOGL"}


def test_sec_title_alias_added(linker):
    got = by_ticker(linker.link(title="Nvidia unveils chip", body="", source_tickers=[]))
    assert "NVDA" in got  # from 'NVIDIA CORP' -> 'Nvidia'


def test_best_method_wins_and_mentions_accumulate(linker):
    got = by_ticker(
        linker.link(title="Apple (NASDAQ: AAPL)", body="Apple. $AAPL", source_tickers=["AAPL"])
    )
    assert got["AAPL"].method == "source"
    assert got["AAPL"].n_mentions == 4  # exchange tag + title name + body name + cashtag


def test_linker_id_changes_with_inputs():
    a = TickerLinker(UNIVERSE, ALIASES, TITLES).linker_id
    b = TickerLinker(UNIVERSE, {**ALIASES, "MSFT": ["Microsoft", "MSFT Corp"]}, TITLES).linker_id
    assert a != b and a == TickerLinker(UNIVERSE, ALIASES, TITLES).linker_id


def test_alias_for_unknown_ticker_rejected():
    with pytest.raises(KeyError):
        TickerLinker(UNIVERSE, {"ZZZZ": ["Zed"]})


@pytest.mark.parametrize(
    "title,clean",
    [
        ("Apple Inc.", "Apple"),
        ("MICROSOFT CORP", "Microsoft"),
        ("BERKSHIRE HATHAWAY INC", "Berkshire Hathaway"),
        ("Meta Platforms, Inc.", "Meta"),
        ("AMAZON COM INC", "Amazon"),
        ("JOHNSON & JOHNSON", "Johnson & Johnson"),
        ("Inc.", None),
    ],
)
def test_clean_company_name(title, clean):
    assert clean_company_name(title) == clean


def test_social_media_boilerplate_is_not_a_mention():
    lk = TickerLinker(Universe({1: ["META"], 2: ["WMT"]}), {"META": ["Meta", "Facebook"]}, None)
    body = (
        "More information about Walmart can be found by visiting corporate.walmart.com, "
        "on Facebook at facebook.com/walmart and on LinkedIn at linkedin.com/company/walmart."
    )
    assert lk.link(title="", body=body, source_tickers=[]) == []
    assert lk.link(title="", body="Facebook parent Meta said", source_tickers=[])


def test_source_only_sources_skip_text_matching():
    lk = TickerLinker(
        Universe({1: ["META"], 2: ["WMT"]}), {"WMT": ["Walmart"]}, None, ["transcripts"]
    )
    text = "Our customers include Walmart. Walmart grew. $WMT"
    kept = lk.link(title=None, body=text, source_tickers=["META"], source="transcripts")
    assert [(x.ticker, x.method) for x in kept] == [("META", "source")]
    news = lk.link(title=None, body=text, source_tickers=["META"], source="news_api")
    assert {x.ticker for x in news} == {"META", "WMT"}
    assert (
        lk.linker_id
        != TickerLinker(Universe({1: ["META"], 2: ["WMT"]}), {"WMT": ["Walmart"]}).linker_id
    )


# ------------------------------------------------- tickers held by two companies (2026-10-30)
# Shapes of the real spells: Trane kept Ingersoll-Rand plc's CIK (IR -> TT), Gardner Denver took
# IR; Truist kept BB&T's CIK (BBT -> TFC), Beacon Financial (H3 extra) took BBT in 2026-07;
# Disney's new CIK is listed only from 2020-10 (Wikipedia lag) although it filed from 2019;
# ViacomCBS files under CBS's CIK while VIAC is listed under Viacom's until 2020-10.
TRANE, GD, TRUIST, BEACON = 1466258, 1699150, 92230, 1108134
DIS_OLD, DIS_NEW, CBS, VIACOM = 1001039, 1744489, 813828, 1339947
SPELLS = [
    ("IR", TRANE, date(2010, 11, 17), date(2020, 4, 1)),
    ("TT", TRANE, date(2020, 3, 3), None),
    ("IR", GD, date(2020, 4, 1), None),
    ("BBT", TRUIST, date(1997, 12, 4), date(2019, 12, 9)),
    ("TFC", TRUIST, date(2019, 12, 9), None),
    ("BBT", BEACON, date(2026, 7, 1), None),
    ("DIS", DIS_OLD, date(1996, 1, 2), date(2020, 10, 1)),
    ("DIS", DIS_NEW, date(2020, 10, 1), None),
    ("CBS", CBS, date(1996, 1, 2), date(2019, 12, 5)),
    ("VIAC", CBS, date(2020, 10, 1), date(2022, 2, 17)),
    ("VIAB", VIACOM, date(2006, 1, 3), date(2019, 12, 5)),
    ("VIAC", VIACOM, date(2019, 12, 5), date(2020, 10, 1)),
    ("AAPL", 320193, date(1982, 11, 30), None),
]
HOLDERS = TickerHolders.from_spells(SPELLS)


def test_holder_of_shared_ticker_by_date():
    assert set(HOLDERS.spells) == {"IR", "BBT", "DIS", "VIAC"}  # AAPL etc. are not shared
    assert HOLDERS.holder("AAPL", date(2023, 1, 1)) is None
    assert HOLDERS.holder("IR", date(2019, 6, 1)) == TRANE
    assert HOLDERS.holder("IR", date(2020, 3, 31)) == TRANE  # end is exclusive
    assert HOLDERS.holder("IR", date(2020, 4, 1)) == GD
    assert HOLDERS.holder("IR", None) == GD  # latest spell
    # outside every spell: the nearest one (BBT unlisted 2019-12-09 .. 2026-07-01)
    assert HOLDERS.holder("BBT", date(2020, 6, 1)) == TRUIST
    assert HOLDERS.holder("BBT", date(2026, 4, 15)) == BEACON


@pytest.mark.parametrize(
    "cik, ticker, on, ok",
    [
        (TRANE, "IR", date(2019, 6, 1), True),  # Ingersoll-Rand plc held IR then
        (TRANE, "IR", date(2023, 6, 1), False),  # Trane is TT; IR is Gardner Denver's
        (TRANE, "TT", date(2023, 6, 1), True),  # not shared: always passes
        (GD, "IR", date(2023, 6, 1), True),
        (GD, "IR", date(2019, 6, 1), False),  # GD (not yet a member) vs Ingersoll-Rand plc
        (TRUIST, "BBT", date(2026, 8, 1), False),  # Truist is TFC; BBT is Beacon's
        (BEACON, "BBT", date(2019, 6, 1), False),  # Beacon's 2019 filings are not BB&T's
        (BEACON, "BBT", date(2026, 8, 1), True),
        (BEACON, "BBT", date(2026, 5, 1), True),  # pre-membership baseline history
        (DIS_NEW, "DIS", date(2019, 6, 1), True),  # same company, CIK listed late
        (DIS_OLD, "DIS", date(2023, 6, 1), True),  # old registration still files
        (CBS, "VIAC", date(2020, 5, 1), True),  # ViacomCBS filing as CBS during the lag
    ],
)
def test_events_reach_shared_tickers_only_for_their_company(cik, ticker, on, ok):
    assert HOLDERS.reaches(cik, ticker, on) is ok


def test_restrict_drops_only_foreign_rows():
    df = pl.DataFrame(
        {
            "cik": [TRANE, TRANE, GD, TRANE, 320193],
            "ticker": ["IR", "TT", "IR", "IR", "AAPL"],
            "date": [date(2023, 6, 1)] * 3 + [date(2019, 6, 1), date(2023, 6, 1)],
        }
    )
    got = HOLDERS.restrict(df)
    assert got.rows() == [
        (TRANE, "TT", date(2023, 6, 1)),
        (GD, "IR", date(2023, 6, 1)),
        (TRANE, "IR", date(2019, 6, 1)),
        (320193, "AAPL", date(2023, 6, 1)),
    ]


def test_linker_resolves_shared_tickers_by_date_and_filer():
    u = Universe({TRANE: ["IR", "TT"], GD: ["IR"], TRUIST: ["BBT", "TFC"], BEACON: ["BBT"]})
    lk = TickerLinker(u, {"TT": ["Trane"]}, None, holders=HOLDERS)
    plain = TickerLinker(u, {"TT": ["Trane"]}, None)
    assert lk.linker_id != plain.linker_id

    def links(**kw):
        return sorted((x.ticker, x.cik, x.method) for x in lk.link(title=None, **kw))

    # vendor tag / cashtag: the holder on the publication date
    assert links(body="", source_tickers=["IR"], on=date(2023, 6, 1)) == [("IR", GD, "source")]
    assert links(body="", source_tickers=["IR"], on=date(2019, 6, 1)) == [("IR", TRANE, "source")]
    assert links(body="$BBT up", source_tickers=[], on=date(2026, 8, 1)) == [
        ("BBT", BEACON, "cashtag")
    ]
    # EDGAR: the filer's own CIK, whatever the date (Trane's filings carry IR and TT)
    assert links(
        body="", source_tickers=["IR", "TT"], source="edgar", source_cik=TRANE,
        on=date(2023, 6, 1),
    ) == [("IR", TRANE, "source"), ("TT", TRANE, "source")]  # fmt: skip
    # a name mention links its own company, even next to the other company's cashtag
    assert links(body="Trane said; $IR fell", source_tickers=[], on=date(2023, 6, 1)) == [
        ("IR", TRANE, "name_body"),
        ("IR", GD, "cashtag"),
        ("TT", TRANE, "name_body"),
    ]
