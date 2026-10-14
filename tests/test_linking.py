from __future__ import annotations

import pytest

from newsfeed.tickers import TickerLinker, Universe, clean_company_name

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
