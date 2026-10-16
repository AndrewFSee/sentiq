"""10-K / 10-Q source: section locator, adapter end to end, separation from the 8-K source."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from newsfeed.config import load_config
from newsfeed.eval.prices import CachedPrices, horizon_days
from newsfeed.ingest.base import make_item_id
from newsfeed.ingest.edgar_periodic import EdgarPeriodicAdapter, locate_sections
from newsfeed.storage import RawStore
from newsfeed.tickers import Universe

from .conftest import FixtureTransport, make_http

PARA = "The company sells widgets and services in many markets around the world. " * 40

TEN_K = f"""Apple Inc. Form 10-K
Table of Contents
Part I
Item 1.
Business 1
Item 1A.
Risk Factors 5
Item 7.
Management's Discussion and Analysis 20
Item 7A.
Market Risk 30
Item 8.
Financial Statements 31
PART I
Item 1. Business
{PARA}
Item 1A. Risk Factors
{PARA}{PARA}
Item 1B. Unresolved Staff Comments
None.
PART II
Item 7. Management's Discussion and Analysis of Financial Condition
{PARA}
Item 7A. Quantitative and Qualitative Disclosures About Market Risk
See Item 7 above.
Item 8. Financial Statements
{PARA}
"""

TEN_Q = f"""Table of Contents
Part I - Financial Information
Item 1. Financial Statements
Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations
Item 4. Controls and Procedures
Part II - Other Information
Item 1A. Risk Factors
Item 2. Unregistered Sales of Equity Securities and Use of Proceeds
Item 1. Financial Statements
{PARA}
Item 2. Management's Discussion and Analysis of
Financial Condition and Results of Operations
{PARA}
Item 4. Controls and Procedures
Our controls are effective.
PART
II
Item 1A. Risk Factors
There have been no material changes to our risk factors.
Item 2. Unregistered Sales of Equity Securities and Use of Proceeds
{PARA}
"""


def test_locate_sections_10k_skips_toc_and_short_cross_references():
    s = locate_sections(TEN_K, "10-K")
    assert set(s) == {"1", "1A", "7"}  # 7A is a one-line cross-reference
    a, b = s["1A"]
    assert TEN_K[a:b].strip().startswith("The company sells") and len(TEN_K[a:b]) > 2 * len(PARA)
    assert "Item 1B" not in TEN_K[a:b]  # section ends at the next heading
    a, b = s["7"]
    assert TEN_K[a:b].strip() == PARA.strip()


def test_locate_sections_10q_mda_by_title_not_part_lines():
    s = locate_sections(TEN_Q, "10-Q")
    # MD&A found although no PART I line precedes it and "PART / II" is split over two lines;
    # Part II's item 2 (unregistered sales) is not mistaken for it; a one-line 1A is dropped
    assert set(s) == {"I-2"}
    a, b = s["I-2"]
    assert TEN_Q[a:b].strip().startswith("Financial Condition and Results")
    assert "Unregistered" not in TEN_Q[a:b]


ACC = "0000320193-26-000020"  # a 10-Q listed in the recorded AAPL submissions fixture
FOLDER = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000020"
HDR = f"""<SEC-HEADER>{ACC}.hdr.sgml : 20260731
<ACCEPTANCE-DATETIME>20260731100102
<ACCESSION-NUMBER>{ACC}
<TYPE>10-Q
<PERIOD>20260627
<FILING-DATE>20260731
<FILER>
<COMPANY-DATA>
<CONFORMED-NAME>Apple Inc.
<CIK>0000320193
</COMPANY-DATA>
</FILER>
</SEC-HEADER>
"""
INDEX = """<html><body>
<div class="infoHead">Accepted</div><div class="info">2026-07-31 10:01:02</div>
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>10-Q</td><td><a href="/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm">aapl-20260627.htm</a></td><td>10-Q</td><td>1</td></tr>
<tr><td>2</td><td>EX-31.1</td><td><a href="/Archives/edgar/data/320193/000032019326000020/ex311.htm">ex311.htm</a></td><td>EX-31.1</td><td>1</td></tr>
</table></body></html>"""  # noqa: E501


def _periodic(cfg, ticker_map, tmp_path, clock):
    def serve(content: str):
        return lambda req: httpx.Response(200, content=content.encode())

    html = "<html><body>" + "".join(f"<p>{ln}</p>" for ln in TEN_Q.splitlines()) + "</body></html>"
    transport = FixtureTransport(
        overrides={
            f"{FOLDER}/{ACC}.hdr.sgml": serve(HDR),
            f"{FOLDER}/{ACC}-index.htm": serve(INDEX),
            f"{FOLDER}/aapl-20260627.htm": serve(html),
        }
    )
    adapter = EdgarPeriodicAdapter(
        cfg.sources.edgar_periodic,
        make_http(transport),
        Universe.build(["AAPL"], ticker_map),
        RawStore(tmp_path / "raw_d"),
        clock=clock,
    )
    return adapter, transport


def test_periodic_adapter_end_to_end(cfg, ticker_map, tmp_path, clock):
    adapter, transport = _periodic(cfg, ticker_map, tmp_path, clock)
    refs = adapter.list_filings(320193)
    assert {r.form for r in refs} <= {"10-K", "10-Q"} and "10-K" in {r.form for r in refs}
    ref = next(r for r in refs if r.accession == ACC)
    item = adapter.fetch_filing(ref, ["AAPL"], first_seen_at=clock(), ingest_mode="backfill")
    assert item.source == "edgar_periodic" and item.item_id == make_item_id("edgar_periodic", ACC)
    assert item.item_id != make_item_id("edgar", ACC)  # never collides with the 8-K source
    assert item.published_at == datetime(2026, 7, 31, 14, 1, 2, tzinfo=UTC)  # 10:01:02 EDT
    assert item.title == "Apple Inc. 10-Q for the period ended 2026-06-27"
    assert not item.body_truncated and "Unregistered Sales" in item.body_text  # full text
    assert list(item.meta["sections"]) == ["I-2"]
    a, b = item.meta["sections"]["I-2"]
    assert item.body_text[a:b].strip().startswith("Financial Condition")
    # primary document only: the certification exhibit is never requested
    assert not any("ex311" in str(r.url) for r in transport.requests)
    # raw documents go to this source's own raw store
    assert (tmp_path / "raw_d" / "edgar_periodic" / ACC).is_dir()


def test_periodic_config_defaults_and_8k_untouched():
    cfg = load_config(None)
    per = cfg.sources.edgar_periodic
    assert per.forms == ["10-K", "10-Q"] and per.exhibit_type_prefixes == []
    assert per.ingest_margin_before_days == 400 and per.raw_dir is not None
    assert cfg.sources.edgar.forms == ["8-K", "8-K/A"] and cfg.sources.edgar.raw_dir is None
    assert "edgar_periodic" in cfg.dedupe.isolated_sources
    assert "edgar_periodic" not in cfg.collect.sources
    assert 60 in cfg.eval.horizons


def test_horizon_days_keeps_m6_values_up_to_20_sessions():
    assert horizon_days(20) == (45, 25)  # what every eval through test 9 used
    after, pad = horizon_days(60)
    assert pad >= 61 and after >= 90  # 60 sessions ~ 84-88 calendar days


def test_ended_series_cache_serves_later_windows(tmp_path):
    calls = []

    class Src:
        def daily_bars(self, symbol, start, end):
            import polars as pl

            calls.append((symbol, start, end))
            days = [date(2021, 1, 4), date(2021, 1, 5)]  # trading stopped in early January
            cols = {c: [1.0, 1.0] for c in ("open", "high", "low", "close", "volume")}
            return pl.DataFrame({"date": days, **cols})

    cache = CachedPrices(Src(), tmp_path)
    cache.daily_bars("OLD", date(2021, 1, 1), date(2021, 6, 1))
    assert cache.daily_bars("OLD", date(2021, 1, 1), date(2021, 9, 1)).height == 2
    assert len(calls) == 1  # no refetch for a series that had ended
    cache.daily_bars("OLD", date(2020, 1, 1), date(2021, 9, 1))  # earlier start: must refetch
    assert len(calls) == 2


def test_live_scoring_hook_only_plans_collected_sources(cfg, store, monkeypatch):
    """A 10-K backfill running next to the collector must never be scored by the live hook."""
    from contextlib import ExitStack
    from types import SimpleNamespace

    import newsfeed.cli as cli

    seen = {}

    def fake_plan(job, c2t, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(cli, "plan_targets", fake_plan)
    ctx = cli.Context(cfg, store, ExitStack(), None)
    monkeypatch.setattr(ctx, "scoring_job", lambda *a: None)
    monkeypatch.setattr(ctx, "llm_client", lambda *a: None)
    monkeypatch.setattr(ctx, "universe", lambda full=False: SimpleNamespace(cik_to_tickers={}))
    assert ctx.score_recent("r") == {"calls": 0}
    assert seen["sources"] == list(cfg.collect.sources)
    assert "edgar_periodic" not in seen["sources"]
