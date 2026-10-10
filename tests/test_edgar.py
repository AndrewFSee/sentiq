from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from newsfeed.ingest.edgar import (
    EdgarAdapter,
    columnar_to_rows,
    parse_header,
    parse_index,
    split_items,
)
from newsfeed.ingest.http import HttpClient, RateLimiter
from newsfeed.ingest.runner import build_row
from newsfeed.textutil import html_to_text, truncate_tokens
from newsfeed.tickers import Universe

from .conftest import EDGAR_FIXTURES, FixtureTransport, make_http

APPLE_DIR = EDGAR_FIXTURES / "www.sec.gov" / "Archives" / "edgar" / "data" / "320193"
JUL26 = "0000320193-26-000018"


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


# --------------------------------------------------------------------- parsers
def test_parse_header():
    hdr = parse_header((APPLE_DIR / "000032019326000018" / f"{JUL26}.hdr.sgml").read_text())
    assert hdr["ACCEPTANCE-DATETIME"] == "20260730163028"
    assert hdr["ITEMS"] == ["2.02", "9.01"]
    assert hdr["CONFORMED-NAME"] == "Apple Inc."  # first occurrence, not a former name


def test_parse_header_requires_acceptance():
    with pytest.raises(ValueError):
        parse_header("<SEC-HEADER>\n<TYPE>8-K\n")


def test_parse_index():
    docs, accepted = parse_index(
        (APPLE_DIR / "000032019326000018" / f"{JUL26}-index.htm").read_bytes()
    )
    assert accepted == "2026-07-30 16:30:28"
    by_type = {d.doc_type: d.filename for d in docs}
    assert by_type["8-K"] == "aapl-20260730.htm"  # the /ix?doc= viewer link is stripped
    assert by_type["EX-99.1"] == "a8-kex991q3202606272026.htm"


def test_html_to_text_drops_hidden_ixbrl():
    html = (APPLE_DIR / "000032019326000018" / "aapl-20260730.htm").read_bytes()
    text = html_to_text(html)
    assert "Results of Operations and Financial Condition" in text
    assert "dei:" not in text and "xbrli:" not in text  # hidden ix:header facts removed


def test_split_items_on_real_8k():
    text = html_to_text((APPLE_DIR / "000032019326000018" / "aapl-20260730.htm").read_bytes())
    items = split_items(text)
    assert set(items) == {"2.02", "9.01"}
    assert "Exhibit 99.1" in items["2.02"] or "99.1" in items["2.02"]
    assert "SIGNATURE" not in items["9.01"].upper().split("\n")[-1]


def test_split_items_synthetic():
    text = (
        "cover\nItem 5.02. Departure of Directors\nCEO left.\n"
        "ITEM 9.01 — Exhibits\n99.1\nSIGNATURES\nsig"
    )
    assert split_items(text) == {
        "5.02": "Departure of Directors\nCEO left.",
        "9.01": "Exhibits\n99.1",
    }


def test_truncate_tokens():
    text = "word " * 1000
    out, trunc = truncate_tokens(text, max_tokens=100, chars_per_token=4)
    assert trunc and len(out) <= 400 and not out.endswith(" ")
    assert truncate_tokens("short", 100, 4) == ("short", False)


def test_columnar_to_rows_rejects_ragged():
    assert columnar_to_rows({"a": [1, 2], "b": [3, 4]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    with pytest.raises(ValueError):
        columnar_to_rows({"a": [1, 2], "b": [3]})


# --------------------------------------------------------------------- adapter
def test_fetch_filing_end_to_end(edgar: EdgarAdapter, clock):
    ref = next(r for r in edgar.list_filings(320193) if r.accession == JUL26)
    item = edgar.fetch_filing(ref, ["AAPL"], first_seen_at=clock(), ingest_mode="backfill")

    # authoritative time from the SGML header, NOT the JSON value or filingDate
    assert item.published_at == utc(2026, 7, 30, 20, 30, 28)
    assert item.meta["acceptance_json_raw"] == "2026-07-31T00:30:28.000Z"
    assert item.meta["json_convention"] == "double_offset"
    assert item.meta["acceptance_index_et"] == "2026-07-30 16:30:28"
    assert item.meta["items"] == ["2.02", "9.01"]
    assert item.meta["exhibits"] == [{"type": "EX-99.1", "file": "a8-kex991q3202606272026.htm"}]
    assert item.title.startswith("Apple Inc. 8-K: Item 2.02 Results of Operations")
    assert "[ITEM 2.02" in item.body_text and "[EXHIBIT EX-99.1" in item.body_text
    assert item.body_tokens_est <= edgar.cfg.max_tokens
    assert item.meta["body_tokens_est_full"] >= item.body_tokens_est
    assert json.loads(item.raw_json)["submissions_row"]["accessionNumber"] == JUL26
    # raw documents cached for re-parsing
    assert edgar.raw.get("edgar", JUL26, "a8-kex991q3202606272026.htm") is not None


def test_after_530pm_filing_uses_acceptance_not_filing_date(edgar: EdgarAdapter, clock):
    ref = next(r for r in edgar.list_filings(320193) if r.accession == "0001140361-25-025275")
    assert ref.filing_date == "2025-07-09"  # next-day filingDate
    item = edgar.fetch_filing(ref, ["AAPL"], first_seen_at=clock(), ingest_mode="backfill")
    assert item.published_at == utc(2025, 7, 9, 1, 18, 13)  # 2025-07-08 21:18:13 EDT
    assert item.meta["items"] == ["5.02"]
    assert item.meta["exhibits"] == []


def test_backfill_window_and_idempotency(edgar: EdgarAdapter, store, calendar, transport):
    start, end = utc(2026, 7, 1), utc(2026, 8, 1)
    lag = timedelta(seconds=60)

    def run() -> int:
        items = list(edgar.backfill(start, end, skip=store.has_item))
        return store.write_items(
            [
                build_row(
                    i, run_id="r", calendar=calendar, backfill_lag=lag, ingested_at=utc(2026, 10, 8)
                )
                for i in items
            ],
            "r",
        )

    assert run() == 1
    df = store.query("SELECT native_id, tradable_at, ingest_mode, first_seen_at FROM items")
    assert df["native_id"].to_list() == [JUL26]
    assert df["ingest_mode"][0] == "backfill"
    # 16:30:28 EDT + 60s lag is after the close -> next session open (Fri 09:30 EDT)
    assert df["tradable_at"][0] == utc(2026, 7, 31, 13, 30)

    n_before = len(transport.requests)
    assert run() == 0
    archive_hits = [r for r in transport.requests[n_before:] if "/Archives/" in str(r.url)]
    assert archive_hits == []  # resumable: stored filings are not re-fetched


def test_backfill_continues_past_failures(edgar: EdgarAdapter, store):
    """Window includes filings with no recorded fixtures (404): they are reported, not fatal."""
    errors: list[str] = []
    items = list(
        edgar.backfill(
            utc(2025, 12, 1),
            utc(2026, 8, 1),
            skip=store.has_item,
            on_error=lambda acc, exc: errors.append(acc),
        )
    )
    got = {i.native_id for i in items}
    assert got == {JUL26, "0000320193-26-000005"}
    assert "0001140361-26-015711" in errors  # an unrecorded 8-K in the window
    assert all(e not in got for e in errors)


def test_backfill_without_on_error_raises(edgar: EdgarAdapter, store):
    with pytest.raises(httpx.HTTPStatusError):
        list(edgar.backfill(utc(2025, 12, 1), utc(2026, 8, 1), skip=store.has_item))


def test_truncation_cap_applies(cfg, http, ticker_map, store, clock):
    small = cfg.sources.edgar.model_copy(update={"max_tokens": 300})
    adapter = EdgarAdapter(
        small, http, Universe.build(["AAPL"], ticker_map), store.raw, clock=clock
    )
    ref = next(r for r in adapter.list_filings(320193) if r.accession == JUL26)
    item = adapter.fetch_filing(ref, ["AAPL"], first_seen_at=clock(), ingest_mode="backfill")
    assert item.body_truncated
    assert item.body_tokens_est <= 300 < item.meta["body_tokens_est_full"]
    assert item.body_text.startswith("[ITEM 2.02")  # items come before exhibits


def test_catch_up_sweep_marks_live_and_stamps_first_seen_after_listing(
    edgar: EdgarAdapter, store, clock
):
    # fixture listing as of 2026-10-08; lookback reaches the July earnings 8-K
    edgar.poll_lookback = timedelta(days=75)
    errors: list[str] = []
    items = list(
        edgar.poll(skip=store.has_item, on_error=lambda a, e: errors.append(a), catch_up=True)
    )
    assert [i.native_id for i in items] == [JUL26]
    assert errors == ["0001140361-26-035325"]  # Sep 8-K/A in window, not recorded as fixture
    assert items[0].ingest_mode == "live"
    assert items[0].first_seen_at == clock.now
    assert items[0].first_seen_at >= items[0].published_at


def test_share_classes_resolve_to_one_cik(ticker_map):
    universe = Universe.build(["GOOGL", "GOOG"], ticker_map)
    assert universe.cik_to_tickers == {1652044: ["GOOG", "GOOGL"]}


def test_co_registrant_filing_fetched_once_with_merged_tickers(cfg, ticker_map, store, clock):
    """Same accession listed under two universe CIKs -> one item, tickers unioned."""
    apple_json = (
        EDGAR_FIXTURES / "data.sec.gov" / "submissions" / "CIK0000320193.json"
    ).read_bytes()
    url = "https://data.sec.gov/submissions/CIK0009999999.json"
    transport = FixtureTransport(overrides={url: lambda r: httpx.Response(200, content=apple_json)})
    universe = Universe({320193: ["AAPL"], 9999999: ["SUBCO"]})
    adapter = EdgarAdapter(
        cfg.sources.edgar, make_http(transport), universe, store.raw, clock=clock
    )
    items = list(adapter.backfill(utc(2026, 7, 1), utc(2026, 8, 1), skip=store.has_item))
    assert [(i.native_id, i.tickers) for i in items] == [(JUL26, ["AAPL", "SUBCO"])]
    hdr_hits = [r for r in transport.requests if str(r.url).endswith(".hdr.sgml")]
    assert len(hdr_hits) == 1


def test_older_submission_pages_fetched_only_when_needed(cfg, ticker_map, store, clock):
    older = {
        "accessionNumber": ["0000320193-15-000001"],
        "filingDate": ["2015-01-27"],
        "acceptanceDateTime": ["2015-01-28T02:30:00.000Z"],
        "form": ["8-K"],
        "items": ["2.02"],
        "primaryDocument": ["x.htm"],
    }
    url = "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
    transport = FixtureTransport(overrides={url: lambda r: httpx.Response(200, json=older)})
    http = make_http(transport)
    adapter = EdgarAdapter(
        cfg.sources.edgar, http, Universe.build(["AAPL"], ticker_map), store.raw, clock=clock
    )

    adapter.list_filings(320193, utc(2026, 1, 1), utc(2026, 2, 1))
    assert all(str(r.url) != url for r in transport.requests)
    refs = adapter.list_filings(320193, utc(2015, 1, 1), utc(2015, 2, 1))
    assert any(str(r.url) == url for r in transport.requests)
    assert [r.accession for r in refs] == ["0000320193-15-000001"]


def test_form_filter(edgar: EdgarAdapter):
    forms = {r.form for r in edgar.list_filings(320193)}
    assert forms <= {"8-K", "8-K/A"} and "8-K" in forms


# --------------------------------------------------------------------- http client
def test_http_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, text="ok")

    c = make_http(httpx.MockTransport(handler))
    assert c.get("https://example.com/x").text == "ok"
    assert calls["n"] == 3


def test_http_does_not_retry_404():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    c = make_http(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://example.com/x")
    assert calls["n"] == 1


def test_http_sends_user_agent():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200)

    make_http(httpx.MockTransport(handler)).get("https://example.com/")
    assert seen["ua"] == "test-suite test@example.com"


def test_rate_limiter_spacing():
    now = {"t": 0.0}
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        now["t"] += s

    lim = RateLimiter(8, clock=lambda: now["t"], sleep=sleep)
    for _ in range(9):
        lim.acquire()
    assert now["t"] == pytest.approx(1.0)  # 9 acquisitions at 8/s span exactly 1 second
    assert all(s == pytest.approx(0.125) for s in sleeps)


def test_rate_limiter_wired_into_client():
    acquired = {"n": 0}

    class Counting(RateLimiter):
        def acquire(self):
            acquired["n"] += 1

    c = HttpClient(
        user_agent="x x@y.z",
        rate_per_sec=8,
        limiter=Counting(8),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    c.get("https://example.com/a")
    c.get("https://example.com/b")
    assert acquired["n"] == 2 == c.request_count


# --------------------------------------------------------------------- table of contents
TOC_DOC = EDGAR_FIXTURES / "toc" / "amzn-0001104659-26-082293-8k.htm"


def test_split_items_skips_table_of_contents_real_filing():
    """Regression (found by the M3 smoke test): a TOC listing 'ITEM 8.01 ... SIGNATURES' used to
    end parsing before the real sections, leaving bodies like 'OTHER EVENTS. 3'."""
    items = split_items(html_to_text(TOC_DOC.read_bytes()))
    assert set(items) == {"8.01", "9.01"}
    assert "closed the sale" in items["8.01"] and len(items["8.01"]) > 2000
    assert "Underwriting Agreement" in items["9.01"]


def test_split_items_toc_synthetic():
    text = (
        "Table of Contents\nITEM 8.01. OTHER EVENTS. 3\nITEM 9.01. EXHIBITS. 4\nSIGNATURES 5\n"
        "ITEM 8.01.\nThe company issued notes.\nITEM 9.01.\n99.1 Press release\n"
        "SIGNATURES\nPursuant to the requirements..."
    )
    assert split_items(text) == {"8.01": "The company issued notes.", "9.01": "99.1 Press release"}
