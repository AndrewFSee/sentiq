"""Federal Reserve documents (source fed): listings, feeds, pages, timing, derive exclusion,
and the Fed scoring path. Fixtures recorded by scripts/record_fed_fixtures.py (2026-11-18)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from newsfeed.dedupe import rebuild_stories
from newsfeed.derive import relink
from newsfeed.ingest.fed import (
    FedAdapter,
    page_parts,
    parse_feed,
    parse_listing,
    parse_listing_time,
)
from newsfeed.ingest.runner import ingest, new_stats
from newsfeed.scoring.fed import FED_OUTPUT_JSON_SCHEMA, FedScorer, parse_fed_output
from newsfeed.scoring.prompt import GenerationSettings, load_prompt
from newsfeed.tickers import TickerLinker, Universe
from tests.conftest import FIXTURES, FakeClock, make_http
from tests.fake_llm import FakeLLM, make_message

FED = FIXTURES / "fed"
BASE = "https://www.federalreserve.gov"
ROUTES = {
    "/feeds/press_monetary.xml": "press_monetary.xml",
    "/feeds/speeches_and_testimony.xml": "speeches_and_testimony.xml",
    "/json/ne-press.json": "ne-press.json",
    "/json/ne-speeches.json": "ne-speech.json",
    "/json/ne-testimony.json": "ne-testimony.json",
    "/newsevents/pressreleases/monetary20261028a.htm": "statement.htm",
    "/newsevents/pressreleases/monetary20261118a.htm": "minutes_release.htm",
    "/monetarypolicy/fomcminutes20261028.htm": "minutes_page.htm",
    "/newsevents/speech/waller20261117a.htm": "speech.htm",
    "/newsevents/testimony/warsh20260714a.htm": "testimony.htm",
}


class FedTransport(httpx.MockTransport):
    """Serves the recorded Fed fixtures; pages that were not recorded get a minimal article.
    ``on_listing`` runs when a feed / listing is requested (to move a fake clock)."""

    def __init__(self, on_listing=None):
        self.paths: list[str] = []
        self.on_listing = on_listing
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path.startswith(("/feeds/", "/json/")) and self.on_listing:
            self.on_listing()
        if path in ROUTES:
            return httpx.Response(200, content=(FED / ROUTES[path]).read_bytes())
        if path.endswith(".htm"):
            html = (
                '<html><body><div id="article"><div class="heading col-md-8"><h3>T</h3></div>'
                '<div class="col-md-8"><p>Remarks on the economic outlook.</p></div>'
                "</div></body></html>"
            )
            return httpx.Response(200, text=html)
        return httpx.Response(404)


def _adapter(cfg, store, transport, clock) -> FedAdapter:
    return FedAdapter(cfg.sources.fed, make_http(transport), store.raw, clock=clock)


# ------------------------------------------------------------------ parsing
def test_listing_time_is_new_york_wall_clock():
    assert parse_listing_time("11/18/2026 2:00:00 PM") == datetime(2026, 11, 18, 19, tzinfo=UTC)
    assert parse_listing_time("7/14/2026 8:30:00 AM") == datetime(2026, 7, 14, 12, 30, tzinfo=UTC)


def test_press_listing_keeps_only_statements_and_minutes():
    refs = parse_listing((FED / "ne-press.json").read_bytes(), "press")
    kinds = {r.kind for r in refs}
    assert kinds <= {"fomc_statement", "fomc_minutes"}
    by_path = {r.path: r for r in refs}
    st = by_path["/newsevents/pressreleases/monetary20261028a.htm"]
    assert st.kind == "fomc_statement"
    assert st.published_at == datetime(2026, 10, 28, 18, tzinfo=UTC)  # 2:00 PM EDT
    mn = by_path["/newsevents/pressreleases/monetary20261118a.htm"]
    assert mn.kind == "fomc_minutes" and mn.native_id == "fomc_minutes_monetary20261118a"
    raw = json.loads((FED / "ne-press.json").read_text(encoding="utf-8"))
    others = [e for e in raw if e.get("l") and e["l"] not in by_path]
    assert others, "fixture should include press releases that are not collected"


def test_speech_listing_has_speaker_and_location():
    refs = parse_listing((FED / "ne-speech.json").read_bytes(), "speech")
    w = next(r for r in refs if r.path.endswith("waller20261117a.htm"))
    assert w.kind == "speech" and w.native_id == "speech_waller20261117a"
    assert w.speaker == "Governor Christopher J. Waller"
    assert w.published_at == datetime(2026, 11, 17, 14, 10, tzinfo=UTC)  # 9:10 AM EST
    assert not any("&#" in r.title or "&quot;" in r.title or "&amp;" in r.title for r in refs)


def test_feed_times_match_listing_times():
    """The live poll (RSS, GMT) and the backfill (JSON, New York time) date documents alike."""
    feed = {r.native_id: r for r in parse_feed((FED / "speeches_and_testimony.xml").read_bytes(),
                                               BASE)}  # fmt: skip
    feed |= {r.native_id: r for r in parse_feed((FED / "press_monetary.xml").read_bytes(), BASE)}
    listed = {}
    for name, kind in (("ne-press", "press"), ("ne-speech", "speech")):
        listed |= {r.native_id: r for r in parse_listing((FED / f"{name}.json").read_bytes(), kind)}
    common = set(feed) & set(listed)
    assert len(common) >= 5
    gap = {n: abs(feed[n].published_at - listed[n].published_at) for n in common}
    off = {n: g for n, g in gap.items() if g > timedelta(minutes=1)}
    assert not off, off
    assert feed["speech_waller20261117a"].title.startswith("Considerations")  # "Waller, " cut


def test_page_parts_reads_heading_and_body():
    p = page_parts((FED / "speech.htm").read_bytes())
    assert p["speaker"] == "Governor Christopher J. Waller"
    assert p["location"] and p["location"].startswith("At ")
    assert "Share" not in p["body"].splitlines()[:3]
    assert len(p["body"].split()) > 1000
    st = page_parts((FED / "statement.htm").read_bytes())
    assert "federal funds rate" in st["body"] and st["speaker"] is None


def test_page_parts_skips_attachment_panels():
    """fed/1 took the last content column: on pages with charts that is the attachments panel
    ("View speech charts and figures"), which left 13-token bodies for some speeches."""
    html = (
        '<div id="article"><div class="heading col-md-8"><p class="speaker">Chair X</p></div>'
        '<div class="col-xs-12 col-md-8"><p>Inflation has come down a long way.</p>'
        "<p>Policy is well positioned.</p></div>"
        '<div class="panel panel-attachments col-md-8"><p>View speech charts and figures</p>'
        "<p>Accessible Version</p></div></div>"
    )
    p = page_parts(html)
    assert p["speaker"] == "Chair X"
    assert "Inflation has come down" in p["body"] and "charts" not in p["body"]


def test_reparse_keeps_timing_and_fixes_text(cfg, store, calendar):
    from newsfeed.ingest.reparse import reparse_source

    clock = FakeClock(datetime(2026, 11, 18, 22, tzinfo=UTC))
    a = _adapter(cfg, store, FedTransport(), clock)
    ingest(a.backfill(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC),
                      skip=store.has_item),
           store=store, cfg=cfg, calendar=calendar, run_id="r", flush_every=10,
           stats=new_stats())  # fmt: skip
    before = store.query("SELECT item_id, tradable_at, first_seen_at FROM items ORDER BY item_id")
    out = reparse_source(store, "fed", a, "fed/99", calendar=calendar, run_id="rp")
    assert out["candidates"] == before.height and not out["failed"]
    after = store.query("SELECT item_id, tradable_at, first_seen_at FROM items ORDER BY item_id")
    assert after.equals(before)


# ------------------------------------------------------------------ adapter
def test_backfill_fetches_documents_and_caches_pages(cfg, store):
    t = FedTransport()
    clock = FakeClock(datetime(2026, 11, 18, 22, tzinfo=UTC))
    a = _adapter(cfg, store, t, clock)
    start, end = datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC)
    items = list(a.backfill(start, end, skip=lambda _id: False))
    by = {i.native_id: i for i in items}
    mn = by["fomc_minutes_monetary20261118a"]
    assert mn.url.endswith("/monetarypolicy/fomcminutes20261028.htm")
    assert "Staff Review" in mn.body_text or len(mn.body_text.split()) > 3000
    assert by["fomc_statement_monetary20261028a"].meta["kind"] == "fomc_statement"
    sp = by["speech_waller20261117a"]
    assert sp.meta["speaker"] == "Governor Christopher J. Waller"
    assert all(i.tickers == [] and i.ingest_mode == "backfill" for i in items)
    assert all(start <= i.published_at < end for i in items)
    n_pages = len([p for p in t.paths if p.endswith(".htm")])
    list(a.backfill(start, end, skip=lambda _id: False))  # pages now come from the raw store
    assert len([p for p in t.paths if p.endswith(".htm")]) == n_pages


def test_poll_stamps_first_seen_after_the_listing_response(cfg, store):
    clock = FakeClock(datetime(2026, 11, 18, 19, 0, 30, tzinfo=UTC))

    def advance():  # the listing response arrives 20 s after the request
        clock.now += timedelta(seconds=20)

    a = _adapter(cfg, store, FedTransport(on_listing=advance), clock)
    items = list(a.poll(skip=lambda _id: False))
    assert items and all(i.ingest_mode == "live" for i in items)
    assert all(i.first_seen_at == datetime(2026, 11, 18, 19, 1, 10, tzinfo=UTC) for i in items)
    assert a.last_activity_at == max(i.published_at for i in items)
    # nothing older than poll_lookback_days
    assert min(i.published_at for i in items) >= clock.now - timedelta(days=14)


def test_tradable_at_live_and_backfill(cfg, store, calendar):
    clock = FakeClock(datetime(2026, 11, 18, 19, 1, tzinfo=UTC))
    a = _adapter(cfg, store, FedTransport(), clock)
    stats = new_stats()
    ingest(a.poll(skip=store.has_item), store=store, cfg=cfg, calendar=calendar, run_id="r",
           flush_every=10, stats=stats)  # fmt: skip
    row = store.query(
        "SELECT published_at, tradable_at, tradable_rule FROM items "
        "WHERE native_id = 'fomc_minutes_monetary20261118a'"
    ).row(0, named=True)
    # minutes at 14:00 ET, first seen 14:01 ET (in session) -> usable when we saw them
    assert row["published_at"] == datetime(2026, 11, 18, 19, tzinfo=UTC)
    assert row["tradable_at"] == datetime(2026, 11, 18, 19, 1, tzinfo=UTC)
    assert row["tradable_rule"].startswith("live:")
    a2 = _adapter(cfg, store, FedTransport(), clock)
    day = (datetime(2026, 10, 28, tzinfo=UTC), datetime(2026, 10, 29, tzinfo=UTC))
    ingest(a2.backfill(*day, skip=store.has_item), store=store, cfg=cfg, calendar=calendar,
           run_id="r2", flush_every=10, stats=stats)  # fmt: skip
    st = store.query(
        "SELECT tradable_at, tradable_rule FROM items "
        "WHERE native_id = 'fomc_statement_monetary20261028a'"
    ).row(0, named=True)
    assert st["tradable_at"] == datetime(2026, 10, 28, 18, 1, tzinfo=UTC)  # 14:00 EDT + 60 s
    assert st["tradable_rule"] == "backfill:published+60s+session"


def test_fed_items_are_never_linked_or_in_stories(cfg, store, calendar):
    clock = FakeClock(datetime(2026, 11, 18, 22, tzinfo=UTC))
    a = _adapter(cfg, store, FedTransport(), clock)
    ingest(a.backfill(datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC),
                      skip=store.has_item),
           store=store, cfg=cfg, calendar=calendar, run_id="r", flush_every=10,
           stats=new_stats())  # fmt: skip
    n = store.query("SELECT count(*) AS n FROM items WHERE source = 'fed'")["n"][0]
    assert n >= 3
    # the minutes mention banks, payments, "Apple"-like names would match aliases: never linked
    linker = TickerLinker(Universe({320193: ["AAPL"]}), {"AAPL": ["Federal", "Committee"]})
    assert relink(store, linker, "r") == 0
    rebuild_stories(store, cfg.dedupe, linker.linker_id)
    assert store.query("SELECT count(*) AS n FROM item_story")["n"][0] == 0


# ------------------------------------------------------------------ scoring
def _fed_out(**kw):
    return {"policy_relevance": "high", "stance": 0.4, "rate_path": "hold", "inflation": 0.6,
            "growth": 0.3, "labor": 0.0, "confidence": 0.8, "rationale": "Test.", **kw}  # fmt: skip


def test_company_prompt_hash_is_unchanged():
    """The Fed path generalised build_params / PromptSpec.hash: the frozen company prompt (H1,
    H2, H3 depend on it) must keep its recorded hash."""
    from newsfeed.config import load_config

    s = load_config(None).scoring
    settings = GenerationSettings(s.effort, s.thinking, s.max_tokens)
    assert load_prompt("score_v1").hash(settings) == "6108c555090fa4b3"


def test_fed_scoring_end_to_end(cfg, store, calendar):
    clock = FakeClock(datetime(2026, 11, 18, 22, tzinfo=UTC))
    a = _adapter(cfg, store, FedTransport(), clock)
    ingest(a.backfill(datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC),
                      skip=store.has_item),
           store=store, cfg=cfg, calendar=calendar, run_id="r", flush_every=10,
           stats=new_stats())  # fmt: skip
    llm = FakeLLM(respond=lambda p: make_message(_fed_out()))
    sc = FedScorer(cfg, store, llm, run_id="s1")
    targets = sc.plan()
    assert len(targets) == store.query("SELECT count(*) AS n FROM items")["n"][0]
    est = sc.estimate(targets)
    assert est.n_requests == len(targets) and est.cost_live_no_cache > 0
    sc.run_live(targets)
    df = store.query("SELECT * FROM fed_scores")
    assert df.height == len(targets) and set(df["status"]) == {"ok"}
    assert set(df["rate_path"]) == {"hold"}
    req = llm.calls[0]
    assert req["output_config"]["format"]["schema"] == FED_OUTPUT_JSON_SCHEMA
    users = [c["messages"][0]["content"] for c in llm.calls]
    assert any("SPEAKER: Federal Open Market Committee" in u for u in users)
    assert any("SPEAKER: Governor Christopher J. Waller" in u for u in users)
    usage = store.ops_query("SELECT count(*) AS n FROM llm_usage WHERE run_id = 's1'")
    assert usage["n"][0] == len(targets)
    # cached: nothing left to score, and the prompt hash differs from the company prompt's
    assert FedScorer(cfg, store, llm, run_id="s2").plan() == []
    assert sc.prompt_hash != "6108c555090fa4b3"


def test_fed_output_validation_rejects_out_of_range():
    ok, err = parse_fed_output(json.dumps(_fed_out(stance=None, inflation=None)))
    assert ok is not None and err is None
    bad, err = parse_fed_output(json.dumps(_fed_out(stance=1.5)))
    assert bad is None and "outside" in err


def test_invalid_and_error_outcomes(cfg, store, calendar):
    clock = FakeClock(datetime(2026, 11, 18, 22, tzinfo=UTC))
    a = _adapter(cfg, store, FedTransport(), clock)
    ingest(a.backfill(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 19, tzinfo=UTC),
                      skip=store.has_item),
           store=store, cfg=cfg, calendar=calendar, run_id="r", flush_every=10,
           stats=new_stats())  # fmt: skip
    targets = FedScorer(cfg, store, None, run_id="x").plan()
    assert len(targets) >= 2

    def fails(p) -> bool:
        return "Considerations" in p["messages"][0]["content"]

    llm = FakeLLM(respond=lambda p: make_message(_fed_out(stance=3.0)), fail_on=fails)
    sc = FedScorer(cfg, store, llm, run_id="s")
    sc.run_live(targets)
    n_err = sc.stats.by_status.get("error", 0)
    assert 1 <= n_err < len(targets)
    stored = store.query("SELECT item_id, status FROM fed_scores")
    assert set(stored["status"]) == {"invalid"}  # out of range: stored as final, not clamped
    assert stored.height == len(targets) - n_err  # API errors are not stored ...
    left = FedScorer(cfg, store, None, run_id="y").plan()
    assert len(left) == n_err  # ... so exactly those are planned again
    assert not {t.item_id for t in left} & set(stored["item_id"])


def test_fixtures_exist():
    for name in set(ROUTES.values()):
        assert (FED / name).is_file(), name
    assert Path(FED / "speech.htm").stat().st_size < 200_000
