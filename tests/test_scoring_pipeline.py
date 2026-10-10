"""End-to-end scoring over the synthetic news corpus with a fake LLM (no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from newsfeed.dedupe import rebuild_stories
from newsfeed.derive import relink
from newsfeed.ingest.news_api import FixtureVendor, NewsApiAdapter
from newsfeed.ingest.runner import ingest, new_stats
from newsfeed.scoring import agreement as agr
from newsfeed.scoring import batch as batch_mod
from newsfeed.scoring.client import BatchResult, call_cost
from newsfeed.scoring.mask import MaskLexicon
from newsfeed.scoring.runner import ScoringJob, estimate_cost, plan_targets
from newsfeed.tickers import TickerLinker, Universe

from .conftest import FIXTURES, FakeClock
from .fake_llm import FakeLLM, default_output, make_message, user_text

NAMES = {320193: "Apple Inc.", 789019: "MICROSOFT CORP", 1652044: "Alphabet Inc."}


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


@pytest.fixture
def env(cfg, store, calendar, ticker_map):
    base = Universe.build(["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA"], ticker_map)
    extra = {101: ["AMZN"], 102: ["META"], 103: ["JPM"], 104: ["WMT"], 105: ["JNJ"], 106: ["XOM"]}
    universe = Universe({**base.cik_to_tickers, **extra})
    tick = {t for ts in universe.cik_to_tickers.values() for t in ts}
    ucfg = cfg.universe.model_copy(
        update={
            k: {t: v for t, v in getattr(cfg.universe, k).items() if t in tick}
            for k in ("aliases", "products", "people", "places")
        }
    )
    cfg = cfg.model_copy(update={"universe": ucfg})

    clock = FakeClock(utc(2026, 10, 8))
    adapter = NewsApiAdapter(
        cfg.sources.news_api,
        FixtureVendor(FIXTURES / "news" / "articles.jsonl", clock=clock),
        universe,
        clock=clock,
    )
    items = adapter.backfill(
        utc(2026, 7, 1), utc(2026, 9, 1), skip=store.has_item, on_error=lambda *_: None
    )
    ingest(
        items,
        store=store,
        cfg=cfg,
        calendar=calendar,
        run_id="ing",
        flush_every=100,
        stats=new_stats(),
    )
    linker = TickerLinker(universe, ucfg.aliases)
    relink(store, linker, "ing")
    rebuild_stories(store, cfg.dedupe, linker.linker_id)

    def job(client=None, model=None, run_id="run1", **overrides):
        c = cfg
        if overrides:
            c = cfg.model_copy(update={"scoring": cfg.scoring.model_copy(update=overrides)})
        return ScoringJob(
            c,
            store,
            client,
            run_id=run_id,
            model=model or c.scoring.model,
            company_names=NAMES,
            lexicon=MaskLexicon.build(ucfg, universe, NAMES),
        )

    return cfg, store, job, universe.cik_to_tickers


def native(store, item_id):
    return store.query("SELECT native_id FROM items WHERE item_id = $i", {"i": item_id})[
        "native_id"
    ][0]


# ------------------------------------------------------------------ planning
def test_plan_targets_founders_policy_and_share_classes(env):
    _, store, job, c2t = env
    targets = plan_targets(job(), c2t)
    pairs = {(native(store, t.item_id), t.ticker) for t in targets}
    assert ("fixture:a1", "AAPL") in pairs
    assert not {p for p in pairs if p[0] in ("fixture:a2", "fixture:a3", "fixture:a4")}
    assert ("fixture:a6", "MSFT") in pairs and ("fixture:a6", "GOOGL") in pairs
    assert not [p for p in pairs if p[1] == "GOOG"]  # share class scored once, canonical GOOGL
    assert not [p for p in pairs if p[0] == "fixture:a12" and p[1] == "AAPL"]  # "apple orchards"
    assert len(targets) == 11


# ------------------------------------------------------------------ live
def test_live_scores_cache_and_ledger(env):
    cfg, store, job, c2t = env
    fake = FakeLLM()
    j = job(fake)
    stats = j.run_live(plan_targets(j, c2t))
    assert len(fake.calls) == 11 and stats.by_status == {"ok": 11}
    scores = store.query("SELECT * FROM scores")
    assert scores.height == 11 and set(scores["status"]) == {"ok"}
    assert scores["raw_response"][0].startswith("{")
    assert (scores["prompt_hash"] == j.prompt_hash).all()
    usage = store.ops_query("SELECT * FROM llm_usage")
    assert usage.height == 11
    one = call_cost(
        cfg.pricing,
        "claude-haiku-5-5",
        {
            "input_tokens": 1000,
            "output_tokens": 120,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2000,
        },
        batch=False,
    )
    assert usage["cost_usd"].sum() == pytest.approx(11 * one)
    assert store.ops_query("SELECT count(*) AS n FROM experiments")["n"][0] == 1

    # the cache: a second run plans nothing and calls nothing
    j2 = job(FakeLLM(), run_id="run2")
    assert plan_targets(j2, c2t) == []


def test_request_content(env):
    _, _, job, c2t = env
    fake = FakeLLM()
    j = job(fake)
    j.run_live(plan_targets(j, c2t))
    apple = [p for p in fake.calls if "TARGET: Apple Inc. (ticker: AAPL)" in user_text(p)]
    assert len(apple) == 2  # a1 and a5
    p = apple[0]
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in p
    assert "2026-07" not in user_text(p)  # no publication timestamps sent


def test_failed_calls_are_retried_but_refusals_are_final(env):
    _, store, job, c2t = env

    def respond(params):
        if "Chip stocks" in user_text(params):  # a7 -> refusal
            return make_message(None, stop_reason="refusal")
        return make_message(default_output(user_text(params)))

    fake = FakeLLM(respond, fail_on=lambda p: "Analyst raises" in user_text(p))  # a5 -> error
    j = job(fake)
    stats = j.run_live(plan_targets(j, c2t))
    assert stats.by_status == {"ok": 9, "refusal": 1, "error": 1}
    statuses = store.ops_query("SELECT status, count(*) AS n FROM llm_usage GROUP BY 1")
    assert dict(zip(statuses["status"], statuses["n"], strict=True)) == {
        "ok": 9,
        "refusal": 1,
        "error": 1,
    }
    retry = plan_targets(job(FakeLLM(), run_id="run2"), c2t)
    assert [native(store, t.item_id) for t in retry] == ["fixture:a5"]  # refusal not retried


# ------------------------------------------------------------------ studies
def test_masked_study_hides_identities_and_pairs_with_primary(env):
    _, store, job, c2t = env
    j = job(FakeLLM())
    j.run_live(plan_targets(j, c2t))
    fake = FakeLLM()
    jm = job(fake, run_id="mask")
    targets = plan_targets(jm, c2t, masked=True)
    assert len(targets) == 11 and all(t.masked for t in targets)
    jm.run_live(targets)
    for p in fake.calls:
        u = user_text(p)
        assert "TARGET: [COMPANY_A] (ticker: [TICKER_A])" in u
        for leak in ("Apple", "AAPL", "Microsoft", "Alphabet", "Walmart", "JPMorgan"):
            assert leak not in u, leak
    rows = store.query("SELECT mask_residuals FROM scores WHERE masked")
    assert rows.height == 11 and rows["mask_residuals"].max() >= 1  # 'apple orchards' (a12)
    m = agr.metrics(
        agr.paired(
            store, agr.Side(j.model, j.prompt_hash), agr.Side(j.model, j.prompt_hash, masked=True)
        )
    )
    assert m["n_pairs"] == 11


def test_studies_only_cover_documents_with_primary_scores(env):
    _, _, job, c2t = env
    j = job(FakeLLM())
    assert plan_targets(j, c2t, masked=True) == []  # nothing scored yet
    j.run_live(plan_targets(j, c2t, limit=3))
    assert len(plan_targets(job(FakeLLM(), run_id="m"), c2t, masked=True)) == 3
    sonnet = job(FakeLLM(), model="claude-sonnet-5-5", run_id="v")
    assert sonnet.prompt_hash == j.prompt_hash  # model is a separate cache-key component
    assert len(plan_targets(sonnet, c2t)) == 3


def test_replicates_noise_study(env):
    _, store, job, c2t = env
    j = job(FakeLLM())
    j.run_live(plan_targets(j, c2t))
    jr = job(FakeLLM(), run_id="rep")
    targets = plan_targets(jr, c2t, replicates=2, sample=3, seed=7)
    assert len(targets) == 6 and {t.replicate for t in targets} == {1, 2}
    assert len({t.cache_key for t in targets}) == 6
    assert targets == plan_targets(jr, c2t, replicates=2, sample=3, seed=7)  # stable sample
    jr.run_live(targets)
    m = agr.metrics(
        agr.paired(
            store,
            agr.Side(j.model, j.prompt_hash),
            agr.Side(j.model, j.prompt_hash, replicate=None),
        )
    )
    assert m["n_pairs"] == 6
    assert m["sentiment_mae"] == 0  # the fake is deterministic


# ------------------------------------------------------------------ batch
def test_batch_lifecycle(env):
    cfg, store, job, c2t = env
    fake = FakeLLM()
    j = job(fake, batch_max_requests=4)
    targets = plan_targets(j, c2t)
    ids = batch_mod.submit(j, targets)
    assert len(ids) == 3 and not fake.calls  # nothing via the live API
    assert sum(len(r) for r in fake.batches.values()) == 11
    assert plan_targets(job(FakeLLM(), run_id="x"), c2t) == []  # in flight -> not re-planned

    # not ended yet -> nothing collected
    s = batch_mod.collect(j)
    assert s["collected"] == [] and len(s["still_running"]) == 3

    expired, errored = targets[0].custom_id, targets[1].custom_id
    fake.batch_overrides[expired] = BatchResult(expired, "expired", None, None)
    fake.batch_overrides[errored] = BatchResult(errored, "errored", None, "overloaded")
    for b in ids:
        fake.batch_state[b] = "ended"
    s = batch_mod.collect(j)
    assert sorted(s["collected"]) == sorted(ids)
    scores = store.query("SELECT * FROM scores")
    assert scores.height == 9 and set(scores["mode"]) == {"batch"}
    # batch discount applied on each row
    live_cost = call_cost(
        cfg.pricing,
        "claude-haiku-5-5",
        {
            "input_tokens": 1000,
            "output_tokens": 120,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2000,
        },
        batch=False,
    )
    assert scores["cost_usd"][0] == pytest.approx(live_cost / 2)
    # collection is idempotent; the expired/errored pair is re-planned
    assert batch_mod.collect(j)["collected"] == []
    assert store.query("SELECT count(*) AS n FROM scores_all")["n"][0] == 9
    again = plan_targets(job(FakeLLM(), run_id="y"), c2t)
    assert {t.custom_id for t in again} == {expired, errored}


def test_batch_collect_wait_polls_until_done(env):
    _, _, job, c2t = env
    fake = FakeLLM()
    j = job(fake)
    (bid,) = batch_mod.submit(j, plan_targets(j, c2t, limit=2))
    polls = []

    def sleep(s):
        polls.append(s)
        fake.batch_state[bid] = "ended"

    s = batch_mod.collect(j, wait=True, sleep=sleep)
    assert s["collected"] == [bid] and len(polls) == 1


# ------------------------------------------------------------------ estimates
def test_estimate_cost(env):
    _, _, job, c2t = env
    fake = FakeLLM()
    j = job(fake)
    targets = plan_targets(j, c2t)
    est = estimate_cost(j, targets)
    assert est.method == "count_tokens" and est.n_requests == 11
    assert len(fake.counted) == 12  # system baseline + 11 sampled (sample size 25 > 11)
    assert not fake.calls  # estimating never calls the model
    assert est.cost_live_cached < est.cost_live_no_cache
    assert est.cost_batch_no_cache == pytest.approx(est.cost_live_no_cache / 2)
    offline = estimate_cost(job(None), targets)
    assert offline.method == "chars" and offline.output_source == "config"


def test_agreement_pairs_only_scores_of_the_same_text(env):
    """Regression (smoke test): stale primary rows for an older text of the same item doubled
    the pair counts. Pairs must match on (item, ticker, text_hash)."""
    _, store, job, c2t = env
    j = job(FakeLLM())
    j.run_live(plan_targets(j, c2t))
    jm = job(FakeLLM(), run_id="mask")
    jm.run_live(plan_targets(jm, c2t, masked=True, sample=3))
    stale = store.query("SELECT * FROM scores WHERE NOT masked").to_dicts()[:5]
    for r in stale:  # pretend these were scored on an earlier parse of the same items
        r.update(cache_key=r["cache_key"][::-1], text_hash="stale", run_id="old")
    store.write_scores(stale, "old")
    m = agr.metrics(
        agr.paired(
            store, agr.Side(j.model, j.prompt_hash), agr.Side(j.model, j.prompt_hash, masked=True)
        )
    )
    assert m["n_pairs"] == 3
