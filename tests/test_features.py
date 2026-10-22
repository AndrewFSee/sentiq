"""Point-in-time features: values, availability rules, and strict causality."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from newsfeed import coverage
from newsfeed.features.aggregate import (
    _volume_z,
    as_of_grid,
    build_features,
    feature_set_id,
)
from newsfeed.features.baselines import (
    compute_baselines,
    finbert_score,
    lm_counts,
    load_lm_dictionary,
)
from newsfeed.ingest.base import RawItem
from newsfeed.ingest.runner import build_row
from newsfeed.scoring.runner import plan_targets
from newsfeed.storage import Store

from .fake_llm import FakeLLM, default_output, user_text
from .news_env import (
    HashEmbedder,
    copy_items,
    ingest_corpus,
    make_env,
    record_corpus_coverage,
)

US = 1_000_000


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


@pytest.fixture
def env(cfg, store, calendar, ticker_map):
    e = make_env(cfg, store, ticker_map)
    e.embedder = HashEmbedder()
    ingest_corpus(e, calendar)
    e.derive()
    e.score_all()
    return e


def feats(env, times, tickers=None):
    return build_features(env.store, env.cfg, times, env.c2t, env.sources(), tickers=tickers)


def row(df, ticker, t):
    out = df.filter((pl.col("ticker") == ticker) & (pl.col("as_of") == t)).to_dicts()
    assert len(out) == 1
    return out[0]


# ------------------------------------------------------------------ baselines
def test_lm_counts_negation_categories_and_removed_words(cfg):
    lm = load_lm_dictionary(cfg.features.lm_dictionary)
    text = "Revenue GAIN was strong but not profitable; litigation risk may rise. BREAKING news"
    c = lm_counts(text, lm)
    assert (c["n_positive"], c["n_negative"]) == (2, 2)  # 'not profitable' -> negative
    assert (c["n_uncertainty"], c["n_litigious"]) == (2, 1)  # 'BREAKING' is a removed word
    assert c["score"] == 0.0 and c["n_words"] == 13


def test_baselines_are_incremental_and_follow_text(env):
    lm = load_lm_dictionary(env.cfg.features.lm_dictionary)
    assert compute_baselines(env.store, lm=lm, run_id="again") == {"lm": 0}
    n_founders = env.store.query("SELECT count(*) AS n FROM item_story WHERE story_id = item_id")[
        "n"
    ][0]
    assert env.store.query("SELECT count(*) AS n FROM baselines")["n"][0] == n_founders


def test_finbert_score_weights_chunks():
    def classify(chunks):
        return [
            {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
            if "good" in c
            else {"positive": 0.1, "negative": 0.8, "neutral": 0.1}
            for c in chunks
        ]

    out = finbert_score("good " * 400 + "bad " * 200, classify, words_per_chunk=400)
    assert out["p_positive"] == pytest.approx((0.9 * 400 + 0.1 * 200) / 600)
    assert out["score"] == pytest.approx(out["p_positive"] - out["p_negative"])


# ------------------------------------------------------------------ grid & helpers
def test_as_of_grid_session_close(calendar):
    grid = as_of_grid(calendar, date(2025, 11, 26), date(2025, 12, 2), "session_close", 10)
    assert grid == [
        utc(2025, 11, 26, 20, 50),  # regular close 16:00 EST
        utc(2025, 11, 28, 17, 50),  # half day: 13:00 EST close (Thanksgiving skipped)
        utc(2025, 12, 1, 20, 50),  # weekend skipped
    ]
    opens = as_of_grid(calendar, date(2025, 12, 1), date(2025, 12, 2), "session_open", 15)
    assert opens == [utc(2025, 12, 1, 14, 45)]


def test_volume_z():
    day = 86400 * US
    T = 100 * day
    baseline = [T - k * day - day // 2 for k in range(1, 61)]  # one story per earlier day
    now = [T - 3600 * US * h for h in (1, 2, 3, 4, 5)]  # five in the current day
    f_t = np.array(sorted(baseline + now))
    z = _volume_z(f_t, T, day, 60, sigma_floor=0.5)
    assert z == pytest.approx((5 - 1) / 0.5)  # baseline mean 1, std 0 -> floor
    assert _volume_z(f_t, T, day, 1, sigma_floor=0.5) is None  # too few baseline windows


def test_feature_set_id_tracks_config_and_sources(env):
    src = env.sources()
    a = feature_set_id(env.cfg, src)
    other = env.cfg.model_copy(
        update={"features": env.cfg.features.model_copy(update={"min_relevance": "passing"})}
    )
    assert feature_set_id(other, src) != a
    assert feature_set_id(env.cfg, src.__class__(**{**src.__dict__, "llm_prompt_hash": "x"})) != a
    # machine paths are not (the project moved from C:\Users\Andrew to C:\Users\andre)
    moved = env.cfg.model_copy(
        update={
            "features": env.cfg.features.model_copy(
                update={
                    "embedding_cache": Path("Z:/elsewhere/models"),
                    "lm_dictionary": Path("Z:/elsewhere/lm.csv"),
                }
            )
        }
    )
    assert feature_set_id(moved, src) == a
    # the universe is part of the id (a 12-ticker and an S&P 500 build must not collide)
    small = env.cfg.model_copy(
        update={"universe": env.cfg.universe.model_copy(update={"tickers": ["AAPL"]})}
    )
    assert feature_set_id(small, src) != a
    assert feature_set_id(env.cfg, src, "members-v2") != a


# ------------------------------------------------------------------ values
def test_feature_values_on_corpus(env):
    T = utc(2026, 7, 31, 14, 0)  # a1-a4 (one Apple story, 3 outlets) became tradable at 13:30
    df = feats(env, [T])
    r = row(df, "AAPL", T)
    assert (r["1h_n_items"], r["1h_n_stories"], r["1h_n_outlets"]) == (4, 1, 3)
    assert r["1h_llm_n"] == 1 and r["1h_lm_n"] == 1 and r["1h_llm_coverage"] == 1.0
    a1_sent = env.store.query(
        "SELECT s.sentiment FROM scores s JOIN items i USING (item_id) "
        "WHERE i.native_id = 'fixture:a1'"
    )["sentiment"][0]
    assert r["1h_llm_sent_decay"] == pytest.approx(a1_sent * 0.5)  # 30 min, half-life 30 min
    assert r["1h_llm_sent_cwmean"] == pytest.approx(a1_sent)
    assert r["1h_llm_sent_disp"] is None  # one event
    assert r["1h_volume_z"] is None  # less than 60 days of history
    # share classes get identical company-level features
    g, gg = row(df, "GOOGL", T), row(df, "GOOG", T)
    assert {k: v for k, v in g.items() if k != "ticker"} == {
        k: v for k, v in gg.items() if k != "ticker"
    }
    assert g["5d_n_stories"] == 1  # a6 (Microsoft + Alphabet)
    assert row(df, "XYZQ", T) if False else True  # non-universe tickers never appear
    assert set(df["ticker"]) == {t for ts in env.c2t.values() for t in ts}


def test_member_spans_restrict_rows_without_changing_values(env):
    # 02:00 UTC on 07-31 is still 07-30 in New York
    times = [utc(2026, 7, 30, 19, 50), utc(2026, 7, 31, 2, 0), utc(2026, 7, 31, 19, 50)]
    full = feats(env, times)
    spans = {"AAPL": [(date(2026, 7, 31), None)], "MSFT": [(date(2020, 1, 1), date(2026, 7, 31))]}
    df = build_features(env.store, env.cfg, times, env.c2t, env.sources(), member_spans=spans)
    assert df.select("ticker", "as_of").rows() == [
        ("MSFT", times[0]),
        ("MSFT", times[1]),
        ("AAPL", times[2]),
    ]
    keyed = full.join(df.select("ticker", "as_of"), on=["ticker", "as_of"], how="semi")
    assert_frame_equal(df, keyed.sort("as_of", "ticker"))


def test_window_boundaries_exact(env):
    t_a5 = utc(2026, 7, 31, 14, 10)  # a5 published 14:05 + 5 min backfill lag
    eps = timedelta(microseconds=1)
    df = feats(env, [t_a5 - eps, t_a5], tickers=["AAPL"])
    before, at = row(df, "AAPL", t_a5 - eps), row(df, "AAPL", t_a5)
    assert at["1h_n_stories"] == before["1h_n_stories"] + 1
    assert at["1h_llm_n"] == before["1h_llm_n"] + 1
    # lower edge is open: an event exactly w before T is out of the window
    df = feats(env, [t_a5 + timedelta(hours=1)], tickers=["AAPL"])
    assert row(df, "AAPL", t_a5 + timedelta(hours=1))["1h_n_stories"] == 0


def test_relevance_filter(env):
    # a6 mentions Alphabet; make its Alphabet score 'passing' and check it drops out of LLM feats
    passing = env.store.query("SELECT * FROM scores WHERE ticker = 'GOOGL'").with_columns(
        pl.lit("passing").alias("relevance"), pl.lit("rerun").alias("run_id")
    )
    T = utc(2026, 7, 31, 14, 0)
    before = row(feats(env, [T], ["GOOGL"]), "GOOGL", T)
    assert before["5d_llm_n"] == 1
    env.store.scores_dir.joinpath("x").mkdir()
    env.store.write_scores(
        [{**r, "scored_at": r["scored_at"] - timedelta(days=1)} for r in passing.to_dicts()], "x"
    )  # earlier ok draw for the same key now becomes canonical
    after = row(feats(env, [T], ["GOOGL"]), "GOOGL", T)
    assert after["5d_llm_n"] == 0 and after["5d_n_stories"] == 1 and after["5d_lm_n"] == 1


# ------------------------------------------------------------------ causality
CAUSAL_TIMES = [
    utc(2026, 7, 29, 16, 5),  # a7 exactly at its tradable_at
    utc(2026, 7, 31, 13, 30),  # the Apple story's open
    utc(2026, 7, 31, 14, 10),  # a5 boundary
    utc(2026, 7, 31, 15, 30),  # after a11, before the weekend item (Mon open)
]


@pytest.mark.parametrize("T", CAUSAL_TIMES)
def test_features_equal_a_store_frozen_at_T(env, cfg, calendar, ticker_map, tmp_path, T):
    """The strongest check: rebuild the whole pipeline (links, stories, scores, baselines)
    from only the items tradable by T, in a fresh store, and require identical features."""
    paths = cfg.paths.model_copy(
        update={k: tmp_path / "frozen" / k for k in ("data_dir", "lake_dir", "raw_dir", "ref_dir")}
        | {"db_path": tmp_path / "frozen" / "ops.duckdb"}
    )
    frozen = make_env(cfg.model_copy(update={"paths": paths}), Store(paths), ticker_map)
    frozen.embedder = HashEmbedder()
    n = copy_items(env.store, frozen.store, "tradable_at <= $T", {"T": T})
    record_corpus_coverage(frozen, calendar)
    assert 0 < n < env.store.query("SELECT count(*) AS n FROM items")["n"][0]
    frozen.derive()
    frozen.score_all()
    # Build the full store's features the way production does -- one call over a grid whose
    # last time is after every item -- so per-T slicing (not the loader) must hide the future.
    grid = sorted({*CAUSAL_TIMES, utc(2026, 9, 30)})
    full_df = feats(env, grid).filter(pl.col("as_of") == T)
    frozen_df = build_features(frozen.store, frozen.cfg, [T], frozen.c2t, env.sources())
    assert_frame_equal(full_df, frozen_df, check_exact=False, rel_tol=1e-12)
    frozen.store.close()


def test_future_data_cannot_change_past_features(env, calendar):
    T = utc(2026, 7, 31, 14, 0)
    before = feats(env, [T])
    # new Apple items after T, including one *published* before T but only seen (live) after T
    for nid, pub, seen in (
        ("late-live", utc(2026, 7, 31, 13, 45), utc(2026, 7, 31, 14, 30)),
        ("future", utc(2026, 7, 31, 15, 0), utc(2026, 7, 31, 15, 1)),
    ):
        item = RawItem(
            source="news_api",
            native_id=f"test:{nid}",
            publisher="Late Wire",
            url=None,
            title="Apple (NASDAQ: AAPL) announces something new",
            body_text="x " * 200,
            body_tokens_est=100,
            body_truncated=False,
            tickers=["AAPL"],
            published_at=pub,
            published_at_raw=pub.isoformat(),
            first_seen_at=seen,
            ingest_mode="live",
            meta={},
            raw_json="{}",
            parser_version="t",
        )
        env.store.write_items(
            [
                build_row(
                    item,
                    run_id="late",
                    calendar=calendar,
                    backfill_lag=timedelta(0),
                    ingested_at=seen,
                )
            ],
            "late",
        )
    env.derive()
    env.score_all()
    after = feats(env, [T])
    assert_frame_equal(before, after)


def test_live_scores_count_only_after_they_exist(env, calendar):
    """A live item is tradable at 14:21, but its score was computed 'now' (long after):
    the item counts toward volume at T, its LLM score does not."""
    item = RawItem(
        source="news_api",
        native_id="test:live",
        publisher="Live Wire",
        url=None,
        title="Apple (NASDAQ: AAPL) unveils a new product",
        body_text="y " * 200,
        body_tokens_est=100,
        body_truncated=False,
        tickers=["AAPL"],
        published_at=utc(2026, 7, 31, 14, 20),
        published_at_raw="x",
        first_seen_at=utc(2026, 7, 31, 14, 21),
        ingest_mode="live",
        meta={},
        raw_json="{}",
        parser_version="t",
    )
    env.store.write_items(
        [
            build_row(
                item,
                run_id="live",
                calendar=calendar,
                backfill_lag=timedelta(0),
                ingested_at=utc(2026, 7, 31, 14, 21),
            )
        ],
        "live",
    )
    env.derive()
    env.score_all()
    T = utc(2026, 7, 31, 15, 0)
    r = row(feats(env, [T], ["AAPL"]), "AAPL", T)
    assert r["1h_n_stories"] == 2  # a5 (14:10) + the live item (14:21)
    assert r["1h_llm_n"] == 1  # only a5: the live item's score did not exist at T
    assert r["1h_llm_coverage"] == pytest.approx(1.0)  # both scored by now, as of storage
    assert r["1h_lm_n"] == 2  # baselines are instant


def test_default_fake_is_stable_under_rescoring():
    # guard for the frozen-store test: identical text -> identical fake scores
    llm = FakeLLM()
    p = {"messages": [{"role": "user", "content": "same"}]}
    assert user_text(p) == "same"
    assert llm.respond(p).content[1].text == llm.respond(p).content[1].text
    assert default_output("same") == default_output("same")


# ------------------------------------------------------------------ coverage
def test_backfill_coverage_starts_after_items_published_before_start(store, calendar):
    # backfill from Saturday: Friday-evening items (not collected) become tradable Monday 09:30,
    # so coverage starts there -- windows are (lo, hi], so lo = Monday 13:30 UTC is fine
    coverage.record_backfill(
        store,
        "edgar",
        ["AAPL"],
        utc(2026, 8, 1),
        utc(2026, 9, 1),
        lag=timedelta(minutes=5),
        calendar=calendar,
        run_id="r",
    )
    ((s, e),) = coverage.merged(store, "edgar")["AAPL"]
    assert (s, e) == (utc(2026, 8, 3, 13, 30), utc(2026, 9, 1))
    assert coverage.is_covered([(s, e)], s, s + timedelta(hours=1))
    assert not coverage.is_covered([(s, e)], s - timedelta(microseconds=1), s)
    assert not coverage.is_covered([(s, e)], e - timedelta(hours=1), e)  # end exclusive


def test_adjacent_backfills_merge_without_phantom_gap(store, calendar):
    lag = timedelta(seconds=60)
    for a, b in ((utc(2025, 1, 1), utc(2026, 1, 1)), (utc(2026, 1, 1), utc(2026, 2, 1))):
        coverage.record_backfill(
            store, "edgar", ["AAPL"], a, b, lag=lag, calendar=calendar, run_id="r"
        )
    assert coverage.merged(store, "edgar")["AAPL"] == [(utc(2025, 1, 2, 14, 30), utc(2026, 2, 1))]
    # a real gap stays a gap (with the conservative restart after it)
    coverage.record_backfill(
        store,
        "edgar",
        ["MSFT"],
        utc(2025, 1, 1),
        utc(2025, 3, 1),
        lag=lag,
        calendar=calendar,
        run_id="r",
    )
    coverage.record_backfill(
        store,
        "edgar",
        ["MSFT"],
        utc(2025, 3, 8),
        utc(2025, 4, 1),
        lag=lag,
        calendar=calendar,
        run_id="r",
    )
    assert coverage.merged(store, "edgar")["MSFT"] == [
        (utc(2025, 1, 2, 14, 30), utc(2025, 3, 1)),
        (utc(2025, 3, 10, 13, 30), utc(2025, 4, 1)),  # Sat 8 Mar -> Mon open (EDT)
    ]


def test_uncovered_windows_are_null_not_zero(env):
    T = utc(2026, 7, 2, 18, 0)  # inside coverage, no news: zeros
    r = row(feats(env, [T], ["AAPL"]), "AAPL", T)
    assert r["1d_n_stories"] == 0 and r["1d_llm_n"] == 0
    T0 = utc(2026, 6, 30, 18, 0)  # before collection started: unknown
    r0 = row(feats(env, [T0], ["AAPL"]), "AAPL", T0)
    assert all(v is None for k, v in r0.items() if k not in ("ticker", "as_of"))
    assert r["1d_volume_z"] is None  # baseline (60 days back) not covered either


def test_volume_z_needs_covered_baseline(env, calendar):
    T = utc(2026, 7, 31, 14, 0)
    assert row(feats(env, [T], ["AAPL"]), "AAPL", T)["1d_volume_z"] is None
    tickers = [t for ts in env.c2t.values() for t in ts]
    coverage.record_backfill(
        env.store,
        "news_api",
        tickers,
        utc(2026, 4, 1),
        utc(2026, 7, 1),
        lag=timedelta(minutes=5),
        calendar=calendar,
        run_id="old",
    )  # an earlier backfill ending where the corpus begins: merges without a gap
    z = row(feats(env, [T], ["AAPL"]), "AAPL", T)["1d_volume_z"]
    # one story in the window, none in the 60 preceding days -> (1 - 0) / sigma_floor
    assert z == pytest.approx(1 / env.cfg.features.volume_sigma_floor)


def test_live_coverage_extends_one_row(store):
    a, b, c = utc(2026, 8, 3, 14), utc(2026, 8, 3, 14, 1), utc(2026, 8, 3, 14, 2)
    coverage.record_live(store, "edgar", ["AAPL", "MSFT"], a, b, "r")
    coverage.record_live(store, "edgar", ["AAPL", "MSFT"], b, c, "r")
    assert store.ops_query("SELECT count(*) AS n FROM coverage")["n"][0] == 2
    assert coverage.merged(store, "edgar")["AAPL"] == [(a, c)]


def test_rebuild_coverage_from_runs(store, cfg, calendar):
    run = store.start_run(
        "backfill",
        cfg,
        {"source": "edgar", "start": "2025-01-01", "end": "2026-01-01", "tickers": "AAPL,MSFT"},
    )
    store.finish_run(run, "ok", {})
    run2 = store.start_run(
        "backfill",
        cfg,
        {"source": "edgar", "start": "2026-01-01", "end": "2026-02-01", "tickers": None},
    )
    store.finish_run(run2, "partial", {})
    n = coverage.rebuild_from_runs(store, calendar, {"edgar": 60}, ["AAPL", "MSFT", "JPM"])
    assert n == 2
    m = coverage.merged(store, "edgar")
    assert m["AAPL"] == [(utc(2025, 1, 2, 14, 30), utc(2026, 2, 1))]  # 1 Jan holiday
    assert m["JPM"] == [(utc(2026, 1, 2, 14, 30), utc(2026, 2, 1))]
    assert coverage.rebuild_from_runs(store, calendar, {"edgar": 60}, ["AAPL"]) == 2  # idempotent


# ------------------------------------------------------------------ embeddings / novelty
def test_max_prior_similarity_is_strictly_backward():
    from newsfeed.features.embeddings import max_prior_similarity

    t = np.array([0, 10, 10, 50, 200]) * US
    v = np.eye(3, dtype=np.float32)[[0, 0, 0, 1, 0]]
    sims = max_prior_similarity(t, v, lookback_us=100 * US)
    # t=0: no prior. t=10 (x2): see t=0 (same vector) but not each other (not strictly earlier).
    # t=50: priors exist but are orthogonal. t=200: look-back [100, 200) is empty.
    assert sims.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_embeddings_incremental(env):
    from newsfeed.features.embeddings import compute_embeddings

    founders = env.store.query("SELECT count(*) AS n FROM item_story WHERE story_id = item_id")
    assert env.store.query("SELECT count(*) AS n FROM embeddings")["n"][0] == founders["n"][0]
    assert compute_embeddings(env.store, env.embedder, "again") == 0
    v = env.store.query("SELECT vector FROM embeddings LIMIT 1")["vector"][0]
    assert abs(np.linalg.norm(np.array(v)) - 1) < 1e-5


def test_novelty_features(env):
    T = utc(2026, 7, 31, 14, 10)  # a1 (13:30, first Apple story) and a5 (14:10, follow-up)
    r = row(feats(env, [T], ["AAPL"]), "AAPL", T)
    assert r["1h_emb_n"] == 2
    vecs = env.store.query(
        "SELECT i.native_id, e.vector FROM embeddings e JOIN items i USING (item_id) "
        "WHERE i.native_id IN ('fixture:a1', 'fixture:a5') ORDER BY i.native_id"
    )["vector"].to_list()
    sim = float(np.dot(vecs[0], vecs[1]))
    assert r["1h_emb_novelty_mean"] == pytest.approx(((1 - 0.0) + (1 - sim)) / 2, rel=1e-5)
    assert r["1h_emb_n_novel"] == (1 + (sim < env.cfg.features.novelty_sim_threshold))
    # look-back not covered (corpus coverage starts 2026-07-01): novelty is unknown
    early = utc(2026, 7, 2, 18, 0)
    assert row(feats(env, [early], ["AAPL"]), "AAPL", early)["5d_emb_novelty_mean"] is None


def test_feature_sources_filter(env):
    T = utc(2026, 7, 31, 14, 0)
    assert row(feats(env, [T]), "AAPL", T)["1h_n_items"] == 4
    other = env.cfg.model_copy(
        update={"features": env.cfg.features.model_copy(update={"sources": ["edgar"]})}
    )
    df = build_features(env.store, other, [T], env.c2t, env.sources())
    assert row(df, "AAPL", T)["1h_n_items"] == 0  # corpus is news_api only
    assert feature_set_id(other, env.sources()) != feature_set_id(env.cfg, env.sources())


def test_features_can_use_masked_scores(env):
    T = utc(2026, 7, 31, 14, 0)
    base = row(feats(env, [T]), "AAPL", T)
    masked_cfg = env.cfg.model_copy(
        update={"features": env.cfg.features.model_copy(update={"llm_masked": True})}
    )
    df = build_features(env.store, masked_cfg, [T], env.c2t, env.sources())
    assert base["1h_llm_n"] == 1 and row(df, "AAPL", T)["1h_llm_n"] == 0  # none masked yet
    j = env.job(FakeLLM())
    j.run_live(plan_targets(j, env.c2t, masked=True))
    df = build_features(env.store, masked_cfg, [T], env.c2t, env.sources())
    r = row(df, "AAPL", T)
    assert r["1h_llm_n"] == 1 and r["1h_llm_sent_cwmean"] is not None
    assert r["1h_lm_n"] == base["1h_lm_n"]  # non-LLM features unchanged
    assert feature_set_id(masked_cfg, env.sources()) != feature_set_id(env.cfg, env.sources())


def test_llm_event_type_and_title_filters(env):
    T = utc(2026, 7, 31, 14, 0)
    base = row(feats(env, [T]), "AAPL", T)
    assert base["1h_n_items"] == 4 and base["1h_llm_n"] == 1
    titles = env.store.query("SELECT title FROM items WHERE native_id = 'fixture:a1'")["title"]
    word = titles[0].split()[0]
    no_a1 = env.cfg.model_copy(
        update={"features": env.cfg.features.model_copy(update={"exclude_title_regex": word})}
    )
    r = row(build_features(env.store, no_a1, [T], env.c2t, env.sources()), "AAPL", T)
    assert r["1h_n_items"] < 4
    other_types = env.cfg.model_copy(
        update={"features": env.cfg.features.model_copy(update={"llm_event_types": ["m&a"]})}
    )
    r = row(build_features(env.store, other_types, [T], env.c2t, env.sources()), "AAPL", T)
    assert r["1h_llm_n"] == 0 and r["1h_n_items"] == 4  # only LLM aggregates are filtered
