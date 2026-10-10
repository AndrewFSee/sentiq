"""Reusable pipeline over the synthetic news corpus: ingest -> link -> stories -> score ->
baselines, with a fake LLM. Used by feature tests, including the 'store frozen at T' check."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from newsfeed import coverage
from newsfeed.config import AppConfig
from newsfeed.dedupe import rebuild_stories
from newsfeed.derive import derive_baselines, relink
from newsfeed.features.aggregate import FeatureSources
from newsfeed.features.baselines import load_lm_dictionary
from newsfeed.features.embeddings import compute_embeddings
from newsfeed.ingest.news_api import FixtureVendor, NewsApiAdapter
from newsfeed.ingest.runner import ingest, new_stats
from newsfeed.scoring.mask import MaskLexicon
from newsfeed.scoring.runner import ScoringJob, plan_targets
from newsfeed.storage import Store
from newsfeed.tickers import TickerLinker, TickerMap, Universe

from .conftest import FIXTURES, FakeClock
from .fake_llm import FakeLLM

NAMES = {320193: "Apple Inc.", 789019: "MICROSOFT CORP", 1652044: "Alphabet Inc."}


@dataclass
class NewsEnv:
    cfg: AppConfig
    store: Store
    universe: Universe
    linker: TickerLinker
    embedder: Any = None  # tests opt in with HashEmbedder

    @property
    def c2t(self) -> dict[int, list[str]]:
        return self.universe.cik_to_tickers

    def job(self, client: Any = None, run_id: str = "score") -> ScoringJob:
        return ScoringJob(
            self.cfg,
            self.store,
            client,
            run_id=run_id,
            model=self.cfg.scoring.model,
            company_names=NAMES,
            lexicon=MaskLexicon.build(self.cfg.universe, self.universe, NAMES),
        )

    def derive(self) -> None:
        relink(self.store, self.linker, "derive")
        rebuild_stories(self.store, self.cfg.dedupe, self.linker.linker_id)
        derive_baselines(self.store, self.cfg, "derive")
        if self.embedder is not None:
            compute_embeddings(self.store, self.embedder, "derive")

    def score_all(self, llm: FakeLLM | None = None) -> None:
        j = self.job(llm or FakeLLM())
        targets = plan_targets(j, self.c2t)
        if targets:
            j.run_live(targets)

    def sources(self) -> FeatureSources:
        return FeatureSources(
            llm_model=self.cfg.scoring.model,
            llm_prompt_hash=self.job().prompt_hash,
            lm_version=load_lm_dictionary(self.cfg.features.lm_dictionary).version,
            finbert_version=None,
            embedding_model=self.embedder.name if self.embedder is not None else None,
        )


def make_env(cfg: AppConfig, store: Store, ticker_map: TickerMap) -> NewsEnv:
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
    feats = cfg.features.model_copy(update={"coverage_sources": ["news_api"]})
    cfg = cfg.model_copy(update={"universe": ucfg, "features": feats})
    return NewsEnv(cfg, store, universe, TickerLinker(universe, ucfg.aliases))


CORPUS_START = datetime(2026, 7, 1, tzinfo=UTC)
CORPUS_END = datetime(2026, 9, 1, tzinfo=UTC)


def record_corpus_coverage(env: NewsEnv, calendar: Any) -> None:
    lag = timedelta(seconds=env.cfg.timing.lag_for("news_api"))
    tickers = [t for ts in env.c2t.values() for t in ts]
    coverage.record_backfill(
        env.store,
        "news_api",
        tickers,
        CORPUS_START,
        CORPUS_END,
        lag=lag,
        calendar=calendar,
        run_id="cov",
    )


def ingest_corpus(env: NewsEnv, calendar: Any) -> None:
    clock = FakeClock(datetime(2026, 10, 8, tzinfo=UTC))
    adapter = NewsApiAdapter(
        env.cfg.sources.news_api,
        FixtureVendor(FIXTURES / "news" / "articles.jsonl", clock=clock),
        env.universe,
        clock=clock,
    )
    items = adapter.backfill(
        CORPUS_START,
        CORPUS_END,
        skip=env.store.has_item,
        on_error=lambda *_: None,
    )
    ingest(
        items,
        store=env.store,
        cfg=env.cfg,
        calendar=calendar,
        run_id="ing",
        flush_every=100,
        stats=new_stats(),
    )
    record_corpus_coverage(env, calendar)


def copy_items(src: Store, dst: Store, rows_where: str, params: dict[str, Any]) -> int:
    """Copy stored item rows (verbatim, incl. timing) into another store."""
    rows: Sequence[dict[str, Any]] = src.query(
        f"SELECT * EXCLUDE (text_hash) FROM items WHERE {rows_where}", params
    ).to_dicts()
    return dst.write_items(list(rows), "copy")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder: similar wording -> similar vectors."""

    name = "test-hash-64"

    def embed(self, texts: list[str]) -> Any:
        import hashlib

        import numpy as np

        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
