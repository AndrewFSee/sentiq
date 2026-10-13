"""Configuration: YAML file for everything tunable, .env for secrets.

All models use ``extra="forbid"`` so a typo in the YAML fails loudly instead of silently
falling back to a default.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_dir: Path
    lake_dir: Path
    raw_dir: Path
    db_path: Path
    ref_dir: Path

    def resolved(self, root: Path) -> PathsConfig:
        def fix(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        return PathsConfig(**{k: fix(v) for k, v in self.model_dump().items()})


class UniverseConfig(_Strict):
    # static: the `tickers` list below. sp500_pit: point-in-time S&P 500 membership built by
    # `newsfeed universe build` (universe_pit.py); `tickers` is then only an optional subset.
    mode: Literal["static", "sp500_pit"] = "static"
    tickers: list[str] = Field(default_factory=list)
    pit_dir: Path = Path("data/ref/universe")
    pit_start: date = date(2019, 1, 1)
    fja_sha: str = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"  # github.com/fja05680/sp500
    ingest_margin_before_days: int = Field(90, ge=0)  # history before joining (baselines)
    ingest_margin_after_days: int = Field(30, ge=0)
    # sec_names: also match cleaned company names; aliases_only: only `aliases` (safer for a
    # 500-name universe where names like "Target" or "Ball" collide with ordinary words)
    name_linking: Literal["sec_names", "aliases_only"] = "sec_names"
    # sources whose items link only through their source tags (no name/cashtag matching)
    source_only_linking: list[str] = Field(default_factory=lambda: ["transcripts"])
    # Company-name aliases for text linking, matched case-sensitively on word boundaries.
    # A cleaned SEC company name is added automatically.
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    # Extra identifying terms replaced in masked scoring (leakage check): product names,
    # people (executives) and places (HQ towns). Matched case-sensitively on word boundaries.
    products: dict[str, list[str]] = Field(default_factory=dict)
    people: dict[str, list[str]] = Field(default_factory=dict)
    places: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("tickers")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        out = [t.strip().upper() for t in v if t.strip()]
        if len(set(out)) != len(out):
            raise ValueError("duplicate tickers in universe")
        return out


class MarketConfig(_Strict):
    calendar: str = "XNYS"
    calendar_start: str = "2000-01-01"


class TimingConfig(_Strict):
    backfill_lag_seconds: dict[str, int]

    def lag_for(self, source: str) -> int:
        if source not in self.backfill_lag_seconds:
            raise KeyError(f"no backfill_lag_seconds configured for source {source!r}")
        return self.backfill_lag_seconds[source]


class EdgarConfig(_Strict):
    forms: list[str] = Field(default_factory=lambda: ["8-K", "8-K/A"])
    max_requests_per_second: float = Field(8.0, gt=0, le=10)
    timeout_seconds: float = 30.0
    max_retries: int = 5
    submissions_base_url: str = "https://data.sec.gov/submissions"
    archives_base_url: str = "https://www.sec.gov/Archives/edgar/data"
    ticker_map_url: str = "https://www.sec.gov/files/company_tickers.json"
    exhibit_type_prefixes: list[str] = Field(default_factory=lambda: ["EX-99"])
    max_tokens: int = Field(6000, gt=0)
    chars_per_token: float = Field(4.0, gt=0)
    flush_every: int = Field(25, gt=0)
    # live polling via EDGAR's "latest filings" Atom feed (all companies, newest first)
    current_feed_url: str = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K"
        "&company=&dateb=&owner=include&output=atom"
    )
    current_feed_page_size: int = Field(100, gt=0, le=100)
    current_feed_max_pages: int = Field(5, gt=0)
    poll_lookback_hours: float = Field(48, gt=0)  # entries older than this are ignored by poll
    catch_up_every_minutes: float = Field(60, gt=0)  # per-CIK submissions sweep (missed entries)
    silence_alert_minutes: float = Field(45, gt=0)  # market-open minutes w/o feed activity


class NewsApiConfig(_Strict):
    vendor: Literal["fixture", "massive"] = "fixture"
    massive_base_url: str = "https://api.massive.com"  # Massive (ex-Polygon); MASSIVE_API_KEY
    massive_requests_per_minute: float = Field(5, gt=0)  # free Basic tier: 5 req/min
    fixture_path: Path = Path("tests/fixtures/news/articles.jsonl")  # relative to project root
    page_size: int = Field(100, gt=0)
    poll_overlap_minutes: float = Field(15, ge=0)  # re-request this much before the watermark
    poll_lookback_hours: float = Field(24, gt=0)  # first poll / catch-up window
    catch_up_every_minutes: float = Field(60, gt=0)
    filter_to_universe: bool = True  # drop articles whose vendor tickers miss the universe
    max_tokens: int = Field(3000, gt=0)
    chars_per_token: float = Field(4.0, gt=0)
    flush_every: int = Field(100, gt=0)
    silence_alert_minutes: float = Field(30, gt=0)


class TranscriptsConfig(_Strict):
    """Earnings-call transcripts from a pinned local copy of the defeatbeta dataset."""

    path: Path = Path("data/ref/defeatbeta/stock_earning_call_transcripts.parquet")
    dataset_commit: str = "860d42ac253f8df98c93c181b0d873f21e982e48"
    max_tokens: int = Field(16000, gt=0)  # median call ~10k tokens
    chars_per_token: float = Field(4.0, gt=0)
    drop_speakers: list[str] = Field(default_factory=lambda: ["AI Insights", "Disclaimer*"])
    flush_every: int = Field(200, gt=0)
    poll_lookback_days: int = Field(10, gt=0)


class SourcesConfig(_Strict):
    edgar: EdgarConfig = Field(default_factory=EdgarConfig)
    news_api: NewsApiConfig = Field(default_factory=NewsApiConfig)
    transcripts: TranscriptsConfig = Field(default_factory=TranscriptsConfig)


class CollectConfig(_Strict):
    interval_seconds: float = Field(60, gt=0)
    sources: list[str] = Field(default_factory=lambda: ["edgar"])
    max_consecutive_failures: int = Field(3, gt=0)  # alert threshold
    stalled_after_intervals: float = Field(3, gt=1)  # heartbeat older than N intervals = stalled
    realert_minutes: float = Field(60, gt=0)  # repeat an ongoing alert at most this often


class DedupeConfig(_Strict):
    near_dup_threshold: float = Field(0.8, gt=0, le=1)  # estimated Jaccard of word shingles
    num_perm: int = Field(128, gt=0)
    shingle_words: int = Field(5, gt=0)
    max_chars: int = Field(5000, gt=0)  # of title + body used for near-dup detection
    window_hours: float = Field(72, gt=0)  # exact/near matches only within this window
    require_ticker_overlap: bool = True  # when both items have tickers
    seed: int = 1


class LinkPolicy(_Strict):
    """Which ticker links get an LLM score (user decision, M3)."""

    methods: list[str] = Field(
        default_factory=lambda: ["source", "exchange_tag", "cashtag", "name_title"]
    )
    name_body_min_mentions: int = Field(2, ge=1)  # body-only name links need >= this many


class PriceCard(_Strict):
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_read: float


class ModelPricing(_Strict):
    standard: PriceCard
    long: PriceCard | None = None  # prompts above long_threshold_tokens (Haiku 5.5)
    long_threshold_tokens: int | None = None


class PricingConfig(_Strict):
    as_of: str  # date the table was checked against Anthropic's pricing page
    batch_discount: float = Field(0.5, gt=0, le=1)  # stacks with cache multipliers
    models: dict[str, ModelPricing] = Field(default_factory=dict)


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ScoringConfig(_Strict):
    model: str = "claude-haiku-5-5"
    validation_model: str = "claude-sonnet-5-5"
    prompt_version: str = "score_v1"
    effort: Effort = "low"
    thinking: Literal["adaptive", "disabled"] = "adaptive"
    max_tokens: int = Field(2048, gt=0)  # includes thinking tokens
    link_policy: LinkPolicy = Field(default_factory=LinkPolicy)
    founders_only: bool = True  # score each story once, on its first appearance
    live_concurrency: int = Field(4, ge=1)
    batch_max_requests: int = Field(10_000, ge=1, le=100_000)
    batch_max_bytes: int = Field(150_000_000, ge=1, le=250_000_000)  # API limit 256 MB
    batch_poll_seconds: float = Field(60, gt=0)
    dry_run_sample: int = Field(25, ge=1)  # requests token-counted to estimate a job
    est_output_tokens: int = Field(300, gt=0)  # used until the ledger has real averages
    confirm_above_usd: float = Field(1.0, ge=0)  # jobs estimated above this need --yes
    live_in_collect: bool = False  # score new story founders inside the collect loop
    live_lookback_hours: float = Field(48, gt=0)


class FeaturesConfig(_Strict):
    """Per (ticker, as-of time T, window w) aggregates over events in (T - w, T]."""

    windows_hours: dict[str, float] = Field(
        default_factory=lambda: {"1h": 1, "1d": 24, "5d": 168}  # 5d = one calendar week
    )
    half_life_hours: dict[str, float] = Field(
        default_factory=lambda: {"1h": 0.5, "1d": 6, "5d": 36}
    )
    min_relevance: Literal["primary", "secondary", "passing"] = "secondary"
    baseline_days: int = Field(60, gt=1)  # abnormal-volume baseline length
    volume_sigma_floor: float = Field(0.5, gt=0)  # avoids infinite z on quiet tickers
    grid: Literal["session_close", "session_open"] = "session_close"
    decision_lag_minutes: float = Field(10, ge=0)  # close: T = close - lag; open: T = open + lag
    lm_dictionary: Path = Path("data/ref/lm/Loughran-McDonald_MasterDictionary_1993-2025.csv")
    # sources whose coverage a window must have (else its features are null, not zero)
    coverage_sources: list[str] = Field(default_factory=lambda: ["edgar"])
    # item sources that feed events ([] = all). E.g. [news_api] isolates vendor news (test 4).
    sources: list[str] = Field(default_factory=list)
    # use masked-text LLM scores (leakage check: names/tickers/products/people/places hidden)
    llm_masked: bool = False
    # LLM aggregates only over these event types ([] = all), e.g. company-specific news
    llm_event_types: list[str] = Field(default_factory=list)
    # items whose title matches (Python regex, case-insensitive) are dropped from all events,
    # e.g. law-firm class-action solicitations and market-research report releases
    exclude_title_regex: str | None = None
    embeddings: bool = True  # local fastembed model; vectors for story founders
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache: Path = Path("data/ref/models")
    novelty_lookback_days: float = Field(30, gt=0)
    novelty_sim_threshold: float = Field(0.85, gt=0, le=1)  # max_sim below this = novel
    finbert: bool = False  # needs `uv sync --extra finbert`
    finbert_model: str = "ProsusAI/finbert"

    @field_validator("half_life_hours")
    @classmethod
    def _same_keys(cls, v: dict[str, float], info: Any) -> dict[str, float]:
        windows = info.data.get("windows_hours", {})
        if set(v) != set(windows):
            raise ValueError("half_life_hours must have the same keys as windows_hours")
        return v


class CVConfig(_Strict):
    mode: Literal["walk_forward", "blocked"] = "walk_forward"
    n_folds: int = Field(5, ge=2)
    embargo_sessions: int = Field(5, ge=0)  # gap after each test block (blocked mode)
    min_train_sessions: int = Field(60, ge=1)  # walk-forward: first fold needs this history


class ModelsConfig(_Strict):
    ridge_alpha: float = Field(10.0, gt=0)
    lgbm: dict[str, float | int] = Field(
        default_factory=lambda: {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": 7,
        }
    )


class EvalConfig(_Strict):
    tradier_base_url: str = "https://api.tradier.com"
    tradier_requests_per_second: float = Field(1.5, gt=0)  # account limit is 120/min
    # Tiingo free tier: 50 requests/hour, 1000/day, 500 unique symbols/month
    tiingo_requests_per_hour: float = Field(48, gt=0)
    # universe ticker -> "tradier:SYM" | "tiingo:SYM" where routing by SEC CIK gets it wrong
    price_overrides: dict[str, str] = Field(default_factory=dict)
    market_etf: str = "SPY"
    sector_etfs: dict[str, str] = Field(default_factory=dict)  # ticker -> SPDR sector ETF
    # sp500_pit mode: point-in-time GICS sector -> SPDR sector ETF (overrides sector_etfs)
    gics_sector_etfs: dict[str, str] = Field(
        default_factory=lambda: {
            "Information Technology": "XLK",
            "Financials": "XLF",
            "Health Care": "XLV",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
            "Communications Services": "XLC",  # one Wikipedia revision's spelling
        }
    )
    horizons: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 20])  # sessions
    entries: list[Literal["close", "next_open"]] = Field(
        default_factory=lambda: ["close", "next_open"]
    )
    adjustments: list[Literal["market", "sector"]] = Field(
        default_factory=lambda: ["market", "sector"]
    )
    holdout_start: date  # data on/after this date is frozen; eval refuses it without a flag
    # news = ticker-days with >= 1 story in the window; llm = >= 1 LLM event that passes the
    # relevance (and event-type) filters, i.e. the rows where LLM sentiment is defined
    sample: Literal["news", "all", "llm"] = "news"
    sample_window: str = "1d"
    # optional event-type subset (LLM event_type of scored founders in the sample window), e.g.
    # [earnings]: keep ticker-days with >= 1 such event. Defines a *different* test sample.
    sample_event_types: list[str] = Field(default_factory=list)
    # optional size subset: PIT mcap terciles among all members on the date
    sample_size_buckets: list[Literal["small", "mid", "large", "unknown"]] = Field(
        default_factory=list
    )
    # exploratory per-type analysis: rows whose LLM events in the window are ALL of this type
    sample_pure_event_type: str | None = None
    cv: CVConfig = Field(default_factory=CVConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    fdr_q: float = Field(0.10, gt=0, lt=1)  # Benjamini-Hochberg level across the grid
    bootstrap_reps: int = Field(1000, ge=100)
    seed: int = 7


class Secrets(BaseSettings):
    """Read from environment / .env. Never logged, never hashed into config_hash."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    sec_user_agent: str | None = None
    anthropic_api_key: str | None = None
    tradier_token: str | None = Field(
        None, validation_alias=AliasChoices("TRADIER_TOKEN", "TRADIER_API_TOKEN")
    )
    massive_api_key: str | None = Field(
        None, validation_alias=AliasChoices("MASSIVE_API_KEY", "POLYGON_API_KEY")
    )
    tiingo_token: str | None = Field(
        None, validation_alias=AliasChoices("TIINGO_TOKEN", "TIINGO_API_TOKEN", "TIINGO_API_KEY")
    )

    def require_sec_user_agent(self) -> str:
        ua = (self.sec_user_agent or "").strip()
        # SEC asks for "Company/Name contact@domain"; insist on at least an email-like token.
        if not ua or "@" not in ua:
            raise RuntimeError(
                "SEC_USER_AGENT is not set (or has no contact email). Add e.g. "
                "'SEC_USER_AGENT=\"YourOrg research you@example.edu\"' to .env"
            )
        return ua


class AppConfig(_Strict):
    paths: PathsConfig
    universe: UniverseConfig
    market: MarketConfig = Field(default_factory=MarketConfig)
    timing: TimingConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    eval: EvalConfig
    pricing: PricingConfig

    def config_hash(self) -> str:
        """Stable hash of the (non-secret, machine-independent) config, recorded on every run."""
        blob = json.dumps(self.model_dump(mode="json", exclude={"paths"}), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: Path | str | None = None, root: Path | None = None) -> AppConfig:
    """Load YAML config. Relative paths resolve against ``root`` (default: project root)."""
    path = Path(path or os.environ.get("NEWSFEED_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig.model_validate(raw)
    paths = cfg.paths.resolved(root or PROJECT_ROOT)
    news = cfg.sources.news_api
    if not news.fixture_path.is_absolute():  # fixtures ship with the code: resolve vs. project
        news = news.model_copy(update={"fixture_path": PROJECT_ROOT / news.fixture_path})
    trans = cfg.sources.transcripts
    if not trans.path.is_absolute():  # reference data lives with the project
        trans = trans.model_copy(update={"path": PROJECT_ROOT / trans.path})
    sources = cfg.sources.model_copy(update={"news_api": news, "transcripts": trans})
    feats = cfg.features
    uni = cfg.universe
    if not uni.pit_dir.is_absolute():
        uni = uni.model_copy(update={"pit_dir": (root or PROJECT_ROOT) / uni.pit_dir})
    for key in ("lm_dictionary", "embedding_cache"):  # reference data lives with the project
        if not getattr(feats, key).is_absolute():
            feats = feats.model_copy(update={key: PROJECT_ROOT / getattr(feats, key)})
    return cfg.model_copy(
        update={"paths": paths, "sources": sources, "features": feats, "universe": uni}
    )


def load_secrets() -> Secrets:
    return Secrets()
