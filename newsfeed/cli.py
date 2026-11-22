"""Command-line entry point: ``newsfeed <command>``."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import polars as pl
import typer

from newsfeed import coverage
from newsfeed.collect import Collector, check_states
from newsfeed.config import MARKET_SOURCES, AppConfig, load_config, load_secrets
from newsfeed.derive import derive as run_derive
from newsfeed.distill import dataset as distill_ds
from newsfeed.eval.caps import load_shares, market_caps
from newsfeed.eval.prices import (
    CachedPrices,
    PriceRoute,
    TiingoPrices,
    TradierPrices,
    build_panel,
    horizon_days,
    route_prices,
)
from newsfeed.eval.report import (
    DEFAULT_UNIVARIATE,
    EvalInputs,
    add_derived,
    assemble,
    benjamini_hochberg,
    bp,
    decision_dates,
    eval_config_id,
    p_floor_note,
    paired_ic_comparison,
    partial_ic_cells,
    render_markdown,
    run_eval,
    univariate_cells,
)
from newsfeed.eval.targets import (
    adjusted_targets,
    controls,
    forward_returns,
    volatility_targets,
)
from newsfeed.features.aggregate import (
    FeatureSources,
    as_of_grid,
    build_features,
    feature_set_id,
)
from newsfeed.features.baselines import load_lm_dictionary
from newsfeed.ingest.base import SourceAdapter
from newsfeed.ingest.edgar import PARSER_VERSION as EDGAR_PARSER_VERSION
from newsfeed.ingest.edgar import EdgarAdapter
from newsfeed.ingest.http import HttpClient
from newsfeed.ingest.news_api import NewsApiAdapter, make_vendor
from newsfeed.ingest.reparse import reparse_source
from newsfeed.ingest.runner import ingest, make_error_recorder, new_stats
from newsfeed.scoring import agreement as agr
from newsfeed.scoring import batch as batch_mod
from newsfeed.scoring.client import AnthropicClient, LLMClient
from newsfeed.scoring.mask import MaskLexicon
from newsfeed.scoring.prompt import load_prompt
from newsfeed.scoring.runner import ScoringJob, estimate_cost, plan_targets
from newsfeed.storage import FileLock, Store
from newsfeed.tickers import TickerHolders, TickerLinker, TickerMap, Universe, load_ticker_map
from newsfeed.timeutil import MarketCalendar, get_calendar, utcnow
from newsfeed.universe_pit import PitUniverse

NY_TZ = ZoneInfo("America/New_York")

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("newsfeed")

SOURCES = (
    "edgar",
    "edgar_periodic",
    "fed",
    "news_api",
    "transcripts",
    "transcripts_qa",
    "transcripts_prep",
)
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="YAML config path")]
TickersOpt = Annotated[
    str | None, typer.Option("--tickers", help="Comma-separated subset of the universe")
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _parse_day(s: str) -> datetime:
    return datetime.combine(date.fromisoformat(s), datetime.min.time(), tzinfo=UTC)


def _split(s: str | None) -> list[str]:
    return [t.strip() for t in (s or "").split(",") if t.strip()]


class Context:
    """Lazily-built shared objects for one command invocation."""

    def __init__(self, cfg: AppConfig, store: Store, stack: ExitStack, tickers: str | None) -> None:
        self.cfg = cfg
        self.store = store
        self.stack = stack
        self.calendar: MarketCalendar = get_calendar(cfg.market.calendar, cfg.market.calendar_start)
        self._http: HttpClient | None = None
        self._tmap: TickerMap | None = None
        self._pit: PitUniverse | None = None
        self.subset = [t.upper() for t in _split(tickers)]
        if cfg.universe.mode == "static":
            unknown = set(self.subset) - set(cfg.universe.tickers)
            if unknown:
                raise typer.BadParameter(f"not in configured universe: {sorted(unknown)}")
        self.ingest_tickers = self.subset or cfg.universe.tickers

    def http(self) -> HttpClient:
        """SEC client, created on first use (fixture-only runs never need a User-Agent)."""
        if self._http is None:
            e = self.cfg.sources.edgar
            self._http = self.stack.enter_context(
                HttpClient(
                    user_agent=load_secrets().require_sec_user_agent(),
                    rate_per_sec=e.max_requests_per_second,
                    timeout=e.timeout_seconds,
                    max_retries=e.max_retries,
                )
            )
        return self._http

    @property
    def tmap(self) -> TickerMap:
        if self._tmap is None:
            self._tmap = load_ticker_map(
                self.http, self.cfg.sources.edgar.ticker_map_url, self.cfg.paths.ref_dir
            )
        return self._tmap

    @property
    def pit(self) -> PitUniverse:
        if self._pit is None:
            d = self.cfg.universe.pit_dir
            if not (d / "sp500_members.parquet").exists():
                raise typer.BadParameter(f"no PIT universe in {d}; run `newsfeed universe build`")
            self._pit = PitUniverse.load_for(self.cfg.universe)
        return self._pit

    def universe(self, full: bool = False) -> Universe:
        if self.cfg.universe.mode == "sp500_pit":
            u = self.pit.as_universe(None if full or not self.subset else self.subset)
            if not u.cik_to_tickers:
                raise typer.BadParameter(f"none of {self.subset} were ever S&P 500 members")
            return u
        return Universe.build(self.cfg.universe.tickers if full else self.ingest_tickers, self.tmap)

    def titles(self, u: Universe) -> dict[int, str]:
        """Company names by CIK: PIT snapshots (include delisted firms) or SEC's current map."""
        if self.cfg.universe.mode == "sp500_pit":
            t = self.pit.titles()
            return {c: t.get(c, "") for c in u.ciks}
        return {c: self.tmap.cik_to_titles.get(c, "") for c in u.ciks}

    def linker(self) -> TickerLinker:
        u = self.universe(full=True)  # linking always uses the whole configured universe
        titles = None if self.cfg.universe.name_linking == "aliases_only" else self.titles(u)
        return TickerLinker(
            u,
            self.cfg.universe.aliases,
            titles,
            self.cfg.universe.source_only_linking,
            holders=self.holders(),
        )

    def holders(self) -> TickerHolders | None:
        """sp500_pit: who held each shared ticker when (None in static mode)."""
        return self.pit.holders() if self.cfg.universe.mode == "sp500_pit" else None

    def fan_out(
        self, df: pl.DataFrame, cik_col: str = "cik", date_col: str = "date"
    ) -> pl.DataFrame:
        """Company events -> one row per ticker: every ticker the CIK ever had, minus shared
        tickers held by another company on ``date_col`` (New York date), and, in PIT mode,
        only tickers that are index members on that date."""
        c2t = self.universe(full=True).cik_to_tickers
        fan = pl.DataFrame(
            [{cik_col: c, "ticker": t} for c, ts in c2t.items() for t in ts],
            schema={cik_col: pl.Int64, "ticker": pl.Utf8},
        )
        out = df.with_columns(pl.col(cik_col).cast(pl.Int64)).join(fan, on=cik_col)
        if self.cfg.universe.mode != "sp500_pit":
            return out
        out = self.pit.holders().restrict(out, cik_col, date_col)
        return out.filter(self.pit.is_member(out.select("ticker", pl.col(date_col).alias("date"))))

    def universe_sig(self) -> str:
        """Content hash of the PIT membership table ("" in static mode)."""
        if self.cfg.universe.mode != "sp500_pit":
            return ""
        import hashlib

        p = self.cfg.universe.pit_dir / "sp500_members.parquet"
        h = hashlib.sha256(p.read_bytes())
        uh = self.cfg.universe.hashable()
        if self.cfg.universe.extra_members:  # absent -> same signature as before extras existed
            h.update(json.dumps(uh["extra_members"]).encode())
        if "extra_members_file_sha256" in uh:
            h.update(uh["extra_members_file_sha256"].encode())
        return h.hexdigest()[:16]

    def cik_windows(
        self, start: datetime, end: datetime, before_days: int | None = None
    ) -> dict[int, tuple[datetime, datetime]] | None:
        """sp500_pit: per-CIK ingestion windows (membership +- margins) inside [start, end);
        ``before_days`` overrides ``universe.ingest_margin_before_days``."""
        if self.cfg.universe.mode != "sp500_pit":
            return None
        uc = self.cfg.universe
        u = self.universe()
        before = uc.ingest_margin_before_days if before_days is None else before_days
        w = self.pit.ingest_windows(start.date(), end.date(), before, uc.ingest_margin_after_days)
        return {
            c: (_parse_day(s.isoformat()), _parse_day(e.isoformat()))
            for c, (s, e) in w.items()
            if c in u.cik_to_tickers
        }

    def adapter(self, source: str) -> SourceAdapter:
        if source == "fed":
            from newsfeed.ingest.fed import FedAdapter

            fc = self.cfg.sources.fed
            http = self.stack.enter_context(
                HttpClient(
                    user_agent=fc.user_agent,
                    rate_per_sec=fc.max_requests_per_second,
                    timeout=fc.timeout_seconds,
                    max_retries=fc.max_retries,
                )
            )
            return FedAdapter(fc, http, self.store.raw)
        if source == "edgar":
            return EdgarAdapter(
                self.cfg.sources.edgar, self.http(), self.universe(), self.store.raw
            )
        if source == "edgar_periodic":
            from newsfeed.ingest.edgar_periodic import EdgarPeriodicAdapter
            from newsfeed.storage import RawStore

            pcfg = self.cfg.sources.edgar_periodic
            raw = RawStore(pcfg.raw_dir) if pcfg.raw_dir is not None else self.store.raw
            return EdgarPeriodicAdapter(pcfg, self.http(), self.universe(), raw)
        if source == "news_api":
            ncfg = self.cfg.sources.news_api

            def watermark() -> datetime | None:
                df = self.store.query(
                    "SELECT max(published_at) AS m FROM items WHERE source = 'news_api'"
                )
                return df["m"][0]

            return NewsApiAdapter(ncfg, make_vendor(ncfg), self.universe(), watermark=watermark)
        if source in ("transcripts", "transcripts_qa", "transcripts_prep"):
            from newsfeed.ingest.transcripts import TranscriptsAdapter

            spells = None
            if self.cfg.universe.mode == "sp500_pit":
                spells = {}
                for r in self.pit.members.select("ticker", "start", "end", "cik").iter_rows():
                    spells.setdefault(r[0], []).append((r[1], r[2], int(r[3])))
            return TranscriptsAdapter(
                getattr(self.cfg.sources, source),
                self.universe(),
                self.tmap,
                spells=spells,
                section={"transcripts": "full", "transcripts_qa": "qa"}.get(source, "prep"),
            )
        raise typer.BadParameter(f"unknown source {source!r}; choose from {SOURCES}")

    def derive_fn(
        self,
        run_id: str,
        full: bool = False,
        relink_sources: list[str] | None = None,
        *,
        if_changed: bool = False,
    ) -> Callable[[], dict[str, Any]]:
        linker = self.linker()
        retry = self.store.state_dir / "live-scoring-retry"

        def fn() -> dict[str, Any]:
            out = run_derive(
                self.store,
                self.cfg,
                linker,
                run_id,
                full_stories=full,
                relink_sources=relink_sources,
                if_changed=if_changed,
            )
            # nothing new -> nothing to score, unless the last attempt hit API errors
            new_or_retry = not out.get("unchanged") or retry.exists()
            if self.cfg.scoring.live_in_collect and not out.get("skipped") and new_or_retry:
                out["scoring"] = self.score_recent(run_id)
                if out["scoring"].get("by_status", {}).get("error"):
                    retry.touch()
                else:
                    retry.unlink(missing_ok=True)
            # Fed documents are not derived (market source), so the derive fingerprint never
            # sees them: plan them every cycle (a cheap read of the small fed partition)
            if self.cfg.scoring.live_in_collect and "fed" in self.cfg.collect.sources:
                out["fed_scoring"] = self.score_fed_recent(run_id)
            return out

        return fn

    # ------------------------------------------------------------ scoring
    def llm_client(self, required: bool = True) -> LLMClient | None:
        key = load_secrets().anthropic_api_key
        if not key:
            if required:
                raise typer.BadParameter("ANTHROPIC_API_KEY is not set (add it to .env)")
            return None
        return AnthropicClient(api_key=key)

    def scoring_job(self, run_id: str, model: str | None, client: LLMClient | None) -> ScoringJob:
        u = self.universe(full=True)
        titles = self.titles(u)
        return ScoringJob(
            self.cfg,
            self.store,
            client,
            run_id=run_id,
            model=model or self.cfg.scoring.model,
            company_names=titles,
            lexicon=MaskLexicon.build(self.cfg.universe, u, titles),
        )

    def score_recent(self, run_id: str) -> dict[str, Any]:
        """Collect-loop hook: score new story founders live (scoring.live_in_collect)."""
        job = self.scoring_job(run_id, None, self.llm_client())
        since = utcnow() - timedelta(hours=self.cfg.scoring.live_lookback_hours)
        # only what the loop collects: a backfill running alongside (e.g. 10-K full text, ~100k
        # tokens each) must never be scored by the live hook
        job.item_sources = list(self.cfg.collect.sources)
        targets = plan_targets(
            job,
            self.universe(full=True).cik_to_tickers,
            since_tradable=since,
            sources=job.item_sources,
        )
        return job.run_live(targets).as_dict() if targets else {"calls": 0}

    def score_fed_recent(self, run_id: str) -> dict[str, Any]:
        """Collect-loop hook: score new Fed documents (same look-back as company scoring)."""
        from newsfeed.scoring.fed import FedScorer

        since = utcnow() - timedelta(hours=self.cfg.scoring.live_lookback_hours)
        scorer = FedScorer(self.cfg, self.store, None, run_id=run_id)
        targets = scorer.plan(start=since)
        if not targets:
            return {"calls": 0}
        scorer.client = self.llm_client()
        return scorer.run_live(targets).as_dict()


def _run(
    command: str,
    cfg: AppConfig,
    params: dict[str, Any],
    body: Callable[[Context, str, dict[str, Any]], None],
    tickers: str | None = None,
) -> None:
    with ExitStack() as stack:
        store = stack.enter_context(Store(cfg.paths))
        ctx = Context(cfg, store, stack, tickers)
        run_id = store.start_run(command, cfg, params)
        stats = new_stats()
        status = "error"
        try:
            body(ctx, run_id, stats)
            status = "ok" if stats["failed"] == 0 else "partial"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            store.finish_run(run_id, status, stats)
            typer.echo(f"run {run_id}: {status} {stats}")


@app.command()
def backfill(
    start: Annotated[str, typer.Option(help="Inclusive UTC date, YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="Exclusive UTC date, YYYY-MM-DD")],
    source: Annotated[str, typer.Option(help="edgar | news_api")] = "edgar",
    tickers: TickersOpt = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Historical ingestion, then derived tables. Idempotent and resumable."""
    _setup_logging(verbose)
    cfg = load_config(config)
    t0, t1 = _parse_day(start), _parse_day(end)
    if t1 <= t0:
        raise typer.BadParameter("--end must be after --start")
    if source not in SOURCES:
        raise typer.BadParameter(f"unknown source {source!r}; choose from {SOURCES}")

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        adapter = ctx.adapter(source)
        before = getattr(getattr(cfg.sources, source), "ingest_margin_before_days", None)
        company = source not in ("news_api", *MARKET_SOURCES)
        windows = ctx.cik_windows(t0, t1, before) if company else None
        if windows is not None:
            adapter.cik_windows = windows
            stats["cik_windows"] = len(windows)
        on_error = make_error_recorder(ctx.store, run_id, source, stats)
        items = adapter.backfill(t0, t1, skip=ctx.store.has_item, on_error=on_error)
        ingest(
            items,
            store=ctx.store,
            cfg=cfg,
            calendar=ctx.calendar,
            run_id=run_id,
            flush_every=getattr(cfg.sources, source).flush_every,
            stats=stats,
        )
        stats["adapter"] = dict(getattr(adapter, "stats", {}))
        lag = timedelta(seconds=cfg.timing.lag_for(source))
        u = adapter.universe
        spans = windows or {cik: (t0, t1) for cik in u.ciks}
        covered = set()
        for cik, (ws, we) in spans.items():
            tks = u.tickers_for(cik)
            covered.update(tks)
            coverage.record_backfill(
                ctx.store, source, tks, ws, we, lag=lag, calendar=ctx.calendar, run_id=run_id
            )
        covered = sorted(covered)
        stats["covered_tickers"] = covered
        if ctx._http is not None:
            stats["http_requests"] = ctx._http.request_count
        stats["derive"] = ctx.derive_fn(run_id)()

    params = {"source": source, "start": start, "end": end, "tickers": tickers}
    _run("backfill", cfg, params, body, tickers)


@app.command()
def collect(
    once: Annotated[bool, typer.Option("--once", help="One cycle (cron-friendly)")] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Evaluate health from saved state; exit 1 if bad")
    ] = False,
    sources: Annotated[
        str | None, typer.Option(help="Comma-separated sources; default from config")
    ] = None,
    max_cycles: Annotated[int | None, typer.Option(help="Stop after N cycles")] = None,
    tickers: TickersOpt = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Live collection: poll -> ingest -> derive -> heartbeat/health, every interval."""
    _setup_logging(verbose)
    cfg = load_config(config)
    srcs = _split(sources) or cfg.collect.sources
    if bad := set(srcs) - set(SOURCES):
        raise typer.BadParameter(f"unknown sources {sorted(bad)}")

    if check:
        with Store(cfg.paths) as store:
            cal = get_calendar(cfg.market.calendar, cfg.market.calendar_start)
            problems = check_states(cfg, store, cal, srcs, utcnow())
        for p in problems:
            typer.echo(f"UNHEALTHY [{p.source}] {p.kind}: {p.message}")
        if problems:
            raise typer.Exit(1)
        typer.echo("healthy")
        return

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        adapters = {s: ctx.adapter(s) for s in srcs}
        collector = Collector(
            cfg,
            ctx.store,
            adapters,
            ctx.calendar,
            run_id,
            derive_fn=ctx.derive_fn(run_id, if_changed=True),
        )
        try:
            collector.run(max_cycles=1 if once else max_cycles)
        finally:
            for k in ("seen", "written", "failed", "truncated", "cycles"):
                stats[k] = collector.stats[k]

    _run("collect", cfg, {"sources": srcs, "once": once, "tickers": tickers}, body, tickers)


@app.command()
def derive(
    full: Annotated[bool, typer.Option("--full", help="Rebuild stories from scratch")] = False,
    relink_source: Annotated[
        str | None,
        typer.Option(help="Force a fresh link batch for these sources (comma-separated)"),
    ] = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Rebuild derived tables (ticker links, stories). Normally runs automatically."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        stats["derive"] = ctx.derive_fn(
            run_id, full=full, relink_sources=_split(relink_source) or None
        )()

    _run("derive", cfg, {"full": full}, body)


@app.command()
def inspect(config: ConfigOpt = None, limit: int = 10) -> None:
    """Summarise what is stored: items, links, stories, runs, failures, alerts."""
    cfg = load_config(config)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")  # Windows consoles choke on box drawing
    pl.Config.set_tbl_rows(50)
    pl.Config.set_fmt_str_lengths(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    with Store(cfg.paths) as store:
        sections = {
            "items": """SELECT source, strftime(published_at, '%Y-%m') AS month, count(*) AS n,
                               count(*) FILTER (WHERE ingest_mode = 'live') AS live,
                               count(*) FILTER (WHERE body_truncated) AS truncated
                        FROM items GROUP BY ALL ORDER BY ALL""",
            "ticker links": """SELECT ticker, method, count(*) AS n,
                                      round(avg(confidence), 2) AS conf
                               FROM item_tickers GROUP BY ALL ORDER BY ticker, n DESC""",
            "stories": """SELECT reason, count(*) AS n, round(avg(similarity), 3) AS avg_sim
                          FROM item_story GROUP BY ALL ORDER BY n DESC""",
        }
        for name, sql in sections.items():
            typer.echo(f"\n== {name}")
            typer.echo(store.query(sql))
        ops = {
            "runs": "SELECT run_id, command, status, stats FROM runs "
            "ORDER BY started_at DESC LIMIT ?",
            "failures": "SELECT run_id, native_id, error FROM failures "
            "ORDER BY failed_at DESC LIMIT ?",
            "alerts": "SELECT raised_at, source, kind, message FROM alerts "
            "ORDER BY raised_at DESC LIMIT ?",
        }
        for name, sql in ops.items():
            typer.echo(f"\n== {name}")
            typer.echo(store.ops_query(sql, [limit]))


@app.command()
def reparse(
    source: Annotated[str, typer.Option(help="edgar | fed")] = "edgar",
    everything: Annotated[bool, typer.Option("--all", help="Also items already current")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Re-parse stored items with the current parser (from the raw cache; timing preserved),
    then rebuild links and stories (edgar; fed items have neither). Scores of changed items
    are re-planned automatically (their text hash changes)."""
    from newsfeed.ingest.fed import PARSER_VERSION as FED_PARSER_VERSION

    _setup_logging(verbose)
    cfg = load_config(config)
    versions = {"edgar": EDGAR_PARSER_VERSION, "fed": FED_PARSER_VERSION}
    if source not in versions:
        raise typer.BadParameter("only --source edgar or fed supports re-parsing")

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        adapter = ctx.adapter(source)
        stats["reparse"] = reparse_source(
            ctx.store,
            source,
            adapter,
            versions[source],
            calendar=ctx.calendar,
            run_id=run_id,
            everything=everything,
        )
        stats["failed"] = len(stats["reparse"]["failed"])
        if source not in MARKET_SOURCES:
            stats["derive"] = ctx.derive_fn(run_id, full=True)()

    _run("reparse", cfg, {"source": source, "all": everything}, body)


@app.command()
def score(
    mode: Annotated[
        str, typer.Option(help="live (normal API) | batch (Batches API, -50%)")
    ] = "live",
    model: Annotated[str | None, typer.Option(help="Default: scoring.model")] = None,
    masked: Annotated[bool, typer.Option(help="Score masked text (leakage check)")] = False,
    start: Annotated[str | None, typer.Option(help="tradable_at >= this UTC date")] = None,
    end: Annotated[str | None, typer.Option(help="tradable_at < this UTC date")] = None,
    sources: Annotated[str | None, typer.Option(help="Comma-separated sources")] = None,
    sample: Annotated[int | None, typer.Option(help="Deterministic sample of N pairs")] = None,
    event_types: Annotated[
        str | None,
        typer.Option(help="Only pairs whose primary score has these event types (comma-sep.)"),
    ] = None,
    seed: int = 1,
    unpaired: Annotated[
        bool,
        typer.Option("--unpaired", help="Masked/model runs without requiring a primary score"),
    ] = False,
    replicates: Annotated[int, typer.Option(help="Noise study: re-draw K times")] = 0,
    limit: Annotated[int | None, typer.Option(help="Cap the number of requests")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Estimate cost, call nothing")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm jobs above the cost limit")] = False,
    wait: Annotated[bool, typer.Option(help="batch: wait for and collect results")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Score (story founder, company) pairs with the LLM. Cached: re-runs never re-pay.

    Studies: --masked (vs unmasked), --model <validation model>, --replicates K. Studies only
    cover documents that already have an ok primary score; combine with --sample N.
    """
    _setup_logging(verbose)
    cfg = load_config(config)
    if mode not in ("live", "batch"):
        raise typer.BadParameter("--mode must be live or batch")

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        client = ctx.llm_client(required=not dry_run)
        job = ctx.scoring_job(run_id, model, client)
        targets = plan_targets(
            job,
            ctx.universe(full=True).cik_to_tickers,
            start=_parse_day(start) if start else None,
            end=_parse_day(end) if end else None,
            sources=_split(sources) or None,
            masked=masked,
            replicates=replicates,
            sample=sample,
            seed=seed,
            limit=limit,
            event_types=_split(event_types) or None,
            paired=not unpaired,
        )
        typer.echo(
            f"model={job.model} prompt={job.prompt.version}/{job.prompt_hash} "
            f"masked={masked} replicates={replicates} -> {len(targets)} requests to make"
        )
        if not targets:
            return
        est = estimate_cost(job, targets)
        typer.echo(est.summary())
        stats["estimate"] = est.__dict__
        if dry_run:
            return
        upper = est.cost_batch_no_cache if mode == "batch" else est.cost_live_no_cache
        if upper > cfg.scoring.confirm_above_usd and not yes:
            typer.echo(
                f"Estimated up to ${upper:.2f} > scoring.confirm_above_usd "
                f"(${cfg.scoring.confirm_above_usd:.2f}). Re-run with --yes to proceed."
            )
            raise typer.Exit(3)
        if mode == "live":
            job.run_live(targets)
        else:
            stats["batches"] = batch_mod.submit(job, targets)
            if wait:
                stats["collect"] = batch_mod.collect(job, wait=True)
        stats["scoring"] = job.stats.as_dict()
        stats["failed"] = job.stats.by_status.get("error", 0)

    params = {
        "mode": mode,
        "model": model,
        "masked": masked,
        "start": start,
        "end": end,
        "sample": sample,
        "seed": seed,
        "replicates": replicates,
        "limit": limit,
        "dry_run": dry_run,
    }
    _run("score", cfg, params, body)


@app.command("score-batches")
def score_batches(
    wait: Annotated[bool, typer.Option(help="Poll until every batch has ended")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Collect results of submitted batches (safe to run any time; idempotent)."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        job = ctx.scoring_job(run_id, None, ctx.llm_client())
        stats["collect"] = batch_mod.collect(job, wait=wait)
        stats["scoring"] = job.stats.as_dict()
        typer.echo(stats["collect"])

    _run("score-batches", cfg, {"wait": wait}, body)


@app.command("score-fed")
def score_fed(
    model: Annotated[str | None, typer.Option(help="Default: scoring.model")] = None,
    start: Annotated[str | None, typer.Option(help="tradable_at >= this UTC date")] = None,
    end: Annotated[str | None, typer.Option(help="tradable_at < this UTC date")] = None,
    limit: Annotated[int | None, typer.Option(help="Cap the number of requests")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Estimate cost, call nothing")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm jobs above the cost limit")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Score Federal Reserve documents (source fed) with the Fed prompt (live API, cached:
    re-runs never re-pay). One score per document: policy relevance, hawkish/dovish stance,
    signalled rate path, inflation / growth / labor assessments."""
    from newsfeed.scoring.fed import FedScorer

    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        scorer = FedScorer(
            cfg, ctx.store, ctx.llm_client(required=not dry_run), run_id=run_id, model=model
        )
        targets = scorer.plan(
            start=_parse_day(start) if start else None,
            end=_parse_day(end) if end else None,
            limit=limit,
        )
        typer.echo(
            f"model={scorer.model} prompt={scorer.prompt.version}/{scorer.prompt_hash} -> "
            f"{len(targets)} Fed documents to score"
        )
        if not targets:
            return
        est = scorer.estimate(targets)
        typer.echo(est.summary())
        stats["estimate"] = est.__dict__
        if dry_run:
            return
        if est.cost_live_no_cache > cfg.scoring.confirm_above_usd and not yes:
            typer.echo(
                f"Estimated up to ${est.cost_live_no_cache:.2f} > scoring.confirm_above_usd "
                f"(${cfg.scoring.confirm_above_usd:.2f}). Re-run with --yes to proceed."
            )
            raise typer.Exit(3)
        scorer.run_live(targets)
        stats["scoring"] = scorer.stats.as_dict()
        stats["failed"] = scorer.stats.by_status.get("error", 0)

    params = {"model": model, "start": start, "end": end, "limit": limit, "dry_run": dry_run}
    _run("score-fed", cfg, params, body)


@app.command()
def agreement(
    compare: Annotated[str, typer.Option(help="masked | model | replicates")],
    model_b: Annotated[
        str | None, typer.Option(help="model: default scoring.validation_model")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Agreement between the primary scores and a study (same documents, same prompt)."""
    cfg = load_config(config)
    with ExitStack() as stack:
        store = stack.enter_context(Store(cfg.paths))
        ctx = Context(cfg, store, stack, None)
        job = ctx.scoring_job("agreement", None, None)
        primary = agr.Side(cfg.scoring.model, job.prompt_hash)
        sides = {
            "masked": agr.Side(cfg.scoring.model, job.prompt_hash, masked=True),
            "model": agr.Side(model_b or cfg.scoring.validation_model, job.prompt_hash),
            "replicates": agr.Side(cfg.scoring.model, job.prompt_hash, replicate=None),
        }
        if compare not in sides:
            raise typer.BadParameter("--compare must be masked, model or replicates")
        title = f"primary ({primary.model}, prompt {job.prompt_hash}) vs {compare}"
        typer.echo(
            agr.format_metrics(title, agr.metrics(agr.paired(store, primary, sides[compare])))
        )
        if compare != "replicates":  # the noise floor any study should be judged against
            floor = agr.metrics(agr.paired(store, primary, sides["replicates"]))
            if floor["n_pairs"]:
                typer.echo(agr.format_metrics("noise floor: primary vs its own re-draws", floor))
            else:
                typer.echo(
                    "(no replicates yet: run `score --replicates 2 --sample N` for a noise floor)"
                )


@app.command()
def cost(
    run: Annotated[str | None, typer.Option(help="Only this run_id")] = None,
    limit: int = 20,
    config: ConfigOpt = None,
) -> None:
    """LLM spend per run from the usage ledger (every call, including failures)."""
    cfg = load_config(config)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_rows(100)
    where, params = ("WHERE u.run_id = ?", [run]) if run else ("", [])
    with Store(cfg.paths) as store:
        df = store.ops_query(
            f"""
            SELECT u.run_id, any_value(r.command) AS command, u.mode, u.model,
                   count(*) AS calls,
                   count(*) FILTER (WHERE u.status = 'ok') AS ok,
                   count(*) FILTER (WHERE u.status NOT IN ('ok')) AS not_ok,
                   sum(u.input_tokens) AS input, sum(u.cache_read_input_tokens) AS cache_read,
                   sum(u.cache_creation_input_tokens) AS cache_write,
                   sum(u.output_tokens) AS output, round(sum(u.cost_usd), 4) AS usd,
                   min(u.called_at) AS first_call
            FROM llm_usage u LEFT JOIN runs r USING (run_id) {where}
            GROUP BY u.run_id, u.mode, u.model ORDER BY first_call DESC LIMIT {int(limit)}
            """,
            params,
        )
        total = store.ops_query(
            f"SELECT round(sum(cost_usd), 4) AS usd FROM llm_usage u {where}", params
        )
    typer.echo(df.drop("first_call") if df.height else "no LLM calls recorded")
    typer.echo(f"total: ${total['usd'][0] or 0:.4f} (pricing as of {cfg.pricing.as_of})")


def feature_sources(ctx: Context) -> FeatureSources:
    cfg = ctx.cfg
    job = ctx.scoring_job("features", None, None)
    try:
        lm_version: str | None = load_lm_dictionary(cfg.features.lm_dictionary).version
    except FileNotFoundError:
        lm_version = None
    return FeatureSources(
        llm_model=cfg.scoring.model,
        llm_prompt_hash=job.prompt_hash,
        lm_version=lm_version,
        finbert_version=cfg.features.finbert_model if cfg.features.finbert else None,
        embedding_model=cfg.features.embedding_model if cfg.features.embeddings else None,
    )


@app.command()
def features(
    start: Annotated[str, typer.Option(help="First session date (UTC), YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="Exclusive end date")],
    tickers: TickersOpt = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Build point-in-time features on the session grid (features.grid) and store them."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        fc = cfg.features
        grid = as_of_grid(
            ctx.calendar,
            date.fromisoformat(start),
            date.fromisoformat(end),
            fc.grid,
            fc.decision_lag_minutes,
        )
        src = feature_sources(ctx)
        set_id = feature_set_id(cfg, src, ctx.universe_sig())
        c2t = ctx.universe(full=True).cik_to_tickers
        spans = None
        if cfg.universe.mode == "sp500_pit":  # rows only while the ticker is an index member
            spans = {}
            for r in ctx.pit.member_frame().to_dicts():
                spans.setdefault(r["ticker"], []).append((r["start"], r["end"]))
        df = build_features(
            ctx.store,
            cfg,
            grid,
            c2t,
            src,
            tickers=[t.upper() for t in _split(tickers)] or None,
            member_spans=spans,
            holders=ctx.holders(),
        )
        with ctx.store.ops() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiments VALUES ('feature_set', ?, ?, ?, ?)",
                [
                    set_id,
                    utcnow(),
                    run_id,
                    json.dumps({**src.__dict__, "features": fc.model_dump(mode="json")}),
                ],
            )
        stats["feature_set_id"] = set_id
        stats["rows"] = ctx.store.write_features(df, set_id, run_id)
        stats["as_of_times"] = len(grid)
        if df.height:
            nonzero = {
                c: round(float((df[c].fill_null(0) != 0).mean()), 3)
                for c in ("1d_n_stories", "1d_llm_n", "1d_lm_n", "5d_n_stories")
                if c in df.columns
            }
            stats["share_nonzero"] = nonzero
        typer.echo(f"feature set {set_id}: {stats['rows']} rows ({len(grid)} as-of times)")

    _run("features", cfg, {"start": start, "end": end, "tickers": tickers}, body, tickers)


@app.command("universe")
def universe_cmd(
    build: Annotated[
        bool, typer.Option("--build", help="Fetch sources and (re)build tables")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Point-in-time S&P 500 universe: build from fja05680/sp500 + Wikipedia, or summarise."""
    cfg = load_config(config)
    uc = cfg.universe
    d = uc.pit_dir
    if build:
        from newsfeed.universe_pit import build_tables, fetch_fja, fetch_wiki_monthly

        today = utcnow().date()
        months = [
            date(y, m, 1)
            for y in range(uc.pit_start.year, today.year + 1)
            for m in range(1, 13)
            if uc.pit_start <= date(y, m, 1) <= today
        ]
        fja = fetch_fja(uc.fja_sha, d)
        typer.echo(
            f"fja05680/sp500 @ {uc.fja_sha[:12]} -> {fja.name}; fetching "
            f"{len(months)} monthly Wikipedia snapshots (incremental)..."
        )
        wiki = fetch_wiki_monthly(months, d / "wiki_monthly.parquet")
        members, sectors, stats = build_tables(fja.read_bytes(), wiki, uc.pit_start)
        PitUniverse(members, sectors).save(d)
        typer.echo(f"built: {stats}")
    u = PitUniverse.load(d)
    m = u.members
    typer.echo(
        f"{m['ticker'].n_unique()} tickers, {m['cik'].n_unique()} CIKs, "
        f"{m.filter(pl.col('end').is_null()).height} current members"
    )
    for e in uc.extra_members:
        typer.echo(f"  + extra ({e.index}, not PIT): {e.ticker} {e.name} from {e.start}")
    if uc.extra_members_file is not None:
        by_index = Counter(e.index for e in uc.file_extra_members())
        typer.echo(
            f"  + {uc.extra_members_file.name}: "
            + ", ".join(f"{k} {v}" for k, v in sorted(by_index.items()))
        )


@app.command("universe-freeze")
def universe_freeze_cmd(
    as_of: Annotated[str, typer.Option(help="Freeze date (members of the last revision before)")],
    out: Annotated[str, typer.Option(help="CSV to write (versioned, e.g. config/universe/...)")],
    indices: Annotated[str, typer.Option(help="Comma-separated: sp400, sp600")] = "sp400,sp600",
    cik_overrides: Annotated[
        str | None,
        typer.Option(
            help="TICKER=CIK,... for members whose ticker is gone from SEC's current map "
            "(acquired / delisted after the freeze date; verify each against its filings)"
        ),
    ] = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Freeze S&P 400 / 600 membership as of a date into an extra-members CSV (H3, 2026-10-29):
    Wikipedia revisions before ``as_of``, CIKs from the page or SEC's ticker map, companies in
    the S&P 500 on that date or in ``universe.extra_members`` excluded."""
    from newsfeed.universe_pit import (
        INDEX_PAGES,
        freeze_index_members,
        parse_index_table,
        wiki_revision_html,
    )
    from newsfeed.universe_pit import UA as WIKI_UA

    _setup_logging(verbose)
    cfg = load_config(config)
    day = date.fromisoformat(as_of)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        sp500 = PitUniverse.load(cfg.universe.pit_dir).members
        nasdaq = {e.cik for e in cfg.universe.extra_members}
        # manual CIKs only fill tickers the current SEC map lacks (never override a mapping)
        manual = {
            t.strip().upper(): int(c) for t, c in (p.split("=") for p in _split(cik_overrides))
        }
        t2c = {**manual, **ctx.tmap.ticker_to_cik}
        client = httpx.Client(headers={"User-Agent": WIKI_UA}, timeout=120, follow_redirects=True)
        frames = []
        for idx in _split(indices):
            revid, ts, html = wiki_revision_html(INDEX_PAGES[idx], day, client)
            rows = parse_index_table(html)
            df, unmapped, excluded = freeze_index_members(
                rows, idx, day, t2c, sp500, nasdaq, note=f"wiki rev {revid} {ts}"
            )
            typer.echo(
                f"{idx}: revision {revid} ({ts}): {len(rows)} rows -> {df.height} kept, "
                f"{len(excluded)} excluded (S&P 500 / extras) {excluded}, "
                f"{len(unmapped)} without CIK {unmapped}; "
                f"{df.filter(pl.col('end').is_not_null()).height} later joined the S&P 500"
            )
            frames.append(df)
        frozen = pl.concat(frames)
        dup = frozen.group_by("ticker").len().filter(pl.col("len") > 1)
        if dup.height:
            typer.echo(
                f"WARNING: tickers in both indices, keeping the first: {dup['ticker'].to_list()}"
            )
            frozen = frozen.unique("ticker", keep="first", maintain_order=True)
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        frozen.sort("index", "ticker").write_csv(path)
        stats["written"] = frozen.height
        typer.echo(f"{frozen.height} companies -> {path}")

    params = {"as_of": as_of, "out": out, "indices": indices, "cik_overrides": cik_overrides}
    _run("universe-freeze", cfg, params, body)


@app.command("coverage")
def coverage_cmd(
    rebuild_from_runs: Annotated[
        bool, typer.Option("--rebuild-from-runs", help="Recreate backfill coverage from runs")
    ] = False,
    bridge_live_since: Annotated[
        str | None,
        typer.Option(
            help="Add live coverage for poll gaps <= collect.coverage_max_gap_minutes after "
            "this date (YYYY-MM-DD); rows are added, never edited"
        ),
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Show collected coverage (tradable-time ranges) per source and ticker."""
    cfg = load_config(config)
    with Store(cfg.paths) as store:
        if bridge_live_since:
            tag = f"bridge-{utcnow():%Y%m%dT%H%M%SZ}"
            gap = timedelta(minutes=cfg.collect.coverage_max_gap_minutes)
            for src in cfg.collect.sources:
                n = coverage.bridge_live_gaps(store, src, gap, _parse_day(bridge_live_since), tag)
                typer.echo(f"{src}: {n} bridge rows ({tag})")
        if rebuild_from_runs:
            cal = get_calendar(cfg.market.calendar, cfg.market.calendar_start)
            n = coverage.rebuild_from_runs(
                store, cal, cfg.timing.backfill_lag_seconds, cfg.universe.tickers
            )
            typer.echo(f"rebuilt backfill coverage from {n} runs")
        sources = store.ops_query("SELECT DISTINCT source FROM coverage")["source"].to_list()
        for src in sorted(sources):
            typer.echo(f"== {src}")
            for tk, ivs in sorted(coverage.merged(store, src).items()):
                spans = ", ".join(f"{s:%Y-%m-%d %H:%M} .. {e:%Y-%m-%d %H:%M}" for s, e in ivs)
                typer.echo(f"  {tk:6s} {spans}")


def _sample_companies(ctx: Context, cfg: AppConfig, feats: pl.DataFrame) -> pl.DataFrame:
    """``eval.sample_companies_file``: keep ticker-days whose ticker belongs to a listed CIK on
    that date (New York date of ``as_of``), whatever index the company is in by then."""
    ciks = cfg.eval.sample_ciks()
    if ciks is None:
        return feats
    if cfg.universe.mode != "sp500_pit":
        raise typer.BadParameter("eval.sample_companies_file needs universe.mode sp500_pit")
    spans = ctx.pit.members.filter(pl.col("cik").is_in(sorted(ciks))).select(
        "ticker", "start", "end"
    )
    keyed = feats.select(
        "ticker", pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("_d")
    ).with_row_index("_r")
    hit = keyed.join(spans, on="ticker").filter(
        (pl.col("start") <= pl.col("_d"))
        & (pl.col("end").is_null() | (pl.col("_d") < pl.col("end")))
    )["_r"]
    return feats.with_row_index("_r").filter(pl.col("_r").is_in(hit.implode())).drop("_r")


def _price_caches(cfg: AppConfig) -> dict[str, CachedPrices]:
    ec, secrets = cfg.eval, load_secrets()
    return {
        "tradier": CachedPrices(
            TradierPrices(
                secrets.tradier_token, ec.tradier_base_url, ec.tradier_requests_per_second
            )
            if secrets.tradier_token
            else None,
            cfg.paths.ref_dir / "prices",
        ),
        "tiingo": CachedPrices(
            TiingoPrices(secrets.tiingo_token, ec.tiingo_requests_per_hour / 3600)
            if secrets.tiingo_token
            else None,
            cfg.paths.ref_dir / "prices_tiingo",
        ),
    }


def _eval_inputs(
    ctx: Context, cfg: AppConfig, feats: pl.DataFrame, *, refresh_prices: bool = False
) -> tuple[EvalInputs, list[dict]]:
    """Prices (routed, cached), targets, controls and PIT size for a feature table (restricted
    to ``eval.sample_companies_file`` first, so sizes are ranked within that sample)."""
    ec = cfg.eval
    feats = _sample_companies(ctx, cfg, feats)
    if feats.height == 0:
        raise typer.BadParameter("no feature rows left in the sample")
    dates = decision_dates(feats)
    caches = _price_caches(cfg)
    after_days, max_pad = horizon_days(max(ec.horizons))
    lo, end = dates[0], min(dates[-1] + timedelta(days=after_days), utcnow().date())
    tickers = sorted(feats["ticker"].unique().to_list())
    bench = [ec.market_etf, *sorted(set(ec.sector_etfs.values()))]
    pit = cfg.universe.mode == "sp500_pit"
    if pit:
        bench += sorted(set(ec.gics_sector_etfs.values()) - set(bench))
        members = ctx.pit.members.filter(pl.col("ticker").is_in(tickers))
        routes = route_prices(
            members,
            ctx.tmap.ticker_to_cik,
            lo,
            end,
            after_days=after_days,
            overrides=ec.price_overrides,
        )
    else:
        routes = [PriceRoute(t, t, "tradier", lo, end, "", "current") for t in tickers]
    routes += [PriceRoute(b, b, "tradier", lo, end, b, "benchmark") for b in bench]
    # sessions = the market ETF's trading days (fetched first)
    spy, _, _ = build_panel(routes[-len(bench) :][:1], caches, [], refresh=refresh_prices)
    sessions = sorted(spy["date"].to_list())
    panel, price_report, splits = build_panel(
        routes, caches, sessions, refresh=refresh_prices, max_pad=max_pad
    )
    panel = panel.drop("synthetic")
    fwd = forward_returns(panel, ec.horizons, ec.entries)
    sector_of = ctx.pit.sector_etf_frame(ec.gics_sector_etfs) if pit else ec.sector_etfs
    targets = adjusted_targets(fwd, tickers, ec.market_etf, sector_of)
    ctl = controls(panel, ec.market_etf)
    shares: dict[str, pl.DataFrame] = {}
    t2c = (
        {
            t: sorted({int(c) for c in g["cik"]})
            for (t,), g in ctx.pit.members.filter(pl.col("ticker").is_in(tickers)).group_by(
                "ticker"
            )
        }
        if pit
        else {t: [c] for c, tks in ctx.universe(full=True).cik_to_tickers.items() for t in tks}
    )
    for t, ciks in sorted(t2c.items()):
        hist = [
            load_shares(ctx.http(), c, cfg.paths.ref_dir / "sec_shares").with_columns(
                pl.lit(c, pl.Int64).alias("cik")
            )
            for c in ciks
        ]
        shares[t] = pl.concat(hist).sort("available_from") if hist else pl.DataFrame()
    decisions = feats.select(
        "ticker",
        pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date"),
    )
    caps = market_caps(decisions, panel, shares, splits, ctx.holders())
    vol = volatility_targets(panel, [5, 20]).filter(pl.col("ticker").is_in(tickers))
    return EvalInputs(feats, ctl, targets, caps, sessions, vol), price_report


@app.command("eval")
def eval_cmd(
    feature_set: Annotated[
        str | None, typer.Option(help="Feature set id (default: most recently built)")
    ] = None,
    use_holdout: Annotated[
        bool, typer.Option("--use-holdout", help="Evaluate on the frozen holdout (final test)")
    ] = False,
    refresh_prices: Annotated[bool, typer.Option(help="Re-download prices/dividends")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Rank IC and LLM incremental lift over the entry x horizon x adjustment grid."""
    _setup_logging(verbose)
    cfg = load_config(config)
    ec = cfg.eval

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        store = ctx.store
        set_id = (
            feature_set
            or store.ops_query(
                "SELECT key FROM experiments WHERE kind = 'feature_set' "
                "ORDER BY first_seen_at DESC LIMIT 1"
            )["key"][0]
        )
        feats = store.features(set_id)
        if feats.height == 0:
            raise typer.BadParameter(f"feature set {set_id} has no rows; run `features` first")
        feats = add_derived(feats.drop("feature_set_id", "build_run_id", "built_at", strict=False))
        prior_holdout = store.ops_query(
            "SELECT count(*) AS n FROM experiments WHERE kind = 'eval' "
            "AND json_extract(details, '$.holdout') = true"
        )["n"][0]
        if use_holdout:
            typer.echo(
                f"WARNING: evaluating on the frozen holdout (from {ec.holdout_start}); it has "
                f"been evaluated {prior_holdout} time(s) before. Each look spends its value."
            )
        inp, price_report = _eval_inputs(ctx, cfg, feats, refresh_prices=refresh_prices)
        univariate = [f for f in DEFAULT_UNIVARIATE if f in feats.columns]
        eval_id = eval_config_id(cfg, set_id, univariate)
        result = run_eval(inp, cfg, univariate=univariate, use_holdout=use_holdout)
        with store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('eval', ?, ?, ?, ?)",
                [
                    f"{eval_id}:{run_id}",
                    utcnow(),
                    run_id,
                    json.dumps(
                        {
                            "eval_id": eval_id,
                            "feature_set": set_id,
                            "holdout": use_holdout,
                            "n_tests": result["n_tests"],
                        }
                    ),
                ],
            )
        exp_counts = store.ops_query(
            "SELECT kind, count(*) AS n FROM experiments GROUP BY 1 ORDER BY 1"
        ).to_dicts()
        meta = {
            "run_id": run_id,
            "feature_set_id": set_id,
            "eval_id": eval_id,
            "llm_model": cfg.scoring.model,
            "prompt_hash": ctx.scoring_job(run_id, None, None).prompt_hash,
            "holdout_start": ec.holdout_start,
            "fdr_q": ec.fdr_q,
            "experiments": ", ".join(f"{r['kind']}={r['n']}" for r in exp_counts),
        }
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        md = render_markdown(result, meta)
        (out_dir / f"eval-{run_id}.md").write_text(md, encoding="utf-8")
        pl.DataFrame(price_report).write_csv(out_dir / f"eval-{run_id}-prices.csv")
        by_src = pl.DataFrame(price_report).group_by("source", "reason").len().sort("len")
        stats["prices"] = {f"{r['source']}/{r['reason']}": r["len"] for r in by_src.to_dicts()}
        for name in ("univariate", "models", "buckets"):
            if result[name].height:
                result[name].write_parquet(out_dir / f"eval-{run_id}-{name}.parquet")
        stats.update(
            {
                "report": str(out_dir / f"eval-{run_id}.md"),
                "n_tests": result["n_tests"],
                "feature_set": set_id,
                "eval_id": eval_id,
            }
        )
        typer.echo(md)

    _run("eval", cfg, {"feature_set": feature_set, "use_holdout": use_holdout}, body)


@app.command("migrate-text-hash")
def migrate_text_hash_cmd(config: ConfigOpt = None, verbose: bool = False) -> None:
    """One-time: store text_hash in item files written before the column existed (values are
    unchanged), then switch the items view to the stored column. Idempotent."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        def progress(k: int, n: int) -> None:
            if k % 500 == 0 or k == n:
                typer.echo(f"  {k}/{n} files", err=True)

        # the collector's derive (long scans holding files open) must not run meanwhile:
        # Windows can't replace a file another process is reading
        lock = FileLock(ctx.store.state_dir / "derive.lock")
        if not lock.acquire():
            raise typer.BadParameter("derive is running (lock held); try again when it's done")
        try:
            out = ctx.store.migrate_text_hash(progress)
        finally:
            lock.release()
        stats["migrate"] = out
        stats["written"] = out["migrated"]
        typer.echo(f"{out}; stored-hash view {'ON' if ctx.store.text_hash_migrated else 'OFF'}")

    _run("migrate-text-hash", cfg, {}, body)


@app.command("distill-data")
def distill_data_cmd(
    sources: Annotated[str, typer.Option(help="Comma-separated item sources")] = "edgar,news_api",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Build the distillation dataset: pre-holdout unmasked LLM labels + the exact text the
    LLM saw, split train/val/test per source by time; hand-labelled pairs held out as gold."""
    _setup_logging(verbose)
    cfg = load_config(config)
    src = _split(sources)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        pairs = distill_ds.load_pairs(ctx.store, src, cfg.scoring.model)
        dup = pairs.group_by("item_id", "ticker").len().filter(pl.col("len") > 1)
        if dup.height:
            raise typer.BadParameter(f"{dup.height} (item, ticker) pairs have several scores")
        pre = distill_ds.pre_holdout(pairs, cfg.eval.holdout_start)
        hashes = pre["prompt_hash"].unique().to_list()
        if len(hashes) != 1:
            raise typer.BadParameter(f"expected one prompt version, found {hashes}")
        prompt = load_prompt(cfg.scoring.prompt_version)
        names = ctx.titles(ctx.universe(full=True))
        df = distill_ds.render_texts(pre, prompt, names)
        gold = distill_ds.gold_pairs(cfg.paths.data_dir / "labeling")
        df = distill_ds.finalize(distill_ds.assign_splits(df, gold))
        out = cfg.paths.data_dir / "distill" / f"v{distill_ds.DISTILL_VERSION}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "dataset.parquet", compression="zstd")
        summary = (
            df.group_by("source", "split")
            .agg(
                pl.len().alias("n"),
                pl.col("text_ok").mean().round(4).alias("text_ok"),
                pl.col("tradable_at").min().dt.date().alias("from"),
                pl.col("tradable_at").max().dt.date().alias("to"),
            )
            .sort("source", "from")
        )
        meta = {
            "run_id": run_id,
            "version": distill_ds.DISTILL_VERSION,
            "sources": src,
            "model": cfg.scoring.model,
            "prompt_hash": hashes[0],
            "holdout_start": str(cfg.eval.holdout_start),
            "dropped_holdout_rows": pairs.height - pre.height,
            "gold_pairs_found": int((df["split"] == "gold").sum()),
            "splits": summary.with_columns(pl.col("from", "to").cast(pl.String)).to_dicts(),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        stats["written"] = df.height
        with pl.Config(
            tbl_rows=20,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
            tbl_formatting="ASCII_MARKDOWN",  # Windows consoles can't print box drawing
        ):
            typer.echo(summary)
        typer.echo(
            f"{df.height} rows -> {out}; dropped {pairs.height - pre.height} holdout rows; "
            f"gold {meta['gold_pairs_found']}; text matches stored hash: "
            f"{df['text_ok'].mean():.2%}"
        )

    _run("distill-data", cfg, {"sources": src}, body)


@app.command("leakage")
def leakage_cmd(
    feature_set: Annotated[str, typer.Option(help="Unmasked feature set (defines the sample)")],
    masked_set: Annotated[str, typer.Option(help="Masked feature set (same features config)")],
    column: Annotated[str, typer.Option(help="Feature compared")] = "1d_llm_sent_cwmean",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Leakage check: IC of masked vs unmasked LLM features on identical pre-holdout rows."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        drop = ("feature_set_id", "build_run_id", "built_at")
        a = ctx.store.features(feature_set).drop(*drop, strict=False)
        b = ctx.store.features(masked_set).select("ticker", "as_of", column)
        if a.height == 0 or b.height == 0:
            raise typer.BadParameter("both feature sets must exist")
        feats = a.join(b.rename({column: f"masked__{column}"}), on=["ticker", "as_of"], how="left")
        inp, _ = _eval_inputs(ctx, cfg, feats)
        res = paired_ic_comparison(inp, cfg, column, f"masked__{column}")
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(out_dir / f"leakage-{run_id}.parquet")
        lines = [
            f"# Leakage check {run_id}",
            "",
            f"- unmasked `{feature_set}` vs masked `{masked_set}`, feature `{column}`, "
            f"identical pre-holdout rows (sample: {cfg.eval.sample}, "
            f"events {cfg.eval.sample_event_types or 'any'})",
            "- diff = IC(masked) - IC(unmasked); paired circular block bootstrap",
            "",
            "| entry | h | adj | n | corr(a,b) | IC unmasked | IC masked | diff | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['entry']} | {r['horizon']} | {r['adjustment']} | {r['n']} | "
                f"{r['corr_ab']:.3f} | {r['ic_a']:+.3f} | {r['ic_b']:+.3f} | {r['diff']:+.3f} | "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.3f} |"
            )
        md = "\n".join(lines) + "\n"
        (out_dir / f"leakage-{run_id}.md").write_text(md, encoding="utf-8")
        with ctx.store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('leakage', ?, ?, ?, ?)",
                [
                    f"{feature_set}:{masked_set}:{run_id}",
                    utcnow(),
                    run_id,
                    json.dumps({"column": column, "eval": cfg.eval.model_dump(mode="json")}),
                ],
            )
        stats["report"] = str(out_dir / f"leakage-{run_id}.md")
        typer.echo(md)

    _run("leakage", cfg, {"feature_set": feature_set, "masked_set": masked_set}, body)


@app.command("explore-types")
def explore_types_cmd(
    feature_set: Annotated[
        list[str], typer.Option(help="label=feature_set_id (repeatable), e.g. news=57330b614405")
    ],
    types: Annotated[str, typer.Option(help="Comma-separated LLM event types")],
    feature: Annotated[str, typer.Option(help="Feature tested")] = "1d_llm_sent_cwmean",
    cells: Annotated[
        str, typer.Option(help="entry/horizon/adjustment cells, comma-separated")
    ] = "close/1/market,close/2/market",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """EXPLORATORY per-event-type IC on 'pure' samples (all LLM events in the window of one
    type), pre-holdout; one Benjamini-Hochberg correction across every set x type x cell."""
    _setup_logging(verbose)
    cfg = load_config(config)
    cell_list = [(e, int(h), a) for e, h, a in (c.split("/") for c in _split(cells))]
    type_list = _split(types)
    sets = dict(fs.split("=", 1) for fs in feature_set)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        rows: list[dict[str, Any]] = []
        for label, set_id in sets.items():
            feats = ctx.store.features(set_id).drop(
                "feature_set_id", "build_run_id", "built_at", strict=False
            )
            if feats.height == 0:
                raise typer.BadParameter(f"feature set {set_id} has no rows")
            inp, _ = _eval_inputs(ctx, cfg, feats)
            for t in type_list:
                c = cfg.model_copy(
                    update={"eval": cfg.eval.model_copy(update={"sample_pure_event_type": t})}
                )
                for r in univariate_cells(inp, c, feature, cell_list):
                    rows.append({"set": label, "feature_set": set_id, "event_type": t, **r})
        res = pl.DataFrame(rows)
        p = res["p"].to_numpy().astype(float)
        ok = ~np.isnan(p)
        disc = np.zeros(len(p), dtype=bool)
        disc[ok] = benjamini_hochberg(p[ok], cfg.eval.fdr_q)
        res = res.with_columns(pl.Series("bh_discovery", disc))
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(out_dir / f"explore-types-{run_id}.parquet")
        lines = [
            f"# EXPLORATORY per-type IC {run_id}",
            "",
            f"- feature `{feature}`, pure-type samples, pre-holdout; BH q={cfg.eval.fdr_q} "
            f"across {int(ok.sum())} tests. Exploratory: survivors are candidates for the "
            "live/holdout data, not findings.",
            "",
            "| set | type | cell | n | IC | 95% CI | p | BH |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['set']} | {r['event_type']} | {r['entry']}/{r['horizon']}/{r['adjustment']} "
                f"| {r['n']} | {r['ic']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                f"{r['p']:.3f} | {'*' if r['bh_discovery'] else ''} |"
            )
        md = "\n".join(lines) + "\n"
        (out_dir / f"explore-types-{run_id}.md").write_text(md, encoding="utf-8")
        with ctx.store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('explore_types', ?, ?, ?, ?)",
                [
                    run_id,
                    utcnow(),
                    run_id,
                    json.dumps({"sets": sets, "types": type_list, "cells": cells, "n": len(p)}),
                ],
            )
        stats["report"] = str(out_dir / f"explore-types-{run_id}.md")
        typer.echo(md)

    _run("explore-types", cfg, {"sets": sets, "types": types}, body)


@app.command("cluster-explore")
def cluster_explore_cmd(
    source: Annotated[str, typer.Option(help="Item source to cluster")] = "news_api",
    k: Annotated[int, typer.Option(help="Number of text clusters")] = 25,
    cells: Annotated[
        str, typer.Option(help="entry/horizon/adjustment cells, comma-separated")
    ] = "close/1/market,close/2/market",
    seed: int = 7,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """EXPLORATORY: cluster documents by text (TF-IDF -> SVD -> k-means, fitted pre-holdout),
    then rank IC of LLM sentiment per cluster; one BH correction across clusters x cells."""
    from newsfeed.eval.clusters import cluster_ic, decision_times, fit_text_clusters
    from newsfeed.features.aggregate import RELEVANCE_RANK

    _setup_logging(verbose)
    cfg = load_config(config)
    cell_list = [(e, int(h), a) for e, h, a in (c.split("/") for c in _split(cells))]

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        store, fc, ec = ctx.store, cfg.features, cfg.eval
        now = utcnow()
        items = store.items_as_of(
            now,
            source=source,
            columns="item_id, title, body_text, publisher, published_at, tradable_at, ingest_mode",
        )
        founders = store.query("SELECT item_id FROM item_story WHERE story_id = item_id")
        items = items.join(founders, on="item_id", how="semi").sort("item_id")
        holdout = datetime.combine(ec.holdout_start, datetime.min.time(), tzinfo=UTC)
        fit_mask = (items["published_at"] < holdout).to_numpy()
        texts = [f"{t or ''}. {b or ''}" for t, b in items.select("title", "body_text").rows()]
        labels, top = fit_text_clusters(texts, fit_mask, k=k, seed=seed)
        items = items.with_columns(pl.Series("cluster", labels))
        src = feature_sources(ctx)
        scores = store.query(
            """
            SELECT item_id, ticker AS canon, sentiment, relevance, scored_at FROM scores
            WHERE status = 'ok' AND masked = $masked AND replicate = 0
              AND model = $m AND prompt_hash = $h
            """,
            {"masked": fc.llm_masked, "m": src.llm_model, "h": src.llm_prompt_hash},
        )
        min_rank = RELEVANCE_RANK[fc.min_relevance]
        ev = items.join(scores, on="item_id").filter(
            pl.col("relevance").replace_strict(RELEVANCE_RANK, default=-1) >= min_rank
        )
        ev = ev.with_columns(
            pl.when(pl.col("ingest_mode") == "live")
            .then(pl.max_horizontal("tradable_at", "scored_at"))
            .otherwise(pl.col("tradable_at"))
            .alias("t_use")
        )
        t0 = ev["t_use"].min().date()
        grid = as_of_grid(ctx.calendar, t0, now.date(), fc.grid, fc.decision_lag_minutes)
        ev = ev.with_columns(
            pl.Series(
                "as_of", decision_times(ev["t_use"].to_list(), grid), dtype=pl.Datetime("us", "UTC")
            )
        ).drop_nulls("as_of")
        ev = ev.with_columns(
            pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
        )
        # scored company -> its tickers, kept only where it is a member
        ev = ctx.fan_out(ev.join(_canon_ciks(ctx, cfg), on=["item_id", "canon"]))
        feats = ev.select("ticker", "as_of").unique()
        inp, _ = _eval_inputs(ctx, cfg, feats)
        res = pl.DataFrame(
            cluster_ic(ev.select("ticker", "date", "cluster", "sentiment"), inp.targets,
                       inp.sessions, cfg, cell_list)
        )  # fmt: skip
        pv = res["p"].to_numpy().astype(float)
        ok = ~np.isnan(pv)
        disc = np.zeros(len(pv), dtype=bool)
        disc[ok] = benjamini_hochberg(pv[ok], ec.fdr_q)
        res = res.with_columns(pl.Series("bh_discovery", disc))
        info = (
            items.group_by("cluster")
            .agg(pl.len().alias("docs"), pl.col("publisher").mode().first().alias("top_publisher"))
            .join(
                ev.group_by("cluster").agg(pl.col("sentiment").mean().alias("mean_sent")),
                on="cluster",
                how="left",
            )
            .sort("cluster")
        )
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(out_dir / f"clusters-{run_id}.parquet")
        info.with_columns(
            pl.Series("top_terms", [", ".join(top[c]) for c in info["cluster"].to_list()])
        ).write_csv(out_dir / f"clusters-{run_id}-info.csv")
        lines = [
            f"# EXPLORATORY text clusters {run_id}",
            "",
            f"- source `{source}`, k={k}, TF-IDF -> SVD -> k-means fitted on pre-holdout "
            f"documents; LLM scores masked={fc.llm_masked}, relevance >= {fc.min_relevance}; "
            f"BH q={ec.fdr_q} across {int(ok.sum())} tests. Candidates only.",
            "",
            "| cluster | docs | top publisher | mean sent | top terms | cell | n | IC | p | BH |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        meta = {r["cluster"]: r for r in info.iter_rows(named=True)}
        for r in res.sort("cluster", "horizon").iter_rows(named=True):
            m = meta[r["cluster"]]
            ms = m["mean_sent"]
            lines.append(
                f"| {r['cluster']} | {m['docs']} | {m['top_publisher']} | "
                f"{'' if ms is None else f'{ms:+.2f}'} | {', '.join(top[r['cluster']][:5])} | "
                f"{r['entry']}/{r['horizon']}/{r['adjustment']} | {r['n']} | {r['ic']:+.3f} | "
                f"{r['p']:.3f} | {'*' if r['bh_discovery'] else ''} |"
            )
        md = "\n".join(lines) + "\n"
        (out_dir / f"clusters-{run_id}.md").write_text(md, encoding="utf-8")
        with store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('explore_clusters', ?, ?, ?, ?)",
                [
                    run_id,
                    utcnow(),
                    run_id,
                    json.dumps({"source": source, "k": k, "seed": seed, "cells": cells,
                                "masked": fc.llm_masked, "n_tests": int(ok.sum())}),
                ],
            )  # fmt: skip
        stats["report"] = str(out_dir / f"clusters-{run_id}.md")
        typer.echo(md)

    _run("cluster-explore", cfg, {"source": source, "k": k, "cells": cells}, body)


@app.command("partial-ic")
def partial_ic_cmd(
    feature_set: Annotated[str, typer.Option(help="Feature set defining the sample")],
    control_set: Annotated[str, typer.Option(help="Feature set holding the control column")],
    column: Annotated[str, typer.Option(help="Tested feature")] = "1d_llm_sent_cwmean",
    control_column: Annotated[str, typer.Option(help="Control feature")] = "5d_llm_sent_cwmean",
    cells: Annotated[
        str, typer.Option(help="entry/horizon/adjustment cells, comma-separated")
    ] = "close/1/market,close/2/market",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Partial rank IC of a feature given a control from another feature set (identical rows,
    pre-holdout): does the tested source add information beyond the control source?"""
    _setup_logging(verbose)
    cfg = load_config(config)
    cell_list = [(e, int(h), a) for e, h, a in (c.split("/") for c in _split(cells))]

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        drop = ("feature_set_id", "build_run_id", "built_at")
        a = ctx.store.features(feature_set).drop(*drop, strict=False)
        b = ctx.store.features(control_set).select("ticker", "as_of", control_column)
        if a.height == 0 or b.height == 0:
            raise typer.BadParameter("both feature sets must exist")
        ctrl = f"control__{control_column}"
        feats = a.join(b.rename({control_column: ctrl}), on=["ticker", "as_of"], how="left")
        inp, _ = _eval_inputs(ctx, cfg, feats)
        res = pl.DataFrame(partial_ic_cells(inp, cfg, column, ctrl, cell_list))
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(out_dir / f"partial-ic-{run_id}.parquet")
        lines = [
            f"# Partial IC {run_id}",
            "",
            f"- `{column}` from `{feature_set}` given `{control_column}` from `{control_set}`; "
            f"sample: {cfg.eval.sample}; rows with both present; pre-holdout.",
            "",
            "| cell | sample rows | rows used | IC tested | IC control | partial IC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['entry']}/{r['horizon']}/{r['adjustment']} | {r['n_sample']} | {r['n']} | "
                f"{r['ic_feature']:+.3f} | {r['ic_control']:+.3f} | {r['partial_ic']:+.3f} | "
                f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p']:.3f} |"
            )
        md = "\n".join(lines) + "\n"
        (out_dir / f"partial-ic-{run_id}.md").write_text(md, encoding="utf-8")
        with ctx.store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('partial_ic', ?, ?, ?, ?)",
                [
                    f"{feature_set}:{control_set}:{run_id}",
                    utcnow(),
                    run_id,
                    json.dumps({"column": column, "control": control_column, "cells": cells}),
                ],
            )
        stats["report"] = str(out_dir / f"partial-ic-{run_id}.md")
        typer.echo(md)

    _run("partial-ic", cfg, {"feature_set": feature_set, "control_set": control_set}, body)


def _canon_ciks(ctx: Context, cfg: AppConfig) -> pl.DataFrame:
    """(item_id, canon, cik): the item's qualified links with the canonical ticker their score
    is stored under -- as the features join them, so a score follows the item's *current* link
    (a stored score's own ``cik`` can predate a relink)."""
    from newsfeed.features.aggregate import _qualified_links
    from newsfeed.scoring.targets import canonical_tickers

    canon = canonical_tickers(cfg, ctx.universe(full=True).cik_to_tickers)
    canon_df = pl.DataFrame(
        {"cik": list(canon), "canon": list(canon.values())},
        schema={"cik": pl.Int64, "canon": pl.Utf8},
    )
    links = _qualified_links(ctx.store, cfg).with_columns(pl.col("cik").cast(pl.Int64))
    return links.join(canon_df, on="cik")


def _llm_events(
    ctx: Context,
    cfg: AppConfig,
    source: str,
    event_types: list[str],
    *,
    masked: bool | None = None,
) -> tuple[pl.DataFrame, list[datetime]]:
    """Scored story-founder events of one source at relevance >= ``features.min_relevance``:
    (events with canon ticker, cik, score fields and t_use = tradable_at, or max(tradable_at,
    scored_at) for live items; the decision grid)."""
    from newsfeed.features.aggregate import RELEVANCE_RANK

    store, fc = ctx.store, cfg.features
    now = utcnow()
    items = store.items_as_of(
        now, source=source, columns="item_id, tradable_at, ingest_mode, published_at"
    )
    founders = store.query("SELECT item_id FROM item_story WHERE story_id = item_id")
    items = items.join(founders, on="item_id", how="semi")
    src = feature_sources(ctx)
    scores = store.query(
        """
        SELECT item_id, ticker AS canon, sentiment, relevance, event_type, scored_at,
               surprise_vs_expectations, materiality
        FROM scores
        WHERE status = 'ok' AND masked = $masked AND replicate = 0
          AND model = $m AND prompt_hash = $h
        """,
        {
            "masked": fc.llm_masked if masked is None else masked,
            "m": src.llm_model,
            "h": src.llm_prompt_hash,
        },
    )
    ev = (
        items.join(scores, on="item_id")
        .filter(
            pl.col("relevance").replace_strict(RELEVANCE_RANK, default=-1)
            >= RELEVANCE_RANK[fc.min_relevance]
        )
        .join(_canon_ciks(ctx, cfg), on=["item_id", "canon"])
    )
    if event_types:
        ev = ev.filter(pl.col("event_type").is_in(event_types))
    ev = ev.with_columns(
        pl.when(pl.col("ingest_mode") == "live")
        .then(pl.max_horizontal("tradable_at", "scored_at"))
        .otherwise(pl.col("tradable_at"))
        .alias("t_use"),
        pl.lit(0).alias("cluster"),
    )
    grid = as_of_grid(
        ctx.calendar, ev["t_use"].min().date(), now.date(), fc.grid, fc.decision_lag_minutes
    )
    return ev, grid


def _dated_events(
    ctx: Context,
    ev: pl.DataFrame,
    grid: list[datetime],
    *,
    delay: int = 0,
) -> pl.DataFrame:
    """Events (t_use, cik) -> as_of = first decision >= t_use (+ ``delay`` sessions), date in
    New York, fanned out to the company's tickers that are members on that date."""
    from newsfeed.eval.clusters import decision_times

    e = ev.with_columns(
        pl.Series(
            "as_of",
            decision_times(ev["t_use"].to_list(), grid, delay=delay),
            dtype=pl.Datetime("us", "UTC"),
        )
    ).drop_nulls("as_of")
    return ctx.fan_out(
        e.with_columns(
            pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
        )
    )


@app.command("delay-sensitivity")
def delay_sensitivity_cmd(
    source: Annotated[str, typer.Option(help="Item source")] = "transcripts",
    delays: Annotated[str, typer.Option(help="Extra sessions of delay, comma-separated")] = (
        "0,1,2,5"
    ),
    cells: Annotated[
        str, typer.Option(help="entry/horizon/adjustment cells, comma-separated")
    ] = "close/1/market,close/2/market",
    event_types: Annotated[
        str | None, typer.Option(help="Only LLM events of these types (comma-separated)")
    ] = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Event-level rank IC of LLM sentiment when every event becomes usable N sessions later
    than its tradable time: how fast does the signal decay if the data arrives late?"""
    from newsfeed.eval.clusters import cluster_ic

    _setup_logging(verbose)
    cfg = load_config(config)
    cell_list = [(e, int(h), a) for e, h, a in (c.split("/") for c in _split(cells))]
    delay_list = [int(x) for x in _split(delays)]

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        fc = cfg.features
        ev, grid = _llm_events(ctx, cfg, source, _split(event_types))
        rows = []
        inp_cache = None
        for d in delay_list:
            e = _dated_events(ctx, ev, grid, delay=d)
            if inp_cache is None:  # prices/targets once, over the widest set of decisions
                wide = _dated_events(ctx, ev, grid, delay=max(delay_list))
                feats = pl.concat(
                    [e.select("ticker", "as_of"), wide.select("ticker", "as_of")]
                ).unique()
                inp_cache, _ = _eval_inputs(ctx, cfg, feats)
            for r in cluster_ic(
                e.select("ticker", "date", "cluster", "sentiment"),
                inp_cache.targets,
                inp_cache.sessions,
                cfg,
                cell_list,
            ):
                rows.append({"delay_sessions": d, **{k: v for k, v in r.items() if k != "cluster"}})
        res = pl.DataFrame(rows)
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        res.write_parquet(out_dir / f"delay-{run_id}.parquet")
        lines = [
            f"# Delay sensitivity {run_id}",
            "",
            f"- source `{source}`, events {event_types or 'all'}, LLM scores "
            f"masked={fc.llm_masked}, relevance >= "
            f"{fc.min_relevance}; event-level IC, pre-holdout; delay = extra sessions after the "
            "first usable decision.",
            "",
            "| delay (sessions) | cell | n | IC | 95% CI | p |",
            "|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['delay_sessions']} | {r['entry']}/{r['horizon']}/{r['adjustment']} | "
                f"{r['n']} | {r['ic']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                f"{r['p']:.3f} |"
            )
        md = "\n".join(lines) + "\n"
        (out_dir / f"delay-{run_id}.md").write_text(md, encoding="utf-8")
        stats["report"] = str(out_dir / f"delay-{run_id}.md")
        typer.echo(md)

    params = {"source": source, "delays": delays, "event_types": event_types}
    _run("delay-sensitivity", cfg, params, body)


@app.command("transcripts-latency")
def transcripts_latency_cmd(config: ConfigOpt = None) -> None:
    """How late do live-collected calls arrive? For each LIVE transcript: first_seen vs the
    company's earnings 8-K (acceptance time, within 3 days before the call date) and whether
    it was in hand by the next open and by the next 15:50 decision after the release."""
    cfg = load_config(config)
    with Store(cfg.paths) as store:
        cal = get_calendar(cfg.market.calendar, cfg.market.calendar_start)
        tr = store.query(
            """
            SELECT item_id, published_at_raw AS call_date, first_seen_at, meta
            FROM items WHERE source = 'transcripts' AND ingest_mode = 'live'
            """
        )
        if tr.height == 0:
            typer.echo("no live transcripts yet")
            return
        tr = tr.with_columns(
            pl.col("meta").str.json_path_match("$.cik").cast(pl.Int64).alias("cik"),
            pl.col("call_date").str.slice(0, 10).str.to_date().alias("call_date"),
        )
        k = store.query(
            """
            SELECT DISTINCT t.cik, i.published_at AS release_at FROM items i
            JOIN item_tickers t USING (item_id)
            WHERE i.source = 'edgar' AND i.body_text LIKE '%[ITEM 2.02%'
            """
        ).with_columns(pl.col("cik").cast(pl.Int64))
        rows = []
        for r in tr.iter_rows(named=True):
            rel = k.filter(pl.col("cik") == r["cik"])["release_at"].to_list()
            rel = [x for x in rel if 0 <= (r["call_date"] - x.astimezone(NY_TZ).date()).days <= 3]
            release = max(rel) if rel else None
            seen = r["first_seen_at"]
            nxt_open = cal.push_to_session(release + timedelta(seconds=1)) if release else None
            dec = None
            if release is not None:
                d = release.astimezone(NY_TZ).date()
                for _ in range(10):
                    b = cal.session_bounds(d)
                    if b is not None and b[1] - timedelta(minutes=10) > release:
                        dec = b[1] - timedelta(minutes=10)
                        break
                    d += timedelta(days=1)
            rows.append(
                {
                    "call_date": r["call_date"],
                    "cik": r["cik"],
                    "release_at": release,
                    "first_seen_at": seen,
                    "hours_after_release": (seen - release).total_seconds() / 3600
                    if release
                    else None,
                    "by_next_open": seen <= nxt_open if nxt_open else None,
                    "by_next_decision": seen <= dec if dec else None,
                }
            )
        df = pl.DataFrame(rows)
        out = cfg.paths.data_dir / "reports" / "transcripts-latency.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(out)
        m = df.drop_nulls("hours_after_release")
        typer.echo(f"live transcripts: {df.height}; matched to an earnings 8-K: {m.height}")
        if m.height:
            q = m["hours_after_release"].quantile
            typer.echo(
                f"hours after release: median {q(0.5):.1f}, p75 {q(0.75):.1f}, "
                f"p90 {q(0.9):.1f}, max {m['hours_after_release'].max():.1f}"
            )
            typer.echo(
                f"in hand by the next open: {m['by_next_open'].mean():.0%}; "
                f"by the next 15:50 decision: {m['by_next_decision'].mean():.0%}"
            )
        typer.echo(f"rows: {out}")


def _cells(spec: str) -> list[tuple[str, int, str]]:
    return [(e, int(h), a) for e, h, a in (c.split("/") for c in _split(spec))]


def _write_report(cfg: AppConfig, name: str, md: str, frame: pl.DataFrame) -> Path:
    out_dir = cfg.paths.data_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if frame.height:
        frame.write_parquet(out_dir / f"{name}.parquet")
    path = out_dir / f"{name}.md"
    path.write_text(md, encoding="utf-8")
    typer.echo(md)
    return path


def _num(v: Any, fmt: str = "+.3f") -> str:
    return "-" if v is None or v != v else format(v, fmt)


@app.command("event-ic")
def event_ic_cmd(
    source: Annotated[str, typer.Option(help="Item source")] = "edgar",
    event_types: Annotated[
        str | None, typer.Option(help="Only LLM events of these types (comma-separated)")
    ] = "earnings",
    value: Annotated[
        str, typer.Option(help="Score field tested (sentiment, surprise_vs_expectations)")
    ] = "surprise_vs_expectations",
    control: Annotated[
        str | None, typer.Option(help="Semi-partial IC: residualise the value on this field")
    ] = None,
    by: Annotated[
        str | None, typer.Option(help="Break down by 'materiality' (high vs medium/low)")
    ] = None,
    masked: Annotated[
        bool | None, typer.Option("--masked/--unmasked", help="Default: features.llm_masked")
    ] = None,
    cells: Annotated[str, typer.Option(help="entry/horizon/adjustment cells")] = (
        "close/1/market,close/2/market"
    ),
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Event-level rank IC of one LLM score field (test 6: earnings surprise), each event dated
    to the first decision at or after it became usable; pre-holdout."""
    from newsfeed.eval.clusters import cluster_ic

    _setup_logging(verbose)
    cfg = load_config(config)
    cell_list = _cells(cells)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        ev, grid = _llm_events(ctx, cfg, source, _split(event_types), masked=masked)
        if by == "materiality":
            ev = ev.filter(pl.col("materiality").is_not_null()).with_columns(
                pl.when(pl.col("materiality") == "high")
                .then(pl.lit("high"))
                .otherwise(pl.lit("medium/low"))
                .alias("cluster")
            )
        elif by:
            raise typer.BadParameter(f"unknown --by {by!r}")
        e = _dated_events(ctx, ev, grid)
        inp, _ = _eval_inputs(ctx, cfg, e.select("ticker", "as_of").unique())
        cols = ["ticker", "date", "cluster", value] + ([control] if control else [])
        res = pl.DataFrame(
            cluster_ic(
                e.select(cols),
                inp.targets,
                inp.sessions,
                cfg,
                cell_list,
                value=value,
                control=control,
            )
        )
        is_masked = cfg.features.llm_masked if masked is None else masked
        what = f"semi-partial IC of `{value}` given `{control}`" if control else f"`{value}`"
        head = "| cell | group | n | IC | 95% CI | p |"
        if control:
            head += " IC value | IC control |"
        lines = [
            f"# Event IC {run_id}",
            "",
            f"- {what}; source `{source}`, events {event_types or 'all'}, masked={is_masked}, "
            f"relevance >= {cfg.features.min_relevance}; event-level, pre-holdout; rows with the "
            "value present.",
            "",
            head,
            "|---" * head.count(" |") + "|",
        ]
        for r in res.iter_rows(named=True):
            line = (
                f"| {r['entry']}/{r['horizon']}/{r['adjustment']} | {r['cluster']} | {r['n']} | "
                f"{_num(r['ic'])} | [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} |"
            )
            if control:
                line += f" {_num(r.get('ic_value'))} | {_num(r.get('ic_control'))} |"
            lines.append(line)
        path = _write_report(cfg, f"event-ic-{run_id}", "\n".join(lines) + "\n", res)
        stats["report"] = str(path)

    params = {
        "source": source,
        "event_types": event_types,
        "value": value,
        "control": control,
        "by": by,
        "masked": masked,
        "cells": cells,
    }
    _run("event-ic", cfg, params, body)


@app.command("recall-check")
def recall_check_cmd(
    predictions: Annotated[
        str, typer.Option(help="Comma-separated walk-forward student runs under data/distill/runs")
    ] = "wf1,wf2,wf3,wf4",
    start: Annotated[str, typer.Option(help="Events with t_use >= this UTC date")] = "2021-01-01",
    end: Annotated[str, typer.Option(help="Events with t_use < this UTC date")] = "2026-07-01",
    cells: Annotated[str, typer.Option(help="entry/horizon/adjustment cells")] = (
        "close/2/market,close/1/market,close/1/sector,close/2/sector"
    ),
    reps: Annotated[int, typer.Option(help="Bootstrap repetitions")] = 10_000,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Distillation step 3 (pre-registered 2026-10-24): on identical 8-K events, rank IC of
    Haiku's sentiment vs a walk-forward student's (each document scored by a student trained
    only on documents tradable before its period). Earnings (primary) and guidance events."""
    from newsfeed.eval.clusters import paired_event_ic

    _setup_logging(verbose)
    cfg = load_config(config)
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"bootstrap_reps": reps})})
    cell_list = _cells(cells)
    runs = _split(predictions)
    t0, t1 = _parse_day(start), _parse_day(end)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        if cfg.universe.all_extra_members():
            raise typer.BadParameter("use an S&P 500-only config (the pre-registered universe)")
        d = cfg.paths.data_dir / "distill" / "runs"
        preds = pl.concat(
            [
                pl.read_parquet(d / r / "predictions.parquet")
                .filter(pl.col("source") == "edgar")
                .select("item_id", pl.col("ticker").alias("canon"), "pred_sent")
                for r in runs
            ]
        )
        dup = preds.group_by("item_id", "canon").len().filter(pl.col("len") > 1)
        if dup.height:
            raise typer.BadParameter(f"{dup.height} documents scored by several students")
        lines = [
            f"# Recall check {run_id}",
            "",
            f"- events: 8-K founders, unmasked Haiku labels, relevance >= "
            f"{cfg.features.min_relevance}, t_use in [{start}, {end}), dated to the first "
            "decision; student = walk-forward runs " + ", ".join(runs),
            f"- a = Haiku sentiment, b = student sentiment; diff = IC(b) - IC(a), paired block "
            f"bootstrap over sessions, {reps} reps",
            "",
            "| events | cell | n | corr(a,b) | IC Haiku | IC student | diff | 95% CI diff | "
            "p diff | p Haiku | p student |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        frames = []
        for etype in ("earnings", "guidance"):
            ev, grid = _llm_events(ctx, cfg, "edgar", [etype], masked=False)
            ev = ev.filter((pl.col("t_use") >= t0) & (pl.col("t_use") < t1))
            n_ev = ev.height
            ev = ev.join(preds, on=["item_id", "canon"], how="inner")
            stats[f"{etype}_events"] = n_ev
            stats[f"{etype}_with_student"] = ev.height
            e = _dated_events(ctx, ev, grid)
            inp, _ = _eval_inputs(ctx, cfg, e.select("ticker", "as_of").unique())
            res = pl.DataFrame(
                paired_event_ic(
                    e.select("ticker", "date", "sentiment", "pred_sent"),
                    inp.targets,
                    inp.sessions,
                    cfg,
                    cell_list,
                    "sentiment",
                    "pred_sent",
                )
            ).with_columns(pl.lit(etype).alias("events"))
            frames.append(res)
            for r in res.iter_rows(named=True):
                lines.append(
                    f"| {etype} | {r['entry']}/{r['horizon']}/{r['adjustment']} | {r['n']} | "
                    f"{_num(r.get('corr_ab'), '.3f')} | {_num(r['ic_a'])} | {_num(r['ic_b'])} | "
                    f"{_num(r['diff'])} | [{_num(r.get('ci_lo_diff'))}, "
                    f"{_num(r.get('ci_hi_diff'))}] | {_num(r.get('p_diff'), '.3f')} | "
                    f"{_num(r.get('p_a'), '.3f')} | {_num(r.get('p_b'), '.3f')} |"
                )
            lines.append(
                f"\n_{etype}: {n_ev} Haiku events in the window, {ev.height} with a student "
                "score_\n"
            )
        out = pl.concat(frames, how="diagonal_relaxed")
        path = _write_report(cfg, f"recall-check-{run_id}", "\n".join(lines) + "\n", out)
        stats["report"] = str(path)
        with ctx.store.ops() as con:
            con.execute(
                "INSERT INTO experiments VALUES ('recall_check', ?, ?, ?, ?)",
                [
                    f"recall:{','.join(runs)}:{run_id}",
                    utcnow(),
                    run_id,
                    json.dumps(
                        {"runs": runs, "start": start, "end": end, "cells": cells, "reps": reps}
                    ),
                ],
            )

    params = {"predictions": predictions, "start": start, "end": end, "cells": cells, "reps": reps}
    _run("recall-check", cfg, params, body)


# test 7 controls: trailing volatility, |overnight gap|, |yesterday's return|, liquidity, volume
VOL_CONTROLS = ["ctl_vol_20d", "abs_gap", "abs_ret_1d", "ctl_log_dvol_20d", "ctl_volume_ratio"]


@app.command("vol-test")
def vol_test_cmd(
    feature_set: Annotated[str, typer.Option(help="Feature set (all member ticker-days)")],
    features: Annotated[str, typer.Option(help="Attention features, comma-separated")] = (
        "1d_n_stories,1d_volume_z,5d_n_stories,1d_llm_sent_maxabs"
    ),
    targets: Annotated[str, typer.Option(help="rv_5, rv_20, absret_1")] = "rv_5,rv_20,absret_1",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 7: does news attention predict realised volatility beyond trailing volatility,
    the overnight gap, yesterday's move and trading volume? Partial rank correlation on every
    member ticker-day (pre-holdout), all days and non-earnings days."""
    from newsfeed.eval.report import partial_rank_corr

    _setup_logging(verbose)
    cfg = load_config(config)
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"sample": "all"})})
    feat_list, target_list = _split(features), _split(targets)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        drop = ("feature_set_id", "build_run_id", "built_at")
        feats = ctx.store.features(feature_set).drop(*drop, strict=False)
        inp, _ = _eval_inputs(ctx, cfg, feats.select("ticker", "as_of"))
        inp.features = _sample_companies(ctx, cfg, feats)
        base = assemble(inp, cfg).join(inp.vol_targets, on=["ticker", "date"], how="left")
        base = base.with_columns(
            pl.col("ctl_gap").abs().alias("abs_gap"),
            pl.col("ctl_ret_1d").abs().alias("abs_ret_1d"),
        )
        session_idx = {d: i for i, d in enumerate(inp.sessions)}
        hold = next(
            (i for i, d in enumerate(inp.sessions) if d >= cfg.eval.holdout_start),
            len(inp.sessions),
        )
        s_all = np.array([session_idx.get(d, -1) for d in base["date"].to_list()])
        earn = base["1d_llm_evt_earnings"].fill_null(0).to_numpy() > 0
        ctl = base.select(VOL_CONTROLS).to_numpy().astype(float)
        rng = np.random.default_rng(cfg.eval.seed)
        rows = []
        for tgt in target_list:
            h = 1 if tgt == "absret_1" else int(tgt.split("_")[1])
            y = base[tgt].to_numpy().astype(float)
            pre = (s_all >= 0) & (s_all + h < hold)  # the label ends before the holdout
            for feat in feat_list:
                x = base[feat].to_numpy().astype(float)
                for subset, m in (("all", pre), ("non-earnings", pre & ~earn)):
                    r = partial_rank_corr(
                        x[m],
                        y[m],
                        ctl[m],
                        s_all[m],
                        block=h + 1,
                        reps=cfg.eval.bootstrap_reps,
                        rng=rng,
                    )
                    rows.append({"feature": feat, "target": tgt, "subset": subset, **r})
                    log.info("vol-test %s %s %s: %s", feat, tgt, subset, r)
        res = pl.DataFrame(rows)
        lines = [
            f"# Attention -> volatility {run_id}",
            "",
            f"- feature set `{feature_set}`, every member ticker-day with the feature present, "
            "pre-holdout; close entry, raw returns. rv_h = sd of daily log returns over the next "
            "h sessions; absret_1 = |next-day return|.",
            f"- partial IC = rank correlation after regressing out {', '.join(VOL_CONTROLS)} "
            "(ranks); block bootstrap over sessions. Primary (pre-registered): 1d_n_stories vs "
            "rv_5, all days.",
            "",
            "| feature | target | subset | n | raw IC | partial IC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['feature']} | {r['target']} | {r['subset']} | {r['n']} | "
                f"{_num(r['ic'])} | {_num(r['partial_ic'])} | [{_num(r['ci_lo'])}, "
                f"{_num(r['ci_hi'])}] | {_num(r['p'], '.3f')} |"
            )
        path = _write_report(cfg, f"vol-{run_id}", "\n".join(lines) + "\n", res)
        stats["report"] = str(path)

    _run("vol-test", cfg, {"feature_set": feature_set, "features": features}, body)


ITEM_CODES = "2.02,7.01,8.01,5.02,1.01,5.07,2.03,5.03,1.02,2.01,3.03,3.02,2.05,3.01,2.06"


@app.command("item-study")
def item_study_cmd(
    feature_set: Annotated[
        str, typer.Option(help="Feature set whose rows define index members per day")
    ],
    codes: Annotated[str, typer.Option(help="8-K item codes")] = ITEM_CODES,
    horizons: Annotated[str, typer.Option(help="Close-entry horizons")] = "1,5,20",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 8: mean abnormal return after 8-Ks by item code (drift from the first decision
    after the filing); market-adjusted cells tested with one BH, sector-adjusted reported."""
    from newsfeed.eval.clusters import decision_times
    from newsfeed.eval.events import abnormal_returns, event_mean_cells

    _setup_logging(verbose)
    cfg = load_config(config)
    code_list = _split(codes)
    hs = [int(h) for h in _split(horizons)]

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        fc = cfg.features
        now = utcnow()
        items = ctx.store.items_as_of(now, source="edgar", columns="item_id, tradable_at, meta")
        meta = pl.Struct({"form": pl.Utf8, "cik": pl.Int64, "items": pl.List(pl.Utf8)})
        items = (
            items.with_columns(pl.col("meta").str.json_decode(meta).alias("_m"))
            .unnest("_m")
            .filter(pl.col("form") == "8-K")
            .explode("items")
            .rename({"items": "group"})
            .filter(pl.col("group").is_in(code_list))
        )
        grid = as_of_grid(
            ctx.calendar,
            items["tradable_at"].min().date(),
            now.date(),
            fc.grid,
            fc.decision_lag_minutes,
        )
        ev = ctx.fan_out(
            items.with_columns(
                pl.Series(
                    "as_of",
                    decision_times(items["tradable_at"].to_list(), grid),
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
            .drop_nulls("as_of")
            .with_columns(
                pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
        )
        members = ctx.store.features(feature_set).select("ticker", "as_of")
        inp, _ = _eval_inputs(ctx, cfg, members)
        members = members.select(
            "ticker",
            pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date"),
        )
        ar = abnormal_returns(inp.targets, members)
        cells = [("close", h, a) for h in hs for a in ("market", "sector")]
        res = pl.DataFrame(event_mean_cells(ev, ar, inp.sessions, cfg, cells))
        mk = (res["adjustment"] == "market").to_numpy()
        p_mk = res["p"].to_numpy().astype(float)[mk]
        flags = np.zeros(res.height, dtype=bool)
        flags[np.flatnonzero(mk)] = benjamini_hochberg(p_mk, cfg.eval.fdr_q)
        res = res.with_columns(pl.Series("bh_discovery", flags))
        n_tests = int(np.sum(~np.isnan(p_mk)))
        lines = [
            f"# 8-K item event study {run_id}",
            "",
            "- original 8-Ks; one event per (member ticker, decision date, item code); decision "
            "= first 15:50 ET at or after tradable_at; close entry; pre-holdout.",
            "- abnormal return (AR) = benchmark-adjusted return minus the same-date "
            "equal-weighted mean over all index members (members from feature set "
            f"`{feature_set}`); mean vs 0, block bootstrap over sessions.",
            f"- BH (q {cfg.eval.fdr_q}) over the {n_tests} market-adjusted tests; "
            "sector-adjusted columns are descriptive.",
            *p_floor_note(cfg.eval.bootstrap_reps, n_tests, cfg.eval.fdr_q),
            "",
            "| item | h | n | sessions | mean AR (mkt) | 95% CI | p | BH | share > 0 | "
            "mean AR (sector) | p (sector) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        sec = res.filter(~pl.Series(mk))
        for r in res.filter(pl.Series(mk)).sort("group", "horizon").iter_rows(named=True):
            o = sec.filter((pl.col("group") == r["group"]) & (pl.col("horizon") == r["horizon"]))
            o = o.to_dicts()[0] if o.height else {"mean": float("nan"), "p": float("nan")}
            lines.append(
                f"| {r['group']} | {r['horizon']} | {r['n']} | {r['sessions']} | "
                f"{bp(r['mean'])} | [{bp(r['ci_lo'])}, {bp(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} | {'yes' if r['bh_discovery'] else ''} | "
                f"{_num(r['hit'], '.2f')} | {bp(o['mean'])} | {_num(o['p'], '.3f')} |"
            )
        path = _write_report(cfg, f"items-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), n_tests=n_tests, discoveries=int(flags.sum()))

    params = {"feature_set": feature_set, "codes": codes, "horizons": horizons}
    _run("item-study", cfg, params, body)


@app.command("ic-detail")
def ic_detail_cmd(
    feature_set: Annotated[str, typer.Option(help="Feature set id")],
    feature: Annotated[str, typer.Option(help="Feature column")] = "1d_llm_sent_cwmean",
    cells: Annotated[str, typer.Option(help="entry/horizon/adjustment cells")] = (
        "close/1/market,close/2/market"
    ),
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 9 (descriptive): pooled IC next to the daily cross-sectional IC, ICIR and Q5 - Q1
    spread of one feature on the config's eval sample, pre-holdout."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        drop = ("feature_set_id", "build_run_id", "built_at")
        feats = add_derived(ctx.store.features(feature_set).drop(*drop, strict=False))
        inp, _ = _eval_inputs(ctx, cfg, feats)
        res = pl.DataFrame(univariate_cells(inp, cfg, feature, _cells(cells), detail=True))
        ec = cfg.eval
        sample = f"`{ec.sample}`"
        if ec.sample_event_types:
            sample += f", event types {ec.sample_event_types}"
        if ec.sample_pure_event_type:
            sample += f", pure {ec.sample_pure_event_type}"
        lines = [
            f"# IC detail {run_id}",
            "",
            f"- feature set `{feature_set}`, `{feature}`, sample {sample}; pre-holdout.",
            "- daily view (descriptive, not BH-tested): sessions with >= 10 names; ICIR = mean /"
            " sd of the daily IC (per day); Q5 - Q1 = mean daily spread between the top and "
            "bottom quintile of the feature. CIs: block bootstrap over sessions.",
            "",
            "| cell | rows | pooled IC | p | days | daily IC | 95% CI | p | ICIR | Q5 - Q1 | "
            "95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['entry']}/{r['horizon']}/{r['adjustment']} | {r['n']} | {_num(r['ic'])} | "
                f"{_num(r['p'], '.3f')} | {r.get('d_days', 0)} | {_num(r.get('d_ic_daily'))} | "
                f"[{_num(r.get('d_ic_daily_ci_lo'))}, {_num(r.get('d_ic_daily_ci_hi'))}] | "
                f"{_num(r.get('d_ic_daily_p'), '.3f')} | {_num(r.get('d_icir'))} | "
                f"{bp(r.get('d_q_spread'))} | [{bp(r.get('d_q_spread_ci_lo'))}, "
                f"{bp(r.get('d_q_spread_ci_hi'))}] | {_num(r.get('d_q_spread_p'), '.3f')} |"
            )
        path = _write_report(cfg, f"ic-detail-{run_id}", "\n".join(lines) + "\n", res)
        stats["report"] = str(path)

    _run("ic-detail", cfg, {"feature_set": feature_set, "feature": feature, "cells": cells}, body)


@app.command("doc-change")
def doc_change_cmd(
    primary_cells: Annotated[str, typer.Option(help="Pre-registered cells")] = (
        "close/20/market,close/60/market"
    ),
    secondary_cells: Annotated[str, typer.Option(help="Reported, not primary")] = (
        "close/1/market,close/5/market,close/10/market,close/20/sector,close/60/sector"
    ),
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 10: does a 10-K / 10-Q that changed little from the year-earlier report predict
    higher returns ("Lazy Prices")? Event-level rank IC of document similarity, pre-holdout."""
    from newsfeed.eval.clusters import cluster_ic, decision_times
    from newsfeed.eval.cv import pre_holdout_mask
    from newsfeed.eval.docchange import filing_meta, pair_filings, similarities
    from newsfeed.eval.report import ic_detail

    _setup_logging(verbose)
    cfg = load_config(config)
    prim, sec = _cells(primary_cells), _cells(secondary_cells)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        store, fc, src = ctx.store, cfg.features, "edgar_periodic"
        now = utcnow()
        f = filing_meta(store.items_as_of(now, source=src, columns="item_id, tradable_at, meta"))
        pairs = pair_filings(f)
        log.info("doc-change: %d filings, %d with a year-earlier pair", f.height, pairs.height)

        def load(ids: list[str]) -> dict[str, str]:
            b = store.items_as_of(now, source=src, item_ids=ids, columns="item_id, body_text")
            return dict(zip(b["item_id"].to_list(), b["body_text"].to_list(), strict=True))

        sim = similarities(f, pairs, load)
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        sim.write_parquet(out_dir / f"doc-change-{run_id}-similarity.parquet")
        grid = as_of_grid(
            ctx.calendar, sim["tradable_at"].min().date(), now.date(), fc.grid,
            fc.decision_lag_minutes,
        )  # fmt: skip
        ev = ctx.fan_out(
            sim.with_columns(
                pl.Series(
                    "as_of",
                    decision_times(sim["tradable_at"].to_list(), grid),
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
            .drop_nulls("as_of")
            .with_columns(
                pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
        )
        inp, _ = _eval_inputs(ctx, cfg, ev.select("ticker", "as_of").unique())
        ev = ev.join(inp.caps.select("ticker", "date", "mcap"), on=["ticker", "date"], how="left")
        # NaN (section missing in one filing) -> null, so it counts as absent, not as a value
        ev = ev.with_columns(
            pl.col("mcap").log().alias("log_mcap"),
            *[pl.col(c).fill_nan(None) for c in ("cos", "jac", "cos_mda", "cos_rf")],
        )
        stacked = pl.concat(
            [
                ev.with_columns(pl.lit("all").alias("cluster")),
                ev.with_columns(pl.col("form").alias("cluster")),
            ]
        )
        cols = ["ticker", "date", "cluster", "cos", "jac", "cos_mda", "cos_rf", "log_mcap"]
        stacked = stacked.select(cols)
        rows = []
        for tier, cells in (("primary", prim), ("secondary", sec)):
            for value, control in (("cos", None), ("jac", None), ("cos_mda", None),
                                   ("cos_rf", None), ("cos", "log_mcap")):  # fmt: skip
                if tier == "primary" or value == "cos":
                    for r in cluster_ic(
                        stacked, inp.targets, inp.sessions, cfg, cells, value=value,
                        control=control,
                    ):  # fmt: skip
                        rows.append({"tier": tier, "value": value, "control": control or "", **r})
        res = pl.DataFrame(rows)
        # descriptive daily view for the primary measure / cells, all filings
        session_idx = {d: i for i, d in enumerate(inp.sessions)}
        daily = []
        rng = np.random.default_rng(cfg.eval.seed + 1)
        for entry, h, adj in prim:
            t = inp.targets.filter((pl.col("entry") == entry) & (pl.col("horizon") == h))
            d = ev.join(t, on=["ticker", "date"]).filter(pl.col("cos").is_not_null())
            dec = d["date"].to_numpy().astype("datetime64[D]")
            end = d["label_end"].to_numpy().astype("datetime64[D]")
            d = d.filter(pl.Series(pre_holdout_mask(dec, end, cfg.eval.holdout_start)))
            s = np.array([session_idx.get(x, -1) for x in d["date"].to_list()])
            daily.append(
                {"cell": f"{entry}/{h}/{adj}", **ic_detail(
                    s, d["cos"].to_numpy().astype(float), d[f"ret_{adj}"].to_numpy().astype(float),
                    block=h + 1, reps=cfg.eval.bootstrap_reps, rng=rng,
                )}
            )  # fmt: skip
        n_pre = ev.filter(pl.col("date") < cfg.eval.holdout_start)
        lines = [
            f"# Document change (Lazy Prices) {run_id}",
            "",
            f"- {f.height} 10-K/10-Q filings, {pairs.height} with a year-earlier same-form pair; "
            f"{n_pre.height} member-ticker events before the holdout "
            f"({n_pre['ticker'].n_unique()} tickers).",
            "- similarity: cosine of term-frequency vectors (words of >= 2 letters, numbers "
            "dropped); high = little change; event dated to the first decision after tradable_at;"
            " pre-holdout; block bootstrap over sessions.",
            f"- median cosine: all {sim['cos'].median():.3f}, 10-K "
            f"{sim.filter(pl.col('form') == '10-K')['cos'].median():.3f}, 10-Q "
            f"{sim.filter(pl.col('form') == '10-Q')['cos'].median():.3f}; MD&A found in both "
            f"filings for {sim['cos_mda'].is_not_nan().sum()} pairs, risk factors for "
            f"{sim['cos_rf'].is_not_nan().sum()}.",
            "",
            "| tier | value | control | cell | group | n | IC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['tier']} | {r['value']} | {r['control'] or '-'} | "
                f"{r['entry']}/{r['horizon']}/{r['adjustment']} | {r['cluster']} | {r['n']} | "
                f"{_num(r['ic'])} | [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} |"
            )
        lines += [
            "",
            "Daily view (descriptive; sessions with >= 10 filings):",
            "",
            "| cell | days | daily IC | 95% CI | ICIR | Q5 - Q1 | 95% CI |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in daily:
            lines.append(
                f"| {d['cell']} | {d['days']} | {_num(d['ic_daily'])} | "
                f"[{_num(d['ic_daily_ci_lo'])}, {_num(d['ic_daily_ci_hi'])}] | "
                f"{_num(d['icir'])} | {bp(d['q_spread'])} | [{bp(d['q_spread_ci_lo'])}, "
                f"{bp(d['q_spread_ci_hi'])}] |"
            )
        path = _write_report(cfg, f"doc-change-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), filings=f.height, pairs=pairs.height)

    _run("doc-change", cfg, {"primary": primary_cells, "secondary": secondary_cells}, body)


@app.command("mdna-tone")
def mdna_tone_cmd(
    primary_cells: Annotated[str, typer.Option(help="Pre-registered cells")] = (
        "close/20/market,close/60/market"
    ),
    secondary_cells: Annotated[str, typer.Option(help="Reported, not primary")] = (
        "close/1/market,close/5/market,close/10/market,close/20/sector,close/60/sector"
    ),
    release_set: Annotated[
        str, typer.Option(help="8-K feature set holding the release sentiment control")
    ] = "8990ee7a9775",
    finbert: Annotated[bool, typer.Option(help="Also score the MD&As with FinBERT")] = True,
    tones: Annotated[
        Path | None, typer.Option(help="Reuse a tones parquet from an earlier run")
    ] = None,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 11: does the change in MD&A tone from the year-earlier 10-K / 10-Q predict returns?
    Event-level rank IC of the LM tone change (local scorers only), pre-holdout."""
    from newsfeed.eval.clusters import cluster_ic, decision_times
    from newsfeed.eval.cv import pre_holdout_mask
    from newsfeed.eval.docchange import filing_meta, pair_filings
    from newsfeed.eval.mdna import mdna_tones, tone_changes
    from newsfeed.eval.report import ic_detail
    from newsfeed.features.baselines import make_finbert

    _setup_logging(verbose)
    cfg = load_config(config)
    prim, sec = _cells(primary_cells), _cells(secondary_cells)
    control = "rel_sent"

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        store, fc, src = ctx.store, cfg.features, "edgar_periodic"
        now = utcnow()
        f = filing_meta(store.items_as_of(now, source=src, columns="item_id, tradable_at, meta"))
        pairs = pair_filings(f)
        out_dir = cfg.paths.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        if tones is not None:
            tn = pl.read_parquet(tones)
        else:

            def load(ids: list[str]) -> dict[str, str]:
                # ids come from the as-of read above; items_as_of(item_ids=) would dedupe the
                # whole lake per call (out of memory on 16 GB)
                b = store.items_by_id(ids, sources=[src], columns="item_id, body_text")
                return dict(zip(b["item_id"].to_list(), b["body_text"].to_list(), strict=True))

            classify = None
            if finbert:
                import torch  # type: ignore[import-not-found]

                dev = 0 if torch.cuda.is_available() else None
                classify = make_finbert(fc.finbert_model, device=dev, batch_size=32)
            tn = mdna_tones(f, load, load_lm_dictionary(fc.lm_dictionary), classify=classify)
            # written before any return is joined, so a re-run can reuse it (--tones)
            tn.write_parquet(out_dir / f"mdna-tone-{run_id}-tones.parquet")
        ch = tone_changes(f, tn, pairs)
        log.info(
            "mdna-tone: %d filings, %d with an MD&A, %d with a tone change",
            f.height, tn.height, ch["d_lm_tone"].is_not_null().sum(),
        )  # fmt: skip
        grid = as_of_grid(
            ctx.calendar, ch["tradable_at"].min().date(), now.date(), fc.grid,
            fc.decision_lag_minutes,
        )  # fmt: skip
        ev = ctx.fan_out(
            ch.with_columns(
                pl.Series(
                    "as_of",
                    decision_times(ch["tradable_at"].to_list(), grid),
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
            .drop_nulls("as_of")
            .with_columns(
                pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
        )
        inp, _ = _eval_inputs(ctx, cfg, ev.select("ticker", "as_of").unique())
        rel = store.features(release_set)
        if rel.height == 0:
            raise typer.BadParameter(f"feature set {release_set} not found")
        ev = ev.join(
            rel.select("ticker", "as_of", pl.col("5d_llm_sent_cwmean").alias(control)),
            on=["ticker", "as_of"],
            how="left",
        )
        values = ["d_lm_tone", "lm_tone", "d_lm_neg_prop", "lm_neg_prop", "d_fb_tone", "fb_tone"]
        ev = ev.with_columns(pl.col(c).fill_nan(None) for c in [*values, control])
        stacked = pl.concat(
            [
                ev.with_columns(pl.lit("all").alias("cluster")),
                ev.with_columns(pl.col("form").alias("cluster")),
            ]
        ).select("ticker", "date", "cluster", *values, control)
        plan = [("primary", prim, "d_lm_tone", None)]
        plan += [("secondary", prim, v, None) for v in values[1:]]
        plan += [("secondary", prim, "d_lm_tone", control), ("secondary", sec, "d_lm_tone", None)]
        rows = []
        for tier, cells, value, ctl in plan:
            if stacked[value].is_null().all():
                continue  # e.g. FinBERT not run
            for r in cluster_ic(
                stacked, inp.targets, inp.sessions, cfg, cells, value=value, control=ctl
            ):
                rows.append({"tier": tier, "value": value, "control": ctl or "", **r})
        res = pl.DataFrame(rows)
        # descriptive daily view for the primary measure / cells, all filings
        session_idx = {d: i for i, d in enumerate(inp.sessions)}
        daily = []
        rng = np.random.default_rng(cfg.eval.seed + 1)
        for entry, h, adj in prim:
            t = inp.targets.filter((pl.col("entry") == entry) & (pl.col("horizon") == h))
            d = ev.join(t, on=["ticker", "date"]).filter(pl.col("d_lm_tone").is_not_null())
            dec = d["date"].to_numpy().astype("datetime64[D]")
            end = d["label_end"].to_numpy().astype("datetime64[D]")
            d = d.filter(pl.Series(pre_holdout_mask(dec, end, cfg.eval.holdout_start)))
            s = np.array([session_idx.get(x, -1) for x in d["date"].to_list()])
            daily.append(
                {"cell": f"{entry}/{h}/{adj}", **ic_detail(
                    s, d["d_lm_tone"].to_numpy().astype(float),
                    d[f"ret_{adj}"].to_numpy().astype(float),
                    block=h + 1, reps=cfg.eval.bootstrap_reps, rng=rng,
                )}
            )  # fmt: skip
        n_pre = ev.filter(
            (pl.col("date") < cfg.eval.holdout_start) & pl.col("d_lm_tone").is_not_null()
        )
        med = {k: ch.filter(pl.col("form") == k)["lm_tone"].median() for k in ("10-K", "10-Q")}
        lines = [
            f"# MD&A tone change {run_id}",
            "",
            f"- {f.height} 10-K/10-Q filings, {tn.height} with an MD&A, "
            f"{ch['d_lm_tone'].is_not_null().sum()} with a year-earlier pair that has one; "
            f"{n_pre.height} member-ticker events with a tone change before the holdout "
            f"({n_pre['ticker'].n_unique()} tickers).",
            "- LM tone = (pos - neg) / (pos + neg) of the MD&A (negation rule); `d_` = minus the "
            "year-earlier same-form filing's; `lm_neg_prop` = negative-word share (expected "
            "sign negative); FinBERT `fb_tone` = p_pos - p_neg over 400-word chunks (each chunk "
            "truncated at 512 tokens). Event dated to the first decision after tradable_at; "
            "pre-holdout; block bootstrap over sessions.",
            f"- median MD&A tone 10-K {_num(med['10-K'])}, 10-Q {_num(med['10-Q'])}; median MD&A "
            f"words {tn['mda_words'].median():.0f}. Control `{control}` = 8-K "
            f"`5d_llm_sent_cwmean` from feature set `{release_set}`.",
            "",
            "| tier | value | control | cell | group | n | IC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['tier']} | {r['value']} | {r['control'] or '-'} | "
                f"{r['entry']}/{r['horizon']}/{r['adjustment']} | {r['cluster']} | {r['n']} | "
                f"{_num(r['ic'])} | [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} |"
            )
        lines += [
            "",
            "Daily view of `d_lm_tone` (descriptive; sessions with >= 10 filings):",
            "",
            "| cell | days | daily IC | 95% CI | ICIR | Q5 - Q1 | 95% CI |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in daily:
            lines.append(
                f"| {d['cell']} | {d['days']} | {_num(d['ic_daily'])} | "
                f"[{_num(d['ic_daily_ci_lo'])}, {_num(d['ic_daily_ci_hi'])}] | "
                f"{_num(d['icir'])} | {bp(d['q_spread'])} | [{bp(d['q_spread_ci_lo'])}, "
                f"{bp(d['q_spread_ci_hi'])}] |"
            )
        path = _write_report(cfg, f"mdna-tone-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), filings=f.height, tones=tn.height)

    _run(
        "mdna-tone",
        cfg,
        {"primary": primary_cells, "secondary": secondary_cells, "release_set": release_set},
        body,
    )


def _record_experiment(ctx: Context, kind: str, key: str, run_id: str, params: dict) -> None:
    with ctx.store.ops() as con:
        con.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
            [kind, key, utcnow(), run_id, json.dumps(params, default=str)],
        )


@app.command("tone-market")
def tone_market_cmd(
    reps: Annotated[int, typer.Option(help="Bootstrap repetitions")] = 10_000,
    min_pairs: Annotated[int, typer.Option(help="Months with fewer pairs are dropped")] = 10,
    start: Annotated[str, typer.Option(help="First tone month")] = "2020-01-01",
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 12 (pre-registered 2026-11-18): does the month's same-company change in earnings-
    release tone predict next month's SPY return? Monthly Spearman correlation, block bootstrap
    over months, pre-holdout targets only."""
    from newsfeed.eval.aggregate import (
        corr_test,
        month_index,
        monthly_tone,
        monthly_total_returns,
        tone_market_table,
    )
    from newsfeed.eval.prices import price_panel

    _setup_logging(verbose)
    cfg = load_config(config)
    first = _parse_day(start).date()

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        if cfg.universe.all_extra_members():
            raise typer.BadParameter("use an S&P 500-only config (the pre-registered universe)")
        ev, grid = _llm_events(ctx, cfg, "edgar", ["earnings"], masked=False)
        # member on the release's decision date; one release per (filing, company)
        rel = (
            _dated_events(ctx, ev, grid)
            .unique(["item_id", "cik"])
            .select("item_id", "cik", "t_use", "sentiment")
        )
        tone = monthly_tone(rel, min_pairs=min_pairs).filter(pl.col("month") >= first)
        bars = price_panel(
            _price_caches(cfg)["tradier"],
            [cfg.eval.market_etf],
            first - timedelta(days=10),
            utcnow().date(),
        )
        tab = tone_market_table(tone, monthly_total_returns(bars), cfg.eval.holdout_start)
        months = month_index(tab["month"].to_list())
        rng = np.random.default_rng(cfg.eval.seed)
        plan = [("primary", "yoy", "spy_next", 3)]
        plan += [("secondary (a)", "yoy_3m", "spy_next3", 6)]
        plan += [
            ("secondary (b)", "yoy", "spy_same", 3),
            ("secondary (b)", "yoy_3m", "spy_same", 3),
        ]
        for m in ("level", "raw"):
            plan += [
                (f"secondary ({'c' if m == 'level' else 'd'})", m, "spy_next", 3),
                (f"secondary ({'c' if m == 'level' else 'd'})", f"{m}_3m", "spy_next3", 6),
                (f"secondary ({'c' if m == 'level' else 'd'})", m, "spy_same", 3),
            ]
        rows = []
        for tier, x, y, block in plan:
            r = corr_test(
                tab[x].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                tab[y].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                months,
                block=block,
                reps=reps,
                rng=rng,
            )
            rows.append({"tier": tier, "measure": x, "target": y, "block": block, **r})
            log.info("tone-market %s %s vs %s: %s", tier, x, y, r)
        res = pl.DataFrame(rows)
        used = tab.filter(pl.col("spy_next").is_not_null() & pl.col("yoy").is_not_null())
        lines = [
            f"# Aggregate earnings tone vs the market {run_id}",
            "",
            f"- releases: S&P 500 earnings 8-Ks (test 1 definition), {rel.height} in total; month "
            "= New York month of t_use. `yoy` = mean of (sentiment - same company's release "
            f"320-410 days earlier), months with < {min_pairs} pairs dropped; `level` = mean "
            "sentiment minus the company's mean over earlier releases; `raw` = mean sentiment; "
            "`_3m` = mean of months m-2..m.",
            "- targets: SPY total return in month m + 1 (`spy_next`), m + 1..m + 3 "
            "(`spy_next3`), month m (`spy_same`, contemporaneous, not tradable); only targets "
            f"ending before {cfg.eval.holdout_start}. Spearman correlation across months, "
            f"circular block bootstrap over months, {reps} reps.",
            f"- primary sample: {used.height} months, {used['month'].min()} .. "
            f"{used['month'].max()}; median pairs per month {used['n_pairs'].median():.0f}.",
            "",
            "| tier | measure | target | months | corr | 95% CI | p |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['tier']} | {r['measure']} | {r['target']} | {r['n']} | "
                f"{_num(r['corr'])} | [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} |"
            )
        out = cfg.paths.data_dir / "reports" / f"tone-market-{run_id}-months.parquet"
        tab.write_parquet(out)
        path = _write_report(cfg, f"tone-market-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), months=used.height, releases=rel.height)
        _record_experiment(
            ctx, "tone_market", f"tone-market:{run_id}", run_id,
            {"reps": reps, "min_pairs": min_pairs, "start": start},
        )  # fmt: skip

    _run("tone-market", cfg, {"reps": reps, "min_pairs": min_pairs, "start": start}, body)


@app.command("materiality-impact")
def materiality_impact_cmd(
    reps: Annotated[int, typer.Option(help="Bootstrap repetitions")] = 10_000,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 13 (pre-registered 2026-11-18): does Haiku's materiality label (low < medium < high)
    track the size of the price reaction to an 8-K? Partial rank correlation with
    |market-adjusted return close[i-1] -> close[i]| given trailing volatility and liquidity;
    secondary: realised volatility over the next 5 sessions. Pre-holdout."""
    from newsfeed.eval.aggregate import MATERIALITY_RANK
    from newsfeed.eval.report import partial_rank_corr

    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        if cfg.universe.all_extra_members():
            raise typer.BadParameter("use an S&P 500-only config (the pre-registered universe)")
        ev, grid = _llm_events(ctx, cfg, "edgar", [], masked=False)
        ev = ev.filter(pl.col("materiality").is_in(list(MATERIALITY_RANK)))
        e = _dated_events(ctx, ev, grid).with_columns(
            pl.col("materiality").replace_strict(MATERIALITY_RANK).cast(pl.Float64).alias("mat")
        )
        inp, _ = _eval_inputs(ctx, cfg, e.select("ticker", "as_of").unique())
        sess = inp.sessions
        prev = {sess[i]: sess[i - 1] for i in range(1, len(sess))}
        session_idx = {d: i for i, d in enumerate(sess)}
        hold = next((i for i, d in enumerate(sess) if d >= cfg.eval.holdout_start), len(sess))
        r1 = inp.targets.filter((pl.col("entry") == "close") & (pl.col("horizon") == 1)).select(
            "ticker", pl.col("date").alias("prev_date"), pl.col("ret_market").abs().alias("react")
        )
        d = (
            e.with_columns(pl.col("date").replace_strict(prev, default=None).alias("prev_date"))
            .join(r1, on=["ticker", "prev_date"], how="left")
            .join(inp.controls, on=["ticker", "date"], how="left")
            .join(inp.vol_targets, on=["ticker", "date"], how="left")
            .with_columns(
                pl.col("ctl_gap").abs().alias("abs_gap"),
                pl.col("ctl_ret_1d").abs().alias("abs_ret_1d"),
                (pl.col("event_type") == "earnings").alias("is_earn"),
            )
        )
        s = np.array([session_idx.get(x, -1) for x in d["date"].to_list()])
        x = d["mat"].to_numpy().astype(float)
        earn = d["is_earn"].to_numpy()
        rng = np.random.default_rng(cfg.eval.seed)

        def arr(col: str) -> np.ndarray:
            return d[col].cast(pl.Float64).fill_null(np.nan).to_numpy()

        react_ctl = np.column_stack([arr("ctl_vol_20d"), arr("ctl_log_dvol_20d")])
        vol_ctl = np.column_stack([arr(c) for c in VOL_CONTROLS])
        targets = [
            ("primary", "react", react_ctl, 2, (s >= 1) & (s < hold)),
            ("secondary", "rv_5", vol_ctl, 6, (s >= 0) & (s + 5 < hold)),
        ]
        rows = []
        for tier, tgt, ctl, block, pre in targets:
            y = arr(tgt)
            for subset, m in (
                ("all", pre),
                ("earnings", pre & earn),
                ("non-earnings", pre & ~earn),
            ):
                tier_here = tier if subset == "all" else "secondary"
                r = partial_rank_corr(x[m], y[m], ctl[m], s[m], block=block, reps=reps, rng=rng)
                rows.append({"tier": tier_here, "target": tgt, "subset": subset, **r})
                log.info("materiality-impact %s %s: %s", tgt, subset, r)
        res = pl.DataFrame(rows)
        pre_r = d.filter(pl.Series((s >= 1) & (s < hold)) & pl.col("react").is_not_null())
        desc = (
            pre_r.with_columns(
                pl.when(pl.col("is_earn")).then(pl.lit("earnings")).otherwise(pl.lit("other"))
                .alias("events")
            )
            .group_by("events", "materiality")
            .agg(
                pl.len().alias("n"),
                (pl.col("react").mean() * 1e4).alias("mean_bp"),
                (pl.col("react").median() * 1e4).alias("median_bp"),
                ((pl.col("react") > 0.05).mean()).alias("share_gt_5pct"),
            )
            .sort("events", pl.col("materiality").replace_strict(MATERIALITY_RANK))
        )  # fmt: skip
        lines = [
            f"# Materiality vs price impact {run_id}",
            "",
            f"- events: 8-K founders, all types, unmasked Haiku, relevance >= "
            f"{cfg.features.min_relevance}, materiality present; {d.height} member-ticker events, "
            f"{pre_r.height} with a pre-holdout reaction.",
            "- react = |market-adjusted total return close[i-1] -> close[i]|, i = first decision "
            "session; partial rank correlation of materiality rank (low 0 < medium 1 < high 2) "
            "given ctl_vol_20d, ctl_log_dvol_20d (block 2). rv_5 = sd of daily log returns over "
            f"i+1..i+5 given {', '.join(VOL_CONTROLS)} (block 6). {reps} reps.",
            "",
            "| tier | target | events | n | raw IC | partial IC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['tier']} | {r['target']} | {r['subset']} | {r['n']} | {_num(r['ic'])} | "
                f"{_num(r['partial_ic'])} | [{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | "
                f"{_num(r['p'], '.3f')} |"
            )
        lines += [
            "",
            "Size of the reaction by materiality (descriptive, pre-holdout):",
            "",
            "| events | materiality | n | mean abs. reaction | median | share > 5% |",
            "|---|---|---|---|---|---|",
        ]
        for r in desc.iter_rows(named=True):
            lines.append(
                f"| {r['events']} | {r['materiality']} | {r['n']} | {r['mean_bp']:.0f} bp | "
                f"{r['median_bp']:.0f} bp | {r['share_gt_5pct']:.1%} |"
            )
        out = cfg.paths.data_dir / "reports" / f"materiality-impact-{run_id}-levels.parquet"
        desc.write_parquet(out)
        path = _write_report(cfg, f"materiality-impact-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), events=d.height)
        _record_experiment(
            ctx, "materiality_impact", f"materiality-impact:{run_id}", run_id, {"reps": reps}
        )

    _run("materiality-impact", cfg, {"reps": reps}, body)


def _session_docs(ev: pl.DataFrame, grid: list[datetime], value: str) -> pl.DataFrame:
    """Documents (item_id, t_use, ``value``) -> date = New York date of the first decision
    >= t_use; ``value`` averaged over a document's rows (an article's company pairs)."""
    from newsfeed.eval.clusters import decision_times

    doc = ev.group_by("item_id").agg(pl.col("t_use").first(), pl.col(value).mean())
    as_of = decision_times(doc["t_use"].to_list(), grid)
    return (
        doc.with_columns(pl.Series("as_of", as_of, dtype=pl.Datetime("us", "UTC")))
        .drop_nulls("as_of")
        .select(
            pl.col("as_of").dt.convert_time_zone("America/New_York").dt.date().alias("date"),
            value,
        )
    )


@app.command("media-tone")
def media_tone_cmd(
    reps: Annotated[int, typer.Option(help="Bootstrap repetitions")] = 10_000,
    min_docs: Annotated[int, typer.Option(help="Sessions with fewer documents are dropped")] = 10,
    demean: Annotated[int, typer.Option(help="Tone minus the mean of N prior sessions")] = 20,
    describe: Annotated[bool, typer.Option(help="Document counts only (no prices)")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Test 14 (pre-registered 2026-11-22, after Tetlock 2007): does the day's market-wide news
    tone predict SPY's next-day return? Partial rank correlation across sessions given SPY's
    returns of the last five sessions, block bootstrap over sessions, pre-holdout targets only."""
    from newsfeed.eval.media import (
        CONTROLS,
        CONTROLS_TRADABLE,
        VOLUME_CONTROLS,
        market_daily,
        market_table,
        session_tone,
    )
    from newsfeed.eval.prices import price_panel
    from newsfeed.eval.report import partial_rank_corr

    _setup_logging(verbose)
    cfg = load_config(config)
    rx = cfg.features.exclude_title_regex
    params = {"reps": reps, "min_docs": min_docs, "demean": demean, "describe": describe}

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        if cfg.universe.all_extra_members():
            raise typer.BadParameter("use an S&P 500-only config (the pre-registered universe)")
        if not rx:
            raise typer.BadParameter("config has no features.exclude_title_regex (test 4b filter)")
        store = ctx.store
        news = store.query(
            "SELECT i.item_id, i.title, i.text_hash, i.tradable_at AS t_use "
            f"FROM {store.items_relation(['news_api'])} i "
            "JOIN item_story s ON s.item_id = i.item_id AND s.story_id = i.item_id"
        ).filter(~pl.col("title").str.contains("(?i)" + rx))
        keep = news.select("item_id")
        series: dict[str, pl.DataFrame] = {}
        grids: dict[str, list[datetime]] = {}
        for label, masked in (("news_masked", True), ("news_unmasked", False)):
            ev, grid = _llm_events(ctx, cfg, "news_api", [], masked=masked)
            grids["news"] = grid
            series[label] = _session_docs(ev.join(keep, on="item_id"), grid, "sentiment")
        lm_version = feature_sources(ctx).lm_version
        if lm_version is None:
            raise typer.BadParameter("no LM dictionary (features.lm_dictionary)")
        lm = store.query(
            "SELECT item_id, text_hash, neg_prop, score AS lm_tone FROM baselines "
            "WHERE method = 'lm' AND method_version = $v",
            {"v": lm_version},
        )
        lm_docs = news.join(lm, on=["item_id", "text_hash"])
        series["news_lm_neg"] = _session_docs(lm_docs, grids["news"], "neg_prop")
        series["news_lm_tone"] = _session_docs(lm_docs, grids["news"], "lm_tone")
        ev8, grid8 = _llm_events(ctx, cfg, "edgar", [], masked=False)
        series["8k_all"] = _session_docs(ev8, grid8, "sentiment")
        earn = ev8.filter(pl.col("event_type") == "earnings")
        series["8k_earnings"] = _session_docs(earn, grid8, "sentiment")

        hold = cfg.eval.holdout_start
        if describe:  # documents only: no prices are loaded
            for k, v in series.items():
                pre = v.filter(pl.col("date") < hold)
                per = pre.group_by("date").len()["len"]
                typer.echo(
                    f"{k}: {pre.height} pre-holdout documents, {per.len()} sessions "
                    f"({(per >= min_docs).sum()} with >= {min_docs}), median {per.median():.0f}"
                    f"/session, {pre['date'].min()} .. {pre['date'].max()}"
                )
            return
        first = min(s["date"].min() for s in series.values())
        bars = price_panel(
            _price_caches(cfg)["tradier"],
            [cfg.eval.market_etf],
            first - timedelta(days=60),
            utcnow().date(),
        ).filter(pl.col("symbol") == cfg.eval.market_etf)
        # holdout sessions never enter the statistics or the saved table
        mkt = market_table(market_daily(bars), hold).filter(pl.col("date") < hold)
        sessions = mkt["date"].to_list()
        sidx = np.arange(len(sessions))
        tones = {
            k: session_tone(v, sessions, value=v.columns[1], min_docs=min_docs, demean=demean)
            for k, v in series.items()
        }
        rng = np.random.default_rng(cfg.eval.seed)

        def col(frame: pl.DataFrame, name: str) -> np.ndarray:
            return frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

        def ctl(names: list[str]) -> np.ndarray:
            return np.column_stack([col(mkt, c) for c in names])

        # (tier, label, series, measure, transform, target, controls, block)
        plan: list[tuple[str, str, str, str, str, str, list[str], int]] = [
            ("primary", "Tetlock", "news_masked", "tone_dm", "", "fwd1", CONTROLS, 2),
            ("secondary (a)", "reversal days 2-5", "news_masked", "tone_dm", "", "rev",
             CONTROLS, 5),
            ("secondary (a)", "close[i] -> close[i+5]", "news_masked", "tone_dm", "", "fwd5",
             CONTROLS, 6),
            ("secondary (b)", "tradable controls", "news_masked", "tone_dm", "", "fwd1",
             CONTROLS_TRADABLE, 2),
            ("secondary (c)", "raw tone", "news_masked", "tone", "", "fwd1", CONTROLS, 2),
            ("secondary (c)", "unmasked", "news_unmasked", "tone_dm", "", "fwd1", CONTROLS, 2),
            ("secondary (d)", "LM negative share", "news_lm_neg", "tone_dm", "", "fwd1",
             CONTROLS, 2),
            ("secondary (d)", "LM tone", "news_lm_tone", "tone_dm", "", "fwd1", CONTROLS, 2),
            ("secondary (e)", "8-K all", "8k_all", "tone_dm", "", "fwd1", CONTROLS, 2),
            ("secondary (e)", "8-K all, days 2-5", "8k_all", "tone_dm", "", "rev", CONTROLS, 5),
            # earnings releases cluster in seasons: too few sessions for a 20-session demean
            ("secondary (e)", "8-K earnings", "8k_earnings", "tone", "", "fwd1", CONTROLS, 2),
            ("secondary (f)", "pessimism -> volume", "news_masked", "tone_dm", "neg", "vol_next",
             VOLUME_CONTROLS, 2),
            ("secondary (f)", "|tone| -> volume", "news_masked", "tone_dm", "abs", "vol_next",
             VOLUME_CONTROLS, 2),
        ]  # fmt: skip
        rows = []
        for tier, label, key, measure, tf, target, ctl_names, block in plan:
            x = col(tones[key], measure)
            x = -x if tf == "neg" else np.abs(x) if tf == "abs" else x
            r = partial_rank_corr(
                x, col(mkt, target), ctl(ctl_names), sidx, block=block, reps=reps, rng=rng
            )
            ok = ~(np.isnan(x) | np.isnan(col(mkt, target)))
            rows.append({
                "tier": tier, "label": label, "series": key, "measure": measure + tf,
                "target": target, "controls": ",".join(ctl_names), "block": block,
                "first": sessions[int(np.argmax(ok))] if ok.any() else None,
                "last": sessions[len(ok) - 1 - int(np.argmax(ok[::-1]))] if ok.any() else None,
                **r,
            })  # fmt: skip
            log.info("media-tone %s %s: %s", tier, label, r)
        res = pl.DataFrame(rows)
        prim = res.row(0, named=True)
        docs_per = tones["news_masked"].filter(pl.col("tone").is_not_null())["n"]
        lines = [
            f"# Media tone vs the market {run_id}",
            "",
            "- documents: Massive news story founders without test 4b's law-firm / market-"
            "research titles; article tone = mean Haiku sentiment of its company pairs with "
            f"relevance >= {cfg.features.min_relevance} (masked primary); session i = first "
            "15:50 decision >= t_use. 8-K rows: unmasked 8-K founders. LM rows: all filtered "
            "news founders.",
            f"- tone_dm = session mean minus the mean of the previous {demean} sessions (>= 15 "
            f"with a tone); sessions with < {min_docs} documents dropped. Median "
            f"{docs_per.median():.0f} news articles per kept session.",
            "- targets (SPY total return): fwd1 = close[i] -> close[i+1]; rev = close[i+1] -> "
            "close[i+5]; fwd5 = close[i] -> close[i+5]; vol_next = log volume[i+1] vs its mean "
            f"over i-19..i. Only sessions and targets ending before {hold}.",
            "- statistic: partial rank correlation given the controls (r0 = SPY return of "
            "session i, r1..r5 earlier sessions, gap = today's open vs yesterday's close, "
            f"vol_today), circular block bootstrap over sessions, {reps} reps, two-sided p.",
            f"- primary: {prim['n']} sessions, {prim['first']} .. {prim['last']}.",
            "",
            "| tier | test | series | measure | target | controls | n | raw corr | partial | "
            "95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res.iter_rows(named=True):
            lines.append(
                f"| {r['tier']} | {r['label']} | {r['series']} | {r['measure']} | {r['target']} "
                f"| {r['controls']} | {r['n']} | {_num(r['ic'])} | {_num(r['partial_ic'])} | "
                f"[{_num(r['ci_lo'])}, {_num(r['ci_hi'])}] | {_num(r['p'], '.3f')} |"
            )
        out = cfg.paths.data_dir / "reports" / f"media-tone-{run_id}-sessions.parquet"
        frame = mkt
        for k, t in tones.items():
            frame = frame.join(
                t.select("date", pl.col("n").alias(f"{k}_n"), pl.col("tone").alias(f"{k}_tone"),
                         pl.col("tone_dm").alias(f"{k}_dm")),
                on="date", how="left",
            )  # fmt: skip
        frame.write_parquet(out)
        path = _write_report(cfg, f"media-tone-{run_id}", "\n".join(lines) + "\n", res)
        stats.update(report=str(path), sessions=prim["n"])
        _record_experiment(ctx, "media_tone", f"media-tone:{run_id}", run_id, params)

    _run("media-tone", cfg, params, body)


if __name__ == "__main__":
    app()
