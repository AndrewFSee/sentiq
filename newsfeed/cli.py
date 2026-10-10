"""Command-line entry point: ``newsfeed <command>``."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from newsfeed import coverage
from newsfeed.collect import Collector, check_states
from newsfeed.config import AppConfig, load_config, load_secrets
from newsfeed.derive import derive as run_derive
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
from newsfeed.scoring.runner import ScoringJob, estimate_cost, plan_targets
from newsfeed.storage import Store
from newsfeed.tickers import TickerLinker, TickerMap, Universe, load_ticker_map
from newsfeed.timeutil import MarketCalendar, get_calendar, utcnow

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("newsfeed")

SOURCES = ("edgar", "news_api")
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
        subset = [t.upper() for t in _split(tickers)]
        unknown = set(subset) - set(cfg.universe.tickers)
        if unknown:
            raise typer.BadParameter(f"not in configured universe: {sorted(unknown)}")
        self.ingest_tickers = subset or cfg.universe.tickers

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

    def universe(self, full: bool = False) -> Universe:
        return Universe.build(self.cfg.universe.tickers if full else self.ingest_tickers, self.tmap)

    def linker(self) -> TickerLinker:
        u = self.universe(full=True)  # linking always uses the whole configured universe
        titles = {c: self.tmap.cik_to_titles.get(c, "") for c in u.ciks}
        return TickerLinker(u, self.cfg.universe.aliases, titles)

    def adapter(self, source: str) -> SourceAdapter:
        if source == "edgar":
            return EdgarAdapter(
                self.cfg.sources.edgar, self.http(), self.universe(), self.store.raw
            )
        if source == "news_api":
            ncfg = self.cfg.sources.news_api

            def watermark() -> datetime | None:
                df = self.store.query(
                    "SELECT max(published_at) AS m FROM items WHERE source = 'news_api'"
                )
                return df["m"][0]

            return NewsApiAdapter(ncfg, make_vendor(ncfg), self.universe(), watermark=watermark)
        raise typer.BadParameter(f"unknown source {source!r}; choose from {SOURCES}")

    def derive_fn(self, run_id: str, full: bool = False) -> Callable[[], dict[str, Any]]:
        linker = self.linker()

        def fn() -> dict[str, Any]:
            out = run_derive(self.store, self.cfg, linker, run_id, full_stories=full)
            if self.cfg.scoring.live_in_collect and not out.get("skipped"):
                out["scoring"] = self.score_recent(run_id)
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
        titles = {c: self.tmap.cik_to_titles.get(c, "") for c in u.ciks}
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
        targets = plan_targets(job, self.universe(full=True).cik_to_tickers, since_tradable=since)
        return job.run_live(targets).as_dict() if targets else {"calls": 0}


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
        covered = sorted({t for ts in adapter.universe.cik_to_tickers.values() for t in ts})
        lag = timedelta(seconds=cfg.timing.lag_for(source))
        coverage.record_backfill(
            ctx.store, source, covered, t0, t1, lag=lag, calendar=ctx.calendar, run_id=run_id
        )
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
            cfg, ctx.store, adapters, ctx.calendar, run_id, derive_fn=ctx.derive_fn(run_id)
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
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Rebuild derived tables (ticker links, stories). Normally runs automatically."""
    _setup_logging(verbose)
    cfg = load_config(config)

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        stats["derive"] = ctx.derive_fn(run_id, full=full)()

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
    source: Annotated[str, typer.Option(help="Only edgar supports re-parsing")] = "edgar",
    everything: Annotated[bool, typer.Option("--all", help="Also items already current")] = False,
    config: ConfigOpt = None,
    verbose: bool = False,
) -> None:
    """Re-parse stored items with the current parser (from the raw cache; timing preserved),
    then rebuild links and stories. Scores of changed items are re-planned automatically."""
    _setup_logging(verbose)
    cfg = load_config(config)
    if source != "edgar":
        raise typer.BadParameter("only --source edgar supports re-parsing")

    def body(ctx: Context, run_id: str, stats: dict[str, Any]) -> None:
        adapter = ctx.adapter("edgar")
        stats["reparse"] = reparse_source(
            ctx.store,
            "edgar",
            adapter,
            EDGAR_PARSER_VERSION,
            calendar=ctx.calendar,
            run_id=run_id,
            everything=everything,
        )
        stats["failed"] = len(stats["reparse"]["failed"])
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
    seed: int = 1,
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
        set_id = feature_set_id(cfg, src)
        c2t = ctx.universe(full=True).cik_to_tickers
        df = build_features(
            ctx.store, cfg, grid, c2t, src, tickers=[t.upper() for t in _split(tickers)] or None
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


@app.command("coverage")
def coverage_cmd(
    rebuild_from_runs: Annotated[
        bool, typer.Option("--rebuild-from-runs", help="Recreate backfill coverage from runs")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Show collected coverage (tradable-time ranges) per source and ticker."""
    cfg = load_config(config)
    with Store(cfg.paths) as store:
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


def _later(milestone: int) -> Callable[[], None]:
    def cmd() -> None:
        typer.echo(f"Not implemented yet (Milestone {milestone}).")
        raise typer.Exit(2)

    return cmd


app.command("eval", help="Evaluation (Milestone 5)")(_later(5))


if __name__ == "__main__":
    app()
