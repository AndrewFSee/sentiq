# CLAUDE.md — newsfeed

Research pipeline: financial news/filings → point-in-time (PIT) correct, LLM-scored features →
test for *incremental* tradable signal. **Correctness (no look-ahead, reproducibility) beats
speed and polish.** It is not a trading system.

## Working agreement
- Build one milestone at a time, run `ruff` + `pytest` after each, then stop for review.
- Ask before decisions that change research semantics (timing rules, universe, targets).
- Status: **M1 done** (scaffold, config, storage, EDGAR 8-K). **M2 done** (collect loop +
  heartbeat/outage detection, news adapter on fixtures, ticker linking, dedupe/stories,
  multi-process safety). **M3 done** (LLM scoring: structured outputs, cache, batches, cost,
  masking/replicate/validation studies, re-parse). Next: M4 features/baselines → M5 eval.
- Every command that calls the model costs money: run `score --dry-run` first; jobs estimated
  above `scoring.confirm_above_usd` need `--yes`. Never run paid jobs without the user's OK.

## Commands
```
uv sync                                   # install (Python >=3.11)
uv run pytest                             # no network: pytest-socket --disable-socket
uv run ruff check . && uv run ruff format --check .
uv run newsfeed backfill --start 2025-01-01 --end 2026-01-01 [--tickers AAPL,MSFT] [--source news_api]
uv run newsfeed collect                   # loop; --once for cron; --check = health exit code
uv run newsfeed derive [--full]           # rebuild ticker links + stories (normally automatic)
uv run newsfeed inspect
uv run newsfeed reparse                   # re-parse stored EDGAR items after a parser fix
uv run newsfeed score --dry-run           # plan + cost estimate (count_tokens is free)
uv run newsfeed score [--mode batch --wait] [--start/--end] [--limit N]
uv run newsfeed score --masked --sample 200        # leakage study
uv run newsfeed score --replicates 2 --sample 100  # noise floor
uv run newsfeed score --model claude-sonnet-5-5 --sample 100   # validation
uv run newsfeed score-batches [--wait]    # collect submitted batches
uv run newsfeed agreement --compare masked|model|replicates
uv run newsfeed cost                      # spend per run from the usage ledger
uv run python scripts/record_edgar_fixtures.py   # re-record test fixtures (needs network)
```

## Point-in-time rules (most important)
- All timestamps are tz-aware UTC. Naive datetimes are rejected (`timeutil.ensure_utc`,
  pydantic validators, ruff `DTZ` rules). Parquet columns are `timestamp[us, tz=UTC]`.
- Every item stores `published_at` (source's time), `first_seen_at` (when *we* saw it; always
  truthful, even for backfill), `ingest_mode` (live|backfill), `tradable_at`, `tradable_rule`.
- `tradable_at` (`timeutil.compute_tradable_at`, applied only in `ingest.runner.build_row`):
  - live: `max(published_at, first_seen_at)`
  - backfill: `published_at + timing.backfill_lag_seconds[source]` (user decision, M1)
  - then pushed to the next XNYS regular-session open if outside [open, close). Half days and
    holidays come from `exchange_calendars`. Close is exclusive, open inclusive.
- Anything "as of T" reads items **only** via `Store.items_as_of(T)` (`tradable_at <= T`) or
  `dedupe.stories_as_of(T)`. Do not query the `items` view directly in feature/eval code.
  Causality tests in `tests/test_storage.py` probe every tradable_at boundary.
- `first_seen_at` is stamped *after* the listing response arrives, never before (tests advance
  the clock during the request to enforce this).
- On duplicate item_ids (only possible via a race) the view keeps the **live** row, then the
  earliest `first_seen_at`.
- Story aggregates (n_outlets, first/last seen) are computed as of T from members with
  `tradable_at <= T`. Never store or use a precomputed count.

## EDGAR facts (verified against live SEC data, 2026-10)
- Authoritative acceptance time = `<ACCEPTANCE-DATETIME>` in `<accession>.hdr.sgml`, which is
  **Eastern wall clock**. Convert with `zoneinfo("America/New_York")`, never a fixed offset.
- `data.sec.gov/submissions` `acceptanceDateTime` (ends in `Z`) is **unreliable and
  filer-dependent**: correct UTC for e.g. MSFT/NVDA/GOOGL/WMT/JNJ/XOM, but *true UTC + ET offset*
  (offset applied twice; 4h late in EDT, 5h in EST) for e.g. AAPL/AMZN/BRK/JPM/META. Read
  literally it is always ≥ the true time, so it is only used as an upper bound for coarse
  filtering (≥1-day margins). Each item records `meta.json_convention`; a value other than
  `utc`/`double_offset`/`feed` logs a warning and must be investigated.
- The "latest filings" Atom feed (`browse-edgar?action=getcurrent`) gives `<updated>` with an
  explicit offset, equal to the header acceptance time. Newest first; co-registrants appear
  once per filer CIK.
- `filingDate` is never used for timing: filings accepted after 17:30 ET get the next
  business day's filingDate.
- 8-Ks accepted after 17:30 ET are disseminated the next business day (the live feed shows
  nothing after 17:30); the session push (to 09:30 at the earliest) keeps us conservative.
- SEC fair access: ≤10 req/s across sec.gov hosts (we default to 8), descriptive
  `SEC_USER_AGENT` with contact email (from `.env`). 403 = blocked → check UA / rate.
- Submissions JSON pages older history in `filings.files[]`; fetched only when needed.
- Co-registrant filings appear under several CIKs → one item per accession, tickers unioned.
- `company_tickers.json` is a *current* snapshot (not PIT). Items are labelled with today's
  ticker(s) for the CIK; delisted names are missing (survivorship bias in backfills).

## Dedupe / stories (`dedupe.py`)
- Order: URL match (any distance) → exact content hash (within `window_hours`) → MinHash
  near-dup (word 5-shingles, Jaccard ≥ `near_dup_threshold`, within window) → new story.
  Ticker overlap required when both items have linked tickers.
- Causal assignment: items processed in (tradable_at, published_at, item_id) order; each joins
  ONE earlier item's story or founds its own (`story_id` = founder's item_id). **No retroactive
  merges**: a later bridging article must not regroup earlier ones. Founder = first appearance
  = the item to score once.
- Incremental rebuilds recompute from the earliest unassigned item (an out-of-order backfill
  recomputes everything after it) and rewrite only affected month files. Config or version
  change → full rebuild (`config_hash` on each row).
- Content hash uses the body, or title + body when the body is < 200 chars (headline-only).

## Ticker linking (`tickers.TickerLinker`)
- Only universe tickers. Methods/confidence: source tags 1.0 (vendor tags, EDGAR filer CIK),
  exchange tag "(NASDAQ: X)" 0.95, cashtag 0.9, company-name alias in title 0.8 / body 0.6.
- Names are case-sensitive and word-bounded; config `universe.aliases` + cleaned SEC name. URLs
  and "on Facebook/LinkedIn…" boilerplate are stripped before name matching.
- A name mention links all share classes of the CIK (GOOG + GOOGL); `cik` is stored so M3 can
  score per entity.
- `linker_id` = LINKER_VERSION + hash(universe, aliases, titles); a change triggers a relink.
  The `item_tickers` view keeps the newest batch per item; a NULL-ticker row marks "linked,
  nothing found".

## Live collection (`collect.py`)
- Per cycle: poll each source → ingest → derive (under `state/derive.lock`) → health →
  heartbeat. State in `data/state/collect-<source>.json` makes `--once` (cron) equivalent to
  one loop tick; `--check` evaluates health from state only (exit 1 if unhealthy).
- EDGAR live poll = the Atom feed (all companies, ~12 8-Ks/hour in market hours). Catch-up every
  `catch_up_every_minutes` = feed + per-CIK submissions sweep (recovers anything beyond the
  feed's paging window). Catch-up items are still `live` (we genuinely see them only now).
- Outage = no *market-wide* feed activity (`adapter.last_activity_at`, any company) for
  `silence_alert_minutes` of **regular-session time**: a quiet universe is normal, a quiet feed
  is not. Also `poll_failing` (N consecutive failures) and `stalled` (no successful poll for N
  intervals of market time). Alerts → log WARNING + `alerts` table, re-raised at most every
  `realert_minutes`, cleared on recovery.
- News: vendor-agnostic `NewsVendor` protocol; `FixtureVendor` until a vendor is licensed.
  Live polls re-request `poll_overlap_minutes` before the watermark (vendors back-date and
  index late); the watermark comes from storage, so cron runs resume without local state.

## LLM scoring (`scoring/`)
- Unit = (story founder, company): founders only (score once, on first appearance); links per
  `scoring.link_policy`; one canonical ticker per CIK (first in `universe.tickers` order), so
  GOOG/GOOGL is scored once and features fan out to both.
- Request: static system prompt from `scoring/prompts/<version>.md` with `cache_control`
  (≥512 tokens, so it caches); user turn = target + doc type + source + title + body (no dates).
  `output_config.format` JSON schema (structured outputs), `effort` from config, adaptive
  thinking. **No temperature/top_p/top_k** (Haiku 5.5 / Sonnet 5.5 reject non-defaults) and
  **no forced tool_choice** (Sonnet 5.5 rejects it). Read response blocks by type.
- Ranges and the 25-word rationale are validated client-side; out-of-range = `invalid`, never
  clamped. Statuses: ok | invalid | refusal | max_tokens | empty are final (cached);
  API errors and expired/errored batch requests are not stored, so they are retried.
- Cache key = sha256(item_id, ticker, model, prompt_hash, masked, replicate, text_hash,
  mask_id). `prompt_hash` covers prompt text + schema + effort/thinking/max_tokens (not the
  model). `text_hash` = items view hash of title+body, so a re-parsed item is re-scored.
  The `scores` view keeps one row per key (first ok draw); `scores_all` keeps every attempt.
- Never edit a prompt version in place once used; copy to a new version. Each new prompt_hash
  is recorded in the `experiments` table (multiple-testing accounting).
- Ledger `llm_usage` has one row per call incl. failures; `cost_usd` uses `pricing` in config
  (Haiku 5.5 switches rate card for prompts >100K tokens; batch discount stacks with cache).
- Batches: submission is recorded right after the API accepts it; uncollected requests are
  excluded from planning; collection is idempotent (re-collect may double-count the ledger).
- Studies only cover documents with an ok primary score on the same text; pair on
  (item_id, ticker, text_hash). Judge masked/model gaps against the replicate noise floor.
- Masking replaces universe company names, tickers, products, people and places (config
  `universe.products/people/places`) with [COMPANY_A]/[TICKER_A]/[PRODUCT_n]…; the target is
  always A. Non-universe companies and dates stay visible. `mask_residuals` counts leftovers.
- Timing caveat for M4: a score is a deterministic function of text available at `tradable_at`,
  but in live use it only exists at `scored_at`; features for the live/holdout period should use
  max(tradable_at, scored_at).

## Re-parsing
- Items are append-only; `reparse` writes a new row per item with the current parser, copying
  `published_at`, `published_at_raw`, `first_seen_at`, `ingest_mode`, `tradable_at`,
  `tradable_rule` **verbatim** (never recomputed). The view prefers the newer parser_version
  among rows with identical timing. Links follow `text_hash`; stories are rebuilt in full.

## Storage conventions
- `data/lake/items/src=<source>/ym=<YYYY-MM>/part-<run_id>-<seq>-<rand>.parquet`, append-only,
  written to a temp name then atomically renamed. DuckDB view `items` reads them.
- Derived Parquet: `lake/item_tickers/` (append-only parts), `lake/item_story/ym=*/stories.parquet`
  (month files replaced atomically). All are rebuildable from `items`.
- `data/raw/<source>/<native_id>/<file>.gz`: verbatim source documents. Adapters read through
  this cache, so re-parsing never needs to re-download. `raw_json` on each row holds source
  metadata (submissions row or feed entry, SGML header, index listing; vendor record).
- `data/newsfeed.duckdb`: operational tables (`runs`, `failures`, `alerts`), opened per
  operation with lock retry. Parquet is read via an in-memory DuckDB (no file lock), so a
  `collect` loop and a `backfill` can run at the same time (verified live).
- Windows: `os.replace` onto a file another process is reading can fail; `write_table_atomic`
  retries.
- `item_id = sha256("<source>:<native_id>")[:32]`, deterministic.
- Every run records `run_id`, `config_hash` (non-secret, path-independent), git SHA, versions.
- Bump `PARSER_VERSION` in an adapter when its parsing output changes.

## Code conventions
- Python 3.11+, type hints everywhere, pydantic models with `extra="forbid"`, config-driven
  (`config/default.yaml`). Secrets only in `.env` (gitignored), see `.env.example`.
- Source adapters implement `ingest.base.SourceAdapter` (`backfill`, `poll(catch_up=)`,
  `last_activity_at`), produce `RawItem`, and never compute `tradable_at` themselves.
- HTTP goes through `ingest.http.HttpClient` (rate limiter + tenacity retries on 429/5xx/transport).
- Tests use recorded fixtures under `tests/fixtures/` replayed via `httpx.MockTransport`; no
  network. Clocks, sleeps and rate limiters are injectable for determinism.
- Ruff line length 100. Shell heredocs mangle escapes: a `\u` escape became a literal character,
  and a regex `\b` became a backspace byte that silently broke matching. Use the Edit/Write tools
  for regexes, `\N{NAME}` for non-ASCII, and keep `tests/test_hygiene.py` passing.

## Decisions log
- 2026-10-08: backfill `tradable_at` = published + per-source lag (EDGAR 60s, news 300s).
- 2026-10-08: no news vendor yet → news adapter built against fixtures (`tests/fixtures/news`).
- 2026-10-08: price data for M5 = **Tradier** (token via `TRADIER_TOKEN`).
- 2026-10-08 (M2): outage detection uses market-wide feed activity, not universe items.
- 2026-10-08 (M2): stories never merge retroactively (PIT); n_outlets is computed as of T.
- 2026-10-09 (M3): scoring model `claude-haiku-5-5` (effort low, adaptive thinking),
  validation `claude-sonnet-5-5`.
- 2026-10-09 (M3): links scored = source/exchange/cashtag/title + body-name with ≥2 mentions
  (revisit after looking at relevance labels).
- 2026-10-09 (M3): structured outputs (`output_config.format`) instead of forced tool use.
- 2026-10-09 (M3): sampled scores (temperature 0 unavailable) + replicate noise study.
- 2026-10-09 (M3): masking covers names, tickers, products, people, places (not dates).
- 2026-10-09 (M3): EDGAR parser v2 fixes table-of-contents filings (20/252 items had truncated
  or TOC-only bodies); items re-parsed with timing preserved.
