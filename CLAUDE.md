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
  masking/replicate/validation studies, re-parse). **M4 done** (coverage tracking, LM/FinBERT
  baselines, embeddings + novelty, point-in-time features with frozen-store causality tests).
  **M5 done** (Tradier prices, total-return targets over an entry x horizon x adjustment grid,
  PIT controls, purged CV, frozen holdout, rank IC + incremental lift, bootstrap + BH).
  **M6 done** (branch `m6-sp500`): point-in-time S&P 500 universe 2019–present, per-CIK
  backfill windows, price routing (Tradier / renamed / Tiingo for delisted), split-consistent
  market caps, PIT sector ETFs, member-only feature rows, Massive vendor news, masked-score
  leakage checks, pre-registered tests 1–9 (incl. earnings-call transcripts, partial IC vs the
  release, delay sensitivity, LLM surprise, attention → volatility, 8-K item event study,
  daily IC / quintile view) + exploratory per-type and text-cluster breakdowns (results in
  the decisions log). Ongoing: live collection of 8-Ks and transcripts
  (`scripts\collect_window.cmd`) for the holdout; `transcripts-latency` once the Q3-2026
  season has data; the single pre-registered holdout look (H1 earnings, H2 guidance, ≥ 10,000
  bootstrap reps) is on/after 2027-08-31.
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
uv run newsfeed coverage [--rebuild-from-runs]   # collected ranges per source/ticker
uv run newsfeed features --start 2025-01-01 --end 2026-10-01   # build + store a feature set
uv run newsfeed eval [--feature-set ID]   # pre-holdout grid report -> data/reports/
uv run newsfeed eval --use-holdout        # FINAL test on the frozen holdout (each look counts)
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

## Coverage (`coverage.py`)
- "No rows" must not be read as "no news": the ops `coverage` table records what was collected.
  Backfills store their published-time range; merged per ticker on that axis, a range [S, E)
  covers tradable time [push(S + lag), E). Live polls extend [previous ok poll, now]; stalls
  leave gaps. `backfill` records coverage automatically; `coverage --rebuild-from-runs`
  recreates it from the runs log (assumes the configured universe hasn't changed).
- Feature windows (lo, hi] outside coverage of every `features.coverage_sources` are null;
  volume_z needs its whole baseline covered.

## Features (`features/`)
- `build_features(store, cfg, as_of_times, …)` → one row per (ticker, as_of), columns
  `<window>_<feature>` for windows `features.windows_hours` (1h, 1d, 5d = one calendar week),
  each over (T − w, T]. Grid: each session close − `decision_lag_minutes` (or open + lag).
- Events: volume from all qualified links (same link policy as scoring) fanned out to all share
  classes; LLM/LM/FinBERT from story founders. Availability: items at `tradable_at`; LLM
  scores at `tradable_at` (backfill) or max(tradable_at, scored_at) (live); baselines instant.
- LLM sentiment aggregates skip relevance < `features.min_relevance`; `llm_coverage` = share of
  founder events in the window with a score (refusals etc. show up here).
- Loads via `items_as_of(T_max)` then slices per T. Tests: a frozen-store test rebuilds the
  whole pipeline from items tradable by T and requires identical features, with the full store
  built over a multi-T grid (so per-T slicing, not the loader, must hide the future); boundary
  tests at T and T − 1µs; future-perturbation invariance; live scored_at. Mutations (1h
  look-ahead, ignoring scored_at) are caught — keep it that way.
- Fixed dtypes (counts Int64, else Float64) so all-null columns don't change the schema.
- `feature_set_id` = hash(FEATURE_VERSION, features config, link policy, LLM model + prompt hash,
  LM version, FinBERT model); every built set is recorded in `experiments`.
- Baselines (`features/baselines.py`): LM tone (pos − neg)/(pos + neg) with negation rule,
  removed words excluded; current dictionary for all dates (small look-ahead for words added
  after 2009 — irrelevant for 2024+ data). FinBERT optional (`features.finbert: true`,
  `uv sync --extra finbert`). Both document-level, computed in `derive` for story founders,
  keyed by (item_id, text_hash, method, method_version). LM CSV lives in `data/ref/lm/`
  (free for academic use, not committed); tests use a 34-word excerpt fixture.

## Embeddings (`features/embeddings.py`)
- fastembed BAAI/bge-small-en-v1.5 (local ONNX, cached in `data/ref/models`), vectors for
  story founders in `derive`. Novelty = 1 − max cosine sim to the ticker's strictly earlier
  stories within 30 days (null if the look-back isn't covered). Tests use `HashEmbedder`.

## Evaluation (`eval/`)
- Prices (`eval/prices.py`): Tradier daily bars are split-adjusted, NOT dividend-adjusted;
  dividends endpoint amounts are split-adjusted too → total returns add dividends on ex-dates.
  Intraday history is only 20–40 sessions, so only daily horizons are testable. Cache in
  `data/ref/prices/`. Token env var: TRADIER_TOKEN or TRADIER_API_TOKEN.
- Targets (`eval/targets.py`): decision 15:50 ET (= feature as-of = NYSE MOC cutoff). Entry
  `close`: close[i] → close[i+h]; `next_open`: open[i+1] → open[i+1+h]. Dividends count if held
  over the ex-date. Adjusted = minus SPY / sector ETF over the same window. `label_end` = exit.
- Controls use only closes/volumes through i−1 plus today's open (gap); intraday move to 15:50
  is not observable with daily bars (not a leak; just an uncontrolled factor).
- Size buckets: PIT market cap = SEC shares (filed date + 1 day) × previous close; terciles
  within the universe; Berkshire = unknown (custom share tag).
- CV (`eval/cv.py`): walk-forward (train labels end before the test block) or blocked with
  purge + embargo. Holdout: rows whose decision or label end ≥ `eval.holdout_start` are dropped
  unless `--use-holdout`; holdout looks are counted in `experiments` and warned about.
- Report (`eval/report.py`): univariate pooled rank IC (+ by size bucket) and OOS ridge/LGBM
  on M0 controls / M1 + non-LLM news / M2 + LLM; lift = IC(M2) − IC(M1). Inference: circular
  block bootstrap over *calendar* sessions with block = label length (h+1 close, h+2 open);
  BH at `fdr_q` across all tests of the run. Every eval run is an `experiments` row.
- Bootstrap gotcha (fixed 2026-10-10): blocks over *sampled* news days + non-circular blocks
  made 20-day CIs narrower than the naive SE and produced 5 spurious "discoveries" in the first
  run. `test_block_bootstrap_matches_analytic_se_for_overlapping_labels` guards it.
- Placebo check (shuffled targets): OOS IC centres at −0.01..−0.03 (sd 0.07–0.10), so small
  negative M0 ICs are noise; lifts are differences and cancel this bias.

## Universe (M6, `universe_pit.py`, `universe.mode: sp500_pit`)
- Membership: github fja05680/sp500 (MIT) pinned at `universe.fja_sha` (daily ticker lists,
  last snapshot 2026-08-18); after that, monthly Wikipedia snapshots. CIK, name and GICS sector
  come from monthly revisions of Wikipedia's constituents table (`data/ref/universe/`).
  `newsfeed universe --build` refetches; `newsfeed universe` summarises.
- Spells: (ticker, cik, start, end exclusive). A ticker's CIK on day x = latest Wikipedia
  snapshot ≤ x, else its first snapshot. 18 tickers changed CIK (reorgs, e.g. Disney 2019).
- Ingestion: per-CIK windows = the *ticker's* whole membership window (all its CIKs) widened
  by `ingest_margin_before/after_days` (90/30) — immune to Wikipedia lagging a CIK change.
  Items outside a CIK's window are skipped at backfill; coverage is recorded per window.
- Linking: `name_linking: aliases_only` in PIT mode (filer CIK + aliases; no SEC-name body
  matching across 650 names). Features: rows only while the ticker is a member on the
  decision date (`build_features(member_spans=...)`).
- `PitUniverse.as_universe()` maps each CIK to *all* tickers it ever had (FB and META); the
  membership filter decides which ticker carries a row on a given day.

## Prices (M6, `eval/prices.py`)
- `route_prices`: SEC's current map still gives the ticker to one of its CIKs → Tradier
  (`current`); else a current ticker of the same CIK → Tradier under that symbol (`renamed`:
  FB→META, BK→BNY, EQR→VMRK); else Tiingo under the old ticker (`delisted`). A ticker reused
  by an unrelated company is never fetched from Tradier under that ticker. Each route covers
  only its membership window (+45 days); `build_panel` falls back to Tiingo when Tradier has
  < 50% of the window's sessions (e.g. DISCA→WBD: WBD history starts 2022-04).
- Delisted: after the last real bar, up to 25 synthetic sessions at the last close (cash).
  Caveats: bankruptcies' final recovery may be below the last exchange trade; stock-for-stock
  deals ignore the acquirer's returns after the close. Non-current series that don't reach
  back to their window start are dropped (unpriced), not padded.
- Vendor facts (verified 2026-10-10): Tradier has deterministic gaps for some *listed* names
  (NOC none before 2020-08-24, CSX missing 2024-06..2026-06, BEN/TDG/ROK/DG/AIG start late)
  → any Tradier series that starts late or is < 98% dense is replaced by Tiingo's if better.
  Tiingo files a security under its *last* ticker (FRC→FRCB, SIVB→SIVBQ, CDAY→DAY,
  COG→CTRA, DISCA→WBD, HCP/PEAK→DOC); same-CIK tickers and Tiingo name search are tried
  automatically, the rest are `eval.price_overrides` (each verified against a known price).
- Unpriced (no free source): APC, NFX, SCG, STI, INFO, VIAC (2019-12..2022-02), WRK, and
  FOX/FOXA before 2019-03-12 (then 21st Century Fox) — ~13 of ~3700 member ticker-years.
  `scripts/prefetch_prices.py` warms both caches and writes `data/reports/prices-prefetch.csv`.
- Tiingo bars are unadjusted + `divCash`/`splitFactor`; converted to Tradier's convention
  (split-adjusted bars and dividends, no dividend adjustment). Free tier: 50 req/hour, 1000/day,
  500 symbols/month → `eval.tiingo_requests_per_hour: 48`. Cache: `data/ref/prices_tiingo/`.
- Splits: Tradier `/beta/markets/fundamentals/corporate_actions`; Tiingo `splitFactor`.
  Market cap = adj prev close × reported shares × prod(split factors with ex_date ≥
  available_from). Before this fix (M5) pre-split caps were understated by the split ratio.
- Each eval writes `eval-<run>-prices.csv` (source, reason, coverage, last real bar, padding).
- Sector ETF per (ticker, date) from the monthly GICS snapshots (`gics_sector_etfs`; the first
  snapshot also covers earlier dates). Static mode keeps `eval.sector_etfs`.

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
- Archive (moved off C: 2026-10-14, see `D:\sentiq-archive\README.md`): the defeatbeta pin
  860d42ac253f.parquet (transcripts backfill version), full SEC companyfacts JSON (the
  project keeps slim copies with only the three share-count tags; verified identical share
  histories), the fastembed model, M6 run logs. The v1 transcript backup and `data/configs/`
  (copies of `config/variants/`) were deleted. `data/raw` stays: re-parsing and live
  collection use it.
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
- 2026-10-10 (M3 studies, n = 150/200/60): re-draw noise floor sentiment r 0.989 (MAE 0.020);
  masked vs unmasked r 0.939 (MAE 0.053, mean diff 0.000); Sonnet vs Haiku r 0.915 (Haiku
  +0.04 more positive). Novelty is the noisiest field (kappa 0.69 re-draw, 0.18 vs Sonnet).
- 2026-10-10 (M5): grid = entries {close, next_open} × horizons {1,2,5,10,20} sessions ×
  {market, sector}-adjusted; holdout from 2026-07-01 (never evaluated yet); sample = news days.
- 2026-10-10 (M5, eval 2b0709, pre-holdout, ~175 OOS rows/cell, ~120 decision days): no
  univariate IC or LLM lift survives BH (q 0.10) out of 200 tests. LLM sentiment IC is negative
  at 1–2 day horizons (≈ −0.2, reversal after the move) but not significant. Power is the
  binding constraint: 10 tickers, 8-Ks only.
- 2026-10-10 (M4): windows 1h/1d/5d(=168h) wall-clock; LLM aggregates exclude 'passing';
  features null outside coverage. Feature set 58e14e54d4af was built before coverage gating
  (volume_z wrong in early 2025) — superseded by 95e2118e52e1; both count as tried variants.
- 2026-10-10 (M6): scale to the PIT S&P 500 from 2019-01-01 (fja05680/sp500 + Wikipedia);
  keep the full entry × horizon × adjustment grid; delisted prices from Tiingo (free tier).
- 2026-10-10 (M6): full EDGAR backfill = 651 CIKs, ~48k 8-Ks; ~5–6 SEC requests per filing at
  8 req/s → ~15–17 h. Don't run other SEC-heavy jobs (companyfacts for eval) concurrently.
  Actual: 53,050 8-Ks in ~12 h, 0 failures. Embedding derive is the slow part (bge-small on
  CPU, ~1–3 chunks/s × ~355k 300-word chunks, single write at the end): left running.
- 2026-10-11 (M6): batch scoring 55,493 requests (7 batches, byte-capped), done in ~25 min,
  $15.91 (55,471 ok, 22 max_tokens). Feature set 8990ee7a9775 (embeddings off, config
  `config/variants/sp500_noemb.yaml`): 984,567 rows, 1,954 sessions, ~504 members/day.
- 2026-10-11 (M6, eval e1a1af, pre-holdout, ~52k news ticker-days, 1,878 sessions, 659
  tickers): no BH discovery out of 180 tests. LLM sentiment univariate IC is small and
  positive for `close` entry (≈ +0.01–0.016 at 1–2d; best p 0.01, sector-adj), ≈ 0 for
  `next_open` (the effect, if any, is overnight), stronger in the small tercile (≈ +0.02 vs
  +0.004 large). LLM lift over controls + LM/volume: |lift| ≤ 0.01 IC, none significant. The
  M5 negative 1–2d IC (10 tickers) did not replicate — it was noise.
- 2026-10-11 PRE-REGISTERED (before running) — follow-up tests, run one at a time, each a new
  eval config (all count as tried variants):
  1. **Earnings-only**: `eval.sample_event_types: [earnings]` (ticker-days with ≥1 LLM
     event_type=earnings founder in the 1d window), feature set 8990ee7a9775, same grid/models.
     Primary: `1d_llm_sent_cwmean` IC > 0 at close entry, h ∈ {1, 2}, market-adjusted.
     After-close releases become tradable at the next open → this is post-announcement drift
     from the first close after tradability, not the announcement reaction.
  2. Size-stratified (small tercile), 3. novelty/embeddings set, 4. news vendor — each
     pre-registered here before it runs.
- 2026-10-11 RESULT test 1 (eval a82bca, eval config f6fefba49188; ~15.7k earnings ticker-days,
  1,544 sessions, 653 tickers): primary `1d_llm_sent_cwmean` close/market IC h1 +0.018
  (p 0.096), h2 +0.032 (CI [+0.006, +0.056], p 0.018; ×2 primary cells = 0.036). No BH
  discovery across the run's 180 tests (min p 0.002 vs first BH threshold 0.00056). Sentiment
  IC ≈ +0.03 at close/2–10d (both adjustments), ≈ half that at next_open → part of the drift is
  overnight. LM tone is *negative* on the same sample (−0.01..−0.02). LLM lift over M1 small
  (+0.005..+0.014; best p 0.010). By size: mid/large ≈ +0.03..+0.046, small ≈ +0.01..+0.02 —
  the opposite of the full-sample size pattern.
- 2026-10-11 PRE-REGISTERED test 2 (size): `eval.sample_size_buckets: [small]` on the full news
  sample (not earnings-only), feature set 8990ee7a9775, same grid/models. Primary:
  `1d_llm_sent_cwmean` IC > 0 at close entry, h ∈ {1, 2}, market-adjusted. Buckets are PIT
  terciles of mcap among all index members on each date (`market_caps`).
- 2026-10-12 RESULT test 2 (eval aea12f, config 33cda6f54b63; ~17.2k small-tercile news
  ticker-days, 372 tickers): primary FAILS — close/market IC h1 +0.013 (p 0.26), h2 +0.017
  (p 0.17). Sector-adjusted h1 +0.024 (p 0.03) is not a primary cell. No BH discovery (best
  lift p 0.002, close/2 sector lgbm). The full-sample small-cap hint does not hold up.
- 2026-10-12 PRE-REGISTERED test 3 (novelty): embeddings done (51,915 founder vectors,
  bge-small, full documents; derive took ~24 h on CPU). New feature set = default config
  (embeddings on), full news sample, same grid/models. Primary: rank IC of
  `1d_llm_sent_x_novelty` (= `1d_llm_sent_cwmean` × `1d_emb_novelty_mean`, `add_derived`)
  > 0 at close entry, h ∈ {1, 2}, market-adjusted. Novelty is null when its 30-day look-back
  isn't covered, so the first month has no interaction values.
- 2026-10-12 RESULT test 3 (feature set c35f16b5bccb rebuilt, eval 9909b6, config
  ec5249afad6f, 220 tests): primary FAILS — `1d_llm_sent_x_novelty` close/market IC h1
  +0.006 (p 0.39), h2 +0.013 (p 0.084), i.e. no better than sentiment alone (+0.007 / +0.013).
  Novelty alone ≈ +0.003..+0.008. No BH discovery (best lift p 0.004, next_open/20 sector).
- 2026-10-12 BUG (fixed): `feature_set_id` did not include the universe, so the S&P 500
  embeddings build reused the 12-ticker M5 id c35f16b5bccb. Reads keep the latest build per
  (ticker, as_of) and all 984,567 rows came from the new build, so eval 9909b6 is valid; the
  M5 rows are shadowed. Ids now hash `cfg.universe` + the PIT membership table.
- 2026-10-12 HOLDOUT PLAN (user: wait & collect; holdout NOT looked at). Holdout so far has
  533 earnings ticker-days / 58 sessions (≈ one season; ~530 per quarter) → ~15–20% power
  for IC 0.03, so it is not spent yet. PRE-REGISTERED confirmatory test, ONE look only:
  earnings sample (`sample_event_types: [earnings]`), `1d_llm_sent_cwmean`, close entry,
  h = 2, market-adjusted; pass = IC > 0 with one-sided bootstrap p < 0.05 (two-sided p < 0.10).
  Same feature config / link policy / Haiku 5.5 prompt 6108c555090fa4b3 (if the model is
  retired, the test is re-planned, not silently switched). Look date: first run on or after
  **2027-08-31** (after the Q2-2027 season, expected n ≈ 2,600, power still only ~40%); not
  earlier, regardless of interim results. The other 179 cells are reported as exploratory.
- 2026-10-12: live collection for the holdout: `scoring.live_in_collect: true` (≈ $0.50/month,
  live features need scores at `scored_at`), `collect.interval_seconds: 120` = period of the
  Windows scheduled task running `collect --once` (live coverage needs ok polls ≤ 3 intervals
  apart). The holdout from 2026-07-01 to 2026-10-10 is backfill-rule data; later is live-rule.
- 2026-10-12: run live collection in ONE labelled console window, `scripts\collect_window.cmd`
  (title "sentiq - COLLECTING 8-Ks", one `collect --once` per 120 s, `[FAILED]`/`[UNHEALTHY]`
  appended to the title), not a 2-minute scheduled task (that flashes a console window every
  run). EDGAR silence on US federal holidays (`timeutil.edgar_closed_dates`: NYSE open, EDGAR
  closed — Columbus/Veterans Day) no longer raises `silent_feed`; found live on 2026-10-12.
- 2026-10-12 test 4 vendor: Massive (ex-Polygon) `/v2/reference/news`, free Basic key (2 years
  of history, 5 req/min). WRDS/RavenPack not available (alumni). Tiingo News: 3 months only.
  Measured: ~170–250 articles/day market-wide, ~75–105/day tagged with S&P 500 tickers; mostly
  Motley Fool, GlobeNewswire (incl. law-firm notices), Benzinga, Investing.com — no Reuters/DJ.
  Text = title + `description` (a machine-written summary; no body). Only `published_utc`
  (minute precision), no vendor first-seen time. Massive's per-ticker `insights` sentiment is
  dropped at ingest (look-ahead risk). Config variants used by tests live in `config/variants/`.
- 2026-10-12 PRE-REGISTERED test 4 (vendor news): `config/variants/news_massive.yaml`
  (features.sources = coverage_sources = [news_api], embeddings off), backfill 2024-10-13 →
  2026-10-12, Haiku 5.5 prompt 6108c555090fa4b3 on title + summary, news sample = ticker-days
  with ≥1 vendor-news story in 1d, pre-holdout only. Primary: `1d_llm_sent_cwmean` IC > 0 at
  close entry, h ∈ {1, 2}, market-adjusted. Timing: backfill rule published + 300 s → session.
  If the primary passes, pre-planned robustness checks: (a) longer lag (published + 1 h) and
  (b) headline-only scoring (rules out hindsight in vendor summaries) before any holdout use.
- 2026-10-12 LEAKAGE (user question): Haiku 5.5's training data very likely covers most or all
  of the pre-holdout period, so the model may "remember" outcomes; that bias looks exactly like
  positive IC. All pre-holdout ICs are upper bounds; the holdout (after the training cutoff,
  especially the live part) is the clean test. Masking (names/tickers/products/people/places;
  dates and numbers stay) reduces but cannot remove recall.
- 2026-10-12 PRE-REGISTERED leakage check A (earnings, before scoring): masked rescore of the
  pre-holdout earnings founders (`score --masked --event-types earnings --end 2026-07-01`),
  masked feature set (`features.llm_masked: true`). On the identical rows of the test-1 earnings
  sample, compare IC(masked) vs IC(unmasked) of `1d_llm_sent_cwmean`, primary cell close/2/
  market, paired block bootstrap of the difference. Reading: masked IC keeps ≥ half of the
  unmasked IC and the difference CI covers 0 → no evidence that recall drives test 1; masked IC
  ≈ 0 with the difference CI excluding 0 → test 1 is likely recall, not reading.
- 2026-10-12 AMENDED test 4 (before any scoring or outcomes): score news founders unmasked and
  masked; the **masked** feature set is primary (same primary cells), unmasked is secondary.
- 2026-10-12 RESULT leakage check A (masked rescore 16,210 requests, $8.11; masked feature set
  ba818990bf17; `newsfeed leakage` run 73e752, 15,720 identical rows): primary close/2/market
  IC unmasked +0.032, masked +0.033, diff +0.000 [−0.005, +0.006], p 0.87 → by the
  pre-registered reading, no evidence that recall drives test 1. Masked/unmasked sentiment
  correlate 0.94. Longer horizons (10–20d) show small consistent negative diffs (−0.003 to
  −0.005, p ≈ 0.06–0.13): at most mild recall there. Limits: dates and numbers stay visible, so
  recall through them is not tested; the holdout remains the clean test. The unmasked IC
  reproduces test 1 exactly (eval-inputs refactor check).
- 2026-10-12 RESULT test 4 (vendor news; scoring 112,698 unmasked + 112,959 masked requests,
  $12.05; feature sets masked 57330b614405 / unmasked 9fb97ec0eadc; evals 3ee8cf / af6e68;
  ~38k news ticker-days, 426 sessions, 539 tickers, 2024-10-15 → 2026-06-29): primary FAILS —
  masked `1d_llm_sent_cwmean` close/market IC h1 +0.002 (p 0.81), h2 +0.003 (p 0.72); unmasked
  −0.000 / +0.003. No BH discovery in either run (best univariate p ≈ 0.09). The conditional
  robustness checks (longer lag, headline-only) are not needed. Caveat: Massive's feed is
  Motley Fool / GlobeNewswire / Benzinga / Investing.com headlines + machine summaries, not a
  real-time wire, so this is a null for that feed, not for news in general.
- 2026-10-12 Test-4 data profile (user asked about cleaning/categories): of 113k scored news
  pairs 60% are 'passing' mentions, 35% event_type macro, 34% other; Motley Fool = 54%; 15% of
  articles tag ≥4 tickers; ~6.6% of titles are law-firm class-action solicitations (almost all
  GlobeNewswire), plus "Market Size … by 2035" research releases. Test 4 pooled all of it.
- 2026-10-12 PRE-REGISTERED test 4b (confirmatory, cleaned news; no new LLM cost):
  `config/variants/news_clean_masked.yaml` = test-4 masked set + `llm_event_types`
  [analyst_action, product, earnings, legal_regulatory, m&a, guidance] (relevance ≥ secondary
  as always) + `exclude_title_regex` (law firms, market-research releases) + `eval.sample: llm`
  (days with ≥1 qualifying LLM event). Primary: masked `1d_llm_sent_cwmean` IC > 0 at close,
  h ∈ {1, 2}, market-adjusted. Robustness checks (lag, headline-only) only if it passes.
- 2026-10-12 PRE-REGISTERED tier 2 (EXPLORATORY, `newsfeed explore-types`): per event type on
  "pure" samples (all qualifying LLM events in the 1d window of that type), news = test-4b
  feature set (masked) × [analyst_action, product, earnings, legal_regulatory, m&a, guidance],
  8-K = 8990ee7a9775 (unmasked; only earnings has a masked rescore) × [earnings, guidance, m&a,
  legal_regulatory, product, other]; cells close/1/market and close/2/market → 24 tests, one BH
  (q 0.10). Survivors are candidates for live/holdout confirmation only, not findings.
- 2026-10-12 RESULT test 4b (cleaned news, masked; feature set 237fddb5e2f7, eval faee2b;
  ~12.1k ticker-days, 425 sessions, 512 tickers): primary FAILS — close/market IC h1 −0.001
  (p 0.99), h2 +0.003 (p 0.75). No BH discovery (180 tests). Not pre-registered, noted only:
  20d IC +0.039 (close/market p 0.004; LM tone +0.029, p 0.27).
- 2026-10-12 RESULT tier 2 (explore-types 86a94c, 24 tests): no BH survivor. News: every type
  ≈ 0 except analyst_action (+0.020 / +0.027, p 0.29 / 0.15). 8-K: earnings +0.019 / +0.031
  (p 0.09 / 0.03, = test 1), guidance +0.037 / +0.078 (n 1,406; p 0.31 / 0.03), m&a, legal,
  product ≈ 0, other (26.7k rows) +0.007 / +0.005 — the full-sample 8-K IC comes from
  earnings + guidance. Candidates only (8-K guidance), not findings.
- 2026-10-12 Categorisation notes: event_type and relevance are Haiku labels from the same
  scoring call (definitions in `scoring/prompts/score_v1.md`; "results + guidance together →
  earnings"). Not human-validated. Spot checks: 8-K guidance = outlook raises/cuts/withdrawals,
  pre-announcements, investor-day targets (some also contain results). News analyst_action is
  mostly Motley Fool / Zacks "which stock is a better buy" pieces, not broker rating changes.
- 2026-10-12 HOLDOUT PLAN AMENDED (holdout still unseen; user decision): two pre-registered
  hypotheses at the same single look (on/after 2027-08-31), α split by Bonferroni:
  **H1 earnings** — unchanged definition (test-1 sample, `1d_llm_sent_cwmean`, close/2/market),
  pass = IC > 0 with one-sided p < 0.025 (two-sided p < 0.05; was one-sided 0.05).
  **H2 guidance** — `config/variants/sp500_noemb_guidance.yaml` (pure-guidance sample:
  all qualifying LLM events in the 1d window are 8-K guidance; unmasked, same feature config as
  test 1), `1d_llm_sent_cwmean`, close/2/market, pass = IC > 0 with one-sided p < 0.025. H2 was
  found by exploration (pre-holdout IC +0.078, n 1,411 — likely inflated by selection), so its
  holdout test is the first real test. Expected holdout n ≈ 210 (≈188/yr; 38 so far) → power
  roughly 20% even if the true IC were 0.078.
- 2026-10-12 Hand-labelling: private artifact "News Label Desk" (claude.ai/artifact/
  7ptLUVXwLY4oxSHYEZxVdb), 200 news pairs stratified 25 per model event type (seed 2026,
  `scripts/make_label_sample.py`), blind; answer key `data/labeling/news_sample_v1_key.csv`.
  Labels land in the artifact's db collection `labels`.
- 2026-10-12 PRE-REGISTERED (EXPLORATORY) text clusters, `newsfeed cluster-explore` with
  `config/variants/news_massive_masked.yaml`: news founders, title + summary, TF-IDF (1–2-grams,
  min_df 5, max_df 0.5) → SVD 100 → k-means k = 25, seed 7, fitted on pre-holdout documents;
  events = masked LLM scores with relevance ≥ secondary, dated to the first decision at or after
  tradable_at, fanned out to the member ticker; per-cluster IC at close/1 and close/2 market;
  one BH across ≤ 50 tests. Candidates only.
- 2026-10-12 RESULT text clusters (run 01c43f, 50 tests): no BH survivor (smallest p 0.010 vs
  first threshold 0.002). Clusters are interpretable themes — ETFs, 13F stake changes,
  market-research releases, "article discusses" listicles, two law-firm clusters, IPOs/deals,
  dividends, AI press releases, generic stock picks, Tesla, Lilly, Nvidia, biotech trials, big
  tech AI, earnings recaps, Buffett, dividend declarations, data centres, quantum, oil/market,
  tariffs, analyst-opinion pieces. Largest (nominal) ICs: law-firm Rosen cluster +0.121 at 1d
  (p 0.010, n 534), IPO/deals +0.062 at 2d (p 0.010), biotech trials −0.101 at 2d (p 0.026),
  tariffs −0.094 at 2d (p 0.038); all within what 50 tests produce by chance.
- 2026-10-13 Transcripts source (`ingest/transcripts.py`, `sources.transcripts`): defeatbeta
  `stock_earning_call_transcripts`, pinned commit 860d42a, local copy in `data/ref/defeatbeta/`.
  12,415 S&P 500 calls 2019 → 2026-10 (11,873 after merging share-class duplicates); 1,419 hit
  the 16k-token cap. Date-only timing → published = end of call date (NY), tradable next open
  (all 09:30). Linking: `universe.source_only_linking: [transcripts]` (own company only).
  Relink now streams items on its own cursor (bodies exceeded a 2 GB Arrow buffer; writing on
  the reading connection had ended the stream after one batch).
- 2026-10-13 PRE-REGISTERED test 5 (earnings calls, before scoring): score transcript founders
  unmasked and masked (Haiku 5.5, prompt 6108c555090fa4b3, doc type "earnings call
  transcript"); feature sets `config/variants/transcripts_masked.yaml` (PRIMARY) and
  `transcripts_unmasked.yaml` (features.sources = coverage_sources = [transcripts], embeddings
  off, `eval.sample: llm`), 2019-01-01 → 2026-10-10, pre-holdout eval. Primary: masked
  `1d_llm_sent_cwmean` IC > 0 at close entry, h ∈ {1, 2}, market-adjusted (decision = first
  15:50 after the 09:30 open following the call, i.e. the day after the call). Secondary 5b
  (incremental over the press release): on the primary rows, partial rank IC = Spearman of the
  return with the residual of rank(transcript sentiment) regressed on rank(8-K
  `5d_llm_sent_cwmean`, feature set 8990ee7a9775, same ticker/as_of); same cells, block
  bootstrap. Robustness (only if the primary passes): headline-free check not applicable;
  instead Q&A-only text vs prepared remarks.
- 2026-10-13 SHARED TICKERS (found by the scoring duplicate check): 17 universe tickers belong
  to two CIKs. Most are re-registrations of one company (DIS, CI, MDT, AVGO, BLK, XOM, XRX,
  APA, BG, …); IR (old Ingersoll-Rand plc → TT vs Gardner Denver → IR from 2020-04) and
  FOX/FOXA (21st Century Fox vs Fox Corp) are different companies. The linker's ticker→CIK
  lookup is ambiguous for them. Fixes: (1) transcripts parser v2 tags a call with ticker T only
  if T's membership spell on the call date belongs to the same CIK or to a predecessor CIK
  with no current ticker (5 Gardner Denver 2019 calls dropped, 26 IR tags removed from them and
  from TT's post-2020 calls); (2) linker v3 takes `source_cik` for source tags (transcripts
  pass `meta.cik`); (3) `candidate_pairs` skips duplicate (item, canonical ticker) pairs with
  a warning; (4) relink also redoes items whose links predate their stored row; (5)
  `derive --relink-source <src>` forces a fresh link batch (needed once: a collector cycle had
  relinked the v1 rows under linker v3 while they were being replaced). Lesson: don't move
  lake files while a collector process is running. 8-K and news links are unchanged.
- 2026-10-13 RESULT test 5 (transcripts; scoring 11,871 + 11,853 requests, $20.40; feature
  sets masked 09557d983019 / unmasked b695ae2d4c10; evals a92fbd / 9b6eb3; ~11.3k call-days,
  1,130 sessions, 565 tickers). Primary (masked) close/market IC h1 +0.029 (CI [+0.004,
  +0.057], p 0.026), h2 +0.035 (CI [+0.004, +0.067], p 0.024) → passes nominally (×2 for the
  two primary cells: 0.052 / 0.048). Unmasked +0.029 / +0.031 (p 0.024 / 0.054): masking does
  not lower it. LM tone negative again (−0.01..−0.03). Masked run: first BH discoveries of the
  project, both LightGBM lifts (M2 − M1) — close/20/market +0.016 and next_open/1/sector
  +0.019, bootstrap p "0.000" = < 1/1000, at the 1000-rep resolution limit vs the first BH
  threshold 0.00056 → needs more reps before it counts; not in the unmasked run.
  Test 5b partial IC (transcript sentiment given the 8-K `5d_llm_sent_cwmean`, 11,182 rows):
  h1 +0.012 (p 0.30), h2 +0.015 (p 0.21); on the same rows the 8-K sentiment alone has IC
  +0.031 / +0.037 → the call mostly repeats the press release; no significant increment.
- 2026-10-13 defeatbeta latency: the HF repo keeps no history (single squashed commit "Compact
  finance data history <date>", ~05:29 UTC daily), so past arrival times can't be
  reconstructed. Forward: collector checks hourly (`sources.transcripts.catch_up_every_minutes:
  60`), `newsfeed transcripts-latency` reports hours from the Item 2.02 8-K to first_seen and
  the share in hand by the next open / next 15:50 decision (Q3 season from 2026-10-14).
- 2026-10-13 RESULT delay sensitivity (masked transcripts, event-level, run 8969bd): close/
  market IC h1/h2 at +0 sessions +0.030 / +0.035 (reproduces test 5), +1: +0.018 / +0.017
  (p 0.16 / 0.24), +2: +0.014 / +0.023, +5: +0.010 / +0.016. About half the signal is gone if
  the call is usable one session later than assumed → a source that lags a day loses most of
  the significant part.
- 2026-10-13 CORRECTION: test 5's two "BH discoveries" (masked LightGBM lifts) were a
  bootstrap-resolution artefact: with 1,000 reps they had p = 0, below any BH threshold. Rerun
  at 10,000 reps (configs `transcripts_masked_boot_*`): close/20/market lift +0.016, p 0.0016;
  next_open/1/sector lift +0.019, p 0.005 → BH over the run's 180 tests with these p: no
  discovery. Fix: `boot_summary` uses (k + 1)/(B + 1) (p is never 0; floor 2/(B+1)), and each
  report states whether the floor is below the first BH threshold q/n (1,000 reps over 180
  tests is not: cells at the floor need ≥ 3,600 reps). Use ≥ 10,000 reps for the holdout look.
- 2026-10-13 RESULT delay sensitivity, earnings 8-Ks (event-level; unmasked / masked): close/
  market IC h1/h2 at +0 sessions +0.020 / +0.031 (reproduces test 1), +1: +0.029 / +0.031
  (p 0.016 / 0.036), +2: +0.021 / +0.020 (p ≈ 0.09 / 0.14), +5: +0.023 / +0.013; masked the
  same. The release-based signal holds for ~1–2 sessions, then fades. Transcript delay 0 is
  already ~1 session after the release (date-only timing), so the transcript curve is this
  curve shifted: it drops once it passes ~2 sessions after the release. Conclusion: what
  matters is reading the release within a session or two; the 8-K arrives within minutes via
  EDGAR (live collector + live scoring), so live call transcription would add little.
- 2026-10-14 PRE-REGISTERED test 5 robustness (owed: the test-5 primary passed nominally) —
  Q&A vs prepared remarks, before any backfill/scoring. Section views of each call from the
  pin (D:\sentiq-archive\defeatbeta_pin\860d42ac253f.parquet): sources `transcripts_qa` /
  `transcripts_prep`; split at the first paragraph after the opening that says questions start
  now (`transcripts.split_sections`, found in 95.5% of calls; calls without it are skipped);
  each section capped at 16k tokens; same date-only timing; `dedupe.isolated_sources` keeps
  them out of the full call's story. Scored MASKED only (`score --masked --unpaired`), Haiku
  5.5, prompt 6108c555090fa4b3. Feature sets `config/variants/transcripts_{qa,prep}_masked.yaml`.
  Readout: masked `1d_llm_sent_cwmean` IC of each section at close/1 and close/2 market, and
  the paired difference IC(Q&A) − IC(prepared) on identical rows (`newsfeed leakage` used as a
  generic paired comparison: --feature-set prep --masked-set qa). Reading: Q&A carries the
  signal if its IC ≥ the full-call IC and the difference CI excludes 0; otherwise no evidence
  that either section dominates. Descriptive robustness, not a new BH family.
- 2026-10-15 RESULT test 5 robustness (sections: 11,866 items each, 544 calls without a Q&A
  boundary skipped; masked scoring 11,355 + 11,354 requests, $10.59; feature sets Q&A
  a8fa32903f0c / prepared 69b725996572). Identical rows (10,811), close/market IC h1/h2:
  prepared +0.028 / +0.031, Q&A +0.020 / +0.030, diff Q&A − prepared −0.008 (p 0.28) / −0.001
  (p 0.87); vs the full call (10,773 rows, +0.028 / +0.034): Q&A −0.008 / −0.004, n.s. Section
  sentiments correlate 0.68. By the pre-registered reading: no evidence that either section
  dominates; both carry similar signal (Q&A slightly higher at 5–20d, +0.011..+0.016,
  p 0.10–0.20, n.s.). Test 5's signal is not an artefact of one part of the call.
- 2026-10-15 PRE-REGISTERED tests 6–9 (signal families of "Machine Learning for Trading"
  §10.5 that were not yet tested; no LLM cost, existing scores and feature sets only;
  pre-holdout; each run counts as a tried variant):
  6. **LLM earnings surprise** (`newsfeed event-ic`): events = 8-K story founders with LLM
     event_type earnings, relevance ≥ secondary, unmasked Haiku 5.5 prompt 6108c555090fa4b3,
     non-null `surprise_vs_expectations` (≈ 22% of releases: the prompt asks for null unless the
     text compares results with expectations); event-level, dated to the first 15:50 decision
     ≥ tradable_at (as `delay-sensitivity` at delay 0), fanned out to member tickers. Primary:
     rank IC of surprise > 0 at close entry, h ∈ {1, 2}, market-adjusted. Secondary: (a)
     semi-partial IC of surprise given LLM sentiment on the same rows (beyond sentiment?), (b)
     masked surprise, (c) transcript surprise. Exploratory: sentiment IC by materiality (high
     vs medium/low).
  7. **Attention → volatility** (`newsfeed vol-test`): all member ticker-days of feature set
     8990ee7a9775 (8-K attention) where the feature is non-null. Targets, close entry, raw:
     RV_h = sd of daily log close-to-close returns over sessions i+1..i+h, h ∈ {5, 20}, and
     |r| close[i] → close[i+1]. Statistic: partial rank correlation (Pearson of the rank
     residuals of feature and target, each regressed on the ranks of ctl_vol_20d, |ctl_gap|,
     |ctl_ret_1d|, ctl_log_dvol_20d, ctl_volume_ratio), circular block bootstrap over sessions
     (block h + 1). Primary: `1d_n_stories` vs RV5, partial IC > 0. Secondary: `1d_volume_z`,
     `5d_n_stories`, `1d_llm_sent_maxabs` (news days only), targets RV20 and |r1|, and every
     cell again on non-earnings days (`1d_llm_evt_earnings` = 0). Expectation: the primary is
     positive mostly through earnings releases (known event-volatility effect), so the
     non-earnings rows are the informative ones. A pass says attention predicts risk, not
     direction (not a return signal).
  8. **8-K item event study** (`newsfeed item-study`): original 8-Ks (form 8-K, not 8-K/A),
     one event per (member ticker, decision date, item code) from `meta.items`; codes with
     ≥ 100 pre-holdout filings except 9.01 (exhibits): 2.02, 7.01, 8.01, 5.02, 1.01, 5.07, 2.03,
     5.03, 1.02, 2.01, 3.03, 3.02, 2.05, 3.01, 2.06 (15 codes; 4.02 restatements has only 14,
     not tested). Decision = first 15:50 ≥ tradable_at. Statistic: mean abnormal return =
     market-adjusted return minus the same-date equal-weighted mean market-adjusted return of
     all index members (removes the equal- vs cap-weighted drift), close entry, h ∈ {1, 5, 20};
     circular block bootstrap over sessions; two-sided; one BH (q 0.10) over the 45 tests.
     Sector-adjusted abnormal returns reported, not tested. This is drift after the first
     tradable decision, not the announcement reaction.
  9. **ICIR + quintile spreads** (reporting, no new hypothesis): for univariate cells, daily
     cross-sectional rank IC on sessions with ≥ 10 names (mean, sd, ICIR = mean/sd, days) and
     the Q5 − Q1 mean return (quintiles by within-day rank), averaged over days, CIs by
     circular block bootstrap over sessions. Descriptive only, never added to a BH family; the
     pooled IC stays the pre-registered statistic. Reported by `newsfeed ic-detail` for the
     primaries of test 1, test 5 (masked) and H2 guidance (pre-holdout), and in future eval
     reports.
- 2026-10-15 RESULT test 6 (LLM earnings surprise, event-level; runs 6ef6e6 primary, 3cbc44
  semi-partial, 21de45 masked, d5c0ac transcripts, ea04fb materiality; sanity: `--value
  sentiment` run ebe354 reproduces the 8-K earnings delay-0 IC +0.020 / +0.031). Surprise is
  present on 3,504 of 15,938 earnings events. Primary close/market IC h1 +0.033 (p 0.078), h2
  +0.040 (CI [+0.001, +0.075], p 0.044; ×2 primary cells = 0.088) → weak: h2 nominal only.
  (a) Beyond sentiment: semi-partial IC −0.011 / −0.015 (p 0.52 / 0.35); on the same rows
  sentiment alone has IC +0.047 / +0.060 → surprise adds nothing; sentiment already contains
  it. (b) Masked surprise +0.045 / +0.044 (p 0.006 / 0.018): masking does not lower it. (c)
  Transcript surprise +0.007 / +0.021 (n 6,502, n.s.). Materiality split uninformative: 94% of
  earnings releases are "high" (high +0.018 / +0.029; medium/low n 904, ≈ 0). Not
  pre-registered, noted only: sentiment IC is about twice as large on releases that state a
  comparison with expectations (+0.060 at h2) as on all releases (+0.031) — a candidate
  subgroup, not a finding.
- 2026-10-15 RESULT test 7 (attention → volatility, run e58be9; ~0.93M member ticker-days,
  1,000 reps so p ≥ 0.002): primary FAILS — `1d_n_stories` vs rv_5 partial IC −0.001
  (p 0.64; raw +0.012): once trailing volatility, |gap|, |yesterday's return|, dollar volume
  and volume ratio are controlled, the 8-K count adds nothing. Secondary (no BH family, read
  with care): `1d_volume_z` partial +0.047 (rv_5) / +0.097 (rv_20), same on non-earnings days;
  `5d_n_stories` −0.036 / −0.077; `1d_llm_sent_maxabs` (news days only, n ≈ 47k) +0.026
  (rv_5), +0.048 (|r1|; non-earnings +0.024). Caveat: volume_z = (count − own baseline mean)
  / sd is mostly a *cross-sectional* filing-frequency measure on the ~95% of days with no
  8-K (frequent filers sit below zero), so its sign may reflect who files often rather than
  attention shocks; checking it only on days with an 8-K would separate the two. Strong LLM
  |sentiment| days are followed by bigger moves, as expected for material news.
- 2026-10-15 RESULT test 8 (8-K item event study, run 81e0c0; 15 codes × close/{1,5,20},
  45 market-adjusted tests): no BH discovery (smallest p 0.052: 3.01 delisting notices, 5d,
  −118 bp, n 88). The large codes (2.02 n 15.8k, 8.01 13.3k, 7.01 11.6k, 5.02 9.5k, 1.01 5.1k,
  5.07 4.1k) sit within a few bp of zero at 1–5 days (1.01 −11 bp at 5d, p 0.08). For S&P 500
  members the filing type alone carries no drift after the first tradable decision; whatever
  the market does with the item happens before 15:50 on the first day.
- 2026-10-15 RESULT test 9 (descriptive daily view, `ic-detail`; `1d_llm_sent_cwmean`,
  close/market h1 / h2 / h5). Test 1 earnings (run 817cdc): pooled +0.019 / +0.032 / +0.031;
  only 464 sessions have ≥ 10 earnings names; daily IC +0.011 / +0.012 / +0.011 (all n.s.),
  ICIR ≈ 0.04 per day, Q5 − Q1 +8 / +9 / +1 bp (n.s.). Test 5 transcripts masked (run
  322f51): pooled +0.031 / +0.036 / +0.032; 332 days; daily IC +0.037 (p 0.022) / +0.036
  (p 0.050) / +0.026, ICIR 0.13 / 0.13 / 0.09 per day, Q5 − Q1 +16 / +5 / −3 bp (n.s.). H2
  guidance (run a99584): pooled +0.041 / +0.082 / +0.054, but no session has ≥ 10 guidance
  names, so no daily view. Reading: for earnings 8-Ks most of the pooled IC is not a same-day
  cross-sectional ranking on busy days — either it lives on thin days (< 10 releases) or in the
  between-day component (days with more positive releases are followed by better
  market-adjusted returns); the pooled statistic of H1 stays as pre-registered. Quintile
  spreads are small and noisy everywhere (a few bp per 1–2 days, CIs ±20–40 bp).
- 2026-10-15 (user decision) eval horizons += 60 sessions (default config; variants keep their
  own lists, so earlier eval ids are unchanged). Price windows and post-delisting cash padding
  now scale with the longest horizon (`prices.horizon_days`: 20 → 45 days / 25 sessions as
  before, 60 → 105 days / 65 sessions). Caveat grows with the horizon: delisted names are held
  as cash at the last trade for up to 65 sessions. Cache: a series that stopped > 30 days
  before the end of the window it was fetched for is treated as ended (no refetch for a later
  end).
- 2026-10-15 (user: "that order sounds good") new source `edgar_periodic` = 10-K / 10-Q
  originals (no amendments), primary document only, full visible text (no truncation in
  practice), section offsets in `meta.sections` (10-K items 1, 1A, 7, 7A; 10-Q MD&A "I-2" found
  by its title, risk factors "II-1A"; cross-referenced sections, e.g. banks' MD&A, are
  missing). Same header timing as 8-Ks (backfill lag 60 s). Raw HTML on D:
  (`sources.edgar_periodic.raw_dir`); ingest windows start 400 days before membership so each
  filing has its year-earlier comparison. Isolated in dedupe; not in `collect.sources`; the
  live scoring hook now plans only `collect.sources` (guard: a 10-K is ~100k tokens). Not added
  to `universe.source_only_linking` (that would change the linker id → full relink); analyses
  use the filer CIK from `meta.cik`.
- 2026-10-15 Infra: Tiingo name searches and 404 (unknown ticker) answers are now cached in
  the price cache (`search/`, meta `not_found`); before, every eval repeated about a dozen
  throttled requests for unpriced names (~25 min, and 429s when runs overlapped).
