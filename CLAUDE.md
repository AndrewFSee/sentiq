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
  **M6 in progress** (branch `m6-sp500`): point-in-time S&P 500 universe 2019–present,
  per-CIK backfill windows, price routing (Tradier / renamed / Tiingo for delisted),
  split-consistent market caps, PIT sector ETFs, member-only feature rows. Remaining: full
  backfill → derive → batch scoring (ask before paying) → features → eval (pre-holdout).
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
  `data/configs/sp500_noemb.yaml`): 984,567 rows, 1,954 sessions, ~504 members/day.
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
