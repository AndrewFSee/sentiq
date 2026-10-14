# newsfeed

Turns financial news and SEC filings into **point-in-time-correct**, LLM-scored features, to test
whether news carries incremental tradable signal. A research tool, not a trading system.

Status: **Milestones 1–6** done: ingestion, live collection, dedupe, ticker linking, LLM
scoring, baselines, embeddings, point-in-time features, evaluation, and (M6) a point-in-time
S&P 500 universe from 2019 with delisted names, vendor news, leakage checks and pre-registered
follow-up tests. See [Findings so far](#findings-so-far), and `CLAUDE.md` for conventions, the
PIT rules and the full decisions log.

## Setup
```bash
uv sync                      # Python >= 3.11
cp .env.example .env         # then set SEC_USER_AGENT="YourProject research you@example.edu"
uv run pytest                # offline; uses recorded fixtures
```
Configuration lives in `config/default.yaml` (universe + name aliases, rate limits, token caps,
lags, collection and dedupe settings, paths). Point at another file with `--config path.yaml`
or `NEWSFEED_CONFIG`.

## Quickstart by stage

### 1. Backfill
```bash
uv run newsfeed backfill --start 2025-01-01 --end 2026-01-01                 # EDGAR 8-Ks
uv run newsfeed backfill --start 2025-01-01 --end 2026-01-01 --tickers AAPL,MSFT
uv run newsfeed backfill --start 2024-10-13 --end 2026-10-12 --source news_api \
    --config config/variants/news_massive.yaml                               # Massive news
uv run newsfeed backfill --start 2019-01-01 --end 2026-10-12 --source transcripts  # earnings calls
```
Backfill is idempotent and resumable. Re-running skips stored items without re-downloading
them, and failures are logged to the `failures` table and retried on the next run. Each run
ends by refreshing the derived tables (ticker links and stories).

> `sources.news_api.vendor` is `fixture` by default (synthetic records in
> `tests/fixtures/news/`) or `massive` (Massive, ex-Polygon, `/v2/reference/news`; set
> `MASSIVE_API_KEY`; the free tier gives 2 years of history at 5 requests/min, so use
> `page_size: 1000`). Massive supplies headline + machine-written summary, publisher time only;
> its per-ticker "insights" sentiment is dropped at ingest.

> `transcripts` reads earnings-call transcripts from a local copy of the Hugging Face dataset
> `defeatbeta/yahoo-finance-data` (`data/ref/defeatbeta/`, content-addressed files plus a
> `CURRENT` pointer). The dataset has the call date only, so a call becomes tradable at the next
> session open. Each call is tagged with the ticker its company held on that date (tickers shared
> by two companies, such as IR, are resolved by date) and links only to its own company.
> `transcripts_qa` / `transcripts_prep` are backfill-only views of the Q&A and prepared-remarks
> sections of the same calls.

### 2. Live collection
```bash
uv run newsfeed collect                  # loop every collect.interval_seconds
uv run newsfeed collect --once           # one cycle, for cron (state persists in data/state/)
uv run newsfeed collect --check          # health from saved state; exit 1 if unhealthy
uv run newsfeed collect --sources edgar,news_api --max-cycles 10
uv run newsfeed transcripts-latency      # how late live-collected calls arrive vs the 8-K release
```
On Windows, `scripts\collect_window.cmd` runs one `collect --once` per 120 s in a single
console window titled "sentiq - COLLECTING 8-Ks" (problems show in the title); put a shortcut
in `shell:startup` to start it at logon. Keep `collect.interval_seconds` equal to the period:
live coverage is only recorded when successful polls are at most 3 intervals apart.

Example cron entry (every 2 minutes, plus a health check that alerts via your mailer):
```
*/2 * * * * cd /path/to/sentiq && uv run newsfeed collect --once >> data/collect.log 2>&1
*/5 * * * * cd /path/to/sentiq && uv run newsfeed collect --check || <notify>
```
Each cycle polls each source (`collect.sources`: EDGAR and transcripts) and ingests new items,
refreshes the derived tables, evaluates health and writes `data/state/heartbeat.json`. For
transcripts, catch-up polls (hourly) check Hugging Face with one HEAD request and download a new
copy only when the transcripts file itself changed. Health problems:

| kind | meaning |
|---|---|
| `silent_feed` | no new entries from **any company** in the source feed for N market-open minutes, i.e. an outage rather than "no news" (EDGAR: US federal holidays don't count — NYSE may trade while EDGAR is closed) |
| `poll_failing` | N consecutive polls failed |
| `stalled` | no successful poll for N intervals during market hours |

Alerts are logged and written to the `alerts` table, and they clear on recovery.
A `collect` loop and a `backfill` can run at the same time against the same data directory.

### 3. Derived tables
```bash
uv run newsfeed derive          # incremental (runs automatically after backfill/collect)
uv run newsfeed derive --full   # rebuild stories from scratch
uv run newsfeed derive --relink-source transcripts   # force a fresh link batch for one source
```
- **Ticker links** (`item_tickers`): from vendor tags or the EDGAR filer, plus exchange tags,
  cashtags and company-name aliases. Each link records its method and confidence.
- **Stories** (`item_story`): exact and near-duplicate clustering. Each item joins one *earlier*
  item's story, so later articles can never regroup earlier ones.

### 4. Reading data point-in-time
```python
from datetime import datetime, UTC
from newsfeed.config import load_config
from newsfeed.dedupe import stories_as_of
from newsfeed.storage import Store

store = Store(load_config().paths)
t = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)
items = store.items_as_of(t, tickers=["AAPL"])  # only items with tradable_at <= t
stories = stories_as_of(store, t, tickers=["AAPL"])  # n_items / n_outlets as of t
```

### 5. LLM scoring
Needs `ANTHROPIC_API_KEY` in `.env`. Every (story-founding item, company) pair is scored once
by `scoring.model` (Claude Haiku 5.5) with a JSON-schema-constrained output: relevance,
sentiment, confidence, event type, novelty, materiality, surprise vs expectations, and a short
rationale. Results are cached forever (raw responses kept), so re-runs never re-pay.

```bash
uv run newsfeed score --dry-run                        # what would be scored + cost estimate
uv run newsfeed score --start 2026-07-01 --end 2026-08-01    # live API
uv run newsfeed score --mode batch --wait              # Batches API: half price, async
uv run newsfeed score-batches --wait                   # collect batches submitted earlier
uv run newsfeed cost                                   # spend per run
```
Jobs estimated above `scoring.confirm_above_usd` (default $1) refuse to start without `--yes`.

Checks on the scores (each covers only documents that already have a primary score):
```bash
uv run newsfeed score --replicates 2 --sample 100      # re-draw: the noise floor
uv run newsfeed score --masked --sample 200            # names/tickers/products hidden
uv run newsfeed score --model claude-sonnet-5-5 --sample 100   # stronger model
uv run newsfeed agreement --compare masked             # also prints the noise floor
uv run newsfeed agreement --compare model
uv run newsfeed score --masked --event-types earnings --mode batch   # mask one eval sample
```
If masked and unmasked labels disagree beyond the replicate noise floor, the model is likely
using its knowledge of the company (including possibly how the story ended), not just the text.

The prompt lives in `newsfeed/scoring/prompts/score_v1.md`. Never edit a version that has been
used; copy it to `score_v2.md` and set `scoring.prompt_version` (every variant is counted in the
`experiments` table).

### 6. Re-parsing after a parser fix
```bash
uv run newsfeed reparse    # rebuild EDGAR items from cached raw documents; timing unchanged
```

### 7. Baselines and features
Baselines are computed automatically by `derive` (and therefore after every backfill/collect):
- **Loughran-McDonald** tone per story: download the Master Dictionary CSV from
  [SRAF](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) (free for academic use)
  to `data/ref/lm/` (path in `features.lm_dictionary`).
- **FinBERT** (optional): `uv sync --extra finbert` (~2 GB), then `features.finbert: true`.

```bash
uv run newsfeed coverage                                   # what was collected, per ticker
uv run newsfeed coverage --rebuild-from-runs               # for data ingested before M4
uv run newsfeed features --start 2025-01-01 --end 2026-10-01
```
One row per (ticker, session) at the session close minus 10 minutes, with columns per window
(`1h_`, `1d_`, `5d_`): `n_items`, `n_stories`, `n_outlets`, `volume_z` (vs the trailing 60
days), `llm_n`, `llm_coverage`, `llm_sent_decay` (exponentially decayed sum), `llm_sent_cwmean`
(confidence-weighted mean), `llm_sent_maxabs`, `llm_sent_disp`, `llm_evt_<type>` counts, and
the same aggregates for `lm_tone_*` (plus `finbert_*` when enabled). Windows not covered by
collected data are null, not zero. Read a stored set with
`Store(...).features("<feature_set_id>")`.

Feature filters (all part of the feature-set id): `features.sources` (e.g. `[news_api]` for
vendor news only), `features.llm_masked` (use masked-text scores), `features.llm_event_types`
(LLM aggregates over these event types only) and `features.exclude_title_regex` (drop items by
title, e.g. law-firm solicitations). Embeddings are off by default (`features.embeddings`):
computing them on CPU takes ~1 day for 50k documents.

### 8. Evaluation
Needs `TRADIER_TOKEN` (or `TRADIER_API_TOKEN`) in `.env` for daily prices and dividends, and
for the S&P 500 universe also `TIINGO_TOKEN` (free tier) for delisted names.

```bash
uv run newsfeed eval                         # latest feature set, pre-holdout data
uv run newsfeed eval --feature-set <id>      # a specific feature set
uv run newsfeed eval --use-holdout           # final test on the frozen holdout (counted)
```
The report (`data/reports/eval-<run>.md`, plus Parquet tables) covers every cell of the grid
entry {same-day close, next open} × horizon {1, 2, 5, 10, 20 sessions} × {market, sector}
adjustment:
- **univariate rank IC** of key features (LLM sentiment, LM tone, volume z, novelty), also by
  size bucket;
- **incremental lift**: out-of-sample rank IC of ridge / LightGBM models with controls only
  (M0), + non-LLM news features (M1), + LLM features (M2); lift = M2 − M1;
- block-bootstrap confidence intervals and p-values, Benjamini-Hochberg across all tests, and
  a count of every prompt, feature set and eval configuration tried so far.

The holdout (`eval.holdout_start`, default 2026-07-01) is excluded unless `--use-holdout`:
use it once, at the end.

Sample filters: `eval.sample` (`news` = ≥1 story in the window, `llm` = ≥1 qualifying LLM
event, `all`), `eval.sample_event_types`, `eval.sample_size_buckets` and
`eval.sample_pure_event_type`. Each test's configuration is versioned in `config/variants/`.

```bash
uv run newsfeed leakage --feature-set <unmasked> --masked-set <masked> --config <variant>
uv run newsfeed explore-types --feature-set news=<id> --feature-set 8k=<id> --types earnings,guidance
```
```bash
uv run newsfeed partial-ic --feature-set <id> --control-set <id> --config <variant>
uv run newsfeed delay-sensitivity --source edgar --event-types earnings --delays 0,1,2,5
uv run newsfeed cluster-explore --k 25 --config <variant>
```
`leakage` compares two LLM features (masked vs unmasked, or any pair) on identical rows with a
paired block bootstrap; `partial-ic` asks whether one source adds information beyond another
(rank IC after regressing out the control); `delay-sensitivity` recomputes the event-level IC
when each document becomes usable N sessions later; `explore-types` and `cluster-explore` are
explicitly exploratory per-type and per-text-cluster breakdowns with one BH correction each.

Bootstrap p-values are (k + 1)/(B + 1), never 0, and each report states whether the smallest
attainable p is below the first Benjamini-Hochberg threshold. With 1,000 reps over 180 tests it
is not: cells at the floor need at least ~3,600 reps (`eval.bootstrap_reps`).

### Universe: point-in-time S&P 500
`universe.mode: sp500_pit` (default) uses historical index membership: the
[fja05680/sp500](https://github.com/fja05680/sp500) daily constituent lists (pinned commit)
plus monthly Wikipedia snapshots for CIK, company name and GICS sector.
```bash
uv run newsfeed universe --build    # fetch sources, build data/ref/universe/*.parquet
uv run newsfeed universe            # summary
```
Backfills fetch each company only around its membership window; feature rows exist only on
days the ticker is a member; prices come from Tradier, under the current ticker for renamed
companies, and from Tiingo for delisted ones (held as cash after the last trade). Set
`universe.mode: static` with `universe.tickers` for a fixed list.

## Findings so far
Pre-holdout, S&P 500, 2019 – June 2026 (details, ids and every variant tried: `CLAUDE.md`
decisions log). Rank IC of LLM sentiment vs market-adjusted returns from the 15:50 close:

| test (pre-registered) | sample | 1 day | 2 days | survives run-wide BH? |
|---|---|---|---|---|
| all 8-K news days | ~52k ticker-days | +0.007 | +0.013 | no |
| **1. earnings 8-Ks** | ~15.7k | +0.018 | **+0.032 (p 0.018)** | no |
| ↳ masked (names hidden) | same rows | +0.023 | +0.033 | — |
| 2. small-cap tercile | ~17.2k | +0.013 | +0.017 | no |
| 3. sentiment × novelty | ~48k | +0.006 | +0.013 | no |
| 4. vendor news (Massive, masked) | ~38k, 2024-10 → | +0.002 | +0.003 | no |
| 4b. vendor news, cleaned | ~12k | −0.001 | +0.003 | no |
| **5. earnings calls (masked)** | ~11.3k call-days | **+0.029 (p 0.026)** | **+0.035 (p 0.024)** | no |
| ↳ 5b. beyond the 8-K release (partial IC) | same rows | +0.012 | +0.015 | — |

- The only effect that holds up is on **earnings 8-Ks**: ≈ +0.03 IC at 2–10 days, half of it
  overnight; hiding identities does not reduce it (no evidence of training-data recall), and
  Loughran-McDonald tone is negative on the same filings. Exploratory per-type analysis adds
  **8-K guidance** (+0.078 at 2 days, n 1,411) as a candidate.
- **Earnings calls** carry about the same signal as the press release (+0.03, robust to
  masking) but add little beyond it (partial IC +0.01, n.s.).
- **Timing:** the release-based signal holds for 1–2 sessions after the release and then
  fades; a transcript usable one session later than assumed loses about half its IC. The 8-K
  arrives within minutes through EDGAR, so speed is not the bottleneck; live call
  transcription would add little.
- Vendor news (headlines + summaries from Motley Fool, GlobeNewswire, Benzinga, Investing.com)
  shows nothing, pooled, cleaned, by type or by text cluster.
- All pre-holdout ICs are upper bounds: the model's training data likely covers the period.
  The confirmatory test is pre-registered on the holdout (from 2026-07-01, growing with live
  collection): H1 earnings and H2 guidance, one look on/after 2027-08-31, α split 0.025/0.025.

## Inspecting
```bash
uv run newsfeed inspect   # items by month, links, story reasons, runs, failures, alerts
```

## Data layout
```
data/lake/items/src=<source>/ym=<YYYY-MM>/part-*.parquet   # items (view: items)
data/lake/item_tickers/part-*.parquet                     # links (views: item_tickers[_all])
data/lake/item_story/ym=<YYYY-MM>/stories.parquet         # stories (view: item_story)
data/raw/<source>/<id>/*.gz                               # verbatim source documents
data/newsfeed.duckdb                                      # runs, failures, alerts
data/lake/scores/part-*.parquet                           # LLM scores (views: scores, scores_all)
data/lake/baselines/part-*.parquet                        # LM / FinBERT scores (view: baselines)
data/lake/features/set=<feature_set_id>/part-*.parquet    # built feature tables
data/lake/embeddings/part-*.parquet                       # story embeddings
data/reports/eval-<run>.md                                # evaluation reports (+ leakage-,
                                                          #   partial-ic-, delay-, clusters-, ...)
data/ref/prices/ data/ref/sec_shares/                     # cached prices, SEC share counts (slim)
data/ref/defeatbeta/                                      # transcripts dataset copy + CURRENT
data/labeling/                                            # hand-labelling sample + answer key
data/ref/prices_tiingo/                                   # cached Tiingo prices (delisted)
data/ref/universe/                                        # PIT S&P 500 tables + sources
data/state/                                               # collector state, heartbeat, locks
data/ref/company_tickers.json                             # cached SEC ticker map
```
Large reference data not needed day to day (the transcripts version the backfill used, full SEC
companyfacts, the embedding model, old run logs) lives in an archive described in `CLAUDE.md`.

## Item timing, in one paragraph
`published_at` for EDGAR is the filing's acceptance time from its SGML header (Eastern,
converted DST-aware to UTC). The submissions-API `acceptanceDateTime` is **not** used, because
its timezone handling differs by filer, and neither is `filingDate`. For news it is the vendor's
timestamp, which must carry an offset. `tradable_at` is `max(published_at, first_seen_at)` for
live items and `published_at + lag` for backfilled ones. Either way it is pushed to the next
NYSE regular-session open when it falls outside trading hours.
