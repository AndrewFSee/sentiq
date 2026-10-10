# newsfeed

Turns financial news and SEC filings into **point-in-time-correct**, LLM-scored features, to test
whether news carries incremental tradable signal. A research tool, not a trading system.

Status: **Milestones 1–3** done (ingestion, live collection, dedupe, ticker linking, LLM
scoring). See `CLAUDE.md` for conventions and the PIT rules.

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
uv run newsfeed backfill --start 2026-07-01 --end 2026-09-01 --source news_api  # fixture vendor
```
Backfill is idempotent and resumable. Re-running skips stored items without re-downloading
them, and failures are logged to the `failures` table and retried on the next run. Each run
ends by refreshing the derived tables (ticker links and stories).

> The `news_api` source currently reads **synthetic fixtures** (`tests/fixtures/news/`). Point
> it at a scratch data directory (via `--config`) if you don't want them mixed into real data.

### 2. Live collection
```bash
uv run newsfeed collect                  # loop every collect.interval_seconds
uv run newsfeed collect --once           # one cycle, for cron (state persists in data/state/)
uv run newsfeed collect --check          # health from saved state; exit 1 if unhealthy
uv run newsfeed collect --sources edgar,news_api --max-cycles 10
```
Example cron entry (every minute, plus a health check that alerts via your mailer):
```
* * * * *   cd /path/to/sentiq && uv run newsfeed collect --once >> data/collect.log 2>&1
*/5 * * * * cd /path/to/sentiq && uv run newsfeed collect --check || <notify>
```
Each cycle polls each source and ingests new items, refreshes the derived tables, evaluates
health and writes `data/state/heartbeat.json`. Health problems:

| kind | meaning |
|---|---|
| `silent_feed` | no new entries from **any company** in the source feed for N market-open minutes, i.e. an outage rather than "no news" |
| `poll_failing` | N consecutive polls failed |
| `stalled` | no successful poll for N intervals during market hours |

Alerts are logged and written to the `alerts` table, and they clear on recovery.
A `collect` loop and a `backfill` can run at the same time against the same data directory.

### 3. Derived tables
```bash
uv run newsfeed derive          # incremental (runs automatically after backfill/collect)
uv run newsfeed derive --full   # rebuild stories from scratch
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

### 7. Features (M4), eval (M5)
Not yet implemented; `newsfeed features|eval` currently exit with a notice.

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
data/state/                                               # collector state, heartbeat, locks
data/ref/company_tickers.json                             # cached SEC ticker map
```

## Item timing, in one paragraph
`published_at` for EDGAR is the filing's acceptance time from its SGML header (Eastern,
converted DST-aware to UTC). The submissions-API `acceptanceDateTime` is **not** used, because
its timezone handling differs by filer, and neither is `filingDate`. For news it is the vendor's
timestamp, which must carry an offset. `tradable_at` is `max(published_at, first_seen_at)` for
live items and `published_at + lag` for backfilled ones. Either way it is pushed to the next
NYSE regular-session open when it falls outside trading hours.
