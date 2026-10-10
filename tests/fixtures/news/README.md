# News fixtures

`articles.jsonl` holds **synthetic** articles in the generic vendor schema read by
`newsfeed.ingest.news_api.article_from_generic`. Outlets and URLs are fictional (`.test`
domains). Lines starting with `//` are comments.

| field | meaning |
|---|---|
| `id` | vendor article id |
| `published_at` | ISO-8601 **with offset** (naive values are rejected) |
| `visible_at` | optional; FixtureVendor hides the record until this time (simulates late indexing) |
| `url`, `title`, `body`, `summary`, `publisher` | as delivered by the vendor |
| `tickers` | vendor ticker tags |

Each record exercises one case: a1–a4 are one story (syndicated exact copy, near-duplicate
rewrite, same URL updated); a5 is a different Apple story; a6 is multi-company with name-only
mentions; a7 uses cashtags plus a non-universe ticker; a8 is outside the universe; a9 has a naive
timestamp; a10 lands on a weekend; a11 is indexed late; a12 contains "apple" as a common noun;
a13 has an exchange tag; a14 and a15 are headline-only.
