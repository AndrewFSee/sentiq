"""Publish SentIQ's scored 8-Ks / earnings calls, collector health and research status to the hub.

Read-only over this project's store (nothing is fetched, scored or written back). The hub
(market-compass, page ``modules/sentiq/page.py``) only displays what this writes under
``<HUB_DATA_DIR>/sentiq/``:

* ``filings``  -- one row per scored (story founder, company): 8-Ks (form 8-K / 8-K/A) and
  earnings-call transcripts, unmasked canonical Haiku scores (replicate 0, first ok draw).
* ``daily``    -- per ET date x group x event type: count and mean sentiment (relevance >=
  secondary), from 2019 on.
* ``fed``      -- one row per scored Fed document (statements, minutes, Governors' speeches and
  testimony; fed_scores, prompt fed_v1): stance, rate path, inflation / growth / labor.
* ``health``   -- collector state per source, cycle durations (24 h), alerts (7 days), LLM spend.
* ``research`` -- the pre-registered tests and their results (a static table kept here, in
  step with the decisions log in CLAUDE.md), plus holdout progress as event *counts* only.

**No returns are computed or published.** The holdout (from 2026-07-01) must stay unseen until
the pre-registered looks (H3 stage 1 on/after 2027-05-31; H1, H2, H3 stage 2 on/after
2027-08-31), so the page shows what the model read, never how prices moved afterwards.

The hub's ``run_jobs.py`` runs this with this project's venv and the hub folder on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from newsfeed.config import load_config  # noqa: E402
from newsfeed.storage import Store  # noqa: E402
from newsfeed.universe_pit import PitUniverse  # noqa: E402

try:
    from core import store as hub  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit(
        "Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
        "or put the market-compass folder on PYTHONPATH."
    )

MODULE = "sentiq"
ET = "America/New_York"
SOURCES = ("edgar", "transcripts")
HOLDOUT_START = date(2026, 7, 1)
FED_LIVE_SINCE = "2026-11-18T22:58:00+00:00"  # fed added to collect.sources (decisions log)
# S&P 500 companies on dates outside their membership spell (ingest margins; scores kept under a
# CIK's current canonical ticker, e.g. FB-era filings under META's ticker history)
OUTSIDE = "S&P 500 (outside membership)"

# The pre-registered tests (CLAUDE.md decisions log). Primary = the pre-registered cell(s); "p" is
# the nominal two-sided bootstrap p of the primary cell(s). Update when a test is added.
RESEARCH = [
    (
        "1",
        "2026-10-11",
        "Earnings 8-Ks: LLM sentiment -> drift",
        "IC h1 +0.018 (p 0.10), h2 +0.032 (p 0.018)",
        "nominal pass, no BH discovery",
    ),
    (
        "2",
        "2026-10-12",
        "Small-cap tercile, all 8-Ks",
        "IC h1 +0.013 (p 0.26), h2 +0.017 (p 0.17)",
        "fail",
    ),
    (
        "3",
        "2026-10-12",
        "Sentiment x novelty (embeddings)",
        "IC h1 +0.006 (p 0.39), h2 +0.013 (p 0.08)",
        "fail",
    ),
    (
        "4",
        "2026-10-12",
        "Vendor news (Massive), masked",
        "IC h1 +0.002 (p 0.81), h2 +0.003 (p 0.72)",
        "fail",
    ),
    (
        "4b",
        "2026-10-12",
        "Vendor news, cleaned + event types",
        "IC h1 -0.001 (p 0.99), h2 +0.003 (p 0.75)",
        "fail",
    ),
    (
        "A",
        "2026-10-12",
        "Leakage: masked vs unmasked earnings 8-Ks",
        "diff +0.000 [-0.005, +0.006]",
        "no evidence of recall",
    ),
    (
        "5",
        "2026-10-13",
        "Earnings-call transcripts, masked",
        "IC h1 +0.029 (p 0.026), h2 +0.035 (p 0.024)",
        "nominal pass; mostly repeats the release (5b n.s.)",
    ),
    (
        "6",
        "2026-10-15",
        "LLM earnings surprise",
        "IC h1 +0.033 (p 0.08), h2 +0.040 (p 0.044)",
        "weak; adds nothing beyond sentiment",
    ),
    ("7", "2026-10-15", "8-K attention -> volatility", "partial IC -0.001 (p 0.64)", "fail"),
    (
        "8",
        "2026-10-15",
        "8-K item-code event study (45 tests)",
        "no BH discovery (min p 0.052)",
        "fail",
    ),
    (
        "10",
        "2026-10-16",
        "10-K/10-Q text change (Lazy Prices)",
        "IC h20 -0.011 (p 0.69), h60 -0.029 (p 0.33)",
        "fail",
    ),
    (
        "RC",
        "2026-10-24",
        "Recall check: walk-forward student vs Haiku",
        "student +0.009 vs Haiku +0.025",
        "ambiguous",
    ),
    (
        "11",
        "2026-11-10",
        "MD&A tone change (LM, FinBERT)",
        "IC h20 -0.016 (p 0.37), h60 -0.017 (p 0.17)",
        "fail",
    ),
    (
        "12",
        "2026-11-18",
        "Aggregate earnings tone (same-company change) -> next month's S&P 500",
        "monthly corr -0.058 (p 0.55)",
        "fail; raw tone leans negative (-0.25, p 0.009, mostly 2020-22)",
    ),
    (
        "13",
        "2026-11-18",
        "Materiality label -> size of the price reaction",
        "partial corr +0.41 (p < 0.001); within type +0.15",
        "pass: tracks move size, not direction",
    ),
]
HOLDOUT = [
    # id, test, look date(s), pass rule, planned n
    (
        "H1",
        "Earnings 8-Ks (S&P 500): 1d LLM sentiment, close entry, 2-day market-adjusted IC",
        "on/after 2027-08-31",
        "IC > 0, one-sided p < 0.025",
        2600,
    ),
    (
        "H2",
        "8-Ks the model labels guidance (S&P 500), same cell",
        "on/after 2027-08-31",
        "IC > 0, one-sided p < 0.025",
        210,
    ),
    (
        "H3",
        "H1 replicated on the S&P 400 + 600 (frozen 2026-07-01 list)",
        "stage 1 on/after 2027-05-31, stage 2 on/after 2027-08-31",
        "stage 1 one-sided p < 0.012, stage 2 p < 0.023",
        5000,
    ),
]


def _ts(v) -> str | None:
    return None if v is None else pd.Timestamp(v).isoformat()


def load_filings(store: Store, cfg) -> pl.DataFrame:
    rel = store.items_relation(list(SOURCES))
    items = store.query(
        f"""
        SELECT i.item_id, i.source, i.title, i.url, i.meta, i.published_at, i.first_seen_at,
               i.tradable_at, i.ingest_mode
        FROM {rel} i JOIN item_story st ON st.item_id = i.item_id AND st.story_id = i.item_id
        """
    )
    scores = store.query(
        """
        SELECT item_id, ticker, cik, relevance, sentiment, confidence, event_type, novelty,
               materiality, surprise_vs_expectations AS surprise, rationale, scored_at
        FROM scores
        WHERE status = 'ok' AND NOT masked AND replicate = 0 AND model = $m
        QUALIFY row_number() OVER (PARTITION BY item_id, ticker ORDER BY scored_at DESC) = 1
        """,
        {"m": cfg.scoring.model},
    )
    df = scores.join(items, on="item_id", how="inner")
    meta = [json.loads(m) if m else {} for m in df["meta"].to_list()]
    form = [m.get("form") or "8-K" for m in meta]
    doc = [
        "Earnings call" if s == "transcripts" else f
        for s, f in zip(df["source"], form, strict=True)
    ]
    codes = [",".join(m.get("items") or []) for m in meta]
    company_meta = [m.get("company") for m in meta]
    df = df.with_columns(
        pl.Series("doc", doc),
        pl.Series("items", codes),
        pl.Series("company_meta", company_meta, dtype=pl.Utf8),
        pl.col("tradable_at").dt.convert_time_zone(ET).dt.date().alias("date"),
    ).drop("meta")
    # 8-Ks (incl. 8-K/A) and calls only
    df = df.filter(pl.col("doc").is_in(["8-K", "8-K/A", "Earnings call"]))

    # group: S&P 500 member on the date (stored PIT tables), else Nasdaq extra, else S&P 400/600
    sp = PitUniverse.load(cfg.universe.pit_dir)
    df = df.with_columns(sp.is_member(df).alias("_sp500"))
    extras = {e.cik for e in cfg.universe.extra_members}
    h3 = {}
    if cfg.universe.extra_members_file:
        f = pl.read_csv(ROOT / cfg.universe.extra_members_file, columns=["cik", "index"])
        h3 = {int(c): ("S&P 400" if i == "sp400" else "S&P 600") for c, i in f.iter_rows()}
    group = [
        "S&P 500" if m else ("Nasdaq-100 extra" if c in extras else h3.get(c, OUTSIDE))
        for m, c in zip(df["_sp500"], df["cik"], strict=True)
    ]
    full = PitUniverse.load_for(cfg.universe)
    titles = full.titles()
    df = df.with_columns(
        pl.Series("group", group),
        pl.Series("company", [titles.get(int(c)) for c in df["cik"]], dtype=pl.Utf8),
    ).drop("_sp500")
    df = df.with_columns(pl.coalesce("company", "company_meta").alias("company")).drop(
        "company_meta"
    )
    sectors = full.sector_frame().rename({"valid_from": "date"}).sort("date")
    df = (
        df.sort("ticker", "date")
        .join_asof(
            sectors.sort("ticker", "date"),
            on="date",
            by="ticker",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(pl.col("sector").fill_null("Unknown"))
    )
    return df.sort("tradable_at", descending=True)


FED_DOC = {
    "fomc_statement": "FOMC statement",
    "fomc_minutes": "FOMC minutes",
    "speech": "Speech",
    "testimony": "Testimony",
}


def load_fed(store: Store, cfg) -> pl.DataFrame:
    """One row per scored Fed document (fed_scores, first ok draw, the configured model)."""
    rel = store.items_relation(["fed"])
    df = store.query(
        f"""
        SELECT i.item_id, f.kind, i.title, i.url, i.meta, i.published_at, i.first_seen_at,
               i.tradable_at, i.ingest_mode, f.scored_at, f.policy_relevance, f.stance,
               f.rate_path, f.inflation, f.growth, f.labor, f.confidence, f.rationale
        FROM fed_scores f JOIN {rel} i ON i.item_id = f.item_id AND i.text_hash = f.text_hash
        WHERE f.status = 'ok' AND f.model = $m
        """,
        {"m": cfg.scoring.model},
    )
    meta = [json.loads(m) if m else {} for m in df["meta"].to_list()]
    speaker = [
        "Committee" if k.startswith("fomc_") else (m.get("speaker") or "")
        for k, m in zip(df["kind"], meta, strict=True)
    ]
    return (
        df.with_columns(
            pl.Series("speaker", speaker, dtype=pl.Utf8),
            pl.col("kind").replace_strict(FED_DOC, default=pl.col("kind")).alias("doc"),
        )
        .drop("meta")
        .sort("tradable_at", descending=True)
    )


def usable_at() -> pl.Expr:
    """When the *score* was usable: tradable_at, or max(tradable_at, scored_at) for live items
    (the research's t_use)."""
    return (
        pl.when(pl.col("ingest_mode") == "live")
        .then(pl.max_horizontal("tradable_at", "scored_at"))
        .otherwise(pl.col("tradable_at"))
    )


def effect_window(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Adds ``fades_by``: the close two sessions after the first 15:50 decision at or after the
    score was usable -- the end of test 1's primary cell (close entry, h = 2), where the
    pre-holdout signal of earnings releases and calls was concentrated (delay sensitivity:
    mostly gone 2-5 sessions later). Only for earnings / guidance 8-Ks and earnings calls with
    relevance >= secondary; null elsewhere (no signal was found for other documents)."""
    import exchange_calendars as xcals
    import numpy as np

    cal = xcals.get_calendar(cfg.market.calendar, start="2018-01-01")
    utc_closes = pd.to_datetime(cal.closes, utc=True).dt.tz_convert(None)  # naive UTC
    closes = utc_closes.to_numpy().astype("datetime64[us]").astype(np.int64)
    lag = cfg.features.decision_lag_minutes * 60 * 1_000_000
    t_use = df.select(usable_at().dt.epoch("us").alias("t"))["t"].to_numpy()
    idx = np.searchsorted(closes - lag, t_use, side="left") + 2
    ok = idx < len(closes)
    fades = np.where(ok, closes[np.minimum(idx, len(closes) - 1)], 0)
    eligible = pl.col("relevance").is_in(["primary", "secondary"]) & (
        (
            pl.col("doc").is_in(["8-K", "8-K/A"])
            & pl.col("event_type").is_in(["earnings", "guidance"])
        )
        | ((pl.col("doc") == "Earnings call") & (pl.col("event_type") == "earnings"))
    )
    f = pl.Series("_f", fades).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    return df.with_columns(
        pl.when(eligible & pl.Series(ok)).then(f).otherwise(None).alias("fades_by")
    )


def tone_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Per month: S&P 500 earnings 8-K tone vs the same companies' releases a year earlier
    (test 12's measure; mean of sentiment minus the same CIK's release 320-410 days earlier),
    plus the raw monthly mean. Tone only; no returns."""
    from newsfeed.eval.aggregate import monthly_tone

    rel = (
        df.filter(
            pl.col("doc").is_in(["8-K", "8-K/A"])
            & (pl.col("event_type") == "earnings")
            & (pl.col("group") == "S&P 500")
            & pl.col("relevance").is_in(["primary", "secondary"])
            & pl.col("sentiment").is_not_null()
        )
        .with_columns(usable_at().alias("t_use"))
        .unique(["item_id", "cik"])
        .select("item_id", "cik", "t_use", "sentiment")
    )
    return monthly_tone(rel, min_pairs=10).select("month", "yoy", "n_pairs", "raw", "n")


def daily(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("relevance").is_in(["primary", "secondary"]))
        .group_by("date", "group", "doc", "event_type")
        .agg(pl.len().alias("n"), pl.col("sentiment").mean().alias("mean_sentiment"))
        .sort("date")
    )


def holdout_progress(df: pl.DataFrame) -> dict[str, int]:
    """Event counts so far in the holdout (approximate sample sizes; no outcomes)."""
    q = df.filter(
        (pl.col("doc") != "Earnings call")
        & pl.col("relevance").is_in(["primary", "secondary"])
        & (pl.col("date") >= HOLDOUT_START)
    )

    def n(frame: pl.DataFrame) -> int:
        return frame.select("ticker", "date").unique().height

    return {
        "H1": n(q.filter((pl.col("event_type") == "earnings") & (pl.col("group") == "S&P 500"))),
        "H2": n(q.filter((pl.col("event_type") == "guidance") & (pl.col("group") == "S&P 500"))),
        "H3": n(
            q.filter(
                (pl.col("event_type") == "earnings") & pl.col("group").is_in(["S&P 400", "S&P 600"])
            )
        ),
    }


def health(store: Store, now: datetime) -> dict:
    state = {}
    for src in (*SOURCES, "fed"):
        p = store.state_dir / f"collect-{src}.json"
        if p.exists():
            s = json.loads(p.read_text())
            state[src] = {
                k: s.get(k)
                for k in (
                    "last_poll_ok_at",
                    "last_new_item_at",
                    "consecutive_failures",
                    "last_error",
                    "open_alerts",
                    "items_written",
                    "polls",
                )
            }
    hb_path = store.state_dir / "heartbeat.json"
    hb = json.loads(hb_path.read_text()) if hb_path.exists() else {}
    runs = store.ops_query(
        """
        SELECT started_at, finished_at, status, stats FROM runs
        WHERE command = 'collect' AND started_at >= $since ORDER BY started_at
        """,
        {"since": now - timedelta(hours=24)},
    ).to_pandas()
    dur = (
        pd.to_datetime(runs["finished_at"]) - pd.to_datetime(runs["started_at"])
    ).dt.total_seconds() / 60
    # a cycle still running has no stats (NULL -> NaN in pandas) and no finished_at
    written = [json.loads(s).get("written", 0) if isinstance(s, str) else 0 for s in runs["stats"]]
    alerts = store.ops_query(
        "SELECT raised_at, source, kind, message FROM alerts WHERE raised_at >= $since "
        "ORDER BY raised_at DESC",
        {"since": now - timedelta(days=7)},
    ).to_pandas()
    # scoring failures (API errors are not stored as scores, so they are retried; the collector's
    # health check does not watch them -- 401s went unnoticed for 3 days, 2026-11-09..12)
    err = store.ops_query(
        """
        SELECT cache_key, min(called_at) AS first_err, max(called_at) AS last_err,
               any_value(error) AS error
        FROM llm_usage WHERE status = 'error' AND called_at >= $since GROUP BY cache_key
        """,
        {"since": now - timedelta(days=7)},
    )
    scored = (
        set(
            store.query(
                "SELECT cache_key FROM scores "
                "WHERE status = 'ok' AND cache_key IN (SELECT unnest($k))",
                {"k": err["cache_key"].to_list()},
            )["cache_key"].to_list()
        )
        if err.height
        else set()
    )
    unresolved = err.filter(~pl.col("cache_key").is_in(list(scored)))
    month_start = now.astimezone().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    spend = store.ops_query(
        """
        SELECT coalesce(sum(cost_usd) FILTER (WHERE called_at >= $m), 0) AS month,
               coalesce(sum(cost_usd) FILTER (WHERE called_at >= $d30), 0) AS last30
        FROM llm_usage
        """,
        {"m": month_start, "d30": now - timedelta(days=30)},
    ).row(0, named=True)
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "heartbeat_at": hb.get("updated_at"),
        "problems": hb.get("problems") or [],
        "sources": state,
        "cycles_24h": len(runs),
        "cycles_failed_24h": int((~runs["status"].isin(["ok", "running"])).sum()),
        "cycle_minutes_median": float(dur.median()) if len(dur) else None,
        "cycle_minutes_max": float(dur.max()) if len(dur) else None,
        "items_written_24h": int(sum(written)),
        "cycles": [
            {"started_at": _ts(s), "minutes": round(float(m), 2), "written": int(w), "status": st}
            for s, m, w, st in zip(runs["started_at"], dur, written, runs["status"], strict=True)
            if pd.notna(m)
        ],
        "alerts_7d": [
            {
                "raised_at": _ts(r.raised_at),
                "source": r.source,
                "kind": r.kind,
                "message": r.message,
            }
            for r in alerts.itertuples()
        ],
        "scoring_errors_7d": int(err.height),
        "scoring_unresolved_7d": int(unresolved.height),
        "scoring_last_error_at": _ts(err["last_err"].max()) if err.height else None,
        "scoring_last_error": (err.sort("last_err")["error"][-1] or "")[:200]
        if err.height
        else None,
        "spend_month_usd": float(spend["month"]),
        "spend_30d_usd": float(spend["last30"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", action="store_true", help="accepted for the hub; reads the local store only"
    )
    parser.parse_args()

    cfg = load_config(None)
    now = datetime.now(UTC)
    with Store(cfg.paths) as store:
        df = effect_window(load_filings(store, cfg), cfg)
        fed = load_fed(store, cfg)
        h = health(store, now)
    out = df.select(
        "item_id",
        "source",
        "doc",
        "ticker",
        "cik",
        "company",
        "group",
        "sector",
        "items",
        "title",
        "url",
        "published_at",
        "first_seen_at",
        "tradable_at",
        "ingest_mode",
        "scored_at",
        "fades_by",
        "date",
        "relevance",
        "event_type",
        "sentiment",
        "confidence",
        "materiality",
        "surprise",
        "novelty",
        "rationale",
    )
    hub.save_frame(MODULE, "filings", out.to_pandas())
    hub.save_frame(MODULE, "daily", daily(df).to_pandas())
    hub.save_frame(MODULE, "tone_monthly", tone_monthly(df).to_pandas())
    hub.save_frame(MODULE, "fed", fed.to_pandas())
    hub.save_json(MODULE, "health", h)
    prog = holdout_progress(df)
    hub.save_frame(
        MODULE,
        "research",
        pd.DataFrame(RESEARCH, columns=["test", "date", "question", "primary", "verdict"]),
    )
    hub.save_json(
        MODULE,
        "holdout",
        {
            "start": HOLDOUT_START.isoformat(),
            "model": cfg.scoring.model,
            "hypotheses": [
                {"id": i, "test": t, "look": lk, "rule": r, "planned_n": pn, "n_so_far": prog[i]}
                for i, t, lk, r, pn in HOLDOUT
            ],
        },
    )
    hub.save_json(
        MODULE,
        "summary",
        {
            "as_of": now.isoformat(timespec="seconds"),
            "filings": int(out.height),
            "latest": _ts(out["tradable_at"].max()) if out.height else None,
            "model": cfg.scoring.model,
            "prompt_version": cfg.scoring.prompt_version,
            "fed_documents": int(fed.height),
            "fed_prompt_version": cfg.scoring.fed_prompt_version,
            "fed_live_since": FED_LIVE_SINCE,
        },
    )
    print(
        f"Published {out.height:,} scored filings/calls (latest {out['tradable_at'].max()}); "
        f"{h['cycles_24h']} collect cycles in 24 h; holdout counts {prog}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
