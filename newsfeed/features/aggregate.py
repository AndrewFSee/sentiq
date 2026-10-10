"""Point-in-time news features per (ticker, as-of time T), over windows (T - w, T].

Inputs come only through ``Store.items_as_of(T_max)`` and are then filtered per T by each
event's *availability* time, so ``build_features(T)`` uses nothing that was not knowable at T:

* volume features (n_items, n_stories, n_outlets, abnormal volume): every linked item, available
  at ``tradable_at``;
* LLM features: story founders with an ok primary score on their current text, available at
  ``tradable_at`` for backfilled items and at ``max(tradable_at, scored_at)`` for live items
  (in live use a score only exists once computed; backfill scores are a deterministic function
  of text that was available at ``tradable_at``);
* baseline features (LM, FinBERT): story founders, available at ``tradable_at``.

Links follow ``scoring.link_policy`` (the same set the LLM scores) and fan out to every ticker
of the company (GOOG and GOOGL get the same events). LLM sentiment aggregates skip relevance
below ``features.min_relevance``; baselines have no relevance and use all founder events.

Abnormal volume: the window's story count, z-scored against the counts of the K preceding
non-overlapping windows of the same length within ``baseline_days`` (all strictly before the
current window).

Coverage: a window whose span is not inside collected coverage (``coverage.py``) for every
source in ``features.coverage_sources`` gets null features -- "not collected" is not "no news".
volume_z additionally needs its whole baseline span covered.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from newsfeed import coverage
from newsfeed.config import AppConfig
from newsfeed.features.embeddings import max_prior_similarity
from newsfeed.scoring.schema import EVENT_TYPES
from newsfeed.scoring.targets import canonical_tickers
from newsfeed.storage import Store
from newsfeed.timeutil import MarketCalendar, ensure_utc

FEATURE_VERSION = 1
RELEVANCE_RANK = {"passing": 0, "secondary": 1, "primary": 2}
_US = 1_000_000  # microseconds per second


@dataclass(frozen=True)
class FeatureSources:
    """Which score/baseline versions feed the features (part of the feature-set id)."""

    llm_model: str
    llm_prompt_hash: str
    lm_version: str | None
    finbert_version: str | None
    embedding_model: str | None = None


def feature_set_id(cfg: AppConfig, src: FeatureSources) -> str:
    blob = json.dumps(
        [
            FEATURE_VERSION,
            cfg.features.model_dump(mode="json", exclude={"lm_dictionary"}),
            cfg.scoring.link_policy.model_dump(),
            src.__dict__,
        ],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def as_of_grid(
    calendar: MarketCalendar, start: date, end: date, grid: str, lag_minutes: float
) -> list[datetime]:
    """One as-of time per regular session in [start, end): close - lag or open + lag."""
    out = []
    d = start
    lag = timedelta(minutes=lag_minutes)
    while d < end:
        bounds = calendar.session_bounds(d)
        if bounds is not None:
            open_, close = bounds
            out.append(close - lag if grid == "session_close" else open_ + lag)
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------- loading
def _qualified_links(store: Store, cfg: AppConfig) -> pl.DataFrame:
    pol = cfg.scoring.link_policy
    return store.query(
        """
        SELECT DISTINCT item_id, cik FROM item_tickers
        WHERE method IN (SELECT unnest($methods))
           OR (method = 'name_body' AND n_mentions >= $min_body)
        """,
        {"methods": pol.methods, "min_body": pol.name_body_min_mentions},
    )


def load_events(
    store: Store,
    cfg: AppConfig,
    as_of_max: datetime,
    cik_to_tickers: dict[int, list[str]],
    src: FeatureSources,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(volume rows, founder event rows), one row per (item, ticker), times in UTC micros."""
    items = store.items_as_of(
        as_of_max,
        columns="item_id, publisher, tradable_at, ingest_mode, text_hash",
    )
    if items.height == 0:
        empty = pl.DataFrame()
        return empty, empty
    links = _qualified_links(store, cfg)
    stories = store.query("SELECT item_id, story_id FROM item_story")
    fan = pl.DataFrame(
        [{"cik": c, "ticker": t} for c, ts in cik_to_tickers.items() for t in ts],
        schema={"cik": pl.Int64, "ticker": pl.Utf8},
    )
    canon = canonical_tickers(cfg, cik_to_tickers)
    canon_df = pl.DataFrame(
        {"cik": list(canon), "canon_ticker": list(canon.values())},
        schema={"cik": pl.Int64, "canon_ticker": pl.Utf8},
    )
    base = (
        items.join(links.with_columns(pl.col("cik").cast(pl.Int64)), on="item_id")
        .join(stories, on="item_id", how="left")
        .join(fan, on="cik")
        .join(canon_df, on="cik")
        .with_columns(
            pl.col("tradable_at").dt.epoch("us").alias("t_trad"),
            pl.coalesce("story_id", "item_id").alias("story_id"),
        )
    )
    volume = base.select("ticker", "item_id", "story_id", "publisher", "t_trad")

    founders = base.filter(pl.col("item_id") == pl.col("story_id"))
    scores = store.query(
        """
        SELECT item_id, ticker AS canon_ticker, text_hash, sentiment, confidence, relevance,
               event_type, scored_at
        FROM scores
        WHERE status = 'ok' AND NOT masked AND replicate = 0
          AND model = $m AND prompt_hash = $h
        """,
        {"m": src.llm_model, "h": src.llm_prompt_hash},
    )
    ev = founders.join(scores, on=["item_id", "canon_ticker", "text_hash"], how="left")
    ev = ev.with_columns(
        pl.when(pl.col("sentiment").is_null())
        .then(None)
        .when(pl.col("ingest_mode") == "live")
        .then(pl.max_horizontal(pl.col("tradable_at"), pl.col("scored_at")).dt.epoch("us"))
        .otherwise(pl.col("t_trad"))
        .alias("t_llm")
    )
    for method, version in (("lm", src.lm_version), ("finbert", src.finbert_version)):
        cols = {"score": f"{method}_score"}
        if method == "lm":
            cols |= {"neg_prop": "lm_neg_prop", "unc_prop": "lm_unc_prop"}
        if version is None:
            ev = ev.with_columns([pl.lit(None, pl.Float64).alias(v) for v in cols.values()])
            continue
        b = store.query(
            f"SELECT item_id, text_hash, {', '.join(cols)} FROM baselines "
            "WHERE method = $m AND method_version = $v",
            {"m": method, "v": version},
        ).rename(cols)
        ev = ev.join(b, on=["item_id", "text_hash"], how="left")
    if src.embedding_model is not None:
        emb = store.query(
            "SELECT item_id, text_hash, vector FROM embeddings WHERE model = $m",
            {"m": src.embedding_model},
        )
        ev = ev.join(emb, on=["item_id", "text_hash"], how="left")
    return volume, ev


# --------------------------------------------------------------------- aggregation
def _window_stats(
    t: np.ndarray, s: np.ndarray, T: int, half_life_us: float, w: np.ndarray | None = None
) -> dict[str, float | None]:
    """Aggregates of values ``s`` at times ``t`` (already restricted to the window)."""
    n = len(s)
    if n == 0:
        return {"n": 0, "decay": 0.0, "mean": None, "maxabs": None, "disp": None}
    decay = float(np.sum(s * np.power(0.5, (T - t) / half_life_us)))
    if w is not None:
        mean = float(np.sum(s * w) / np.sum(w)) if np.sum(w) > 0 else None
    else:
        mean = float(np.mean(s))
    return {
        "n": n,
        "decay": decay,
        "mean": mean,
        "maxabs": float(np.max(np.abs(s))),
        "disp": float(np.std(s)) if n >= 2 else None,
    }


def _slice(times: np.ndarray, lo: int, hi: int) -> slice:
    """Indices with lo < t <= hi in a sorted array."""
    return slice(int(np.searchsorted(times, lo, "right")), int(np.searchsorted(times, hi, "right")))


def build_features(
    store: Store,
    cfg: AppConfig,
    as_of: Sequence[datetime],
    cik_to_tickers: dict[int, list[str]],
    src: FeatureSources,
    tickers: Sequence[str] | None = None,
    member_spans: dict[str, list[tuple[date, date | None]]] | None = None,
) -> pl.DataFrame:
    """Wide feature table: one row per (ticker, as_of); columns ``<window>_<feature>``.

    With ``member_spans`` (ticker -> [(start, end exclusive | None)] on the decision date in New
    York), rows are produced only while the ticker is a universe member; a ticker without
    spans gets none. Rows are built one ticker at a time (memory stays per ticker) and
    returned sorted by (as_of, ticker)."""
    as_of = sorted(ensure_utc(t, name="as_of") for t in as_of)
    if not as_of:
        return pl.DataFrame()
    volume, events = load_events(store, cfg, as_of[-1], cik_to_tickers, src)
    all_tickers = sorted({t for ts in cik_to_tickers.values() for t in ts})
    tickers = sorted(tickers) if tickers is not None else all_tickers
    fc = cfg.features
    min_rank = RELEVANCE_RANK[fc.min_relevance]
    baseline_us = fc.baseline_days * 86400 * _US
    cov = {  # ticker -> per required source -> [(start_us, end_us)]
        tk: [
            [(int(s.timestamp() * _US), int(e.timestamp() * _US)) for s, e in m.get(tk, [])]
            for m in (coverage.merged(store, source) for source in fc.coverage_sources)
        ]
        for tk in tickers
    }

    def covered(tk: str, lo: int, hi: int) -> bool:
        """(lo, hi] inside collected coverage for every required source."""
        return all(any(s <= lo and hi < e for s, e in ivs) for ivs in cov[tk])

    per_ticker: dict[str, dict[str, Any]] = {}
    for tk in tickers:
        v = volume.filter(pl.col("ticker") == tk).sort("t_trad") if volume.height else volume
        e = events.filter(pl.col("ticker") == tk) if events.height else events
        llm = (
            e.filter(
                pl.col("t_llm").is_not_null()
                & (pl.col("relevance").replace_strict(RELEVANCE_RANK, default=-1) >= min_rank)
            ).sort("t_llm")
            if e.height
            else e
        )
        lm = e.filter(pl.col("lm_score").is_not_null()).sort("t_trad") if e.height else e
        fb = e.filter(pl.col("finbert_score").is_not_null()).sort("t_trad") if e.height else e
        founders = e.sort("t_trad") if e.height else e
        per_ticker[tk] = {
            "v_t": v["t_trad"].to_numpy() if v.height else np.array([], dtype=np.int64),
            "v_story": v["story_id"].to_numpy() if v.height else np.array([]),
            "v_pub": v["publisher"].to_numpy() if v.height else np.array([]),
            "llm_t": llm["t_llm"].to_numpy() if llm.height else np.array([], dtype=np.int64),
            "llm_s": llm["sentiment"].to_numpy() if llm.height else np.array([]),
            "llm_c": llm["confidence"].to_numpy() if llm.height else np.array([]),
            "llm_ev": llm["event_type"].to_numpy() if llm.height else np.array([]),
            "lm_t": lm["t_trad"].to_numpy() if lm.height else np.array([], dtype=np.int64),
            "lm_s": lm["lm_score"].to_numpy() if lm.height else np.array([]),
            "lm_neg": lm["lm_neg_prop"].to_numpy() if lm.height else np.array([]),
            "lm_unc": lm["lm_unc_prop"].to_numpy() if lm.height else np.array([]),
            "fb_t": fb["t_trad"].to_numpy() if fb.height else np.array([], dtype=np.int64),
            "fb_s": fb["finbert_score"].to_numpy() if fb.height else np.array([]),
            "f_t": founders["t_trad"].to_numpy()
            if founders.height
            else np.array([], dtype=np.int64),
            "f_scored": founders["sentiment"].is_not_null().to_numpy()
            if founders.height
            else np.array([], dtype=bool),
        }
        per_ticker[tk].update(_novelty(tk, e, src, fc, covered))

    ny = ZoneInfo("America/New_York")
    as_of_dates = [t.astimezone(ny).date() for t in as_of]
    frames = []
    for tk in tickers:
        d = per_ticker[tk]
        if member_spans is None:
            ts = list(as_of)
        else:
            spans = member_spans.get(tk, [])
            ts = [
                t
                for t, dd in zip(as_of, as_of_dates, strict=True)
                if any(s0 <= dd and (e0 is None or dd < e0) for s0, e0 in spans)
            ]
        rows = []
        for T_dt in ts:
            T = int(T_dt.timestamp() * _US)
            row: dict[str, Any] = {"ticker": tk, "as_of": T_dt}
            for wname, hours in fc.windows_hours.items():
                w_us = int(hours * 3600 * _US)
                hl = fc.half_life_hours[wname] * 3600 * _US
                lo = T - w_us
                p = f"{wname}_"
                vs = _slice(d["v_t"], lo, T)
                row[p + "n_items"] = vs.stop - vs.start
                row[p + "n_stories"] = len(set(d["v_story"][vs]))
                row[p + "n_outlets"] = len(set(d["v_pub"][vs]))
                k = int(baseline_us // w_us)
                row[p + "volume_z"] = (
                    _volume_z(d["f_t"], T, w_us, k, fc.volume_sigma_floor)
                    if covered(tk, T - w_us * (k + 1), T)
                    else None
                )
                fs = _slice(d["f_t"], lo, T)
                ls = _slice(d["llm_t"], lo, T)
                st = _window_stats(d["llm_t"][ls], d["llm_s"][ls], T, hl, d["llm_c"][ls])
                row[p + "llm_n"] = st["n"]
                # share of founder events in the window (by tradable_at) that have a score
                n_f = fs.stop - fs.start
                row[p + "llm_coverage"] = float(np.mean(d["f_scored"][fs])) if n_f else None
                row[p + "llm_sent_decay"] = st["decay"]
                row[p + "llm_sent_cwmean"] = st["mean"]
                row[p + "llm_sent_maxabs"] = st["maxabs"]
                row[p + "llm_sent_disp"] = st["disp"]
                evs = d["llm_ev"][ls]
                for et in EVENT_TYPES:
                    row[p + f"llm_evt_{et.replace('&', '')}"] = int(np.sum(evs == et))
                ms = _slice(d["lm_t"], lo, T)
                st = _window_stats(d["lm_t"][ms], d["lm_s"][ms], T, hl)
                row[p + "lm_n"] = st["n"]
                row[p + "lm_tone_decay"] = st["decay"]
                row[p + "lm_tone_mean"] = st["mean"]
                row[p + "lm_tone_maxabs"] = st["maxabs"]
                row[p + "lm_tone_disp"] = st["disp"]
                row[p + "lm_neg_prop_mean"] = float(np.mean(d["lm_neg"][ms])) if st["n"] else None
                row[p + "lm_unc_prop_mean"] = float(np.mean(d["lm_unc"][ms])) if st["n"] else None
                if src.finbert_version is not None:
                    bs = _slice(d["fb_t"], lo, T)
                    st = _window_stats(d["fb_t"][bs], d["fb_s"][bs], T, hl)
                    row[p + "finbert_n"] = st["n"]
                    row[p + "finbert_decay"] = st["decay"]
                    row[p + "finbert_mean"] = st["mean"]
                if src.embedding_model is not None:
                    es = _slice(d["emb_t"], lo, T)
                    sims = d["emb_sim"][es]
                    sims = sims[~np.isnan(sims)]
                    row[p + "emb_n"] = len(sims)
                    row[p + "emb_novelty_mean"] = float(np.mean(1 - sims)) if len(sims) else None
                    row[p + "emb_n_novel"] = int(np.sum(sims < fc.novelty_sim_threshold))
                if not covered(tk, lo, T):  # not collected: unknown, not "no news"
                    row.update({c: None for c in row if c.startswith(p)})
            rows.append(row)
        if rows:
            frames.append(_typed(pl.DataFrame(rows, infer_schema_length=None)))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical").sort("as_of", "ticker")


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    """Fixed dtypes regardless of content: an all-null column must still be Float64, so builds
    over different periods have identical schemas (and compare/append cleanly)."""
    return df.with_columns(
        pl.col("as_of").dt.convert_time_zone("UTC"),
        *[
            pl.col(c).cast(pl.Int64 if _is_count(c) else pl.Float64)
            for c in df.columns
            if c not in ("ticker", "as_of")
        ],
    )


def _is_count(col: str) -> bool:
    name = col.split("_", 1)[1]
    return name in (
        "n_items",
        "n_stories",
        "n_outlets",
        "llm_n",
        "lm_n",
        "finbert_n",
        "emb_n",
        "emb_n_novel",
    ) or (name.startswith("llm_evt_"))


def _volume_z(f_t: np.ndarray, T: int, w_us: int, k: int, sigma_floor: float) -> float | None:
    """Story-count z-score of (T-w, T] vs the k preceding non-overlapping windows.
    The caller checks that all k+1 windows lie inside collected coverage."""
    if k < 2:
        return None
    current = _slice(f_t, T - w_us, T)
    n_now = current.stop - current.start
    edges = T - w_us * np.arange(k + 1, 0, -1)  # ascending: T-(k+1)w ... T-w
    idx = np.searchsorted(f_t, edges, "right")
    counts = np.diff(idx)  # each bucket (edge_i, edge_{i+1}]
    mu, sigma = float(np.mean(counts)), float(np.std(counts))
    return (n_now - mu) / max(sigma, sigma_floor)


def _novelty(
    tk: str, e: pl.DataFrame, src: FeatureSources, fc: Any, covered: Any
) -> dict[str, np.ndarray]:
    """Event-level max prior similarity for one ticker (NaN when the look-back isn't covered)."""
    empty = {"emb_t": np.array([], dtype=np.int64), "emb_sim": np.array([], dtype=np.float64)}
    if src.embedding_model is None or not e.height or "vector" not in e.columns:
        return empty
    ev = e.filter(pl.col("vector").is_not_null()).sort("t_trad")
    if not ev.height:
        return empty
    t = ev["t_trad"].to_numpy()
    vecs = np.asarray(ev["vector"].to_list(), dtype=np.float32)
    lookback = int(fc.novelty_lookback_days * 86400 * _US)
    sims = max_prior_similarity(t, vecs, lookback)
    ok = np.array([covered(tk, int(ti) - lookback, int(ti)) for ti in t])
    return {"emb_t": t, "emb_sim": np.where(ok, sims, np.nan)}
