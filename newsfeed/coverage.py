"""Data coverage: which (source, ticker, tradable-time) ranges were actually collected.

Without this, "no rows" is ambiguous between *no news* and *not collected* -- e.g. a ticker
backfilled from 2025 would look news-free in 2024 and poison 60-day volume baselines. Features
for windows not fully covered are null, not zero.

Coverage is evaluated on the **tradable_at axis**, half-open [start, end):
* A backfill collects items *published* in [start, end). Adjacent/overlapping backfills are
  first merged on that published axis; a merged range [S, E) then covers tradable time
  [push(S + lag), E): every item with tradable_at < E was published < E (tradable >= published)
  and is stored, while items published before S -- not collected -- can become tradable as late
  as push(S + lag). Merging first avoids phantom gaps where one backfill ends and the next
  begins (the earlier one already holds the items published before the later one's start).
* Live collection: each successful poll extends coverage [previous successful poll, now] on the
  tradable axis; a stall leaves a gap (outage periods are excluded from research use).
Feature windows are (lo, hi] with lo exclusive, so an event exactly at a coverage start is never
relied on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta

from newsfeed.storage import Store
from newsfeed.timeutil import MarketCalendar, ensure_utc, utcnow

Interval = tuple[datetime, datetime]


def record_backfill(
    store: Store,
    source: str,
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    *,
    lag: timedelta,
    calendar: MarketCalendar,
    run_id: str,
) -> None:
    """Record a backfill over published time [start, end)."""
    start, end = ensure_utc(start), ensure_utc(end)
    if end <= start:
        return
    trad_start = calendar.push_to_session(start + lag)
    with store.ops() as con:
        con.executemany(
            "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                [source, t, start, end, trad_start, "backfill", run_id, utcnow()]
                for t in sorted(set(tickers))
            ],
        )


def record_live(
    store: Store, source: str, tickers: Iterable[str], start: datetime, end: datetime, run_id: str
) -> None:
    """Record that ``source`` was watched continuously over tradable time [start, end).
    Extends the existing interval ending at ``start`` instead of adding rows."""
    start, end = ensure_utc(start), ensure_utc(end)
    if end <= start:
        return
    todo = sorted(set(tickers))
    with store.ops() as con:
        extendable = {
            r[0]
            for r in con.execute(
                "SELECT ticker FROM coverage WHERE source = ? AND kind = 'live' AND end_at = ?",
                [source, start],
            ).fetchall()
        }
        con.execute(
            "UPDATE coverage SET end_at = ?, recorded_at = ? "
            "WHERE source = ? AND kind = 'live' AND end_at = ?",
            [end, utcnow(), source, start],
        )
        rest = [t for t in todo if t not in extendable]
        if rest:
            con.executemany(
                "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [[source, t, start, end, start, "live", run_id, utcnow()] for t in rest],
            )


def _merge(intervals: list[tuple[datetime, datetime, datetime]]) -> list[Interval]:
    """Merge (axis_start, end, tradable_start) rows on their own axis; keep the first row's
    tradable start for each merged group."""
    out: list[list[datetime]] = []
    for s, e, ts in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, ts])
    return [(ts, e) for _, e, ts in out]


def merged(store: Store, source: str) -> dict[str, list[Interval]]:
    """ticker -> sorted, merged tradable-time intervals for one source."""
    df = store.ops_query(
        "SELECT ticker, kind, start_at, end_at, tradable_start_at FROM coverage WHERE source = ?",
        [source],
    )
    by_ticker: dict[str, dict[str, list]] = defaultdict(lambda: {"backfill": [], "live": []})
    for r in df.to_dicts():
        by_ticker[r["ticker"]][r["kind"]].append(
            (ensure_utc(r["start_at"]), ensure_utc(r["end_at"]), ensure_utc(r["tradable_start_at"]))
        )
    out: dict[str, list[Interval]] = {}
    for tk, kinds in by_ticker.items():
        tradable = _merge(kinds["backfill"]) + [(s, e) for s, e, _ in kinds["live"]]
        out[tk] = _merge([(s, e, s) for s, e in tradable])
    return out


def is_covered(intervals: Sequence[Interval], lo: datetime, hi: datetime) -> bool:
    """True if the window (lo, hi] lies inside one interval [s, e): s <= lo and hi < e."""
    return any(s <= lo and hi < e for s, e in intervals)


def rebuild_from_runs(
    store: Store, calendar: MarketCalendar, lags: dict[str, int], universe: Sequence[str]
) -> int:
    """Recreate backfill coverage from the `runs` log, for data ingested before coverage
    tracking existed (ok/partial backfill runs; failed filings were logged and retried).
    Runs without a --tickers subset are assumed to have covered ``universe`` -- only correct if
    the configured universe has not changed since. Idempotent: clears backfill rows first."""
    with store.ops() as con:
        con.execute("DELETE FROM coverage WHERE kind = 'backfill'")
    runs = store.ops_query(
        "SELECT run_id, params FROM runs WHERE command = 'backfill' "
        "AND status IN ('ok', 'partial') ORDER BY started_at"
    ).to_dicts()

    def day(s: str) -> datetime:
        return datetime.combine(date.fromisoformat(s), datetime.min.time(), tzinfo=UTC)

    for r in runs:
        p = json.loads(r["params"])
        subset = [t.strip().upper() for t in (p.get("tickers") or "").split(",") if t.strip()]
        src = p.get("source", "edgar")
        record_backfill(
            store,
            src,
            p.get("covered_tickers") or subset or list(universe),
            day(p["start"]),
            day(p["end"]),
            lag=timedelta(seconds=lags[src]),
            calendar=calendar,
            run_id=r["run_id"],
        )
    return len(runs)
