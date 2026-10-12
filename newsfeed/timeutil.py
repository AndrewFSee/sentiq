"""Time handling: everything is tz-aware UTC; market-session logic lives here.

The single most important function is :func:`compute_tradable_at`. Every feature computed
"as of" T may only use items with ``tradable_at <= T``.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NY = ZoneInfo("America/New_York")
IngestMode = Literal["live", "backfill"]


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(dt: datetime, *, name: str = "datetime") -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {dt!r}")
    return dt.astimezone(UTC)


def parse_et_wall(s: str) -> datetime:
    """Parse an Eastern wall-clock timestamp ``YYYYMMDDHHMMSS`` (EDGAR SGML header) to UTC.

    EDGAR accepts filings 06:00-22:00 ET on business days, so the DST-ambiguous hour
    (01:00-02:00 on the November Sunday) cannot occur; we still refuse it explicitly.
    """
    naive = datetime.strptime(s.strip(), "%Y%m%d%H%M%S")  # noqa: DTZ007 - zone attached below
    local = naive.replace(tzinfo=NY)
    if local.replace(fold=1).utcoffset() != local.utcoffset():
        raise ValueError(f"ambiguous Eastern wall time {s!r} (DST fall-back hour)")
    return local.astimezone(UTC)


def edgar_json_literal(s: str) -> datetime:
    """``acceptanceDateTime`` from data.sec.gov/submissions, read literally as UTC.

    NOT a trustworthy acceptance time. Verified 2026-10 against SGML headers (see CLAUDE.md):
    the convention differs by filer --
      * most filers (MSFT, NVDA, GOOGL, ...): correct UTC;
      * some (all Apple filings checked, self- and agent-filed): *true UTC + Eastern offset*,
        i.e. the ET->UTC shift applied twice (16:30:28 EDT -> ``...T00:30:28.000Z`` next day).
    In every observed convention the literal value is >= the true time (late by 0, 4 or 5h),
    so it is safe as an *upper bound* for coarse filtering with a >=1 day margin, and nothing
    else. The authoritative time is the SGML header via :func:`parse_et_wall`.
    """
    naive = datetime.fromisoformat(s.strip().rstrip("Z").split(".")[0])
    return naive.replace(tzinfo=UTC)


def undo_double_offset(s: str) -> datetime:
    """Interpret a JSON acceptanceDateTime under the double-offset convention."""
    naive = edgar_json_literal(s)
    for hours in (4, 5):
        cand = naive - timedelta(hours=hours)
        if cand.astimezone(NY).utcoffset() == -timedelta(hours=hours):
            return cand
    raise ValueError(f"cannot decode EDGAR acceptanceDateTime {s!r}")


def classify_edgar_json_convention(json_raw: str, header_utc: datetime) -> str:
    """Label how the JSON acceptanceDateTime relates to the authoritative header time:
    ``utc`` | ``double_offset`` | ``et_wall`` | ``mismatch``. Stored per item for monitoring;
    ``mismatch`` means neither the JSON nor our understanding of it can be trusted."""
    literal = edgar_json_literal(json_raw)
    header_utc = ensure_utc(header_utc)
    if literal == header_utc:
        return "utc"
    if undo_double_offset(json_raw) == header_utc:
        return "double_offset"
    if literal.replace(tzinfo=None) == header_utc.astimezone(NY).replace(tzinfo=None):
        return "et_wall"
    return "mismatch"


class MarketCalendar:
    """Thin wrapper over exchange_calendars for regular-session logic (holidays, half days)."""

    def __init__(self, name: str = "XNYS", start: str = "2000-01-01") -> None:
        self._cal = xcals.get_calendar(name, start=start)

    def _check_range(self, d: date) -> None:
        ts = pd.Timestamp(d)
        if ts < self._cal.first_session or ts > self._cal.last_session:
            raise ValueError(
                f"{d} outside calendar range {self._cal.first_session.date()}.."
                f"{self._cal.last_session.date()}"
            )

    def session_bounds(self, d: date) -> tuple[datetime, datetime] | None:
        """(open, close) in UTC if ``d`` (an exchange-local date) is a session, else None."""
        self._check_range(d)
        ts = pd.Timestamp(d)
        if not self._cal.is_session(ts):
            return None
        return (
            self._cal.session_open(ts).to_pydatetime(),
            self._cal.session_close(ts).to_pydatetime(),
        )

    def next_session_open_after_date(self, d: date) -> datetime:
        self._check_range(d)
        nxt = self._cal.date_to_session(pd.Timestamp(d) + pd.Timedelta(days=1), direction="next")
        return self._cal.session_open(nxt).to_pydatetime()

    def is_open(self, t: datetime) -> bool:
        t = ensure_utc(t)
        bounds = self.session_bounds(t.astimezone(NY).date())
        return bounds is not None and bounds[0] <= t < bounds[1]

    def open_seconds_between(
        self, a: datetime, b: datetime, skip_dates: Collection[date] = ()
    ) -> float:
        """Seconds of regular-session time within [a, b) (0 if b <= a), ignoring sessions on
        ``skip_dates`` (New York dates). Used for silent-feed detection, so overnight/weekend/
        holiday gaps never count as silence."""
        a, b = ensure_utc(a), ensure_utc(b)
        if b <= a:
            return 0.0
        total = 0.0
        d = a.astimezone(NY).date()
        last = b.astimezone(NY).date()
        while d <= last:
            bounds = None if d in skip_dates else self.session_bounds(d)
            if bounds is not None:
                lo, hi = max(a, bounds[0]), min(b, bounds[1])
                if hi > lo:
                    total += (hi - lo).total_seconds()
            d += timedelta(days=1)
        return total

    def push_to_session(self, t: datetime) -> datetime:
        """Return ``t`` if within a regular session [open, close), else the next session open."""
        t = ensure_utc(t)
        local_date = t.astimezone(NY).date()
        bounds = self.session_bounds(local_date)
        if bounds is not None:
            open_, close = bounds
            if t < open_:
                return open_
            if t < close:
                return t
        return self.next_session_open_after_date(local_date)


@lru_cache(maxsize=4)
def get_calendar(name: str = "XNYS", start: str = "2000-01-01") -> MarketCalendar:
    return MarketCalendar(name, start)


def compute_tradable_at(
    *,
    published_at: datetime,
    first_seen_at: datetime,
    ingest_mode: IngestMode,
    backfill_lag: timedelta,
    calendar: MarketCalendar,
) -> datetime:
    """The point-in-time rule.

    - live:     base = max(published_at, first_seen_at)
    - backfill: base = published_at + backfill_lag   (first_seen_at is ingest time, years later)
    Then push ``base`` to the next regular-session open if it falls outside market hours.
    """
    published_at = ensure_utc(published_at, name="published_at")
    first_seen_at = ensure_utc(first_seen_at, name="first_seen_at")
    if ingest_mode == "live":
        base = max(published_at, first_seen_at)
    elif ingest_mode == "backfill":
        if backfill_lag < timedelta(0):
            raise ValueError("backfill_lag must be non-negative")
        base = published_at + backfill_lag
    else:  # pragma: no cover - guarded by Literal type
        raise ValueError(f"unknown ingest_mode {ingest_mode!r}")
    return calendar.push_to_session(base)


@lru_cache(maxsize=1)
def edgar_closed_dates() -> frozenset[date]:
    """US federal holidays (observed), when EDGAR accepts no filings although NYSE may be open
    (Columbus Day, Veterans Day). Verified 2026-10-12 (Columbus Day): the 'latest filings'
    feed stayed silent through the NYSE session."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    days = USFederalHolidayCalendar().holidays("2000-01-01", "2040-12-31")
    return frozenset(d.date() for d in days)
