from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from newsfeed.timeutil import (
    classify_edgar_json_convention,
    compute_tradable_at,
    edgar_json_literal,
    ensure_utc,
    parse_et_wall,
    undo_double_offset,
)


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


# ------------------------------------------------------------------ basic tz hygiene
def test_ensure_utc_rejects_naive():
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc(datetime(2026, 1, 1))


def test_ensure_utc_converts_offsets():
    est = timezone(timedelta(hours=-5))
    assert ensure_utc(datetime(2026, 1, 2, 9, 30, tzinfo=est)) == utc(2026, 1, 2, 14, 30)


# ------------------------------------------------------------------ EDGAR timestamps
# (header ET wall clock, submissions-JSON value, true UTC) -- all observed on real Apple filings.
REAL_CASES = [
    ("20260730163028", "2026-07-31T00:30:28.000Z", utc(2026, 7, 30, 20, 30, 28)),  # EDT
    ("20260129163033", "2026-01-30T02:30:33.000Z", utc(2026, 1, 29, 21, 30, 33)),  # EST
    ("20240910090634", "2024-09-10T17:06:34.000Z", utc(2024, 9, 10, 13, 6, 34)),  # EDT a.m.
    ("20190206080034", "2019-02-06T18:00:34.000Z", utc(2019, 2, 6, 13, 0, 34)),  # EST a.m.
    ("20250708211813", "2025-07-09T05:18:13.000Z", utc(2025, 7, 9, 1, 18, 13)),  # after 17:30
]


@pytest.mark.parametrize("header,json_raw,true_utc", REAL_CASES)
def test_parse_et_wall_is_dst_aware(header, json_raw, true_utc):
    assert parse_et_wall(header) == true_utc


@pytest.mark.parametrize("header,json_raw,true_utc", REAL_CASES)
def test_apple_json_is_double_offset(header, json_raw, true_utc):
    assert undo_double_offset(json_raw) == true_utc
    assert classify_edgar_json_convention(json_raw, true_utc) == "double_offset"
    assert edgar_json_literal(json_raw) >= true_utc  # upper-bound property


# Other filers report correct UTC (observed on live MSFT / NVDA / GOOGL 8-Ks, 2026-10).
UTC_CASES = [
    ("20260729160453", "2026-07-29T20:04:53.000Z"),  # MSFT 0001193125-26-323632
    ("20260702092316", "2026-07-02T13:23:16.000Z"),  # NVDA 0001045810-26-000060
]


@pytest.mark.parametrize("header,json_raw", UTC_CASES)
def test_other_filers_json_is_true_utc(header, json_raw):
    true_utc = parse_et_wall(header)
    assert classify_edgar_json_convention(json_raw, true_utc) == "utc"
    assert edgar_json_literal(json_raw) == true_utc


def test_naive_json_reading_would_be_late():
    """Documents the bug we are defending against: trusting the 'Z' is 4h late in summer."""
    _header, json_raw, true_utc = REAL_CASES[0]
    naive_read = datetime.fromisoformat(json_raw.replace("Z", "+00:00"))
    assert naive_read - true_utc == timedelta(hours=4)


def test_convention_classifier_detects_a_fix():
    true = utc(2026, 7, 30, 20, 30, 28)
    assert classify_edgar_json_convention("2026-07-30T20:30:28.000Z", true) == "utc"
    assert classify_edgar_json_convention("2026-07-30T16:30:28.000Z", true) == "et_wall"
    assert classify_edgar_json_convention("2026-07-30T11:00:00.000Z", true) == "mismatch"


def test_parse_et_wall_rejects_ambiguous_hour():
    with pytest.raises(ValueError, match="ambiguous"):
        parse_et_wall("20251102013000")  # DST fall-back


# ------------------------------------------------------------------ tradable_at
def tradable(pub, seen=None, mode="live", lag=timedelta(0), calendar=None):
    return compute_tradable_at(
        published_at=pub,
        first_seen_at=seen or pub,
        ingest_mode=mode,
        backfill_lag=lag,
        calendar=calendar,
    )


def test_in_session_item_is_tradable_immediately(calendar):
    t = utc(2026, 7, 30, 15, 0)  # 11:00 EDT Thursday
    assert tradable(t, calendar=calendar) == t


def test_live_uses_later_of_published_and_first_seen(calendar):
    pub, seen = utc(2026, 7, 30, 15, 0), utc(2026, 7, 30, 15, 7)
    assert tradable(pub, seen, calendar=calendar) == seen
    # a source claiming a publish time later than our sighting still wins
    assert tradable(seen, pub, calendar=calendar) == seen


def test_after_close_pushes_to_next_open(calendar):
    pub = utc(2026, 7, 30, 20, 30, 28)  # 16:30:28 EDT Thursday -> Friday 09:30 EDT
    assert tradable(pub, calendar=calendar) == utc(2026, 7, 31, 13, 30)


def test_close_boundary_is_exclusive(calendar):
    close = utc(2026, 7, 30, 20, 0)  # 16:00 EDT exactly
    assert tradable(close, calendar=calendar) == utc(2026, 7, 31, 13, 30)
    assert tradable(close - timedelta(seconds=1), calendar=calendar) == close - timedelta(seconds=1)


def test_open_boundary_is_inclusive(calendar):
    open_ = utc(2026, 7, 30, 13, 30)
    assert tradable(open_, calendar=calendar) == open_


def test_premarket_pushes_to_same_day_open(calendar):
    pub = utc(2024, 9, 10, 13, 6, 34)  # 09:06 EDT
    assert tradable(pub, calendar=calendar) == utc(2024, 9, 10, 13, 30)


def test_friday_evening_pushes_to_monday(calendar):
    pub = utc(2026, 7, 31, 22, 0)  # Fri 18:00 EDT
    assert tradable(pub, calendar=calendar) == utc(2026, 8, 3, 13, 30)


def test_late_evening_utc_rollover(calendar):
    """21:18 EDT Tue is already Wednesday in UTC; must map to Wed open, not Thu."""
    pub = utc(2025, 7, 9, 1, 18, 13)
    assert tradable(pub, calendar=calendar) == utc(2025, 7, 9, 13, 30)


def test_holiday_is_skipped(calendar):
    pub = utc(2025, 7, 3, 20, 0)  # Thu Jul 3 2025 (13:00 half day close) -> Jul 4 closed
    assert tradable(pub, calendar=calendar) == utc(2025, 7, 7, 13, 30)


def test_half_day_close(calendar):
    pub = utc(2025, 11, 28, 18, 30)  # day after Thanksgiving, 13:30 EST, close was 13:00
    assert tradable(pub, calendar=calendar) == utc(2025, 12, 1, 14, 30)
    inside = utc(2025, 11, 28, 17, 59)
    assert tradable(inside, calendar=calendar) == inside


def test_backfill_uses_lag_not_ingest_time(calendar):
    pub = utc(2026, 7, 30, 15, 0)
    years_later = utc(2030, 1, 1)
    got = tradable(pub, years_later, mode="backfill", lag=timedelta(seconds=60), calendar=calendar)
    assert got == pub + timedelta(seconds=60)


def test_backfill_lag_can_cross_the_close(calendar):
    pub = utc(2026, 7, 30, 19, 59, 30)  # 15:59:30 EDT + 60s lag -> after close
    got = tradable(pub, mode="backfill", lag=timedelta(seconds=60), calendar=calendar)
    assert got == utc(2026, 7, 31, 13, 30)


def test_tradable_never_precedes_published(calendar):
    """Property check over a sweep of times across a week incl. DST change."""
    t = utc(2026, 3, 5, 0, 0)
    while t < utc(2026, 3, 12):
        for mode in ("live", "backfill"):
            got = tradable(t, mode=mode, lag=timedelta(seconds=60), calendar=calendar)
            assert got >= t
            assert calendar.is_open(got)
        t += timedelta(minutes=17)


def test_naive_inputs_rejected(calendar):
    with pytest.raises(ValueError):
        tradable(datetime(2026, 7, 30, 15, 0), calendar=calendar)


def test_outside_calendar_range_raises(calendar):
    with pytest.raises(ValueError, match="outside calendar range"):
        tradable(utc(1999, 6, 1, 15, 0), calendar=calendar)
