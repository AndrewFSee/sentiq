from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from newsfeed.collect import Collector, SourceState, check_states, evaluate_health
from newsfeed.ingest.base import RawItem

# Mon 2026-08-03 is a regular session (09:30-16:00 EDT = 13:30-20:00 UTC)
MON_OPEN = datetime(2026, 8, 3, 13, 30, tzinfo=UTC)


class FakeAdapter:
    name = "edgar"

    def __init__(self) -> None:
        self.last_activity_at: datetime | None = None
        self.fail = False
        self.activity: datetime | None = None
        self.items: list[RawItem] = []
        self.calls: list[bool] = []

    def poll(self, skip, on_error=None, *, catch_up=False) -> Iterator[RawItem]:
        self.calls.append(catch_up)
        if self.fail:
            raise ConnectionError("feed down")
        if self.activity is not None:
            self.last_activity_at = self.activity
        yield from (i for i in self.items if not skip(i.item_id))

    def backfill(self, *a, **k):  # pragma: no cover
        return iter(())


def item(nid: str, t: datetime) -> RawItem:
    return RawItem(
        source="edgar",
        native_id=nid,
        url=None,
        title="t",
        body_text="b",
        body_tokens_est=1,
        body_truncated=False,
        tickers=["AAPL"],
        published_at=t,
        published_at_raw="x",
        first_seen_at=t,
        ingest_mode="live",
        meta={},
        raw_json="{}",
        parser_version="t",
    )


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


@pytest.fixture
def setup(cfg, store, calendar):
    clock = Clock(MON_OPEN + timedelta(minutes=30))
    adapter = FakeAdapter()

    def make() -> Collector:  # a fresh process (cron) shares only the on-disk state
        return Collector(
            cfg, store, {"edgar": adapter}, calendar, "run", clock=clock, sleep=lambda s: None
        )

    return make, adapter, clock


def health(st, calendar, now, **kw):
    args = dict(
        silence_minutes=45,
        interval_seconds=60,
        max_consecutive_failures=3,
        stalled_after_intervals=3,
    )
    args.update(kw)
    return {p.kind for p in evaluate_health(st, now=now, calendar=calendar, **args)}


# ------------------------------------------------------------------ evaluate_health
def test_silence_counts_only_market_time(calendar):
    st = SourceState(
        "edgar",
        first_poll_at="2026-07-31T00:00:00+00:00",
        last_poll_ok_at=(MON_OPEN + timedelta(minutes=10)).isoformat(),
        last_activity_at="2026-07-31T21:30:00+00:00",
    )  # Fri 17:30 EDT
    # Monday 09:40: 10 market minutes since Friday's last filing -> fine
    assert health(st, calendar, MON_OPEN + timedelta(minutes=10)) == set()
    # Monday 10:30: 60 market minutes of nothing from ANY filer -> outage
    st.last_poll_ok_at = (MON_OPEN + timedelta(minutes=60)).isoformat()
    assert health(st, calendar, MON_OPEN + timedelta(minutes=60)) == {"silent_feed"}


def test_no_silence_alert_outside_market_hours(calendar):
    st = SourceState(
        "edgar",
        first_poll_at="2026-08-01T00:00:00+00:00",
        last_poll_ok_at="2026-08-02T12:00:00+00:00",
        last_activity_at="2026-07-31T21:30:00+00:00",
    )
    assert health(st, calendar, datetime(2026, 8, 2, 12, 0, tzinfo=UTC)) == set()  # Sunday


def test_stalled_and_failing(calendar):
    now = MON_OPEN + timedelta(minutes=30)
    st = SourceState(
        "edgar",
        first_poll_at=MON_OPEN.isoformat(),
        last_poll_ok_at=(now - timedelta(minutes=10)).isoformat(),
        last_activity_at=(now - timedelta(minutes=1)).isoformat(),
        consecutive_failures=3,
        last_error="boom",
    )
    assert health(st, calendar, now) == {"stalled", "poll_failing"}


def test_never_ran(calendar):
    assert health(SourceState("edgar"), calendar, MON_OPEN) == {"never_ran"}


# ------------------------------------------------------------------ Collector
def test_cycle_ingests_and_writes_state_and_heartbeat(setup, store):
    make, adapter, clock = setup
    adapter.activity = clock.t - timedelta(minutes=2)
    adapter.items = [item("acc-1", clock.t - timedelta(minutes=5))]
    problems = make().cycle()
    assert problems == []
    assert store.query("SELECT count(*) AS n FROM items")["n"][0] == 1
    st = json.loads((store.state_dir / "collect-edgar.json").read_text())
    assert st["items_written"] == 1 and st["consecutive_failures"] == 0
    assert st["last_activity_at"] == adapter.activity.isoformat()
    hb = json.loads((store.state_dir / "heartbeat.json").read_text())
    assert hb["sources"]["edgar"]["polls"] == 1 and hb["problems"] == []


def test_failures_accumulate_then_alert_once_then_recover(setup, store):
    make, adapter, clock = setup
    adapter.fail = True
    c = make()
    kinds = []
    for _ in range(4):
        kinds.append({p.kind for p in c.cycle()})
        clock.t += timedelta(minutes=1)
    assert "poll_failing" not in kinds[1] and "poll_failing" in kinds[2]
    alerts = store.ops_query("SELECT kind FROM alerts WHERE kind = 'poll_failing'")
    assert alerts.height == 1  # re-alert throttled (realert_minutes=60)
    adapter.fail = False
    adapter.activity = clock.t
    assert c.cycle() == []
    st = SourceState.load(store.state_dir / "collect-edgar.json", "edgar")
    assert st.open_alerts == {}


def test_silent_feed_alert_from_activity_not_universe_items(setup, store):
    make, adapter, clock = setup
    adapter.activity = MON_OPEN - timedelta(hours=40)  # nothing since Friday
    c = make()
    assert c.cycle() == []  # 30 market minutes so far
    clock.t += timedelta(minutes=20)
    assert {p.kind for p in c.cycle()} == {"silent_feed"}
    adapter.activity = clock.t  # other companies are filing again; universe still quiet
    clock.t += timedelta(minutes=1)
    assert c.cycle() == []


def test_catch_up_schedule_persists_across_processes(setup):
    make, adapter, clock = setup
    make().cycle()  # first ever: catch-up
    clock.t += timedelta(minutes=1)
    make().cycle()  # new process 1 minute later: regular poll
    clock.t += timedelta(minutes=61)
    make().cycle()  # an hour later: catch-up again
    assert adapter.calls == [True, False, True]


def test_check_states_without_polling(setup, cfg, store, calendar):
    make, adapter, clock = setup
    assert {p.kind for p in check_states(cfg, store, calendar, ["edgar"], clock.t)} == {"never_ran"}
    adapter.activity = clock.t
    make().cycle()
    assert check_states(cfg, store, calendar, ["edgar"], clock.t) == []
    later = clock.t + timedelta(minutes=10)  # cron job died
    assert {p.kind for p in check_states(cfg, store, calendar, ["edgar"], later)} == {"stalled"}


def test_run_loop_respects_max_cycles(setup):
    make, adapter, _ = setup
    c = make()
    c.run(max_cycles=3)
    assert c.stats["cycles"] == 3 and len(adapter.calls) == 3


def test_derive_failure_does_not_kill_cycle(setup, cfg, store, calendar):
    _, adapter, clock = setup

    def boom():
        raise RuntimeError("derive broke")

    c = Collector(cfg, store, {"edgar": adapter}, calendar, "run", derive_fn=boom, clock=clock)
    adapter.activity = clock.t
    assert c.cycle() == []


def test_live_polls_record_coverage_and_stalls_leave_gaps(setup, store):
    from newsfeed import coverage
    from newsfeed.tickers import Universe

    make, adapter, clock = setup
    adapter.universe = Universe({1: ["AAPL"]})
    adapter.activity = clock.t
    t0 = clock.t
    make().cycle()  # first poll: nothing to cover yet
    clock.t += timedelta(minutes=1)
    make().cycle()
    clock.t += timedelta(minutes=1)
    make().cycle()
    assert coverage.merged(store, "edgar")["AAPL"] == [(t0, t0 + timedelta(minutes=2))]
    clock.t += timedelta(minutes=30)  # a late poll that paged back: no gap
    make().cycle()
    assert coverage.merged(store, "edgar")["AAPL"] == [(t0, t0 + timedelta(minutes=32))]
    clock.t += timedelta(minutes=90)  # stalled beyond collect.coverage_max_gap_minutes: gap
    make().cycle()
    clock.t += timedelta(minutes=1)
    make().cycle()
    assert coverage.merged(store, "edgar")["AAPL"] == [
        (t0, t0 + timedelta(minutes=32)),
        (t0 + timedelta(minutes=122), t0 + timedelta(minutes=123)),
    ]
    # a poll whose feed did not reach back to the previous poll leaves a gap even if prompt
    # (FakeAdapter.poll never runs a sweep after the first cycle's catch-up)
    clock.t += timedelta(minutes=2)
    adapter.last_poll_reach = clock.t - timedelta(minutes=1)
    make().cycle()
    assert coverage.merged(store, "edgar")["AAPL"][-1] == (
        t0 + timedelta(minutes=122),
        t0 + timedelta(minutes=123),
    )


def test_bridge_live_gaps_adds_rows_only_for_short_gaps(store):
    from newsfeed import coverage

    t = datetime(2026, 10, 12, 14, 30, tzinfo=UTC)
    m = timedelta(minutes=1)
    coverage.record_live(store, "edgar", ["AAPL"], t, t + 10 * m, "r")
    coverage.record_live(store, "edgar", ["AAPL"], t + 17 * m, t + 30 * m, "r")  # 7-min gap
    coverage.record_live(store, "edgar", ["AAPL"], t + 130 * m, t + 140 * m, "r")  # 100-min gap
    n = coverage.bridge_live_gaps(store, "edgar", timedelta(minutes=60), t, "bridge-x")
    assert n == 1
    assert coverage.merged(store, "edgar")["AAPL"] == [(t, t + 30 * m), (t + 130 * m, t + 140 * m)]
    # nothing before ``since`` is bridged; re-running adds nothing new to bridge
    assert coverage.bridge_live_gaps(store, "edgar", timedelta(minutes=60), t, "b2") == 0


def test_edgar_silence_ignores_federal_holidays_when_nyse_is_open(calendar):
    # Columbus Day 2026-10-12: NYSE open, EDGAR closed -> no filings all day is not an outage
    st = SourceState(
        "edgar",
        first_poll_at="2026-10-01T00:00:00+00:00",
        last_activity_at="2026-10-09T21:30:00+00:00",  # Fri 17:30 EDT
    )
    holiday = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)  # 14:00 EDT, NYSE session
    st.last_poll_ok_at = holiday.isoformat()
    assert calendar.is_open(holiday)
    assert health(st, calendar, holiday) == set()
    # Tuesday 10:30 EDT: 60 minutes of Tuesday's session (Monday's doesn't count) -> outage
    tue = datetime(2026, 10, 13, 14, 30, tzinfo=UTC)
    st.last_poll_ok_at = tue.isoformat()
    assert health(st, calendar, tue) == {"silent_feed"}
    # a news vendor publishes on federal holidays: silence there still counts
    st_news = SourceState(
        "news_api",
        first_poll_at="2026-10-01T00:00:00+00:00",
        last_poll_ok_at=holiday.isoformat(),
        last_activity_at="2026-10-09T21:30:00+00:00",
    )
    assert health(st_news, calendar, holiday) == {"silent_feed"}
