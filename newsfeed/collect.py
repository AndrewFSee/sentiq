"""Continuous / cron-friendly live collection with heartbeat and outage detection.

Each cycle, per source: poll (periodically a wider catch-up sweep) -> ingest -> update a
persisted state file. Then derived tables are refreshed and health is evaluated.

State lives in ``data/state/collect-<source>.json`` (plus ``heartbeat.json``), so
``collect --once`` from cron behaves exactly like one iteration of the loop, and
``collect --check`` can evaluate health without polling (exit code for monitors).

"Silent feed = outage, not 'no news'": silence is measured on the source's *market-wide*
activity (``adapter.last_activity_at``: newest entry for any company), counting only
regular-session time. A quiet universe is normal; a quiet EDGAR feed at 11:00 ET is not.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from newsfeed import coverage
from newsfeed.config import AppConfig
from newsfeed.ingest.base import SourceAdapter
from newsfeed.ingest.runner import ingest, make_error_recorder, new_stats
from newsfeed.storage import Store
from newsfeed.timeutil import MarketCalendar, edgar_closed_dates, ensure_utc, utcnow

log = logging.getLogger(__name__)


def _iso(dt: datetime | None) -> str | None:
    return ensure_utc(dt).isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


@dataclass
class SourceState:
    source: str
    first_poll_at: str | None = None
    last_poll_started_at: str | None = None
    last_poll_ok_at: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    last_new_item_at: str | None = None  # when we last *stored* a universe item
    last_activity_at: str | None = None  # newest market-wide feed timestamp
    last_catch_up_at: str | None = None
    polls: int = 0
    items_written: int = 0
    open_alerts: dict[str, str] = field(default_factory=dict)  # kind -> last raised (iso)

    @classmethod
    def load(cls, path: Path, source: str) -> SourceState:
        if not path.exists():
            return cls(source=source)
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        _write_json(path, asdict(self))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class Problem:
    source: str
    kind: str  # poll_failing | silent_feed | stalled | never_ran
    message: str


def evaluate_health(
    state: SourceState,
    *,
    now: datetime,
    calendar: MarketCalendar,
    silence_minutes: float,
    interval_seconds: float,
    max_consecutive_failures: int,
    stalled_after_intervals: float,
) -> list[Problem]:
    src = state.source
    if state.first_poll_at is None:
        return [Problem(src, "never_ran", "collector has never polled this source")]
    out: list[Problem] = []
    if state.consecutive_failures >= max_consecutive_failures:
        out.append(
            Problem(
                src,
                "poll_failing",
                f"{state.consecutive_failures} consecutive failed polls; last: {state.last_error}",
            )
        )
    # Stalls only matter in market time: an after-hours gap is harmless because tradable_at
    # is pushed to the next open anyway, but it must be fixed before the open.
    last_ok = _dt(state.last_poll_ok_at) or _dt(state.first_poll_at)
    assert last_ok is not None
    stalled = calendar.open_seconds_between(last_ok, now)
    if calendar.is_open(now) and stalled > stalled_after_intervals * interval_seconds:
        out.append(Problem(src, "stalled", f"no successful poll since {last_ok.isoformat()}"))
    ref = _dt(state.last_activity_at) or _dt(state.first_poll_at)
    assert ref is not None
    # EDGAR is closed on federal holidays even when NYSE trades: those sessions aren't silence
    closed = edgar_closed_dates() if src == "edgar" else frozenset()
    silent = calendar.open_seconds_between(ref, now, closed) / 60
    today_closed = now.astimezone(ZoneInfo("America/New_York")).date() in closed
    if calendar.is_open(now) and not today_closed and silent > silence_minutes:
        out.append(
            Problem(
                src,
                "silent_feed",
                f"no feed activity for {silent:.0f} market-open minutes "
                f"(threshold {silence_minutes:.0f}); newest seen {ref.isoformat()}",
            )
        )
    return out


class Collector:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        adapters: dict[str, SourceAdapter],
        calendar: MarketCalendar,
        run_id: str,
        *,
        derive_fn: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.adapters = adapters
        self.calendar = calendar
        self.run_id = run_id
        self.derive_fn = derive_fn
        self.clock = clock
        self.sleep = sleep
        self.stats: dict[str, Any] = {"cycles": 0, **new_stats()}

    def state_path(self, source: str) -> Path:
        return self.store.state_dir / f"collect-{source}.json"

    def _source_cfg(self, source: str) -> Any:
        return getattr(self.cfg.sources, source)

    # ---- one iteration
    def poll_source(self, source: str, adapter: SourceAdapter) -> SourceState:
        path = self.state_path(source)
        st = SourceState.load(path, source)
        now = self.clock()
        st.first_poll_at = st.first_poll_at or _iso(now)
        st.last_poll_started_at = _iso(now)
        scfg = self._source_cfg(source)
        last_cu = _dt(st.last_catch_up_at)
        catch_up = last_cu is None or now - last_cu >= timedelta(
            minutes=scfg.catch_up_every_minutes
        )
        stats = new_stats()
        try:
            items = adapter.poll(
                self.store.has_item,
                make_error_recorder(self.store, self.run_id, source, stats),
                catch_up=catch_up,
            )
            ingest(
                items,
                store=self.store,
                cfg=self.cfg,
                calendar=self.calendar,
                run_id=self.run_id,
                flush_every=scfg.flush_every,
                stats=stats,
            )
        except Exception as exc:  # whole poll failed (network, parse of the listing, ...)
            log.exception("poll failed for %s", source)
            st.consecutive_failures += 1
            st.last_error = repr(exc)[:500]
        else:
            done = self.clock()
            prev_ok = _dt(st.last_poll_ok_at)
            max_gap = timedelta(
                seconds=self.cfg.collect.stalled_after_intervals * self.cfg.collect.interval_seconds
            )
            universe = getattr(adapter, "universe", None)
            if prev_ok is not None and done - prev_ok <= max_gap and universe is not None:
                # watched continuously since the previous successful poll
                tickers = [t for ts in universe.cik_to_tickers.values() for t in ts]
                coverage.record_live(self.store, source, tickers, prev_ok, done, self.run_id)
            st.consecutive_failures = 0
            st.last_error = None
            st.last_poll_ok_at = _iso(done)
            if catch_up:
                st.last_catch_up_at = _iso(done)
            if stats["written"]:
                st.last_new_item_at = _iso(done)
        if adapter.last_activity_at is not None:
            prev = _dt(st.last_activity_at)
            newest = adapter.last_activity_at
            st.last_activity_at = _iso(max(prev, newest) if prev else newest)
        st.polls += 1
        st.items_written += stats["written"]
        for k in ("seen", "written", "failed", "truncated"):
            self.stats[k] += stats[k]
        st.save(path)
        return st

    def check_and_alert(self, states: dict[str, SourceState]) -> list[Problem]:
        now = self.clock()
        problems: list[Problem] = []
        for source, st in states.items():
            found = evaluate_health(
                st,
                now=now,
                calendar=self.calendar,
                silence_minutes=self._source_cfg(source).silence_alert_minutes,
                interval_seconds=self.cfg.collect.interval_seconds,
                max_consecutive_failures=self.cfg.collect.max_consecutive_failures,
                stalled_after_intervals=self.cfg.collect.stalled_after_intervals,
            )
            problems += found
            kinds = {p.kind for p in found}
            for p in found:
                last = _dt(st.open_alerts.get(p.kind))
                if last is None or now - last >= timedelta(
                    minutes=self.cfg.collect.realert_minutes
                ):
                    log.warning("ALERT [%s] %s: %s", p.source, p.kind, p.message)
                    self.store.record_alert(p.source, p.kind, p.message, self.run_id)
                    st.open_alerts[p.kind] = _iso(now) or ""
            for kind in list(st.open_alerts):
                if kind not in kinds:
                    log.info("RECOVERED [%s] %s", source, kind)
                    del st.open_alerts[kind]
            st.save(self.state_path(source))
        return problems

    def cycle(self) -> list[Problem]:
        self.store.refresh_known_ids()  # other processes may have written
        states = {s: self.poll_source(s, a) for s, a in self.adapters.items()}
        if self.derive_fn is not None:
            try:
                self.stats["derive"] = self.derive_fn()
            except Exception:
                log.exception("derive failed (will retry next cycle)")
        problems = self.check_and_alert(states)
        self.stats["cycles"] += 1
        _write_json(
            self.store.state_dir / "heartbeat.json",
            {
                "updated_at": _iso(self.clock()),
                "pid": os.getpid(),
                "run_id": self.run_id,
                "problems": [asdict(p) for p in problems],
                "sources": {s: asdict(st) for s, st in states.items()},
            },
        )
        self.store.update_run_stats(self.run_id, self.stats)
        return problems

    def run(self, max_cycles: int | None = None) -> None:
        interval = self.cfg.collect.interval_seconds
        n = 0
        while max_cycles is None or n < max_cycles:
            started = time.monotonic()
            self.cycle()
            n += 1
            if max_cycles is not None and n >= max_cycles:
                break
            self.sleep(max(0.0, interval - (time.monotonic() - started)))


def check_states(
    cfg: AppConfig, store: Store, calendar: MarketCalendar, sources: list[str], now: datetime
) -> list[Problem]:
    """Health from persisted state only (no polling) -- for ``collect --check``."""
    problems: list[Problem] = []
    for s in sources:
        st = SourceState.load(store.state_dir / f"collect-{s}.json", s)
        problems += evaluate_health(
            st,
            now=now,
            calendar=calendar,
            silence_minutes=getattr(cfg.sources, s).silence_alert_minutes,
            interval_seconds=cfg.collect.interval_seconds,
            max_consecutive_failures=cfg.collect.max_consecutive_failures,
            stalled_after_intervals=cfg.collect.stalled_after_intervals,
        )
    return problems
