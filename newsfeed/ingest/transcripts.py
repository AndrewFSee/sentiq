"""Earnings-call transcripts from the defeatbeta dataset (Hugging Face
``defeatbeta/yahoo-finance-data``, table ``stock_earning_call_transcripts``; ODC-By, research
use; source stated as Yahoo Finance), read from a local Parquet pinned to a dataset commit.

Facts (checked 2026-10-12 at commit 860d42a): one row per (symbol, fiscal_year,
fiscal_quarter) with ``report_date`` (call DATE only, no time) and speaker-labelled paragraphs;
``symbol`` is the company's CURRENT ticker (FB-era calls are under META), so symbols map to a
company through SEC's current ticker map; delisted companies are largely absent. report_date
equals the earnings 8-K date for 78% of calls and the next day for 20% (after-close releases,
next-morning calls). Some 2025 rows start with empty "AI Insights" paragraphs.

Timing (conservative, no look-ahead): with no time of day and vendor publication lagging the
call, ``published_at`` is set to the END of report_date in New York; the standard rules then
make a backfilled call tradable at the next session open. ``published_at_raw`` keeps the date
and ``meta.timing = "date_only_end_of_day"`` records the convention. Live polls (after
refreshing the Parquet) stamp the true first-seen time instead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from newsfeed.config import TranscriptsConfig
from newsfeed.ingest.base import ErrorFn, RawItem, SkipFn, make_item_id
from newsfeed.ingest.hf_refresh import current_file, refresh
from newsfeed.textutil import estimate_tokens, truncate_tokens
from newsfeed.tickers import TickerMap, Universe, normalize_ticker
from newsfeed.timeutil import IngestMode, ensure_utc, utcnow

log = logging.getLogger(__name__)

SOURCE = "transcripts"
PARSER_VERSION = "transcripts/2"  # v2: date-aware tags
NY = ZoneInfo("America/New_York")


def end_of_day_ny(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=NY).astimezone(ZoneInfo("UTC"))


# Q&A start = the first paragraph (not the opening one) that says questions begin NOW
# ("Our first question comes from ...", "We will now begin the question-and-answer session"),
# ignoring announcements of what comes later ("after the speakers' remarks, there will be a
# question-and-answer session"). Checked on the S&P 500 calls: found in 95.5%.
_QA_START = re.compile(
    r"(our|the) first question|first question (comes|is) from"
    r"|we will now (begin|start|open|take|conduct)\b.{0,40}question"
    r"|(now|ready to) (open|take|begin)\b.{0,40}questions"
    r"|now (open|turn)\b.{0,30}\b(line|lines|floor|call)\b.{0,25}questions"
    r"|(begin|start|open) the question[- ]and[- ]answer"
    r"|question[- ]and[- ]answer (session|portion)\b.{0,40}\bnow\b",
    re.I,
)
_QA_FUTURE = re.compile(
    r"will follow|after (the|today's) (speakers|prepared|presentation|remarks)"
    r"|at the (end|conclusion) of|later in the call",
    re.I,
)


def split_sections(paras: list[str]) -> tuple[list[str], list[str]] | None:
    """(prepared remarks, Q&A) of 'Speaker: text' paragraphs, or None if no Q&A start."""
    for i, p in enumerate(paras):
        if i > 0 and _QA_START.search(p) and not _QA_FUTURE.search(p):
            return paras[:i], paras[i:]
    return None


def transcript_text(paragraphs: list[dict], drop_speakers: set[str]) -> str:
    out = []
    for p in sorted(paragraphs or [], key=lambda x: x.get("paragraph_number") or 0):
        speaker = (p.get("speaker") or "").strip()
        content = (p.get("content") or "").strip()
        if not content or speaker in drop_speakers:
            continue
        out.append(f"{speaker or 'Unknown'}: {content}")
    return "\n\n".join(out)


class TranscriptsAdapter:
    name = SOURCE

    def __init__(
        self,
        cfg: TranscriptsConfig,
        universe: Universe,
        tmap: TickerMap,
        *,
        spells: dict[str, list[tuple[date, date | None, int]]] | None = None,
        section: str = "full",
        clock=utcnow,
    ) -> None:
        """``spells``: ticker -> [(start, end exclusive | None, cik)] index-membership spells.
        With them, a call is tagged with ticker T only if, on the call date, T belonged to the
        same company: the spell's CIK is the call's CIK, or a predecessor CIK that no longer
        has a ticker of its own (a re-registration such as Disney 2019). A ticker then held by
        a different, still-trading company (IR in 2019 = old Ingersoll-Rand plc, now TT) is
        never put on another company's call (Gardner Denver, whose symbol is IR today)."""
        if section not in ("full", "qa", "prep"):
            raise ValueError(f"unknown transcript section {section!r}")
        self.section = section
        # section views are separate sources (own items, scores and features); the full call
        # stays "transcripts"
        self.name = SOURCE if section == "full" else f"{SOURCE}_{section}"
        self.cfg = cfg
        self.universe = universe
        self.tmap = tmap
        self.spells = spells
        self._http = None  # tests inject an httpx.Client for refresh()
        self._live_ciks = set(tmap.ticker_to_cik.values())
        self.clock = clock
        self.last_activity_at: datetime | None = None
        self.cik_windows: dict[int, tuple[datetime, datetime]] | None = None
        self.stats: dict[str, int] = {}
        self._drop = set(cfg.drop_speakers)

    @property
    def file(self) -> Path:
        return self._resolve()[0]

    @property
    def commit(self) -> str:
        return self._resolve()[1]

    def _resolve(self) -> tuple[Path, str]:
        p = Path(self.cfg.path)
        if p.is_dir():
            return current_file(p, self.cfg.dataset_commit)
        return p, self.cfg.dataset_commit  # a single file (tests, ad-hoc runs)

    def refresh(self) -> dict:
        """Download a newer dataset version if one exists (directory layout only)."""
        p = Path(self.cfg.path)
        if not (self.cfg.auto_refresh and p.is_dir()):
            return {"changed": False, "skipped": True}
        out = refresh(
            p,
            self.cfg.hf_repo,
            self.cfg.hf_file,
            fallback_commit=self.cfg.dataset_commit,
            keep_files=[f"{self.cfg.dataset_commit[:12]}.parquet"],
            client=self._http,
        )
        self.stats["refresh"] = out
        return out

    def _tags(self, cik: int, d: date) -> list[str]:
        tickers = self.universe.tickers_for(cik)
        if self.spells is None:
            return tickers
        out = []
        for t in tickers:
            holder = next(
                (c for s, e, c in self.spells.get(t, []) if s <= d and (e is None or d < e)),
                None,
            )
            if holder is None or holder == cik or holder not in self._live_ciks:
                out.append(t)
            else:
                self._bump(f"tag_dropped_{t}")
        return out

    def _bump(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def _rows(self, d0: date, d1: date) -> Iterator[dict]:
        path = self.file
        if not path.exists():
            raise FileNotFoundError(
                f"transcripts Parquet not found at {path}; download the pinned file "
                f"(dataset commit {self.cfg.dataset_commit}) first"
            )
        con = duckdb.connect()
        # 1) metadata only (the text column is ~2 GB: never sort or scan it in bulk)
        meta = con.execute(
            """
            SELECT transcripts_id, symbol, report_date FROM read_parquet(?)
            WHERE report_date >= ? AND report_date < ?
            ORDER BY report_date, symbol, transcripts_id
            """,
            [path.as_posix(), d0.isoformat(), d1.isoformat()],
        ).fetchall()
        # 2) keep calls of universe companies; fetch their text in ONE streaming pass (semi-join,
        #    no sort: ingest order doesn't matter, stories are rebuilt in time order anyway)
        wanted = [
            tid
            for tid, sym, _ in meta
            if self.tmap.ticker_to_cik.get(normalize_ticker(str(sym)))
            in self.universe.cik_to_tickers
        ]
        self.stats["not_in_universe"] = (
            self.stats.get("not_in_universe", 0) + len(meta) - len(wanted)
        )
        if not wanted:
            con.close()
            return
        con.execute("SET preserve_insertion_order = false")
        con.execute("CREATE TEMP TABLE wanted AS SELECT unnest(?) AS tid", [wanted])
        cur = con.execute(
            """
            SELECT symbol, fiscal_year, fiscal_quarter, report_date, transcripts_id, transcripts
            FROM read_parquet(?) t WHERE t.transcripts_id IN (SELECT tid FROM wanted)
            """,
            [path.as_posix()],
        )
        cols = [c[0] for c in cur.description]
        while batch := cur.fetchmany(100):
            for r in batch:
                yield dict(zip(cols, r, strict=True))
        con.close()

    def _item(self, r: dict, *, mode: IngestMode, first_seen: datetime) -> RawItem | None:
        sym = normalize_ticker(str(r["symbol"]))
        cik = self.tmap.ticker_to_cik.get(sym)
        if cik is None or cik not in self.universe.cik_to_tickers:
            self._bump("not_in_universe")
            return None
        d = date.fromisoformat(str(r["report_date"])[:10])
        published = end_of_day_ny(d)
        if self.cik_windows is not None:
            win = self.cik_windows.get(cik)
            if win is None or not (win[0] <= published < win[1]):
                self._bump("outside_window")
                return None
        tags = self._tags(cik, d)
        if not tags:
            self._bump("no_valid_ticker")
            return None
        text = transcript_text(r["transcripts"], self._drop)
        if self.section != "full":
            parts = split_sections(text.split("\n\n")) if text else None
            if parts is None:
                self._bump("no_qa_boundary")
                return None
            text = "\n\n".join(parts[1] if self.section == "qa" else parts[0])
        if not text:
            self._bump("empty")
            return None
        body, truncated = truncate_tokens(text, self.cfg.max_tokens, self.cfg.chars_per_token)
        q, y = r["fiscal_quarter"], r["fiscal_year"]
        self._bump("items")
        suffix = "" if self.section == "full" else f":{self.section}"
        label = {"full": "", "qa": " (Q&A only)", "prep": " (prepared remarks only)"}
        return RawItem(
            source=self.name,
            native_id=f"defeatbeta:{r['transcripts_id']}{suffix}",
            publisher="Earnings call (defeatbeta)",
            url=None,
            title=f"{sym} earnings call transcript, fiscal Q{q} {y}{label[self.section]}",
            body_text=body,
            body_tokens_est=estimate_tokens(body, self.cfg.chars_per_token),
            body_truncated=truncated,
            tickers=tags,
            published_at=published,
            published_at_raw=str(r["report_date"]),
            first_seen_at=first_seen,
            ingest_mode=mode,
            meta={
                "symbol": sym,
                "cik": cik,
                "fiscal_year": y,
                "fiscal_quarter": q,
                "timing": "date_only_end_of_day",
                "dataset_commit": self.commit,
                "n_paragraphs": len(r["transcripts"] or []),
            },
            raw_json="{}",  # full text lives in the pinned Parquet (not duplicated)
            parser_version=PARSER_VERSION + ("" if self.section == "full" else f"+{self.section}"),
        )

    # ---- adapter interface
    def backfill(
        self, start: datetime, end: datetime, skip: SkipFn, on_error: ErrorFn | None = None
    ) -> Iterator[RawItem]:
        start, end = ensure_utc(start, name="start"), ensure_utc(end, name="end")
        d0 = start.astimezone(NY).date() - timedelta(days=1)
        d1 = end.astimezone(NY).date() + timedelta(days=1)
        for r in self._rows(d0, d1):
            native = f"defeatbeta:{r['transcripts_id']}" + (
                "" if self.section == "full" else f":{self.section}"
            )
            try:
                d = date.fromisoformat(str(r["report_date"])[:10])
                if not (start <= end_of_day_ny(d) < end):
                    continue
                if skip(make_item_id(self.name, native)):
                    continue
                item = self._item(r, mode="backfill", first_seen=self.clock())
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(native, exc)
                continue
            if item is not None:
                yield item

    def poll(
        self, skip: SkipFn, on_error: ErrorFn | None = None, *, catch_up: bool = False
    ) -> Iterator[RawItem]:
        """Calls in the last ``poll_lookback_days`` not stored yet (refresh the Parquet first).
        Live items get first_seen_at = now, so tradable_at is honest even when the dataset
        added a call late."""
        if catch_up:
            try:
                self.refresh()
            except Exception as exc:  # a failed refresh must not stop polling the local copy
                if on_error is None:
                    raise
                on_error("refresh", exc)
        today = self.clock().astimezone(NY).date()
        seen_at = self.clock()
        newest: datetime | None = None
        for r in self._rows(
            today - timedelta(days=self.cfg.poll_lookback_days), today + timedelta(1)
        ):
            native = f"defeatbeta:{r['transcripts_id']}" + (
                "" if self.section == "full" else f":{self.section}"
            )
            d = end_of_day_ny(date.fromisoformat(str(r["report_date"])[:10]))
            newest = d if newest is None else max(newest, d)
            if skip(make_item_id(self.name, native)):
                continue
            try:
                item = self._item(r, mode="live", first_seen=seen_at)
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(native, exc)
                continue
            if item is not None:
                yield item
        self.last_activity_at = newest
