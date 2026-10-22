"""Storage: append-only Parquet lake queried through DuckDB, plus small operational tables.

Layout::

    lake/items/src=<source>/ym=<YYYY-MM of published_at>/part-<run_id>-<seq>.parquet
    lake/item_tickers/part-<run_id>-<seq>.parquet          # derived: ticker links (append-only)
    lake/item_story/ym=<YYYY-MM of tradable_at>/stories.parquet   # derived: dedupe clusters
    raw/<source>/<native_id>/<name>.gz          # verbatim source documents for re-parsing
    newsfeed.duckdb                              # runs, failures, alerts
    state/                                       # collector heartbeat/state JSON, lock files

Concurrency (several processes, e.g. ``collect`` loop + a ``backfill``):
  * Parquet is read through an *in-memory* DuckDB connection, so readers never take the
    database-file lock. Parquet parts have unique names and are written to a temp file then
    atomically renamed, so concurrent writers never clobber or half-write each other.
  * The operational DuckDB file is opened per operation and closed immediately, retrying
    while another process holds it.
  * Derived-table rebuilds are serialised with a lock file (:class:`FileLock`).

Idempotency: rows whose item_id already exists are dropped before writing; if a race still
produces duplicates, the ``items`` view keeps one row per item_id (live rows first, then the
earliest ``first_seen_at``).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import platform
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from newsfeed.config import AppConfig, PathsConfig
from newsfeed.timeutil import ensure_utc, utcnow

log = logging.getLogger(__name__)

TS = pa.timestamp("us", tz="UTC")
ITEMS_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("source", pa.string()),
        ("native_id", pa.string()),
        ("publisher", pa.string()),
        ("url", pa.string()),
        ("url_norm", pa.string()),
        ("title", pa.string()),
        ("body_text", pa.string()),
        ("body_tokens_est", pa.int32()),
        ("body_truncated", pa.bool_()),
        ("content_hash", pa.string()),
        ("tickers", pa.list_(pa.string())),
        ("published_at", TS),
        ("published_at_raw", pa.string()),
        ("first_seen_at", TS),
        ("tradable_at", TS),
        ("ingest_mode", pa.string()),
        ("tradable_rule", pa.string()),
        ("meta", pa.string()),
        ("raw_json", pa.string()),
        ("parser_version", pa.string()),
        ("run_id", pa.string()),
        ("ingested_at", TS),
        # sha256 hex of title + "\n" + body (item_text_hash); stored since 2026-10-22 so that
        # lookups by text never read or hash the bodies of the whole lake (older files gained it
        # via `newsfeed migrate-text-hash`; same value the view used to compute)
        pa.field("text_hash", pa.string(), nullable=False),
    ]
)


TEXT_HASH_MARKER = "_text_hash_migrated"  # in lake/items/: every file stores text_hash


def item_text_hash(title: str | None, body: str | None) -> str:
    """The items' text hash, identical to DuckDB's
    sha256(coalesce(title, '') || chr(10) || coalesce(body_text, '')) (verified on stored rows,
    incl. non-ASCII text)."""
    return hashlib.sha256(((title or "") + "\n" + (body or "")).encode("utf-8")).hexdigest()


def _with_text_hash(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "text_hash": item_text_hash(r.get("title"), r.get("body_text"))} for r in rows]


ITEM_TICKERS_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("ticker", pa.string()),  # NULL row = "linked, nothing found" marker
        ("cik", pa.int64()),
        ("method", pa.string()),  # source | exchange_tag | cashtag | name
        ("confidence", pa.float32()),
        ("n_mentions", pa.int32()),
        ("in_title", pa.bool_()),
        ("linker_id", pa.string()),
        ("text_hash", pa.string()),  # items.text_hash this link was computed from
        ("linked_at", TS),
    ]
)

ITEM_STORY_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("story_id", pa.string()),  # = item_id of the story's first (founding) item
        ("reason", pa.string()),  # new | url | exact | near
        ("similarity", pa.float32()),
        ("matched_item_id", pa.string()),
        ("tradable_at", TS),
        ("published_at", TS),
        ("dedupe_version", pa.int32()),
        ("config_hash", pa.string()),
        ("built_at", TS),
    ]
)

SCORES_SCHEMA = pa.schema(
    [
        (
            "cache_key",
            pa.string(),
        ),  # sha256(item_id, ticker, model, prompt_hash, masked, replicate)
        ("item_id", pa.string()),
        ("ticker", pa.string()),
        ("cik", pa.int64()),
        ("model", pa.string()),
        ("prompt_version", pa.string()),
        ("prompt_hash", pa.string()),
        ("schema_version", pa.string()),
        ("masked", pa.bool_()),
        ("replicate", pa.int32()),  # 0 = the canonical draw; 1..K = noise-study re-draws
        ("run_id", pa.string()),
        ("mode", pa.string()),  # live | batch
        ("batch_id", pa.string()),
        ("request_id", pa.string()),
        ("status", pa.string()),  # ok | invalid | refusal | max_tokens | empty
        ("error", pa.string()),
        ("relevance", pa.string()),
        ("sentiment", pa.float64()),
        ("confidence", pa.float64()),
        ("event_type", pa.string()),
        ("novelty", pa.string()),
        ("materiality", pa.string()),
        ("surprise_vs_expectations", pa.float64()),
        ("rationale", pa.string()),
        ("rationale_words", pa.int32()),
        ("mask_residuals", pa.int32()),  # identifying terms still present after masking
        ("stop_reason", pa.string()),
        ("input_tokens", pa.int64()),
        ("cache_creation_input_tokens", pa.int64()),
        ("cache_read_input_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("cost_usd", pa.float64()),
        ("text_hash", pa.string()),  # items.text_hash of the scored text (part of cache_key)
        ("mask_id", pa.string()),  # mask lexicon hash for masked rows (part of cache_key)
        ("user_text_hash", pa.string()),  # hash of the exact user message sent
        ("raw_response", pa.string()),  # full API response JSON
        ("scored_at", TS),
    ]
)

BASELINES_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("text_hash", pa.string()),
        ("method", pa.string()),  # lm | finbert
        ("method_version", pa.string()),  # LM csv sha256[:12] | model name
        ("computed_at", TS),
        ("score", pa.float64()),  # lm_tone or p_pos - p_neg, in [-1, 1]
        ("n_words", pa.int64()),
        ("n_positive", pa.int64()),
        ("n_negative", pa.int64()),
        ("n_uncertainty", pa.int64()),
        ("n_litigious", pa.int64()),
        ("neg_prop", pa.float64()),
        ("pos_prop", pa.float64()),
        ("unc_prop", pa.float64()),
        ("lit_prop", pa.float64()),
        ("p_positive", pa.float64()),
        ("p_negative", pa.float64()),
        ("p_neutral", pa.float64()),
    ]
)

EMBEDDINGS_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("text_hash", pa.string()),
        ("model", pa.string()),
        ("dim", pa.int32()),
        ("vector", pa.list_(pa.float32())),
        ("computed_at", TS),
    ]
)

_OPS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    command VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status VARCHAR NOT NULL,
    config_hash VARCHAR,
    git_sha VARCHAR,
    versions JSON,
    params JSON,
    stats JSON
);
CREATE TABLE IF NOT EXISTS failures (
    run_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    native_id VARCHAR NOT NULL,
    error VARCHAR,
    failed_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    raised_at TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    message VARCHAR,
    run_id VARCHAR
);
-- one row per LLM call attempt (incl. failures) -> `newsfeed cost`
CREATE TABLE IF NOT EXISTS llm_usage (
    called_at TIMESTAMPTZ NOT NULL,
    run_id VARCHAR,
    mode VARCHAR,                 -- live | batch | count_tokens
    model VARCHAR,
    cache_key VARCHAR,
    batch_id VARCHAR,
    request_id VARCHAR,
    status VARCHAR,               -- ok|invalid|refusal|max_tokens|empty|error|expired|...
    input_tokens BIGINT,
    cache_creation_input_tokens BIGINT,
    cache_read_input_tokens BIGINT,
    output_tokens BIGINT,
    cost_usd DOUBLE,
    error VARCHAR
);
CREATE TABLE IF NOT EXISTS llm_batches (
    batch_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    model VARCHAR,
    prompt_hash VARCHAR,
    n_requests INTEGER,
    submitted_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    processing_status VARCHAR,
    collected BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS llm_batch_requests (
    batch_id VARCHAR NOT NULL,
    custom_id VARCHAR NOT NULL,
    cache_key VARCHAR NOT NULL,
    target JSON NOT NULL          -- everything needed to build the score row on collection
);
-- collected ranges on the tradable_at axis, [start_at, end_at) -- see coverage.py
CREATE TABLE IF NOT EXISTS coverage (
    source VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    start_at TIMESTAMPTZ NOT NULL,   -- backfill: published-time start; live: tradable time
    end_at TIMESTAMPTZ NOT NULL,
    tradable_start_at TIMESTAMPTZ NOT NULL,  -- backfill: push(start + lag); live: = start_at
    kind VARCHAR NOT NULL,        -- backfill | live
    run_id VARCHAR,
    recorded_at TIMESTAMPTZ NOT NULL
);
-- every prompt variant / feature variant ever tried (multiple-testing accounting)
CREATE TABLE IF NOT EXISTS experiments (
    kind VARCHAR NOT NULL,        -- prompt | feature_set | model | ...
    key VARCHAR NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    run_id VARCHAR,
    details JSON,
    PRIMARY KEY (kind, key)
);
"""


def new_run_id(now: datetime | None = None) -> str:
    now = ensure_utc(now or utcnow())
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _versions() -> dict[str, str]:
    pkgs = ["newsfeed", "exchange-calendars", "duckdb", "pyarrow", "polars", "httpx", "datasketch"]
    out = {"python": platform.python_version()}
    for p in pkgs:
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[p] = metadata.version(p)
    return out


def _replace_with_retry(src: Path, dst: Path, attempts: int = 20) -> None:
    """os.replace, retrying briefly: on Windows a reader holding ``dst`` open blocks it."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def write_table_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    pq.write_table(table, tmp)
    _replace_with_retry(tmp, path)


class FileLock:
    """Cross-process exclusive lock via O_EXCL lock file. Stale locks (not touched for
    ``stale_after`` seconds, e.g. from a killed process) are broken. The holder touches the file
    every ``stale_after / 4`` seconds (at most every 60 s) from a daemon thread, so a long job
    is never taken for dead: on 2026-10-21 a 35-min relink lost its lock to the collector,
    which started a second full relink next to it (two ~15 GB processes on a 16 GB laptop).
    PID liveness isn't used: on Windows ``os.kill(pid, 0)`` terminates the process."""

    def __init__(self, path: Path, stale_after: float = 1800.0) -> None:
        self.path = path
        self.stale_after = stale_after
        self._held = False
        self._stop: threading.Event | None = None
        self._beat: threading.Thread | None = None

    def _heartbeat(self, stop: threading.Event) -> None:
        every = min(60.0, self.stale_after / 4)
        while not stop.wait(every):
            with contextlib.suppress(OSError):
                os.utime(self.path, None)

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.stale_after:
                    log.warning("breaking stale lock %s (age %.0fs)", self.path, age)
                    with contextlib.suppress(FileNotFoundError):
                        self.path.unlink()
                    continue
                return False
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"pid": os.getpid(), "at": utcnow().isoformat()}))
            self._held = True
            self._stop = threading.Event()
            self._beat = threading.Thread(
                target=self._heartbeat, args=(self._stop,), daemon=True, name="lock-heartbeat"
            )
            self._beat.start()
            return True
        return False

    def release(self) -> None:
        if self._held:
            if self._stop is not None and self._beat is not None:
                self._stop.set()
                self._beat.join(timeout=5)
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
            self._held = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


class RawStore:
    """Gzipped verbatim documents, addressable by (source, native_id, name)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, source: str, native_id: str, name: str) -> Path:
        for part in (source, native_id, name):
            if not part or "/" in part or "\\" in part or part in (".", ".."):
                raise ValueError(f"unsafe raw-store path component {part!r}")
        return self.root / source / native_id / f"{name}.gz"

    def put(self, source: str, native_id: str, name: str, data: bytes) -> Path:
        p = self.path(source, native_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:8]}.tmp")
        with gzip.open(tmp, "wb") as f:
            f.write(data)
        _replace_with_retry(tmp, p)
        return p

    def get(self, source: str, native_id: str, name: str) -> bytes | None:
        p = self.path(source, native_id, name)
        if not p.exists():
            return None
        with gzip.open(p, "rb") as f:
            return f.read()


class Store:
    def __init__(self, paths: PathsConfig, *, lock_timeout: float = 60.0) -> None:
        self.paths = paths
        self.lake = paths.lake_dir
        self.items_dir = self.lake / "items"
        self.tickers_dir = self.lake / "item_tickers"
        self.story_dir = self.lake / "item_story"
        self.scores_dir = self.lake / "scores"
        self.baselines_dir = self.lake / "baselines"
        self.embeddings_dir = self.lake / "embeddings"
        self.features_dir = self.lake / "features"
        self.state_dir = paths.data_dir / "state"
        for d in (
            self.items_dir,
            self.tickers_dir,
            self.story_dir,
            self.baselines_dir,
            self.embeddings_dir,
            self.features_dir,
            self.scores_dir,
            self.state_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        paths.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = RawStore(paths.raw_dir)
        self.lock_timeout = lock_timeout
        self.q = duckdb.connect()  # in-memory; reads Parquet only, holds no file locks
        self.q.execute("SET TimeZone = 'UTC'")
        with self.ops() as con:
            con.execute(_OPS_DDL)
        self._seq = 0
        self._known_ids: set[str] | None = None
        self.refresh_views()

    # ------------------------------------------------------------- ops database
    @contextlib.contextmanager
    def ops(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Short-lived connection to the operational DuckDB file (retries while locked)."""
        deadline = time.monotonic() + self.lock_timeout
        delay = 0.05
        while True:
            try:
                con = duckdb.connect(str(self.paths.db_path))
                break
            except duckdb.IOException:
                if time.monotonic() > deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
        try:
            con.execute("SET TimeZone = 'UTC'")
            yield con
        finally:
            con.close()

    def ops_query(self, sql: str, params: Any = None) -> pl.DataFrame:
        with self.ops() as con:
            return con.execute(sql, params or []).pl()

    # ----------------------------------------------------------------- views
    def _empty_path(self, name: str, schema: pa.Schema) -> Path:
        """An empty, correctly-typed Parquet file for ``schema``, named by a hash of the schema:
        a schema change writes a *new* file instead of replacing one that another process may
        be reading (on Windows that replace fails; it crashed a migration on 2026-10-22)."""
        tag = hashlib.sha256(schema.to_string().encode()).hexdigest()[:8]
        empty = self.lake / f"_empty_{name}_{tag}.parquet"
        if not empty.exists():
            write_table_atomic(schema.empty_table(), empty)
        return empty

    def _source_sql(self, name: str, directory: Path, schema: pa.Schema) -> str:
        """read_parquet over ``directory`` unioned with an empty, correctly-typed file, so the
        view exists before the first write and always exposes every current column."""
        files = [self._empty_path(name, schema).as_posix()]
        if any(directory.rglob("*.parquet")):
            files.append((directory / "**" / "*.parquet").as_posix())
        return f"read_parquet({files!r}, union_by_name = true)"

    @property
    def text_hash_migrated(self) -> bool:
        """True once `newsfeed migrate-text-hash` has verified every item file stores
        text_hash (marker file); writers always store it, so the view can then use the column
        without reading any body text."""
        return (self.items_dir / TEXT_HASH_MARKER).exists()

    def _items_sql(self, src: str, where: str = "") -> str:
        """The ``items`` view body: one canonical row per item_id. ``where`` may only filter on
        item_id (the window's partition key), so filtering before the window is identical to
        filtering the view. text_hash: the stored column after the migration; before it,
        computed where a file predates the column (same value)."""
        th = (
            "text_hash"
            if self.text_hash_migrated
            else "coalesce(text_hash, "
            "sha256(coalesce(title, '') || chr(10) || coalesce(body_text, '')))"
        )
        return f"""
            SELECT * EXCLUDE (_rn) REPLACE (coalesce(publisher, source) AS publisher,
                                            {th} AS text_hash)
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY item_id
                    -- live row first; then earliest sighting; among rows with identical timing
                    -- (a re-parse copies it) the newest parser version wins
                    ORDER BY (ingest_mode = 'live') DESC, first_seen_at, parser_version DESC,
                             ingested_at, run_id
                ) AS _rn
                FROM {src} {where}
            ) WHERE _rn = 1
            """

    def migrate_text_hash(self, progress: Callable[[int, int], None] | None = None) -> dict:
        """Add the stored ``text_hash`` column to item files written before it existed. Each
        file: read without partition columns, hash = ``item_text_hash`` (checked against
        DuckDB's sha256 on the same table), atomic replace; nothing else changes. Idempotent.
        Writes the marker only when no item row anywhere lacks a hash."""
        files = sorted(self.items_dir.rglob("*.parquet"))
        done = skipped = 0
        con = duckdb.connect()
        for k, f in enumerate(files, 1):
            with pq.ParquetFile(f) as pf:  # closed before the replace (Windows locks open files)
                nulls = None
                if "text_hash" in pf.schema_arrow.names:
                    col = pf.schema_arrow.names.index("text_hash")
                    md = pf.metadata
                    nulls = sum(
                        md.row_group(g).column(col).statistics.null_count
                        if md.row_group(g).column(col).statistics is not None
                        else 1
                        for g in range(md.num_row_groups)
                    )
                tbl = None if nulls == 0 else pf.read()
            if tbl is None:
                skipped += 1
                continue
            if "text_hash" in tbl.column_names:
                tbl = tbl.drop_columns(["text_hash"])
            titles, bodies = tbl.column("title").to_pylist(), tbl.column("body_text").to_pylist()
            hashes = pa.array(
                [item_text_hash(t, b) for t, b in zip(titles, bodies, strict=True)], pa.string()
            )
            tbl = tbl.append_column(pa.field("text_hash", pa.string(), nullable=False), hashes)
            con.register("t", tbl)
            bad = con.execute(
                "SELECT count(*) FROM t WHERE text_hash IS DISTINCT FROM "
                "sha256(coalesce(title, '') || chr(10) || coalesce(body_text, ''))"
            ).fetchone()[0]
            con.unregister("t")
            if bad:
                raise RuntimeError(f"{f}: {bad} hashes differ from DuckDB's sha256; not written")
            write_table_atomic(tbl, f)
            done += 1
            if progress is not None:
                progress(k, len(files))
        missing = self.rows_without_text_hash()
        if missing == 0:
            (self.items_dir / TEXT_HASH_MARKER).write_text(utcnow().isoformat(), encoding="utf-8")
            self.refresh_views()
        return {"files": len(files), "migrated": done, "already": skipped, "missing": missing}

    def rows_without_text_hash(self) -> int:
        """Item rows (raw files, not the view) with no stored hash; reads only that column."""
        if not any(self.items_dir.rglob("*.parquet")):
            return 0
        glob = (self.items_dir / "**" / "*.parquet").as_posix()
        empty = self._empty_path("items", ITEMS_SCHEMA).as_posix()
        return int(
            self.q.execute(
                f"SELECT count(*) FROM read_parquet([{empty!r}, {glob!r}], union_by_name = true) "
                "WHERE text_hash IS NULL"
            ).fetchone()[0]
        )

    def check_text_hash(self) -> bool:
        """Guard for the stored-hash view: if any row lacks a hash (a file written by code from
        before the column), drop the marker so the view computes hashes again. True if OK."""
        if not self.text_hash_migrated:
            return False
        n = self.rows_without_text_hash()
        if n:
            log.error("%d item rows lack text_hash; view falls back to computing it -- run "
                      "`newsfeed migrate-text-hash`", n)  # fmt: skip
            with contextlib.suppress(FileNotFoundError):
                (self.items_dir / TEXT_HASH_MARKER).unlink()
            self.refresh_views()
            return False
        return True

    def refresh_views(self) -> None:
        items_src = self._source_sql("items", self.items_dir, ITEMS_SCHEMA)
        self.q.execute(f"CREATE OR REPLACE VIEW items AS {self._items_sql(items_src)}")
        tick_src = self._source_sql("item_tickers", self.tickers_dir, ITEM_TICKERS_SCHEMA)
        self.q.execute(
            f"""
            CREATE OR REPLACE VIEW item_tickers_all AS
            SELECT * FROM {tick_src}
            QUALIFY linked_at = max(linked_at) OVER (PARTITION BY item_id)
            """
        )
        self.q.execute(
            "CREATE OR REPLACE VIEW item_tickers AS "
            "SELECT * FROM item_tickers_all WHERE ticker IS NOT NULL"
        )
        story_src = self._source_sql("item_story", self.story_dir, ITEM_STORY_SCHEMA)
        self.q.execute(f"CREATE OR REPLACE VIEW item_story AS SELECT * FROM {story_src}")
        # scores_all: every attempt; scores: one row per cache_key -- the first `ok` draw if any
        # (re-tries of invalid/refused rows can only add, never replace, a canonical result).
        base_src = self._source_sql("baselines", self.baselines_dir, BASELINES_SCHEMA)
        self.q.execute(
            f"""
            CREATE OR REPLACE VIEW baselines AS SELECT * FROM {base_src}
            QUALIFY row_number() OVER (
                PARTITION BY item_id, text_hash, method, method_version ORDER BY computed_at
            ) = 1
            """
        )
        emb_src = self._source_sql("embeddings", self.embeddings_dir, EMBEDDINGS_SCHEMA)
        self.q.execute(
            f"""
            CREATE OR REPLACE VIEW embeddings AS SELECT * FROM {emb_src}
            QUALIFY row_number() OVER (
                PARTITION BY item_id, text_hash, model ORDER BY computed_at
            ) = 1
            """
        )
        scores_src = self._source_sql("scores", self.scores_dir, SCORES_SCHEMA)
        self.q.execute(f"CREATE OR REPLACE VIEW scores_all AS SELECT * FROM {scores_src}")
        self.q.execute(
            """
            CREATE OR REPLACE VIEW scores AS
            SELECT * FROM scores_all
            QUALIFY row_number() OVER (
                PARTITION BY cache_key ORDER BY (status = 'ok') DESC, scored_at, run_id
            ) = 1
            """
        )

    # ----------------------------------------------------------------- items
    def known_ids(self) -> set[str]:
        if self._known_ids is None:
            rows = self.q.execute("SELECT DISTINCT item_id FROM items").fetchall()
            self._known_ids = {r[0] for r in rows}
        return self._known_ids

    def refresh_known_ids(self) -> None:
        """Forget the cached id set (another process may have written since)."""
        self._known_ids = None

    def has_item(self, item_id: str) -> bool:
        return item_id in self.known_ids()

    def _part_path(self, directory: Path, run_id: str) -> Path:
        self._seq += 1
        return directory / f"part-{run_id}-{self._seq:05d}-{uuid.uuid4().hex[:6]}.parquet"

    def write_items(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        """Append rows not already stored. Returns number written."""
        known = self.known_ids()
        fresh: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r["item_id"] not in known and r["item_id"] not in fresh:
                fresh[r["item_id"]] = r
        if not fresh:
            return 0
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in fresh.values():
            groups[(r["source"], f"{r['published_at']:%Y-%m}")].append(r)
        for (source, ym), grp in groups.items():
            d = self.items_dir / f"src={source}" / f"ym={ym}"
            write_table_atomic(
                pa.Table.from_pylist(_with_text_hash(grp), schema=ITEMS_SCHEMA),
                self._part_path(d, run_id),
            )
        known.update(fresh)
        self.refresh_views()
        return len(fresh)

    def write_reparsed_items(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        """Append new parses of *existing* items (bypasses the anti-join). Rows must carry the
        original timing fields verbatim; the view then prefers the newer parser_version."""
        if not rows:
            return 0
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[(r["source"], f"{r['published_at']:%Y-%m}")].append(r)
        for (source, ym), grp in groups.items():
            d = self.items_dir / f"src={source}" / f"ym={ym}"
            write_table_atomic(
                pa.Table.from_pylist(_with_text_hash(grp), schema=ITEMS_SCHEMA),
                self._part_path(d, run_id),
            )
        self.refresh_views()
        return len(rows)

    def write_ticker_links(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        if not rows:
            return 0
        table = pa.Table.from_pylist(list(rows), schema=ITEM_TICKERS_SCHEMA)
        write_table_atomic(table, self._part_path(self.tickers_dir, run_id))
        self.refresh_views()
        return len(rows)

    def write_features(self, df: pl.DataFrame, feature_set_id: str, run_id: str) -> int:
        """Append a built feature table (wide; schema varies by feature set)."""
        if df.height == 0:
            return 0
        out = df.with_columns(
            pl.lit(feature_set_id).alias("feature_set_id"),
            pl.lit(run_id).alias("build_run_id"),
            pl.lit(utcnow()).dt.convert_time_zone("UTC").alias("built_at"),
        )
        d = self.features_dir / f"set={feature_set_id}"
        write_table_atomic(out.to_arrow(), self._part_path(d, run_id))
        return out.height

    def features(self, feature_set_id: str) -> pl.DataFrame:
        """Latest build per (ticker, as_of) for one feature set."""
        files = list((self.features_dir / f"set={feature_set_id}").glob("*.parquet"))
        if not files:
            return pl.DataFrame()
        glob = (self.features_dir / f"set={feature_set_id}" / "*.parquet").as_posix()
        return self.q.execute(
            f"""SELECT * FROM read_parquet('{glob}', union_by_name = true,
                                             hive_partitioning = false)
                QUALIFY row_number() OVER (PARTITION BY ticker, as_of ORDER BY built_at DESC) = 1
                ORDER BY as_of, ticker"""
        ).pl()

    def write_embeddings(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        if not rows:
            return 0
        table = pa.Table.from_pylist(list(rows), schema=EMBEDDINGS_SCHEMA)
        write_table_atomic(table, self._part_path(self.embeddings_dir, run_id))
        self.refresh_views()
        return len(rows)

    def write_baselines(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        if not rows:
            return 0
        table = pa.Table.from_pylist(list(rows), schema=BASELINES_SCHEMA)
        write_table_atomic(table, self._part_path(self.baselines_dir, run_id))
        self.refresh_views()
        return len(rows)

    def write_scores(self, rows: Sequence[dict[str, Any]], run_id: str) -> int:
        """Append score rows (append-only; the `scores` view picks the canonical row)."""
        if not rows:
            return 0
        table = pa.Table.from_pylist(list(rows), schema=SCORES_SCHEMA)
        write_table_atomic(table, self._part_path(self.scores_dir, run_id))
        self.refresh_views()
        return len(rows)

    def replace_story_months(self, months: dict[str, Sequence[dict[str, Any]]]) -> None:
        """Atomically replace whole month files of the item_story table."""
        for ym, rows in months.items():
            path = self.story_dir / f"ym={ym}" / "stories.parquet"
            write_table_atomic(pa.Table.from_pylist(list(rows), schema=ITEM_STORY_SCHEMA), path)
        self.refresh_views()

    def items_by_id(
        self, item_ids: Iterable[str], *, sources: Iterable[str] | None = None, columns: str = "*"
    ) -> pl.DataFrame:
        """Rows of the ``items`` view for these ids, without scanning the whole lake: ids are
        filtered before the dedupe window (same result, see ``_items_sql``) and, with
        ``sources``, only those ``src=`` partitions are read (the 10-K / 10-Q full texts are
        most of the lake). Measured 2026-10-21: 5 ids 204 s via the view, 4.2 s here. A lookup
        by id, not an as-of read: callers that need point-in-time selection still go through
        ``items_as_of``."""
        ids = list(item_ids)
        if not ids:
            return self.query(f"SELECT {columns} FROM items LIMIT 0")
        rel = self.items_relation(sources, where="WHERE item_id IN (SELECT unnest($ids))")
        return self.query(f"SELECT {columns} FROM {rel}", {"ids": ids})

    def items_relation(self, sources: Iterable[str] | None = None, *, where: str = "") -> str:
        """SQL relation equal to ``items`` restricted to ``sources`` (None = all), reading only
        their ``src=<source>`` partitions: every item is stored under its own source, so this is
        the same as filtering the view on ``source`` but without scanning (and hashing) the
        rest of the lake. ``where`` may only filter on item_id (see ``_items_sql``)."""
        if sources is None:
            src = self._source_sql("items", self.items_dir, ITEMS_SCHEMA)
        else:
            files = [self._empty_path("items", ITEMS_SCHEMA).as_posix()]
            for s in sources:
                d = self.items_dir / f"src={s}"
                if any(d.rglob("*.parquet")):
                    files.append((d / "**" / "*.parquet").as_posix())
            src = f"read_parquet({files!r}, union_by_name = true)"
        return f"({self._items_sql(src, where)})"

    def items_as_of(
        self,
        as_of: datetime,
        *,
        tickers: Iterable[str] | None = None,
        source: str | None = None,
        columns: str = "*",
        item_ids: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """THE point-in-time accessor: only items with ``tradable_at <= as_of``.

        Downstream code (features, scoring selection) must read items through this method
        (or SQL built on the same predicate, e.g. ``dedupe.stories_as_of``), never through the
        raw view, so the causality guarantee lives in one place.
        """
        as_of = ensure_utc(as_of, name="as_of")
        where = ["tradable_at <= $as_of"]
        params: dict[str, Any] = {"as_of": as_of}
        if tickers is not None:
            where.append("list_has_any(tickers, $tickers)")
            params["tickers"] = list(tickers)
        if source is not None:
            where.append("source = $source")
            params["source"] = source
        if item_ids is not None:
            where.append("item_id IN (SELECT unnest($item_ids))")
            params["item_ids"] = list(item_ids)
        sql = f"SELECT {columns} FROM items WHERE {' AND '.join(where)} ORDER BY tradable_at"
        return self.q.execute(sql, params).pl()

    def query(self, sql: str, params: Any = None) -> pl.DataFrame:
        """Ad-hoc read over the Parquet views (items, item_tickers, item_story)."""
        return self.q.execute(sql, params or []).pl()

    # ------------------------------------------------------------------ runs
    def start_run(self, command: str, cfg: AppConfig, params: dict[str, Any]) -> str:
        run_id = new_run_id()
        with self.ops() as con:
            con.execute(
                "INSERT INTO runs VALUES (?, ?, ?, NULL, 'running', ?, ?, ?, ?, NULL)",
                [
                    run_id,
                    command,
                    utcnow(),
                    cfg.config_hash(),
                    _git_sha(),
                    json.dumps(_versions()),
                    json.dumps(params, default=str),
                ],
            )
        return run_id

    def update_run_stats(self, run_id: str, stats: dict[str, Any]) -> None:
        with self.ops() as con:
            con.execute(
                "UPDATE runs SET stats = ? WHERE run_id = ?",
                [json.dumps(stats, default=str), run_id],
            )

    def finish_run(self, run_id: str, status: str, stats: dict[str, Any]) -> None:
        with self.ops() as con:
            con.execute(
                "UPDATE runs SET finished_at = ?, status = ?, stats = ? WHERE run_id = ?",
                [utcnow(), status, json.dumps(stats, default=str), run_id],
            )

    def record_failure(self, run_id: str, source: str, native_id: str, error: str) -> None:
        with self.ops() as con:
            con.execute(
                "INSERT INTO failures VALUES (?, ?, ?, ?, ?)",
                [run_id, source, native_id, error[:2000], utcnow()],
            )

    def record_alert(self, source: str, kind: str, message: str, run_id: str | None) -> None:
        with self.ops() as con:
            con.execute(
                "INSERT INTO alerts VALUES (?, ?, ?, ?, ?)",
                [utcnow(), source, kind, message[:2000], run_id],
            )

    def close(self) -> None:
        self.q.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def month_key(ts: datetime) -> str:
    return f"{ensure_utc(ts):%Y-%m}"


__all__ = [
    "ITEMS_SCHEMA",
    "ITEM_STORY_SCHEMA",
    "ITEM_TICKERS_SCHEMA",
    "FileLock",
    "RawStore",
    "Store",
    "month_key",
    "new_run_id",
    "write_table_atomic",
]
