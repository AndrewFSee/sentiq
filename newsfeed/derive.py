"""Derived tables, rebuilt incrementally after ingestion: ticker links, then stories.

Both are deterministic functions of the stored items (+ config), so they can always be
rebuilt from scratch; incremental runs only touch what changed. A lock file serialises
rebuilds across processes (``collect`` loop and a concurrent ``backfill``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from newsfeed.config import AppConfig
from newsfeed.dedupe import rebuild_stories
from newsfeed.features.baselines import compute_baselines, load_lm_dictionary, make_finbert
from newsfeed.features.embeddings import Embedder, FastEmbedder, compute_embeddings
from newsfeed.storage import FileLock, Store
from newsfeed.tickers import TickerLinker
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)


def relink(
    store: Store,
    linker: TickerLinker,
    run_id: str,
    batch_size: int = 5000,
    force_sources: list[str] | None = None,
) -> int:
    """Link every item without links from the current ``linker_id`` for its current text and
    newer than its stored row (a re-parsed item gets relinked even when only its source tags
    changed). Returns items linked."""
    # streamed in batches: all bodies at once (8-Ks + news + ~40k-char transcripts) exceed a
    # single 2 GB Arrow buffer
    cur = store.q.cursor()  # own connection: writes below must not end this stream
    reader = cur.execute(
        """
        SELECT i.item_id, i.source, i.title, i.body_text, i.tickers, i.text_hash, i.meta
        FROM items i
        WHERE list_contains($force, i.source) OR NOT EXISTS (
            SELECT 1 FROM item_tickers_all t
            WHERE t.item_id = i.item_id AND t.linker_id = $lid AND t.text_hash = i.text_hash
              -- links must postdate the item row they describe: a re-parsed item can change
              -- its source tags without changing its text
              AND t.linked_at >= i.ingested_at
        )
        """,
        {"lid": linker.linker_id, "force": list(force_sources or [])},
    ).to_arrow_reader(batch_size)
    total = 0
    for batch in reader:
        pending = pl.from_arrow(batch).to_dicts()
        if pending:
            _link_batch(store, linker, pending, run_id)
            total += len(pending)
    cur.close()
    return total


def _meta_cik(r: dict[str, Any]) -> int | None:
    try:
        cik = json.loads(r.get("meta") or "{}").get("cik")
        return int(cik) if cik is not None else None
    except (ValueError, TypeError):
        return None


def _link_batch(
    store: Store, linker: TickerLinker, pending: list[dict[str, Any]], run_id: str
) -> None:
    now = utcnow()
    rows: list[dict[str, Any]] = []
    for r in pending:
        links = linker.link(
            title=r["title"],
            body=r["body_text"],
            source_tickers=r["tickers"],
            source=r["source"],
            source_cik=_meta_cik(r) if r["source"] in linker.source_only_sources else None,
        )
        base = {
            "item_id": r["item_id"],
            "linker_id": linker.linker_id,
            "text_hash": r["text_hash"],
            "linked_at": now,
        }
        if not links:  # marker so the item is not re-linked every cycle
            rows.append(
                {
                    **base,
                    "ticker": None,
                    "cik": None,
                    "method": "none",
                    "confidence": 0.0,
                    "n_mentions": 0,
                    "in_title": False,
                }
            )
        for lk in links:
            rows.append(
                {
                    **base,
                    "ticker": lk.ticker,
                    "cik": lk.cik,
                    "method": lk.method,
                    "confidence": lk.confidence,
                    "n_mentions": lk.n_mentions,
                    "in_title": lk.in_title,
                }
            )
    store.write_ticker_links(rows, run_id)


def derive(
    store: Store,
    cfg: AppConfig,
    linker: TickerLinker,
    run_id: str,
    *,
    full_stories: bool = False,
    wait: bool = False,
    embedder: Embedder | None = None,
    relink_sources: list[str] | None = None,
    if_changed: bool = False,
) -> dict[str, Any]:
    """Relink, rebuild stories, compute text baselines for story founders. If another process
    holds the lock, skip (or raise if ``wait``) -- every step is self-healing next cycle.

    ``if_changed`` (the collect loop): return ``{"unchanged": True}`` without touching anything
    when the item lake and the derive inputs are exactly as at the last successful derive.
    Each step otherwise scans every stored item (~2 min at 300k items), which made collect
    cycles overrun their interval although almost every cycle ingests nothing."""
    lock = FileLock(store.state_dir / "derive.lock")
    if not lock.acquire():
        if wait:
            raise RuntimeError(f"derive lock held: {lock.path}")
        log.info("derive skipped: lock held by another process")
        return {"skipped": True}
    try:
        fp = lake_fingerprint(store, derive_inputs(cfg, linker), cfg.dedupe.excluded_sources)
        fp_path = store.state_dir / "derive-fingerprint.json"
        quick = if_changed and not full_stories and not relink_sources
        if quick and fp_path.exists() and json.loads(fp_path.read_text())["fingerprint"] == fp:
            return {"unchanged": True}
        linked = relink(store, linker, run_id, force_sources=relink_sources)
        stories = rebuild_stories(store, cfg.dedupe, linker.linker_id, full=full_stories)
        out = {
            "linked": linked,
            **stories,
            "baselines": derive_baselines(store, cfg, run_id),
            "embeddings": derive_embeddings(store, cfg, run_id, embedder),
        }
        # taken before the steps ran: items written meanwhile change it, so they are derived
        # next time
        fp_path.write_text(json.dumps({"fingerprint": fp, "run_id": run_id}))
        return out
    finally:
        lock.release()


def derive_inputs(cfg: AppConfig, linker: TickerLinker) -> str:
    """Everything besides the items that decides what derive produces."""
    fc = cfg.features
    return json.dumps(
        [
            linker.linker_id,
            cfg.dedupe.model_dump(mode="json"),
            str(fc.lm_dictionary),
            fc.finbert,
            fc.finbert_model,
            fc.embeddings,
            fc.embedding_model,
        ],
        sort_keys=True,
    )


def lake_fingerprint(store: Store, inputs: str, ignore_sources: list[str] | None = None) -> str:
    """Hash of the item lake's Parquet files (count, total size, newest mtime) + ``inputs``.
    Items are append-only and written by atomic rename, so any new item changes it.
    ``ignore_sources`` (``dedupe.excluded_sources``: no stories, not scored live) are left out,
    so a long 10-K backfill does not make every collect cycle re-derive; their links are made
    by the backfill's own derive or the next derive that runs for other items."""
    skip = {f"src={s}" for s in ignore_sources or []}
    n = size = newest = 0
    for root, dirs, files in os.walk(store.items_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            if name.endswith(".parquet"):
                st = os.stat(os.path.join(root, name))
                n, size, newest = n + 1, size + st.st_size, max(newest, st.st_mtime_ns)
    blob = f"{inputs}|{n}|{size}|{newest}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def derive_baselines(store: Store, cfg: AppConfig, run_id: str) -> dict[str, int]:
    fc = cfg.features
    try:
        lm = load_lm_dictionary(fc.lm_dictionary)
    except FileNotFoundError as exc:
        log.warning("LM baseline skipped: %s", exc)
        lm = None
    fb = (fc.finbert_model, make_finbert(fc.finbert_model)) if fc.finbert else None
    return compute_baselines(store, lm=lm, finbert=fb, run_id=run_id)


@lru_cache(maxsize=2)
def _fast_embedder(model: str, cache_dir: str) -> FastEmbedder:
    return FastEmbedder(model, Path(cache_dir))  # model load takes seconds: once per process


def derive_embeddings(
    store: Store, cfg: AppConfig, run_id: str, embedder: Embedder | None = None
) -> int:
    fc = cfg.features
    if not fc.embeddings:
        return 0
    emb = embedder or _fast_embedder(fc.embedding_model, str(fc.embedding_cache))
    return compute_embeddings(store, emb, run_id)
