"""What to score: (story-founding item, company) pairs, minus anything cached or in flight.

Selection rules (config ``scoring``):
  * founders_only: one score per story, on its first appearance (the story's founding item).
  * link_policy: a link qualifies if its method is in ``methods`` (source, exchange_tag,
    cashtag, name_title) or it is a body-only name mention with ``n_mentions >= min``.
  * One target per (item, company): share classes (GOOG/GOOGL) are scored once, under the
    company's canonical ticker = first of its tickers in ``universe.tickers`` config order.
    Features fan the score out to every ticker of that CIK.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from newsfeed.config import AppConfig
from newsfeed.storage import Store
from newsfeed.timeutil import ensure_utc

CUSTOM_ID_LEN = 48  # Message Batches custom_id: <= 64 chars of [A-Za-z0-9_-]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Target:
    item_id: str
    ticker: str
    cik: int
    model: str
    prompt_version: str
    prompt_hash: str
    text_hash: str  # items.text_hash: a re-parsed item (new text) is a new cache entry
    masked: bool = False
    mask_id: str = ""  # MaskLexicon.mask_id when masked: new mask terms = new cache entry
    replicate: int = 0

    @property
    def cache_key(self) -> str:
        """The spec's (item_id, ticker, model, prompt_hash, masked) plus the replicate index and
        the exact inputs that can change under a fixed item_id: its text and the mask terms."""
        blob = json.dumps(
            [
                self.item_id,
                self.ticker,
                self.model,
                self.prompt_hash,
                self.masked,
                self.replicate,
                self.text_hash,
                self.mask_id if self.masked else "",
            ]
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def custom_id(self) -> str:
        return self.cache_key[:CUSTOM_ID_LEN]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> Target:
        return cls(**json.loads(s))


def canonical_tickers(cfg: AppConfig, cik_to_tickers: dict[int, list[str]]) -> dict[int, str]:
    order = {t: i for i, t in enumerate(cfg.universe.tickers)}
    # config order first (static universe), then alphabetical -- deterministic either way
    return {
        cik: min(ts, key=lambda t: (order.get(t, 10**6), t)) for cik, ts in cik_to_tickers.items()
    }


def candidate_pairs(
    store: Store,
    cfg: AppConfig,
    canonical: dict[int, str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    sources: list[str] | None = None,
    since_tradable: datetime | None = None,
) -> list[dict[str, Any]]:
    """Distinct (item_id, cik) pairs eligible for scoring, oldest first."""
    pol = cfg.scoring.link_policy
    where = [
        "(t.method IN (SELECT unnest($methods)) "
        "OR (t.method = 'name_body' AND t.n_mentions >= $min_body))"
    ]
    params: dict[str, Any] = {"methods": pol.methods, "min_body": pol.name_body_min_mentions}
    if cfg.scoring.founders_only:
        where.append("i.item_id IN (SELECT item_id FROM item_story WHERE story_id = item_id)")
    if start is not None:
        where.append("i.tradable_at >= $start")
        params["start"] = ensure_utc(start)
    if end is not None:
        where.append("i.tradable_at < $end")
        params["end"] = ensure_utc(end)
    if since_tradable is not None:
        where.append("i.tradable_at >= $since")
        params["since"] = ensure_utc(since_tradable)
    if sources:
        where.append("i.source IN (SELECT unnest($sources))")
        params["sources"] = sources
    df = store.query(
        f"""
        SELECT DISTINCT i.item_id, t.cik, i.tradable_at, i.text_hash
        FROM item_tickers t JOIN items i USING (item_id)
        WHERE {" AND ".join(where)}
        ORDER BY i.tradable_at, i.item_id, t.cik
        """,
        params,
    )
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in df.to_dicts():
        cik = int(r["cik"])
        if cik not in canonical:
            continue
        key = (r["item_id"], canonical[cik])
        if key in seen:  # two CIKs sharing a canonical ticker (e.g. a re-registered company)
            log.warning("duplicate scoring pair %s skipped (cik %s)", key, cik)
            continue
        seen.add(key)
        out.append(
            {"item_id": r["item_id"], "cik": cik, "ticker": key[1], "text_hash": r["text_hash"]}
        )
    return out


def done_or_inflight_keys(store: Store) -> set[str]:
    """Cache keys with a stored result (any final status) or sitting in an uncollected batch."""
    done = set(store.query("SELECT DISTINCT cache_key FROM scores_all")["cache_key"].to_list())
    inflight = store.ops_query(
        "SELECT r.cache_key FROM llm_batch_requests r JOIN llm_batches b USING (batch_id) "
        "WHERE NOT b.collected"
    )["cache_key"].to_list()
    return done | set(inflight)


def ok_keys(store: Store) -> set[tuple[str, str, str, str, str]]:
    """(item_id, ticker, model, prompt_hash, text_hash) with an ok canonical unmasked score --
    used to restrict validation / masked / replicate runs to documents (in their current text)
    that the primary setup has scored."""
    df = store.query(
        "SELECT DISTINCT item_id, ticker, model, prompt_hash, text_hash FROM scores "
        "WHERE status = 'ok' AND NOT masked AND replicate = 0"
    )
    return {
        (r["item_id"], r["ticker"], r["model"], r["prompt_hash"], r["text_hash"])
        for r in df.to_dicts()
    }


def stable_sample(pairs: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample: rank by sha256(seed, item, cik). Membership is stable as the
    candidate set grows (a pair sampled once stays sampled), which keeps studies comparable."""

    def rank(p: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{p['item_id']}:{p['cik']}".encode()).hexdigest()

    return sorted(pairs, key=rank)[:n]
