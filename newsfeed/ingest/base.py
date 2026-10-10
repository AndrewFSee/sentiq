"""Source-adapter interface and the canonical item record every adapter produces."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from newsfeed.timeutil import IngestMode, ensure_utc


def make_item_id(source: str, native_id: str) -> str:
    """Deterministic id: the same source document always maps to the same item_id."""
    return hashlib.sha256(f"{source}:{native_id}".encode()).hexdigest()[:32]


class RawItem(BaseModel):
    """One source document, as observed. Adapters build these; storage derives the rest
    (item_id, url_norm, content_hash, tradable_at)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    native_id: str
    publisher: str | None = None  # outlet (for n_outlets); EDGAR = "SEC EDGAR"
    url: str | None
    title: str | None
    body_text: str
    body_tokens_est: int
    body_truncated: bool
    tickers: list[str]
    published_at: datetime
    published_at_raw: str  # exactly as the source reported it, for audit
    first_seen_at: datetime
    ingest_mode: IngestMode
    meta: dict[str, Any]
    raw_json: str  # source metadata payload(s) verbatim; large documents live in raw_dir
    parser_version: str

    @field_validator("published_at", "first_seen_at")
    @classmethod
    def _utc(cls, v: datetime, info: Any) -> datetime:
        return ensure_utc(v, name=info.field_name)

    @property
    def item_id(self) -> str:
        return make_item_id(self.source, self.native_id)


SkipFn = Callable[[str], bool]
"""Given an item_id, return True if it is already stored (adapter should not re-fetch)."""


ErrorFn = Callable[[str, Exception], None]
"""Called with (native_id, exception) when one document fails; ingestion continues."""


class SourceAdapter(Protocol):
    name: str
    last_activity_at: datetime | None
    """Newest timestamp seen in the source's feed by the last poll -- for *any* company, not just
    the universe. Drives silent-feed (outage) detection; None if the poll saw nothing."""

    def backfill(
        self, start: datetime, end: datetime, skip: SkipFn, on_error: ErrorFn | None = None
    ) -> Iterator[RawItem]:
        """Historical items with published_at in [start, end). ingest_mode='backfill'."""
        ...

    def poll(
        self, skip: SkipFn, on_error: ErrorFn | None = None, *, catch_up: bool = False
    ) -> Iterator[RawItem]:
        """Items newly visible now. ingest_mode='live'; first_seen_at is stamped after the
        response that revealed the item arrived. ``catch_up=True`` does a slower, wider sweep
        to recover anything the cheap poll could have missed (still labelled live: we really
        are only seeing it now)."""
        ...
