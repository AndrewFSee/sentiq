"""Scoring of Federal Reserve documents (source ``fed``): one score per document (no company),
own prompt family (``scoring.fed_prompt_version``, default ``fed_v1``), own schema (``fed/1``)
and own table (``fed_scores``). Same client, settings, cost accounting and usage ledger
(``llm_usage``) as company scoring.

Cache key = sha256(item_id, model, prompt_hash, text_hash); final statuses (ok, invalid,
refusal, max_tokens, empty) are stored and never re-requested, API errors are not stored (the
next run retries them). ``prompt_hash`` covers the prompt text, the Fed schema and the
generation settings; each new hash is recorded in ``experiments``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from newsfeed.config import AppConfig
from newsfeed.scoring.client import LLMClient, Outcome, build_params, call_cost, parse_message
from newsfeed.scoring.prompt import GenerationSettings, load_prompt
from newsfeed.scoring.runner import Estimate, JobStats
from newsfeed.storage import Store
from newsfeed.timeutil import ensure_utc, utcnow

log = logging.getLogger(__name__)

FED_SCHEMA_VERSION = "fed/1"
MAX_RATIONALE_WORDS = 25
POLICY_RELEVANCE = ["high", "medium", "low"]
RATE_PATH = ["hike", "hold", "cut", "unclear"]
DOC_TYPES = {
    "fomc_statement": "FOMC monetary policy statement",
    "fomc_minutes": "minutes of a Federal Open Market Committee meeting",
    "speech": "speech by a member of the Federal Reserve Board",
    "testimony": "congressional testimony by a member of the Federal Reserve Board",
    "presconf_opening": "opening statement of the Federal Reserve Chair at the FOMC press "
    "conference (released the same day; no questions and answers)",
    "presconf_transcript": "transcript of the Federal Reserve Chair's FOMC press conference "
    "(opening statement and answers to reporters' questions)",
}
COMMITTEE = "Federal Open Market Committee"


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


FED_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "policy_relevance": {"type": "string", "enum": POLICY_RELEVANCE},
        "stance": _nullable({"type": "number", "description": "In [-1, 1]; -1 dovish."}),
        "rate_path": {"type": "string", "enum": RATE_PATH},
        "inflation": _nullable({"type": "number", "description": "In [-1, 1]."}),
        "growth": _nullable({"type": "number", "description": "In [-1, 1]."}),
        "labor": _nullable({"type": "number", "description": "In [-1, 1]."}),
        "confidence": {"type": "number", "description": "In [0, 1]."},
        "rationale": {"type": "string", "description": "At most 25 words."},
    },
    "required": [
        "policy_relevance",
        "stance",
        "rate_path",
        "inflation",
        "growth",
        "labor",
        "confidence",
        "rationale",
    ],
    "additionalProperties": False,
}


class FedScoreOutput(BaseModel):
    """Client-side validation: ranges are checked, never clamped (out of range = invalid)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_relevance: Literal["high", "medium", "low"]
    stance: float | None
    rate_path: Literal["hike", "hold", "cut", "unclear"]
    inflation: float | None
    growth: float | None
    labor: float | None
    confidence: float
    rationale: str

    @field_validator("stance", "inflation", "growth", "labor")
    @classmethod
    def _signed_unit(cls, v: float | None) -> float | None:
        if v is not None and not -1.0 <= v <= 1.0:
            raise ValueError(f"{v} outside [-1, 1]")
        return v

    @field_validator("confidence")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{v} outside [0, 1]")
        return v


def parse_fed_output(text: str) -> tuple[FedScoreOutput | None, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json: {exc}"
    try:
        return FedScoreOutput.model_validate(data), None
    except ValidationError as exc:
        return None, f"validation: {exc.errors(include_url=False)}"


@dataclass(frozen=True)
class FedTarget:
    item_id: str
    text_hash: str
    model: str
    prompt_version: str
    prompt_hash: str

    @property
    def cache_key(self) -> str:
        blob = json.dumps([self.item_id, self.model, self.prompt_hash, self.text_hash])
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class FedPrepared:
    target: FedTarget
    kind: str
    params: dict[str, Any]
    user_text: str


class FedScorer:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        client: LLMClient | None,
        *,
        run_id: str,
        model: str | None = None,
    ) -> None:
        self.cfg, self.store, self.client, self.run_id = cfg, store, client, run_id
        self.model = model or cfg.scoring.model
        s = cfg.scoring
        self.prompt = load_prompt(s.fed_prompt_version)
        self.settings = GenerationSettings(s.effort, s.thinking, s.max_tokens)
        self.prompt_hash = self.prompt.hash(
            self.settings, schema=FED_OUTPUT_JSON_SCHEMA, schema_version=FED_SCHEMA_VERSION
        )
        self.stats = JobStats()

    # ------------------------------------------------------------------ planning
    def plan(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[FedTarget]:
        """Fed items (tradable_at in [start, end)) without a stored result for this model,
        prompt and text, oldest first."""
        where, params = ["TRUE"], {}
        if start is not None:
            where.append("tradable_at >= $start")
            params["start"] = ensure_utc(start)
        if end is not None:
            where.append("tradable_at < $end")
            params["end"] = ensure_utc(end)
        rel = self.store.items_relation(["fed"])
        df = self.store.query(
            f"SELECT item_id, text_hash FROM {rel} WHERE {' AND '.join(where)} "
            "ORDER BY tradable_at, item_id",
            params,
        )
        done = set(self.store.query("SELECT DISTINCT cache_key FROM fed_scores_all")["cache_key"])
        out = []
        for r in df.iter_rows(named=True):
            t = FedTarget(
                r["item_id"], r["text_hash"], self.model, self.prompt.version, self.prompt_hash
            )
            if t.cache_key not in done:
                out.append(t)
        return out[:limit] if limit is not None else out

    # ------------------------------------------------------------------ requests
    def prepare(self, targets: Sequence[FedTarget]) -> list[FedPrepared]:
        items = {
            r["item_id"]: r
            for r in self.store.items_by_id(
                [t.item_id for t in targets],
                sources=["fed"],
                columns="item_id, title, body_text, body_truncated, meta, publisher, text_hash",
            ).to_dicts()
        }
        out = []
        for t in targets:
            it = items[t.item_id]
            if it["text_hash"] != t.text_hash:
                raise ValueError(f"item {t.item_id} text changed since planning; re-plan")
            meta = json.loads(it["meta"] or "{}")
            kind = meta.get("kind", "speech")
            speaker = COMMITTEE if kind.startswith("fomc_") else (meta.get("speaker") or "unknown")
            user = self.prompt.render_user(
                target=speaker,
                doc_type=DOC_TYPES.get(kind, kind),
                publisher=it["publisher"],
                title=it["title"],
                body=it["body_text"] or "",
                truncated=bool(it["body_truncated"]),
            )
            params = build_params(
                model=t.model,
                system=self.prompt.system,
                user=user,
                settings=self.settings,
                schema=FED_OUTPUT_JSON_SCHEMA,
            )
            out.append(FedPrepared(t, kind, params, user))
        return out

    # ------------------------------------------------------------------ cost
    def estimate(self, targets: Sequence[FedTarget], *, seed: int = 0) -> Estimate:
        """Same method as company scoring: count_tokens on a sample, scale by length."""
        prepared = self.prepare(targets)
        n = len(prepared)
        sample = random.Random(seed).sample(prepared, min(self.cfg.scoring.dry_run_sample, n))
        if self.client is not None and sample:
            method = "count_tokens"
            base = build_params(
                model=self.model,
                system=self.prompt.system,
                user=".",
                settings=self.settings,
                schema=FED_OUTPUT_JSON_SCHEMA,
            )
            system_tokens = self.client.count_tokens(base)
            counted = [self.client.count_tokens(p.params) - system_tokens for p in sample]
            ratio = sum(counted) / max(1, sum(len(p.user_text) for p in sample))
        else:
            method = "chars"
            system_tokens = math.ceil(len(self.prompt.system) / 4) + 300
            ratio = 0.25
        user_tokens = [len(p.user_text) * ratio for p in prepared]
        hist = self.store.query(
            "SELECT avg(output_tokens) AS m, count(*) AS c FROM fed_scores_all "
            "WHERE model = $m AND prompt_hash = $h AND status = 'ok'",
            {"m": self.model, "h": self.prompt_hash},
        )
        if hist["c"][0] and hist["c"][0] >= 20:
            out_per, out_src = float(hist["m"][0]), "ledger"
        else:
            out_per, out_src = float(self.cfg.scoring.est_output_tokens), "config"

        def total(*, cached: bool, batch: bool) -> float:
            usd = 0.0
            for i, u in enumerate(user_tokens):
                sys_in = 0 if cached else system_tokens
                usage = {
                    "input_tokens": int(u) + sys_in,
                    "cache_creation_input_tokens": system_tokens if cached and i == 0 else 0,
                    "cache_read_input_tokens": system_tokens if cached and i > 0 else 0,
                    "output_tokens": int(out_per),
                }
                usd += call_cost(self.cfg.pricing, self.model, usage, batch=batch)
            return usd

        return Estimate(
            n_requests=n,
            method=method,
            system_tokens=system_tokens,
            user_tokens_total=int(sum(user_tokens)),
            output_tokens_per_request=out_per,
            output_source=out_src,
            cost_live_no_cache=total(cached=False, batch=False),
            cost_live_cached=total(cached=True, batch=False),
            cost_batch_no_cache=total(cached=False, batch=True),
            cost_batch_cached=total(cached=True, batch=True),
        )

    # ------------------------------------------------------------------ recording
    def _row(self, p: FedPrepared, o: Outcome, cost: float) -> dict[str, Any]:
        t, out = p.target, o.output
        return {
            "cache_key": t.cache_key,
            "item_id": t.item_id,
            "kind": p.kind,
            "model": t.model,
            "prompt_version": t.prompt_version,
            "prompt_hash": t.prompt_hash,
            "schema_version": FED_SCHEMA_VERSION,
            "text_hash": t.text_hash,
            "run_id": self.run_id,
            "mode": "live",
            "request_id": o.request_id,
            "status": o.status,
            "error": o.error,
            **{
                f: getattr(out, f) if out else None
                for f in (
                    "policy_relevance",
                    "stance",
                    "rate_path",
                    "inflation",
                    "growth",
                    "labor",
                    "confidence",
                    "rationale",
                )
            },
            "stop_reason": o.stop_reason,
            **o.usage,
            "cost_usd": cost,
            "user_text_hash": hashlib.sha256(p.user_text.encode()).hexdigest()[:16],
            "raw_response": o.raw,
            "scored_at": utcnow(),
        }

    def _usage(
        self, t: FedTarget, status: str, usage: dict[str, int], cost: float, **kw: Any
    ) -> list[Any]:
        return [
            utcnow(),
            self.run_id,
            "live",
            t.model,
            t.cache_key,
            None,
            kw.get("request_id"),
            status,
            usage.get("input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
            usage.get("output_tokens", 0),
            cost,
            kw.get("error"),
        ]

    def _log_usage(self, rows: list[list[Any]]) -> None:
        if not rows:
            return
        cols = (
            "called_at, run_id, mode, model, cache_key, batch_id, request_id, status, "
            "input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens, "
            "cost_usd, error"
        )
        with self.store.ops() as con:
            con.executemany(f"INSERT INTO llm_usage ({cols}) VALUES ({', '.join('?' * 14)})", rows)

    def register_experiment(self) -> None:
        with self.store.ops() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiments VALUES ('prompt', ?, ?, ?, ?)",
                [
                    self.prompt_hash,
                    utcnow(),
                    self.run_id,
                    json.dumps(
                        {
                            "prompt_version": self.prompt.version,
                            "schema_version": FED_SCHEMA_VERSION,
                            **self.settings.__dict__,
                        }
                    ),
                ],
            )

    # ------------------------------------------------------------------ live
    def run_live(self, targets: Sequence[FedTarget], *, flush_every: int = 25) -> JobStats:
        if self.client is None:
            raise RuntimeError("no LLM client configured")
        self.register_experiment()
        prepared = self.prepare(targets)
        rows: list[dict[str, Any]] = []
        usage: list[list[Any]] = []

        def flush() -> None:
            self.store.write_fed_scores(rows, self.run_id)
            self._log_usage(usage)
            rows.clear()
            usage.clear()

        def call(p: FedPrepared) -> tuple[FedPrepared, Any, Exception | None]:
            try:
                return p, self.client.create(p.params), None  # type: ignore[union-attr]
            except Exception as exc:  # the SDK already retried transient errors
                return p, None, exc

        with ThreadPoolExecutor(max_workers=self.cfg.scoring.live_concurrency) as pool:
            for fut in as_completed([pool.submit(call, p) for p in prepared]):
                p, message, exc = fut.result()
                if exc is not None:  # not stored: retried by the next run
                    log.error("fed scoring failed for %s: %r", p.target.item_id, exc)
                    self.stats.add("error", {}, 0.0)
                    usage.append(self._usage(p.target, "error", {}, 0.0, error=repr(exc)[:500]))
                    continue
                o = parse_message(message, parse_fed_output)
                cost = call_cost(self.cfg.pricing, p.target.model, o.usage, batch=False)
                self.stats.add(o.status, o.usage, cost)
                rows.append(self._row(p, o, cost))
                usage.append(
                    self._usage(p.target, o.status, o.usage, cost, request_id=o.request_id)
                )
                if len(rows) >= flush_every:
                    flush()
        flush()
        return self.stats
