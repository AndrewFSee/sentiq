"""Scoring jobs: build requests, call the model (live), record results + usage, estimate cost."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from newsfeed.config import AppConfig
from newsfeed.scoring.client import LLMClient, Outcome, build_params, call_cost, parse_message
from newsfeed.scoring.mask import MaskLexicon
from newsfeed.scoring.prompt import GenerationSettings, PromptSpec, doc_type_for, load_prompt
from newsfeed.scoring.schema import SCHEMA_VERSION
from newsfeed.scoring.targets import (
    Target,
    candidate_pairs,
    canonical_tickers,
    done_or_inflight_keys,
    ok_keys,
    stable_sample,
)
from newsfeed.storage import Store
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    target: Target
    params: dict[str, Any]
    user_text: str
    user_text_hash: str
    mask_residuals: int | None


@dataclass
class JobStats:
    calls: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0

    def add(self, status: str, usage: dict[str, int], cost: float) -> None:
        self.calls += 1
        self.by_status[status] = self.by_status.get(status, 0) + 1
        if not math.isnan(cost):
            self.cost_usd += cost
        for k in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        ):
            setattr(self, k, getattr(self, k) + usage.get(k, 0))

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "cost_usd": round(self.cost_usd, 6)}


class ScoringJob:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        client: LLMClient | None,
        *,
        run_id: str,
        model: str,
        company_names: dict[int, str],
        lexicon: MaskLexicon,
        prompt: PromptSpec | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.client = client
        self.run_id = run_id
        self.model = model
        self.company_names = company_names
        self.lexicon = lexicon
        self.prompt = prompt or load_prompt(cfg.scoring.prompt_version)
        s = cfg.scoring
        self.settings = GenerationSettings(s.effort, s.thinking, s.max_tokens)
        self.prompt_hash = self.prompt.hash(self.settings)
        self.stats = JobStats()
        self._items: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- targets
    def target(
        self,
        item_id: str,
        ticker: str,
        cik: int,
        text_hash: str,
        *,
        masked: bool = False,
        replicate: int = 0,
    ) -> Target:
        return Target(
            item_id=item_id,
            ticker=ticker,
            cik=cik,
            model=self.model,
            prompt_version=self.prompt.version,
            prompt_hash=self.prompt_hash,
            text_hash=text_hash,
            masked=masked,
            mask_id=self.lexicon.mask_id if masked else "",
            replicate=replicate,
        )

    def register_experiment(self) -> None:
        """Count every prompt/settings variant ever used (multiple-testing accounting)."""
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
                            "schema_version": SCHEMA_VERSION,
                            **self.settings.__dict__,
                        }
                    ),
                ],
            )

    # ---------------------------------------------------------------- requests
    def _load_items(self, item_ids: Iterable[str]) -> None:
        need = sorted(set(item_ids) - set(self._items))
        if not need:
            return
        df = self.store.query(
            "SELECT item_id, source, publisher, title, body_text, body_truncated, meta, text_hash "
            "FROM items WHERE item_id IN (SELECT unnest($ids))",
            {"ids": need},
        )
        for r in df.to_dicts():
            r["meta"] = json.loads(r["meta"] or "{}")
            self._items[r["item_id"]] = r

    def prepare(self, targets: Sequence[Target]) -> list[Prepared]:
        self._load_items(t.item_id for t in targets)
        out = []
        for t in targets:
            it = self._items[t.item_id]
            if it["text_hash"] != t.text_hash:  # item re-parsed since planning
                raise ValueError(f"item {t.item_id} text changed since planning; re-plan")
            title, body = it["title"] or "", it["body_text"] or ""
            residuals = None
            if t.masked:
                m = self.lexicon.mask(title, body, t.cik)
                title, body, label, residuals = m.title, m.body, m.target_label, m.residuals
            else:
                name = self.company_names.get(t.cik) or t.ticker
                label = f"{name} (ticker: {t.ticker})"
            user = self.prompt.render_user(
                target=label,
                doc_type=doc_type_for(it["source"], it["meta"]),
                publisher=it["publisher"],
                title=title,
                body=body,
                truncated=bool(it["body_truncated"]),
            )
            params = build_params(
                model=t.model, system=self.prompt.system, user=user, settings=self.settings
            )
            out.append(
                Prepared(t, params, user, hashlib.sha256(user.encode()).hexdigest()[:16], residuals)
            )
        return out

    # ---------------------------------------------------------------- recording
    def score_row(
        self,
        t: Target,
        o: Outcome,
        *,
        mode: str,
        batch_id: str | None,
        user_text_hash: str,
        mask_residuals: int | None,
        cost: float,
    ) -> dict[str, Any]:
        out = o.output
        return {
            "cache_key": t.cache_key,
            "item_id": t.item_id,
            "ticker": t.ticker,
            "cik": t.cik,
            "model": t.model,
            "prompt_version": t.prompt_version,
            "prompt_hash": t.prompt_hash,
            "schema_version": SCHEMA_VERSION,
            "text_hash": t.text_hash,
            "mask_id": t.mask_id,
            "masked": t.masked,
            "replicate": t.replicate,
            "run_id": self.run_id,
            "mode": mode,
            "batch_id": batch_id,
            "request_id": o.request_id,
            "status": o.status,
            "error": o.error,
            "relevance": out.relevance if out else None,
            "sentiment": out.sentiment if out else None,
            "confidence": out.confidence if out else None,
            "event_type": out.event_type if out else None,
            "novelty": out.novelty if out else None,
            "materiality": out.materiality if out else None,
            "surprise_vs_expectations": out.surprise_vs_expectations if out else None,
            "rationale": out.rationale if out else None,
            "rationale_words": out.rationale_words if out else None,
            "mask_residuals": mask_residuals,
            "stop_reason": o.stop_reason,
            **o.usage,
            "cost_usd": cost,
            "user_text_hash": user_text_hash,
            "raw_response": o.raw,
            "scored_at": utcnow(),
        }

    def log_usage(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = [
            "called_at",
            "run_id",
            "mode",
            "model",
            "cache_key",
            "batch_id",
            "request_id",
            "status",
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
            "cost_usd",
            "error",
        ]
        with self.store.ops() as con:
            con.executemany(
                f"INSERT INTO llm_usage ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [[r.get(c) for c in cols] for r in rows],
            )

    def usage_row(
        self,
        t: Target,
        *,
        mode: str,
        status: str,
        usage: dict[str, int],
        cost: float,
        batch_id: str | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "called_at": utcnow(),
            "run_id": self.run_id,
            "mode": mode,
            "model": t.model,
            "cache_key": t.cache_key,
            "batch_id": batch_id,
            "request_id": request_id,
            "status": status,
            **{
                k: usage.get(k, 0)
                for k in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "output_tokens",
                )
            },
            "cost_usd": cost,
            "error": error,
        }

    def record(
        self,
        prepared: Prepared,
        outcome: Outcome,
        *,
        mode: str,
        batch_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        t = prepared.target
        cost = call_cost(self.cfg.pricing, t.model, outcome.usage, batch=mode == "batch")
        self.stats.add(outcome.status, outcome.usage, cost)
        row = self.score_row(
            t,
            outcome,
            mode=mode,
            batch_id=batch_id,
            user_text_hash=prepared.user_text_hash,
            mask_residuals=prepared.mask_residuals,
            cost=cost,
        )
        usage = self.usage_row(
            t,
            mode=mode,
            status=outcome.status,
            usage=outcome.usage,
            cost=cost,
            batch_id=batch_id,
            request_id=outcome.request_id,
            error=outcome.error,
        )
        return row, usage

    # ---------------------------------------------------------------- live
    def run_live(self, targets: Sequence[Target], *, flush_every: int = 50) -> JobStats:
        if self.client is None:
            raise RuntimeError("no LLM client configured")
        self.register_experiment()
        prepared = self.prepare(targets)
        rows: list[dict[str, Any]] = []
        usage: list[dict[str, Any]] = []

        def flush() -> None:
            self.store.write_scores(rows, self.run_id)
            self.log_usage(usage)
            rows.clear()
            usage.clear()

        def call(p: Prepared) -> tuple[Prepared, Any, Exception | None]:
            try:
                return p, self.client.create(p.params), None  # type: ignore[union-attr]
            except Exception as exc:  # SDK already retried transient errors
                return p, None, exc

        with ThreadPoolExecutor(max_workers=self.cfg.scoring.live_concurrency) as pool:
            futures = [pool.submit(call, p) for p in prepared]
            for fut in as_completed(futures):
                p, message, exc = fut.result()
                if exc is not None:
                    log.error("scoring call failed for %s: %r", p.target.cache_key[:12], exc)
                    self.stats.add("error", {}, 0.0)
                    usage.append(
                        self.usage_row(
                            p.target,
                            mode="live",
                            status="error",
                            usage={},
                            cost=0.0,
                            error=repr(exc)[:500],
                        )
                    )
                    continue
                row, u = self.record(p, parse_message(message), mode="live")
                rows.append(row)
                usage.append(u)
                if len(rows) >= flush_every:
                    flush()
        flush()
        return self.stats


# -------------------------------------------------------------------- estimates
@dataclass
class Estimate:
    n_requests: int
    method: str  # count_tokens | chars
    system_tokens: int
    user_tokens_total: int
    output_tokens_per_request: float
    output_source: str  # ledger | config
    cost_live_no_cache: float
    cost_live_cached: float
    cost_batch_no_cache: float
    cost_batch_cached: float

    def summary(self) -> str:
        return (
            f"{self.n_requests} requests | input ~{self.system_tokens} system + "
            f"{self.user_tokens_total:,} user tokens ({self.method}) | output ~"
            f"{self.output_tokens_per_request:.0f}/req ({self.output_source})\n"
            f"  live : ${self.cost_live_cached:.4f} (system prompt cached) .. "
            f"${self.cost_live_no_cache:.4f} (no cache hits)\n"
            f"  batch: ${self.cost_batch_cached:.4f} .. ${self.cost_batch_no_cache:.4f}"
        )


def estimate_cost(job: ScoringJob, targets: Sequence[Target], *, seed: int = 0) -> Estimate:
    """Token-count a sample of the actual requests (free endpoint) and extrapolate by length.
    Falls back to chars/4 when no client is available. Output tokens come from the usage
    ledger for this model + prompt if there is history, else ``scoring.est_output_tokens``."""
    n = len(targets)
    prepared = job.prepare(targets)
    user_chars = [len(p.user_text) for p in prepared]
    sample = (
        random.Random(seed).sample(prepared, min(job.cfg.scoring.dry_run_sample, n)) if n else []
    )
    if job.client is not None and sample:
        method = "count_tokens"
        base = build_params(
            model=job.model, system=job.prompt.system, user=".", settings=job.settings
        )
        system_tokens = job.client.count_tokens(base)
        counted = [job.client.count_tokens(p.params) - system_tokens for p in sample]
        ratio = sum(counted) / max(1, sum(len(p.user_text) for p in sample))
    else:
        method = "chars"
        system_tokens = math.ceil(len(job.prompt.system) / 4) + 300  # + schema/format overhead
        ratio = 0.25
    user_tokens = [c * ratio for c in user_chars]

    hist = job.store.query(
        "SELECT avg(output_tokens) AS m, count(*) AS c FROM scores_all "
        "WHERE model = $m AND prompt_hash = $h AND status = 'ok'",
        {"m": job.model, "h": job.prompt_hash},
    )
    if hist["c"][0] and hist["c"][0] >= 20:
        out_per, out_src = float(hist["m"][0]), "ledger"
    else:
        out_per, out_src = float(job.cfg.scoring.est_output_tokens), "config"

    def total(*, cached: bool, batch: bool) -> float:
        usd = 0.0
        for i, u in enumerate(user_tokens):
            if cached:
                usage = {
                    "input_tokens": int(u),
                    "cache_creation_input_tokens": system_tokens if i == 0 else 0,
                    "cache_read_input_tokens": 0 if i == 0 else system_tokens,
                }
            else:
                usage = {
                    "input_tokens": int(u) + system_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            usage["output_tokens"] = int(out_per)
            usd += call_cost(job.cfg.pricing, job.model, usage, batch=batch)
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


# -------------------------------------------------------------------- planning
def plan_targets(
    job: ScoringJob,
    cik_to_tickers: dict[int, list[str]],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    sources: list[str] | None = None,
    since_tradable: datetime | None = None,
    masked: bool = False,
    replicates: int = 0,
    sample: int | None = None,
    seed: int = 1,
    limit: int | None = None,
) -> list[Target]:
    """Targets still to score for this job.

    Studies (masked run, a non-primary model, or replicates) are restricted to documents that
    already have an ok primary score (same prompt + settings), so every study row has a pair.
    ``replicates=K`` creates re-draws 1..K (replicate 0 is the canonical primary draw).
    """
    cfg = job.cfg
    canonical = canonical_tickers(cfg, cik_to_tickers)
    pairs = candidate_pairs(
        job.store,
        cfg,
        canonical,
        start=start,
        end=end,
        sources=sources,
        since_tradable=since_tradable,
    )
    is_study = masked or replicates > 0 or job.model != cfg.scoring.model
    if is_study:
        ok = ok_keys(job.store)
        pairs = [
            p
            for p in pairs
            if (p["item_id"], p["ticker"], cfg.scoring.model, job.prompt_hash, p["text_hash"]) in ok
        ]
    if sample is not None:
        pairs = stable_sample(pairs, sample, seed)
    reps = range(1, replicates + 1) if replicates > 0 else [0]
    targets = [
        job.target(p["item_id"], p["ticker"], p["cik"], p["text_hash"], masked=masked, replicate=r)
        for p in pairs
        for r in reps
    ]
    skip = done_or_inflight_keys(job.store)
    targets = [t for t in targets if t.cache_key not in skip]
    return targets[:limit] if limit is not None else targets
