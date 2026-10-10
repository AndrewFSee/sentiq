"""Anthropic API access: request construction, response parsing, cost accounting.

All model calls go through :class:`LLMClient`, a narrow protocol implemented by
:class:`AnthropicClient` (the official SDK) and by fakes in tests (no network in tests).

API facts this module relies on (Claude API docs, checked 2026-10-09):
  * Haiku 5.5 / Sonnet 5.5 reject non-default temperature/top_p/top_k -> we send none; scores
    are sampled draws, made reproducible by caching every result with its raw response.
  * Structured outputs (``output_config.format``) guarantee schema-valid JSON in the first text
    block; forced ``tool_choice`` is rejected by Sonnet 5.5, so tool use is not used.
  * Thinking: adaptive by default; ``disabled`` is accepted by Haiku 5.5 at effort <= high but
    rejected by Sonnet 5.5. Thinking tokens are billed as output tokens.
  * Responses can start with ``thinking`` blocks -- read blocks by type, never by position.
  * A refusal is HTTP 200 with ``stop_reason == "refusal"``; check before reading content.
  * Prompt caching: the static system block carries ``cache_control``; minimum cacheable
    prefix on these models is 512 tokens.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from newsfeed.config import PricingConfig
from newsfeed.scoring.prompt import GenerationSettings
from newsfeed.scoring.schema import OUTPUT_JSON_SCHEMA, ScoreOutput, parse_output

log = logging.getLogger(__name__)

NO_THINKING_OFF = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"}


def build_params(
    *, model: str, system: str, user: str, settings: GenerationSettings
) -> dict[str, Any]:
    """Messages API request body (identical for live, batch and count_tokens)."""
    if settings.thinking == "disabled" and model in NO_THINKING_OFF:
        raise ValueError(f"{model} rejects thinking={{'type': 'disabled'}}; use adaptive")
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": settings.thinking},
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": OUTPUT_JSON_SCHEMA},
        },
    }
    return params


def count_params(params: dict[str, Any]) -> dict[str, Any]:
    """count_tokens takes the same body minus max_tokens."""
    return {k: v for k, v in params.items() if k != "max_tokens"}


# ------------------------------------------------------------------------ parsing
@dataclass
class Outcome:
    status: str  # ok | invalid | refusal | max_tokens | empty
    output: ScoreOutput | None
    error: str | None
    stop_reason: str | None
    usage: dict[str, int]
    request_id: str | None
    raw: str


def usage_dict(usage: Any) -> dict[str, int]:
    def g(name: str) -> int:
        v = getattr(usage, name, None) if usage is not None else None
        return int(v or 0)

    return {
        "input_tokens": g("input_tokens"),
        "cache_creation_input_tokens": g("cache_creation_input_tokens"),
        "cache_read_input_tokens": g("cache_read_input_tokens"),
        "output_tokens": g("output_tokens"),
    }


def parse_message(message: Any) -> Outcome:
    """Classify an SDK ``Message`` (or equivalent) into an Outcome."""
    raw = message.to_json() if hasattr(message, "to_json") else json.dumps(message, default=str)
    usage = usage_dict(getattr(message, "usage", None))
    stop = getattr(message, "stop_reason", None)
    rid = getattr(message, "_request_id", None) or getattr(message, "id", None)
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        cat = getattr(details, "category", None) if details is not None else None
        return Outcome("refusal", None, f"refusal category={cat}", stop, usage, rid, raw)
    text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
    if stop == "max_tokens":
        return Outcome("max_tokens", None, "hit max_tokens", stop, usage, rid, raw)
    if not text:
        return Outcome("empty", None, "no text block", stop, usage, rid, raw)
    out, err = parse_output(text)
    if out is None:
        return Outcome("invalid", None, err, stop, usage, rid, raw)
    return Outcome("ok", out, None, stop, usage, rid, raw)


# ------------------------------------------------------------------------ cost
def call_cost(pricing: PricingConfig, model: str, usage: dict[str, int], *, batch: bool) -> float:
    """USD for one call. Haiku 5.5 switches rate card when the whole prompt (uncached + cache
    writes + cache reads) exceeds the threshold. Batch discount stacks with cache multipliers."""
    mp = pricing.models.get(model)
    if mp is None:
        log.warning("no pricing configured for %s; cost recorded as NaN", model)
        return math.nan
    prompt = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    card = mp.standard
    if mp.long is not None and mp.long_threshold_tokens and prompt > mp.long_threshold_tokens:
        card = mp.long
    usd = (
        usage["input_tokens"] * card.input
        + usage["cache_creation_input_tokens"] * card.cache_write_5m
        + usage["cache_read_input_tokens"] * card.cache_read
        + usage["output_tokens"] * card.output
    ) / 1e6
    return usd * pricing.batch_discount if batch else usd


# ------------------------------------------------------------------------ clients
@dataclass
class BatchResult:
    custom_id: str
    type: str  # succeeded | errored | canceled | expired
    message: Any | None
    error: str | None


class LLMClient(Protocol):
    def create(self, params: dict[str, Any]) -> Any: ...

    def count_tokens(self, params: dict[str, Any]) -> int: ...

    def batch_create(self, requests: list[dict[str, Any]]) -> str: ...

    def batch_status(self, batch_id: str) -> str: ...

    def batch_results(self, batch_id: str) -> Iterator[BatchResult]: ...


class AnthropicClient:
    """Official SDK. The SDK retries 408/409/429/5xx and connection errors itself."""

    def __init__(self, api_key: str | None = None, max_retries: int = 4) -> None:
        import anthropic  # deferred: tests and non-scoring commands never need it

        self._sdk = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def create(self, params: dict[str, Any]) -> Any:
        return self._sdk.messages.create(**params)

    def count_tokens(self, params: dict[str, Any]) -> int:
        return self._sdk.messages.count_tokens(**count_params(params)).input_tokens

    def batch_create(self, requests: list[dict[str, Any]]) -> str:
        return self._sdk.messages.batches.create(requests=requests).id

    def batch_status(self, batch_id: str) -> str:
        return self._sdk.messages.batches.retrieve(batch_id).processing_status

    def batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        for r in self._sdk.messages.batches.results(batch_id):
            res = r.result
            if res.type == "succeeded":
                yield BatchResult(r.custom_id, "succeeded", res.message, None)
            elif res.type == "errored":
                yield BatchResult(r.custom_id, "errored", None, str(res.error))
            else:
                yield BatchResult(r.custom_id, res.type, None, None)
