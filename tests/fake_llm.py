"""In-process stand-in for the Anthropic API: returns real SDK ``Message`` objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from typing import Any

from anthropic.types import Message

from newsfeed.scoring.client import BatchResult


def default_output(user: str) -> dict[str, Any]:
    """Deterministic pseudo-labels derived from the user text (stable across runs)."""
    h = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
    return {
        "relevance": "primary",
        "sentiment": round((h % 200) / 100 - 1, 2),
        "confidence": 0.7,
        "event_type": "earnings" if "Results" in user or "results" in user else "other",
        "novelty": "new",
        "materiality": "medium",
        "surprise_vs_expectations": None,
        "rationale": "Synthetic label for tests.",
    }


def make_message(
    payload: dict[str, Any] | str | None,
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
    thinking_first: bool = True,
    model: str = "claude-haiku-5-5",
) -> Message:
    content: list[dict[str, Any]] = []
    if thinking_first:
        content.append({"type": "thinking", "thinking": "", "signature": "sig"})
    if payload is not None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        content.append({"type": "text", "text": text})
    u = {
        "input_tokens": 1000,
        "output_tokens": 120,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2000,
        **(usage or {}),
    }
    data: dict[str, Any] = {
        "id": "msg_" + hashlib.sha256(json.dumps(content).encode()).hexdigest()[:20],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": u,
    }
    if stop_reason == "refusal":
        data["stop_details"] = {"type": "refusal", "category": "general_harms", "explanation": "x"}
    return Message.model_validate(data)


class FakeLLM:
    def __init__(
        self,
        respond: Callable[[dict[str, Any]], Message] | None = None,
        *,
        fail_on: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.respond = respond or (lambda p: make_message(default_output(user_text(p))))
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []
        self.counted: list[dict[str, Any]] = []
        self.batches: dict[str, list[dict[str, Any]]] = {}
        self.batch_state: dict[str, str] = {}
        self.batch_overrides: dict[str, BatchResult] = {}  # custom_id -> forced result

    # --- messages
    def create(self, params: dict[str, Any]) -> Message:
        self.calls.append(params)
        if self.fail_on and self.fail_on(params):
            raise ConnectionError("simulated outage")
        return self.respond(params)

    def count_tokens(self, params: dict[str, Any]) -> int:
        self.counted.append(params)
        sys_text = params["system"][0]["text"]
        return len(sys_text) // 4 + 400 + len(user_text(params)) // 4

    # --- batches
    def batch_create(self, requests: list[dict[str, Any]]) -> str:
        bid = f"msgbatch_{len(self.batches) + 1:03d}"
        self.batches[bid] = requests
        self.batch_state[bid] = "in_progress"
        return bid

    def batch_status(self, batch_id: str) -> str:
        return self.batch_state[batch_id]

    def batch_results(self, batch_id: str) -> Iterator[BatchResult]:
        for req in reversed(self.batches[batch_id]):  # any order: match by custom_id only
            cid = req["custom_id"]
            if cid in self.batch_overrides:
                yield self.batch_overrides[cid]
            else:
                yield BatchResult(cid, "succeeded", self.respond(req["params"]), None)


def user_text(params: dict[str, Any]) -> str:
    return params["messages"][0]["content"]
