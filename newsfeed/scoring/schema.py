"""Output schema for per-(article, ticker) scores.

The JSON schema is sent as ``output_config.format`` (structured outputs), which guarantees
schema-valid JSON. Structured outputs cannot express numeric ranges or length limits, so those
are validated client-side by :class:`ScoreOutput`; out-of-range values make a row ``invalid``
(never clamped -- we do not silently alter model output).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

SCHEMA_VERSION = "score/1"
MAX_RATIONALE_WORDS = 25

RELEVANCE = ["primary", "secondary", "passing"]
EVENT_TYPES = [
    "earnings",
    "guidance",
    "m&a",
    "legal_regulatory",
    "analyst_action",
    "product",
    "macro",
    "other",
]
NOVELTY = ["new", "update", "rehash"]
MATERIALITY = ["low", "medium", "high"]


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


# Every property is required; "null where the text doesn't support it" applies to the nullable
# ones. event_type has an explicit "other", so it is never null.
OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance": {"type": "string", "enum": RELEVANCE},
        "sentiment": {"type": "number", "description": "In [-1, 1]."},
        "confidence": {"type": "number", "description": "In [0, 1]."},
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "novelty": _nullable({"type": "string", "enum": NOVELTY}),
        "materiality": _nullable({"type": "string", "enum": MATERIALITY}),
        "surprise_vs_expectations": _nullable(
            {"type": "number", "description": "In [-1, 1]; null if no expectation is stated."}
        ),
        "rationale": {"type": "string", "description": "At most 25 words."},
    },
    "required": [
        "relevance",
        "sentiment",
        "confidence",
        "event_type",
        "novelty",
        "materiality",
        "surprise_vs_expectations",
        "rationale",
    ],
    "additionalProperties": False,
}


class ScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: Literal["primary", "secondary", "passing"]
    sentiment: float
    confidence: float
    event_type: Literal[
        "earnings",
        "guidance",
        "m&a",
        "legal_regulatory",
        "analyst_action",
        "product",
        "macro",
        "other",
    ]
    novelty: Literal["new", "update", "rehash"] | None
    materiality: Literal["low", "medium", "high"] | None
    surprise_vs_expectations: float | None
    rationale: str

    @field_validator("sentiment", "surprise_vs_expectations")
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

    @property
    def rationale_words(self) -> int:
        return len(self.rationale.split())


def parse_output(text: str) -> tuple[ScoreOutput | None, str | None]:
    """Returns (parsed, error). Over-long rationales are kept (flagged via rationale_words)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json: {exc}"
    try:
        return ScoreOutput.model_validate(data), None
    except ValidationError as exc:
        return None, f"validation: {exc.errors(include_url=False)}"
