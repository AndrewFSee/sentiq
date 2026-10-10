"""Versioned prompts. One file per version under ``prompts/``; never edit a version that has been
used for scoring -- copy it to a new version instead (the hash changes either way, so edits
silently become a new, separately-cached experiment).

``prompt_hash`` covers everything sent to the model except the article itself and the model id:
system text, user template, output schema, and generation settings (effort, thinking,
max_tokens). The model id is a separate component of the score cache key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from newsfeed.scoring.schema import OUTPUT_JSON_SCHEMA, SCHEMA_VERSION

PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_MARK = "=== SYSTEM ===\n"
_USER_MARK = "=== USER ===\n"


@dataclass(frozen=True)
class GenerationSettings:
    effort: str
    thinking: str  # adaptive | disabled
    max_tokens: int


@dataclass(frozen=True)
class PromptSpec:
    version: str
    system: str
    user_template: str

    def render_user(
        self,
        *,
        target: str,
        doc_type: str,
        publisher: str | None,
        title: str | None,
        body: str,
        truncated: bool,
    ) -> str:
        note = "\nNOTE: the document text was truncated at a length limit." if truncated else ""
        return self.user_template.format(
            target=target,
            doc_type=doc_type,
            publisher=publisher or "unknown",
            title=(title or "").strip() or "(none)",
            truncation_note=note,
            body=body.strip(),
        )

    def hash(self, settings: GenerationSettings) -> str:
        blob = json.dumps(
            {
                "system": self.system,
                "user_template": self.user_template,
                "schema_version": SCHEMA_VERSION,
                "schema": OUTPUT_JSON_SCHEMA,
                "settings": settings.__dict__,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@lru_cache(maxsize=8)
def load_prompt(version: str, directory: Path = PROMPTS_DIR) -> PromptSpec:
    text = (directory / f"{version}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.startswith(_SYSTEM_MARK) or _USER_MARK not in text:
        raise ValueError(f"prompt {version}: expected '{_SYSTEM_MARK.strip()}' then a user section")
    system, user = text[len(_SYSTEM_MARK) :].split(_USER_MARK, 1)
    for field in ("{target}", "{body}", "{doc_type}", "{publisher}", "{title}"):
        if field not in user:
            raise ValueError(f"prompt {version}: user template lacks {field}")
    return PromptSpec(version=version, system=system.strip(), user_template=user.rstrip() + "\n")


def doc_type_for(source: str, meta: dict) -> str:
    if source == "edgar":
        return f"SEC Form {meta.get('form', '8-K')} filing (current report)"
    return "news article"
