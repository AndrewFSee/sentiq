"""Non-LLM text baselines on the same items, so the LLM's *incremental* value can be tested.

* Loughran-McDonald (LM) dictionary: counts of positive / negative / uncertainty / litigious
  words. ``lm_tone = (pos - neg) / max(pos + neg, 1)`` in [-1, 1] (same scale as LLM sentiment);
  proportions are per total words. Following Loughran & McDonald, a positive word preceded
  within three words by a negation (no, not, none, neither, never, nobody) counts as negative.
  Words the dictionary marks as removed (negative year) are excluded.
  Caveat: the current dictionary is applied to all dates; a few words were added after 2009,
  which is a (small) look-ahead for documents older than the year they were added.
* FinBERT (optional, ``features.finbert``; ``uv sync --extra finbert``): ProsusAI/finbert
  probabilities averaged over ~400-word chunks, ``finbert_score = p_pos - p_neg``.

Both are document-level (not target-specific): every ticker linked to an item gets the same
baseline score. That is a real difference from the LLM and part of what the test measures.
Results live in ``lake/baselines`` keyed by (item_id, text_hash, method, method_version).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from newsfeed.storage import Store
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)

LM_CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious")
NEGATIONS = {"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"}
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class LMDictionary:
    version: str  # first 12 hex chars of the CSV's sha256
    words: dict[str, frozenset[str]]  # category -> upper-case words

    @classmethod
    def from_rows(cls, rows: dict[str, Sequence[str]], version: str) -> LMDictionary:
        return cls(version, {c: frozenset(w.upper() for w in ws) for c, ws in rows.items()})


@lru_cache(maxsize=2)
def load_lm_dictionary(path: Path) -> LMDictionary:
    """Official SRAF CSV: a category column holds the year a word was added (0 = not in the
    category; negative = removed in that year)."""
    if not path.exists():
        raise FileNotFoundError(
            f"LM dictionary not found at {path}. Download the Master Dictionary CSV from "
            "https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (free for academic use)."
        )
    version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    df = pl.read_csv(path, infer_schema_length=0, columns=["Word", *LM_CATEGORIES])
    words = {
        c: frozenset(df.filter(pl.col(c).cast(pl.Int64) > 0)["Word"].str.to_uppercase().to_list())
        for c in LM_CATEGORIES
    }
    return LMDictionary(version, words)


def lm_counts(text: str, lm: LMDictionary) -> dict[str, Any]:
    tokens = [t.upper() for t in _WORD.findall(text)]
    pos = neg = unc = lit = 0
    for i, tok in enumerate(tokens):
        if tok in lm.words["Positive"]:
            if NEGATIONS & set(tokens[max(0, i - 3) : i]):
                neg += 1
            else:
                pos += 1
        elif tok in lm.words["Negative"]:
            neg += 1
        if tok in lm.words["Uncertainty"]:
            unc += 1
        if tok in lm.words["Litigious"]:
            lit += 1
    n = len(tokens)
    return {
        "n_words": n,
        "n_positive": pos,
        "n_negative": neg,
        "n_uncertainty": unc,
        "n_litigious": lit,
        "score": (pos - neg) / max(pos + neg, 1),
        "neg_prop": neg / n if n else 0.0,
        "pos_prop": pos / n if n else 0.0,
        "unc_prop": unc / n if n else 0.0,
        "lit_prop": lit / n if n else 0.0,
    }


# ------------------------------------------------------------------ FinBERT
Classifier = Callable[[list[str]], list[dict[str, float]]]  # chunks -> [{label: prob}]


def make_finbert(model: str) -> Classifier:  # pragma: no cover - heavy optional dependency
    try:
        from transformers import pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("FinBERT needs `uv sync --extra finbert`") from exc
    pipe = pipeline("text-classification", model=model, top_k=None, truncation=True)

    def classify(chunks: list[str]) -> list[dict[str, float]]:
        return [{d["label"].lower(): d["score"] for d in out} for out in pipe(chunks)]

    return classify


def finbert_score(text: str, classify: Classifier, words_per_chunk: int = 400) -> dict[str, Any]:
    words = text.split()
    chunks = [
        " ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)
    ] or [""]
    probs = classify(chunks)
    weights = [max(len(c.split()), 1) for c in chunks]
    tot = sum(weights)

    def avg(label: str) -> float:
        return sum(p.get(label, 0.0) * w for p, w in zip(probs, weights, strict=True)) / tot

    p_pos, p_neg, p_neu = avg("positive"), avg("negative"), avg("neutral")
    return {"score": p_pos - p_neg, "p_positive": p_pos, "p_negative": p_neg, "p_neutral": p_neu}


# ------------------------------------------------------------------ incremental compute
def compute_baselines(
    store: Store,
    *,
    lm: LMDictionary | None,
    finbert: tuple[str, Classifier] | None = None,
    run_id: str,
) -> dict[str, int]:
    """Score story founders that lack a baseline for their current text + method version."""
    methods: list[tuple[str, str]] = []
    if lm is not None:
        methods.append(("lm", lm.version))
    if finbert is not None:
        methods.append(("finbert", finbert[0]))
    out: dict[str, int] = {}
    for method, version in methods:
        # two steps: ids first (no text columns, so the view never materialises the bodies of
        # the whole lake -- that took ~15 GB and hours on a 16 GB laptop, 2026-10-23), then the
        # text of just those items
        ids = store.query(
            """
            SELECT i.item_id FROM items i
            WHERE i.item_id IN (SELECT item_id FROM item_story WHERE story_id = item_id)
              AND NOT EXISTS (
                SELECT 1 FROM baselines b WHERE b.item_id = i.item_id
                  AND b.text_hash = i.text_hash AND b.method = $m AND b.method_version = $v)
            """,
            {"m": method, "v": version},
        )["item_id"].to_list()
        todo = store.items_by_id(ids, columns="item_id, text_hash, title, body_text").to_dicts()
        rows = []
        now = utcnow()
        for r in todo:
            text = f"{r['title'] or ''}\n{r['body_text'] or ''}"
            if method == "lm":
                assert lm is not None
                vals = lm_counts(text, lm)
            else:
                assert finbert is not None
                vals = finbert_score(text, finbert[1])
            rows.append(
                {
                    "item_id": r["item_id"],
                    "text_hash": r["text_hash"],
                    "method": method,
                    "method_version": version,
                    "computed_at": now,
                    **{k: vals.get(k) for k in BASELINE_VALUE_COLS},
                }
            )
        out[method] = store.write_baselines(rows, run_id)
    return out


BASELINE_VALUE_COLS = (
    "score",
    "n_words",
    "n_positive",
    "n_negative",
    "n_uncertainty",
    "n_litigious",
    "neg_prop",
    "pos_prop",
    "unc_prop",
    "lit_prop",
    "p_positive",
    "p_negative",
    "p_neutral",
)
