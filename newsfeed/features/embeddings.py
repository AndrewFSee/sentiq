"""Text embeddings for story founders, and an embedding-based novelty measure.

Model: ``features.embedding_model`` (default BAAI/bge-small-en-v1.5 via fastembed: local ONNX,
no API key, deterministic). A document's vector is the L2-normalised mean of its chunk vectors
(~300-word chunks of title + body). Stored in ``lake/embeddings`` keyed by
(item_id, text_hash, model).

Novelty (computed in features.aggregate): for a founder event of ticker k at time t,
``max_sim`` = max cosine similarity to k's founder events with tradable_at in
[t - novelty_lookback_days, t) -- strictly earlier, so it only uses what was known at t.
Window features: ``emb_novelty_mean`` = mean(1 - max_sim), ``emb_n_novel`` = events with
max_sim < novelty_sim_threshold. Null when the look-back span is not inside coverage.
This is an objective alternative to the LLM's ``novelty`` label, which proved unreliable.

PIT note: the embedding model is fixed (trained before our sample), so it encodes general
language, not knowledge of specific later events in a way the features could exploit beyond
what the text says -- but it is a pretrained model, like the LLM; the masking study is the check
for the LLM, and embedding features enter the eval as their own block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from newsfeed.storage import Store
from newsfeed.timeutil import utcnow

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, d) float32, rows L2-normalised."""
        ...


class FastEmbedder:
    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # deferred: model load takes seconds

        self.name = model
        self._m = TextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self._m.embed(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def document_vector(text: str, embedder: Embedder, words_per_chunk: int = 300) -> np.ndarray:
    words = text.split()
    chunks = [
        " ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)
    ] or [""]
    v = embedder.embed(chunks).mean(axis=0)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def compute_embeddings(store: Store, embedder: Embedder, run_id: str) -> int:
    """Embed story founders lacking a vector for their current text and this model."""
    todo = store.query(
        """
        SELECT i.item_id, i.text_hash, i.title, i.body_text FROM items i
        WHERE i.item_id IN (SELECT item_id FROM item_story WHERE story_id = item_id)
          AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.item_id = i.item_id
                          AND e.text_hash = i.text_hash AND e.model = $m)
        """,
        {"m": embedder.name},
    ).to_dicts()
    now = utcnow()
    rows = []
    for r in todo:
        v = document_vector(f"{r['title'] or ''}\n{r['body_text'] or ''}", embedder)
        rows.append(
            {
                "item_id": r["item_id"],
                "text_hash": r["text_hash"],
                "model": embedder.name,
                "dim": int(v.shape[0]),
                "vector": v.tolist(),
                "computed_at": now,
            }
        )
    return store.write_embeddings(rows, run_id)


def max_prior_similarity(t: np.ndarray, vecs: np.ndarray, lookback_us: int) -> np.ndarray:
    """For events sorted by time ``t`` (micros) with unit vectors ``vecs``: max cosine
    similarity to any event with t_j in [t_i - lookback, t_i). 0.0 when there is none."""
    out = np.zeros(len(t), dtype=np.float64)
    for i in range(len(t)):
        lo = np.searchsorted(t, t[i] - lookback_us, "left")
        hi = np.searchsorted(t, t[i], "left")  # strictly earlier
        if hi > lo:
            out[i] = float(np.max(vecs[lo:hi] @ vecs[i]))
    return out
