"""Agreement between two scoring setups on the same (item, ticker) pairs.

Used for three studies:
  * masked vs unmasked (same model/prompt) -- look-ahead/leakage check: a model that relies on
    knowing *which* company it is (possibly including how the story ended) disagrees with
    itself when names are hidden.
  * primary vs validation model (Haiku vs Sonnet) -- label quality.
  * replicate 0 vs replicates 1..K -- test-retest noise floor of a single sampled draw.
Interpretation: the masked-vs-unmasked gap should be judged against the replicate noise floor;
agreement no worse than re-sampling noise is evidence against identity-driven scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from newsfeed.storage import Store

CATEGORICAL = ["relevance", "event_type", "novelty", "materiality"]
VALUE_COLS = ("sentiment", "confidence", *CATEGORICAL, "surprise_vs_expectations")


@dataclass(frozen=True)
class Side:
    model: str
    prompt_hash: str
    masked: bool = False
    replicate: int | None = 0  # None = any replicate >= 1


def _side_sql(alias: str, s: Side) -> tuple[str, dict[str, Any]]:
    rep = "replicate >= 1" if s.replicate is None else f"replicate = {int(s.replicate)}"
    cols = ", ".join(f"{c} AS {c}_{alias}" for c in ("replicate", *VALUE_COLS))
    sql = (
        f"SELECT item_id, ticker, text_hash, {cols} "
        f"FROM scores WHERE status = 'ok' AND model = ${alias}_m AND prompt_hash = ${alias}_h "
        f"AND masked = ${alias}_k AND {rep}"
    )
    return sql, {f"{alias}_m": s.model, f"{alias}_h": s.prompt_hash, f"{alias}_k": s.masked}


def paired(store: Store, a: Side, b: Side) -> pl.DataFrame:
    sa, pa_ = _side_sql("a", a)
    sb, pb = _side_sql("b", b)
    return store.query(
        f"SELECT * FROM ({sa}) a JOIN ({sb}) b USING (item_id, ticker, text_hash)", {**pa_, **pb}
    )


def _kappa(x: list[Any], y: list[Any]) -> float:
    """Cohen's kappa; None is treated as its own category."""
    cats = sorted({str(v) for v in x + y})
    if len(x) == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for u, v in zip(x, y, strict=True):
        m[idx[str(u)], idx[str(v)]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    return (sums / counts)[inv]


def metrics(df: pl.DataFrame, *, neutral_band: float = 0.1) -> dict[str, Any]:
    """``df`` from :func:`paired`: value columns suffixed ``_a`` / ``_b``."""
    n = df.height
    out: dict[str, Any] = {"n_pairs": n}
    if n == 0:
        return out
    sa = df["sentiment_a"].to_numpy().astype(float)
    sb = df["sentiment_b"].to_numpy().astype(float)
    if n >= 3 and sa.std() > 0 and sb.std() > 0:
        out["sentiment_pearson"] = float(np.corrcoef(sa, sb)[0, 1])
        out["sentiment_spearman"] = float(np.corrcoef(_rank(sa), _rank(sb))[0, 1])
    out["sentiment_mae"] = float(np.abs(sa - sb).mean())
    out["sentiment_mean_diff_a_minus_b"] = float((sa - sb).mean())

    def sign(v: np.ndarray) -> np.ndarray:
        return np.where(v > neutral_band, 1, np.where(v < -neutral_band, -1, 0))

    out["sentiment_sign_agreement"] = float((sign(sa) == sign(sb)).mean())
    ca, cb = df["confidence_a"].to_numpy(), df["confidence_b"].to_numpy()
    out["confidence_mae"] = float(np.abs(ca - cb).mean())
    for col in CATEGORICAL:
        x, y = df[f"{col}_a"].to_list(), df[f"{col}_b"].to_list()
        out[f"{col}_agreement"] = float(
            np.mean([str(u) == str(v) for u, v in zip(x, y, strict=True)])
        )
        out[f"{col}_kappa"] = _kappa(x, y)
    xa = df["surprise_vs_expectations_a"].to_list()
    xb = df["surprise_vs_expectations_b"].to_list()
    out["surprise_null_agreement"] = float(
        np.mean([(u is None) == (v is None) for u, v in zip(xa, xb, strict=True)])
    )
    both = [(u, v) for u, v in zip(xa, xb, strict=True) if u is not None and v is not None]
    if len(both) >= 3:
        u, v = np.array(both).T
        if u.std() > 0 and v.std() > 0:
            out["surprise_pearson"] = float(np.corrcoef(u, v)[0, 1])
    return out


def format_metrics(title: str, m: dict[str, Any]) -> str:
    lines = [f"== {title}"]
    for k, v in m.items():
        lines.append(f"  {k:34s} {v:.3f}" if isinstance(v, float) else f"  {k:34s} {v}")
    return "\n".join(lines)
