"""Exploratory: cluster documents by text, then ask whether LLM sentiment predicts returns
within any cluster.

Clusters are unsupervised (TF-IDF over title + body -> TruncatedSVD -> k-means), fitted on
pre-holdout documents only, and never see returns. Each scored (document, company) event is
dated to the first decision time (session close - lag) at or after it became usable
(``tradable_at``; live items ``max(tradable_at, scored_at)``), so the test is causal. Results
are a breakdown with one Benjamini-Hochberg correction across clusters x cells: candidates for
fresh data, not findings.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from newsfeed.config import AppConfig
from newsfeed.eval.cv import pre_holdout_mask
from newsfeed.eval.report import block_bootstrap, boot_summary, spearman


def fit_text_clusters(
    texts: Sequence[str],
    fit_mask: np.ndarray,
    *,
    k: int,
    seed: int,
    n_components: int = 100,
    min_df: int = 5,
) -> tuple[np.ndarray, list[list[str]]]:
    """Cluster label per text (fit on ``fit_mask`` rows, predict for all) and the top terms of
    each cluster centre."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [t or "" for t in texts]
    fit_texts = [t for t, m in zip(texts, fit_mask, strict=True) if m]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.5,
        stop_words="english",
        sublinear_tf=True,
        max_features=100_000,
    )
    x_fit = vec.fit_transform(fit_texts)
    n_comp = max(2, min(n_components, x_fit.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    z_fit = normalize(svd.fit_transform(x_fit))
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(z_fit)
    labels = km.predict(normalize(svd.transform(vec.transform(texts))))
    centres = svd.inverse_transform(km.cluster_centers_)
    terms = vec.get_feature_names_out()
    top = [[str(terms[j]) for j in np.argsort(-c)[:8]] for c in centres]
    return labels.astype(int), top


def decision_times(t_usable: Sequence[datetime], grid: Sequence[datetime]) -> list[datetime | None]:
    """First grid time >= each usable time (None beyond the grid). ``grid`` sorted."""
    out: list[datetime | None] = []
    for t in t_usable:
        i = bisect.bisect_left(grid, t)
        out.append(grid[i] if i < len(grid) else None)
    return out


def cluster_ic(
    events: pl.DataFrame,
    targets: pl.DataFrame,
    sessions: list,
    cfg: AppConfig,
    cells: list[tuple[str, int, str]],
    *,
    min_n: int = 30,
) -> list[dict[str, Any]]:
    """events: ticker, date, cluster, sentiment. Rank IC of sentiment vs the target per cluster
    and cell, pre-holdout only, circular block bootstrap over calendar sessions."""
    ec = cfg.eval
    rng = np.random.default_rng(ec.seed)
    session_idx = {d: i for i, d in enumerate(sessions)}
    out = []
    for entry, h, adj in cells:
        t = targets.filter((pl.col("entry") == entry) & (pl.col("horizon") == h))
        df = events.join(t, on=["ticker", "date"], how="inner")
        if df.height:
            dec = df["date"].to_numpy().astype("datetime64[D]")
            end = df["label_end"].to_numpy().astype("datetime64[D]")
            df = df.filter(pl.Series(pre_holdout_mask(dec, end, ec.holdout_start)))
        for c in sorted(events["cluster"].unique().to_list()):
            sub = df.filter(pl.col("cluster") == c) if df.height else df
            row: dict[str, Any] = {
                "cluster": c,
                "entry": entry,
                "horizon": h,
                "adjustment": adj,
                "n": sub.height,
            }
            if sub.height < min_n:
                nan = float("nan")
                out.append({**row, "ic": nan, "ci_lo": nan, "ci_hi": nan, "p": nan})
                continue
            s = np.array([session_idx.get(d, -1) for d in sub["date"].to_list()])
            x = sub["sentiment"].to_numpy().astype(float)
            y = sub[f"ret_{adj}"].to_numpy().astype(float)
            ic = spearman(x, y)
            boots = block_bootstrap(
                s,
                lambda idx, x=x, y=y: spearman(x[idx], y[idx]),
                block=h + 1 if entry == "close" else h + 2,
                reps=ec.bootstrap_reps,
                rng=rng,
            )
            out.append({**row, "ic": ic, **boot_summary(ic, boots)})
    return out
