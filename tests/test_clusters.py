"""Exploratory text clusters: separation, causal decision times, per-cluster IC."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from newsfeed.eval.clusters import cluster_ic, decision_times, fit_text_clusters


def test_fit_text_clusters_separates_topics_and_predicts_unseen():
    rng = np.random.default_rng(0)
    a = ["merger acquisition deal shareholders approve tender offer premium"]
    b = ["quarterly revenue earnings per share beat guidance raised outlook"]
    texts = []
    for i in range(120):
        base = (a if i % 2 == 0 else b)[0].split()
        texts.append(" ".join(rng.permutation(base)[:6]))
    fit = np.array([i < 100 for i in range(120)])  # last 20 only predicted
    labels, top = fit_text_clusters(texts, fit, k=2, seed=1, min_df=2, n_components=5)
    assert len(set(labels[0::2])) == 1 and len(set(labels[1::2])) == 1
    assert labels[0] != labels[1] and len(top) == 2


def test_decision_times_never_before_the_event():
    grid = [datetime(2025, 1, d, 20, 50, tzinfo=UTC) for d in (2, 3, 6)]
    t = [
        datetime(2025, 1, 2, 20, 50, tzinfo=UTC),  # exactly at the decision: usable
        datetime(2025, 1, 2, 20, 51, tzinfo=UTC),  # one minute late -> next session
        datetime(2025, 1, 7, tzinfo=UTC),  # beyond the grid
    ]
    assert decision_times(t, grid) == [grid[0], grid[1], None]


def test_cluster_ic_finds_signal_only_in_its_cluster(cfg):
    rng = np.random.default_rng(3)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(400)]
    rows, tgt = [], []
    for d in days:
        for j, cl in enumerate((0, 1)):
            s = float(rng.normal())
            r = 0.01 * rng.normal() + (0.01 * s if cl == 0 else 0.0)
            tk = f"T{j}"
            rows.append({"ticker": tk, "date": d, "cluster": cl, "sentiment": s})
            tgt.append({"ticker": tk, "date": d, "entry": "close", "horizon": 1,
                        "label_end": d + timedelta(days=1), "ret_market": r})  # fmt: skip
    ec = cfg.eval.model_copy(update={"holdout_start": date(2030, 1, 1), "bootstrap_reps": 200})
    out = cluster_ic(
        pl.DataFrame(rows),
        pl.DataFrame(tgt),
        days,
        cfg.model_copy(update={"eval": ec}),
        [("close", 1, "market")],
    )
    by = {r["cluster"]: r for r in out}
    assert by[0]["ic"] > 0.3 and by[0]["p"] < 0.01
    assert abs(by[1]["ic"]) < 0.15


def test_decision_times_delay_shifts_by_sessions():
    grid = [datetime(2025, 1, d, 20, 50, tzinfo=UTC) for d in (2, 3, 6, 7)]
    t = [datetime(2025, 1, 2, 14, 30, tzinfo=UTC)]
    assert decision_times(t, grid, delay=0) == [grid[0]]
    assert decision_times(t, grid, delay=2) == [grid[2]]
    assert decision_times(t, grid, delay=4) == [None]
