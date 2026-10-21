"""Agreement between hand labels (News Label Desk) and the model's labels for the news sample.

    uv run python scripts/label_agreement.py [--sample news_sample_v1] [--reps 2000]

Reads data/labeling/<sample>_labels.csv (hand labels: direction / event_type / relevance) and
<sample>_key.csv (model labels, unmasked and masked, from make_label_sample.py). The sample is
stratified 25 per *model* event type, so overall rates are also reported reweighted to the
model event-type mix of the population the sample was drawn from (same filter as the sampler).
Per-model-type precision needs no weights. CIs: stratified bootstrap over items.
No network, no model calls.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from newsfeed.config import load_config
from newsfeed.storage import Store

TYPES = ["earnings", "guidance", "m&a", "legal_regulatory", "analyst_action", "product",
         "macro", "other"]  # fmt: skip
RELEVANCE = ["passing", "secondary", "primary"]
DIRECTION = {"negative": -1, "neutral": 0, "positive": 1}
BANDS = (0.0, 0.1, 0.2)  # model |sentiment| <= band -> neutral


def population_shares(cfg_path: str | None) -> dict[str, float]:
    """Model event-type mix of the population make_label_sample.py drew from."""
    cfg = load_config(cfg_path)
    with Store(cfg.paths) as s:
        df = s.query(
            """
            SELECT u.event_type, count(*) AS n
            FROM scores u JOIN items i USING (item_id)
            WHERE i.source = 'news_api' AND NOT u.masked AND u.status = 'ok'
              AND u.replicate = 0 AND i.published_at < TIMESTAMP '2026-07-01'
            GROUP BY 1
            """
        )
    total = df["n"].sum()
    return {r["event_type"]: r["n"] / total for r in df.to_dicts()}


def to_direction(s: np.ndarray, band: float) -> np.ndarray:
    return np.where(np.abs(s) <= band + 1e-9, 0, np.sign(s)).astype(int)


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def boot_ci(
    stat: Callable[[np.ndarray], float], strata: np.ndarray, reps: int, seed: int = 7
) -> tuple[float, float]:
    """Percentile CI of stat(idx) under resampling items within each stratum."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    vals = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(v: float, ci: tuple[float, float] | None = None) -> str:
    return f"{v:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def confusion(df: pl.DataFrame, rows: str, cols: str, order: list[str]) -> pl.DataFrame:
    seen = [c for c in order if c in set(df[cols].to_list())]
    return (
        df.group_by(rows, cols)
        .len()
        .pivot(on=cols, index=rows, values="len")
        .fill_null(0)
        .select([rows, *seen])
        .sort(pl.col(rows).replace_strict({t: i for i, t in enumerate(order)}, default=99))
    )


def sentiment_block(df: pl.DataFrame, col: str, w: np.ndarray, reps: int) -> None:
    sub = df.filter(pl.col(col).is_not_null())
    keep = df[col].is_not_null().to_numpy()
    s, h, ww = sub[col].to_numpy(), sub["h_dir"].to_numpy(), w[keep]
    strata = sub["m_type"].to_numpy()
    print(f"  n = {len(sub)}")
    rho = spearmanr(s, h).statistic
    ci = boot_ci(lambda i: spearmanr(s[i], h[i]).statistic, strata, reps)
    print(f"  Spearman(model sentiment, hand direction)  {fmt(rho, ci)}")
    nz = h != 0
    hn, sn = h[nz] > 0, s[nz]
    auc = roc_auc_score(hn, sn)
    ci = boot_ci(
        lambda i: roc_auc_score(hn[i], sn[i]) if 0 < hn[i].sum() < len(i) else np.nan,
        strata[nz],
        reps,
    )
    print(f"  AUC positive vs negative (hand), n = {nz.sum()}  {fmt(auc, ci)}")
    for d, k in DIRECTION.items():
        m = h == k
        print(f"    hand {d:8s} n={m.sum():3d}  mean model sentiment {np.mean(s[m]):+.3f}")
    print("  3-class agreement (model neutral if |s| <= band): raw / weighted / kappa")
    for b in BANDS:
        md = to_direction(s, b)
        acc = float(np.mean(md == h))
        wacc = wmean((md == h).astype(float), ww)
        kap = cohen_kappa_score(md, h)
        opp = int(np.sum(md * h == -1))
        print(
            f"    band {b:.1f}: {acc:.3f} / {wacc:.3f} / kappa {kap:+.3f}  (opposite sign: {opp})"
        )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="news_sample_v1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = Path(cfg.paths.data_dir) / "labeling"
    hand = pl.read_csv(d / f"{args.sample}_labels.csv")
    key = pl.read_csv(d / f"{args.sample}_key.csv", infer_schema_length=None)
    df = key.join(hand, on="id", how="inner", suffix="_h").rename(
        {
            "event_type": "m_type",
            "relevance": "m_rel",
            "sentiment": "m_sent",
            "m_event_type": "mm_type",
            "m_relevance": "mm_rel",
            "m_sentiment": "mm_sent",
            "event_type_h": "h_type",
            "relevance_h": "h_rel",
        }
    )
    df = df.with_columns(pl.col("direction").replace_strict(DIRECTION).alias("h_dir"))
    print(f"{args.sample}: {len(df)} labelled pairs ({len(key)} in key, {len(hand)} labels)")
    if len(df) != len(key):
        print("  WARNING: labels and key do not cover the same ids")

    shares = population_shares(args.config)
    n_by = df.group_by("m_type").len()
    n_map = dict(zip(n_by["m_type"], n_by["len"], strict=True))
    w = np.array([shares.get(t, 0.0) / n_map[t] for t in df["m_type"]])
    strata = df["m_type"].to_numpy()
    print("  population model event-type mix: "
          + ", ".join(f"{t} {shares.get(t, 0):.1%}" for t in TYPES))  # fmt: skip

    print("\n== Event type ==")
    for col, label in (("m_type", "unmasked"), ("mm_type", "masked")):
        sub = df.filter(pl.col(col).is_not_null())
        m, h = sub[col].to_numpy(), sub["h_type"].to_numpy()
        agree = (m == h).astype(float)
        keep = df[col].is_not_null().to_numpy()
        ci = boot_ci(lambda i, a=agree: float(np.mean(a[i])), strata[keep], args.reps)
        print(
            f"  {label:8s} n={len(sub)}  agreement {np.mean(agree):.3f}"
            f" [{ci[0]:.3f}, {ci[1]:.3f}]  population-weighted {wmean(agree, w[keep]):.3f}"
            f"  kappa {cohen_kappa_score(m, h):+.3f}"
        )
    print("  precision per model type (share the hand label agrees with; n = 25 each):")
    for t in TYPES:
        sub = df.filter(pl.col("m_type") == t)
        top = sub.group_by("h_type").len().sort("len", descending=True).head(3).to_dicts()
        alt = ", ".join(f"{r['h_type']} {r['len']}" for r in top)
        print(f"    {t:17s} {(sub['h_type'] == t).mean():.2f}   hand: {alt}")
    print("  confusion (rows = model unmasked, columns = hand):")
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, tbl_width_chars=160):  # fmt: skip
        print(confusion(df, "m_type", "h_type", TYPES))

    print("\n== Relevance ==")
    rank = {r: i for i, r in enumerate(RELEVANCE)}
    for col, label in (("m_rel", "unmasked"), ("mm_rel", "masked")):
        sub = df.filter(pl.col(col).is_not_null())
        keep = df[col].is_not_null().to_numpy()
        m = np.array([rank[x] for x in sub[col]])
        h = np.array([rank[x] for x in sub["h_rel"]])
        qm, qh = m >= 1, h >= 1  # qualifies for LLM features (>= secondary)
        exact_w = wmean((m == h) * 1.0, w[keep])
        print(
            f"  {label:8s} exact {np.mean(m == h):.3f} (weighted {exact_w:.3f})"
            f"  linear-weighted kappa {cohen_kappa_score(m, h, weights='linear'):+.3f}"
        )
        print(
            f"           qualifies (>= secondary): agreement {wmean((qm == qh) * 1.0, w[keep]):.3f}"
            f" weighted; of model-qualifying, hand agrees {wmean(qh[qm] * 1.0, w[keep][qm]):.3f};"
            f" of hand-qualifying, model agrees {wmean(qm[qh] * 1.0, w[keep][qh]):.3f}"
        )
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(confusion(df, "m_rel", "h_rel", RELEVANCE))

    print("\n== Sentiment ==")
    for col, label in (("m_sent", "unmasked"), ("mm_sent", "masked")):
        print(f" {label}, all pairs")
        sentiment_block(df, col, w, args.reps)
    q = df.filter(pl.col("m_rel").is_in(["primary", "secondary"]))
    wq = w[df["m_rel"].is_in(["primary", "secondary"]).to_numpy()]
    print(" unmasked, model relevance >= secondary (the pairs LLM features use)")
    sentiment_block(q, "m_sent", wq, args.reps)

    unsure = df.filter(pl.col("unsure"))
    if len(unsure):
        print(f"\nunsure labels: {', '.join(unsure['id'])}")


if __name__ == "__main__":
    main()
