"""8-K hand labels vs the model (pre-registered 2026-10-20, descriptive; no BH family).

    uv run python scripts/label_agreement_8k.py [--sample edgar_8k_sample_v1]

Reads data/labeling/<sample>_labels.csv (hand: has_results, has_outlook, event_type,
direction, relevance, unsure) and <sample>_key.csv (model labels; the sample is 60 filings the
model called guidance + 40 it called earnings). Readouts, each on all labels and on the
confident ones (unsure = False):
  (1) precision of model guidance / model earnings vs the hand event type;
  (2) among model-guidance filings: share with reported results, and share that is
      outlook-only (has_outlook yes, has_results no) -- pre-registered reading: if fewer than
      half are outlook-only, H2 is described as "8-Ks the model labels guidance (mostly results
      + outlook)", not as guidance-only releases;
  (3) sentiment agreement (model sentiment vs hand direction), and the distilled student
      (run full_v1, which never trained on these gold filings) on the same rows.
Proportions get Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DIRECTION = {"negative": -1, "neutral": 0, "positive": 1}


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    p, z = k / n, 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{k}/{n} = {p:.2f} [{mid - half:.2f}, {mid + half:.2f}]"


def sentiment(df: pl.DataFrame, col: str) -> str:
    s, h = df[col].to_numpy().astype(float), df["h_dir"].to_numpy()
    rho = spearmanr(s, h).statistic
    nz = h != 0
    auc = roc_auc_score(h[nz] > 0, s[nz]) if 0 < (h[nz] > 0).sum() < nz.sum() else float("nan")
    means = ", ".join(
        f"{d} {np.mean(s[h == k]):+.2f}" for d, k in DIRECTION.items() if (h == k).any()
    )
    return (
        f"Spearman {rho:+.3f}, AUC pos vs neg {auc:.3f} (n {nz.sum()}); mean by hand label: {means}"
    )


def report(df: pl.DataFrame, label: str) -> None:
    print(f"\n=== {label}: {df.height} filings ===")
    mg, me = df.filter(pl.col("m_type") == "guidance"), df.filter(pl.col("m_type") == "earnings")
    print("(1) precision")
    prec_g = wilson((mg["h_type"] == "guidance").sum(), mg.height)
    prec_e = wilson((me["h_type"] == "earnings").sum(), me.height)
    print(f"    model guidance -> hand guidance: {prec_g}")
    print(f"    model earnings -> hand earnings: {prec_e}")
    print("    hand types for model guidance:", dict(mg["h_type"].value_counts().iter_rows()))
    print("    hand types for model earnings:", dict(me["h_type"].value_counts().iter_rows()))
    res = (mg["has_results"] == "yes").sum()
    only = ((mg["has_outlook"] == "yes") & (mg["has_results"] == "no")).sum()
    neither = ((mg["has_outlook"] == "no") & (mg["has_results"] == "no")).sum()
    print("(2) model-guidance filings")
    print(f"    with reported results:          {wilson(res, mg.height)}")
    print(f"    outlook-only (no results):      {wilson(only, mg.height)}")
    print(f"    neither results nor outlook:    {wilson(neither, mg.height)}")
    verdict = "mostly results + outlook" if only < mg.height / 2 else "mostly guidance-only"
    print(f"    pre-registered reading -> {verdict}")
    print("(3) sentiment vs hand direction")
    print(f"    model (Haiku): {sentiment(df, 'm_sent')}")
    st = df.filter(pl.col("pred_sent").is_not_null())
    if st.height:
        print(f"    student full_v1 (n {st.height}): {sentiment(st, 'pred_sent')}")
        stu = wilson((st["pred_event_type"] == st["h_type"]).sum(), st.height)
        mod = wilson((st["m_type"] == st["h_type"]).sum(), st.height)
        print(f"    event type = hand: student {stu}; model on the same rows {mod}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="edgar_8k_sample_v1")
    ap.add_argument("--student", default="full_v1")
    args = ap.parse_args()
    d = ROOT / "data" / "labeling"
    hand = pl.read_csv(d / f"{args.sample}_labels.csv")
    key = pl.read_csv(d / f"{args.sample}_key.csv", infer_schema_length=None)
    df = key.join(hand, on="id", how="inner", suffix="_h").select(
        "id", "item_id", "ticker",
        pl.col("event_type").alias("m_type"), pl.col("sentiment").alias("m_sent"),
        pl.col("event_type_h").alias("h_type"),
        pl.col("direction").replace_strict(DIRECTION).alias("h_dir"),
        "has_results", "has_outlook", "unsure",
    )  # fmt: skip
    pred = ROOT / "data" / "distill" / "runs" / args.student / "predictions.parquet"
    if pred.exists():
        p = pl.read_parquet(pred).select("item_id", "ticker", "pred_sent", "pred_event_type")
        df = df.join(p, on=["item_id", "ticker"], how="left")
    else:
        df = df.with_columns(pl.lit(None).alias("pred_sent"), pl.lit(None).alias("pred_event_type"))
    print(f"{args.sample}: {df.height} labelled filings ({key.height} in key); "
          f"unsure {df['unsure'].sum()}")  # fmt: skip
    report(df, "all labels")
    report(df.filter(~pl.col("unsure")), "confident labels (unsure = False)")


if __name__ == "__main__":
    main()
