from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from newsfeed.distill.dataset import assign_splits, finalize, pre_holdout, render_texts
from newsfeed.scoring.prompt import doc_type_for, load_prompt

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def _frame(n: int, source: str = "news_api", start: datetime = T0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": [f"{source}{k:03d}" for k in range(n)],
            "ticker": ["AAA"] * n,
            "source": [source] * n,
            "tradable_at": [start + timedelta(days=k) for k in range(n)],
        }
    )


def test_pre_holdout_drops_rows_tradable_at_or_after_the_start() -> None:
    cut = datetime(2026, 7, 1, tzinfo=UTC)
    df = pl.DataFrame({"tradable_at": [cut - timedelta(microseconds=1), cut, cut + timedelta(1)]})
    out = pre_holdout(df, date(2026, 7, 1))
    assert out["tradable_at"].to_list() == [cut - timedelta(microseconds=1)]


def test_splits_are_chronological_per_source_and_gold_is_held_out() -> None:
    df = pl.concat([_frame(100, "news_api"), _frame(50, "edgar", T0 - timedelta(days=900))])
    gold = {("news_api010", "AAA"), ("edgar049", "AAA")}
    out = assign_splits(df, gold)
    assert out.filter(pl.col("split") == "gold").height == 2
    assert set(out.filter(pl.col("item_id").is_in(["news_api010", "edgar049"]))["split"]) == {
        "gold"
    }
    for src in ("news_api", "edgar"):
        s = out.filter(pl.col("source") == src)
        last = None
        for name in ("train", "val", "test"):
            part = s.filter(pl.col("split") == name)["tradable_at"]
            assert part.len() > 0, (src, name)
            if last is not None:
                assert part.min() > last
            last = part.max()
    news = out.filter((pl.col("source") == "news_api") & (pl.col("split") != "gold"))
    counts = dict(news["split"].value_counts().iter_rows())
    assert counts == {"train": 70, "val": 10, "test": 19}  # 99 non-gold rows, cuts at 70% / 80%


def test_split_cuts_never_separate_items_with_the_same_tradable_at() -> None:
    df = _frame(10).with_columns(pl.lit(T0).alias("tradable_at"))
    out = assign_splits(df, set())
    assert out["split"].n_unique() == 1


def test_bad_split_fractions_are_rejected() -> None:
    with pytest.raises(ValueError):
        assign_splits(_frame(5), set(), (0.8, 0.7))


def test_render_texts_reproduces_the_scored_user_turn_and_flags_changes() -> None:
    prompt = load_prompt("score_v1")
    meta = {"form": "8-K"}
    expected = prompt.render_user(
        target="Acme Corp (ticker: AAA)",
        doc_type=doc_type_for("edgar", meta),
        publisher="SEC EDGAR",
        title="ACME 8-K",
        body="Results were strong.",
        truncated=False,
    )
    h = hashlib.sha256(expected.encode()).hexdigest()[:16]
    row = {
        "cik": 1, "ticker": "AAA", "source": "edgar", "publisher": "SEC EDGAR",
        "title": "ACME 8-K", "body_text": "Results were strong.", "body_truncated": False,
        "meta": json.dumps(meta), "user_text_hash": h,
    }  # fmt: skip
    df = pl.DataFrame([row, {**row, "user_text_hash": "0" * 16}])
    out = render_texts(df, prompt, {1: "Acme Corp"})
    assert out["text"][0] == expected
    assert out["text_ok"].to_list() == [True, False]


def test_finalize_encodes_labels_and_rejects_unknown_ones() -> None:
    base = {
        "item_id": "x", "ticker": "AAA", "cik": 1, "source": "edgar", "published_at": T0,
        "tradable_at": T0, "split": "train", "text": "t", "text_ok": True, "sentiment": 0.5,
        "relevance": "primary", "event_type": "guidance", "prompt_hash": "p",
    }  # fmt: skip
    out = finalize(pl.DataFrame([base]))
    assert out.select("relevance_id", "event_type_id", "year").row(0) == (2, 1, 2026)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        finalize(pl.DataFrame([{**base, "event_type": "rumour"}]))
