"""Test 11 machinery: MD&A tone of 10-K / 10-Q filings and its year-over-year change."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from newsfeed.eval.docchange import pair_filings
from newsfeed.eval.mdna import mdna_text, mdna_tones, tone_changes
from newsfeed.features.baselines import LMDictionary

LM = LMDictionary.from_rows(
    {"Positive": ["strong", "improved"], "Negative": ["decline", "loss"], "Uncertainty": [],
     "Litigious": []},
    "test",
)  # fmt: skip


def t(y, m, d):
    return datetime(y, m, d, 15, tzinfo=UTC)


def _filing(item_id, ts, form, period, body, key):
    head = "Item 1. Business. We make widgets. "
    sections = {key: [len(head), len(head) + len(body)]} if key else {}
    row = {"item_id": item_id, "tradable_at": ts, "cik": 7, "form": form, "period": period,
           "sections": json.dumps(sections)}  # fmt: skip
    return row, head + body + " Item 8. Statements: loss loss."


def _setup(rows):
    f = pl.DataFrame(
        [r for r, _ in rows],
        schema={"item_id": pl.Utf8, "tradable_at": pl.Datetime("us", "UTC"), "cik": pl.Int64,
                "form": pl.Utf8, "period": pl.Date, "sections": pl.Utf8},
    )  # fmt: skip
    bodies = {r["item_id"]: b for r, b in rows}
    loads = []

    def load(ids):
        loads.append(ids)
        return {i: bodies[i] for i in ids}

    return f, load, loads


def test_mdna_text_uses_the_form_s_section():
    body = "intro MDA-TEXT outro"
    assert mdna_text(body, "10-K", {"7": [6, 14]}) == "MDA-TEXT"
    assert mdna_text(body, "10-Q", {"I-2": [6, 14]}) == "MDA-TEXT"
    assert mdna_text(body, "10-Q", {"7": [6, 14]}) is None  # a 10-Q has no item 7
    assert mdna_text(body, "10-K", {}) is None


def test_tones_read_only_the_mdna_and_change_is_vs_year_earlier_pair():
    rows = [
        _filing("k20", t(2021, 2, 1), "10-K", date(2020, 12, 31),
                "Sales were strong and margins improved.", "7"),
        _filing("k21", t(2022, 2, 1), "10-K", date(2021, 12, 31),
                "Sales were strong, but a decline and a loss followed.", "7"),
        _filing("q21", t(2021, 5, 1), "10-Q", date(2021, 3, 31), "No section found.", None),
        _filing("q22", t(2022, 5, 1), "10-Q", date(2022, 3, 31), "Results improved.", "I-2"),
    ]  # fmt: skip
    f, load, loads = _setup(rows)
    calls = []

    def classify(chunks):
        calls.append(chunks)
        return [{"positive": 0.6, "negative": 0.1, "neutral": 0.3} for _ in chunks]

    tn = mdna_tones(f, load, LM, classify=classify)
    by = {r["item_id"]: r for r in tn.to_dicts()}
    assert set(by) == {"k20", "k21", "q22"}  # q21 has no MD&A
    assert by["k20"]["lm_tone"] == 1.0  # "loss loss" in item 8 is not read
    assert by["k21"]["lm_tone"] == pytest.approx((1 - 2) / 3)
    assert by["k21"]["lm_neg_prop"] == pytest.approx(2 / 10)
    assert by["q22"]["fb_tone"] == pytest.approx(0.5)
    assert all("Item 8" not in c for cs in calls for c in cs)
    assert len(loads) == 1  # only filings with sections are loaded, in one chunk

    ch = {r["item_id"]: r for r in tone_changes(f, tn, pair_filings(f)).to_dicts()}
    assert ch["k21"]["prior_item_id"] == "k20"
    assert ch["k21"]["d_lm_tone"] == pytest.approx((1 - 2) / 3 - 1.0)
    assert ch["k21"]["d_fb_tone"] == pytest.approx(0.0)
    assert ch["k20"]["d_lm_tone"] is None  # no year-earlier filing
    # paired, but the pair has no MD&A -> no change
    assert ch["q22"]["prior_item_id"] == "q21" and ch["q22"]["d_lm_tone"] is None
    assert ch["q22"]["lm_tone"] == 1.0  # the level still counts


def test_tones_without_finbert_leave_it_null():
    rows = [_filing("k20", t(2021, 2, 1), "10-K", date(2020, 12, 31), "A decline.", "7")]
    f, load, _ = _setup(rows)
    tn = mdna_tones(f, load, LM)
    assert tn["lm_tone"].to_list() == [-1.0] and tn["fb_tone"].to_list() == [None]
