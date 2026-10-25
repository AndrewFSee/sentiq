"""Test 10 machinery: bag-of-words similarity and year-earlier pairing of 10-K / 10-Q."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from newsfeed.eval.docchange import cosine, jaccard, pair_filings, similarities, term_counts


def test_term_counts_drop_numbers_and_single_letters():
    c = term_counts("Revenue rose 12% to $3.4 billion in Q3; a revenue record.")
    assert c["revenue"] == 2 and "12" not in c and "a" not in c and "q" not in c


def test_cosine_and_jaccard():
    a, b = term_counts("risk risk demand supply"), term_counts("risk demand supply supply")
    assert cosine(a, a) == pytest.approx(1.0) and jaccard(a, b) == 1.0
    assert 0 < cosine(a, b) < 1
    assert cosine(a, term_counts("tariffs litigation")) == 0.0
    assert cosine(a, term_counts("")) != cosine(a, term_counts(""))  # NaN for an empty text


def _f(rows):
    return pl.DataFrame(
        rows,
        schema={"item_id": pl.Utf8, "tradable_at": pl.Datetime("us", "UTC"), "cik": pl.Int64,
                "form": pl.Utf8, "period": pl.Date, "sections": pl.Utf8},
    )  # fmt: skip


def t(y, m, d):
    return datetime(y, m, d, 15, tzinfo=UTC)


def test_pair_filings_year_earlier_same_form_same_quarter():
    rows = [
        ("k21", t(2021, 2, 1), 1, "10-K", date(2020, 12, 31)),
        ("k22", t(2022, 2, 1), 1, "10-K", date(2021, 12, 31)),
        ("q21", t(2021, 5, 1), 1, "10-Q", date(2021, 3, 31)),
        ("q21b", t(2021, 8, 1), 1, "10-Q", date(2021, 6, 30)),
        ("q22", t(2022, 5, 1), 1, "10-Q", date(2022, 3, 31)),
        ("x22", t(2022, 2, 1), 2, "10-K", date(2021, 12, 31)),  # other company, no prior
    ]
    f = _f([{"item_id": a, "tradable_at": b, "cik": c, "form": d, "period": e,
             "sections": "{}"} for a, b, c, d, e in rows])  # fmt: skip
    p = {r["item_id"]: r["prior_item_id"] for r in pair_filings(f).to_dicts()}
    # 10-K pairs with the prior 10-K, a 10-Q with the same quarter a year earlier (not the
    # adjacent quarter, and never with a 10-K)
    assert p == {"k22": "k21", "q22": "q21"}


def test_pair_requires_the_comparison_to_be_tradable_first():
    f = _f([
        {"item_id": "late", "tradable_at": t(2023, 1, 1), "cik": 1, "form": "10-K",
         "period": date(2020, 12, 31), "sections": "{}"},  # filed very late
        {"item_id": "new", "tradable_at": t(2022, 2, 1), "cik": 1, "form": "10-K",
         "period": date(2021, 12, 31), "sections": "{}"},
    ])  # fmt: skip
    assert pair_filings(f).height == 0  # never compare with a document that came later


def test_similarities_whole_document_and_sections():
    risk_a = "competition supply chain cyber attack regulation " * 30
    risk_b = "competition supply chain pandemic war inflation " * 30
    mda = "revenue increased driven by services growth margin " * 30
    txt_a = mda + "\n" + risk_a
    txt_b = mda + "\n" + risk_b
    sec = lambda txt: json.dumps({"7": [0, len(mda)], "1A": [len(mda) + 1, len(txt)]})  # noqa: E731
    f = _f([
        {"item_id": "old", "tradable_at": t(2021, 2, 1), "cik": 7, "form": "10-K",
         "period": date(2020, 12, 31), "sections": sec(txt_a)},
        {"item_id": "new", "tradable_at": t(2022, 2, 1), "cik": 7, "form": "10-K",
         "period": date(2021, 12, 31), "sections": sec(txt_b)},
    ])  # fmt: skip
    loads = []

    def load(ids):
        loads.append(ids)
        return {"old": txt_a, "new": txt_b}

    out = similarities(f, pair_filings(f), load).to_dicts()
    assert len(out) == 1 and len(loads) == 1
    r = out[0]
    assert r["cos_mda"] == pytest.approx(1.0)  # MD&A unchanged
    assert r["cos_rf"] < 0.7  # risk factors rewritten
    assert r["cos_rf"] < r["cos"] < 1.0 and 0 < r["jac"] < 1
    assert r["item_id"] == "new" and r["prior_item_id"] == "old" and r["form"] == "10-K"
