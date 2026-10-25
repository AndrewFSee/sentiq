"""Document change between consecutive annual / quarterly reports (test 10, "Lazy Prices").

Cohen, Malloy and Nguyen (2020) find that firms whose 10-K / 10-Q text changes a lot from the
previous year's report underperform over the following months. Here each filing is paired with
the same company's report of the same form for the period one year earlier, and the change is
measured on bags of words (no model, no LLM): cosine similarity of term-frequency vectors
(primary) and Jaccard similarity of word sets, for the whole document and for the MD&A and
risk-factor sections where both filings have them.

Point in time: the comparison filing is always older than the filing scored (tradable earlier),
and similarity is a function of the two texts only, so a filing's value exists at its own
``tradable_at``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

WORD = re.compile(r"[a-z]{2,}")
PAIR_MIN_DAYS, PAIR_MAX_DAYS = 300, 430
# section keys per form: (MD&A, risk factors)
SECTION_KEYS = {"10-K": ("7", "1A"), "10-Q": ("I-2", "II-1A")}


def term_counts(text: str) -> Counter[str]:
    return Counter(WORD.findall(text.lower()))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return float("nan")
    small, large = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * large.get(k, 0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return float("nan")
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _period(p: str | None) -> date | None:
    if not p or len(p) != 8 or not p.isdigit():
        return None
    try:
        return date(int(p[:4]), int(p[4:6]), int(p[6:]))
    except ValueError:
        return None


def filing_meta(items: pl.DataFrame) -> pl.DataFrame:
    """items (item_id, tradable_at, meta) -> one row per filing: item_id, tradable_at, cik,
    form, period (date), sections (JSON)."""
    rows = []
    for r in items.iter_rows(named=True):
        m = json.loads(r["meta"])
        rows.append(
            {
                "item_id": r["item_id"],
                "tradable_at": r["tradable_at"],
                "cik": int(m["cik"]),
                "form": m["form"],
                "period": _period(m.get("period")),
                "sections": json.dumps(m.get("sections") or {}),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "item_id": pl.Utf8,
            "tradable_at": pl.Datetime("us", "UTC"),
            "cik": pl.Int64,
            "form": pl.Utf8,
            "period": pl.Date,
            "sections": pl.Utf8,
        },
    )


def pair_filings(f: pl.DataFrame) -> pl.DataFrame:
    """For each filing, the same CIK + form filing whose period is 300-430 days earlier and
    which became tradable before it (closest to 365 days; ties -> the more recent one). Returns
    item_id, prior_item_id, gap_days for filings that have a pair."""
    out = []
    for (_, _), g in f.filter(pl.col("period").is_not_null()).group_by("cik", "form"):
        rows = g.sort("period", "tradable_at").to_dicts()
        for r in rows:
            best: tuple[int, int, dict] | None = None
            for q in rows:
                gap = (r["period"] - q["period"]).days
                if (
                    not (PAIR_MIN_DAYS <= gap <= PAIR_MAX_DAYS)
                    or q["tradable_at"] >= r["tradable_at"]
                ):
                    continue
                key = (abs(gap - 365), gap)
                if best is None or key < best[:2]:
                    best = (*key, q)
            if best is not None:
                q = best[2]
                gap = (r["period"] - q["period"]).days
                out.append(
                    {"item_id": r["item_id"], "prior_item_id": q["item_id"], "gap_days": gap}
                )
    return pl.DataFrame(
        out, schema={"item_id": pl.Utf8, "prior_item_id": pl.Utf8, "gap_days": pl.Int64}
    )


def _section(text: str, sections: dict[str, list[int]], key: str) -> str | None:
    span = sections.get(key)
    return text[span[0] : span[1]] if span else None


def similarities(
    f: pl.DataFrame,
    pairs: pl.DataFrame,
    load_bodies: Callable[[list[str]], dict[str, str]],
    *,
    chunk_ciks: int = 25,
) -> pl.DataFrame:
    """Similarity of each paired filing to its comparison filing. Bodies are loaded a few
    companies at a time (``load_bodies(item_ids) -> {item_id: text}``), so memory holds one
    chunk of documents, not all ~15k reports."""
    meta = {r["item_id"]: r for r in f.to_dicts()}
    by_cik: dict[int, list[dict[str, Any]]] = {}
    for p in pairs.to_dicts():
        by_cik.setdefault(meta[p["item_id"]]["cik"], []).append(p)
    ciks = sorted(by_cik)
    out = []
    for c0 in range(0, len(ciks), chunk_ciks):
        chunk = [p for c in ciks[c0 : c0 + chunk_ciks] for p in by_cik[c]]
        ids = sorted({p["item_id"] for p in chunk} | {p["prior_item_id"] for p in chunk})
        bodies = load_bodies(ids)
        counts = {i: term_counts(bodies.get(i, "")) for i in ids}
        for p in chunk:
            a, b = p["item_id"], p["prior_item_id"]
            ma, mb = meta[a], meta[b]
            row = {
                **p,
                "cos": cosine(counts[a], counts[b]),
                "jac": jaccard(counts[a], counts[b]),
                "n_words": sum(counts[a].values()),
                "n_words_prior": sum(counts[b].values()),
            }
            sa, sb = json.loads(ma["sections"]), json.loads(mb["sections"])
            for name, key in zip(
                ("mda", "rf"), SECTION_KEYS.get(ma["form"], ("", "")), strict=True
            ):
                ta, tb = _section(bodies.get(a, ""), sa, key), _section(bodies.get(b, ""), sb, key)
                row[f"cos_{name}"] = (
                    cosine(term_counts(ta), term_counts(tb)) if ta and tb else float("nan")
                )
            out.append(row)
        log.info("doc-change: %d / %d companies", min(c0 + chunk_ciks, len(ciks)), len(ciks))
    res = pl.DataFrame(out) if out else pl.DataFrame()
    return res.join(f.select("item_id", "tradable_at", "cik", "form", "period"), on="item_id")
