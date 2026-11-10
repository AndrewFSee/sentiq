"""MD&A tone of 10-K / 10-Q filings and its change from the year-earlier report (test 11).

Feldman, Govindaraj, Livnat and Segal (2010) find that a change in the tone of the MD&A from
the previous comparable report predicts returns after the filing. Here tone is read locally
(no LLM): the Loughran-McDonald dictionary (``lm_counts``: tone = (pos - neg) / (pos + neg) with
the baseline negation rule, and the negative-word share) and optionally FinBERT
(``finbert_score``: p_pos - p_neg averaged over 400-word chunks). Pairs come from test 10's
``pair_filings`` (same CIK and form, period one year earlier, tradable first).

Point in time: a filing's tone is a function of its own text, and the comparison filing is
always older, so the change exists at the filing's ``tradable_at``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import polars as pl

from newsfeed.eval.docchange import SECTION_KEYS, _section
from newsfeed.features.baselines import Classifier, LMDictionary, finbert_score, lm_counts

log = logging.getLogger(__name__)

TONE_SCHEMA = {
    "item_id": pl.Utf8,
    "mda_words": pl.Int64,
    "lm_tone": pl.Float64,
    "lm_neg_prop": pl.Float64,
    "fb_tone": pl.Float64,
}
MEASURES = ("lm_tone", "lm_neg_prop", "fb_tone")


def mdna_text(body: str, form: str, sections: dict[str, list[int]]) -> str | None:
    """The MD&A section (10-K item 7, 10-Q part I item 2), or None if the parser found none."""
    key = SECTION_KEYS.get(form, ("", ""))[0]
    txt = _section(body, sections, key) if key else None
    return txt if txt and txt.strip() else None


def mdna_tones(
    f: pl.DataFrame,
    load_bodies: Callable[[list[str]], dict[str, str]],
    lm: LMDictionary,
    *,
    classify: Classifier | None = None,
    chunk: int = 200,
) -> pl.DataFrame:
    """One row per filing with an MD&A: item_id, mda_words, lm_tone, lm_neg_prop, fb_tone
    (null without ``classify``). Bodies are loaded ``chunk`` filings at a time."""
    meta = {r["item_id"]: r for r in f.select("item_id", "form", "sections").to_dicts()}
    ids = sorted(i for i, r in meta.items() if json.loads(r["sections"]))
    out = []
    for c0 in range(0, len(ids), chunk):
        part = ids[c0 : c0 + chunk]
        bodies = load_bodies(part)
        for i in part:
            m = meta[i]
            txt = mdna_text(bodies.get(i, ""), m["form"], json.loads(m["sections"]))
            if txt is None:
                continue
            lc = lm_counts(txt, lm)
            fb = finbert_score(txt, classify)["score"] if classify is not None else None
            out.append(
                {
                    "item_id": i,
                    "mda_words": lc["n_words"],
                    "lm_tone": lc["score"],
                    "lm_neg_prop": lc["neg_prop"],
                    "fb_tone": fb,
                }
            )
        log.info("mdna-tone: %d / %d filings", min(c0 + chunk, len(ids)), len(ids))
    return pl.DataFrame(out, schema=TONE_SCHEMA)


def tone_changes(f: pl.DataFrame, tones: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Every filing with an MD&A tone: item_id, tradable_at, cik, form, the levels and, where
    the year-earlier pair also has an MD&A, ``d_<measure>`` = level - the pair's level (null
    otherwise: no pair or no MD&A in it)."""
    prior = tones.select(
        pl.col("item_id").alias("prior_item_id"), *[pl.col(m).alias(f"prior_{m}") for m in MEASURES]
    )
    return (
        tones.join(f.select("item_id", "tradable_at", "cik", "form"), on="item_id")
        .join(pairs.select("item_id", "prior_item_id"), on="item_id", how="left")
        .join(prior, on="prior_item_id", how="left")
        .with_columns([(pl.col(m) - pl.col(f"prior_{m}")).alias(f"d_{m}") for m in MEASURES])
        .drop([f"prior_{m}" for m in MEASURES])
    )
