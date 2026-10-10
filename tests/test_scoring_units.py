"""Schema, prompt, request construction, response parsing, cost, masking, agreement metrics."""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

from newsfeed.config import PriceCard, PricingConfig
from newsfeed.scoring.agreement import metrics
from newsfeed.scoring.client import build_params, call_cost, parse_message
from newsfeed.scoring.mask import MaskLexicon
from newsfeed.scoring.prompt import GenerationSettings, load_prompt
from newsfeed.scoring.schema import OUTPUT_JSON_SCHEMA, parse_output
from newsfeed.tickers import Universe

from .fake_llm import default_output, make_message

SETTINGS = GenerationSettings("low", "adaptive", 2048)
GOOD = {
    "relevance": "primary",
    "sentiment": 0.4,
    "confidence": 0.8,
    "event_type": "earnings",
    "novelty": None,
    "materiality": "high",
    "surprise_vs_expectations": None,
    "rationale": "Revenue rose and guidance was raised.",
}


# ------------------------------------------------------------------ schema
def test_json_schema_is_strict_and_complete():
    s = OUTPUT_JSON_SCHEMA
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])
    for nullable in ("novelty", "materiality", "surprise_vs_expectations"):
        assert {"type": "null"} in s["properties"][nullable]["anyOf"]
    assert "m&a" in s["properties"]["event_type"]["enum"]


def test_parse_output_validates_ranges_without_clamping():
    assert parse_output(json.dumps(GOOD))[0].sentiment == 0.4
    for bad in ({"sentiment": 1.5}, {"confidence": -0.1}, {"surprise_vs_expectations": -2}):
        out, err = parse_output(json.dumps({**GOOD, **bad}))
        assert out is None and "outside" in err
    assert parse_output("{not json")[1].startswith("json")
    assert parse_output(json.dumps({**GOOD, "extra": 1}))[0] is None


def test_long_rationale_is_kept_but_counted():
    out, _ = parse_output(json.dumps({**GOOD, "rationale": "word " * 30}))
    assert out is not None and out.rationale_words == 30


# ------------------------------------------------------------------ prompt
def test_prompt_loads_and_hash_covers_everything():
    p = load_prompt("score_v1")
    assert "Do not use outside knowledge" in p.system
    assert len(p.system) > 2000  # comfortably above the 512-token cacheable minimum
    h = p.hash(SETTINGS)
    assert h == p.hash(GenerationSettings("low", "adaptive", 2048))
    assert h != p.hash(GenerationSettings("medium", "adaptive", 2048))
    assert h != p.hash(GenerationSettings("low", "adaptive", 4096))
    altered = type(p)(p.version, p.system + " ", p.user_template)
    assert altered.hash(SETTINGS) != h


def test_render_user_has_no_date_and_flags_truncation():
    p = load_prompt("score_v1")
    u = p.render_user(
        target="Apple Inc. (ticker: AAPL)",
        doc_type="news article",
        publisher="Wire",
        title="T",
        body="{braces} ok",
        truncated=True,
    )
    assert "TARGET: Apple Inc. (ticker: AAPL)" in u
    assert "{braces} ok" in u and "truncated" in u
    assert "2026" not in u  # publication date is deliberately not sent


# ------------------------------------------------------------------ requests
def test_build_params_shape():
    p = build_params(model="claude-haiku-5-5", system="S", user="U", settings=SETTINGS)
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert p["output_config"]["format"]["schema"] is OUTPUT_JSON_SCHEMA
    assert p["output_config"]["effort"] == "low"
    assert p["thinking"] == {"type": "adaptive"}
    for banned in ("temperature", "top_p", "top_k", "tools", "tool_choice"):
        assert banned not in p


def test_disabled_thinking_rejected_for_sonnet_5_5():
    off = GenerationSettings("low", "disabled", 2048)
    build_params(model="claude-haiku-5-5", system="S", user="U", settings=off)  # ok
    with pytest.raises(ValueError, match="rejects"):
        build_params(model="claude-sonnet-5-5", system="S", user="U", settings=off)


# ------------------------------------------------------------------ parsing
def test_parse_message_reads_blocks_by_type():
    o = parse_message(make_message(GOOD, thinking_first=True))
    assert o.status == "ok" and o.output.sentiment == 0.4
    assert o.usage["cache_read_input_tokens"] == 2000
    assert json.loads(o.raw)["content"][0]["type"] == "thinking"


@pytest.mark.parametrize(
    "msg,status",
    [
        (make_message(None, stop_reason="refusal"), "refusal"),
        (make_message('{"relevance": "prim', stop_reason="max_tokens"), "max_tokens"),
        (make_message(None), "empty"),
        (make_message({**GOOD, "sentiment": 3}), "invalid"),
    ],
)
def test_parse_message_statuses(msg, status):
    assert parse_message(msg).status == status


# ------------------------------------------------------------------ cost
def test_call_cost_rate_cards_and_batch(cfg):
    u = {
        "input_tokens": 50_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 50_000,
    }
    assert call_cost(cfg.pricing, "claude-haiku-5-5", u, batch=False) == pytest.approx(0.03)
    assert call_cost(cfg.pricing, "claude-haiku-5-5", u, batch=True) == pytest.approx(0.015)
    small = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 3000,
        "output_tokens": 100,
    }
    exp = (1000 * 0.10 + 2000 * 0.125 + 3000 * 0.01 + 100 * 0.50) / 1e6
    assert call_cost(cfg.pricing, "claude-haiku-5-5", small, batch=False) == pytest.approx(exp)
    # > 100k-token prompt (cache reads count toward it) -> long-context card for the whole call
    big = {**small, "cache_read_input_tokens": 99_000}
    exp_big = (1000 * 0.50 + 2000 * 0.625 + 99_000 * 0.05 + 100 * 2.50) / 1e6
    assert call_cost(cfg.pricing, "claude-haiku-5-5", big, batch=False) == pytest.approx(exp_big)
    assert math.isnan(call_cost(cfg.pricing, "unknown-model", small, batch=False))


def test_pricing_config_sonnet():
    card = PriceCard(input=2, output=10, cache_write_5m=2.5, cache_read=0.1)
    pc = PricingConfig.model_validate({"as_of": "x", "models": {"m": {"standard": card}}})
    u = {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 0,
    }
    assert call_cost(pc, "m", u, batch=False) == pytest.approx(0.10)


# ------------------------------------------------------------------ masking
UNIVERSE = Universe(
    {320193: ["AAPL"], 789019: ["MSFT"], 1652044: ["GOOG", "GOOGL"], 1067983: ["BRK-B"]}
)


@pytest.fixture
def lexicon(cfg):
    ucfg = cfg.universe.model_copy(
        update={
            "aliases": {
                "AAPL": ["Apple"],
                "MSFT": ["Microsoft"],
                "GOOGL": ["Alphabet", "Google"],
                "BRK-B": ["Berkshire Hathaway"],
            },
            "products": {"AAPL": ["iPhone", "Apple Watch"], "MSFT": ["Azure"]},
            "people": {"AAPL": ["Tim Cook"], "BRK-B": ["Warren Buffett", "Buffett"]},
            "places": {"AAPL": ["Cupertino"]},
        }
    )
    titles = {320193: "Apple Inc.", 789019: "MICROSOFT CORP"}
    return MaskLexicon.build(ucfg, UNIVERSE, titles)


def test_mask_replaces_all_identifiers(lexicon):
    title = "Apple (NASDAQ: AAPL) beats; Microsoft's Azure grows"
    body = (
        "APPLE INC. said iPhone and Apple Watch sales rose, CEO Tim Cook told staff in "
        "Cupertino. Shares of $AAPL and MSFT moved; Google and Berkshire Hathaway "
        "(Warren Buffett) were mentioned. Apple's outlook improved."
    )
    m = lexicon.mask(title, body, target_cik=789019)  # target = Microsoft
    assert m.target_label == "[COMPANY_A] (ticker: [TICKER_A])"
    assert "[COMPANY_A]'s [PRODUCT_1]" in m.title  # Microsoft is A even though Apple is first
    assert m.title.startswith("[COMPANY_B] (NASDAQ: [TICKER_B])")
    assert "[PRODUCT_2] and [PRODUCT_3]" in m.body  # iPhone, Apple Watch (longest match wins)
    assert "[PERSON_1]" in m.body and "[PLACE_1]" in m.body
    assert "$[TICKER_B]" in m.body and "[TICKER_A] moved" in m.body
    assert "[COMPANY_B]'s outlook" in m.body
    for leaked in (
        "Apple",
        "APPLE",
        "Microsoft",
        "iPhone",
        "Cook",
        "Cupertino",
        "Buffett",
        "Google",
        "Berkshire",
        "AAPL",
        "MSFT",
    ):
        assert leaked not in m.title + m.body, leaked
    assert m.residuals == 0


def test_mask_is_deterministic_and_counts_residuals(lexicon):
    a = lexicon.mask("Apple", "Apple and Microsoft", 320193)
    b = lexicon.mask("Apple", "Apple and Microsoft", 320193)
    assert a == b and a.body == "[COMPANY_A] and [COMPANY_B]"
    assert lexicon.residuals("an apple a day; microsoft") == 2  # case-insensitive diagnostic


def test_mask_unknown_ticker_in_config_raises(cfg):
    bad = cfg.universe.model_copy(update={"products": {"ZZZZ": ["Widget"]}})
    with pytest.raises(KeyError):
        MaskLexicon.build(bad, UNIVERSE, {})


# ------------------------------------------------------------------ agreement
def _pairs(sa, sb, ra=None, rb=None):
    n = len(sa)
    ra = ra or ["primary"] * n
    rb = rb or ra
    base = {
        "confidence": [0.5] * n,
        "event_type": ["other"] * n,
        "novelty": [None] * n,
        "materiality": ["low"] * n,
        "surprise_vs_expectations": [None] * n,
    }
    cols = {"item_id": [str(i) for i in range(n)], "ticker": ["AAPL"] * n}
    for side, s, r in (("a", sa, ra), ("b", sb, rb)):
        cols[f"sentiment_{side}"] = s
        cols[f"relevance_{side}"] = r
        for k, v in base.items():
            cols[f"{k}_{side}"] = v
    return pl.DataFrame(cols)


def test_agreement_identical_is_perfect():
    m = metrics(
        _pairs(
            [0.5, -0.2, 0.0, 0.9],
            [0.5, -0.2, 0.0, 0.9],
            ["primary", "passing", "secondary", "primary"],
        )
    )
    assert m["sentiment_pearson"] == pytest.approx(1.0)
    assert m["sentiment_mae"] == 0 and m["sentiment_sign_agreement"] == 1.0
    assert m["relevance_kappa"] == pytest.approx(1.0)
    assert m["surprise_null_agreement"] == 1.0


def test_agreement_detects_bias_and_disagreement():
    m = metrics(
        _pairs(
            [0.6, 0.4, 0.8, 0.5],
            [0.1, -0.1, 0.3, 0.0],
            ["primary"] * 4,
            ["primary", "passing", "primary", "passing"],
        )
    )
    assert m["sentiment_mean_diff_a_minus_b"] == pytest.approx(0.5)
    assert m["sentiment_pearson"] == pytest.approx(1.0)  # perfectly correlated, shifted
    assert m["relevance_agreement"] == 0.5
    assert m["sentiment_sign_agreement"] == 0.25  # only 0.3 clears the +-0.1 neutral band


def test_default_fake_output_is_schema_valid():
    assert parse_output(json.dumps(default_output("x results")))[0] is not None


def test_cache_key_covers_text_and_mask_terms(lexicon, cfg):
    from newsfeed.scoring.targets import Target

    base = Target("i", "AAPL", 1, "m", "v1", "ph", text_hash="t1")
    assert base.cache_key != Target("i", "AAPL", 1, "m", "v1", "ph", text_hash="t2").cache_key
    assert base.cache_key != Target("i", "AAPL", 1, "m", "v1", "ph", "t1", replicate=1).cache_key
    m1 = Target("i", "AAPL", 1, "m", "v1", "ph", "t1", masked=True, mask_id="a")
    m2 = Target("i", "AAPL", 1, "m", "v1", "ph", "t1", masked=True, mask_id="b")
    assert m1.cache_key != m2.cache_key != base.cache_key
    # mask_id ignored when unmasked
    assert base.cache_key == Target("i", "AAPL", 1, "m", "v1", "ph", "t1", mask_id="zz").cache_key
    more = MaskLexicon.build(
        cfg.universe.model_copy(
            update={
                "aliases": {"AAPL": ["Apple", "Apple Computer"]},
                "products": {},
                "people": {},
                "places": {},
            }
        ),
        UNIVERSE,
        {},
    )
    assert more.mask_id != lexicon.mask_id
    assert len(base.custom_id) == 48
