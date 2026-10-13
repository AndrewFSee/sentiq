"""Story clustering: correctness of each matching rule, and point-in-time properties."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from newsfeed.config import DedupeConfig
from newsfeed.dedupe import StoryAssigner, _Node, make_minhash, rebuild_stories, stories_as_of
from newsfeed.derive import relink
from newsfeed.ingest.base import RawItem
from newsfeed.ingest.runner import build_row
from newsfeed.storage import Store
from newsfeed.tickers import TickerLinker, Universe

CFG = DedupeConfig()
T0 = datetime(2026, 7, 28, 14, 0, tzinfo=UTC)  # Tue 10:00 EDT, in session

WORDS = [
    "revenue",
    "guidance",
    "margin",
    "quarter",
    "growth",
    "services",
    "demand",
    "outlook",
    "capital",
    "board",
    "dividend",
    "repurchase",
    "supply",
    "chain",
    "inventory",
    "pricing",
    "customers",
    "analysts",
    "estimates",
    "shares",
    "market",
    "segment",
    "operating",
    "cash",
    "flow",
    "acquisition",
    "regulatory",
    "approval",
    "product",
    "launch",
    "cloud",
    "chips",
]


def text(seed: int, n: int = 80) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def node(item_id: str, minutes: float, body: str, *, url=None, h=None, tickers=("AAPL",)):
    t = T0 + timedelta(minutes=minutes)
    return _Node(
        key=(t, t, item_id),
        item_id=item_id,
        story_id="",
        url_norm=url,
        content_hash=h,
        tickers=frozenset(tickers),
        mh=make_minhash(body, CFG),
    )


def edit(body: str, k: int) -> str:
    w = body.split()
    for i in range(k):
        w[i * 7 % len(w)] = "EDITED"
    return " ".join(w)


# ------------------------------------------------------------------ assigner rules
def test_new_story_founded_by_first_item():
    s = StoryAssigner(CFG)
    a = s.add(node("a", 0, text(1)))
    assert (a.story_id, a.reason) == ("a", "new")


def test_url_match_any_distance():
    s = StoryAssigner(CFG)
    s.add(node("a", 0, text(1), url="https://x.test/1"))
    b = s.add(node("b", 60 * 24 * 30, text(2), url="https://x.test/1"))  # 30 days later
    assert (b.story_id, b.reason) == ("a", "url")


def test_exact_hash_only_within_window():
    s = StoryAssigner(CFG)
    s.add(node("a", 0, text(1), h="H"))
    b = s.add(node("b", 60, text(1), h="H"))
    c = s.add(node("c", 60 * 24 * 10, text(1), h="H"))  # 10 days: recurring boilerplate
    assert (b.story_id, b.reason) == ("a", "exact")
    assert (c.story_id, c.reason) == ("c", "new")


def test_near_duplicate_rewrite_joins_but_different_text_does_not():
    s = StoryAssigner(CFG)
    base = text(1)
    s.add(node("a", 0, base))
    b = s.add(node("b", 5, edit(base, 1)))
    c = s.add(node("c", 10, text(2)))
    assert b.reason == "near" and b.story_id == "a" and b.similarity >= CFG.near_dup_threshold
    assert c.reason == "new"


def test_ticker_overlap_required():
    s = StoryAssigner(CFG)
    s.add(node("a", 0, text(1), h="H", tickers=("AAPL",)))
    b = s.add(node("b", 5, text(1), h="H", tickers=("MSFT",)))
    c = s.add(node("c", 6, text(1), h="H", tickers=()))  # untagged: allowed
    assert b.reason == "new"
    assert c.reason == "exact"


def test_no_retroactive_merge():
    """A and C are dissimilar; B resembles both. B must not merge A and C after the fact."""
    rng = random.Random(7)
    left = [rng.choice(WORDS) for _ in range(60)]
    right = [rng.choice(WORDS) for _ in range(60)]
    a_txt = " ".join(left + right[:10])
    c_txt = " ".join(left[:10] + right)
    b_txt = " ".join(left + right)
    s = StoryAssigner(DedupeConfig(near_dup_threshold=0.4))  # b~a 0.61, b~c 0.50, a~c 0.12
    a = s.add(node("a", 0, a_txt))
    c = s.add(node("c", 1, c_txt))
    b = s.add(node("b", 2, b_txt))
    assert c.story_id == "c"  # unchanged by B's later arrival
    assert b.story_id in {"a", "c"}
    assert a.story_id == "a"


def test_ties_go_to_earliest_candidate():
    s = StoryAssigner(CFG)
    base = text(3)
    s.add(node("a", 0, base, tickers=("AAPL",)))
    s.add(node("z", 1, text(4), tickers=("AAPL",)))
    s.add(node("b", 2, base, tickers=("MSFT",)))  # same text, separate story (no overlap)
    d = s.add(node("d", 3, base, tickers=()))  # matches a and b equally
    assert d.story_id == "a"


def test_out_of_order_add_rejected():
    s = StoryAssigner(CFG)
    s.add(node("b", 5, text(1)))
    with pytest.raises(ValueError, match="sort-key order"):
        s.add(node("a", 0, text(2)))


# ------------------------------------------------------------------ store-level rebuild
UNIVERSE = Universe({320193: ["AAPL"], 789019: ["MSFT"]})


def raw(nid: str, pub: datetime, body: str, *, publisher="P1", url=None, tickers=("AAPL",)):
    return RawItem(
        source="test",
        native_id=nid,
        publisher=publisher,
        url=url,
        title=f"t{nid}",
        body_text=body,
        body_tokens_est=1,
        body_truncated=False,
        tickers=list(tickers),
        published_at=pub,
        published_at_raw=pub.isoformat(),
        first_seen_at=pub,
        ingest_mode="live",
        meta={},
        raw_json="{}",
        parser_version="t",
    )


def write(store: Store, calendar, items):
    rows = [
        build_row(i, run_id="r", calendar=calendar, backfill_lag=timedelta(0), ingested_at=T0)
        for i in items
    ]
    store.write_items(rows, "r")


def derive(store: Store, full=False):
    linker = TickerLinker(UNIVERSE)
    relink(store, linker, "r")
    return rebuild_stories(store, CFG, linker.linker_id, full=full)


def assignments(store: Store) -> dict[str, str]:
    df = store.query(
        "SELECT i.native_id, s2.native_id AS founder FROM item_story s "
        "JOIN items i USING (item_id) JOIN items s2 ON s2.item_id = s.story_id"
    )
    return dict(zip(df["native_id"].to_list(), df["founder"].to_list(), strict=True))


@pytest.fixture
def corpus():
    base = text(11)
    m = timedelta(minutes=1)
    return [
        raw("a1", T0, base, publisher="Wire"),
        raw("a2", T0 + 5 * m, base, publisher="Biz"),  # exact syndication
        raw("a3", T0 + 9 * m, edit(base, 1), publisher="Daily"),  # near rewrite
        raw("b1", T0 + 20 * m, text(12), publisher="Wire"),  # different story
        raw("m1", T0 + 30 * m, base, publisher="Wire", tickers=("MSFT",)),  # other company
    ]


def test_rebuild_groups_corpus(store, calendar, corpus):
    write(store, calendar, corpus)
    stats = derive(store)
    assert stats["assigned"] == 5
    assert assignments(store) == {"a1": "a1", "a2": "a1", "a3": "a1", "b1": "b1", "m1": "m1"}


def test_incremental_equals_full_even_out_of_order(store, calendar, corpus, cfg, tmp_path):
    # ingest in a scrambled order with derives in between (live + late backfill)
    write(store, calendar, [corpus[3], corpus[2]])
    derive(store)
    write(store, calendar, [corpus[4]])
    derive(store)
    write(store, calendar, [corpus[0], corpus[1]])  # the founding article arrives last
    derive(store)
    incremental = assignments(store)
    derive(store, full=True)
    assert assignments(store) == incremental
    assert incremental["a3"] == "a1"  # re-assigned once the earlier article showed up


def test_rebuild_is_noop_when_nothing_new(store, calendar, corpus):
    write(store, calendar, corpus)
    derive(store)
    assert derive(store)["assigned"] == 0


def test_stories_as_of_counts_only_visible_members(store, calendar, corpus):
    write(store, calendar, corpus)
    derive(store)
    m = timedelta(minutes=1)

    def story(t):
        df = stories_as_of(store, t, tickers=["AAPL"])
        return {r["founder_item_id"]: r for r in df.to_dicts()}

    s = story(T0 + 6 * m)  # a1, a2 visible
    (st,) = s.values()
    assert (st["n_items"], st["n_outlets"]) == (2, 2)
    s = story(T0 + 10 * m)  # + a3
    (st,) = s.values()
    assert (st["n_items"], st["n_outlets"], st["last_tradable_at"]) == (3, 3, T0 + 9 * m)
    assert len(story(T0 + 25 * m)) == 2  # + b1 story
    assert story(T0 - m) == {}


def test_version_change_forces_full_rebuild(store, calendar, corpus):
    write(store, calendar, corpus)
    derive(store)
    linker = TickerLinker(UNIVERSE)
    other = CFG.model_copy(update={"near_dup_threshold": 0.95})
    stats = rebuild_stories(store, other, linker.linker_id)
    assert stats["assigned"] == 5


def test_relink_streams_every_batch(store, calendar, corpus):
    # regression: writing links on the reading connection ended the stream after batch 1
    write(store, calendar, corpus)
    linker = TickerLinker(UNIVERSE)
    assert relink(store, linker, "r", batch_size=2) == 5
    linked = store.query("SELECT count(DISTINCT item_id) AS n FROM item_tickers_all")["n"][0]
    assert linked == 5 and relink(store, linker, "r", batch_size=2) == 0
