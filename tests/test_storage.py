from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyarrow.parquet as pq
import pytest

from newsfeed.dedupe import content_hash, normalize_url
from newsfeed.ingest.base import RawItem
from newsfeed.ingest.runner import build_row
from newsfeed.storage import Store


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def item(native_id: str, published: datetime, *, seen=None, mode="live", tickers=("AAPL",)):
    return RawItem(
        source="test",
        native_id=native_id,
        url=f"https://Example.com/a/{native_id}/?utm_source=x",
        title=f"t{native_id}",
        body_text=f"body {native_id}",
        body_tokens_est=2,
        body_truncated=False,
        tickers=list(tickers),
        published_at=published,
        published_at_raw=published.isoformat(),
        first_seen_at=seen or published,
        ingest_mode=mode,
        meta={},
        raw_json="{}",
        parser_version="test/1",
    )


def rows(items, calendar, run_id="run1", lag=60):
    return [
        build_row(
            i,
            run_id=run_id,
            calendar=calendar,
            backfill_lag=timedelta(seconds=lag),
            ingested_at=utc(2026, 10, 8),
        )
        for i in items
    ]


def test_rawitem_rejects_naive_timestamps():
    with pytest.raises(ValueError):
        item("x", datetime(2026, 7, 30, 15))


def test_write_is_idempotent(store: Store, calendar):
    batch = rows([item("1", utc(2026, 7, 30, 15)), item("2", utc(2026, 7, 30, 16))], calendar)
    assert store.write_items(batch, "run1") == 2
    assert store.write_items(batch, "run2") == 0
    # a fresh Store instance (new process) also sees them
    store2 = Store(store.paths)
    assert store2.write_items(batch, "run3") == 0
    assert store2.query("SELECT count(*) AS n FROM items")["n"][0] == 2
    store2.close()


def test_view_keeps_earliest_sighting_even_if_duplicate_files_exist(store: Store, calendar):
    early = rows([item("1", utc(2026, 7, 30, 15), seen=utc(2026, 7, 30, 15, 1))], calendar)
    late = rows([item("1", utc(2026, 7, 30, 15), seen=utc(2026, 7, 30, 15, 9))], calendar)
    store.write_items(late, "r1")
    store._known_ids = set()  # simulate a racing writer that bypassed the anti-join
    store.write_items(early, "r2")
    df = store.query("SELECT first_seen_at FROM items")
    assert df.height == 1
    assert df["first_seen_at"][0] == utc(2026, 7, 30, 15, 1)


def test_items_by_id_and_source_relation_match_the_view(store: Store, calendar):
    # duplicate sightings of item 1 (the view must keep the earliest), plus another source
    early = rows([item("1", utc(2026, 7, 30, 15), seen=utc(2026, 7, 30, 15, 1))], calendar)
    late = rows([item("1", utc(2026, 7, 30, 15), seen=utc(2026, 7, 30, 15, 9))], calendar)
    other = item("3", utc(2026, 7, 30, 17)).model_copy(update={"source": "other"})
    store.write_items(late + rows([item("2", utc(2026, 7, 30, 16)), other], calendar), "r1")
    store._known_ids = set()
    store.write_items(early, "r2")
    ids = store.query("SELECT item_id FROM items ORDER BY item_id")["item_id"].to_list()
    assert len(ids) == 3

    def view(where: str) -> list[dict]:
        return store.query(f"SELECT * FROM items WHERE {where} ORDER BY item_id").to_dicts()

    by_id = store.items_by_id(ids).sort("item_id").to_dicts()
    assert by_id == view("true")
    assert by_id[0]["first_seen_at"] == utc(2026, 7, 30, 15, 1)  # dedupe rule still applied
    only_test = store.items_by_id(ids, sources=["test"]).sort("item_id").to_dicts()
    assert only_test == view("source = 'test'") and len(only_test) == 2
    rel = store.query(f"SELECT * FROM {store.items_relation(['test'])} ORDER BY item_id")
    assert rel.to_dicts() == view("source = 'test'")
    assert store.items_by_id([]).height == 0


def test_timestamps_roundtrip_as_utc(store: Store, calendar):
    store.write_items(rows([item("1", utc(2026, 7, 30, 20, 30, 28))], calendar), "r")
    f = next(store.items_dir.rglob("*.parquet"))
    schema = pq.read_schema(f)
    for col in ("published_at", "first_seen_at", "tradable_at", "ingested_at"):
        assert str(schema.field(col).type) == "timestamp[us, tz=UTC]"
    df = store.query("SELECT published_at, tradable_at, tradable_rule FROM items")
    assert df["published_at"][0] == utc(2026, 7, 30, 20, 30, 28)
    assert df["tradable_at"][0] == utc(2026, 7, 31, 13, 30)
    assert df["tradable_rule"][0].startswith("live:")


def test_items_as_of_is_strictly_causal(store: Store, calendar):
    """The core PIT guarantee: nothing with tradable_at > T is ever returned for as-of T."""
    pubs = [utc(2026, 7, 27, 13, 0) + timedelta(hours=7 * k) for k in range(30)]
    its = [item(str(k), p, mode="live") for k, p in enumerate(pubs)]
    its += [item(f"b{k}", p, seen=utc(2030, 1, 1), mode="backfill") for k, p in enumerate(pubs)]
    store.write_items(rows(its, calendar), "r")
    all_rows = store.query("SELECT item_id, tradable_at FROM items")
    assert all_rows.height == 60

    # Probe exactly at, and 1us before, every distinct tradable_at: the boundaries where a
    # `<` vs `<=` or timezone slip would show up.
    eps = timedelta(microseconds=1)
    probes = sorted({t + d for t in all_rows["tradable_at"].to_list() for d in (-eps, 0 * eps)})
    for t in [utc(2026, 7, 26), *probes, utc(2026, 8, 12)]:
        got = store.items_as_of(t)
        assert (got["tradable_at"] <= t).all(), f"look-ahead at {t}"
        expected = int((all_rows["tradable_at"] <= t).sum())
        assert got.height == expected, f"missing items at {t}"
        # every item is also never visible before it was published
        assert (got["published_at"] <= t).all()


def test_backfilled_items_are_not_hidden_by_late_first_seen(store: Store, calendar):
    pub = utc(2026, 7, 30, 15)
    store.write_items(rows([item("b", pub, seen=utc(2030, 1, 1), mode="backfill")], calendar), "r")
    assert store.items_as_of(pub + timedelta(seconds=59)).height == 0
    assert store.items_as_of(pub + timedelta(seconds=60)).height == 1


def test_items_as_of_filters(store: Store, calendar):
    store.write_items(
        rows(
            [
                item("1", utc(2026, 7, 30, 15), tickers=("AAPL",)),
                item("2", utc(2026, 7, 30, 15), tickers=("MSFT", "GOOGL")),
            ],
            calendar,
        ),
        "r",
    )
    t = utc(2026, 8, 1)
    assert store.items_as_of(t, tickers=["GOOGL"])["native_id"].to_list() == ["2"]
    assert store.items_as_of(t, source="nope").height == 0
    with pytest.raises(ValueError):
        store.items_as_of(datetime(2026, 8, 1))


def test_empty_store_queries_work(store: Store):
    assert store.items_as_of(utc(2026, 1, 1)).height == 0
    assert store.known_ids() == set()


def test_raw_store_roundtrip_and_path_safety(store: Store):
    store.raw.put("edgar", "0000320193-26-000018", "a.htm", b"<html>x</html>")
    assert store.raw.get("edgar", "0000320193-26-000018", "a.htm") == b"<html>x</html>"
    assert store.raw.get("edgar", "nope", "a.htm") is None
    with pytest.raises(ValueError):
        store.raw.put("edgar", "..", "a.htm", b"")
    with pytest.raises(ValueError):
        store.raw.put("edgar", "x", "../../evil", b"")


def test_runs_table(store: Store, cfg):
    run_id = store.start_run("backfill", cfg, {"start": "2026-01-01"})
    store.finish_run(run_id, "ok", {"written": 3})
    df = store.ops_query("SELECT status, config_hash, stats FROM runs")
    assert df["status"][0] == "ok"
    assert df["config_hash"][0] == cfg.config_hash()


def test_normalize_url():
    assert normalize_url("HTTPS://www.Example.com:443/a/b/?utm_source=x&b=2&a=1#frag") == (
        "https://example.com/a/b?a=1&b=2"
    )
    assert normalize_url(None) is None


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Apple  reports\nRESULTS") == content_hash("apple reports results")
    assert content_hash("a") != content_hash("b")
