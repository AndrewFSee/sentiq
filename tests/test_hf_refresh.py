"""Content-addressed refresh of a Hugging Face dataset file (transcripts)."""

from __future__ import annotations

import contextlib
import json

import httpx

from newsfeed.ingest.hf_refresh import current_file, refresh

PIN = "a" * 40
PIN_FILE = f"{PIN[:12]}.parquet"


def client(commit: str, etag: str, body: bytes, calls: list[str]) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(f"{req.method} {req.url.path}")
        if req.method == "HEAD":
            return httpx.Response(
                302, headers={"x-repo-commit": commit, "x-linked-etag": f'"{etag}"'}
            )
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def run(tmp_path, commit, etag, body, calls):
    return refresh(
        tmp_path,
        "o/r",
        "f.parquet",
        fallback_commit=PIN,
        keep_files=[PIN_FILE],
        client=client(commit, etag, body, calls),
    )


def test_new_content_downloads_new_commit_same_content_only_moves_pointer(tmp_path):
    (tmp_path / PIN_FILE).write_bytes(b"pinned")
    assert current_file(tmp_path, PIN) == (tmp_path / PIN_FILE, PIN)  # before any refresh
    calls: list[str] = []
    e1 = "1" * 64
    out = run(tmp_path, "b" * 40, e1, b"v1", calls)
    assert out["changed"] and calls[-1] == f"GET /datasets/o/r/resolve/{'b' * 40}/f.parquet"
    path, commit = current_file(tmp_path, PIN)
    assert path.name == f"{e1[:12]}.parquet" and path.read_bytes() == b"v1"
    assert commit == "b" * 40 and (tmp_path / PIN_FILE).exists()  # the pin is never pruned

    n = len(calls)
    out = run(tmp_path, "c" * 40, e1, b"never fetched", calls)  # repo moved, file did not
    assert not out["changed"] and len(calls) == n + 1  # one HEAD, no GET
    assert current_file(tmp_path, PIN) == (path, "c" * 40)
    assert json.loads((tmp_path / "CURRENT").read_text())["etag"] == e1

    e2 = "2" * 64
    run(tmp_path, "d" * 40, e2, b"v2", calls)
    names = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert names == sorted([PIN_FILE, f"{e2[:12]}.parquet"])  # v1 pruned, pin kept
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_leaves_current_version_untouched(tmp_path):
    (tmp_path / PIN_FILE).write_bytes(b"pinned")

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "HEAD":
            headers = {"x-repo-commit": "d" * 40, "x-linked-etag": "e" * 64}
            return httpx.Response(302, headers=headers)
        return httpx.Response(500)

    with contextlib.suppress(httpx.HTTPStatusError):
        refresh(
            tmp_path,
            "o/r",
            "f.parquet",
            fallback_commit=PIN,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert current_file(tmp_path, PIN) == (tmp_path / PIN_FILE, PIN)
    assert not (tmp_path / "CURRENT").exists()
