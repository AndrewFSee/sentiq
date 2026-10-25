"""Content-addressed refresh of a Hugging Face dataset file (transcripts)."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator

import httpx
import pytest

from newsfeed.ingest.hf_refresh import current_file, pending_download, refresh

PIN = "a" * 40
PIN_FILE = f"{PIN[:12]}.parquet"
MB = 1 << 20


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Dropping(httpx.SyncByteStream):
    """Yields ``data`` in 1 MB pieces, then drops the connection after ``drop_after`` bytes."""

    def __init__(self, data: bytes, drop_after: int | None = None) -> None:
        self.data, self.drop_after = data, drop_after

    def __iter__(self) -> Iterator[bytes]:
        for k in range(0, len(self.data), MB):
            if self.drop_after is not None and k >= self.drop_after:
                raise httpx.RemoteProtocolError("peer closed connection")
            yield self.data[k : k + MB]


def ranged_server(
    body: bytes, etag: str, *, drops: list[int], honor_range: bool = True, log: list | None = None
) -> httpx.Client:
    """HEAD gives version headers; GET honours Range (206) and drops after drops.pop(0) bytes."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "HEAD":
            return httpx.Response(302, headers={"x-repo-commit": "d" * 40, "x-linked-etag": etag})
        rng = req.headers.get("range")
        if log is not None:
            log.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and honor_range else 0
        rest = body[start:]
        drop = drops.pop(0) if drops else None
        if start:
            headers = {"content-range": f"bytes {start}-{len(body) - 1}/{len(body)}"}
            return httpx.Response(206, headers=headers, stream=Dropping(rest, drop))
        headers = {"content-length": str(len(body))}
        return httpx.Response(200, headers=headers, stream=Dropping(rest, drop))

    return httpx.Client(transport=httpx.MockTransport(handler))


def refresh_with(tmp_path, client, **kw):
    return refresh(tmp_path, "o/r", "f.parquet", fallback_commit=PIN, client=client, **kw)


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
    e1 = sha(b"v1")
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

    e2 = sha(b"v2")
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


def test_dropped_connection_resumes_with_a_range_request(tmp_path):
    body = bytes(range(256)) * (5 * MB // 256)  # 5 MB
    sent: list = []
    client = ranged_server(body, sha(body), drops=[2 * MB, 2 * MB], log=sent)
    out = refresh_with(tmp_path, client)
    assert out["changed"]
    assert sent == [None, f"bytes={2 * MB}-", f"bytes={4 * MB}-"]
    path, _ = current_file(tmp_path, PIN)
    assert path.read_bytes() == body and not pending_download(tmp_path)


def test_time_cap_leaves_a_resumable_partial_file_and_the_old_version_in_use(tmp_path):
    (tmp_path / PIN_FILE).write_bytes(b"pinned")
    body = b"x" * (4 * MB)
    ticks = iter(range(1000))
    clock = lambda: float(next(ticks))  # noqa: E731 - one tick per chunk
    out = refresh_with(tmp_path, ranged_server(body, sha(body), drops=[]), max_seconds=1.5,
                       clock=clock)  # fmt: skip
    assert out["partial"] and 0 < out["bytes"] < len(body)
    assert pending_download(tmp_path)
    assert current_file(tmp_path, PIN) == (tmp_path / PIN_FILE, PIN)  # old version still used
    sent: list = []
    out = refresh_with(tmp_path, ranged_server(body, sha(body), drops=[], log=sent))
    assert out["changed"] and sent == [f"bytes={2 * MB}-"]
    assert current_file(tmp_path, PIN)[0].read_bytes() == body


def test_server_that_ignores_range_restarts_from_zero(tmp_path):
    body = b"y" * (3 * MB)
    client = ranged_server(body, sha(body), drops=[MB], honor_range=False)
    assert refresh_with(tmp_path, client)["changed"]
    assert current_file(tmp_path, PIN)[0].read_bytes() == body


def test_corrupt_download_is_rejected_and_discarded(tmp_path):
    (tmp_path / PIN_FILE).write_bytes(b"pinned")
    body = b"z" * MB
    with pytest.raises(RuntimeError, match="sha256"):
        refresh_with(tmp_path, ranged_server(body, sha(b"other"), drops=[]))
    assert not pending_download(tmp_path) and not (tmp_path / "CURRENT").exists()
    assert current_file(tmp_path, PIN) == (tmp_path / PIN_FILE, PIN)


def test_partial_file_of_a_superseded_version_is_discarded(tmp_path):
    (tmp_path / "000000000000.parquet.part").write_bytes(b"old version, half done")
    body = b"new"
    assert refresh_with(tmp_path, ranged_server(body, sha(body), drops=[]))["changed"]
    assert not pending_download(tmp_path)
