"""Keep a local copy of one file of a Hugging Face dataset current.

Layout under ``<dir>``: content-addressed copies ``<sha256[:12]>.parquet`` (the LFS
``x-linked-etag``) plus a ``CURRENT`` pointer ``{"commit", "etag", "file"}``. A repo commit
that leaves this file unchanged (other tables updated) only moves the pointer: nothing is
downloaded. Otherwise the file is streamed to ``<sha256[:12]>.parquet.part`` and renamed into
place, so a reader never sees a partial file. The pinned version a backfill used is never pruned
(``keep_files``). Legacy layout (``COMMIT`` + ``<commit[:12]>.parquet``) is still read.

Slow links (found 2026-10-21: ~0.45 MB/s, and the server closes a 2.3 GB response after
~110 MB): a dropped connection resumes from the ``.part`` file with an HTTP Range request, and
``max_seconds`` caps one call so a collect cycle isn't blocked for an hour; the next call
continues where this one stopped (``pending_download``). A finished file is checked against the
LFS sha256 before it is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets"


def _pointer(directory: Path) -> dict | None:
    p = directory / "CURRENT"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def current_file(directory: Path, fallback_commit: str) -> tuple[Path, str]:
    """(file, commit) of the current local version; the pinned commit before any refresh."""
    ptr = _pointer(directory)
    if ptr is not None:
        return directory / ptr["file"], ptr["commit"]
    legacy = directory / "COMMIT"
    commit = legacy.read_text(encoding="utf-8").strip() if legacy.exists() else fallback_commit
    return directory / f"{commit[:12]}.parquet", commit


def remote_version(client: httpx.Client, repo: str, file: str) -> tuple[str, str]:
    """(commit, content sha256) of the file on the repo's main branch (one HEAD request)."""
    r = client.head(f"{HF}/{repo}/resolve/main/{file}", follow_redirects=False)
    commit = (r.headers.get("x-repo-commit") or "").strip()
    etag = (r.headers.get("x-linked-etag") or r.headers.get("etag") or "").strip().strip('"')
    if not commit or not etag:
        raise RuntimeError(f"no commit/etag headers for {repo}/{file} (HTTP {r.status_code})")
    return commit, etag


def _replace(src: Path, dst: Path, attempts: int = 10) -> None:
    for i in range(attempts):  # Windows: the target may be briefly open in a reader
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def _write_pointer(directory: Path, ptr: dict) -> None:
    tmp = directory / "CURRENT.part"
    tmp.write_text(json.dumps(ptr, sort_keys=True), encoding="utf-8")
    _replace(tmp, directory / "CURRENT")


def pending_download(directory: Path) -> bool:
    """True if a partial download is waiting to be resumed."""
    return directory.is_dir() and any(directory.glob("*.parquet.part"))


def _total_size(r: httpx.Response, offset: int) -> int | None:
    """Full file size from a 206 Content-Range or a 200 Content-Length."""
    cr = r.headers.get("content-range", "")
    if "/" in cr and not cr.endswith("/*"):
        return int(cr.rsplit("/", 1)[1])
    cl = r.headers.get("content-length")
    return int(cl) + offset if cl is not None else None


def _download(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    deadline: float | None,
    clock: Callable[[], float],
    max_resumes: int = 50,
) -> bool:
    """Append to ``part`` until complete (True) or the deadline passes (False, resumable)."""
    total: int | None = None
    for _ in range(max_resumes):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            return True
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416:  # nothing left after `have`: complete
                    return True
                r.raise_for_status()
                if have and r.status_code == 200:  # server ignored the range: start over
                    have = 0
                total = _total_size(r, have)
                with part.open("ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        if deadline is not None and clock() > deadline:
                            return False
            if total is None or part.stat().st_size >= total:
                return True
        except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
            log.info("download of %s interrupted at %d bytes (%s); resuming", url, have, exc)
        if deadline is not None and clock() > deadline:
            return False
    raise RuntimeError(f"download of {url} did not finish after {max_resumes} resumes")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def refresh(
    directory: Path,
    repo: str,
    file: str,
    *,
    fallback_commit: str,
    keep_files: Iterable[str] = (),
    client: httpx.Client | None = None,
    max_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Bring the local copy to the repo's current version of ``file``. With ``max_seconds``,
    a download that doesn't finish in time returns ``{"partial": True, ...}`` and resumes on
    the next call; the current version stays in use meanwhile."""
    directory.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)
    deadline = clock() + max_seconds if max_seconds is not None else None
    try:
        _, local_commit = current_file(directory, fallback_commit)
        ptr = _pointer(directory) or {}
        commit, etag = remote_version(client, repo, file)
        if etag == ptr.get("etag"):
            if commit != local_commit:  # repo moved, this file did not
                _write_pointer(directory, {**ptr, "commit": commit})
            for stale in directory.glob("*.parquet.part"):
                stale.unlink(missing_ok=True)
            return {"changed": False, "commit": commit, "etag": etag}
        target = directory / f"{etag[:12]}.parquet"
        if not target.exists():
            part = target.with_suffix(".parquet.part")
            for stale in directory.glob("*.parquet.part"):  # a superseded version's download
                if stale != part:
                    stale.unlink(missing_ok=True)
            url = f"{HF}/{repo}/resolve/{commit}/{file}"
            if not _download(client, url, part, deadline=deadline, clock=clock):
                size = part.stat().st_size if part.exists() else 0
                log.info("transcripts download paused at %d bytes; resumes next poll", size)
                return {"changed": False, "partial": True, "bytes": size, "etag": etag}
            if len(etag) == 64 and _sha256(part) != etag:  # LFS oid = sha256 of the content
                part.unlink()
                raise RuntimeError(f"downloaded {file} does not match sha256 {etag[:12]}")
            _replace(part, target)
        _write_pointer(directory, {"commit": commit, "etag": etag, "file": target.name})
        keep = {*keep_files, target.name}
        for old in directory.glob("*.parquet"):
            if old.name not in keep:
                try:
                    old.unlink()
                except PermissionError:  # still open somewhere: prune next time
                    log.info("could not prune %s yet", old)
        log.warning("transcripts dataset updated %s -> %s", local_commit[:12], commit[:12])
        return {"changed": True, "commit": commit, "etag": etag, "previous": local_commit}
    finally:
        if own:
            client.close()
