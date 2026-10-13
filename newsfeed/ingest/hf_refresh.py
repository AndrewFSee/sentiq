"""Keep a local copy of one file of a Hugging Face dataset current.

Layout under ``<dir>``: content-addressed copies ``<sha256[:12]>.parquet`` (the LFS
``x-linked-etag``) plus a ``CURRENT`` pointer ``{"commit", "etag", "file"}``. A repo commit
that leaves this file unchanged (other tables updated) only moves the pointer: nothing is
downloaded. Otherwise the file is streamed to ``*.part`` and renamed into place, so a reader
never sees a partial file. The pinned version a backfill used is never pruned (``keep_files``).
Legacy layout (``COMMIT`` + ``<commit[:12]>.parquet``) is still read.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
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


def refresh(
    directory: Path,
    repo: str,
    file: str,
    *,
    fallback_commit: str,
    keep_files: Iterable[str] = (),
    client: httpx.Client | None = None,
) -> dict:
    """Bring the local copy to the repo's current version of ``file``."""
    directory.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)
    try:
        _, local_commit = current_file(directory, fallback_commit)
        ptr = _pointer(directory) or {}
        commit, etag = remote_version(client, repo, file)
        if etag == ptr.get("etag"):
            if commit != local_commit:  # repo moved, this file did not
                _write_pointer(directory, {**ptr, "commit": commit})
            return {"changed": False, "commit": commit, "etag": etag}
        target = directory / f"{etag[:12]}.parquet"
        if not target.exists():
            part = target.with_suffix(".parquet.part")
            with client.stream("GET", f"{HF}/{repo}/resolve/{commit}/{file}") as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
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
