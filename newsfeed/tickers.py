"""Universe and ticker <-> CIK mapping.

Caveat: SEC's company_tickers.json is a *current* snapshot, not point-in-time. Historical
filings are fetched by CIK (stable), and labelled with today's ticker(s) for that CIK.
Renamed tickers (FB -> META) therefore get the current symbol, which is what current-symbol
price APIs expect; delisted companies are absent (survivorship bias -- see CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from newsfeed.ingest.http import HttpClient
from newsfeed.timeutil import utcnow


def normalize_ticker(t: str) -> str:
    return t.strip().upper().replace(".", "-")


@dataclass(frozen=True)
class TickerMap:
    ticker_to_cik: dict[str, int]
    cik_to_titles: dict[int, str] = field(default_factory=dict)
    fetched_at: datetime | None = None

    @classmethod
    def from_sec_json(cls, payload: dict, fetched_at: datetime | None = None) -> TickerMap:
        t2c: dict[str, int] = {}
        titles: dict[int, str] = {}
        for row in payload.values():
            cik = int(row["cik_str"])
            t2c[normalize_ticker(row["ticker"])] = cik
            titles.setdefault(cik, row.get("title", ""))
        return cls(t2c, titles, fetched_at)

    def cik(self, ticker: str) -> int:
        try:
            return self.ticker_to_cik[normalize_ticker(ticker)]
        except KeyError:
            raise KeyError(f"ticker {ticker!r} not in SEC ticker map") from None


@dataclass(frozen=True)
class Universe:
    """The configured tickers, resolved to CIKs. Several tickers may share a CIK (GOOG/GOOGL)."""

    cik_to_tickers: dict[int, list[str]]

    @classmethod
    def build(cls, tickers: list[str], tmap: TickerMap) -> Universe:
        out: dict[int, list[str]] = defaultdict(list)
        missing = []
        for t in tickers:
            try:
                out[tmap.cik(t)].append(normalize_ticker(t))
            except KeyError:
                missing.append(t)
        if missing:
            raise KeyError(f"tickers not found in SEC map: {missing}")
        return cls({cik: sorted(ts) for cik, ts in out.items()})

    @property
    def ciks(self) -> list[int]:
        return sorted(self.cik_to_tickers)

    def tickers_for(self, cik: int) -> list[str]:
        return self.cik_to_tickers.get(cik, [])


def load_ticker_map(
    http: HttpClient | Callable[[], HttpClient], url: str, cache_dir: Path, *, refresh: bool = False
) -> TickerMap:
    """Fetch SEC's ticker map, caching it on disk (with fetch time) under ``cache_dir``.
    ``http`` may be a factory so no client (and no User-Agent) is needed on a cache hit."""
    cache = cache_dir / "company_tickers.json"
    if cache.exists() and not refresh:
        blob = json.loads(cache.read_text(encoding="utf-8"))
        return TickerMap.from_sec_json(blob["data"], datetime.fromisoformat(blob["fetched_at"]))
    client = http if isinstance(http, HttpClient) else http()
    payload = client.get(url).json()
    now = utcnow()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fetched_at": now.isoformat(), "data": payload}), encoding="utf-8")
    return TickerMap.from_sec_json(payload, now)


# --------------------------------------------------------------------------- text linking
LINKER_VERSION = 3  # bump when linking logic changes (3: source_cik for source tags)

# Confidence by method. Source tags (vendor ticker lists, EDGAR filer CIK) are authoritative;
# exchange tags like "(NASDAQ: AAPL)" are near-certain; cashtags slightly less (spam, typos);
# company-name aliases can collide with common words, more so when only in the body.
CONFIDENCE = {
    "source": 1.0,
    "exchange_tag": 0.95,
    "cashtag": 0.9,
    "name_title": 0.8,
    "name_body": 0.6,
}
_EXCHANGE_TAG = re.compile(
    r"\b(?:NASDAQ|Nasdaq|NYSE|Nyse|NYSE\s+American|NYSE\s+Arca|Cboe|CBOE)"
    r"(?:\s?(?:GS|GM|CM))?\s*:\s*\$?([A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z])?)\b"
)
# Removed before *name* matching only: press-release boilerplate such as
# "find Walmart on Facebook at facebook.com/walmart" is not a mention of Meta.
_NAME_NOISE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w.-]+\.(?:com|net|org|io)\b(?:/\S*)?"
    r"|\b(?:on|via|at)\s+(?:Facebook|Instagram|LinkedIn|Twitter|YouTube|Threads|TikTok)\b",
    re.I,
)
_CASHTAG = re.compile(r"(?<![\w$])\$([A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z])?)\b")
_NAME_SUFFIX = re.compile(
    r"[,.]?\s+(?:inc|incorporated|corp|corporation|co|company|ltd|plc|holdings?|group|"
    r"platforms|com|/de/|/new/)\.?$",
    re.I,
)


def clean_company_name(title: str) -> str | None:
    """'Apple Inc.' -> 'Apple'; 'MICROSOFT CORP' -> 'Microsoft'; drops '& Co' etc."""
    name = re.sub(r"\s*&\s*co\.?$", "", title.strip(), flags=re.I)
    if _NAME_SUFFIX.fullmatch(" " + name):  # nothing but a suffix ("Inc.")
        return None
    for _ in range(3):
        name = _NAME_SUFFIX.sub("", name).strip(" ,.")
    if not name:
        return None
    if name.isupper():
        name = " ".join(w.capitalize() if len(w) > 3 else w for w in name.split())
    return name if len(name) >= 3 else None


@dataclass(frozen=True)
class TickerLink:
    ticker: str
    cik: int
    method: str
    confidence: float
    n_mentions: int
    in_title: bool


class TickerLinker:
    """Deterministic text -> universe-ticker linking. Only universe tickers are ever emitted.

    A company-level mention (name alias) links every share class of that CIK (GOOG and GOOGL);
    downstream scoring can de-duplicate by ``cik``. Exchange tags / cashtags link the exact
    ticker written (if in the universe).
    """

    def __init__(
        self,
        universe: Universe,
        aliases: dict[str, list[str]] | None = None,
        company_titles: dict[int, str] | None = None,
        source_only_sources: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.universe = universe
        # items from these sources link ONLY through their source tags (no text matching):
        # an earnings call names competitors and customers, but is about the reporting company
        self.source_only_sources = tuple(sorted(set(source_only_sources)))
        self.ticker_to_cik = {t: c for c, ts in universe.cik_to_tickers.items() for t in ts}
        self._aliases_in = {k: sorted(v) for k, v in sorted((aliases or {}).items())}
        self._titles_in = {str(k): v for k, v in sorted((company_titles or {}).items())}
        names: dict[int, set[str]] = {c: set() for c in universe.cik_to_tickers}
        for t, al in (aliases or {}).items():
            cik = self.ticker_to_cik.get(normalize_ticker(t))
            if cik is None:
                raise KeyError(f"alias given for ticker {t!r} not in universe")
            names[cik].update(a.strip() for a in al if a.strip())
        for cik, title in (company_titles or {}).items():
            if cik in names and (clean := clean_company_name(title)):
                names[cik].add(clean)
        # longest alias first so "JPMorgan Chase" wins over "JPMorgan" in counting
        self._name_res = {
            cik: re.compile(
                r"(?<![\w&])(?:"
                + "|".join(re.escape(n) for n in sorted(ns, key=len, reverse=True))
                + r")(?![\w&])"
            )
            for cik, ns in names.items()
            if ns
        }

    @property
    def linker_id(self) -> str:
        """Version + hash of every input that affects output; changes trigger a relink."""
        blob = json.dumps(
            [LINKER_VERSION, self.universe.cik_to_tickers, self._aliases_in, self._titles_in]
            + ([list(self.source_only_sources)] if self.source_only_sources else []),
            sort_keys=True,
            default=str,
        )
        return f"v{LINKER_VERSION}-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"

    def link(
        self,
        *,
        title: str | None,
        body: str | None,
        source_tickers: list[str],
        source: str | None = None,
        source_cik: int | None = None,
    ) -> list[TickerLink]:
        """``source_cik``: the company the source tags belong to, when the source knows it
        (a ticker shared by two CIKs in the universe is otherwise resolved arbitrarily)."""
        title = title or ""
        body = body or ""
        hits: dict[str, dict[str, Any]] = {}

        def add(ticker: str, method: str, n: int, in_title: bool) -> None:
            ticker = normalize_ticker(ticker)
            cik = self.ticker_to_cik.get(ticker)
            if (
                method == "source"
                and source_cik is not None
                and ticker in self.universe.tickers_for(source_cik)
            ):
                cik = source_cik
            if cik is None:
                return
            h = hits.setdefault(
                ticker, {"cik": cik, "method": method, "conf": 0.0, "n": 0, "title": False}
            )
            if CONFIDENCE[method] > h["conf"]:
                h["conf"], h["method"] = CONFIDENCE[method], method
            h["n"] += n
            h["title"] = h["title"] or in_title

        for t in source_tickers:
            add(t, "source", 0, False)
        texts = [] if source in self.source_only_sources else [(title, True), (body, False)]
        for text, is_title in texts:
            for rx, method in ((_EXCHANGE_TAG, "exchange_tag"), (_CASHTAG, "cashtag")):
                for m in rx.finditer(text):
                    add(m.group(1), method, 1, is_title)
            clean = _NAME_NOISE.sub(" ", text)
            for cik, rx in self._name_res.items():
                n = len(rx.findall(clean))
                if n:
                    for t in self.universe.tickers_for(cik):
                        add(t, "name_title" if is_title else "name_body", n, is_title)
        return [
            TickerLink(t, h["cik"], h["method"], h["conf"], h["n"], h["title"])
            for t, h in sorted(hits.items())
        ]
