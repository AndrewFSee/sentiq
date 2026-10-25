"""Point-in-time S&P 500 universe (membership, CIK, GICS sector by date).

Sources (both free; recorded with exact versions so a build is reproducible):
* Membership: github.com/fja05680/sp500 (MIT) "S&P 500 Historical Components & Changes
  (Updated).csv" pinned to a commit SHA -- one row per change date with the full ticker list;
  built from a 1996-2019 base list plus Wikipedia changes verified by the maintainer.
  Validated 2026-10: 503-507 members on every date since 2019 (another public dataset dropped
  real members such as FOX/FOXA/DOW).
* Identity: monthly revisions of Wikipedia's "List of S&P 500 companies" (revision ids stored):
  ticker -> CIK, security name, GICS sector as listed at the time. All 671 tickers that were
  members at any point 2019-2026 have a CIK there. Tickers whose CIK changed (re-domiciles,
  holding-company reorganisations: DIS 2019, XOM 2025, ...) are split by snapshot date.
After the last fja snapshot, membership comes from the Wikipedia snapshots (monthly
granularity -- an add/drop is dated to the snapshot that first shows it).
Caveats: Wikipedia is community-edited (small date errors possible); monthly snapshots date
CIK/sector changes to within a month.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

import httpx
import polars as pl
from bs4 import BeautifulSoup

from newsfeed.config import ExtraMember, UniverseConfig
from newsfeed.tickers import Universe, normalize_ticker

log = logging.getLogger(__name__)

FJA_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_TITLE = "List_of_S&P_500_companies"
UA = "sentiq-research/0.1 (https://github.com/AndrewFSee/sentiq)"


# --------------------------------------------------------------------- fetching
def fetch_fja(sha: str, out_dir: Path, client: httpx.Client | None = None) -> Path:
    out = out_dir / f"fja_{sha[:12]}.csv"
    if out.exists():
        return out
    url = f"https://raw.githubusercontent.com/fja05680/sp500/{sha}/{quote(FJA_FILE)}"
    c = client or httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True)
    r = c.get(url)
    r.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def parse_wiki_table(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", {"id": "constituents"}) or soup.find("table", class_="wikitable")
    if t is None:
        raise ValueError("constituents table not found")
    hdr = [th.get_text(" ", strip=True).lower() for th in t.find("tr").find_all("th")]

    def col(*names: str) -> int:
        return next(i for i, h in enumerate(hdr) if any(n in h for n in names))

    ix = {
        "ticker": col("symbol", "ticker"),
        "name": col("security"),
        "sector": col("gics sector"),
        "cik": col("cik"),
    }
    rows = []
    for tr in t.find_all("tr")[1:]:
        td = [x.get_text(" ", strip=True) for x in tr.find_all("td")]
        if len(td) > max(ix.values()):
            rows.append({k: td[i] for k, i in ix.items()})
    return rows


def fetch_wiki_monthly(
    months: list[date], out_path: Path, *, pause: float = 0.5, client: httpx.Client | None = None
) -> pl.DataFrame:
    """One revision per month (the last revision before the 1st). Incremental: months already
    in ``out_path`` are not fetched again."""
    have = pl.read_parquet(out_path) if out_path.exists() else None
    done = set(have["snapshot"].to_list()) if have is not None else set()
    c = client or httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True)
    new = []
    for m in months:
        if m in done:
            continue
        r = c.get(
            WIKI_API,
            params={
                "action": "query",
                "prop": "revisions",
                "titles": WIKI_TITLE,
                "rvlimit": 1,
                "rvstart": f"{m.isoformat()}T00:00:00Z",
                "rvdir": "older",
                "rvprop": "ids|timestamp",
                "format": "json",
            },
        ).json()
        rev = next(iter(r["query"]["pages"].values()))["revisions"][0]
        html = c.get("https://en.wikipedia.org/w/index.php", params={"oldid": rev["revid"]}).text
        for row in parse_wiki_table(html):
            new.append({"snapshot": m, "revid": int(rev["revid"]), **row})
        time.sleep(pause)
    df = pl.DataFrame(new) if new else None
    if df is not None and have is not None:
        df = pl.concat([have, df.select(have.columns)])
    df = df if df is not None else have
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df


# --------------------------------------------------------------------- building
def fja_snapshots(csv_bytes: bytes) -> pl.DataFrame:
    d = pl.read_csv(io.BytesIO(csv_bytes))
    return d.select(
        pl.col("date").str.to_date(),
        pl.col("tickers").str.split(",").list.eval(pl.element().str.strip_chars()),
    )


def membership_intervals(snaps: list[tuple[date, set[str]]]) -> pl.DataFrame:
    """[start, end) per contiguous membership spell; end null = still a member at the last
    snapshot."""
    rows = []
    open_since: dict[str, date] = {}
    prev: set[str] = set()
    for d, members in sorted(snaps, key=lambda x: x[0]):
        for t in members - prev:
            open_since[t] = d
        for t in prev - members:
            rows.append({"ticker": t, "start": open_since.pop(t), "end": d})
        prev = members
    rows += [{"ticker": t, "start": s, "end": None} for t, s in open_since.items()]
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "start": pl.Date, "end": pl.Date})


def build_tables(
    fja_csv: bytes, wiki: pl.DataFrame, start: date
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """(members: ticker, cik, name, start, end) and (sectors: ticker, valid_from, sector)."""
    snaps = fja_snapshots(fja_csv)
    fja = [(r["date"], {normalize_ticker(t) for t in r["tickers"]}) for r in snaps.to_dicts()]
    last_fja = max(d for d, _ in fja)
    if wiki["snapshot"].dtype == pl.Utf8:
        wiki = wiki.with_columns(pl.col("snapshot").str.to_date())
    w = wiki.with_columns(
        pl.col("ticker")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(".", "-", literal=True),
        pl.col("cik").str.extract(r"(\d+)").cast(pl.Int64),
    ).filter(pl.col("cik").is_not_null())
    later = [
        (d, set(g["ticker"].to_list()))
        for (d,), g in w.filter(pl.col("snapshot") > last_fja).group_by("snapshot")
    ]
    spells = membership_intervals(fja + later)
    spells = spells.filter(pl.col("end").is_null() | (pl.col("end") > start))
    # identity history per ticker from the snapshots: (ticker, valid_from, cik, name, sector)
    ident = w.select(
        "ticker", pl.col("snapshot").alias("valid_from"), "cik", "name", "sector"
    ).sort("ticker", "valid_from")
    rows = []
    unmapped = 0
    hist: dict[str, list[dict]] = {}
    for r in ident.to_dicts():
        hist.setdefault(r["ticker"], []).append(r)
    for sp in spells.to_dicts():
        h = hist.get(sp["ticker"])
        if not h:
            unmapped += 1
            continue
        rows += _cik_segments(sp["ticker"], sp["start"], sp["end"], h)
    members = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "cik": pl.Int64,
            "name": pl.Utf8,
            "start": pl.Date,
            "end": pl.Date,
        },
    ).filter(pl.col("end").is_null() | (pl.col("end") > pl.col("start")))
    sectors = ident.select("ticker", "valid_from", "sector")
    stats = {
        "spells": spells.height,
        "member_rows": members.height,
        "unmapped_spells": unmapped,
        "last_fja_snapshot": str(last_fja),
        "ciks": members["cik"].n_unique(),
    }
    return members, sectors, stats


def _cik_segments(ticker: str, start: date, end: date | None, hist: list[dict]) -> list[dict]:
    """Split one membership spell [start, end) where the ticker's CIK changes.

    CIK on day x = CIK of the latest snapshot <= x (the ticker's first snapshot if none, e.g.
    an addition dated between monthly snapshots). Segment boundaries are the snapshot dates
    inside (start, end) at which the CIK differs from the previous snapshot's."""
    hist = sorted(hist, key=lambda r: r["valid_from"])

    def ident_at(x: date) -> dict:
        prior = [r for r in hist if r["valid_from"] <= x]
        return prior[-1] if prior else hist[0]

    cuts = [
        r["valid_from"]
        for prev, r in pairwise(hist)
        if r["cik"] != prev["cik"]
        and start < r["valid_from"]
        and (end is None or r["valid_from"] < end)
    ]
    bounds = [start, *cuts, end]
    out = []
    for s, e in pairwise(bounds):
        idn = ident_at(s)
        out.append({"ticker": ticker, "cik": idn["cik"], "name": idn["name"], "start": s, "end": e})
    return out


# --------------------------------------------------------------------- runtime view
@dataclass
class PitUniverse:
    members: pl.DataFrame  # ticker, cik, name, start, end (end exclusive, null = open)
    sectors: pl.DataFrame  # ticker, valid_from, sector

    @classmethod
    def load(cls, directory: Path) -> PitUniverse:
        return cls(
            pl.read_parquet(directory / "sp500_members.parquet"),
            pl.read_parquet(directory / "sp500_sectors.parquet"),
        )

    @classmethod
    def load_for(cls, ucfg: UniverseConfig) -> PitUniverse:
        """The configured universe: the stored S&P 500 tables plus ``extra_members`` (added in
        memory only; the stored tables, and so their content hash, never include extras)."""
        return cls.load(ucfg.pit_dir).with_extras(ucfg.extra_members)

    def with_extras(self, extras: list[ExtraMember]) -> PitUniverse:
        """Add non-PIT extra members (open spells from ``start``, sector valid from ``start``).
        A ticker or CIK that is already in the S&P 500 tables is rejected: the extras are
        meant to be companies the index never held, so no spell can overlap."""
        if not extras:
            return self
        known_t, known_c = set(self.members["ticker"]), set(self.members["cik"])
        for e in extras:
            if e.ticker in known_t or e.cik in known_c:
                raise ValueError(f"extra member {e.ticker} (CIK {e.cik}) is already in the S&P 500")
        if len({e.ticker for e in extras}) != len(extras):
            raise ValueError("duplicate ticker in extra_members")
        members = pl.concat(
            [
                self.members,
                pl.DataFrame(
                    [(e.ticker, e.cik, e.name, e.start, None) for e in extras],
                    schema=self.members.schema,
                    orient="row",
                ),
            ]
        )
        sectors = pl.concat(
            [
                self.sectors,
                pl.DataFrame(
                    [(e.ticker, e.start, e.sector) for e in extras],
                    schema=self.sectors.schema,
                    orient="row",
                ),
            ]
        )
        return PitUniverse(members, sectors)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.members.write_parquet(directory / "sp500_members.parquet")
        self.sectors.write_parquet(directory / "sp500_sectors.parquet")

    def as_universe(self, tickers: list[str] | None = None) -> Universe:
        """Static CIK -> tickers map over all ever-members (rows are filtered by membership
        at T downstream). A CIK may carry several tickers over time (FB -> META) and a ticker
        several CIKs (re-domiciles); both are fine for linking by filer CIK."""
        m = (
            self.members
            if tickers is None
            else self.members.filter(pl.col("ticker").is_in(tickers))
        )
        c2t: dict[int, list[str]] = {}
        for r in m.select("cik", "ticker").unique().to_dicts():
            c2t.setdefault(int(r["cik"]), []).append(r["ticker"])
        return Universe({k: sorted(v) for k, v in c2t.items()})

    def titles(self) -> dict[int, str]:
        return {
            int(r["cik"]): r["name"]
            for r in self.members.sort("start").unique("cik", keep="last").to_dicts()
        }

    def member_frame(self) -> pl.DataFrame:
        return self.members.select("ticker", "start", "end")

    def is_member(self, df: pl.DataFrame, date_col: str = "date") -> pl.Series:
        """Boolean per row of ``df`` (ticker, date_col): member on that date."""
        j = (
            df.with_row_index("_r")
            .join(self.member_frame(), on="ticker", how="left")
            .filter(
                (pl.col(date_col) >= pl.col("start"))
                & (pl.col("end").is_null() | (pl.col(date_col) < pl.col("end")))
            )
        )
        hit = set(j["_r"].to_list())
        return pl.Series([i in hit for i in range(df.height)])

    def ingest_windows(
        self, start: date, end: date, before_days: int, after_days: int
    ) -> dict[int, tuple[date, date]]:
        """Per CIK: the *ticker's* whole membership window (over all CIKs that ticker ever had)
        widened by margins, clipped to [start, end). Using the ticker's window, not the CIK's
        dated segment, makes ingestion immune to Wikipedia lagging a CIK change (Disney's new
        CIK took effect 2019-03 but appears on Wikipedia only in 2020-10): an old CIK simply
        stops filing, so fetching both over the full window is harmless. Margins give history
        for the 60-day volume and 30-day novelty baselines."""
        from datetime import timedelta

        far = date(9999, 1, 1)
        tick_win = {
            r["ticker"]: (r["start"], r["end"] or far)
            for r in self.members.with_columns(pl.col("end").fill_null(far))
            .group_by("ticker")
            .agg(pl.col("start").min(), pl.col("end").max())
            .to_dicts()
        }
        out: dict[int, tuple[date, date]] = {}
        for r in self.members.select("cik", "ticker").unique().to_dicts():
            s0, e0 = tick_win[r["ticker"]]
            s = max(start, s0 - timedelta(days=before_days))
            e = end if e0 == far else min(end, e0 + timedelta(days=after_days))
            cik = int(r["cik"])
            if cik in out:
                s, e = min(s, out[cik][0]), max(e, out[cik][1])
            if e > s:
                out[cik] = (s, e)
        return out

    def sector_frame(self) -> pl.DataFrame:
        return self.sectors.sort("ticker", "valid_from")

    def sector_etf_frame(self, etf_of: dict[str, str]) -> pl.DataFrame:
        """ticker, valid_from, etf: the sector ETF as of each monthly snapshot. A ticker's first
        snapshot also covers earlier dates (valid_from moved to date.min). Unknown sectors are
        dropped with a warning (rows fall back to the previous snapshot)."""
        s = self.sector_frame()
        unknown = sorted(set(s["sector"].unique().to_list()) - set(etf_of))
        if unknown:
            log.warning("no sector ETF for GICS sector(s) %s", unknown)
        s = s.filter(pl.col("sector").is_in(list(etf_of))).with_columns(
            pl.col("sector").replace_strict(etf_of).alias("etf")
        )
        first = pl.col("valid_from") == pl.col("valid_from").min().over("ticker")
        return s.select(
            "ticker",
            pl.when(first)
            .then(pl.lit(date.min))
            .otherwise(pl.col("valid_from"))
            .alias("valid_from"),
            "etf",
        ).sort("ticker", "valid_from")
