"""Masking: replace identifying terms with placeholders for the look-ahead/leakage check.

If a model scores a document differently when it can see *who* the company is, it may be using
memorised knowledge (including what happened afterwards) rather than the text. Scoring masked
and unmasked versions of the same documents and measuring agreement (``scoring.agreement``)
quantifies that.

Replaced (universe companies only; other companies' names stay visible):
  company names (aliases, SEC names, cleaned SEC names, upper-case variants) -> [COMPANY_X]
  tickers (incl. BRK.B / BRK/B variants)                                      -> [TICKER_X]
  products / people / places from config                                     -> [PRODUCT_n], ...
The TARGET is always company A; other companies get B, C, ... in order of first appearance.
Longest match wins ("Apple Watch" is a product, not "Apple" + "Watch").
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass, field

from newsfeed.config import UniverseConfig
from newsfeed.tickers import Universe, clean_company_name, normalize_ticker

_KIND_PREFIX = {"product": "PRODUCT", "person": "PERSON", "place": "PLACE"}
_BOUND_L = r"(?<![\w&])"
_BOUND_R = r"(?![\w&])"


def trie_pattern(words: list[str]) -> str:
    """Regex matching exactly the alternation of ``words``, preferring the longest word at a
    position (greedy optional tails; backtracks to shorter words if a later boundary check
    fails), compiled as a character trie. A flat ``a|b|...`` over ~2,700 names is tried at every
    position and made masking ~0.5 s per 8-K; the trie is equivalent and far faster."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = True  # end-of-word marker ("" is never a real character key)

    def build(node: dict) -> str:
        children = sorted((ch, sub) for ch, sub in node.items() if ch != "")
        if not children:
            return ""
        alts = [re.escape(ch) + build(sub) for ch, sub in children]
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return f"(?:{body})?" if "" in node else body

    return build(trie)


@dataclass(frozen=True)
class Term:
    text: str
    kind: str  # company | ticker | product | person | place
    cik: int


@dataclass
class MaskLexicon:
    terms: list[Term]
    _rx: re.Pattern[str] = field(init=False, repr=False)
    _lookup: dict[str, Term] = field(init=False, repr=False)
    _residual_rx: re.Pattern[str] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lookup = {}
        for t in sorted(self.terms, key=lambda t: (-len(t.text), t.text)):
            self._lookup.setdefault(t.text, t)
        self._rx = re.compile(f"{_BOUND_L}(?:{trie_pattern(list(self._lookup))}){_BOUND_R}")
        long_terms = sorted({t.text.lower() for t in self.terms if len(t.text) >= 4})
        self._residual_rx = (
            re.compile(f"{_BOUND_L}(?:{trie_pattern(long_terms)}){_BOUND_R}", re.I)
            if long_terms
            else None
        )

    @classmethod
    def build(
        cls, ucfg: UniverseConfig, universe: Universe, titles: dict[int, str] | None = None
    ) -> MaskLexicon:
        titles = titles or {}
        t2c = {t: c for c, ts in universe.cik_to_tickers.items() for t in ts}
        terms: list[Term] = []

        def add(text: str, kind: str, cik: int, upper: bool = False) -> None:
            text = text.strip()
            if not text:
                return
            terms.append(Term(text, kind, cik))
            if upper and text.upper() != text:
                terms.append(Term(text.upper(), kind, cik))

        for cik, tickers in universe.cik_to_tickers.items():
            for t in tickers:
                for v in {t, t.replace("-", "."), t.replace("-", "/")}:
                    add(v, "ticker", cik)
            if title := titles.get(cik):
                add(title, "company", cik, upper=True)
                if clean := clean_company_name(title):
                    add(clean, "company", cik, upper=True)
        for src, kind in (
            (ucfg.aliases, "company"),
            (ucfg.products, "product"),
            (ucfg.people, "person"),
            (ucfg.places, "place"),
        ):
            for ticker, words in src.items():
                cik = t2c.get(normalize_ticker(ticker))
                if cik is None:
                    raise KeyError(f"{kind} terms given for {ticker!r}, not in universe")
                for w in words:
                    add(w, kind, cik, upper=kind in ("company", "person"))
        return cls(terms)

    @property
    def mask_id(self) -> str:
        """Hash of every term and its mapping; changes when config mask terms change."""
        blob = json.dumps(sorted((t.text, t.kind, t.cik) for t in self.terms))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def mask(self, title: str, body: str, target_cik: int) -> MaskedDoc:
        letters: dict[int, str] = {target_cik: "A"}
        numbers: dict[tuple[str, str], int] = {}
        counters: dict[str, int] = {}
        joined = f"{title}\n{body}"
        # assign labels in order of first appearance (target is always A)
        for m in self._rx.finditer(joined):
            t = self._lookup[m.group(0)]
            if t.kind in ("company", "ticker"):
                if t.cik not in letters:
                    letters[t.cik] = _letter(len(letters))
            elif (t.kind, t.text) not in numbers:
                counters[t.kind] = counters.get(t.kind, 0) + 1
                numbers[(t.kind, t.text)] = counters[t.kind]

        def repl(m: re.Match[str]) -> str:
            t = self._lookup[m.group(0)]
            if t.kind == "company":
                return f"[COMPANY_{letters[t.cik]}]"
            if t.kind == "ticker":
                return f"[TICKER_{letters[t.cik]}]"
            return f"[{_KIND_PREFIX[t.kind]}_{numbers[(t.kind, t.text)]}]"

        m_title = self._rx.sub(repl, title)
        m_body = self._rx.sub(repl, body)
        return MaskedDoc(
            title=m_title,
            body=m_body,
            target_label="[COMPANY_A] (ticker: [TICKER_A])",
            n_replacements=len(self._rx.findall(joined)),
            residuals=self.residuals(f"{m_title}\n{m_body}"),
        )

    def residuals(self, text: str) -> int:
        """Case-insensitive count of identifying terms (>= 4 chars) left after masking.
        Diagnostic only: includes harmless hits such as the fruit 'apple'."""
        return len(self._residual_rx.findall(text)) if self._residual_rx is not None else 0


@dataclass(frozen=True)
class MaskedDoc:
    title: str
    body: str
    target_label: str
    n_replacements: int
    residuals: int


def _letter(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = string.ascii_uppercase[r] + out
    return out
