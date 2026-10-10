"""HTML -> text, offline token estimates and truncation."""

from __future__ import annotations

import math
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

_WS = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n{3,}")
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.I)


def html_to_text(html: str | bytes) -> str:
    """Visible text of an (i)XBRL/HTML document, whitespace-normalised, one block per line."""
    with warnings.catch_warnings():
        # EDGAR XHTML/iXBRL starts with an XML declaration; the HTML parser is intended.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    # Inline XBRL hides machine-readable facts in <ix:header> inside a display:none div.
    for tag in soup.find_all(lambda t: t.name in ("ix:header",)):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()
    text = soup.get_text("\n")
    text = unicodedata.normalize("NFKC", text)
    lines = (_WS.sub(" ", ln).strip() for ln in text.splitlines())
    text = "\n".join(ln for ln in lines if ln)
    return _BLANKS.sub("\n\n", text).strip()


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token) if text else 0


def truncate_tokens(text: str, max_tokens: int, chars_per_token: float) -> tuple[str, bool]:
    """Truncate to ~max_tokens on a whitespace boundary. Returns (text, was_truncated)."""
    limit = int(max_tokens * chars_per_token)
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    ws = max(cut.rfind(" "), cut.rfind("\n"))
    if ws > limit * 0.8:
        cut = cut[:ws]
    return cut.rstrip(), True
