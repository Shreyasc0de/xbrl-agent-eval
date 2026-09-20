"""Inline XBRL removal: the boundary between what the agent sees and the answer key.

An iXBRL filing carries the tagged values inside the HTML itself, as ``ix:``
elements and ``contextRef`` / ``unitRef`` / ``scale`` attributes. Handing that
document to an agent would hand it the answer key. This module reduces a filing
to what a human reader actually sees on the page, and nothing else.

Three categories are removed:

1. Machine-readable scaffolding: ``ix:header``, ``ix:references``,
   ``ix:resources``, and the ``xbrli:`` context and unit declarations.
2. Hidden facts: ``ix:hidden``, and any element hidden with ``display:none``.
   These carry tagged values that no reader of the rendered filing can see, so
   they are answer-key material by definition.
3. Fact markup on displayed values: the ``ix:`` wrapper elements are unwrapped,
   keeping their displayed text and dropping every attribute that says what the
   value means.

What survives is the rendered text and the surrounding HTML structure.

See ``tests/test_no_ixbrl_leak.py``. That test is the contract; this module is
an implementation of it.
"""

from __future__ import annotations

import re
from typing import Final

from lxml import etree, html

__all__ = ["IXBRLLeakError", "assert_no_ixbrl_leak", "filing_to_text", "strip_ixbrl"]

# Tags whose boundaries are meaningful to a reader. Financial statements are
# tables, so row and cell boundaries have to survive into the text or every
# figure runs into its neighbour and the retrieval layer sees noise.
_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "dd",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    }
)
_CELL_TAGS: Final[frozenset[str]] = frozenset({"td", "th"})


class IXBRLLeakError(AssertionError):
    """Raised when machine-readable XBRL survives into agent-visible text."""


# Subtrees removed whole. Matched on localname, case-insensitively, with or
# without a namespace prefix, since the HTML parser keeps prefixes in the tag.
_DROP_SUBTREE_LOCALNAMES: Final[frozenset[str]] = frozenset(
    {
        "header",
        "hidden",
        "references",
        "resources",
        "context",
        "unit",
        "schemaref",
        "relationship",
        "denominator",
        "numerator",
    }
)

# Namespace prefixes whose elements are never part of the rendered document.
_MACHINE_PREFIXES: Final[tuple[str, ...]] = ("ix", "xbrli", "xbrldi", "link", "xlink", "iso4217")

# Attributes stripped from every element, whatever the tag.
_DROP_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "contextref",
        "unitref",
        "scale",
        "decimals",
        "precision",
        "sign",
        "format",
        "escape",
        "continuedat",
        "tupleref",
        "tupleid",
        "order",
        "target",
    }
)

# Tokens that must never appear in agent-visible text.
#
# Each entry carries its own flags, because these patterns do not all want the
# same matching. Markup names are case-insensitive in practice; a taxonomy
# QName is not, and matching it case-insensitively is what made this check
# reject real filings.
_LEAK_PATTERNS: Final[tuple[tuple[str, str, int], ...]] = (
    (r"<\s*/?\s*ix\s*:", "ix: element", re.IGNORECASE),
    (r"xmlns\s*:\s*ix", "ix namespace declaration", re.IGNORECASE),
    (r"inlineXBRL", "inline XBRL namespace URI", re.IGNORECASE),
    (r"\bcontextRef\b", "contextRef attribute", re.IGNORECASE),
    (r"\bunitRef\b", "unitRef attribute", re.IGNORECASE),
    (r"\bxbrli\s*:", "xbrli: element or namespace", re.IGNORECASE),
    (r"\bxbrldi\s*:", "xbrldi: element or namespace", re.IGNORECASE),
    (r"\bxlink\s*:", "xlink: attribute or namespace", re.IGNORECASE),
    (r"\bschemaRef\b", "schemaRef element", re.IGNORECASE),
    # A taxonomy QName is prefix:LocalName with no whitespace around the colon
    # and an upper camel case local name. Both details matter.
    #
    # "currency", "country" and "invest" are taxonomy prefixes and also ordinary
    # English words, and filings are full of lines like "Foreign currency: Net
    # increase (decrease) in fair value". An earlier version of this pattern
    # allowed whitespace after the colon and matched case-insensitively, so it
    # read that sentence as a leak and refused to cache two real filings. Fail
    # closed is the right default, but a check that fires on English prose
    # stops being a check and becomes an outage.
    #
    # The local name is two or more characters, not three: country:US and
    # currency:US are real QNames, and requiring three silently exempted them.
    (
        r"\b(?:us-gaap|dei|srt|ifrs-full|invest|country|currency|exch|naics|sic|stpr|utr)"
        r":[A-Z][A-Za-z0-9]+\b",
        "taxonomy QName",
        0,
    ),
)


def _localname(tag: str) -> str:
    """Localname of a tag, handling both ``{uri}name`` and ``prefix:name`` forms."""
    if tag.startswith("{"):
        tag = tag.rsplit("}", 1)[-1]
    return tag.rsplit(":", 1)[-1].lower()


def _prefix(tag: str) -> str:
    if tag.startswith("{") or ":" not in tag:
        return ""
    return tag.split(":", 1)[0].lower()


def _is_hidden(el: etree._Element) -> bool:
    style = el.get("style")
    if not style:
        return False
    return re.search(r"display\s*:\s*none", style, flags=re.IGNORECASE) is not None


def _drop_subtree(el: etree._Element) -> None:
    """Remove an element and its descendants, preserving the tail text.

    The tail belongs to the surrounding prose, not to the removed element, so
    discarding it would silently delete readable text.
    """
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail or ""
    if tail:
        previous = el.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


def _unwrap(el: etree._Element) -> None:
    """Replace an element with its own contents, keeping all displayed text."""
    parent = el.getparent()
    if parent is None:
        return
    index = parent.index(el)
    text = el.text or ""
    children = list(el)

    if children:
        if text:
            previous = el.getprevious()
            if previous is not None:
                previous.tail = (previous.tail or "") + text
            else:
                parent.text = (parent.text or "") + text
        for offset, child in enumerate(children):
            parent.insert(index + offset, child)
        last = children[-1]
        last.tail = (last.tail or "") + (el.tail or "")
    else:
        merged = text + (el.tail or "")
        if merged:
            previous = el.getprevious()
            if previous is not None:
                previous.tail = (previous.tail or "") + merged
            else:
                parent.text = (parent.text or "") + merged

    parent.remove(el)


def _clean_attributes(el: etree._Element) -> None:
    for name in list(el.attrib):
        lowered = name.lower()
        bare = lowered.rsplit(":", 1)[-1] if not lowered.startswith("{") else _localname(lowered)
        if (
            lowered.startswith("xmlns")
            or lowered.startswith("{")
            or ":" in lowered
            or bare in _DROP_ATTRS
        ):
            del el.attrib[name]


_XML_DECLARATION: Final[re.Pattern[str]] = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)


def _parseable(source: str | bytes) -> str | bytes:
    """Give lxml something it will accept, without changing the encoding story.

    lxml refuses a str that carries an XML encoding declaration, and SEC
    filings routinely begin with one. There are two cases and they want
    opposite treatment.

    Bytes pass through untouched, so lxml reads the declaration and decodes
    accordingly. Encoding a str to bytes instead would throw away the encoding
    lxml had already resolved, and it silently mangled non-ASCII characters on
    a second pass, which is how the idempotency test caught it.

    A str has its declaration removed and is parsed as a str.
    """
    if isinstance(source, bytes):
        return source
    return _XML_DECLARATION.sub("", source, count=1)


def strip_ixbrl(source: str | bytes) -> str:
    """Return the filing as a human reader sees it, with all XBRL markup removed.

    Idempotent: stripping an already-stripped document returns it unchanged.
    Raises ``IXBRLLeakError`` if anything machine-readable survives, so a parser
    edge case fails loudly instead of silently poisoning a benchmark run.
    """
    if not source.strip():
        raise ValueError("empty document passed to strip_ixbrl")

    parser = html.HTMLParser(remove_comments=True, remove_pis=True, huge_tree=True)
    root = html.document_fromstring(_parseable(source), parser=parser)

    # Pass 1: remove whole subtrees that are never rendered. Iterate over a
    # materialized list because the tree is mutated during the walk.
    for el in list(root.iter()):
        if el.getparent() is None:
            continue
        tag = el.tag
        if not isinstance(tag, str):
            _drop_subtree(el)
            continue
        prefix = _prefix(tag)
        local = _localname(tag)
        is_non_rendered = local in ("script", "style")
        is_machine_scaffolding = prefix in _MACHINE_PREFIXES and local in _DROP_SUBTREE_LOCALNAMES
        if is_non_rendered or is_machine_scaffolding or _is_hidden(el):
            _drop_subtree(el)

    # Pass 2: unwrap the remaining fact elements, keeping their displayed text.
    # Deepest first, so a nested ix:exclude inside an ix:nonNumeric is handled
    # before its parent disappears.
    for el in reversed(list(root.iter())):
        tag = el.tag
        if not isinstance(tag, str):
            continue
        if _prefix(tag) in _MACHINE_PREFIXES:
            _unwrap(el)

    # Pass 3: strip attributes everywhere, including the namespace declarations
    # that the HTML parser keeps as ordinary attributes on the root element.
    for el in root.iter():
        if isinstance(el.tag, str):
            _clean_attributes(el)

    cleaned = str(html.tostring(root, encoding="unicode", method="html"))
    assert_no_ixbrl_leak(cleaned)
    return cleaned


def assert_no_ixbrl_leak(document: str) -> None:
    """Raise ``IXBRLLeakError`` if the document still contains XBRL markup.

    Call this at every point where text crosses into agent-visible territory,
    not only after stripping. A leak that reaches an agent invalidates the run
    it appears in and every comparison that run takes part in.
    """
    for pattern, description, flags in _LEAK_PATTERNS:
        match = re.search(pattern, document, flags=flags)
        if match is not None:
            start = max(0, match.start() - 60)
            context = document[start : match.end() + 60].replace("\n", " ")
            raise IXBRLLeakError(
                f"{description} survived preprocessing at offset {match.start()}: ...{context}..."
            )


def filing_to_text(document: str) -> str:
    """Render a stripped filing as plain text, preserving table structure.

    The agent reads this, so it is the last point where a leak would matter.
    ``assert_no_ixbrl_leak`` runs on the result, not only on the HTML, because a
    conversion step is exactly the kind of place an attribute can reappear in
    the output as visible text.

    Cells are separated by tabs and rows by newlines. A financial statement is a
    table, and a converter that flattens it to a stream of numbers destroys the
    association between a line label and its figure, which is most of what the
    agent needs to read.
    """
    assert_no_ixbrl_leak(document)

    parser = html.HTMLParser(remove_comments=True, remove_pis=True, huge_tree=True)
    root = html.document_fromstring(_parseable(document), parser=parser)

    # Source newlines are layout, not structure: a filing wraps a table cell
    # across several lines of HTML for readability. Collapsing them before
    # inserting real separators is what keeps a line label attached to its
    # figure instead of splitting them onto separate lines.
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        if element.text:
            element.text = re.sub(r"\s+", " ", element.text)
        if element.tail:
            element.tail = re.sub(r"\s+", " ", element.tail)

    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        local = _localname(tag)
        if local in _CELL_TAGS:
            element.tail = (element.tail or "") + "\t"
        elif local in _BLOCK_TAGS:
            element.tail = (element.tail or "") + "\n"

    raw = str(root.text_content())

    lines: list[str] = []
    for line in raw.replace("\xa0", " ").split("\n"):
        cells = [re.sub(r"[ \t]+", " ", cell).strip() for cell in line.split("\t")]
        cleaned = "\t".join(cell for cell in cells if cell)
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    assert_no_ixbrl_leak(text)
    return text
