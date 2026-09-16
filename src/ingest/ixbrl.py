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

__all__ = ["IXBRLLeakError", "assert_no_ixbrl_leak", "strip_ixbrl"]


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

# Tokens that must never appear in agent-visible text. Checked case-insensitively
# except where the token is only meaningful in its given case.
_LEAK_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (r"<\s*/?\s*ix\s*:", "ix: element"),
    (r"xmlns\s*:\s*ix", "ix namespace declaration"),
    (r"inlineXBRL", "inline XBRL namespace URI"),
    (r"\bcontextRef\b", "contextRef attribute"),
    (r"\bunitRef\b", "unitRef attribute"),
    (r"\bxbrli\s*:", "xbrli: element or namespace"),
    (r"\bxbrldi\s*:", "xbrldi: element or namespace"),
    (r"\bxlink\s*:", "xlink: attribute or namespace"),
    (r"\bschemaRef\b", "schemaRef element"),
    (r"\b(?:us-gaap|dei|srt|ifrs-full|invest|country|currency)\s*:\s*[A-Za-z]", "taxonomy QName"),
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


def strip_ixbrl(source: str) -> str:
    """Return the filing as a human reader sees it, with all XBRL markup removed.

    Idempotent: stripping an already-stripped document returns it unchanged.
    Raises ``IXBRLLeakError`` if anything machine-readable survives, so a parser
    edge case fails loudly instead of silently poisoning a benchmark run.
    """
    if not source.strip():
        raise ValueError("empty document passed to strip_ixbrl")

    parser = html.HTMLParser(remove_comments=True, remove_pis=True, huge_tree=True)
    root = html.document_fromstring(source, parser=parser)

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
    for pattern, description in _LEAK_PATTERNS:
        match = re.search(pattern, document, flags=re.IGNORECASE)
        if match is not None:
            start = max(0, match.start() - 60)
            context = document[start : match.end() + 60].replace("\n", " ")
            raise IXBRLLeakError(
                f"{description} survived preprocessing at offset {match.start()}: ...{context}..."
            )
