"""Markdown block/inline model shared by the PDF and DOCX dossier renderers.

The dossier is generated as GitHub-flavored Markdown, so both exporters need the
same understanding of headings, pipe tables, nested lists and inline spans.
Parsing lives here, free of reportlab and python-docx imports, so it can be unit
tested directly and so the two renderers can never drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

CJK_PATTERN = re.compile(
    "[\u1100-\u11ff\u2e80-\u303f\u3040-\u30ff\u3130-\u318f"
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]"
)


_CJK_RUN = re.compile(f"{CJK_PATTERN.pattern}+")


def has_cjk(text: str) -> bool:
    """Report whether a string needs the CID font to avoid missing glyphs."""
    return CJK_PATTERN.search(text) is not None


@dataclass(frozen=True)
class Span:
    """A run of text carrying resolved (already flattened) inline styling."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    href: str | None = None


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class Quote:
    text: str


@dataclass
class Rule:
    pass


@dataclass
class Code:
    text: str
    language: str = ""


@dataclass
class ListItem:
    level: int
    ordered: bool
    marker: str
    text: str


@dataclass
class ListBlock:
    items: list[ListItem] = field(default_factory=list)


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    aligns: list[str]


@dataclass
class Details:
    summary: str
    blocks: list[Block] = field(default_factory=list)


Block = Heading | Para | Quote | Rule | Code | ListBlock | Table | Details

# Ordered alternation: code spans win over emphasis so identifiers such as
# `deep_research_unavailable` survive, and `**` is matched before `*`.
# Underscore emphasis requires non-word boundaries for the same reason.
_INLINE_PATTERN = re.compile(
    r"(?P<fence>`+)(?P<code>.+?)(?P=fence)"
    r"|\[(?P<link_text>[^\]\n]*)\]\((?P<href>[^)\s]*)(?:\s+\"[^\"]*\")?\)"
    r"|\*\*(?P<strong>\S(?:[^\n]*?\S)?)\*\*"
    r"|(?<![\w])__(?P<strong_us>\S(?:[^\n]*?\S)?)__(?!\w)"
    r"|(?<!\*)\*(?P<em>[^*\n]*[^*\s])\*(?!\*)"
    r"|(?<![\w])_(?P<em_us>[^_\n]*[^_\s])_(?!\w)"
)


def parse_inline(text: str) -> list[Span]:
    """Flatten Markdown inline markup into styled spans (nesting is merged)."""
    return _parse_inline(text, bold=False, italic=False, href=None)


def _parse_inline(
    text: str, *, bold: bool, italic: bool, href: str | None
) -> list[Span]:
    spans: list[Span] = []
    position = 0
    for match in _INLINE_PATTERN.finditer(text):
        if match.start() > position:
            spans.append(
                Span(text[position : match.start()], bold, italic, False, href)
            )
        position = match.end()
        if match.group("code") is not None:
            spans.append(Span(match.group("code"), bold, italic, True, href))
        elif match.group("link_text") is not None:
            spans.extend(
                _parse_inline(
                    match.group("link_text"),
                    bold=bold,
                    italic=italic,
                    href=match.group("href") or href,
                )
            )
        elif match.group("strong") is not None or match.group("strong_us") is not None:
            inner = match.group("strong")
            if inner is None:
                inner = match.group("strong_us")
            spans.extend(_parse_inline(inner, bold=True, italic=italic, href=href))
        else:
            inner = match.group("em")
            if inner is None:
                inner = match.group("em_us")
            spans.extend(_parse_inline(inner, bold=bold, italic=True, href=href))
    if position < len(text):
        spans.append(Span(text[position:], bold, italic, False, href))
    return [span for span in spans if span.text]


def cjk_markup(escaped: str, cjk_font: str | None) -> str:
    """Pin the CID face onto CJK runs only, so Latin text keeps the serif face."""
    if not cjk_font or not has_cjk(escaped):
        return escaped
    return _CJK_RUN.sub(
        lambda match: f'<font face="{cjk_font}">{match.group(0)}</font>', escaped
    )


def inline_markup(text: str, cjk_font: str | None = None) -> str:
    """Render inline spans as reportlab mini-HTML with everything XML-escaped."""
    return "".join(_span_markup(span, cjk_font) for span in parse_inline(text))


def _span_markup(span: Span, cjk_font: str | None = None) -> str:
    body = escape(span.text, quote=False).replace("\n", "<br/>")
    body = cjk_markup(body, cjk_font)
    if span.code:
        # Relative size keeps Courier optically matched to the surrounding serif.
        body = f'<font face="Courier" size="-1.2">{body}</font>'
    if span.bold:
        body = f"<b>{body}</b>"
    if span.italic:
        body = f"<i>{body}</i>"
    if span.href:
        href = escape(span.href, quote=False).replace('"', "&quot;")
        body = f'<link href="{href}" color="blue">{body}</link>'
    return body


def plain_text(text: str) -> str:
    """Strip inline markup, for titles, bookmarks and word-processor fallbacks."""
    return "".join(span.text for span in parse_inline(text))


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_LIST_ITEM = re.compile(r"^(\s*)(?:([-*+])|(\d{1,9})[.)])\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{1,}:?\s*\|?\s*$")
_DETAILS_OPEN = re.compile(r"^\s*<details[^>]*>\s*(.*)$", re.IGNORECASE)
_SUMMARY = re.compile(r"^\s*<summary[^>]*>(.*?)</summary>\s*(.*)$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(\S*)")


def parse_blocks(content: str) -> list[Block]:
    """Parse a Markdown document into a flat list of renderable blocks."""
    lines = content.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            index, text = _consume_fence(lines, index, fence.group(1))
            blocks.append(Code(text, fence.group(2)))
            continue

        details = _DETAILS_OPEN.match(line)
        if details:
            index, block = _consume_details(lines, index, details.group(1))
            blocks.append(block)
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(Heading(len(heading.group(1)), heading.group(2)))
            index += 1
            continue

        if _RULE.match(line):
            blocks.append(Rule())
            index += 1
            continue

        if line.lstrip().startswith("|") and index + 1 < len(lines):
            if _TABLE_SEPARATOR.match(lines[index + 1]):
                index, table = _consume_table(lines, index)
                blocks.append(table)
                continue

        if line.lstrip().startswith(">"):
            index, text = _consume_quote(lines, index)
            blocks.append(Quote(text))
            continue

        if _LIST_ITEM.match(line):
            index, block = _consume_list(lines, index)
            blocks.append(block)
            continue

        if line.strip() in {"{", "["}:
            index, text = _consume_json(lines, index)
            blocks.append(Code(text, "json"))
            continue

        index, text = _consume_paragraph(lines, index)
        blocks.append(Para(text))
    return blocks


def _consume_fence(lines: list[str], index: int, fence: str) -> tuple[int, str]:
    body: list[str] = []
    index += 1
    while index < len(lines) and not lines[index].strip().startswith(fence[0] * 3):
        body.append(lines[index])
        index += 1
    return index + 1, "\n".join(body)


def _consume_details(
    lines: list[str], index: int, remainder: str
) -> tuple[int, Details]:
    summary = ""
    body: list[str] = []
    pending = remainder
    index += 1
    while True:
        match = _SUMMARY.match(pending)
        if match:
            summary = match.group(1).strip()
            pending = match.group(2)
        closing = re.split(r"</details\s*>", pending, maxsplit=1, flags=re.IGNORECASE)
        if len(closing) == 2:
            body.append(closing[0])
            break
        if pending.strip():
            body.append(pending)
        if index >= len(lines):
            break
        pending = lines[index]
        index += 1
    return index, Details(summary or "Details", parse_blocks("\n".join(body)))


def _consume_table(lines: list[str], index: int) -> tuple[int, Table]:
    header = _split_row(lines[index])
    aligns = _row_alignments(_split_row(lines[index + 1]), len(header))
    index += 2
    rows: list[list[str]] = []
    while index < len(lines) and lines[index].lstrip().startswith("|"):
        cells = _split_row(lines[index])
        cells = (cells + [""] * len(header))[: len(header)]
        rows.append(cells)
        index += 1
    return index, Table(header, rows, aligns)


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [
        cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)
    ]


def _row_alignments(cells: list[str], width: int) -> list[str]:
    aligns = []
    for cell in cells:
        marker = cell.strip()
        if marker.startswith(":") and marker.endswith(":"):
            aligns.append("center")
        elif marker.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    return (aligns + ["left"] * width)[:width]


def _consume_quote(lines: list[str], index: int) -> tuple[int, str]:
    body: list[str] = []
    while index < len(lines) and lines[index].lstrip().startswith(">"):
        body.append(re.sub(r"^\s*>\s?", "", lines[index]))
        index += 1
    return index, "\n".join(body).strip()


def _consume_list(lines: list[str], index: int) -> tuple[int, ListBlock]:
    block = ListBlock()
    indents: list[int] = []
    while index < len(lines):
        match = _LIST_ITEM.match(lines[index])
        if not match:
            if lines[index].strip() and block.items and lines[index].startswith(" "):
                block.items[-1].text += "\n" + lines[index].strip()
                index += 1
                continue
            if not lines[index].strip() and index + 1 < len(lines):
                if _LIST_ITEM.match(lines[index + 1]):
                    index += 1
                    continue
            break
        indent = len(match.group(1).expandtabs(4))
        while indents and indent < indents[-1]:
            indents.pop()
        if not indents or indent > indents[-1]:
            indents.append(indent)
        ordered = match.group(3) is not None
        block.items.append(
            ListItem(
                level=len(indents) - 1,
                ordered=ordered,
                marker=match.group(3) or match.group(2),
                text=match.group(4).strip(),
            )
        )
        index += 1
    return index, block


def _consume_json(lines: list[str], index: int) -> tuple[int, str]:
    """Treat an unfenced balanced JSON literal as a code block, not as prose."""
    body: list[str] = []
    depth = 0
    while index < len(lines):
        line = lines[index]
        body.append(line)
        depth += line.count("{") + line.count("[") - line.count("}") - line.count("]")
        index += 1
        if depth <= 0:
            break
    return index, "\n".join(body)


def _consume_paragraph(lines: list[str], index: int) -> tuple[int, str]:
    body: list[str] = []
    while index < len(lines) and lines[index].strip():
        line = lines[index]
        if body and (
            _HEADING.match(line)
            or _RULE.match(line)
            or _LIST_ITEM.match(line)
            or _FENCE.match(line)
            or _DETAILS_OPEN.match(line)
            or line.lstrip().startswith(("|", ">"))
        ):
            break
        body.append(line.strip())
        index += 1
    return index, "\n".join(body)
