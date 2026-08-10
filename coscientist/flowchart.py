"""A specialist's workflow diagram, drawn as a diagram.

Every generator is asked for ``workflow_diagram_mermaid`` and most return one, and
the exporters printed the answer the way it arrived: eleven lines of ``F -->|Yes|
H[Randomized Long-term Cycling]`` set in Courier, under a caption reading
"**Figure 1.** Proposed Workflow". Markdown gets away with it, because the viewers a
reader is likely to open it in draw a Mermaid fence themselves. The PDF is the
deliverable that has to stand next to a published report, and it cannot: what it put
on the page was the drawing's source code labelled as the drawing.

So the source is parsed here into nodes and arrows, laid out in layers, and handed to
the PDF exporter as vector shapes. The parser is deliberately narrow -- flowcharts, one
arrow per statement, the four node shapes the generators actually emit -- and returns
``None`` on anything it does not recognise rather than guessing, which puts the old
verbatim listing back on the page for that one figure. Word takes the same graph as a
list of steps, since a .docx cannot hold a vector drawing without an image and this
image would have to be rasterised by a library the deployment does not carry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

# Mermaid's own header. ``graph`` and ``flowchart`` are the same diagram; the four
# directions are the axis the arrows run along.
_HEADER = re.compile(r"^(?:graph|flowchart)\s+(TD|TB|LR|RL|BT)\s*;?$", re.IGNORECASE)
# Presentation-only statements, which change nothing a reader of the drawing sees at
# this level of fidelity, and comments.
_IGNORED = re.compile(
    r"^(?:%%|style\s|classDef\s|class\s|click\s|linkStyle\s|direction\s)",
    re.IGNORECASE,
)
# Everything the layout below cannot honour. A subgraph is a box around a region and
# dropping it would silently redraw the author's grouping as no grouping at all.
_UNSUPPORTED = re.compile(r"^(?:subgraph\b|end\b)", re.IGNORECASE)
# ``A --> B``, ``A -.-> B``, ``A ==> B``, ``A --- B``, each optionally carrying a
# ``|label|`` after the arrow. Two dashes minimum before the head, three for a plain
# link, so a node label holding "HR-TEM" or "1-10" is not read as an arrow.
_LINK = re.compile(r"\s*(?:-{2,}>|-\.-+>|={2,}>|-{3,}|-\.-+)\s*(?:\|([^|]*)\|\s*)?")
# Mermaid's other way of labelling an arrow: the label sits in the middle of the link,
# ``D -- Yes --> E``, rather than in a pipe pair after the head. Half the live figures
# label their decision branches that way, and reading only the pipe form left their
# source printed on the page as the drawing -- the very thing this module exists to
# stop. The dotted and thick links carry theirs the same way, ``-. Yes .->`` and
# ``== Yes ==>``, so all three openers are read and rewritten into the pipe form the
# splitter above already understands.
_MID_LABEL = re.compile(
    r"(?:-{2,}|={2,}|-\.)\s*(?P<label>[^-=.|<>\s][^|<>]*?)\s*"
    r"(?P<head>-{2,}>|={2,}>|\.-+>|-{3,})"
)
_NODE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Longest opener first: "([" has to be tried before "(" or the label keeps a bracket.
_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("([", "])", "round"),
    ("[[", "]]", "box"),
    ("((", "))", "round"),
    ("{{", "}}", "decision"),
    ("[/", "/]", "box"),
    ("[", "]", "box"),
    ("(", ")", "round"),
    ("{", "}", "decision"),
)


@dataclass(frozen=True)
class FlowNode:
    id: str
    label: str
    shape: str


@dataclass(frozen=True)
class FlowEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class Flowchart:
    direction: str
    nodes: list[FlowNode]
    edges: list[FlowEdge]


def _node(token: str, seen: dict[str, FlowNode]) -> FlowNode | None:
    """One ``A[Label]`` reference, which after its first use may be bare ``A``."""
    token = token.strip().rstrip(";").strip()
    if not token:
        return None
    for opener, closer, shape in _SHAPES:
        head, found, rest = token.partition(opener)
        if not found or not rest.endswith(closer):
            continue
        identifier = head.strip()
        if not _NODE_ID.match(identifier):
            return None
        label = _BREAK.sub(" ", rest[: -len(closer)]).strip().strip("\"'").strip()
        node = FlowNode(identifier, " ".join(label.split()) or identifier, shape)
        # A node declared twice keeps its first shape and label, which is where
        # Mermaid puts them too: later references are usually the bare id.
        return seen.setdefault(identifier, node)
    if not _NODE_ID.match(token):
        return None
    return seen.setdefault(token, FlowNode(token, token, "box"))


def _pipe_labels(line: str) -> str:
    """``D -- Yes --> E`` rewritten as ``D -->|Yes| E``, outside node labels.

    A node's own text is left alone. Nothing stops a generator writing an arrow inside
    a bracket, and rewriting there would move a word out of the label it belongs to,
    so the depth of the brackets is tracked and only the gaps between nodes are read.
    """
    depth, inside = 0, []
    for character in line:
        if character in "[({":
            depth += 1
        inside.append(depth > 0)
        if character in "])}":
            depth = max(0, depth - 1)

    def rewrite(match: re.Match[str]) -> str:
        if inside[match.start()]:
            return match.group(0)
        head = match.group("head")
        # The dotted head arrives as ``.->`` with its opening dash consumed above.
        return f"{'-' + head if head.startswith('.') else head}|{match.group('label')}|"

    return _MID_LABEL.sub(rewrite, line)


def parse_mermaid(source: str) -> Flowchart | None:
    """The subset of Mermaid the generators write, or ``None`` for anything else."""
    lines = [line.strip() for line in source.splitlines()]
    lines = [line for line in lines if line and not _IGNORED.match(line)]
    if not lines:
        return None
    header = _HEADER.match(lines[0])
    if not header:
        return None
    direction = header.group(1).upper()
    seen: dict[str, FlowNode] = {}
    edges: list[FlowEdge] = []
    for line in lines[1:]:
        if _UNSUPPORTED.match(line):
            return None
        parts = _LINK.split(_pipe_labels(line))
        if len(parts) == 1:
            # A statement declaring a node without linking it. Anything else on its
            # own line is syntax this parser does not model.
            if _node(line, seen) is None:
                return None
            continue
        # split() with one capture group interleaves: node, label, node, label, node.
        nodes = [_node(part, seen) for part in parts[0::2]]
        if any(node is None for node in nodes):
            return None
        labels = [(part or "").strip() for part in parts[1::2]]
        for index, label in enumerate(labels):
            source_node, target_node = nodes[index], nodes[index + 1]
            assert source_node is not None and target_node is not None
            if source_node.id == target_node.id:
                continue
            edges.append(FlowEdge(source_node.id, target_node.id, label))
    if not edges:
        return None
    return Flowchart(direction, list(seen.values()), edges)


# --- layout -------------------------------------------------------------------

# Points. A box wider than this wraps instead; the gap along the flow axis has to
# hold an edge label, which is why it is more than twice the cross-axis gap.
MAX_LABEL_WIDTH = 148.0
MIN_BOX_WIDTH = 52.0
PAD_X = 9.0
PAD_Y = 6.0
GAP_CROSS = 16.0
GAP_FLOW = 34.0
# A back edge is routed out to a lane beside the drawing rather than straight through
# whatever sits between its ends: the one live diagram with a loop sent its arrow up
# through two boxes it had nothing to do with.
LANE = 24.0
LEADING = 1.18


@dataclass
class PlacedNode:
    node: FlowNode
    lines: list[str]
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class PlacedEdge:
    edge: FlowEdge
    points: list[tuple[float, float]]
    label_at: tuple[float, float] | None = None


@dataclass
class FlowLayout:
    width: float
    height: float
    nodes: list[PlacedNode] = field(default_factory=list)
    edges: list[PlacedEdge] = field(default_factory=list)
    font_size: float = 8.0


def _wrapped(
    label: str, measure: Callable[[str], float], limit: float
) -> tuple[list[str], float]:
    lines: list[str] = []
    current = ""
    for word in label.split():
        trial = f"{current} {word}".strip()
        if current and measure(trial) > limit:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if not lines:
        lines = [label]
    return lines, max(measure(line) for line in lines)


def _layers(chart: Flowchart) -> tuple[list[list[str]], set[tuple[str, str]]] | None:
    """Longest-path layering, with the edges that close a cycle set aside.

    A back edge is found by a depth-first walk: an arrow into a node already open on
    the walk's own stack is the arrow that closes the loop, and it is the one the
    author drew as the loop rather than as the flow.
    """
    order = [node.id for node in chart.nodes]
    outgoing: dict[str, list[str]] = {item: [] for item in order}
    for edge in chart.edges:
        outgoing[edge.source].append(edge.target)
    back: set[tuple[str, str]] = set()
    state: dict[str, int] = {}

    def walk(start: str) -> None:
        stack: list[tuple[str, int]] = [(start, 0)]
        state[start] = 1
        while stack:
            node, index = stack.pop()
            if index >= len(outgoing[node]):
                state[node] = 2
                continue
            stack.append((node, index + 1))
            nxt = outgoing[node][index]
            if state.get(nxt) == 1:
                back.add((node, nxt))
            elif state.get(nxt) is None:
                state[nxt] = 1
                stack.append((nxt, 0))

    incoming_count = {item: 0 for item in order}
    for edge in chart.edges:
        incoming_count[edge.target] += 1
    for item in order:
        if not incoming_count[item] and state.get(item) is None:
            walk(item)
    for item in order:
        if state.get(item) is None:
            walk(item)

    forward = [(edge.source, edge.target) for edge in chart.edges]
    forward = [pair for pair in forward if pair not in back]
    depth = {item: 0 for item in order}
    for _ in range(len(order)):
        changed = False
        for source, target in forward:
            if depth[target] < depth[source] + 1:
                depth[target] = depth[source] + 1
                changed = True
        if not changed:
            break
    else:
        return None
    rows: list[list[str]] = [[] for _ in range(max(depth.values()) + 1)]
    for item in order:
        rows[depth[item]].append(item)
    return rows, back


def _ordered(rows: list[list[str]], edges: list[FlowEdge]) -> None:
    """Put each row in its parents' order, so the arrows between rows cross less."""
    parents: dict[str, list[str]] = {}
    for edge in edges:
        parents.setdefault(edge.target, []).append(edge.source)
    for index in range(1, len(rows)):
        above = {item: position for position, item in enumerate(rows[index - 1])}
        appearance = {item: position for position, item in enumerate(rows[index])}
        rows[index].sort(
            key=lambda item: (
                sum(
                    above[parent] for parent in parents.get(item, []) if parent in above
                )
                / max(1, len([p for p in parents.get(item, []) if p in above]))
                if any(parent in above for parent in parents.get(item, []))
                else appearance[item],
                appearance[item],
            )
        )


def _boundary(box: PlacedNode, toward: tuple[float, float]) -> tuple[float, float]:
    """Where the line from the box's centre to ``toward`` leaves the box."""
    cx, cy = box.center
    dx, dy = toward[0] - cx, toward[1] - cy
    if not dx and not dy:
        return (cx, cy)
    half_w, half_h = box.width / 2, box.height / 2
    if box.node.shape == "decision":
        # |dx|/w + |dy|/h = 1 is the diamond's own edge; a rectangle's would leave a
        # visible gap at every corner of it.
        scale = 1.0 / (abs(dx) / half_w + abs(dy) / half_h)
    else:
        candidates = []
        if dx:
            candidates.append(half_w / abs(dx))
        if dy:
            candidates.append(half_h / abs(dy))
        scale = min(candidates)
    return (cx + dx * scale, cy + dy * scale)


def layout_flowchart(
    chart: Flowchart,
    *,
    measure: Callable[[str, float], float],
    font_size: float = 8.0,
) -> FlowLayout | None:
    """Place every node and route every arrow, in points with y running downward."""
    layered = _layers(chart)
    if layered is None:
        return None
    rows, back = layered
    _ordered(
        rows, [edge for edge in chart.edges if (edge.source, edge.target) not in back]
    )
    by_id = {node.id: node for node in chart.nodes}
    placed: dict[str, PlacedNode] = {}
    line_height = font_size * LEADING
    for identifier in by_id:
        node = by_id[identifier]
        lines, widest = _wrapped(
            node.label,
            lambda text: measure(text, font_size),
            MAX_LABEL_WIDTH - 2 * PAD_X,
        )
        width = max(MIN_BOX_WIDTH, widest + 2 * PAD_X)
        height = len(lines) * line_height + 2 * PAD_Y
        if node.shape == "decision":
            # A diamond only holds half the area its bounding box does.
            width, height = width * 1.5 + 8, height * 1.6 + 4
        placed[identifier] = PlacedNode(node, lines, 0.0, 0.0, width, height)

    flow = 0.0
    content = 0.0
    for row in rows:
        span = sum(placed[item].width for item in row) + GAP_CROSS * (len(row) - 1)
        content = max(content, span)
    for row in rows:
        span = sum(placed[item].width for item in row) + GAP_CROSS * (len(row) - 1)
        cross = (content - span) / 2
        tallest = max(placed[item].height for item in row)
        for item in row:
            box = placed[item]
            box.x = cross
            box.y = flow + (tallest - box.height) / 2
            cross += box.width + GAP_CROSS
        flow += tallest + GAP_FLOW
    height = flow - GAP_FLOW
    lane = LANE if back else 0.0

    routed: list[PlacedEdge] = []
    for edge in chart.edges:
        head, tail = placed[edge.source], placed[edge.target]
        if (edge.source, edge.target) in back:
            lane_x = content + lane / 2
            start = (head.x + head.width, head.center[1])
            end = (tail.x + tail.width, tail.center[1])
            points = [start, (lane_x, start[1]), (lane_x, end[1]), end]
            routed.append(PlacedEdge(edge, points, (lane_x, (start[1] + end[1]) / 2)))
            continue
        start = _boundary(head, tail.center)
        end = _boundary(tail, head.center)
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        routed.append(PlacedEdge(edge, [start, end], middle if edge.label else None))

    result = FlowLayout(
        content + lane, height, list(placed.values()), routed, font_size
    )
    if chart.direction in {"LR", "RL"}:
        result = _transposed(result)
    if chart.direction in {"BT", "RL"}:
        result = _reversed_axis(result, chart.direction)
    return result


def _transposed(layout: FlowLayout) -> FlowLayout:
    """The same graph laid out along the other axis, which is what LR asks for."""
    for box in layout.nodes:
        box.x, box.y = box.y, box.x
    for link in layout.edges:
        link.points = [(y, x) for x, y in link.points]
        if link.label_at:
            link.label_at = (link.label_at[1], link.label_at[0])
    layout.width, layout.height = layout.height, layout.width
    return layout


def _reversed_axis(layout: FlowLayout, direction: str) -> FlowLayout:
    """BT and RL are TD and LR read from the far end."""
    vertical = direction == "BT"
    for box in layout.nodes:
        if vertical:
            box.y = layout.height - box.y - box.height
        else:
            box.x = layout.width - box.x - box.width
    for link in layout.edges:
        link.points = [
            (x, layout.height - y) if vertical else (layout.width - x, y)
            for x, y in link.points
        ]
        if link.label_at:
            x, y = link.label_at
            link.label_at = (
                (x, layout.height - y) if vertical else (layout.width - x, y)
            )
    return layout


# --- exporters ----------------------------------------------------------------

_BOX_FILL = "#f1f3f4"
_DECISION_FILL = "#e8f0fe"
_STROKE = "#5f6368"
_TEXT = "#202124"


def flowchart_drawing(source: str, available: float, *, font: str = "Helvetica"):
    """The diagram as a reportlab drawing, or ``None`` if the source is not one."""
    chart = parse_mermaid(source)
    if chart is None:
        return None
    from reportlab.graphics.shapes import (
        Drawing,
        Group,
        Line,
        Polygon,
        Rect,
        String,
    )
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics

    font_size = 8.0
    layout = layout_flowchart(
        chart,
        measure=lambda text, size: pdfmetrics.stringWidth(text, font, size),
        font_size=font_size,
    )
    if layout is None or not layout.nodes:
        return None
    # Only ever shrunk. A four-box diagram blown up to the width of the frame reads
    # as a poster, and the type inside it would grow with the boxes.
    scale = min(1.0, available / layout.width) if layout.width else 1.0
    group = Group()
    line_height = font_size * LEADING

    def at(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0], layout.height - point[1])

    for link in layout.edges:
        points = [at(point) for point in link.points]
        for start, end in pairwise(points):
            group.add(
                Line(
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    strokeColor=colors.HexColor(_STROKE),
                    strokeWidth=0.8,
                )
            )
        group.add(_arrow_head(points[-2], points[-1], Polygon, colors))
        if link.label_at and link.edge.label:
            x, y = at(link.label_at)
            width = pdfmetrics.stringWidth(link.edge.label, font, font_size - 1) + 4
            group.add(
                Rect(
                    x - width / 2,
                    y - (font_size - 1) / 2 - 1,
                    width,
                    font_size + 1,
                    fillColor=colors.white,
                    strokeColor=None,
                )
            )
            group.add(
                String(
                    x,
                    y - (font_size - 1) / 3,
                    link.edge.label,
                    fontName=font,
                    fontSize=font_size - 1,
                    fillColor=colors.HexColor(_STROKE),
                    textAnchor="middle",
                )
            )

    for box in layout.nodes:
        cx, cy = box.center
        cy = layout.height - cy
        fill = colors.HexColor(
            _DECISION_FILL if box.node.shape == "decision" else _BOX_FILL
        )
        if box.node.shape == "decision":
            group.add(
                Polygon(
                    [
                        cx,
                        cy + box.height / 2,
                        cx + box.width / 2,
                        cy,
                        cx,
                        cy - box.height / 2,
                        cx - box.width / 2,
                        cy,
                    ],
                    fillColor=fill,
                    strokeColor=colors.HexColor(_STROKE),
                    strokeWidth=0.8,
                )
            )
        else:
            group.add(
                Rect(
                    box.x,
                    layout.height - box.y - box.height,
                    box.width,
                    box.height,
                    rx=4 if box.node.shape == "round" else 0,
                    ry=4 if box.node.shape == "round" else 0,
                    fillColor=fill,
                    strokeColor=colors.HexColor(_STROKE),
                    strokeWidth=0.8,
                )
            )
        top = cy + len(box.lines) * line_height / 2
        for index, text in enumerate(box.lines):
            group.add(
                String(
                    cx,
                    top - line_height * (index + 0.8),
                    text,
                    fontName=font,
                    fontSize=font_size,
                    fillColor=colors.HexColor(_TEXT),
                    textAnchor="middle",
                )
            )

    group.transform = (scale, 0, 0, scale, 0, 0)
    drawing = Drawing(layout.width * scale, layout.height * scale)
    drawing.add(group)
    return drawing


def _arrow_head(start, end, polygon_class, colors):
    length, half = 6.5, 3.0
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / span, dy / span
    base = (end[0] - ux * length, end[1] - uy * length)
    return polygon_class(
        [
            end[0],
            end[1],
            base[0] - uy * half,
            base[1] + ux * half,
            base[0] + uy * half,
            base[1] - ux * half,
        ],
        fillColor=colors.HexColor(_STROKE),
        strokeColor=colors.HexColor(_STROKE),
    )


def flowchart_steps(source: str) -> list[str] | None:
    """The same graph as steps, for the export that cannot hold a drawing.

    Word takes an image or nothing, and the deployment carries no rasteriser, so the
    choice there is between the arrows read out and the Mermaid source printed. The
    arrows read out is a workflow; the source is a listing of one.
    """
    chart = parse_mermaid(source)
    if chart is None:
        return None
    labels = {node.id: node.label for node in chart.nodes}
    steps = []
    for edge in chart.edges:
        head, tail = labels[edge.source], labels[edge.target]
        condition = f" (if {edge.label.strip()})" if edge.label.strip() else ""
        steps.append(f"{head}{condition} → {tail}")
    return steps
