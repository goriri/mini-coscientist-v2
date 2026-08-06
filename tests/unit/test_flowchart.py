"""The specialist's workflow diagram, drawn instead of listed.

Every generator returns ``workflow_diagram_mermaid`` and every export printed it as
its own source: eleven lines of ``F -->|Yes| H[...]`` set in Courier under a caption
reading "Figure 1. Proposed Workflow". These pin the parse, the layout and the two
ways out -- vector shapes for the PDF, the arrows read out for Word -- and, as
importantly, pin the fallback: a source this parser does not recognise goes back to
being printed verbatim rather than half-drawn.
"""

from __future__ import annotations

from io import BytesIO

from coscientist.flowchart import (
    flowchart_drawing,
    flowchart_steps,
    layout_flowchart,
    parse_mermaid,
)

LIVE = """graph TD
    A[NMC811 Powder] --> B[Powder ALD LiNbO3 2.5nm]
    A --> C[Uncoated Control]
    B --> D[HR-TEM/XPS Check]
    D --> E[Blinded Coin Cell Assembly n=10]
    C --> E
    E --> F{O2/CO2 Reduced >70%?}
    F -->|Yes| G[Randomized Long-term Cycling]
    F -->|No| H[Falsified]
"""

LOOP = """graph TD
    A[NMC811 Cathode] --> B[ALD TMA/H2O Cycle]
    B --> C[0.5 nm Al2O3 Layer]
    C --> F{Repeat 5x}
    F -->|Yes| B
    F -->|No| G[2.5 nm Nanolaminate Coating]
"""


def _measure(text: str, size: float) -> float:
    """A stand-in for the font metrics, so the layout tests need no PDF backend."""
    return len(text) * size * 0.5


def test_the_nodes_and_arrows_of_a_live_diagram_are_read_off_it():
    chart = parse_mermaid(LIVE)

    assert chart is not None
    assert chart.direction == "TD"
    assert [node.id for node in chart.nodes] == list("ABCDEFGH")
    assert {node.id: node.label for node in chart.nodes}["F"] == "O2/CO2 Reduced >70%?"
    assert [edge.label for edge in chart.edges if edge.label] == ["Yes", "No"]


def test_a_brace_node_is_a_decision_and_a_bracket_node_is_a_step():
    chart = parse_mermaid(LIVE)

    shapes = {node.id: node.shape for node in chart.nodes}
    assert shapes["F"] == "decision"
    assert shapes["A"] == "box"


def test_a_node_named_once_keeps_its_label_where_it_is_referred_to_bare():
    """ "C --> E" carries neither label, and both were declared a line earlier."""
    chart = parse_mermaid(LIVE)

    labels = {node.id: node.label for node in chart.nodes}
    assert labels["C"] == "Uncoated Control"
    assert labels["E"] == "Blinded Coin Cell Assembly n=10"


def test_a_chain_written_on_one_line_is_read_as_two_arrows():
    chart = parse_mermaid("graph LR\n  A[One] --> B[Two] -->|then| C[Three]\n")

    assert [(edge.source, edge.target, edge.label) for edge in chart.edges] == [
        ("A", "B", ""),
        ("B", "C", "then"),
    ]


def test_a_hyphen_inside_a_label_is_not_read_as_an_arrow():
    """ "HR-TEM/XPS Check" and "Cycles 1-10" are labels, not links."""
    chart = parse_mermaid("graph TD\n  A[HR-TEM/XPS Check] --> B[Cycles 1-10]\n")

    assert [node.label for node in chart.nodes] == ["HR-TEM/XPS Check", "Cycles 1-10"]


def test_styling_statements_and_comments_are_skipped_rather_than_refused():
    chart = parse_mermaid(
        "graph TD\n  %% the happy path\n  A[One] --> B[Two]\n"
        "  style A fill:#f9f\n  classDef done fill:#bbf\n"
    )

    assert chart is not None and len(chart.edges) == 1


def test_something_that_is_not_a_flowchart_is_not_guessed_at():
    for source in (
        "sequenceDiagram\n  Alice->>John: Hello\n",
        "graph TD\n  subgraph Phase One\n  A[One] --> B[Two]\n  end\n",
        '{"question": "why"}',
        "graph TD\n  A[One]\n",
    ):
        assert parse_mermaid(source) is None, source


def test_the_layout_puts_every_node_in_the_row_its_arrows_put_it_in():
    chart = parse_mermaid(LIVE)
    layout = layout_flowchart(chart, measure=_measure)

    rows: dict[str, float] = {box.node.id: box.y for box in layout.nodes}
    assert rows["A"] < rows["B"] == rows["C"] < rows["D"]
    assert rows["E"] < rows["F"] < rows["G"] == rows["H"]


def test_no_two_boxes_overlap():
    layout = layout_flowchart(parse_mermaid(LIVE), measure=_measure)

    for first in layout.nodes:
        for second in layout.nodes:
            if first is second:
                continue
            apart = (
                first.x + first.width <= second.x
                or second.x + second.width <= first.x
                or first.y + first.height <= second.y
                or second.y + second.height <= first.y
            )
            assert apart, f"{first.node.id} overlaps {second.node.id}"


def test_a_loop_is_routed_beside_the_drawing_and_not_through_it():
    """ "F -->|Yes| B" runs back up past two boxes it has nothing to do with, and a
    straight line between the two would be drawn over both of them."""
    chart = parse_mermaid(LOOP)
    layout = layout_flowchart(chart, measure=_measure)

    boxes = {box.node.id: box for box in layout.nodes}
    assert boxes["B"].y < boxes["F"].y, "the loop's target sits above its source"
    (back,) = [
        link
        for link in layout.edges
        if (link.edge.source, link.edge.target) == ("F", "B")
    ]
    assert len(back.points) == 4, "the back edge is not routed around anything"
    lane = max(point[0] for point in back.points)
    assert lane > max(box.x + box.width for box in layout.nodes) - 0.01
    assert back.points[-1][1] == boxes["B"].center[1], "it arrives at the wrong box"


def test_a_left_to_right_diagram_runs_left_to_right():
    chart = parse_mermaid("graph LR\n  A[One] --> B[Two] --> C[Three]\n")
    layout = layout_flowchart(chart, measure=_measure)

    boxes = {box.node.id: box for box in layout.nodes}
    assert boxes["A"].x < boxes["B"].x < boxes["C"].x
    assert boxes["A"].y == boxes["B"].y == boxes["C"].y


def test_a_bottom_to_top_diagram_ends_at_the_top():
    chart = parse_mermaid("graph BT\n  A[Start] --> B[Middle] --> C[End]\n")
    layout = layout_flowchart(chart, measure=_measure)

    boxes = {box.node.id: box for box in layout.nodes}
    assert boxes["C"].y < boxes["B"].y < boxes["A"].y


def test_a_long_label_wraps_rather_than_widening_the_page():
    chart = parse_mermaid(
        "graph TD\n  A[Blinded randomized long-term cycling at one C for five "
        "hundred cycles at twenty five degrees] --> B[Done]\n"
    )
    layout = layout_flowchart(chart, measure=_measure)

    (wide,) = [box for box in layout.nodes if box.node.id == "A"]
    assert len(wide.lines) > 1
    assert wide.width <= 150


def test_the_drawing_is_shapes_and_words_rather_than_the_source_text():
    from reportlab.graphics.shapes import Polygon, Rect, String

    drawing = flowchart_drawing(LIVE, 460)

    assert drawing is not None
    (group,) = drawing.contents
    kinds = {type(shape).__name__ for shape in group.contents}
    said = [shape.text for shape in group.contents if isinstance(shape, String)]

    assert {"Line", "Rect", "Polygon"} <= kinds
    assert "Falsified" in said and "Yes" in said
    assert not any("-->" in text for text in said)
    # The diamond and the arrowheads are polygons; the steps are rectangles.
    assert sum(isinstance(shape, Polygon) for shape in group.contents) >= 9
    assert sum(isinstance(shape, Rect) for shape in group.contents) >= 7


def test_a_drawing_wider_than_the_frame_is_scaled_down_and_never_up():
    narrow = flowchart_drawing(LIVE, 120)
    roomy = flowchart_drawing(LIVE, 900)

    assert narrow.width <= 120.001
    assert roomy.width == flowchart_drawing(LIVE, 460).width, (
        "a four-box diagram was blown up to the width of the frame"
    )


def test_a_source_the_parser_declines_produces_no_drawing_at_all():
    assert flowchart_drawing("sequenceDiagram\n  A->>B: hi\n", 460) is None


def test_word_gets_the_arrows_read_out_in_order():
    steps = flowchart_steps(LIVE)

    assert steps[0] == "NMC811 Powder → Powder ALD LiNbO3 2.5nm"
    assert "O2/CO2 Reduced >70%? (if Yes) → Randomized Long-term Cycling" in steps
    assert flowchart_steps("not a diagram") is None


def test_the_pdf_export_draws_the_diagram_instead_of_printing_its_source():
    from pypdf import PdfReader

    from coscientist.dossier import render_pdf

    rendered = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(
            BytesIO(render_pdf(f"# D\n\n## Workflow\n\n```mermaid\n{LIVE}```\n"))
        ).pages
    )

    assert "Uncoated Control" in rendered
    assert "-->" not in rendered and "graph TD" not in rendered
    assert "Figure 1." in rendered


def test_an_unparseable_fence_still_reaches_the_pdf_verbatim():
    from pypdf import PdfReader

    from coscientist.dossier import render_pdf

    source = "sequenceDiagram\n    Alice->>John: Hello John\n"
    rendered = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(
            BytesIO(render_pdf(f"# D\n\n## Workflow\n\n```mermaid\n{source}```\n"))
        ).pages
    )

    assert "sequenceDiagram" in rendered
