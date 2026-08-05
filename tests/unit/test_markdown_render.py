from coscientist.markdown_render import (
    Code,
    Details,
    Heading,
    ListBlock,
    Para,
    Quote,
    Rule,
    Span,
    Table,
    cjk_markup,
    has_cjk,
    inline_markup,
    parse_blocks,
    parse_inline,
    plain_text,
)


def test_inline_parser_flattens_nested_styles():
    spans = parse_inline("plain **bold with *inner* text** tail")

    assert Span("plain ") in spans
    assert Span("bold with ", bold=True) in spans
    assert Span("inner", bold=True, italic=True) in spans
    assert Span(" tail") in spans


def test_inline_parser_keeps_code_spans_literal():
    spans = parse_inline("id `deep_research_unavailable` here")

    assert Span("deep_research_unavailable", code=True) in spans


def test_inline_markup_escapes_before_converting():
    markup = inline_markup("a < b & c > d **bold**")

    assert markup == "a &lt; b &amp; c &gt; d <b>bold</b>"


def test_inline_markup_emits_reportlab_tags_in_precedence_order():
    markup = inline_markup("**b** *i* `c` [t](http://x?a=1&b=2)")

    assert "<b>b</b>" in markup
    assert "<i>i</i>" in markup
    assert '<font face="Courier" size="-1.2">c</font>' in markup
    assert '<link href="http://x?a=1&amp;b=2" color="blue">t</link>' in markup


def test_intraword_underscores_are_not_emphasis():
    assert inline_markup("snake_case_name") == "snake_case_name"
    assert inline_markup("a _real_ emphasis") == "a <i>real</i> emphasis"


def test_plain_text_strips_all_marks():
    assert plain_text("**a** `b` [c](http://d)") == "a b c"


def test_cjk_detection_and_font_tagging():
    assert has_cjk("鉴定治疗") and not has_cjk("PD-1")
    assert cjk_markup("PD-1 耐药 assay", "STSong-Light") == (
        'PD-1 <font face="STSong-Light">耐药</font> assay'
    )
    assert cjk_markup("latin only", "STSong-Light") == "latin only"


def test_table_parser_reads_header_rows_and_alignment():
    (table,) = parse_blocks(
        "| Rank | Candidate | Elo |\n"
        "| ---: | --- | :---: |\n"
        "| 1 | `cand_a` | 1500.0 |\n"
        "| 2 | `cand_b` | 1484.0 |\n"
    )

    assert isinstance(table, Table)
    assert table.header == ["Rank", "Candidate", "Elo"]
    assert table.aligns == ["right", "left", "center"]
    assert table.rows == [["1", "`cand_a`", "1500.0"], ["2", "`cand_b`", "1484.0"]]


def test_table_parser_pads_and_trims_ragged_rows():
    (table,) = parse_blocks("| a | b |\n| --- | --- |\n| 1 |\n| 1 | 2 | 3 |\n")

    assert table.rows == [["1", ""], ["1", "2"]]


def test_pipe_without_separator_is_not_a_table():
    blocks = parse_blocks("| not a table\nstill prose\n")

    assert [type(block) for block in blocks] == [Para]


def test_list_parser_tracks_nesting_and_ordering():
    (block,) = parse_blocks("- a\n  - b\n    - c\n- d\n")

    assert isinstance(block, ListBlock)
    assert [(item.level, item.text) for item in block.items] == [
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (0, "d"),
    ]
    assert not any(item.ordered for item in block.items)


def test_ordered_list_is_recognised():
    (block,) = parse_blocks("1. first\n2. second\n")

    assert all(item.ordered for item in block.items)


def test_block_parser_covers_every_construct():
    blocks = parse_blocks(
        "# Title\n\n"
        "Prose line one\nProse line two\n\n"
        "> quoted\n\n"
        "---\n\n"
        '```json\n{"a": 1}\n```\n\n'
        "<details><summary>Payload</summary>\n\n"
        "```json\n{}\n```\n\n"
        "</details>\n"
    )

    kinds = [type(block) for block in blocks]
    assert kinds == [Heading, Para, Quote, Rule, Code, Details]
    assert blocks[0].level == 1
    assert blocks[1].text == "Prose line one\nProse line two"
    assert blocks[2].text == "quoted"
    assert blocks[4].language == "json"
    assert blocks[5].summary == "Payload"
    assert isinstance(blocks[5].blocks[0], Code)


def test_unfenced_json_becomes_a_code_block():
    (block,) = parse_blocks('{\n  "question": "why",\n  "n": [1, 2]\n}\n')

    assert isinstance(block, Code)
    assert block.text.startswith("{") and block.text.rstrip().endswith("}")
