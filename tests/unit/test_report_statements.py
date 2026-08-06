"""What a specialist writes into a prose field, printed as prose.

Three defects on the run that finished today, all of them the report repeating a
model's output shape instead of the field's declared one. A specialist answered a
prose field with a Markdown table, and the table went through the sentence writer
as one line of pipes and dashes. The same finding came back from more than one
Deep Research pass and was printed once per pass, sometimes tagged as supporting
in one copy and contradicting in another with nothing saying so. And a research
direction restating the run's own question was listed as a direction to pursue.
"""

from __future__ import annotations

from coscientist.models import (
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    Session,
)
from coscientist.narrative import (
    ResearchRecord,
    _blocks_as_prose,
    _evidence_statements,
    _sentence,
    synthesize_overview,
)

TABLE = """High-nickel cathodes suffer from instability.

## Evaluation of Idea

| Criterion | Description | Judgment |
| --- | --- | --- |
| Aggregation Control | Fluidized bed ALD ensures discrete particle coating | High |
| Purity Potential | Amorphous, defect-free layer | Moderate |
"""


def test_a_table_answering_a_prose_field_is_read_out_as_clauses():
    said = _blocks_as_prose(TABLE)

    assert "|" not in said and "---" not in said
    assert (
        "Aggregation Control (Description: Fluidized bed ALD ensures discrete "
        "particle coating; Judgment: High)." in said
    )
    assert "Evaluation of Idea." in said, "the heading was dropped rather than said"


def test_a_two_column_row_is_read_as_a_colon_rather_than_a_parenthesis():
    said = _blocks_as_prose(
        "| Criterion | Judgment |\n| --- | --- |\n| Purity | High |"
    )

    assert said == "Purity: High."


def test_a_row_with_nothing_but_a_name_keeps_the_name():
    said = _blocks_as_prose("| Criterion | Judgment |\n| --- | --- |\n| Purity |  |")

    assert said == "Purity."


def test_ordinary_prose_is_left_exactly_as_written():
    text = "The coating blocks the reaction that drives fade."

    assert _blocks_as_prose(text) == text


def test_the_sentence_writer_collapses_a_table_into_one_line_of_prose():
    said = _sentence(TABLE)

    assert "\n" not in said
    assert said.startswith("High-nickel cathodes suffer from instability.")
    assert said.endswith("Judgment: Moderate).")


def _statement(
    text: str, *, relation: str = "supports", url: str = ""
) -> DiscoveryStatement:
    return DiscoveryStatement(
        text=text,
        facet="supporting",
        relation=relation,
        source_urls=[url] if url else [],
        originating_pass=1,
    )


def _record(*statements: DiscoveryStatement) -> ResearchRecord:
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.discovery = DiscoveryManifest(
        question="Can a coating help?",
        narratives=[
            DiscoveryNarrative(
                question="Can a coating help?",
                summary="A report.",
                statements=list(statements),
            )
        ],
    )
    return record


FINDING = "An alumina coating raises retention by ten points."


def test_one_finding_returned_by_two_passes_is_printed_once():
    statements = _evidence_statements(
        _record(
            _statement(FINDING, url="https://example.org/a"),
            _statement(FINDING, url="https://example.org/b"),
        )
    )

    assert len(statements) == 1
    assert statements[0].urls == ["https://example.org/a", "https://example.org/b"]


def test_two_copies_that_disagree_about_direction_say_that_they_disagree():
    """Printed twice, one tagged supporting and one contradicting, the reader has
    two findings and no way to tell that they are the same sentence."""
    statements = _evidence_statements(
        _record(
            _statement(FINDING, relation="supports"),
            _statement(FINDING, relation="contradicts"),
        )
    )

    assert len(statements) == 1
    assert statements[0].relation == "recorded_both_ways"


def test_copies_that_agree_keep_the_direction_they_agree_on():
    statements = _evidence_statements(
        _record(
            _statement(FINDING, relation="contradicts"),
            _statement(FINDING, relation="contradicts"),
        )
    )

    assert [item.relation for item in statements] == ["contradicts"]


def test_a_finding_with_no_text_is_not_carried_as_an_empty_entry():
    assert not _evidence_statements(_record(_statement("   ")))


def _directions(record: ResearchRecord) -> list[str]:
    return synthesize_overview(record).research_directions


def test_a_research_direction_restating_the_question_is_not_listed_as_one():
    record = _record(_statement(FINDING))
    record.discovery.narratives[0].research_directions = [
        "Can a coating help?",
        "Measure the coating thickness that stops losing capacity.",
    ]

    said = " ".join(_directions(record))

    assert "Measure the coating thickness" in said
    assert "Can a coating help" not in said


def test_the_same_direction_returned_by_two_passes_is_listed_once():
    record = _record(_statement(FINDING))
    record.discovery.narratives[0].research_directions = [
        "Measure the coating thickness that stops losing capacity.",
        "Measure the coating thickness that stops losing capacity.",
    ]

    assert len(_directions(record)) == 1
