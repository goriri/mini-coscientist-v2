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
    IdeaReview,
    ResearchRecord,
    _blocks_as_prose,
    _coherence,
    _evidence_statements,
    _section_three,
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


# --- the relation clause is a property of the group, not of each finding ------


def _directions_section(*statements: DiscoveryStatement) -> str:
    return "\n".join(_section_three(_record(*statements)).core)


ALPHA = "An alumina coating raises retention by ten points."
BETA = "A titania coating raises retention by four points."
GAMMA = "Retention falls once the coating passes forty nanometres."


def test_a_relation_the_findings_share_is_stated_once_over_them():
    """ "Discovery returned this finding more than once and read it differently each
    time" stood under eleven consecutive findings of a live report, with the same
    sentence for the neutral ones four times further down."""
    said = _directions_section(
        _statement(ALPHA, relation="contradicts"),
        _statement(BETA, relation="contradicts"),
        _statement(GAMMA, relation="contradicts"),
    )

    assert said.count("arguing against the hypothesis") == 1
    assert "Discovery recorded the next three findings as arguing against" in said
    assert "Discovery recorded this finding as arguing against" not in said


def test_a_relation_carried_by_one_finding_stays_under_that_finding():
    """A lead-in over a group of one is a paragraph saying what the next one says."""
    said = _directions_section(
        _statement(ALPHA),
        _statement(BETA, relation="contradicts"),
    )

    assert "Discovery recorded this finding as arguing against" in said
    assert "Discovery recorded the next" not in said


def test_the_findings_that_share_a_relation_are_printed_together():
    """Scattered through the section, a reader looking for the case against the goal
    has to read every finding to find the three that make it."""
    said = _directions_section(
        _statement(ALPHA, relation="contradicts"),
        _statement(BETA),
        _statement(GAMMA, relation="contradicts"),
    )

    assert said.index(BETA) < said.index(ALPHA) < said.index(GAMMA)


def test_each_way_the_findings_were_read_gets_its_own_lead_in():
    said = _directions_section(
        _statement(ALPHA, relation="neutral"),
        _statement(BETA, relation="neutral"),
        _statement(GAMMA, relation="contradicts"),
    )

    assert "Discovery recorded the next two findings as bearing on the question" in said
    assert "Discovery recorded this finding as arguing against" in said


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


# --- coherence: the falsifier settles a dispute only where there is one -------


def _review(section: str, score: int, recommendation: str = "revise") -> IdeaReview:
    return IdeaReview(
        section=section,
        lead_in="",
        question="",
        findings=[],
        objections=[],
        rebuttals=[],
        answer="",
        score=score,
        recommendation=recommendation,
    )


def test_a_falsifier_settles_a_disagreement_only_where_the_reviews_disagree():
    lines, _ = _coherence(
        [_review("Feasibility", 2), _review("Correctness", 5)],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert "settle the disagreement between the reviews above" in lines[-1]


def test_a_falsifier_under_agreeing_reviews_tests_the_reading_they_share():
    """ "the disagreement between the reviews above" printed two lines under "They
    agree" -- the sentence named a dispute the paragraph above had just denied."""
    lines, _ = _coherence(
        [_review("Correctness", 4), _review("Feasibility", 4)],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert "put the reading they share to the test" in lines[-1]
    assert "disagreement" not in lines[-1]


def test_reviews_that_score_alike_but_ask_for_different_things_still_disagree():
    lines, _ = _coherence(
        [
            _review("Correctness", 4, "revise"),
            _review("Feasibility", 4, "advance"),
        ],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert "settle the disagreement between the reviews above" in lines[-1]


def test_agreeing_reviews_with_no_falsifier_are_not_told_they_disagree():
    lines, _ = _coherence(
        [_review("Correctness", 4), _review("Feasibility", 4)],
        {"Falsifier": "Not stated by the specialist."},
    )

    assert "test the reading the reviews share" in lines[-1]
    assert (
        "Their agreement is about a claim nothing can yet be held against"
        in (lines[-1])
    )
