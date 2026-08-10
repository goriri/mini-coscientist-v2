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

import dataclasses

from coscientist.models import (
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    Session,
)
from coscientist.narrative import (
    IdeaReview,
    ResearchRecord,
    _authors_own_parts,
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


def test_a_heading_over_prose_introduces_it_rather_than_standing_alone():
    """ "... isolate the true kinetic penalty of the coating. Critical Scientific
    Judgment. TMA is highly pyrophoric" -- a title stranded mid-paragraph between two
    sentences it belongs to neither of."""
    said = _sentence(
        "The binder masks the penalty.\n\n"
        "## Critical Scientific Judgment\n\n"
        "TMA is pyrophoric. Cutoffs must be set to 4.5 V."
    )

    assert "Critical Scientific Judgment: TMA is pyrophoric." in said
    assert "Judgment. TMA" not in said


def test_a_heading_over_a_table_keeps_its_own_full_stop():
    """The rows below are sentences of their own, and a colon would read as though
    only the first of them were what the heading introduced."""
    said = _sentence(TABLE)

    assert "Evaluation of Idea. Aggregation Control" in said


def test_a_field_name_out_of_the_contract_is_printed_as_words():
    """ "included a structured Evaluation Table in the mechanism_model" -- an
    identifier out of a JSON schema, printed to a reader who has never seen it."""
    said = _sentence("Added a table to mechanism_model and a test to go_no_go_tests.")

    assert said == "Added a table to mechanism model and a test to go/no-go tests."


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


def test_a_space_left_in_front_of_a_comma_is_closed_up():
    """ "in the voltage range of 2-4.8 V , compared to" stood in four places of a
    live report, each a quotation of the same claim. The space is an artefact of the
    field the sentence arrived in, not part of what was said."""
    said = _sentence("Retention held at 2-4.8 V , and rose at 30 C .")

    assert said == "Retention held at 2-4.8 V, and rose at 30 C."


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


def test_a_finding_arrives_without_the_tags_the_pass_wrote_for_the_pipeline():
    """The reproduced pass reports have these stripped on the way in, and these are
    the same sentences out of the same reports. A live run opened Main Research
    Directions on "... is the sample size requirement. `[Facet: contradictory]`"."""
    statements = _evidence_statements(
        _record(_statement("Retention falls past forty nanometres. `[Facet: methods]`"))
    )

    assert [item.text for item in statements] == [
        "Retention falls past forty nanometres."
    ]


def test_the_same_finding_tagged_in_one_copy_and_not_the_other_is_one_finding():
    """The merge keys on the text, so a tag left on one copy made two findings."""
    statements = _evidence_statements(
        _record(
            _statement(FINDING),
            _statement(f"{FINDING.rstrip('.')} [cite: 2]."),
        )
    )

    assert len(statements) == 1


# --- the relation clause is a property of the group, not of each finding ------


def _directions_section(*statements: DiscoveryStatement) -> str:
    return "\n".join(_section_three(_record(*statements)).core)


ALPHA = "An alumina coating raises retention by ten points."
BETA = "A titania coating raises retention by four points."
GAMMA = "Retention falls once the coating passes forty nanometres."


def test_the_findings_under_the_directions_heading_are_introduced_as_findings():
    """Section three opens on the research directions and points at a list of them a
    page below -- and then prints thirty-five cited findings with nothing between the
    two saying what they are. Under the same heading, the reader meets the first
    finding as though the report had changed the subject without saying so."""
    record = _record(
        _statement(ALPHA, url="https://example.org/a"),
        _statement(BETA, url="https://example.org/b"),
    )
    record.discovery.narratives[0].research_directions = [
        "Interfacial coatings on high-nickel cathodes."
    ]

    said = "\n".join(_section_three(record).core)

    assert "listed under Research directions below" in said
    assert "two findings from the literature" in said
    assert said.index("two findings") < said.index("An alumina coating")


def test_nothing_introduces_the_findings_twice_where_the_opening_already_did():
    """The opening of a run with no directions already says what follows is individual
    findings, and a second sentence saying it would be the paragraph repeated."""
    said = _directions_section(_statement(ALPHA))

    assert "what follows is individual findings" in said
    assert "from the literature, set out below" not in said


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


def test_reviews_that_all_scored_alike_are_not_reported_as_a_span():
    """ "The five reviews of this idea span 5 to 5 of five" is a range with one end,
    and takes three numbers to say what one says."""
    lines, _ = _coherence([_review("Correctness", 5), _review("Feasibility", 5)], {})

    assert "span 5 to 5" not in lines[0]
    assert "all came in at 5 of five" in lines[0]


def test_reviews_that_did_not_score_alike_still_report_both_ends():
    lines, _ = _coherence([_review("Correctness", 2), _review("Feasibility", 5)], {})

    assert "span 2 to 5 of five" in lines[0]


def test_what_a_recorded_falsifier_is_for_is_said_once_above_the_ideas():
    """It was said under each idea instead, in the standing note's own words: four of
    eight ideas closed on a sentence stating no fact about the idea it closed."""
    from coscientist.narrative import COHERENCE_FALSIFIER_NOTE

    lines, notes = _coherence(
        [_review("Feasibility", 2), _review("Correctness", 5)],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert COHERENCE_FALSIFIER_NOTE in notes
    assert not any("falsifier" in line.lower() for line in lines)


def test_a_falsifier_under_agreeing_reviews_names_no_disagreement():
    """ "the disagreement between the reviews above" printed two lines under "They
    agree" -- the sentence named a dispute the paragraph above had just denied."""
    from coscientist.narrative import COHERENCE_FALSIFIER_NOTE

    lines, notes = _coherence(
        [_review("Correctness", 4), _review("Feasibility", 4)],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert COHERENCE_FALSIFIER_NOTE in notes
    assert not any("disagreement" in line for line in lines)


def test_reviews_that_score_alike_but_ask_for_different_things_still_disagree():
    lines, _ = _coherence(
        [
            _review("Correctness", 4, "revise"),
            _review("Feasibility", 4, "advance"),
        ],
        {"Falsifier": "No capacity difference at 500 cycles."},
    )

    assert "the recommendations do not" in lines[0]


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


def test_what_a_bottom_evidence_review_means_is_left_to_the_standing_note():
    """ "The lowest of them is the evidence and correctness review, at 2 of five: what
    it faults is the grounding, not the experiment" stood under six of eight ideas, and
    the clause after the colon is the opening of the standing note hoisted above them
    all. What is this idea's own is which review is at the bottom and how far down."""
    from coscientist.narrative import COHERENCE_EVIDENCE_NOTE

    lines, notes = _coherence(
        [_review("Correctness", 2), _review("Feasibility", 5)], {}
    )

    assert lines[-1] == (
        "The lowest of them is the evidence and correctness review, at 2 of five."
    )
    assert COHERENCE_EVIDENCE_NOTE in notes
    assert "the disagreement is about the grounding" in COHERENCE_EVIDENCE_NOTE


def test_a_bottom_the_evidence_review_shares_is_not_called_the_lowest_of_them():
    """The position was read off ``min``, which answers with one review whether or not
    one review holds it. Three reviews of a live idea scored two -- the same chapter's
    conclusion said as much -- and this sentence named whichever of the three came
    first as "the lowest of them"."""
    from coscientist.narrative import COHERENCE_EVIDENCE_NOTE

    lines, notes = _coherence(
        [
            _review("Correctness", 2),
            _review("Novelty", 2),
            _review("Feasibility", 2),
            _review("Impact", 5),
        ],
        {},
    )

    assert lines[-1] == (
        "Nothing scored below the evidence and correctness review, at 2 of five, and "
        "the Novelty and Feasibility reviews are level with it."
    )
    assert COHERENCE_EVIDENCE_NOTE in notes

    # One review level with it takes the singular, and the note still fires.
    pair, _ = _coherence(
        [_review("Correctness", 2), _review("Impact", 2), _review("Novelty", 5)], {}
    )
    assert "the Impact review is level with it" in pair[-1]


# --- a review nobody wrote is not a judgement of the idea it sits under -------


def test_a_review_that_stood_in_for_a_reviewer_is_named_as_a_placeholder():
    """The rank-1 idea of a live run carried a feasibility review no reviewer wrote,
    scored, counted in its span and printed as one of the four judgements of it."""
    from coscientist.narrative import COHERENCE_STOOD_IN_NOTE

    lines, notes = _coherence(
        [
            _review("Correctness", 5),
            dataclasses.replace(_review("Feasibility", 3), stood_in=True),
        ],
        {},
    )

    said = " ".join(lines)
    assert "The feasibility review is a placeholder" in said
    assert "nobody reviewed it on that criterion" in said
    assert COHERENCE_STOOD_IN_NOTE in notes


def test_reviews_a_reviewer_wrote_are_not_called_placeholders():
    from coscientist.narrative import COHERENCE_STOOD_IN_NOTE

    lines, notes = _coherence(
        [_review("Correctness", 5), _review("Feasibility", 3)], {}
    )

    assert not any("placeholder" in line for line in lines)
    assert COHERENCE_STOOD_IN_NOTE not in notes


def test_a_placeholders_objection_is_not_counted_as_one_nobody_answered():
    """ "One review raised one objection and recorded no response to it -- the
    Feasibility review" was printed about a sentence no reviewer wrote."""
    stood = dataclasses.replace(
        _review("Feasibility", 3),
        objections=["The candidate's feasibility is not established."],
        stood_in=True,
    )

    lines, _ = _coherence([_review("Correctness", 5), stood], {})

    assert not any("recorded no response" in line for line in lines)


# --- a table the specialist appended is its table, not the mechanism ----------


MECHANISM_WITH_A_SCORECARD = (
    "High-nickel cathodes suffer from instability, and the coating slows it.\n\n"
    "## Evaluation of Idea\n\n"
    "| Criterion | Description | Judgment |\n"
    "| --- | --- | --- |\n"
    "| Aggregation Control | Fluidized bed ALD coats discrete particles | High |\n"
    "| Purity Potential | Amorphous, defect-free layer | Exceptional |\n"
)


def _rated_candidate():
    from coscientist.models import Candidate

    return Candidate(
        id="candidate_0001",
        title="A 2.5 nm LiNbO3 Coating",
        claim="A 2.5 nm coating raises retention.",
        # The table goes where the live specialists put it: five of eight put theirs
        # in mechanism_model and three in rationale, and the report has to find it
        # either way.
        rationale="The literature reports gains from thin oxide layers.",
        mechanism_model=MECHANISM_WITH_A_SCORECARD,
        validation_protocol="Coin cells against an uncoated control.",
        falsifier="No difference at 500 cycles.",
    )


def test_the_mechanism_cell_holds_the_mechanism_and_not_the_self_rating():
    """ "Stereochemical Integrity (Description: ...; Judgment: Exceptional)" ran on
    inside the Mechanism cell of a live comparison grid, where a rating the specialist
    awarded itself reads as one of the fields the run filled in."""
    from coscientist.narrative import _table_rows

    mechanism = dict(_table_rows(_rated_candidate()))["Mechanism"]

    assert mechanism == (
        "High-nickel cathodes suffer from instability, and the coating slows it."
    )
    assert "Judgment" not in mechanism
    assert "Evaluation of Idea" not in mechanism


def test_the_mechanism_comes_from_the_field_the_specialist_wrote_it_in():
    """``mechanism_model`` is a required field of the contract, every generation
    prompt asks for the mechanism in it, and no exporter read it. On a live run all
    eight ideas filled it and filled ``rationale`` with something else, so the report
    printed the motivation under the heading Mechanism and the mechanism -- up to
    3,330 characters of reaction pathway and precursor chemistry -- reached no export
    at all."""
    from coscientist.models import Candidate
    from coscientist.narrative import _authored_extras, _mechanism

    candidate = Candidate(
        title="A 2.5 nm LiNbO3 Coating",
        claim="A 2.5 nm coating raises retention.",
        # The heading the prompt asks the specialist to put over the field. It names
        # the field rather than a section inside it, so it is not a label to print.
        mechanism_model=(
            "### Rich Technical Narrative\nLiNbO3 conducts lithium and blocks HF."
        ),
        rationale=(
            "### Motivation and Supporting Evidence\nThin niobate layers have been "
            "reported to raise retention."
        ),
        validation_protocol="Coin cells against an uncoated control.",
        falsifier="No difference at 500 cycles.",
    )

    assert _mechanism(candidate) == "LiNbO3 conducts lithium and blocks HF."
    # And what the specialist headed in the other field is still its own section,
    # rather than being displaced by the field the mechanism came from.
    assert _authored_extras(candidate)[2] == [
        (
            "Motivation and Supporting Evidence",
            "Thin niobate layers have been reported to raise retention.",
        )
    ]


def test_a_judgment_written_in_front_of_the_protocol_is_not_the_protocol():
    """One live Validation Protocol section opened "Critical Scientific Judgment:
    While HfON offers superior dielectric shielding, the primary risk is ..." and
    reached the bench steps 300 words later behind a bold "Experimental Protocol:".
    The same label is printed under The Specialist's Own Sections for the ideas whose
    specialist wrote it in the mechanism field, so one report filed one label two
    ways."""
    from coscientist.models import Candidate
    from coscientist.narrative import _authored_extras, _protocol_steps

    candidate = Candidate(
        title="A 2 nm HfON Coating",
        claim="A 2 nm HfON coating raises retention.",
        mechanism_model="HfON screens the interfacial field.",
        rationale="High-k dielectrics block electron leakage.",
        validation_protocol=(
            "Critical Scientific Judgment: While HfON offers superior dielectric "
            "shielding, the primary risk is a higher charge-transfer resistance. "
            "**Experimental Protocol:** 1. Synthesize 2 nm HfON-coated NMC811 via "
            "MLD. 2. Assemble CR2032 coin cells (n=5 per group). 3. Cycle at 1C."
        ),
        falsifier="No difference at 500 cycles.",
    )

    assert _protocol_steps(candidate.validation_protocol) == [
        "Synthesize 2 nm HfON-coated NMC811 via MLD.",
        "Assemble CR2032 coin cells (n=5 per group).",
        "Cycle at 1C.",
    ]
    assert _authored_extras(candidate)[2] == [
        (
            "Critical Scientific Judgment",
            "While HfON offers superior dielectric shielding, the primary risk is a "
            "higher charge-transfer resistance.",
        )
    ]


def test_a_rating_appended_after_the_bench_steps_is_lifted_out_of_the_protocol():
    """Two of eight live ideas answered the whole generation prompt in
    ``validation_protocol``, so their Validation Protocol ran on from "Quantify the
    depth and frequency of pitting corrosion" into a rating the specialist awarded
    itself, read out as clauses -- "Ligation Strategy (Description: ALD coating
    adhesion under mechanical stress; Judgment: Brittle Al2O3 fractures)" -- and then
    a Critical Scientific Judgment. The other six had those same two artefacts
    printed as a captioned table and a labelled section of their own."""
    from coscientist.dossier import _self_rating
    from coscientist.models import Candidate
    from coscientist.narrative import IdeaBrief, _authored_extras, _protocol_steps

    candidate = Candidate(
        title="Defect-Induced Current Focusing",
        claim="Pinholes funnel current into hotspots.",
        mechanism_model="A resistive coating leaves the pinholes as the only path.",
        rationale="Corrosion science reports pitting at pinholes.",
        validation_protocol=(
            "**Experimental Protocol:**\n"
            "Prepare NMC811 electrodes with ALD Al2O3 coatings (1-5 nm). Quantify "
            "the depth and frequency of pitting corrosion.\n\n"
            "**Evaluation of Idea Table:**\n"
            "| Category | Description | Judgment |\n"
            "| :--- | :--- | :--- |\n"
            "| Ligation Strategy | ALD coating adhesion under mechanical stress | "
            "Brittle Al2O3 fractures |\n\n"
            "**Critical Scientific Judgment:**\n"
            "The strength of the idea lies in its appraisal of operational defects."
        ),
        falsifier="Defect-coated cells retain more capacity than bare ones.",
    )

    title, table, sections, source = _authored_extras(candidate)

    assert _protocol_steps(candidate.validation_protocol) == [
        "Prepare NMC811 electrodes with ALD Al2O3 coatings (1-5 nm). Quantify the "
        "depth and frequency of pitting corrosion."
    ]
    assert title == "Evaluation of Idea Table"
    assert source == "validation protocol"
    assert table == [
        ["Category", "Description", "Judgment"],
        [
            "Ligation Strategy",
            "ALD coating adhesion under mechanical stress",
            "Brittle Al2O3 fractures",
        ],
    ]
    assert sections == [
        (
            "Critical Scientific Judgment",
            "The strength of the idea lies in its appraisal of operational defects.",
        )
    ]

    said = "\n".join(
        _self_rating(
            IdeaBrief(
                title=candidate.title,
                candidate_id="candidate_0004",
                rank=4,
                elo=1184,
                category="",
                proposal="",
                description=[],
                facts={},
                summary={},
                table_rows=[],
                reviews=[],
                coherence=[],
                deep_verification=[],
                matches=[],
                wins=0,
                losses=0,
                ties=0,
                shortlisted=False,
                self_rating_title=title,
                self_rating=table,
                self_rating_source=source,
            )
        )
    )
    # Where the specialist appended it, not where six of eight happened to.
    assert (
        "a table of its own — headed Evaluation of Idea Table — to the validation "
        "protocol." in said
    )


def test_a_table_of_bench_conditions_stays_inside_the_protocol():
    """The lift is keyed to the heading the generation prompts ask for by name. A
    protocol that tabulates its own arms is protocol, and moving it under The
    Specialist's Own Rating would file the run's conditions as a self-assessment."""
    from coscientist.models import Candidate
    from coscientist.narrative import _authored_extras

    candidate = Candidate(
        title="A 2 nm Al2O3 Coating",
        claim="A 2 nm coating raises retention.",
        mechanism_model="Alumina scavenges HF.",
        rationale="Scavenging slows transition metal dissolution.",
        validation_protocol=(
            "Cycle each arm at 1C to 500 cycles.\n\n"
            "| Arm | Coating | Cells |\n"
            "| --- | --- | --- |\n"
            "| A | none | 5 |\n"
        ),
        falsifier="No difference at 500 cycles.",
    )

    assert _authored_extras(candidate)[1] == []


def test_a_protocol_whose_preamble_is_nobodys_section_is_left_whole():
    """The split above moves the preamble out of the protocol, so it only runs where
    every word of that preamble is a section the specialist headed. Anything else in
    front of the label is protocol prose, and moving it out would lose it."""
    from coscientist.narrative import _protocol_steps

    written = (
        "Cells are built in an argon glovebox. Experimental Protocol: Cycle at 1C."
    )

    assert _protocol_steps(written) == [written]


def test_the_protocol_is_printed_as_the_steps_the_specialist_numbered():
    """``validation_protocol`` is required by the contract, asked for by every
    generation prompt and checked by normalisation, and no exporter read it: the
    sample size, its power rationale, the blinding and the abort limits were in the
    saved session for all eight live ideas and in none of the three exports."""
    from coscientist.dossier import _validation_protocol
    from coscientist.narrative import IdeaBrief, _protocol_steps

    steps = _protocol_steps(
        "1. Fabricate NMC811 cathodes and apply 2 nm Al2O3 by ALD. "
        "2. Assemble CR2032 coin cells (n=5 per arm) in an argon glovebox. "
        "3. Cycle at 1C to 500 cycles and abort above 45 degrees."
    )
    assert steps == [
        "Fabricate NMC811 cathodes and apply 2 nm Al2O3 by ALD.",
        "Assemble CR2032 coin cells (n=5 per arm) in an argon glovebox.",
        "Cycle at 1C to 500 cycles and abort above 45 degrees.",
    ]
    # A number inside a step is not a step: a protocol that says "4.3 V" or "n=5"
    # would be cut into pieces at the measurement, which is worse than one paragraph.
    assert _protocol_steps(
        "Charge to 4.3 V and hold. 25 cells per arm are cycled to 500 cycles."
    ) == ["Charge to 4.3 V and hold. 25 cells per arm are cycled to 500 cycles."]

    lines = _validation_protocol(
        IdeaBrief(
            title="A 2.5 nm LiNbO3 Coating",
            candidate_id="candidate_0001",
            rank=1,
            elo=1200,
            category="",
            proposal="",
            description=[],
            facts={},
            summary={},
            table_rows=[],
            reviews=[],
            coherence=[],
            deep_verification=[],
            matches=[],
            wins=0,
            losses=0,
            ties=0,
            shortlisted=False,
            validation_protocol=steps,
        )
    )
    assert lines[0] == "### Validation Protocol"
    assert lines[2:5] == [f"{index}. {step}" for index, step in enumerate(steps, 1)]


def _brief_with_own_sections(title: str) -> object:
    from coscientist.narrative import IdeaBrief

    return IdeaBrief(
        title=title,
        candidate_id="candidate_0001",
        rank=1,
        elo=1200,
        category="",
        proposal="",
        description=[],
        facts={},
        summary={},
        table_rows=[],
        reviews=[],
        coherence=[],
        deep_verification=[],
        matches=[],
        wins=0,
        losses=0,
        ties=0,
        shortlisted=False,
        authors_own_sections=[
            ("Critical Scientific Judgment", "TMA is pyrophoric and needs a hood.")
        ],
    )


def test_what_the_specialists_own_sections_are_is_explained_once_above_the_ideas():
    """The note under the heading describes the generation contract -- one prose field
    for a mechanism the prompt asks four things of -- rather than the idea it stands
    under, and it stood in the same words under four of eight ideas."""
    from coscientist.dossier import _authors_own_sections, shared_authors_own_note

    briefs = [_brief_with_own_sections("One"), _brief_with_own_sections("Two")]
    hoisted = "\n".join(shared_authors_own_note(briefs))

    assert "not a finding of the run" in hoisted
    under = "\n".join(_authors_own_sections(briefs[0], hoisted=True))
    assert under.startswith("### The Specialist's Own Sections")
    assert "headed a section of its own" not in under
    assert "**Critical Scientific Judgment.** TMA is pyrophoric" in under


def test_one_idea_carrying_its_own_sections_keeps_the_note_under_it():
    """Hoisting a note over a single idea puts it two pages away from the only thing
    it describes, and saves nothing."""
    from coscientist.dossier import _authors_own_sections, shared_authors_own_note

    brief = _brief_with_own_sections("One")

    assert not shared_authors_own_note([brief])
    assert "headed a section of its own" in "\n".join(_authors_own_sections(brief))


def test_the_self_rating_is_printed_as_the_specialists_own_table():
    from coscientist.dossier import _self_rating
    from coscientist.narrative import IdeaBrief, _authors_own_table

    _, title, rating = _authors_own_table(MECHANISM_WITH_A_SCORECARD)
    brief = IdeaBrief(
        title="A 2.5 nm LiNbO3 Coating",
        candidate_id="candidate_0001",
        rank=1,
        elo=1200,
        category="",
        proposal="",
        description=[],
        facts={},
        summary={},
        table_rows=[],
        reviews=[],
        coherence=[],
        deep_verification=[],
        matches=[],
        wins=0,
        losses=0,
        ties=0,
        shortlisted=False,
        self_rating_title=title,
        self_rating=rating,
    )

    said = "\n".join(_self_rating(brief))

    assert "a table of its own — headed Evaluation of Idea — to the mechanism." in said
    assert "not a result of the reviews or the tournament below" in said
    assert "| Criterion | Description | Judgment |" in said
    assert "| Purity Potential | Amorphous, defect-free layer | Exceptional |" in said


def test_the_heading_belongs_to_the_table_and_not_to_the_field_it_was_appended_to():
    """At the end of the sentence the nearest noun takes it, and that is the field.

    Two of eight live ideas appended the rating to their validation protocol, and the
    sentence read "appended a table of its own to the validation protocol, headed
    Evaluation of Idea Table" -- which gives the heading to the protocol, a field of
    the idea that has no heading of its own to give.
    """
    from dataclasses import replace

    from coscientist.dossier import _self_rating
    from coscientist.narrative import _authors_own_table

    _, title, rating = _authors_own_table(MECHANISM_WITH_A_SCORECARD)
    brief = replace(
        _brief_without_a_rating(),
        self_rating_title=title,
        self_rating=rating,
        self_rating_source="validation protocol",
    )

    said = "\n".join(_self_rating(brief))

    assert "to the validation protocol, headed" not in said
    assert (
        "a table of its own — headed Evaluation of Idea — to the validation protocol."
        in said
    )


def test_a_rating_the_specialist_left_unheaded_still_names_the_field_it_sits_under():
    from dataclasses import replace

    from coscientist.dossier import _self_rating
    from coscientist.narrative import _authors_own_table

    _, _, rating = _authors_own_table(MECHANISM_WITH_A_SCORECARD)
    brief = replace(_brief_without_a_rating(), self_rating=rating)

    said = "\n".join(_self_rating(brief))

    assert "a table of its own to the mechanism." in said
    assert "—" not in said.partition("\n")[0]


def test_an_idea_whose_specialist_rated_nothing_gets_no_rating_section():
    from coscientist.dossier import _self_rating
    from coscientist.narrative import _authors_own_table

    kept, title, rating = _authors_own_table("The coating slows the fade.")

    assert (kept, title, rating) == ("The coating slows the fade.", "", [])
    assert _self_rating(_brief_without_a_rating()) == []


def _brief_without_a_rating():
    from coscientist.narrative import IdeaBrief

    return IdeaBrief(
        title="Idea",
        candidate_id="candidate_0002",
        rank=1,
        elo=1200,
        category="",
        proposal="",
        description=[],
        facts={},
        summary={},
        table_rows=[],
        reviews=[],
        coherence=[],
        deep_verification=[],
        matches=[],
        wins=0,
        losses=0,
        ties=0,
        shortlisted=False,
    )


FOUR_PART = """The coating suppresses lattice oxygen release at high states of charge.

## Motivation and Supporting Evidence

Literature shows a bare cell has 43% shorter cycle life than a coated one.

## Critical Scientific Judgment

The annealing step can form a resistive rock-salt phase at the boundary.
"""


def test_the_sections_a_specialist_headed_itself_are_not_read_as_the_mechanism():
    """A live Mechanism cell opened "Motivation and Supporting Evidence:", ran 1,475
    characters through a "Critical Scientific Judgment:" and never said what the
    mechanism was. Four parts are asked for and one prose field is given to hold them."""
    lead, parts = _authors_own_parts(FOUR_PART)

    assert lead.strip() == (
        "The coating suppresses lattice oxygen release at high states of charge."
    )
    assert [label for label, _ in parts] == [
        "Motivation and Supporting Evidence",
        "Critical Scientific Judgment",
    ]
    assert parts[0][1].startswith("Literature shows a bare cell")
    assert parts[1][1].startswith("The annealing step")


def test_the_labels_are_recognised_however_the_specialist_wrote_them():
    """Numbered, bolded, inline or as a heading -- the prompt numbers them and the
    specialists answer in whichever of those shapes they please."""
    _, parts = _authors_own_parts(
        "It suppresses oxygen release.\n\n"
        "**2. Motivation and Supporting Evidence:** The literature agrees.\n\n"
        "#### critical scientific judgment\n\nThe anneal is the risk.\n"
    )

    assert [label for label, _ in parts] == [
        # Printed in the canonical spelling: the words are the specialist's, the
        # capitalisation of a heading in this report is the report's.
        "Motivation and Supporting Evidence",
        "Critical Scientific Judgment",
    ]
    assert parts[0][1] == "The literature agrees."


def test_a_mechanism_with_no_headed_sections_is_left_exactly_as_written():
    text = "High-nickel cathodes suffer from instability. The coating blocks it."
    assert _authors_own_parts(text) == (text, [])


def test_a_specialist_that_wrote_only_headed_sections_is_not_given_a_mechanism():
    """The lead is empty here, and the Mechanism cell has to say so rather than
    print the motivation section under a label that is not what it is."""
    from coscientist.models import Candidate
    from coscientist.narrative import _UNSTATED, _idea_facts

    facts = _idea_facts(
        Candidate(
            id="cand_a",
            title="A coating",
            claim="A coating helps.",
            rationale="## Motivation and Supporting Evidence\n\nThe literature agrees.",
            mechanism_model="",
            validation_protocol="",
            falsifier="Retention is unchanged.",
        )
    )

    assert facts["Mechanism and rationale"] == _UNSTATED["Mechanism and rationale"]
