"""The Knowledge Base prints seven literature reports, not one paragraph.

Each Deep Research pass returns a whole Markdown document: its own title, its own
heading tree, a facet tag on most sentences and citation markers numbered against
a source list only that pass held. The dossier used to collapse every report's
whitespace and join all seven with a space. On a live run that produced a single
19,348-character line whose first characters were ``# Protective Coatings`` -- so
the table of contents grew one entry holding the entire literature review, and the
Knowledge Base rendered as one unbroken block of prose that ended mid-word.

These tests hold the four properties that block has to have: line structure
survives, the reports stay separated and labelled, the provider's markup for the
pipeline does not reach the reader, and a truncated report says it is truncated.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from coscientist.evidence import SUMMARY_CHARACTER_LIMIT, _report_summary
from coscientist.models import (
    DeepResearchRun,
    DiscoveryNarrative,
    DiscoveryStatement,
    SourceLead,
)
from coscientist.narrative import _deep_research_prose, _knowledge_summary

REPORT = """# Protective Coatings and Cycle Life

## Summary

An exhaustive review finds no evidence meeting every constraint [supporting].

### ALD Al2O3 on LiFePO4

The researchers achieved 5 nm coatings [cite: 2] [supporting]. Testing ran at
40 C [cite: 2, 8] [Facet: Supporting].
"""


def _narrative(**fields) -> DiscoveryNarrative:
    defaults = {
        "question": "Does a coating improve cycle life?",
        "summary": REPORT,
        "statements": [
            DiscoveryStatement(
                text="The researchers achieved 5 nm coatings.",
                facet="supporting",
                source_urls=["https://example.org/a"],
                originating_pass=1,
            )
        ],
    }
    return DiscoveryNarrative(**{**defaults, **fields})


def _record(
    *narratives: DiscoveryNarrative,
    checked: bool = True,
    ran: int | None = None,
    leads: Sequence[SourceLead] = (),
    numbers: dict[str, int] | None = None,
    seeded_from: str = "",
) -> SimpleNamespace:
    """``ran`` is how many passes the manifest recorded, which need not be how many
    of them came back with a report to print. ``leads`` and ``numbers`` are what a
    pass turned up and which of it the running text has cited by the time this
    section is built. ``seeded_from`` is the earlier run this one forked its whole
    corpus from, which is what everything below this heading then belongs to."""
    passes = len(narratives) if ran is None else ran
    assigned = numbers or {}
    return SimpleNamespace(
        session=SimpleNamespace(seeded_evidence_from=seeded_from),
        citations=SimpleNamespace(numbered=assigned.get),
        discovery=SimpleNamespace(
            narratives=list(narratives),
            source_leads=list(leads),
            runs=[
                DeepResearchRun(pass_number=number, status="completed")
                for number in range(1, passes + 1)
            ],
        ),
        evidence=SimpleNamespace(
            claims=[
                SimpleNamespace(
                    verification_status="verified" if checked else "unverified"
                )
            ]
        ),
    )


def test_the_reports_keep_the_line_structure_they_arrived_with():
    section = _knowledge_summary(_record(_narrative()))
    assert "\n" in section
    assert max(len(line) for line in section.splitlines()) < 400, (
        "a report was flattened back into one line"
    )


def test_a_reports_own_headings_are_demoted_rather_than_inlined():
    """They nest under ``## Knowledge Summary``; at level one they own the page."""
    prose = _deep_research_prose(REPORT)
    assert "#### Protective Coatings and Cycle Life" in prose
    assert "##### Summary" in prose
    assert "###### ALD Al2O3 on LiFePO4" in prose
    assert not any(
        line.startswith(("# ", "## ", "### ")) for line in prose.splitlines()
    ), "a heading left at level three or above escapes the section it belongs to"


def test_the_pipelines_own_markup_does_not_reach_the_reader():
    prose = _deep_research_prose(REPORT)
    assert "[supporting]" not in prose
    assert "[Facet: Supporting]" not in prose
    assert "[cite:" not in prose
    assert "achieved 5 nm coatings." in prose, "the sentence lost more than its tags"


def test_a_tag_the_pass_wrote_as_inline_code_is_still_the_pipelines_markup():
    """A live pass wrote every tag as ``[Facet: contradictory]`` in backticks. The
    trailing lookahead then saw a backtick where it wanted a space, and fourteen of
    them went to the reader."""
    prose = _deep_research_prose(
        "The sample size requirement is the failure point. `[Facet: contradictory]` "
        "There are zero instances of it. `[cite: 3, 4]`\n"
    )

    assert "Facet:" not in prose
    assert "[cite:" not in prose
    assert "`" not in prose, "the backticks the tag was wrapped in stayed behind"
    assert "The sample size requirement is the failure point." in prose
    assert "There are zero instances of it." in prose


def test_each_pass_is_labelled_with_the_evidence_it_was_sent_to_find():
    section = _knowledge_summary(
        _record(
            _narrative(facet="supporting"),
            _narrative(facet="corrections_retractions", statements=[]),
        )
    )
    assert "### Pass 1: Supporting evidence" in section
    assert (
        "### Pass 2: Corrections or retractions affecting the sources used" in section
    )


def test_a_pass_that_wrote_no_report_is_still_counted_among_the_passes_that_ran():
    """ "The search ran as six separate passes" stood in the body of a live report
    whose appendix said Deep Research ran seven. Both counts were right about
    different things, and only one of them was printed."""
    section = _knowledge_summary(_record(_narrative(), _narrative(), ran=3))

    assert "The search ran as three separate passes" in section
    assert "two of them wrote a report" in section
    assert "The other recorded none" in section
    # And the pass that recorded none is headed with the rest. The sentence above
    # says a pass that found nothing is a finding that disappears when the reports
    # are merged, and printing only the reports is one way of merging them.
    assert "### Pass 3" in section
    assert "This pass ran and recorded no report" in section


def test_a_run_where_every_pass_reported_does_not_count_them_twice():
    section = _knowledge_summary(_record(_narrative(), _narrative()))

    assert "The search ran as two separate passes" in section
    assert "each wrote its own report" in section
    assert "wrote a report" not in section.split("each wrote its own report")[0]


def test_a_report_is_headed_with_the_pass_that_wrote_it_not_its_place_in_the_list():
    """The discovery appendix numbers the passes as they ran, and the body tells the
    reader to compare the two. Renumbering from one skips whatever came back empty."""
    section = _knowledge_summary(
        _record(
            _narrative(facet="supporting", pass_number=2),
            _narrative(facet="methods", pass_number=5),
            ran=5,
        )
    )

    assert "### Pass 2: Supporting evidence" in section
    assert "### Pass 5: " in section
    # Pass 1 ran and wrote nothing. It is headed, but no report is printed under it
    # and nothing else in the section is numbered as though it were pass 1.
    assert "### Pass 1" in section
    assert (
        "no report to reproduce under this heading" in (section.split("### Pass 2")[0])
    )
    assert "Protective Coatings" not in section.split("### Pass 2")[0]


def test_the_facet_is_recovered_from_the_statements_when_the_pass_did_not_record_it():
    """Sessions written before the narrative carried its own facet."""
    section = _knowledge_summary(_record(_narrative(), _narrative(facet="methods")))
    assert "### Pass 1: Supporting evidence" in section


def test_the_pass_that_closed_the_gaps_is_headed_with_what_it_went_looking_for():
    """A bare "### Pass 8" ended a live Knowledge Base under seven headings that each
    named their pass's subject, reading as a heading whose subject had gone missing.

    The gap pass has one. It is planned with no facet because it covers whatever the
    fan-out left open rather than one of the seven, and it goes last for that reason.
    """
    section = _knowledge_summary(
        _record(
            _narrative(facet="supporting", pass_number=1),
            _narrative(facet="methods", pass_number=2),
            _narrative(pass_number=3),
        )
    )

    assert "### Pass 3: The gaps the fan-out left open" in section


def test_a_pass_before_the_fan_out_recorded_anything_is_not_read_as_the_gap_pass():
    """Ahead of every facet the run recorded, an unlabelled pass is an old session's
    -- the field did not exist when it was saved -- and its facet is still inferred."""
    section = _knowledge_summary(
        _record(_narrative(pass_number=1), _narrative(facet="methods", pass_number=2))
    )

    assert "### Pass 1: Supporting evidence" in section
    assert "the fan-out left open" not in section


def test_a_single_pass_is_not_given_a_heading_that_counts_it():
    assert "### Pass 1" not in _knowledge_summary(_record(_narrative()))


def test_the_reader_is_told_the_removed_citation_markers_were_pass_local():
    section = _knowledge_summary(_record(_narrative()))
    assert "not the numbering under References" in section


def test_a_pass_that_grounded_nothing_says_so_under_its_own_report():
    section = _knowledge_summary(_record(_narrative(statements=[])))
    assert "no finding from it was carried into the evidence base" in section


def test_the_note_about_ungrounded_passes_is_made_once_over_the_passes_it_holds_for():
    """The same 45-word note stood under five of seven passes, in a section a reader
    goes through in one pass of their own. What it says is about how this run matched
    statements to sources, not about any one pass."""
    section = _knowledge_summary(
        _record(
            _narrative(statements=[], pass_number=1),
            _narrative(pass_number=2),
            _narrative(statements=[], pass_number=3),
        )
    )

    assert section.count("could be tied to a source the provider also returned") == 1
    assert "*No statement in passes 1 and 3 could be tied to a source" in section
    # Said above the reports rather than after the last of them.
    assert section.index("No statement in passes") < section.index("### Pass 1")


def _lead(url: str, *passes: int) -> SourceLead:
    return SourceLead(canonical_url=url, originating_passes=list(passes))


def test_a_pass_whose_findings_went_nowhere_is_not_said_to_have_gone_uncited():
    """ "Nothing from those passes was carried into the evidence base or cited
    elsewhere in this report" stood over four passes whose leads the appendix two
    pages down reports as cited nine times between them. A pass contributes a
    statement and a source separately, and only the statement failed here."""
    section = _knowledge_summary(
        _record(
            _narrative(statements=[], pass_number=1),
            _narrative(pass_number=2),
            _narrative(statements=[], pass_number=3),
            leads=[
                _lead("https://example.org/a", 1),
                _lead("https://example.org/b", 2),
            ],
            numbers={"https://example.org/a": 4},
        )
    )

    assert "or cited elsewhere in this report" not in section
    assert "no finding from those passes was carried into the evidence base" in section
    assert (
        "Sources they returned are cited here even so, where a finding another pass "
        "recorded rests on one." in section
    )


def test_nothing_is_claimed_for_a_pass_whose_leads_carry_no_number_yet():
    """A number is assigned the first time the running text cites a URL, so a lead
    with none here may still be cited further down: the absence proves nothing and
    the sentence says nothing."""
    section = _knowledge_summary(
        _record(
            _narrative(statements=[], pass_number=1),
            _narrative(pass_number=2),
            _narrative(statements=[], pass_number=3),
            leads=[_lead("https://example.org/a", 1)],
        )
    )

    assert "cited here even so" not in section
    assert "no finding from those passes was carried into the evidence base" in section


def test_a_forked_corpus_says_it_was_searched_by_another_run():
    """Everything under this heading -- the passes, what each was asked, what each
    returned -- is the record of a search that happened in a different session. Left
    unsaid it reads as this run's work, and two forks of one corpus read as two
    independent searches that happened to agree."""
    section = _knowledge_summary(
        _record(_narrative(), checked=False, seeded_from="session_abc123")
    )

    assert section.startswith("This run did not search the literature")
    assert "session_abc123" in section
    # And the report of the search is still printed below it, unedited: the note
    # says whose search this was, not what the search found.
    assert "What follows is the literature search's own report" in section
    assert "achieved 5 nm coatings." in section


def test_a_run_that_did_its_own_searching_is_not_told_it_did_not():
    assert "did not search the literature" not in _knowledge_summary(
        _record(_narrative())
    )


def test_an_unchecked_search_still_opens_with_whose_claim_this_is():
    section = _knowledge_summary(_record(_narrative(), checked=False))
    assert section.startswith("What follows is the literature search's own report")


def test_a_truncated_report_says_it_is_truncated():
    section = _knowledge_summary(_record(_narrative(truncated=True)))
    assert "cut off above" in section
    assert "cut off above" not in _knowledge_summary(_record(_narrative()))


def test_the_stored_report_is_cut_on_a_boundary_a_reader_can_see():
    """The bare slice ended a live run's Knowledge Base on "ambient thermal fl"."""
    text = ("Ambient thermal fluctuation is eliminated. " * 600).strip()
    assert len(text) > SUMMARY_CHARACTER_LIMIT
    summary, truncated = _report_summary(text)
    assert truncated
    assert len(summary) <= SUMMARY_CHARACTER_LIMIT
    assert summary.endswith("eliminated.")


def test_a_report_within_the_limit_is_kept_whole_and_not_flagged():
    summary, truncated = _report_summary(REPORT)
    assert not truncated
    assert summary == REPORT.strip()


def test_a_pass_that_calls_its_own_document_the_report_is_told_which_one():
    """Three of six live passes opened "The report evaluates whether...", "The report
    outlines...", "The report overwhelmingly supports..." -- inside a document that
    calls itself the report on every other page, under a heading reading "Pass 5"."""
    prose = _deep_research_prose(
        "The report evaluates whether a coating improves cycle life.\n\n"
        "## Findings\n\nThe report by the Hanyang team is the exception.\n"
    )

    assert prose.startswith("This pass's report evaluates whether")
    # Only the opening. Further in, "the report" is as likely to name a paper the
    # pass is discussing, which is not this section's to rename.
    assert "The report by the Hanyang team is the exception." in prose


def test_a_report_the_pass_is_citing_is_not_renamed_as_the_passs_own():
    """ "The report by X" names somebody else's work, and renaming it would put this
    section's words inside a claim about a paper."""
    prose = _deep_research_prose(
        "# Protective Coatings\n\nThe report by Kim et al. finds no null result.\n"
    )
    assert "The report by Kim et al." in prose
    assert "This pass's report" not in prose


def test_a_list_the_provider_marked_with_asterisks_is_set_like_every_other_list():
    """Both markers are Markdown and both render, so the exporters never saw it.

    The Markdown is a deliverable of its own, and a live one carried "*   **Al2O3 on
    NCM622 Cathodes:**" three lines under "- Pass 2 returned" -- one document setting
    its lists two ways, in the one section written by the provider rather than here.
    """
    prose = _deep_research_prose(
        "## Findings\n"
        "*   **Al2O3 on NCM622:** it outperformed the uncoated control.\n"
        "    *   Measured over three cells.\n"
        "+   A marker written as a plus.\n"
        "- Already a hyphen.\n"
    )

    assert "- **Al2O3 on NCM622:** it outperformed the uncoated control." in prose
    # The indent is kept, so a nested list is still nested.
    assert "    - Measured over three cells." in prose
    assert "- A marker written as a plus." in prose
    assert "- Already a hyphen." in prose
    assert "*   " not in prose


def test_what_is_not_a_list_marker_is_left_where_the_provider_put_it():
    """A bold label opening a line, an emphasised sentence, and both spellings of a
    thematic break all begin with the character a bullet begins with."""
    prose = _deep_research_prose(
        "**Bold label:** this is a paragraph.\n*An emphasised sentence.*\n***\n* * *\n"
    )

    assert "**Bold label:** this is a paragraph." in prose
    assert "*An emphasised sentence.*" in prose
    assert "\n***\n" in prose
    assert "* * *" in prose
    assert "- " not in prose
