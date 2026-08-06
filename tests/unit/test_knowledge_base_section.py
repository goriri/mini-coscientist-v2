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

from types import SimpleNamespace

from coscientist.evidence import SUMMARY_CHARACTER_LIMIT, _report_summary
from coscientist.models import (
    DeepResearchRun,
    DiscoveryNarrative,
    DiscoveryStatement,
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
    *narratives: DiscoveryNarrative, checked: bool = True, ran: int | None = None
) -> SimpleNamespace:
    """``ran`` is how many passes the manifest recorded, which need not be how many
    of them came back with a report to print."""
    passes = len(narratives) if ran is None else ran
    return SimpleNamespace(
        discovery=SimpleNamespace(
            narratives=list(narratives),
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
    assert "### Pass 1" not in section


def test_the_facet_is_recovered_from_the_statements_when_the_pass_did_not_record_it():
    """Sessions written before the narrative carried its own facet."""
    section = _knowledge_summary(_record(_narrative(), _narrative(facet="methods")))
    assert "### Pass 1: Supporting evidence" in section


def test_a_single_pass_is_not_given_a_heading_that_counts_it():
    assert "### Pass 1" not in _knowledge_summary(_record(_narrative()))


def test_the_reader_is_told_the_removed_citation_markers_were_pass_local():
    section = _knowledge_summary(_record(_narrative()))
    assert "not the numbering under References" in section


def test_a_pass_that_grounded_nothing_says_so_under_its_own_report():
    section = _knowledge_summary(_record(_narrative(statements=[])))
    assert "nothing from it was carried into the evidence base" in section


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
