"""The warnings chapter, and what the body keeps once the warnings leave it.

Each of these paragraphs used to be printed beside the material it qualifies. That
was the right call one warning at a time and the wrong one in aggregate: a live
report opened with three of them before its first sentence about the science, and
the per-idea copy of the fallback notice was printed eight times. They are collected
into one appendix chapter now, so what these tests hold is the two halves of that
bargain -- nothing run-level is left loose in the body, and nothing idea-level was
swept into the appendix with it.
"""

from __future__ import annotations

import pytest

from coscientist.advisories import advisory_pointer, run_advisories
from coscientist.dossier import compile_dossier
from coscientist.models import (
    Artifact,
    ArtifactStatus,
    DecisionAction,
    HumanDecision,
    Session,
)
from coscientist.narrative import (
    BlockerNote,
    ResearchRecord,
    build_idea_briefs,
    load_record,
)

CHAPTER = "\n# Warnings and Limitations"


def _halves(session: Session) -> tuple[str, str]:
    report = compile_dossier(session)
    return report.split(CHAPTER)[0], report[report.index(CHAPTER) :]


def _auto_approved(session: Session) -> Session:
    session.decisions.append(
        HumanDecision(
            action=DecisionAction.ACCEPT,
            stage="generate",
            actor="auto_policy",
            automatic=True,
            session_version=session.version,
        )
    )
    return session


def test_the_chapter_is_present_on_a_run_with_nothing_wrong_with_it():
    """The standing limits are true of every run, so the chapter is never absent.

    A chapter that appears only on troubled runs teaches a reader that its absence
    means the report was checked, which is the one thing it does not mean.
    """
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    advisories = run_advisories(record)

    assert [item.title for item in advisories] == ["What this report is and is not"]
    assert not any(item.blocking for item in advisories)
    pointer = advisory_pointer(advisories)
    assert pointer.startswith("One limitation applies")
    assert "it does not block the work proposed below" in pointer


def test_the_body_pointer_names_the_blocking_warnings_rather_than_counting_them(
    rich_session: Session,
):
    """A bare count gives a reader no reason to turn to the appendix.

    Two reports can carry the same number of limitations and differ entirely in
    whether either of them says to stop, so the count that leads is the blocking one
    and the blockers are named where the reader is.
    """
    rich_session.exploratory_evidence_accepted = True
    body, appendix = _halves(rich_session)

    assert "a waived evidence gate" in body
    assert "ideas citing evidence that is absent or retracted" in body
    assert "should not proceed on the material they name" in body
    # Named, not reproduced: the paragraph itself is in one place only.
    assert "Waiving the gate is a distinct act" not in body
    assert "Waiving the gate is a distinct act" in appendix


@pytest.mark.parametrize(
    ("mutate", "moved"),
    [
        (
            lambda session: setattr(session, "exploratory_evidence_accepted", True),
            "The evidence gate for this run was waived",
        ),
        (_auto_approved, "was approved automatically under the auto approval"),
        (lambda session: None, "Nothing in this document is a finding"),
        (lambda session: None, "No experiment was performed in this run"),
        (lambda session: None, "a fixed template, not the specialist's own reasoning"),
        (
            lambda session: None,
            "cite evidence that does not exist in this session",
        ),
    ],
)
def test_every_run_level_warning_is_in_the_chapter_and_not_in_the_body(
    rich_session: Session, mutate, moved: str
):
    mutate(rich_session)
    body, appendix = _halves(rich_session)

    assert moved in appendix
    assert moved not in body


def test_a_caveat_about_one_idea_stays_under_that_idea(rich_session: Session):
    """The line the collection must not cross.

    A grounding verdict and a fatal flaw are findings about one hypothesis. Hoisting
    those into a chapter at the end would mean a reader comparing two ideas could not
    see which of them carries one.
    """
    body, appendix = _halves(rich_session)
    briefs = build_idea_briefs(load_record(rich_session))
    alarming = [brief for brief in briefs if brief.support_is_alarming]

    assert alarming, "the fixture no longer exercises a broken grounding verdict"
    for brief in alarming:
        assert brief.title in body
    assert "Evidence support:" in body
    # The run-level roll-up names them too, which is the point of a roll-up; what it
    # must not do is be the only place they are named.
    assert "unsupported rather than evidence-backed" in appendix


def test_an_unanswered_governance_block_leads_the_chapter(rich_session: Session):
    """Order is by consequence. Nothing outranks work a reviewer said to stop."""
    rich_session.exploratory_evidence_accepted = True
    record = load_record(rich_session)
    record.open_governance_blocks = [
        BlockerNote(
            candidate_id="candidate_0001",
            title="A Conformal Alumina Coating Suppresses Electrolyte Decomposition",
            fatal_flaws=["The disposal route for the fluorinated coating is unstated"],
        )
    ]
    advisories = run_advisories(
        record, briefs=tuple(build_idea_briefs(load_record(rich_session)))
    )

    assert advisories[0].title == "Unanswered fatal governance findings"
    assert advisories[0].blocking
    assert "blocked, not approved" in advisories[0].body
    assert [item.blocking for item in advisories].count(True) == 3


def test_the_chapter_opens_the_appendix_rather_than_closing_it(rich_session: Session):
    """Provenance is for auditing the report; this is for deciding whether to act on
    it, so a reader who stops after one chapter has read the right one."""
    report = compile_dossier(rich_session)

    assert report.index(CHAPTER) < report.index("\n# Provenance")


def test_a_fallback_is_warned_about_once_however_many_ideas_the_run_produced(
    rich_session: Session,
):
    """It used to be printed under every overview section the stage fed and again
    under every idea, which on the live report was six copies of one paragraph."""
    report = compile_dossier(rich_session)
    notice = "a fixed template, not the specialist's own reasoning"

    assert report.count(notice) == 1


def test_a_stage_that_did_not_fall_back_raises_no_template_warning():
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.provenance = []

    titles = [item.title for item in run_advisories(record)]

    assert "Stages that produced a template rather than reasoning" not in titles


def test_a_report_with_no_accepted_artifacts_still_compiles_its_chapter():
    """The empty run is the one a compiler is most likely to trip over, and it is
    also the run whose reader most needs to be told nothing here is a finding."""
    session = Session(question="Can a coating help?")
    session.artifacts.append(
        Artifact(
            stage="scope",
            agent="scope_agent",
            artifact_type="stage_bundle",
            schema_name="ResearchPlan",
            content="",
            payload={},
            status=ArtifactStatus.DRAFT,
        )
    )

    report = compile_dossier(session)

    assert CHAPTER in report
    assert "Nothing in this document is a finding" in report


def test_the_waived_gate_says_why_its_total_is_not_the_references_total():
    """Two counts of two populations, four chapters apart, opening the same way.

    Section eight counts the documents the literature search reached and the waiver
    counts the sources the evidence stage admitted. On a live run those were eighty
    and seventy-seven, both introduced as "Twenty-five of the", and nothing anywhere
    said they were counts of different things.
    """
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[
            SourceRecord(url="https://a.example/paper", verification_status="verified"),
            SourceRecord(url="https://b.example/paper"),
            # Recorded without a link, so the reference list cannot number it.
            SourceRecord(url=""),
        ],
    )
    record.citations = CitationRegistry(
        [
            SourceLead(canonical_url="https://a.example/paper", title="A"),
            SourceLead(canonical_url="https://b.example/paper", title="B"),
            SourceLead(canonical_url="https://c.example/paper", title="C"),
            SourceLead(canonical_url="https://d.example/paper", title="D"),
        ]
    )
    body = _waived_gate_advisory(record)[0].body

    assert "One of the three sources admitted was retrieved and checked" in body
    assert "that one is marked as verified where it is cited" in body
    assert (
        "That total counts what the evidence stage admitted, which is not the " in body
    )
    assert (
        "four documents the literature search reached that Key Findings counts" in body
    )
    assert "two documents there are leads the search returned that no admitted " in body
    assert "one source admitted here names no document that can be numbered" in body


def test_the_waived_gate_reconciles_nothing_where_the_two_totals_agree():
    """A sentence explaining a difference, printed where there is none, invents one."""
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import EvidencePacket, SourceLead, SourceRecord
    from coscientist.narrative import CitationRegistry

    session = Session(question="Does a coating change cycle life?")
    session.exploratory_evidence_accepted = True
    record = ResearchRecord(session=session)
    record.evidence = EvidencePacket(
        question=session.question,
        sources=[SourceRecord(url="https://a.example/paper")],
    )
    record.citations = CitationRegistry(
        [SourceLead(canonical_url="https://a.example/paper", title="A")]
    )

    assert "different population" not in _waived_gate_advisory(record)[0].body
