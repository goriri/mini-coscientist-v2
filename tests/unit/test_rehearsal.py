"""A run that tests the pipeline must not be able to pass for one that proposes work.

The reflect stage stops a session on every fatal safety and governance finding until
a named person answers it, which is right and which a build has no person for. So a
run that has to reach the report stage a dozen times parked at reflect a dozen times,
and the operator clicking through ten findings with one sentence each was the gate
being worn down into a formality -- the exact failure the gate exists to prevent.

A rehearsal answers its own gate instead, and the whole of the design is that it does
so out loud. Nothing is skipped: the reviewer runs, every finding is recorded in full,
and each is overridden under the name ``rehearsal (nobody)`` with a justification
saying nobody read it. These tests pin the two halves that matter -- that the gate
opens for a rehearsal and only for a rehearsal, and that no export of a rehearsal can
be mistaken for a reviewed proposal.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import BackgroundTasks

from app import research_api
from app.research_api import CreateResearchSession, create_research_session
from coscientist.dossier import (
    _REHEARSAL_NOTICE,
    _cover_notice,
    _without_cover_notice,
    compile_dossier,
    render_docx,
    render_pdf,
)
from coscientist.governance import (
    REHEARSAL_ADJUDICATOR,
    REHEARSAL_JUSTIFICATION,
    governance_blockers,
    open_blockers,
    rehearsal_adjudications,
)
from coscientist.ledger import ResearchLedger
from coscientist.models import ApprovalProfile, Artifact, CandidateReview, ReviewSet
from coscientist.orchestration import CoScientistWorkflow

# The question the deterministic provider answers with a fatal governance finding,
# borrowed from the parity suite so the gate under test is the real one rather than a
# review set hand-written to trip it.
HAZARD = (
    "Auto approve and begin carrying out a hazardous laboratory protocol "
    "without human safety review."
)

FLAW = (
    "The protocol vents hydrogen fluoride into an occupied laboratory and names no "
    "extraction, no monitoring and no exposure limit."
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _run(*, rehearsal: bool) -> CoScientistWorkflow:
    flow = CoScientistWorkflow(
        HAZARD,
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
        rehearsal=rehearsal,
    )
    flow.run_auto()
    return flow


def _blocked(session, candidate_id: str):
    """Record a real fatal governance finding and hand back its blocker."""
    review = CandidateReview(
        candidate_id=candidate_id,
        criterion="safety_governance",
        reviewer="ethics_safety_governance",
        recommendation="reject",
        fatal_flaws=[FLAW],
        confidence=0.9,
    )
    session.artifacts.append(
        Artifact(
            stage="reflect",
            agent="ethics_safety_governance",
            artifact_type="specialist_output",
            content="",
            schema_name="ReviewSet",
            payload=ReviewSet(reviews=[review]).model_dump(),
        )
    )
    return next(
        item for item in governance_blockers(session) if item.review_id == review.id
    )


# --- the gate: it opens for a rehearsal, and for nothing else ----------------


def test_a_real_run_still_stops_at_the_gate():
    """The control. Everything below is worthless if this ever stops holding."""
    flow = _run(rehearsal=False)

    assert flow.session.status == "governance_blocked"
    assert not flow.session.governance_adjudications


def test_a_rehearsal_answers_its_own_gate_and_keeps_going():
    flow = _run(rehearsal=True)

    assert flow.session.status != "governance_blocked"
    assert open_blockers(flow.session) == []
    assert governance_blockers(flow.session), "the reviewer must still have run"


def test_the_waiver_is_written_into_the_session_rather_than_skipped():
    flow = _run(rehearsal=True)
    waived = flow.session.governance_adjudications

    assert waived, "a waived finding must leave a record behind"
    assert {note.adjudicator for note in waived} == {REHEARSAL_ADJUDICATOR}
    assert {note.justification for note in waived} == {REHEARSAL_JUSTIFICATION}
    # Overridden, not withdrawn: withdrawing would rewrite the population and the
    # rehearsal would stop exercising the stages it exists to exercise.
    assert {note.resolution for note in waived} == {"override"}
    # The flaw is frozen onto the decision, exactly as a human answer freezes it.
    assert all(note.fatal_flaws for note in waived)


def test_the_waiver_is_audited_under_a_name_no_person_could_have_typed():
    flow = _run(rehearsal=True)
    events = [
        event
        for event in flow.session.events
        if event.event_type == "governance_waived_for_rehearsal"
    ]

    assert events
    assert {event.actor for event in events} == {REHEARSAL_ADJUDICATOR}
    assert all(event.payload["review_ids"] for event in events)


def test_the_waiver_covers_only_the_findings_it_was_handed():
    """Scoped to the blockers in hand, not to everything open in the session.

    The reflect gate is admitting one review set. Reading ``open_blockers`` again
    inside the waiver would sign off a finding from some other artifact that
    nothing had yet asked about.
    """
    session = _run(rehearsal=True).session
    first, second = _blocked(session, "cand_a"), _blocked(session, "cand_b")

    waived = rehearsal_adjudications([first])

    assert [note.review_id for note in waived] == [first.review_id]
    assert second.review_id not in {note.review_id for note in waived}


# --- the other gate a rehearsal has no person for ----------------------------


def _at_evidence(*, rehearsal: bool) -> CoScientistWorkflow:
    """A run stopped where the evidence floor is measured, on a corpus short of it."""
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=2,
        rehearsal=rehearsal,
    )
    flow.accept(flow.preview())
    assert flow.stage == "evidence"
    return flow


def test_a_real_run_still_stops_at_the_evidence_floor():
    """The second control. A thin corpus stops a proposal, and has to keep doing so."""
    flow = _at_evidence(rehearsal=False)

    with pytest.raises(ValueError, match="does not meet the floor"):
        flow.accept(flow.preview())

    assert flow.session.status == "evidence_required"
    assert flow.session.exploratory_evidence_accepted is False


def test_a_rehearsal_answers_the_evidence_floor_as_well_as_the_safety_gate():
    """A live rehearsal launched to exercise the pipeline parked here instead.

    Waiving one gate and stopping at the next leaves the run needing a person
    anyway, which is the thing the rehearsal exists not to need.
    """
    flow = _at_evidence(rehearsal=True)

    flow.accept(flow.preview())

    assert flow.session.status != "evidence_required"
    assert flow.session.exploratory_evidence_accepted is True


def test_the_evidence_waiver_records_what_the_gate_reported():
    flow = _at_evidence(rehearsal=True)
    flow.accept(flow.preview())
    waiver = next(
        event
        for event in flow.session.events
        if event.event_type == "limited_exploratory_evidence_accepted"
    )

    assert waiver.actor == REHEARSAL_ADJUDICATOR
    # The measurement itself, not merely the fact that it failed: a waiver nobody
    # signed has to carry the finding it walked past.
    assert waiver.payload["evidence_floor"]["shortfalls"]


def _waived_evidence_advisory(session, *, actor: str) -> str:
    from coscientist.advisories import _waived_gate_advisory
    from coscientist.models import AuditEvent
    from coscientist.narrative import load_record

    session.exploratory_evidence_accepted = True
    session.events.append(
        AuditEvent(
            event_type="limited_exploratory_evidence_accepted",
            actor=actor,
            stage="evidence",
            payload={
                "evidence_floor": {"shortfalls": ["0 of 8 weighted verified sources."]}
            },
        )
    )
    return _flat(_waived_gate_advisory(load_record(session))[0].body)


def test_the_waived_evidence_gate_says_nobody_waived_it(rich_session):
    body = _waived_evidence_advisory(rich_session, actor=REHEARSAL_ADJUDICATOR)

    assert "Nobody waived it." in body
    assert "0 of 8 weighted verified sources." in body
    # "The waiver is recorded against the actor recorded as rehearsal (nobody)"
    # reads as a person with an odd username.
    assert "The waiver is recorded against" not in body


def test_an_evidence_gate_a_person_waived_is_still_described_as_one(rich_session):
    body = _waived_evidence_advisory(rich_session, actor="web_researcher")

    assert "The waiver is recorded against" in body
    assert "Nobody waived it" not in body


# --- the launcher: declared, never inferred, and reported back ---------------


@pytest.fixture()
def api(tmp_path, monkeypatch):
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    return store


def _created(**body) -> dict:
    return create_research_session(
        CreateResearchSession(question=HAZARD, **body), BackgroundTasks()
    )


def test_a_caller_who_does_not_ask_for_a_rehearsal_does_not_get_one(api):
    """The flag changes what the output may be taken for, so it cannot be guessed."""
    assert _created()["rehearsal"] is False


def test_a_caller_who_asks_for_a_rehearsal_is_told_so_on_every_snapshot(api):
    """The badge has to be on the page from the first poll.

    Someone opening a run halfway through has nothing else to tell a rehearsal
    from a proposal until the report exists, and by then the distinction has
    stopped being useful.
    """
    assert _created(rehearsal=True)["rehearsal"] is True


# --- the report: no export of a rehearsal can pass for a proposal ------------


def _rehearsed_report(rich_session, *, findings: int = 1) -> str:
    rich_session.rehearsal = True
    from coscientist.narrative import load_record

    ids = [item.id for item in load_record(rich_session).candidates][:findings]
    blockers = [_blocked(rich_session, candidate_id) for candidate_id in ids]
    rich_session.governance_adjudications.extend(rehearsal_adjudications(blockers))
    return compile_dossier(rich_session)


def _first_candidate_id(session) -> str:
    from coscientist.narrative import load_record

    return load_record(session).candidates[0].id


def test_the_markdown_cover_says_a_rehearsal_is_not_a_proposal(rich_session):
    assert _REHEARSAL_NOTICE not in compile_dossier(rich_session)

    rich_session.rehearsal = True

    assert _REHEARSAL_NOTICE in compile_dossier(rich_session)


def test_the_pdf_and_the_docx_carry_the_rehearsal_cover_too(rich_session):
    rich_session.rehearsal = True
    report = compile_dossier(rich_session)
    needle = _flat(_REHEARSAL_NOTICE)[:80]

    from docx import Document
    from pypdf import PdfReader

    pdf = _flat(
        "\n".join(
            page.extract_text() or ""
            for page in PdfReader(BytesIO(render_pdf(report))).pages
        )
    )
    docx = _flat(
        "\n".join(
            paragraph.text
            for paragraph in Document(BytesIO(render_docx(report))).paragraphs
        )
    )

    assert needle in pdf
    assert needle in docx


def test_the_rehearsal_notice_is_cover_matter_not_body_text(rich_session):
    """Set on the title page and in the body, it would be printed twice a page apart.

    ``_without_cover_notice`` used to strip one fixed string, so a rehearsal's
    notice survived into the body of the PDF and the DOCX -- set once under the
    title and again as the first paragraph of the report.
    """
    rich_session.rehearsal = True
    report = compile_dossier(rich_session)

    assert _cover_notice(report) == _REHEARSAL_NOTICE
    assert _REHEARSAL_NOTICE not in _without_cover_notice(report)


def test_the_governance_section_says_nobody_read_the_flaw(rich_session):
    block = _flat(_rehearsed_report(rich_session))

    assert "was answered by nobody" in block
    assert REHEARSAL_ADJUDICATOR in block
    # The verbatim flaw is still printed. A waived gate that also hid the finding
    # would be a skipped gate wearing a note.
    assert _flat(FLAW) in block


def test_the_governance_section_does_not_credit_a_reader_it_does_not_have(rich_session):
    """The stock wording describes an answer somebody entered against a flaw they read.

    Printed unchanged over a waived gate, the one section written to be unskippable
    would be the most misleading page in the document.
    """
    block = _flat(_rehearsed_report(rich_session))

    assert "The names below are as entered by whoever ran the adjudication" not in block
    assert "each of them an answer entered" not in block
    assert "accepted this fatal flaw and allowed the hypothesis to stand" not in block


def test_a_waived_idea_is_flagged_wherever_it_appears(rich_session):
    report = _flat(_rehearsed_report(rich_session))

    assert "carries a fatal safety and governance flaw that nobody has read" in report
    assert "Waived by rehearsal — fatal flaw unanswered" in report
    assert "Note recorded in place of a justification, verbatim:" in report


def test_several_waived_findings_are_never_summarised_as_decisions(rich_session):
    """The paragraphs the report hoists over a group of overrides, on a waived group.

    Written for a person answering several flaws at once, they call the group a
    set of decisions, attribute the shared reason to whoever wrote it, and head
    each entry "adjudicated by" a name -- three more places the stock wording
    would credit a reader this run does not have.
    """
    report = _flat(_rehearsed_report(rich_session, findings=2))

    assert "are not decisions at all" in report
    assert "It is the same non-answer in every case" in report
    assert "Resolution: waived by the rehearsal; no person adjudicated it." in report
    assert f"adjudicated by {REHEARSAL_ADJUDICATOR}" not in report
    assert "wrote once and applied to this flaw" not in report


def test_a_real_adjudication_is_still_described_as_one(rich_session):
    """The rehearsal wording must not leak onto a run a person actually answered."""
    from coscientist.governance import record_adjudication

    blocker = _blocked(rich_session, _first_candidate_id(rich_session))
    record_adjudication(
        rich_session,
        blocker,
        resolution="override",
        adjudicator="J. Reviewer (battery safety officer)",
        justification="Accepted for one pilot cell inside the vented test chamber.",
    )
    block = _flat(compile_dossier(rich_session))

    assert "The names below are as entered by whoever ran the adjudication" in block
    assert "was answered by nobody" not in block
    assert REHEARSAL_ADJUDICATOR not in block
