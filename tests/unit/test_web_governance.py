"""The way out of a safety block, from the browser.

The workflow could always be unblocked, and the CLI and the TUI both do it. The
web could not: there was no route, no action and no control. A researcher whose
reflect stage recorded a fatal flaw saw "Governance review blocked", pressed
Accept -- which returns silently, because the gate is checked before anything
is recorded -- and had a session that could never move again. The browser is
the primary interface, so the one state the reflect stage is designed to enter
was a dead end in it.

These tests hold the route, and hold that it refuses the two things that would
make the record worthless: an anonymous decision and an unexplained one.
"""

from __future__ import annotations

import pytest
from fastapi import BackgroundTasks, HTTPException

from app import research_api
from app.research_api import (
    ResearchDecision,
    decide_research_session,
    get_research_session,
)
from coscientist.governance import latest_population, withdrawn_candidate_ids
from coscientist.ledger import ResearchLedger
from coscientist.models import (
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    ReviewSet,
    Session,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective coating extend battery cycle life?"
FLAW = (
    "Annealing PVDF-containing electrodes at 400 C decomposes the binder and "
    "releases hydrogen fluoride."
)
REASON = "Confirmed against the binder datasheet; the protocol cannot be run safely."
NAME = "Dr. Ada Lovelace"


def _candidate(candidate_id: str, title: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        title=title,
        claim=f"{title}.",
        rationale="Because the mechanism predicts it.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        predictions=["Capacity retention improves."],
        falsifier="Retention does not improve.",
    )


@pytest.fixture()
def blocked(tmp_path, monkeypatch) -> str:
    """A persisted session halted on one fatal governance finding."""
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)

    session = Session(question=QUESTION)
    population = CandidatePopulation(
        candidates=[
            _candidate("cand_1", "A conformal alumina coating passivates the surface"),
            _candidate("cand_2", "Anneal the assembled electrode at 400 C"),
        ],
        target_size=2,
    )
    reviews = ReviewSet(
        reviews=[
            CandidateReview(
                id="rev_1",
                candidate_id="cand_2",
                criterion="safety_governance",
                reviewer="ethics_safety_governance",
                recommendation="reject",
                fatal_flaws=[FLAW],
                objections=["No fume extraction is specified."],
            )
        ]
    )
    session.artifacts = [
        Artifact(
            stage="generate",
            agent="generation",
            content="",
            schema_name="CandidatePopulation",
            payload=population.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        ),
        Artifact(
            stage="reflect",
            agent="ethics_safety_governance",
            content="",
            schema_name="ReviewSet",
            payload=reviews.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        ),
    ]
    session.status = "governance_blocked"
    store.save(session)
    return session.id


def _decide(session_id: str, tasks: BackgroundTasks | None = None, **fields) -> dict:
    return decide_research_session(
        session_id, ResearchDecision(**fields), tasks or BackgroundTasks()
    )


def test_the_blocked_session_carries_the_flaw_a_researcher_has_to_answer(blocked):
    """A review id is not something anyone can make a safety decision from."""
    snapshot = get_research_session(blocked, BackgroundTasks())
    assert snapshot["status"] == "governance_blocked"
    blocker = snapshot["governance_blockers"][0]
    assert blocker["review_id"] == "rev_1"
    assert blocker["candidate_title"] == "Anneal the assembled electrode at 400 C"
    assert blocker["fatal_flaws"] == [FLAW]
    assert blocker["objections"] == ["No fume extraction is specified."]


def test_withdrawing_from_the_browser_clears_the_block(blocked):
    snapshot = _decide(
        blocked,
        action="withdraw_hypothesis",
        review_id="rev_1",
        feedback=REASON,
        actor=NAME,
    )
    assert snapshot["status"] == "active"
    assert snapshot["governance_blockers"] == []
    workflow = CoScientistWorkflow.load_from_ledger(blocked, research_api._ledger())
    surviving = CandidatePopulation.model_validate(
        latest_population(workflow.session).payload
    )
    assert [item.id for item in surviving.candidates] == ["cand_1"]
    assert withdrawn_candidate_ids(workflow.session) == {"cand_2"}


def test_overriding_from_the_browser_keeps_the_hypothesis_on_the_record(blocked):
    snapshot = _decide(
        blocked,
        action="override_governance",
        review_id="rev_1",
        feedback="Accepted with a fume hood and a written control plan.",
        actor=NAME,
    )
    assert snapshot["status"] == "active"
    workflow = CoScientistWorkflow.load_from_ledger(blocked, research_api._ledger())
    adjudication = workflow.session.governance_adjudications[0]
    assert adjudication.resolution == "override"
    assert adjudication.adjudicator == NAME
    assert adjudication.fatal_flaws == [FLAW]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"feedback": REASON, "actor": NAME}, "review_id is required"),
        ({"review_id": "rev_1", "actor": NAME}, "written justification"),
        ({"review_id": "rev_1", "feedback": REASON}, "Name the person"),
        (
            {"review_id": "rev_1", "feedback": REASON, "actor": "   "},
            "Name the person",
        ),
    ],
)
def test_a_governance_decision_cannot_be_anonymous_or_unexplained(
    blocked, fields, expected
):
    """The default web actor is a role, not a person.

    "web_researcher" is what every other decision is filed under, and it is
    fine for accepting a stage. Signing off on a hypothesis a reviewer called
    fatally flawed under a generic label is not, so this is the one action that
    refuses it.
    """
    with pytest.raises(HTTPException) as raised:
        _decide(blocked, action="override_governance", **fields)
    assert raised.value.status_code == 409
    assert expected in raised.value.detail
    assert (
        get_research_session(blocked, BackgroundTasks())["status"]
        == "governance_blocked"
    ), "a refused adjudication must leave the block standing"


def test_answering_the_last_finding_starts_the_workflow_moving_again(blocked):
    """Clearing the block leaves the draft the block interrupted.

    A live run proved it: three findings answered, status back to active, and
    the session then sat on an unaccepted reflect draft on a profile that was
    going to accept it automatically. Nothing was blocked and nothing moved --
    a quieter dead end than the one the action exists to clear.
    """
    tasks = BackgroundTasks()
    _decide(
        blocked,
        tasks,
        action="override_governance",
        review_id="rev_1",
        feedback="Accepted with a fume hood and a written control plan.",
        actor=NAME,
    )
    assert tasks.tasks, "the cleared session was left with nothing scheduled"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [(ApprovalProfile.AUTO, "auto"), (ApprovalProfile.MILESTONE, "next")],
)
def test_an_unattended_run_resumes_with_the_worker_that_answers_its_own_gates(
    blocked, profile, expected
):
    """Resuming an auto run with the gate-drafting worker parks it.

    The "next" worker prepares the next gate and stops, which is right for a
    profile with a human on the other side of it and wrong for one without: the
    run comes back from the block, drafts reflect and waits for an acceptance
    that, on this profile, only the auto worker was ever going to give it.
    Three live sessions sat at reflect that way with nothing blocked.
    """
    workflow = CoScientistWorkflow.load_from_ledger(blocked, research_api._ledger())
    workflow.session.approval_profile = profile
    research_api._ledger().save(
        workflow.session, expected_version=workflow.session.version
    )

    tasks = BackgroundTasks()
    _decide(
        blocked,
        tasks,
        action="withdraw_hypothesis",
        review_id="rev_1",
        feedback=REASON,
        actor=NAME,
    )
    assert [task.kwargs["kind"] for task in tasks.tasks] == [expected]


def test_a_finding_answered_while_others_remain_does_not_resume_the_run(blocked):
    session = CoScientistWorkflow.load_from_ledger(
        blocked, research_api._ledger()
    ).session
    reviews = ReviewSet.model_validate(session.artifacts[1].payload)
    reviews.reviews.append(
        CandidateReview(
            id="rev_2",
            candidate_id="cand_1",
            criterion="safety_governance",
            reviewer="ethics_safety_governance",
            recommendation="reject",
            fatal_flaws=["No thermal runaway screening is scheduled."],
        )
    )
    session.artifacts[1].payload = reviews.model_dump(mode="json")
    research_api._ledger().save(session, expected_version=session.version)

    tasks = BackgroundTasks()
    snapshot = _decide(
        blocked,
        tasks,
        action="override_governance",
        review_id="rev_1",
        feedback="Accepted with a fume hood and a written control plan.",
        actor=NAME,
    )
    assert snapshot["status"] == "governance_blocked"
    assert not tasks.tasks
    # Both findings are still on the card, and only one of them still blocks.
    # Sending the open one alone is what made the browser tear the card down and
    # rebuild it as a new card of what was left.
    findings = snapshot["governance_blockers"]
    assert [item["review_id"] for item in findings] == ["rev_1", "rev_2"]
    assert findings[1]["resolution"] is None
    assert findings[0]["resolution"] == {
        "action": "override",
        "actor": NAME,
        "reason": "Accepted with a fume hood and a written control plan.",
        "decided_at": findings[0]["resolution"]["decided_at"],
    }
    # And the answered finding still says which hypothesis it was about.
    assert findings[0]["candidate_title"] == "Anneal the assembled electrode at 400 C"


def test_a_withdrawn_finding_keeps_its_title_after_the_population_is_rewritten(
    blocked,
):
    """Withdrawal rewrites the population without the candidate it dropped.

    Titles were read from the live population only, so the row for the finding
    that had just been answered fell back to a bare ``cand_2`` -- the identifier
    the card exists to avoid making anyone reason from.
    """
    session = CoScientistWorkflow.load_from_ledger(
        blocked, research_api._ledger()
    ).session
    reviews = ReviewSet.model_validate(session.artifacts[1].payload)
    reviews.reviews.append(
        CandidateReview(
            id="rev_2",
            candidate_id="cand_1",
            criterion="safety_governance",
            reviewer="ethics_safety_governance",
            recommendation="reject",
            fatal_flaws=["No thermal runaway screening is scheduled."],
        )
    )
    session.artifacts[1].payload = reviews.model_dump(mode="json")
    research_api._ledger().save(session, expected_version=session.version)

    snapshot = _decide(
        blocked,
        action="withdraw_hypothesis",
        review_id="rev_1",
        feedback=REASON,
        actor=NAME,
    )

    settled = snapshot["governance_blockers"][0]
    assert settled["candidate_id"] == "cand_2"
    assert settled["candidate_title"] == "Anneal the assembled electrode at 400 C"
    assert settled["resolution"]["action"] == "withdraw"


def test_a_stale_review_id_is_reported_rather_than_silently_ignored(blocked):
    with pytest.raises(HTTPException) as raised:
        _decide(
            blocked,
            action="withdraw_hypothesis",
            review_id="rev_gone",
            feedback=REASON,
            actor=NAME,
        )
    assert raised.value.status_code == 409
    assert "rev_1" in raised.value.detail
