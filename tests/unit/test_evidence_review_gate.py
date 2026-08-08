"""The opt-in stop on the evidence base, before the generators reason over it.

The milestone profile does not count evidence as a milestone: discovery runs as
internal work and the first thing the researcher is handed is eight hypotheses
built on whatever came back. That is right when the corpus is sound. When it is
thin it is only visible by reading it, and by the generate gate four strategies
have already spent themselves on it. This gate is the cheap place to find out.
"""

import pytest
from fastapi import BackgroundTasks

from app import research_api
from app.research_api import CreateResearchSession, create_research_session
from coscientist.cli import main as cli_main
from coscientist.ledger import ResearchLedger
from coscientist.models import ApprovalProfile, Session
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective coating improve battery cycle life?"


def _at_evidence(**kwargs) -> CoScientistWorkflow:
    flow = CoScientistWorkflow(
        QUESTION,
        workflow_version=2,
        **kwargs,
    )
    flow.accept(flow.preview())
    assert flow.stage == "evidence"
    return flow


def test_milestone_runs_through_evidence_when_the_gate_was_not_asked_for():
    flow = _at_evidence(approval_profile=ApprovalProfile.MILESTONE)

    assert flow.session.evidence_review is False
    assert flow.requires_human_approval is False


def test_the_run_that_opted_in_stops_on_its_evidence_base():
    flow = _at_evidence(
        approval_profile=ApprovalProfile.MILESTONE, evidence_review=True
    )

    assert flow.session.evidence_review is True
    assert flow.requires_human_approval is True
    # Not a new milestone for everything else: the toggle buys one stop, and the
    # stages the profile already ran internally still run internally.
    flow.accept_exploratory_evidence()
    flow.accept(flow.preview())
    assert flow.stage == "generate"
    assert flow.requires_human_approval is False


def test_the_gate_refuses_to_be_stepped_over_automatically():
    """The whole point is that somebody reads it, so the auto path is closed."""
    flow = _at_evidence(
        approval_profile=ApprovalProfile.MILESTONE, evidence_review=True
    )
    flow.accept_exploratory_evidence()

    with pytest.raises(ValueError, match="Automatic decisions are disabled"):
        flow.accept(flow.preview(), automatic=True)


def test_advancing_to_the_next_human_gate_stops_at_evidence():
    """``advance_to_human_gate`` is what the web API calls after every accept."""
    flow = CoScientistWorkflow(
        QUESTION,
        approval_profile=ApprovalProfile.MILESTONE,
        evidence_review=True,
        workflow_version=2,
    )
    flow.accept(flow.preview())
    flow.accept_exploratory_evidence()
    flow.advance_to_human_gate()

    assert flow.stage == "evidence"


def test_an_unattended_run_does_not_record_a_gate_it_will_never_stop_at():
    """``run_auto`` accepts every draft it drafts; nobody is standing there.

    Recording the request anyway would put a promise in the session that the run
    is built to break, and the launcher reads this field back to tell the
    researcher what their run is actually going to do.
    """
    flow = CoScientistWorkflow(
        QUESTION,
        approval_profile=ApprovalProfile.AUTO,
        evidence_review=True,
        workflow_version=2,
    )

    assert flow.session.evidence_review is False


def test_a_session_saved_before_the_gate_existed_loads_as_a_run_without_it():
    saved = Session(question="Can a coating help?").to_dict()
    del saved["evidence_review"]

    assert Session.from_dict(saved).evidence_review is False


# ---------------------------------------------------------------------------
# The two surfaces that set it
# ---------------------------------------------------------------------------


def _created(store, **body) -> dict:
    return create_research_session(
        CreateResearchSession(question=QUESTION, **body), BackgroundTasks()
    )


@pytest.fixture()
def api(tmp_path, monkeypatch):
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    return store


def test_an_api_caller_that_does_not_ask_for_the_gate_does_not_get_one(api):
    """A script polling this API to completion would park at evidence forever."""
    snapshot = _created(api)

    assert snapshot["evidence_review"] is False


def test_the_launcher_gets_the_gate_it_asked_for_back_in_the_snapshot(api):
    snapshot = _created(api, evidence_review=True)

    assert snapshot["evidence_review"] is True


def test_the_snapshot_reports_the_gate_the_run_has_rather_than_the_one_requested(api):
    """Auto drops it, and the launcher shows what the run will actually do."""
    snapshot = _created(api, approval_profile="auto", evidence_review=True)

    assert snapshot["evidence_review"] is False


def test_the_cli_refuses_the_gate_flag_on_a_resumed_run(tmp_path):
    """The gate is configured with the run, like the model and the language: a
    session parked mid-pipeline has already passed the stage it would stop at."""
    path = tmp_path / "session.json"
    CoScientistWorkflow(QUESTION).save(path)

    with pytest.raises(SystemExit, match="configure a new run"):
        cli_main(["run", "--resume", str(path), "--evidence-review"])
