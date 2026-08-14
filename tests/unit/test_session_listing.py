"""Every run this server holds, to whoever visits it.

The history panel was a browser's own localStorage. A visitor arriving at the
service saw an empty page however much was running on it -- the question that
started this was "why the sessions not visible when visiting the URL?" -- and a
researcher who opened the site on a second machine could not reach a single run
they had started on the first.

These hold the two things that route has to be: complete, and cheap. A finished
run must be listed beside a running one and a blocked one, and listing must not
go through the loader, which builds a workflow and renders its whole dossier.
"""

from __future__ import annotations

import pytest
from fastapi import BackgroundTasks, HTTPException

from app import research_api
from app.research_api import (
    CreateResearchSession,
    create_research_session,
    delete_research_session,
    get_research_session,
    list_research_sessions,
)
from coscientist.ledger import ResearchLedger
from coscientist.models import ApprovalMode
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Does a protective coating extend cathode cycle life?"


@pytest.fixture()
def ledger(tmp_path, monkeypatch) -> ResearchLedger:
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    return store


def _finished(ledger: ResearchLedger, question: str) -> CoScientistWorkflow:
    flow = CoScientistWorkflow(
        question,
        approval_mode=ApprovalMode.AUTO,
        workflow_version=1,
        ledger=ledger,
    )
    while not flow.done:
        flow.accept(flow.preview(), actor="auto")
    return flow


def test_a_visitor_who_started_nothing_still_sees_what_is_running(ledger):
    create_research_session(
        CreateResearchSession(question=QUESTION, approval_profile="milestone"),
        BackgroundTasks(),
    )

    ((entry,),) = (list_research_sessions()["sessions"],)

    assert entry["question"] == QUESTION
    assert entry["status"] == "active"
    assert entry["stage"] == "scope"
    assert entry["report_available"] is False


def test_a_finished_run_is_listed_as_finished_rather_than_left_out(ledger):
    flow = _finished(ledger, QUESTION)

    ((entry,),) = (list_research_sessions()["sessions"],)

    assert entry["id"] == flow.session.id
    assert entry["report_available"] is True
    assert entry["stage"] == "report"
    assert entry["stage_number"] == entry["stage_count"]
    # Which is what tells the panel to offer the dossier rather than a progress
    # bar, and it is not the "active" every run is created holding.
    assert entry["status"] == "ready_for_report"


def test_the_newest_run_is_first_because_that_is_the_one_being_watched(ledger):
    for question in ("First question?", "Second question?", "Third question?"):
        create_research_session(
            CreateResearchSession(question=question), BackgroundTasks()
        )

    listing = list_research_sessions()["sessions"]

    assert [entry["question"] for entry in listing] == [
        "Third question?",
        "Second question?",
        "First question?",
    ]


def test_the_listing_reports_what_each_run_is_waiting_on(ledger):
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION), BackgroundTasks()
    )
    ledger.set_operation(snapshot["id"], "running", "Searching the literature.", "next")

    ((entry,),) = (list_research_sessions()["sessions"],)

    assert entry["operation"]["status"] == "running"
    assert entry["operation"]["detail"] == "Searching the literature."


def test_listing_does_not_load_a_single_session(ledger, monkeypatch):
    """``load`` deserialises about a megabyte per run once one carries a corpus,
    and the route above it renders the dossier of every run that has finished."""
    _finished(ledger, QUESTION)

    def refuse(session_id: str):
        raise AssertionError(f"the listing loaded {session_id}")

    monkeypatch.setattr(ledger, "load", refuse)

    assert len(list_research_sessions()["sessions"]) == 1


def test_a_request_for_more_than_the_ceiling_gets_the_ceiling(ledger):
    """The limit is a caller's, so it is clamped rather than trusted: this route
    is public, and one request for every session ever run is a page of database
    work anyone passing by can ask for."""
    calls: list[int] = []
    original = ledger.recent_sessions
    ledger.recent_sessions = lambda limit=50: calls.append(limit) or original(limit)

    list_research_sessions(limit=10_000)
    list_research_sessions(limit=0)

    assert calls == [200, 1]


def test_a_run_this_browser_did_not_start_can_be_deleted_by_the_operator(
    ledger, monkeypatch
):
    """The per-session credential is handed out once, to whoever created the run.

    Nothing hands it out again, so a run started on another machine, from the
    command line, or before this service issued tokens at all could be deleted
    by nobody -- and the live deployment reached sixty-nine sessions with no way
    to clear one of them from the page that lists them.
    """
    monkeypatch.setenv("COSCIENTIST_ADMIN_TOKEN", "operator-secret")
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION), BackgroundTasks()
    )
    # The credential this run was born with, gone the way it goes in practice.
    ledger.set_delete_token_hash(snapshot["id"], "")

    with pytest.raises(HTTPException) as refused:
        delete_research_session(snapshot["id"], "not-the-operator")
    assert refused.value.status_code == 403

    delete_research_session(snapshot["id"], "operator-secret")

    assert list_research_sessions()["sessions"] == []


def test_without_an_operator_key_configured_only_the_session_token_deletes(
    ledger, monkeypatch
):
    """Local runs and tests set no key, and an empty one must not become a
    password that any empty header matches."""
    monkeypatch.delenv("COSCIENTIST_ADMIN_TOKEN", raising=False)
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION), BackgroundTasks()
    )

    for credential in ("", "anything"):
        with pytest.raises(HTTPException):
            delete_research_session(snapshot["id"], credential)

    delete_research_session(snapshot["id"], snapshot["deletion_token"])

    assert list_research_sessions()["sessions"] == []


def test_the_operator_is_told_when_the_run_is_already_gone(ledger, monkeypatch):
    monkeypatch.setenv("COSCIENTIST_ADMIN_TOKEN", "operator-secret")

    with pytest.raises(HTTPException) as missing:
        delete_research_session("session_none", "operator-secret")

    assert missing.value.status_code == 404


def test_a_run_that_is_not_here_says_so_whatever_credential_is_offered(
    ledger, monkeypatch
):
    """A row can outlive the run it names, and the answer has to say which it is.

    The ledger reports "no such session" and "wrong token" as the same false, so
    both used to come back 403 -- and a browser holding a stale token for a run
    deleted somewhere else read that as "you may not", kept the row, and offered
    the button again. One row on the live deployment was refused eight times in
    ninety seconds. Which sessions exist is on the public listing, so answering
    404 here tells a caller nothing it could not already read.
    """
    monkeypatch.delenv("COSCIENTIST_ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as missing:
        delete_research_session("session_none", "a-token-from-a-run-long-gone")

    assert missing.value.status_code == 404


def test_a_wrong_token_for_a_run_that_is_here_is_still_refused(ledger, monkeypatch):
    """Saying 404 for what is absent must not soften what is present."""
    monkeypatch.delenv("COSCIENTIST_ADMIN_TOKEN", raising=False)
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION), BackgroundTasks()
    )

    with pytest.raises(HTTPException) as refused:
        delete_research_session(snapshot["id"], "not-this-run's-token")

    assert refused.value.status_code == 403
    assert [entry["id"] for entry in list_research_sessions()["sessions"]] == [
        snapshot["id"]
    ]


def test_a_run_whose_worker_died_stops_calling_itself_active(ledger):
    """A worker records why it stopped and leaves the session where it was.

    Nothing said so. The run stayed ``active``, the workspace headed it
    "Workflow active", and the history panel offered it as work in progress --
    four runs on the live deployment sat like that for hours, three of them
    killed by a specialist closing its event stream without answering. The
    landing screen then opened one of them and watched a spinner that had
    nothing behind it.
    """
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION, approval_profile="milestone"),
        BackgroundTasks(),
    )
    ledger.set_operation(
        snapshot["id"], "failed", "The specialist closed its stream.", "next"
    )

    ((entry,),) = (list_research_sessions()["sessions"],)
    assert entry["status"] == "failed"
    assert get_research_session(snapshot["id"], BackgroundTasks())["status"] == (
        "failed"
    )


def test_a_run_picked_back_up_is_active_again(ledger):
    """The failure is read, never written, so a retry needs nothing undone."""
    snapshot = create_research_session(
        CreateResearchSession(question=QUESTION, approval_profile="milestone"),
        BackgroundTasks(),
    )
    ledger.set_operation(snapshot["id"], "failed", "Gone.", "next")
    ledger.set_operation(snapshot["id"], "queued", "Trying again.", "next")

    ((entry,),) = (list_research_sessions()["sessions"],)
    assert entry["status"] == "active"


def test_a_finished_run_is_not_relabelled_by_a_failure_behind_it(ledger):
    """Only a run that still claims to be running is corrected.

    A run stopped by a researcher, or one holding a finished dossier, says
    something the operation cannot improve on -- and an export that failed
    after the fact must not turn a readable report into a failure.
    """
    flow = _finished(ledger, QUESTION)
    ledger.set_operation(flow.session.id, "failed", "The PDF export died.", "export")

    ((entry,),) = (list_research_sessions()["sessions"],)
    assert entry["status"] == flow.session.status
    assert entry["status"] != "failed"
