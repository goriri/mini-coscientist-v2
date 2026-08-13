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
from fastapi import BackgroundTasks

from app import research_api
from app.research_api import (
    CreateResearchSession,
    create_research_session,
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
