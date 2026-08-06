"""The evidence worker's wait loop, with no task queue behind it."""

from __future__ import annotations

import pytest

from app import research_api
from coscientist.evidence import EvidenceStillRunning
from coscientist.ledger import ResearchLedger
from coscientist.orchestration import CoScientistWorkflow


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    monkeypatch.setattr(research_api.time, "sleep", lambda seconds: None)
    return store


def test_a_still_running_pass_is_polled_again_by_the_same_worker(ledger, monkeypatch):
    """It used to call the task enqueuer, which raises when no queue exists.

    The RuntimeError escaped from inside the ``except EvidenceStillRunning``
    handler and killed the worker with the operation left ``queued``, so the
    evidence stage only advanced when a lease expiry happened to restart it --
    one poll every five minutes.
    """
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")

    attempts: list[int] = []

    def drafts(workflow):
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise EvidenceStillRunning("pass 1 is still running")

    monkeypatch.setattr(research_api, "_draft_next_gate", drafts)

    research_api._advance_in_background(session_id, kind="evidence")

    assert len(attempts) == 3
    operation = ledger.operation(session_id)
    assert operation["status"] == "completed"


def test_the_lease_is_held_across_the_wait(ledger, monkeypatch):
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")

    seen: list[str] = []

    def drafts(workflow):
        seen.append(ledger.operation(session_id)["detail"])
        if len(seen) < 2:
            raise EvidenceStillRunning("still running")

    monkeypatch.setattr(research_api, "_draft_next_gate", drafts)
    monkeypatch.setattr(
        ResearchLedger,
        "requeue_expired_operation",
        lambda self, session: pytest.fail("the worker's lease was declared dead"),
    )

    research_api._advance_in_background(session_id, kind="evidence")

    # The operator is told what the wait is for, not just that something is
    # happening.
    assert "Deep Research is still running" in seen[1]


def test_a_worker_that_lost_its_lease_stops(ledger, monkeypatch):
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")

    attempts: list[int] = []

    def drafts(workflow):
        attempts.append(len(attempts))
        raise EvidenceStillRunning("still running")

    monkeypatch.setattr(research_api, "_draft_next_gate", drafts)
    monkeypatch.setattr(
        ResearchLedger, "renew_operation", lambda *args, **kwargs: False
    )

    research_api._advance_in_background(session_id, kind="evidence")

    assert len(attempts) == 1
