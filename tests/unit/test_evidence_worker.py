"""The evidence worker's wait loop, with no task queue behind it."""

from __future__ import annotations

import threading
import time
from time import sleep as real_sleep

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


def test_the_lease_is_renewed_while_a_long_stage_is_still_working(ledger, monkeypatch):
    """The evidence stage is one call that runs for many minutes.

    It outlived its five-minute lease, so the expiry sweep declared a working
    worker dead and started a second one beside it every five minutes: attempt
    five, twenty-five minutes in, still on Deep Research pass seven and no
    closer to a gate.
    """
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "next")

    monkeypatch.setattr(research_api, "OPERATION_LEASE_SECONDS", 1)
    monkeypatch.setattr(research_api, "OPERATION_HEARTBEAT_SECONDS", 0.05)

    def slow_stage(workflow):
        # Three lease lifetimes' worth of work.
        real_sleep(3.2)

    monkeypatch.setattr(research_api, "_draft_next_gate", slow_stage)

    swept: list[bool] = []
    original = ResearchLedger.requeue_expired_operation

    def sweep(self, session):
        expired = original(self, session)
        swept.append(expired)
        return expired

    monkeypatch.setattr(ResearchLedger, "requeue_expired_operation", sweep)

    worker = threading.Thread(
        target=research_api._advance_in_background,
        args=(session_id,),
        kwargs={"kind": "next"},
    )
    worker.start()
    # What a browser polling the session does throughout.
    while worker.is_alive():
        real_sleep(0.2)
        ledger.requeue_expired_operation(session_id)
    worker.join()

    assert swept and not any(swept), "a working worker was declared dead"
    assert ledger.operation(session_id)["status"] == "completed"


def test_the_heartbeat_leaves_the_researchers_message_alone(ledger, monkeypatch):
    """It has nothing to add to what the worker last said it was doing."""
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")
    assert ledger.claim_operation(session_id, "worker-a", detail="Polling pass 3")

    stop = threading.Event()
    beat = threading.Thread(
        target=research_api._hold_operation_lease,
        args=(session_id, "worker-a", stop),
        daemon=True,
    )
    monkeypatch.setattr(research_api, "OPERATION_HEARTBEAT_SECONDS", 0.05)
    beat.start()
    real_sleep(0.2)
    stop.set()
    beat.join(timeout=2)

    assert ledger.operation(session_id)["detail"] == "Polling pass 3"


def test_a_stage_that_outlives_its_lease_is_not_declared_dead(ledger, monkeypatch):
    """Discovery polls every Deep Research pass inside one call.

    That is many minutes against a five-minute lease. Without a heartbeat the
    expiry sweep started a second worker beside the first every five minutes,
    and the evidence stage restarted forever instead of finishing.
    """
    monkeypatch.setattr(research_api, "OPERATION_LEASE_SECONDS", 1)
    monkeypatch.setattr(research_api, "OPERATION_HEARTBEAT_SECONDS", 0.05)
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "next")

    def slow(workflow):
        # Three lease lifetimes' worth of work.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            assert not ledger.requeue_expired_operation(session_id), (
                "a working worker was declared dead"
            )
            time.sleep(0.05)

    monkeypatch.setattr(research_api, "_draft_next_gate", slow)

    research_api._advance_in_background(session_id, kind="next")

    assert ledger.operation(session_id)["status"] == "completed"


def test_the_heartbeat_leaves_the_message_the_worker_wrote(ledger, monkeypatch):
    """The researcher is reading it. A beat has nothing to add."""
    monkeypatch.setattr(research_api, "OPERATION_HEARTBEAT_SECONDS", 0.05)
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")

    seen: list[str] = []

    def drafts(workflow):
        if not seen:
            seen.append("first")
            raise EvidenceStillRunning("still running")
        seen.append(ledger.operation(session_id)["detail"])

    monkeypatch.setattr(research_api, "_draft_next_gate", drafts)
    # Real sleeping, so the heartbeat gets several beats inside the wait. Held
    # by reference first: research_api.time is the time module, so patching it
    # would replace the sleep this lambda is about to call with itself.
    sleep = time.sleep
    monkeypatch.setattr(research_api.time, "sleep", lambda seconds: sleep(0.3))

    research_api._advance_in_background(session_id, kind="evidence")

    assert "Deep Research is still running" in seen[1]


def test_a_worker_stops_beating_once_its_stage_is_done(ledger, monkeypatch):
    """Otherwise a finished session keeps a lease nothing is holding."""
    monkeypatch.setattr(research_api, "OPERATION_LEASE_SECONDS", 1)
    monkeypatch.setattr(research_api, "OPERATION_HEARTBEAT_SECONDS", 0.05)
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "next")
    monkeypatch.setattr(research_api, "_draft_next_gate", lambda workflow: None)

    research_api._advance_in_background(session_id, kind="next")

    before = ledger.operation(session_id)["lease_expires_at"]
    time.sleep(0.3)
    assert ledger.operation(session_id)["lease_expires_at"] == before


def test_a_stage_narrating_its_progress_keeps_the_lease_it_is_holding(ledger):
    """Through ``renew_operation``, and the distinction is the whole test.

    ``set_operation`` clears ``lease_owner`` and ``lease_expires_at`` on its way
    past. A stage that used it to say "Checked 12 of 28 sources" would, with
    that sentence, hand its own session to the expiry sweep -- and the sweep
    starts a second worker on an evidence stage that is forty minutes and
    twenty-four dollars into the first one.
    """
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")
    assert ledger.claim_operation(session_id, "worker-a", detail="Polling pass 8")

    research_api._progress_writer(session_id, "worker-a", "evidence")(
        "Checked 12 of 28 sources against what the document actually says."
    )

    operation = ledger.operation(session_id)
    assert operation["detail"] == (
        "Checked 12 of 28 sources against what the document actually says."
    )
    assert operation["lease_owner"] == "worker-a"
    assert operation["lease_expires_at"]


def test_the_stage_the_worker_drives_is_given_a_way_to_speak(ledger, monkeypatch):
    """The writer is attached to the workflow the worker loaded, or the stage
    holds whatever sentence the last caller left over a stage that has moved
    on -- twenty-five minutes of "Deep Research is still running" on a live run
    that had finished searching."""
    flow = CoScientistWorkflow("Question", ledger=ledger)
    session_id = flow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "next")

    def drafts(workflow):
        workflow._note("Opening 28 sources to check what the documents say.")

    monkeypatch.setattr(research_api, "_draft_next_gate", drafts)
    seen: list[str] = []
    monkeypatch.setattr(
        research_api,
        "_set_operation",
        lambda *args, **kwargs: seen.append(ledger.operation(session_id)["detail"]),
    )

    research_api._advance_in_background(session_id, kind="next")

    assert seen == ["Opening 28 sources to check what the documents say."]
