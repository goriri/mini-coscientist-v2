import hashlib
from pathlib import Path

import pytest

from coscientist.ledger import ConcurrentSessionUpdate, ResearchLedger
from coscientist.models import ApprovalMode
from coscientist.orchestration import CoScientistWorkflow


def test_sqlite_restart_recovery_for_human_mode(tmp_path: Path):
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow(
        "Does intervention X change outcome Y?",
        approval_mode=ApprovalMode.HUMAN,
        workflow_version=1,
        ledger=ledger,
    )
    draft = flow.preview()
    flow.accept(draft, actor="researcher")

    resumed = CoScientistWorkflow.load_from_ledger(flow.session.id, ledger)
    assert resumed.stage == "generate"
    assert resumed.session.decisions[-1].automatic is False
    assert [event.event_type for event in ledger.events(flow.session.id)] == [
        "session_created",
        "stage_drafted",
        "stage_accepted",
    ]


def test_sqlite_restart_recovery_for_auto_mode(tmp_path: Path):
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow(
        "Does intervention X change outcome Y?",
        approval_mode=ApprovalMode.AUTO,
        workflow_version=1,
        ledger=ledger,
    )
    flow.run_auto()

    resumed = CoScientistWorkflow.load_from_ledger(flow.session.id, ledger)
    assert resumed.done
    assert all(
        decision.automatic
        for decision in resumed.session.decisions
        if decision.action == "accept"
    )


def test_optimistic_lock_rejects_stale_writer(tmp_path: Path):
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow("Question", ledger=ledger)
    stale = ledger.load(flow.session.id)
    flow.preview()
    stale.status = "stale_write"
    with pytest.raises(ConcurrentSessionUpdate):
        ledger.save(stale)


def test_operation_lease_and_bearer_deletion(tmp_path: Path):
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow("Question", ledger=ledger)
    token_hash = hashlib.sha256(b"delete-secret").hexdigest()
    ledger.set_delete_token_hash(flow.session.id, token_hash)

    ledger.set_operation(flow.session.id, "queued", "Waiting", "initial")
    assert ledger.claim_operation(
        flow.session.id, "worker-a", detail="Working", lease_seconds=60
    )
    assert not ledger.claim_operation(
        flow.session.id, "worker-b", detail="Duplicate", lease_seconds=60
    )
    operation = ledger.operation(flow.session.id)
    assert operation["status"] == "running"
    assert operation["detail"] == "Working"
    assert operation["attempt"] == 1

    assert ledger.delete_session(flow.session.id, "wrong") is False
    assert ledger.delete_session(flow.session.id, token_hash) is True
    with pytest.raises(KeyError):
        ledger.load(flow.session.id)


def test_a_renewal_keeps_the_lease_with_the_worker_that_holds_it(tmp_path: Path):
    """Polling Deep Research outruns a five-minute lease.

    Without a renewal the worker is declared dead mid-wait and a second one is
    started beside it, both writing to the same session.
    """
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow("Question", ledger=ledger)
    ledger.set_operation(flow.session.id, "queued", "Waiting", "evidence")
    assert ledger.claim_operation(
        flow.session.id, "worker-a", detail="Working", lease_seconds=-1
    )
    # The lease is already expired, so the session is up for grabs.
    assert ledger.requeue_expired_operation(flow.session.id)

    assert ledger.claim_operation(
        flow.session.id, "worker-a", detail="Working", lease_seconds=-1
    )
    assert ledger.renew_operation(
        flow.session.id, "worker-a", detail="Polling", lease_seconds=600
    )
    operation = ledger.operation(flow.session.id)
    assert operation["status"] == "running"
    assert operation["detail"] == "Polling"
    # And now it is not: the renewal is what stops the second worker.
    assert not ledger.requeue_expired_operation(flow.session.id)
    assert not ledger.claim_operation(
        flow.session.id, "worker-b", detail="Duplicate", lease_seconds=60
    )


def test_a_worker_whose_lease_was_taken_away_cannot_renew_it(tmp_path: Path):
    """Which is how it learns to stop rather than write over the new owner."""
    ledger = ResearchLedger(tmp_path / "research.db")
    flow = CoScientistWorkflow("Question", ledger=ledger)
    ledger.set_operation(flow.session.id, "queued", "Waiting", "evidence")
    assert ledger.claim_operation(
        flow.session.id, "worker-a", detail="Working", lease_seconds=-1
    )
    assert ledger.requeue_expired_operation(flow.session.id)
    assert ledger.claim_operation(
        flow.session.id, "worker-b", detail="Taking over", lease_seconds=600
    )

    assert not ledger.renew_operation(
        flow.session.id, "worker-a", detail="Polling", lease_seconds=600
    )
    assert ledger.operation(flow.session.id)["detail"] == "Taking over"
