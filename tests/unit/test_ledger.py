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
