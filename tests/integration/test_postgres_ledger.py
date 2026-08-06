import hashlib
import os

import pytest

from coscientist.ledger import PostgresResearchLedger
from coscientist.orchestration import CoScientistWorkflow


@pytest.mark.skipif(
    not os.getenv("COSCIENTIST_TEST_POSTGRES_URL"),
    reason="Set COSCIENTIST_TEST_POSTGRES_URL to exercise PostgreSQL parity.",
)
def test_postgres_restart_operations_and_deletion():
    ledger = PostgresResearchLedger(os.environ["COSCIENTIST_TEST_POSTGRES_URL"])
    workflow = CoScientistWorkflow(
        "Does intervention X change outcome Y?", ledger=ledger
    )
    draft = workflow.preview()
    workflow.accept(draft, actor="researcher")

    resumed = CoScientistWorkflow.load_from_ledger(workflow.session.id, ledger)
    assert resumed.stage == "generate"
    assert ledger.events(workflow.session.id)[-1].event_type == "stage_accepted"

    ledger.set_operation(workflow.session.id, "queued", "Waiting", "next")
    assert ledger.claim_operation(
        workflow.session.id,
        "integration-worker",
        detail="Working",
        lease_seconds=60,
    )
    assert ledger.operation(workflow.session.id)["status"] == "running"

    token_hash = hashlib.sha256(b"postgres-delete-token").hexdigest()
    ledger.set_delete_token_hash(workflow.session.id, token_hash)
    assert ledger.delete_session(workflow.session.id, token_hash)


@pytest.mark.skipif(
    not os.getenv("COSCIENTIST_TEST_POSTGRES_URL"),
    reason="Set COSCIENTIST_TEST_POSTGRES_URL to exercise PostgreSQL parity.",
)
def test_postgres_lease_renewal_holds_off_the_recovery_sweep():
    """The deployed ledger is this one, so the renewal has to work here too."""
    ledger = PostgresResearchLedger(os.environ["COSCIENTIST_TEST_POSTGRES_URL"])
    workflow = CoScientistWorkflow("Does a coating change cycle life?", ledger=ledger)
    session_id = workflow.session.id
    ledger.set_operation(session_id, "queued", "Waiting", "evidence")
    assert ledger.claim_operation(
        session_id, "worker-a", detail="Working", lease_seconds=-1
    )

    assert ledger.renew_operation(
        session_id, "worker-a", detail="Polling", lease_seconds=600
    )
    assert ledger.operation(session_id)["detail"] == "Polling"
    assert not ledger.requeue_expired_operation(session_id)
    assert not ledger.renew_operation(
        session_id, "worker-b", detail="Not mine", lease_seconds=600
    )

    token_hash = hashlib.sha256(b"postgres-renewal-token").hexdigest()
    ledger.set_delete_token_hash(session_id, token_hash)
    assert ledger.delete_session(session_id, token_hash)
