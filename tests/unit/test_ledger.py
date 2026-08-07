import contextlib
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


def test_the_postgres_ledger_holds_a_bounded_share_of_the_server(monkeypatch):
    """One pool per process, sized, waiting rather than refusing.

    Every operation used to open its own connection and close it again, so the
    only bound on how many this process held at once was the server's own
    ceiling -- twenty-five on the machine it runs against, three of them
    reserved. Three concurrent report exports reached it: the PDF came back a
    500 in forty milliseconds reading "remaining connection slots are reserved
    for non-replication superuser connections", at the end of an hour-long run.
    """
    import psycopg_pool

    from coscientist.ledger import (
        DEFAULT_POOL_MAX_SIZE,
        POOL_WAIT_SECONDS,
        PostgresResearchLedger,
    )

    built: list[dict] = []
    handed_out: list[int] = []

    liveness_check = psycopg_pool.ConnectionPool.check_connection

    class _Pool:
        check_connection = liveness_check

        def __init__(self, conninfo, **kwargs):
            built.append({"conninfo": conninfo, **kwargs})

        def connection(self):
            handed_out.append(1)
            return contextlib.nullcontext(object())

        def close(self):
            built.append({"closed": True})

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _Pool)
    monkeypatch.setattr(PostgresResearchLedger, "_initialize", lambda self: None)

    ledger = PostgresResearchLedger("postgresql:///coscientist")
    with ledger._connect():
        pass
    with ledger._connect():
        pass

    assert len(built) == 1, "A pool per operation is the churn this replaced."
    assert built[0]["conninfo"] == "postgresql:///coscientist"
    assert built[0]["max_size"] == DEFAULT_POOL_MAX_SIZE
    # Queue the burst, do not refuse it: a reader clicking three export links
    # should wait for the third, not be handed a 500 for it.
    assert built[0]["timeout"] == POOL_WAIT_SECONDS
    # Cloud SQL drops an idle connection without telling the holder.
    assert built[0]["check"] is liveness_check
    assert len(handed_out) == 2

    ledger.close()
    assert built[-1] == {"closed": True}
    assert ledger._pool is None


def test_the_connection_share_is_configurable_for_a_bigger_server(monkeypatch):
    """The default is a share of a small server's ceiling, not a fixed truth."""
    from coscientist.ledger import DEFAULT_POOL_MAX_SIZE, _pool_max_size

    monkeypatch.delenv("LEDGER_POOL_MAX_SIZE", raising=False)
    assert _pool_max_size() == DEFAULT_POOL_MAX_SIZE
    monkeypatch.setenv("LEDGER_POOL_MAX_SIZE", "12")
    assert _pool_max_size() == 12
    # A misconfigured value must not leave the process with an unusable pool.
    monkeypatch.setenv("LEDGER_POOL_MAX_SIZE", "0")
    assert _pool_max_size() == DEFAULT_POOL_MAX_SIZE
    monkeypatch.setenv("LEDGER_POOL_MAX_SIZE", "not a number")
    assert _pool_max_size() == DEFAULT_POOL_MAX_SIZE


def test_the_adk_engine_takes_a_share_of_the_same_server():
    """The SQLAlchemy engine draws on the ceiling the ledger draws on.

    SQLAlchemy holds five and overflows to fifteen by default, and there were two
    engines here: thirty from one process, against a budget of twenty-two shared
    with every other instance of the service.
    """
    from app.app_utils.services import _engine_options

    postgres = _engine_options("postgresql+asyncpg://user@/coscientist")
    assert postgres["pool_size"] + postgres["max_overflow"] <= 5
    assert postgres["pool_pre_ping"] is True
    # SQLite has no server and no ceiling, and its pool arguments differ.
    assert _engine_options("sqlite+aiosqlite:///runtime.db") == {}


def test_two_instances_at_peak_leave_the_server_connections_to_spare():
    """Eleven per process was called room for a second instance. Two elevens is
    the whole budget, and a tie is not room: a live run lost it, the session
    database refused the A2A server a connection mid-turn, and the generate stage
    failed behind an hour of Deep Research."""
    from app.app_utils.services import _engine_options
    from coscientist.ledger import DEFAULT_POOL_MAX_SIZE

    engine = _engine_options("postgresql+asyncpg://user@/coscientist")
    per_process = engine["pool_size"] + engine["max_overflow"] + DEFAULT_POOL_MAX_SIZE
    # Twenty-five on a db-f1-micro, three reserved for the superuser.
    assert 2 * per_process <= 22 - 4


def test_the_session_service_and_the_task_store_share_one_engine(monkeypatch):
    """Two engines against the same server is two pools where one will do."""
    import app.app_utils.services as services

    monkeypatch.setattr(
        services, "_database_url", lambda: "postgresql+asyncpg://user@/coscientist"
    )
    services._shared_engine.cache_clear()
    services.get_session_service.cache_clear()
    services.get_task_store.cache_clear()
    monkeypatch.delenv("SESSION_SERVICE_URI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
    try:
        assert services.get_session_service().db_engine is services._shared_engine()
        assert services.get_task_store().engine is services._shared_engine()
    finally:
        services._shared_engine.cache_clear()
        services.get_session_service.cache_clear()
        services.get_task_store.cache_clear()
