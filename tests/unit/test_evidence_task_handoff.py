"""Which of the two workers an evidence stage is handed to."""

from __future__ import annotations

import pytest
from fastapi import BackgroundTasks

from app import research_api
from coscientist.ledger import ResearchLedger
from coscientist.models import ApprovalProfile
from coscientist.narrative import stage_name
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a thin oxide coating suppress zinc dendrites?"


def _interrupted(store, session_id: str, kind: str) -> None:
    """Leave ``session_id`` held by a worker that died with its instance."""
    store.set_operation(session_id, "queued", "Waiting", kind)
    assert store.claim_operation(
        session_id, "worker-gone", detail="Polling pass 4", lease_seconds=-1
    )


@pytest.fixture()
def with_a_queue(tmp_path, monkeypatch):
    """A deployment that has somewhere to send an evidence poll."""
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: True)
    return store


@pytest.fixture()
def queued(with_a_queue):
    """A session interrupted mid-evidence, on that deployment."""
    flow = CoScientistWorkflow(
        QUESTION, ledger=with_a_queue, approval_profile=ApprovalProfile.AUTO
    )
    flow.accept(flow.preview(), automatic=True)
    assert flow.stage == "evidence"

    _interrupted(with_a_queue, flow.session.id, "evidence")
    return with_a_queue, flow.session.id


def test_a_recovered_evidence_stage_is_handed_back_to_the_queue(queued, monkeypatch):
    """Recovery used to add the background task itself, and a background task is
    the thing the queue exists to replace: it outlives the request that started
    it, so the instance serving it can be reclaimed mid-stage -- which is what
    leaves an evidence stage to be recovered in the first place. A live run went
    straight back onto that path: the sweep declared the interrupted worker
    dead, started attempt three inside the main service, and the queue standing
    ready beside it was never asked."""
    store, session_id = queued
    enqueued: list[str] = []
    monkeypatch.setattr(
        research_api,
        "enqueue_evidence_step",
        lambda session, **kwargs: enqueued.append(session),
    )
    background = BackgroundTasks()

    research_api.get_research_session(session_id, background)

    assert enqueued == [session_id]
    assert not background.tasks, "the stage was also started inside this process"
    assert store.operation(session_id)["status"] == "queued"


def test_a_recovered_stage_runs_here_where_there_is_no_queue(queued, monkeypatch):
    """The other half. A deployment without Cloud Tasks has nowhere to hand the
    work to, and leaving it queued forever is worse than a background task."""
    _store, session_id = queued
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    background = BackgroundTasks()

    research_api.get_research_session(session_id, background)

    assert [task.func for task in background.tasks] == [
        research_api._advance_in_background
    ]


def test_a_stage_that_is_not_evidence_is_recovered_here(with_a_queue, monkeypatch):
    """The queue is for one stage. The worker behind it starts and polls Deep
    Research interactions and nothing else, so handing it a scoping stage would
    lose the session down a route that has no idea what to do with it."""
    flow = CoScientistWorkflow(
        QUESTION, ledger=with_a_queue, approval_profile=ApprovalProfile.AUTO
    )
    assert flow.stage == "scope"
    _interrupted(with_a_queue, flow.session.id, "next")
    enqueued: list[str] = []
    monkeypatch.setattr(
        research_api,
        "enqueue_evidence_step",
        lambda session, **kwargs: enqueued.append(session),
    )
    background = BackgroundTasks()

    research_api.get_research_session(flow.session.id, background)

    assert enqueued == []
    assert [task.func for task in background.tasks] == [
        research_api._advance_in_background
    ]


def test_a_stage_that_is_still_held_by_a_living_worker_is_left_alone(
    queued, monkeypatch
):
    """The sweep is what decides a worker is gone. Without that, a browser
    polling every two seconds would enqueue a second worker onto a session that
    already has one."""
    store, session_id = queued
    assert store.claim_operation(
        session_id, "worker-alive", detail="Polling pass 5", lease_seconds=300
    )
    enqueued: list[str] = []
    monkeypatch.setattr(
        research_api,
        "enqueue_evidence_step",
        lambda session, **kwargs: enqueued.append(session),
    )
    background = BackgroundTasks()

    research_api.get_research_session(session_id, background)

    assert enqueued == []
    assert not background.tasks


# ---------------------------------------------------------------------------
# How much of the run one task is
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_worker(with_a_queue, monkeypatch):
    """An autonomous run on the process that serves the queue."""
    monkeypatch.setenv("EVIDENCE_TASK_STEP_MODE", "true")
    monkeypatch.setenv("INTEGRATION_TEST", "TRUE")
    flow = CoScientistWorkflow(
        QUESTION,
        ledger=with_a_queue,
        approval_profile=ApprovalProfile.AUTO,
        # Version 1 has no Deep Research evidence stage, which the offline suite
        # has no way to satisfy. What is under test is how much of a run one
        # task does, and that is the same question on either version.
        workflow_version=1,
    )
    with_a_queue.set_operation(flow.session.id, "queued", "Waiting", "evidence")
    return with_a_queue, flow.session.id


def test_a_task_does_one_stage_and_hands_the_next_one_back(task_worker, monkeypatch):
    """run_auto here would carry the whole rest of the pipeline -- generate,
    five reviewers, a tournament, evolution, a meta-review -- inside a single
    request, and Cloud Run cuts a request off at three hundred seconds. The
    retry then lands on a lease the killed instance holds for five minutes more,
    so the run advances in five-minute stalls. The poll loop is already bounded
    this way; this is the same bound on the stages after it."""
    store, session_id = task_worker
    before = research_api._load(session_id)
    stages = before.workflow_stages
    enqueued: list[str] = []
    monkeypatch.setattr(
        research_api,
        "enqueue_evidence_step",
        lambda session, **kwargs: enqueued.append(session),
    )

    research_api._advance_in_background(session_id, kind="evidence")

    after = research_api._load(session_id)
    assert stages.index(after.stage) == stages.index(before.stage) + 1
    assert enqueued == [session_id]
    assert store.operation(session_id)["status"] == "queued"
    assert (
        store.operation(session_id)["detail"] == f"Queued: {stage_name(after.stage)}."
    )


def test_the_web_service_still_runs_the_whole_pipeline(task_worker, monkeypatch):
    """The other half. Without a queue in front of it there is no next task to
    hand the next stage to, so stopping after one would strand the run."""
    store, session_id = task_worker
    monkeypatch.setenv("EVIDENCE_TASK_STEP_MODE", "false")

    research_api._advance_in_background(session_id, kind="evidence")

    assert research_api._load(session_id).done
    assert store.operation(session_id)["status"] == "completed"
