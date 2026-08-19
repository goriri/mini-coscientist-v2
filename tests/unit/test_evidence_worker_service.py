"""The private Cloud Run app the queue delivers evidence tasks to."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

QUEUE_SETTINGS = {
    "EVIDENCE_WORKER_URL": "https://coscientist-evidence-worker.example.run.app",
    "EVIDENCE_CLOUD_TASKS_QUEUE": "coscientist-evidence",
    "EVIDENCE_CLOUD_TASKS_LOCATION": "us-east1",
    "EVIDENCE_TASKS_SERVICE_ACCOUNT": "worker@example.iam.gserviceaccount.com",
    "GOOGLE_CLOUD_PROJECT": "example-project",
}


def _worker(monkeypatch, settings: dict[str, str]):
    """The worker app, with ``settings`` in the environment and nothing else."""
    monkeypatch.setenv("EVIDENCE_TASK_STEP_MODE", "true")
    for name in QUEUE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    return importlib.import_module("app.evidence_worker")


def test_a_worker_with_a_queue_to_answer_to_starts(monkeypatch):
    """The other half of the refusal below: it has to be about the queue and
    not about this app being unable to start at all."""
    worker = _worker(monkeypatch, QUEUE_SETTINGS)

    with TestClient(worker.app):
        pass


def test_a_worker_with_nowhere_to_send_the_next_poll_refuses_to_start(monkeypatch):
    """One task is one poll, and the poll after it is enqueued by this process,
    which asks configured() first. The first revision of this service carried
    the database, the bucket and the model settings but none of the four
    EVIDENCE_ variables, so configured() was false inside it and every task took
    the branch written for a deployment with no queue at all: poll, sleep, poll,
    until Cloud Run cut the request off at three hundred seconds with a 504.
    Cloud Tasks retried it into an operation lease the killed instance was still
    holding and was answered 200 for doing nothing, and the evidence stage moved
    again only when that lease expired. Starting is where this costs nothing to
    catch."""
    worker = _worker(
        monkeypatch,
        {
            name: value
            for name, value in QUEUE_SETTINGS.items()
            if name != "EVIDENCE_CLOUD_TASKS_QUEUE"
        },
    )

    with pytest.raises(RuntimeError, match="no queue to hand the next poll to"):
        with TestClient(worker.app):
            pass
