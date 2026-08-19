"""The private Cloud Run app the queue delivers evidence tasks to."""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

from app.evidence_tasks import EVIDENCE_TASK_DEADLINE_SECONDS
from coscientist.model_catalog import DEFAULT_MODEL, specialist_agent_name

PROVISION = pathlib.Path(__file__).parents[2] / "scripts/provision_evidence_worker.sh"

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


def test_the_worker_publishes_the_specialists_a_stage_dials(monkeypatch):
    """A stage reaches its specialist over loopback, on this process's own port,
    so whichever process runs the stage has to be serving the cards. This one
    was not: it served a task endpoint and nothing else. A live run polled seven
    Deep Research passes to completion, read all seven reports, and then failed
    on a 404 for
    /a2a/specialists/evidence_discovery/.well-known/agent-card.json -- a card
    the main service publishes and this service had never been told to. Every
    stage after evidence dials the same way, and a task is now a whole stage."""
    monkeypatch.setenv("INTEGRATION_TEST", "TRUE")
    worker = _worker(monkeypatch, QUEUE_SETTINGS)
    name = specialist_agent_name("evidence_discovery", DEFAULT_MODEL)

    with TestClient(worker.app) as client:
        card = client.get(f"/a2a/specialists/{name}/.well-known/agent-card.json")

    assert card.status_code == 200, card.text


def test_the_queue_waits_as_long_as_the_worker_is_allowed_to_take():
    """Two ceilings on one request, and the lower one decides. Both were five
    minutes, and a task is not the twenty-second poll it is named after: folding
    a finished wave in is a model call per pass over reports of thirty thousand
    characters, then a fetch of every source they named, and one live wave spent
    six and a half minutes on it. Cloud Run cut that request off mid-read, and
    the retry landed on the lease the killed instance still held."""
    deployed = re.search(r"--timeout (\d+)", PROVISION.read_text())

    assert deployed, "the worker deploy no longer sets a Cloud Run request timeout"
    assert int(deployed.group(1)) == EVIDENCE_TASK_DEADLINE_SECONDS


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
