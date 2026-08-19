"""Private Cloud Run entry point for resumable Evidence polling tasks."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure each request performs only one Interactions API poll before returning.
os.environ.setdefault("EVIDENCE_TASK_STEP_MODE", "true")

from app.evidence_tasks import configured as evidence_tasks_configured
from app.research_api import _advance_in_background, _ledger, _operation


@asynccontextmanager
async def _only_where_the_next_poll_can_be_handed_on(_: FastAPI) -> AsyncIterator[None]:
    """Refuse to serve unless this worker can enqueue the poll after this one.

    One task is one poll, and the task that carries the next one is enqueued by
    the same code path that decided a poll was still needed -- which asks
    ``configured()`` first, in this process. The first deployment of this
    service was given the database, the bucket and the model settings but not
    the four EVIDENCE_ variables, so ``configured()`` was false inside it and
    every task took the branch written for a deployment with no queue at all:
    sleep, poll again, sleep, for the whole three hundred seconds Cloud Run
    allows a request. The task then failed with a 504, Cloud Tasks retried it
    into an operation lease the killed instance was still holding, and got back
    a 200 for having done nothing at all. The evidence stage moved again only
    when that lease expired five minutes later.

    Starting is the last moment this is cheap to notice, so it is noticed here:
    the container exits, the revision never takes traffic, and the deployment
    that would have degraded silently fails loudly instead.
    """
    if not evidence_tasks_configured():
        raise RuntimeError(
            "The evidence worker has no queue to hand the next poll to. It needs "
            "EVIDENCE_WORKER_URL, EVIDENCE_CLOUD_TASKS_QUEUE and "
            "GOOGLE_CLOUD_PROJECT, the same three the main service needs."
        )
    yield


app = FastAPI(
    title="Co-Scientist Evidence Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=_only_where_the_next_poll_can_be_handed_on,
)


class EvidenceTask(BaseModel):
    session_id: str


@app.post("/tasks/evidence")
def process_evidence_task(task: EvidenceTask) -> dict:
    """Run one idempotent start/poll/normalize step for a private Cloud Task."""
    _advance_in_background(task.session_id, kind="evidence")
    return _operation(task.session_id)


@app.get("/health")
def health() -> dict:
    return {"ready": bool(_ledger().healthcheck())}
