"""Private Cloud Run entry point for resumable Evidence polling tasks."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure each request performs only one Interactions API poll before returning.
os.environ.setdefault("EVIDENCE_TASK_STEP_MODE", "true")

from app.research_api import _advance_in_background, _ledger, _operation

app = FastAPI(
    title="Co-Scientist Evidence Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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
