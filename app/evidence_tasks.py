"""Cloud Tasks dispatch for short, idempotent Evidence worker steps."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta

EVIDENCE_TASK_DEADLINE_SECONDS = 1800
"""How long one task may take. Thirty minutes is the ceiling Cloud Tasks allows
an HTTP target, and ``scripts/provision_evidence_worker.sh`` gives the worker
the same number as its Cloud Run request timeout."""


def configured() -> bool:
    return bool(
        os.environ.get("EVIDENCE_WORKER_URL")
        and os.environ.get("EVIDENCE_CLOUD_TASKS_QUEUE")
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    )


def enqueue_evidence_step(
    session_id: str,
    *,
    session_version: int,
    delay_seconds: int = 15,
) -> str:
    """Create one deterministic task; AlreadyExists is successful idempotency."""
    if not configured():
        raise RuntimeError("The Evidence Cloud Tasks worker is not configured.")

    from google.api_core.exceptions import AlreadyExists
    from google.cloud import tasks_v2
    from google.protobuf import duration_pb2, timestamp_pb2

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("EVIDENCE_CLOUD_TASKS_LOCATION", "us-east1")
    queue = os.environ["EVIDENCE_CLOUD_TASKS_QUEUE"]
    worker_url = os.environ["EVIDENCE_WORKER_URL"].rstrip("/")
    service_account = os.environ.get(
        "EVIDENCE_TASKS_SERVICE_ACCOUNT",
        os.environ.get("K_SERVICE_ACCOUNT", ""),
    )
    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)
    safe_session = re.sub(r"[^a-z0-9-]", "-", session_id.lower())[-80:]
    task_id = f"{safe_session}-v{session_version}"
    task_name = client.task_path(project, location, queue, task_id)
    request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": f"{worker_url}/tasks/evidence",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"session_id": session_id}).encode(),
    }
    if service_account:
        request["oidc_token"] = {
            "service_account_email": service_account,
            "audience": worker_url,
        }
    schedule_time = timestamp_pb2.Timestamp()
    schedule_time.FromDatetime(
        datetime.now(UTC) + timedelta(seconds=max(0, delay_seconds))
    )
    # Long enough for the longest single unit a task carries, which is not the
    # poll it is named after. A poll is twenty seconds; folding a finished wave
    # in is a model call per pass over reports that run to thirty thousand
    # characters, and then the fetch of every source they named. One live wave
    # took six and a half minutes of that, and at a five-minute deadline Cloud
    # Tasks cut the request off mid-read, retried it into the lease the killed
    # instance still held, and was answered 200 for doing nothing. Cloud Run's
    # request timeout on the worker has to match, or the same cut happens one
    # layer down.
    deadline = duration_pb2.Duration(seconds=EVIDENCE_TASK_DEADLINE_SECONDS)
    task = {
        "name": task_name,
        "http_request": request,
        "schedule_time": schedule_time,
        "dispatch_deadline": deadline,
    }
    try:
        created = client.create_task(parent=parent, task=task)
        return created.name
    except AlreadyExists:
        return task_name
