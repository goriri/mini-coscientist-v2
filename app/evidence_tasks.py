"""Cloud Tasks dispatch for short, idempotent Evidence worker steps."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta


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
    deadline = duration_pb2.Duration(seconds=300)
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
