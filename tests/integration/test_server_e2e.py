# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    TextPart,
)
from requests.exceptions import RequestException

from coscientist.agents import A2AProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_PORT = int(os.getenv("COSCIENTIST_TEST_PORT", "8765"))
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
RUN_SSE_URL = BASE_URL + "/run_sse"
A2A_RPC_URL = BASE_URL + "/a2a/app/"
AGENT_CARD_URL = A2A_RPC_URL + ".well-known/agent-card.json"
FEEDBACK_URL = BASE_URL + "/feedback"

HEADERS = {"Content-Type": "application/json"}
LIVE_TEST = pytest.mark.skipif(
    os.getenv("COSCIENTIST_LIVE_TESTS", "false").lower() != "true",
    reason="Set COSCIENTIST_LIVE_TESTS=true to call Vertex AI.",
)


def log_output(pipe: Any, log_func: Any) -> None:
    """Log the output from the given pipe."""
    for line in iter(pipe.readline, ""):
        log_func(line.strip())


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.fast_api_app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
    ]
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "TRUE"
    env["APP_URL"] = BASE_URL
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Start threads to log stdout and stderr in real-time
    threading.Thread(
        target=log_output, args=(process.stdout, logger.info), daemon=True
    ).start()
    threading.Thread(
        target=log_output, args=(process.stderr, logger.error), daemon=True
    ).start()

    return process


def wait_for_server(timeout: int = 90, interval: int = 1) -> bool:
    """Wait for the server to be ready (agent card requires the lifespan to run)."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(AGENT_CARD_URL, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:
    """Pytest fixture to start and stop the server for testing."""
    logger.info("Starting server process")
    server_process = start_server()
    if not wait_for_server():
        pytest.fail("Server failed to start")
    logger.info("Server process started")

    def stop_server() -> None:
        logger.info("Stopping server process")
        server_process.terminate()
        server_process.wait()
        logger.info("Server process stopped")

    request.addfinalizer(stop_server)
    yield server_process


@LIVE_TEST
def test_adk_run_sse(server_fixture: subprocess.Popen[str]) -> None:
    """Test the native ADK route (/run_sse) end to end."""
    logger.info("Starting ADK /run_sse test")
    user_id = f"user_{uuid.uuid4()}"
    session_data = {"state": {"preferred_language": "English", "visit_count": 1}}

    session_response = requests.post(
        f"{BASE_URL}/apps/app/users/{user_id}/sessions",
        headers=HEADERS,
        json=session_data,
        timeout=60,
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["id"]

    data = {
        "app_name": "app",
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {"role": "user", "parts": [{"text": "Hi!"}]},
        "streaming": True,
    }
    response = requests.post(
        RUN_SSE_URL, headers=HEADERS, json=data, stream=True, timeout=60
    )
    assert response.status_code == 200

    events = []
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                events.append(json.loads(line_str[6:]))

    assert events, "No events received from stream"
    has_text_content = any(
        (content := event.get("content"))
        and content.get("parts")
        and any(part.get("text") for part in content["parts"])
        for event in events
    )
    assert has_text_content, "Expected at least one event with text content"


@LIVE_TEST
def test_a2a_chat_stream(server_fixture: subprocess.Popen[str]) -> None:
    """Test the A2A route using the JSON-RPC streaming protocol."""
    logger.info("Starting A2A chat stream test")

    message = Message(
        message_id=f"msg-user-{uuid.uuid4()}",
        role=Role.user,
        parts=[Part(root=TextPart(text="Hi!"))],
    )
    request = SendStreamingMessageRequest(
        id="test-req-001",
        params=MessageSendParams(message=message),
    )
    response = requests.post(
        A2A_RPC_URL,
        headers=HEADERS,
        json=request.model_dump(mode="json", exclude_none=True),
        stream=True,
        timeout=60,
    )
    assert response.status_code == 200

    responses: list[SendStreamingMessageResponse] = []
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                responses.append(
                    SendStreamingMessageResponse.model_validate(
                        json.loads(line_str[6:])
                    )
                )

    assert responses, "No responses received from stream"

    final_responses = [
        r.root
        for r in responses
        if hasattr(r.root, "result")
        and hasattr(r.root.result, "final")
        and r.root.result.final is True
    ]
    assert final_responses, "No final response received"
    assert final_responses[-1].result.status.state == "completed"


def test_agent_card(server_fixture: subprocess.Popen[str]) -> None:
    """Test that the A2A agent card is served at the well-known URI."""
    response = requests.get(AGENT_CARD_URL, timeout=10)
    assert response.status_code == 200, f"A2A endpoint returned {response.status_code}"

    served_agent_card = response.json()
    for field in ("name", "description", "skills", "capabilities", "url", "version"):
        assert field in served_agent_card, f"Missing field in agent card: {field}"


def test_public_research_workspace(server_fixture: subprocess.Popen[str]) -> None:
    """The custom UI loads while ADK development controls remain disabled."""
    response = requests.get(BASE_URL, timeout=10)
    assert response.status_code == 200
    assert "Turn uncertainty into" in response.text
    assert "/assets/styles.css" in response.text

    stylesheet = requests.get(f"{BASE_URL}/assets/styles.css", timeout=10)
    script = requests.get(f"{BASE_URL}/assets/app.js", timeout=10)
    assert stylesheet.status_code == 200
    assert script.status_code == 200
    assert "streamResearch" in script.text
    assert "height: 100dvh" in stylesheet.text
    assert "nearConversationBottom" in script.text
    assert "renderReportCompletion" in script.text
    assert "editable Word document for Google Docs" in script.text
    assert "Guided HITL" in response.text

    assert requests.get(f"{BASE_URL}/dev-ui", timeout=10).status_code == 404
    assert (
        requests.get(f"{BASE_URL}/dev/apps/app/builder", timeout=10).status_code == 404
    )


def test_guided_hitl_workflow_end_to_end(
    server_fixture: subprocess.Popen[str],
) -> None:
    """A researcher can revise, approve milestones, and obtain a dossier."""
    create_started = time.time()
    created = requests.post(
        f"{BASE_URL}/api/research/sessions",
        headers=HEADERS,
        json={
            "question": (
                "Does a protective coating improve rechargeable battery cycle "
                "life compared with an uncoated control?"
            ),
            "approval_profile": "milestone",
        },
        timeout=30,
    )
    assert created.status_code == 201
    assert time.time() - create_started < 2
    workflow = created.json()
    assert workflow["deletion_token"]
    assert workflow["stage"] == "scope"
    assert workflow["requires_human_approval"] is True
    assert workflow["operation"]["status"] in {"queued", "running", "completed"}
    started = time.time()
    while (
        workflow["operation"]["status"] in {"queued", "running"}
        or workflow["pending_draft"] is None
    ):
        assert time.time() - started < 20
        time.sleep(0.05)
        workflow = requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
        ).json()
    assert workflow["pending_draft"]["version"] == 1

    revision_started = time.time()
    revision = requests.post(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/decisions",
        headers=HEADERS,
        json={
            "action": "revise",
            "feedback": "Add a prespecified cycle-life endpoint and control.",
        },
        timeout=10,
    )
    assert revision.status_code == 200
    assert time.time() - revision_started < 2
    workflow = revision.json()
    assert workflow["pending_draft"] is None
    assert workflow["operation"]["status"] == "queued"
    started = time.time()
    while (
        workflow["operation"]["status"] in {"queued", "running"}
        or workflow["pending_draft"] is None
    ):
        assert time.time() - started < 20
        time.sleep(0.05)
        workflow = requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
        ).json()
    assert workflow["pending_draft"]["version"] == 2

    edited_content = (
        workflow["pending_draft"]["content"]
        + "\n\nResearcher edit: use a prespecified cycle-life endpoint."
    )
    edited = requests.post(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/decisions",
        headers=HEADERS,
        json={
            "action": "edit",
            "content": edited_content,
        },
        timeout=30,
    )
    assert edited.status_code == 200
    workflow = edited.json()
    assert workflow["pending_draft"]["version"] == 3
    assert workflow["pending_draft"]["content"] == edited_content
    assert workflow["decisions"][-1]["action"] == "revise"

    human_gate_count = 0
    while workflow["status"] == "active":
        assert workflow["pending_draft"] is not None
        accepted = requests.post(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}/decisions",
            headers=HEADERS,
            json={"action": "accept"},
            timeout=10,
        )
        assert accepted.status_code == 200, accepted.text
        workflow = accepted.json()
        human_gate_count += 1
        assert human_gate_count <= 5
        started = time.time()
        while workflow["status"] == "active" and (
            workflow["operation"]["status"] in {"queued", "running"}
            or workflow["pending_draft"] is None
        ):
            assert time.time() - started < 20
            time.sleep(0.05)
            workflow = requests.get(
                f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
            ).json()
        if workflow["status"] == "evidence_required":
            fallback = requests.post(
                f"{BASE_URL}/api/research/sessions/{workflow['id']}/decisions",
                headers=HEADERS,
                json={"action": "exploratory_evidence"},
                timeout=10,
            )
            assert fallback.status_code == 200, fallback.text
            workflow = fallback.json()
            started = time.time()
            while workflow["status"] == "active" and (
                workflow["operation"]["status"] in {"queued", "running"}
                or workflow["pending_draft"] is None
            ):
                assert time.time() - started < 20
                time.sleep(0.05)
                workflow = requests.get(
                    f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
                ).json()

    assert workflow["status"] == "ready_for_report"
    assert workflow["stage"] == "report"
    assert workflow["report"]
    assert [item["format"] for item in workflow["report_exports"]] == [
        "docx",
        "pdf",
        "md",
    ]
    assert human_gate_count in {4, 5}
    assert any(decision["automatic"] for decision in workflow["decisions"])
    assert all(
        decision["actor"] == "web_researcher"
        for decision in workflow["decisions"]
        if not decision["automatic"]
    )

    resumed = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "ready_for_report"

    markdown = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/report/md",
        timeout=30,
    )
    assert markdown.status_code == 200
    assert markdown.content.startswith(b"# Co-Scientist Research Dossier")
    assert markdown.headers["content-disposition"].endswith('.md"')

    pdf = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/report/pdf",
        timeout=120,
    )
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")
    assert pdf.headers["content-type"] == "application/pdf"

    docx = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/report/docx",
        timeout=120,
    )
    assert docx.status_code == 200
    assert docx.content.startswith(b"PK")
    assert "openxmlformats" in docx.headers["content-type"]


def test_stage_previews_are_read_only(
    server_fixture: subprocess.Popen[str],
) -> None:
    """Only saved stage bundles are exposed, and browsing does not mutate state."""
    created = requests.post(
        f"{BASE_URL}/api/research/sessions",
        headers=HEADERS,
        json={
            "question": "Which controls distinguish adsorption from catalytic turnover?",
            "approval_profile": "stage",
        },
        timeout=10,
    )
    assert created.status_code == 201
    workflow = created.json()
    deletion_token = workflow["deletion_token"]
    started = time.time()
    while workflow["pending_draft"] is None:
        assert time.time() - started < 20
        time.sleep(0.05)
        workflow = requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
        ).json()

    scope_metadata = next(
        item for item in workflow["stage_previews"] if item["stage"] == "scope"
    )
    assert scope_metadata["available"] is True
    assert scope_metadata["is_current"] is True
    assert all(
        not item["available"]
        for item in workflow["stage_previews"]
        if item["stage"] != "scope"
    )

    before = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
    ).json()
    preview = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}/stages/scope",
        timeout=10,
    )
    assert preview.status_code == 200
    assert preview.json()["read_only"] is True
    assert preview.json()["artifact_id"] == before["pending_draft"]["id"]
    assert preview.json()["presentation"]["kind"] == "scope"

    after = requests.get(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
    ).json()
    assert after["decisions"] == before["decisions"]
    assert after["pending_draft"]["id"] == before["pending_draft"]["id"]
    assert after["pending_draft"]["version"] == before["pending_draft"]["version"]

    assert (
        requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}/stages/generate",
            timeout=10,
        ).status_code
        == 404
    )
    assert (
        requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}/stages/not-a-stage",
            timeout=10,
        ).status_code
        == 404
    )
    assert (
        requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}/report/pdf",
            timeout=10,
        ).status_code
        == 409
    )

    denied = requests.delete(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}",
        headers={"X-Session-Delete-Token": "wrong"},
        timeout=10,
    )
    assert denied.status_code == 403
    deleted = requests.delete(
        f"{BASE_URL}/api/research/sessions/{workflow['id']}",
        headers={"X-Session-Delete-Token": deletion_token},
        timeout=10,
    )
    assert deleted.status_code == 204
    assert (
        requests.get(
            f"{BASE_URL}/api/research/sessions/{workflow['id']}", timeout=10
        ).status_code
        == 404
    )


def test_storage_health(server_fixture: subprocess.Popen[str]) -> None:
    response = requests.get(f"{BASE_URL}/api/research/health/storage", timeout=10)
    assert response.status_code == 200
    assert response.json() == {"ready": True, "backend": "sqlite"}


def test_specialist_agent_cards(server_fixture: subprocess.Popen[str]) -> None:
    """Every purpose-built specialist exposes a narrow generated A2A card."""
    specialists = (
        "goal_manager",
        "evidence_discovery",
        "source_verification",
        "generation",
        "generation_evidence_first",
        "generation_mechanism_first",
        "generation_analogy_transfer",
        "generation_competing_explanation",
        "reflection",
        "novelty_review",
        "methods_statistics",
        "ethics_safety_governance",
        "impact_review",
        "ranking",
        "evolution",
        "proximity",
        "meta_reviewer",
    )
    for specialist in specialists:
        response = requests.get(
            f"{BASE_URL}/a2a/specialists/{specialist}/.well-known/agent-card.json",
            timeout=10,
        )
        assert response.status_code == 200
        assert response.json()["name"] == specialist


@LIVE_TEST
def test_a2a_provider_invokes_narrow_specialist(
    server_fixture: subprocess.Popen[str],
) -> None:
    provider = A2AProvider(BASE_URL)
    response = provider.complete(
        role="goal_manager",
        prompt=(
            "Research question: Does a coating improve battery cycle life?\n"
            "Return only a concise, falsifiable objective and constraints."
        ),
    )
    assert response.strip()
    assert "coating" in response.lower() or "cycle" in response.lower()


def test_collect_feedback(server_fixture: subprocess.Popen[str]) -> None:
    """Test the feedback collection endpoint (/feedback)."""
    feedback_data = {
        "score": 4,
        "user_id": "test-user-456",
        "session_id": "test-session-456",
        "text": "Great response!",
    }
    response = requests.post(
        FEEDBACK_URL, json=feedback_data, headers=HEADERS, timeout=10
    )
    assert response.status_code == 200
