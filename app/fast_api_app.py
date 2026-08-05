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

import contextlib
import logging
import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.typing import Feedback
from app.research_api import router as research_router

load_dotenv()
logger = logging.getLogger(__name__)
cloud_logger = None
if os.getenv("COSCIENTIST_CLOUD_LOGGING", "false").lower() == "true":
    from google.cloud import logging as google_cloud_logging

    cloud_logger = google_cloud_logging.Client().logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import MODEL_TREES, root_agent
    from app.agent import app as adk_app

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    task_store = services.get_task_store()
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=task_store,
        rpc_path=f"/a2a/{adk_app.name}",
    )
    # Each purpose-specific child also publishes a narrow Agent Card/A2A skill.
    # The route attachment itself remains the generated agents-cli transport.
    #
    # Every model on the allowlist gets its own set of cards, because a run
    # selects its model by choosing which card to call. The agent name already
    # carries the model for every tree but the default one, so the paths cannot
    # collide and the default model's paths are exactly what they were before
    # the choice existed.
    specialist_runners = {}
    for tree in MODEL_TREES.values():
        for specialist in tree.sub_agents:
            specialist_runner = Runner(
                agent=specialist,
                app_name=f"specialist_{specialist.name}",
                session_service=services.get_session_service(),
                artifact_service=services.get_artifact_service(),
                auto_create_session=True,
            )
            specialist_runners[specialist.name] = specialist_runner
            await attach_a2a_routes(
                app,
                agent=specialist,
                runner=specialist_runner,
                task_store=task_store,
                rpc_path=f"/a2a/specialists/{specialist.name}",
            )
    app.state.specialist_runners = specialist_runners
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    # The generated ADK development console includes builder and evaluation
    # routes that must not be exposed by the public Cloud Run service.
    web=False,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=os.getenv("COSCIENTIST_OTEL_TO_CLOUD", "false").lower() == "true",
    lifespan=lifespan,
)
app.title = "coscientist"
app.description = "API for interacting with the Agent coscientist"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="web-assets")
app.include_router(research_router)


@app.get("/", include_in_schema=False)
def research_workspace() -> FileResponse:
    """Serve the public Co-Scientist research workspace."""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/robots.txt", include_in_schema=False)
def robots() -> Response:
    """Keep the development research service out of search indexes."""
    return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    payload = feedback.model_dump()
    if cloud_logger is not None:
        cloud_logger.log_struct(payload, severity="INFO")
    else:
        logger.info(
            "feedback score=%s session_id=%s", feedback.score, feedback.session_id
        )
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
