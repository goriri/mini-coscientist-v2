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

"""Process-wide ADK session/artifact services shared by every serving surface.

Registered under ``shared://`` so the ADK web routes, the A2A path, and the
reasoning_engine adapter share one instance: a session created on any surface
is visible to the others.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from urllib.parse import quote_plus

from a2a.server.tasks import DatabaseTaskStore
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.sessions import DatabaseSessionService
from sqlalchemy.ext.asyncio import create_async_engine

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_STATE_DIR = Path(
    os.environ.get("COSCIENTIST_STATE_DIR", os.path.join(_AGENT_DIR, ".coscientist"))
)


@functools.cache
def _database_url() -> str:
    if database_url := os.environ.get("SESSION_DATABASE_URL"):
        return database_url
    if connection_name := os.environ.get("CLOUD_SQL_CONNECTION_NAME"):
        user = quote_plus(os.environ.get("DATABASE_USER", "coscientist_app"))
        password = quote_plus(os.environ["DATABASE_PASSWORD"])
        database = quote_plus(
            os.environ.get(
                "SESSION_DATABASE_NAME",
                os.environ.get("DATABASE_NAME", "coscientist"),
            )
        )
        socket = quote_plus(f"/cloudsql/{connection_name}")
        return f"postgresql+asyncpg://{user}:{password}@/{database}?host={socket}"
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{_STATE_DIR / 'adk_runtime.db'}"


@functools.cache
def get_session_service():
    """Process-wide session service shared across every serving surface."""
    if uri := os.environ.get("SESSION_SERVICE_URI"):
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=uri
        )
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

        return VertexAiSessionService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            # Runtime-injected agent-engine region, not GOOGLE_CLOUD_LOCATION
            # (which agent.py pins to "global").
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )
    return DatabaseSessionService(db_url=_database_url())


@functools.cache
def get_task_store():
    """Durable A2A task store shared by the generated JSON-RPC surface."""
    engine = create_async_engine(_database_url())
    return DatabaseTaskStore(engine=engine, create_table=True)


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
