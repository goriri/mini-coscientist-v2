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


# Every pool in this process draws on one server's connection ceiling, and that
# ceiling is twenty-five on the machine this runs against, three of them reserved
# for the superuser. SQLAlchemy defaults to five held plus ten overflow per
# engine, and there were two engines here beside the research ledger's own pool
# -- thirty-five from one process against a twenty-two-connection budget. The
# exports stage of a live run collected the resulting "remaining connection slots
# are reserved" 500.
#
# Rationing them left eleven per process, which the note here called room for a
# second instance. Two instances of eleven is twenty-two, and twenty-two of
# twenty-two is not room, it is a tie -- pool_recycle and pool_pre_ping both
# replace a connection, and a replacement overlaps its predecessor for as long as
# the old one takes to close. The tie lost: with exactly two instances up, the
# session database refused a connection to the A2A server mid-turn, which closed
# the specialist's event stream having sent nothing and failed the generate stage
# of a run with an hour and twenty-four dollars of Deep Research behind it.
#
# So the two engines are now one, shared by the session service and the task
# store (see ``get_task_store``), and it holds two rather than four. Peak is five
# here plus three in the ledger: eight per process, sixteen at the two instances
# that were up, six spare. Steady state is three, because an overflow connection
# is closed when it is handed back.
#
# What this still does not do is bound instances times pool. Nothing in the
# process knows how many instances are up, maxScale is ten, and three at peak
# would be twenty-four. These are shares of the server's budget: raise them with
# its max_connections, not before it.
_ENGINE_POOL = {
    "pool_size": 2,
    "max_overflow": 3,
    # Cloud SQL closes an idle connection without telling the holder.
    "pool_pre_ping": True,
    "pool_recycle": 1800,
}


def _engine_options(url: str) -> dict:
    """Pool settings for a server that has them, and none for SQLite, which has not."""
    return {} if url.startswith("sqlite") else dict(_ENGINE_POOL)


@functools.cache
def _shared_engine():
    """One engine for every pool in this process that talks to the same server.

    SQLite gets none: it has no connection ceiling to ration, and ADK attaches a
    foreign-key pragma to an engine it builds itself and not to one handed to it,
    so the local runtime keeps the engine ADK makes.
    """
    url = _database_url()
    if url.startswith("sqlite"):
        return None
    return create_async_engine(url, **_engine_options(url))


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
    if engine := _shared_engine():
        return DatabaseSessionService(db_engine=engine)
    url = _database_url()
    return DatabaseSessionService(db_url=url, **_engine_options(url))


@functools.cache
def get_task_store():
    """Durable A2A task store shared by the generated JSON-RPC surface."""
    url = _database_url()
    engine = _shared_engine() or create_async_engine(url, **_engine_options(url))
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
