"""Google ADK and A2A entry point for Co-Scientist.

The live agent tree exposes purpose-specific scientific specialists. The
durable ``coscientist.orchestration.CoScientistWorkflow`` remains the authority
for stage transitions and approval; model instructions cannot forge them.
"""

from google.adk.apps import App, ResumabilityConfig

from coscientist.agents import (
    GEMINI_MODEL,
    VERTEX_LOCATION,
    build_adk_workflow,
    configure_vertex_ai_global_endpoint,
)

configure_vertex_ai_global_endpoint()
root_agent = build_adk_workflow(model=GEMINI_MODEL)

# The app name must match this package directory for ADK sessions/evaluation.
app = App(
    name="app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

__all__ = [
    "GEMINI_MODEL",
    "VERTEX_LOCATION",
    "app",
    "root_agent",
]
