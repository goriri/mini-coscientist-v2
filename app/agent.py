"""Google ADK and A2A entry point for Co-Scientist.

The live agent tree exposes purpose-specific scientific specialists. The
durable ``coscientist.orchestration.CoScientistWorkflow`` remains the authority
for stage transitions and approval; model instructions cannot forge them.

One tree is built per model on the allowlist. An ``LlmAgent`` binds its model
when it is constructed, so letting a run choose a model means choosing between
trees that already exist rather than configuring one at call time.
"""

from google.adk.apps import App, ResumabilityConfig

from coscientist.agents import (
    GEMINI_MODEL,
    VERTEX_LOCATION,
    build_adk_workflows,
    configure_vertex_ai_global_endpoint,
)

configure_vertex_ai_global_endpoint()
MODEL_TREES = build_adk_workflows()

# The default model's tree is the root. It is the one the conversational ADK
# surface and the top-level A2A card address, and it is what an unconfigured
# client gets, so moving it would break every caller that predates the choice.
root_agent = MODEL_TREES[GEMINI_MODEL]

# The app name must match this package directory for ADK sessions/evaluation.
app = App(
    name="app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

__all__ = [
    "GEMINI_MODEL",
    "MODEL_TREES",
    "VERTEX_LOCATION",
    "app",
    "root_agent",
]
