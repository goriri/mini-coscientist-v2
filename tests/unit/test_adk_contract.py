from google.adk.agents import SequentialAgent

from app.agent import GEMINI_MODEL, VERTEX_LOCATION, app, root_agent


def test_adk_app_is_resumable_and_global():
    assert app.name == "app"
    assert app.resumability_config.is_resumable is True
    assert GEMINI_MODEL == "gemini-3.1-pro-preview"
    assert VERTEX_LOCATION == "global"


def test_live_tree_is_not_an_automatic_sequential_workflow():
    assert not isinstance(root_agent, SequentialAgent)
    assert len(root_agent.sub_agents) == 13
    search_agents = []
    for agent in root_agent.sub_agents:
        tool_names = [getattr(tool, "name", "") for tool in getattr(agent, "tools", [])]
        if "google_search" in tool_names:
            search_agents.append(agent.name)
    assert search_agents == ["evidence_discovery"]
