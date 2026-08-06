import httpx
import pytest

from app.app_utils.a2a import default_app_url
from coscientist.agents import A2AProvider


def test_a2a_provider_removes_exact_and_trailing_prompt_echoes():
    prompt = "Research question: Does a coating improve cycle life?"
    response = '{"question":"Does a coating improve cycle life?"}'

    assert A2AProvider._without_prompt_echo([response, prompt], prompt) == response
    assert (
        A2AProvider._without_prompt_echo([f"{response}\n{prompt}"], prompt) == response
    )


def test_the_advertised_card_url_is_one_a_client_can_actually_dial(monkeypatch):
    """A card that advertises a bind address breaks every caller, us included."""
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.setenv("PORT", "8080")
    url = default_app_url()
    assert "0.0.0.0" not in url
    # httpx builds the request the A2A client would send; an unroutable host
    # fails here rather than at connect time in production.
    assert httpx.URL(url).host == "127.0.0.1"
    assert httpx.URL(url).port == 8080


def test_an_explicit_app_url_wins_and_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv("APP_URL", "https://coscientist.example.run.app/")
    # rpc paths are appended verbatim, so a trailing slash would double up.
    assert default_app_url() == "https://coscientist.example.run.app"


@pytest.mark.parametrize("port", ["8080", "9000"])
def test_the_card_follows_the_port_the_container_was_given(monkeypatch, port):
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.setenv("PORT", port)
    assert default_app_url().endswith(f":{port}")


def test_a_role_no_agent_serves_is_refused_before_the_card_is_fetched():
    """It used to surface as a 404 on a card URL, which reads as a broken deploy.

    The actor-critic loop addressed ``<role>_critic``; only the seventeen
    specialists are published, so every session against the deployment died at
    the first stage.
    """
    provider = A2AProvider(base_url="https://example.invalid")
    with pytest.raises(ValueError, match="is not a published specialist"):
        provider.complete(role="goal_manager_critic", prompt="anything")
