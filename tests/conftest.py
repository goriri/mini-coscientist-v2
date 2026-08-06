"""Shared test isolation.

Deep Research now prefers Vertex AI with Application Default Credentials, so on a
GCP machine the workflow would otherwise start a real, billable Deep Research
interaction during the test suite. Every test therefore runs with both backends
explicitly unavailable; tests that exercise the transport opt back in by
monkeypatching the resolver themselves.

The verification sweep is held off for the same reason, minus the billing: it
fetches every locator in a packet and asks Crossref and OpenAlex about it, so the
evidence tests were opening real sockets to real registries -- twenty-five
seconds for one test, and a different answer whenever a registry was slow or a
publisher blocked the fetch. Retrieval is exercised directly against fakes in
``test_retrieval.py`` instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_deep_research_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr("coscientist.evidence.resolve_vertex_project", lambda: None)


@pytest.fixture(autouse=True)
def _no_ambient_source_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_retrieval(targets, *, retriever=None):
        return {}

    monkeypatch.setattr("coscientist.evidence.assess_sources", _no_retrieval)
