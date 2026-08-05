"""Shared test isolation.

Deep Research now prefers Vertex AI with Application Default Credentials, so on a
GCP machine the workflow would otherwise start a real, billable Deep Research
interaction during the test suite. Every test therefore runs with both backends
explicitly unavailable; tests that exercise the transport opt back in by
monkeypatching the resolver themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_deep_research_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr("coscientist.evidence.resolve_vertex_project", lambda: None)
