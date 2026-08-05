from __future__ import annotations

import json

import pytest

from coscientist.evidence import (
    EvidenceArtifactStore,
    GeminiDeepResearchTransport,
    IterativeEvidenceDiscovery,
    _extract_report,
    canonicalize_url,
    normalize_report,
)
from coscientist.models import ResearchPlan, Session

# Shape recorded from a completed Vertex AI Deep Research interaction
# (agent deep-research-preview-04-2026, project cellular-cider-495602-r9). Vertex
# returns the report in output_text and repeats it as a typed model_output step
# whose annotations carry the cited URLs and their titles.
VERTEX_COMPLETED_PAYLOAD = {
    "object": "interaction",
    "id": "ChAyODc1NjAyMGViMDg2MTM4EAgaATAqBG1haW4",
    "agent": "deep-research-preview-04-2026",
    "status": "completed",
    "output_text": (
        "# Report\n\nSupporting evidence was demonstrated [cite: 1].\n"
        "See also https://pubs.acs.org/doi/10.1021/example\n"
    ),
    "steps": [
        {
            "type": "model_output",
            "content": [
                {
                    "type": "text",
                    "text": "# Report\n\nSupporting evidence was demonstrated.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 5331,
                            "end_index": 5347,
                            "title": "Development of ArgTag for Scalable Synthesis\nacs.org",
                            "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE",
                        }
                    ],
                }
            ],
        }
    ],
    "usage": {
        "total_input_tokens": 642026,
        "total_output_tokens": 42929,
        "total_tokens": 1153471,
        "grounding_tool_count": [{"type": "google_search", "count": 132}],
    },
}


class _FakeGenaiClient:
    """Record how the transport asked google-genai to authenticate."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.interactions = None


class FakeTransport:
    def __init__(self, reports: list[str]):
        self.reports = reports
        self.starts: list[dict] = []

    def start(self, *, prompt: str, pass_number: int, session_id: str) -> dict:
        self.starts.append(
            {"prompt": prompt, "pass_number": pass_number, "session_id": session_id}
        )
        return {
            "id": f"interaction-{pass_number}",
            "status": "completed",
            "output_text": self.reports[pass_number - 1],
        }

    def get(self, interaction_id: str) -> dict:
        raise AssertionError("Completed interactions must not be polled.")


def _plan(session: Session) -> ResearchPlan:
    return ResearchPlan(question=session.question, intended_claim="testable claim")


@pytest.fixture
def fake_genai_client(monkeypatch: pytest.MonkeyPatch) -> list[_FakeGenaiClient]:
    from google import genai

    created: list[_FakeGenaiClient] = []

    def factory(**kwargs):
        client = _FakeGenaiClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(genai, "Client", factory)
    return created


def test_transport_uses_vertex_adc_without_any_api_key(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: list[_FakeGenaiClient]
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "coscientist.evidence.resolve_vertex_project", lambda: "project-from-adc"
    )

    transport = GeminiDeepResearchTransport()

    assert transport.backend == "vertex"
    assert transport.project == "project-from-adc"
    assert fake_genai_client[0].kwargs == {
        "vertexai": True,
        "project": "project-from-adc",
        "location": "global",
    }


def test_transport_falls_back_to_api_key_when_vertex_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: list[_FakeGenaiClient]
):
    monkeypatch.setenv("GEMINI_API_KEY", "key-123")

    transport = GeminiDeepResearchTransport()

    assert transport.backend == "api_key"
    assert fake_genai_client[0].kwargs == {"api_key": "key-123"}


def test_transport_error_names_both_missing_credentials():
    with pytest.raises(RuntimeError) as error:
        GeminiDeepResearchTransport()

    message = str(error.value)
    assert "GOOGLE_CLOUD_PROJECT" in message
    assert "GEMINI_API_KEY" in message


def test_extract_report_reads_the_vertex_interaction_shape():
    report = _extract_report(VERTEX_COMPLETED_PAYLOAD)
    assert report.startswith("# Report")

    step_only = dict(VERTEX_COMPLETED_PAYLOAD)
    step_only.pop("output_text")
    assert "Supporting evidence" in _extract_report(step_only)


def test_vertex_citation_annotations_become_titled_source_leads():
    class VertexTransport:
        def start(self, **_):
            return dict(VERTEX_COMPLETED_PAYLOAD)

        def get(self, interaction_id: str):  # pragma: no cover - never polled
            raise AssertionError("Completed interactions must not be polled.")

    session = Session(question="Vertex shaped research")
    manifest = IterativeEvidenceDiscovery(
        VertexTransport(),
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        max_passes=1,
    ).run(session, _plan(session))

    leads = {lead.canonical_url: lead for lead in manifest.source_leads}
    grounded = next(
        lead for url, lead in leads.items() if "grounding-api-redirect" in url
    )
    assert grounded.title.startswith("Development of ArgTag")
    assert manifest.runs[0].usage["total_tokens"] == 1153471
    assert manifest.estimated_cost_usd > 0
    assert manifest.stored_interaction_notice is True


def test_deep_research_never_starts_unless_someone_asked_for_it(monkeypatch):
    """The switch protecting a billable, uncancellable call must default shut.

    ADC is ambient on developer machines, CI, and Cloud Run, so an opt-out
    default means importing and running the workflow spends money before
    anyone has read a line of output. That happened once; this pins it.
    """
    from coscientist.orchestration import _deep_research_enabled

    monkeypatch.delenv("COSCIENTIST_DEEP_RESEARCH", raising=False)
    assert _deep_research_enabled() is False
    for value in ("on", "ON", "true", "1", "yes"):
        monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", value)
        assert _deep_research_enabled() is True, value
    # Anything that is not an explicit yes is a no, including typos: a
    # misspelled opt-in must fail closed rather than bill.
    for value in ("off", "false", "0", "no", "onn", ""):
        monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", value)
        assert _deep_research_enabled() is False, value


def test_discovery_is_attempted_whenever_vertex_adc_is_reachable(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: list[_FakeGenaiClient]
):
    """No GEMINI_API_KEY must no longer mean "deep_research_unavailable"."""
    from coscientist import orchestration
    from coscientist.agents import DeterministicProvider
    from coscientist.models import DiscoveryManifest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Deep Research is opt-in because it is billable and uncancellable, so a
    # test about building its transport has to say so out loud. The recording
    # double below is what keeps this from costing anything.
    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "on")
    monkeypatch.setattr(
        "coscientist.evidence.resolve_vertex_project", lambda: "adc-project"
    )
    built: list[GeminiDeepResearchTransport] = []

    class RecordingDiscovery:
        def __init__(self, transport, artifact_store, **kwargs):
            self.transport = transport
            built.append(transport)

        def run(self, session, plan, *, manifest=None, **kwargs):
            manifest = manifest or DiscoveryManifest(question=session.question)
            manifest.convergence_reason = "coverage_sufficient"
            return manifest

    monkeypatch.setattr(orchestration, "IterativeEvidenceDiscovery", RecordingDiscovery)
    flow = orchestration.CoScientistWorkflow(
        "Can a coating improve cycle life?", DeterministicProvider()
    )
    flow.accept(flow.preview(), actor="test_researcher")
    assert flow.stage == "evidence"
    flow.preview()

    assert [transport.backend for transport in built] == ["vertex"]
    assert built[0].project == "adc-project"


def test_sufficient_first_pass_does_not_repeat():
    question = "Does treatment X improve outcome Y?"
    paragraphs = [
        f"{question} Supporting evidence from a primary study https://pubmed.ncbi.nlm.nih.gov/1/",
        "Contradictory evidence was reported https://doi.org/10.1000/conflict",
        "A negative null result found no effect https://pubmed.ncbi.nlm.nih.gov/2/",
        "An independent replication is available https://pubmed.ncbi.nlm.nih.gov/3/",
        "Methods and measurement bias are described https://pubmed.ncbi.nlm.nih.gov/4/",
        "Safety toxicity and governance evidence https://www.fda.gov/example",
        "Corrections and retractions were checked https://retractionwatch.com/example",
    ]
    transport = FakeTransport(["\n\n".join(paragraphs)])
    session = Session(question=question)
    manifest = IterativeEvidenceDiscovery(
        transport, EvidenceArtifactStore(bucket_name=""), poll_interval_seconds=0
    ).run(session, _plan(session))

    assert len(transport.starts) == 1
    assert manifest.coverage_history[-1].sufficient
    assert manifest.convergence_reason == "coverage_sufficient"
    assert all(
        lead.verification_status == "discovered_unverified"
        for lead in manifest.source_leads
    )


def test_low_value_second_pass_stops_and_queues_targeted_enrichment():
    report = (
        "Supporting evidence exists, but the landscape is incomplete "
        "https://pubmed.ncbi.nlm.nih.gov/1/"
    )
    transport = FakeTransport([report, report])
    session = Session(question="Test an incomplete evidence landscape")
    manifest = IterativeEvidenceDiscovery(
        transport, EvidenceArtifactStore(bucket_name=""), poll_interval_seconds=0
    ).run(session, _plan(session))

    assert len(transport.starts) == 2
    assert manifest.convergence_reason == "coverage_improvement_below_threshold"
    assert 1 <= len(manifest.enrichment_requests) <= 6
    assert all(
        request.provider == "google_search" for request in manifest.enrichment_requests
    )


def test_normalization_rejects_citations_not_in_originating_report():
    report_url = "https://pubmed.ncbi.nlm.nih.gov/1/"
    invented_url = "https://example.com/fabricated"
    normalized = {
        "question": "Q",
        "research_directions": ["Q"],
        "statements": [
            {
                "text": "fabricated citation",
                "facet": "supporting",
                "source_urls": [invented_url],
                "originating_pass": 1,
            },
            {
                "text": "grounded citation",
                "facet": "supporting",
                "source_urls": [report_url],
                "originating_pass": 1,
            },
        ],
    }
    narrative = normalize_report(
        question="Q",
        report=f"Grounded source {report_url}",
        pass_number=1,
        normalizer=lambda _: json.dumps(normalized),
    )

    assert len(narrative.statements) == 1
    assert narrative.statements[0].source_urls == [report_url]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/report",
        "http://10.2.3.4/report",
        "http://[::1]/report",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_source_url_controls_reject_non_public_networks(url: str):
    with pytest.raises(ValueError):
        canonicalize_url(url)


def test_canonicalization_removes_tracking_without_removing_identity():
    assert (
        canonicalize_url("HTTPS://Example.org/paper?utm_source=test&id=7#abstract")
        == "https://example.org/paper?id=7"
    )


def test_retry_resumes_the_same_stored_interaction():
    report = (
        "Supporting evidence https://pubmed.ncbi.nlm.nih.gov/1/ and contradictory "
        "negative null replication methods safety correction evidence "
        "https://www.fda.gov/example"
    )

    class InterruptedTransport:
        starts = 0

        def start(self, **_):
            self.starts += 1
            return {"id": "stable-interaction", "status": "in_progress"}

        def get(self, interaction_id: str):
            assert interaction_id == "stable-interaction"
            return {"id": interaction_id, "status": "completed", "output_text": report}

    session = Session(question="Resumable research")
    transport = InterruptedTransport()
    captured = None

    def interrupt(manifest):
        nonlocal captured
        captured = manifest.model_copy(deep=True)
        raise RuntimeError("worker terminated")

    controller = IterativeEvidenceDiscovery(
        transport, EvidenceArtifactStore(bucket_name=""), poll_interval_seconds=0
    )
    with pytest.raises(RuntimeError, match="worker terminated"):
        controller.run(session, _plan(session), manifest_callback=interrupt)

    assert captured is not None
    result = controller.run(session, _plan(session), manifest=captured)
    assert transport.starts == 1
    assert result.runs[0].interaction_id == "stable-interaction"


def test_short_worker_steps_start_then_poll_without_duplicate_interaction():
    report = (
        "Supporting contradictory negative null replication methods safety "
        "correction evidence https://pubmed.ncbi.nlm.nih.gov/1/"
    )

    class StepTransport:
        starts = 0
        polls = 0

        def start(self, **_):
            self.starts += 1
            return {"id": "step-interaction", "status": "in_progress"}

        def get(self, interaction_id: str):
            self.polls += 1
            return {
                "id": interaction_id,
                "status": "completed",
                "output_text": report,
            }

    transport = StepTransport()
    session = Session(question="Short task research")
    first = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=0,
    ).run(session, _plan(session))
    assert first.runs[0].status == "in_progress"

    second = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=1,
    ).run(session, _plan(session), manifest=first)
    assert transport.starts == 1
    assert transport.polls == 1
    assert second.runs[0].status == "completed"
