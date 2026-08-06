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
from coscientist.models import EVIDENCE_FACETS, ResearchPlan, Session
from coscientist.orchestration import _deep_research_enabled

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


def test_deep_research_runs_by_default_and_can_still_be_switched_off(monkeypatch):
    """Deep Research is on unless a deployer turns it off, and bounded either way.

    The switch used to default shut because the call is billable and cannot be
    cancelled. It bought nothing: a run without it discovered leads that no tool
    could verify, so the money it saved was the money that made the stage worth
    running. What replaced it is a ceiling the code enforces --
    ``MAX_DISCOVERY_PASSES`` interactions and ``DEFAULT_COST_LIMIT_USD`` -- which
    bounds an anonymous request whether or not anybody is watching the switch.
    """
    from coscientist.evidence import DEFAULT_COST_LIMIT_USD, MAX_DEEP_RESEARCH_PASSES
    from coscientist.orchestration import _deep_research_enabled

    monkeypatch.delenv("COSCIENTIST_DEEP_RESEARCH", raising=False)
    assert _deep_research_enabled() is True
    for value in ("on", "ON", "true", "1", "yes", "", "anything"):
        monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", value)
        assert _deep_research_enabled() is True, value
    for value in ("off", "OFF", "false", "0", "no"):
        monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", value)
        assert _deep_research_enabled() is False, value

    assert MAX_DEEP_RESEARCH_PASSES == 8
    assert DEFAULT_COST_LIMIT_USD == 24.0


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


FAN_OUT_REPORTS = [
    "Supporting evidence from a primary study https://pubmed.ncbi.nlm.nih.gov/1/",
    "Contradictory evidence was reported https://doi.org/10.1000/conflict",
    "A negative null result found no effect https://pubmed.ncbi.nlm.nih.gov/2/",
    "An independent replication is available https://pubmed.ncbi.nlm.nih.gov/3/",
    "Methods and measurement bias are described https://pubmed.ncbi.nlm.nih.gov/4/",
    "Safety toxicity and governance evidence https://www.fda.gov/example",
    "Corrections and retractions were checked https://retractionwatch.com/example",
]
"""One report per facet, in the order the fan-out asks for them."""


def test_the_first_wave_asks_every_facet_at_once_and_stops_when_all_answer():
    """Seven interactions, one per facet, and no second wave once all came back.

    The sequential shape this replaces asked one broad question and scored the
    answer: two live runs came back with the supporting literature and empty
    everywhere else, because that is what a single question about seven things
    gets answered with.
    """
    question = "Does treatment X improve outcome Y?"
    transport = FakeTransport(list(FAN_OUT_REPORTS))
    session = Session(question=question)
    manifest = IterativeEvidenceDiscovery(
        transport, EvidenceArtifactStore(bucket_name=""), poll_interval_seconds=0
    ).run(session, _plan(session))

    assert len(transport.starts) == len(EVIDENCE_FACETS)
    assert [run.facet for run in manifest.runs] == list(EVIDENCE_FACETS)
    # Each pass is told which facet is its own and told the others are covered.
    assert (
        "THIS PASS COVERS ONE FACET ONLY: contradictory"
        in (transport.starts[1]["prompt"])
    )
    assert manifest.coverage_history[-1].sufficient
    assert manifest.convergence_reason == "coverage_sufficient"
    assert manifest.estimated_cost_usd == pytest.approx(21.0)
    assert all(
        lead.verification_status == "discovered_unverified"
        for lead in manifest.source_leads
    )


def test_a_facet_whose_pass_returned_nothing_citable_is_not_scored_as_covered():
    """A pass that reports "no such literature" must not close its own facet.

    Scoring a facet from the tag its pass carries makes the fan-out grade
    itself: seven passes go out tagged, seven come back tagged, and coverage is
    1.0 before anybody has cited anything.
    """
    reports = list(FAN_OUT_REPORTS)
    reports[1] = "No contradictory literature exists for this question."
    transport = FakeTransport([*reports, "Still nothing contradictory was found."])
    session = Session(question="Test a facet with no literature")
    manifest = IterativeEvidenceDiscovery(
        transport, EvidenceArtifactStore(bucket_name=""), poll_interval_seconds=0
    ).run(session, _plan(session))

    coverage = manifest.coverage_history[-1]
    assert coverage.facet_scores["contradictory"] == 0.0
    assert coverage.facet_scores["supporting"] == 1.0
    gap = next(item for item in coverage.gaps if item.facet == "contradictory")
    # Searched and empty is a different report from never asked about.
    assert gap.description.startswith("A pass dedicated to")
    # The gap-closing pass is the eighth and last interaction the ceiling allows.
    assert len(transport.starts) == 8
    assert manifest.estimated_cost_usd == pytest.approx(24.0)
    assert manifest.convergence_reason == "maximum_passes_reached"
    assert 1 <= len(manifest.enrichment_requests) <= 6
    assert all(
        request.provider == "google_search" for request in manifest.enrichment_requests
    )


def test_the_fan_out_says_which_facets_the_budget_dropped():
    """A run that cannot afford seven passes must name the three it never ran.

    Otherwise the facets it skipped are indistinguishable from facets with no
    literature, and the panel reports an absence of evidence that is really an
    absence of searching.
    """
    transport = FakeTransport(list(FAN_OUT_REPORTS))
    session = Session(question="Test a truncated fan-out")
    manifest = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        max_passes=3,
    ).run(session, _plan(session))

    assert len(transport.starts) == 3
    assert manifest.convergence_reason.startswith("fan_out_truncated_by_budget:")
    dropped = manifest.convergence_reason.split(":", 1)[1].split(",")
    assert dropped == list(EVIDENCE_FACETS[3:])


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

        def start(self, *, pass_number: int, **_):
            self.starts += 1
            return {"id": f"stable-interaction-{pass_number}", "status": "in_progress"}

        def get(self, interaction_id: str):
            assert interaction_id.startswith("stable-interaction-")
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
    # The resumed run adopts the whole wave that was already paid for rather
    # than starting a second one: seven interactions exist, seven were started.
    assert transport.starts == len(EVIDENCE_FACETS)
    assert result.runs[0].interaction_id == "stable-interaction-1"


def test_short_worker_steps_start_then_poll_without_duplicate_interaction():
    report = (
        "Supporting contradictory negative null replication methods safety "
        "correction evidence https://pubmed.ncbi.nlm.nih.gov/1/"
    )

    class StepTransport:
        starts = 0
        polls = 0

        def start(self, *, pass_number: int, **_):
            self.starts += 1
            return {"id": f"step-interaction-{pass_number}", "status": "in_progress"}

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
    assert [run.status for run in first.runs] == ["in_progress"] * len(EVIDENCE_FACETS)

    second = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=1,
    ).run(session, _plan(session), manifest=first)
    # The second invocation polls the wave it inherited -- one poll per live
    # interaction -- instead of starting a second wave beside it.
    assert transport.starts == len(EVIDENCE_FACETS)
    assert transport.polls == len(EVIDENCE_FACETS)
    assert [run.status for run in second.runs] == ["completed"] * len(EVIDENCE_FACETS)


def test_gemini_deep_research_transport_uses_adc_when_no_api_key(monkeypatch):
    """No API key on a project means Vertex with Application Default Credentials."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-adc")
    # The Deep Research agent is served from "global" and nowhere else, so the
    # deployment's own region must not be read here: honouring it would point
    # every interaction at an endpoint that does not host the agent.
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # The suite stubs project resolution to None so that no test can reach a
    # billable interaction by accident. This one is about what happens when a
    # project does resolve, so it puts the real answer back for its own scope.
    monkeypatch.setattr(
        "coscientist.evidence.resolve_vertex_project", lambda: "test-project-adc"
    )
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("google.genai.Client", FakeClient)
    transport = GeminiDeepResearchTransport()

    assert len(calls) == 1
    assert calls[0].get("vertexai") is True
    assert calls[0].get("project") == "test-project-adc"
    assert calls[0].get("location") == "global"
    assert transport.backend == "vertex"


def test_deep_research_is_on_unless_the_deployment_turns_it_off(monkeypatch):
    """The default is on; only an explicit off switch substitutes grounded search."""
    monkeypatch.delenv("COSCIENTIST_DEEP_RESEARCH", raising=False)
    assert _deep_research_enabled() is True

    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "off")
    assert _deep_research_enabled() is False
