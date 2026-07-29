from __future__ import annotations

import json

import pytest

from coscientist.evidence import (
    EvidenceArtifactStore,
    IterativeEvidenceDiscovery,
    canonicalize_url,
    normalize_report,
)
from coscientist.models import ResearchPlan, Session


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
