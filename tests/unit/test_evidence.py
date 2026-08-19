from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from coscientist import orchestration
from coscientist.collaboration import TaskResult
from coscientist.evidence import (
    EvidenceArtifactStore,
    GeminiDeepResearchTransport,
    IterativeEvidenceDiscovery,
    build_research_prompt,
    canonicalize_url,
    extract_report,
    merge_leads,
    normalize_report,
)
from coscientist.models import (
    EVIDENCE_FACETS,
    Artifact,
    DeepResearchRun,
    DiscoveryManifest,
    EvidenceClaim,
    EvidencePacket,
    ResearchPlan,
    Session,
    SourceLead,
    SourceRecord,
    TaskRecord,
    TaskState,
)
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
    report = extract_report(VERTEX_COMPLETED_PAYLOAD)
    assert report.startswith("# Report")

    step_only = dict(VERTEX_COMPLETED_PAYLOAD)
    step_only.pop("output_text")
    assert "Supporting evidence" in extract_report(step_only)


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
    from coscientist.agents import DeterministicProvider

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


class SlowTransport:
    """A wave where every pass but one comes back on the first poll.

    The shape of the live failure: seven interactions start together, six finish,
    and the seventh is still running when the wall clock runs out.
    """

    def __init__(self, reports: list[str], *, laggard: int):
        self.reports = reports
        self.laggard = laggard

    def start(self, *, prompt: str, pass_number: int, session_id: str) -> dict:
        return {"id": f"interaction-{pass_number}", "status": "in_progress"}

    def get(self, interaction_id: str) -> dict:
        number = int(interaction_id.rsplit("-", 1)[1])
        if number == self.laggard:
            return {"id": interaction_id, "status": "in_progress"}
        return {
            "id": interaction_id,
            "status": "completed",
            "output_text": self.reports[number - 1],
        }


def _clock(monkeypatch, step: float):
    """A monotonic clock that moves ``step`` seconds every time it is read."""
    import coscientist.evidence

    ticks = [0.0]

    def monotonic() -> float:
        ticks[0] += step
        return ticks[0] - step

    monkeypatch.setattr(coscientist.evidence.time, "monotonic", monotonic)


def test_a_wave_cut_off_by_the_deadline_still_yields_the_passes_that_finished(
    monkeypatch,
):
    """Six completed Deep Research reports were discarded because a seventh ran long.

    Seen on a live production run: the stage reported "7 passes attempted, 0 source
    leads, $0.00" and an evidence floor met by nothing, after thirty-three minutes
    and six finished searches it had already paid for. The poller returned the
    whole wave unread the moment one pass passed the deadline.
    """
    _clock(monkeypatch, step=5.0)
    transport = SlowTransport(list(FAN_OUT_REPORTS), laggard=2)
    session = Session(question="Does treatment X improve outcome Y?")

    manifest = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        pass_timeout_seconds=6,
    ).run(session, _plan(session))

    assert manifest.convergence_reason == "deep_research_timed_out"
    assert len(manifest.source_leads) == len(EVIDENCE_FACETS) - 1
    assert [run.status for run in manifest.runs].count("completed") == 6
    # The pass that ran out of time says so in words rather than as an empty object.
    laggard = next(run for run in manifest.runs if run.pass_number == 2)
    assert laggard.status == "timed_out"
    assert laggard.error == "Deep Research exceeded the local pass deadline."
    # And the deadline ends the search: no second wave is bought after it.
    assert len(manifest.runs) == len(EVIDENCE_FACETS)


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


def test_a_pass_records_the_facet_it_was_planned_with_and_not_one_read_off_it():
    """The gap-closing pass is planned with no facet, and that is a fact about it.

    The normalizer fills the field in anyway -- it is reading a report, not the
    plan -- and a live gap pass came back labelled ``long_term_safety``, a facet
    nothing in this run plans, searches or scores. Everything downstream reads the
    field as what the pass was sent to cover, so an invented one puts the pass
    inside the fan-out it was run to finish.
    """
    invented = json.dumps(
        {
            "question": "Q",
            "facet": "long_term_safety",
            "research_directions": ["Q"],
            "statements": [],
        }
    )
    session = Session(question="Q")
    manifest = DiscoveryManifest(question="Q")
    wave = [
        DeepResearchRun(pass_number=1, facet="supporting", interaction_id="a"),
        DeepResearchRun(pass_number=2, facet="", interaction_id="b"),
    ]
    payloads = {
        "a": {"status": "completed", "output_text": "A supporting report."},
        "b": {"status": "completed", "output_text": "A gap-closing report."},
    }

    IterativeEvidenceDiscovery(
        FakeTransport([]),
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
    )._ingest_wave(session, wave, payloads, manifest, normalizer=lambda _: invented)

    assert [item.facet for item in manifest.narratives] == ["supporting", ""]


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


def test_a_pass_out_of_time_is_cut_off_by_the_worker_that_inherits_it(monkeypatch):
    """The deadline used to be measured from the top of the call it was set in.
    Under Cloud Tasks a call is one poll and about twenty seconds long, so it
    could never arrive: a live pass sat in_progress for forty minutes with its
    six siblings long finished, and would have been re-polled once a minute for
    as long as Vertex kept saying "in_progress". The clock the budget is spent
    against belongs to the interaction, not to whichever worker is holding it."""

    class NeverFinishes:
        def start(self, *, pass_number: int, **_):
            return {"id": f"slow-{pass_number}", "status": "in_progress"}

        def get(self, interaction_id: str):
            return {"id": interaction_id, "status": "in_progress"}

    transport = NeverFinishes()
    session = Session(question="A pass that will not come back")
    started = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=0,
        pass_timeout_seconds=1800,
    ).run(session, _plan(session))
    assert [run.status for run in started.runs] == ["in_progress"] * len(
        EVIDENCE_FACETS
    )

    # An hour later, on whichever instance the next task lands on.
    an_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    for run in started.runs:
        run.started_at = an_hour_ago

    timed_out = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        polls_per_invocation=1,
        pass_timeout_seconds=1800,
    ).run(session, _plan(session), manifest=started)

    assert timed_out.convergence_reason != "interaction_in_progress"
    assert [run.status for run in timed_out.runs] == ["timed_out"] * len(
        EVIDENCE_FACETS
    )
    assert timed_out.runs[0].error == "Deep Research exceeded the local pass deadline."


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


# The passes that came back while nobody was listening
#
# A wave is read all at once, when every interaction in it is terminal, and the
# statuses on the way there are persisted after every poll. Between those two
# facts sits the whole cost of a restart: a pass that finished before the
# instance died is on the record as completed and nothing was taken from it, and
# the run that picks the manifest back up has to know that.


def _finished(pass_number: int) -> str:
    return (
        "Supporting contradictory negative null replication methods safety "
        f"correction evidence https://pubmed.ncbi.nlm.nih.gov/{pass_number}/"
    )


class _RestartTransport:
    """Starts an interaction in flight, and answers with a report once asked."""

    def __init__(self) -> None:
        self.starts = 0
        self.asked: list[str] = []

    def start(self, *, pass_number: int, **_) -> dict:
        self.starts += 1
        return {"id": f"restart-{pass_number}", "status": "in_progress"}

    def get(self, interaction_id: str) -> dict:
        self.asked.append(interaction_id)
        number = int(interaction_id.rsplit("-", 1)[1])
        return {
            "id": interaction_id,
            "status": "completed",
            "output_text": _finished(number),
        }


def test_a_pass_that_finished_before_the_restart_is_read_after_it():
    """Six of a live seven-pass fan-out were started, paid for, completed, and
    never read: the instance died with one pass still running, and the invocation
    that resumed looked only for interactions still in flight. The panel got the
    literature of the straggler, and the report told the reader the other six
    passes had returned no source leads."""
    transport = _RestartTransport()
    session = Session(question="Restarted research")
    interrupted = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=0,
    ).run(session, _plan(session))
    # What the poll loop had written when the instance went away: every pass but
    # the last had come back, and none of them had been folded in.
    for run in interrupted.runs[:-1]:
        run.status = "completed"
    assert not any(run.raw_artifact_reference for run in interrupted.runs)

    resumed = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
    ).run(session, _plan(session), manifest=interrupted)

    fanned = [run for run in resumed.runs if run.pass_number <= len(EVIDENCE_FACETS)]
    assert transport.starts == len(EVIDENCE_FACETS)
    # Each finished pass is asked about once, rather than skipped as terminal.
    assert sorted(set(transport.asked)) == sorted(
        f"restart-{run.pass_number}" for run in fanned
    )
    assert all(run.raw_artifact_reference for run in fanned)
    assert {narrative.pass_number for narrative in resumed.narratives} >= {
        run.pass_number for run in fanned
    }
    # And the corpus holds what each of them found, not only the straggler's.
    assert {lead.canonical_url for lead in resumed.source_leads} >= {
        f"https://pubmed.ncbi.nlm.nih.gov/{run.pass_number}/" for run in fanned
    }


def test_an_interaction_that_cannot_be_fetched_back_says_which_failed():
    """A stored interaction that has expired is a fetch this run could not make.
    Recorded as "Deep Research completed without a report" it reads as a provider
    that produced nothing, and the pass gets written off rather than retried."""

    class _GoneTransport(_RestartTransport):
        def get(self, interaction_id: str) -> dict:
            self.asked.append(interaction_id)
            raise RuntimeError("interaction 404: not found")

    transport = _GoneTransport()
    session = Session(question="Expired research")
    interrupted = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        polls_per_invocation=0,
    ).run(session, _plan(session))
    for run in interrupted.runs:
        run.status = "completed"

    resumed = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
    ).run(session, _plan(session), manifest=interrupted)

    assert [run.status for run in resumed.runs] == ["failed"] * len(EVIDENCE_FACETS)
    assert all("404" in run.error for run in resumed.runs)


def test_a_pass_finished_and_unread_is_not_counted_as_one_the_run_has():
    """The card said "8 completed of 8 attempted" over a corpus built from two of
    them, because a pass was counted the moment the provider called it done."""
    from coscientist.evidence import unread_passes

    manifest = DiscoveryManifest(
        question="Restarted research",
        runs=[
            DeepResearchRun(pass_number=1, status="completed", interaction_id="a"),
            DeepResearchRun(
                pass_number=2,
                status="completed",
                interaction_id="b",
                raw_artifact_reference="interaction://b",
            ),
        ],
    )

    assert unread_passes(manifest) == {1}
    said = orchestration.CoScientistWorkflow._evidence_summary(manifest)
    assert "1 completed of 2 attempted" in said
    assert "1 finished and could not be read back" in said
    # A manifest from before the field was recorded says nothing either way, so
    # every pass on it keeps the standing it always had.
    for run in manifest.runs:
        run.raw_artifact_reference = ""
    assert unread_passes(manifest) == set()
    assert "2 completed of 2 attempted" in (
        orchestration.CoScientistWorkflow._evidence_summary(manifest)
    )


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


def test_a_facet_pass_is_told_not_to_hand_the_pipelines_words_back_as_prose():
    """The instruction's own phrasing came back as the finding.

    Two facets of a live run reported nothing found, and both reported it in the
    words the prompt had used: "In strict adherence to the defined evaluation
    parameters, this constitutes a genuinely empty facet", and "An exhaustive
    analysis of the available literature reveals an **honest empty facet**". A
    reader of the report has no idea what a facet is.
    """
    session = Session(question="Can a coating extend lithium-ion cycle life?")
    plan = ResearchPlan(question=session.question, intended_claim="hypothesis")
    prompt = build_research_prompt(session, plan, pass_number=1, facet="contradictory")

    # The tag is still demanded, because a statement is filed by it.
    assert "tag every statement you return with the facet contradictory" in prompt
    # And it is confined to the field it is read from.
    assert "never in its prose" in prompt
    assert "mean nothing to a reader" in " ".join(prompt.split())
    assert "honest empty facet" not in prompt
    assert "genuinely does not exist" not in prompt


@pytest.mark.parametrize(
    ("language", "reported_in"),
    [("en", "Summarise it in English"), ("zh-Hans", "Summarise it in Simplified")],
)
def test_the_one_prompt_that_searches_the_open_web_is_told_what_to_report_in(
    language: str, reported_in: str
):
    """It was the one prompt in the system carrying no working language at all.
    A Chinese run's whole Knowledge Base came back in English because nothing
    here said otherwise, and an English run carried a German paper's own sentence
    into its findings: "Die Lebensdauer der Elektrodenmaterialien wird durch die
    Beschichtung stark erhöht" stood among the evidence findings."""
    session = Session(
        question="Can a coating extend lithium-ion cycle life?", language=language
    )
    plan = ResearchPlan(question=session.question, intended_claim="hypothesis")
    prompt = build_research_prompt(session, plan, pass_number=1, facet="supporting")

    assert reported_in in prompt
    assert "never carry a sentence across in the language you found it in" in prompt
    # The title names a document that exists, and translating it names one that
    # does not.
    assert "title, authors, and publisher are the exception" in prompt
    assert ("简体中文" in prompt) is (language == "zh-Hans")


@pytest.mark.parametrize(
    ("language", "reported_in"),
    [("en", "Summarise it in English"), ("zh-Hans", "Summarise it in Simplified")],
)
def test_the_extractor_is_told_the_working_language_of_the_run_it_extracts_for(
    language: str, reported_in: str
):
    """ "Copied verbatim" is asked of the URLs, and this is where it was taken to
    cover the prose as well."""
    seen: dict[str, str] = {}

    def _capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            {"question": "Q", "research_directions": [], "statements": []}
        )

    normalize_report(
        question="Q",
        report="A German paper reports longer electrode life.",
        pass_number=1,
        normalizer=_capture,
        language=language,
    )

    assert reported_in in seen["prompt"]
    assert ("简体中文" in seen["prompt"]) is (language == "zh-Hans")


def test_a_doi_found_by_retrieval_folds_the_two_leads_it_makes_one_document():
    """The corpus settles before a researcher is shown its size, not after.

    A DOI is what makes two leads one document, and retrieval is where a lead
    that was found by its address learns one. Left as written, the manifest
    carried both rows past the evidence gate and the next merge folded them --
    and the next merge is the gap search a researcher asks for at that gate. So
    a live run answered "search for long-term safety" by reporting the corpus
    had shrunk from 88 leads to 85, having removed nothing at all.
    """
    flow = orchestration.CoScientistWorkflow("Can a coating improve cycle life?")
    manifest = DiscoveryManifest(
        question=flow.session.question,
        source_leads=[
            SourceLead(
                canonical_url="https://doi.org/10.1039/d5ta02510a",
                title="Atomic layer deposition of TiO2 on NCM83",
                identifiers={"doi": "10.1039/d5ta02510a"},
                facets=["supporting"],
            ),
            SourceLead(
                canonical_url="https://pubs.rsc.org/en/content/articlehtml/2024/ta/x",
                title="Atomic layer deposition of TiO2 on NCM83 - RSC",
                facets=["methods"],
            ),
        ],
    )
    packet = EvidencePacket(
        question=flow.session.question,
        sources=[
            SourceRecord(
                url="https://pubs.rsc.org/en/content/articlehtml/2024/ta/x",
                title="Atomic layer deposition of TiO2 on NCM83",
                verification_status="verified",
                identifiers={"doi": "10.1039/d5ta02510a"},
            )
        ],
    )

    updated = flow._manifest_with_verification(manifest, [packet])

    assert len(updated.source_leads) == 1
    lead = updated.source_leads[0]
    # The verified copy is what the reader can stand on, and the facets are why
    # neither row could simply be dropped.
    assert lead.verification_status == "verified"
    assert lead.facets == ["supporting", "methods"]
    assert updated.verification_handoff_source_ids == [lead.id]
    # And it stays settled: the gap search at the gate merges this corpus again.
    assert len(merge_leads(updated.source_leads, [])) == 1


def test_a_gap_search_that_returns_a_paper_already_held_adds_no_second_row():
    """The corpus a revision hands back is never smaller than the one it took.

    A lead built from a gap search carried the address and not the DOI the
    address states, so the paper the corpus already held under that DOI was
    added a second time. What makes this the researcher's problem rather than a
    tidiness one is the merge: the count they were shown at the gate is settled
    only when the gate is answered, so the number moved under the very request
    they made to improve it.
    """
    flow = orchestration.CoScientistWorkflow("Can a coating improve cycle life?")
    plan = ResearchPlan(
        question=flow.session.question,
        success_criteria=["Cycle life against an uncoated control"],
    )
    held = SourceLead(
        canonical_url="https://pubs.rsc.org/en/content/articlehtml/2024/ta/x",
        title="Atomic layer deposition of TiO2 on NCM83",
        identifiers={"doi": "10.1039/d5ta02510a"},
        facets=["supporting"],
    )
    manifest = DiscoveryManifest(question=flow.session.question, source_leads=[held])
    discovery = Artifact(
        stage="evidence",
        agent="supervisor",
        artifact_type="specialist_output",
        content="### Evidence Discovery",
        schema_name="DiscoveryManifest",
        payload=manifest.model_dump(mode="json"),
    )
    # The same paper, reached the other way round: the gap search cites its DOI.
    packet = EvidencePacket(
        question=flow.session.question,
        sources=[
            SourceRecord(
                id="src_1",
                url="https://doi.org/10.1039/d5ta02510a",
                title="Atomic layer deposition of TiO2 on NCM83",
            )
        ],
        claims=[
            EvidenceClaim(
                claim="A 2 nm TiO2 layer held 92% capacity after 500 cycles.",
                source_id="src_1",
                relation="supports",
            )
        ],
    )
    answer = Artifact(
        stage="evidence",
        agent="evidence_discovery",
        artifact_type="specialist_output",
        content="### Gap search",
        schema_name="EvidencePacket",
        payload=packet.model_dump(mode="json"),
    )

    async def _dispatch(session, specialists, *, feedback="", revision=1):
        return [
            TaskResult(
                task=TaskRecord(
                    context_id=session.id,
                    stage="evidence",
                    agent="evidence_discovery",
                    idempotency_key=f"{session.id}:gap:{revision}",
                    state=TaskState.COMPLETED,
                    output_artifact_id=answer.id,
                ),
                artifact=answer,
            )
        ]

    flow.task_bus.dispatch_stage = _dispatch
    updated, _ = asyncio.run(
        flow._gap_directed_search(
            plan,
            manifest,
            discovery,
            feedback="Nothing here covers long-term safety. Search for that.",
            revision=2,
        )
    )

    assert len(updated.source_leads) == 1
    lead = updated.source_leads[0]
    assert lead.canonical_url == held.canonical_url
    # Merged rather than ignored: the gap search is what says this paper answers
    # the researcher's question as well as the one it was found under.
    assert lead.claim_relations == ["supports"]


def _cited_payload(text: str, annotations: list[dict]) -> dict:
    return {
        "output_text": text,
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text, "annotations": annotations}],
            }
        ],
    }


CITED_REPORT = "Coatings help [cite: 9]. Onset is early [cite: 2, 4].\n"
FIRST = "https://pubmed.ncbi.nlm.nih.gov/1/"
SECOND = "https://pubmed.ncbi.nlm.nih.gov/2/"


def _renumbering_payload() -> tuple[dict, list[str]]:
    """A pass whose own numbering agrees with nothing.

    ``[cite: 9]`` is the ninth entry of a source list the provider keeps to
    itself and never returns; the second span names two documents under two
    numbers that are not two of anything the caller holds. Both were resolved
    against the URL list by position, which on the live report measured here
    disagreed with the annotations on a hundred and seventeen of a hundred and
    twenty-nine spans.
    """
    first = CITED_REPORT.index("[cite: 9]")
    second = CITED_REPORT.index("[cite: 2, 4]")
    payload = _cited_payload(
        CITED_REPORT,
        [
            {
                "type": "url_citation",
                "url": SECOND,
                "title": "Onset",
                "start_index": first,
                "end_index": first + len("[cite: 9]"),
            },
            {
                "type": "url_citation",
                "url": FIRST,
                "title": "Coatings",
                "start_index": second,
                "end_index": second + len("[cite: 2, 4]"),
            },
        ],
    )
    return payload, [SECOND, FIRST]


def test_a_markers_number_is_rewritten_onto_the_list_it_is_resolved_against():
    payload, cited = _renumbering_payload()
    seen = {}

    normalize_report(
        question="Q",
        report=extract_report(payload),
        pass_number=1,
        normalizer=lambda prompt: seen.setdefault("prompt", prompt) and "{}",
        citation_urls=cited,
        payload=payload,
    )

    # Not [cite: 9] and [cite: 2, 4]: the first span names the first entry of the
    # list beneath the report and the second names the second.
    assert "Coatings help [cite: 1]." in seen["prompt"]
    assert "Onset is early [cite: 2]." in seen["prompt"]
    assert f"[1] {SECOND}" in seen["prompt"]
    assert f"[2] {FIRST}" in seen["prompt"]


def test_the_fallback_reads_a_marker_as_the_source_the_annotation_named():
    """No normalizer, so the paragraph scanner is what attaches the sources."""
    payload, cited = _renumbering_payload()

    narrative = normalize_report(
        question="Q",
        report=extract_report(payload),
        pass_number=1,
        citation_urls=cited,
        payload=payload,
    )

    ((statement,),) = (narrative.statements,)
    assert sorted(statement.source_urls) == sorted([FIRST, SECOND])


def test_a_report_with_no_annotations_keeps_the_prose_it_arrived_with():
    """The rewrite has nothing to say about a payload it can read no report out
    of, and what it must not do is hand back the empty string it made of it. The
    report reaching here was assembled by the caller and may have come from
    somewhere this rewrite cannot follow."""
    report = "Coatings help. See https://pubmed.ncbi.nlm.nih.gov/1/"

    narrative = normalize_report(
        question="Q",
        report=report,
        pass_number=1,
        citation_urls=[FIRST],
        payload={"status": "completed"},
    )

    ((statement,),) = (narrative.statements,)
    assert statement.source_urls == [FIRST]
    assert "Coatings help." in narrative.summary
