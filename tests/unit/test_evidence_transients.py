"""One bad second at Vertex used to cost a run its whole corpus.

``session_c0f2b7ce70104cb2`` on the live deployment: seven Deep Research passes
launched, seven fetched, and while the normalizer was reading the reports back
Vertex answered ``503 UNAVAILABLE``. That sentence travelled up through the
extractor, through the wave, through the evidence stage and out to the worker,
which recorded it and stopped. Seven completed searches, already paid for, were
still sitting in the payloads nobody read.

A run is hours long and hundreds of Vertex calls deep, so meeting one bad second
is not the unlikely case. These are the three places that used to make it fatal:
the call itself never asked twice, the extractor caught only parse failures, and
the poll loop -- alone among the three polls in the file -- was unguarded.
"""

from __future__ import annotations

import json

import pytest
from google.genai.errors import ClientError, ServerError

from coscientist.evidence import (
    EvidenceArtifactStore,
    IterativeEvidenceDiscovery,
    call_vertex,
    normalize_report,
)
from coscientist.models import ResearchPlan, Session

QUESTION = "Does a protective coating improve rechargeable battery cycle life?"
ACS = "https://pubs.acs.org/doi/10.1021/acsami.4c13335"
REPORT = f"Alumina coatings halve first-cycle loss ([ACS]({ACS}))."


def _unavailable() -> ServerError:
    """The exact error off the live traceback, reconstructed."""
    return ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "The service is currently unavailable.",
                "status": "UNAVAILABLE",
            }
        },
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The waits are real seconds and this suite is about what is asked, not when."""
    import coscientist.evidence

    monkeypatch.setattr(coscientist.evidence.time, "sleep", lambda _: None)


# ---------------------------------------------------------------------------
# Asking again
# ---------------------------------------------------------------------------


def test_a_transient_refusal_is_asked_again_and_answered():
    calls = []

    def flaky():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise _unavailable()
        return "the answer"

    assert call_vertex("a test call", flaky) == "the answer"
    assert calls == [1, 2, 3]


def test_a_refusal_about_the_request_is_not_asked_again():
    """A 400 says the request is wrong. Repeating it buys identical refusals."""
    calls = []

    def rejected():
        calls.append(1)
        raise ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})

    with pytest.raises(ClientError):
        call_vertex("a test call", rejected)
    assert calls == [1]


def test_a_service_that_stays_down_still_raises():
    """Retrying is not pretending. Four refusals are a refusal."""

    def down():
        raise _unavailable()

    with pytest.raises(ServerError):
        call_vertex("a test call", down)


# ---------------------------------------------------------------------------
# Reading a report back
# ---------------------------------------------------------------------------


def test_an_extractor_that_dies_on_vertex_falls_back_to_the_deterministic_read():
    """The live failure. The report is in hand and readable without a model.

    Only parse failures used to reach the fallback, so an extractor that raised
    a ``ServerError`` instead of returning bad JSON escaped the stage entirely.
    """

    def raises(_prompt: str) -> str:
        raise _unavailable()

    narrative = normalize_report(
        question=QUESTION,
        report=REPORT,
        pass_number=3,
        normalizer=raises,
        citation_urls=[ACS],
    )

    assert narrative.pass_number == 3
    assert [statement.source_urls for statement in narrative.statements] == [[ACS]]


def test_an_extractor_that_answers_is_still_preferred_over_the_fallback():
    """The widened clause must not swallow a working extractor's answer."""
    normalized = {
        "question": QUESTION,
        "statements": [
            {
                "text": "Alumina coatings halve first-cycle loss.",
                "facet": "supporting",
                "source_urls": [ACS],
                "originating_pass": 3,
            }
        ],
    }

    narrative = normalize_report(
        question=QUESTION,
        report=REPORT,
        pass_number=3,
        normalizer=lambda _: json.dumps(normalized),
        citation_urls=[ACS],
    )

    assert [statement.text for statement in narrative.statements] == [
        "Alumina coatings halve first-cycle loss."
    ]


# ---------------------------------------------------------------------------
# Polling a wave
# ---------------------------------------------------------------------------


class FlakyPollTransport:
    """A wave whose passes are polled twice, and one poll is refused.

    The interaction behind an unanswered poll is untouched by it: still running
    on Vertex, still being paid for, and answerable on the next tick.
    """

    def __init__(self, reports: list[str], *, refuse_on: int):
        self.reports = reports
        self.refuse_on = refuse_on
        self.polls = 0

    def start(self, *, prompt: str, pass_number: int, session_id: str) -> dict:
        return {"id": f"interaction-{pass_number}", "status": "in_progress"}

    def get(self, interaction_id: str) -> dict:
        self.polls += 1
        if self.polls == self.refuse_on:
            raise _unavailable()
        number = int(interaction_id.rsplit("-", 1)[1])
        return {
            "id": interaction_id,
            "status": "completed",
            "output_text": self.reports[number - 1],
        }


def test_one_unanswered_poll_does_not_take_the_wave_with_it():
    """Raised through, a single refused poll failed a wave of running passes."""
    from coscientist.models import EVIDENCE_FACETS

    reports = [
        f"Coatings change cycle life on facet {index} ([ACS]({ACS})). "
        "The effect is measured against an uncoated control."
        for index in range(1, len(EVIDENCE_FACETS) + 2)
    ]
    transport = FlakyPollTransport(reports, refuse_on=2)
    session = Session(question=QUESTION)

    manifest = IterativeEvidenceDiscovery(
        transport,
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        pass_timeout_seconds=600,
    ).run(session, ResearchPlan(question=QUESTION, intended_claim="testable claim"))

    assert manifest.convergence_reason != "deep_research_timed_out"
    assert [run.status for run in manifest.runs].count("completed") == len(
        manifest.runs
    )
    assert manifest.source_leads
