"""What the workspace says while the evidence stage is opening its sources.

A live run finished its eighth Deep Research pass at 12:37 and spent the next
twenty-five minutes verifying fifty-six leads. The workspace held the sentence
the last poll had written -- "Deep Research is still running; next status check
in 60 seconds" -- for the whole of it, over a stage that had stopped searching
half an hour earlier. The run was healthy; the page had no way to say so.

Where a task queue drives the stage each poll returns through the caller, which
rewrites the line on its way past. Where one is not configured -- the deployed
service -- discovery is inside the same single call as verification, and the run
after that one sat nine minutes on "Specialists are preparing the next research
gate." with three of its eight searches already back. Both halves have to
narrate themselves from the inside.
"""

from __future__ import annotations

import json

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.evidence import (
    EVIDENCE_FACETS,
    EvidenceArtifactStore,
    IterativeEvidenceDiscovery,
    discovery_progress_sentence,
)
from coscientist.models import (
    VERIFICATION_BATCH_SIZE,
    Artifact,
    DeepResearchRun,
    DiscoveryManifest,
    SourceLead,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Does a protective coating improve rechargeable battery cycle life?"


class _QuietVerifier(DeterministicProvider):
    """Answers the verification role with an empty packet, and quickly.

    What each batch returns is not what these tests are about; that it returned,
    and that the count on the page moved when it did, is.
    """

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "source_verification":
            return json.dumps({"question": QUESTION, "sources": [], "claims": []})
        return super().complete(role=role, prompt=prompt)


def _manifest(count: int) -> DiscoveryManifest:
    return DiscoveryManifest(
        question=QUESTION,
        source_leads=[
            SourceLead(
                canonical_url=f"https://doi.org/10.1000/{index}",
                title=f"Paper {index}",
            )
            for index in range(count)
        ],
    )


def _run(monkeypatch: pytest.MonkeyPatch, leads: int) -> CoScientistWorkflow:
    """A run whose literature search returned a manifest, as Deep Research does."""
    manifest = _manifest(leads)
    flow = CoScientistWorkflow(QUESTION, _QuietVerifier())

    async def _discovered(self, plan, *, feedback, revision):
        artifact = Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content="Deep Research returned.",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
        )
        self.session.artifacts.append(artifact)
        return manifest, artifact

    monkeypatch.setattr(CoScientistWorkflow, "_discovered_evidence", _discovered)
    return flow


def test_the_verification_fan_out_counts_the_sources_it_has_opened(monkeypatch):
    """Twenty-eight leads, three batches, and a line that moves three times."""
    flow = _run(monkeypatch, VERIFICATION_BATCH_SIZE * 2 + 4)
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    assert said[0] == "Opening 28 sources to check what the documents say."
    checked = [line for line in said if line.startswith("Checked ")]
    assert len(checked) == 3
    # However the batches interleave, the last one to land says the whole corpus
    # was opened -- the number the reader is waiting to see reach itself.
    assert checked[-1] == (
        "Checked 28 of 28 sources against what the document actually says."
    )


def test_one_source_is_opened_rather_than_one_sources(monkeypatch):
    """A count of one is the case a formatted count gets wrong, and it is also
    the case a rehearsal produces."""
    flow = _run(monkeypatch, 1)
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    assert said[0] == "Opening 1 source to check what the documents say."


def test_a_narrator_that_raises_does_not_cost_the_stage_its_research(monkeypatch):
    """The writer goes to the database the lease lives in, and a database says
    no for reasons that have nothing to do with this run. Losing eight paid
    Deep Research passes over a sentence would be the more expensive failure."""
    flow = _run(monkeypatch, 6)

    def _refuse(detail: str) -> None:
        raise RuntimeError("the operation row is gone")

    flow.progress = _refuse

    flow.accept(flow.preview(), actor="test_researcher")
    draft = flow.preview()

    assert draft is not None
    assert flow.stage == "evidence"


def test_a_run_with_nobody_listening_narrates_nothing_and_still_finishes(monkeypatch):
    """The CLI and every test construct a workflow with no writer attached."""
    flow = _run(monkeypatch, 6)

    assert flow.progress is None
    flow.accept(flow.preview(), actor="test_researcher")

    assert flow.preview() is not None


def test_the_same_sentence_is_not_written_twice(monkeypatch):
    """The discovery poll loop asks every fifteen seconds and usually has the
    same answer. A sentence that has not changed is not news, and a run polling
    for forty minutes would otherwise write two hundred identical rows."""
    flow = _run(monkeypatch, 6)
    said: list[str] = []
    flow.progress = said.append

    flow._note("Deep Research has finished 3 of 8 searches; 41 sources so far.")
    flow._note("Deep Research has finished 3 of 8 searches; 41 sources so far.")
    flow._note("Deep Research has finished 4 of 8 searches; 41 sources so far.")

    assert said == [
        "Deep Research has finished 3 of 8 searches; 41 sources so far.",
        "Deep Research has finished 4 of 8 searches; 41 sources so far.",
    ]


def _with_runs(*statuses: str, leads: int = 0) -> DiscoveryManifest:
    manifest = _manifest(leads)
    manifest.runs = [
        DeepResearchRun(pass_number=number, status=status)
        for number, status in enumerate(statuses, start=1)
    ]
    return manifest


def test_a_wave_still_out_is_reported_by_how_much_of_it_has_landed():
    manifest = _with_runs(
        "completed", "completed", "completed", "in_progress", "queued", leads=41
    )

    assert discovery_progress_sentence(manifest) == (
        "Deep Research has finished 3 of 5 searches; 41 sources so far."
    )


def test_a_pass_that_timed_out_or_failed_is_not_still_searching():
    """Counted by what is in flight rather than by what succeeded. Neither
    status is in the controller's terminal set, and counting the complement
    would have reported a refused pass as still searching for the rest of the
    run -- a stage that never finishes, on a page that never says why."""
    manifest = _with_runs("timed_out", "failed", "completed", "in_progress")

    assert discovery_progress_sentence(manifest).startswith(
        "Deep Research has finished 3 of 4 searches;"
    )


def test_the_last_pass_landing_says_the_searches_are_back():
    """No count of sources on this branch. It fires in the gap between the last
    pass going terminal and the fold that reads it, where the corpus on the
    manifest is still the previous wave's and any number would be a stale one."""
    manifest = _with_runs("completed", "completed", leads=41)

    assert discovery_progress_sentence(manifest) == (
        "All 2 searches are back; folding what they found in."
    )


def test_a_manifest_with_no_passes_yet_says_so():
    assert (
        discovery_progress_sentence(_manifest(0)) == "Choosing which searches to run."
    )


def test_one_search_and_one_source_are_counted_in_the_singular():
    """The grounded fallback runs a single pass, so this is not a corner case."""
    assert discovery_progress_sentence(_with_runs("in_progress", leads=1)) == (
        "Deep Research has finished 0 of 1 search; 1 source so far."
    )
    # "All 1 search are back" is what a count dropped into a plural sentence
    # produces; one search gets its own sentence rather than a patched verb.
    assert discovery_progress_sentence(_with_runs("completed")) == (
        "The search is back; folding what it found in."
    )


class _StaggeredTransport:
    """Seven interactions that come back one poll apart, as a real wave does."""

    REPORT = (
        "Supporting evidence https://pubmed.ncbi.nlm.nih.gov/1/ and contradictory "
        "negative null replication methods safety correction evidence "
        "https://www.fda.gov/example"
    )

    def __init__(self) -> None:
        self.polls: dict[str, int] = {}

    def start(self, *, pass_number: int, **_):
        return {"id": f"interaction-{pass_number}", "status": "in_progress"}

    def get(self, interaction_id: str):
        seen = self.polls.get(interaction_id, 0) + 1
        self.polls[interaction_id] = seen
        due = int(interaction_id.rsplit("-", 1)[1])
        if seen < due:
            return {"id": interaction_id, "status": "in_progress"}
        return {"id": interaction_id, "status": "completed", "output_text": self.REPORT}


def test_the_poll_loop_narrates_the_wave_it_is_waiting_on():
    """The wiring, through the real controller rather than a stub of it: the
    sentence is written by the callback the poll loop already calls to save the
    manifest, so if that call is ever removed the page goes quiet again."""
    flow = CoScientistWorkflow(QUESTION, _QuietVerifier())
    flow.evidence_discovery = IterativeEvidenceDiscovery(
        _StaggeredTransport(),
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
    )
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    total = len(EVIDENCE_FACETS)
    # The whole stage, in order and without a gap: the wave going out, one line
    # for each pass that came back, the fold, and then verification. Nine
    # minutes of "Specialists are preparing the next research gate." is what
    # this same run said before the callback wrote anything.
    assert said == [
        *(
            f"Deep Research has finished {done} of {total} searches; 0 sources so far."
            for done in range(total)
        ),
        f"All {total} searches are back; folding what they found in.",
        "Opening 2 sources to check what the documents say.",
        "Checked 2 of 2 sources against what the document actually says.",
    ]
