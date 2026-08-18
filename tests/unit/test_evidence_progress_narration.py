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
    EnrichmentRequest,
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


def _run(
    monkeypatch: pytest.MonkeyPatch,
    leads: int,
    *,
    outstanding: int = 0,
) -> CoScientistWorkflow:
    """A run whose literature search returned a manifest, as Deep Research does."""
    manifest = _manifest(leads)
    manifest.enrichment_requests = [
        EnrichmentRequest(
            provider="google_search",
            gap_ids=[f"gap-{index}"],
            query=f"{QUESTION} gap {index}",
            status="queued",
        )
        for index in range(outstanding)
    ]
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

    assert "Opening 28 sources to check what the documents say." in said
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

    assert "Opening 1 source to check what the documents say." in said


def test_the_searches_that_close_a_gap_say_how_many_are_running(monkeypatch):
    """A coverage score short of sufficient queues up to six more searches, and
    they run between the corpus being written down and the first document being
    opened -- a model fan-out with nothing on the manifest to show for it yet."""
    flow = _run(monkeypatch, 6, outstanding=4)
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    assert "Running 4 follow-up searches on what the corpus left open." in said


def test_one_gap_search_is_not_four(monkeypatch):
    flow = _run(monkeypatch, 6, outstanding=1)
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    assert "Running 1 follow-up search on what the corpus left open." in said


def test_a_corpus_that_left_nothing_open_says_nothing_about_gaps(monkeypatch):
    flow = _run(monkeypatch, 6)
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    assert not [line for line in said if line.startswith("Running ")]


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


def test_a_sentence_does_not_come_back_after_one_other_line(monkeypatch):
    """Two callers sit one line apart around the registry lookup and both read
    the same manifest. Against the previous line alone, the page said what the
    fold had produced, said what the lookup was doing, and then said what the
    fold had produced again -- going backwards, which reads as a stall."""
    flow = _run(monkeypatch, 6)
    said: list[str] = []
    flow.progress = said.append

    flow._note("All 7 searches came back; 114 sources to work with.")
    flow._note("Looking up publication details for 114 sources.")
    flow._note("All 7 searches came back; 114 sources to work with.")
    # Far enough back to be news again, which is what a revisited stage is.
    flow._note("Opening 114 sources to check what the documents say.")
    flow._note("Checked 114 of 114 sources against what the document actually says.")
    flow._note("All 7 searches came back; 114 sources to work with.")

    assert said == [
        "All 7 searches came back; 114 sources to work with.",
        "Looking up publication details for 114 sources.",
        "Opening 114 sources to check what the documents say.",
        "Checked 114 of 114 sources against what the document actually says.",
        "All 7 searches came back; 114 sources to work with.",
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


def test_a_wave_that_has_been_read_does_not_say_it_is_being_folded():
    """The same callback is called again on the far side of the fold. Both times
    every pass is terminal, so both times the sentence above was written -- the
    page announcing, after seven reports had been read, work already done."""
    manifest = _with_runs("completed", "completed", leads=41)
    for index, run in enumerate(manifest.runs, start=1):
        run.raw_artifact_reference = f"gs://bucket/pass-{index}.json"

    assert discovery_progress_sentence(manifest) == (
        "All 2 searches came back; 41 sources to work with."
    )


def test_a_wave_half_read_is_still_being_folded():
    """The ingest reads its passes one at a time and writes the reference for
    each as it goes. Until the last one is written the fold is still running,
    and the count on the manifest is part of a wave rather than all of it."""
    manifest = _with_runs("completed", "completed", leads=41)
    manifest.runs[0].raw_artifact_reference = "gs://bucket/pass-1.json"

    assert discovery_progress_sentence(manifest) == (
        "All 2 searches are back; folding what they found in."
    )


def test_a_single_read_search_keeps_its_own_verb():
    manifest = _with_runs("completed", leads=1)
    manifest.runs[0].raw_artifact_reference = "gs://bucket/pass-1.json"

    assert discovery_progress_sentence(manifest) == (
        "The search came back; 1 source to work with."
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
    # The whole stage, in order and without a gap: the wave going out, the fold,
    # one line for each report as it is read, what the fold produced, the corpus
    # being written down, verification, the merge and the survey. Nine minutes
    # of "Specialists are preparing the next research gate." is what this same
    # run said before the callback wrote anything, and every line below that is
    # not a poll stands over a stretch that was silent after it started writing:
    # a model call per report, then two more over the whole corpus.
    assert said == [
        *(
            f"Deep Research has finished {done} of {total} searches; 0 sources so far."
            for done in range(total)
        ),
        f"All {total} searches are back; folding what they found in.",
        *(
            f"Reading what search {position} of {total} came back with."
            for position in range(1, total + 1)
        ),
        f"All {total} searches came back; 2 sources to work with.",
        "Writing down what the searches found, source by source.",
        "Opening 2 sources to check what the documents say.",
        "Checked 2 of 2 sources against what the document actually says.",
        "Merging 2 sets of findings into one corpus.",
        "Writing the knowledge survey over 2 sources.",
    ]


class _RedirectedTransport(_StaggeredTransport):
    """A wave whose reports cite through the grounding redirector, as Deep
    Research does: it mints a fresh token for every citation it prints."""

    REPORT = (
        "Supporting evidence https://vertexaisearch.cloud.google.com/"
        "grounding-api-redirect/AbC1 and contradictory negative null replication "
        "methods safety correction evidence https://vertexaisearch.cloud.google."
        "com/grounding-api-redirect/AbC2"
    )


class _CountingEnricher:
    """Stands in for the registry lookup, which is one network round trip per
    lead against Crossref and its neighbours."""

    def enrich(self, leads):
        return leads


def test_the_two_stretches_between_the_fold_and_the_verifier_are_narrated():
    """Neither is on the manifest and neither is a model call the caller makes,
    so neither was visible: a registry round trip per retained lead, and a
    redirector followed per citation Deep Research printed. On a live run of a
    hundred and fourteen sources they are the bulk of the ten silent minutes
    between the last search landing and the first document being opened."""
    flow = CoScientistWorkflow(QUESTION, _QuietVerifier())
    flow.evidence_discovery = IterativeEvidenceDiscovery(
        _RedirectedTransport(),
        EvidenceArtifactStore(bucket_name=""),
        poll_interval_seconds=0,
        registry_enricher=_CountingEnricher(),
    )
    said: list[str] = []
    flow.progress = said.append

    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()

    # One lead, because a redirector is a fresh token per citation and the two
    # in this report resolve to the same unfollowable address -- which is also
    # why the singular matters here: "1 search link back to the documents they
    # name" is what a count dropped into a fixed sentence produces.
    lookup = "Looking up publication details for 1 source."
    following = "Following 1 search link back to the document it names."
    assert lookup in said
    assert following in said
    # In that order, and both after the reports have been read: the leads have
    # to exist before either can count them.
    total = len(EVIDENCE_FACETS)
    assert said.index(lookup) > said.index(
        f"Reading what search {total} of {total} came back with."
    )
    assert said.index(following) > said.index(lookup)


def test_a_corpus_of_plain_links_is_not_told_they_are_being_followed():
    """The grounded fallback and a resumed manifest both arrive with their
    locators already naming documents. A sentence about following redirectors
    would stand over no work at all, and the next one is the honest one."""
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

    assert not [line for line in said if line.startswith("Following ")]
