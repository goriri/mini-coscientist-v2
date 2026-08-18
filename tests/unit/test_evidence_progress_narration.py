"""What the workspace says while the evidence stage is opening its sources.

A live run finished its eighth Deep Research pass at 12:37 and spent the next
twenty-five minutes verifying fifty-six leads. The workspace held the sentence
the last poll had written -- "Deep Research is still running; next status check
in 60 seconds" -- for the whole of it, over a stage that had stopped searching
half an hour earlier. The run was healthy; the page had no way to say so.

Discovery never had this problem because each poll returns through the caller,
which rewrites the line on its way past. Verification is one call, and the
narration has to come from inside it.
"""

from __future__ import annotations

import json

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.models import (
    VERIFICATION_BATCH_SIZE,
    Artifact,
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
