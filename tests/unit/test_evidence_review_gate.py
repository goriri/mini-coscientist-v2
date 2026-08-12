"""The opt-in stop on the evidence base, before the generators reason over it.

The milestone profile does not count evidence as a milestone: discovery runs as
internal work and the first thing the researcher is handed is eight hypotheses
built on whatever came back. That is right when the corpus is sound. When it is
thin it is only visible by reading it, and by the generate gate four strategies
have already spent themselves on it. This gate is the cheap place to find out.
"""

import pytest
from fastapi import BackgroundTasks

from app import research_api
from app.research_api import CreateResearchSession, create_research_session
from coscientist.cli import main as cli_main
from coscientist.ledger import ResearchLedger
from coscientist.models import (
    ApprovalProfile,
    Artifact,
    ArtifactStatus,
    DiscoveryManifest,
    EvidenceClaim,
    EvidencePacket,
    Session,
    SourceLead,
    SourceRecord,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective coating improve battery cycle life?"


def _at_evidence(**kwargs) -> CoScientistWorkflow:
    flow = CoScientistWorkflow(
        QUESTION,
        workflow_version=2,
        **kwargs,
    )
    flow.accept(flow.preview())
    assert flow.stage == "evidence"
    return flow


def test_milestone_runs_through_evidence_when_the_gate_was_not_asked_for():
    flow = _at_evidence(approval_profile=ApprovalProfile.MILESTONE)

    assert flow.session.evidence_review is False
    assert flow.requires_human_approval is False


def test_the_run_that_opted_in_stops_on_its_evidence_base():
    flow = _at_evidence(
        approval_profile=ApprovalProfile.MILESTONE, evidence_review=True
    )

    assert flow.session.evidence_review is True
    assert flow.requires_human_approval is True
    # Not a new milestone for everything else: the toggle buys one stop, and the
    # stages the profile already ran internally still run internally.
    flow.accept_exploratory_evidence()
    flow.accept(flow.preview())
    assert flow.stage == "generate"
    assert flow.requires_human_approval is False


def test_the_gate_refuses_to_be_stepped_over_automatically():
    """The whole point is that somebody reads it, so the auto path is closed."""
    flow = _at_evidence(
        approval_profile=ApprovalProfile.MILESTONE, evidence_review=True
    )
    flow.accept_exploratory_evidence()

    with pytest.raises(ValueError, match="Automatic decisions are disabled"):
        flow.accept(flow.preview(), automatic=True)


def test_advancing_to_the_next_human_gate_stops_at_evidence():
    """``advance_to_human_gate`` is what the web API calls after every accept."""
    flow = CoScientistWorkflow(
        QUESTION,
        approval_profile=ApprovalProfile.MILESTONE,
        evidence_review=True,
        workflow_version=2,
    )
    flow.accept(flow.preview())
    flow.accept_exploratory_evidence()
    flow.advance_to_human_gate()

    assert flow.stage == "evidence"


def test_an_unattended_run_does_not_record_a_gate_it_will_never_stop_at():
    """``run_auto`` accepts every draft it drafts; nobody is standing there.

    Recording the request anyway would put a promise in the session that the run
    is built to break, and the launcher reads this field back to tell the
    researcher what their run is actually going to do.
    """
    flow = CoScientistWorkflow(
        QUESTION,
        approval_profile=ApprovalProfile.AUTO,
        evidence_review=True,
        workflow_version=2,
    )

    assert flow.session.evidence_review is False


def _batched_evidence_draft(flow: CoScientistWorkflow) -> Artifact:
    """The artifacts a batched verification leaves behind, in the order it leaves them.

    Leads go out twelve at a time; each batch answers with its own packet, and
    the corpus that merges them is appended last and supersedes them. All of it
    is listed on the draft the researcher is shown.
    """
    facets = (
        "supporting",
        "contradictory",
        "negative_null",
        "replication",
        "methods",
        "safety_governance",
        "corrections_retractions",
        "supporting",
    )
    merged = EvidencePacket(
        question=QUESTION,
        sources=[
            SourceRecord(
                id=f"src_{index}",
                url=f"https://doi.org/10.1000/{index}",
                title=f"Paper {index}",
                source_type="primary_study",
                verification_status="verified",
                facet=facet,
            )
            for index, facet in enumerate(facets, start=1)
        ],
        claims=[
            EvidenceClaim(
                id="claim_2",
                claim="Thick coatings showed no benefit.",
                source_id="src_2",
                relation="contradicts",
                verification_status="verified",
            )
        ],
    )
    # The first batch of a real run is a twelfth of the corpus, and nothing says
    # the twelve it happens to hold were reachable that day.
    thin = EvidencePacket(question=QUESTION, sources=[], claims=[])
    manifest = DiscoveryManifest(
        question=QUESTION,
        discovery_angles=["supporting", "contradictory"],
        source_leads=[
            SourceLead(canonical_url=source.url, title=source.title)
            for source in merged.sources
        ],
    )

    def _artifact(schema: str, payload, status: ArtifactStatus) -> Artifact:
        item = Artifact(
            stage="evidence",
            agent="source_verification",
            artifact_type="specialist_output",
            content="",
            schema_name=schema,
            payload=payload.model_dump(mode="json"),
            status=status,
        )
        flow.session.artifacts.append(item)
        return item

    discovery = _artifact("DiscoveryManifest", manifest, ArtifactStatus.DRAFT)
    batch = _artifact("EvidencePacket", thin, ArtifactStatus.SUPERSEDED)
    corpus = _artifact("EvidencePacket", merged, ArtifactStatus.DRAFT)
    draft = Artifact(
        stage="evidence",
        agent="supervisor",
        content="Discovery and verification.",
        input_artifact_ids=[discovery.id, batch.id, corpus.id],
    )
    flow.session.artifacts.append(draft)
    return draft


def test_the_gate_measures_the_merged_corpus_and_not_the_first_batch():
    """A gate that reads a twelfth of the corpus can only ever refuse.

    Verification is batched, and the draft lists every batch's packet ahead of
    the merge of them. Taking the first one listed meant a live run stopped on
    "0 of 8 weighted verified sources ... 0 of 4 required evidence facets" while
    the evidence page beside it, reading the merge, said "Twenty-six usable
    sources across seven facets -- the evidence floor is met". One corpus, two
    counts, and no way past the gate but the exploratory escape hatch.
    """
    flow = _at_evidence(
        approval_profile=ApprovalProfile.MILESTONE, evidence_review=True
    )
    draft = _batched_evidence_draft(flow)

    flow.accept(draft, actor="test_researcher")

    assert flow.stage == "generate"
    assert flow.session.status == "active"


def test_a_session_saved_before_the_gate_existed_loads_as_a_run_without_it():
    saved = Session(question="Can a coating help?").to_dict()
    del saved["evidence_review"]

    assert Session.from_dict(saved).evidence_review is False


# ---------------------------------------------------------------------------
# The two surfaces that set it
# ---------------------------------------------------------------------------


def _created(store, **body) -> dict:
    return create_research_session(
        CreateResearchSession(question=QUESTION, **body), BackgroundTasks()
    )


@pytest.fixture()
def api(tmp_path, monkeypatch):
    store = ResearchLedger(tmp_path / "research.db")
    monkeypatch.setattr(research_api, "_ledger", lambda: store)
    monkeypatch.setattr(research_api, "evidence_tasks_configured", lambda: False)
    return store


def test_an_api_caller_that_does_not_ask_for_the_gate_does_not_get_one(api):
    """A script polling this API to completion would park at evidence forever."""
    snapshot = _created(api)

    assert snapshot["evidence_review"] is False


def test_the_launcher_gets_the_gate_it_asked_for_back_in_the_snapshot(api):
    snapshot = _created(api, evidence_review=True)

    assert snapshot["evidence_review"] is True


def test_the_snapshot_reports_the_gate_the_run_has_rather_than_the_one_requested(api):
    """Auto drops it, and the launcher shows what the run will actually do."""
    snapshot = _created(api, approval_profile="auto", evidence_review=True)

    assert snapshot["evidence_review"] is False


def test_the_cli_refuses_the_gate_flag_on_a_resumed_run(tmp_path):
    """The gate is configured with the run, like the model and the language: a
    session parked mid-pipeline has already passed the stage it would stop at."""
    path = tmp_path / "session.json"
    CoScientistWorkflow(QUESTION).save(path)

    with pytest.raises(SystemExit, match="configure a new run"):
        cli_main(["run", "--resume", str(path), "--evidence-review"])
