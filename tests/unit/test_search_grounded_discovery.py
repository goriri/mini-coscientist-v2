"""Evidence discovery when Deep Research is switched off.

Deep Research is opt-in because it is billable and uncancellable, so the common
case is that it never runs. That used to empty the whole evidence stage: no
leads, nothing to verify, and every downstream hypothesis honestly citing
nothing. The grounded-search specialist covers that case, and these tests hold
it to the difference that matters -- material discovered by search is discovered,
never verified, and is labelled with the provider that found it.
"""

from __future__ import annotations

import json

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.evidence import discovery_angles
from coscientist.models import (
    EVIDENCE_FACETS,
    Artifact,
    ArtifactStatus,
    DeepResearchRun,
    DiscoveryManifest,
    EvidencePacket,
    ResearchPlan,
    SourceLead,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective interphase coating extend lithium-ion cycle life?"

DISCOVERED = {
    "question": QUESTION,
    "sources": [
        {
            "id": "src_alumina",
            "url": "https://pubmed.ncbi.nlm.nih.gov/28001/",
            "title": "Atomic layer deposition of alumina on silicon anodes",
            "source_type": "primary_study",
            "verification_status": "verified",
            "supports_claim_ids": ["claim_alumina"],
        },
        {
            "id": "src_null",
            "url": "https://doi.org/10.1000/null-result",
            "title": "No cycle-life gain from thick coatings",
            "source_type": "primary_study",
            "verification_status": "discovered_unverified",
        },
    ],
    "claims": [
        {
            "id": "claim_alumina",
            "claim": "A 2 nm alumina layer halves first-cycle capacity loss.",
            "source_id": "src_alumina",
            "exact_location": "Figure 3",
            "relation": "supports",
            "verification_status": "verified",
            "confidence": 0.7,
        },
        {
            "id": "claim_null",
            "claim": "Coatings above 20 nm showed no measurable benefit.",
            "source_id": "src_null",
            "exact_location": "Table 2",
            "relation": "contradicts",
            "verification_status": "discovered_unverified",
            "confidence": 0.5,
        },
    ],
    "limitations": ["Search results were read; no source was opened."],
}


class SearchingProvider(DeterministicProvider):
    """Offline everywhere except discovery, which answers like a live searcher."""

    def __init__(self, payload: dict | str = DISCOVERED):
        self.payload = payload
        self.discovery_prompts: list[str] = []
        self.verifier_prompts: list[str] = []

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "source_verification":
            self.verifier_prompts.append(prompt)
        if role == "evidence_discovery":
            self.discovery_prompts.append(prompt)
            if isinstance(self.payload, str):
                return self.payload
            return json.dumps(self.payload)
        return super().complete(role=role, prompt=prompt)


@pytest.fixture(autouse=True)
def _deep_research_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "off")


def _at_evidence(provider: DeterministicProvider) -> CoScientistWorkflow:
    flow = CoScientistWorkflow(QUESTION, provider)
    flow.accept(flow.preview(), actor="test_researcher")
    assert flow.stage == "evidence"
    return flow


def _manifest(flow: CoScientistWorkflow) -> DiscoveryManifest:
    artifact = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.schema_name == "DiscoveryManifest"
    )
    return DiscoveryManifest.model_validate(artifact.payload)


def test_search_discovers_leads_when_deep_research_is_off():
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    manifest = _manifest(flow)
    assert [lead.canonical_url for lead in manifest.source_leads] == [
        "https://pubmed.ncbi.nlm.nih.gov/28001/",
        "https://doi.org/10.1000/null-result",
    ]
    assert manifest.convergence_reason == "search_grounded_fallback"


def test_every_search_lead_names_the_provider_that_found_it():
    """A search hit and a Deep Research finding must not read alike later."""
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    manifest = _manifest(flow)
    assert {lead.provider for lead in manifest.source_leads} == {"google_search"}
    assert all(
        lead.verification_status == "discovered_unverified"
        for lead in manifest.source_leads
    )


def test_discovery_cannot_verify_its_own_finds():
    """The packet claimed one source and one claim were verified. They are not."""
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    packet = next(
        EvidencePacket.model_validate(item.payload)
        for item in flow.session.artifacts
        if item.agent == "evidence_discovery" and item.payload
    )
    assert {source.verification_status for source in packet.sources} == {
        "discovered_unverified"
    }
    assert {claim.verification_status for claim in packet.claims} == {
        "discovered_unverified"
    }


def test_the_manifest_records_why_deep_research_did_not_run():
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    manifest = _manifest(flow)
    assert [run.status for run in manifest.runs] == ["failed"]
    assert "opt-in" in manifest.runs[0].error
    assert manifest.estimated_cost_usd == 0.0
    assert manifest.stored_interaction_notice is False


def test_the_stage_summary_does_not_claim_a_stored_interaction():
    """Nothing was stored on Gemini, so the report must not say something was."""
    flow = _at_evidence(SearchingProvider())
    draft = flow.preview()

    assert "Stored interaction notice" not in draft.content
    assert "Discovery provider: google_search" in draft.content
    assert "0 completed of 1 attempted" in draft.content


def test_discovery_that_finds_nothing_still_reports_the_outage():
    flow = _at_evidence(SearchingProvider("no sources were located"))
    flow.preview()

    manifest = _manifest(flow)
    assert manifest.source_leads == []
    assert manifest.convergence_reason == "deep_research_unavailable"


def test_verification_runs_against_the_search_leads():
    """Discovery is worth nothing if the verifier never sees it."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    draft = flow.preview()

    assert len(provider.verifier_prompts) == 1
    assert "https://pubmed.ncbi.nlm.nih.gov/28001/" in provider.verifier_prompts[0], (
        "the verifier was asked to verify without being shown what was found"
    )
    assert "Source Verification" in draft.content


def test_discovered_claims_survive_an_unusable_verification_pass():
    """The offline verifier returns prose. Discovery must not be deleted by it."""
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    final = next(
        EvidencePacket.model_validate(item.payload)
        for item in reversed(flow.session.artifacts)
        if item.agent == "source_verification" and item.payload
    )
    assert {claim.id for claim in final.claims} == {"claim_alumina", "claim_null"}
    assert {claim.verification_status for claim in final.claims} == {"inaccessible"}
    # The claims name documents, so those documents have to come across as well.
    assert {"src_alumina", "src_null"} <= {source.id for source in final.sources}


def test_a_paid_deep_research_manifest_is_never_replaced_by_a_search_pass():
    """Re-previewing with the switch off must not delete discovery already bought."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    paid = DiscoveryManifest(
        question=QUESTION,
        runs=[DeepResearchRun(pass_number=1, status="completed")],
        source_leads=[
            SourceLead(
                canonical_url="https://pubmed.ncbi.nlm.nih.gov/1/",
                title="Bought and paid for",
            )
        ],
        convergence_reason="coverage_sufficient",
        estimated_cost_usd=3.0,
    )
    flow.session.artifacts.append(
        Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            artifact_type="specialist_output",
            content="earlier pass",
            schema_name="DiscoveryManifest",
            payload=paid.model_dump(mode="json"),
        )
    )

    flow.preview()

    manifest = _manifest(flow)
    assert manifest.estimated_cost_usd == 3.0
    assert [lead.provider for lead in manifest.source_leads] == ["deep_research"]
    assert provider.discovery_prompts == []


def test_an_accepted_manifest_is_not_mistaken_for_a_draft():
    """Only the draft of the stage being previewed may be reused."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    stale = Artifact(
        stage="evidence",
        agent="deep_research_discovery",
        artifact_type="specialist_output",
        content="accepted earlier",
        schema_name="DiscoveryManifest",
        payload=DiscoveryManifest(
            question=QUESTION,
            source_leads=[SourceLead(canonical_url="https://example.org/old")],
        ).model_dump(mode="json"),
    )
    stale.status = ArtifactStatus.ACCEPTED
    flow.session.artifacts.append(stale)

    flow.preview()

    assert provider.discovery_prompts, "a fresh search pass should have run"
    assert _manifest(flow).convergence_reason == "search_grounded_fallback"


def test_the_dispatch_tells_the_specialist_which_mode_it_is_in():
    """The same agent enriches gaps in one mode and searches broadly in the other."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    flow.preview()

    assert provider.discovery_prompts
    for prompt in provider.discovery_prompts:
        assert "PRIMARY DISCOVERY PASS" in prompt
        assert "Deep Research is unavailable" in prompt
        assert QUESTION in prompt


def test_the_pass_is_decomposed_into_one_search_per_angle():
    """One query asked for the mechanism, the studies for and against it,
    replications, negative results, retractions and the measurement standards at
    once, and a live run answered it with four sources, all of them supporting.
    Each of those is a search."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    flow.preview()

    plan = ResearchPlan.model_validate(
        next(
            item.payload
            for item in flow.session.artifacts
            if item.schema_name == "ResearchPlan" and item.payload
        )
    )
    angles = discovery_angles(plan)

    assert len(provider.discovery_prompts) == len(angles) > 1
    for angle, prompt in zip(angles, provider.discovery_prompts, strict=True):
        assert f"({angle.key})" in prompt
        assert angle.brief in prompt
    # Every facet coverage is scored on is asked for by name, so a facet that is
    # missing from the corpus is missing from the literature rather than from
    # the query.
    for facet in EVIDENCE_FACETS:
        assert any(f"({facet})" in prompt for prompt in provider.discovery_prompts)


def test_the_angles_are_folded_into_one_corpus_the_verifier_can_read():
    """Ten drafts with colliding identifiers are not a corpus. The verifier used
    to be handed the manifest instead, which is titles and URLs with every claim
    discovery found stripped out of it."""
    provider = SearchingProvider()
    flow = _at_evidence(provider)
    flow.preview()

    corpus = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.agent == "evidence_discovery"
        and item.schema_name == "EvidencePacket"
        and item.status != ArtifactStatus.SUPERSEDED
    )
    packet = EvidencePacket.model_validate(corpus.payload)

    # Every angle returned the same two sources here, so the corpus is two.
    assert [source.id for source in packet.sources] == ["src_alumina", "src_null"]
    assert [claim.id for claim in packet.claims] == ["claim_alumina", "claim_null"]
    # The angle drafts stay as the record of which search found what.
    superseded = [
        item
        for item in flow.session.artifacts
        if item.agent == "evidence_discovery"
        and item.status == ArtifactStatus.SUPERSEDED
    ]
    assert len(superseded) == len(provider.discovery_prompts)
    assert set(corpus.input_artifact_ids) == {item.id for item in superseded}
    assert "https://pubmed.ncbi.nlm.nih.gov/28001/" in provider.verifier_prompts[0]
    assert "A 2 nm alumina layer halves" in provider.verifier_prompts[0], (
        "the verifier was shown the sources and not what was claimed of them"
    )


def test_the_search_pass_is_recorded_as_a_task_with_its_artifact():
    """Provenance: the manifest has to say which artifact produced its leads."""
    flow = _at_evidence(SearchingProvider())
    flow.preview()

    discovery = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.schema_name == "DiscoveryManifest"
    )
    manifest = DiscoveryManifest.model_validate(discovery.payload)
    produced = {lead.raw_artifact_reference for lead in manifest.source_leads}
    # Every lead names the angle that found it, and the merged corpus is an input
    # of the manifest too, so the leads account for all but that one.
    assert produced <= set(discovery.input_artifact_ids)
    assert produced
    assert any(task.agent == "evidence_discovery" for task in flow.session.tasks)
    assert manifest.verification_handoff_source_ids == [
        lead.id for lead in manifest.source_leads
    ]
