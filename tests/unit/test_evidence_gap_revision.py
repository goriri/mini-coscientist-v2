"""Sending the evidence base back searches the gaps, and only the gaps.

A researcher who reads the corpus at the evidence gate and asks for more has
named a hole in something that already exists. The stage used to answer that by
running discovery again from nothing -- on a deployment with Deep Research, seven
fresh interactions, roughly twenty-one dollars and forty minutes that cannot be
cancelled, to re-find the papers already on the page.

These tests hold the revision to what was asked for: targeted searches with the
grounded specialist, merged into the corpus that is already there, with the money
path never touched and nothing already found thrown away.
"""

from __future__ import annotations

import json

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.models import (
    ArtifactStatus,
    DiscoveryCoverage,
    DiscoveryManifest,
    EvidencePacket,
    ResearchGap,
)
from coscientist.orchestration import MAX_GAP_SEARCHES, CoScientistWorkflow

QUESTION = "Can a protective interphase coating extend lithium-ion cycle life?"

FOUND = {
    "question": QUESTION,
    "sources": [
        {
            "id": "src_alumina",
            "url": "https://pubmed.ncbi.nlm.nih.gov/28001/",
            "title": "Atomic layer deposition of alumina on silicon anodes",
            "source_type": "primary_study",
            "supports_claim_ids": ["claim_alumina"],
        }
    ],
    "claims": [
        {
            "id": "claim_alumina",
            "claim": "A 2 nm alumina layer halves first-cycle capacity loss.",
            "source_id": "src_alumina",
            "exact_location": "Figure 3",
            "relation": "supports",
            "confidence": 0.7,
        }
    ],
    "limitations": ["Search results were read; no source was opened."],
}

FILLED = {
    "question": QUESTION,
    "sources": [
        {
            "id": "src_retracted",
            "url": "https://doi.org/10.1000/retraction-notice",
            "title": "Retraction: coating thickness and cycle life",
            "source_type": "primary_study",
            "supports_claim_ids": ["claim_retracted"],
        }
    ],
    "claims": [
        {
            "id": "claim_retracted",
            "claim": "The 2019 thickness study was retracted for image reuse.",
            "source_id": "src_retracted",
            "exact_location": "Notice",
            "relation": "contradicts",
            "confidence": 0.6,
        }
    ],
    "limitations": ["Search results were read; no source was opened."],
}


class SearchingProvider(DeterministicProvider):
    """Offline everywhere except discovery, which answers like a live searcher."""

    def __init__(self) -> None:
        self.discovery_prompts: list[str] = []
        self.payload: dict = FOUND

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "evidence_discovery":
            self.discovery_prompts.append(prompt)
            return json.dumps(self.payload)
        return super().complete(role=role, prompt=prompt)


class RefusingController:
    """A Deep Research controller that fails the test if anything reaches it."""

    transport = None

    def run(self, *args, **kwargs):  # pragma: no cover - the point is not to run
        raise AssertionError(
            "A revision at the evidence gate started a paid Deep Research wave."
        )


@pytest.fixture(autouse=True)
def _deep_research_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "off")


def _discovered(provider: SearchingProvider) -> CoScientistWorkflow:
    """A workflow whose evidence base has been searched once and drafted."""
    flow = CoScientistWorkflow(QUESTION, provider)
    flow.accept(flow.preview(), actor="test_researcher")
    assert flow.stage == "evidence"
    flow.preview()
    provider.discovery_prompts.clear()
    provider.payload = FILLED
    return flow


def _standing(flow: CoScientistWorkflow):
    return next(
        item
        for item in reversed(flow.session.artifacts)
        if item.schema_name == "DiscoveryManifest"
        and item.status == ArtifactStatus.DRAFT
    )


def _manifest(flow: CoScientistWorkflow) -> DiscoveryManifest:
    return DiscoveryManifest.model_validate(_standing(flow).payload)


def _corpus(flow: CoScientistWorkflow) -> EvidencePacket:
    artifact = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.agent == "evidence_discovery"
        and item.schema_name == "EvidencePacket"
        and item.status != ArtifactStatus.SUPERSEDED
    )
    return EvidencePacket.model_validate(artifact.payload)


def _with_gaps(flow: CoScientistWorkflow, count: int) -> None:
    """Give the standing manifest a coverage audit that names open gaps."""
    standing = _standing(flow)
    manifest = DiscoveryManifest.model_validate(standing.payload)
    manifest.coverage_history = [
        DiscoveryCoverage(
            facet_scores={facet: 1.0 for facet in ("supporting", "methods")},
            weighted_score=0.4,
            gaps=[
                ResearchGap(
                    direction="Evidence landscape",
                    facet="corrections_retractions",
                    description=f"Nothing was found on open question {index}.",
                    decision_impact="high" if index == 0 else "low",
                    priority=5 if index == 0 else 1,
                )
                for index in range(count)
            ],
        )
    ]
    standing.payload = manifest.model_dump(mode="json")


def test_a_revision_searches_the_gaps_instead_of_rerunning_discovery():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 2)

    flow.revise("The corpus has no retraction check at all.")

    assert provider.discovery_prompts, "the revision searched nothing"
    assert len(provider.discovery_prompts) <= MAX_GAP_SEARCHES
    for prompt in provider.discovery_prompts:
        assert "GAP-DIRECTED PASS" in prompt
        assert "PRIMARY DISCOVERY PASS" not in prompt


def test_the_revision_never_reaches_the_paid_provider():
    """The whole point: a second look at the corpus must not cost another wave."""
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 1)
    flow.evidence_discovery = RefusingController()

    flow.revise("Find the negative results.")

    manifest = _manifest(flow)
    assert [run.status for run in manifest.runs] == ["failed"], (
        "a gap search was filed as a Deep Research pass, which the panel bills"
    )


def test_what_the_researcher_asked_for_becomes_its_own_search():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 2)

    flow.revise("Nothing here covers solid-state cells.")

    asked = [
        prompt
        for prompt in provider.discovery_prompts
        if "Nothing here covers solid-state cells." in prompt
    ]
    assert len(asked) == 1, (
        "the researcher's own words were folded into every gap prompt, or into none"
    )


def test_the_gap_searches_are_told_what_is_already_in_the_corpus():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 1)

    flow.revise("Check the retractions.")

    for prompt in provider.discovery_prompts:
        assert "Already in the corpus" in prompt
        assert "Atomic layer deposition of alumina" in prompt


def test_the_revision_keeps_everything_the_first_pass_found():
    provider = SearchingProvider()
    flow = _discovered(provider)
    before = {lead.canonical_url for lead in _manifest(flow).source_leads}
    _with_gaps(flow, 1)

    flow.revise("Check the retractions.")

    after = {lead.canonical_url for lead in _manifest(flow).source_leads}
    assert before <= after, "a revision deleted sources the first pass had found"
    assert "https://doi.org/10.1000/retraction-notice" in after
    corpus = _corpus(flow)
    assert {source.url for source in corpus.sources} >= before | {
        "https://doi.org/10.1000/retraction-notice"
    }
    assert {claim.id for claim in corpus.claims} == {"claim_alumina", "claim_retracted"}


def test_only_one_packet_claims_to_be_the_corpus_after_a_revision():
    """Everything downstream reads the newest one, so two of them is a coin toss."""
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 1)

    flow.revise("Check the retractions.")

    live = [
        item
        for item in flow.session.artifacts
        if item.agent == "evidence_discovery"
        and item.schema_name == "EvidencePacket"
        and item.status != ArtifactStatus.SUPERSEDED
    ]
    assert len(live) == 1


def test_each_gap_search_is_recorded_as_a_resolved_enrichment_request():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 2)

    flow.revise("Check the retractions.")

    requests = _manifest(flow).enrichment_requests
    assert len(requests) == len(provider.discovery_prompts)
    assert {request.provider for request in requests} == {"google_search"}
    # Left queued, the residual-enrichment dispatch downstream would run every
    # one of them a second time, on this revision and on every later one.
    assert {request.status for request in requests} == {"completed"}
    assert all(request.result_artifact_reference for request in requests)
    assert not any(
        "Resolve only these residual searches" in prompt
        for prompt in provider.discovery_prompts
    )


def test_a_second_revision_does_not_re_run_the_first_ones_searches():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 1)
    flow.revise("Check the retractions.")
    first = len(provider.discovery_prompts)
    provider.discovery_prompts.clear()

    flow.revise("Now check the replications.")

    assert len(provider.discovery_prompts) <= first, (
        "each revision inherited the previous revision's searches and re-ran them"
    )
    assert any(
        "Now check the replications." in prompt for prompt in provider.discovery_prompts
    )


def test_coverage_is_re_scored_against_the_whole_corpus_not_just_the_new_part():
    """Direction scores are computed from statements, and the manifest keeps
    none of them -- they are built at dispatch and dropped. Scored against the
    gap pass alone, a revision that added sources reported coverage falling."""
    provider = SearchingProvider()
    flow = _discovered(provider)
    before = _manifest(flow).coverage_history[-1]

    flow.revise("Check the retractions.")

    history = _manifest(flow).coverage_history
    assert len(history) == 2
    after = history[-1]
    assert after.weighted_score >= before.weighted_score
    assert set(after.direction_scores) == set(before.direction_scores)
    for direction, score in before.direction_scores.items():
        assert after.direction_scores[direction] >= score, (
            f"adding sources made {direction!r} look less covered than before"
        )
    # A facet the first wave searched and the revision did not must not fall
    # back to unsearched: the searching happened, and the corpus still holds
    # what it returned.
    for facet, score in before.facet_scores.items():
        assert after.facet_scores[facet] >= score


def test_a_capped_revision_says_what_it_left_unsearched():
    """No silent caps: nine gaps and six searches is arithmetic a reader needs."""
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 9)

    flow.revise("And the safety literature.")

    assert len(provider.discovery_prompts) == MAX_GAP_SEARCHES
    manifest = _manifest(flow)
    assert manifest.gap_searches_deferred == 4
    assert f"at {MAX_GAP_SEARCHES} searches" in manifest.synthesis_report
    # And on the page, not only in the record: the gap list shows the searched
    # and the unsearched side by side and cannot tell them apart.
    shown = _standing(flow).content
    assert "Gaps not searched: 4" in shown
    assert f"Gap-directed searches: {MAX_GAP_SEARCHES} run" in shown


def test_the_most_damaging_gaps_are_the_ones_that_get_searched():
    provider = SearchingProvider()
    flow = _discovered(provider)
    _with_gaps(flow, 9)

    flow.revise("And the safety literature.")

    # Gap 0 is the only "high" impact one; the rest are "low". Nine gaps do not
    # fit in six searches, so which six is a decision, and it has to fall on the
    # damaging ones.
    assert any("open question 0." in prompt for prompt in provider.discovery_prompts), (
        "the cap dropped the high-impact gap and searched the trivial ones"
    )
    assert not any(
        "open question 8." in prompt for prompt in provider.discovery_prompts
    )


def test_a_revision_against_a_corpus_with_no_named_gaps_still_searches():
    """Sending the corpus back has to do something, or the button is a lie.

    The first wave here covered every facet it was asked about, so the coverage
    audit named no gap at all. What the researcher said is then the whole
    work-list, and it is enough of one.
    """
    provider = SearchingProvider()
    flow = _discovered(provider)
    assert _manifest(flow).coverage_history[-1].gaps == []

    flow.revise("Nothing here is about solid-state cells.")

    assert len(provider.discovery_prompts) == 1
    assert "solid-state cells" in provider.discovery_prompts[0]


def _keyed_on_the_question(flow: CoScientistWorkflow) -> DiscoveryCoverage:
    """Score the standing corpus the way a Deep Research wave scores it.

    The two discovery paths key their research directions differently: a Deep
    Research narrative is keyed on the question and the grounded fallback on the
    scope's success criteria. Every test above runs the grounded path, where the
    keys happen to match whatever a revision computes -- which is why none of
    them caught a revision re-keying them.
    """
    standing = _standing(flow)
    manifest = DiscoveryManifest.model_validate(standing.payload)
    manifest.coverage_history = [
        DiscoveryCoverage(
            direction_scores={QUESTION: 1.0},
            facet_scores={facet: 1.0 for facet in ("supporting", "methods")},
            weighted_score=0.875,
        )
    ]
    standing.payload = manifest.model_dump(mode="json")
    return manifest.coverage_history[-1]


def test_a_revision_scores_the_directions_discovery_scored():
    """A live revision published 70% coverage under the 88% it started from.

    The corpus came from Deep Research, whose directions are keyed on the
    question; the revision re-scored against the success criteria instead. Keys
    the previous audit never named have no previous score to be held at, so the
    floor that exists to prevent exactly this passed straight through, and the
    panel told the researcher that asking for more evidence made the evidence
    base worse.
    """
    provider = SearchingProvider()
    flow = _discovered(provider)
    before = _keyed_on_the_question(flow)

    flow.revise("Add the long-term safety literature.")

    after = _manifest(flow).coverage_history[-1]
    assert set(after.direction_scores) == set(before.direction_scores)
    assert after.weighted_score >= before.weighted_score, (
        "the revision published a lower coverage than the corpus it grew"
    )


def test_the_researchers_own_search_counts_toward_coverage():
    """The one target with no facet is the researcher's own request.

    Statements were collected only from facet-tagged searches, so the common
    revision -- no gaps named, the request the only search -- contributed none,
    and every research direction scored zero for want of a statement to count.
    """
    provider = SearchingProvider()
    flow = _discovered(provider)
    standing = _standing(flow)
    manifest = DiscoveryManifest.model_validate(standing.payload)
    # Floored at zero, so nothing can be inherited from the previous audit and
    # the score can only come from the statements this revision counted.
    manifest.coverage_history = [
        DiscoveryCoverage(
            direction_scores={QUESTION: 0.0},
            facet_scores={facet: 0.0 for facet in ("supporting", "methods")},
            weighted_score=0.0,
        )
    ]
    standing.payload = manifest.model_dump(mode="json")
    # One direction, so the score is the statement count over two, and the
    # standing corpus alone is what it has to beat.
    standing_only = min(1.0, len(_corpus(flow).claims) / 2)

    flow.revise("Nothing here covers long-term safety. Search for that.")

    after = _manifest(flow).coverage_history[-1]
    assert len(provider.discovery_prompts) == 1
    assert after.direction_scores[QUESTION] > standing_only, (
        "the researcher's own search returned sources that counted for nothing"
    )
