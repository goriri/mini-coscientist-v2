"""Where a live run lost its corpus between discovery and the panel.

Eight paid Deep Research passes returned ninety citable papers -- real DOIs on
ACS, Frontiers, MDPI -- and the evidence stage reported "Coverage: 0%", told the
reader that every one of the seven facet passes "returned no citable source",
and handed the gate a corpus it measured as one usable source. Nothing was
broken in the sense of raising; each stage quietly dropped material the stage
before it had found.

Four separate leaks, tested here in the order they happen: the provider's
citations not being recognised as citations, coverage being scored on prose that
therefore had none, one specialist being asked to enumerate ninety sources in a
single answer, and -- the widest of them -- the search's own findings never
being written down at all, so verification was handed a list of addresses and
the panel got ten claims out of ninety-two papers.
"""

from __future__ import annotations

import json
import re

import pytest

from coscientist.agents import DeterministicProvider
from coscientist.evidence import (
    _fallback_narrative,
    audit_coverage,
    discovered_corpus,
    normalize_report,
)
from coscientist.models import (
    MAX_VERIFICATION_BATCHES,
    VERIFICATION_BATCH_SIZE,
    ArtifactStatus,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EvidencePacket,
    SourceLead,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Does a protective coating improve rechargeable battery cycle life?"
ACS = "https://pubs.acs.org/doi/10.1021/acsami.4c13335"
FRONTIERS = "https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2019.00267/full"


# ---------------------------------------------------------------------------
# A citation the report did not spell out is still a citation
# ---------------------------------------------------------------------------


def test_a_statement_citing_the_provider_list_is_not_discarded_as_invented():
    """The live failure. The guard that stops a normalizer inventing a URL was
    reading the report text as the whole of what the provider said, so a report
    that cites by number lost every statement it had."""
    normalized = {
        "question": QUESTION,
        "statements": [
            {
                "text": "Alumina coatings halve first-cycle loss [1].",
                "facet": "supporting",
                "source_urls": [ACS],
                "originating_pass": 3,
            }
        ],
    }

    narrative = normalize_report(
        question=QUESTION,
        report="Alumina coatings halve first-cycle loss [1].",
        pass_number=3,
        normalizer=lambda _: json.dumps(normalized),
        citation_urls=[ACS],
    )

    assert [statement.source_urls for statement in narrative.statements] == [[ACS]]


def test_a_url_in_neither_the_report_nor_the_citation_list_is_still_refused():
    """Widening the allow-list must not turn it off."""
    normalized = {
        "question": QUESTION,
        "statements": [
            {
                "text": "Invented.",
                "facet": "supporting",
                "source_urls": ["https://example.org/never-cited"],
                "originating_pass": 1,
            }
        ],
    }

    narrative = normalize_report(
        question=QUESTION,
        report="A report citing [1].",
        pass_number=1,
        normalizer=lambda _: json.dumps(normalized),
        citation_urls=[ACS],
    )

    assert narrative.statements == []


def test_the_normalizer_is_shown_the_numbered_list_it_is_told_to_resolve():
    """ "Copy the URL verbatim" is unfollowable when the prose holds no URL."""
    seen: list[str] = []

    def _capture(prompt: str) -> str:
        seen.append(prompt)
        return "{}"

    normalize_report(
        question=QUESTION,
        report="Coatings help [1]. Thick ones do not [2].",
        pass_number=1,
        normalizer=_capture,
        citation_urls=[ACS, FRONTIERS],
    )

    assert f"[1] {ACS}" in seen[0]
    assert f"[2] {FRONTIERS}" in seen[0]


def test_the_fallback_resolves_a_citation_marker_against_the_provider_list():
    narrative = _fallback_narrative(
        QUESTION,
        "Coated cells lasted longer [1].\n\nThick coatings did not [2, 1].",
        4,
        [ACS, FRONTIERS],
    )

    assert [statement.source_urls for statement in narrative.statements] == [
        [ACS],
        [FRONTIERS, ACS],
    ]


def test_the_fallback_ignores_a_marker_that_names_no_citation():
    """An out-of-range marker is a numbering the report got wrong, not a source."""
    narrative = _fallback_narrative(QUESTION, "Coatings help [9].", 1, [ACS])

    assert narrative.statements == []


# ---------------------------------------------------------------------------
# Coverage is what a pass returned, not how the report chose to cite
# ---------------------------------------------------------------------------


def _narrative(*statements: DiscoveryStatement) -> DiscoveryNarrative:
    return DiscoveryNarrative(
        question=QUESTION, research_directions=[QUESTION], statements=list(statements)
    )


def test_a_facet_whose_pass_returned_a_lead_is_covered_with_no_prose_at_all():
    """What the panel got wrong: seven passes returned dozens of papers each and
    all seven were reported as having returned no citable source."""
    leads = [
        SourceLead(canonical_url=ACS, title="Alumina", facets=["supporting"]),
        SourceLead(canonical_url=FRONTIERS, title="Thick", facets=["contradictory"]),
    ]

    coverage = audit_coverage(
        _narrative(),
        leads,
        searched_facets={"supporting", "contradictory", "replication"},
    )

    assert coverage.facet_scores["supporting"] == 1.0
    assert coverage.facet_scores["contradictory"] == 1.0
    assert coverage.facet_scores["replication"] == 0.0
    assert [gap.facet for gap in coverage.gaps if gap.facet == "supporting"] == []


def test_a_facet_whose_pass_returned_nothing_still_scores_zero():
    """The property that stops a fan-out certifying itself: seven passes go out
    tagged with seven facets and all seven come back tagged, so a tag alone can
    never be coverage. Only a citable source the pass returned counts."""
    coverage = audit_coverage(
        _narrative(),
        [SourceLead(canonical_url=ACS, title="Alumina", facets=["supporting"])],
        searched_facets=set(dict.fromkeys(["supporting", "negative_null"])),
    )

    assert coverage.facet_scores["negative_null"] == 0.0
    assert any(gap.facet == "negative_null" for gap in coverage.gaps)


def test_a_lead_tagged_with_something_that_is_not_a_facet_earns_nothing():
    coverage = audit_coverage(
        _narrative(),
        [SourceLead(canonical_url=ACS, title="Alumina", facets=["interesting"])],
        searched_facets={"supporting"},
    )

    assert coverage.facet_scores["supporting"] == 0.0


# ---------------------------------------------------------------------------
# One specialist, one list it can finish
# ---------------------------------------------------------------------------


def _manifest(count: int, **overrides) -> DiscoveryManifest:
    return DiscoveryManifest(
        question=QUESTION,
        source_leads=[
            SourceLead(
                canonical_url=f"https://doi.org/10.1000/{index}",
                title=f"Paper {index}",
                **overrides,
            )
            for index in range(count)
        ],
    )


def test_ninety_leads_become_batches_a_specialist_can_enumerate():
    """The live failure: shown ninety leads in one dispatch, the specialist
    returned five sources and said nothing about the other eighty-five."""
    manifest = _manifest(90)

    batches = CoScientistWorkflow._verification_batches(manifest)

    assert all(len(batch) <= VERIFICATION_BATCH_SIZE for batch in batches)
    assert sum(len(batch) for batch in batches) == 90
    assert manifest.leads_sent_to_verification == 90
    assert manifest.leads_beyond_verification_ceiling == 0


def test_a_corpus_past_the_ceiling_records_what_it_left_behind():
    """A cap that reports nothing reads as complete coverage."""
    over = VERIFICATION_BATCH_SIZE * MAX_VERIFICATION_BATCHES + 7
    manifest = _manifest(over)

    batches = CoScientistWorkflow._verification_batches(manifest)

    assert len(batches) == MAX_VERIFICATION_BATCHES
    assert manifest.leads_beyond_verification_ceiling == 7
    assert manifest.leads_sent_to_verification == over - 7


def test_a_lead_carrying_an_identifier_outranks_one_that_carries_none():
    """What the ceiling drops should be the weakest material, not the tail."""
    manifest = DiscoveryManifest(
        question=QUESTION,
        source_leads=[
            SourceLead(canonical_url="https://blog.example.org/post", title="Blog"),
            SourceLead(
                canonical_url=ACS,
                title="Alumina",
                identifiers={"doi": "10.1021/acsami.4c13335"},
            ),
        ],
    )

    batches = CoScientistWorkflow._verification_batches(manifest)

    assert [lead.canonical_url for lead in batches[0]] == [
        ACS,
        "https://blog.example.org/post",
    ]


def test_a_bare_domain_is_never_given_to_a_verifier_as_a_document():
    manifest = DiscoveryManifest(
        question=QUESTION,
        source_leads=[
            SourceLead(canonical_url="https://www.mdpi.com", title="MDPI"),
            SourceLead(canonical_url=ACS, title="Alumina"),
        ],
    )

    batches = CoScientistWorkflow._verification_batches(manifest)

    assert [lead.canonical_url for lead in batches[0]] == [ACS]


def test_a_manifest_with_no_locators_falls_back_to_the_corpus_packet():
    """Discovery without URLs still has claims worth checking, and an empty
    work-list must not mean an empty dispatch."""
    manifest = DiscoveryManifest(question=QUESTION)

    assert CoScientistWorkflow._verification_batches(manifest) == [[]]


def test_each_batch_names_its_sources_and_forbids_shortening_them():
    """Three of the five sources a live run returned named a bare domain the
    model had shortened for itself."""
    batch = [SourceLead(canonical_url=ACS, title="Alumina interphases")]

    feedback = CoScientistWorkflow._verification_feedback("Original.", batch, 2, 8)

    assert f"- {ACS} -- Alumina interphases" in feedback
    assert "batch 2 of 8" in feedback
    assert "do not shorten a URL to its domain" in feedback
    assert "including the ones you could not reach" in feedback
    assert feedback.endswith("Original.")


def test_an_empty_batch_passes_the_original_feedback_through_untouched():
    assert CoScientistWorkflow._verification_feedback("Original.", [], 1, 1) == (
        "Original."
    )


# ---------------------------------------------------------------------------
# What the search found is what the verifier is asked about
# ---------------------------------------------------------------------------


def _statement(text: str, urls: list[str], **overrides) -> DiscoveryStatement:
    return DiscoveryStatement(
        text=text,
        facet=overrides.pop("facet", "supporting"),
        source_urls=urls,
        originating_pass=overrides.pop("originating_pass", 1),
        **overrides,
    )


def _searched(*statements: DiscoveryStatement, leads: list[str] | None = None):
    urls = leads if leads is not None else [ACS, FRONTIERS]
    return DiscoveryManifest(
        question=QUESTION,
        source_leads=[
            SourceLead(canonical_url=url, title=f"Paper on {url.split('/')[2]}")
            for url in urls
        ],
        narratives=[DiscoveryNarrative(question=QUESTION, statements=list(statements))],
    )


def test_what_a_search_pass_reported_becomes_a_claim_against_the_paper_it_named():
    """The live failure. Deep Research writes a manifest, the grounded path
    writes a packet, and nothing turned one into the other -- so the verifier
    was handed titles and URLs and the panel got ten findings out of
    ninety-two sources."""
    manifest = _searched(
        _statement("Alumina halves first-cycle loss.", [ACS]),
        _statement("Cycling above 4.4 V undoes the gain.", [FRONTIERS]),
    )

    corpus = discovered_corpus(QUESTION, manifest)

    by_id = {source.id: source.url for source in corpus.sources}
    assert [(claim.claim, by_id[claim.source_id]) for claim in corpus.claims] == [
        ("Alumina halves first-cycle loss.", ACS),
        ("Cycling above 4.4 V undoes the gain.", FRONTIERS),
    ]
    # Discovery says where a finding came from; the status is verification's.
    assert {claim.verification_status for claim in corpus.claims} == {
        "discovered_unverified"
    }
    assert {source.verification_status for source in corpus.sources} == {
        "discovered_unverified"
    }
    assert [source.supports_claim_ids for source in corpus.sources] == [
        [corpus.claims[0].id],
        [corpus.claims[1].id],
    ]


def test_a_finding_two_passes_both_reported_is_one_claim_not_two():
    """Seven passes cover overlapping ground and report the same result in the
    same words. The merge keys claims on their text, so duplicates that get this
    far are dropped there anyway -- having taken up room in the batches on the
    way."""
    manifest = _searched(
        _statement("Alumina halves first-cycle loss.", [ACS]),
        _statement(
            "Alumina  halves first-cycle loss.", [FRONTIERS], originating_pass=2
        ),
    )

    corpus = discovered_corpus(QUESTION, manifest)

    assert [claim.claim for claim in corpus.claims] == [
        "Alumina halves first-cycle loss."
    ]
    # The paper the surviving claim did not name stays citable in its own right.
    assert len(corpus.sources) == 2


def test_a_finding_whose_papers_were_all_dropped_is_not_carried_unsourced():
    """A claim with no source is the shape of the thing the badge complains
    about, and the retention ceiling cuts leads out from under statements."""
    manifest = _searched(
        _statement("Coatings help.", ["https://example.org/cut-by-the-ceiling"]),
        leads=[ACS],
    )

    assert discovered_corpus(QUESTION, manifest).claims == []


def test_the_search_corpus_is_recorded_as_the_artifact_the_stage_carries():
    """It has to be an artifact, not a local: the gate reads what the run wrote
    down, and a corpus the verifier saw but nothing recorded is a corpus the
    reader cannot check the verification against."""
    flow = CoScientistWorkflow(QUESTION, DeterministicProvider())
    manifest = _searched(_statement("Alumina halves first-cycle loss.", [ACS]))

    artifact = flow._record_discovered_corpus(manifest, feedback="Original.")

    assert artifact is not None
    assert artifact in flow.session.artifacts
    assert (artifact.stage, artifact.agent) == ("evidence", "evidence_discovery")
    packet = EvidencePacket.model_validate(artifact.payload)
    assert [claim.claim for claim in packet.claims] == [
        "Alumina halves first-cycle loss."
    ]


def test_a_grounded_manifest_carries_no_narrative_and_writes_no_second_corpus():
    """The grounded specialist returns the packet itself. A second corpus
    written over the top of it supersedes the one the run already has, and that
    one holds the findings its specialist wrote."""
    flow = CoScientistWorkflow(QUESTION, DeterministicProvider())
    manifest = DiscoveryManifest(
        question=QUESTION, source_leads=[SourceLead(canonical_url=ACS, title="Alumina")]
    )

    assert flow._record_discovered_corpus(manifest, feedback="Original.") is None
    assert flow.session.artifacts == []


def test_the_verifier_is_shown_what_the_search_says_each_paper_shows():
    """Handed addresses, a verifier returns addresses. The finding has to
    travel with the URL for the check to be a check of anything."""
    corpus = discovered_corpus(
        QUESTION, _searched(_statement("Alumina halves first-cycle loss.", [ACS]))
    )
    batch = [SourceLead(canonical_url=ACS, title="Alumina interphases")]

    feedback = CoScientistWorkflow._verification_feedback(
        "Original.", batch, 1, 1, corpus
    )

    assert (
        f"- {ACS} -- Alumina interphases\n"
        "    - What the search says it shows: Alumina halves first-cycle loss."
    ) in feedback
    assert "carry into your packet with a status of its own" in feedback


def test_a_batch_with_nothing_stated_about_it_asks_the_question_it_always_did():
    """Grounded runs and revisions reach here with no manifest findings, and an
    instruction about a list that is not printed is an instruction to invent
    one."""
    batch = [SourceLead(canonical_url=ACS, title="Alumina interphases")]

    feedback = CoScientistWorkflow._verification_feedback("Original.", batch, 1, 1)

    assert "What the search says it shows" not in feedback
    assert "carry into your packet" not in feedback
    assert f"- {ACS} -- Alumina interphases" in feedback


# ---------------------------------------------------------------------------
# The batches are one verification of one corpus
# ---------------------------------------------------------------------------

WIDE = 20
_LISTED_URL_RE = re.compile(r"^- (https://\S+)", re.M)


class _BatchAwareProvider(DeterministicProvider):
    """Discovers a wide corpus; verifies exactly the batch it was handed.

    A verifier that answers the same way whatever it is asked cannot show
    whether batching reached the whole corpus, which is the only thing these
    tests are about.
    """

    def __init__(self) -> None:
        self.verifier_prompts: list[str] = []

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "evidence_discovery":
            return json.dumps(
                {
                    "question": QUESTION,
                    "sources": [
                        {
                            "id": f"src_{index}",
                            "url": f"https://doi.org/10.1000/paper-{index}",
                            "title": f"Paper {index}",
                            "source_type": "primary_study",
                            "verification_status": "discovered_unverified",
                        }
                        for index in range(WIDE)
                    ],
                    "claims": [],
                }
            )
        if role == "source_verification":
            self.verifier_prompts.append(prompt)
            listed = _LISTED_URL_RE.findall(prompt)
            return json.dumps(
                {
                    "question": QUESTION,
                    "sources": [
                        {
                            "id": f"ver_{url.rsplit('-', 1)[-1]}",
                            "url": url,
                            "title": f"Paper {url.rsplit('-', 1)[-1]}",
                            "verification_status": "metadata_verified",
                            "verification_note": "crossref confirms this record.",
                        }
                        for url in listed
                    ],
                    "claims": [],
                }
            )
        return super().complete(role=role, prompt=prompt)


@pytest.fixture
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "off")


def _wide_run(_offline) -> tuple[CoScientistWorkflow, _BatchAwareProvider]:
    provider = _BatchAwareProvider()
    flow = CoScientistWorkflow(QUESTION, provider)
    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()
    return flow, provider


def test_a_wide_corpus_is_split_across_dispatches_that_each_name_their_sources(
    _offline,
):
    _, provider = _wide_run(_offline)

    assert len(provider.verifier_prompts) == 2
    listed = [_LISTED_URL_RE.findall(prompt) for prompt in provider.verifier_prompts]
    assert [len(batch) for batch in listed] == [
        VERIFICATION_BATCH_SIZE,
        WIDE - VERIFICATION_BATCH_SIZE,
    ]
    # Every discovered source reaches a verifier, and none is asked about twice.
    flattened = [url for batch in listed for url in batch]
    assert len(set(flattened)) == WIDE


def test_the_batches_are_folded_into_the_one_packet_everything_downstream_reads(
    _offline,
):
    """Left as separate artifacts, the gate, the panel and the list of citable
    ids would each see whichever batch happened to finish last."""
    flow, _ = _wide_run(_offline)

    newest = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.schema_name == "EvidencePacket" and item.payload
    )
    packet = EvidencePacket.model_validate(newest.payload)

    assert newest.agent == "source_verification"
    assert len(packet.sources) == WIDE
    assert {source.verification_status for source in packet.sources} == {
        "metadata_verified"
    }


def test_the_per_batch_artifacts_are_superseded_rather_than_deleted(_offline):
    """They are the record of which batch checked what."""
    flow, _ = _wide_run(_offline)

    batch_artifacts = [
        item
        for item in flow.session.artifacts
        if item.agent == "source_verification"
        and item.status == ArtifactStatus.SUPERSEDED
    ]

    assert len(batch_artifacts) == 2


def test_the_stage_artifact_carries_the_corpus_once_and_no_fetched_page_text(
    _offline,
):
    """A live run ended its answer by pasting a university site's navigation
    menu, and it landed verbatim in the artifact a researcher reads."""
    flow, _ = _wide_run(_offline)

    draft = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.stage == "evidence" and item.agent == "supervisor"
    )

    assert draft.content.count("### Source Verification") == 1
    assert "Skip to main content" not in draft.content
    assert "10.1000/paper-19" in draft.content


def test_the_summary_reports_the_real_pass_ceiling(_offline):
    """It said "(limit 3)" on a run that had made eight passes."""
    flow, _ = _wide_run(_offline)

    draft = next(
        item
        for item in reversed(flow.session.artifacts)
        if item.stage == "evidence" and item.agent == "supervisor"
    )

    assert "(limit 3)" not in draft.content
    assert f"- Sent to verification: {WIDE} " in draft.content
