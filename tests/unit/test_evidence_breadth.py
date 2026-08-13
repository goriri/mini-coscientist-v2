"""How much literature the evidence stage looks for, and how much of it survives.

Two live runs put four sources and six claims in front of the reader, and every
one of the six supported the question. Nothing had failed. One query asked for
the mechanism, the studies for and against it, replications, negative results,
retractions and the measurement standards all at once, and one competent answer
came back; then the verifier -- shown a list of titles and URLs with every claim
stripped out of it -- re-invented a handful of claims to check.

So there are two numbers here. What the search asks for, which is now one query
per facet rather than one for all of them, and what reaches the report, which is
everything the angles found rather than whatever one merge happened to keep. The
last group holds the countervailing rule: more is not better if it is not real,
so a locator that reaches no document cannot have been verified against one.
"""

from __future__ import annotations

import json

import pytest

from coscientist.agents import STRUCTURED_OUTPUT_INSTRUCTIONS, DeterministicProvider
from coscientist.dossier import compile_dossier
from coscientist.evidence import (
    ANGLE_SOURCE_TARGET,
    CORPUS_SOURCE_TARGET,
    MAX_DISCOVERY_ANGLES,
    discovery_angles,
    downgrade_unlocatable_sources,
    merge_evidence_packets,
    names_a_document,
    retain_leads,
)
from coscientist.models import (
    EVIDENCE_FACETS,
    EvidenceClaim,
    EvidencePacket,
    ResearchPlan,
    SourceLead,
    SourceRecord,
)
from coscientist.orchestration import CoScientistWorkflow

QUESTION = "Can a protective interphase coating extend lithium-ion cycle life?"


def _plan(criteria: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        question=QUESTION,
        research_mode="literature_synthesis",
        intended_claim="A coating extends cycle life.",
        success_criteria=["Cycle life at 500 cycles"] if criteria is None else criteria,
    )


def _packet(
    sources: list[tuple[str, str]] = (),
    claims: list[tuple[str, str, str]] = (),
    limitations: list[str] = (),
) -> EvidencePacket:
    """A packet from one angle, written the way an angle writes one."""
    return EvidencePacket(
        question=QUESTION,
        sources=[
            SourceRecord(id=id_, url=url, title="A coating study")
            for id_, url in sources
        ],
        claims=[
            EvidenceClaim(id=id_, claim=text, source_id=source_id)
            for id_, text, source_id in claims
        ],
        limitations=list(limitations),
    )


# ---------------------------------------------------------------------------
# What the search asks for
# ---------------------------------------------------------------------------


def test_every_facet_the_coverage_audit_scores_gets_a_search_of_its_own():
    """A facet missing from the corpus should be missing from the literature, not
    from the query set that went looking for it."""
    angles = discovery_angles(_plan())

    assert [angle.key for angle in angles][: len(EVIDENCE_FACETS)] == list(
        EVIDENCE_FACETS
    )
    assert all(angle.brief.strip() for angle in angles)


def test_the_plans_own_success_criteria_are_searched_after_the_facets():
    """The facets are what any question needs. The criteria are what this one has
    to be able to show, and no fixed list of facets knows them."""
    angles = discovery_angles(_plan(["Cycle life beyond 500 cycles", "Cost per kWh"]))
    criteria = [angle for angle in angles if angle.key.startswith("criterion_")]

    assert len(criteria) == 2
    assert "Cycle life beyond 500 cycles" in criteria[0].brief
    assert "Cost per kWh" in criteria[1].brief


def test_a_plan_with_many_criteria_does_not_fan_out_without_limit():
    """Each angle is a live grounded search and the bus runs four at a time, so
    the cap is the difference between three waves and twenty."""
    angles = discovery_angles(_plan([f"Criterion {index}" for index in range(20)]))

    assert len(angles) == MAX_DISCOVERY_ANGLES
    # The facets are never what gets dropped: they are the axes the coverage
    # audit scores, so losing one makes the audit unanswerable.
    assert set(EVIDENCE_FACETS) <= {angle.key for angle in angles}


def test_a_plan_with_no_criteria_still_searches_every_facet():
    assert len(discovery_angles(_plan([]))) == len(EVIDENCE_FACETS)


def test_the_corpus_size_is_asked_for_as_a_target_and_not_as_a_quota():
    """A contract that rejects a short packet does not produce a longer one. It
    produces an invented one, and every entry in that is something a reviewer has
    to check before finding out it was never there."""
    instruction = STRUCTURED_OUTPUT_INSTRUCTIONS["evidence_discovery"]

    assert f"at least {ANGLE_SOURCE_TARGET} distinct sources" in instruction
    assert f"{CORPUS_SOURCE_TARGET} across all its searches" in instruction
    assert "That is a target, not a quota" in instruction
    # The whole-pass target has to be reachable from the per-angle one, or the two
    # halves of the same sentence are asking for different things.
    assert ANGLE_SOURCE_TARGET * len(EVIDENCE_FACETS) >= CORPUS_SOURCE_TARGET


def test_the_verifier_is_told_it_decides_a_status_and_never_membership():
    """The live failure this sentence exists for: a verification pass that returned
    five sources against fifty-three leads, shrinking the literature the report
    showed to whatever one model chose to mention."""
    instruction = STRUCTURED_OUTPUT_INSTRUCTIONS["source_verification"]

    assert "decides a status, never membership" in instruction
    assert "omitting an entry deletes the record" in instruction
    # The status it may assign is capped by an actual retrieval, so the sentence
    # above only means something if the verifier is told to fetch first.
    assert "Call fetch_source_document on every source" in instruction


# ---------------------------------------------------------------------------
# What survives the merge
# ---------------------------------------------------------------------------


def test_two_angles_that_find_the_same_paper_cite_it_once():
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet([("src_1", "https://doi.org/10.1000/a")]),
            _packet([("src_1", "https://doi.org/10.1000/a?utm_source=scholar")]),
        ],
    )

    assert [source.url for source in merged.sources] == ["https://doi.org/10.1000/a"]


def test_two_angles_that_find_different_papers_do_not_collide_on_one_id():
    """Each packet is written independently, so ``src_1`` in one has nothing to do
    with ``src_1`` in the next, and a naive merge silently drops a paper."""
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet(
                [("src_1", "https://doi.org/10.1000/a")], [("c_1", "Alpha", "src_1")]
            ),
            _packet(
                [("src_1", "https://doi.org/10.1000/b")], [("c_1", "Beta", "src_1")]
            ),
        ],
    )

    assert [source.id for source in merged.sources] == ["src_1", "src_1_2"]
    assert [claim.id for claim in merged.claims] == ["c_1", "c_1_2"]
    # Each claim followed its own source across the rename.
    assert {claim.claim: claim.source_id for claim in merged.claims} == {
        "Alpha": "src_1",
        "Beta": "src_1_2",
    }
    assert {source.id: source.supports_claim_ids for source in merged.sources} == {
        "src_1": ["c_1"],
        "src_1_2": ["c_1_2"],
    }


def test_an_identifier_no_other_angle_wanted_is_left_exactly_as_it_was():
    """``src_alumina`` says what it is and ``src_003`` says where it landed in a
    merge, and these ids are what a reader follows from a hypothesis to a paper."""
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet(
                [("src_alumina", "https://doi.org/10.1000/a")],
                [("claim_alumina", "Alpha", "src_alumina")],
            )
        ],
    )

    assert [source.id for source in merged.sources] == ["src_alumina"]
    assert [claim.id for claim in merged.claims] == ["claim_alumina"]


def test_the_same_finding_reported_by_two_angles_is_one_claim():
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet(
                [("src_1", "https://doi.org/10.1000/a")], [("c_1", "Same", "src_1")]
            ),
            _packet(
                [("src_1", "https://doi.org/10.1000/a")], [("c_2", "  same  ", "src_1")]
            ),
        ],
    )

    assert [claim.claim for claim in merged.claims] == ["Same"]


def _tiered(url: str, status: str, claim: tuple[str, str] | None = None):
    """One packet's word on one document, at the tier it claims for it."""
    packet = _packet([("src_1", url)], [("c_1", claim[0], "src_1")] if claim else [])
    packet.sources[0].verification_status = status
    if claim:
        packet.claims[0].verification_status = claim[1]
    return packet


def test_the_batch_that_opened_the_paper_outranks_the_nine_that_did_not():
    """Every batch is shown the whole discovered corpus and carries forward what
    it was not asked about, so the same paper arrives checked from one batch and
    unreachable from the rest. First-wins made the status a lottery on which
    batch finished first: a paper retrieved and read in batch three was printed
    as not retrieved because batch one had said nothing about it."""
    merged = merge_evidence_packets(
        QUESTION,
        [
            _tiered("https://doi.org/10.1000/a", "inaccessible"),
            _tiered("https://doi.org/10.1000/a", "verified"),
        ],
    )

    assert [source.verification_status for source in merged.sources] == ["verified"]


def test_a_finding_one_batch_confirmed_is_not_overwritten_by_one_that_carried_it():
    merged = merge_evidence_packets(
        QUESTION,
        [
            _tiered("https://doi.org/10.1000/a", "verified", ("Alpha", "inaccessible")),
            _tiered("https://doi.org/10.1000/a", "verified", ("alpha", "verified")),
        ],
    )

    assert [claim.claim for claim in merged.claims] == ["Alpha"]
    assert merged.claims[0].verification_status == "verified"


def test_a_retraction_is_never_traded_away_for_a_higher_tier():
    """It is not a rung on that ladder. A withdrawn paper one batch happened to
    read is still withdrawn."""
    merged = merge_evidence_packets(
        QUESTION,
        [
            _tiered("https://doi.org/10.1000/a", "retracted"),
            _tiered("https://doi.org/10.1000/a", "verified"),
        ],
    )

    assert [source.verification_status for source in merged.sources] == ["retracted"]


def test_a_retraction_found_late_still_overrides_what_was_read_earlier():
    merged = merge_evidence_packets(
        QUESTION,
        [
            _tiered("https://doi.org/10.1000/a", "verified"),
            _tiered("https://doi.org/10.1000/a", "retracted"),
        ],
    )

    assert [source.verification_status for source in merged.sources] == ["retracted"]


def test_a_limitation_one_angle_recorded_is_neither_lost_nor_repeated():
    shared = "Search results were read; no source was opened."
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet(limitations=[shared, "No retraction notice was searched."]),
            _packet(limitations=[shared]),
        ],
    )

    assert merged.limitations == [shared, "No retraction notice was searched."]


def test_a_source_whose_locator_will_not_parse_is_kept_rather_than_dropped():
    """It is still a record of what the search saw, and deleting it hides the gap
    instead of reporting it."""
    merged = merge_evidence_packets(
        QUESTION,
        [
            _packet([("src_1", "not a url at all")]),
            _packet([("src_2", "also not a url")]),
        ],
    )

    assert [source.url for source in merged.sources] == [
        "not a url at all",
        "also not a url",
    ]


# ---------------------------------------------------------------------------
# What may not be called verified
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "reaches"),
    [
        ("https://pubmed.ncbi.nlm.nih.gov/28001/", True),
        ("https://doi.org/10.1000/abc", True),
        ("https://www.researchgate.net", False),
        ("https://www.researchgate.net/", False),
        ("https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC", False),
        ("researchgate.net/article/12", False),
        ("", False),
    ],
)
def test_a_locator_either_reaches_a_document_or_only_names_a_website(
    url: str, reaches: bool
):
    assert names_a_document(url) is reaches


def test_a_bare_domain_cannot_be_verified_however_the_packet_labels_it():
    """Verification means someone opened the document and found the claim where
    they said it was. Nobody has done that with a publisher's front page."""
    packet = _packet(
        [("src_bare", "https://www.researchgate.net")],
        [("claim_bare", "Coatings help.", "src_bare")],
    )
    packet.sources[0].verification_status = "verified"
    packet.claims[0].verification_status = "verified"

    checked = downgrade_unlocatable_sources(packet)

    assert checked.sources[0].verification_status == "inaccessible"
    # A claim is only as verified as the document it rests on.
    assert checked.claims[0].verification_status == "inaccessible"
    assert any(
        "named a website rather than a document" in item for item in checked.limitations
    )


def test_a_downgraded_source_stays_in_the_corpus():
    """Dropping it would shrink the literature the report shows below what the run
    actually saw, which is the failure this whole area exists to fix."""
    packet = _packet([("src_bare", "https://www.researchgate.net")])
    packet.sources[0].verification_status = "verified"

    checked = downgrade_unlocatable_sources(packet)

    assert [source.id for source in checked.sources] == ["src_bare"]


def test_a_packet_that_never_claimed_verification_is_returned_untouched():
    """The guard corrects an assertion. It has no opinion about a source honestly
    reported as unread."""
    packet = _packet([("src_bare", "https://www.researchgate.net")])

    assert downgrade_unlocatable_sources(packet) is packet


def test_a_locator_that_does_reach_a_document_survives_the_guard():
    packet = _packet([("src_ok", "https://doi.org/10.1000/abc")])
    packet.sources[0].verification_status = "verified"

    checked = downgrade_unlocatable_sources(packet)

    assert checked.sources[0].verification_status == "verified"
    assert checked.limitations == []


# ---------------------------------------------------------------------------
# Through the stage
# ---------------------------------------------------------------------------


DISCOVERED = {
    "question": QUESTION,
    "sources": [
        {
            "id": "src_real",
            "url": "https://pubmed.ncbi.nlm.nih.gov/28001/",
            "title": "Atomic layer deposition of alumina on silicon anodes",
            "source_type": "primary_study",
        }
    ],
    "claims": [
        {
            "id": "claim_real",
            "claim": "A 2 nm alumina layer halves first-cycle capacity loss.",
            "source_id": "src_real",
            "relation": "supports",
        }
    ],
}

VERIFIED_A_WEBSITE = {
    "question": QUESTION,
    "sources": [
        {
            "id": "src_real",
            "url": "https://pubmed.ncbi.nlm.nih.gov/28001/",
            "title": "Atomic layer deposition of alumina on silicon anodes",
            "source_type": "primary_study",
            "verification_status": "verified",
        },
        {
            "id": "src_bare",
            "url": "https://www.researchgate.net",
            "title": "Cycle-life benchmarking across coating thicknesses",
            "source_type": "primary_study",
            "verification_status": "verified",
        },
    ],
    "claims": [
        {
            "id": "claim_real",
            "claim": "A 2 nm alumina layer halves first-cycle capacity loss.",
            "source_id": "src_real",
            "exact_location": "Figure 3",
            "relation": "supports",
            "verification_status": "verified",
            "confidence": 0.8,
        },
        {
            "id": "claim_bare",
            "claim": "Thicker coatings improve cycle life across chemistries.",
            "source_id": "src_bare",
            "exact_location": "Table 1",
            "relation": "supports",
            "verification_status": "verified",
            "confidence": 0.6,
        },
    ],
}


class _Searching(DeterministicProvider):
    """Offline everywhere except the two evidence roles, which answer live."""

    def complete(self, *, role: str, prompt: str) -> str:
        if role == "evidence_discovery":
            return json.dumps(DISCOVERED)
        if role == "source_verification":
            return json.dumps(VERIFIED_A_WEBSITE)
        return super().complete(role=role, prompt=prompt)


@pytest.fixture(autouse=True)
def _deep_research_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSCIENTIST_DEEP_RESEARCH", "off")


def _through_evidence() -> CoScientistWorkflow:
    """Run the evidence stage, but do not accept it.

    The gate refuses a packet with an unverified source in it, which is the whole
    point of the downgrade: what these tests need is the stage's own output, not
    the supervisor's opinion of it.
    """
    flow = CoScientistWorkflow(QUESTION, _Searching())
    flow.accept(flow.preview(), actor="test_researcher")
    flow.preview()
    return flow


def test_the_stage_corrects_a_verifier_that_verified_a_website():
    """The contract asks the verifier not to do this. A live run showed it doing so
    anyway when the grounding metadata gave it nothing better, and everything
    downstream reads the status field and nothing else."""
    flow = _through_evidence()

    packet = next(
        EvidencePacket.model_validate(item.payload)
        for item in reversed(flow.session.artifacts)
        if item.agent == "source_verification" and item.payload
    )

    assert {source.id: source.verification_status for source in packet.sources} == {
        "src_real": "verified",
        "src_bare": "inaccessible",
    }
    assert {claim.id: claim.verification_status for claim in packet.claims} == {
        "claim_real": "verified",
        "claim_bare": "inaccessible",
    }


def test_the_appendix_says_how_the_literature_was_actually_searched():
    """ "A single search-grounded pass, from one set of queries" was honest when the
    stage ran one query. Repeating it now would understate the search by an order
    of magnitude -- the opposite failure, and just as misleading."""
    report = compile_dossier(_through_evidence().session)

    assert "parallel searches" in report
    assert "from one set of queries" not in report
    assert "What no amount of parallel search does is iterate" in report


def test_the_contract_asks_for_a_title_and_says_what_a_title_is_not():
    """A live run put a sentence from an abstract, a finding, and a
    volume-and-page line in the column a reader scans for paper titles."""
    instruction = STRUCTURED_OUTPUT_INSTRUCTIONS["evidence_discovery"]

    assert "the document's own title and nothing else" in instruction
    assert "not a sentence from its abstract" in instruction


# ---------------------------------------------------------------------------
# What survives the retention ceiling
# ---------------------------------------------------------------------------


def _facet_leads(facet: str, count: int) -> list[SourceLead]:
    return [
        SourceLead(
            canonical_url=f"https://example.org/{facet}/{index}",
            title=f"{facet} {index}",
            facets=[facet],
        )
        for index in range(count)
    ]


def test_the_retention_ceiling_keeps_every_facet_the_run_paid_to_search():
    """It used to be ``leads[:90]``, over a list in pass-ingestion order.

    The broad supporting pass is ingested first and returns the most, so a live
    seven-facet fan-out kept ninety leads that were all its -- and discarded
    every lead the contradictory, negative-null, replication, methods, safety
    and retraction passes had found. Twenty-one dollars of research, six facets
    of it gone, and the panel then reported one facet covered.
    """
    leads = [
        lead
        for facet, count in zip(
            EVIDENCE_FACETS, (200, 40, 30, 20, 15, 10, 5), strict=True
        )
        for lead in _facet_leads(facet, count)
    ]

    kept = retain_leads(leads, 90)

    assert len(kept) == 90
    assert {facet for lead in kept for facet in lead.facets} == set(EVIDENCE_FACETS)
    # The scarce facets are kept whole rather than sampled; the broad one gives
    # up the slots, because it is the one with leads to spare.
    assert sum(lead.facets == ["corrections_retractions"] for lead in kept) == 5


def test_a_corpus_inside_the_ceiling_is_left_in_the_order_it_was_found():
    leads = _facet_leads("supporting", 10)

    assert retain_leads(leads, 90) == leads


def test_an_undecomposed_pass_still_fills_the_manifest():
    """One broad search tags nothing, and round-robin over one queue is a slice."""
    leads = [
        SourceLead(canonical_url=f"https://example.org/{index}") for index in range(120)
    ]

    kept = retain_leads(leads, 90)

    assert kept == leads[:90]
