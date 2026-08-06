"""What a corpus has to be before a hypothesis may rest on it.

Two separate jobs are tested here. The sweep caps every status in a packet at
what a retrieval actually supports, because a run whose only fetch tool raised
``ImportError`` still returned a packet full of confident statuses and nothing
downstream could tell those from real ones. The floor then measures the capped
corpus on the three axes that decide whether a literature was read: how much was
checked, how many kinds of evidence were found, and whether anything was found
that disagrees.

The floor reports all three whether or not it is met, because the researcher at
the gate is being asked to decide, not informed of a verdict.
"""

from __future__ import annotations

import pytest

from coscientist.evidence import (
    apply_retrieval_outcomes,
    evaluate_evidence_floor,
    sweep_verification,
)
from coscientist.models import (
    EVIDENCE_FLOOR_CREDIT,
    EVIDENCE_FLOOR_FACETS,
    DiscoveryManifest,
    EvidenceClaim,
    EvidencePacket,
    SourceLead,
    SourceRecord,
)
from coscientist.retrieval import FetchedDocument, RetrievalOutcome, SourceMetadata

QUESTION = "Can a protective interphase coating extend lithium-ion cycle life?"


def _source(index: int, *, status: str = "verified", facet: str = "supporting"):
    return SourceRecord(
        id=f"src_{index}",
        url=f"https://doi.org/10.1000/{index}",
        title=f"Paper {index}",
        source_type="primary_study",
        verification_status=status,
        facet=facet,
    )


def _packet(*sources: SourceRecord, claims: list[EvidenceClaim] | None = None):
    return EvidencePacket(question=QUESTION, sources=list(sources), claims=claims or [])


def _outcome(url: str, tier: str, *, metadata: SourceMetadata | None = None):
    return RetrievalOutcome(
        url=url,
        tier=tier,
        document=FetchedDocument(url=url, status=200 if tier == "verified" else 403),
        metadata=metadata or SourceMetadata(),
        reason=f"tier={tier}",
    )


# ---------------------------------------------------------------------------
# The sweep: a status is capped at what retrieval established
# ---------------------------------------------------------------------------


def test_a_status_no_fetch_supports_is_demoted_and_the_demotion_is_recorded():
    """The live failure: confident statuses from a tool that never ran."""
    packet = _packet(
        _source(1, status="verified"),
        claims=[
            EvidenceClaim(
                id="claim_1",
                claim="A 2 nm layer halves capacity loss.",
                source_id="src_1",
                relation="supports",
                verification_status="verified",
            )
        ],
    )

    updated = apply_retrieval_outcomes(
        packet,
        {
            "https://doi.org/10.1000/1": _outcome(
                "https://doi.org/10.1000/1", "inaccessible"
            )
        },
    )

    assert updated.sources[0].verification_status == "inaccessible"
    # A claim is only as verified as the document it rests on.
    assert updated.claims[0].verification_status == "inaccessible"
    assert any("downgraded" in line for line in updated.limitations)
    # The original is untouched, so a caller can still see what was claimed.
    assert packet.sources[0].verification_status == "verified"


def test_a_registry_hit_promotes_a_source_the_specialist_never_reached():
    packet = _packet(_source(1, status="discovered_unverified"))

    updated = apply_retrieval_outcomes(
        packet,
        {
            "https://doi.org/10.1000/1": _outcome(
                "https://doi.org/10.1000/1",
                "metadata_verified",
                metadata=SourceMetadata(
                    title="Alumina interphases",
                    authors=["Wei Chen"],
                    year=2023,
                    container="Nature Energy",
                    identifiers={"doi": "10.1000/1"},
                    registries=["crossref"],
                ),
            )
        },
    )

    assert updated.sources[0].verification_status == "metadata_verified"
    assert updated.sources[0].authors == ["Wei Chen"]
    assert updated.sources[0].year == 2023
    assert updated.sources[0].container == "Nature Energy"
    assert updated.sources[0].identifiers["doi"] == "10.1000/1"


def test_a_reading_of_the_text_survives_a_retrieval_that_also_read_it():
    """The specialist decides meaning; the sweep only decides reachability."""
    packet = _packet(_source(1, status="corrected"))

    updated = apply_retrieval_outcomes(
        packet,
        {
            "https://doi.org/10.1000/1": _outcome(
                "https://doi.org/10.1000/1", "verified"
            )
        },
    )

    assert updated.sources[0].verification_status == "corrected"
    assert updated.limitations == []


def test_a_retraction_overrides_a_document_that_was_read_in_full():
    packet = _packet(
        _source(1, status="verified"),
        claims=[
            EvidenceClaim(
                id="claim_1",
                claim="Coatings triple cycle life.",
                source_id="src_1",
                relation="supports",
                verification_status="verified",
            )
        ],
    )

    updated = apply_retrieval_outcomes(
        packet,
        {
            "https://doi.org/10.1000/1": _outcome(
                "https://doi.org/10.1000/1", "retracted"
            )
        },
    )

    assert updated.sources[0].verification_status == "retracted"
    assert updated.claims[0].verification_status == "retracted"
    assert any("must not be cited as support" in line for line in updated.limitations)


def test_a_locator_naming_only_a_website_is_marked_without_a_fetch():
    packet = _packet(
        SourceRecord(
            id="src_1",
            url="https://www.nature.com",
            title="Nature",
            verification_status="verified",
        )
    )

    updated = apply_retrieval_outcomes(packet, {})

    assert updated.sources[0].verification_status == "inaccessible"
    assert "names a website rather than a document" in (
        updated.sources[0].verification_note
    )


async def test_the_sweep_returns_the_packet_untouched_when_nothing_can_be_reached(
    monkeypatch: pytest.MonkeyPatch,
):
    """No outbound access must cost the run its statuses, not its evidence."""

    async def _nothing(_targets, *, retriever=None):
        return {}

    monkeypatch.setattr("coscientist.evidence.assess_sources", _nothing)
    packet = _packet(_source(1))

    assert (await sweep_verification(packet)).sources[0].verification_status == (
        "verified"
    )


async def test_the_sweep_only_asks_about_locators_that_name_a_document(
    monkeypatch: pytest.MonkeyPatch,
):
    asked: list[tuple[str, str]] = []

    async def _record(targets, *, retriever=None):
        asked.extend(targets)
        return {}

    monkeypatch.setattr("coscientist.evidence.assess_sources", _record)
    await sweep_verification(
        _packet(
            _source(1),
            SourceRecord(id="src_2", url="https://www.nature.com", title="Nature"),
        )
    )

    assert asked == [("https://doi.org/10.1000/1", "Paper 1")]


async def test_an_empty_packet_costs_no_requests():
    assert (await sweep_verification(_packet())).sources == []


# ---------------------------------------------------------------------------
# The floor: three named checks, always reported
# ---------------------------------------------------------------------------


def _wide_corpus() -> tuple[EvidencePacket, DiscoveryManifest]:
    facets = (
        "supporting",
        "supporting",
        "contradictory",
        "negative_null",
        "replication",
        "methods",
        "safety_governance",
        "corrections_retractions",
    )
    sources = [
        _source(index, facet=facet) for index, facet in enumerate(facets, start=1)
    ]
    claims = [
        EvidenceClaim(
            id="claim_3",
            claim="Thick coatings showed no benefit.",
            source_id="src_3",
            relation="contradicts",
            verification_status="verified",
        )
    ]
    manifest = DiscoveryManifest(
        question=QUESTION, discovery_angles=["supporting", "contradictory"]
    )
    return _packet(*sources, claims=claims), manifest


def test_a_corpus_that_clears_every_check_is_met_with_no_shortfalls():
    packet, manifest = _wide_corpus()

    floor = evaluate_evidence_floor(packet, manifest)

    assert floor.met is True
    assert floor.shortfalls == []
    assert floor.verified_sources == 8
    assert floor.weighted_credit == 8.0
    assert len(floor.facets_covered) == 7
    assert floor.facets_missing == []
    assert floor.disconfirming_sources == 1
    assert floor.searched_for_disconfirming is True


def test_a_registry_confirmed_paper_counts_half_and_the_shortfall_says_so():
    """Paywalled-but-real is not the same as uncitable, and not the same as read."""
    packet = _packet(
        *[_source(index, status="verified") for index in range(1, 5)],
        *[
            _source(index, status="metadata_verified", facet="contradictory")
            for index in range(5, 11)
        ],
    )

    floor = evaluate_evidence_floor(packet)

    assert floor.verified_sources == 4
    assert floor.metadata_verified_sources == 6
    assert floor.weighted_credit == 7.0
    assert floor.credit_met is False
    assert (
        f"7 of {EVIDENCE_FLOOR_CREDIT:g} weighted verified sources"
        in (floor.shortfalls[0])
    )
    assert "count for half each" in floor.shortfalls[0]


def test_the_floor_names_the_kinds_of_evidence_that_are_missing_in_prose():
    packet, manifest = _wide_corpus()
    narrow = _packet(
        *[source for source in packet.sources if source.facet == "supporting"],
    )

    floor = evaluate_evidence_floor(narrow, manifest)

    assert floor.facets_met is False
    shortfall = next(line for line in floor.shortfalls if "evidence facets" in line)
    assert f"1 of {EVIDENCE_FLOOR_FACETS} required" in shortfall
    # The facet tokens are an enum; a researcher reads the phrases.
    assert "negative or null results" in shortfall
    assert "negative_null" not in shortfall


def test_a_quarantined_source_earns_no_credit_and_covers_no_facet():
    packet = _packet(
        *[
            _source(index, status="inaccessible", facet="contradictory")
            for index in range(1, 12)
        ]
    )

    floor = evaluate_evidence_floor(packet)

    assert floor.weighted_credit == 0.0
    assert floor.facets_covered == []
    assert floor.inaccessible_sources == 11


def test_finding_no_contradiction_is_a_finding_only_once_the_search_ran():
    packet, manifest = _wide_corpus()
    without_contradiction = _packet(*packet.sources)

    unsearched = evaluate_evidence_floor(
        without_contradiction, DiscoveryManifest(question=QUESTION)
    )
    searched = evaluate_evidence_floor(without_contradiction, manifest)

    assert unsearched.met is False
    assert unsearched.credit_met is True
    assert unsearched.facets_met is True
    assert "No search was run for contradictory" in unsearched.shortfalls[-1]
    assert searched.met is True
    assert searched.disconfirming_sources == 0
    assert searched.shortfalls == []


def test_the_facet_a_lead_was_discovered_under_counts_toward_coverage():
    """The search that found a source is better evidence of what it is than a
    keyword guess, and the packet does not always carry the facet back."""
    packet = _packet(
        *[_source(index, facet="") for index in range(1, 9)],
    )
    manifest = DiscoveryManifest(
        question=QUESTION,
        discovery_angles=["contradictory"],
        source_leads=[
            SourceLead(
                canonical_url=f"https://doi.org/10.1000/{index}",
                title=f"Paper {index}",
                facets=[facet],
            )
            for index, facet in enumerate(
                ("supporting", "contradictory", "replication", "methods"), start=1
            )
        ],
    )

    floor = evaluate_evidence_floor(packet, manifest)

    assert floor.facets_covered == [
        "contradictory",
        "methods",
        "replication",
        "supporting",
    ]
    assert floor.met is True


def test_a_retraction_is_counted_even_though_it_is_not_a_shortfall():
    packet = _packet(_source(1, status="retracted"))

    floor = evaluate_evidence_floor(packet)

    assert floor.retracted_sources == 1
    assert floor.weighted_credit == 0.0
    assert floor.met is False
