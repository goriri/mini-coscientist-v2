"""A candidate may not borrow authority from evidence that is not there."""

from coscientist.citations import (
    integrity_warnings,
    resolve_candidate,
    resolve_population,
)
from coscientist.models import (
    Candidate,
    CandidatePopulation,
    EvidenceClaim,
    EvidencePacket,
    SourceRecord,
)


def _candidate(candidate_id: str, *evidence_ids: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation_strategy="evidence_first",
        title="A 2 nm alumina coating slows transition-metal dissolution.",
        claim="A 2 nm alumina coating slows transition-metal dissolution.",
        rationale="The layer blocks HF attack on the cathode surface.",
        mechanism_model="The layer blocks HF attack on the cathode surface.",
        validation_protocol="Coin cells against an uncoated control.",
        predictions=["Coated cells hold 80% capacity at cycle 500."],
        falsifier="Coated cells fade at or below the uncoated rate.",
        evidence_ids=list(evidence_ids),
    )


def _packet(*claims: EvidenceClaim, sources: list[SourceRecord] | None = None):
    return EvidencePacket(
        question="Can a coating extend cycle life?",
        claims=list(claims),
        sources=sources or [],
    )


def test_an_id_that_matches_nothing_is_reported_rather_than_ignored():
    # The live failure: generation cited claim_001 into an empty packet.
    citations = resolve_candidate(_candidate("cand_1", "claim_001"), _packet())
    assert citations.unresolved == ["claim_001"]
    assert citations.support == "unsupported"
    assert citations.grounded == []


def test_an_empty_packet_and_a_missing_packet_are_treated_alike():
    assert resolve_candidate(_candidate("cand_1", "claim_001"), None).support == (
        "unsupported"
    )


def test_citing_nothing_is_honest_and_distinct_from_citing_wrongly():
    """Silence about evidence must not be scored the same as a false citation."""
    assert resolve_candidate(_candidate("cand_1"), _packet()).support == "uncited"


def test_a_discovered_but_unverified_claim_grounds_nothing():
    claim = EvidenceClaim(
        id="claim_001",
        claim="Alumina coatings reduce metal dissolution.",
        verification_status="discovered_unverified",
    )
    citations = resolve_candidate(_candidate("cand_1", "claim_001"), _packet(claim))
    assert citations.unresolved == []
    assert citations.grounded == []
    assert citations.support == "unverified"


def test_a_verified_claim_grounds_the_candidate_that_cites_it():
    claim = EvidenceClaim(
        id="claim_001",
        claim="Alumina coatings reduce metal dissolution.",
        source_id="src_1",
        exact_location="Fig. 3",
        verification_status="verified",
    )
    citations = resolve_candidate(_candidate("cand_1", "claim_001"), _packet(claim))
    assert citations.support == "grounded"
    assert citations.grounded == ["claim_001"]


def test_one_bad_id_among_good_ones_still_condemns_the_candidate():
    claim = EvidenceClaim(id="claim_001", claim="Real.", verification_status="verified")
    citations = resolve_candidate(
        _candidate("cand_1", "claim_001", "claim_999"), _packet(claim)
    )
    assert citations.unresolved == ["claim_999"]
    assert citations.support == "unsupported"


def test_a_partly_verified_set_is_named_as_such():
    verified = EvidenceClaim(id="c1", claim="Real.", verification_status="verified")
    pending = EvidenceClaim(
        id="c2", claim="Unchecked.", verification_status="discovered_unverified"
    )
    citations = resolve_candidate(
        _candidate("cand_1", "c1", "c2"), _packet(verified, pending)
    )
    assert citations.support == "partially_grounded"


def test_a_retracted_source_is_worse_than_an_unchecked_one():
    """Retraction is positive evidence against; it must not read as 'pending'."""
    source = SourceRecord(
        id="src_1",
        title="Withdrawn study",
        url="https://example.org/withdrawn",
        verification_status="retracted",
    )
    citations = resolve_candidate(
        _candidate("cand_1", "src_1"), _packet(sources=[source])
    )
    assert citations.support == "discredited"
    assert citations.discredited == ["src_1"]
    assert citations.grounded == []


def test_warnings_name_the_hypothesis_a_reader_must_distrust():
    verified = EvidenceClaim(
        id="claim_001",
        claim="Alumina coatings reduce metal dissolution.",
        verification_status="verified",
        source_id="src_1",
        confidence=0.9,
    )
    source = SourceRecord(
        id="src_1",
        title="Alumina coatings",
        url="https://example.org/alumina",
        verification_status="verified",
    )
    population = CandidatePopulation(
        candidates=[
            _candidate("cand_bad", "claim_404"),
            _candidate("cand_ok", "claim_001"),
        ],
        target_size=2,
    )
    warnings = integrity_warnings(
        resolve_population(population, _packet(verified, sources=[source]))
    )
    assert len(warnings) == 1
    assert "cand_bad" in warnings[0] and "claim_404" in warnings[0]
    assert "unsupported" in warnings[0]


def test_an_uncited_hypothesis_is_reported_as_ungrounded_rather_than_omitted():
    """Citing nothing is honest, which is not the same as being grounded. Only the
    candidates that cited something were reported, so the one hypothesis in the run
    with no grounding at all was the one missing from the list of ungrounded ones --
    and on a live run it was the hypothesis the tournament ranked first."""
    population = CandidatePopulation(
        candidates=[_candidate("cand_silent")], target_size=1
    )
    warnings = integrity_warnings(resolve_population(population, _packet()))

    assert len(warnings) == 1
    assert "cand_silent" in warnings[0]
    assert "cites no evidence at all" in warnings[0]
    assert "conjecture" in warnings[0]
