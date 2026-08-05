"""A verification pass may downgrade a claim. It may not make one vanish.

Taken from a live run: the verifier returned six claims -- each with a source,
an exact location, a relation and its own limitations, two of them contradicting
the hypothesis -- and honestly marked them unverified because its only URLs were
opaque search redirectors. The gate discarded the whole packet and rebuilt it
from a URL regex, so the stage recorded zero claims. These tests pin the
corrected behaviour: honesty is audited, not punished.
"""

from __future__ import annotations

import json

from coscientist.models import (
    Artifact,
    ArtifactStatus,
    EvidenceClaim,
    EvidencePacket,
    Session,
)
from coscientist.parity import evidence_packet

QUESTION = "Can a protective interphase coating extend lithium-ion battery cycle life?"


def _session(*claims: EvidenceClaim) -> Session:
    """A session whose accepted evidence artifact is the discovery packet."""
    session = Session(question=QUESTION)
    if claims:
        packet = EvidencePacket(question=QUESTION, claims=list(claims))
        session.artifacts.append(
            Artifact(
                stage="evidence",
                agent="evidence_discovery",
                content="",
                schema_name="EvidencePacket",
                payload=packet.model_dump(mode="json"),
                status=ArtifactStatus.ACCEPTED,
            )
        )
    return session


def _packet_json(*, claims: list[dict], sources: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "question": QUESTION,
            "sources": sources
            if sources is not None
            else [{"id": "src_1", "url": "https://example.org/a", "title": "A"}],
            "claims": claims,
            "limitations": ["Queries were narrow."],
        }
    )


def test_an_honest_unverified_packet_survives_the_gate():
    """The live regression: six good claims became zero for admitting a gap."""
    content = _packet_json(
        claims=[
            {
                "id": f"claim_{n}",
                "claim": f"Finding {n}.",
                "source_id": "src_1",
                "exact_location": f"Paragraph {n}",
                "verification_status": "discovered_unverified",
            }
            for n in range(1, 7)
        ]
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert [claim.id for claim in packet.claims] == [f"claim_{n}" for n in range(1, 7)]
    assert packet.limitations[0] == "Queries were narrow."


def test_claiming_verification_without_receipts_is_downgraded_not_deleted():
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Asserted without a location.",
                "source_id": "src_1",
                "exact_location": "",
                "verification_status": "verified",
            }
        ]
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert len(packet.claims) == 1
    assert packet.claims[0].verification_status == "discovered_unverified"
    assert "Downgraded to unverified" in packet.claims[0].limitations[0]


def test_a_claim_citing_a_source_outside_its_own_packet_cannot_be_verified():
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Points at a source nobody listed.",
                "source_id": "src_absent",
                "exact_location": "Figure 2",
                "verification_status": "verified",
            }
        ]
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert packet.claims[0].verification_status == "discovered_unverified"


def test_a_properly_evidenced_claim_keeps_its_verified_status():
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Confirmed against the original.",
                "source_id": "src_1",
                "exact_location": "Table 3",
                "verification_status": "verified",
            }
        ]
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert packet.claims[0].verification_status == "verified"
    assert packet.claims[0].limitations == []


def test_a_discovered_claim_the_verifier_forgot_is_carried_forward_unreachable():
    """Deleting an unreached claim makes it indistinguishable from one never made."""
    discovered = EvidenceClaim(id="claim_9", claim="Found during discovery.")
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "The only one the pass returned.",
                "source_id": "src_1",
                "exact_location": "Paragraph 1",
                "verification_status": "discovered_unverified",
            }
        ]
    )
    packet = evidence_packet(_session(discovered), content, verified=True)
    carried = {claim.id: claim for claim in packet.claims}["claim_9"]
    assert carried.verification_status == "inaccessible"
    assert "unreachable rather than dropped" in carried.limitations[0]
    assert any("carried forward as unreachable" in note for note in packet.limitations)


def test_nothing_is_carried_forward_when_the_pass_returned_everything():
    discovered = EvidenceClaim(id="claim_1", claim="Found during discovery.")
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Found during discovery.",
                "source_id": "src_1",
                "exact_location": "Paragraph 1",
                "verification_status": "discovered_unverified",
            }
        ]
    )
    packet = evidence_packet(_session(discovered), content, verified=True)
    assert len(packet.claims) == 1
    assert not any("carried forward" in note for note in packet.limitations)


def test_a_search_redirect_link_is_named_as_uncitable():
    """It resolves in a browser, expires, and names no document."""
    content = _packet_json(
        claims=[],
        sources=[
            {
                "id": "src_1",
                "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ",
                "title": "nih.gov",
            }
        ],
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert any("redirect links" in note for note in packet.limitations)


def test_a_real_url_is_not_flagged_as_a_redirect():
    content = _packet_json(
        claims=[],
        sources=[
            {"id": "src_1", "url": "https://doi.org/10.1021/example", "title": "A"}
        ],
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert not any("redirect links" in note for note in packet.limitations)


def test_unparseable_output_still_falls_back_to_scraped_leads():
    """The audit path must not swallow the case it was never meant to handle."""
    packet = evidence_packet(
        _session(),
        "I read https://example.org/paper and it looked broadly supportive.",
        verified=True,
    )
    assert [source.url for source in packet.sources] == ["https://example.org/paper"]
    assert packet.claims == []
    assert any("discovery leads" in note for note in packet.limitations)


def test_discovery_still_forces_every_status_down_to_unverified():
    """Discovery has opened nothing; it may not self-certify."""
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Optimistically self-certified.",
                "source_id": "src_1",
                "exact_location": "Table 1",
                "verification_status": "verified",
            }
        ],
        sources=[
            {
                "id": "src_1",
                "url": "https://example.org/a",
                "title": "A",
                "verification_status": "verified",
            }
        ],
    )
    packet = evidence_packet(_session(), content, verified=False)
    assert packet.claims[0].verification_status == "discovered_unverified"
    assert packet.sources[0].verification_status == "discovered_unverified"


def test_the_verified_property_still_refuses_an_audited_packet():
    """The gate's own verdict must not soften just because the packet survived."""
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Honest and unconfirmed.",
                "source_id": "src_1",
                "exact_location": "Paragraph 1",
                "verification_status": "discovered_unverified",
            }
        ]
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert packet.verified is False


def test_a_fully_verified_packet_passes_the_gate_end_to_end():
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Confirmed against the original.",
                "source_id": "src_1",
                "exact_location": "Table 3",
                "verification_status": "verified",
            }
        ],
        sources=[
            {
                "id": "src_1",
                "url": "https://doi.org/10.1021/example",
                "title": "A",
                "verification_status": "verified",
            }
        ],
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert packet.verified is True


def test_a_source_the_pass_marks_retracted_keeps_that_verdict():
    """Retraction is a finding. Normalising it away would erase the finding."""
    content = _packet_json(
        claims=[],
        sources=[
            {
                "id": "src_1",
                "url": "https://example.org/withdrawn",
                "title": "Withdrawn",
                "verification_status": "retracted",
            }
        ],
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert packet.sources[0].verification_status == "retracted"


def test_carried_claims_are_not_citable_grounding():
    """An unreachable claim must not become a hypothesis's evidence."""
    from coscientist.citations import GROUNDED_STATUSES

    discovered = EvidenceClaim(id="claim_9", claim="Found during discovery.")
    packet = evidence_packet(
        _session(discovered),
        _packet_json(claims=[]),
        verified=True,
    )
    assert all(
        claim.verification_status not in GROUNDED_STATUSES for claim in packet.claims
    )


def test_sources_survive_the_audit_intact():
    content = _packet_json(
        claims=[],
        sources=[
            {"id": "src_1", "url": "https://example.org/a", "title": "First"},
            {"id": "src_2", "url": "https://example.org/b", "title": "Second"},
        ],
    )
    packet = evidence_packet(_session(), content, verified=True)
    assert [source.title for source in packet.sources] == ["First", "Second"]


def test_an_empty_verification_packet_is_kept_rather_than_rescraped():
    """Returning nothing is a result; it must not be overwritten by a regex."""
    packet = evidence_packet(_session(), _packet_json(claims=[]), verified=True)
    assert packet.claims == []
    assert [source.id for source in packet.sources] == ["src_1"]


def test_a_pass_that_verified_nothing_cannot_flatten_a_contradiction():
    """Three findings, two of them contradicting, came back neutral and located
    nowhere. Nothing was confirmed, so nothing earned the right to overwrite
    what discovery recorded -- and the report had described a literature that
    agreed with itself."""
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim="Al2O3 coating showed little to no improvement in capacity fade.",
            source_id="src_1",
            exact_location="Abstract / Summary",
            relation="contradicts",
        ),
        EvidenceClaim(
            id="claim_2",
            claim="A thin ALD coating eliminates fading on LiCoO2.",
            source_id="src_1",
            exact_location="Abstract",
            relation="supports",
        ),
    )
    content = _packet_json(
        claims=[
            {
                "id": claim_id,
                "claim": "Restated without a stance.",
                "source_id": "src_1",
                "exact_location": "",
                "relation": "neutral",
                "verification_status": "discovered_unverified",
            }
            for claim_id in ("claim_1", "claim_2")
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    restored = {claim.id: claim for claim in packet.claims}
    assert restored["claim_1"].relation == "contradicts"
    assert restored["claim_1"].exact_location == "Abstract / Summary"
    assert restored["claim_2"].relation == "supports"
    assert restored["claim_2"].exact_location == "Abstract"


def test_a_pass_that_did_verify_keeps_its_own_reading_of_the_source():
    """Restoration is for a pass with nothing to say, not one that read the paper."""
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim="The coating suppresses fade.",
            source_id="src_1",
            exact_location="Abstract",
            relation="supports",
        )
    )
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "The full text reports no significant effect.",
                "source_id": "src_1",
                "exact_location": "Table 3",
                "relation": "neutral",
                "verification_status": "corrected",
            }
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    assert packet.claims[0].relation == "neutral"
    assert packet.claims[0].exact_location == "Table 3"


def test_a_pass_that_verified_nothing_cannot_rewrite_the_finding():
    """The live failure this pins: every discovered claim came back under its
    own id with the text replaced by a paraphrase of the source's title --
    "atomic layer deposition can be used to apply surface coatings", still
    labelled contradicts. The stance no longer described the sentence it was
    attached to, and the measured result was gone."""
    measured = (
        "Excessive or nonuniform CEI growth from coatings increases impedance "
        "and aggravates local polarization."
    )
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim=measured,
            source_id="src_1",
            exact_location="Introduction/Discussion",
            relation="contradicts",
        )
    )
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": (
                    "Atomic layer deposition can be used to apply surface "
                    "coatings on Ni-rich cathodes."
                ),
                "source_id": "src_1",
                "exact_location": "Introduction/Discussion",
                "relation": "contradicts",
                "verification_status": "discovered_unverified",
            }
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    assert packet.claims[0].claim == measured
    assert any("restated this claim" in item for item in packet.claims[0].limitations)


def test_a_pass_that_read_the_paper_may_correct_the_finding():
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim="The coating suppresses fade.",
            source_id="src_1",
            exact_location="Abstract",
            relation="supports",
        )
    )
    corrected = "Table 3 reports a 2% difference, within the stated error."
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": corrected,
                "source_id": "src_1",
                "exact_location": "Table 3",
                "relation": "neutral",
                "verification_status": "corrected",
            }
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    assert packet.claims[0].claim == corrected
    assert not packet.claims[0].limitations


def test_a_claim_the_pass_left_untouched_collects_no_restoration_note():
    """Repeating the discovered wording is agreement, not a rewrite."""
    wording = "A thin ALD coating eliminates fading on LiCoO2."
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim=wording,
            source_id="src_1",
            exact_location="Abstract",
            relation="supports",
        )
    )
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": wording,
                "source_id": "src_1",
                "exact_location": "Abstract",
                "relation": "supports",
                "verification_status": "discovered_unverified",
            }
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    assert packet.claims[0].claim == wording
    assert not packet.claims[0].limitations


def test_a_pass_that_verified_nothing_cannot_drop_the_scope_discovery_recorded():
    """Discovery writes what each finding does not cover -- "specific to NCM811
    cathodes and dry vs wet coating methods" -- and it is the only qualification on
    the number anywhere in the run. A live pass returned each claim under its own id
    carrying a line of its own boilerplate instead, so all six scopes left the record
    and the report printed six retention figures unqualified."""
    scope = "Specific to NCM811 cathodes and dry vs wet coating methods."
    session = _session(
        EvidenceClaim(
            id="claim_1",
            claim="Dry-coated NCM811 cathodes retained 80.8% after 150 cycles.",
            source_id="src_1",
            exact_location="Table 2",
            relation="supports",
            limitations=[scope],
        )
    )
    content = _packet_json(
        claims=[
            {
                "id": "claim_1",
                "claim": "Dry-coated NCM811 cathodes retained 80.8% after 150 cycles.",
                "source_id": "src_1",
                "exact_location": "Table 2",
                "relation": "supports",
                "verification_status": "discovered_unverified",
                "limitations": ["Unverified claim inferred from search snippet."],
            }
        ]
    )

    packet = evidence_packet(session, content, verified=True)

    assert scope in packet.claims[0].limitations
    # What the pass had to say about its own reach is kept alongside, not replaced.
    assert "Unverified claim inferred from search snippet." in (
        packet.claims[0].limitations
    )
    assert packet.claims[0].limitations.count(scope) == 1
