"""The evidence panel, as a reader sees it.

What this replaces listed every lead's title and URL in one flat block: forty-four
rows, all marked unverified, next to a "Coverage by facet" box and an "Unresolved
gaps" box that were both empty, on a run where neither was measured. A reader
could not tell which claim any source was found for, which sources disagreed with
the rest, or whether an empty box meant perfect coverage or none.

So these tests hold the panel to what it must be able to say: how far each source
can be trusted, what it was found for, which kinds of evidence are missing, and
-- before the verifier has run -- that nothing has been checked yet.
"""

from __future__ import annotations

from coscientist.models import (
    Artifact,
    ArtifactStatus,
    DeepResearchRun,
    DiscoveryCoverage,
    DiscoveryManifest,
    EvidenceClaim,
    EvidencePacket,
    ResearchGap,
    Session,
    SourceLead,
    SourceRecord,
)
from coscientist.presentation import build_stage_presentation

QUESTION = "Can a protective interphase coating extend lithium-ion cycle life?"

ALUMINA = "https://doi.org/10.1000/alumina"
THICK = "https://doi.org/10.1000/thick"
PAYWALLED = "https://doi.org/10.1000/paywalled"
DEAD = "https://example.org/gone"


def _manifest(**overrides) -> DiscoveryManifest:
    base = {
        "question": QUESTION,
        "source_leads": [
            SourceLead(
                canonical_url=ALUMINA,
                title="Alumina interphases on silicon anodes",
                provider="deep_research",
                facets=["supporting"],
                source_type="primary_study",
            ),
            SourceLead(
                canonical_url=THICK,
                title="No cycle-life gain from thick coatings",
                provider="deep_research",
                facets=["contradictory"],
                source_type="primary_study",
            ),
            SourceLead(
                canonical_url=PAYWALLED,
                title="Interphase formation reviewed",
                provider="google_search",
                facets=["methods"],
            ),
            SourceLead(
                canonical_url=DEAD,
                title="Coating overview",
                provider="google_search",
                facets=["supporting"],
            ),
        ],
        "coverage_history": [
            DiscoveryCoverage(
                facet_scores={"supporting": 1.0, "contradictory": 1.0, "methods": 0.5},
                weighted_score=0.61,
                gaps=[
                    ResearchGap(
                        direction="primary",
                        facet="replication",
                        description=(
                            "A pass dedicated to independent replication returned "
                            "no citable source."
                        ),
                        decision_impact="high",
                    )
                ],
            )
        ],
    }
    base.update(overrides)
    return DiscoveryManifest(**base)


def _packet() -> EvidencePacket:
    return EvidencePacket(
        question=QUESTION,
        sources=[
            SourceRecord(
                id="src_alumina",
                url=ALUMINA,
                title="Alumina interphases on silicon anodes",
                source_type="primary_study",
                verification_status="verified",
                verification_note="Retrieved 42,000 characters of text.",
                authors=["Wei Chen", "Mei Lin"],
                year=2023,
                container="Nature Energy",
                identifiers={"doi": "10.1000/alumina"},
                facet="supporting",
            ),
            SourceRecord(
                id="src_thick",
                url=THICK,
                title="No cycle-life gain from thick coatings",
                verification_status="verified",
                facet="contradictory",
            ),
            SourceRecord(
                id="src_paywalled",
                url=PAYWALLED,
                title="Interphase formation reviewed",
                verification_status="metadata_verified",
                verification_note="crossref confirms this record.",
                facet="methods",
            ),
            SourceRecord(
                id="src_dead",
                url=DEAD,
                title="Coating overview",
                verification_status="inaccessible",
                verification_note="HTTP 404",
            ),
        ],
        claims=[
            EvidenceClaim(
                id="claim_alumina",
                claim="A 2 nm alumina layer halves first-cycle capacity loss.",
                source_id="src_alumina",
                exact_location="Figure 3",
                relation="supports",
                verification_status="verified",
                confidence=0.8,
            ),
            EvidenceClaim(
                id="claim_thick",
                claim="Coatings above 20 nm showed no measurable benefit.",
                source_id="src_thick",
                exact_location="Table 2",
                relation="contradicts",
                verification_status="verified",
                confidence=0.6,
            ),
        ],
    )


def _session(
    manifest: DiscoveryManifest, packet: EvidencePacket | None = None
) -> Session:
    session = Session(question=QUESTION)
    session.artifacts.append(
        Artifact(
            stage="evidence",
            agent="deep_research_discovery",
            content="",
            schema_name="DiscoveryManifest",
            payload=manifest.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        )
    )
    if packet is not None:
        session.artifacts.append(
            Artifact(
                stage="evidence",
                agent="source_verification",
                content="",
                schema_name="EvidencePacket",
                payload=packet.model_dump(mode="json"),
                status=ArtifactStatus.ACCEPTED,
            )
        )
    return session


def _evidence(session: Session) -> dict:
    presentation = build_stage_presentation(session, "evidence")
    assert presentation is not None
    return presentation["evidence"]


def _facet(evidence: dict, name: str) -> dict:
    return next(entry for entry in evidence["facets"] if entry["facet"] == name)


# ---------------------------------------------------------------------------
# Before the verifier has run
# ---------------------------------------------------------------------------


def test_nothing_is_called_quarantined_before_anything_has_been_checked():
    """Reporting the stage's own incompleteness as a finding about the
    literature is how a panel comes to say four sources are unusable when the
    verifier has not started."""
    presentation = build_stage_presentation(_session(_manifest()), "evidence")
    assert presentation is not None
    evidence = presentation["evidence"]

    assert evidence["verification_ran"] is False
    assert evidence["floor"] is None
    assert evidence["floor_details"] == []
    assert evidence["headline"] == ""
    assert [metric["label"] for metric in presentation["metrics"]] == [
        "Deep Research passes",
        "Source leads",
        "Coverage",
        "Estimated cost",
    ]
    assert presentation["summary"].startswith("Knowledge landscape from")


def test_a_discovered_lead_is_shown_as_unverified_rather_than_as_usable():
    evidence = _evidence(_session(_manifest()))

    assert evidence["verified_count"] == 0
    assert {card["status_label"] for card in evidence["quarantine"]} == {"Unverified"}
    assert all(not facet["sources"] for facet in evidence["facets"])


# ---------------------------------------------------------------------------
# After verification
# ---------------------------------------------------------------------------


def test_the_headline_says_what_is_usable_and_whether_the_floor_is_met():
    evidence = _evidence(_session(_manifest(), _packet()))

    assert evidence["verification_ran"] is True
    assert evidence["headline"] == (
        "Three usable sources, three facets covered — the evidence floor is not met"
    )
    assert evidence["verified_count"] == 2
    assert evidence["metadata_verified_count"] == 1
    assert evidence["quarantined_count"] == 1


def test_the_floor_is_three_named_checks_and_not_a_pass_fail_bit():
    evidence = _evidence(_session(_manifest(), _packet()))

    assert [detail["label"] for detail in evidence["floor_details"]] == [
        "Weighted credit",
        "Facets covered",
        "Disconfirming evidence",
    ]
    credit, facets, disconfirming = evidence["floor_details"]
    assert credit["value"] == "2.5 of 8 required"
    assert credit["met"] is False
    assert facets["value"] == "3 of 4 required"
    assert disconfirming["value"] == "1 found"
    assert disconfirming["met"] is True
    assert evidence["shortfalls"]


def test_every_source_carries_the_claim_it_was_found_for_and_its_relation():
    """A title and a URL is a reading list; this is what a reader can act on."""
    evidence = _evidence(_session(_manifest(), _packet()))
    card = _facet(evidence, "contradictory")["sources"][0]

    assert card["title"] == "No cycle-life gain from thick coatings"
    assert card["relations"] == ["contradicts"]
    assert card["claims"] == [
        {
            "text": "Coatings above 20 nm showed no measurable benefit.",
            "relation": "contradicts",
            "relation_label": "Contradicts",
            "location": "Table 2",
            "confidence": 0.6,
            "limitations": [],
        }
    ]


def test_a_source_is_named_the_way_it_would_be_cited():
    evidence = _evidence(_session(_manifest(), _packet()))
    card = _facet(evidence, "supporting")["sources"][0]

    assert (
        card["citation"]
        == "Wei Chen et al. · 2023 · Nature Energy · doi:10.1000/alumina"
    )
    assert card["status_label"] == "Verified"
    assert card["status_tone"] == "verified"
    assert card["provider"] == "deep_research"


def test_a_registry_confirmed_source_says_that_nothing_has_read_it():
    evidence = _evidence(_session(_manifest(), _packet()))
    card = _facet(evidence, "methods")["sources"][0]

    assert card["status_label"] == "Registry-confirmed"
    assert card["status_tone"] == "partial"
    assert "nothing has checked what it says" in card["status_meaning"]
    assert card["verification_note"] == "crossref confirms this record."


def test_an_empty_facet_stays_visible_and_carries_the_gap_that_explains_it():
    """An absent box reads as perfect coverage. A visible empty one reads as a
    finding, which is what it is."""
    evidence = _evidence(_session(_manifest(), _packet()))
    replication = _facet(evidence, "replication")

    assert replication["label"] == "Independent replication"
    assert replication["sources"] == []
    assert replication["score"] == 0
    assert replication["gaps"] == [
        {
            "description": (
                "A pass dedicated to independent replication returned no citable "
                "source."
            ),
            "impact": "high",
        }
    ]
    # All seven facets are shown whether or not anything was found for them.
    assert len(evidence["facets"]) == 7


def test_what_cannot_be_relied_on_is_kept_where_it_can_still_be_read():
    evidence = _evidence(_session(_manifest(), _packet()))

    assert [card["url"] for card in evidence["quarantine"]] == [DEAD]
    assert evidence["quarantine"][0]["status_label"] == "Unreachable"
    assert evidence["quarantine"][0]["verification_note"] == "HTTP 404"
    # And it is not counted anywhere as usable evidence.
    assert all(
        DEAD not in [card["url"] for card in facet["sources"]]
        for facet in evidence["facets"]
    )


def test_a_verified_source_the_manifest_never_held_is_still_shown():
    """A corpus that disagrees with the discovery record is worth seeing."""
    packet = _packet()
    packet.sources.append(
        SourceRecord(
            id="src_extra",
            url="https://doi.org/10.1000/extra",
            title="Late addition",
            verification_status="verified",
            facet="replication",
        )
    )

    evidence = _evidence(_session(_manifest(), packet))

    card = _facet(evidence, "replication")["sources"][0]
    assert card["title"] == "Late addition"
    assert card["provider"] == "verification"


def test_a_source_nobody_attributed_to_a_facet_is_shown_rather_than_dropped():
    packet = _packet()
    packet.sources[0].facet = ""
    manifest = _manifest()
    manifest.source_leads[0].facets = []

    evidence = _evidence(_session(manifest, packet))
    unattributed = _facet(evidence, "unattributed")

    assert unattributed["label"] == "Not attributed to a facet"
    assert [card["url"] for card in unattributed["sources"]] == [ALUMINA]
    assert unattributed["score"] is None


def test_every_status_a_source_can_hold_is_explained_in_the_panel():
    evidence = _evidence(_session(_manifest(), _packet()))

    legend = {entry["status"]: entry for entry in evidence["legend"]}
    assert set(legend) == {
        "verified",
        "corrected",
        "metadata_verified",
        "discovered_unverified",
        "inaccessible",
        "retracted",
    }
    assert all(entry["meaning"] for entry in legend.values())
    assert legend["retracted"]["tone"] == "retracted"


def test_each_pass_names_the_facet_it_was_sent_to_cover():
    """Eight identical rows saying "pass 3 of 8" told a reader nothing about
    what the fan-out actually asked."""
    manifest = _manifest(
        runs=[
            DeepResearchRun(pass_number=1, facet="supporting", status="completed"),
            DeepResearchRun(pass_number=8, facet="", status="completed"),
        ]
    )
    presentation = build_stage_presentation(_session(manifest, _packet()), "evidence")
    assert presentation is not None

    passes = next(
        detail for detail in presentation["details"] if detail["label"] == "Passes"
    )
    assert [entry["facet"] for entry in passes["value"]] == [
        "supporting",
        "gap-closing",
    ]
