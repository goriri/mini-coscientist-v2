"""The Evidence Assessment block, read against the records it names.

Two defects off page 72 of the dossier the live run produced. Twenty-eight of the
thirty bullets were badged "[Unsourced claim]" on a run whose evidence base was
mostly verified, and one bullet was the whole of "**[Unsourced claim]** The
unverified cited claim." -- a sentence the naming table writes for an id, standing
where a finding should be. Both came from the same ordering: ``load_record`` rewrites
every id out of stored prose before this block is built, so the block that was written
to resolve the ids found none, could not read a status off anything, and fell through
to the badge of last resort. These pin the snapshot that keeps the ids available and
the three shapes a specialist writes them in.
"""

from __future__ import annotations

from coscientist.models import (
    Artifact,
    Candidate,
    CandidatePopulation,
    DiscoveryManifest,
    DiscoveryNarrative,
    DiscoveryStatement,
    EvidenceClaim,
    EvidencePacket,
    Session,
    SourceRecord,
)
from coscientist.narrative import (
    DISCREDITED_BADGE,
    LEAD_BADGE,
    UNSOURCED_BADGE,
    VERIFIED_BADGE,
    _evidence_notes,
    load_record,
)

QUESTION = "Does a thicker coating buy any further cycle life?"
CLAIM_TEXT = (
    "A conformal alumina layer of five to ten nanometres reduces first-cycle "
    "irreversible capacity loss in silicon-containing anodes"
)
STATEMENT_TEXT = (
    "Increased coating thickness causes severe mass transfer resistance at the "
    "particle surface"
)
WITHDRAWN_TEXT = (
    "Some studies report that even a one-nanometre coating is detrimental to "
    "cycling performance"
)
TITLE = "Ultrathin Al2O3 coatings on high-nickel cathodes"
SECOND_TITLE = (
    "Tailoring performance of the LiNi0.8Mn0.1Co0.1O2 cathode by plasma-enhanced "
    "atomic layer deposition"
)


def _record(*, evidence_for=(), evidence_against=(), evidence_gaps=()):
    """A run whose evidence base holds one verified claim and one search finding."""
    evidence = EvidencePacket(
        question=QUESTION,
        sources=[
            SourceRecord(
                id="source_1",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                title=TITLE,
                verification_status="verified",
            ),
            SourceRecord(
                id="source_2",
                url="https://pubmed.ncbi.nlm.nih.gov/23456789/",
                title=SECOND_TITLE,
                verification_status="discovered_unverified",
            ),
        ],
        claims=[
            EvidenceClaim(
                id="claim_1",
                claim=CLAIM_TEXT,
                source_id="source_1",
                verification_status="verified",
                confidence=0.9,
            ),
            EvidenceClaim(
                id="claim_2",
                claim=WITHDRAWN_TEXT,
                source_id="source_1",
                verification_status="inaccessible",
                confidence=0.4,
            ),
            EvidenceClaim(
                id="claim_3",
                claim="ALD alumina suppresses surface corrosion",
                source_id="source_1",
                verification_status="discovered_unverified",
                confidence=0.5,
            ),
            EvidenceClaim(
                id="claim_4",
                claim="A PEALD interphase layer suppresses thermal runaway",
                source_id="source_2",
                verification_status="discovered_unverified",
                confidence=0.5,
            ),
        ],
    )
    discovery = DiscoveryManifest(
        question=QUESTION,
        narratives=[
            DiscoveryNarrative(
                question=QUESTION,
                summary="Thickness and chemistry are reported together.",
                statements=[
                    DiscoveryStatement(
                        id="pass4_stmt_5",
                        text=STATEMENT_TEXT,
                        facet="contradictory",
                        originating_pass=4,
                    )
                ],
            )
        ],
    )
    population = CandidatePopulation(
        candidates=[
            Candidate(
                id="candidate_0001",
                title="Coating thickness has an optimum rather than a floor",
                claim="Coating thickness has an optimum rather than a floor",
                rationale="Transport resistance grows with thickness faster than the "
                "passivation benefit does",
                mechanism_model="Lithium crosses the coating by a hopping mechanism "
                "whose resistance scales with path length",
                validation_protocol="Coin cells at four thicknesses against an "
                "uncoated control, cycled to a prespecified retention endpoint",
                falsifier="Retention keeps rising with thickness past forty nanometres",
                evidence_for=list(evidence_for),
                evidence_against=list(evidence_against),
                evidence_gaps=list(evidence_gaps),
            )
        ]
    )
    session = Session(question=QUESTION)
    session.artifacts = [
        Artifact(
            stage="evidence",
            agent="discovery_agent",
            artifact_type="specialist_output",
            content="",
            schema_name="DiscoveryManifest",
            payload=discovery.model_dump(),
        ),
        Artifact(
            stage="evidence",
            agent="evidence_agent",
            artifact_type="specialist_output",
            content="",
            schema_name="EvidencePacket",
            payload=evidence.model_dump(),
        ),
        Artifact(
            stage="generate",
            agent="generation_agent",
            artifact_type="specialist_output",
            content="",
            schema_name="CandidatePopulation",
            payload=population.model_dump(),
        ),
    ]
    return load_record(session)


def _notes(**kwargs):
    record = _record(**kwargs)
    return _evidence_notes(record, record.candidates[0])


def test_a_statement_that_is_only_an_id_is_printed_as_what_the_record_holds():
    """ "Evidence for: claim_1." is the specialist naming a record. The reader cannot
    look the id up anywhere in the document, so the claim itself is the finding."""
    ((heading, badge, said),) = _notes(evidence_for=["claim_1"])

    assert heading == "Evidence for"
    assert said == f"{CLAIM_TEXT}."
    assert badge == VERIFIED_BADGE


def test_a_verified_claim_cited_by_id_is_not_badged_as_unsourced():
    """The badge is decided by looking for a locator in the sentence, and an id is not
    one. Reading the run's own verdict off the cited record is the whole point."""
    ((_heading, badge, _said),) = _notes(evidence_for=["claim_1"])

    assert badge != UNSOURCED_BADGE


def test_an_unverified_finding_cited_by_id_is_a_lead_and_not_a_verified_source():
    ((_heading, badge, said),) = _notes(evidence_against=["pass4_stmt_5"])

    assert badge == LEAD_BADGE
    assert said == f"{STATEMENT_TEXT}."


def test_an_id_written_in_front_of_the_sentence_it_cites_is_dropped():
    """ "pass4_stmt_5: Increased coating thickness ..." is a footnote marker in a
    format with no footnotes, and the badge beside the bullet already says as much."""
    ((_heading, badge, said),) = _notes(
        evidence_against=[f"pass4_stmt_5: {STATEMENT_TEXT}."]
    )

    assert said == f"{STATEMENT_TEXT}."
    assert badge == LEAD_BADGE


def test_a_statement_id_carrying_its_pass_number_is_named_like_any_other():
    """An underscore is a word character, so there is no boundary in front of the
    "stmt" in "pass4_stmt_5" and the pattern matched nothing. Chapters five and six
    of a live report printed these raw while every other chapter named the finding."""
    ((_heading, _badge, said),) = _notes(
        evidence_against=["The thickness limit is set by pass4_stmt_5."]
    )

    assert "pass4_stmt_5" not in said
    assert said == (
        "The thickness limit is set by the finding that increased coating thickness "
        "causes severe mass transfer resistance at the particle surface."
    )


def test_a_prefix_that_names_no_record_is_left_where_the_specialist_put_it():
    """ "Cycle_life: retention held" is a phrase with a colon after it, not a citation.
    Only a prefix the run can resolve is furniture the report may cut."""
    ((_heading, _badge, said),) = _notes(evidence_for=["Cycle_life: retention held."])

    assert said == "Cycle_life: retention held."


def test_an_id_inside_a_sentence_is_read_out_as_the_record_it_names():
    ((_heading, badge, said),) = _notes(
        evidence_for=["The retention gain in claim_1 has never been replicated."]
    )

    assert "claim_1" not in said
    assert said == (
        f"The retention gain in the claim drawn from {TITLE} has never been replicated."
    )
    assert badge == VERIFIED_BADGE


def test_an_id_the_specialist_opened_the_statement_with_takes_the_sentence_capital():
    """A live bullet read "**[Literature Lead]** the unverified source Molecular Layer
    Deposition of Organic-Inorganic Hafnium Oxynitride Hybrid Films for Electrochemical
    Applications demonstrates MLD of HfON for electrochemical systems" -- a lower-case
    article opening the sentence, beside bullets whose statements held no id and so
    began with a capital. The phrase standing in for an id opens on its article, and
    the capital belongs to whatever word opens the statement."""
    ((_heading, badge, said),) = _notes(
        evidence_against=["pass4_stmt_5 is why thickness cannot be pushed further."]
    )

    assert badge == LEAD_BADGE
    assert said == (
        "The finding that increased coating thickness causes severe mass transfer "
        "resistance at the particle surface is why thickness cannot be pushed further."
    )


def test_a_run_of_ids_of_one_kind_says_what_they_are_once_and_then_lists_them():
    """A live bullet read "**[Literature Lead]** The unverified claim drawn from
    Identification of the dual roles of Al2O3 coatings on NMC811-cathodes via theory
    and experiment and the unverified claim drawn from Tailoring Performance of the
    LiNi0.8Mn0.1Co0.1O2 Cathode by Al2O3 and MoO3 artificial cathode electrolyte
    interphase (CEI) layers through plasma-enhanced atomic layer deposition (PEALD)
    Coating show ALD Al2O3 prevents surface corrosion" -- the same eight words of
    standing around each of two titles that between them carry four further "and"s,
    so the reader has no way to see where one name ends. The idea's own prose already
    collapsed this; the bullets went by a different path and did not."""
    ((_heading, badge, said),) = _notes(
        evidence_for=["claim_3 and claim_4 show the coating prevents corrosion."]
    )

    assert badge == LEAD_BADGE
    assert said == (
        f"The unverified claims drawn from {TITLE} and {SECOND_TITLE} show the "
        "coating prevents corrosion."
    )


def test_a_run_of_ids_the_run_holds_differently_is_named_one_by_one():
    """A plural drawn over a mixture says of each record what is true of one of them:
    claim_1 was verified and claim_2 could not be retrieved, and "the unverified claims
    drawn from ..." would be wrong about both."""
    ((_heading, _badge, said),) = _notes(
        evidence_for=["claim_1 and claim_2 disagree about the floor."]
    )

    assert said == (
        f"The claim drawn from {TITLE} and the unretrieved claim drawn from {TITLE} "
        "disagree about the floor."
    )


def test_an_id_naming_nothing_this_run_holds_says_so_rather_than_standing_alone():
    """The naming table answers an unresolvable id with "the unverified cited claim",
    which printed alone was a bullet that said nothing at all."""
    ((_heading, badge, said),) = _notes(evidence_for=["claim_missing"])

    assert said.startswith('The specialist gave the record id "claim_missing" here')
    assert "no record of that id exists" in said
    assert badge == UNSOURCED_BADGE


def test_a_gap_is_a_statement_that_no_evidence_exists_and_carries_no_badge():
    ((heading, badge, said),) = _notes(
        evidence_gaps=["No study reports thicknesses above forty nanometres."]
    )

    assert heading == "Evidence gaps"
    assert badge == ""
    assert said == "No study reports thicknesses above forty nanometres."


def test_the_ids_are_kept_for_this_block_after_the_prose_has_been_scrubbed_of_them():
    """The naming pass rewrites the candidate's stored fields, and it must: those are
    read elsewhere as prose. This block needs what was written, so it is snapshotted
    before the pass rather than re-derived from what the pass left behind."""
    record = _record(evidence_for=["claim_1"])

    assert record.candidates[0].evidence_for == [f"The claim drawn from {TITLE}"]
    assert record.cited_evidence["candidate_0001"][0] == ["claim_1"]


def test_a_record_the_run_could_not_stand_behind_is_not_badged_as_a_lead():
    """A live chapter carried "[Literature Lead] Some studies suggest even 1 nm
    coatings can be detrimental" over the very record whose citation, two hundred
    lines below, was declared to discredit the grounding of three other ideas. A
    lead is something to follow; this is something the run went back to and could
    not stand behind."""
    ((_heading, badge, said),) = _notes(evidence_against=["claim_2"])

    assert badge == DISCREDITED_BADGE
    assert said == f"{WITHDRAWN_TEXT}."


def test_the_reading_guide_says_what_the_new_badge_means():
    """Every other badge is explained where the report explains how to read itself."""
    from coscientist.narrative import DEEP_DIVE_PREAMBLE

    assert any(DISCREDITED_BADGE in note for note in DEEP_DIVE_PREAMBLE)
