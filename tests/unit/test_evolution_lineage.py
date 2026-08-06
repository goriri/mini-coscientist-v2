"""A rewrite has to resolve back to the idea it is a rewrite of.

Every downstream sentence about evolution is keyed on that mapping: which idea was
revised, what the rewrite changed, which idea the meta-review recommended, and which
of two rankings applies. On the run that finished today the mapping came out empty,
and the report printed all four of those wrong at once -- a closing section
recommending four ideas under titles that appear nowhere else in the document, one of
them the exact negation of the ranked idea it was a rewrite of; a promise, made twice,
that the revised text was set out under each idea, over eight ideas that carried no
such section; and a paragraph naming a reordering whose second order was the empty
string, printed as "ranked on the proposals ... they come out ."

The contract puts the parent in ``EvolutionRecord.parent_ids``. These tests hold the
places a live specialist puts it instead, and hold the report honest where it is
nowhere at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from coscientist.models import (
    Candidate,
    CandidatePopulation,
    DossierManifest,
    EvolutionCycle,
    EvolutionRecord,
    Session,
    SourceLead,
    TournamentState,
)
from coscientist.narrative import (
    CitationRegistry,
    ResearchRecord,
    _post_evolution_reordering,
    _reference_standing,
    _trace_lineage,
    synthesize_overview,
)


def _candidate(candidate_id: str, claim: str, parents: list[str] | None = None):
    return Candidate(
        id=candidate_id,
        parent_ids=parents or [],
        title=claim,
        claim=claim,
        rationale="Because the coating blocks the reaction.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        falsifier="Retention does not improve.",
    )


def _record(
    *,
    revision_id: str = "rev_1",
    record_parents: list[str] | None = None,
    candidate_parents: list[str] | None = None,
) -> ResearchRecord:
    """One ranked idea and one rewrite of it, with the parent wherever asked for."""
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_1", "A thin alumina coating raises retention")]
    )
    record.titles = {
        "cand_1": "A Thin Alumina Coating",
        revision_id: "A Thin Alumina Coating, Preregistered",
    }
    record.evolution = EvolutionCycle(
        records=[
            EvolutionRecord(
                parent_ids=list(record_parents or []),
                candidate=_candidate(
                    revision_id,
                    "A thin alumina coating raises retention, preregistered",
                    parents=candidate_parents,
                ),
                changes=["Preregistered the discriminating design."],
                new_prediction="Retention improves by ten points.",
            )
        ]
    )
    return record


def test_the_parent_is_read_off_the_record_where_the_contract_puts_it():
    record = _record(record_parents=["cand_1"])
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("rev_1") == "cand_1"


def test_the_parent_is_read_off_the_evolved_candidate_when_the_record_omits_it():
    """The reshaper that unwraps a by-round payload reads it off the candidate."""
    record = _record(candidate_parents=["cand_1"])
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("rev_1") == "cand_1"


def test_a_parent_named_by_title_resolves_to_the_idea_that_carries_that_title():
    record = _record(record_parents=["A Thin  Alumina   Coating"])
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("rev_1") == "cand_1"


def test_the_parent_is_recovered_from_the_revisions_own_id_as_a_last_resort():
    record = _record(revision_id="cand_1_evolved_2")
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("cand_1_evolved_2") == "cand_1"


def test_one_ranked_id_is_not_read_out_of_another_that_extends_it():
    """``cand_1`` is a prefix of ``cand_11``, and a prefix is not a parent."""
    record = _record(revision_id="cand_11_evolved_2")
    record.population = CandidatePopulation(
        candidates=[
            _candidate("cand_1", "A thin alumina coating raises retention"),
            _candidate("cand_11", "A thin zirconia coating raises retention"),
        ]
    )
    _trace_lineage(record, {"cand_1", "cand_11"})
    assert record.ranked_id("cand_11_evolved_2") == "cand_11"


def test_a_second_revision_resolves_through_the_first():
    record = _record(revision_id="rev_1", record_parents=["cand_1"])
    record.evolution.records.append(
        EvolutionRecord(
            parent_ids=["rev_1"],
            candidate=_candidate("rev_2", "Preregistered twice", parents=["rev_1"]),
            changes=["Tightened the loading."],
            new_prediction="Retention improves by twelve points.",
        )
    )
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("rev_2") == "cand_1"


def test_a_rewrite_that_names_no_parent_anywhere_is_left_unmapped():
    record = _record(revision_id="offspring_zeta")
    _trace_lineage(record, {"cand_1"})
    assert record.ranked_id("offspring_zeta") == "offspring_zeta"


def _section_nine_prose(record: ResearchRecord) -> str:
    return " ".join(
        paragraph
        for section in synthesize_overview(record).sections
        if section.number == 9
        for paragraph in section.paragraphs
    )


def test_an_unresolvable_recommendation_is_reported_as_unmatched():
    """It used to be printed as an idea, under a title from the rewritten claim."""
    record = _record(revision_id="offspring_zeta")
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["offspring_zeta"]
    )
    _trace_lineage(record, {"cand_1"})
    prose = _section_nine_prose(record)
    assert "cannot match to any idea it ranked" in prose
    assert "`offspring_zeta`" in prose
    assert "The meta-review recommends carrying" not in prose


def test_a_recommendation_that_resolved_names_the_ranked_idea():
    record = _record(record_parents=["cand_1"])
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["rev_1"]
    )
    _trace_lineage(record, {"cand_1"})
    prose = _section_nine_prose(record)
    assert "A Thin Alumina Coating" in prose
    assert "cannot match to any idea it ranked" not in prose


def test_a_reordering_is_not_printed_with_one_of_its_two_orders_empty():
    """ "Ranked on the proposals ... they come out ." reached a live report."""
    record = SimpleNamespace(
        post_evolution_order=["rev_2", "rev_1"],
        title_for={"rev_1": "Alumina", "rev_2": "Zirconia"}.__getitem__,
        ranked_id=lambda item: item,
    )
    briefs = [SimpleNamespace(candidate_id="cand_1")]
    assert not _post_evolution_reordering(record, ["rev_1", "rev_2"], briefs)


def test_a_reordering_compares_the_two_orders_through_the_lineage():
    record = _record(record_parents=["cand_1"])
    record.population.candidates.append(
        _candidate("cand_2", "A thin zirconia coating raises retention")
    )
    record.titles["cand_2"] = "A Thin Zirconia Coating"
    record.evolution.records.append(
        EvolutionRecord(
            parent_ids=["cand_2"],
            candidate=_candidate("rev_2", "Zirconia, preregistered"),
            changes=["Preregistered the design."],
            new_prediction="Retention improves.",
        )
    )
    record.evolution.ranking_history = [
        TournamentState(ratings={"rev_2": 1300.0, "rev_1": 1200.0})
    ]
    _trace_lineage(record, {"cand_1", "cand_2"})
    briefs = [
        SimpleNamespace(candidate_id="cand_1"),
        SimpleNamespace(candidate_id="cand_2"),
    ]
    said = _post_evolution_reordering(record, ["cand_2", "cand_1"], briefs)
    assert "A Thin Zirconia Coating" in said and "A Thin Alumina Coating" in said
    assert "they come out ." not in said


def _standing(*statuses: str) -> ResearchRecord:
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.citations = CitationRegistry(
        [
            SourceLead(
                canonical_url=f"https://example.org/{index}",
                title=f"Paper {index}",
                verification_status=status,
            )
            for index, status in enumerate(statuses, start=1)
        ]
    )
    return record


def test_the_reference_standing_says_nothing_was_checked_when_nothing_was():
    said = _reference_standing(_standing("discovered_unverified", "metadata_verified"))
    assert said.startswith("Every one of them is a lead")


def test_the_reference_standing_says_everything_was_checked_when_it_was():
    """The flat "every one of them is a lead" contradicted the grounded verdicts."""
    said = _reference_standing(_standing("verified", "corrected"))
    assert "checked against the document it names" in said
    assert "lead" not in said


def test_the_reference_standing_counts_a_mixed_list_rather_than_flattening_it():
    said = _reference_standing(
        _standing("verified", "discovered_unverified", "inaccessible")
    )
    assert "One of the three was retrieved and checked" in said
    assert "remaining two sources" in said


def test_the_reference_standing_spells_every_count_in_its_sentence():
    """House style writes counts above twelve in figures, and the two counts that open
    clauses in this sentence are spelled whatever their size. A live run put both rules
    in one sentence: "Twenty-five of the eighty were retrieved ... For the remaining 55
    sources" -- the same kind of number written two ways eleven words apart."""
    said = _reference_standing(
        _standing(*(["verified"] * 25 + ["discovered_unverified"] * 55))
    )
    assert "Twenty-five of the eighty were retrieved" in said
    assert "For the remaining fifty-five sources" in said
    assert "55" not in said
