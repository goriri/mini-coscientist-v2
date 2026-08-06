"""What the report says has to be what the record holds.

The tests in ``test_report_structure`` pin the document's shape. These pin something
narrower and harder to see: every sentence the report writes about the run has to be
recoverable from the artifacts, and no sentence may assert a relation the contracts do
not carry. Each case below was a live report stating something the session data
contradicted — a recommendation for an idea the reader had never been shown, a fatal
flaw against an idea no reviewer faulted, an objection declared answered by a rebuttal
written about something else. They read as facts on the page, which is what makes them
worth a test apiece.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coscientist.models import (
    Candidate,
    CandidatePopulation,
    CandidateReview,
    EvolutionCycle,
    EvolutionRecord,
    ResearchCluster,
    ResearchLandscape,
    ReviewSet,
    Session,
    TournamentState,
)
from coscientist.narrative import (
    _UNSTATED,
    AdjudicationNote,
    IdeaBrief,
    IdeaReview,
    ResearchRecord,
    _category_path,
    _idea_reviews,
    _joined_titles,
    _lead_over_rival,
    _objection_spread,
    _objections_raised,
    _placed,
    _recurring_objections,
    _trace_lineage,
)
from coscientist.parity import ELO_K


def _candidate(candidate_id: str, version: int = 1, parents: list[str] | None = None):
    return Candidate(
        id=candidate_id,
        version=version,
        parent_ids=parents or [],
        title=f"{candidate_id} raises retention",
        claim=f"{candidate_id} raises retention",
        rationale="Because the coating blocks the reaction.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        falsifier="Retention does not improve.",
    )


def _evolved(candidate_id: str, parent: str, version: int) -> EvolutionRecord:
    return EvolutionRecord(
        parent_ids=[parent],
        candidate=_candidate(candidate_id, version=version, parents=[parent]),
        changes=[f"Tightened the loading for {candidate_id}."],
        new_prediction="Retention improves by ten points.",
    )


def _review(candidate_id: str, **kwargs) -> CandidateReview:
    fields = {
        "candidate_id": candidate_id,
        "criterion": "novelty",
        "recommendation": "revise",
        "reviewer": "reviewer",
    }
    fields.update(kwargs)
    return CandidateReview(**fields)


def _idea_review(**kwargs) -> IdeaReview:
    fields = {
        "section": "Novelty",
        "lead_in": "",
        "question": "",
        "findings": [],
        "objections": [],
        "rebuttals": [],
        "answer": "",
        "score": 3,
    }
    fields.update(kwargs)
    return IdeaReview(**fields)


def _facts() -> dict[str, str]:
    """The idea-fact grid a summary subsection reads, with every field stated."""
    from coscientist.narrative import _idea_facts

    return _idea_facts(
        SimpleNamespace(
            claim="A coating extends cycle life.",
            rationale="It blocks the electrolyte.",
            mechanism_model="",
            validation_protocol="Cycle ten cells per arm and compare retention.",
            predictions=["Coated cells outlast uncoated cells by fifteen per cent."],
            alternatives=["The gain comes from the binder."],
            falsifier="No difference at ten cells per arm.",
            dependencies=["An ALD reactor."],
            risks=["The coating cracks."],
            go_no_go_tests=["Thickness within two nanometres by TEM."],
        )
    )


def _brief(
    title: str,
    reviews: list[IdeaReview],
    elo: float = 1200.0,
    *,
    shortlisted: bool = False,
    matches: list | None = None,
    facts: dict[str, str] | None = None,
    candidate_id: str = "",
    revised_form: list[tuple[str, str]] | None = None,
    contradicting_claims: list[str] | None = None,
) -> IdeaBrief:
    return IdeaBrief(
        title=title,
        candidate_id=candidate_id or title,
        rank=1,
        elo=elo,
        category="",
        proposal="",
        description=[],
        facts=facts if facts is not None else {},
        summary={},
        table_rows=[],
        reviews=reviews,
        coherence=[],
        deep_verification=[],
        matches=list(matches or []),
        wins=0,
        losses=0,
        ties=0,
        shortlisted=shortlisted,
        revised_form=list(revised_form or []),
        contradicting_claims=list(contradicting_claims or []),
    )


def test_a_second_revision_resolves_to_the_idea_the_tournament_ranked():
    """A v3 names its v2 as parent, so one pass leaves it pointing at an unranked id."""
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evolution = EvolutionCycle(
        records=[
            # Deliberately out of order: the specialist emits them however it likes.
            _evolved("cand_a_v3", "cand_a_v2", 3),
            _evolved("cand_a_v2", "cand_a", 2),
        ]
    )
    _trace_lineage(record, {"cand_a"})

    assert record.ranked_id("cand_a_v3") == "cand_a"
    assert record.ranked_id("cand_a_v2") == "cand_a"
    assert record.ranked_id("cand_a") == "cand_a"


def test_a_revision_descending_from_outside_the_population_is_left_unmapped():
    """Guessing an ancestor is worse than admitting the chain does not reach one."""
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evolution = EvolutionCycle(records=[_evolved("cand_x_v2", "cand_other", 2)])
    _trace_lineage(record, {"cand_a"})

    assert record.ranked_id("cand_x_v2") == "cand_x_v2"


def test_the_latest_revision_of_an_idea_is_the_one_reported():
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evolution = EvolutionCycle(
        records=[
            _evolved("cand_a_v2", "cand_a", 2),
            _evolved("cand_a_v3", "cand_a_v2", 3),
        ]
    )
    _trace_lineage(record, {"cand_a"})

    revision = record.revision_of("cand_a_v3")
    assert revision is not None
    assert revision.candidate.version == 3


def _clustered(name: str, *candidate_ids: str) -> ResearchLandscape:
    return ResearchLandscape(
        clusters=[
            ResearchCluster(
                name=name,
                candidate_ids=list(candidate_ids),
                shared_mechanism="A coating suppresses the interfacial reaction.",
                shared_outcome="Retention holds past five hundred cycles.",
            )
        ]
    )


def test_the_rewrite_on_the_page_is_credited_only_with_its_own_re_reviews():
    """Three rounds of verdicts were attributed to the one rewrite the report prints.

    Every round mints a revision and re-reviews it, and all of them resolve back to the
    same ranked ancestor. A run of three rounds at five reviewers told the reader "it
    was re-reviewed after the rewrite: 15 reviews said revise it first" under a diff
    five of them had seen.
    """
    from coscientist.narrative import _revised_form

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evolution = EvolutionCycle(
        records=[
            _evolved("cand_a_v2", "cand_a", 2),
            _evolved("cand_a_v3", "cand_a_v2", 3),
        ],
        rereviews=[
            _review("cand_a_v2", recommendation="revise"),
            _review("cand_a_v2", recommendation="revise"),
            _review("cand_a_v3", recommendation="advance"),
        ],
    )
    _trace_lineage(record, {"cand_a"})

    lead_in, _, _ = _revised_form(record, _candidate("cand_a"))

    # One re-review, and named: a lone check is reported by what it checked.
    assert "the novelty review said advance the idea as written" in lead_in
    assert "reviews said revise" not in lead_in


def test_a_rewrite_checked_on_one_criterion_says_which_four_were_not_re_run():
    """ "It was re-reviewed after the rewrite: one review said advance the idea as
    written" sat under a live recommendation to carry the rewrite forward. Five
    criteria judged the ranked form and one judged the rewrite, so four of the five
    scores the reader was carrying forward were scores of the text it replaced."""
    from coscientist.narrative import _revised_form

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", criterion=criterion)
                for criterion in (
                    "evidence_correctness",
                    "novelty",
                    "methods_feasibility",
                    "impact_safety",
                    "safety_governance",
                )
            ]
        )
    ]
    record.evolution = EvolutionCycle(
        records=[_evolved("cand_a_v2", "cand_a", 2)],
        rereviews=[_review("cand_a_v2", criterion="novelty", recommendation="advance")],
    )
    _trace_lineage(record, {"cand_a"})
    assert len(_idea_reviews(record, "cand_a")) == 5

    lead_in, _, _ = _revised_form(record, _candidate("cand_a"))

    assert "The other four criteria that judged it" in lead_in
    assert "correctness, feasibility, impact, and safety" in lead_in
    assert "were not run again" in lead_in
    assert "the scores above are scores of the form the rewrite replaced" in lead_in


def test_a_rewrite_every_criterion_saw_again_says_nothing_about_what_was_missed():
    """The clause is only worth printing where something was in fact not re-run."""
    from coscientist.narrative import _revised_form

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evolution = EvolutionCycle(
        records=[_evolved("cand_a_v2", "cand_a", 2)],
        rereviews=[_review("cand_a_v2", recommendation="advance")],
    )
    _trace_lineage(record, {"cand_a"})

    lead_in, _, _ = _revised_form(record, _candidate("cand_a"))

    assert "not run again" not in lead_in


def test_clusters_with_one_mechanism_between_them_are_not_said_to_each_have_one():
    """A fallback clustering writes the same sentence into every cluster it makes.

    The paragraph promised "each named here with the mechanism its members share" above
    four copies of "candidates share a generation lens but retain distinct
    predictions", then told the reader that two ideas in the same cluster fail for the
    same reason. Nothing in that grouping supports either claim.
    """
    from coscientist.narrative import _section_three

    filler = "Candidates share a generation lens but retain distinct predictions."
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.landscape = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name=name,
                candidate_ids=[f"cand_{name.lower()}"],
                shared_mechanism=filler,
                shared_outcome="Retention holds.",
            )
            for name in ("Evidence First", "Mechanism First")
        ]
    )

    three = " ".join(_section_three(record).core)
    assert three.count("generation lens") == 0
    assert "recorded no mechanism that tells them apart" in three
    assert "fail for the same reason" not in three
    assert "Evidence First holds one idea." in three


def test_an_ideas_category_names_the_cluster_that_claimed_its_revision():
    """Clustering runs after evolution, so it names ``cand_a_v3``, not ``cand_a``."""
    record = ResearchRecord(
        session=Session(question="Can a coating help?", research_mode="experimental")
    )
    record.evolution = EvolutionCycle(records=[_evolved("cand_a_v2", "cand_a", 2)])
    _trace_lineage(record, {"cand_a"})
    record.landscape = _clustered("Physical Barrier Coatings", "cand_a_v2")

    path = _category_path(record, _candidate("cand_a"))

    assert path == "Experimental > Physical Barrier Coatings > Mechanism-led"


def test_an_idea_no_cluster_claimed_is_not_padded_out_to_three_levels():
    """The old middle level was the generation strategy the posture already states."""
    record = ResearchRecord(
        session=Session(question="Can a coating help?", research_mode="experimental")
    )
    record.landscape = _clustered("Physical Barrier Coatings", "cand_b")

    path = _category_path(record, _candidate("cand_a"))

    assert path == "Experimental > Mechanism-led"


def test_a_fatal_flaw_is_counted_from_the_reviews_not_from_the_summary():
    """The meta-review's exclusion list disagreed with the reviews in both directions."""
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", fatal_flaws=["The mechanism is already published."]),
                _review("cand_b"),
                _review("cand_c_v2", fatal_flaws=["No safety case."]),
            ]
        )
    ]
    record.evolution = EvolutionCycle(records=[_evolved("cand_c_v2", "cand_c", 2)])
    _trace_lineage(record, {"cand_a", "cand_b", "cand_c"})

    # Resolved back to the ranked id, so a flaw recorded against a revision is not
    # reported as a flaw against an idea the reader has not been shown.
    assert record.recorded_fatal_flaw_ids == {"cand_a", "cand_c"}


def test_an_objection_is_not_paired_with_a_rebuttal_the_record_never_linked():
    """Two independent lists were zipped, inventing both halves of the pairing."""
    reviews = [
        _idea_review(
            section="Novelty",
            objections=["No power rationale.", "No blinding protocol."],
            rebuttals=["The effect size is large."],
        ),
        _idea_review(section="Impact", objections=["No safety case."], rebuttals=[]),
    ]

    assert _objections_raised(reviews) == [
        ("Novelty", "No power rationale.", True),
        ("Novelty", "No blinding protocol.", True),
        ("Impact", "No safety case.", False),
    ]


def test_an_objection_raised_twice_keeps_the_mention_that_drew_a_response():
    reviews = [
        _idea_review(
            section="Novelty", objections=["No power rationale."], rebuttals=[]
        ),
        _idea_review(
            section="Impact",
            objections=["No power rationale."],
            rebuttals=["N=40 was chosen for 80% power."],
        ),
    ]

    assert _objections_raised(reviews) == [("Impact", "No power rationale.", True)]


def test_a_whitespace_only_rebuttal_is_not_a_response():
    reviews = [_idea_review(objections=["No safety case."], rebuttals=["   "])]

    assert _objections_raised(reviews) == [("Novelty", "No safety case.", False)]


@pytest.mark.parametrize(
    ("titles", "expected"),
    [
        (["One Idea"], "One Idea scored highest at 5"),
        (["One Idea", "Another"], "One Idea and Another tied highest at 5"),
        (["A", "B", "C"], "A; B; and C tied highest at 5"),
        (list("ABCDEFG"), "seven of the ideas tied highest at 5"),
    ],
)
def test_a_tie_at_the_top_of_a_criterion_is_reported_as_a_tie(titles, expected):
    """``max`` names one member of a tie, and naming it read as a finding."""
    assert _placed(titles, 5, "highest") == expected


def test_the_same_objection_phrased_five_ways_is_counted_once_per_idea():
    """Exact text cannot count recurrence: no two reviewers write the same sentence."""
    phrasings = [
        "Lacks randomization and blinding",
        "Missing explicit randomization and blinding protocols",
        "Fails to explicitly state that cell assignment will be randomized and blinded",
        "No randomization or blinding is described",
    ]
    briefs = [
        _brief(f"Idea {index}", [_idea_review(objections=[text])])
        for index, text in enumerate(phrasings)
    ]

    recurring = _recurring_objections(briefs)
    assert len(recurring) == 1
    objection, count = recurring[0]
    # The shortest phrasing represents the group, and the count is a floor.
    assert objection == "Lacks randomization and blinding"
    assert count == 4


def test_the_representative_phrasing_does_not_carry_one_ideas_sample_size():
    """A number from one protocol is the part of the wording the group does not share.

    "No power rationale for N=10" was the shortest phrasing in its group, so it was
    printed as the objection raised against seven of eight ideas -- three of which
    propose N=15. The figure reads as the field's, and it is one idea's.
    """
    briefs = [
        _brief("A", [_idea_review(objections=["No power rationale for N=10"])]),
        _brief("B", [_idea_review(objections=["No power rationale for N=15"])]),
        _brief("C", [_idea_review(objections=["No power rationale is given"])]),
    ]

    objection, count = _recurring_objections(briefs)[0]
    assert objection == "No power rationale is given"
    assert count == 3


def test_two_objections_sharing_one_word_are_not_merged():
    """A single shared token merged "no power rationale" with "no statistical plan"."""
    briefs = [
        _brief("A", [_idea_review(objections=["No statistical power rationale"])]),
        _brief("B", [_idea_review(objections=["No thermal power budget"])]),
    ]

    assert _recurring_objections(briefs) == []


def test_an_objection_three_ideas_escape_is_not_called_a_property_of_the_goal():
    """The gloss was written for the near-unanimous case and printed unconditionally.

    An objection recurs at half the field, so a live report closed on "an objection
    raised against most of the field is a property of the goal ... and cannot be
    resolved by choosing differently among them" over an objection raised against four
    of seven ideas. Three ideas do not carry it, and choosing one of those three
    resolves it exactly.
    """
    spread = [("The expected outcomes are speculative", 4)]

    paragraph = _objection_spread(spread, 7)
    assert "raised against at least four of the seven ideas" in paragraph
    assert (
        "Three of the ideas escaped it, so it is not a property of the goal"
        in paragraph
    )
    assert "cannot be resolved by choosing differently" not in paragraph


def test_an_objection_every_idea_but_one_carries_is_a_property_of_the_goal():
    spread = [("Lacks randomization", 7), ("No power rationale", 8)]

    paragraph = _objection_spread(spread, 8)
    assert "cannot be resolved by choosing differently among them." in paragraph
    assert "escaped" not in paragraph


def test_one_count_shared_by_every_objection_is_stated_once():
    """The per-item clause was the same eight words three times in one sentence."""
    spread = [
        ("Lacks randomization", 7),
        ("No statistical test", 7),
        ("No power case", 7),
    ]

    paragraph = _objection_spread(spread, 8)
    assert paragraph.count("raised against at least") == 1
    assert "Each was raised against at least seven of the eight ideas: " in paragraph
    # A one-item list under a colon reads as a mistake, so the hoist waits for two.
    single = _objection_spread([("Lacks randomization", 7)], 8)
    assert "Each was raised" not in single
    assert (
        "Lacks randomization — raised against at least seven of the eight ideas"
        in single
    )


def test_an_objection_raised_against_one_idea_is_not_called_recurring():
    briefs = [
        _brief("A", [_idea_review(objections=["No safety case"])]),
        _brief("B", [_idea_review(objections=["The coating is too thick"])]),
    ]

    assert _recurring_objections(briefs) == []


def _played(opponent: str, before: float, after: float):
    from coscientist.narrative import IdeaMatch

    return IdeaMatch(
        round_number=1,
        opponent_title=opponent,
        outcome="win",
        elo_before=before,
        elo_after=after,
        confidence=0.7,
        rationale="",
        judge="llm_comparison",
    )


def test_a_lead_inside_one_matchs_worth_of_rating_is_called_level():
    """The old thirty-point noise floor was invented; the played matches fix a real one."""
    leader = _brief(
        "Leader", [], elo=1215.0, matches=[_played("Rival", 1200.0, 1216.0)]
    )
    rival = _brief("Rival", [], elo=1200.0, matches=[_played("Leader", 1200.0, 1184.0)])

    line = _lead_over_rival(leader, [leader, rival])
    assert (
        "no single match in this tournament moved a rating by more than 16 points"
        in (line.lower())
    )
    assert "read as level rather than ordered" in line


def test_a_lead_wider_than_any_played_match_does_not_claim_a_multiple_of_it():
    """ "1.8 times what any single result could account for" was arithmetic on a bad bound."""
    leader = _brief(
        "Leader", [], elo=1328.0, matches=[_played("Rival", 1200.0, 1216.0)]
    )
    rival = _brief("Rival", [], elo=1200.0, matches=[_played("Leader", 1200.0, 1184.0)])

    line = _lead_over_rival(leader, [leader, rival])
    assert "wider than any single result here produced" in line
    assert "times what any single result" not in line
    assert "read as level" not in line


def test_the_bound_is_read_off_the_matches_rather_than_asserted_from_k():
    """A decided match moves the winner by K x (1 - expected), which is not K."""
    leader = _brief(
        "Leader", [], elo=1216.0, matches=[_played("Rival", 1200.0, 1216.0)]
    )
    rival = _brief("Rival", [], elo=1200.0, matches=[_played("Leader", 1200.0, 1184.0)])

    line = _lead_over_rival(leader, [leader, rival])
    # The tables in the same report show no move above 16, so 32 cannot be stated as
    # what one match does -- only as the ceiling it never came near.
    assert f"more than {round(ELO_K)} points" not in line
    assert f"K factor of {round(ELO_K)}" in line


def test_rivals_the_leader_never_played_are_named_as_never_played():
    """Wins and losses without the pairing graph read as a head-to-head ordering."""
    leader = _brief(
        "Leader", [], elo=1216.0, matches=[_played("Rival", 1200.0, 1216.0)]
    )
    rival = _brief("Rival", [], elo=1200.0)
    unmet = _brief("Stranger", [], elo=1190.0)

    line = _lead_over_rival(leader, [leader, rival, unmet])
    assert "never paired against Stranger" in line
    assert "Rival" in line.split("never paired against")[0]
    assert "shared opponents" in line


def test_a_field_of_one_claims_no_lead_at_all():
    leader = _brief("Leader", [])

    assert _lead_over_rival(leader, [leader]) == "No other idea was ranked against it."


def test_the_final_rounds_movement_is_reported_in_points_not_as_a_fraction():
    """0.0381 is 3.8% of the 1200 start -- forty-six points, not four hundredths of one."""
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381, ranking_stable_rounds=1),
        [_brief("Leader", [], elo=1290.0), _brief("Rival", [], elo=1234.0)],
    )
    assert "46 points" in line
    assert "0.04" not in line
    assert "did not converge" in line


def test_a_movement_inside_the_limit_is_not_offered_as_the_reason_it_failed():
    """ "The ranking did not converge ... The final round moved one rating by 3.7 per
    cent of that" reads as the reason, and 3.7 is inside the five the sentence above
    names. The run failed the other half of the rule."""
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381, ranking_stable_rounds=1),
        [_brief("Leader", [], elo=1290.0), _brief("Rival", [], elo=1234.0)],
    )

    assert "46 points, which is inside that limit" in line
    assert "No two consecutive rounds ended on the same top four" in line


def test_a_movement_over_the_limit_is_reported_as_over_it():
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.09, ranking_stable_rounds=2),
        [_brief("Leader", [], elo=1290.0), _brief("Rival", [], elo=1234.0)],
    )

    assert "108 points, which is over that limit" in line


def test_a_round_is_said_to_hold_more_matches_than_one():
    """45 points stood a paragraph away from "no single match moved a rating by more
    than 17 points", with nothing saying the 45 is a round rather than a match."""
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381, ranking_stable_rounds=1),
        [_brief("Leader", [], elo=1290.0), _brief("Rival", [], elo=1234.0)],
    )

    assert (
        "A round is several matches, so a rating can move further across a round "
        "than any one match moves it." in line
    )


def test_the_stated_rule_carries_the_threshold_the_tournament_judged_it_by():
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence
    from coscientist.parity import SETTLED_MOVEMENT

    line = _convergence(TournamentState(score_movement=0.0381), [])

    assert SETTLED_MOVEMENT == 0.05, "the printed rule spells this number out"
    assert "no rating by more than five per cent of the 1200" in line


def test_a_single_stable_round_is_not_reported_as_rounds_that_held():
    """``_stable_rounds`` floors at one, so one is a baseline rather than a finding."""
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381, ranking_stable_rounds=1), []
    )
    assert "No two consecutive rounds ended on the same top four" in line
    assert "1 round in which the order did not change" not in line
    # The floor is not an observation about the final round either.
    assert "in the final round only" not in line


def test_movement_wider_than_a_gap_in_the_standings_says_which_positions_it_covers():
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381),
        # Gaps of 56 and 12; the final round moved a rating by 46, so one of the two.
        [
            _brief("A", [], elo=1290.0),
            _brief("B", [], elo=1234.0),
            _brief("C", [], elo=1222.0),
        ],
    )
    assert "further than one of the two gaps" in line


def test_every_gap_cleared_is_reported_as_all_of_them():
    """ "further than six of the six gaps" is a construction nobody writes."""
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence

    line = _convergence(
        TournamentState(score_movement=0.0381),
        [_brief("A", [], elo=1290.0), _brief("B", [], elo=1289.0)],
    )
    assert "further than all one gap" not in line
    assert "further than one of the one gaps" not in line


def test_an_unmeasured_rating_movement_is_not_printed_as_a_rating_that_moved():
    """A tournament that measured nothing reported moving a rating by the whole 1200.

    ``score_movement`` defaults to and falls back to 1.0, which is a "not measured"
    sentinel and not a fraction. Multiplied out, a live report said "the final round
    moved one rating by 100.0 per cent of that, or about 1200 points" -- two
    paragraphs after reporting that no match in the run moved one by more than
    sixteen -- and then called every position in the standings provisional on it.
    """
    from coscientist.dossier import _tournament_facts
    from coscientist.models import TournamentState
    from coscientist.narrative import _convergence
    from coscientist.parity import UNMEASURED_MOVEMENT

    briefs = [
        _brief("A", [], elo=1290.0),
        _brief("B", [], elo=1234.0),
        _brief("C", [], elo=1222.0),
    ]
    line = _convergence(
        TournamentState(score_movement=UNMEASURED_MOVEMENT, ranking_stable_rounds=1),
        briefs,
    )
    assert "1200 points" not in line
    assert "100.0 per cent" not in line
    # Nor as a rating that measurably held still: "recorded no rating change" reads as
    # a movement of zero, which would satisfy the second half of the convergence rule
    # rather than leave it untested.
    assert "did not record how far its final round moved the ratings" in line
    assert "no rating change" not in line
    # Nothing was measured, so nothing about the gaps follows from it either.
    assert "gap" not in line

    facts = _tournament_facts(
        SimpleNamespace(
            swiss_rounds=1,
            top_round_robin_size=4,
            converged=False,
            score_movement=UNMEASURED_MOVEMENT,
            comparisons=[
                SimpleNamespace(round_number=1, candidate_a_id="a", candidate_b_id="b")
            ],
        )
    )
    assert not any("1200 points" in fact for fact in facts)
    assert any("no rating change was recorded" in fact for fact in facts)


def test_a_constraint_reviewers_wrote_about_is_not_reported_as_unchecked():
    """Section 1 said none of them checked a constraint; section 6 then reported a breach."""
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    # A review counts as reachable only under an idea the report still prints.
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="methods_feasibility",
                    objections=[
                        "No uncoated control cells are described for the comparison."
                    ],
                )
            ]
        )
    ]

    line = _constraint_coverage(
        record,
        [
            "Must include uncoated control cells for direct comparison",
            "Must specify exact charge/discharge rates, voltage windows, and temperature",
        ],
    )
    assert "constraint one, in one review under Feasibility" in line
    assert "reaches constraint two" in line
    assert "none of them writes about any of the constraints" not in line.lower()


def test_a_withdrawn_ideas_reviews_are_not_counted_as_printed_under_an_idea():
    """A withdrawn idea keeps its reviews and loses its section, so the counts part.

    A live run that withdrew one of eight ideas opened this paragraph with "what the
    record does hold is 40 reviews of the ideas, printed in full under each", above 35
    printed reviews. Any sentence that tells the reader where to look has to count what
    is in that place.
    """
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="methods_feasibility",
                    objections=["No uncoated control cells are described."],
                ),
                # Withdrawn before the report was written: no section carries it.
                _review(
                    "cand_gone",
                    criterion="methods_feasibility",
                    objections=["No uncoated control cells are described."],
                ),
            ]
        )
    ]

    line = _constraint_coverage(
        record, ["Must include uncoated control cells for direct comparison"]
    )
    assert "one review of the ideas, printed in full under each" in line
    assert "two reviews" not in line


def test_one_shared_word_does_not_count_as_reaching_a_constraint():
    """ "Temperature" appears in a review of every idea and says nothing about the rule."""
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    # A review counts as reachable only under an idea the report still prints.
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="impact_safety",
                    findings=["Elevated temperature accelerates the side reaction."],
                )
            ]
        )
    ]

    line = _constraint_coverage(
        record,
        ["Must specify exact charge/discharge rates, voltage windows, and temperature"],
    )
    assert "Not one of them writes about any of the constraints" in line


def _planned(**kwargs) -> ResearchRecord:
    from coscientist.models import ResearchPlan

    fields = {
        "question": "Can a coating help?",
        "intended_claim": "hypothesis",
    }
    fields.update(kwargs)
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.plan = ResearchPlan(**fields)
    return record


def test_a_stopping_rule_is_not_left_to_the_word_budget():
    """The budget dropped the rule that stops the work on a thermal runaway."""
    from coscientist.narrative import _section_one

    draft = _section_one(
        _planned(stopping_criteria=["Thermal runaway or venting of any test cell"])
    )

    assert any("Thermal runaway" in paragraph for paragraph in draft.core)
    assert not any("Thermal runaway" in paragraph for paragraph in draft.extra)


def test_the_question_the_run_executed_is_printed_when_it_is_not_the_goal():
    """Scoping narrows the goal, and only the goal was ever on the page."""
    from coscientist.narrative import _section_one

    record = _planned(
        question="Does a cathode-surface coating raise cycles to 80% retention "
        "against an uncoated control?"
    )

    draft = _section_one(record)

    assert any("uncoated control" in paragraph for paragraph in draft.core)


def test_a_goal_scoping_did_not_narrow_is_not_printed_twice():
    from coscientist.narrative import _section_one

    draft = _section_one(_planned(question="Can a coating help?"))

    assert sum("Can a coating help?" in item for item in draft.core) == 1


def test_labelled_criteria_are_set_as_a_list_not_as_a_semicolon_chain():
    """Seven Label: value items ran together into one two-hundred-word sentence."""
    from coscientist.narrative import _labelled_bullets

    text = _labelled_bullets(
        ["Evidence: Alignment with the provided claims", "Safety: Thermal runaway risk"]
    )

    assert text == (
        "- **Evidence** — Alignment with the provided claims.\n"
        "- **Safety** — Thermal runaway risk."
    )


def test_an_unlabelled_criterion_is_still_a_bullet():
    from coscientist.narrative import _labelled_bullets

    assert _labelled_bullets(["It has to beat the uncoated control"]) == (
        "- It has to beat the uncoated control."
    )


def test_a_colon_inside_a_sentence_is_not_mistaken_for_a_label():
    """ "The rule is this: ..." is prose, and bolding its first six words is not a label."""
    from coscientist.narrative import _labelled_bullets

    text = _labelled_bullets(["What the run has to show is this: a real difference"])

    assert text.startswith("- What the run")
    assert "**" not in text


def test_a_pair_of_titles_is_joined_without_the_list_semicolon():
    """The semicolon punctuates a pair as though a third item were coming."""
    assert _joined_titles(["One", "Two"]) == "One and Two"
    assert _joined_titles(["One", "Two", "Three"]) == "One; Two; and Three"


def test_one_match_moves_both_sides_by_the_same_number_of_points():
    """Rounding each endpoint separately gave the winner +13 and the loser -14."""
    from coscientist.narrative import IdeaMatch

    def match(before: float, after: float) -> IdeaMatch:
        return IdeaMatch(
            round_number=1,
            opponent_title="Other",
            outcome="win",
            elo_before=before,
            elo_after=after,
            confidence=0.7,
            rationale="",
            judge="llm",
        )

    winner = match(1200.4, 1213.9)
    loser = match(1199.6, 1186.1)

    assert winner.swing == -loser.swing == 14
    # The printed endpoints reach the printed swing, rather than being a third
    # independent rounding that the arithmetic on the page does not support.
    assert winner.shown_after - winner.shown_before == winner.swing
    assert loser.shown_after - loser.shown_before == loser.swing


def test_a_shortlist_cut_inside_a_tie_says_the_last_place_was_a_tie_break():
    from coscientist.narrative import _shortlist_caveats

    briefs = [
        _brief("Leader", [], elo=1300, shortlisted=True),
        _brief("Included", [], elo=1184, shortlisted=True),
        _brief("Excluded", [], elo=1184),
        _brief("Below", [], elo=1100),
    ]

    caveats = _shortlist_caveats([briefs[0], briefs[1]], briefs)
    assert len(caveats) == 1
    assert "The cut fell inside a tie." in caveats[0]
    assert "Included made the shortlist on 1184" in caveats[0]
    assert "Excluded finished on the same 1184 and did not" in caveats[0]


def test_a_clean_cut_draws_no_tie_caveat():
    from coscientist.narrative import _shortlist_caveats

    briefs = [
        _brief("Leader", [], elo=1300, shortlisted=True),
        _brief("Below", [], elo=1100),
    ]

    assert _shortlist_caveats([briefs[0]], briefs) == []


def test_a_shortlisted_idea_carrying_a_rejection_is_reconciled_on_the_page():
    """Nothing in the pipeline reads the recommendations, and nothing said so."""
    from coscientist.narrative import _shortlist_caveats

    rejected = _brief(
        "Rejected", [_idea_review(recommendation="reject")], elo=1300, shortlisted=True
    )
    caveats = _shortlist_caveats([rejected], [rejected])

    assert len(caveats) == 1
    assert "Rejected was shortlisted while carrying a reviewer's" in caveats[0]
    assert "nothing reconciled them" in caveats[0]


def test_a_citation_that_argues_against_its_own_idea_is_declared():
    """A candidate lists its evidence without saying which way each piece cuts."""
    from coscientist.models import EvidenceClaim, EvidencePacket
    from coscientist.narrative import _contradicting_claims

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        claims=[
            EvidenceClaim(
                id="claim_1",
                source_id="src_1",
                claim="Thick coatings reduce ionic conductivity",
                relation="contradicts",
            ),
            EvidenceClaim(
                id="claim_2",
                source_id="src_1",
                claim="Thin coatings extend cycle life",
                relation="supports",
            ),
        ],
    )
    candidate = _candidate("cand_a")
    candidate.evidence_ids = ["claim_1", "claim_2"]

    assert _contradicting_claims(record, candidate) == [
        "Thick coatings reduce ionic conductivity."
    ]


def test_a_reviewers_evaluation_table_is_printed_as_a_table():
    """The discipline critics answer the findings field with a Markdown table under
    "**Structured Evaluation Table:**". Fourteen findings on a live run carried one,
    and the flattener read each row out as a run of clauses -- "Aggregation Control
    (Description: ALD on pre-fabricated electrodes prevents agglomeration; Judgment:
    High)" -- in the one section a reader consults to find what was wrong with an
    idea."""
    from coscientist.dossier import _review_finding_tables
    from coscientist.narrative import IdeaReview, _review_findings

    split = _review_findings(
        [
            "**Structured Evaluation Table:**\n"
            "| Category | Description | Judgment |\n"
            "|---|---|---|\n"
            "| Aggregation Control | ALD prevents agglomeration. | High |\n"
            "| Purity Potential | ALD provides exceptional purity. | High |",
            "**Critical Scientific Judgment:** The coating may act as an insulator.",
            "TOF-SIMS is appropriate for detecting AlO- fragments.",
        ]
    )

    # The label over nothing but a table becomes that table's heading rather than a
    # paragraph of its own, and a label over prose is set the way the report sets
    # every other label a specialist wrote.
    assert split["findings"] == [
        "**Critical Scientific Judgment.** The coating may act as an insulator.",
        "TOF-SIMS is appropriate for detecting AlO- fragments.",
    ]
    assert split["finding_tables"] == [
        (
            "Structured Evaluation Table",
            [
                ["Category", "Description", "Judgment"],
                ["Aggregation Control", "ALD prevents agglomeration.", "High"],
                ["Purity Potential", "ALD provides exceptional purity.", "High"],
            ],
        )
    ]

    lines = _review_finding_tables(
        IdeaReview(
            section="Correctness",
            lead_in="Reviewer:",
            question="Is it correct?",
            objections=[],
            rebuttals=[],
            answer="Yes.",
            score=4,
            **split,
        )
    )
    assert lines[0] == "**Structured Evaluation Table.**"
    assert lines[2] == "| Category | Description | Judgment |"
    assert lines[4] == "| Aggregation Control | ALD prevents agglomeration. | High |"


def test_a_statement_is_badged_by_the_weakest_record_it_names():
    """The badge was the best standing among the records a statement names, so a live
    bullet read "**[Verified Source]** Al2O3 coatings improve capacity retention (the
    unverified claim drawn from ..., the claim drawn from ...)" -- asserting a check
    of two records in the label and naming one of them as unchecked in the same line.
    The same rule would have called a statement verified over a retracted record cited
    beside a sound one."""
    from coscientist.narrative import (
        DISCREDITED_BADGE,
        LEAD_BADGE,
        VERIFIED_BADGE,
        _EvidenceRecord,
        _grounding_badge,
    )

    checked = _EvidenceRecord("claim_1", "Retention improves", "verified")
    unchecked = _EvidenceRecord(
        "claim_2", "Retention improves", "discovered_unverified"
    )
    pulled = _EvidenceRecord("claim_3", "Retention improves", "retracted")

    assert _grounding_badge("s", set(), set(), [checked]) == VERIFIED_BADGE
    assert _grounding_badge("s", set(), set(), [checked, unchecked]) == LEAD_BADGE
    assert _grounding_badge("s", set(), set(), [checked, pulled]) == DISCREDITED_BADGE


def test_an_evidence_bullet_that_is_only_an_id_says_so():
    """ "- **[Unsourced claim]** stmt5." was the whole of an evidence bullet on the
    live run, eight times across four ideas. The resolver reads a token as a record
    id only where it carries an underscore, and the specialists wrote "stmt5" and
    "stmt3" where the base holds "stmt_5" -- so the id was neither read out, nor
    named by the invented-citation audit, which reads the same pattern."""
    from coscientist.models import EvidenceClaim, EvidencePacket
    from coscientist.narrative import _evidence_notes, _invented_ids

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        claims=[EvidenceClaim(id="stmt_5", claim="Thin coatings extend cycle life")],
    )
    candidate = _candidate("cand_a")
    candidate.evidence_for = [
        "stmt5",
        "Al2O3 coatings suppress LiPF6 hydrolysis in NCM811 cells",
        "stmt_5",
    ]

    stated = [text for _, _, text in _evidence_notes(record, candidate)]
    assert "stmt5" in stated[0]
    assert "no record of that id exists in this run's evidence base" in stated[0]
    # A formula is the same shape as an id and is what the ideas are about, so the
    # letters have to be the leading word of an id this run actually recorded.
    assert stated[1] == "Al2O3 coatings suppress LiPF6 hydrolysis in NCM811 cells."
    # An id that does resolve is still read out rather than printed.
    assert stated[2] == "Thin coatings extend cycle life."

    assert _invented_ids(record, candidate, None) == ["stmt5"]


def test_an_idea_citing_sources_is_not_told_it_cites_no_evidence():
    """ "No finding in this report's evidence is cited for this idea." was printed
    under two live ideas that cite a source and a neutral claim apiece, every id of
    them resolving, with the same sources listed by name in their Evidence
    Assessment. An ``evidence_ids`` entry names a claim, a source, a discovery lead
    or a discovery statement -- and only a claim carries a relation, whose default is
    ``neutral``, so reading the supporting claims alone found nothing at all."""
    from coscientist.models import EvidenceClaim, EvidencePacket, SourceRecord
    from coscientist.narrative import _cited_evidence, _motivation

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(
                id="src_1",
                url="https://example.org/alumina",
                title="Atomic layer deposition of alumina",
                verification_status="verified",
            )
        ],
        claims=[
            EvidenceClaim(
                id="claim_1",
                source_id="src_1",
                claim="Coating uniformity varies with precursor dose",
            )
        ],
    )
    candidate = _candidate("cand_a")
    candidate.evidence_ids = ["src_1", "claim_1"]

    cited = _cited_evidence(record, candidate)
    assert cited.supports == []
    # The source itself and the claim the extraction stage left at its neutral
    # default: both are cited, neither is support, and the report has to say so.
    assert cited.undirected == [
        "Atomic layer deposition of alumina.",
        "Coating uniformity varies with precursor dose.",
    ]

    motivation = _motivation({"Core idea": "A coating helps."}, cited)
    assert "No finding in this report's evidence is cited" not in motivation
    assert "with no direction recorded either way" in motivation
    assert "Atomic layer deposition of alumina" in motivation

    # An idea whose every citation cuts the other way is told that, rather than
    # being told it cites nothing.
    against = _candidate("cand_b")
    against.evidence_ids = ["claim_2"]
    record.evidence.claims.append(
        EvidenceClaim(
            id="claim_2",
            source_id="src_1",
            claim="Thick coatings reduce ionic conductivity",
            relation="contradicts",
        )
    )
    only_against = _motivation({}, _cited_evidence(record, against))
    assert "cutting against the research question rather than for it" in only_against

    # And an idea that really cites nothing still gets the plain sentence.
    assert "No finding in this report's evidence is cited" in _motivation(
        {}, _cited_evidence(record, _candidate("cand_c"))
    )


CUTS_AGAINST = (
    "Even coatings of one nanometre were detrimental to the cycling performance "
    "of LNMO."
)


# Every optional field left at its unstated fallback, so the chapters print only
# what these tests are about.
FACTS = {"Core idea": "A coating helps.", **_UNSTATED}


def _section_four_text(briefs) -> tuple[str, list[str]]:
    from coscientist.narrative import _section_four

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    four = _section_four(record, briefs)
    return " ".join(four.core), [
        paragraph
        for subsection in four.subsections
        for paragraph in subsection.paragraphs
    ]


def test_a_finding_that_cuts_against_more_than_one_idea_is_stated_once():
    """Ninety identical words -- the finding and the two sentences that read it --
    stood under three of the eight chapters of a live report."""
    briefs = [
        _brief(
            name,
            [],
            facts=FACTS,
            contradicting_claims=[CUTS_AGAINST] if shared else None,
        )
        for name, shared in (("Alpha", True), ("Beta", True), ("Gamma", False))
    ]

    core, chapters = _section_four_text(briefs)

    assert core.count(CUTS_AGAINST) == 1
    assert "Cited by Alpha and Beta." in core
    assert not any(CUTS_AGAINST in paragraph for paragraph in chapters)
    assert sum("stated at the head of this section" in p for p in chapters) == 2


def test_a_finding_that_cuts_against_one_idea_stays_under_that_idea():
    briefs = [
        _brief("Alpha", [], facts=FACTS, contradicting_claims=[CUTS_AGAINST]),
        _brief("Beta", [], facts=FACTS),
    ]

    core, chapters = _section_four_text(briefs)

    assert CUTS_AGAINST not in core
    assert any(CUTS_AGAINST in paragraph for paragraph in chapters)
    assert not any("stated at the head of this section" in p for p in chapters)


def test_an_idea_citing_both_a_shared_finding_and_its_own_says_both():
    own = "Coatings thicker than five nanometres raise the overpotential."
    briefs = [
        _brief("Alpha", [], facts=FACTS, contradicting_claims=[CUTS_AGAINST, own]),
        _brief("Beta", [], facts=FACTS, contradicting_claims=[CUTS_AGAINST]),
    ]

    core, chapters = _section_four_text(briefs)

    assert core.count(CUTS_AGAINST) == 1
    assert own not in core
    alpha = " ".join(chapters)
    assert own in alpha
    # The chapter counts only the findings it prints, not the hoisted one as well.
    assert "One claim this idea cites" in alpha
    assert "One finding this idea cites" in alpha


def test_a_count_that_opens_a_sentence_is_spelled_out():
    """ "3 objections recurred" reads as a list item that lost its bullet."""
    from coscientist.narrative import _opening, _placed

    assert _opening(3, "objection") == "Three objections"
    assert _opening(1, "review") == "One review"
    # The table stopped at twelve, which is where house style stops spelling counts
    # out mid-sentence -- but that put "15 reviews closed at two or below" at the head
    # of a paragraph on a live run. A count that opens a sentence is a word at any
    # size; only figures over nine hundred and ninety-nine are left as they are.
    assert _opening(15, "review") == "Fifteen reviews"
    assert _opening(41, "objection") == "Forty-one objections"
    assert _opening(102, "objection") == "One hundred and two objections"
    # Mid-sentence the word is lower-cased; only the opening clause capitalises it.
    assert _placed(list("ABCD"), 5, "highest", opening=True).startswith("Four of")
    assert _placed(list("ABCD"), 5, "lowest").startswith("four of")


def test_a_reviewer_recorded_fatal_flaw_is_printed_somewhere():
    """The finding a reviewer called disqualifying reached the reader nowhere.

    Critical Flaws keyed off the score, and a score is what a fatal flaw causes rather
    than what it is; Deep Verification listed the objections and dropped the flaw. So
    the strongest thing said about an idea in the whole run was the one thing the
    report would not print.
    """
    from coscientist.narrative import (
        DEEP_VERIFICATION_FATAL_LEAD_IN,
        _deep_verification,
        _summary_sections,
    )

    flawed = _idea_review(
        section="Correctness",
        score=2,
        objections=["The loading is not stated."],
        fatal_flaws=["The cited source measures a different chemistry."],
    )
    lead_in, checks = _deep_verification([flawed], {})
    assert lead_in == DEEP_VERIFICATION_FATAL_LEAD_IN
    assert checks[0] == (
        "Fatal flaw recorded by the correctness review",
        "The cited source measures a different chemistry.",
    )
    # The objections keep their place behind it rather than being displaced by it.
    assert checks[1][1] == "The loading is not stated."

    critical = _summary_sections(
        _facts(), [flawed], rank=1, elo=1200, shortlisted=False
    )["Critical Flaws"]
    assert "recorded a fatal flaw against this idea" in critical
    assert "Deep Verification below" in critical


def test_an_objection_opening_on_a_measurement_is_carried_by_a_lead_in():
    """ "15 nm might still be thin enough…" set as a paragraph of its own.

    Prose does not open a sentence on a numeral, and this one is the specialist's
    sentence about a measurement: the number cannot be spelled the way a count the
    report wrote is, and putting a noun in front of it -- "A 15 nm layer might still
    be" -- would be the report deciding what the reviewer meant.
    """
    from coscientist.narrative import _deep_verification

    _, checks = _deep_verification(
        [
            _idea_review(
                section="Impact",
                score=2,
                objections=["15 nm might still be thin enough to allow tunneling."],
                fatal_flaws=["2 nm is below the reproducible deposition limit."],
            )
        ],
        {},
    )
    assert checks[0][1] == (
        "The flaw is that 2 nm is below the reproducible deposition limit."
    )
    assert checks[1][1] == (
        "The objection is that 15 nm might still be thin enough to allow tunneling."
    )

    # A sentence that does not open on a numeral is printed as the reviewer wrote it.
    _, plain = _deep_verification(
        [_idea_review(section="Impact", score=2, objections=["No reactor is named."])],
        {},
    )
    assert plain[0][1] == "No reactor is named."


def test_a_flaw_and_the_score_it_caused_are_not_reported_as_two_findings():
    """A fatal flaw caps its review at two, so the cap is the flaw restated.

    Printing both put "the correctness review recorded a fatal flaw" beside "the
    correctness review scored this idea at or below two of five" and invited the
    reader to count two reviewers' worth of trouble where there was one.
    """
    from coscientist.narrative import _summary_sections

    reviews = [
        _idea_review(
            section="Correctness",
            score=2,
            fatal_flaws=["The rationale does not hold."],
        ),
        _idea_review(section="Feasibility", score=2, objections=["No reactor."]),
    ]
    critical = _summary_sections(
        _facts(), reviews, rank=1, elo=1200, shortlisted=False
    )["Critical Flaws"]
    assert "The correctness review recorded a fatal flaw" in critical
    assert "The feasibility review scored this idea at or below two" in critical
    assert "The correctness and feasibility reviews scored" not in critical


def test_one_flaw_reached_by_two_reviews_is_one_finding():
    """Printed under both headings it reads as independent corroboration."""
    from coscientist.narrative import _deep_verification

    flaw = "The cited source measures a different chemistry."
    _, checks = _deep_verification(
        [
            _idea_review(section="Correctness", score=2, fatal_flaws=[flaw]),
            _idea_review(section="Novelty", score=2, fatal_flaws=[flaw]),
        ],
        {},
    )
    assert len(checks) == 1
    assert checks[0][0] == "Fatal flaw recorded by the correctness review"


def test_what_an_untested_objection_is_worth_is_said_once_over_all_of_them():
    """The reading guide states it above the ideas, over every list it is true of.
    Critical Flaws restated it verbatim under six of eight ideas -- forty words each,
    and none of the six copies said anything the guide had not."""
    from coscientist.narrative import DEEP_DIVE_PREAMBLE, _summary_sections

    guide = " ".join(DEEP_DIVE_PREAMBLE)
    assert "Nothing in this run tested any item in those lists" in guide

    critical = _summary_sections(
        _facts(),
        [
            _idea_review(section="Impact", score=2, objections=["The gain is small."]),
            _idea_review(section="Correctness", score=5),
        ],
        rank=1,
        elo=1200,
        shortlisted=False,
    )["Critical Flaws"]
    # Which review scored it down is this idea's fact and stays here. Where to read
    # the objection is a pointer, which is short; what the objection is worth untested
    # is the guide's sentence and is not printed a second time.
    assert critical.startswith("The impact review scored this idea at or below two")
    assert "What it objected to is set out under Deep Verification below." in critical
    assert "unresolved rather than established" not in critical
    assert "Nothing else in this run has tested" not in critical


def test_a_recorded_fatal_flaw_does_not_restate_the_guides_sentence_either():
    from coscientist.narrative import _summary_sections

    critical = _summary_sections(
        _facts(),
        [_idea_review(section="Correctness", score=1, fatal_flaws=["No such cell."])],
        rank=1,
        elo=1200,
        shortlisted=False,
    )["Critical Flaws"]
    assert (
        "printed in full under Deep Verification below, and no reviewer withdrew it"
        in critical
    )
    assert "Nothing in this run tested it" not in critical


def test_no_flaw_and_no_low_score_still_says_what_that_is_worth():
    """The clean case has to stay a statement about the reviews, not about the idea."""
    from coscientist.narrative import _summary_sections

    critical = _summary_sections(
        _facts(), [_idea_review(score=4)], rank=1, elo=1200, shortlisted=True
    )["Critical Flaws"]
    assert critical.startswith("No reviewer recorded a fatal flaw")


def test_inputs_the_rewrite_added_are_not_denied_by_the_reviewed_form():
    """ "No input or dependency was recorded for it" stood some thirty lines under a
    Revised Form block listing two of them. True of the reviewed form, and read as a
    claim about the idea by anyone who had just scrolled past the list."""
    from coscientist.narrative import _summary_sections

    bare = dict(_facts())
    bare["Required inputs and dependencies"] = _UNSTATED[
        "Required inputs and dependencies"
    ]

    feasibility = _summary_sections(
        bare,
        [_idea_review(section="Feasibility", score=4)],
        revised=_facts(),
        rank=1,
        elo=1200,
        shortlisted=True,
    )["Feasibility Assessment (Go/No-Go Decision)"]

    assert "No input or dependency was recorded for the form these reviews" in (
        feasibility
    )
    assert "Revised Form above does list what it would need" in feasibility


def test_an_idea_with_no_inputs_on_either_form_still_says_none_were_recorded():
    from coscientist.narrative import _summary_sections

    bare = dict(_facts())
    bare["Required inputs and dependencies"] = _UNSTATED[
        "Required inputs and dependencies"
    ]

    feasibility = _summary_sections(
        bare,
        [_idea_review(section="Feasibility", score=4)],
        revised=bare,
        rank=1,
        elo=1200,
        shortlisted=True,
    )["Feasibility Assessment (Go/No-Go Decision)"]

    assert "No input or dependency was recorded for it, so nothing here states" in (
        feasibility
    )


def _nine(record: ResearchRecord, briefs: list[IdeaBrief]) -> str:
    from coscientist.narrative import _section_nine

    return " ".join(_section_nine(record, briefs).core)


def _twice_revised() -> ResearchRecord:
    """An idea evolution rewrote in two rounds, the first round the material one."""
    from coscientist.models import CandidatePopulation, DossierManifest

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.titles = {"cand_a": "A TiO2 coating"}
    second = _candidate("cand_a_v2", version=2, parents=["cand_a"])
    second.go_no_go_tests = ["Confirm loading within five per cent by ICP-OES."]
    third = _candidate("cand_a_v3", version=3, parents=["cand_a_v2"])
    third.go_no_go_tests = ["Confirm loading within five per cent by ICP-OES."]
    record.evolution = EvolutionCycle(
        records=[
            EvolutionRecord(
                parent_ids=["cand_a"],
                candidate=second,
                round_number=1,
                changes=["Changed TiO2 to Al2O3 to match the baseline material."],
                critiques_addressed=["the wrong baseline material"],
                new_prediction="Retention improves by ten points.",
            ),
            EvolutionRecord(
                parent_ids=["cand_a_v2"],
                candidate=third,
                round_number=2,
                changes=["Specified H14-grade HEPA filtration."],
                critiques_addressed=["the inhalation risk"],
                new_prediction="Retention improves by ten points.",
            ),
        ]
    )
    _trace_lineage(record, {"cand_a"})
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["cand_a"]
    )
    return record


def test_the_change_log_covers_every_round_of_a_rewrite_not_the_last_one():
    """A two-round rewrite was described by its smaller half.

    Round one changed the coating material and round two added a filter specification;
    the recommendation reported only round two, so the reader was told the recommended
    rewrite was a filtration tweak on the idea that had been ranked.
    """
    core = _nine(
        _twice_revised(),
        [
            _brief(
                "A TiO2 coating",
                [],
                facts=_facts(),
                candidate_id="cand_a",
                # The change log is only promised where the idea's own section will
                # carry the rewrite, which is what this brief says it does.
                revised_form=[("Claim", "An Al2O3 coating raises retention.")],
            )
        ],
    )
    assert "TiO2 to Al2O3 to match the baseline material" in core
    assert "H14-grade HEPA filtration" in core


def test_a_rewrite_made_over_two_rounds_is_not_attributed_to_one():
    """The diff is against the ranked form, so it spans every round that touched it."""
    from coscientist.narrative import _revised_form

    lead_in, _, _ = _revised_form(_twice_revised(), _candidate("cand_a"))
    assert "in rounds one and two" in lead_in
    assert "the wrong baseline material" in lead_in
    assert "the inhalation risk" in lead_in
    # The critiques run to a four-item semicolon series on a live run, and closing it
    # with ", and this is the cumulative result" hung the point of the sentence off
    # the end of a list the reader was still working through.
    assert "and this is the cumulative result. The rounds addressed" in lead_in
    critiques = lead_in.split("The rounds addressed, between them, ")[1]
    assert critiques.split(". ")[0].endswith("the inhalation risk")


def test_a_critique_recorded_with_its_remedy_stays_inside_the_sentence():
    """Evolution records what it addressed as "Missing Structured Evaluation Table:
    Added to mechanism_model", and "to address" in front of four of those produced a
    sentence that stopped at the first colon and three more starting at a remedy."""
    from coscientist.narrative import _revised_form

    record = _twice_revised()
    record.evolution.records[0].critiques_addressed = [
        "Missing Structured Evaluation Table: Added to mechanism_model",
        "Incomplete Synthetic Routes: Specified ALD conditions",
    ]
    record.evolution.records[1].critiques_addressed = []

    lead_in, _, _ = _revised_form(record, _candidate("cand_a"))

    assert (
        "The rounds addressed, between them, missing Structured Evaluation Table — "
        "added to mechanism model; and incomplete Synthetic Routes — specified ALD "
        "conditions." in lead_in
    )
    assert "mechanism_model" not in lead_in
    assert ": Added to" not in lead_in


def test_a_rewrite_nobody_recommends_is_not_headed_as_a_recommendation():
    """Evolution rewrites the whole shortlist; the meta-review recommends a subset.

    The section was headed "Revised Form Recommended" and opened "This is the form the
    meta-review recommends" under every rewritten idea, so the run where every
    candidate carried a fatal flaw said no idea cleared the bar in section 9 and told
    four ideas' readers the opposite in their own sections.
    """
    from dataclasses import replace

    from coscientist.dossier import _revised_form_block
    from coscientist.narrative import _revised_form

    record = _twice_revised()
    candidate = _candidate("cand_a")
    brief = _brief("A TiO2 coating", [], facts=_facts(), candidate_id="cand_a")

    carried, changed, _ = _revised_form(record, candidate, recommended=True)
    assert changed, "the fixture must give the idea a rewrite to print"
    assert carried.startswith("This is the form the meta-review recommends, and it is")
    block = "\n".join(
        _revised_form_block(
            replace(
                brief,
                revised_lead_in=carried,
                revised_form=changed,
                revised_is_recommended=True,
            )
        )
    )
    assert block.startswith("### Revised Form Recommended\n")

    dropped, changed, _ = _revised_form(record, candidate, recommended=False)
    assert dropped.startswith(
        "This is the form evolution produced, which the meta-review does not "
        "recommend, and it is"
    )
    block = "\n".join(
        _revised_form_block(
            replace(brief, revised_lead_in=dropped, revised_form=changed)
        )
    )
    assert block.startswith("### Revised Form\n")


def test_the_go_no_go_the_reader_is_sent_to_run_belongs_to_the_recommended_form():
    """It quoted the ranked candidate's test two paragraphs after saying the revised
    form is what is recommended, and the rewrite had changed the test."""
    core = _nine(
        _twice_revised(),
        [
            _brief(
                "A TiO2 coating",
                [],
                facts=_facts(),
                candidate_id="cand_a",
                # The change log is only promised where the idea's own section will
                # carry the rewrite, which is what this brief says it does.
                revised_form=[("Claim", "An Al2O3 coating raises retention.")],
            )
        ],
    )
    assert "Confirm loading within five per cent by ICP-OES" in core
    assert "as the rewrite recommended below states it" in core
    assert "Thickness within two nanometres by TEM" not in core


def test_a_rank_the_tournament_did_not_decide_is_not_reported_as_a_result():
    """Three live ideas held an Elo of 1184 and were printed as ranks four, five and
    six, each in its own subsection, with the tie disclosed a section away."""
    import dataclasses

    from coscientist.narrative import _section_four

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    tied = dataclasses.replace(
        _brief("A coating", [], elo=1184.0, facts=_facts()),
        rank=5,
        tied_with=2,
        predictions=["Retention improves by ten points."],
    )
    clear = dataclasses.replace(
        _brief("Another coating", [], elo=1290.0, facts=_facts()),
        predictions=["Retention improves by twenty points."],
    )

    four = " ".join(
        paragraph
        for section in _section_four(record, [clear, tied]).subsections
        for paragraph in section.paragraphs
    )
    assert "It finished level with two other ideas on an Elo of 1184" in four
    assert "shares position 5 with them" in four
    assert "It finished rank 5 on an Elo" not in four
    # The idea that shares its rating with nobody still states the rank plainly.
    assert "It finished rank 1 on an Elo of 1290." in four


def test_the_prose_above_a_grid_points_forward_to_it_and_not_back():
    """ "Beyond the prediction in the grid, <twenty-word title> is separated from…"

    Two defects in one clause: the grid is printed below this paragraph, not above it,
    and the subject was the idea's full title set two lines under a heading that is the
    same title, where every other sentence in the paragraph already says "It".
    """
    import dataclasses

    from coscientist.narrative import _section_four

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    title = (
        "A Self-healing Polyurethane-based Interphase Coating Containing "
        "Microencapsulated Lithium Salts"
    )
    several = dataclasses.replace(
        _brief(title, [], facts=_facts()),
        predictions=["It survives ten cycles.", "Retention improves by ten points."],
    )
    lone = dataclasses.replace(
        _brief("A coating", [], facts=_facts()),
        predictions=["Retention improves by ten points."],
    )

    four = " ".join(
        paragraph
        for section in _section_four(record, [several, lone]).subsections
        for paragraph in section.paragraphs
    )
    assert "Beyond the prediction in the grid below, it is separated from its" in four
    assert "It rests on the single prediction in the grid below and records no other."
    assert title not in four
    assert "in the grid," not in four

    lead_in = _section_four(record, [several, lone]).core[0]
    # The five rows are five items, and only the last pair went uncommaed.
    assert "the dependency it turns on, and its principal risk" in lead_in
    assert "The prose above each grid" in lead_in


def _recommended_pair(claim_status: str) -> ResearchRecord:
    """Two recommended ideas: the leading one cites nothing, the other cites a claim."""
    from coscientist.citations import CandidateCitations, Citation
    from coscientist.models import (
        CandidatePopulation,
        DossierManifest,
        EvidenceClaim,
        TournamentState,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a"), _candidate("cand_b")]
    )
    record.titles = {"cand_a": "An uncited conjecture", "cand_b": "A cited idea"}
    record.tournament = TournamentState(ratings={"cand_a": 1240.0, "cand_b": 1180.0})
    record.evidence_support = {
        "cand_a": CandidateCitations(candidate_id="cand_a", citations=[]),
        "cand_b": CandidateCitations(
            candidate_id="cand_b",
            citations=[
                Citation(
                    reference="claim_1",
                    claim=EvidenceClaim(
                        id="claim_1",
                        source_id="src_1",
                        claim="Thin coatings extend cycle life",
                        relation="supports",
                        verification_status=claim_status,
                    ),
                )
            ],
        ),
    }
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["cand_a", "cand_b"]
    )
    return record


def test_the_recommendation_carries_the_grounding_the_appendix_records():
    """A reader acting on the recommendation alone was not told what it rests on.

    The appendix said the leading recommended idea cites no evidence at all and that
    its claim is a conjecture; this section's only qualifications were protocol
    drafting and outside review, so the reader had to reach the appendix to learn the
    top of the recommendation was unevidenced.
    """
    core = _nine(
        _recommended_pair("discovered_unverified"),
        [_brief("An uncited conjecture", [], facts=_facts(), candidate_id="cand_a")],
    )
    assert "None of the two rests on verified evidence." in core
    assert "An uncited conjecture, which leads the recommendation, cites none" in core
    assert "Evidence integrity in the appendix" in core


def test_a_recommendation_resting_on_checked_evidence_is_not_qualified_as_if_it_did_not():
    """The clause states the run's grounding, so a verified run has to read as one."""
    core = _nine(
        _recommended_pair("verified"),
        [_brief("An uncited conjecture", [], facts=_facts(), candidate_id="cand_a")],
    )
    # cand_a still cites nothing, so this is the mixed case, not the clean one.
    assert "One of the two rests on verified evidence and the other does not." in core
    assert "None of the two" not in core


def test_a_recommendation_with_no_citation_resolution_makes_no_grounding_claim():
    """Without resolution there is no verdict to report, and silence is the verdict."""
    record = _recommended_pair("discovered_unverified")
    record.evidence_support = {}
    core = _nine(
        record,
        [_brief("An uncited conjecture", [], facts=_facts(), candidate_id="cand_a")],
    )
    assert "rests on verified evidence" not in core
    assert "Evidence integrity in the appendix" not in core


def test_the_rereview_count_and_its_verdict_are_not_run_together():
    """ "Eight re-reviews every one of which returned advance" read as a restriction.

    Unpunctuated, the clause picks out which re-reviews are meant, implying there were
    others that returned something else. There is only the one set, so it takes the
    comma -- and the comma then promotes the two-item series to semicolons, which is
    what keeps "; and two further ranking rounds" legible.
    """
    record = _twice_revised()
    record.evolution.rereviews = [
        _review("cand_a_v3", recommendation="advance"),
        _review("cand_a_v3", recommendation="advance"),
    ]
    record.evolution.ranking_history = [TournamentState(ratings={"cand_a_v3": 1200.0})]
    core = _nine(
        record,
        [
            _brief(
                "A TiO2 coating",
                [],
                facts=_facts(),
                candidate_id="cand_a",
                revised_form=[("Claim", "An Al2O3 coating raises retention.")],
            )
        ],
    )
    assert (
        "Both checks were run again on the rewrites: two re-reviews, every one of "
        "which returned advance; and one further ranking round." in core
    )


def test_the_governance_obligations_are_a_list_the_lead_in_carries():
    """Set after a full stop, the obligations were a sentence with no verb in it.

    They are noun phrases as the planner states them -- "adherence to laboratory safety
    protocols", "proper disposal procedures" -- so "...discharged by a named owner.
    Adherence to laboratory safety protocols, proper disposal procedures, and use of
    calibrated equipment." asked the reader to find a predicate that was not there.
    """
    from coscientist.models import CandidatePopulation, ResearchPlan

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.plan = ResearchPlan(
        question=record.session.question,
        intended_claim="hypothesis",
        governance_requirements=[
            "Adherence to laboratory safety protocols",
            "Proper disposal procedures for hazardous battery materials",
        ],
    )
    core = _nine(record, [_brief("A coating", [], facts=_facts())])
    assert (
        "discharged by a named owner: adherence to laboratory safety protocols, and "
        "proper disposal procedures for hazardous battery materials." in core
    )
    assert "named owner. Adherence" not in core


def test_only_one_thing_in_the_closing_section_is_said_to_come_first():
    """Three sentences on one page each named a different thing to do first.

    "closing that comes before any of the work below" two paragraphs above, "Running
    that first" here, and "Before any physical, clinical or data-access step" below.
    The go/no-go is first among the work; the other two are what has to be cleared
    before the work starts, so this is the sentence that gives way.
    """
    from coscientist.models import (
        CandidatePopulation,
        DossierManifest,
        ResearchPlan,
        ReviewSet,
    )

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    briefs = [_brief("A coating", [], facts=_facts(), candidate_id="cand_a")]

    unblocked = _nine(record, briefs)
    assert "Running that first is what converts this report" in unblocked

    record.plan = ResearchPlan(
        question=record.session.question,
        intended_claim="hypothesis",
        governance_requirements=["Adherence to laboratory safety protocols"],
    )
    gated = _nine(record, briefs)
    assert (
        "Running that, once the governance obligations below are discharged, is what "
        "converts this report" in gated
    )

    record.reviews = [_review("cand_a", fatal_flaws=["The coating dissolves."])]
    record.reviews = [ReviewSet(reviews=[record.reviews[0]])]
    record.manifest = DossierManifest(
        title="Dossier", sections=[], unresolved_fatal_flaw_candidate_ids=["cand_a"]
    )
    both = _nine(record, briefs)
    assert "closing that comes before any of the work below" in both
    assert (
        "Running that, once the flaw and the exclusion above are closed and the "
        "governance obligations below are discharged, is what converts this report"
        in both
    )
    assert "Running that first" not in both


def test_a_recommendation_is_not_followed_by_a_paragraph_denying_it():
    """The fallback was the else of "were any revised", not of "was anything
    recommended", so a run that recommended four ideas and revised none of them said
    both things on the same page."""
    from coscientist.models import CandidatePopulation, DossierManifest

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["cand_a"]
    )
    core = _nine(record, [_brief("A coating", [], facts=_facts())])
    assert "The meta-review recommends carrying" in core
    assert "No idea cleared the bar" not in core


def test_the_least_weak_idea_carries_what_stands_against_it():
    """It is named as where to spend the next increment; if it is the idea the
    meta-review excluded, the paragraph cannot send the reader there in silence."""
    from coscientist.models import CandidatePopulation, DossierManifest, ReviewSet

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[_review("cand_a", fatal_flaws=["The rationale does not hold."])]
        )
    ]
    record.manifest = DossierManifest(
        title="Dossier",
        sections=[],
        unresolved_fatal_flaw_candidate_ids=["cand_a"],
    )
    core = _nine(
        record, [_brief("A coating", [], facts=_facts(), candidate_id="cand_a")]
    )
    assert "No idea cleared the bar" in core
    assert "a reviewer recorded a fatal flaw against it" in core
    assert "the meta-review excluded it from any recommendation" in core


def test_an_unflawed_least_weak_idea_is_not_given_a_warning_it_did_not_earn():
    from coscientist.models import CandidatePopulation

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    core = _nine(record, [_brief("A coating", [], facts=_facts())])
    assert "No idea cleared the bar" in core
    assert "standing of the whole set" not in core


def test_an_open_flaw_says_what_it_stops_in_the_terms_of_the_review_that_raised_it():
    """A novelty flaw printed "has to be closed before any work proceeds".

    The only flaw in that branch on the live run was a novelty review's — the idea
    duplicates published work — carrying stop-work language a safety reviewer had not
    written. Escalating a value judgement into a prohibition spends the credibility
    the report needs for the case where the prohibition is real.
    """
    from coscientist.narrative import _open_flaw_consequence

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    # A review counts as reachable only under an idea the report still prints.
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="novelty",
                    fatal_flaws=["It replicates a published study."],
                )
            ]
        )
    ]
    consequence = _open_flaw_consequence(record, ["cand_a"])
    assert "whether the work is worth doing" in consequence
    assert "may begin" not in consequence

    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="safety_governance",
                    fatal_flaws=["It vents at 55C without containment."],
                )
            ]
        )
    ]
    assert "no work on it may begin" in _open_flaw_consequence(record, ["cand_a"])


def test_a_recommended_ideas_open_flaw_points_at_the_section_that_prints_it():
    """The pointer named a section that never prints the flaw, and often is not there.

    Section eight reports that a flaw stands open and which review recorded it, but
    it prints that paragraph only where the meta-review named an exclusion list to be
    checked against the reviews. With no exclusions -- the case here -- the sentence
    sent the reader to a paragraph the report does not contain. Deep Verification
    prints the flaw itself under every ranked idea.
    """
    from coscientist.models import CandidatePopulation, DossierManifest, ReviewSet
    from coscientist.narrative import _section_eight

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.titles = {"cand_a": "A coating"}
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", fatal_flaws=["It replicates a published study."])
            ]
        )
    ]
    record.manifest = DossierManifest(
        title="Dossier", sections=[], recommendation_candidate_ids=["cand_a"]
    )
    briefs = [_brief("A coating", [], facts=_facts(), candidate_id="cand_a")]
    core = _nine(record, briefs)
    assert "recommended while carrying the fatal flaw" in core
    assert "printed in full under Deep Verification in its own section below" in core
    assert "Key Findings and Unexpected Connections" not in core
    eight = " ".join(_section_eight(record, briefs).core)
    assert "It replicates a published study" not in eight


def test_a_withdrawn_hypothesis_is_not_filed_under_the_meta_reviews_exclusions():
    """A person pulled it; the meta-review listed it and took the credit.

    Filed under the model's exclusions it read as one more automatic decision, and
    the one decision in the run a human actually took disappeared into a list of
    seven.
    """
    from coscientist.models import CandidatePopulation, DossierManifest, ReviewSet
    from coscientist.narrative import AdjudicationNote, _section_eight

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a"), _candidate("cand_b")]
    )
    record.titles = {"cand_a": "A coating", "cand_b": "A pre-coating"}
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", fatal_flaws=["It replicates a published study."]),
                _review("cand_b", fatal_flaws=["It vents without containment."]),
            ]
        )
    ]
    record.adjudications = [
        AdjudicationNote(
            candidate_id="cand_b",
            title="A pre-coating",
            resolution="withdraw",
            adjudicator="A named person",
            justification="The hazard is not worth the gain.",
            fatal_flaws=["It vents without containment."],
        )
    ]
    record.manifest = DossierManifest(
        title="Dossier",
        sections=[],
        unresolved_fatal_flaw_candidate_ids=["cand_a", "cand_b"],
    )
    core = " ".join(_section_eight(record, []).core)
    exclusions = next(
        part for part in core.split("\n") if "excluded the following" in part
    )
    assert "A pre-coating" not in exclusions.split("Exclusion here")[0]
    assert "withdrawn from the population by a named person" in core
    assert "Governance adjudications below" in core
    # It is not in contention, so nothing about it stands open against anything.
    assert "A pre-coating. The meta-review did not exclude" not in core


def test_an_idea_on_both_of_the_meta_reviews_lists_is_reported_as_on_both():
    """Section eight said the meta-review excluded every ranked idea, section nine
    recommended four of them by name eight lines later, and the minority note said of
    two of those four that they are "outside any recommendation this report makes".
    Both lists are the model's own and they overlapped on all four ideas; three
    sections read them independently and printed them as disjoint, so a reader could
    not determine whether the report recommends anything at all."""
    from coscientist.models import CandidatePopulation, DossierManifest, ReviewSet
    from coscientist.narrative import _minority_note, _section_eight, _section_nine

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a"), _candidate("cand_b")]
    )
    record.titles = {"cand_a": "A coating", "cand_b": "A pre-coating"}
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", fatal_flaws=["It replicates a published study."]),
                _review("cand_b", fatal_flaws=["It vents without containment."]),
            ]
        )
    ]
    record.manifest = DossierManifest(
        title="Dossier",
        sections=[],
        unresolved_fatal_flaw_candidate_ids=["cand_a", "cand_b"],
        recommendation_candidate_ids=["cand_a"],
    )

    disagreement = "The meta-review both excluded and recommended A coating"
    # Stated where the exclusion is stated and again where the recommendation is,
    # because a reader meets only one of the two paragraphs before acting on it.
    assert disagreement in " ".join(_section_eight(record, []).core)
    assert disagreement in " ".join(_section_nine(record, []).core)

    # And the idea is no longer said to be outside every recommendation the report
    # makes, in a report that recommends it.
    note = _minority_note(record, "cand_a")
    assert "outside any recommendation this report makes" not in note
    assert "excluded it and then recommended carrying it" in note
    # The idea the meta-review only excluded still gets the plain sentence.
    assert "outside any recommendation this report makes" in _minority_note(
        record, "cand_b"
    )


def _section_one_core(record: ResearchRecord) -> str:
    from coscientist.narrative import _section_one

    return " ".join(_section_one(record).core)


def test_no_experiment_was_performed_is_not_gated_on_a_flag_no_run_sets():
    """It was printed only when ``literature_only`` was true, which no ordinary run
    sets — so the one sentence saying nothing was measured appeared in no ordinary
    report, over eight ideas quoting retention to the decimal point.

    It has since moved out of section one and into the standing-limits advisory,
    which is unconditional; the flag still only decides whether the literature-only
    sentence opens it."""
    from coscientist.advisories import _standing_limits_advisory

    record = ResearchRecord(
        session=Session(question="Can a coating help?", research_mode="experimental")
    )
    limits = _standing_limits_advisory(record).body
    assert "No experiment was performed in this run" in limits
    assert "the research mode named on the cover, experimental," in limits
    assert "The run was executed as a literature-only analysis" not in limits

    record.session.literature_only = True
    assert "literature-only analysis" in _standing_limits_advisory(record).body


def test_a_thin_literature_says_which_pass_produced_it():
    """ "Discovery searched the literature and did not synthesise it" reads as a
    finding about the field. It was a fact about the configuration: the agent that
    writes the synthesis never ran."""
    from coscientist.models import DiscoveryManifest, EvidenceClaim, EvidencePacket
    from coscientist.narrative import _knowledge_summary, _section_three

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        claims=[
            EvidenceClaim(
                id="claim_1",
                source_id="src_1",
                claim="Thin coatings extend cycle life",
                relation="supports",
            )
        ],
    )
    record.discovery = DiscoveryManifest(
        question="Can a coating help?",
        convergence_reason="search_grounded_fallback",
    )
    core = " ".join(_section_three(record).core)
    assert "did not run on this goal" in core
    assert "a single search-grounded pass stood in for it" in core
    assert "did not run on this goal" in _knowledge_summary(record)

    record.discovery = DiscoveryManifest(
        question="Can a coating help?", convergence_reason="coverage_converged"
    )
    assert "did not run on this goal" not in " ".join(_section_three(record).core)


@pytest.mark.parametrize(
    ("judge", "expected", "forbidden"),
    [
        (
            "llm_debate",
            "None of these scores enters the ordering directly",
            "was computed from these scores",
        ),
        (
            "deterministic",
            "was computed from these scores",
            "None of these scores enters",
        ),
    ],
)
def test_what_a_criterion_score_feeds_is_stated_once_and_matches_the_judge(
    judge, expected, forbidden
):
    """Each criterion closed on the scores "doing real work in the final ordering".

    Where a judge decided the matches the ordering is a function of them and these
    scores are not an input at all, so the claim was false; and it was made once per
    criterion, three times over in one section of a live report.
    """
    from coscientist.models import PairwiseComparison, TournamentState
    from coscientist.narrative import _section_six

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_a",
                candidate_b_id="cand_b",
                presented_first_id="cand_a",
                winner_id="cand_a",
                rationale="The first idea is the more informative of the two.",
                judge=judge,
            )
        ]
    )
    briefs = [
        _brief("A", [_idea_review(section="Correctness", score=5)]),
        _brief("B", [_idea_review(section="Correctness", score=2)]),
        _brief("C", [_idea_review(section="Safety", score=5)]),
        _brief("D", [_idea_review(section="Safety", score=1)]),
    ]

    six = " ".join(_section_six(record, briefs).core)
    assert "doing real work in the final ordering" not in six
    assert expected in six
    assert forbidden not in six
    # Stated in the lead-in, not once under every criterion that spread.
    assert six.count(expected) == 1


def test_what_sits_at_the_top_of_a_spread_is_an_idea_not_a_review():
    """ "Four of the ideas tied highest at 5 ... a spread of 3 points with four
    reviews at the top of it" names the same set two ways in consecutive clauses,
    the second of them after the instrument rather than the thing measured."""
    from coscientist.narrative import _section_six

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    briefs = [
        _brief(name, [_idea_review(section="Correctness", score=score)])
        for name, score in (("A", 5), ("B", 5), ("C", 5), ("D", 2))
    ]

    six = " ".join(_section_six(record, briefs).core)

    assert "A; B; and C tied highest at 5" in six
    # A point on a five-point review scale is a count of grades, not a measured
    # quantity like an Elo point, so it is spelled: "a spread of 3 points with three
    # ideas at the top of it" wrote the same kind of number both ways in one clause.
    assert "a spread of three points with three ideas at the top of it" in six
    assert "3 points" not in six
    assert "reviews at the top" not in six
    # One criterion is one judgement, and the count and the noun have to agree even
    # where no live run reaches the singular.
    assert "compresses one separate judgement into one number" in six


def test_a_criterion_that_separates_nothing_says_so_once():
    """ "Every review on this criterion came in at 2, so it separates nothing. The
    spread is narrow enough that this criterion did not separate the field and should
    not be used to justify a choice between them."

    A flat criterion has a spread of zero, so it fell through to the narrow-spread
    verdict and the paragraph made the same finding twice in consecutive sentences.
    """
    from coscientist.narrative import _section_six

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    briefs = [
        _brief("A", [_idea_review(section="Correctness", score=2)]),
        _brief("B", [_idea_review(section="Correctness", score=2)]),
    ]

    six = " ".join(_section_six(record, briefs).core)
    assert "Every review on this criterion came in at 2." in six
    # The verdict is made once over all the criteria, below the paragraphs, and the
    # flat one is named there rather than carrying its own copy of it.
    assert six.count("Correctness left the ideas level or within a point") == 1
    assert "The spread is narrow enough" not in six
    # With no highest and no lowest, the same set is not named twice as two things.
    assert "tied highest" not in six


def test_the_criteria_worth_choosing_on_are_named_once_below_the_paragraphs():
    """ "wide enough to separate the field" closed four of five criterion paragraphs
    on a live run, the words identical and only the counts in front of them changing,
    and which criteria a choice may rest on had to be collected paragraph by
    paragraph."""
    from coscientist.narrative import _section_six

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    briefs = [
        _brief(
            name,
            [
                _idea_review(section="Correctness", score=wide),
                _idea_review(section="Novelty", score=wide),
                _idea_review(section="Feasibility", score=4),
            ],
        )
        for name, wide in (("A", 5), ("B", 2))
    ]

    six = " ".join(_section_six(record, briefs).core)

    assert "Correctness and novelty spread the ideas by at least two points" in six
    assert six.count("wide enough to choose on") == 1
    assert (
        "Feasibility did not, and a choice between the ideas should not be justified "
        "on it." in six
    )
    assert "wide enough to separate the field" not in six


def test_reviews_agreeing_at_a_capped_score_are_not_reported_as_a_clearance():
    """A fatal flaw caps its review at two, so a field of capped reviews agrees.

    The coherence line read that agreement off the spread and reported it as a property
    of the idea — "they agree, so the idea presents the same way from every angle that
    was examined" — over a review that had called the idea disqualified.
    """
    from coscientist.narrative import _coherence

    lines, _ = _coherence(
        [
            _idea_review(section="Correctness", score=2, recommendation="reject"),
            _idea_review(
                section="Novelty",
                score=2,
                recommendation="reject",
                fatal_flaws=["It replicates a published study."],
            ),
        ],
        {},
    )

    assert "That agreement is not a clearance" in lines[0]
    assert "the novelty review records a fatal flaw" in lines[0]

    # The same clause over a spread said "They disagree by more than a point ... That
    # agreement is not a clearance" -- there was no agreement to qualify.
    spread, _ = _coherence(
        [
            _idea_review(section="Impact", score=5, recommendation="advance"),
            _idea_review(
                section="Novelty",
                score=2,
                recommendation="reject",
                fatal_flaws=["It replicates a published study."],
            ),
        ],
        {},
    )
    assert "That agreement is not a clearance" not in spread[0]
    assert spread[0].endswith(
        "a disagreement of more than a point, and the novelty review records a fatal "
        "flaw against the idea."
    )


def test_what_a_spread_means_is_explained_above_the_ideas_not_under_each(
    rich_session: Session,
):
    """Three sentences of standing explanation — what a spread of more than a point
    means, and what would settle one — wrapped one clause of idea-specific fact and
    were printed under all seven ideas of a live run."""
    from coscientist.dossier import compile_dossier
    from coscientist.narrative import (
        COHERENCE_SPREAD_NOTE,
        build_idea_briefs,
        load_record,
        shared_coherence_notes,
    )

    briefs = build_idea_briefs(load_record(rich_session))
    spread = [
        brief for brief in briefs if COHERENCE_SPREAD_NOTE in brief.coherence_notes
    ]
    assert len(spread) > 1, "the fixture must give more than one idea a wide spread"

    report = compile_dossier(rich_session)
    assert report.count(COHERENCE_SPREAD_NOTE) == 1
    assert COHERENCE_SPREAD_NOTE in "\n".join(shared_coherence_notes(briefs))
    for brief in spread:
        # What stays under the idea is this idea's spread, not what a spread is.
        assert any(
            "a disagreement of more than a point" in line for line in brief.coherence
        )
        assert not any(
            note in line for note in brief.coherence_notes for line in brief.coherence
        )


def test_why_a_flawed_review_widens_a_spread_is_said_once_and_not_under_each_idea():
    """Six of the seven ideas in a live chapter carried the identical "Part of that
    spread is a disqualification rather than a grade:", and the clause that differed —
    which reviews had recorded the flaw — sat at the end of it, where a reader six
    repetitions in has stopped looking. That is a rule about spreads, so it is raised
    as a standing note and printed with the other rules above the ideas."""
    from coscientist.narrative import (
        _COHERENCE_NOTES,
        COHERENCE_DISQUALIFICATION_NOTE,
        _coherence,
    )

    lines, notes = _coherence(
        [
            _idea_review(section="Impact", score=5, recommendation="advance"),
            _idea_review(
                section="Novelty",
                score=2,
                recommendation="reject",
                fatal_flaws=["It replicates a published study."],
            ),
        ],
        {},
    )

    # What stays under the idea is which review refused it.
    assert "the novelty review records a fatal flaw" in lines[0]
    assert "a disqualification rather than a grade" not in lines[0]
    assert COHERENCE_DISQUALIFICATION_NOTE in notes
    # Hoisting is by membership of this tuple, which is what the caller prints once.
    assert COHERENCE_DISQUALIFICATION_NOTE in _COHERENCE_NOTES


def test_the_fatal_flaws_nobody_ruled_on_are_counted_beside_the_one_somebody_did():
    """The paragraph reported the adjudicated flaws and stopped.

    A run in which a person answered one flaw and 16 others were left standing read as
    a run whose fatal flaws had been dealt with — and said "a fatal flaw against one
    hypothesis ... and each one was answered", disagreeing with itself about how many
    there were.
    """
    from coscientist.models import ReviewSet
    from coscientist.narrative import AdjudicationNote, _section_one

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a"), _candidate("cand_b"), _candidate("cand_c")]
    )
    record.reviews = [
        ReviewSet(
            reviews=[
                _review("cand_a", fatal_flaws=["It replicates a published study."]),
                _review("cand_b", fatal_flaws=["It has no power rationale."]),
                _review("cand_c", fatal_flaws=["It vents without containment."]),
            ]
        )
    ]
    record.adjudications = [
        AdjudicationNote(
            candidate_id="cand_c",
            title="A pre-coating",
            resolution="withdraw",
            adjudicator="A named person",
            justification="The hazard is not worth the gain.",
            fatal_flaws=["It vents without containment."],
        )
    ]

    one = " ".join(_section_one(record).core)
    assert "against one hypothesis in this run, and it was answered" in one
    assert "each one was answered" not in one
    assert "fatal flaws against a further two ideas, which nobody adjudicated" in one
    # The contrast is carried by "the other reviews". Spelling it out a second time
    # put a "that" after the pointer to Governance adjudications, where the nearest
    # noun for it to reach was the reprinting rather than the flaw.
    assert "a person ruled on" not in one


def test_a_tournament_that_never_read_the_criteria_is_not_said_to_have_applied_them():
    """The section described the run that was configured, not the run that happened.

    A tournament with no judge available ranks on the review scores, and the criteria
    are never applied to a match. The opening sentence told that reader the criteria
    were "applied identically to all of them so that the ranking reflects the ideas".
    """
    from coscientist.models import (
        CandidatePopulation,
        PairwiseComparison,
        TournamentState,
    )
    from coscientist.narrative import _section_two

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a"), _candidate("cand_b")],
        comparison_criteria=["Evidence: Alignment with the provided claims"],
    )
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_a",
                candidate_b_id="cand_b",
                presented_first_id="cand_a",
                winner_id="cand_a",
                rationale="The first idea scores higher.",
                judge="deterministic",
            )
        ]
    )

    two = " ".join(_section_two(record).core)
    assert "applied identically to all of them" not in two
    assert "The tournament did not read them" in two
    assert "decided by arithmetic on the review scores" in two


def test_the_goal_details_are_not_restated_in_full_by_the_sections_that_use_them():
    """The cover prints the constraints, the assumptions and the criteria; sections
    one and two printed all three again within two pages, some four hundred words of
    verbatim restatement. They point now, and keep only what the cover cannot say."""
    from coscientist.models import CandidatePopulation, ResearchPlan
    from coscientist.narrative import _section_two

    plan = ResearchPlan(
        question="Can a coating help?",
        constraints=["Must include uncoated control cells"],
        assumptions=["The coating does not alter bulk properties"],
        success_criteria=["Cycle life rises by ten per cent"],
    )
    record = ResearchRecord(session=Session(question="Can a coating help?"), plan=plan)
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a")],
        comparison_criteria=["Evidence: Alignment with the provided claims"],
    )
    one = _section_one_core(record)
    assert "Must include uncoated control cells" not in one
    assert "numbered under Requirements on the cover" in one
    assert "The coating does not alter bulk properties" not in one
    assert "listed under Attributes on the cover" in one

    two = " ".join(_section_two(record).core)
    assert "Alignment with the provided claims" not in two
    assert "comparison criteria set out on the cover" in two
    assert "Cycle life rises by ten per cent" not in two
    assert "stated under Criteria on the cover" in two


def test_the_cover_separates_what_would_meet_the_goal_from_what_ranked_the_ideas():
    """One bulleted list held both, set by different stages and answering different
    questions, with nothing on the page marking where one ended."""
    from coscientist.dossier import _front_matter
    from coscientist.models import CandidatePopulation, ResearchPlan
    from coscientist.narrative import synthesize_overview

    plan = ResearchPlan(
        question="Can a coating help?",
        constraints=["Must include uncoated control cells", "Must state a thickness"],
        success_criteria=["Cycle life rises by ten per cent"],
    )
    record = ResearchRecord(session=Session(question="Can a coating help?"), plan=plan)
    record.population = CandidatePopulation(
        candidates=[_candidate("cand_a")],
        comparison_criteria=["Evidence: Alignment with the provided claims"],
    )
    lines = _front_matter(record, synthesize_overview(record))
    assert "**Success criteria — what would make the goal met:**" in lines
    assert "**Comparison criteria — what every idea was scored against:**" in lines
    # Numbered, because section 1 sends the reader here by number.
    assert "1. Must include uncoated control cells" in lines
    assert "2. Must state a thickness" in lines
    # The label leads, since the report cites the criteria by it.
    assert "- **Evidence** — Alignment with the provided claims." in lines


def test_a_novelty_verdict_carries_the_reservation_its_own_reviewer_recorded():
    """A five out of five and a prior-art objection from the same review are both on
    the record, and the comparison section printed only the first. One live report
    listed an idea as adding something beyond current practice on a score whose own
    reviewer had written "some literature already discusses HF scavenging by
    amphoteric oxides like Al2O3 and ZnO" -- three hundred lines below, under the
    idea, where a reader of the comparison never meets it."""
    from coscientist.narrative import _section_seven

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    reserved = _idea_review(
        score=5,
        objections=["Some literature already discusses HF scavenging by ZnO."],
    )
    clean = _idea_review(score=5)
    briefs = [
        _brief("A Sacrificial ZnO Island Coating", [reserved]),
        _brief("A Self-healing Microcapsule Coating", [clean]),
    ]

    core = " ".join(_section_seven(record, briefs).core)
    assert "add something beyond current practice" in core
    assert "The verdict is not unqualified" in core
    assert "A Sacrificial ZnO Island Coating" in core.split("not unqualified")[1]
    assert "A Self-healing Microcapsule Coating" not in core.split("not unqualified")[1]

    # Every novelty verdict reserved is the common case, and naming all of them back
    # reads as a second list rather than as a qualification of the first.
    both = _section_seven(
        record, [_brief("A Sacrificial ZnO Island Coating", [reserved])]
    )
    assert "an objection against every one of them" in " ".join(both.core)

    # A verdict nobody qualified is not qualified by this report either.
    unreserved = _section_seven(record, [_brief("A Self-healing Coating", [clean])])
    assert "not unqualified" not in " ".join(unreserved.core)


def test_a_findings_own_scope_is_printed_with_it_and_a_runs_scope_only_once():
    """Every claim in the live runs carried a recorded scope and the report printed
    none of them: "specific to NCM811 cathodes and dry vs wet coating methods" sat in
    the artifact while a retention figure went onto the page as a general result. The
    qualification every claim shares is a fact about the run rather than about any
    finding, so it is stated once instead of six times."""
    from coscientist.models import EvidenceClaim, EvidencePacket, SourceRecord
    from coscientist.narrative import _section_three

    shared = "Unverified claim inferred from search snippet and title."
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[
            SourceRecord(id="src_1", url="https://example.org/a", title="First"),
            SourceRecord(id="src_2", url="https://example.org/b", title="Second"),
        ],
        claims=[
            EvidenceClaim(
                id="claim_1",
                source_id="src_1",
                claim="Dry-coated NCM811 cathodes retained 80.8% after 150 cycles",
                relation="supports",
                limitations=[shared, "Specific to NCM811 and dry versus wet coating."],
            ),
            EvidenceClaim(
                id="claim_2",
                source_id="src_2",
                claim="Double-layer coated NCM retained 94.3% after 100 cycles",
                relation="supports",
                limitations=[shared, "Tested at 55 degrees C to accelerate fading."],
            ),
        ],
    )

    core = _section_three(record).core
    stated = [
        paragraph for paragraph in core if "recorded against every finding" in paragraph
    ]
    assert len(stated) == 1
    assert shared in stated[0]

    findings = [
        paragraph
        for paragraph in core
        if "retained 80.8%" in paragraph or "retained 94.3%" in paragraph
    ]
    assert len(findings) == 2
    for paragraph in findings:
        assert shared not in paragraph
    assert "Specific to NCM811 and dry versus wet coating" in findings[0]
    assert "Tested at 55 degrees C" in findings[1]
    assert "Tested at 55 degrees C" not in findings[0]


def test_shared_qualifications_are_not_introduced_by_a_colon_that_never_closes():
    """The colon opened a list the qualifications' own full stops then broke up.

    They come back as whole sentences, one of them with a colon of its own, so
    "recorded against every finding below: Unverified claim inferred from search
    snippet and title. Restored to the discovered wording: the verification pass
    restated this claim without confirming it." left a reader unable to see where the
    first item ended. "That is a statement" then stood over both of them.
    """
    from coscientist.models import EvidenceClaim, EvidencePacket, SourceRecord
    from coscientist.narrative import _section_three

    shared = [
        "Unverified claim inferred from search snippet and title.",
        "Restored to the discovered wording: the verification pass restated this "
        "claim without confirming it.",
    ]
    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.evidence = EvidencePacket(
        question="Can a coating help?",
        sources=[SourceRecord(id="src_1", url="https://example.org/a", title="First")],
        claims=[
            EvidenceClaim(
                id=f"claim_{index}",
                source_id="src_1",
                claim=text,
                relation="supports",
                limitations=list(shared),
            )
            for index, text in enumerate(
                ["Coated cells retained 80.8%", "Uncoated cells retained 61.2%"], 1
            )
        ],
    )

    stated = next(
        paragraph
        for paragraph in _section_three(record).core
        if "recorded against every finding" in paragraph
    )
    assert stated.startswith(
        "The same two qualifications are recorded against every finding below, in the "
        "words the run recorded them. Unverified claim"
    )
    assert "below: Unverified" not in stated
    assert "Those are statements about how this run reached the findings" in stated
    assert "That is a statement" not in stated


def test_a_constraint_a_judge_argued_is_not_reported_as_unreached():
    """The report told a reader "nothing any reviewer wrote reaches constraint three,
    so on that one the report is silent" -- three hundred lines above a tournament
    transcript in which a judge had written that both hypotheses fail the constraint
    requiring exact charge and discharge rates, voltage windows and temperature, and
    quoted it. Silence was the one thing the record did not hold."""
    from coscientist.models import PairwiseComparison, TournamentState
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    # A review counts as reachable only under an idea the report still prints.
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="methods_feasibility",
                    objections=[
                        "No uncoated control cells are described for the comparison."
                    ],
                )
            ]
        )
    ]
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_a",
                candidate_b_id="cand_b",
                presented_first_id="cand_a",
                winner_id="cand_a",
                rationale="Neither states a voltage window.",
                judge="llm_debate",
                debate_turns=[
                    "Both hypotheses fail the constraint requiring exact "
                    "charge/discharge rates, voltage windows and temperature."
                ],
            )
        ]
    )

    line = _constraint_coverage(
        record,
        [
            "Must include uncoated control cells for direct comparison",
            "Must specify exact charge/discharge rates, voltage windows, and temperature",
        ],
    )
    assert "Nothing any reviewer wrote reaches constraint two." in line
    assert "The tournament debates reach constraint two" in line
    # What a judge says about a requirement is said about the pair it is comparing,
    # so the sentence may not be read as a screen of the field.
    assert "said about the pair in front of it" in line
    assert "the report is silent" not in line


def test_a_turn_reaching_two_constraints_is_one_turn():
    """Summing the per-constraint tallies counted a turn once per constraint it reached.

    A judge writing about rates, voltage windows and control cells in one turn was
    three turns to this sentence: a run of 27 debate turns was reported as 33, more
    than the transcript printed below it holds.
    """
    from coscientist.models import PairwiseComparison, TournamentState
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [ReviewSet(reviews=[_review("cand_a", criterion="impact_safety")])]
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_a",
                candidate_b_id="cand_b",
                presented_first_id="cand_a",
                winner_id="cand_a",
                rationale="The first idea is the more informative of the two.",
                judge="llm_debate",
                debate_turns=[
                    "Neither describes uncoated control cells, and neither specifies "
                    "the charge/discharge rates, voltage windows or temperature."
                ],
            )
        ]
    )

    line = _constraint_coverage(
        record,
        [
            "Must include uncoated control cells for direct comparison",
            "Must specify exact charge/discharge rates, voltage windows, and temperature",
        ],
    )
    assert "constraints one and two, in one turn across the matches" in line


def test_a_constraint_neither_a_reviewer_nor_a_judge_reached_is_still_reported_once():
    """The gap and the silence are one fact; stated twice they read as two findings."""
    from coscientist.models import PairwiseComparison, TournamentState
    from coscientist.narrative import _constraint_coverage

    record = ResearchRecord(session=Session(question="Can a coating help?"))
    # A review counts as reachable only under an idea the report still prints.
    record.population = CandidatePopulation(candidates=[_candidate("cand_a")])
    record.reviews = [
        ReviewSet(
            reviews=[
                _review(
                    "cand_a",
                    criterion="methods_feasibility",
                    objections=[
                        "No uncoated control cells are described for the comparison."
                    ],
                )
            ]
        )
    ]
    record.tournament = TournamentState(
        comparisons=[
            PairwiseComparison(
                round_number=1,
                candidate_a_id="cand_a",
                candidate_b_id="cand_b",
                presented_first_id="cand_a",
                winner_id="cand_a",
                rationale="The first idea is the more informative of the two.",
                judge="llm_debate",
                debate_turns=["Both are worth running."],
            )
        ]
    )

    line = _constraint_coverage(
        record,
        [
            "Must include uncoated control cells for direct comparison",
            "Must specify exact charge/discharge rates, voltage windows, and temperature",
        ],
    )
    assert line.count("constraint two") == 1
    assert (
        "reaches constraint two, so on that one this report shows no coverage" in line
    )
    assert "The tournament debates reach" not in line
    # An unreached constraint is the output of a wording match, and the paragraph used
    # to report it as silence -- a property of the record rather than of the test.
    assert "The match is on wording" in line


def _evidence_session(*packets: dict) -> Session:
    """A session whose evidence stage emitted each of these packets, in order."""
    from coscientist.models import Artifact, ArtifactStatus, EvidencePacket

    session = Session(question="Can a coating help?")
    for agent, payload in zip(
        ("evidence_discovery", "source_verification"), packets, strict=False
    ):
        packet = EvidencePacket(question=session.question, **payload)
        session.artifacts.append(
            Artifact(
                stage="evidence",
                agent=agent,
                content="",
                schema_name="EvidencePacket",
                payload=packet.model_dump(mode="json"),
                status=ArtifactStatus.ACCEPTED,
                artifact_type="specialist_output",
            )
        )
    return session


def test_a_verification_pass_that_returns_nothing_does_not_erase_the_evidence():
    """The second evidence packet was assigned over the first.

    Source verification is meant to hand the discovered claims back checked. One live
    run had it come back empty, and the report then stated the run had no evidence in
    it at all while the ideas below went on citing the discarded claims by id.
    """
    from coscientist.narrative import load_record

    record = load_record(
        _evidence_session(
            {
                "sources": [{"id": "src_1", "url": "https://example.org/a"}],
                "claims": [
                    {
                        "id": "claim_1",
                        "claim": "A coating held 95% of capacity after 280 cycles.",
                        "source_id": "src_1",
                    }
                ],
            },
            {
                "sources": [{"id": "src_1", "url": "https://example.org/a"}],
                "claims": [],
            },
        )
    )
    assert record.evidence is not None
    assert [claim.id for claim in record.evidence.claims] == ["claim_1"]


def test_a_verified_claim_replaces_the_discovered_form_of_itself():
    """Merging must not resurrect the unverified version alongside the checked one."""
    from coscientist.narrative import load_record

    claim = {
        "id": "claim_1",
        "claim": "A coating held 95% of capacity after 280 cycles.",
        "source_id": "src_1",
    }
    record = load_record(
        _evidence_session(
            {
                "sources": [{"id": "src_1", "url": "https://example.org/a"}],
                "claims": [claim],
            },
            {
                "sources": [{"id": "src_1", "url": "https://example.org/a"}],
                "claims": [{**claim, "verification_status": "verified"}],
            },
        )
    )
    assert record.evidence is not None
    assert len(record.evidence.claims) == 1
    assert record.evidence.claims[0].verification_status == "verified"


def test_a_source_only_the_evidence_packet_names_can_still_be_numbered():
    """Only a discovery lead could be cited, so a claim whose source discovery never
    listed was printed with no marker and left out of the references, under prose
    saying every claim the ideas cite exists."""
    from coscientist.narrative import load_record

    record = load_record(
        _evidence_session(
            {
                "sources": [{"id": "src_1", "url": "https://example.org/a"}],
                "claims": [
                    {
                        "id": "claim_1",
                        "claim": "A coating held 95% of capacity after 280 cycles.",
                        "source_id": "src_1",
                    }
                ],
            }
        )
    )
    assert record.citations.number("https://example.org/a") == 1


def test_a_name_folded_into_a_sentence_keeps_the_capital_it_carries():
    """The go/no-go tests are spliced into a list, and the test for whether the first
    word was capitalised by the sentence or by itself passed a person's name through:
    "Karl Fischer titration to ensure moisture content ... is below 50 ppm" went to
    the page as "karl Fischer titration"."""
    from coscientist.narrative import _spliced

    assert _spliced("Karl Fischer titration to ensure moisture is below 50 ppm.") == (
        "Karl Fischer titration to ensure moisture is below 50 ppm"
    )
    # The shape of the sentence cannot decide this. Every one of these opens on a
    # capitalised word followed by a second capital, exactly as the name above does,
    # and every one of them is a word the sentence capitalised and has to fold.
    for folded, text in (
        ("initial Coulombic", "Initial Coulombic efficiency will be improved."),
        ("cross-sectional SEM", "Cross-sectional SEM capabilities for post-mortem."),
        ("specified H14-grade", "Specified H14-grade HEPA filtration for safety."),
        ("conduct Electrochemical", "Conduct Electrochemical Impedance Spectroscopy."),
    ):
        assert _spliced(text).startswith(folded)
    assert _spliced("Synthesis of battery-grade PEDOT:PSS.").startswith("synthesis of")


def test_a_colon_inside_a_name_does_not_take_an_item_out_of_its_series():
    """PEDOT:PSS is a polymer, not a sentence opening a list. Read as punctuation, it
    set its own item apart and the required inputs printed as "It cannot start until
    its inputs exist: synthesis of ... PEDOT:PSS. Cross-sectional SEM capabilities for
    post-mortem analysis." -- a colon introducing one item and a fragment after it."""
    from coscientist.narrative import _join

    assert _join(
        [
            "Synthesis of battery-grade, moisture-free PEDOT:PSS",
            "Cross-sectional SEM capabilities for post-mortem analysis",
        ],
        fallback="none.",
    ) == (
        "Synthesis of battery-grade, moisture-free PEDOT:PSS; and cross-sectional "
        "SEM capabilities for post-mortem analysis."
    )
    # A colon that does punctuate still sets its item on its own, which is what the
    # check was written for: what the colon opens runs to the end of the sentence.
    apart = _join(
        [
            "It rests on one measurement: coated cells reach 80% retention later",
            "Cross-sectional SEM shows less cracking",
        ],
        fallback="none.",
    )
    assert "; and" not in apart


def test_the_reviews_that_recorded_a_flaw_are_listed_as_reviews_not_as_titles():
    """The list was punctuated by the helper that sets out idea titles, which are long
    enough to need semicolons: "The correctness review; feasibility review; and novelty
    review" sets three two-word phrases as clauses and says "review" three times. The
    Coherence subsection under the same idea named the same three correctly, and in the
    order the run recorded them rather than in alphabetical order."""
    from coscientist.narrative import _fatal_flaw_notice

    faulted = [
        _idea_review(section=section, fatal_flaws=[f"A flaw the {section} review saw."])
        for section in ("Correctness", "Novelty", "Feasibility")
    ]
    notice = _fatal_flaw_notice(faulted)
    assert notice.startswith(
        "The correctness, novelty, and feasibility reviews recorded three fatal "
        "flaws against this idea."
    )
    assert _fatal_flaw_notice(faulted[:1]).startswith(
        "The correctness review recorded a fatal flaw against this idea."
    )


def test_two_reviews_sharing_the_floor_are_both_of_them_and_not_all_of_them():
    """ "Its lowest score, 2 of five, is shared by the correctness and feasibility
    reviews, all printed in full below" counts a pair with the word for three or more.
    The reviews are also named in the order the run recorded them, which is the order
    they are printed in and the order the Coherence subsection above uses: sorted
    alphabetically, the two sentences listed the same three reviews two ways."""
    from coscientist.narrative import _conclusion

    facts = {"Go/no-go tests": "Soak the coated electrodes.", "Falsifier": "No gain."}
    sections = ("Correctness", "Novelty", "Feasibility", "Impact")

    def conclusion(*scores: int) -> str:
        return _conclusion(
            facts,
            [
                _idea_review(section=section, score=score)
                for section, score in zip(sections, scores, strict=True)
            ],
            shortlisted=True,
            accepted_flaw=None,
        )

    assert (
        "shared by the correctness and feasibility reviews, both printed in full below"
        in conclusion(2, 5, 2, 4)
    )
    assert (
        "shared by the correctness, novelty, and feasibility reviews, all printed in "
        "full below" in conclusion(2, 2, 2, 4)
    )


def test_the_two_orders_a_reordering_reports_are_set_as_two_sentences():
    """The titles in each order are separated by semicolons, because they are long
    enough to need them. Joining the two orders with a semicolon too made the mark
    that ended the first list and the mark inside it the same one, so the second
    clause read as a fifth title -- and the sentence ran to eighty-six words."""
    from coscientist.narrative import _post_evolution_reordering

    titles = {
        "a": "Dry-coating NCM811 Cathodes with a 2 wt% TiO2",
        "b": "A 10 nm Double-layer Coating of Al2O3 Nanoparticles",
        "c": "A 5 nm Uniform LiF Nanoshell Coating",
    }
    record = SimpleNamespace(
        post_evolution_order=["b", "a", "c"],
        title_for=titles.__getitem__,
        ranked_id=lambda item: item,
    )
    briefs = [SimpleNamespace(candidate_id=item) for item in ("a", "b", "c")]
    said = _post_evolution_reordering(record, ["a", "b", "c"], briefs)
    assert "A 5 nm Uniform LiF Nanoshell Coating. Ranked on the proposals" in said
    assert "; ranked on the proposals" not in said
    assert max(len(part.split()) for part in said.split(". ")) < 45

    # The section says nothing at all when the two rounds agree, which is the common
    # case: a reader is told about a reordering only when there was one.
    assert not _post_evolution_reordering(
        SimpleNamespace(
            post_evolution_order=["a", "b", "c"],
            title_for=titles.get,
            ranked_id=lambda item: item,
        ),
        ["a", "b", "c"],
        briefs,
    )


# --- a recorded reason has to stand on its own -------------------------------


def _judged(rationale: str, *, opponent: str = "A LiAlF4 Coating"):
    from coscientist.narrative import IdeaMatch

    return IdeaMatch(
        round_number=2,
        opponent_title=opponent,
        outcome="win",
        elo_before=1200.0,
        elo_after=1216.0,
        confidence=0.75,
        rationale=rationale,
        judge="llm_comparison",
    )


def test_a_reason_opening_on_a_contrast_does_not_start_the_reader_mid_argument():
    """ "However, the deciding factor is safety and feasibility" opened a match bullet
    of a live report. Only the judge's closing paragraphs are recorded, so the thing
    the sentence contrasts with is nowhere on the page."""
    from coscientist.dossier import _match_summary

    said = "\n".join(
        _match_summary(
            _brief(
                "Idea",
                [],
                matches=[
                    _judged(
                        "However, the deciding factor is safety and feasibility. "
                        "The opposing idea uses HF-pyridine precursors."
                    )
                ],
            )
        )
    )

    assert "However" not in said
    assert (
        "**Round 2 against A LiAlF4 Coating (win):** The deciding factor is safety "
        "and feasibility." in said
    )


def test_a_reason_that_already_stands_alone_is_printed_word_for_word():
    from coscientist.dossier import _match_summary

    said = "\n".join(
        _match_summary(
            _brief(
                "Idea",
                [],
                matches=[_judged("The opposing idea uses HF-pyridine precursors.")],
            )
        )
    )

    assert "The opposing idea uses HF-pyridine precursors." in said


def test_the_rematch_note_keeps_its_place_in_front_of_the_reason():
    """The note is the report's own bracketed aside, so the connective to drop is the
    one after it rather than the bracket the line opens on."""
    from coscientist.debate import standalone_opening

    said = standalone_opening(
        "[Rematch: this pair also met in Swiss round 2.] Therefore, this idea wins."
    )

    assert said == "[Rematch: this pair also met in Swiss round 2.] This idea wins."


def test_a_reason_that_is_nothing_but_a_connective_is_left_as_recorded():
    from coscientist.debate import standalone_opening

    assert standalone_opening("However,") == "However,"


def test_a_contrast_the_judge_buried_after_the_subject_is_dropped_too():
    """ "This idea, on the other hand, offers a paradigm-shifting reinterpretation"
    opened a live match bullet with no first hand anywhere on the page. The leading-
    connective rule cannot see this one: it sits after the subject."""
    from coscientist.narrative import _self_contained

    assert (
        _self_contained(
            "The opposing idea, on the other hand, offers a reinterpretation."
        )
        == "The opposing idea offers a reinterpretation."
    )
    assert (
        _self_contained(
            "[Rematch: this pair also met in Swiss round 1.] This idea, by contrast, wins."
        )
        == "[Rematch: this pair also met in Swiss round 1.] This idea wins."
    )


def test_a_flaw_the_reason_points_at_without_naming_is_given_a_side():
    """ "This idea avoids this fatal flaw and provides an equally provocative claim"
    was the whole of what a live bullet said about the deciding flaw. Which side the
    flaw is on follows from which side the sentence is about, and both are resolved
    by the time the bullet is written."""
    from coscientist.narrative import _self_contained

    assert (
        _self_contained("This idea avoids this fatal flaw and is provocative.")
        == "This idea avoids the fatal flaw the judge found in the opposing idea and "
        "is provocative."
    )
    assert (
        _self_contained("The opposing idea lacks these weaknesses.")
        == "The opposing idea lacks the weaknesses the judge found in this idea."
    )


def test_a_reason_that_says_what_the_flaw_was_is_not_told_again():
    """Pointing back at its own sentence is not a dangling pointer."""
    from coscientist.narrative import _self_contained

    said = _self_contained(
        "The opposing idea cites fabricated evidence, a fatal flaw. This idea avoids "
        "this fatal flaw."
    )

    assert said.endswith("This idea avoids this fatal flaw.")


# --- a rewrite's claim about an accepted flaw is the rewrite's claim ----------


def _accepted_flaw() -> AdjudicationNote:
    return AdjudicationNote(
        candidate_id="cand_a",
        title="A TiO2 coating",
        resolution="override",
        adjudicator="Automated verification run",
        justification="Proceeding under containment.",
        fatal_flaws=["The HF precursor cannot be contained at atmospheric pressure."],
    )


def test_a_rewrite_saying_it_removes_an_accepted_flaw_is_not_left_to_stand_alone():
    """A live chapter carried a rewrite that achieves the same layer "while eliminating
    the safety fatal flaw" twenty lines above a conclusion saying the flaw was allowed
    to stand and no work in the run removes it. Both are true of different texts, and
    nothing on the page said which each was about."""
    from coscientist.narrative import _revised_form

    lead_in, _, _ = _revised_form(
        _twice_revised(), _candidate("cand_a"), accepted_flaw=_accepted_flaw()
    )

    assert "accepted against the form ranked above" in lead_in
    assert "was not re-run against the rewrite" in lead_in
    assert "that is the rewrite's own claim and nothing in this run has tested it" in (
        lead_in
    )


def test_a_rewrite_the_safety_review_did_see_is_reported_as_seen():
    from coscientist.narrative import _revised_form

    record = _twice_revised()
    record.evolution.rereviews = [
        _review("cand_a_v3", criterion="safety_governance", recommendation="advance")
    ]

    lead_in, _, _ = _revised_form(
        record, _candidate("cand_a"), accepted_flaw=_accepted_flaw()
    )

    assert "was re-run against the rewrite." in lead_in
    assert "nothing in this run has tested it" not in lead_in


def test_an_idea_with_no_accepted_flaw_says_nothing_about_one():
    from coscientist.narrative import _revised_form

    lead_in, _, _ = _revised_form(_twice_revised(), _candidate("cand_a"))

    assert "fatal flaw" not in lead_in
