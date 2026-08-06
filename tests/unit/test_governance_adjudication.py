"""A safety gate needs an exit a human can actually use.

The reflect stage halts on a fatal governance finding, which is right: a live
run stopped a hypothesis that would have annealed assembled electrodes at 400 C
and vented hydrogen fluoride into a lab. But until now nothing could answer the
block, so the only way past it was to edit the session file. These tests pin
both exits -- withdraw and override -- and, just as importantly, pin that
neither can be taken anonymously or without a written reason.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coscientist.governance import (
    WithdrawalRefused,
    governance_blockers,
    latest_population,
    open_blockers,
    withdraw_candidate,
    withdrawn_candidate_ids,
)
from coscientist.models import (
    Artifact,
    ArtifactStatus,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    DecisionAction,
    GovernanceAdjudication,
    ReviewSet,
    Session,
)
from coscientist.orchestration import CoScientistWorkflow

FLAW = (
    "Annealing PVDF-containing electrodes at 400 C decomposes the binder and "
    "releases hydrogen fluoride."
)
REASON = "Confirmed against the binder datasheet; the protocol cannot be run safely."


def _candidate(candidate_id: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        generation_strategy="mechanism_first",
        title=f"Hypothesis {candidate_id}",
        claim=f"Hypothesis {candidate_id}.",
        rationale="Because the mechanism predicts it.",
        mechanism_model="The coating blocks the reaction that drives fade.",
        validation_protocol="Coin cells against an uncoated control.",
        predictions=["Capacity retention improves."],
        falsifier="Retention does not improve.",
    )


def _review(review_id: str, candidate_id: str, *flaws: str) -> CandidateReview:
    return CandidateReview(
        id=review_id,
        candidate_id=candidate_id,
        criterion="safety_governance",
        reviewer="ethics_safety_governance",
        recommendation="reject" if flaws else "advance",
        fatal_flaws=list(flaws),
    )


def _session(*, candidates: int = 3, reviews: list[CandidateReview] | None = None):
    session = Session(question="Can a coating extend cycle life?")
    population = CandidatePopulation(
        candidates=[_candidate(f"cand_{n}") for n in range(1, candidates + 1)],
        target_size=candidates,
    )
    session.artifacts.append(
        Artifact(
            stage="generate",
            agent="generation",
            content="",
            schema_name="CandidatePopulation",
            payload=population.model_dump(mode="json"),
            status=ArtifactStatus.ACCEPTED,
        )
    )
    if reviews:
        session.artifacts.append(
            Artifact(
                stage="reflect",
                agent="ethics_safety_governance",
                content="",
                schema_name="ReviewSet",
                payload=ReviewSet(reviews=reviews).model_dump(mode="json"),
                status=ArtifactStatus.ACCEPTED,
            )
        )
    return session


def _blocked_flow() -> CoScientistWorkflow:
    session = _session(reviews=[_review("rev_1", "cand_2", FLAW)])
    session.status = "governance_blocked"
    return CoScientistWorkflow("Can a coating extend cycle life?", session=session)


def test_only_a_fatal_flaw_blocks_not_a_mere_objection():
    session = _session(
        reviews=[_review("rev_1", "cand_1"), _review("rev_2", "cand_2", FLAW)]
    )
    assert [item.review_id for item in governance_blockers(session)] == ["rev_2"]


def test_an_unanswered_finding_stays_open():
    flow = _blocked_flow()
    assert [item.review_id for item in open_blockers(flow.session)] == ["rev_1"]


def test_withdrawing_removes_the_hypothesis_and_clears_the_block():
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    assert flow.session.status == "active"
    surviving = CandidatePopulation.model_validate(
        latest_population(flow.session).payload
    )
    assert [item.id for item in surviving.candidates] == ["cand_1", "cand_3"]
    assert withdrawn_candidate_ids(flow.session) == {"cand_2"}


def test_the_superseded_population_is_kept_as_history():
    """What a stage was reviewed against must stay readable afterwards."""
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    populations = [
        item
        for item in flow.session.artifacts
        if item.schema_name == "CandidatePopulation"
    ]
    assert len(populations) == 2
    assert populations[0].status == ArtifactStatus.SUPERSEDED
    assert populations[1].parent_id == populations[0].id
    original = CandidatePopulation.model_validate(populations[0].payload)
    assert [item.id for item in original.candidates] == ["cand_1", "cand_2", "cand_3"]


def test_overriding_keeps_the_hypothesis_and_clears_the_block():
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1",
        "override",
        adjudicator="Safety Officer",
        justification="Accepted with a fume hood and a written control plan.",
    )
    assert flow.session.status == "active"
    surviving = CandidatePopulation.model_validate(
        latest_population(flow.session).payload
    )
    assert [item.id for item in surviving.candidates] == ["cand_1", "cand_2", "cand_3"]
    assert withdrawn_candidate_ids(flow.session) == set()


def test_the_flaw_is_frozen_onto_the_decision():
    """A later revision must not be able to change what was approved."""
    flow = _blocked_flow()
    adjudication = flow.adjudicate_governance(
        "rev_1", "override", adjudicator="Safety Officer", justification=REASON
    )
    assert adjudication.fatal_flaws == [FLAW]
    assert adjudication.candidate_id == "cand_2"


def test_a_session_stays_blocked_until_every_finding_is_answered():
    session = _session(
        candidates=4,
        reviews=[
            _review("rev_1", "cand_2", FLAW),
            _review("rev_2", "cand_3", "Uncontrolled dual-use risk."),
        ],
    )
    session.status = "governance_blocked"
    flow = CoScientistWorkflow("Can a coating extend cycle life?", session=session)
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    assert flow.session.status == "governance_blocked"
    flow.adjudicate_governance(
        "rev_2", "override", adjudicator="Safety Officer", justification=REASON
    )
    assert flow.session.status == "active"


def test_an_anonymous_decision_is_refused():
    with pytest.raises(ValidationError, match="adjudicator"):
        GovernanceAdjudication(
            review_id="rev_1",
            candidate_id="cand_2",
            resolution="override",
            adjudicator="   ",
            justification=REASON,
        )


def test_an_unexplained_decision_is_refused():
    with pytest.raises(ValidationError, match="justification"):
        GovernanceAdjudication(
            review_id="rev_1",
            candidate_id="cand_2",
            resolution="override",
            adjudicator="Safety Officer",
            justification="",
        )


def test_the_same_finding_cannot_be_adjudicated_twice():
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "override", adjudicator="Safety Officer", justification=REASON
    )
    flow.session.status = "governance_blocked"
    with pytest.raises(ValueError, match="already been adjudicated"):
        flow.adjudicate_governance(
            "rev_1", "withdraw", adjudicator="Someone Else", justification=REASON
        )


def test_an_unknown_review_id_names_the_open_ones():
    flow = _blocked_flow()
    with pytest.raises(ValueError, match="rev_1"):
        flow.adjudicate_governance(
            "rev_typo", "withdraw", adjudicator="Safety Officer", justification=REASON
        )


def test_an_invented_resolution_is_refused():
    flow = _blocked_flow()
    with pytest.raises(ValueError, match="Unknown governance resolution"):
        flow.adjudicate_governance(
            "rev_1", "ignore", adjudicator="Safety Officer", justification=REASON
        )
    assert flow.session.governance_adjudications == []


def test_adjudication_is_refused_on_a_session_that_is_not_blocked():
    session = _session(reviews=[_review("rev_1", "cand_2", FLAW)])
    flow = CoScientistWorkflow("Can a coating extend cycle life?", session=session)
    with pytest.raises(ValueError, match="only to a blocked session"):
        flow.adjudicate_governance(
            "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
        )


def test_withdrawing_the_last_hypothesis_is_refused_and_changes_nothing():
    session = _session(candidates=1, reviews=[_review("rev_1", "cand_1", FLAW)])
    session.status = "governance_blocked"
    flow = CoScientistWorkflow("Can a coating extend cycle life?", session=session)
    with pytest.raises(WithdrawalRefused):
        flow.adjudicate_governance(
            "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
        )
    assert flow.session.governance_adjudications == []
    assert flow.session.status == "governance_blocked"
    surviving = CandidatePopulation.model_validate(
        latest_population(flow.session).payload
    )
    assert [item.id for item in surviving.candidates] == ["cand_1"]


def test_the_decision_is_written_into_the_audit_trail():
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    event = next(
        item
        for item in flow.session.events
        if item.event_type == "governance_adjudicated"
    )
    assert event.actor == "Safety Officer"
    assert event.payload["review_id"] == "rev_1"
    assert event.payload["resolution"] == "withdraw"
    assert event.payload["justification"] == REASON
    assert event.payload["remaining_blocker_ids"] == []


def test_withdrawing_a_candidate_absent_from_the_population_records_no_revision():
    """The finding is still answered; there is simply nothing to remove."""
    session = _session(reviews=[_review("rev_1", "cand_absent", FLAW)])
    session.status = "governance_blocked"
    flow = CoScientistWorkflow("Can a coating extend cycle life?", session=session)
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    assert flow.session.status == "active"
    populations = [
        item
        for item in flow.session.artifacts
        if item.schema_name == "CandidatePopulation"
    ]
    assert len(populations) == 1


def test_withdraw_candidate_is_a_no_op_without_a_population():
    assert withdraw_candidate(Session(question="Q"), "cand_1") is None


def test_an_adjudicated_session_survives_a_save_and_reload(tmp_path):
    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    path = tmp_path / "session.json"
    flow.save(path)
    reloaded = CoScientistWorkflow.load(path)
    assert len(reloaded.session.governance_adjudications) == 1
    assert reloaded.session.governance_adjudications[0].justification == REASON
    assert open_blockers(reloaded.session) == []


def test_a_session_written_before_adjudications_existed_still_loads(tmp_path):
    session = _session()
    payload = session.to_dict()
    del payload["governance_adjudications"]
    restored = Session.from_dict(payload)
    assert restored.governance_adjudications == []


# --------------------------------------------------------------------------
# Populations were always even until a withdrawal made one odd.
# --------------------------------------------------------------------------


def _population_session(size: int) -> Session:
    session = _session(candidates=size)
    session.research_mode = "experimental"
    return session


def test_a_deterministic_tournament_measures_the_movement_it_used_to_hard_code():
    """``parity`` returned the sentinel beside real ratings it had just computed.

    ``score_movement=1.0`` and ``ranking_stable_rounds=1`` were placeholders standing
    where the rounds had in fact been played and the ratings recorded. The report read
    both as measurements and printed "the final round moved one rating by 100.0 per
    cent of that, or about 1200 points" -- a rating falling to zero in one round, which
    a K factor of 32 cannot do in any number of matches a round holds.
    """
    from coscientist.parity import (
        DEFAULT_ELO,
        ELO_K,
        UNMEASURED_MOVEMENT,
        tournament_state,
    )

    state = tournament_state(_population_session(8))
    assert state.score_movement < UNMEASURED_MOVEMENT
    # The final round is a round robin over four, so three matches per idea at most.
    assert state.score_movement * DEFAULT_ELO <= 3 * ELO_K


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 8])
def test_the_deterministic_tournament_pairs_any_population_size(size: int):
    """An odd field used to raise IndexError on the last unpaired hypothesis."""
    from coscientist.parity import tournament_state

    state = tournament_state(_population_session(size))
    assert set(state.ratings) == {f"cand_{n}" for n in range(1, size + 1)}
    assert len(state.shortlist_ids) == min(size, 4)


def test_an_unpaired_hypothesis_gets_a_bye_not_a_free_win():
    """Sitting a round out must not move a rating that nobody played against."""
    from coscientist.parity import DEFAULT_ELO, tournament_state

    state = tournament_state(_population_session(3))
    played = {
        candidate_id
        for match in state.comparisons
        for candidate_id in (match.candidate_a_id, match.candidate_b_id)
    }
    for candidate_id, rating in state.ratings.items():
        if candidate_id not in played:
            assert rating == DEFAULT_ELO


def test_a_withdrawal_leaves_a_population_the_tournament_can_still_rank():
    from coscientist.parity import tournament_state

    flow = _blocked_flow()
    flow.adjudicate_governance(
        "rev_1", "withdraw", adjudicator="Safety Officer", justification=REASON
    )
    state = tournament_state(flow.session)
    assert set(state.ratings) == {"cand_1", "cand_3"}


def test_walking_away_from_a_block_is_itself_recorded():
    """``stop`` used to return silently unless the session was active.

    The two states an operator is most likely to abandon -- an unmet evidence
    gate and an open fatal finding -- were exactly the two it ignored, so the
    file kept saying ``governance_blocked`` with no decision and no event, as
    though someone were still coming back to answer it.
    """
    flow = _blocked_flow()

    flow.stop(actor="Night Shift")

    assert flow.session.status == "stopped_by_researcher"
    assert [item.action for item in flow.session.decisions] == [DecisionAction.STOP]
    event = flow.session.events[-1]
    assert event.event_type == "session_stopped"
    assert event.actor == "Night Shift"
    assert event.payload["halted_at"] == "governance_blocked"
    assert event.payload["unanswered_governance_findings"] == ["rev_1"]


def test_an_abandoned_evidence_gate_is_recorded_too():
    flow = _blocked_flow()
    flow.session.status = "evidence_required"

    flow.stop(actor="Night Shift")

    assert flow.session.status == "stopped_by_researcher"
    event = flow.session.events[-1]
    assert event.payload["halted_at"] == "evidence_required"


def test_a_finished_session_is_not_reopened_to_be_stopped():
    flow = _blocked_flow()
    flow.session.status = "ready_for_report"

    flow.stop(actor="Night Shift")

    assert flow.session.status == "ready_for_report"
    assert flow.session.decisions == []
