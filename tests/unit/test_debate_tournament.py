"""The ranking tournament must be judged, not asserted.

Covers the section 9.3 verdict terminator in the spellings models actually
emit, the position-bias rotation, the Elo arithmetic that follows from a
verdict, and the guarantee that the offline provider still gets the
deterministic arithmetic tournament.
"""

from __future__ import annotations

import threading
import time
from itertools import pairwise

import pytest

from coscientist.agents import (
    RANKING_JUDGE_CONTRACT,
    SPECIALISTS,
    ContractViolation,
    DeterministicProvider,
    Specialist,
    output_contract,
)
from coscientist.debate import (
    COMPARISON_PROMPT,
    DEBATE_PROMPT,
    MIN_DEBATE_TURNS,
    NO_VERDICT_RATIONALE,
    JudgeContext,
    comparison_prompt,
    debate_prompt,
    parse_verdict,
    readable_exchange,
    readable_turn,
    run_debate_tournament,
    strip_rationale_label,
    strip_turn_label,
)
from coscientist.models import ApprovalProfile, TournamentState
from coscientist.orchestration import CoScientistWorkflow
from coscientist.parity import DEFAULT_ELO, ELO_K, population_from_artifacts

QUESTION = "Can a protective coating improve lithium-ion battery cycle life?"


class _JudgeProvider:
    """A live-looking provider that answers every match with a scripted verdict."""

    model_id = "scripted-judge"

    def __init__(self, *, verdict: str = "better idea: <1>"):
        self.verdict = verdict
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, *, role: str, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
        return (
            "Turn 1: both hypotheses target interface stabilisation.\n\n"
            "Turn 2: hypothesis 1 states a falsifier the reviews can check.\n\n"
            "Turn 3: hypothesis 2 leaves the mediator undefined.\n\n"
            f"Judgment: hypothesis 1 is more decidable. {self.verdict}"
        )


def _ranked_session():
    """A session advanced far enough that a candidate population exists."""
    flow = CoScientistWorkflow(
        QUESTION,
        approval_profile=ApprovalProfile.AUTO,
        # v1 skips the Deep Research evidence gate, which the offline suite
        # cannot satisfy; the ranking stage's inputs are identical either way.
        workflow_version=1,
    )
    while flow.stage != "rank" and not flow.done:
        flow.accept(flow.preview(), automatic=True)
    return flow.session


# --------------------------------------------------------------------------
# Verdict parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "winner"),
    [
        ("Reasoning here.\n\nbetter idea: <1>", 1),
        ("Reasoning here.\n\nbetter idea: 1", 1),
        ("Reasoning here.\n\nBetter Idea: 2", 2),
        ("Reasoning here.\n\nbetter idea: <2>.", 2),
        ("Reasoning here.\n\n**better idea: 2**", 2),
        # The paper's single-turn prompt closes by asking for this spelling.
        ("Reasoning here.\n\nbetter hypothesis: <1>", 1),
        ("Reasoning here.\n\nBETTER IDEA:2!", 2),
    ],
)
def test_every_accepted_terminator_spelling_is_parsed(text: str, winner: int):
    verdict = parse_verdict(text)
    assert verdict.winner == winner
    assert verdict.decided
    assert verdict.rationale == "Reasoning here."


def test_no_terminator_is_reported_as_no_verdict_rather_than_guessed():
    verdict = parse_verdict(
        "Hypothesis 1 looks stronger overall and I would probably pick it."
    )
    assert verdict.winner is None
    assert not verdict.decided
    assert verdict.confidence == 0.0
    assert verdict.rationale == NO_VERDICT_RATIONALE


def test_an_echoed_instruction_is_not_a_verdict():
    """`better idea: <1 or 2>` is the prompt talking, not the judge."""
    assert parse_verdict('Please conclude with "better idea: <1 or 2>".').winner is None


def test_the_last_terminator_wins_when_the_prompt_is_echoed_first():
    text = (
        'I must end with "better idea: <1>" style output.\n\n'
        "On reflection hypothesis 2 is better.\n\nbetter idea: 2"
    )
    assert parse_verdict(text).winner == 2


def test_a_stated_confidence_is_preferred_over_the_depth_heuristic():
    assert parse_verdict("Sure.\n\nConfidence: 0.4\n\nbetter idea: 1").confidence == 0.4
    assert (
        parse_verdict("Sure.\n\nConfidence: 85%\n\nbetter idea: 1").confidence == 0.85
    )


def test_the_transcript_is_split_into_turns():
    verdict = parse_verdict(
        "Turn 1: summary of both.\n\nTurn 2: critique.\n\n"
        "Turn 3: judgment.\n\nbetter idea: 1"
    )
    assert len(verdict.turns) == 3
    assert verdict.turns[0].startswith("Turn 1")


# --------------------------------------------------------------------------
# Prompt fidelity
# --------------------------------------------------------------------------


def test_the_prompts_are_the_papers_prompts_with_the_real_variables_filled_in():
    session = _ranked_session()
    population = population_from_artifacts(session.artifacts)
    context = JudgeContext.build(session, population)
    first, second = population.candidates[0], population.candidates[1]

    single = comparison_prompt(context, first, second)
    assert single.startswith("You are an expert evaluator tasked with comparing two")
    assert 'concluding with the phrase "better idea: <1 or 2>"' in single
    assert "Disregard these scores in your comparative analysis" in single
    assert first.claim in single and second.claim in single
    assert QUESTION in single

    debate = debate_prompt(context, first, second)
    assert debate.startswith("You are an expert in comparative analysis")
    assert "typically ranging from 3 to 5, with a maximum of 10" in debate
    assert 'writing the phrase "better idea: "' in debate
    assert first.claim in debate and second.claim in debate
    for placeholder in ("{goal}", "{preferences}", "{notes}", "{hypothesis 1}"):
        assert placeholder not in debate
        assert placeholder not in single
    # Both templates keep the paper's placeholder names verbatim.
    assert "{idea_attributes}" in COMPARISON_PROMPT
    assert "{review 2}" in DEBATE_PROMPT


# --------------------------------------------------------------------------
# Tournament mechanics
# --------------------------------------------------------------------------


def test_presentation_order_alternates_across_matches():
    session = _ranked_session()
    state, _ = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    firsts = [
        comparison.presented_first_id == comparison.candidate_a_id
        for comparison in state.comparisons
    ]
    assert firsts[0] is True
    # A judge that always answers "1" would sweep the tournament if the reading
    # order never rotated; alternation is the control for exactly that.
    assert all(earlier != later for earlier, later in pairwise(firsts))
    assert set(firsts) == {True, False}


def test_a_verdict_moves_elo_by_the_shared_constants():
    session = _ranked_session()
    state, _ = run_debate_tournament(session, _JudgeProvider(), max_workers=2)
    first = state.comparisons[0]
    # Comparison 0 is a Swiss round, which is one shot of prose rather than an
    # argument, so it is labelled as such and carries no transcript.
    assert first.judge == "llm_comparison"
    assert first.debate_turns == []
    debated = [item for item in state.comparisons if item.judge == "llm_debate"]
    assert debated and all(item.debate_turns for item in debated)
    assert first.winner_id is not None
    left, right = first.candidate_a_id, first.candidate_b_id
    assert first.elo_before == {left: DEFAULT_ELO, right: DEFAULT_ELO}
    # Equal ratings mean an expected score of 0.5, so the winner takes K/2.
    gain = ELO_K / 2
    winner, loser = (left, right) if first.winner_id == left else (right, left)
    assert first.elo_after[winner] == pytest.approx(DEFAULT_ELO + gain)
    assert first.elo_after[loser] == pytest.approx(DEFAULT_ELO - gain)
    assert sum(state.ratings.values()) == pytest.approx(
        DEFAULT_ELO * len(state.ratings)
    )


def test_an_undecided_match_is_a_draw_and_moves_no_elo():
    session = _ranked_session()
    provider = _JudgeProvider(verdict="I cannot choose between them.")
    state, transcript = run_debate_tournament(session, provider, max_workers=2)
    assert state.comparisons
    for comparison in state.comparisons:
        assert comparison.winner_id is None
        assert comparison.judge in {"llm_comparison", "llm_debate"}
        assert comparison.elo_before == comparison.elo_after
        assert "did not reach a verdict" in comparison.rationale
    assert set(state.ratings.values()) == {DEFAULT_ELO}
    assert "no verdict reached" in transcript


def test_the_swiss_rounds_are_cheap_and_every_finalist_pair_debates():
    session = _ranked_session()
    provider = _JudgeProvider()
    state, _ = run_debate_tournament(session, provider, max_workers=4)
    assert len(provider.prompts) == len(state.comparisons)
    swiss = [
        comparison for comparison in state.comparisons if comparison.round_number <= 3
    ]
    finals = [
        comparison for comparison in state.comparisons if comparison.round_number == 4
    ]
    assert len(swiss) == 12  # eight candidates, three rounds of four matches
    # All six top-four pairs, including any that already met in Swiss: the
    # expensive multi-turn judgement is what the round robin is for.
    assert len(finals) == 6
    single_turn = sum(
        1
        for prompt in provider.prompts
        if prompt.startswith("You are an expert evaluator")
    )
    debates = sum(
        1
        for prompt in provider.prompts
        if prompt.startswith("You are an expert in comparative analysis")
    )
    assert single_turn == 12
    assert debates == 6


def test_a_finalist_pair_that_met_in_swiss_is_debated_again_and_marked():
    session = _ranked_session()
    state, transcript = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    finals = [
        comparison for comparison in state.comparisons if comparison.round_number == 4
    ]
    rematches = [
        comparison for comparison in finals if "[Rematch:" in comparison.rationale
    ]
    assert rematches, "the top four always contain a pair that met during Swiss"
    for comparison in rematches:
        pair = {comparison.candidate_a_id, comparison.candidate_b_id}
        earlier = [
            other
            for other in state.comparisons
            if other.round_number <= 3
            and {other.candidate_a_id, other.candidate_b_id} == pair
        ]
        assert len(earlier) == 1
        # Both judgements survive in the record, and the debate is the later
        # Elo update, so it is the one that decides the standings.
        assert state.comparisons.index(comparison) > state.comparisons.index(earlier[0])
    assert "rematch of Swiss round" in transcript


def test_the_transcript_names_both_ideas_and_carries_the_debate():
    session = _ranked_session()
    state, transcript = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    assert transcript.startswith("# Ranking tournament transcript")
    assert "## Round 1: Swiss round 1 — single-turn comparison" in transcript
    assert "Top-four round robin — simulated scientific debate" in transcript
    assert "Turn 2: hypothesis 1 states a falsifier" in transcript
    population = population_from_artifacts(session.artifacts)
    for candidate in population.candidates[:2]:
        assert candidate.id in transcript
    assert "## Final standings" in transcript
    for candidate_id in state.shortlist_ids:
        assert candidate_id in transcript


def test_matches_in_a_round_are_judged_concurrently_within_the_bound():
    """Independent matches must overlap, and never exceed ``max_workers``."""
    session = _ranked_session()
    peak = 0
    live = 0
    lock = threading.Lock()

    class _ConcurrentJudge(_JudgeProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            nonlocal peak, live
            with lock:
                live += 1
                peak = max(peak, live)
            # Long enough that a serial implementation could never overlap and
            # short enough not to slow the suite.
            time.sleep(0.05)
            with lock:
                live -= 1
            return super().complete(role=role, prompt=prompt)

    run_debate_tournament(session, _ConcurrentJudge(), max_workers=3)
    assert peak == 3


# --------------------------------------------------------------------------
# A degenerate judge must fail loudly
# --------------------------------------------------------------------------

# Verbatim shape of the live failure: the A2A `ranking` specialist carried a
# server-side instruction to emit a TournamentState, which beat the section 9.3
# user prompt. The model returned a whole tournament with invented ids, and its
# nested `rationale` ended in "better idea: 2" -- so the verdict parsed, Elo
# moved, and the tournament looked healthy.
LIVE_DEGENERATE_BLOB = """{
  "ratings": {"hypothesis_1": 1185.0, "hypothesis_2": 1215.0},
  "comparisons": [
    {
      "id": "comp_1",
      "round_number": 1,
      "candidate_a_id": "hypothesis_1",
      "candidate_b_id": "hypothesis_2",
      "presented_first_id": "hypothesis_1",
      "winner_id": "hypothesis_2",
      "criterion_scores": {"novelty": 0.5, "feasibility": 1.0},
      "rationale": "Hypothesis 1 has an unresolvable fatal flaw: the \
microcapsules rupture during calendering. Hypothesis 2 is more feasible. \
better idea: 2",
      "confidence": 0.95,
      "elo_before": {"hypothesis_1": 1200.0, "hypothesis_2": 1200.0},
      "elo_after": {"hypothesis_1": 1185.0, "hypothesis_2": 1215.0},
      "rubric_version": "1",
      "debate_turns": [],
      "judge": "deterministic"
    }
  ],
  "shortlist_ids": ["hypothesis_2"],
  "swiss_rounds": 3,
  "top_round_robin_size": 4,
  "ranking_stable_rounds": 1,
  "score_movement": 15.0,
  "converged": true
}"""


class _ContractEchoProvider(_JudgeProvider):
    """Answers a debate request with a TournamentState, as the live server did."""

    def complete(self, *, role: str, prompt: str) -> str:
        super().complete(role=role, prompt=prompt)
        return LIVE_DEGENERATE_BLOB


def test_a_tournament_state_blob_never_becomes_a_healthy_looking_tournament():
    """The exact live failure: a serialized contract instead of a debate."""
    session = _ranked_session()
    with pytest.raises(ContractViolation) as raised:
        run_debate_tournament(session, _ContractEchoProvider(), max_workers=2)
    assert raised.value.role == "ranking"
    assert "did not judge the match" in str(raised.value)
    assert "JSON document" in raised.value.error
    # The offending payload is preserved for the operator, not swallowed.
    assert "hypothesis_2" in raised.value.content


def test_the_blob_would_otherwise_have_parsed_as_a_clean_win():
    """Pins why the failure was silent: the nested rationale ends correctly."""
    verdict = parse_verdict(LIVE_DEGENERATE_BLOB)
    assert verdict.winner == 2, "regression guard: this is what looked healthy"
    # ...but the rationale must never be the JSON dump itself.
    assert not verdict.rationale.lstrip().startswith("{")


def test_a_contract_shaped_response_wrapped_in_prose_is_also_rejected():
    session = _ranked_session()

    class _FencedProvider(_JudgeProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            super().complete(role=role, prompt=prompt)
            return (
                "Here is the tournament you asked for.\n\n"
                '{"ratings": {"a": 1200.0}, "shortlist_ids": ["a"], '
                '"comparisons": []}\n\nbetter idea: 1'
            )

    with pytest.raises(ContractViolation) as raised:
        run_debate_tournament(session, _FencedProvider(), max_workers=2)
    assert "serialized contract" in raised.value.error


def test_invented_candidate_identifiers_are_rejected():
    session = _ranked_session()

    class _InventedIdProvider(_JudgeProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            super().complete(role=role, prompt=prompt)
            return (
                "Turn 1: comparing cand_alpha_7 and cand_beta_3.\n\n"
                "Turn 2: cand_alpha_7 has the stronger falsifier.\n\n"
                "Turn 3: concluding.\n\nbetter idea: 1"
            )

    with pytest.raises(ContractViolation) as raised:
        run_debate_tournament(session, _InventedIdProvider(), max_workers=2)
    assert "invented" in raised.value.error
    assert "cand_alpha_7" in raised.value.error


def test_a_debate_that_never_debated_is_rejected():
    """One turn is right for a Swiss comparison and wrong for a debate."""
    session = _ranked_session()

    class _OneTurnProvider(_JudgeProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            super().complete(role=role, prompt=prompt)
            return "Hypothesis 1 is better on every criterion. better idea: 1"

    with pytest.raises(ContractViolation) as raised:
        run_debate_tournament(session, _OneTurnProvider(), max_workers=2)
    # The twelve Swiss matches accepted the same single-turn answer; only the
    # debate round demands deliberation.
    assert f"at least {MIN_DEBATE_TURNS}" in raised.value.error
    assert "1 turn(s)" in raised.value.error


def test_judge_supplied_criterion_scores_are_kept():
    session = _ranked_session()

    class _ScoringProvider(_JudgeProvider):
        def complete(self, *, role: str, prompt: str) -> str:
            super().complete(role=role, prompt=prompt)
            return (
                "Turn 1: summary of hypothesis 1.\n"
                "- Novelty: 4/5\n"
                "- Feasibility: 0.9\n\n"
                "Turn 2: summary of hypothesis 2.\n"
                "- Novelty: 2/5\n\n"
                "Turn 3: hypothesis 1 wins on feasibility.\n\n"
                "Confidence: 0.8\n\nbetter idea: 1"
            )

    state, transcript = run_debate_tournament(
        session, _ScoringProvider(), max_workers=2
    )
    scores = state.comparisons[0].criterion_scores
    assert scores, "scores the judge volunteered were being dropped"
    assert scores["hypothesis 1: novelty"] == pytest.approx(0.8)
    assert scores["hypothesis 2: novelty"] == pytest.approx(0.4)
    assert scores["hypothesis 1: feasibility"] == pytest.approx(0.9)
    assert "Judge-supplied criterion scores" in transcript
    # A judge-stated confidence must not be presented as a depth proxy.
    assert state.comparisons[0].confidence == pytest.approx(0.8)
    assert "0.80 (judge-stated)" in transcript


def test_an_unstated_confidence_is_labelled_as_deliberation_depth():
    session = _ranked_session()
    _, transcript = run_debate_tournament(session, _JudgeProvider(), max_workers=2)
    assert "deliberation depth, not a calibrated probability" in transcript
    assert "must not be averaged as one" in transcript


# --------------------------------------------------------------------------
# The server-side contract must not order a TournamentState
# --------------------------------------------------------------------------


def test_the_live_ranking_specialist_is_told_to_judge_one_match_not_rank():
    """The root cause: a system instruction that beat the section 9.3 prompt."""
    assert "TournamentState" not in RANKING_JUDGE_CONTRACT
    assert "ONE pair" in RANKING_JUDGE_CONTRACT
    assert "better idea: <1 or 2>" in RANKING_JUDGE_CONTRACT
    assert "Emit no JSON" in RANKING_JUDGE_CONTRACT
    # The typed artifact contract is unchanged, because the Supervisor still
    # has to produce a TournamentState for every downstream consumer.
    assert "TournamentState" in output_contract("ranking")


def test_the_adk_ranking_agent_carries_the_judge_contract():
    adk = pytest.importorskip("google.adk.agents")
    assert adk
    from coscientist.agents import build_adk_workflow

    workflow = build_adk_workflow()
    ranking = next(item for item in workflow.sub_agents if item.name == "ranking")
    assert "TournamentState" not in ranking.instruction
    assert "better idea: <1 or 2>" in ranking.instruction
    reflection = next(item for item in workflow.sub_agents if item.name == "reflection")
    assert "ReviewSet" in reflection.instruction  # other roles keep their contract


# --------------------------------------------------------------------------
# The offline path must not change
# --------------------------------------------------------------------------


def _ranking_specialist() -> Specialist:
    return next(item for item in SPECIALISTS if item.role == "ranking")


def test_the_deterministic_provider_still_gets_the_arithmetic_tournament():
    session = _ranked_session()
    artifact = _ranking_specialist().run(session, DeterministicProvider())
    assert artifact.schema_name == "TournamentState"
    assert artifact.payload_source == "deterministic_fallback"
    state = TournamentState.model_validate(artifact.payload)
    assert state.comparisons
    assert all(
        comparison.judge == "deterministic" and not comparison.debate_turns
        for comparison in state.comparisons
    )
    # The offline provider's ranking template, not a debate transcript.
    assert artifact.content.startswith("Prioritization (provisional)")


def test_a_live_provider_gets_the_debate_tournament_from_the_ranking_specialist():
    session = _ranked_session()
    provider = _JudgeProvider()
    artifact = _ranking_specialist().run(session, provider)
    assert artifact.schema_name == "TournamentState"
    assert artifact.payload_source == "specialist"
    assert artifact.producer_model == "scripted-judge"
    assert artifact.content.startswith("# Ranking tournament transcript")
    state = TournamentState.model_validate(artifact.payload)
    assert state.comparisons
    # Every match was judged by the model, and exactly the finals were debated.
    assert all(
        comparison.judge in {"llm_comparison", "llm_debate"}
        for comparison in state.comparisons
    )
    assert any(comparison.judge == "llm_debate" for comparison in state.comparisons)
    # The generic single-shot ranking prompt must not have been sent at all.
    assert not any("Prior work:" in prompt for prompt in provider.prompts)


def test_the_protocol_terminator_is_not_part_of_the_transcript():
    """ "... is the vastly superior choice. better idea: 1." was printed verbatim."""
    verdict = parse_verdict(
        "Turn 1: The panel convenes.\n\n"
        "Turn 2: Hypothesis 1 isolates the mechanism, so it wins. better idea: 1."
    )

    assert verdict.winner == 1
    assert all("better idea" not in turn for turn in verdict.turns)
    # The terminator took the sentence's full stop with it; the turn keeps one.
    assert verdict.turns[-1].endswith("so it wins.")


def test_turn_labels_are_normalised_however_the_judge_wrote_them():
    """One transcript alternated "Turn 1:" and "**Turn 2:**" down the same page."""
    verdict = parse_verdict(
        "Turn 1: The panel convenes.\n\n**Turn 2:** The panel decides. better idea: 2."
    )

    assert [turn.split(":")[0] for turn in verdict.turns] == ["Turn 1", "Turn 2"]


def test_the_judges_own_rationale_heading_is_not_kept():
    """Stripping the outer asterisks off "**Rationale:**" left "Rationale:**"."""
    verdict = parse_verdict(
        "Turn 1: The panel convenes.\n\n"
        "Turn 2: Discussion.\n\n"
        "**Rationale:** Hypothesis 1 isolates the mechanism.\n\nbetter idea: 1"
    )

    assert verdict.rationale == "Hypothesis 1 isolates the mechanism."


def test_a_rationale_label_after_a_bracketed_note_is_stripped_too():
    note = "[Rematch: this pair also met in Swiss round 1.] Rationale: H1 is better."

    assert strip_rationale_label(note) == (
        "[Rematch: this pair also met in Swiss round 1.] H1 is better."
    )


def test_a_turn_ends_on_a_full_stop_once_the_terminator_is_gone():
    assert readable_turn("Turn 3: The kinetics argument holds better idea: 2") == (
        "Turn 3: The kinetics argument holds."
    )


def test_a_turn_that_already_ends_on_punctuation_gains_nothing():
    assert readable_turn("Turn 1: Does the coating conduct? better idea: 1") == (
        "Turn 1: Does the coating conduct?"
    )


def test_the_panel_is_named_the_same_way_in_every_transcript():
    """Each match is judged by its own call, and the judge picks A/B/C or 1/2/3. Ten
    transcripts of one run came back lettered and two numbered, so a reader who had
    followed Expert A through four matches met Expert 1 in the fifth."""
    numbered = readable_exchange(
        "Turn 2: Expert 1: The barrier reading is stronger. "
        "Expert 2: The kinetic reading explains the same data."
    )
    assert [prefix for prefix, _ in numbered] == [
        "Turn 2, Expert A",
        "Turn 2, Expert B",
    ]


def test_a_contribution_opens_on_a_capital_once_its_label_is_off_the_front():
    """ "Rationale: Hypothesis 1 provides ..." is one sentence whose capital belonged
    to the label. Split into a label and a body, the body opened a bullet in lower
    case, because the slot name it started on had been rewritten as "this idea"."""
    exchange = readable_exchange(
        "Turn 4: Expert A: Both are plausible. Rationale: this idea is better specified."
    )
    assert exchange[-1] == (
        "Turn 4, Closing rationale",
        "This idea is better specified.",
    )


def test_a_turn_label_carried_into_a_rationale_is_dropped():
    assert strip_turn_label("Turn 4: Final evaluation. Hypothesis 2 is stronger.") == (
        "Final evaluation. Hypothesis 2 is stronger."
    )


# --------------------------------------------------------------------------
# The comparison budget
# --------------------------------------------------------------------------


def test_the_default_budget_buys_the_full_three_swiss_rounds():
    """Eighteen comparisons over eight candidates is exactly the design.

    So the guard below must be inert on a run that is inside its budget: if it
    were to shorten this tournament, every ordinary run would be ranked on less
    evidence than the design calls for.
    """
    session = _ranked_session()
    assert session.budget.max_pairwise_comparisons == 18
    state, _ = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    assert state.swiss_rounds == 3
    assert len(state.comparisons) == 18


def test_a_smaller_budget_buys_fewer_swiss_rounds_and_says_so():
    """The finals are held back from the reduction and the transcript discloses it.

    A shortened tournament separates the field less confidently, so a reader
    comparing Elo across runs has to be told which one they are reading.
    """
    session = _ranked_session()
    session.budget.max_pairwise_comparisons = 12
    state, transcript = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    # (12 - 6 finals) // 4 matches a round.
    assert state.swiss_rounds == 1
    assert len(state.comparisons) == 10
    assert sum(1 for item in state.comparisons if item.round_number == 4) == 6
    assert "Structure: 1 Swiss rounds" in transcript
    assert "to stay inside the session's 12-comparison budget" in transcript


def test_a_budget_too_small_for_any_swiss_round_still_seeds_the_finals():
    """One round survives whatever the budget says.

    With none, the top four would be chosen off the default rating every
    candidate starts on, which is to say alphabetically.
    """
    session = _ranked_session()
    session.budget.max_pairwise_comparisons = 1
    state, _ = run_debate_tournament(session, _JudgeProvider(), max_workers=4)
    assert state.swiss_rounds == 1
