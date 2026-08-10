"""Unit tests for Phase 1 Evidence and Normalization Gates."""

import pytest

from coscientist.models import (
    ApprovalProfile,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    EvidenceGap,
    EvidenceRequest,
    KnowledgeBaseManifest,
    ResearchDirection,
    ReviewSet,
    Session,
    TournamentState,
)
from coscientist.normalization import (
    NormalizationError,
    normalize_specialist_output,
    repair_json_string,
    validate_candidate_comprehensiveness,
    validate_candidate_distinctness,
    validate_no_template_leakage,
)
from coscientist.orchestration import CoScientistWorkflow


def test_knowledge_base_manifest_and_evidence_request_contracts():
    kb = KnowledgeBaseManifest(
        version=1,
        directions=[
            ResearchDirection(
                title="Direction 1",
                scope="Scope",
                mechanism_or_concept="Mech",
                outcome="Outcome",
            )
        ],
        unresolved_gaps=[
            EvidenceGap(
                direction_id="dir_1",
                description="Gap description",
                resolution_query="query",
            )
        ],
    )
    assert kb.version == 1
    assert len(kb.directions) == 1
    assert len(kb.unresolved_gaps) == 1

    ev_req = EvidenceRequest(
        requesting_stage="reflect",
        requesting_agent="reflection",
        claim_to_verify="SHP2 inhibition prevents rebound",
    )
    assert ev_req.status == "submitted"


def test_schema_repair_and_template_leakage_validators():
    malformed_json = '```json\n{"question": "Test", "runs": []\n```'
    repaired = repair_json_string(malformed_json)
    assert repaired.endswith("}")

    with pytest.raises(NormalizationError, match="Template leakage"):
        validate_no_template_leakage({"claim": "This is a TODO item"})

    with pytest.raises(NormalizationError, match="Template leakage"):
        validate_no_template_leakage({"rationale": "Placeholder for [insert here]"})


def test_candidate_distinctness_validator():
    c1 = Candidate(
        title="Candidate 1",
        claim="Same exact claim hypothesis string",
        rationale="Same exact rationale string describing the mechanism",
        mechanism_model="Same exact mechanism model describing the causal pathway",
        validation_protocol="Same exact validation protocol and experimental design",
        falsifier="Same exact falsifier condition string",
    )
    c2 = Candidate(
        title="Candidate 2",
        claim="Same exact claim hypothesis string",
        rationale="Same exact rationale string describing the mechanism",
        mechanism_model="Same exact mechanism model describing the causal pathway",
        validation_protocol="Same exact validation protocol and experimental design",
        falsifier="Same exact falsifier condition string",
    )
    pop = CandidatePopulation(candidates=[c1, c2], target_size=2)
    with pytest.raises(NormalizationError, match="fails distinctness diversity"):
        validate_candidate_distinctness(pop)


def test_v2_normalization_rejects_malformed_live_output_without_fallback():
    session = Session(question="Test question", workflow_version=2)
    malformed_content = "This is definitely not json at all {broken"
    with pytest.raises(NormalizationError, match="Failed to normalize goal_manager"):
        normalize_specialist_output(session, "goal_manager", malformed_content)


def test_evidence_delta_request_and_budget_enforcement():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.session.budget.max_searches = 1
    flow.run_auto()

    ev_req = flow.request_evidence_delta(
        requesting_stage="reflect",
        requesting_agent="reflection",
        claim_to_verify="Coating stability at 4.5V",
    )
    assert ev_req.id.startswith("evreq_")

    with pytest.raises(ValueError, match="max_searches exceeded"):
        flow.request_evidence_delta(
            requesting_stage="reflect",
            requesting_agent="reflection",
            claim_to_verify="Another claim",
        )


def test_v2_evidence_gate_blocks_generation_without_verified_claims():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=2,
    )
    scope_draft = flow.preview()
    flow.accept(scope_draft)
    assert flow.stage == "evidence"
    evidence_draft = flow.preview()
    # The refusal names the shortfall rather than the rule. An operator being
    # asked to decide whether to proceed needs to know what is missing.
    with pytest.raises(
        ValueError, match="does not meet the floor for generating hypotheses"
    ) as raised:
        flow.accept(evidence_draft)
    assert "weighted verified sources" in str(raised.value)
    assert "required evidence facets" in str(raised.value)
    assert flow.session.status == "evidence_required"

    flow.accept_exploratory_evidence()
    assert flow.session.exploratory_evidence_accepted is True


def test_v2_all_insufficient_reviews_block_ranking():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=2,
    )
    flow.session.current_stage = flow.workflow_stages.index("rank")
    from coscientist.models import Artifact, ArtifactStatus

    rev = CandidateReview(
        candidate_id="cand_1",
        criterion="evidence_correctness",
        recommendation="insufficient_evidence",
        reviewer="reflection",
    )
    rs = ReviewSet(reviews=[rev])
    art = Artifact(
        stage="reflect",
        agent="reflection",
        content="All insufficient reviews test",
        schema_name="ReviewSet",
        status=ArtifactStatus.ACCEPTED,
        payload=rs.model_dump(mode="json"),
    )
    flow.session.artifacts.append(art)

    rank_art = Artifact(
        stage="rank",
        agent="ranking",
        content="Tournament rank test",
        schema_name="TournamentState",
        status=ArtifactStatus.DRAFT,
        payload=TournamentState().model_dump(mode="json"),
    )
    flow.session.artifacts.append(rank_art)

    with pytest.raises(ValueError, match="all candidates have insufficient evidence"):
        flow.accept(rank_art)


def test_candidate_comprehensiveness_validator():
    c1 = Candidate(
        title="Candidate 1",
        claim="Claim",
        rationale="Rationale",
        mechanism_model="Too short mechanism",
        validation_protocol="Too short protocol",
        falsifier="Falsifier",
        evidence_for=["Citation 1"],
    )
    pop = CandidatePopulation(candidates=[c1], target_size=1)
    with pytest.raises(NormalizationError, match="mechanism_model has only"):
        validate_candidate_comprehensiveness(pop)

    c1.mechanism_model = (
        "This mechanism model has more than twenty words in total so that it can easily "
        "satisfy the comprehensiveness word count check enforced by our normalization validator."
    )
    with pytest.raises(NormalizationError, match="validation_protocol has only"):
        validate_candidate_comprehensiveness(pop)

    c1.validation_protocol = (
        "This validation protocol also contains more than twenty words describing the experimental "
        "and analytical study design with controls, variables, calibration, and go no go thresholds."
    )
    # Now should pass cleanly
    validate_candidate_comprehensiveness(pop)


def test_v3_candidate_schema_and_mermaid_diagrams():
    """Verify workflow_diagram_mermaid serializes and renders in dossier with summary table and badges."""
    from coscientist.dossier import compile_dossier

    flow = CoScientistWorkflow(
        "Design a 2026-State-of-the-Art Synthesis Strategy for a 45-mer Hydrophobic Therapeutic Peptide",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=2,
    )
    flow.accept_literature_only()
    draft = flow.preview()
    flow.accept(draft)
    flow.accept_exploratory_evidence()
    while not flow.done:
        draft = flow.preview()
        flow.accept(draft)
    report_md = compile_dossier(flow.session)
    assert "## Executive Candidate Summary" in report_md
    assert (
        "| Rank | Candidate Title | Strategy | Primary Claim | Falsifier Summary |"
        in report_md
    )
    assert "```mermaid" in report_md
    assert "graph TD" in report_md
    assert "[Verified Source]" in report_md or "[Literature Lead]" in report_md


def test_v3_hitl_refine_section():
    """A researcher can refine an individual section of a candidate card without regenerating the population."""
    flow = CoScientistWorkflow(
        "Design a 2026-State-of-the-Art Synthesis Strategy for a 45-mer Hydrophobic Therapeutic Peptide",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=2,
    )
    flow.accept_literature_only()
    draft = flow.preview()
    flow.accept(draft)
    flow.accept_exploratory_evidence()
    while flow.stage != "generate":
        draft = flow.preview()
        flow.accept(draft)
    draft = flow.preview()
    assert flow.stage == "generate"
    pop_art = next(
        art
        for art in reversed(flow.session.artifacts)
        if art.stage == "generate" and art.schema_name == "CandidatePopulation"
    )
    pop = CandidatePopulation.model_validate(pop_art.payload)
    target_id = pop.candidates[0].id
    old_claim = pop.candidates[0].claim

    refined_draft = flow.refine_section(
        target_id,
        "validation_protocol",
        "Add a positive control arm with known active enzyme.",
    )
    pop_after = CandidatePopulation.model_validate(refined_draft.payload)
    assert pop_after.candidates[0].id == target_id
    assert pop_after.candidates[0].claim == old_claim
    assert "Add a positive control arm" in pop_after.candidates[0].validation_protocol
    assert flow.session.decisions[-1].feedback.startswith(
        "Refined section 'validation_protocol'"
    )


# ---------------------------------------------------------------------------
# The candidate ceiling
# ---------------------------------------------------------------------------


class _Result:
    """The one attribute the aggregator reads off a dispatched task."""

    def __init__(self, artifact):
        self.artifact = artifact


def _population_artifact(strategy: str, count: int):
    from coscientist.models import Artifact

    candidates = [
        Candidate(
            title=f"{strategy} {index}",
            claim=f"{strategy} claim {index}",
            rationale=f"{strategy} rationale {index}",
            falsifier=f"{strategy} falsifier {index}",
            mechanism_model=f"{strategy} mechanism {index}",
            validation_protocol=f"{strategy} protocol {index}",
        )
        for index in range(count)
    ]
    return Artifact(
        stage="generate",
        agent=f"generation_{strategy}",
        content="",
        schema_name="CandidatePopulation",
        payload=CandidatePopulation(
            candidates=candidates, target_size=count
        ).model_dump(mode="json"),
    )


def _merged(flow, offered_per_strategy: int, strategies: int = 4):
    results = [
        _Result(_population_artifact(f"s{index}", offered_per_strategy))
        for index in range(strategies)
    ]
    return flow._merged_generation_population(results)


def _flow():
    return CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        # v1 skips the comprehensiveness check, which is about prose length and
        # not about how many candidates survive the merge.
        workflow_version=1,
    )


def test_the_generators_output_passes_through_when_it_is_inside_the_ceiling():
    flow = _flow()
    merged = _merged(flow, offered_per_strategy=2)
    population = CandidatePopulation.model_validate(merged.payload)
    assert len(population.candidates) == 8
    assert "8 distinct candidates from 4 generation strategies" in merged.content
    assert "set aside" not in merged.content


def test_over_production_is_held_to_the_budgeted_ceiling():
    """Otherwise four generators of eight give the tournament a field of 32.

    Three Swiss rounds over 32 is 48 matches before the finals, against a
    budget of 18, and a deep-dive section nobody reads.
    """
    flow = _flow()
    assert flow.session.budget.max_candidates == 8
    merged = _merged(flow, offered_per_strategy=8)
    population = CandidatePopulation.model_validate(merged.payload)
    assert len(population.candidates) == 8
    assert population.target_size == 8
    # Said out loud: a truncated field that reads as the whole one would make
    # the ranking look exhaustive over candidates it never saw.
    assert "24 further candidates were set aside" in merged.content
    assert "ceiling of 8" in merged.content


def test_the_ceiling_thins_every_strategy_rather_than_dropping_the_last():
    """Taken a rank at a time, so each strategy keeps its two strongest."""
    flow = _flow()
    merged = _merged(flow, offered_per_strategy=8)
    population = CandidatePopulation.model_validate(merged.payload)
    kept = [candidate.title for candidate in population.candidates]
    assert kept == [f"s{index} {rank}" for rank in (0, 1) for index in range(4)]


def _numbered_from_one(strategy: str, count: int):
    """A generator that numbers its own output, the way the live ones do."""
    from coscientist.models import Artifact

    return Artifact(
        stage="generate",
        agent=f"generation_{strategy}",
        content="",
        schema_name="CandidatePopulation",
        payload=CandidatePopulation(
            candidates=[
                Candidate(
                    id=f"cand_{index}",
                    title=f"{strategy} {index}",
                    claim=f"{strategy} claim {index}",
                    rationale=f"{strategy} rationale {index}",
                    falsifier=f"{strategy} falsifier {index}",
                    mechanism_model=f"{strategy} mechanism {index}",
                    validation_protocol=f"{strategy} protocol {index}",
                )
                for index in range(1, count + 1)
            ],
            target_size=count,
        ).model_dump(mode="json"),
    )


def test_two_strategies_that_both_numbered_from_one_stay_separately_addressable():
    """Eight ideas reached the tournament under six ids on a live production run.

    ``generation`` and ``generation_analogy_transfer`` each minted ``cand_1`` and
    ``cand_2`` for unrelated claims. Downstream, three of eighteen matches were a
    candidate against itself, the ranking showed six rows where generate and
    reflect both showed eight, and the shortlist read ``cand_2, cand_2,
    cand_evidence_1, cand_mechanism_1`` under a summary saying four candidates.
    """
    flow = _flow()
    merged = flow._merged_generation_population(
        [
            _Result(_numbered_from_one("analogy_transfer", 2)),
            _Result(_numbered_from_one("competing_explanation", 2)),
        ]
    )
    population = CandidatePopulation.model_validate(merged.payload)

    identifiers = [candidate.id for candidate in population.candidates]
    assert len(identifiers) == 4
    assert len(set(identifiers)) == 4
    # The strategy that got there first keeps the plain id, so an ordinary run
    # still reads cand_1; the collision is namespaced to whoever wrote it.
    assert sorted(identifiers) == [
        "cand_1",
        "cand_1_competing_explanation",
        "cand_2",
        "cand_2_competing_explanation",
    ]
    # And renaming moved the id only -- every claim still reaches the tournament.
    assert sorted(candidate.claim for candidate in population.candidates) == [
        "analogy_transfer claim 1",
        "analogy_transfer claim 2",
        "competing_explanation claim 1",
        "competing_explanation claim 2",
    ]


def test_a_population_that_reuses_an_id_is_refused_rather_than_ranked():
    """The backstop under the merge: one id, one candidate, or the stage fails."""
    population = CandidatePopulation.model_validate(_numbered_from_one("s0", 2).payload)
    population.candidates.append(population.candidates[0].model_copy(deep=True))

    with pytest.raises(NormalizationError) as excinfo:
        validate_candidate_distinctness(population)
    assert "cand_1" in str(excinfo.value)


def test_a_claim_two_strategies_both_reached_is_carried_once():
    flow = _flow()
    duplicate = _population_artifact("s0", 2)
    merged = flow._merged_generation_population(
        [_Result(duplicate), _Result(_population_artifact("s0", 2))]
    )
    population = CandidatePopulation.model_validate(merged.payload)
    assert len(population.candidates) == 2
    assert "from 1 generation strategy" in merged.content


def test_the_field_the_run_ranks_carries_the_axes_it_is_ranked_on():
    """The merge folded four strategy populations down to their candidates alone.

    So the field reached the tournament with no criteria on it: the judge had none
    to read, the cover printed none, and a live report and a live gate card both
    said "No cross-candidate criterion was recorded" of a run whose criteria are
    fixed.
    """
    from coscientist.parity import COMPARISON_CRITERIA, DIVERSITY_DIMENSIONS

    population = CandidatePopulation.model_validate(_merged(_flow(), 2).payload)

    assert population.comparison_criteria == list(COMPARISON_CRITERIA)
    assert population.diversity_dimensions == list(DIVERSITY_DIMENSIONS)


def test_the_axes_are_the_runs_own_and_not_the_proposing_specialists():
    """The cover says the set was settled before the ideas were written, which it
    cannot have been if a specialist proposing an idea chose what it is judged on."""
    from coscientist.models import Artifact
    from coscientist.parity import COMPARISON_CRITERIA

    written = _population_artifact("s0", 2)
    payload = CandidatePopulation.model_validate(written.payload)
    payload.comparison_criteria = ["whether the idea is mine"]
    self_serving = Artifact(
        stage="generate",
        agent=written.agent,
        content="",
        schema_name="CandidatePopulation",
        payload=payload.model_dump(mode="json"),
    )

    merged = CandidatePopulation.model_validate(
        _flow()._merged_generation_population([_Result(self_serving)]).payload
    )

    assert merged.comparison_criteria == list(COMPARISON_CRITERIA)
