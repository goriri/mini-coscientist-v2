"""Unit tests for Phase 1 Evidence and Normalization Gates."""

import pytest
from pydantic import ValidationError

from coscientist.models import (
    ApprovalProfile,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    DossierManifest,
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
    with pytest.raises(
        ValueError, match="Generation requires completed discovery and claim-level source verification"
    ):
        flow.accept(evidence_draft)

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
    assert "### Executive Candidate Summary" in report_md
    assert "| # | Candidate Title | Strategy | Primary Claim | Falsifier Summary |" in report_md
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
        art for art in reversed(flow.session.artifacts)
        if art.stage == "generate" and art.schema_name == "CandidatePopulation"
    )
    pop = CandidatePopulation.model_validate(pop_art.payload)
    target_id = pop.candidates[0].id
    old_claim = pop.candidates[0].claim
    old_protocol = pop.candidates[0].validation_protocol

    refined_draft = flow.refine_section(
        target_id,
        "validation_protocol",
        "Add a positive control arm with known active enzyme.",
    )
    pop_after = CandidatePopulation.model_validate(refined_draft.payload)
    assert pop_after.candidates[0].id == target_id
    assert pop_after.candidates[0].claim == old_claim
    assert "Add a positive control arm" in pop_after.candidates[0].validation_protocol
    assert flow.session.decisions[-1].feedback.startswith("Refined section 'validation_protocol'")


