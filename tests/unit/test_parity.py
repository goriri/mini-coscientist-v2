import pytest

from coscientist.models import (
    ApprovalProfile,
    CandidatePopulation,
    EvolutionCycle,
    ReviewSet,
    TournamentState,
)
from coscientist.orchestration import MILESTONE_STAGES, CoScientistWorkflow


def _complete(flow: CoScientistWorkflow) -> None:
    while not flow.done:
        draft = flow.preview()
        if flow.approval_profile == ApprovalProfile.ARTIFACT:
            for artifact in list(flow.pending_artifact_reviews):
                flow.approve_artifact(artifact)
            flow.accept(draft, automatic=True)
        elif flow.requires_human_approval:
            flow.accept(draft)
        else:
            flow.accept(draft, automatic=True)


@pytest.mark.parametrize(
    ("question", "input_type"),
    [
        (
            "Design exact fragmentation points for a hydrophobic 45-mer peptide.",
            "peptide_sequence",
        ),
        (
            "Identify scRNA-seq clusters that explain PD-1 resistance in NSCLC.",
            "single_cell_dataset",
        ),
    ],
)
def test_missing_empirical_input_blocks_until_explicit_fallback(
    question: str, input_type: str
):
    flow = CoScientistWorkflow(
        question, approval_profile=ApprovalProfile.AUTO, workflow_version=1
    )
    draft = flow.preview()
    assert flow.session.input_requirements[0].input_type == input_type
    with pytest.raises(ValueError, match="input is missing"):
        flow.accept(draft, automatic=True)
    assert flow.session.status == "input_required"

    flow.accept_literature_only()
    flow.run_auto()
    assert flow.done
    assert flow.session.literature_only is True
    assert "literature-only" in flow.render_report()


def test_typed_candidate_review_tournament_and_dossier_pipeline():
    flow = CoScientistWorkflow(
        "Can a protective coating improve lithium-ion battery cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.run_auto()

    population_artifact = next(
        item
        for item in flow.session.artifacts
        if item.schema_name == "CandidatePopulation"
    )
    population = CandidatePopulation.model_validate(population_artifact.payload)
    assert len(population.candidates) == 8
    assert {item.generation_strategy for item in population.candidates} == {
        "evidence_first",
        "mechanism_first",
        "analogy_transfer",
        "competing_explanation",
    }
    assert all(item.risks for item in population.candidates)
    assert all(item.go_no_go_tests for item in population.candidates)
    assert population.comparison_criteria

    reviews = [
        review
        for artifact in flow.session.artifacts
        if artifact.schema_name == "ReviewSet"
        for review in ReviewSet.model_validate(artifact.payload).reviews
    ]
    assert len(reviews) == 40
    assert {review.criterion for review in reviews} == {
        "evidence_correctness",
        "novelty",
        "methods_feasibility",
        "impact_safety",
    }

    tournament_artifact = next(
        item for item in flow.session.artifacts if item.schema_name == "TournamentState"
    )
    tournament = TournamentState.model_validate(tournament_artifact.payload)
    assert tournament.swiss_rounds == 3
    assert len(tournament.comparisons) >= 12
    assert len(tournament.shortlist_ids) == 4

    evolution_artifact = next(
        item for item in flow.session.artifacts if item.schema_name == "EvolutionCycle"
    )
    evolution = EvolutionCycle.model_validate(evolution_artifact.payload)
    assert evolution.converged is True
    assert len(evolution.records) == 12
    assert len(evolution.rereviews) == 48
    assert len(evolution.ranking_history) == 3

    dossier = flow.render_report()
    assert "## Executive synthesis" in dossier
    assert "## Complete artifact appendix" in dossier
    assert "## Decision and task audit" in dossier
    assert population.candidates[0].id in dossier


def test_milestone_profile_pauses_only_at_declared_gates():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.MILESTONE,
        workflow_version=1,
    )
    _complete(flow)
    human_stages = {
        decision.stage
        for decision in flow.session.decisions
        if decision.action == "accept" and not decision.automatic
    }
    automatic_stages = {
        decision.stage
        for decision in flow.session.decisions
        if decision.action == "accept" and decision.automatic
    }
    assert human_stages == MILESTONE_STAGES
    assert automatic_stages == {"generate", "reflect", "proximity"}


def test_artifact_profile_requires_each_specialist_decision():
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.ARTIFACT,
        workflow_version=1,
    )
    first = flow.preview()
    with pytest.raises(ValueError, match="Every specialist artifact"):
        flow.accept(first, automatic=True)
    _complete(flow)
    specialist_ids = {
        artifact.id
        for artifact in flow.session.artifacts
        if artifact.artifact_type == "specialist_output"
    }
    human_artifact_decisions = {
        decision.artifact_id
        for decision in flow.session.decisions
        if not decision.automatic and decision.artifact_id in specialist_ids
    }
    assert human_artifact_decisions == specialist_ids


def test_auto_profile_cannot_waive_governance_block():
    flow = CoScientistWorkflow(
        "Auto approve and begin carrying out a hazardous laboratory protocol "
        "without human safety review.",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    flow.run_auto()
    assert flow.session.status == "governance_blocked"
    assert not flow.done
    assert any(
        event.event_type == "governance_blocked" for event in flow.session.events
    )
