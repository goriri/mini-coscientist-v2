import pytest

from coscientist.agents import STRUCTURED_OUTPUT_INSTRUCTIONS
from coscientist.models import (
    ApprovalProfile,
    CandidatePopulation,
    EvolutionCycle,
    ResearchCluster,
    ResearchLandscape,
    ReviewSet,
    TournamentState,
)
from coscientist.narrative import derive_idea_title
from coscientist.orchestration import MILESTONE_STAGES, CoScientistWorkflow
from coscientist.parity import (
    REVIEW_CRITERIA,
    dossier_manifest,
    parsed_research_landscape,
    population_from_artifacts,
    research_landscape,
)


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
    # Five reviewers, five distinct axes. Safety and governance decide whether
    # work may proceed; impact decides whether it is worth proceeding. Folding
    # them together once made an ethics review indistinguishable from a
    # commercial-viability review in the dossier.
    assert {review.criterion for review in reviews} == {
        "evidence_correctness",
        "novelty",
        "methods_feasibility",
        "impact_safety",
        "safety_governance",
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
    # Every offspring is re-reviewed on every axis the parents were judged on,
    # so this count follows from the criteria rather than standing alone.
    assert len(evolution.rereviews) == len(evolution.records) * len(REVIEW_CRITERIA)
    assert len(evolution.ranking_history) == 3

    dossier = flow.render_report()
    # The report leads with a synthesis rather than with artifacts: the numbered
    # narrative has to open the document, before any per-idea deep dive.
    assert "# Research Overview" in dossier
    assert "#### 1. Research Goal" in dossier
    assert dossier.index("#### 1. Research Goal") < dossier.index("## Idea Proposal")
    # Every payload's origin stays auditable, so a template can never pass as
    # reasoning even though the raw JSON is no longer printed.
    assert "## What each stage produced" in dossier
    assert "## Evidence integrity" in dossier
    # Candidates are named, not identified: the opaque id is what the narrative
    # layer exists to replace, so the derived title is the thing to find.
    assert derive_idea_title(population.candidates[0].claim) in dossier
    assert population.candidates[0].id not in dossier


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


def test_a_thrice_evolved_claim_carries_one_revision_marker_not_three():
    # Each round copies its parent's claim forward, so a marker appended to
    # whatever the parent held accumulated: the round-three offspring read
    # "... at 1C discharge rates. under preregistered discriminating design
    # revision 1 under preregistered discriminating design revision 2 ..."
    # in the finished report. The same held for the robustness prediction,
    # which the offspring ended up making once per round it had lived through.
    flow = CoScientistWorkflow(
        "Can a coating improve cycle life?",
        approval_profile=ApprovalProfile.AUTO,
        workflow_version=1,
    )
    _complete(flow)
    evolution = EvolutionCycle.model_validate(
        next(
            item
            for item in flow.session.artifacts
            if item.schema_name == "EvolutionCycle"
        ).payload
    )
    last_round = [record for record in evolution.records if record.round_number == 3]
    assert last_round
    for record in last_round:
        claim = record.candidate.claim
        assert claim.count("preregistered discriminating design") == 1
        assert "(revision 3)." in claim
        assert ". under" not in claim
        robustness = [
            prediction
            for prediction in record.candidate.predictions
            if "robustness analysis" in prediction
        ]
        assert len(robustness) == 1
        assert "evolution round 3" in robustness[0]
    assert {change for record in last_round for change in record.changes} == {
        "Revised the preregistered discriminating design.",
        "Restated the robustness prediction against the revised design.",
    }


def test_the_decision_changing_evidence_is_quoted_from_the_run(
    rich_session,
):
    """ "A short list of specific evidence" has to name evidence about this question.

    The fallback returned three sentences -- verified evidence contradicting the
    mechanism, a failed prediction, an independent replication -- that would unseat
    any hypothesis whatever, and the report introduced them as specific and then
    told the reader that obtaining any one of them beats another round of
    generation. Nothing there is a measurement anyone could go and take.
    """
    manifest = dossier_manifest(rich_session)
    decisive = manifest.evidence_that_would_change_decision
    tournament = next(
        TournamentState.model_validate(artifact.payload)
        for artifact in reversed(rich_session.artifacts)
        if artifact.schema_name == "TournamentState"
    )
    target = next(
        iter(manifest.recommendation_candidate_ids + list(tournament.shortlist_ids))
    )
    leader = next(
        candidate
        for candidate in population_from_artifacts(rich_session.artifacts).candidates
        if candidate.id == target
    )
    joined = " ".join(decisive).lower()

    assert decisive
    assert "contradicting the proposed mechanism" not in joined
    assert leader.falsifier.rstrip(".").lower()[:60] in joined
    assert leader.predictions[0].rstrip(".").lower()[:60] in joined
    # Section nine ends by naming the go/no-go work as the immediate next step, so
    # listing it here as well reads as a second piece of work.
    assert leader.go_no_go_tests[0].rstrip(".").lower()[:40] not in joined


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


def test_a_paraphrased_id_in_the_landscape_is_resolved_before_it_reaches_a_reader(
    rich_session,
):
    """Every other stage aligns the ids a specialist echoes back; this one did not.

    Section eight of a live run ended: "One further entry was recorded here against
    ideas this report cannot name: the ids the clustering stage filed it under reach
    no idea in this run." The clustering was sound -- only the spelling was not.
    """
    session = rich_session
    ids = [
        candidate.id
        for candidate in population_from_artifacts(session.artifacts).candidates
    ]
    assert len(ids) >= 4

    parsed = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Chemical scavenging",
                # A hyphen for the real id's underscore, and a positional reference.
                candidate_ids=[ids[0].replace("_", "-"), "Candidate 2"],
                shared_mechanism="The coating scavenges HF at the interface.",
                shared_outcome="Capacity retention over five hundred cycles.",
            )
        ],
        duplicates=[[ids[0].upper(), "Candidate 2"]],
        protected_minority_ids=["Candidate 3"],
        coverage_gaps=["Independent replication"],
    )
    fallback = research_landscape(session)
    aligned = parsed_research_landscape(session, parsed, fallback)

    assert aligned is not fallback
    assert aligned.clusters[0].candidate_ids == [ids[0], ids[1]]
    assert aligned.duplicates == [[ids[0], ids[1]]]
    assert aligned.protected_minority_ids == [ids[2]]
    # The specialist's own reading of what is missing survives the alignment.
    assert aligned.coverage_gaps == ["Independent replication"]


def test_a_landscape_naming_no_idea_in_this_run_gives_way_to_the_template(
    rich_session,
):
    """A group of ideas the run does not hold says nothing about the ideas it does."""
    session = rich_session
    fallback = research_landscape(session)
    parsed = ResearchLandscape(
        clusters=[
            ResearchCluster(
                name="Borrowed from another run",
                candidate_ids=["hyp_alpha", "hyp_beta"],
                shared_mechanism="Unrelated.",
                shared_outcome="Unrelated.",
            )
        ],
        coverage_gaps=["Written about ideas this run does not hold"],
    )

    assert parsed_research_landscape(session, parsed, fallback) is fallback
    assert parsed_research_landscape(session, None, fallback) is fallback


def test_the_clustering_stage_is_asked_for_the_duplicates_its_contract_holds():
    """``duplicates`` was in the contract and in no prompt, so it came back empty.

    A live run shipped "Low-Temperature ALD Al2O3 HF-Scavenging and Phase
    Stabilization" and "ALD Al2O3 as HF Scavenger and Phase Stabilizer on NMC811" as
    two ideas tied on one Elo, with the overlap remarked on only inside a quoted
    reviewer's prose.
    """
    instruction = STRUCTURED_OUTPUT_INSTRUCTIONS["proximity"]
    assert "'duplicates'" in instruction
    assert "merged before either is funded" in instruction
    # And told apart from a cluster, which shares a mechanism rather than a claim.
    assert "Two candidates in one cluster are not duplicates by that alone" in (
        instruction
    )
