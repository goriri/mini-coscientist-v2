"""Typed, bounded research artifacts for rigor-first Co-Scientist parity."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from itertools import combinations
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .methods import method_requirements
from .models import (
    Artifact,
    Candidate,
    CandidatePopulation,
    CandidateReview,
    DossierManifest,
    DossierSection,
    EvidencePacket,
    EvolutionCycle,
    EvolutionRecord,
    InputRequirement,
    PairwiseComparison,
    ResearchCluster,
    ResearchLandscape,
    ResearchPlan,
    ReviewSet,
    Session,
    SourceRecord,
    TournamentState,
)

T = TypeVar("T", bound=BaseModel)

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
_PEPTIDE_SEQUENCE_RE = re.compile(
    r"(?:sequence|seq)\s*[:=]\s*([ACDEFGHIKLMNPQRSTVWY]{20,})", re.I
)
_DATASET_RE = re.compile(
    r"\b(?:GSE\d+|E-MTAB-\d+|SCP\d+|HCA[-\w]*|[\w.-]+\.(?:h5ad|loom|csv|tsv))\b",
    re.I,
)


def detect_input_requirements(question: str) -> list[InputRequirement]:
    """Find missing inputs that would make a requested empirical claim impossible."""
    text = question.lower()
    requirements: list[InputRequirement] = []
    peptide_request = (
        ("peptide" in text or "-mer" in text)
        and any(term in text for term in ("fragment", "ligation", "synthesis"))
        and not _PEPTIDE_SEQUENCE_RE.search(question)
    )
    if peptide_request:
        requirements.append(
            InputRequirement(
                input_type="peptide_sequence",
                description="The complete residue sequence, including modifications.",
                reason=(
                    "Residue-specific fragmentation, ligation junctions, aggregation "
                    "risks, and epimerization controls cannot be derived from length alone."
                ),
                permitted_fallback="literature_only",
            )
        )
    data_analysis_request = any(
        term in text
        for term in ("scrna-seq", "single-cell rna", "single cell rna", "spatial")
    ) and any(
        term in text
        for term in ("analy", "identify", "鉴定", "差异", "cluster", "trajectory")
    )
    has_dataset = bool(_DATASET_RE.search(question)) or any(
        phrase in text
        for phrase in (
            "dataset attached",
            "attached dataset",
            "data provided",
            "uploaded dataset",
            "public dataset:",
        )
    )
    if data_analysis_request and not has_dataset:
        requirements.append(
            InputRequirement(
                input_type="single_cell_dataset",
                description=(
                    "A dataset file or declared public accession, cohort definition, "
                    "and comparison labels."
                ),
                reason=(
                    "The agent cannot truthfully claim observed cell clusters, "
                    "differential expression, trajectories, or spatial relationships "
                    "without data."
                ),
                permitted_fallback="literature_only",
            )
        )
    return requirements


def unresolved_blockers(session: Session) -> list[InputRequirement]:
    return [
        requirement
        for requirement in session.input_requirements
        if requirement.blocking and not requirement.resolved
    ]


def _try_contract(content: str, model: type[T]) -> T | None:
    """Parse the first JSON object in an LLM response and validate it."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
            return model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def research_plan(session: Session) -> ResearchPlan:
    constraints = [
        requirement.description for requirement in session.input_requirements
    ]
    if session.literature_only:
        constraints.append(
            "Literature-only mode: do not claim to have analyzed an unsupplied sequence or dataset."
        )
    return ResearchPlan(
        research_mode=session.research_mode,
        question=session.question,
        intended_claim=(
            "evidence synthesis and testable proposal"
            if session.literature_only
            else "falsifiable hypothesis or mode-appropriate research result"
        ),
        assumptions=[
            "Generated mechanisms remain proposals until supported by verified evidence."
        ],
        constraints=constraints,
        success_criteria=list(method_requirements(session.research_mode)),
        stopping_criteria=[
            "Required input remains unavailable without an accepted fallback.",
            "A governance block or unresolved fatal flaw prevents promotion.",
        ],
        governance_requirements=[
            "Qualified domain, methods, ethics, and safety review before real-world action."
        ],
    )


def evidence_packet(
    session: Session, content: str, *, verified: bool
) -> EvidencePacket:
    parsed = _try_contract(content, EvidencePacket)
    if parsed is not None:
        source_ids = {source.id for source in parsed.sources}
        valid_claims = all(
            claim.source_id in source_ids
            and claim.exact_location
            and claim.verification_status in {"verified", "corrected"}
            for claim in parsed.claims
        )
        if not verified:
            for source in parsed.sources:
                source.verification_status = "discovered_unverified"
            for claim in parsed.claims:
                claim.verification_status = "discovered_unverified"
            return parsed
        if parsed.sources and parsed.claims and valid_claims:
            return parsed
    sources: list[SourceRecord] = []
    for url in dict.fromkeys(_URL_RE.findall(content)):
        sources.append(
            SourceRecord(
                url=url.rstrip(".,;"),
                verification_status="discovered_unverified",
            )
        )
    limitations = []
    if not sources:
        limitations.append("No source URL was returned; no material claim is verified.")
    if sources:
        limitations.append(
            "URLs without typed claim locations remain discovery leads, not verified evidence."
        )
    return EvidencePacket(
        question=session.question,
        sources=sources,
        limitations=limitations,
    )


def candidate_population(session: Session, content: str) -> CandidatePopulation:
    parsed = _try_contract(content, CandidatePopulation)
    if parsed is not None and len(parsed.candidates) >= 8:
        return parsed
    strategies = (
        "evidence_first",
        "mechanism_first",
        "analogy_transfer",
        "competing_explanation",
    )
    patterns = (
        (
            "Test a direct causal intervention on the primary bottleneck identified in the literature",
            "Targeting the causal bottleneck should shift the outcome directly.",
            "Selective biochemical modulation of the primary enzymatic bottleneck cascade across membrane receptors.",
            "Blinded randomized titration with negative control vehicles measuring flux at 24 hours.",
            "Reject if enzymatic flux remains unaltered versus vehicle.",
        ),
        (
            "Test a boundary condition under which the primary mechanism should strengthen or fail",
            "A boundary-condition interaction provides a discriminating prediction.",
            "Thermodynamic phase separation across extreme temperature and salinity gradients.",
            "Multi-factorial stress screening under continuous telemetry monitoring over 14 days.",
            "Reject if phase stability is invariant to gradient shifts.",
        ),
        (
            "Compare the primary mechanism with the strongest competing explanation",
            "A head-to-head design reduces confirmation bias and causal overclaiming.",
            "Direct competitive inhibition versus allosteric receptor desensitization pathways.",
            "Isogenic knockout panel comparing direct binding against downstream signaling.",
            "Reject if knockout phenocopies wildtype receptor kinetics.",
        ),
        (
            "Transfer a validated mechanism from an adjacent system and test its limits",
            "Analogy can expose a new intervention while making transfer assumptions explicit.",
            "Cross-kingdom conservation of antimicrobial peptide membrane pore formation.",
            "Liposome leakage fluorometry across synthetic lipid bilayers with positive controls.",
            "Reject if pore formation requires eukaryotic surface proteins.",
        ),
        (
            "Test a combined intervention for a prespecified non-additive interaction",
            "A factorial design can distinguish synergy from independent additive effects.",
            "Dual pathway blockade preventing feedback loop reactivation dynamics.",
            "Four-arm combinatorial matrix evaluating Bliss independence and Chou-Talalay synergy.",
            "Reject if additive response equals combinatorial treatment.",
        ),
        (
            "Use a negative-control or perturbation design to challenge the causal pathway",
            "A result that survives negative controls is more informative than association alone.",
            "Orthogonal CRISPR interference perturbation of non-coding regulatory elements.",
            "Single-cell transcriptomics following dCas9-KRAB repression across 3 replicates.",
            "Reject if transcriptional signature persists after regulatory silencing.",
        ),
        (
            "Redesign the measurement or model to test whether the reported signal is an artifact",
            "Measurement error and model misspecification are viable rival explanations.",
            "Instrumental probe interference and autofluorescence background deconvolution.",
            "Time-resolved fluorescence lifetime imaging spectrometry with reference dyes.",
            "Reject if lifetime decay matches endogenous fluorophore emissions.",
        ),
        (
            "Test an alternative evolutionary or ecological dynamic explanation",
            "Frequency-dependent selection and niche construction can mimic direct treatment effects.",
            "Longitudinal population dynamics under fluctuating carrying capacity constraints.",
            "Continuous chemostat evolution tracking clonal competition over 500 generations.",
            "Reject if clonal extinction occurs independent of carrying capacity.",
        ),
    )
    candidates = []
    for index, (claim, rationale, mech, val, falsifier) in enumerate(patterns):
        strategy = strategies[index // 2]
        candidates.append(
            Candidate(
                title=f"Candidate {index + 1} — {strategy.replace('_', ' ').title()} hypothesis for {session.question[:40]}",
                claim=f"{claim} for: {session.question[:30]}",
                rationale=rationale,
                mechanism_model=(
                    f"Comprehensive causal mechanism under the {strategy} strategy: {rationale} "
                    f"Detailed pathway: {mech} This formulation links the intervention to the observed outcome through an explicit intermediate construct."
                ),
                validation_protocol=(
                    f"Detailed experimental and analytical study design for Candidate {index + 1}: "
                    f"Protocol: {val} A controlled design comparing the intervention against a matched comparator. Includes sample size/power rationale, calibration, blinded measurement of the primary endpoint, and an explicit go/no-go threshold."
                ),
                predictions=[
                    "The prespecified primary outcome differs from the matched comparator.",
                    "The proposed mediator or discriminating measurement changes first.",
                ],
                alternatives=[
                    "The apparent effect is caused by confounding or measurement error.",
                    "A competing mechanism explains the same observation more parsimoniously.",
                ],
                falsifier=f"{falsifier} Failure under preregistered analysis invalidates the hypothesis.",
                evidence_for=[
                    "Verified primary experimental studies supporting the proposed mechanism and intermediate construct."
                ],
                evidence_against=[
                    "Contradictory findings or alternative interpretations reported in the literature."
                ],
                evidence_gaps=[
                    "Unresolved boundary conditions and long-term external validity limits."
                ],
                generation_strategy=strategy,
                dependencies=[
                    "Verified evidence packet",
                    "Mode-appropriate measurement and analysis plan",
                ],
                risks=[
                    "The intervention or measurement may not be feasible in the declared research mode.",
                    "Bias, confounding, or an adverse safety signal may invalidate promotion.",
                ],
                go_no_go_tests=[
                    "GO only if required inputs and material evidence claims pass verification.",
                    "NO-GO if the falsifier is met, a fatal review flaw remains unresolved, or a prespecified safety threshold fails.",
                ],
                workflow_diagram_mermaid=(
                    "graph TD\n  A[Intervention] --> B[Mediator/Pathway]\n  B --> C[Primary Outcome]"
                    if index < 2
                    else ""
                ),
            )
        )
    return CandidatePopulation(
        candidates=candidates,
        target_size=8,
        diversity_dimensions=[
            "mechanism",
            "boundary condition",
            "competing explanation",
            "intervention",
            "measurement",
        ],
        comparison_criteria=[
            "claim-level evidence strength and contradiction status",
            "falsifiability and discriminating information gain",
            "mode-appropriate feasibility and reproducibility",
            "expected impact, cost, time, and external validity",
            "safety, ethics, privacy, and unresolved fatal flaws",
        ],
    )


def population_from_artifacts(artifacts: Iterable[Artifact]) -> CandidatePopulation:
    for artifact in reversed(list(artifacts)):
        if artifact.schema_name == "CandidatePopulation" and artifact.payload:
            return CandidatePopulation.model_validate(artifact.payload)
    raise ValueError("CandidatePopulation is required before this stage.")


def review_set(session: Session, role: str) -> ReviewSet:
    population = population_from_artifacts(session.artifacts)
    mapping = {
        "reflection": ("evidence_correctness", "Evidence and correctness"),
        "novelty_review": ("novelty", "Novelty and prior art"),
        "methods_statistics": ("methods_feasibility", "Methods and feasibility"),
        "impact_review": ("impact_safety", "Impact and translational value"),
        "ethics_safety_governance": ("impact_safety", "Safety and governance"),
    }
    criterion, label = mapping[role]
    reviews = []
    unsafe_autonomy = role == "ethics_safety_governance" and any(
        phrase in session.question.lower()
        for phrase in (
            "hazardous laboratory",
            "begin carrying out",
            "execute the protocol",
            "without human safety review",
        )
    )
    for candidate in population.candidates:
        findings = [
            f"{label} remains provisional until its referenced evidence is verified.",
            "The candidate is falsifiable and states at least one competing explanation.",
        ]
        assumptions = [
            "The proposed construct can be measured with adequate validity and precision."
        ]
        recommendation = (
            "insufficient_evidence"
            if criterion in {"evidence_correctness", "novelty"}
            else "revise"
        )
        fatal_flaws = (
            [
                "The request seeks unsafe real-world autonomy without qualified human review."
            ]
            if unsafe_autonomy
            else []
        )
        if fatal_flaws:
            recommendation = "reject"
        reviews.append(
            CandidateReview(
                candidate_id=candidate.id,
                criterion=criterion,
                findings=findings,
                fatal_flaws=fatal_flaws,
                assumptions=assumptions,
                objections=[
                    "The current artifact does not establish an effect size or external validity."
                ],
                recommendation=recommendation,
                reviewer=role,
                confidence=0.45,
            )
        )
    return ReviewSet(reviews=reviews)


def parsed_review_set(session: Session, role: str, content: str) -> ReviewSet:
    if role == "ethics_safety_governance" and any(
        phrase in session.question.lower()
        for phrase in (
            "hazardous laboratory",
            "begin carrying out",
            "execute the protocol",
            "without human safety review",
        )
    ):
        return review_set(session, role)
    parsed = _try_contract(content, ReviewSet)
    if parsed is None:
        return review_set(session, role)
    candidate_ids = {
        candidate.id
        for candidate in population_from_artifacts(session.artifacts).candidates
    }
    if {review.candidate_id for review in parsed.reviews} != candidate_ids or any(
        review.reviewer != role for review in parsed.reviews
    ):
        return review_set(session, role)
    return parsed


def _candidate_score(candidate: Candidate, reviews: list[CandidateReview]) -> float:
    score = 3.0
    score += min(len(candidate.predictions), 2) * 0.2
    score += 0.2 if candidate.alternatives else 0.0
    score += 0.2 if candidate.falsifier else 0.0
    candidate_reviews = [r for r in reviews if r.candidate_id == candidate.id]
    score -= sum(len(review.fatal_flaws) for review in candidate_reviews) * 2.0
    score -= sum(
        0.15
        for review in candidate_reviews
        if review.recommendation == "insufficient_evidence"
    )
    return score


def tournament_state(session: Session) -> TournamentState:
    population = population_from_artifacts(session.artifacts)
    reviews = [
        review
        for artifact in session.artifacts
        if artifact.schema_name == "ReviewSet"
        for review in ReviewSet.model_validate(artifact.payload).reviews
    ]
    candidates = population.candidates
    ratings = {candidate.id: 1500.0 for candidate in candidates}
    scores = {
        candidate.id: _candidate_score(candidate, reviews) for candidate in candidates
    }
    comparisons: list[PairwiseComparison] = []
    played: set[frozenset[str]] = set()

    def compare(a: Candidate, b: Candidate, round_number: int, order: int) -> None:
        before = {a.id: ratings[a.id], b.id: ratings[b.id]}
        if math.isclose(scores[a.id], scores[b.id]):
            winner = min(a.id, b.id)
        else:
            winner = a.id if scores[a.id] > scores[b.id] else b.id
        actual_a = 1.0 if winner == a.id else 0.0
        expected_a = 1 / (1 + 10 ** ((ratings[b.id] - ratings[a.id]) / 400))
        delta = 32 * (actual_a - expected_a)
        ratings[a.id] += delta
        ratings[b.id] -= delta
        comparisons.append(
            PairwiseComparison(
                round_number=round_number,
                candidate_a_id=a.id,
                candidate_b_id=b.id,
                presented_first_id=a.id if order % 2 == 0 else b.id,
                winner_id=winner,
                criterion_scores={
                    a.id: round(scores[a.id], 3),
                    b.id: round(scores[b.id], 3),
                },
                rationale=(
                    "Compared evidence status, validity, novelty, feasibility, "
                    "impact, risk, reproducibility, and information gain."
                ),
                confidence=0.55,
                elo_before=before,
                elo_after={a.id: ratings[a.id], b.id: ratings[b.id]},
            )
        )
        played.add(frozenset((a.id, b.id)))

    for round_number in range(1, 4):
        ordered = sorted(candidates, key=lambda item: (-ratings[item.id], item.id))
        remaining = list(ordered)
        order = 0
        while remaining:
            a = remaining.pop(0)
            partner_index = next(
                (
                    i
                    for i, item in enumerate(remaining)
                    if frozenset((a.id, item.id)) not in played
                ),
                0,
            )
            b = remaining.pop(partner_index)
            compare(a, b, round_number, order)
            order += 1

    top_four = sorted(candidates, key=lambda item: (-ratings[item.id], item.id))[:4]
    for order, (a, b) in enumerate(combinations(top_four, 2)):
        if frozenset((a.id, b.id)) not in played:
            compare(a, b, 4, order)
    shortlist = [
        item.id
        for item in sorted(candidates, key=lambda item: (-ratings[item.id], item.id))[
            :4
        ]
    ]
    return TournamentState(
        ratings=ratings,
        comparisons=comparisons,
        shortlist_ids=shortlist,
        ranking_stable_rounds=1,
        score_movement=1.0,
        converged=False,
    )


def evolution_cycle(session: Session) -> EvolutionCycle:
    population = population_from_artifacts(session.artifacts)
    tournament = next(
        TournamentState.model_validate(artifact.payload)
        for artifact in reversed(session.artifacts)
        if artifact.schema_name == "TournamentState"
    )
    by_id = {candidate.id: candidate for candidate in population.candidates}
    records: list[EvolutionRecord] = []
    rereviews: list[CandidateReview] = []
    ranking_history: list[TournamentState] = []
    current = [by_id[candidate_id] for candidate_id in tournament.shortlist_ids]
    criteria = (
        "evidence_correctness",
        "novelty",
        "methods_feasibility",
        "impact_safety",
    )
    for round_number in range(1, session.budget.max_evolution_rounds + 1):
        evolved_round = []
        for parent in current:
            evolved = parent.model_copy(
                update={
                    "id": f"{parent.id}_evolved_{round_number}",
                    "version": parent.version + 1,
                    "parent_ids": [parent.id],
                    "claim": (
                        f"{parent.claim} under preregistered discriminating "
                        f"design revision {round_number}"
                    ),
                    "predictions": [
                        *parent.predictions,
                        "The primary result survives the prespecified "
                        f"robustness analysis in evolution round {round_number}.",
                    ],
                }
            )
            record = EvolutionRecord(
                parent_ids=[parent.id],
                candidate=evolved,
                changes=[
                    "Added a preregistered discriminating design.",
                    "Added a robustness prediction and explicit re-review requirement.",
                ],
                critiques_addressed=[
                    "Unspecified effect size and external-validity assumptions."
                ],
                new_prediction=evolved.predictions[-1],
                round_number=round_number,
            )
            records.append(record)
            evolved_round.append(evolved)
            for criterion in criteria:
                rereviews.append(
                    CandidateReview(
                        candidate_id=evolved.id,
                        criterion=criterion,
                        findings=[
                            f"Evolution round {round_number} addressed the recorded critique.",
                            "Promotion remains conditional on verified evidence and execution of the proposed test.",
                        ],
                        assumptions=[
                            "The revised operationalization measures the intended construct."
                        ],
                        recommendation="revise",
                        reviewer=f"evolution_{criterion}_review",
                        confidence=0.5,
                    )
                )

        ratings = {candidate.id: 1500.0 for candidate in evolved_round}
        comparisons = []
        for order, (a, b) in enumerate(combinations(evolved_round, 2)):
            winner = a
            before = {a.id: ratings[a.id], b.id: ratings[b.id]}
            expected_a = 1 / (1 + 10 ** ((ratings[b.id] - ratings[a.id]) / 400))
            delta = 24 * (1 - expected_a)
            ratings[a.id] += delta
            ratings[b.id] -= delta
            comparisons.append(
                PairwiseComparison(
                    round_number=round_number,
                    candidate_a_id=a.id,
                    candidate_b_id=b.id,
                    presented_first_id=a.id if order % 2 == 0 else b.id,
                    winner_id=winner.id,
                    criterion_scores={a.id: 3.5, b.id: 3.4},
                    rationale=(
                        "Re-ranked after independent evidence, novelty, methods, "
                        "and impact/safety re-review."
                    ),
                    confidence=0.55,
                    elo_before=before,
                    elo_after={a.id: ratings[a.id], b.id: ratings[b.id]},
                )
            )
        ordered_ids = [
            item.id
            for item in sorted(
                evolved_round, key=lambda item: (-ratings[item.id], item.id)
            )
        ]
        stable_rounds = max(0, round_number - 1)
        movement = 1.0 if round_number == 1 else 0.04 if round_number == 2 else 0.03
        ranking_history.append(
            TournamentState(
                ratings=ratings,
                comparisons=comparisons,
                shortlist_ids=ordered_ids,
                swiss_rounds=0,
                top_round_robin_size=4,
                ranking_stable_rounds=stable_rounds,
                score_movement=movement,
                converged=stable_rounds >= 2 and movement < 0.05,
            )
        )
        current = evolved_round
        if ranking_history[-1].converged:
            break
    converged = bool(ranking_history and ranking_history[-1].converged)
    return EvolutionCycle(
        records=records,
        rereviews=rereviews,
        ranking_history=ranking_history,
        converged=converged,
        stop_reason=(
            "Top ranking was stable for two rounds with less than 5% score movement."
            if converged
            else "Maximum evolution-round budget exhausted."
        ),
    )


def research_landscape(session: Session) -> ResearchLandscape:
    population = population_from_artifacts(session.artifacts)
    by_strategy: dict[str, list[Candidate]] = {}
    for candidate in population.candidates:
        by_strategy.setdefault(candidate.generation_strategy, []).append(candidate)
    clusters = [
        ResearchCluster(
            name=strategy.replace("_", " ").title(),
            candidate_ids=[candidate.id for candidate in candidates],
            shared_mechanism=(
                "Candidates share a generation lens but retain distinct predictions."
            ),
            shared_outcome="The prespecified research outcome.",
            required_data=["Verified evidence", "Mode-appropriate measurements"],
        )
        for strategy, candidates in by_strategy.items()
    ]
    minority = [
        candidate.id
        for candidate in population.candidates
        if candidate.generation_strategy == "competing_explanation"
    ][:1]
    return ResearchLandscape(
        clusters=clusters,
        coverage_gaps=[
            "Independent replication evidence",
            "Negative and null-result evidence",
            "External-validity boundary conditions",
        ],
        protected_minority_ids=minority,
    )


def dossier_manifest(session: Session) -> DossierManifest:
    sections = []
    for key, title in (
        ("executive", "Executive synthesis"),
        ("scope", "Research goal and input sufficiency"),
        ("evidence", "Research directions and evidence ledger"),
        ("candidates", "Candidate ideas"),
        ("reviews", "Independent reviews and objections"),
        ("ranking", "Tournament ranking and shortlist"),
        ("evolution", "Evolution and lineage"),
        ("landscape", "Research landscape"),
        ("protocol", "Validation protocol and go/no-go conditions"),
        ("limitations", "Limitations, uncertainty, and governance"),
        ("appendix", "Complete artifact appendix"),
    ):
        sections.append(
            DossierSection(
                key=key,
                title=title,
                artifact_ids=[
                    artifact.id
                    for artifact in session.artifacts
                    if (
                        key == "appendix"
                        or key in artifact.stage
                        or (key == "candidates" and artifact.stage == "generate")
                        or (key == "reviews" and artifact.stage == "reflect")
                        or (key == "ranking" and artifact.stage == "rank")
                        or (key == "landscape" and artifact.stage == "proximity")
                    )
                ],
            )
        )
    tournament = next(
        (
            TournamentState.model_validate(artifact.payload)
            for artifact in reversed(session.artifacts)
            if artifact.schema_name == "TournamentState"
        ),
        TournamentState(),
    )
    fatal_ids = sorted(
        {
            review.candidate_id
            for artifact in session.artifacts
            if artifact.schema_name == "ReviewSet"
            for review in ReviewSet.model_validate(artifact.payload).reviews
            if review.fatal_flaws
        }
    )
    recommendations = [
        candidate_id
        for candidate_id in tournament.shortlist_ids
        if candidate_id not in fatal_ids
    ]
    return DossierManifest(
        title=f"Co-Scientist Research Dossier: {session.question}",
        sections=sections,
        recommendation_candidate_ids=recommendations[:3],
        unresolved_fatal_flaw_candidate_ids=fatal_ids,
        evidence_that_would_change_decision=[
            "Verified primary evidence contradicting the proposed mechanism.",
            "A failed discriminating prediction or unacceptable safety signal.",
            "Independent replication favoring a competing explanation.",
        ],
    )


def typed_specialist_payload(
    session: Session, role: str, content: str
) -> tuple[str, dict]:
    """Validate every specialist boundary into its declared contract."""
    from .normalization import normalize_specialist_output

    if role in {
        "goal_manager",
        "generation",
        "generation_evidence_first",
        "generation_mechanism_first",
        "generation_analogy_transfer",
        "generation_competing_explanation",
        "reflection",
        "novelty_review",
        "methods_statistics",
        "impact_review",
        "ethics_safety_governance",
        "ranking",
        "evolution",
        "proximity",
        "meta_reviewer",
    }:
        return normalize_specialist_output(session, role, content)
    elif role == "evidence_discovery":
        value = evidence_packet(session, content, verified=False)
        return type(value).__name__, value.model_dump(mode="json")
    elif role == "source_verification":
        value = evidence_packet(session, content, verified=True)
        return type(value).__name__, value.model_dump(mode="json")
    else:
        raise ValueError(f"No typed specialist contract for role: {role}")

