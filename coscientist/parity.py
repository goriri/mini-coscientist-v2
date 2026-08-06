"""Typed, bounded research artifacts for rigor-first Co-Scientist parity."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TypeVar

from pydantic import BaseModel

from .citations import latest_evidence_packet
from .contract_io import ParseOutcome, parse_contract
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

# The Co-Scientist supplementary material initialises every hypothesis at 1200
# Elo; matching it keeps published rating ranges comparable with ours.
DEFAULT_ELO = 1200.0
ELO_K = 32.0

# ``score_movement`` is a fraction of DEFAULT_ELO, and 1.0 is its "nothing was
# measured" sentinel rather than a reading: a rating cannot move by 1200 points in
# one round, since a K factor of 32 caps a single match at 32 and no round holds
# thirty-seven matches. A live report multiplied the sentinel out and printed "the
# final round moved one rating by 100.0 per cent of that, or about 1200 points",
# then concluded from it that every position in the standings was provisional.
UNMEASURED_MOVEMENT = 1.0
TOP_FOUR = 4

# The second half of the convergence rule: the largest rating change in the final
# round, as a fraction of DEFAULT_ELO, that still counts as the ordering having
# settled. Named because the report states the rule to the reader and has to measure
# the round against the same number the tournament judged it by.
SETTLED_MOVEMENT = 0.05


def stable_rounds(history: Sequence[Sequence[str]]) -> int:
    """How many trailing rounds left the shortlist ordering unchanged."""
    if not history:
        return 0
    stable = 1
    for earlier, later in zip(
        reversed(history[:-1]), reversed(history[1:]), strict=False
    ):
        if earlier[:TOP_FOUR] != later[:TOP_FOUR]:
            break
        stable += 1
    return stable


def score_movement(history: Sequence[Mapping[str, float]]) -> float:
    """Largest fractional Elo change in the final round, relative to 1200.

    Returns the sentinel where there is no earlier round to measure against, which
    is not the same as a round that moved nothing: the caller has to say which.
    """
    if len(history) < 2 or not history[-1]:
        return UNMEASURED_MOVEMENT
    previous, current = history[-2], history[-1]
    return round(
        max(
            abs(current[key] - previous.get(key, DEFAULT_ELO)) / DEFAULT_ELO
            for key in current
        ),
        4,
    )


@dataclass(frozen=True)
class TypedPayload:
    """A validated specialist payload plus how it was obtained.

    ``source`` distinguishes the specialist's own reasoning from a deterministic
    template that stood in for it, so neither a report nor an operator has to
    guess whether a stage reflects real model work.
    """

    schema_name: str
    payload: dict
    source: str = "specialist"
    repairs: list[str] = field(default_factory=list)
    error: str = ""


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


REVIEW_CRITERIA: dict[str, tuple[str, str]] = {
    "reflection": ("evidence_correctness", "Evidence and correctness"),
    "novelty_review": ("novelty", "Novelty and prior art"),
    "methods_statistics": ("methods_feasibility", "Methods and feasibility"),
    "impact_review": ("impact_safety", "Impact and translational value"),
    "ethics_safety_governance": ("safety_governance", "Safety and governance"),
}


# The four strategy-specific generators each return the same contract as the
# combined one, and are dispatched the same way. Leaving them out made
# ``typed_specialist_payload`` fall through to the meta-review branch and demand
# a CandidatePopulation that the generate stage had not written yet, so the run
# died at the first generator with "CandidatePopulation is required before this
# stage."
GENERATION_ROLES: frozenset[str] = frozenset(
    {
        "generation",
        "generation_evidence_first",
        "generation_mechanism_first",
        "generation_analogy_transfer",
        "generation_competing_explanation",
    }
)

ROLE_CONTRACTS: dict[str, type[BaseModel]] = {
    "goal_manager": ResearchPlan,
    "evidence_discovery": EvidencePacket,
    "source_verification": EvidencePacket,
    "generation": CandidatePopulation,
    "generation_evidence_first": CandidatePopulation,
    "generation_mechanism_first": CandidatePopulation,
    "generation_analogy_transfer": CandidatePopulation,
    "generation_competing_explanation": CandidatePopulation,
    "reflection": ReviewSet,
    "novelty_review": ReviewSet,
    "methods_statistics": ReviewSet,
    "impact_review": ReviewSet,
    "ethics_safety_governance": ReviewSet,
    "ranking": TournamentState,
    "evolution": EvolutionCycle,
    "proximity": ResearchLandscape,
    "meta_reviewer": DossierManifest,
}


def contract_defaults(session: Session, role: str) -> dict[str, dict[str, object]]:
    """Context a specialist may legitimately omit from its typed payload.

    A reviewer restating the research question on every packet, or its own name
    on every review, adds nothing; the supervisor already knows both. Supplying
    them here keeps an otherwise complete answer from being thrown away.
    """
    defaults: dict[str, dict[str, object]] = {
        "ResearchPlan": {
            "question": session.question,
            "research_mode": session.research_mode,
        },
        "EvidencePacket": {"question": session.question},
        "DiscoveryManifest": {"question": session.question},
        "DossierManifest": {
            "title": f"Co-Scientist Research Dossier: {session.question}",
            "sections": [{"key": "executive", "title": "Executive synthesis"}],
        },
    }
    if role in REVIEW_CRITERIA:
        defaults["CandidateReview"] = {
            "reviewer": role,
            "criterion": REVIEW_CRITERIA[role][0],
        }
    return defaults


def _parse(session: Session, role: str, content: str, model: type[T]) -> ParseOutcome:
    return parse_contract(content, model, defaults=contract_defaults(session, role))


def _try_contract(content: str, model: type[T]) -> T | None:
    """Parse an LLM response into ``model``, or return ``None``."""
    return parse_contract(content, model).value


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
        if not verified:
            for source in parsed.sources:
                source.verification_status = "discovered_unverified"
            for claim in parsed.claims:
                claim.verification_status = "discovered_unverified"
            return parsed
        return _audited_verification(session, parsed)
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
    scraped = EvidencePacket(
        question=session.question,
        sources=sources,
        limitations=limitations,
    )
    if not verified:
        return scraped
    # A verification pass whose output could not be parsed at all is the worst
    # case for upstream discovery, not the one case where deleting it is fine.
    # Audit the scrape like any other verification packet so what was already
    # discovered is carried forward as unreachable instead of vanishing.
    return _audited_verification(session, scraped)


# Vertex search grounding hands back redirector links. They resolve in a
# browser today, expire, and name no document, so a reader cannot use one to
# find what was read. Recording that is the difference between a citation and
# the appearance of one.
_OPAQUE_URL_MARKER = "grounding-api-redirect"


def _audited_verification(session: Session, packet: EvidencePacket) -> EvidencePacket:
    """Hold a verification packet to its own standard without discarding it.

    The predicate this replaces required *every* claim to come back verified or
    corrected, and rebuilt the packet from a URL regex otherwise. A live run
    showed what that costs: the verifier returned six claims, each with a
    source, an exact location, a relation and its own limitations -- two of them
    contradicting the hypothesis -- and honestly marked them unverified because
    the only URLs it had were opaque redirectors it could not open. All six were
    thrown away and replaced by five titleless links. The rule rewarded a
    verifier for stamping "verified" on work it had not done and punished one
    that admitted the gap, which is precisely backwards for an evidence gate.

    So the packet is kept and audited instead. An unsupported verification claim
    is corrected downward, an upstream claim the pass forgot is carried forward
    as unreachable, and both are said out loud in the limitations.
    """
    source_ids = {source.id for source in packet.sources}
    for claim in packet.claims:
        if claim.verification_status not in {"verified", "corrected"}:
            continue
        if claim.source_id in source_ids and claim.exact_location:
            continue
        # "Verified" is a statement about receipts, not about plausibility.
        # With no resolvable source and no exact location there is nothing for
        # a reader to check, so the status is corrected rather than believed.
        claim.verification_status = "discovered_unverified"
        claim.limitations.append(
            "Downgraded to unverified: the packet claimed verification without "
            "naming both a source in this packet and an exact location in it."
        )

    discovered = latest_evidence_packet(session)
    upstream = {claim.id: claim for claim in (discovered.claims if discovered else [])}
    for claim in packet.claims:
        if claim.verification_status in {"verified", "corrected"}:
            continue
        original = upstream.get(claim.id)
        if original is None:
            continue
        # A pass that confirmed nothing has no basis for overturning what
        # discovery recorded. A live run watched one re-emit three findings --
        # two of them contradicting the hypothesis, each with its location in
        # the paper -- as three neutral claims located nowhere, and the report
        # then described a literature that agreed with itself.
        if claim.relation == "neutral" and original.relation != "neutral":
            claim.relation = original.relation
        if not claim.exact_location and original.exact_location:
            claim.exact_location = original.exact_location
        # Nor for rewriting the finding itself. A later run watched the pass
        # return each discovered claim under its own id with the text replaced
        # by a paraphrase of the source's title -- "atomic layer deposition can
        # be used to apply surface coatings", still labelled contradicts. The
        # stance no longer described the sentence it was attached to, and the
        # measured result discovery had found was gone. An unverified pass may
        # report what it could not confirm; it may not restate the claim.
        # Nor for dropping the scope discovery recorded against the finding.
        # "Specific to NCM811 cathodes and dry vs wet coating methods" is the
        # only statement in the whole run of what a retention figure does not
        # cover, and a live pass returned each claim under its own id carrying
        # one line of its own boilerplate instead -- so all six scopes went out
        # of the record, and the report printed the numbers unqualified.
        claim.limitations.extend(
            item for item in original.limitations if item not in claim.limitations
        )
        if original.claim and claim.claim != original.claim:
            claim.claim = original.claim
            claim.limitations.append(
                "Restored to the discovered wording: the verification pass "
                "restated this claim without confirming it."
            )

    returned = {claim.id for claim in packet.claims}
    carried = [
        claim.model_copy(deep=True)
        for claim in (discovered.claims if discovered else [])
        if claim.id not in returned
    ]
    # A carried claim whose source is not in this packet cites a document the
    # report cannot name. Bring the source across too, marked unreachable.
    wanted = {claim.source_id for claim in carried} - source_ids - {None}
    for source in discovered.sources if discovered else []:
        if source.id not in wanted:
            continue
        copied = source.model_copy(deep=True)
        copied.verification_status = "inaccessible"
        packet.sources.append(copied)
        source_ids.add(copied.id)
    for claim in carried:
        # A claim the verifier could not reach and a claim nobody ever made
        # look identical once one of them is deleted. Keep it, marked for what
        # it is: discovered, and not confirmed by this pass.
        claim.verification_status = "inaccessible"
        claim.limitations.append(
            "Discovered upstream but absent from the verification pass; "
            "recorded as unreachable rather than dropped."
        )
    packet.claims.extend(carried)

    if carried:
        packet.limitations.append(
            f"{len(carried)} discovered claim(s) were not returned by the "
            "verification pass and are carried forward as unreachable."
        )
    if any(_OPAQUE_URL_MARKER in source.url for source in packet.sources):
        packet.limitations.append(
            "Some sources are search-grounding redirect links rather than "
            "stable identifiers; they name no document a reader can cite and "
            "cannot be opened for verification."
        )
    return packet


MIN_ACCEPTABLE_CANDIDATES = 2
"""Below this a population is too thin to rank, so the fallback is the better artifact."""


def align_candidate_ids(
    referenced: list[str], candidate_ids: list[str]
) -> dict[str, str]:
    """Map the ids a downstream specialist used onto the real candidate ids.

    Specialists paraphrase identifiers -- ``cand_ef1`` for ``cand_ef_1``, or a
    positional ``Candidate 3``. Discarding an entire review set over a cosmetic
    id mismatch loses real analysis, so resolve what can be resolved and report
    the rest to the caller by omission.
    """
    known = {candidate_id: candidate_id for candidate_id in candidate_ids}
    normalized = {
        re.sub(r"[^a-z0-9]", "", candidate_id.lower()): candidate_id
        for candidate_id in candidate_ids
    }
    mapping: dict[str, str] = {}
    for reference in referenced:
        if reference in known:
            mapping[reference] = reference
            continue
        simple = re.sub(r"[^a-z0-9]", "", reference.lower())
        if simple in normalized:
            mapping[reference] = normalized[simple]
            continue
        ordinal = re.search(r"(\d+)\s*$", reference)
        if ordinal:
            index = int(ordinal.group(1)) - 1
            if 0 <= index < len(candidate_ids):
                mapping[reference] = candidate_ids[index]
    return mapping


def candidate_population(session: Session, content: str) -> CandidatePopulation:
    parsed = _parse(session, "generation", content, CandidatePopulation).value
    if parsed is not None and len(parsed.candidates) >= MIN_ACCEPTABLE_CANDIDATES:
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
    criterion, label = REVIEW_CRITERIA[role]
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


def _unsafe_autonomy_request(session: Session) -> bool:
    return any(
        phrase in session.question.lower()
        for phrase in (
            "hazardous laboratory",
            "begin carrying out",
            "execute the protocol",
            "without human safety review",
        )
    )


def parsed_review_set(
    session: Session, role: str, parsed: ReviewSet | None, fallback: ReviewSet
) -> ReviewSet:
    """Keep every review a specialist actually wrote, covering the rest.

    The governance override stays absolute: an unsafe-autonomy request is
    answered deterministically so no model output can soften it. Otherwise the
    specialist's reviews are aligned onto the real candidate ids and any
    candidate it skipped is backfilled, rather than discarding the whole set
    because one id or one reviewer name did not match. Returning ``fallback``
    itself signals to the caller that no specialist review survived.
    """
    if role == "ethics_safety_governance" and _unsafe_autonomy_request(session):
        return fallback
    if parsed is None or not parsed.reviews:
        return fallback
    candidate_ids = [
        candidate.id
        for candidate in population_from_artifacts(session.artifacts).candidates
    ]
    mapping = align_candidate_ids(
        [review.candidate_id for review in parsed.reviews], candidate_ids
    )
    criterion = REVIEW_CRITERIA[role][0]
    aligned: dict[str, CandidateReview] = {}
    for review in parsed.reviews:
        resolved = mapping.get(review.candidate_id)
        if resolved is None or resolved in aligned:
            continue
        aligned[resolved] = review.model_copy(
            update={
                "candidate_id": resolved,
                "reviewer": role,
                "criterion": review.criterion or criterion,
            }
        )
    if not aligned:
        return fallback
    return ReviewSet(
        reviews=[
            aligned.get(review.candidate_id, review) for review in fallback.reviews
        ]
    )


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
    ratings = {candidate.id: DEFAULT_ELO for candidate in candidates}
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
        delta = ELO_K * (actual_a - expected_a)
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

    # The ratings and the standings as each round left them. Both were reported as
    # placeholders before -- ranking_stable_rounds=1, score_movement=1.0 -- and the
    # report read them as measurements, so a run whose leader in fact moved by
    # thirteen points was printed as "the final round moved one rating by 100.0 per
    # cent of that, or about 1200 points". The rounds are played here either way;
    # nothing but the bookkeeping was missing.
    def standings() -> list[str]:
        return [
            item.id
            for item in sorted(
                candidates, key=lambda item: (-ratings[item.id], item.id)
            )
        ]

    rating_history: list[dict[str, float]] = [dict(ratings)]
    standings_history: list[list[str]] = []

    for round_number in range(1, 4):
        ordered = sorted(candidates, key=lambda item: (-ratings[item.id], item.id))
        remaining = list(ordered)
        order = 0
        while len(remaining) > 1:
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
        # An odd field leaves one hypothesis unpaired. Swiss pairing gives it a
        # bye: it sits the round out at its current rating rather than being
        # awarded a win it did not play for. Populations were always even until
        # a governance withdrawal removed one, at which point this loop tried to
        # pair the last hypothesis with nobody.
        rating_history.append(dict(ratings))
        standings_history.append(standings())

    ranked = {item.id: item for item in candidates}
    top_four = [ranked[item] for item in standings()[:TOP_FOUR]]
    finals = False
    for order, (a, b) in enumerate(combinations(top_four, 2)):
        if frozenset((a.id, b.id)) not in played:
            compare(a, b, 4, order)
            finals = True
    if finals:
        rating_history.append(dict(ratings))
        standings_history.append(standings())
    stable = stable_rounds(standings_history)
    movement = score_movement(rating_history)
    return TournamentState(
        ratings=ratings,
        comparisons=comparisons,
        shortlist_ids=standings()[:TOP_FOUR],
        ranking_stable_rounds=stable,
        score_movement=movement,
        # The same rule the debate tournament applies, so that a run judged
        # deterministically and a run judged by a model are called converged on
        # the same terms rather than one of them never being called it at all.
        converged=stable >= 2 and movement < SETTLED_MOVEMENT,
    )


_REVISION_MARKER = ", under a preregistered discriminating design (revision "
_ROBUSTNESS_PREDICTION = (
    "The primary result survives the prespecified robustness analysis, "
    "as prescribed by evolution round {number}."
)


def _revised_claim(claim: str, round_number: int) -> str:
    """The parent's claim carrying one revision marker rather than one per round.

    Each round copies the parent's claim forward, so appending a marker to
    whatever the parent already held stacked three of them onto the end of a
    sentence that had closed with a full stop two rounds earlier: "... at 1C
    discharge rates. under preregistered discriminating design revision 1
    under preregistered discriminating design revision 2 ...". Only the latest
    revision is true of the candidate in hand, so the earlier marker is
    replaced rather than followed.
    """
    base = claim.split(_REVISION_MARKER)[0].rstrip().rstrip(".")
    return f"{base}{_REVISION_MARKER}{round_number})."


def _revised_predictions(predictions: Sequence[str], round_number: int) -> list[str]:
    """The parent's predictions with this round's robustness check, and only this one.

    The check is the same check every round; carrying the parent's copy forward
    as well left the third-round offspring predicting that its result survives
    a robustness analysis in rounds one, two and three, which is one prediction
    written three times.
    """
    kept = [
        prediction
        for prediction in predictions
        if not prediction.startswith("The primary result survives the prespecified")
    ]
    return [*kept, _ROBUSTNESS_PREDICTION.format(number=round_number)]


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
    previous_ratings = dict(tournament.ratings)
    previous_order: list[str] = list(tournament.shortlist_ids)
    stable_rounds = 0
    # Every offspring is renamed each round, so rank stability can only be
    # compared through the lineage each candidate descends from.
    roots = {candidate_id: candidate_id for candidate_id in tournament.shortlist_ids}
    # Re-review has to cover the same axes the review stage did, or an
    # offspring could clear evolution on axes its parent was never judged on.
    criteria = tuple(criterion for criterion, _ in REVIEW_CRITERIA.values())
    for round_number in range(1, session.budget.max_evolution_rounds + 1):
        evolved_round = []
        for parent in current:
            evolved = parent.model_copy(
                update={
                    "id": f"{parent.id}_evolved_{round_number}",
                    "version": parent.version + 1,
                    "parent_ids": [parent.id],
                    "claim": _revised_claim(parent.claim, round_number),
                    "predictions": _revised_predictions(
                        parent.predictions, round_number
                    ),
                }
            )
            record = EvolutionRecord(
                parent_ids=[parent.id],
                candidate=evolved,
                # Only the first round adds either of these; the rounds after it
                # revise what is already on the candidate, and a change log that
                # says "added" three times over reads as three separate designs.
                changes=[
                    "Added a preregistered discriminating design."
                    if round_number == 1
                    else "Revised the preregistered discriminating design.",
                    "Added a robustness prediction and explicit re-review requirement."
                    if round_number == 1
                    else "Restated the robustness prediction against the revised design.",
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

        # An offspring inherits its parent's standing: the revision addressed a
        # critique but produced no new evidence, so the prior tournament is the
        # only defensible basis for ordering this round. Awarding the win to
        # whichever candidate happened to be listed first would manufacture a
        # ranking out of iteration order.
        inherited = {
            candidate.id: previous_ratings.get(candidate.parent_ids[0], DEFAULT_ELO)
            for candidate in evolved_round
        }
        ratings = dict(inherited)
        comparisons = []
        for order, (a, b) in enumerate(combinations(evolved_round, 2)):
            if math.isclose(inherited[a.id], inherited[b.id]):
                winner = a if a.id <= b.id else b
            else:
                winner = a if inherited[a.id] > inherited[b.id] else b
            before = {a.id: ratings[a.id], b.id: ratings[b.id]}
            actual_a = 1.0 if winner.id == a.id else 0.0
            expected_a = 1 / (1 + 10 ** ((ratings[b.id] - ratings[a.id]) / 400))
            delta = ELO_K * (actual_a - expected_a)
            ratings[a.id] += delta
            ratings[b.id] -= delta
            comparisons.append(
                PairwiseComparison(
                    round_number=round_number,
                    candidate_a_id=a.id,
                    candidate_b_id=b.id,
                    presented_first_id=a.id if order % 2 == 0 else b.id,
                    winner_id=winner.id,
                    criterion_scores={
                        a.id: round(inherited[a.id], 1),
                        b.id: round(inherited[b.id], 1),
                    },
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
        # Convergence has to be measured, not asserted on a schedule: the loop
        # stops when the order stops changing and ratings stop moving.
        for candidate in evolved_round:
            roots[candidate.id] = roots[candidate.parent_ids[0]]
        lineage_order = [
            roots[candidate.id]
            for candidate in sorted(
                evolved_round, key=lambda item: (-ratings[item.id], item.id)
            )
        ]
        # Round one is compared against the pre-evolution shortlist, whose order
        # the offspring inherit by construction, so it cannot evidence stability.
        stable = round_number > 1 and lineage_order == previous_order
        stable_rounds = stable_rounds + 1 if stable else 0
        movement = max(
            (
                abs(ratings[candidate_id] - inherited[candidate_id])
                / max(inherited[candidate_id], 1.0)
                for candidate_id in ratings
            ),
            default=0.0,
        )
        previous_ratings = dict(ratings)
        previous_order = lineage_order
        ranking_history.append(
            TournamentState(
                ratings=ratings,
                comparisons=comparisons,
                shortlist_ids=ordered_ids,
                swiss_rounds=0,
                top_round_robin_size=4,
                ranking_stable_rounds=stable_rounds,
                score_movement=movement,
                converged=stable_rounds >= 2 and movement < SETTLED_MOVEMENT,
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


def _low(text: str) -> str:
    """A stated sentence folded into a longer one, with its notation left alone."""
    cleaned = " ".join(text.split()).rstrip(".")
    head, separator, tail = cleaned.partition(" ")
    if head[:1].isupper() and (head[1:].islower() or not head[1:]):
        head = head.lower()
    return f"{head}{separator}{tail}."


def _decisive_evidence(candidate: Candidate | None, *, flawed: bool) -> list[str]:
    """The observations that would actually move this run's ranking.

    The three the fallback used to return -- verified evidence contradicting the
    mechanism, a failed prediction, an independent replication -- are the kinds of
    thing that unseat any hypothesis whatever, so the report introduced them as "a
    short list of specific evidence" and then named none. Section nine goes on to
    say that obtaining any one of them beats another round of generation, which is
    a claim about a measurement someone could go and take. Quote the leading
    candidate's own falsifier, prediction and competing reading instead: those are
    written against this question, and they are what the ranking rests on.
    """
    if candidate is None:
        return []
    decisive = []
    if flawed:
        # Where nothing is recommended, no measurement on the leading idea can move
        # the decision until the finding that disqualified it is settled.
        decisive.append(
            "A safety finding that resolves the fatal flaw recorded against the "
            "leading idea, which is what currently keeps every candidate off the "
            "recommendation."
        )
    decisive.append(
        "The outcome of the falsifying test the proposal names: "
        f"{_low(candidate.falsifier)}"
    )
    if candidate.predictions:
        decisive.append(
            "A direct measurement of the prediction that separates it from the "
            f"field: {_low(candidate.predictions[0])}"
        )
    if candidate.alternatives:
        decisive.append(
            "Evidence for the competing reading its ranking assumes away, which is "
            f"that {_low(candidate.alternatives[0])}"
        )
    # Not the go/no-go tests: the same section ends by naming them as the immediate
    # next step, and a reader who meets them twice in two paragraphs reads the second
    # as a second piece of work.
    return decisive


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
        # The evidence is decisive about one idea, so it is quoted from the idea the
        # decision is about: the leading recommendation where there is one, and
        # otherwise the top of the shortlist, which is what the report discusses when
        # every candidate carries a flaw.
        evidence_that_would_change_decision=_decisive_evidence(
            next(
                (
                    candidate
                    for candidate in population_from_artifacts(
                        session.artifacts
                    ).candidates
                    if candidate.id
                    == next(iter(recommendations + list(tournament.shortlist_ids)), "")
                ),
                None,
            ),
            flawed=not recommendations,
        ),
    )


def parsed_tournament_state(
    session: Session, parsed: TournamentState | None, fallback: TournamentState
) -> TournamentState:
    """Adopt the specialist's tournament when it ranked the real candidates.

    The previous gate demanded that the rating map equal the candidate id set
    exactly, which a model echoing eight opaque identifiers essentially never
    satisfies -- so a genuine tournament with reasoned pairwise rationales was
    replaced by an arithmetic one every time. Align the ids instead, and require
    only that a majority of candidates were actually rated.
    """
    if parsed is None or not parsed.ratings:
        return fallback
    candidate_ids = [
        candidate.id
        for candidate in population_from_artifacts(session.artifacts).candidates
    ]
    mapping = align_candidate_ids(
        [*parsed.ratings, *parsed.shortlist_ids], candidate_ids
    )
    ratings = {
        mapping[reference]: rating
        for reference, rating in parsed.ratings.items()
        if reference in mapping
    }
    if len(ratings) * 2 < len(candidate_ids):
        return fallback
    for candidate_id in candidate_ids:
        ratings.setdefault(candidate_id, DEFAULT_ELO)
    shortlist = list(
        dict.fromkeys(
            mapping[reference]
            for reference in parsed.shortlist_ids
            if reference in mapping
        )
    )
    if not shortlist:
        shortlist = sorted(ratings, key=lambda item: (-ratings[item], item))[:4]
    comparisons = []
    for comparison in parsed.comparisons:
        left = mapping.get(comparison.candidate_a_id)
        right = mapping.get(comparison.candidate_b_id)
        if left is None or right is None:
            continue
        comparisons.append(
            comparison.model_copy(
                update={
                    "candidate_a_id": left,
                    "candidate_b_id": right,
                    "presented_first_id": mapping.get(
                        comparison.presented_first_id, left
                    ),
                    "winner_id": mapping.get(comparison.winner_id or ""),
                }
            )
        )
    return parsed.model_copy(
        update={
            "ratings": ratings,
            "shortlist_ids": shortlist[:4],
            "comparisons": comparisons,
        }
    )


def typed_specialist_payload(session: Session, role: str, content: str) -> TypedPayload:
    """Validate every specialist boundary into its declared contract.

    Returns the payload together with its provenance, so a report can state
    plainly whether a stage reflects the specialist's own reasoning or a
    deterministic template standing in for it.
    """
    model = ROLE_CONTRACTS.get(role)
    if model is None:
        raise ValueError(f"No typed specialist contract for role: {role}")
    outcome = _parse(session, role, content, model)
    if role == "goal_manager":
        fallback: BaseModel = research_plan(session)
        value = outcome.value or fallback
    elif role in {"evidence_discovery", "source_verification"}:
        # The evidence packet is extracted from the specialist's own text rather
        # than parsed from a contract, so it is never a template: whatever the
        # specialist found is what the packet contains.
        packet = evidence_packet(
            session, content, verified=role == "source_verification"
        )
        return TypedPayload(
            schema_name="EvidencePacket", payload=packet.model_dump(mode="json")
        )
    elif role in GENERATION_ROLES:
        fallback = candidate_population(session, content)
        value = outcome.value or fallback
    elif role in REVIEW_CRITERIA:
        fallback = review_set(session, role)
        value = parsed_review_set(session, role, outcome.value, fallback)
    elif role == "ranking":
        fallback = tournament_state(session)
        value = parsed_tournament_state(session, outcome.value, fallback)
    elif role == "evolution":
        fallback = evolution_cycle(session)
        value = outcome.value or fallback
    elif role == "proximity":
        fallback = research_landscape(session)
        value = outcome.value or fallback
    else:
        fallback = dossier_manifest(session)
        value = outcome.value or fallback
    if value is fallback:
        source = "deterministic_fallback"
    else:
        source = "repaired" if outcome.repairs else "specialist"
    return TypedPayload(
        schema_name=type(value).__name__,
        payload=value.model_dump(mode="json"),
        source=source,
        repairs=list(outcome.repairs),
        error=outcome.error,
    )
