"""Versioned contracts for the local supervisor and A2A task boundaries."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model_catalog import DEFAULT_LANGUAGE, DEFAULT_MODEL

SCHEMA_VERSION = "1.0"
STAGES = (
    "scope",
    "evidence",
    "generate",
    "reflect",
    "rank",
    "evolve",
    "proximity",
    "meta_review",
    "report",
)
FORKED_STAGES = ("scope", "evidence")
"""The stages a fork carries over from the run it forked, rather than running.

Named here because the report has to disclaim exactly these: a fork's stage count
includes them, and the models that produced them land in its "Produced by" list
beside the models it called itself.
"""
RESEARCH_MODES = (
    "experimental",
    "observational",
    "computational",
    "theory_simulation",
    "systematic_review",
    "measurement_field",
)
DISCIPLINES = (
    "chemistry_materials",
    "biology_medicine",
    "physics_engineering",
    "computer_science_ai",
    "mathematics_statistics",
    "earth_climate_sciences",
    "neuroscience_cognitive",
    "astronomy_astrophysics",
    "social_science_economics",
    "environmental_ecology",
    "pharmacology_toxicology",
    "general_interdisciplinary",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class ApprovalMode(StrEnum):
    HUMAN = "human"
    AUTO = "auto"


class ApprovalProfile(StrEnum):
    """Researcher interaction policy for promotion decisions."""

    AUTO = "auto"
    MILESTONE = "milestone"
    STAGE = "stage"
    ARTIFACT = "artifact"


class DecisionAction(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    STOP = "stop"
    REFINE_SECTION = "refine_section"


class ArtifactStatus(StrEnum):
    DRAFT = "draft"
    ACCEPTED = "accepted"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class TaskState(StrEnum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Contract(BaseModel):
    """Base contract that is strict at service boundaries."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    schema_version: str = SCHEMA_VERSION


class ResearchPlan(Contract):
    research_mode: str = "experimental"
    question: str
    intended_claim: str = "hypothesis"
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    stopping_criteria: list[str] = Field(default_factory=list)
    governance_requirements: list[str] = Field(default_factory=list)


class InputRequirement(Contract):
    id: str = Field(default_factory=lambda: new_id("input"))
    input_type: str
    description: str
    reason: str
    blocking: bool = True
    permitted_fallback: Literal["literature_only", "none"] = "none"
    status: Literal["missing", "provided", "fallback_accepted"] = "missing"
    provided_reference: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status in {"provided", "fallback_accepted"}


VerificationStatus = Literal[
    "discovered_unverified",
    "metadata_verified",
    "verified",
    "inaccessible",
    "retracted",
    "corrected",
]
"""What is known about a source, in ascending order of what it licenses.

``metadata_verified`` is the tier this system was missing, and its absence was
distorting every evidence count. A paywalled paper whose DOI resolves, whose
title and authors a registry confirms, and against which no retraction is
recorded is not the same object as a citation nobody can find -- but with only
``verified`` and ``inaccessible`` available it had to be recorded as the second,
which made the evidence floor a measure of open-access availability rather than
of scholarship. It is reported separately and counted at a discount, because
nothing has checked that the paper says what was attributed to it.
"""

VERIFIED_STATUSES = frozenset({"verified", "corrected"})
"""Statuses that mean someone read the document and found the passage."""

CREDITED_STATUSES = frozenset({"verified", "corrected", "metadata_verified"})
"""Statuses that count toward the evidence floor, at their respective weights."""

METADATA_VERIFIED_WEIGHT = 0.5
"""What an unread but registry-confirmed source is worth against the floor.

Half, because half of what verification promises has been done: the document
provably exists and is the one that was cited, and nothing has confirmed that it
supports the claim resting on it.
"""


class SourceRecord(Contract):
    id: str = Field(default_factory=lambda: new_id("src"))
    url: str
    title: str = ""
    source_type: str = "unknown"
    accessed_at: str = Field(default_factory=utc_now)
    verification_status: VerificationStatus = "discovered_unverified"
    verification_note: str = ""
    """Why the source holds the status it does, in one sentence a reader can act on.

    "Inaccessible" alone tells a researcher nothing about whether to chase the
    paper down themselves; "HTTP 403 from the publisher, and no open-access copy
    is registered" tells them exactly.
    """
    facet: str = ""
    """Which evidence facet this source was found under, when it is known."""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    container: str = ""
    supports_claim_ids: list[str] = Field(default_factory=list)


class EvidenceClaim(Contract):
    id: str = Field(default_factory=lambda: new_id("claim"))
    claim: str
    source_id: str | None = None
    exact_location: str = ""
    relation: Literal["supports", "contradicts", "neutral"] = "neutral"
    verification_status: VerificationStatus = "discovered_unverified"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class EvidencePacket(Contract):
    id: str = Field(default_factory=lambda: new_id("evidence"))
    question: str
    sources: list[SourceRecord] = Field(default_factory=list)
    claims: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def verified(self) -> bool:
        """Whether every single record in the packet was checked against a document.

        Retained because it is the strictest thing that can be said about a
        corpus, and the dossier reports it. It is no longer what the generation
        gate consults: an all-or-nothing test over ninety sources fails on the
        one paper whose publisher was down, so it was refusing corpora that were
        overwhelmingly sound. :class:`EvidenceFloor` is the gate now.
        """
        verified_sources = bool(self.sources) and all(
            source.verification_status in VERIFIED_STATUSES for source in self.sources
        )
        verified_claims = bool(self.claims) and all(
            claim.source_id
            and claim.exact_location
            and claim.verification_status in VERIFIED_STATUSES
            for claim in self.claims
        )
        return verified_sources and verified_claims


EVIDENCE_FLOOR_CREDIT = 8.0
"""Weighted verified sources a corpus needs before hypotheses may be generated.

Eight is what it takes for a hypothesis to rest on a literature rather than on a
handful of papers that happened to be open access. It is weighted rather than
counted so that a field where most of the relevant work is paywalled can still
clear it -- sixteen registry-confirmed papers are worth the same as eight read
ones, and the shortfall message says which kind the corpus is made of.
"""

EVIDENCE_FLOOR_FACETS = 4
"""How many of the seven evidence facets must be non-empty.

Four rather than seven because ``corrections_retractions`` is legitimately empty
for most healthy literatures, and requiring it would make the gate unclearable
on sound evidence.
"""

VERIFICATION_BATCH_SIZE = 12
"""How many discovered sources one verification specialist is asked to check.

Twelve is a list a model will enumerate. Handed all ninety leads of a live
fan-out at once, the specialist returned five sources and said nothing about the
rest, and the evidence floor then measured a literature of ninety papers as one
usable source.
"""

MAX_VERIFICATION_BATCHES = 10
"""Ceiling on concurrent verification batches, so a wide corpus stays bounded.

Ten batches is a hundred and twenty sources, past the reach of any single
fan-out so far. A corpus that exceeds it is truncated by lead quality rather
than arrival order, and the manifest records that the ceiling was reached --
a cap that reports nothing reads as complete coverage.
"""


class EvidenceFloor(Contract):
    """Whether a corpus is strong enough to generate hypotheses from.

    Every field is reported to the researcher whether the floor is met or not,
    because the decision the gate offers -- keep searching, or proceed knowing
    what is thin -- cannot be made from a pass/fail bit.
    """

    verified_sources: int = 0
    metadata_verified_sources: int = 0
    weighted_credit: float = 0.0
    required_credit: float = EVIDENCE_FLOOR_CREDIT
    facets_covered: list[str] = Field(default_factory=list)
    facets_missing: list[str] = Field(default_factory=list)
    required_facets: int = EVIDENCE_FLOOR_FACETS
    disconfirming_sources: int = 0
    retracted_sources: int = 0
    inaccessible_sources: int = 0
    searched_for_disconfirming: bool = False
    """Whether the run actually went looking for evidence against its direction.

    The disconfirming requirement is soft, but only after the search has been
    made: "we found none" and "we never looked" are different states, and only
    the first is a finding.
    """
    met: bool = False
    shortfalls: list[str] = Field(default_factory=list)

    @property
    def credit_met(self) -> bool:
        return self.weighted_credit >= self.required_credit

    @property
    def facets_met(self) -> bool:
        return len(self.facets_covered) >= self.required_facets


EVIDENCE_FACETS = (
    "supporting",
    "contradictory",
    "negative_null",
    "replication",
    "methods",
    "safety_governance",
    "corrections_retractions",
)

FACET_PHRASES: dict[str, str] = {
    "supporting": "supporting evidence",
    "contradictory": "evidence contradicting the leading direction",
    "negative_null": "negative or null results",
    "replication": "independent replication",
    "methods": "methodological detail",
    "safety_governance": "safety or governance evidence",
    "corrections_retractions": "corrections or retractions affecting the sources used",
}
"""How each facet is named in prose a person reads.

The facet tokens are an enum, and both the coverage gaps and the discovery prompt
printed them with their underscores swapped for spaces. That is fine for
``supporting`` and produces "No adequate corrections retractions evidence was
discovered" for the facet that matters most to a reader deciding whether to trust
a citation.
"""


MAX_DISCOVERY_PASSES = 8
"""The hard ceiling on Deep Research interactions one run may start.

Exactly the seven-facet fan-out plus one gap-closing pass. A pass costs roughly
three dollars and cannot be cancelled -- Vertex answers ``interactions.cancel()``
with 501 UNIMPLEMENTED -- and the deployed service takes anonymous requests, so
the bound is in the contract rather than in an operator's attention. A run that
reaches it records that it did instead of quietly stopping.
"""


class DeepResearchRun(Contract):
    id: str = Field(default_factory=lambda: new_id("deep_research"))
    pass_number: int = Field(ge=1, le=MAX_DISCOVERY_PASSES)
    facet: str = ""
    """Which evidence facet this pass was sent to cover, on a fan-out pass.

    Empty on the gap-closing pass, which is aimed at whatever the fan-out left
    open rather than at one axis.
    """
    interaction_id: str = ""
    status: Literal[
        "queued",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
        "incomplete",
        "requires_action",
        "budget_exceeded",
        "timed_out",
    ] = "queued"
    prompt_gap_ids: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now)
    completed_at: str | None = None
    error: str = ""
    raw_artifact_reference: str = ""
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    usage: dict[str, Any] = Field(default_factory=dict)
    poll_count: int = Field(default=0, ge=0)


class DiscoveryStatement(Contract):
    id: str = Field(default_factory=lambda: new_id("statement"))
    text: str
    facet: str
    source_urls: list[str] = Field(default_factory=list)
    originating_pass: int = Field(ge=1, le=MAX_DISCOVERY_PASSES)
    relation: Literal["supports", "contradicts", "neutral"] = "neutral"
    uncertainty: str = ""


class DiscoveryNarrative(Contract):
    question: str
    research_directions: list[str] = Field(default_factory=list)
    statements: list[DiscoveryStatement] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    summary: str = ""
    facet: str = ""
    """Which facet the pass that produced this narrative was sent to cover.

    The fan-out asks seven different questions and gets seven reports back. Only
    the statements carried the facet, so a narrative whose paragraphs all failed
    to resolve a citation lost every trace of what it had been asked -- and the
    report, which prints the reports themselves, had nothing to label them with
    and ran all seven together as one block of prose.
    """
    pass_number: int = 0
    """Which search pass wrote this report, or 0 on a session saved before this field.

    The Knowledge Base numbered the reports as it printed them, so a run whose third
    pass came back empty had its fourth report headed "Pass 3" -- against a discovery
    appendix that numbers the passes as they ran. The two sections disagreed about
    which pass found what, in the one place the report tells the reader to compare
    them.
    """

    truncated: bool = False
    """Whether ``summary`` is a prefix of the report rather than the whole of it.

    The store cuts the report at a fixed length, which lands mid-word. Printing
    that without saying so presents a sentence the provider never wrote as the
    end of its findings.
    """


class SourceLead(Contract):
    id: str = Field(default_factory=lambda: new_id("lead"))
    canonical_url: str
    title: str = ""
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    source_type: str = "unknown"
    provider: str = "deep_research"
    originating_passes: list[int] = Field(default_factory=list)
    originating_statement_ids: list[str] = Field(default_factory=list)
    facets: list[str] = Field(default_factory=list)
    """Which evidence facets this source was found under.

    A lead is a search result, and the search that found it is the only thing
    that says what kind of evidence it is. Discarding that was why the reader
    was shown forty-four titles in one undifferentiated list: nothing left in
    the record could group them, so nothing did.
    """
    verification_status: VerificationStatus = "discovered_unverified"
    """What verification later concluded about this lead, once it has run.

    Previously pinned to ``discovered_unverified`` by its own type, which was
    accurate at the moment a lead is created and made it impossible to ever
    write the answer back. The manifest is what the evidence panel reads, so a
    lead that cannot record its outcome is a panel that cannot show one.
    """
    verification_note: str = ""
    claim_relations: list[str] = Field(default_factory=list)
    """Whether the claims resting on this source support or contradict the direction.

    A reference list that does not distinguish the two is a reading list. The
    thing a researcher needs to see at a glance is which of these papers
    disagrees with where the run is heading.
    """
    raw_artifact_reference: str = ""


class ResearchGap(Contract):
    id: str = Field(default_factory=lambda: new_id("gap"))
    direction: str
    facet: str
    description: str
    decision_impact: Literal["low", "medium", "high", "blocking"] = "medium"
    priority: int = Field(default=1, ge=1, le=5)
    status: Literal["open", "closed", "unavailable"] = "open"


class DiscoveryCoverage(Contract):
    direction_scores: dict[str, float] = Field(default_factory=dict)
    facet_scores: dict[str, float] = Field(default_factory=dict)
    weighted_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficient: bool = False
    authoritative_source_count: int = Field(default=0, ge=0)
    new_authoritative_source_count: int = Field(default=0, ge=0)
    material_gaps_closed: int = Field(default=0, ge=0)
    gaps: list[ResearchGap] = Field(default_factory=list)


class EnrichmentRequest(Contract):
    id: str = Field(default_factory=lambda: new_id("enrichment"))
    provider: Literal[
        "google_search",
        "crossref",
        "openalex",
        "datacite",
        "pubmed",
        "agent_search",
    ]
    gap_ids: list[str] = Field(default_factory=list)
    query: str
    status: Literal["queued", "working", "completed", "failed", "skipped"] = "queued"
    result_artifact_reference: str = ""


class DiscoveryManifest(Contract):
    question: str
    runs: list[DeepResearchRun] = Field(
        default_factory=list, max_length=MAX_DISCOVERY_PASSES
    )
    discovery_angles: list[str] = Field(default_factory=list)
    """Which sub-searches a decomposed grounded pass ran, in the order they ran.

    Empty on a Deep Research pass, which is one search that iterates rather than
    several that fan out. The appendix reads this to say how the literature was
    found: "one set of queries" was true of the pass this replaced and would be a
    false modesty about the pass that replaced it.
    """
    narratives: list[DiscoveryNarrative] = Field(default_factory=list)
    source_leads: list[SourceLead] = Field(default_factory=list)
    coverage_history: list[DiscoveryCoverage] = Field(default_factory=list)
    enrichment_requests: list[EnrichmentRequest] = Field(default_factory=list)
    convergence_reason: str = ""
    verification_handoff_source_ids: list[str] = Field(default_factory=list)
    stored_interaction_notice: bool = True
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    # How many discovered leads were sent to a verification specialist, and how
    # many the batch ceiling left behind. A truncation nobody records reads, in
    # every panel downstream, as a corpus that was checked in full.
    leads_sent_to_verification: int = Field(default=0, ge=0)
    leads_beyond_verification_ceiling: int = Field(default=0, ge=0)
    # And how many discovery found but the manifest could not hold. Same reason:
    # a corpus of three hundred cut to ninety is a decision about the evidence,
    # and every number downstream is computed from what survived it.
    leads_beyond_retention_ceiling: int = Field(default=0, ge=0)
    # And how many gaps the last revision of this stage named but could not
    # search, having reached its ceiling. Same reason again: the gaps stay
    # listed either way, so without this a gap that was searched and came back
    # empty is indistinguishable from one nothing was spent on.
    gap_searches_deferred: int = Field(default=0, ge=0)
    synthesis_report: str = ""


class ResearchDirection(Contract):
    id: str = Field(default_factory=lambda: new_id("dir"))
    title: str
    scope: str
    mechanism_or_concept: str
    outcome: str
    competing_explanations: list[str] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)
    search_questions: list[str] = Field(default_factory=list)


class EvidenceGap(Contract):
    id: str = Field(default_factory=lambda: new_id("gap"))
    direction_id: str
    description: str
    decision_impact: Literal["low", "medium", "high", "blocking"] = "medium"
    resolution_query: str


class EvidenceRequest(Contract):
    id: str = Field(default_factory=lambda: new_id("evreq"))
    requesting_stage: str
    requesting_agent: str
    claim_to_verify: str
    priority: int = Field(default=1, ge=1, le=5)
    budget_usd: float = Field(default=1.0, ge=0.0)
    status: Literal["submitted", "working", "completed", "failed", "rejected"] = (
        "submitted"
    )
    resulting_manifest_version: int | None = None


class CitationAnchor(Contract):
    id: str = Field(default_factory=lambda: new_id("cite"))
    claim_id: str
    human_citation_number: int
    report_location: str
    display_text: str


class KnowledgeBaseManifest(Contract):
    id: str = Field(default_factory=lambda: new_id("kb"))
    version: int = 1
    parent_version: int | None = None
    directions: list[ResearchDirection] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    coverage_matrix: dict[str, float] = Field(default_factory=dict)
    contradiction_graph: list[tuple[str, str]] = Field(default_factory=list)
    unresolved_gaps: list[EvidenceGap] = Field(default_factory=list)
    search_cutoff_date: str = Field(default_factory=utc_now)
    checksum: str = ""
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


class Candidate(Contract):
    id: str = Field(default_factory=lambda: new_id("candidate"))
    version: int = 1
    parent_ids: list[str] = Field(default_factory=list)
    title: str
    claim: str
    rationale: str
    mechanism_model: str
    validation_protocol: str
    predictions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    falsifier: str
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    generation_strategy: Literal[
        "evidence_first",
        "mechanism_first",
        "analogy_transfer",
        "competing_explanation",
    ] = "mechanism_first"
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    go_no_go_tests: list[str] = Field(default_factory=list)
    workflow_diagram_mermaid: str = ""
    # Unscored means unscored. These defaulted to 4, so a generator that returned
    # no self-assessment still had "4/5 novelty" printed under its name in the
    # dossier, and a reader had no way to tell a judgement from a filled-in blank.
    score_novelty: int | None = Field(default=None, ge=1, le=5)
    score_feasibility: int | None = Field(default=None, ge=1, le=5)
    score_impact: int | None = Field(default=None, ge=1, le=5)
    score_correctness: int | None = Field(default=None, ge=1, le=5)
    score_verification: int | None = Field(default=None, ge=1, le=5)


class CandidatePopulation(Contract):
    candidates: list[Candidate] = Field(min_length=1)
    target_size: int = Field(default=8, ge=1)
    diversity_dimensions: list[str] = Field(default_factory=list)
    comparison_criteria: list[str] = Field(default_factory=list)


class CandidateReview(Contract):
    id: str = Field(default_factory=lambda: new_id("review"))
    candidate_id: str
    criterion: Literal[
        "evidence_correctness",
        "novelty",
        "methods_feasibility",
        "impact_safety",
        # Safety and governance decide whether work may proceed at all; impact
        # decides whether it is worth proceeding. Sessions written before this
        # split still carry "impact_safety" for both, so that value stays legal.
        "safety_governance",
    ]
    findings: list[str] = Field(default_factory=list)
    fatal_flaws: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    rebuttals: list[str] = Field(default_factory=list)
    recommendation: Literal["advance", "revise", "reject", "insufficient_evidence"]
    reviewer: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    stood_in: bool = False
    """Whether this review is the fixed placeholder written for a skipped idea.

    A reviewer that answers for seven of eight ideas has its seven kept and the
    eighth backfilled, because discarding the set over one missing id costs more
    than it saves. Nothing recorded which was which, so a live run printed a
    placeholder as the rank-1 idea's feasibility review -- with a score, a stated
    confidence of 0.45 and an objection -- and the conclusion under it sent the
    reader to that review as the one to read before commissioning the work.
    """


class ReviewSet(Contract):
    reviews: list[CandidateReview] = Field(default_factory=list)


class PairwiseComparison(Contract):
    id: str = Field(default_factory=lambda: new_id("comparison"))
    round_number: int = Field(ge=1)
    candidate_a_id: str
    candidate_b_id: str
    presented_first_id: str
    winner_id: str | None = None
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    elo_before: dict[str, float] = Field(default_factory=dict)
    elo_after: dict[str, float] = Field(default_factory=dict)
    rubric_version: str = "1"
    # Turn-by-turn transcript when the match was decided by a simulated
    # scientific debate rather than by arithmetic. Empty for computed matches,
    # which is itself the signal that no reasoning backs the result.
    debate_turns: list[str] = Field(default_factory=list)
    # A single-turn comparison and a multi-turn debate are not equally strong
    # evidence, so they are not recorded under one name. Only "llm_debate"
    # means the two hypotheses were actually argued against each other.
    judge: Literal["deterministic", "llm_comparison", "llm_debate"] = "deterministic"


class TournamentState(Contract):
    ratings: dict[str, float] = Field(default_factory=dict)
    comparisons: list[PairwiseComparison] = Field(default_factory=list)
    shortlist_ids: list[str] = Field(default_factory=list)
    swiss_rounds: int = 3
    top_round_robin_size: int = 4
    ranking_stable_rounds: int = 0
    score_movement: float = 1.0
    converged: bool = False
    # What the tournament decided, in the judge's own words. Eighteen match
    # rationales behind a fold and four numbers above it told a reader the
    # ranking happened without telling them what it found: which candidate
    # separated from the field, on what, and where the order is too close to
    # act on. Written after the last match, by the model that judged them.
    briefing: str = ""
    # Who wrote it. The fallback is arithmetic over the match record: true, but
    # not a reading of the hypotheses, and it repeats the standings table it
    # would sit under. Surfaces that can only afford one of the two need to know
    # which one they are holding.
    briefing_author: Literal["judge", "computed"] = "computed"


class EvolutionRecord(Contract):
    id: str = Field(default_factory=lambda: new_id("evolution"))
    parent_ids: list[str]
    candidate: Candidate
    changes: list[str]
    critiques_addressed: list[str] = Field(default_factory=list)
    new_prediction: str
    requires_rereview: bool = True
    round_number: int = Field(default=1, ge=1, le=3)


class EvolutionCycle(Contract):
    records: list[EvolutionRecord] = Field(default_factory=list)
    rereviews: list[CandidateReview] = Field(default_factory=list)
    ranking_history: list[TournamentState] = Field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""


class ResearchCluster(Contract):
    id: str = Field(default_factory=lambda: new_id("cluster"))
    name: str
    candidate_ids: list[str]
    shared_mechanism: str
    shared_outcome: str
    evidence_overlap: list[str] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)


class ResearchLandscape(Contract):
    clusters: list[ResearchCluster] = Field(default_factory=list)
    duplicates: list[list[str]] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    protected_minority_ids: list[str] = Field(default_factory=list)


class DossierSection(Contract):
    key: str
    title: str
    artifact_ids: list[str] = Field(default_factory=list)


class DossierManifest(Contract):
    title: str
    sections: list[DossierSection]
    recommendation_candidate_ids: list[str] = Field(default_factory=list)
    unresolved_fatal_flaw_candidate_ids: list[str] = Field(default_factory=list)
    evidence_that_would_change_decision: list[str] = Field(default_factory=list)


class ResearchBudget(Contract):
    max_candidates: int = Field(default=8, ge=1)
    max_pairwise_comparisons: int = Field(default=18, ge=1)
    max_searches: int = Field(default=20, ge=0)
    max_evolution_rounds: int = Field(default=3, ge=1, le=3)
    max_concurrency: int = Field(default=4, ge=1)


class AgentCard(Contract):
    name: str
    purpose: str
    stage: str
    accepts: str = "application/json"
    produces: str = "application/json"
    tools: list[str] = Field(default_factory=list)


class TaskRecord(Contract):
    id: str = Field(default_factory=lambda: new_id("task"))
    context_id: str
    stage: str
    agent: str
    idempotency_key: str
    state: TaskState = TaskState.SUBMITTED
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_id: str | None = None
    error: str | None = None
    attempt: int = Field(default=1, ge=1)
    submitted_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


MERGE_PRODUCER = "in-process-merge"
"""``producer_model`` for a step that folds other agents' answers and calls no model.

Such a step is neither a model's work nor a substitute for one that failed, and the
field's default said the second of those.
"""


class Artifact(Contract):
    id: str = Field(default_factory=lambda: new_id("artifact"))
    version: int = 1
    parent_id: str | None = None
    stage: str
    agent: str
    content: str
    artifact_type: Literal["specialist_output", "stage_bundle"] = "stage_bundle"
    status: ArtifactStatus = ArtifactStatus.DRAFT
    created_at: str = Field(default_factory=utc_now)
    feedback: str = ""
    producer_model: str = "deterministic-offline"
    prompt_version: str = "1"
    input_artifact_ids: list[str] = Field(default_factory=list)
    schema_name: str = "markdown"
    payload: dict[str, Any] = Field(default_factory=dict)
    # Where the typed payload came from. A deterministic fallback discards the
    # specialist's actual reasoning, so it must be recorded rather than silently
    # substituted: a reader cannot otherwise tell a template from real work.
    payload_source: Literal["specialist", "repaired", "deterministic_fallback"] = (
        "specialist"
    )
    payload_repairs: list[str] = Field(default_factory=list)
    payload_error: str = ""
    checksum: str = ""

    @model_validator(mode="after")
    def populate_checksum(self) -> Artifact:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.checksum and self.checksum != expected:
            raise ValueError("Artifact checksum does not match its content.")
        self.checksum = expected
        return self

    def revise(self, content: str, payload: dict[str, Any] | None = None) -> None:
        """Replace this artifact's body and re-seal it.

        Assigning to ``content`` directly leaves the checksum describing the old
        text, so the artifact validates at write time and fails on the next load.
        Every in-place update has to go through here.
        """
        self.content = content
        if payload is not None:
            self.payload = payload
        self.checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()


class HumanDecision(Contract):
    id: str = Field(default_factory=lambda: new_id("decision"))
    action: DecisionAction
    artifact_id: str | None = None
    artifact_version: int | None = None
    stage: str
    actor: str
    automatic: bool = False
    feedback: str = ""
    created_at: str = Field(default_factory=utc_now)
    session_version: int


class GovernanceAdjudication(Contract):
    """A named person's answer to a governance block.

    A safety gate that can halt a run but offers no way to resolve one is a gate
    operators eventually switch off, so both exits are recorded here rather than
    left to whoever edits the session file. ``withdraw`` drops the hypothesis;
    ``override`` keeps it and accepts the flaw. Neither is anonymous and neither
    is silent: the system cannot judge whether a justification is a good one,
    only that somebody signed it, so the dossier reprints it verbatim next to
    the fatal flaw it answers and lets a reader judge.

    ``fatal_flaws`` is copied rather than referenced because a later revision
    can supersede the review, and an approval must stay attached to the exact
    words it approved.
    """

    id: str = Field(default_factory=lambda: new_id("adjudication"))
    review_id: str
    candidate_id: str
    resolution: Literal["withdraw", "override"]
    adjudicator: str
    justification: str
    fatal_flaws: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _require_attribution(self) -> GovernanceAdjudication:
        if not self.adjudicator.strip():
            raise ValueError("A governance adjudication must name its adjudicator.")
        if not self.justification.strip():
            raise ValueError(
                "A governance adjudication must state a written justification."
            )
        return self


class AuditEvent(Contract):
    id: str = Field(default_factory=lambda: new_id("event"))
    event_type: str
    actor: str
    stage: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class Session(Contract):
    question: str
    id: str = Field(default_factory=lambda: new_id("session"))
    context_id: str = Field(default_factory=lambda: new_id("context"))
    research_mode: str = "experimental"
    # Configured once when the run is created and read back on every resume.
    # A run that started on one model and finished on another would report a
    # single producer_model per artifact and no way to tell which was which,
    # so the choice lives with the session rather than with whichever provider
    # object happens to be reconstructed to advance the next stage.
    #
    # Neither field is validated against the catalogue here on purpose: a saved
    # session must still load after a model is retired from the allowlist, and
    # a Session that refuses to deserialise is a dossier nobody can reprint.
    # The CLI and the web API validate at the point the value is chosen.
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    discipline: str = "general_interdisciplinary"
    approval_mode: ApprovalMode = ApprovalMode.HUMAN
    approval_profile: ApprovalProfile = ApprovalProfile.MILESTONE
    input_requirements: list[InputRequirement] = Field(default_factory=list)
    literature_only: bool = False
    workflow_version: int = Field(default=2, ge=1, le=2)
    # Stop after discovery and show the researcher what was actually found,
    # before four generators spend the rest of the run reasoning over it. The
    # milestone profile treats evidence as internal work, which is right when
    # the corpus is sound and wrong the one time it is thin -- and thin is only
    # visible by reading it. Chosen per run at launch rather than globally: an
    # API caller driving the pipeline unattended has nobody at that gate.
    evidence_review: bool = False
    seeded_evidence_from: str = ""
    """The earlier session this run took its scope and evidence base from.

    Gathering the corpus is the long, expensive half of a run: eight Deep Research
    passes against one question, which a second run of that same question would buy
    again to reach the same place. A fork skips them and starts at generation.

    Recorded on the session because the report has to say it. Without this field the
    dossier reprints the corpus's own provenance -- eight passes, twenty-four dollars
    -- as though this run had done that work, and a reader comparing two forks of one
    corpus would take them for two independent searches agreeing with each other.
    """
    exploratory_evidence_accepted: bool = False
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    artifacts: list[Artifact] = Field(default_factory=list)
    tasks: list[TaskRecord] = Field(default_factory=list)
    decisions: list[HumanDecision] = Field(default_factory=list)
    governance_adjudications: list[GovernanceAdjudication] = Field(default_factory=list)
    events: list[AuditEvent] = Field(default_factory=list)
    current_stage: int = 0
    status: str = "active"
    version: int = 0
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def populate_discipline_if_default(self) -> Session:
        if self.discipline == "general_interdisciplinary" and self.question:
            try:
                from .disciplines import classify_discipline

                classified = classify_discipline(self.question)
                if classified != "general_interdisciplinary":
                    self.discipline = classified
            except ImportError:
                pass
        return self

    def artifact(self, stage: str, *, accepted_only: bool = True) -> Artifact | None:
        return next(
            (
                item
                for item in reversed(self.artifacts)
                if item.stage == stage
                and (not accepted_only or item.status == ArtifactStatus.ACCEPTED)
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Load current sessions and migrate the original linear JSON format."""
        migrated = dict(data)
        migrated.setdefault("schema_version", SCHEMA_VERSION)
        migrated.setdefault("context_id", new_id("context"))
        migrated.setdefault("discipline", "general_interdisciplinary")
        migrated.setdefault("approval_mode", ApprovalMode.HUMAN)
        if "approval_profile" not in migrated:
            migrated["approval_profile"] = (
                ApprovalProfile.AUTO
                if migrated["approval_mode"] == ApprovalMode.AUTO
                else ApprovalProfile.STAGE
            )
        migrated.setdefault("input_requirements", [])
        # A session saved before the choice existed was produced on the default
        # model in English, so that is what it is recorded as rather than
        # "unknown": the run really did happen that way.
        migrated.setdefault("model", DEFAULT_MODEL)
        migrated.setdefault("language", DEFAULT_LANGUAGE)
        migrated.setdefault("literature_only", False)
        migrated.setdefault("workflow_version", 1)
        # A session saved before the gate existed ran straight through evidence,
        # which is what off means.
        migrated.setdefault("evidence_review", False)
        migrated.setdefault("exploratory_evidence_accepted", False)
        migrated.setdefault("budget", {})
        migrated.setdefault("tasks", [])
        migrated.setdefault("decisions", [])
        migrated.setdefault("events", [])
        migrated.setdefault("version", 0)
        migrated.setdefault("created_at", utc_now())
        migrated.setdefault("updated_at", utc_now())
        for artifact in migrated.get("artifacts", []):
            artifact.setdefault("schema_version", SCHEMA_VERSION)
            artifact.setdefault("id", new_id("artifact"))
            artifact.setdefault("version", 1)
            artifact.setdefault("parent_id", None)
            artifact.setdefault("status", ArtifactStatus.ACCEPTED)
            artifact.setdefault("artifact_type", "stage_bundle")
            artifact.setdefault("producer_model", "legacy-unverified")
            artifact.setdefault("prompt_version", "legacy")
            artifact.setdefault("input_artifact_ids", [])
            artifact.setdefault("schema_name", "markdown")
            artifact.setdefault("payload", {})
            artifact.setdefault("payload_source", "specialist")
            artifact.setdefault("payload_repairs", [])
            artifact.setdefault("payload_error", "")
            artifact.setdefault(
                "checksum",
                hashlib.sha256(artifact.get("content", "").encode("utf-8")).hexdigest(),
            )
        for task in migrated.get("tasks", []):
            task.setdefault("attempt", 1)
        return cls.model_validate(migrated)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
