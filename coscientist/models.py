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
RESEARCH_MODES = (
    "experimental",
    "observational",
    "computational",
    "theory_simulation",
    "systematic_review",
    "measurement_field",
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


class SourceRecord(Contract):
    id: str = Field(default_factory=lambda: new_id("src"))
    url: str
    title: str = ""
    source_type: str = "unknown"
    accessed_at: str = Field(default_factory=utc_now)
    verification_status: Literal[
        "discovered_unverified", "verified", "inaccessible", "retracted", "corrected"
    ] = "discovered_unverified"
    supports_claim_ids: list[str] = Field(default_factory=list)


class EvidenceClaim(Contract):
    id: str = Field(default_factory=lambda: new_id("claim"))
    claim: str
    source_id: str | None = None
    exact_location: str = ""
    relation: Literal["supports", "contradicts", "neutral"] = "neutral"
    verification_status: Literal[
        "discovered_unverified",
        "verified",
        "inaccessible",
        "retracted",
        "corrected",
    ] = "discovered_unverified"
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
        verified_sources = bool(self.sources) and all(
            source.verification_status in {"verified", "corrected"}
            for source in self.sources
        )
        verified_claims = bool(self.claims) and all(
            claim.source_id
            and claim.exact_location
            and claim.verification_status in {"verified", "corrected"}
            for claim in self.claims
        )
        return verified_sources and verified_claims


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


class DeepResearchRun(Contract):
    id: str = Field(default_factory=lambda: new_id("deep_research"))
    pass_number: int = Field(ge=1, le=3)
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
    originating_pass: int = Field(ge=1, le=3)
    relation: Literal["supports", "contradicts", "neutral"] = "neutral"
    uncertainty: str = ""


class DiscoveryNarrative(Contract):
    question: str
    research_directions: list[str] = Field(default_factory=list)
    statements: list[DiscoveryStatement] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    summary: str = ""


class SourceLead(Contract):
    id: str = Field(default_factory=lambda: new_id("lead"))
    canonical_url: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    source_type: str = "unknown"
    provider: str = "deep_research"
    originating_passes: list[int] = Field(default_factory=list)
    originating_statement_ids: list[str] = Field(default_factory=list)
    verification_status: Literal["discovered_unverified"] = "discovered_unverified"
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
    runs: list[DeepResearchRun] = Field(default_factory=list, max_length=3)
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


class Candidate(Contract):
    id: str = Field(default_factory=lambda: new_id("candidate"))
    version: int = 1
    parent_ids: list[str] = Field(default_factory=list)
    claim: str
    rationale: str
    predictions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    falsifier: str
    evidence_ids: list[str] = Field(default_factory=list)
    generation_strategy: Literal[
        "evidence_first",
        "mechanism_first",
        "analogy_transfer",
        "competing_explanation",
    ] = "mechanism_first"
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    go_no_go_tests: list[str] = Field(default_factory=list)


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
    approval_mode: ApprovalMode = ApprovalMode.HUMAN
    approval_profile: ApprovalProfile = ApprovalProfile.MILESTONE
    input_requirements: list[InputRequirement] = Field(default_factory=list)
    literature_only: bool = False
    workflow_version: int = Field(default=2, ge=1, le=2)
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
